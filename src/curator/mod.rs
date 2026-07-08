//! Curator: Partition-tree index for low-selectivity filtered vector search.
//!
//! When fewer than ~5% of vectors match a filter predicate, graph-based
//! indexes (HNSW, Vamana) fail because the matching subgraph is disconnected.
//! Curator uses a hierarchical k-means tree with per-label sorted-ID buffers
//! and Bloom filters to find matching vectors by spatial containment rather
//! than graph traversal.
//!
//! # Feature Flag
//!
//! ```toml
//! vicinity = { version = "0.10.5", features = ["curator"] }
//! ```
//!
//! # Quick Start
//!
//! ```ignore
//! use vicinity::curator::{CuratorIndex, CuratorParams};
//!
//! let params = CuratorParams::default();
//! let mut index = CuratorIndex::new(128, params)?;
//!
//! // Add vectors with labels
//! index.add(0, vec![0.1; 128], vec!["color:red".into()])?;
//! index.add(1, vec![0.2; 128], vec!["color:blue".into()])?;
//! index.build()?;
//!
//! // Filtered search: only "color:red" vectors
//! let results = index.search_filtered(&query, 10, "color:red")?;
//! ```
//!
//! # Architecture
//!
//! ```text
//! Base index T (shared k-means tree, all vectors)
//!   └── Per-label index T_l (embedded sorted-ID buffers + Bloom filters)
//! ```
//!
//! - No vector duplication: per-label indexes store only integer IDs
//! - Memory overhead: ~4% over base tree
//! - Complement to graph indexes: dispatch by selectivity threshold
//!
//! # References
//!
//! - Jin et al. (2026). "Curator: Efficient Vector Search with
//!   Low-Selectivity Filters." SIGMOD 2026. arXiv:2601.01291.

use crate::distance::cosine_distance_normalized;
use crate::distance::FloatOrd;
use crate::RetrieveError;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

const CURATOR_FORMAT_VERSION: u32 = 1;

/// Curator parameters.
#[derive(Clone, Debug)]
pub struct CuratorParams {
    /// Branching factor for k-means tree. Default: 16.
    pub branching_factor: usize,
    /// Maximum vectors per leaf. Default: 128.
    pub max_leaf_size: usize,
    /// Search ef (exploration budget). Default: 256.
    pub ef_search: usize,
    /// Beam width for tree descent initialization. Default: 4.
    pub beam_width: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PersistedCuratorParams {
    branching_factor: usize,
    max_leaf_size: usize,
    ef_search: usize,
    beam_width: usize,
}

impl From<&CuratorParams> for PersistedCuratorParams {
    fn from(params: &CuratorParams) -> Self {
        Self {
            branching_factor: params.branching_factor,
            max_leaf_size: params.max_leaf_size,
            ef_search: params.ef_search,
            beam_width: params.beam_width,
        }
    }
}

impl PersistedCuratorParams {
    fn into_params(self) -> CuratorParams {
        CuratorParams {
            branching_factor: self.branching_factor,
            max_leaf_size: self.max_leaf_size,
            ef_search: self.ef_search,
            beam_width: self.beam_width,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CuratorManifest {
    version: u32,
    dimension: usize,
    num_vectors: usize,
    params: PersistedCuratorParams,
}

impl Default for CuratorParams {
    fn default() -> Self {
        Self {
            branching_factor: 16,
            max_leaf_size: 128,
            ef_search: 256,
            beam_width: 4,
        }
    }
}

/// Compact Bloom filter backed by a fixed-size bit array.
#[derive(Debug)]
struct BloomFilter {
    bits: Vec<u64>,
    num_bits: usize,
}

impl BloomFilter {
    fn new(num_bits: usize) -> Self {
        let words = num_bits.div_ceil(64);
        Self {
            bits: vec![0u64; words],
            num_bits,
        }
    }

    fn insert(&mut self, hash: u64) {
        // Use two independent hash positions (double hashing)
        let h1 = hash as usize % self.num_bits;
        let h2 = (hash.wrapping_shr(32) as usize).wrapping_mul(0x9e3779b9) % self.num_bits;
        self.bits[h1 / 64] |= 1u64 << (h1 % 64);
        self.bits[h2 / 64] |= 1u64 << (h2 % 64);
    }

    fn may_contain(&self, hash: u64) -> bool {
        let h1 = hash as usize % self.num_bits;
        let h2 = (hash.wrapping_shr(32) as usize).wrapping_mul(0x9e3779b9) % self.num_bits;
        (self.bits[h1 / 64] & (1u64 << (h1 % 64))) != 0
            && (self.bits[h2 / 64] & (1u64 << (h2 % 64))) != 0
    }

    fn owned_bytes(&self) -> usize {
        self.bits.capacity() * std::mem::size_of::<u64>()
    }
}

/// A node in the k-means tree.
#[derive(Debug)]
struct TreeNode {
    /// Centroid of this node's subtree (d-dimensional).
    centroid: Vec<f32>,
    /// Child node indices (empty for leaf nodes).
    children: Vec<usize>,
    /// Vector IDs at this node (non-empty only for leaves).
    vector_ids: Vec<u32>,
    /// Bloom filter: which labels have vectors in this subtree (256-bit).
    label_bloom: BloomFilter,
    /// Per-label sorted buffers: label_hash -> sorted vec of vector IDs.
    /// Non-empty only at "leaf" nodes of each label's sub-index.
    label_buffers: HashMap<u64, Vec<u32>>,
}

impl TreeNode {
    fn owned_bytes(&self) -> usize {
        self.centroid.capacity() * std::mem::size_of::<f32>()
            + self.children.capacity() * std::mem::size_of::<usize>()
            + self.vector_ids.capacity() * std::mem::size_of::<u32>()
            + self.label_bloom.owned_bytes()
            + self.label_buffers.capacity() * std::mem::size_of::<(u64, Vec<u32>)>()
            + self
                .label_buffers
                .values()
                .map(|ids| ids.capacity() * std::mem::size_of::<u32>())
                .sum::<usize>()
    }
}

/// Curator index.
pub struct CuratorIndex {
    dimension: usize,
    params: CuratorParams,
    built: bool,

    /// Flat vector storage.
    vectors: Vec<f32>,
    num_vectors: usize,
    doc_ids: Vec<u32>,

    /// Labels per vector (staging, pre-build).
    staging_labels: Vec<Vec<String>>,

    /// The k-means tree.
    nodes: Vec<TreeNode>,
    /// Root node index.
    root: usize,
}

impl CuratorIndex {
    /// Create a new Curator index.
    pub fn new(dimension: usize, params: CuratorParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be > 0".into(),
            ));
        }
        Ok(Self {
            dimension,
            params,
            built: false,
            vectors: Vec::new(),
            num_vectors: 0,
            doc_ids: Vec::new(),
            staging_labels: Vec::new(),
            nodes: Vec::new(),
            root: 0,
        })
    }

    /// Add a vector with associated labels.
    pub fn add(
        &mut self,
        doc_id: u32,
        vector: Vec<f32>,
        labels: Vec<String>,
    ) -> Result<(), RetrieveError> {
        if self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot add after build".into(),
            ));
        }
        if vector.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: vector.len(),
                doc_dim: self.dimension,
            });
        }

        // L2-normalize
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            self.vectors.extend(vector.iter().map(|x| x / norm));
        } else {
            self.vectors.extend_from_slice(&vector);
        }
        self.doc_ids.push(doc_id);
        self.staging_labels.push(labels);
        self.num_vectors += 1;
        Ok(())
    }

    /// Build the index: construct k-means tree, populate per-label buffers.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        self.nodes.clear();
        let all_ids: Vec<u32> = (0..self.num_vectors as u32).collect();
        self.root = self.build_tree(&all_ids, 0);

        // Populate per-label buffers (bottom-up)
        let labels = std::mem::take(&mut self.staging_labels);
        for (internal_id, doc_labels) in labels.iter().enumerate() {
            for label in doc_labels {
                let label_hash = hash_label(label);
                self.insert_label_entry(self.root, internal_id as u32, label_hash);
            }
        }
        self.staging_labels = labels;

        self.built = true;
        Ok(())
    }

    /// Save a built Curator index to a directory.
    pub fn save_to_dir(&self, output_dir: impl AsRef<Path>) -> Result<(), RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot save unbuilt curator index".into(),
            ));
        }
        if self.staging_labels.len() != self.num_vectors {
            return Err(RetrieveError::InvalidParameter(
                "curator labels are unavailable for persistence".into(),
            ));
        }

        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;
        let manifest = CuratorManifest {
            version: CURATOR_FORMAT_VERSION,
            dimension: self.dimension,
            num_vectors: self.num_vectors,
            params: PersistedCuratorParams::from(&self.params),
        };

        write_json_atomic(&output_dir.join("manifest.json"), &manifest)?;
        write_json_atomic(&output_dir.join("labels.json"), &self.staging_labels)?;
        write_f32_atomic(&output_dir.join("vectors.bin"), &self.vectors)?;
        write_u32_atomic(&output_dir.join("doc_ids.bin"), &self.doc_ids)?;
        Ok(())
    }

    /// Load a Curator index saved by [`Self::save_to_dir`].
    pub fn load_from_dir(input_dir: impl AsRef<Path>) -> Result<Self, RetrieveError> {
        let input_dir = input_dir.as_ref();
        let manifest: CuratorManifest = read_json(&input_dir.join("manifest.json"))?;
        if manifest.version != CURATOR_FORMAT_VERSION {
            return Err(RetrieveError::FormatError(format!(
                "unsupported curator format version {}",
                manifest.version
            )));
        }
        if manifest.dimension == 0 {
            return Err(RetrieveError::FormatError(
                "curator manifest has zero dimension".into(),
            ));
        }
        if manifest.num_vectors == 0 {
            return Err(RetrieveError::FormatError(
                "curator manifest has zero vectors".into(),
            ));
        }

        let vectors = read_f32_exact(
            &input_dir.join("vectors.bin"),
            manifest.num_vectors * manifest.dimension,
        )?;
        let doc_ids = read_u32_exact(&input_dir.join("doc_ids.bin"), manifest.num_vectors)?;
        let labels: Vec<Vec<String>> = read_json(&input_dir.join("labels.json"))?;
        if labels.len() != manifest.num_vectors {
            return Err(RetrieveError::FormatError(format!(
                "curator labels count {} does not match vector count {}",
                labels.len(),
                manifest.num_vectors
            )));
        }

        let mut index = Self {
            dimension: manifest.dimension,
            params: manifest.params.into_params(),
            built: false,
            vectors,
            num_vectors: manifest.num_vectors,
            doc_ids,
            staging_labels: labels,
            nodes: Vec::new(),
            root: 0,
        };
        index.build()?;
        Ok(index)
    }

    /// Search with a label filter.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        label: &str,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "index must be built before search".into(),
            ));
        }
        if query.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.dimension,
            });
        }

        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let query_normalized: Vec<f32> = if query_norm > 1e-10 {
            query.iter().map(|x| x / query_norm).collect()
        } else {
            query.to_vec()
        };

        let label_hash = hash_label(label);
        self.best_first_search(&query_normalized, k, label_hash)
    }

    /// Unfiltered search (scans all leaves).
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "index must be built before search".into(),
            ));
        }
        if query.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.dimension,
            });
        }

        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let query_normalized: Vec<f32> = if query_norm > 1e-10 {
            query.iter().map(|x| x / query_norm).collect()
        } else {
            query.to_vec()
        };

        // Unfiltered: traverse all leaves via best-first on centroids
        self.best_first_unfiltered(&query_normalized, k)
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.num_vectors
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    /// Estimated heap memory used by this index.
    pub fn memory_usage(&self) -> crate::memory::MemoryReport {
        let vectors_bytes = self.vectors.capacity() * std::mem::size_of::<f32>();
        let graph_bytes = self.nodes.capacity() * std::mem::size_of::<TreeNode>()
            + self.nodes.iter().map(TreeNode::owned_bytes).sum::<usize>();
        let metadata_bytes = self.doc_ids.capacity() * std::mem::size_of::<u32>()
            + self.staging_labels.capacity() * std::mem::size_of::<Vec<String>>()
            + label_payload_bytes(&self.staging_labels);

        crate::memory::MemoryReport {
            vectors_bytes,
            graph_bytes,
            quantized_bytes: 0,
            metadata_bytes,
        }
    }

    // ── Internal: tree construction ────────────────────────────────────

    fn build_tree(&mut self, ids: &[u32], depth: usize) -> usize {
        let dim = self.dimension;

        // Compute centroid
        let mut centroid = vec![0.0f32; dim];
        for &id in ids {
            let v = self.get_vector(id as usize);
            for (j, &val) in v.iter().enumerate() {
                centroid[j] += val;
            }
        }
        let n = ids.len() as f32;
        for c in &mut centroid {
            *c /= n;
        }

        let node_idx = self.nodes.len();

        if ids.len() <= self.params.max_leaf_size || depth > 15 {
            // Leaf node
            self.nodes.push(TreeNode {
                centroid,
                children: Vec::new(),
                vector_ids: ids.to_vec(),
                label_bloom: BloomFilter::new(256),
                label_buffers: HashMap::new(),
            });
            return node_idx;
        }

        // Internal node: k-means split
        let k = self.params.branching_factor.min(ids.len());
        let assignments = self.simple_kmeans(ids, k);

        // Create placeholder node (will fill children after recursion)
        self.nodes.push(TreeNode {
            centroid,
            children: Vec::new(),
            vector_ids: Vec::new(),
            label_bloom: BloomFilter::new(256),
            label_buffers: HashMap::new(),
        });

        // Build child subtrees
        let mut children = Vec::with_capacity(k);
        for cluster in &assignments {
            if cluster.is_empty() {
                continue;
            }
            let child_idx = self.build_tree(cluster, depth + 1);
            children.push(child_idx);
        }

        self.nodes[node_idx].children = children;
        node_idx
    }

    /// Simple k-means: one pass (good enough for tree construction).
    fn simple_kmeans(&self, ids: &[u32], k: usize) -> Vec<Vec<u32>> {
        let dim = self.dimension;
        let n = ids.len();

        // Initialize centroids: pick k evenly-spaced points
        let step = n / k;
        let mut centroids: Vec<Vec<f32>> = (0..k)
            .map(|i| {
                let idx = ids[(i * step).min(n - 1)] as usize;
                self.get_vector(idx).to_vec()
            })
            .collect();

        let mut assignments = vec![Vec::new(); k];

        // 3 iterations of Lloyd's algorithm
        for _ in 0..3 {
            // Clear assignments
            for a in &mut assignments {
                a.clear();
            }

            // Assign each point to nearest centroid
            for &id in ids {
                let v = self.get_vector(id as usize);
                let mut best_c = 0;
                let mut best_d = f32::INFINITY;
                for (ci, c) in centroids.iter().enumerate() {
                    let d = cosine_distance_normalized(v, c);
                    if d < best_d {
                        best_d = d;
                        best_c = ci;
                    }
                }
                assignments[best_c].push(id);
            }

            // Update centroids
            for (ci, cluster) in assignments.iter().enumerate() {
                if cluster.is_empty() {
                    continue;
                }
                let mut new_centroid = vec![0.0f32; dim];
                for &id in cluster {
                    let v = self.get_vector(id as usize);
                    for (j, &val) in v.iter().enumerate() {
                        new_centroid[j] += val;
                    }
                }
                let cn = cluster.len() as f32;
                for c in &mut new_centroid {
                    *c /= cn;
                }
                centroids[ci] = new_centroid;
            }
        }

        assignments
    }

    // ── Internal: per-label buffer insertion ───────────────────────────

    /// Insert a (vector_id, label_hash) pair into the tree's per-label structure.
    fn insert_label_entry(&mut self, node_idx: usize, vector_id: u32, label_hash: u64) {
        // Update Bloom filter
        self.nodes[node_idx].label_bloom.insert(label_hash);

        if self.nodes[node_idx].children.is_empty() {
            // Leaf: insert into per-label buffer
            self.nodes[node_idx]
                .label_buffers
                .entry(label_hash)
                .or_default()
                .push(vector_id);
        } else {
            // Internal: recurse into the child that contains this vector
            let children = self.nodes[node_idx].children.clone();
            for &child_idx in &children {
                if self.subtree_contains(child_idx, vector_id) {
                    self.insert_label_entry(child_idx, vector_id, label_hash);
                    return;
                }
            }
        }
    }

    /// Check if a subtree contains a given vector ID.
    fn subtree_contains(&self, node_idx: usize, vector_id: u32) -> bool {
        let node = &self.nodes[node_idx];
        if node.children.is_empty() {
            return node.vector_ids.contains(&vector_id);
        }
        for &child in &node.children {
            if self.subtree_contains(child, vector_id) {
                return true;
            }
        }
        false
    }

    // ── Internal: search ───────────────────────────────────────────────

    /// Best-first search with label filter and Bloom filter pruning.
    fn best_first_search(
        &self,
        query: &[f32],
        k: usize,
        label_hash: u64,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        let num_nodes = self.nodes.len();

        thread_local! {
            static VISITED: std::cell::RefCell<(Vec<u8>, u8)> =
                const { std::cell::RefCell::new((Vec::new(), 1)) };
        }

        VISITED.with(|cell| {
            let (marks, gen) = &mut *cell.borrow_mut();
            if marks.len() < num_nodes {
                marks.resize(num_nodes, 0);
            }
            if let Some(next) = gen.checked_add(1) {
                *gen = next;
            } else {
                marks.fill(0);
                *gen = 1;
            }
            let generation = *gen;

            let mut visited_insert = |idx: usize| -> bool {
                if idx < marks.len() && marks[idx] != generation {
                    marks[idx] = generation;
                    true
                } else {
                    idx >= marks.len()
                }
            };

            let mut heap: BinaryHeap<std::cmp::Reverse<(FloatOrd, usize)>> = BinaryHeap::new();
            let mut results: Vec<(u32, f32)> = Vec::new();
            let mut explored = 0usize;

            // Start from root
            let root_dist = cosine_distance_normalized(query, &self.nodes[self.root].centroid);
            heap.push(std::cmp::Reverse((FloatOrd(root_dist), self.root)));

            while let Some(std::cmp::Reverse((_, node_idx))) = heap.pop() {
                if !visited_insert(node_idx) {
                    continue;
                }
                explored += 1;

                if explored > self.params.ef_search {
                    break;
                }

                let node = &self.nodes[node_idx];

                // Bloom filter check: skip if this label has no vectors here
                if !node.label_bloom.may_contain(label_hash) {
                    continue;
                }

                if node.children.is_empty() {
                    // Leaf node: check per-label buffer
                    if let Some(buffer) = node.label_buffers.get(&label_hash) {
                        for &vid in buffer {
                            let v = self.get_vector(vid as usize);
                            let dist = cosine_distance_normalized(query, v);
                            let doc_id = self.doc_ids[vid as usize];
                            results.push((doc_id, dist));
                        }
                    }
                } else {
                    // Internal node: expand children
                    for &child_idx in &node.children {
                        let child_dist =
                            cosine_distance_normalized(query, &self.nodes[child_idx].centroid);
                        heap.push(std::cmp::Reverse((FloatOrd(child_dist), child_idx)));
                    }
                }
            }

            results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            results.dedup_by_key(|c| c.0);
            results.truncate(k);
            Ok(results)
        })
    }

    /// Unfiltered best-first search.
    fn best_first_unfiltered(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        let num_nodes = self.nodes.len();

        thread_local! {
            static VISITED_UF: std::cell::RefCell<(Vec<u8>, u8)> =
                const { std::cell::RefCell::new((Vec::new(), 1)) };
        }

        VISITED_UF.with(|cell| {
            let (marks, gen) = &mut *cell.borrow_mut();
            if marks.len() < num_nodes {
                marks.resize(num_nodes, 0);
            }
            if let Some(next) = gen.checked_add(1) {
                *gen = next;
            } else {
                marks.fill(0);
                *gen = 1;
            }
            let generation = *gen;

            let mut visited_insert = |idx: usize| -> bool {
                if idx < marks.len() && marks[idx] != generation {
                    marks[idx] = generation;
                    true
                } else {
                    idx >= marks.len()
                }
            };

            let mut heap: BinaryHeap<std::cmp::Reverse<(FloatOrd, usize)>> = BinaryHeap::new();
            let mut results: Vec<(u32, f32)> = Vec::new();
            let mut explored = 0usize;

            let root_dist = cosine_distance_normalized(query, &self.nodes[self.root].centroid);
            heap.push(std::cmp::Reverse((FloatOrd(root_dist), self.root)));

            while let Some(std::cmp::Reverse((_, node_idx))) = heap.pop() {
                if !visited_insert(node_idx) {
                    continue;
                }
                explored += 1;

                if explored > self.params.ef_search {
                    break;
                }

                let node = &self.nodes[node_idx];

                if node.children.is_empty() {
                    // Leaf: scan all vectors
                    for &vid in &node.vector_ids {
                        let v = self.get_vector(vid as usize);
                        let dist = cosine_distance_normalized(query, v);
                        let doc_id = self.doc_ids[vid as usize];
                        results.push((doc_id, dist));
                    }
                } else {
                    for &child_idx in &node.children {
                        let child_dist =
                            cosine_distance_normalized(query, &self.nodes[child_idx].centroid);
                        heap.push(std::cmp::Reverse((FloatOrd(child_dist), child_idx)));
                    }
                }
            }

            results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            results.dedup_by_key(|c| c.0);
            results.truncate(k);
            Ok(results)
        })
    }

    #[inline]
    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        &self.vectors[start..start + self.dimension]
    }
}

/// Hash a label string to u64 (FNV-1a).
fn hash_label(label: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in label.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn label_payload_bytes(labels: &[Vec<String>]) -> usize {
    labels
        .iter()
        .map(|row| {
            row.capacity() * std::mem::size_of::<String>()
                + row.iter().map(String::capacity).sum::<usize>()
        })
        .sum::<usize>()
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), RetrieveError> {
    write_atomic(path, |writer| {
        serde_json::to_writer_pretty(writer, value)
            .map_err(|e| std::io::Error::other(e.to_string()))
    })
}

fn write_f32_atomic(path: &Path, values: &[f32]) -> Result<(), RetrieveError> {
    write_atomic(path, |writer| {
        for value in values {
            writer.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    })
}

fn write_u32_atomic(path: &Path, values: &[u32]) -> Result<(), RetrieveError> {
    write_atomic(path, |writer| {
        for value in values {
            writer.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    })
}

fn write_atomic(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<std::fs::File>) -> std::io::Result<()>,
) -> Result<(), RetrieveError> {
    let tmp_path = path.with_extension("tmp");
    {
        let file = std::fs::File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);
        write(&mut writer)?;
        writer.flush()?;
    }
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, RetrieveError> {
    let file = std::fs::File::open(path)?;
    serde_json::from_reader(BufReader::new(file))
        .map_err(|e| RetrieveError::FormatError(e.to_string()))
}

fn read_f32_exact(path: &Path, expected_len: usize) -> Result<Vec<f32>, RetrieveError> {
    let bytes = std::fs::read(path)?;
    let expected_bytes = expected_len
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| RetrieveError::FormatError("curator f32 byte length overflow".into()))?;
    if bytes.len() != expected_bytes {
        return Err(RetrieveError::FormatError(format!(
            "{} size mismatch: expected {} bytes, got {}",
            path.display(),
            expected_bytes,
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn read_u32_exact(path: &Path, expected_len: usize) -> Result<Vec<u32>, RetrieveError> {
    let bytes = std::fs::read(path)?;
    let expected_bytes = expected_len
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| RetrieveError::FormatError("curator u32 byte length overflow".into()))?;
    if bytes.len() != expected_bytes {
        return Err(RetrieveError::FormatError(format!(
            "{} size mismatch: expected {} bytes, got {}",
            path.display(),
            expected_bytes,
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn make_vector(dim: usize, seed: u32) -> Vec<f32> {
        (0..dim)
            .map(|i| (seed as f32 * 0.1 + i as f32 * 0.01).sin())
            .collect()
    }

    #[test]
    fn build_and_filtered_search() {
        let dim = 16;
        let mut index = CuratorIndex::new(dim, CuratorParams::default()).unwrap();

        // Add 50 vectors, half "red", half "blue"
        for i in 0..50u32 {
            let label = if i % 2 == 0 { "red" } else { "blue" };
            index
                .add(i, make_vector(dim, i), vec![label.into()])
                .unwrap();
        }
        index.build().unwrap();

        // Search for "red" only
        let query = make_vector(dim, 0);
        let results = index.search_filtered(&query, 5, "red").unwrap();

        assert!(!results.is_empty());
        // All results should be "red" (even doc_ids)
        for (doc_id, _) in &results {
            assert_eq!(doc_id % 2, 0, "expected even doc_id (red), got {}", doc_id);
        }
    }

    #[test]
    fn unfiltered_search() {
        let dim = 16;
        let mut index = CuratorIndex::new(dim, CuratorParams::default()).unwrap();

        for i in 0..30u32 {
            index
                .add(i, make_vector(dim, i), vec!["any".into()])
                .unwrap();
        }
        index.build().unwrap();

        let query = make_vector(dim, 0);
        let results = index.search(&query, 5).unwrap();
        assert!(!results.is_empty());
        // doc_id 0 should be closest (self-match)
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn nonexistent_label_returns_empty() {
        let dim = 16;
        let mut index = CuratorIndex::new(dim, CuratorParams::default()).unwrap();

        for i in 0..20u32 {
            index
                .add(i, make_vector(dim, i), vec!["exists".into()])
                .unwrap();
        }
        index.build().unwrap();

        let query = make_vector(dim, 0);
        let results = index.search_filtered(&query, 5, "nonexistent").unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn multi_label_vectors() {
        let dim = 16;
        let mut index = CuratorIndex::new(dim, CuratorParams::default()).unwrap();

        // Vectors with multiple labels
        for i in 0..20u32 {
            let mut labels = vec!["all".to_string()];
            if i % 3 == 0 {
                labels.push("mod3".into());
            }
            index.add(i, make_vector(dim, i), labels).unwrap();
        }
        index.build().unwrap();

        // Search "mod3": should return 0, 3, 6, 9, 12, 15, 18
        let query = make_vector(dim, 0);
        let results = index.search_filtered(&query, 10, "mod3").unwrap();
        for (doc_id, _) in &results {
            assert_eq!(doc_id % 3, 0, "expected mod3 doc_id, got {}", doc_id);
        }

        // Search "all": should return any vector
        let results = index.search_filtered(&query, 5, "all").unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn low_selectivity_recall() {
        // Simulate low selectivity: 5 vectors out of 100 match
        let dim = 16;
        let mut index = CuratorIndex::new(
            dim,
            CuratorParams {
                max_leaf_size: 32,
                ef_search: 512,
                ..Default::default()
            },
        )
        .unwrap();

        for i in 0..100u32 {
            let label = if i < 5 { "rare" } else { "common" };
            index
                .add(i, make_vector(dim, i), vec![label.into()])
                .unwrap();
        }
        index.build().unwrap();

        let query = make_vector(dim, 2); // close to doc_id 2
        let results = index.search_filtered(&query, 3, "rare").unwrap();

        // Should find some of the 5 rare vectors
        assert!(!results.is_empty());
        for (doc_id, _) in &results {
            assert!(*doc_id < 5, "expected rare doc_id < 5, got {}", doc_id);
        }
    }

    #[test]
    fn save_load_roundtrip_preserves_filtered_search() {
        let dim = 16;
        let mut index = CuratorIndex::new(
            dim,
            CuratorParams {
                max_leaf_size: 16,
                ef_search: 128,
                ..Default::default()
            },
        )
        .unwrap();

        for i in 0..64u32 {
            let mut labels = vec![if i % 2 == 0 { "even" } else { "odd" }.to_string()];
            if i < 8 {
                labels.push("rare".to_string());
            }
            index.add(i, make_vector(dim, i), labels).unwrap();
        }
        index.build().unwrap();

        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let loaded = CuratorIndex::load_from_dir(dir.path()).unwrap();

        let query = make_vector(dim, 2);
        assert_eq!(
            index.search_filtered(&query, 5, "rare").unwrap(),
            loaded.search_filtered(&query, 5, "rare").unwrap()
        );
        assert_eq!(
            index.search(&query, 5).unwrap(),
            loaded.search(&query, 5).unwrap()
        );
    }

    #[test]
    fn load_rejects_future_manifest_version() {
        let dim = 8;
        let mut index = CuratorIndex::new(dim, CuratorParams::default()).unwrap();
        for i in 0..16u32 {
            index
                .add(i, make_vector(dim, i), vec!["label".to_string()])
                .unwrap();
        }
        index.build().unwrap();

        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let manifest_path = dir.path().join("manifest.json");
        let mut manifest: CuratorManifest = read_json(&manifest_path).unwrap();
        manifest.version = CURATOR_FORMAT_VERSION + 1;
        write_json_atomic(&manifest_path, &manifest).unwrap();

        match CuratorIndex::load_from_dir(dir.path()) {
            Err(RetrieveError::FormatError(message)) => {
                assert!(message.contains("unsupported curator format version"));
            }
            Err(err) => panic!("expected format error, got {err:?}"),
            Ok(_) => panic!("expected format error, got successful load"),
        }
    }

    #[test]
    fn load_rejects_truncated_vectors() {
        let dim = 8;
        let mut index = CuratorIndex::new(dim, CuratorParams::default()).unwrap();
        for i in 0..16u32 {
            index
                .add(i, make_vector(dim, i), vec!["label".to_string()])
                .unwrap();
        }
        index.build().unwrap();

        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        std::fs::write(dir.path().join("vectors.bin"), [1u8, 2, 3]).unwrap();

        match CuratorIndex::load_from_dir(dir.path()) {
            Err(RetrieveError::FormatError(message)) => {
                assert!(message.contains("vectors.bin size mismatch"));
            }
            Err(err) => panic!("expected format error, got {err:?}"),
            Ok(_) => panic!("expected format error, got successful load"),
        }
    }

    #[test]
    fn empty_index_errors() {
        let mut index = CuratorIndex::new(8, CuratorParams::default()).unwrap();
        assert!(index.build().is_err());
    }
}
