//! File-backed HNSW segment searcher.
//!
//! [`HNSWSegmentSearcher`] answers k-NN queries directly over the files a
//! `HNSWSegmentWriter` wrote
//! into `segments/segment_hnsw_<id>/`, without loading the graph or the full
//! vector matrix into the heap. It mirrors
//! [`HNSWIndex::search`](crate::hnsw::HNSWIndex::search) layer by layer so that
//! recall and result ordering match the in-memory index at the same `ef`.
//!
//! # Memory shape
//!
//! The searcher holds only per-node bookkeeping plus a bounded block cache:
//!
//! - layer 0: one `u32` file offset per node (`4 * n` bytes);
//! - each upper layer: `(node id, file offset)` pairs for the nodes that
//!   actually occur in that layer (about `n / m_l` entries total across all
//!   upper layers, `8` bytes each);
//! - `layer_assignments.bin`, `doc_ids.bin`, `tombstones.bin` in full
//!   (`1 + 4 + 1` bytes per node, plus 4 bytes per node for `doc_ids`);
//! - one dense visited-generation byte per node;
//! - reusable read buffers and the block cache configured by
//!   [`FileCacheConfig`].
//!
//! None of that is the graph: neighbour ids are never held, only the offset at
//! which each node's neighbour list starts. [`HNSWSegmentSearcher::resident_bytes`] reports
//! the total.
//!
//! # Vector layout: AoS sidecar by default
//!
//! `vectors.bin` is stored dimension-major (SoA): reading one vector means
//! `dimension` four-byte reads at a stride of `n * 4` bytes. With a 4 KiB
//! block cache and `n = 100k`, `dim = 96`, the components of a single vector
//! land roughly `400 KiB` apart, so almost every component is a cache miss:
//! 96 block reads to fetch 384 bytes. The measured comparison in this module
//! (`soa_vs_aos_single_vector_reads`, ignored by default) reads 2 000 random
//! vectors from a 38.4 MB file through an 8 MiB cache: SoA costs roughly
//! `dim` block reads per vector, AoS one.
//!
//! Because of that, [`HNSWSegmentSearcher::open`] prefers an AoS sidecar,
//! `vectors_aos.bin`, written by
//! `HNSWSegmentWriter::write_hnsw_index_aos`.
//! One AoS vector is contiguous, so a single cache block usually holds it.
//! If the sidecar is absent the searcher falls back to the SoA `vectors.bin`
//! so legacy segments still answer queries, but
//! [`HNSWSegmentSearcher::uses_soa_fallback`] then reports `true`.
//!
//! # Threading and cache ownership
//!
//! The block cache is owned by the searcher and searched through `&mut self`,
//! exactly like the DiskANN searchers. One searcher serves one thread at a
//! time; sharing it across threads needs external locking. Parallel search
//! should create one searcher per thread, and each searcher carries its own
//! cache budget — `threads * budget_bytes` is the total resident cache.
//!
//! # Metric support
//!
//! The metric comes from `params.bin`. L2 and Cosine are supported. Angular and
//! InnerProduct return [`RetrieveError::Unsupported`] instead of silently
//! applying a different metric.
//!
//! # Layer layout
//!
//! Every layer stores `num_vectors` neighbour-list records indexed by internal
//! node id, with an empty list for a node that does not occur at that layer.
//! That holds for a `compact-hnsw` index too: the compaction is a resident-memory
//! layout (CSR over eligible nodes), but the writer walks node ids and emits an
//! empty list for the rest, so the file stays addressable by id. A compacted
//! segment therefore reads like any other; what compaction changes is the
//! in-memory shape, not the save format.

use std::borrow::Cow;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io;
use std::path::Path;

use crate::distance::{cosine_distance_normalized, l2_distance, normalize, DistanceMetric};
use crate::error::RetrieveError;
use crate::file_io::{CachedFile, FileCacheConfig, FileCacheStats};

/// Magic bytes for HNSW segment metadata.
///
/// Kept in sync with `persistence::format::HNSW_SEGMENT_MAGIC`. Defined locally
/// so the searcher compiles under the `hnsw` feature without requiring the
/// `persistence` feature (the format module is gated behind it).
const HNSW_SEGMENT_MAGIC: [u8; 8] = *b"VCNHNSW\x01";
/// Segment format version; kept in sync with `persistence::format::FORMAT_VERSION`.
const FORMAT_VERSION: u32 = 1;

/// Number of neighbour distances computed before updating the heaps, matching
/// the in-memory beam search's batching so push order is identical.
const DISTANCE_BATCH_SIZE: usize = 8;
/// Beam width above which the in-memory search caches the worst result.
const CACHED_WORST_MIN_EF: usize = 64;
/// Beam width above which the in-memory search prunes the frontier.
const FRONTIER_PRUNE_MIN_EF: usize = 64;
/// Candidate pops between frontier-pruning attempts.
const FRONTIER_PRUNE_INTERVAL: usize = 64;
/// Upper bound on serialized layer count (HNSW layers are logarithmic).
const MAX_LAYERS: usize = 256;
/// Upper bound on a single neighbour count, mirroring the persistence loader.
const MAX_NEIGHBORS: u32 = 65_536;
/// Guard against crafted metadata claiming enormous indexes.
const MAX_VECTORS: usize = 100_000_000;
/// Guard against crafted metadata claiming enormous dimensions.
const MAX_DIMENSION: usize = 65_536;

/// Which on-disk vector layout the searcher reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VectorLayout {
    /// `vectors_aos.bin`: all components of one vector are contiguous.
    Aos,
    /// `vectors.bin`: dimension-major, components of one vector are strided.
    Soa,
}

/// Sparse offset index for one upper layer.
///
/// Upper layers are saved as dense `num_vectors`-entry records, but only the
/// nodes whose layer assignment covers that layer have non-empty lists. The
/// pairs are stored in ascending node-id order so lookup is a binary search.
#[derive(Debug, Default)]
struct UpperLayerIndex {
    node_ids: Vec<u32>,
    offsets: Vec<u32>,
}

/// Diagnostics from [`HNSWSegmentSearcher::open`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HNSWSegmentOpenStats {
    /// Layers read from `layers.bin`.
    pub layers: usize,
    /// Neighbour-list records scanned (one per node per layer).
    pub records: usize,
    /// Neighbour ids present across all records (the edge count).
    pub edges: usize,
    /// Four-byte header fields read while building the offsets table.
    pub header_bytes_read: u64,
    /// Bytes occupied by the in-memory offsets table.
    pub offsets_bytes: usize,
}

/// k-NN searcher over a saved HNSW segment directory.
///
/// See the [module documentation](self) for the memory shape, the AoS sidecar
/// preference, and the `&mut self` cache limitation.
pub struct HNSWSegmentSearcher {
    dimension: usize,
    num_vectors: usize,
    num_layers: usize,
    metric: DistanceMetric,
    auto_normalize: bool,
    built: bool,
    dist_fn: fn(&[f32], &[f32]) -> f32,

    /// Byte offset of each layer-0 neighbour-list record, indexed by node id.
    base_offsets: Vec<u32>,
    /// Sparse offset indexes for layers 1..num_layers (index 0 is layer 1).
    upper_layers: Vec<UpperLayerIndex>,
    layer_assignments: Vec<u8>,
    doc_ids: Vec<u32>,
    tombstones: Vec<u8>,
    entry_point: u32,
    entry_layer: usize,

    vectors: CachedFile,
    layers_file: CachedFile,
    vector_layout: VectorLayout,

    read_buf: Vec<u8>,
    vec_buf: Vec<f32>,
    batch_buf: Vec<f32>,
    neighbor_buf: Vec<u32>,
    neighbor_bytes: Vec<u8>,
    visited_marks: Vec<u8>,
    visited_generation: u8,
    open_stats: HNSWSegmentOpenStats,
}

/// Candidate for the min-heap (explore closest first).
#[derive(PartialEq)]
struct MinCandidate {
    id: u32,
    distance: f32,
}
impl Eq for MinCandidate {}
impl Ord for MinCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.distance.total_cmp(&self.distance)
    }
}
impl PartialOrd for MinCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Result for the max-heap (track worst result for pruning).
#[derive(PartialEq)]
struct MaxResult {
    id: u32,
    distance: f32,
}
impl Eq for MaxResult {}
impl Ord for MaxResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.total_cmp(&other.distance)
    }
}
impl PartialOrd for MaxResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl HNSWSegmentSearcher {
    /// Open a saved HNSW segment directory with the default block cache.
    ///
    /// `dir` is the segment directory itself (the one containing
    /// `metadata.bin`, `layers.bin`, `vectors.bin`, ...), not its parent.
    ///
    /// # Errors
    ///
    /// Returns an error when a required file is missing or malformed, when the
    /// format version is newer than this crate understands, when the saved
    /// metric is neither L2 nor Cosine, or when the segment uses the
    /// `compact-hnsw` upper-layer layout this reader cannot address.
    pub fn open(dir: impl AsRef<Path>, config: FileCacheConfig) -> Result<Self, RetrieveError> {
        Self::open_with_stats(dir, config).map(|(searcher, _)| searcher)
    }

    /// Same as [`Self::open`], returning the open-time scan diagnostics too.
    pub fn open_with_stats(
        dir: impl AsRef<Path>,
        config: FileCacheConfig,
    ) -> Result<(Self, HNSWSegmentOpenStats), RetrieveError> {
        let dir = dir.as_ref();

        // ── metadata.bin ────────────────────────────────────────────────────
        let metadata = read_file(&dir.join("metadata.bin"))?;
        let (dimension, num_vectors, built) = parse_metadata(&metadata)?;

        // ── params.bin ──────────────────────────────────────────────────────
        let params = read_file(&dir.join("params.bin"))?;
        let params = parse_params(&params)?;
        let metric = match params.metric {
            DistanceMetric::L2 => DistanceMetric::L2,
            DistanceMetric::Cosine => DistanceMetric::Cosine,
            DistanceMetric::Angular => {
                return Err(RetrieveError::Unsupported {
                    feature: "Angular metric in HNSWSegmentSearcher",
                    hint: "the file-backed HNSW searcher supports L2 and Cosine; \
                           use HNSWIndex (in memory) for Angular",
                })
            }
            DistanceMetric::InnerProduct => {
                return Err(RetrieveError::Unsupported {
                    feature: "InnerProduct metric in HNSWSegmentSearcher",
                    hint: "the file-backed HNSW searcher supports L2 and Cosine; \
                           use HNSWIndex (in memory) for InnerProduct",
                })
            }
        };
        let dist_fn: fn(&[f32], &[f32]) -> f32 = match metric {
            DistanceMetric::L2 => l2_distance,
            DistanceMetric::Cosine => cosine_distance_normalized,
            _ => unreachable!("unsupported metrics are rejected above"),
        };

        // ── layer_assignments.bin ───────────────────────────────────────────
        let layer_assignments = read_file(&dir.join("layer_assignments.bin"))?;
        if layer_assignments.len() != num_vectors {
            return Err(RetrieveError::FormatError(format!(
                "layer_assignments.bin size mismatch: expected {} bytes, got {}",
                num_vectors,
                layer_assignments.len()
            )));
        }

        // ── doc_ids.bin (optional; identity fallback for legacy segments) ───
        let doc_ids_path = dir.join("doc_ids.bin");
        let doc_ids = if doc_ids_path.exists() {
            let bytes = read_file(&doc_ids_path)?;
            if bytes.len() != num_vectors * 4 {
                return Err(RetrieveError::FormatError(format!(
                    "doc_ids.bin size mismatch: expected {} bytes, got {}",
                    num_vectors * 4,
                    bytes.len()
                )));
            }
            decode_u32s(&bytes)
        } else {
            (0..num_vectors as u32).collect()
        };

        // ── tombstones.bin (optional; legacy segments have no flags) ────────
        let tombstones_path = dir.join("tombstones.bin");
        let tombstones = if tombstones_path.exists() {
            let bytes = read_file(&tombstones_path)?;
            if bytes.len() != num_vectors {
                return Err(RetrieveError::FormatError(format!(
                    "tombstones.bin size mismatch: expected {} bytes, got {}",
                    num_vectors,
                    bytes.len()
                )));
            }
            bytes
        } else {
            Vec::new()
        };

        // ── vectors: prefer the AoS sidecar ─────────────────────────────────
        let aos_path = dir.join("vectors_aos.bin");
        let soa_path = dir.join("vectors.bin");
        let (vectors_path, vector_layout) = if aos_path.exists() {
            (aos_path, VectorLayout::Aos)
        } else {
            (soa_path, VectorLayout::Soa)
        };
        let vectors_len = file_len(&vectors_path)?;
        let expected_vector_bytes = (num_vectors as u64)
            .checked_mul(dimension as u64 * 4)
            .ok_or_else(|| RetrieveError::FormatError("vector byte count overflow".into()))?;
        if vectors_len < expected_vector_bytes {
            return Err(RetrieveError::FormatError(format!(
                "{} too short: expected at least {} bytes, got {}",
                vectors_path.display(),
                expected_vector_bytes,
                vectors_len
            )));
        }

        let layers_path = dir.join("layers.bin");
        let layers_len = file_len(&layers_path)?;

        // Split one cache budget across the two files in proportion to size,
        // matching the DiskANN searchers.
        let budgets = config.split(&[vectors_len, layers_len]);
        let vectors = CachedFile::new(File::open(&vectors_path)?, budgets[0])?;
        let layers_file = CachedFile::new(File::open(&layers_path)?, budgets[1])?;

        // ── scan layers.bin once for the offsets table ──────────────────────
        let mut scanner = HeaderScanner::new(File::open(&layers_path)?, layers_len);
        let scan = scan_layers(&mut scanner, num_vectors, &layer_assignments)?;
        if scan.base_offsets.len() != num_vectors {
            return Err(RetrieveError::FormatError(format!(
                "layer 0 offsets has {} entries, expected {}",
                scan.base_offsets.len(),
                num_vectors
            )));
        }

        // ── entry point: first node with the highest layer assignment ───────
        let (entry_point, entry_layer) = compute_entry_point(&layer_assignments);

        let mut stats = scan.stats;
        stats.offsets_bytes = scan.base_offsets.len() * 4
            + scan
                .upper_layers
                .iter()
                .map(|layer| layer.node_ids.len() * 4 + layer.offsets.len() * 4)
                .sum::<usize>();

        let dimension_guard = dimension;
        let searcher = Self {
            dimension,
            num_vectors,
            num_layers: scan.num_layers,
            metric,
            auto_normalize: params.auto_normalize,
            built,
            dist_fn,
            base_offsets: scan.base_offsets,
            upper_layers: scan.upper_layers,
            layer_assignments,
            doc_ids,
            tombstones,
            entry_point,
            entry_layer,
            vectors,
            layers_file,
            vector_layout,
            read_buf: vec![0u8; dimension_guard * 4],
            vec_buf: vec![0.0f32; dimension_guard],
            batch_buf: vec![0.0f32; dimension_guard * DISTANCE_BATCH_SIZE],
            neighbor_buf: Vec::with_capacity(32),
            neighbor_bytes: Vec::new(),
            visited_marks: vec![0u8; num_vectors],
            visited_generation: 1,
            open_stats: stats,
        };
        Ok((searcher, stats))
    }

    /// Search for the `k` nearest neighbours of `query`.
    ///
    /// Returns external document ids with their distances, sorted ascending,
    /// exactly like [`HNSWIndex::search`](crate::hnsw::HNSWIndex::search).
    /// Tombstoned documents are traversable but never returned.
    ///
    /// # Errors
    ///
    /// Returns [`RetrieveError::NotBuilt`] when the saved index was not built,
    /// [`RetrieveError::DimensionMismatch`] when `query.len()` differs from the
    /// saved dimension, or [`RetrieveError::EmptyIndex`] when there are no
    /// vectors.
    pub fn search(
        &mut self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::NotBuilt {
                component: "HNSW segment",
                operation: "search",
            });
        }
        if query.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.dimension,
            });
        }
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }
        if k == 0 || self.num_layers == 0 {
            return Ok(Vec::new());
        }

        // Mirror HNSWIndex::search's query preparation: only auto-normalize
        // cosine queries; L2 is used as-is.
        let prepared = self.prepare_query(query);
        let query = prepared.as_ref();

        // `ef` counts candidates; requesting k > ef is structurally unsound.
        let ef = ef.max(k);

        // SeedSelectionStrategy::StackedNSW (the default, and the only strategy
        // the save format's params carry implicitly): descend from the node
        // with the highest layer assignment.
        let mut current = self.entry_point;
        let mut current_dist = self.distance_to(query, current)?;
        for layer_idx in (1..=self.entry_layer).rev() {
            if layer_idx >= self.num_layers {
                continue;
            }
            current = self.descend_layer(query, current, current_dist, layer_idx, ef)?;
            current_dist = self.distance_to(query, current)?;
        }

        let base = self.search_base_layer(query, current, ef)?;
        let mut results = Vec::with_capacity(base.len().min(k));
        for (id, distance) in base {
            if !self.is_live(id) {
                continue;
            }
            if let Some(&doc_id) = self.doc_ids.get(id as usize) {
                results.push((doc_id, distance));
                if results.len() == k {
                    break;
                }
            }
        }
        Ok(results)
    }

    /// Block cache counters summed over `vectors` and `layers.bin`.
    pub fn cache_stats(&self) -> FileCacheStats {
        self.vectors.stats().merge(self.layers_file.stats())
    }

    /// Resident bytes owned by the searcher: offsets table, metadata vectors,
    /// visited marks, read buffers, and the current cache blocks.
    ///
    /// This excludes the file system's page cache. It is the memory the
    /// searcher itself accounts for, and is far below the size of the segment
    /// files for a non-trivial index once the configured cache budget is small
    /// relative to the index.
    pub fn resident_bytes(&self) -> usize {
        self.structural_bytes()
            + self.vectors.stats().resident_bytes
            + self.layers_file.stats().resident_bytes
    }

    /// Resident bytes excluding the block cache.
    ///
    /// This is the graph-independent state the searcher must keep regardless of
    /// cache budget: the offsets table, metadata vectors, visited marks, and
    /// reusable buffers. It is `O(nodes)` and never `O(edges)`.
    pub fn structural_bytes(&self) -> usize {
        let upper: usize = self
            .upper_layers
            .iter()
            .map(|layer| layer.node_ids.len() * 4 + layer.offsets.len() * 4)
            .sum();
        self.base_offsets.len() * 4
            + upper
            + self.layer_assignments.len()
            + self.doc_ids.len() * 4
            + self.tombstones.len()
            + self.read_buf.len()
            + self.vec_buf.len() * 4
            + self.batch_buf.len() * 4
            + self.neighbor_buf.capacity() * 4
            + self.neighbor_bytes.capacity()
            + self.visited_marks.len()
    }

    /// Number of entries in the in-memory offsets table.
    ///
    /// Layer 0 contributes one entry per node; each upper layer contributes
    /// only the nodes that occur in it. This is `O(nodes)`, not `O(edges)`.
    pub fn offsets_table_entries(&self) -> usize {
        self.base_offsets.len()
            + self
                .upper_layers
                .iter()
                .map(|layer| layer.node_ids.len())
                .sum::<usize>()
    }

    /// Diagnostic counters from the open-time `layers.bin` scan.
    pub fn open_stats(&self) -> HNSWSegmentOpenStats {
        self.open_stats
    }

    /// True when no `vectors_aos.bin` sidecar was found and the searcher reads
    /// the dimension-major `vectors.bin` instead. That fallback is correct but
    /// costs up to `dimension` block reads per vector.
    pub fn uses_soa_fallback(&self) -> bool {
        self.vector_layout == VectorLayout::Soa
    }

    /// The metric read from `params.bin`.
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    // ── internals ───────────────────────────────────────────────────────────

    fn prepare_query<'a>(&self, query: &'a [f32]) -> Cow<'a, [f32]> {
        if self.auto_normalize && self.metric == DistanceMetric::Cosine {
            Cow::Owned(normalize(query))
        } else {
            Cow::Borrowed(query)
        }
    }

    #[inline]
    fn is_live(&self, id: u32) -> bool {
        self.tombstones
            .get(id as usize)
            .map(|&flag| flag == 0)
            .unwrap_or(true)
    }

    /// Distance from `query` to the stored vector for internal id `id`.
    fn distance_to(&mut self, query: &[f32], id: u32) -> Result<f32, RetrieveError> {
        // Copy the fn pointer out before the mutable borrow of `read_vector`.
        let dist_fn = self.dist_fn;
        let vector = self.read_vector(id)?;
        Ok(dist_fn(query, vector))
    }

    /// Read one vector into the reusable buffer and return it as a slice.
    fn read_vector(&mut self, id: u32) -> Result<&[f32], RetrieveError> {
        let idx = id as usize;
        if idx >= self.num_vectors {
            return Err(RetrieveError::OutOfBounds(idx));
        }
        match self.vector_layout {
            VectorLayout::Aos => {
                let offset = (idx * self.dimension * 4) as u64;
                self.vectors.read_exact_at(offset, &mut self.read_buf)?;
            }
            VectorLayout::Soa => {
                let n = self.num_vectors;
                for d in 0..self.dimension {
                    let offset = ((d * n + idx) * 4) as u64;
                    self.vectors
                        .read_exact_at(offset, &mut self.read_buf[d * 4..d * 4 + 4])?;
                }
            }
        }
        decode_floats(&self.read_buf, &mut self.vec_buf);
        Ok(&self.vec_buf)
    }

    /// Byte offset of node `node`'s neighbour-list record in layer `layer`.
    ///
    /// Returns `None` for a node that does not occur in an upper layer (its
    /// saved list is empty).
    fn layer_offset(&self, layer: usize, node: u32) -> Option<u32> {
        if layer == 0 {
            self.base_offsets.get(node as usize).copied()
        } else {
            let index = self.upper_layers.get(layer - 1)?;
            match index.node_ids.binary_search(&node) {
                Ok(position) => index.offsets.get(position).copied(),
                Err(_) => None,
            }
        }
    }

    /// Read node `node`'s neighbour ids in `layer` into `out`.
    fn read_neighbors_into(
        &mut self,
        layer: usize,
        node: u32,
        out: &mut Vec<u32>,
    ) -> Result<(), RetrieveError> {
        out.clear();
        let Some(offset) = self.layer_offset(layer, node) else {
            return Ok(());
        };
        let mut header = [0u8; 4];
        self.layers_file.read_exact_at(offset as u64, &mut header)?;
        let count = u32::from_le_bytes(header);
        if count > MAX_NEIGHBORS {
            return Err(RetrieveError::FormatError(format!(
                "unreasonable neighbour count {count} for node {node} in layer {layer}"
            )));
        }
        let bytes = count as usize * 4;
        let mut buffer = std::mem::take(&mut self.neighbor_bytes);
        buffer.resize(bytes, 0);
        self.layers_file
            .read_exact_at(offset as u64 + 4, &mut buffer)?;
        for chunk in buffer.as_chunks::<4>().0.iter() {
            out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        self.neighbor_bytes = buffer;
        Ok(())
    }

    /// One greedy descent step in a single upper layer, mirroring
    /// `HNSWIndex::descend_upper_layers`.
    fn descend_layer(
        &mut self,
        query: &[f32],
        mut current: u32,
        mut current_dist: f32,
        layer: usize,
        ef: usize,
    ) -> Result<u32, RetrieveError> {
        let mut changed = true;
        let mut visited = std::collections::HashSet::with_capacity(ef.min(100));
        while changed {
            changed = false;
            visited.insert(current);
            let mut neighbors = std::mem::take(&mut self.neighbor_buf);
            if let Err(error) = self.read_neighbors_into(layer, current, &mut neighbors) {
                self.neighbor_buf = neighbors;
                return Err(error);
            }
            for &neighbor in neighbors.iter() {
                if visited.contains(&neighbor) {
                    continue;
                }
                let distance = self.distance_to(query, neighbor)?;
                if distance < current_dist {
                    current_dist = distance;
                    current = neighbor;
                    changed = true;
                }
            }
            self.neighbor_buf = neighbors;
        }
        Ok(current)
    }

    /// Base-layer beam search mirroring `greedy_search_layer_live`.
    fn search_base_layer(
        &mut self,
        query: &[f32],
        entry_point: u32,
        ef: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.reset_visited();
        let mut candidates: BinaryHeap<MinCandidate> = BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<MaxResult> = BinaryHeap::with_capacity(ef + 1);

        let entry_distance = self.distance_to(query, entry_point)?;
        candidates.push(MinCandidate {
            id: entry_point,
            distance: entry_distance,
        });
        if self.is_live(entry_point) {
            results.push(MaxResult {
                id: entry_point,
                distance: entry_distance,
            });
        }
        self.mark_visited(entry_point);

        let should_prune = ef >= FRONTIER_PRUNE_MIN_EF;
        let mut pops_since_prune = 0usize;
        while let Some(candidate) = candidates.pop() {
            let worst = results
                .peek()
                .map(|result| result.distance)
                .unwrap_or(f32::INFINITY);
            if candidate.distance > worst && results.len() >= ef {
                break;
            }
            if should_prune {
                pops_since_prune += 1;
            }

            let mut neighbors = std::mem::take(&mut self.neighbor_buf);
            if let Err(error) = self.read_neighbors_into(0, candidate.id, &mut neighbors) {
                self.neighbor_buf = neighbors;
                return Err(error);
            }

            let mut batch_ids = [0u32; DISTANCE_BATCH_SIZE];
            let mut batch_count = 0usize;
            for &neighbor in neighbors.iter() {
                if self.mark_visited(neighbor) {
                    batch_ids[batch_count] = neighbor;
                    batch_count += 1;
                    if batch_count == DISTANCE_BATCH_SIZE {
                        self.flush_batch(
                            query,
                            &batch_ids,
                            batch_count,
                            &mut candidates,
                            &mut results,
                            ef,
                        )?;
                        batch_count = 0;
                    }
                }
            }
            if batch_count > 0 {
                self.flush_batch(
                    query,
                    &batch_ids,
                    batch_count,
                    &mut candidates,
                    &mut results,
                    ef,
                )?;
            }
            self.neighbor_buf = neighbors;

            if should_prune && pops_since_prune >= FRONTIER_PRUNE_INTERVAL {
                prune_unpromising_candidates(&mut candidates, &results, ef);
                pops_since_prune = 0;
            }
        }

        let mut output: Vec<(u32, f32)> = results
            .drain()
            .map(|result| (result.id, result.distance))
            .collect();
        output.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn flush_batch(
        &mut self,
        query: &[f32],
        batch_ids: &[u32; DISTANCE_BATCH_SIZE],
        count: usize,
        candidates: &mut BinaryHeap<MinCandidate>,
        results: &mut BinaryHeap<MaxResult>,
        ef: usize,
    ) -> Result<(), RetrieveError> {
        let dist_fn = self.dist_fn;
        let mut distances = [0.0f32; DISTANCE_BATCH_SIZE];
        for i in 0..count {
            let vector = self.read_vector(batch_ids[i])?;
            distances[i] = dist_fn(query, vector);
        }

        if ef < CACHED_WORST_MIN_EF {
            for i in 0..count {
                if admit_candidate(
                    results,
                    ef,
                    batch_ids[i],
                    distances[i],
                    self.is_live(batch_ids[i]),
                ) {
                    candidates.push(MinCandidate {
                        id: batch_ids[i],
                        distance: distances[i],
                    });
                }
            }
            return Ok(());
        }

        let mut worst_dist = if results.len() < ef {
            f32::INFINITY
        } else {
            results
                .peek()
                .map(|result| result.distance)
                .unwrap_or(f32::INFINITY)
        };
        for i in 0..count {
            if results.len() < ef || distances[i] < worst_dist {
                if admit_candidate(
                    results,
                    ef,
                    batch_ids[i],
                    distances[i],
                    self.is_live(batch_ids[i]),
                ) {
                    candidates.push(MinCandidate {
                        id: batch_ids[i],
                        distance: distances[i],
                    });
                }
                worst_dist = if results.len() < ef {
                    f32::INFINITY
                } else {
                    results
                        .peek()
                        .map(|result| result.distance)
                        .unwrap_or(f32::INFINITY)
                };
            }
        }
        Ok(())
    }

    fn reset_visited(&mut self) {
        if let Some(next) = self.visited_generation.checked_add(1) {
            self.visited_generation = next;
        } else {
            self.visited_marks.fill(0);
            self.visited_generation = 1;
        }
    }

    #[inline]
    fn mark_visited(&mut self, id: u32) -> bool {
        match self.visited_marks.get_mut(id as usize) {
            Some(mark) => {
                if *mark == self.visited_generation {
                    false
                } else {
                    *mark = self.visited_generation;
                    true
                }
            }
            // Mirror VisitedSet::insert: an out-of-bounds id is treated as
            // unvisited; the subsequent vector read reports OutOfBounds.
            None => true,
        }
    }
}

// ── heap helpers mirrored from hnsw::search ──────────────────────────────────

#[inline]
fn insert_result_if_accepted(
    results: &mut BinaryHeap<MaxResult>,
    ef: usize,
    id: u32,
    distance: f32,
) -> bool {
    if results.len() < ef {
        results.push(MaxResult { id, distance });
        return true;
    }
    let Some(mut worst) = results.peek_mut() else {
        return false;
    };
    if distance >= worst.distance {
        return false;
    }
    *worst = MaxResult { id, distance };
    true
}

#[inline]
fn admit_candidate(
    results: &mut BinaryHeap<MaxResult>,
    ef: usize,
    id: u32,
    distance: f32,
    live: bool,
) -> bool {
    if live {
        return insert_result_if_accepted(results, ef, id, distance);
    }
    let Some(worst) = results.peek() else {
        return true;
    };
    !(results.len() >= ef && distance >= worst.distance)
}

#[inline]
fn prune_unpromising_candidates(
    candidates: &mut BinaryHeap<MinCandidate>,
    results: &BinaryHeap<MaxResult>,
    ef: usize,
) {
    if ef < FRONTIER_PRUNE_MIN_EF || results.len() < ef || candidates.len() < ef {
        return;
    }
    let Some(worst) = results.peek() else {
        return;
    };
    let worst_dist = worst.distance;
    candidates.retain(|candidate| candidate.distance <= worst_dist);
}

// ── open-time parsing and scanning ──────────────────────────────────────────

struct ParsedParams {
    metric: DistanceMetric,
    auto_normalize: bool,
}

fn parse_metadata(bytes: &[u8]) -> Result<(usize, usize, bool), RetrieveError> {
    if bytes.len() >= 21 && bytes[..8] == HNSW_SEGMENT_MAGIC {
        let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        if version > FORMAT_VERSION {
            return Err(RetrieveError::FormatError(format!(
                "unsupported format version {version}, expected {FORMAT_VERSION}"
            )));
        }
        let dimension = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        let num_vectors = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize;
        let built = bytes[20] != 0;
        validate_metadata(dimension, num_vectors)?;
        return Ok((dimension, num_vectors, built));
    }

    // Legacy v0: dimension | num_vectors | is_built, no magic or version.
    if bytes.len() >= 9 {
        let dimension = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let num_vectors = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let built = bytes[8] != 0;
        validate_metadata(dimension, num_vectors)?;
        return Ok((dimension, num_vectors, built));
    }

    Err(RetrieveError::FormatError(
        "metadata.bin too short to be an HNSW segment header".into(),
    ))
}

fn validate_metadata(dimension: usize, num_vectors: usize) -> Result<(), RetrieveError> {
    if dimension == 0 || dimension > MAX_DIMENSION {
        return Err(RetrieveError::FormatError(format!(
            "unreasonable dimension {dimension}"
        )));
    }
    if num_vectors > MAX_VECTORS {
        return Err(RetrieveError::FormatError(format!(
            "unreasonable vector count {num_vectors}"
        )));
    }
    Ok(())
}

fn parse_params(bytes: &[u8]) -> Result<ParsedParams, RetrieveError> {
    // A v3 params file starts with a marker; a legacy one starts directly with
    // the `m` field. Without this, a truncated v3 file reads `m_l` as the metric
    // byte and can load as the wrong metric instead of failing.
    const PARAMS_MAGIC: &[u8; 4] = b"HNP3";
    let bytes = if bytes.starts_with(PARAMS_MAGIC) {
        let body = &bytes[PARAMS_MAGIC.len()..];
        if !(26..=27).contains(&body.len()) {
            return Err(RetrieveError::FormatError(format!(
                "params.bin has the v3 marker but {} bytes of body; expected 26 or 27",
                body.len()
            )));
        }
        body
    } else {
        if !(24..=27).contains(&bytes.len()) {
            return Err(RetrieveError::FormatError(format!(
                "params.bin is {} bytes; expected 24-27 (legacy)",
                bytes.len()
            )));
        }
        bytes
    };
    // m(4) m_max(4) m_l(8) ef_construction(4) ef_search(4), then optional
    // metric(1), auto_normalize(1), and compact_upper_layers(1).
    let metric = match bytes.get(24).copied() {
        Some(0) => DistanceMetric::L2,
        Some(1) => DistanceMetric::Cosine,
        Some(2) => DistanceMetric::Angular,
        Some(3) => DistanceMetric::InnerProduct,
        Some(other) => {
            return Err(RetrieveError::FormatError(format!(
                "unknown distance metric byte {other} in params.bin"
            )))
        }
        // Legacy v1 file without the metric byte defaulted to Cosine.
        None => DistanceMetric::Cosine,
    };
    let auto_normalize = bytes.get(25).copied().unwrap_or(0) != 0;
    Ok(ParsedParams {
        metric,
        auto_normalize,
    })
}

struct LayerScan {
    num_layers: usize,
    base_offsets: Vec<u32>,
    upper_layers: Vec<UpperLayerIndex>,
    stats: HNSWSegmentOpenStats,
}

/// Scan `layers.bin` once, reading only neighbour-count headers.
///
/// The saved layout is dense: each layer writes `num_vectors` records indexed
/// by internal node id. A record is `u32 neighbour_count` followed by that many
/// `u32` ids. The scanner reads each count and skips the ids with positional
/// reads, so the bytes read are the headers (`4 * records`), not the edge data.
fn scan_layers(
    scanner: &mut HeaderScanner,
    num_vectors: usize,
    layer_assignments: &[u8],
) -> Result<LayerScan, RetrieveError> {
    let mut stats = HNSWSegmentOpenStats::default();
    let mut position = 0u64;

    let num_layers = scanner.read_u32_at(position)? as usize;
    position += 4;
    if num_layers > MAX_LAYERS {
        return Err(RetrieveError::FormatError(format!(
            "unreasonable layer count {num_layers}"
        )));
    }
    stats.layers = num_layers;

    let mut base_offsets: Vec<u32> = Vec::new();
    let mut upper_layers: Vec<UpperLayerIndex> = Vec::new();

    for layer in 0..num_layers {
        let node_count = scanner.read_u32_at(position)? as usize;
        position += 4;
        if node_count != num_vectors {
            return Err(RetrieveError::FormatError(format!(
                "layer {layer} stores {node_count} neighbour-list records but the index has \
                 {num_vectors} vectors; the file-backed searcher only supports the dense \
                 layer layout (compact_upper_layers segments cannot be addressed by node id)"
            )));
        }

        let mut layer_offsets: Vec<u32> = Vec::with_capacity(num_vectors);
        if layer == 0 {
            base_offsets = Vec::with_capacity(num_vectors);
        } else {
            upper_layers.push(UpperLayerIndex::default());
        }

        let mut node = 0usize;
        while node < num_vectors {
            if layer > 0 && (layer_assignments[node] as usize) < layer {
                // Absent nodes save a zero-count record: four contiguous bytes
                // each. Batch a run of them into one read and verify they are
                // really empty.
                let mut run = 0usize;
                while node + run < num_vectors
                    && (layer_assignments[node + run] as usize) < layer
                    && run < 1024
                {
                    run += 1;
                }
                let bytes = run * 4;
                let mut buffer = vec![0u8; bytes];
                scanner.read_bytes_at(position, &mut buffer)?;
                for (offset, chunk) in buffer.as_chunks::<4>().0.iter().enumerate() {
                    if u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) != 0 {
                        return Err(RetrieveError::FormatError(format!(
                            "node {} is absent from layer {layer} but its saved neighbour \
                             list is non-empty",
                            node + offset
                        )));
                    }
                }
                stats.records += run;
                position += bytes as u64;
                node += run;
                continue;
            }

            let record = position;
            let count = scanner.read_u32_at(position)?;
            if count > MAX_NEIGHBORS {
                return Err(RetrieveError::FormatError(format!(
                    "unreasonable neighbour count {count} for node {node} in layer {layer}"
                )));
            }
            if layer == 0 || layer_assignments[node] as usize >= layer {
                if position > u32::MAX as u64 {
                    return Err(RetrieveError::FormatError(
                        "layers.bin exceeds 4 GiB; u32 offsets cannot address it".into(),
                    ));
                }
                layer_offsets.push(record as u32);
            }
            stats.records += 1;
            stats.edges += count as usize;
            position += 4 + count as u64 * 4;
            node += 1;
        }

        if layer == 0 {
            base_offsets = layer_offsets;
        } else if let Some(upper) = upper_layers.last_mut() {
            // Sparse pairs in ascending node order.
            upper.node_ids = (0..num_vectors)
                .filter(|&node| layer_assignments[node] as usize >= layer)
                .map(|node| node as u32)
                .collect();
            upper.offsets = layer_offsets;
            debug_assert_eq!(upper.node_ids.len(), upper.offsets.len());
        }
    }

    if position != scanner.len {
        return Err(RetrieveError::FormatError(format!(
            "layers.bin has {} trailing bytes after the last layer",
            scanner.len.saturating_sub(position)
        )));
    }

    stats.header_bytes_read = scanner.bytes_read;
    Ok(LayerScan {
        num_layers,
        base_offsets,
        upper_layers,
        stats,
    })
}

fn compute_entry_point(layer_assignments: &[u8]) -> (u32, usize) {
    let mut entry = 0u32;
    let mut layer = 0u8;
    for (index, &assignment) in layer_assignments.iter().enumerate() {
        if assignment > layer {
            layer = assignment;
            entry = index as u32;
        }
    }
    (entry, layer as usize)
}

/// Positional 4-byte header reader used only during open.
///
/// This deliberately bypasses [`CachedFile`]: reading a header through the
/// block cache would pull the neighbour ids in the same block and make open
/// read the whole file. A raw positional read touches exactly the four bytes
/// it needs.
struct HeaderScanner {
    file: File,
    len: u64,
    bytes_read: u64,
}

impl HeaderScanner {
    fn new(file: File, len: u64) -> Self {
        Self {
            file,
            len,
            bytes_read: 0,
        }
    }

    fn read_u32_at(&mut self, position: u64) -> io::Result<u32> {
        if position + 4 > self.len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "layers.bin ended inside a neighbour-list header",
            ));
        }
        let mut buffer = [0u8; 4];
        self.read_bytes_at(position, &mut buffer)?;
        Ok(u32::from_le_bytes(buffer))
    }

    fn read_bytes_at(&mut self, position: u64, buffer: &mut [u8]) -> io::Result<()> {
        let end = position
            .checked_add(buffer.len() as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "offset overflow"))?;
        if end > self.len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "layers.bin ended inside a neighbour-list record",
            ));
        }
        read_exact_at(&self.file, position, buffer)?;
        self.bytes_read += buffer.len() as u64;
        Ok(())
    }
}

#[cfg(unix)]
fn read_exact_at(file: &File, offset: u64, buffer: &mut [u8]) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    let mut read = 0;
    while read < buffer.len() {
        let n = file.read_at(&mut buffer[read..], offset + read as u64)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "failed to fill buffer",
            ));
        }
        read += n;
    }
    Ok(())
}

#[cfg(windows)]
fn read_exact_at(file: &File, offset: u64, buffer: &mut [u8]) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut read = 0;
    while read < buffer.len() {
        let n = file.seek_read(&mut buffer[read..], offset + read as u64)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "failed to fill buffer",
            ));
        }
        read += n;
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn read_exact_at(file: &File, offset: u64, buffer: &mut [u8]) -> io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let mut clone = file.try_clone()?;
    clone.seek(SeekFrom::Start(offset))?;
    clone.read_exact(buffer)
}

// ── small helpers ───────────────────────────────────────────────────────────

fn read_file(path: &Path) -> Result<Vec<u8>, RetrieveError> {
    std::fs::read(path).map_err(|source| RetrieveError::IoAt {
        path: path.to_path_buf(),
        source: std::sync::Arc::new(source),
    })
}

fn file_len(path: &Path) -> Result<u64, RetrieveError> {
    let metadata = std::fs::metadata(path).map_err(|source| RetrieveError::IoAt {
        path: path.to_path_buf(),
        source: std::sync::Arc::new(source),
    })?;
    Ok(metadata.len())
}

fn decode_u32s(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn decode_floats(bytes: &[u8], out: &mut [f32]) {
    for (value, chunk) in out.iter_mut().zip(bytes.as_chunks::<4>().0.iter()) {
        *value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::DistanceMetric;
    use crate::hnsw::{HNSWIndex, HNSWParams};
    use std::io::Write;
    use std::path::Path;
    use std::time::{Duration, Instant};

    /// Deterministic pseudo-random vector in roughly `[-1, 1]`.
    fn lcg_vector(seed: u64, dimension: usize) -> Vec<f32> {
        let mut state = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (0..dimension)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect()
    }

    fn build_index(
        n: usize,
        dimension: usize,
        metric: DistanceMetric,
        auto_normalize: bool,
    ) -> HNSWIndex {
        let params = HNSWParams {
            m: 8,
            m_max: 16,
            ef_construction: 64,
            ef_search: 50,
            auto_normalize,
            metric,
            seed: Some(42),
            ..Default::default()
        };
        let mut index = HNSWIndex::with_params(dimension, params).unwrap();
        for internal in 0..n as u32 {
            index
                .add(internal * 7 + 3, lcg_vector(internal as u64 + 1, dimension))
                .unwrap();
        }
        index.build().unwrap();
        index
    }

    fn metric_byte(metric: DistanceMetric) -> u8 {
        match metric {
            DistanceMetric::L2 => 0,
            DistanceMetric::Cosine => 1,
            DistanceMetric::Angular => 2,
            DistanceMetric::InnerProduct => 3,
        }
    }

    /// Test-only serializer for the segment format.
    ///
    /// The real writer lives behind the `persistence` feature; these tests run
    /// under `--features hnsw` alone, so they write the same bytes directly.
    fn write_segment(root: &Path, index: &HNSWIndex, write_aos: bool) {
        let segment = root.join("segments/segment_hnsw_0");
        std::fs::create_dir_all(&segment).unwrap();

        let mut metadata = Vec::new();
        metadata.extend_from_slice(&HNSW_SEGMENT_MAGIC);
        metadata.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        metadata.extend_from_slice(&(index.dimension as u32).to_le_bytes());
        metadata.extend_from_slice(&(index.num_vectors as u32).to_le_bytes());
        metadata.push(u8::from(index.is_built()));
        std::fs::write(segment.join("metadata.bin"), metadata).unwrap();

        let mut params = Vec::new();
        params.extend_from_slice(&(index.params.m as u32).to_le_bytes());
        params.extend_from_slice(&(index.params.m_max as u32).to_le_bytes());
        params.extend_from_slice(&index.params.m_l.to_le_bytes());
        params.extend_from_slice(&(index.params.ef_construction as u32).to_le_bytes());
        params.extend_from_slice(&(index.params.ef_search as u32).to_le_bytes());
        params.push(metric_byte(index.params.metric));
        params.push(u8::from(index.params.auto_normalize));
        std::fs::write(segment.join("params.bin"), params).unwrap();

        let mut doc_ids = Vec::with_capacity(index.num_vectors * 4);
        for doc_id in &index.doc_ids {
            doc_ids.extend_from_slice(&doc_id.to_le_bytes());
        }
        std::fs::write(segment.join("doc_ids.bin"), doc_ids).unwrap();
        // `tombstone_flags()` is gated behind the `persistence` feature; derive
        // the same bytes from the public deletion query.
        let tombstone_flags: Vec<u8> = (0..index.num_vectors)
            .map(|internal| u8::from(index.is_deleted(index.doc_ids[internal])))
            .collect();
        std::fs::write(segment.join("tombstones.bin"), tombstone_flags).unwrap();
        std::fs::write(
            segment.join("layer_assignments.bin"),
            &index.layer_assignments,
        )
        .unwrap();

        // vectors.bin: dimension-major (SoA), exactly like the real writer.
        let mut soa =
            std::io::BufWriter::new(std::fs::File::create(segment.join("vectors.bin")).unwrap());
        for d in 0..index.dimension {
            for v in 0..index.num_vectors {
                soa.write_all(&index.vectors[v * index.dimension + d].to_le_bytes())
                    .unwrap();
            }
        }
        soa.flush().unwrap();

        if write_aos {
            let mut aos = std::io::BufWriter::new(
                std::fs::File::create(segment.join("vectors_aos.bin")).unwrap(),
            );
            for value in &index.vectors {
                aos.write_all(&value.to_le_bytes()).unwrap();
            }
            aos.flush().unwrap();
        }

        // layers.bin: dense per-layer records indexed by internal node id.
        let mut layers =
            std::io::BufWriter::new(std::fs::File::create(segment.join("layers.bin")).unwrap());
        layers
            .write_all(&(index.layers.len() as u32).to_le_bytes())
            .unwrap();
        for layer in &index.layers {
            layers
                .write_all(&(layer.len() as u32).to_le_bytes())
                .unwrap();
            for node in 0..layer.len() as u32 {
                let neighbors = layer.get_neighbors(node);
                layers
                    .write_all(&(neighbors.len() as u32).to_le_bytes())
                    .unwrap();
                for &neighbor in neighbors.iter() {
                    layers.write_all(&neighbor.to_le_bytes()).unwrap();
                }
            }
        }
        layers.flush().unwrap();
    }

    fn dir_size(path: &Path) -> u64 {
        let mut total = 0u64;
        for entry in std::fs::read_dir(path).unwrap() {
            let entry = entry.unwrap();
            total += entry.metadata().unwrap().len();
        }
        total
    }

    fn p50(mut samples: Vec<Duration>) -> Duration {
        samples.sort_unstable();
        samples[samples.len() / 2]
    }

    #[test]
    fn segment_searcher_matches_in_memory() {
        for (metric, auto_normalize) in
            [(DistanceMetric::L2, false), (DistanceMetric::Cosine, true)]
        {
            let dimension = 16;
            let index = build_index(600, dimension, metric, auto_normalize);
            let dir = tempfile::tempdir().unwrap();
            write_segment(dir.path(), &index, true);
            let segment = dir.path().join("segments/segment_hnsw_0");
            let mut searcher =
                HNSWSegmentSearcher::open(&segment, FileCacheConfig::with_budget(1 << 20)).unwrap();
            assert!(!searcher.uses_soa_fallback());

            let queries: Vec<Vec<f32>> = [0usize, 17, 199, 599]
                .iter()
                .map(|&q| index.vectors[q * dimension..(q + 1) * dimension].to_vec())
                .chain([lcg_vector(9_999, dimension)])
                .collect();

            for query in &queries {
                for (k, ef) in [(10usize, 50usize), (5, 10), (10, 100)] {
                    let expected = index.search(query, k, ef).unwrap();
                    let actual = searcher.search(query, k, ef).unwrap();
                    assert_eq!(
                        actual, expected,
                        "metric {metric:?} query {query:?} k {k} ef {ef}"
                    );
                }
            }
        }
    }

    #[test]
    fn segment_searcher_matches_in_memory_with_tombstones() {
        let dimension = 8;
        let mut index = build_index(300, dimension, DistanceMetric::L2, false);
        let deleted = *index.doc_ids.get(11).unwrap();
        index.delete(deleted).unwrap();

        let dir = tempfile::tempdir().unwrap();
        write_segment(dir.path(), &index, true);
        let segment = dir.path().join("segments/segment_hnsw_0");
        let mut searcher =
            HNSWSegmentSearcher::open(&segment, FileCacheConfig::with_budget(1 << 20)).unwrap();

        for q in [0usize, 11, 42, 299] {
            let query = index.vectors[q * dimension..(q + 1) * dimension].to_vec();
            let expected = index.search(&query, 10, 50).unwrap();
            let actual = searcher.search(&query, 10, 50).unwrap();
            assert_eq!(actual, expected);
            assert!(actual.iter().all(|&(doc_id, _)| doc_id != deleted));
        }
    }

    #[cfg(feature = "compact-hnsw")]
    #[test]
    fn compacted_upper_layers_are_readable_from_disk() {
        // Compaction changes the resident layout (CSR over eligible nodes) but
        // the writer still emits one record per internal node id, with an empty
        // list for nodes absent from the layer. The searcher treats such a
        // segment like any other, so an index can be compact in memory and still
        // be served from disk.
        let (n, dimension) = (400usize, 12usize);
        let params = HNSWParams {
            m: 8,
            m_max: 16,
            ef_construction: 64,
            ef_search: 50,
            metric: DistanceMetric::L2,
            seed: Some(42),
            compact_upper_layers: true,
            ..HNSWParams::default()
        };
        let mut index = HNSWIndex::with_params(dimension, params).unwrap();
        for internal in 0..n as u32 {
            index
                .add(internal * 7 + 3, lcg_vector(internal as u64 + 1, dimension))
                .unwrap();
        }
        index.build().unwrap();

        let dir = tempfile::tempdir().unwrap();
        write_segment(dir.path(), &index, true);
        let segment = dir.path().join("segments/segment_hnsw_0");
        let mut searcher =
            HNSWSegmentSearcher::open(&segment, FileCacheConfig::with_budget(1 << 20)).unwrap();
        assert!(!searcher.uses_soa_fallback());

        for q in [0usize, 37, 200, 399] {
            let query = index.vectors[q * dimension..(q + 1) * dimension].to_vec();
            let expected = index.search(&query, 10, 50).unwrap();
            let actual = searcher.search(&query, 10, 50).unwrap();
            assert_eq!(actual, expected, "compacted segment query {q} disagreed");
        }
    }

    #[test]
    fn soa_fallback_still_answers_queries() {
        let dimension = 12;
        let index = build_index(200, dimension, DistanceMetric::L2, false);
        let dir = tempfile::tempdir().unwrap();
        // Legacy segment: no vectors_aos.bin sidecar.
        write_segment(dir.path(), &index, false);
        let segment = dir.path().join("segments/segment_hnsw_0");
        let mut searcher =
            HNSWSegmentSearcher::open(&segment, FileCacheConfig::with_budget(1 << 20)).unwrap();
        assert!(searcher.uses_soa_fallback());

        let query = index.vectors[..dimension].to_vec();
        let expected = index.search(&query, 10, 50).unwrap();
        let actual = searcher.search(&query, 10, 50).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn unsupported_metric_is_rejected_at_open() {
        let mut index = build_index(64, 8, DistanceMetric::L2, false);
        index.params.metric = DistanceMetric::Angular;
        let dir = tempfile::tempdir().unwrap();
        write_segment(dir.path(), &index, true);
        let segment = dir.path().join("segments/segment_hnsw_0");
        let error = HNSWSegmentSearcher::open(&segment, FileCacheConfig::default())
            .err()
            .expect("Angular must be rejected");
        assert!(
            matches!(error, RetrieveError::Unsupported { .. }),
            "expected Unsupported, got {error:?}"
        );
    }

    #[test]
    fn truncated_v3_params_is_rejected_at_open() {
        // A v3 params file is the marker plus a 26- or 27-byte body. Truncated to
        // marker + 24 bytes it used to pass the "at least 24 bytes" check and read
        // the metric from offset 24, which in a marker-prefixed file is the top
        // byte of `m_l`. The metric is a small number there, so the file loaded as
        // an index of the wrong metric instead of failing.
        let index = build_index(64, 8, DistanceMetric::L2, false);
        let dir = tempfile::tempdir().unwrap();
        write_segment(dir.path(), &index, true);
        let segment = dir.path().join("segments/segment_hnsw_0");

        let mut params = Vec::new();
        params.extend_from_slice(b"HNP3");
        params.extend_from_slice(&(index.params.m as u32).to_le_bytes());
        params.extend_from_slice(&(index.params.m_max as u32).to_le_bytes());
        params.extend_from_slice(&index.params.m_l.to_le_bytes());
        std::fs::write(segment.join("params.bin"), &params).unwrap();

        let error = HNSWSegmentSearcher::open(&segment, FileCacheConfig::default())
            .err()
            .expect("a v3 marker with a 24-byte body must be rejected");
        assert!(
            matches!(error, RetrieveError::FormatError(ref message)
                if message.contains("v3 marker") && message.contains("expected 26 or 27")),
            "expected FormatError naming the v3 body length, got {error:?}"
        );
    }

    #[test]
    fn v3_params_with_marker_parses_the_metric_after_the_body_prefix() {
        // Cosine with auto_normalize: the builder normalizes each row, so the
        // stored metric byte 1 is honest and the index builds without the
        // L2-normalization panic that un-normalized cosine rows raise.
        let index = build_index(64, 8, DistanceMetric::Cosine, true);
        let dir = tempfile::tempdir().unwrap();
        write_segment(dir.path(), &index, true);
        let segment = dir.path().join("segments/segment_hnsw_0");

        // Marker + 26-byte body: the metric byte sits at body offset 24, which is
        // file offset 28. Reading offset 24 would take the top byte of `m_l`.
        let mut params = Vec::new();
        params.extend_from_slice(b"HNP3");
        params.extend_from_slice(&(index.params.m as u32).to_le_bytes());
        params.extend_from_slice(&(index.params.m_max as u32).to_le_bytes());
        params.extend_from_slice(&index.params.m_l.to_le_bytes());
        params.extend_from_slice(&(index.params.ef_construction as u32).to_le_bytes());
        params.extend_from_slice(&(index.params.ef_search as u32).to_le_bytes());
        params.push(1); // Cosine
        params.push(1); // auto_normalize
        std::fs::write(segment.join("params.bin"), &params).unwrap();

        let searcher = HNSWSegmentSearcher::open(&segment, FileCacheConfig::default())
            .expect("marker + 26-byte body must parse");
        assert_eq!(searcher.metric(), DistanceMetric::Cosine);
    }

    #[test]
    fn searcher_resident_state_is_smaller_than_files() {
        let dimension = 16;
        let index = build_index(4096, dimension, DistanceMetric::L2, false);
        let dir = tempfile::tempdir().unwrap();
        write_segment(dir.path(), &index, true);
        let segment = dir.path().join("segments/segment_hnsw_0");

        let config = FileCacheConfig::with_budget(8 << 10);
        let (mut searcher, stats) = HNSWSegmentSearcher::open_with_stats(&segment, config).unwrap();

        // Run one search so the cache holds its steady-state blocks.
        let query = index.vectors[..dimension].to_vec();
        searcher.search(&query, 10, 50).unwrap();

        let file_bytes = dir_size(&segment) as usize;
        let resident = searcher.resident_bytes();
        assert!(
            resident * 4 < file_bytes,
            "resident {resident} bytes should be well below {file_bytes} bytes of segment files"
        );

        // Offsets are O(nodes), not O(edges).
        assert!(
            searcher.offsets_table_entries() < stats.edges,
            "offsets table has {} entries, edges {}",
            searcher.offsets_table_entries(),
            stats.edges
        );
        let max_nodes = index.num_vectors + index.num_vectors / 2;
        assert!(
            searcher.offsets_table_entries() <= max_nodes,
            "offsets table {} entries exceeds node count budget {}",
            searcher.offsets_table_entries(),
            max_nodes
        );

        // Open read only headers, not the edge data.
        let layers_len = std::fs::metadata(segment.join("layers.bin")).unwrap().len();
        assert!(
            stats.header_bytes_read < layers_len,
            "open read {} header bytes of a {} byte layers.bin",
            stats.header_bytes_read,
            layers_len
        );
        assert_eq!(stats.offsets_bytes, searcher.open_stats().offsets_bytes);
        assert_eq!(stats.layers, index.layers.len());
    }

    /// End-to-end latency at a size where the layout matters.
    ///
    /// Ignored by default: it builds a 100k index (tens of seconds in release)
    /// and is a measurement, not a gate.
    #[test]
    #[ignore = "measurement: builds a 100k index and times 200 queries"]
    fn segment_searcher_100k_latency() {
        use std::time::Instant;

        let (n, dimension, queries, k, ef) = (100_000usize, 96usize, 200usize, 10usize, 50usize);
        let index = build_index(n, dimension, DistanceMetric::L2, false);
        let dir = tempfile::tempdir().unwrap();
        write_segment(dir.path(), &index, true);
        let segment = dir.path().join("segments/segment_hnsw_0");
        let budget_mib: usize = std::env::var("SEG_CACHE_MIB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        let mut searcher =
            HNSWSegmentSearcher::open(&segment, FileCacheConfig::with_budget(budget_mib << 20))
                .unwrap();
        assert!(
            !searcher.uses_soa_fallback(),
            "this run measures the AoS path"
        );
        assert_eq!(
            searcher
                .search(&lcg_vector(7, dimension), k, ef)
                .unwrap()
                .len(),
            k
        );

        let probes: Vec<Vec<f32>> = (0..queries)
            .map(|q| lcg_vector(10_000_000 + q as u64, dimension))
            .collect();

        let mut resident = Vec::with_capacity(queries);
        for q in &probes {
            let t = Instant::now();
            let got = index.search(q, k, ef).unwrap();
            resident.push(t.elapsed().as_micros() as u64);
            assert_eq!(got.len(), k);
        }

        let mut file_backed = Vec::with_capacity(queries);
        for q in &probes {
            let t = Instant::now();
            let got = searcher.search(q, k, ef).unwrap();
            file_backed.push(t.elapsed().as_micros() as u64);
            assert_eq!(got.len(), k);
        }

        let p50 = |v: &mut Vec<u64>| -> u64 {
            v.sort_unstable();
            v[v.len() / 2]
        };
        let (r, f) = (p50(&mut resident), p50(&mut file_backed));

        let segment_bytes: u64 = [
            "layers.bin",
            "vectors.bin",
            "vectors_aos.bin",
            "doc_ids.bin",
        ]
        .iter()
        .filter_map(|name| std::fs::metadata(segment.join(name)).ok())
        .map(|m| m.len())
        .sum();

        eprintln!(
            "100k x {dimension}, k={k}, ef={ef}, {queries} queries, {budget_mib} MiB cache: \
             resident p50 {r} us, file-backed p50 {f} us ({:.1}x), \
             hit rate {:.2}, structural {} B, offsets {} entries, segment {} MiB",
            f as f64 / r.max(1) as f64,
            searcher.cache_stats().hit_rate(),
            searcher.structural_bytes(),
            searcher.offsets_table_entries(),
            segment_bytes / (1 << 20)
        );
    }

    /// Micro-benchmark: read one vector from a dimension-major file versus an
    /// AoS file through the same block cache.
    ///
    /// Run with:
    /// `cargo test --features hnsw --lib soa_vs_aos -- --ignored --nocapture`
    #[test]
    #[ignore = "measurement; run with --ignored --nocapture"]
    fn soa_vs_aos_single_vector_reads() {
        let n = 100_000usize;
        let dimension = 96usize;
        let reads = 2_000usize;
        let dir = tempfile::tempdir().unwrap();

        let soa_path = dir.path().join("soa.bin");
        let aos_path = dir.path().join("aos.bin");
        let vector =
            |v: usize| -> Vec<f32> { (0..dimension).map(|d| (v * dimension + d) as f32).collect() };
        {
            let mut soa = std::io::BufWriter::new(std::fs::File::create(&soa_path).unwrap());
            for d in 0..dimension {
                for v in 0..n {
                    soa.write_all(&vector(v)[d].to_le_bytes()).unwrap();
                }
            }
            soa.flush().unwrap();
            let mut aos = std::io::BufWriter::new(std::fs::File::create(&aos_path).unwrap());
            for v in 0..n {
                for value in vector(v) {
                    aos.write_all(&value.to_le_bytes()).unwrap();
                }
            }
            aos.flush().unwrap();
        }

        let config = FileCacheConfig::with_budget(8 << 20);
        let mut soa = CachedFile::new(std::fs::File::open(&soa_path).unwrap(), config).unwrap();
        let mut aos = CachedFile::new(std::fs::File::open(&aos_path).unwrap(), config).unwrap();
        let ids: Vec<usize> = {
            let mut state = 12345u64;
            (0..reads)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1);
                    (state as usize) % n
                })
                .collect()
        };

        let mut buf = vec![0u8; dimension * 4];
        let soa_start = Instant::now();
        for &id in &ids {
            for d in 0..dimension {
                let offset = ((d * n + id) * 4) as u64;
                soa.read_exact_at(offset, &mut buf[d * 4..d * 4 + 4])
                    .unwrap();
            }
        }
        let soa_elapsed = soa_start.elapsed();
        let soa_stats = soa.stats();

        let aos_start = Instant::now();
        for &id in &ids {
            let offset = (id * dimension * 4) as u64;
            aos.read_exact_at(offset, &mut buf).unwrap();
        }
        let aos_elapsed = aos_start.elapsed();
        let aos_stats = aos.stats();

        println!(
            "read_one_vector n={n} dim={dimension} reads={reads} cache=8MiB\n\
             soa: {:?} total ({:?}/vector), miss rate {:.2}\n\
             aos: {:?} total ({:?}/vector), miss rate {:.2}",
            soa_elapsed,
            soa_elapsed / reads as u32,
            1.0 - soa_stats.hit_rate(),
            aos_elapsed,
            aos_elapsed / reads as u32,
            1.0 - aos_stats.hit_rate()
        );
    }

    /// Synthetic p50 latency: in-memory index versus file-backed searcher at the
    /// same `k` and `ef`, plus the searcher's cache hit rate.
    ///
    /// Run with:
    /// `cargo test --features hnsw --lib resident_vs_file_backed_p50 -- --ignored --nocapture`
    #[test]
    #[ignore = "measurement; run with --ignored --nocapture"]
    fn resident_vs_file_backed_p50() {
        let n = 20_000usize;
        let dimension = 32usize;
        let index = build_index(n, dimension, DistanceMetric::L2, false);
        let dir = tempfile::tempdir().unwrap();
        write_segment(dir.path(), &index, true);
        let segment = dir.path().join("segments/segment_hnsw_0");
        let file_bytes = dir_size(&segment);

        let mut searcher =
            HNSWSegmentSearcher::open(&segment, FileCacheConfig::with_budget(2 << 20)).unwrap();

        let queries: Vec<Vec<f32>> = (0..200)
            .map(|i| lcg_vector(90_000 + i, dimension))
            .collect();

        // Warm up both paths.
        for query in &queries {
            let _ = index.search(query, 10, 50).unwrap();
            let _ = searcher.search(query, 10, 50).unwrap();
        }

        let mut resident = Vec::with_capacity(queries.len());
        let mut file_backed = Vec::with_capacity(queries.len());
        for query in &queries {
            let start = Instant::now();
            let _ = index.search(query, 10, 50).unwrap();
            resident.push(start.elapsed());

            let start = Instant::now();
            let _ = searcher.search(query, 10, 50).unwrap();
            file_backed.push(start.elapsed());
        }

        let stats = searcher.cache_stats();
        let open = searcher.open_stats();
        println!(
            "p50 k=10 ef=50 n={n} dim={dimension}\n\
             resident:    {:?}\n\
             file-backed: {:?}\n\
             cache hit rate {:.2}, resident {} bytes (structural {}, cache {}), segment files {} bytes\n\
             offsets: {} entries / {} bytes, open read {} header bytes of {} layer bytes ({} layers, {} edges)",
            p50(resident),
            p50(file_backed),
            stats.hit_rate(),
            searcher.resident_bytes(),
            searcher.structural_bytes(),
            stats.resident_bytes,
            file_bytes,
            searcher.offsets_table_entries(),
            open.offsets_bytes,
            open.header_bytes_read,
            std::fs::metadata(segment.join("layers.bin")).unwrap().len(),
            open.layers,
            open.edges,
        );
    }
}
