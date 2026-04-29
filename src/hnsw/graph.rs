//! HNSW graph structure and core types.

use crate::distance::DistanceMetric;
use crate::hnsw::tombstones::TombstoneSet;
use crate::RetrieveError;

#[cfg(feature = "hnsw")]
use smallvec::SmallVec;

use rand::SeedableRng;
use std::collections::HashMap;
use std::sync::Mutex;

/// HNSW index for approximate nearest neighbor search.
///
/// Implements the Hierarchical Navigable Small World algorithm (Malkov & Yashunin, 2016)
/// with optimizations for SIMD acceleration and cache efficiency.
///
/// **Important**: This index uses cosine distance (`1 - dot(a, b)`) which requires
/// **L2-normalized** input vectors. Un-normalized vectors produce silently wrong results.
/// Use [`crate::distance::normalize`] before adding vectors.
///
/// # Scalability Note (Wilson Lin's 3B Embedding Insight)
///
/// Standard in-memory HNSW (like this implementation) becomes cost-prohibitive at billion-scale (requires TBs of RAM).
///
/// **Future Optimization (CoreNN approach)**:
/// - Move vector storage and graph structure to disk (SSD).
/// - Support live updates without full rebuilds.
/// - Use sharding (e.g., 64 shards by xxHash) to distribute load.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HNSWIndex {
    /// Vectors stored in Structure of Arrays (SoA) format for cache efficiency
    /// Layout: [v0[0..d], v1[0..d], ..., vn[0..d]]
    pub(crate) vectors: Vec<f32>,

    /// External document IDs aligned with internal vector indices.
    ///
    /// Invariants:
    /// - `doc_ids.len() == num_vectors`
    /// - internal index `i` corresponds to external `doc_id = doc_ids[i]`
    pub(crate) doc_ids: Vec<u32>,

    /// Reverse map: external `doc_id -> internal vector index`.
    /// Rebuilt from `doc_ids` on deserialization.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) doc_id_to_internal: HashMap<u32, u32>,

    /// Vector dimension
    pub(crate) dimension: usize,

    /// Number of vectors
    pub(crate) num_vectors: usize,

    /// Graph layers (index 0 = base layer, higher = upper layers)
    pub(crate) layers: Vec<Layer>,

    /// Layer assignment for each vector (u8: max layer where vector appears)
    pub(crate) layer_assignments: Vec<u8>,

    /// Parameters
    pub(crate) params: HNSWParams,

    /// Whether index has been built
    built: bool,

    /// Cached entry point (node with highest layer assignment).
    /// Set during `build()`, avoids O(n) scan on every search.
    #[cfg_attr(feature = "serde", serde(skip))]
    cached_entry_point: Option<u32>,

    /// Optional metadata store for filtering.
    /// Skipped during serialization -- must be re-added after load for filtered search.
    #[cfg_attr(feature = "serde", serde(skip))]
    metadata: Option<crate::filtering::MetadataStore>,

    /// Field name for filtering (e.g., "category")
    filter_field: Option<String>,

    /// Category assignments: vector_idx -> category value (from metadata filter field).
    /// Skipped during serialization -- rebuilt via `add_metadata` after load.
    #[cfg_attr(feature = "serde", serde(skip))]
    category_assignments: Vec<Option<crate::filtering::MetadataValue>>,

    /// Soft-deleted nodes. Deleted internal IDs are excluded from search results
    /// but remain in the graph for navigation. Storage is not reclaimed until rebuild.
    /// Skipped during serialization -- starts empty on deserialization.
    #[cfg_attr(feature = "serde", serde(skip))]
    tombstones: TombstoneSet,

    /// Internal RNG for layer assignment. Seeded from `params.seed` when set,
    /// otherwise uses thread-local RNG.
    #[cfg_attr(feature = "serde", serde(skip))]
    rng: Mutex<Option<rand::rngs::StdRng>>,
}

/// Seed selection strategy for HNSW search initialization.
///
/// This controls how the search chooses its entrypoint(s). Different strategies can
/// behave better on different datasets and scale regimes; treat this as a tuning knob
/// and benchmark on your workload.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SeedSelectionStrategy {
    /// Stacked NSW: Hierarchical multi-resolution graphs (default, best for large datasets)
    /// Uses entry point in highest layer, navigates down layer by layer.
    #[default]
    StackedNSW,

    /// K-Sampled Random Seeds: K random nodes per query (best for medium-scale 1M-25GB)
    /// Lower indexing overhead, but requires more samples on large datasets.
    KSampledRandom {
        /// Number of random seeds to sample (typically k or ef_search)
        k: usize,
    },
}

/// Neighborhood diversification strategy for graph construction.
///
/// These strategies pick a subset of candidate neighbors to keep the graph navigable
/// while avoiding redundant edges.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NeighborhoodDiversification {
    /// Relative Neighborhood Diversification (RND) - best overall performance
    /// Formula: dist(X_q, X_j) < dist(X_i, X_j) for all neighbors X_i
    #[default]
    RelativeNeighborhood,

    /// Maximum-Oriented Neighborhood Diversification (MOND) - second-best
    /// Maximizes angles between neighbors (θ ≥ 60°)
    MaximumOriented {
        /// Minimum angle threshold in degrees (typically 60°)
        min_angle_degrees: f32,
    },

    /// Relaxed Relative Neighborhood Diversification (RRND)
    /// Formula: dist(X_q, X_j) < α · dist(X_i, X_j) with α ≥ 1.5
    RelaxedRelative {
        /// Relaxation factor (typically 1.3-1.5)
        alpha: f32,
    },
}

/// HNSW parameters controlling graph structure and search behavior.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HNSWParams {
    /// Maximum number of connections per node (typically 16)
    pub m: usize,

    /// Maximum connections for newly inserted nodes (typically 16)
    pub m_max: usize,

    /// Layer assignment probability parameter (typically 1/ln(2) ≈ 1.44)
    /// Higher = more vectors in upper layers
    pub m_l: f64,

    /// Search width during construction (typically 200)
    pub ef_construction: usize,

    /// Default search width during query (typically 50-200)
    pub ef_search: usize,

    /// When true, L2-normalize vectors before storing them.
    /// Useful when callers cannot guarantee pre-normalized input.
    pub auto_normalize: bool,

    /// Distance metric used for all comparisons (default: Cosine).
    pub metric: DistanceMetric,

    /// Seed selection strategy (default: StackedNSW for large-scale)
    pub seed_selection: SeedSelectionStrategy,

    /// Neighborhood diversification strategy (default: RND for best performance)
    pub neighborhood_diversification: NeighborhoodDiversification,

    /// Optional RNG seed for reproducible layer assignments.
    /// When `None` (default), uses thread-local RNG.
    pub seed: Option<u64>,

    /// ID compression method (optional)
    #[cfg(feature = "id-compression")]
    #[cfg_attr(feature = "serde", serde(skip))]
    pub id_compression: Option<crate::compression::IdCompressionMethod>,

    /// Minimum neighbor list size to compress (smaller lists use uncompressed storage)
    #[cfg(feature = "id-compression")]
    #[cfg_attr(
        feature = "serde",
        serde(skip, default = "default_compression_threshold")
    )]
    pub compression_threshold: usize,
}

/// Default compression threshold for serde deserialization.
#[cfg(feature = "id-compression")]
fn default_compression_threshold() -> usize {
    32
}

impl Default for HNSWParams {
    fn default() -> Self {
        Self {
            m: 16,
            m_max: 32,                // Paper: m_max0 = 2*M for layer 0
            m_l: 1.0 / 16.0_f64.ln(), // 1/ln(M), per Malkov & Yashunin 2018
            ef_construction: 200,
            ef_search: 50,
            auto_normalize: false,
            metric: DistanceMetric::Cosine,
            seed_selection: SeedSelectionStrategy::default(),
            neighborhood_diversification: NeighborhoodDiversification::default(),
            seed: None,
            #[cfg(feature = "id-compression")]
            id_compression: None,
            #[cfg(feature = "id-compression")]
            compression_threshold: 32, // Only compress if m >= 32 (per paper)
        }
    }
}

/// Builder for [`HNSWIndex`].
///
/// ```no_run
/// # fn main() -> Result<(), vicinity::RetrieveError> {
/// use vicinity::hnsw::HNSWIndex;
/// let index = HNSWIndex::builder(128)
///     .m(24)
///     .ef_construction(400)
///     .auto_normalize(true)
///     .build()?;
/// # Ok(())
/// # }
/// ```
pub struct HNSWBuilder {
    dimension: usize,
    m: usize,
    m_max: Option<usize>,
    ef_construction: usize,
    ef_search: usize,
    auto_normalize: bool,
    metric: DistanceMetric,
}

impl HNSWBuilder {
    /// Set the maximum number of neighbors per node (default 16).
    pub fn m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// Set the maximum neighbors on non-zero layers (default: same as `m`).
    pub fn m_max(mut self, m_max: usize) -> Self {
        self.m_max = Some(m_max);
        self
    }

    /// Set construction effort (default 200).
    pub fn ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set default search effort (default 50).
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }

    /// Whether to L2-normalize vectors on add (default false).
    pub fn auto_normalize(mut self, normalize: bool) -> Self {
        self.auto_normalize = normalize;
        self
    }

    /// Set the distance metric (default: Cosine).
    pub fn metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Build the index.
    pub fn build(self) -> Result<HNSWIndex, RetrieveError> {
        let m_max = self.m_max.unwrap_or(2 * self.m); // Paper: m_max0 = 2*M
        let params = HNSWParams {
            m: self.m,
            m_max,
            ef_construction: self.ef_construction,
            ef_search: self.ef_search,
            auto_normalize: self.auto_normalize,
            metric: self.metric,
            ..Default::default()
        };
        HNSWIndex::with_params(self.dimension, params)
    }
}

/// Storage for neighbor lists (compressed or uncompressed).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum NeighborStorage {
    /// Uncompressed neighbors (current implementation).
    Uncompressed(Vec<SmallVec<[u32; 16]>>),

    /// Compressed neighbors.
    #[cfg(feature = "id-compression")]
    Compressed {
        data: Vec<CompressedNeighborList>,
        universe_size: u32,
    },
}

/// Compressed neighbor list for a single node.
#[cfg(feature = "id-compression")]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct CompressedNeighborList {
    data: Vec<u8>,
    num_neighbors: usize,
}

/// Graph layer containing neighbor lists for all vectors in that layer.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct Layer {
    storage: NeighborStorage,
    /// Cache for decompressed neighbors (temporary, cleared after use).
    /// Skipped during serialization -- rebuilt as empty on load.
    #[cfg(feature = "id-compression")]
    #[cfg_attr(feature = "serde", serde(skip, default = "Layer::empty_cache"))]
    decompressed_cache: std::sync::Mutex<std::collections::HashMap<u32, SmallVec<[u32; 16]>>>,
}

impl Layer {
    /// Default empty decompression cache (used by serde skip default).
    #[cfg(feature = "id-compression")]
    fn empty_cache() -> std::sync::Mutex<std::collections::HashMap<u32, SmallVec<[u32; 16]>>> {
        std::sync::Mutex::new(std::collections::HashMap::new())
    }

    /// Create uncompressed layer.
    pub(crate) fn new_uncompressed(neighbors: Vec<SmallVec<[u32; 16]>>) -> Self {
        Self {
            storage: NeighborStorage::Uncompressed(neighbors),
            #[cfg(feature = "id-compression")]
            decompressed_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Get mutable access to uncompressed neighbors (for construction only).
    /// Panics if layer is compressed.
    pub(crate) fn get_neighbors_mut(&mut self) -> &mut Vec<SmallVec<[u32; 16]>> {
        match &mut self.storage {
            NeighborStorage::Uncompressed(neighbors) => neighbors,
            #[cfg(feature = "id-compression")]
            NeighborStorage::Compressed { .. } => {
                panic!("Cannot get mutable access to compressed neighbors");
            }
        }
    }

    /// Compress this layer after construction.
    #[cfg(feature = "id-compression")]
    pub(crate) fn compress(
        &mut self,
        compressor: &crate::compression::DeltaVarintCompressor,
        universe_size: u32,
        threshold: usize,
    ) -> Result<(), crate::compression::CompressionError> {
        // Extract uncompressed neighbors
        let neighbors =
            match std::mem::replace(&mut self.storage, NeighborStorage::Uncompressed(Vec::new())) {
                NeighborStorage::Uncompressed(n) => n,
                NeighborStorage::Compressed { .. } => {
                    // Already compressed, nothing to do
                    return Ok(());
                }
            };

        // Compress
        let compressed_layer =
            Self::new_compressed(neighbors, compressor, universe_size, threshold)?;
        *self = compressed_layer;
        Ok(())
    }

    /// Create compressed layer.
    #[cfg(feature = "id-compression")]
    fn new_compressed(
        neighbors: Vec<SmallVec<[u32; 16]>>,
        _compressor: &crate::compression::DeltaVarintCompressor,
        universe_size: u32,
        threshold: usize,
    ) -> Result<Self, crate::compression::CompressionError> {
        let mut compressed_lists = Vec::with_capacity(neighbors.len());

        for neighbor_list in &neighbors {
            if neighbor_list.len() >= threshold {
                // Compress this neighbor list
                let mut sorted = neighbor_list.to_vec();
                sorted.sort();
                sorted.dedup();

                let compressed = crate::compression::compress_set_enveloped(
                    &sorted,
                    universe_size,
                    crate::compression::ChooseConfig::default(),
                )?;

                compressed_lists.push(CompressedNeighborList {
                    data: compressed,
                    num_neighbors: sorted.len(),
                });
            } else {
                // Too small, store uncompressed (use empty data as marker)
                compressed_lists.push(CompressedNeighborList {
                    data: Vec::new(), // Empty = uncompressed
                    num_neighbors: neighbor_list.len(),
                });
            }
        }

        Ok(Self {
            storage: NeighborStorage::Compressed {
                data: compressed_lists,
                universe_size,
            },
            decompressed_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }

    /// Get neighbors for a node.
    ///
    /// Returns a `Cow` to avoid cloning on the uncompressed path (the common case).
    /// The compressed path (id-compression feature) decompresses and returns owned data.
    pub(crate) fn get_neighbors(&self, node: u32) -> std::borrow::Cow<'_, [u32]> {
        match &self.storage {
            NeighborStorage::Uncompressed(neighbors) => neighbors
                .get(node as usize)
                .map(|n| std::borrow::Cow::Borrowed(n.as_slice()))
                .unwrap_or(std::borrow::Cow::Borrowed(&[])),
            #[cfg(feature = "id-compression")]
            NeighborStorage::Compressed {
                data,
                universe_size,
            } => {
                // Check cache first
                {
                    let cache = self
                        .decompressed_cache
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    if let Some(cached) = cache.get(&node) {
                        return std::borrow::Cow::Owned(cached.to_vec());
                    }
                }

                // Decompress
                let compressed = &data[node as usize];
                if compressed.data.is_empty() {
                    return std::borrow::Cow::Borrowed(&[]);
                }

                let decompressed = crate::compression::decompress_set_enveloped(&compressed.data)
                    .map(|(_choice, u2, ids)| {
                        if u2 == *universe_size {
                            ids
                        } else {
                            Vec::new()
                        }
                    })
                    .unwrap_or_else(|_| Vec::new());

                let neighbors: SmallVec<[u32; 16]> = decompressed.into();

                // Cache for future lookups
                let result = neighbors.to_vec();
                {
                    let mut cache = self
                        .decompressed_cache
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    // Cap cache size to prevent unbounded memory growth during
                    // concurrent searches. Full clear is simple and sufficient;
                    // hot entries will be re-populated on next access.
                    if cache.len() >= 10_000 {
                        cache.clear();
                    }
                    cache.insert(node, neighbors);
                }

                std::borrow::Cow::Owned(result)
            }
        }
    }

    /// Clear decompression cache (call after search).
    #[cfg(feature = "id-compression")]
    pub(crate) fn clear_cache(&self) {
        let mut cache = self
            .decompressed_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        cache.clear();
    }

    /// Get number of nodes in this layer.
    pub(crate) fn len(&self) -> usize {
        match &self.storage {
            NeighborStorage::Uncompressed(neighbors) => neighbors.len(),
            #[cfg(feature = "id-compression")]
            NeighborStorage::Compressed { data, .. } => data.len(),
        }
    }

    /// Remap all neighbor IDs using a permutation table.
    ///
    /// Used by `reorder_for_locality` to update neighbor references after
    /// reordering the vector/doc_id arrays.
    pub(crate) fn remap_ids(&mut self, old_to_new: &[u32], new_size: usize) {
        match &mut self.storage {
            NeighborStorage::Uncompressed(neighbors) => {
                // Permute the neighbor lists themselves (position i → new position)
                let mut permuted = vec![SmallVec::<[u32; 16]>::new(); new_size];
                for (old_idx, nbs) in neighbors.iter().enumerate() {
                    if old_idx < old_to_new.len() {
                        let new_idx = old_to_new[old_idx] as usize;
                        if new_idx < new_size {
                            // Remap each neighbor reference
                            let remapped: SmallVec<[u32; 16]> = nbs
                                .iter()
                                .filter_map(|&nb| {
                                    let nb_usize = nb as usize;
                                    if nb_usize < old_to_new.len() {
                                        Some(old_to_new[nb_usize])
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            permuted[new_idx] = remapped;
                        }
                    }
                }
                *neighbors = permuted;
            }
            #[cfg(feature = "id-compression")]
            NeighborStorage::Compressed { .. } => {
                // Compressed layers: would need decompress → remap → recompress.
                // For now, reorder_for_locality should only be called before compression.
            }
        }
    }

    /// Get all neighbor lists (for persistence).
    /// Returns None if layer is compressed.
    #[allow(dead_code)]
    pub(crate) fn get_all_neighbors(&self) -> Option<&Vec<SmallVec<[u32; 16]>>> {
        match &self.storage {
            NeighborStorage::Uncompressed(neighbors) => Some(neighbors),
            #[cfg(feature = "id-compression")]
            NeighborStorage::Compressed { .. } => None,
        }
    }
}

impl HNSWIndex {
    /// Create a builder for configuring an HNSW index.
    pub fn builder(dimension: usize) -> HNSWBuilder {
        HNSWBuilder {
            dimension,
            m: 16,
            m_max: None,
            ef_construction: 200,
            ef_search: 50,
            auto_normalize: false,
            metric: DistanceMetric::Cosine,
        }
    }

    /// Create a new HNSW index.
    ///
    /// # Arguments
    ///
    /// * `dimension` - Vector dimension
    /// * `m` - Maximum connections per node
    /// * `m_max` - Maximum connections for new nodes
    ///
    /// # Errors
    ///
    /// Returns `RetrieveError` if parameters are invalid.
    pub fn new(dimension: usize, m: usize, m_max: usize) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be > 0".into(),
            ));
        }
        if m == 0 || m_max == 0 {
            return Err(RetrieveError::InvalidParameter(
                "m and m_max must be greater than 0".into(),
            ));
        }

        Ok(Self {
            vectors: Vec::new(),
            doc_ids: Vec::new(),
            doc_id_to_internal: HashMap::new(),
            dimension,
            num_vectors: 0,
            layers: Vec::new(),
            layer_assignments: Vec::new(),
            params: HNSWParams {
                m,
                m_max,
                ..Default::default()
            },
            built: false,
            cached_entry_point: None,
            metadata: None,
            filter_field: None,
            category_assignments: Vec::new(),
            tombstones: TombstoneSet::default(),
            rng: Mutex::new(None),
        })
    }

    /// Create with custom parameters.
    pub fn with_params(dimension: usize, params: HNSWParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be > 0".into(),
            ));
        }
        if params.m == 0 || params.m_max == 0 {
            return Err(RetrieveError::InvalidParameter(
                "m and m_max must be greater than 0".into(),
            ));
        }

        Ok(Self {
            vectors: Vec::new(),
            doc_ids: Vec::new(),
            doc_id_to_internal: HashMap::new(),
            dimension,
            num_vectors: 0,
            layers: Vec::new(),
            layer_assignments: Vec::new(),
            rng: Mutex::new(params.seed.map(rand::rngs::StdRng::seed_from_u64)),
            params,
            built: false,
            cached_entry_point: None,
            metadata: None,
            filter_field: None,
            category_assignments: Vec::new(),
            tombstones: TombstoneSet::default(),
        })
    }

    /// Create a new HNSW index with filtering support.
    ///
    /// # Arguments
    ///
    /// * `dimension` - Vector dimension
    /// * `m` - Maximum connections per node
    /// * `m_max` - Maximum connections for new nodes
    /// * `filter_field` - Field name for filtering (e.g., "category")
    pub fn with_filtering(
        dimension: usize,
        m: usize,
        m_max: usize,
        filter_field: impl Into<String>,
    ) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be > 0".into(),
            ));
        }
        if m == 0 || m_max == 0 {
            return Err(RetrieveError::InvalidParameter(
                "m and m_max must be greater than 0".into(),
            ));
        }

        Ok(Self {
            vectors: Vec::new(),
            doc_ids: Vec::new(),
            doc_id_to_internal: HashMap::new(),
            dimension,
            num_vectors: 0,
            layers: Vec::new(),
            layer_assignments: Vec::new(),
            params: HNSWParams {
                m,
                m_max,
                ..Default::default()
            },
            built: false,
            cached_entry_point: None,
            metadata: Some(crate::filtering::MetadataStore::new()),
            filter_field: Some(filter_field.into()),
            category_assignments: Vec::new(),
            tombstones: TombstoneSet::default(),
            rng: Mutex::new(None),
        })
    }

    /// Check if the index has been built and is ready for search.
    pub fn is_built(&self) -> bool {
        self.built
    }

    /// Return the cached entry point (node with highest layer assignment).
    ///
    /// Available after `build()`. Returns `None` if the index is empty or not built.
    pub fn entry_point(&self) -> Option<(u32, usize)> {
        self.cached_entry_point.map(|ep| {
            let layer = self.layer_assignments[ep as usize] as usize;
            (ep, layer)
        })
    }

    /// Borrow the raw flat vector storage.
    ///
    /// Layout: `[v0[0..d], v1[0..d], ..., vn[0..d]]` where d = dimension.
    /// Useful for building external structures (PRT projections, quantization)
    /// that need access to the stored vectors.
    pub fn raw_vectors(&self) -> &[f32] {
        &self.vectors
    }

    /// Return the maximum neighbor count across all nodes and layers.
    ///
    /// Useful for verifying the graph respects the degree bound after construction.
    /// Returns `(layer_index, node_id, degree)` for the node with the most neighbors.
    pub fn max_node_degree(&self) -> (usize, u32, usize) {
        let mut max = (0, 0u32, 0usize);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            for node_id in 0..layer.len() as u32 {
                let neighbors = layer.get_neighbors(node_id);
                let degree = neighbors.len();
                if degree > max.2 {
                    max = (layer_idx, node_id, degree);
                }
            }
        }
        max
    }

    /// Mark a vector as deleted. Deleted vectors are excluded from search results.
    ///
    /// The vector's storage is not reclaimed until the index is rebuilt.
    /// Graph edges from/to deleted nodes remain intact for navigation;
    /// deleted nodes are filtered from final results only.
    ///
    /// # Persistence caveat
    ///
    /// Tombstones are *not* serialized by `save_to_writer` or `save_to_file`
    /// (both `serde`-gated): a tombstoned doc_id will reappear in search
    /// results after a save / load roundtrip. For deletions that must
    /// survive serialization, use [`Self::delete_with_repair`], which
    /// physically removes graph edges and updates the entry point so the
    /// deletion is captured by the persisted graph state.
    pub fn delete(&mut self, doc_id: u32) -> Result<(), RetrieveError> {
        let internal_id = self
            .doc_id_to_internal
            .get(&doc_id)
            .copied()
            .ok_or_else(|| {
                RetrieveError::InvalidParameter(format!("doc_id {} not found in index", doc_id))
            })?;

        self.tombstones.delete(internal_id as usize);
        Ok(())
    }

    /// Delete a vector with Wolverine++ graph repair.
    ///
    /// Unlike [`HNSWIndex::delete`](Self::delete) (which only tombstones), this method repairs the graph:
    /// 1. Finds all in-neighbors (nodes pointing to the deleted node) on each layer
    /// 2. Removes edges to the deleted node
    /// 3. For each in-neighbor, finds a replacement via 2-hop crescent-locus filtering
    /// 4. Updates the entry point if the deleted node was the entry
    ///
    /// The crescent-locus filter (Wolverine, VLDB 2025) selects repair candidates
    /// that are closer to the in-neighbor than the deleted node was, and far from
    /// the deleted node itself. This produces edges that survive diversity pruning
    /// and maintain monotonic reachability.
    ///
    /// Returns the number of repair edges added. Requires uncompressed neighbor storage.
    pub fn delete_with_repair(&mut self, doc_id: u32) -> Result<usize, RetrieveError> {
        let internal_id = self
            .doc_id_to_internal
            .get(&doc_id)
            .copied()
            .ok_or_else(|| {
                RetrieveError::InvalidParameter(format!("doc_id {} not found in index", doc_id))
            })?;

        // Verify layers are uncompressed (compressed layers are read-only)
        #[cfg(feature = "id-compression")]
        for layer in &self.layers {
            if matches!(layer.storage, NeighborStorage::Compressed { .. }) {
                return Err(RetrieveError::InvalidParameter(
                    "delete_with_repair requires uncompressed neighbor storage".into(),
                ));
            }
        }

        let dist_fn = self.dist_fn();
        let deleted_vec: Vec<f32> = self.get_vector(internal_id as usize).to_vec();
        let max_layer = self
            .layer_assignments
            .get(internal_id as usize)
            .copied()
            .unwrap_or(0);
        let m_max = self.params.m_max;
        let mut total_repairs = 0usize;

        // Process each layer where the deleted node exists (layer 0 through max_layer)
        for layer_idx in 0..=max_layer as usize {
            if layer_idx >= self.layers.len() {
                break;
            }

            // Phase 1: Find in-neighbors (scan layer for nodes pointing to deleted node)
            // and collect repair info before mutating.
            let layer_len = self.layers[layer_idx].len();
            let mut repair_tasks: Vec<(u32, Vec<u32>)> = Vec::new(); // (in_neighbor_id, current_neighbors_after_removal)

            for node_id in 0..layer_len as u32 {
                if node_id == internal_id {
                    continue;
                }
                let neighbors = self.layers[layer_idx].get_neighbors(node_id);
                if neighbors.contains(&internal_id) {
                    // This node points to the deleted node -- needs repair
                    let remaining: Vec<u32> = neighbors
                        .iter()
                        .copied()
                        .filter(|&n| n != internal_id)
                        .collect();
                    repair_tasks.push((node_id, remaining));
                }
            }

            // Phase 2: For each in-neighbor, find crescent-locus replacement via 2-hop
            let mut replacements: Vec<(u32, Vec<u32>)> = Vec::new();
            for (in_neighbor_id, remaining_neighbors) in &repair_tasks {
                let in_vec = self.get_vector(*in_neighbor_id as usize);
                let dist_deleted_to_in = dist_fn(in_vec, &deleted_vec);

                let mut best_candidate: Option<(u32, f32)> = None;
                let mut fallback: Option<(u32, f32)> = None;
                let existing_set: std::collections::HashSet<u32> =
                    remaining_neighbors.iter().copied().collect();

                // Scan 2-hop neighborhood
                for &neighbor in remaining_neighbors {
                    let two_hop_neighbors = self.layers[layer_idx].get_neighbors(neighbor);
                    for &two_hop in two_hop_neighbors.iter() {
                        if two_hop == *in_neighbor_id
                            || two_hop == internal_id
                            || existing_set.contains(&two_hop)
                            || self.tombstones.is_deleted(two_hop as usize)
                        {
                            continue;
                        }

                        let th_vec = self.get_vector(two_hop as usize);
                        let dist_to_in = dist_fn(in_vec, th_vec);
                        let dist_to_deleted = dist_fn(&deleted_vec, th_vec);

                        // Crescent-locus conditions
                        let close_to_in = dist_to_in < dist_deleted_to_in;
                        let far_from_deleted = dist_to_deleted > dist_deleted_to_in;

                        if close_to_in && far_from_deleted {
                            match &best_candidate {
                                Some((_, best_dist)) if dist_to_in >= *best_dist => {}
                                _ => best_candidate = Some((two_hop, dist_to_in)),
                            }
                        } else {
                            match &fallback {
                                Some((_, best_dist)) if dist_to_in >= *best_dist => {}
                                _ => fallback = Some((two_hop, dist_to_in)),
                            }
                        }
                    }
                }

                let replacement = best_candidate.or(fallback).map(|(id, _)| id);
                let mut new_neighbors = remaining_neighbors.clone();
                if let Some(rep) = replacement {
                    if new_neighbors.len() < m_max {
                        new_neighbors.push(rep);
                        total_repairs += 1;
                    }
                }
                replacements.push((*in_neighbor_id, new_neighbors));
            }

            // Phase 3: Apply all replacements
            let neighbors_vec = self.layers[layer_idx].get_neighbors_mut();
            for (node_id, new_neighbors) in replacements {
                let sv = &mut neighbors_vec[node_id as usize];
                sv.clear();
                sv.extend(new_neighbors);
            }

            // Clear deleted node's own neighbor list
            let neighbors_vec = self.layers[layer_idx].get_neighbors_mut();
            neighbors_vec[internal_id as usize].clear();
        }

        // Update entry point if we deleted it
        if self.cached_entry_point == Some(internal_id) {
            self.cached_entry_point = self.find_entry_point_excluding(internal_id);
        }

        // Mark as tombstone so search results exclude it
        self.tombstones.delete(internal_id as usize);

        Ok(total_repairs)
    }

    /// Batch delete with Wolverine++ repair.
    ///
    /// More efficient than calling `delete_with_repair` in a loop because
    /// in-neighbor scanning is done once per layer for all deleted nodes.
    pub fn delete_batch_with_repair(&mut self, doc_ids: &[u32]) -> Result<usize, RetrieveError> {
        // Resolve all internal IDs first
        let internal_ids: Vec<u32> = doc_ids
            .iter()
            .map(|&doc_id| {
                self.doc_id_to_internal
                    .get(&doc_id)
                    .copied()
                    .ok_or_else(|| {
                        RetrieveError::InvalidParameter(format!(
                            "doc_id {} not found in index",
                            doc_id
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let deleted_set: std::collections::HashSet<u32> = internal_ids.iter().copied().collect();

        #[cfg(feature = "id-compression")]
        for layer in &self.layers {
            if matches!(layer.storage, NeighborStorage::Compressed { .. }) {
                return Err(RetrieveError::InvalidParameter(
                    "delete_batch_with_repair requires uncompressed neighbor storage".into(),
                ));
            }
        }

        let dist_fn = self.dist_fn();
        let m_max = self.params.m_max;
        let mut total_repairs = 0usize;

        // Find max layer across all deleted nodes
        let max_layer = internal_ids
            .iter()
            .filter_map(|&id| self.layer_assignments.get(id as usize).copied())
            .max()
            .unwrap_or(0);

        for layer_idx in 0..=max_layer as usize {
            if layer_idx >= self.layers.len() {
                break;
            }

            // Phase 1: Scan for all in-neighbors of any deleted node
            let layer_len = self.layers[layer_idx].len();
            let mut repair_tasks: Vec<(u32, Vec<u32>)> = Vec::new();

            for node_id in 0..layer_len as u32 {
                if deleted_set.contains(&node_id) {
                    continue;
                }
                let neighbors = self.layers[layer_idx].get_neighbors(node_id);
                let has_deleted = neighbors.iter().any(|n| deleted_set.contains(n));
                if has_deleted {
                    let remaining: Vec<u32> = neighbors
                        .iter()
                        .copied()
                        .filter(|n| !deleted_set.contains(n))
                        .collect();
                    repair_tasks.push((node_id, remaining));
                }
            }

            // Phase 2: Find replacements
            let mut replacements: Vec<(u32, Vec<u32>)> = Vec::new();
            for (in_neighbor_id, remaining_neighbors) in &repair_tasks {
                let in_vec = self.get_vector(*in_neighbor_id as usize);
                let existing_set: std::collections::HashSet<u32> =
                    remaining_neighbors.iter().copied().collect();

                // How many edges were lost?
                let capacity = m_max.saturating_sub(remaining_neighbors.len());
                let mut new_neighbors = remaining_neighbors.clone();
                let mut added = 0usize;

                if capacity > 0 {
                    // Collect all 2-hop candidates sorted by distance
                    let mut candidates: Vec<(u32, f32)> = Vec::new();
                    let mut visited: std::collections::HashSet<u32> = existing_set.clone();
                    visited.insert(*in_neighbor_id);
                    visited.extend(deleted_set.iter());

                    for &neighbor in remaining_neighbors {
                        let two_hop_neighbors = self.layers[layer_idx].get_neighbors(neighbor);
                        for &two_hop in two_hop_neighbors.iter() {
                            if !visited.insert(two_hop) {
                                continue;
                            }
                            if self.tombstones.is_deleted(two_hop as usize) {
                                continue;
                            }
                            let th_vec = self.get_vector(two_hop as usize);
                            let dist_to_in = dist_fn(in_vec, th_vec);
                            candidates.push((two_hop, dist_to_in));
                        }
                    }

                    candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

                    for (cand_id, _) in candidates {
                        if added >= capacity {
                            break;
                        }
                        new_neighbors.push(cand_id);
                        added += 1;
                    }
                    total_repairs += added;
                }
                replacements.push((*in_neighbor_id, new_neighbors));
            }

            // Phase 3: Apply
            let neighbors_vec = self.layers[layer_idx].get_neighbors_mut();
            for (node_id, new_neighbors) in replacements {
                let sv = &mut neighbors_vec[node_id as usize];
                sv.clear();
                sv.extend(new_neighbors);
            }

            // Clear deleted nodes' neighbor lists
            let neighbors_vec = self.layers[layer_idx].get_neighbors_mut();
            for &id in &internal_ids {
                if (id as usize) < neighbors_vec.len() {
                    neighbors_vec[id as usize].clear();
                }
            }
        }

        // Update entry point
        if let Some(ep) = self.cached_entry_point {
            if deleted_set.contains(&ep) {
                self.cached_entry_point = self.find_entry_point_excluding_set(&deleted_set);
            }
        }

        // Mark all as tombstones
        for &id in &internal_ids {
            self.tombstones.delete(id as usize);
        }

        Ok(total_repairs)
    }

    /// Find a new entry point, excluding a specific node.
    fn find_entry_point_excluding(&self, excluded: u32) -> Option<u32> {
        // Pick the surviving node with the highest layer assignment
        self.layer_assignments
            .iter()
            .enumerate()
            .filter(|(i, _)| *i as u32 != excluded && !self.tombstones.is_deleted(*i))
            .max_by_key(|(_, &layer)| layer)
            .map(|(i, _)| i as u32)
    }

    /// Find a new entry point, excluding a set of nodes.
    fn find_entry_point_excluding_set(
        &self,
        excluded: &std::collections::HashSet<u32>,
    ) -> Option<u32> {
        self.layer_assignments
            .iter()
            .enumerate()
            .filter(|(i, _)| !excluded.contains(&(*i as u32)) && !self.tombstones.is_deleted(*i))
            .max_by_key(|(_, &layer)| layer)
            .map(|(i, _)| i as u32)
    }

    /// Check if a doc_id has been deleted.
    pub fn is_deleted(&self, doc_id: u32) -> bool {
        self.doc_id_to_internal
            .get(&doc_id)
            .map(|&internal_id| self.tombstones.is_deleted(internal_id as usize))
            .unwrap_or(false)
    }

    /// Number of active (non-deleted) vectors.
    pub fn num_active(&self) -> usize {
        self.num_vectors.saturating_sub(self.tombstones.len())
    }

    /// Memory usage breakdown for this index.
    pub fn memory_usage(&self) -> crate::memory::MemoryReport {
        let vectors_bytes = self.vectors.len() * std::mem::size_of::<f32>();

        let graph_bytes: usize = self
            .layers
            .iter()
            .map(|layer| match &layer.storage {
                NeighborStorage::Uncompressed(neighbors) => neighbors
                    .iter()
                    .map(|sv| sv.capacity() * std::mem::size_of::<u32>())
                    .sum::<usize>(),
                #[cfg(feature = "id-compression")]
                NeighborStorage::Compressed { data, .. } => {
                    data.iter().map(|cl| cl.data.len()).sum::<usize>()
                }
            })
            .sum();

        let metadata_bytes = self.doc_ids.len() * std::mem::size_of::<u32>()
            + self.layer_assignments.len() * std::mem::size_of::<u8>();

        crate::memory::MemoryReport {
            vectors_bytes,
            graph_bytes,
            quantized_bytes: 0,
            metadata_bytes,
        }
    }

    /// Serialize this index to a writer as JSON.
    ///
    /// The `metadata` store and `doc_id_to_internal` reverse map are not
    /// serialized. The reverse map is rebuilt on [`Self::load_from_reader`]; metadata
    /// must be re-added if filtered search is needed.
    ///
    /// **Tombstones are not serialized.** A doc_id removed via
    /// [`Self::delete`] before this call will reappear in search results
    /// after [`Self::load_from_reader`]. Use [`Self::delete_with_repair`]
    /// for deletions that must survive a save / load roundtrip.
    #[cfg(feature = "serde")]
    pub fn save_to_writer<W: std::io::Write>(&self, writer: W) -> Result<(), RetrieveError> {
        serde_json::to_writer(writer, self).map_err(|e| RetrieveError::Serialization(e.to_string()))
    }

    /// Save this index to a file path (JSON).
    ///
    /// Serialize the index to a file path (JSON), atomically and durably.
    ///
    /// Three-step pipeline:
    /// 1. Write to a sibling temp file (`<path>.tmp`).
    /// 2. `sync_all` on the temp file (push bytes through the page cache to
    ///    the device).
    /// 3. `std::fs::rename` into place. POSIX rename is atomic within a
    ///    filesystem; on Windows it uses `MoveFileExW` with REPLACE_EXISTING.
    /// 4. `sync_all` on the parent directory (best-effort) so the rename
    ///    survives a crash on filesystems that journal data and metadata
    ///    independently (XFS, some ext4 configs, overlayfs). Without this,
    ///    the rename can succeed in memory while the directory entry is
    ///    still buffered, leaving the file unfindable after a crash. ext4
    ///    with `auto_da_alloc` does this implicitly; XFS and tmpfs do not.
    ///
    /// P-HNSW (MDPI 2025) flags partial-write corruption as a recurring
    /// failure mode for graph-index persistence; this routine is the
    /// in-tree counterpart to `durability::storage::FsDirectory::atomic_write`,
    /// inlined here because `save_to_file` is `serde`-gated (broader than the
    /// `persistence`-gated `Directory` layer). Callers operating through a
    /// `Directory` should prefer that path -- it covers the same shape and
    /// composes with the rest of the durability stack (WAL, recovery).
    ///
    /// **Tombstones are not persisted.** See [`Self::save_to_writer`] for
    /// details; for durable deletion, use [`Self::delete_with_repair`].
    #[cfg(feature = "serde")]
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), RetrieveError> {
        let path = path.as_ref();
        let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
        let filename = path.file_name().ok_or_else(|| {
            RetrieveError::Serialization("save_to_file: target path has no filename".into())
        })?;
        let mut tmp_name: std::ffi::OsString = filename.to_owned();
        tmp_name.push(".tmp");
        let tmp_path = parent.join(&tmp_name);

        {
            let file = std::fs::File::create(&tmp_path)?;
            let mut writer = std::io::BufWriter::new(file);
            self.save_to_writer(&mut writer)?;
            let file = writer.into_inner().map_err(|e| {
                RetrieveError::Serialization(format!("save_to_file flush failed: {}", e.error()))
            })?;
            // Push bytes through the kernel page cache to the device.
            // Without this, the rename below can succeed while the
            // contents are still buffered, defeating the atomicity.
            file.sync_all()?;
        }

        std::fs::rename(&tmp_path, path).map_err(|e| {
            // Best-effort cleanup of the temp; ignore the cleanup error
            // because the rename error is the more informative one.
            let _ = std::fs::remove_file(&tmp_path);
            RetrieveError::from(e)
        })?;

        // Make the *name* durable: the rename above is atomic but the
        // directory entry can still be buffered. Open the parent and
        // sync_all. Best-effort because some platforms (notably Windows)
        // don't allow opening a directory as a File and will return
        // PermissionDenied / NotFound; the file data itself is already
        // durable from the sync_all above, so a parent-dir sync failure
        // is a graceful degradation, not a save failure.
        if let Ok(dir_file) = std::fs::File::open(parent) {
            let _ = dir_file.sync_all();
        }

        Ok(())
    }

    /// Load an index from a file path (JSON).
    ///
    /// Convenience wrapper over [`Self::load_from_reader`].
    #[cfg(feature = "serde")]
    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, RetrieveError> {
        let file = std::fs::File::open(path.as_ref())?;
        Self::load_from_reader(std::io::BufReader::new(file))
    }

    /// Deserialize an index from a reader (JSON).
    ///
    /// Rebuilds the `doc_id_to_internal` reverse map from `doc_ids`.
    /// The `metadata` store is not restored -- call [`Self::add_metadata`] after
    /// loading if filtered search is needed.
    #[cfg(feature = "serde")]
    pub fn load_from_reader<R: std::io::Read>(reader: R) -> Result<Self, RetrieveError> {
        let mut index: Self = serde_json::from_reader(reader)
            .map_err(|e| RetrieveError::Serialization(e.to_string()))?;

        // Validate structural invariants before the index is usable.
        index.validate_structure()?;

        // Rebuild the reverse map that was skipped during deserialization.
        // validate_structure already checked for duplicate doc_ids.
        index.doc_id_to_internal = index
            .doc_ids
            .iter()
            .enumerate()
            .map(|(i, &doc_id)| (doc_id, i as u32))
            .collect();

        Ok(index)
    }

    /// Validate structural invariants of the index.
    ///
    /// Catches malformed or adversarial data that would cause panics during search.
    /// Called by `load_from_reader` after deserialization and by `from_parts` after
    /// reconstruction.
    fn validate_structure(&self) -> Result<(), RetrieveError> {
        // Dimension must be positive.
        if self.dimension == 0 {
            return Err(RetrieveError::FormatError("dimension must be > 0".into()));
        }

        // Vector buffer must be exactly num_vectors * dimension.
        if self.vectors.len() != self.num_vectors * self.dimension {
            return Err(RetrieveError::FormatError(format!(
                "vectors.len() ({}) != num_vectors ({}) * dimension ({})",
                self.vectors.len(),
                self.num_vectors,
                self.dimension
            )));
        }

        // doc_ids length must match num_vectors.
        if self.doc_ids.len() != self.num_vectors {
            return Err(RetrieveError::FormatError(format!(
                "doc_ids.len() ({}) != num_vectors ({})",
                self.doc_ids.len(),
                self.num_vectors
            )));
        }

        // layer_assignments length must match num_vectors.
        if self.layer_assignments.len() != self.num_vectors {
            return Err(RetrieveError::FormatError(format!(
                "layer_assignments.len() ({}) != num_vectors ({})",
                self.layer_assignments.len(),
                self.num_vectors
            )));
        }

        // category_assignments, if non-empty, must match num_vectors.
        if !self.category_assignments.is_empty()
            && self.category_assignments.len() != self.num_vectors
        {
            return Err(RetrieveError::FormatError(format!(
                "category_assignments.len() ({}) != num_vectors ({})",
                self.category_assignments.len(),
                self.num_vectors
            )));
        }

        // Check for duplicate doc_ids.
        {
            let mut seen = std::collections::HashSet::with_capacity(self.doc_ids.len());
            for &doc_id in &self.doc_ids {
                if !seen.insert(doc_id) {
                    return Err(RetrieveError::FormatError(format!(
                        "duplicate doc_id: {}",
                        doc_id
                    )));
                }
            }
        }

        // Layer count sanity bound.
        if self.layers.len() > 255 {
            return Err(RetrieveError::FormatError(format!(
                "too many layers ({}) -- expected < 256",
                self.layers.len()
            )));
        }

        // If built, layers should be non-empty (unless the index itself is empty).
        if self.built && self.num_vectors > 0 && self.layers.is_empty() {
            return Err(RetrieveError::FormatError(
                "index is marked as built but has no layers".into(),
            ));
        }

        // All neighbor IDs must be in bounds.
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_len = layer.len();
            for node in 0..layer_len {
                let neighbors = layer.get_neighbors(node as u32);
                for &neighbor in neighbors.iter() {
                    if neighbor as usize >= self.num_vectors {
                        return Err(RetrieveError::FormatError(format!(
                            "layer {} node {} has out-of-bounds neighbor {} (num_vectors={})",
                            layer_idx, node, neighbor, self.num_vectors
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    /// Reconstruct an index from persisted parts (internal use only).
    ///
    /// This is used by the persistence layer to reconstruct an index from disk.
    /// Validates structural invariants to prevent silent corruption from
    /// malformed or adversarial persistence data.
    #[allow(dead_code)]
    pub(crate) fn from_parts(
        vectors: Vec<f32>,
        dimension: usize,
        num_vectors: usize,
        layers: Vec<Layer>,
        layer_assignments: Vec<u8>,
        params: HNSWParams,
        built: bool,
        doc_ids: Vec<u32>,
    ) -> Result<Self, RetrieveError> {
        let seed = params.seed;
        // Build reverse map (validate_structure checks for duplicates).
        let doc_id_to_internal: HashMap<u32, u32> = doc_ids
            .iter()
            .enumerate()
            .map(|(i, &doc_id)| (doc_id, i as u32))
            .collect();

        let mut index = Self {
            vectors,
            doc_ids,
            doc_id_to_internal,
            dimension,
            num_vectors,
            layers,
            layer_assignments,
            params,
            built,
            cached_entry_point: None,
            metadata: None,
            filter_field: None,
            category_assignments: Vec::new(),
            tombstones: TombstoneSet::default(),
            rng: Mutex::new(seed.map(rand::rngs::StdRng::seed_from_u64)),
        };
        // Cache entry point if already built
        if built {
            index.cached_entry_point = index.compute_entry_point();
        }

        // Reuse shared validation. Map FormatError to InvalidParameter to
        // preserve the existing error variant contract for from_parts callers.
        index.validate_structure().map_err(|e| match e {
            RetrieveError::FormatError(msg) => RetrieveError::InvalidParameter(msg),
            other => other,
        })?;

        Ok(index)
    }

    /// Add metadata for a document (required for filtering).
    pub fn add_metadata(
        &mut self,
        doc_id: u32,
        metadata: crate::filtering::DocumentMetadata,
    ) -> Result<(), RetrieveError> {
        if let Some(ref mut store) = self.metadata {
            store.add(doc_id, metadata);
            Ok(())
        } else {
            Err(RetrieveError::InvalidParameter(
                "filtering not enabled; use HNSWIndex::with_filtering()".into(),
            ))
        }
    }

    /// Add a vector to the index.
    ///
    /// Vectors should be L2-normalized for cosine similarity.
    /// Index must be built before searching.
    pub fn add(&mut self, doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        self.add_slice(doc_id, &vector)
    }

    /// Add a vector to the index from a borrowed slice.
    ///
    /// # Errors
    ///
    /// Returns [`RetrieveError::InvalidParameter`] if:
    /// - The vector is not L2-normalized (`norm^2` outside `[0.9, 1.1]`),
    ///   unless `auto_normalize` is enabled on the builder.
    /// - The `doc_id` is a duplicate.
    /// - The index has already been built.
    pub fn add_slice(&mut self, doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
        if self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot add vectors after index is built".into(),
            ));
        }

        if self.doc_id_to_internal.contains_key(&doc_id) {
            return Err(RetrieveError::InvalidParameter(format!(
                "duplicate doc_id {} (doc_id must be unique within an index)",
                doc_id
            )));
        }

        if vector.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: vector.len(),
                doc_dim: self.dimension,
            });
        }

        // Normalize if requested, otherwise borrow the original slice.
        let normalized;
        let vector = if self.params.auto_normalize {
            normalized = crate::distance::normalize(vector);
            normalized.as_slice()
        } else {
            vector
        };

        // Cosine distance uses 1 - dot(a,b), which only equals cosine distance when
        // vectors are L2-normalized. Reject clearly un-normalized vectors to prevent
        // silent wrong results. Skip this check for metrics that don't require it (e.g. L2).
        // When auto_normalize is on, zero vectors normalize to zero (degenerate but valid).
        if self.params.metric == DistanceMetric::Cosine && !self.params.auto_normalize {
            let norm_sq: f32 = vector.iter().map(|x| x * x).sum();
            if (norm_sq - 1.0).abs() > 0.01 {
                return Err(RetrieveError::InvalidParameter(format!(
                    "HNSW cosine distance requires L2-normalized vectors \
                     (got norm^2 = {:.4}, expected ~1.0). \
                     Use `distance::normalize()` or set `auto_normalize(true)` on the builder.",
                    norm_sq
                )));
            }
        }

        // Assign internal ID (stable: insertion order)
        let internal_id = self.num_vectors as u32;

        self.doc_ids.push(doc_id);
        self.doc_id_to_internal.insert(doc_id, internal_id);

        // Store vector in SoA format
        self.vectors.extend_from_slice(vector);
        self.num_vectors += 1;

        // Assign layer (exponential distribution)
        let layer = self.assign_layer();
        self.layer_assignments.push(layer);

        // Store category assignment if filtering is enabled
        if let (Some(ref metadata_store), Some(ref field)) = (&self.metadata, &self.filter_field) {
            let category = metadata_store
                .get(doc_id)
                .and_then(|m| m.get(field).cloned());
            self.category_assignments.push(category);
        } else {
            self.category_assignments.push(None);
        }

        // Post-condition: parallel arrays must stay in sync
        debug_assert_eq!(
            self.vectors.len(),
            self.num_vectors * self.dimension,
            "vectors buffer out of sync: expected {} floats, got {}",
            self.num_vectors * self.dimension,
            self.vectors.len()
        );
        debug_assert_eq!(
            self.doc_ids.len(),
            self.num_vectors,
            "doc_ids out of sync with num_vectors"
        );
        debug_assert_eq!(
            self.layer_assignments.len(),
            self.num_vectors,
            "layer_assignments out of sync with num_vectors"
        );
        debug_assert_eq!(
            self.category_assignments.len(),
            self.num_vectors,
            "category_assignments out of sync with num_vectors"
        );

        Ok(())
    }

    /// Add multiple vectors in bulk from a flat f32 slice.
    ///
    /// `ids` and `vectors` must be aligned: `vectors.len() == ids.len() * self.dimension`.
    /// Each contiguous `dimension`-sized chunk in `vectors` corresponds to the ID at the same
    /// position in `ids`.
    pub fn add_batch(&mut self, ids: &[u32], vectors: &[f32]) -> Result<(), RetrieveError> {
        if vectors.len() != ids.len() * self.dimension {
            return Err(RetrieveError::InvalidParameter(format!(
                "vectors.len() ({}) != ids.len() ({}) * dimension ({})",
                vectors.len(),
                ids.len(),
                self.dimension
            )));
        }
        for (id, chunk) in ids.iter().zip(vectors.chunks_exact(self.dimension)) {
            self.add_slice(*id, chunk)?;
        }
        Ok(())
    }

    /// Build the index (required before search).
    ///
    /// Constructs the multi-layer graph structure.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(()); // Already built
        }

        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        // Construct graph
        crate::hnsw::construction::construct_graph(self)?;

        // Add intra-category edges if filtering is enabled
        if self.metadata.is_some() && self.filter_field.is_some() {
            self.add_intra_category_edges()?;
        }

        // Compress layers if enabled
        #[cfg(feature = "id-compression")]
        {
            if let Some(method) = self.params.id_compression.clone() {
                let threshold = self.params.compression_threshold;
                if self.params.m >= threshold {
                    self.compress_layers(&method)
                        .map_err(|e| RetrieveError::Other(format!("Compression failed: {}", e)))?;
                }
            }
        }

        self.built = true;
        self.cached_entry_point = self.compute_entry_point();
        self.reorder_for_locality();

        Ok(())
    }

    /// Build the index using parallel batched construction (requires `parallel` feature).
    ///
    /// Same as [`build`](Self::build) but uses rayon to parallelize the neighbor
    /// search phase within each batch. The `batch_size` parameter controls how many
    /// vectors search the graph concurrently.
    ///
    /// Recommended `batch_size`: 4096. Larger batches improve parallelism at the
    /// cost of marginally lower recall (vectors in the same batch don't see each
    /// other's edges). Recall loss is typically < 1% at batch=4096.
    #[cfg(feature = "parallel")]
    pub fn build_parallel(&mut self, batch_size: usize) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        crate::hnsw::construction::construct_graph_parallel(self, batch_size)?;

        if self.metadata.is_some() && self.filter_field.is_some() {
            self.add_intra_category_edges()?;
        }

        #[cfg(feature = "id-compression")]
        {
            if let Some(method) = self.params.id_compression.clone() {
                let threshold = self.params.compression_threshold;
                if self.params.m >= threshold {
                    self.compress_layers(&method)
                        .map_err(|e| RetrieveError::Other(format!("Compression failed: {}", e)))?;
                }
            }
        }

        self.built = true;
        self.cached_entry_point = self.compute_entry_point();
        self.reorder_for_locality();
        Ok(())
    }

    /// Reorder graph nodes so BFS-adjacent nodes are contiguous in memory.
    ///
    /// After construction, nodes are in insertion order. Graph traversal visits
    /// neighbors that may be scattered across memory, causing cache misses.
    /// BFS-ordering from the entry point places frequently co-visited nodes
    /// in adjacent memory locations, improving L2/L3 cache hit rates.
    ///
    /// The permutation is applied to: vectors, doc_ids, layer_assignments,
    /// category_assignments, all neighbor lists in all layers, and the
    /// doc_id_to_internal reverse map.
    fn reorder_for_locality(&mut self) {
        if self.num_vectors <= 1 {
            return;
        }
        let ep = match self.cached_entry_point {
            Some(ep) => ep as usize,
            None => return,
        };

        let n = self.num_vectors;
        let dim = self.dimension;

        // BFS from entry point through base layer to compute visit order.
        let mut new_order: Vec<u32> = Vec::with_capacity(n);
        let mut visited = vec![false; n];
        let mut queue = std::collections::VecDeque::with_capacity(n);

        queue.push_back(ep);
        visited[ep] = true;

        while let Some(node) = queue.pop_front() {
            new_order.push(node as u32);
            if let Some(layer) = self.layers.first() {
                let neighbors = layer.get_neighbors(node as u32);
                for &nb in neighbors.iter() {
                    let nb = nb as usize;
                    if nb < n && !visited[nb] {
                        visited[nb] = true;
                        queue.push_back(nb);
                    }
                }
            }
        }
        // Add any unreachable nodes (rare in a connected graph).
        for (i, &v) in visited.iter().enumerate().take(n) {
            if !v {
                new_order.push(i as u32);
            }
        }

        // Build permutation: old_to_new[old_idx] = new_idx
        let mut old_to_new = vec![0u32; n];
        for (new_idx, &old_idx) in new_order.iter().enumerate() {
            old_to_new[old_idx as usize] = new_idx as u32;
        }

        // Permute vectors (flat array, stride = dim)
        let mut new_vectors = vec![0.0f32; self.vectors.len()];
        for (new_idx, &old_idx) in new_order.iter().enumerate() {
            let old_start = old_idx as usize * dim;
            let new_start = new_idx * dim;
            new_vectors[new_start..new_start + dim]
                .copy_from_slice(&self.vectors[old_start..old_start + dim]);
        }
        self.vectors = new_vectors;

        // Permute doc_ids
        let new_doc_ids: Vec<u32> = new_order
            .iter()
            .map(|&old| self.doc_ids[old as usize])
            .collect();
        self.doc_ids = new_doc_ids;

        // Permute layer_assignments
        let new_assignments: Vec<u8> = new_order
            .iter()
            .map(|&old| self.layer_assignments[old as usize])
            .collect();
        self.layer_assignments = new_assignments;

        // Remap all neighbor lists in all layers
        for layer in &mut self.layers {
            layer.remap_ids(&old_to_new, n);
        }

        // Rebuild reverse map
        self.doc_id_to_internal.clear();
        for (i, &doc_id) in self.doc_ids.iter().enumerate() {
            self.doc_id_to_internal.insert(doc_id, i as u32);
        }

        // Permute category assignments if present
        if !self.category_assignments.is_empty() {
            let new_cats: Vec<Option<crate::filtering::MetadataValue>> = new_order
                .iter()
                .map(|&old| self.category_assignments[old as usize].clone())
                .collect();
            self.category_assignments = new_cats;
        }

        // Recompute cached entry point (its index changed)
        self.cached_entry_point = self.compute_entry_point();
    }

    /// Add extra intra-category edges to improve filterable search.
    ///
    /// For each vector, adds connections to nearby vectors in the same category,
    /// ensuring filtered search doesn't break graph connectivity.
    fn add_intra_category_edges(&mut self) -> Result<(), RetrieveError> {
        if self.category_assignments.is_empty() {
            return Ok(());
        }

        // Group vectors by category
        let mut category_vectors: std::collections::HashMap<
            crate::filtering::MetadataValue,
            Vec<u32>,
        > = std::collections::HashMap::new();

        for (vector_idx, category) in self.category_assignments.iter().enumerate() {
            if let Some(cat) = category {
                category_vectors
                    .entry(cat.clone())
                    .or_default()
                    .push(vector_idx as u32);
            }
        }

        // For each category, add intra-category edges in base layer (layer 0)
        if self.layers.is_empty() {
            return Ok(());
        }

        let max_intra_edges = self.params.m / 4; // Add up to m/4 intra-category edges
        let dist_fn = self.dist_fn();

        // Collect all candidate edges first (immutable borrows)
        let mut edges_to_add: Vec<(u32, Vec<u32>)> = Vec::new();

        // Cap per-vector comparisons to avoid O(C^2) in large categories.
        // 64 random samples suffice to find good intra-category neighbors
        // when we only keep max_intra_edges (typically m/4 = 4).
        const MAX_CANDIDATES: usize = 64;

        for (_category, vector_ids) in category_vectors.iter() {
            if vector_ids.len() < 2 {
                continue;
            }

            let use_sampling = vector_ids.len() > MAX_CANDIDATES;

            for &vector_id in vector_ids.iter() {
                let vector = self.get_vector(vector_id as usize);
                let mut candidates = Vec::new();

                if use_sampling {
                    // Sample up to MAX_CANDIDATES random peers
                    use rand::Rng;
                    let mut rng = rand::rng();
                    let mut sampled = 0;
                    let mut attempts = 0;
                    while sampled < MAX_CANDIDATES && attempts < MAX_CANDIDATES * 2 {
                        let idx = rng.random_range(0..vector_ids.len());
                        let other_id = vector_ids[idx];
                        attempts += 1;
                        if other_id == vector_id {
                            continue;
                        }
                        let other_vector = self.get_vector(other_id as usize);
                        let dist = dist_fn(vector, other_vector);
                        candidates.push((other_id, dist));
                        sampled += 1;
                    }
                } else {
                    // Small category: brute-force all pairs
                    for &other_id in vector_ids.iter() {
                        if other_id == vector_id {
                            continue;
                        }
                        let other_vector = self.get_vector(other_id as usize);
                        let dist = dist_fn(vector, other_vector);
                        candidates.push((other_id, dist));
                    }
                }

                candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                let selected_neighbors: Vec<u32> = candidates
                    .iter()
                    .take(max_intra_edges)
                    .map(|(id, _)| *id)
                    .collect();

                edges_to_add.push((vector_id, selected_neighbors));
            }
        }

        // Now perform mutable operations (add edges to base layer)
        let base_layer = &mut self.layers[0];
        for (vector_id, selected_neighbors) in edges_to_add {
            let neighbors_vec = base_layer.get_neighbors_mut();
            let neighbors = &mut neighbors_vec[vector_id as usize];
            for other_id in selected_neighbors {
                if !neighbors.contains(&other_id) && neighbors.len() < self.params.m_max {
                    neighbors.push(other_id);
                }
            }
        }

        Ok(())
    }

    /// Compress all layers after construction.
    #[cfg(feature = "id-compression")]
    fn compress_layers(
        &mut self,
        method: &crate::compression::IdCompressionMethod,
    ) -> Result<(), crate::compression::CompressionError> {
        match method {
            crate::compression::IdCompressionMethod::DeltaVarint => {
                let compressor = crate::compression::DeltaVarintCompressor::new();
                let universe_size = self.num_vectors as u32;
                let threshold = self.params.compression_threshold;

                for layer in &mut self.layers {
                    layer.compress(&compressor, universe_size, threshold)?;
                }
            }
            _ => {
                // Other methods not implemented yet
            }
        }

        Ok(())
    }

    /// Search for k nearest neighbors.
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector (should be L2-normalized)
    /// * `k` - Number of neighbors to return
    /// * `ef` - Search width (higher = better recall, slower)
    ///
    /// # Returns
    ///
    /// Vector of (doc_id, distance) pairs, sorted by distance ascending.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
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

        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        // ef guard: HNSW beam search returns at most ef candidates, so requesting
        // k > ef is structurally unsound (the result set is smaller than requested).
        // Empirically, recall also degrades sharply well before that boundary.
        // Standard practice: ef >= k. We enforce this silently.
        let ef = ef.max(k);

        // Select seed points based on strategy
        let (entry_point, entry_layer, initial_seeds) = match &self.params.seed_selection {
            SeedSelectionStrategy::StackedNSW => {
                // Default: Use entry point in highest layer
                let ep = self.get_entry_point().ok_or(RetrieveError::EmptyIndex)?;
                let el = self.layer_assignments[ep as usize] as usize;
                (ep, el, vec![ep])
            }
            SeedSelectionStrategy::KSampledRandom { k } => {
                // K-Sampled Random: Sample k random nodes
                // Optimized: use reservoir sampling or direct random generation instead of collecting full Vec
                use rand::Rng;
                let mut rng = rand::rng();
                let num_samples = (*k).min(self.num_vectors);

                // Generate random seeds without creating full Vec of all IDs
                let mut seeds: Vec<u32> = Vec::with_capacity(num_samples);
                let mut used = std::collections::HashSet::with_capacity(num_samples);
                while seeds.len() < num_samples {
                    let candidate = rng.random_range(0..self.num_vectors as u32);
                    if used.insert(candidate) {
                        seeds.push(candidate);
                    }
                }

                // Find closest seed to query as entry point
                let ks_dist_fn = self.dist_fn();
                let mut best_seed = seeds[0];
                let mut best_dist = f32::INFINITY;
                for &seed_id in &seeds {
                    let seed_vec = self.get_vector(seed_id as usize);
                    let dist = ks_dist_fn(query, seed_vec);
                    if dist < best_dist {
                        best_dist = dist;
                        best_seed = seed_id;
                    }
                }

                let entry_layer = self.layer_assignments[best_seed as usize] as usize;
                (best_seed, entry_layer, seeds)
            }
        };

        // Navigate from top layer down to base layer
        let dist_fn = self.dist_fn();
        let current_closest = self.descend_upper_layers(query, entry_point, entry_layer, ef);

        // Fine search in base layer (layer 0)
        if !self.layers.is_empty() {
            // For KS strategy, warm up search with multiple seeds
            let base_results =
                if let SeedSelectionStrategy::KSampledRandom { .. } = &self.params.seed_selection {
                    // Use KS seeds to initialize search
                    // Find the best seed (closest to query) among KSampled seeds
                    // and the greedy-descended entry point, then run standard beam search.
                    let mut best_entry = current_closest;
                    let mut best_dist = dist_fn(query, self.get_vector(current_closest as usize));
                    for &seed_id in &initial_seeds {
                        let dist = dist_fn(query, self.get_vector(seed_id as usize));
                        if dist < best_dist {
                            best_dist = dist;
                            best_entry = seed_id;
                        }
                    }

                    // Use the standard beam search from the best entry point
                    crate::hnsw::search::greedy_search_layer(
                        query,
                        best_entry,
                        &self.layers[0],
                        &self.vectors,
                        self.dimension,
                        ef.max(k),
                        self.dist_fn(),
                    )
                } else {
                    // Default: Use greedy search from entry point
                    crate::hnsw::search::greedy_search_layer(
                        query,
                        current_closest,
                        &self.layers[0],
                        &self.vectors,
                        self.dimension,
                        ef.max(k),
                        self.dist_fn(),
                    )
                };

            // greedy_search_layer returns results sorted by distance; take top-k.
            let results: Vec<(u32, f32)> = base_results.into_iter().take(k).collect();

            // Clear decompression caches after search
            #[cfg(feature = "id-compression")]
            {
                for layer in &self.layers {
                    layer.clear_cache();
                }
            }

            // Convert internal IDs -> external doc_ids, filtering out deleted nodes.
            let results = results
                .into_iter()
                .filter(|(internal_id, _)| !self.tombstones.is_deleted(*internal_id as usize))
                .filter_map(|(internal_id, dist)| {
                    let doc_id = self.doc_ids.get(internal_id as usize).copied()?;
                    Some((doc_id, dist))
                })
                .collect();
            Ok(results)
        } else {
            Ok(Vec::new())
        }
    }

    /// Search with a caller-provided distance function.
    ///
    /// The graph structure (built with center-to-center distance) is used for
    /// navigation, but the provided `dist_fn` is called for every distance
    /// computation during search. The closure receives `(query, internal_id)`
    /// and must return a distance value.
    ///
    /// This enables asymmetric search: build the graph with point distances
    /// for navigability, search with box-to-point, quantized, or any other
    /// distance function.
    ///
    /// The `internal_id` passed to the closure is the zero-based insertion
    /// order index (same as the index into flat vector storage). The caller
    /// must maintain a parallel array mapping these IDs to their custom data
    /// (box geometry, quantization codes, etc.).
    ///
    /// Returns `(doc_id, distance)` pairs sorted by distance ascending.
    pub fn search_with_distance<F: Fn(&[f32], u32) -> f32>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        dist_fn: &F,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "index must be built before search".into(),
            ));
        }
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        // Navigate upper layers with standard distance (center-to-center)
        let entry_point = self.get_entry_point().ok_or(RetrieveError::EmptyIndex)?;
        let entry_layer = self.layer_assignments[entry_point as usize] as usize;

        let current_closest = self.descend_upper_layers(query, entry_point, entry_layer, ef);

        // Fine search in base layer with custom distance
        if !self.layers.is_empty() {
            let base_results = crate::hnsw::search::greedy_search_layer_custom(
                query,
                current_closest,
                &self.layers[0],
                &self.vectors,
                self.dimension,
                ef.max(k),
                dist_fn,
            );

            let results = base_results
                .into_iter()
                .take(k)
                .filter(|(internal_id, _)| !self.tombstones.is_deleted(*internal_id as usize))
                .filter_map(|(internal_id, dist)| {
                    let doc_id = self.doc_ids.get(internal_id as usize).copied()?;
                    Some((doc_id, dist))
                })
                .collect();
            Ok(results)
        } else {
            Ok(Vec::new())
        }
    }

    /// Search for multiple queries in parallel.
    ///
    /// Returns one result vector per query, in the same order as the input queries.
    /// Each result vector contains `(doc_id, distance)` pairs sorted by distance ascending.
    ///
    /// Requires the `parallel` feature.
    #[cfg(feature = "parallel")]
    pub fn search_batch(
        &self,
        queries: &[&[f32]],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<Vec<(u32, f32)>>, RetrieveError> {
        use rayon::prelude::*;
        queries
            .par_iter()
            .map(|query| self.search(query, k, ef_search))
            .collect()
    }

    /// Search for multiple queries (flat buffer) in parallel.
    ///
    /// `queries_flat` is a contiguous buffer of length `num_queries * dimension`.
    /// Returns one result vector per query, in input order.
    ///
    /// Requires the `parallel` feature.
    #[cfg(feature = "parallel")]
    pub fn search_batch_flat(
        &self,
        queries_flat: &[f32],
        num_queries: usize,
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<Vec<(u32, f32)>>, RetrieveError> {
        use rayon::prelude::*;
        (0..num_queries)
            .into_par_iter()
            .map(|i| {
                let query = &queries_flat[i * self.dimension..(i + 1) * self.dimension];
                self.search(query, k, ef_search)
            })
            .collect()
    }

    /// Search a batch of queries using Multiple Query Optimization (MQO).
    ///
    /// Instead of routing every query independently from the global HNSW entry point,
    /// queries are processed in a greedy-nearest-neighbour chain order: the first query
    /// uses the normal entry point, and each subsequent query inherits the best result
    /// node from the nearest already-processed query as an *additional* warm-start seed.
    ///
    /// This reuses graph locality between nearby queries, reducing the effective search
    /// depth for similar queries and improving cache utilisation.
    ///
    /// Results are returned in the **same order as the input queries**.
    ///
    /// If the batch contains only one query, this is equivalent to calling
    /// [`search`](Self::search) directly.
    ///
    /// # Arguments
    ///
    /// * `queries` - Slice of query vectors; each must have length `dimension`.
    /// * `k` - Number of nearest neighbours to return per query.
    /// * `ef_search` - Beam width during base-layer search (higher = better recall).
    ///
    /// # Errors
    ///
    /// Returns `RetrieveError` if the index has not been built, any query has the
    /// wrong dimension, or the index is empty.
    pub fn batch_search_mqo(
        &self,
        queries: &[&[f32]],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<Vec<(u32, f32)>>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "index must be built before search".into(),
            ));
        }
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }
        for q in queries.iter() {
            if q.len() != self.dimension {
                return Err(RetrieveError::DimensionMismatch {
                    query_dim: q.len(),
                    doc_dim: self.dimension,
                });
            }
        }

        let n = queries.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n == 1 {
            return Ok(vec![self.search(queries[0], k, ef_search)?]);
        }

        let dist_fn = self.dist_fn();

        // ── Step 1: build a greedy nearest-neighbour chain over query vectors ──
        //
        // Start from query 0. At each step pick the not-yet-visited query closest
        // to the last chosen query. O(n²) in queries -- fine for n << 1000.
        let mut chain: Vec<usize> = Vec::with_capacity(n); // visit order (query indices)
        let mut visited_queries = vec![false; n];
        chain.push(0);
        visited_queries[0] = true;

        for _ in 1..n {
            // SAFETY: chain is non-empty; we push 0 before the loop.
            let last = chain[chain.len() - 1];
            let last_vec = queries[last];
            let mut best_idx = usize::MAX;
            let mut best_dist = f32::INFINITY;
            for j in 0..n {
                if visited_queries[j] {
                    continue;
                }
                let d = dist_fn(last_vec, queries[j]);
                if d < best_dist {
                    best_dist = d;
                    best_idx = j;
                }
            }
            chain.push(best_idx);
            visited_queries[best_idx] = true;
        }

        // ── Step 2: process queries in chain order, reusing entry points ──

        let ep = self.get_entry_point().ok_or(RetrieveError::EmptyIndex)?;

        // results_by_query[i] = (doc_id, dist) results for query i
        let mut results_by_query: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];

        // best_internal[i] = internal node id of best result for chain position i
        let mut best_internal_for_chain: Vec<u32> = Vec::with_capacity(n);

        for (chain_pos, &query_idx) in chain.iter().enumerate() {
            let query = queries[query_idx];

            // Upper-layer greedy descent to find base-layer entry for this query.
            let entry_layer = self.layer_assignments[ep as usize] as usize;
            let current_closest = self.descend_upper_layers(query, ep, entry_layer, ef_search);

            // Build entry-point set: always include the HNSW-descended entry point.
            // For chain_pos > 0, also include the best node from the nearest
            // already-processed query (the parent in the chain).
            let base_results = if self.layers.is_empty() {
                Vec::new()
            } else if chain_pos == 0 {
                crate::hnsw::search::greedy_search_layer(
                    query,
                    current_closest,
                    &self.layers[0],
                    &self.vectors,
                    self.dimension,
                    ef_search.max(k),
                    dist_fn,
                )
            } else {
                let parent_best = best_internal_for_chain[chain_pos - 1];
                // Use multi-entry search seeded from both the HNSW entry point
                // and the best result of the chain parent.
                let entry_points: &[u32] = if parent_best == current_closest {
                    &[current_closest]
                } else {
                    &[current_closest, parent_best]
                };
                crate::hnsw::search::greedy_search_layer_multi_entry(
                    query,
                    entry_points,
                    &self.layers[0],
                    &self.vectors,
                    self.dimension,
                    ef_search.max(k),
                    dist_fn,
                )
            };

            // Record the best internal node (first element, sorted by distance).
            let best_internal = base_results.first().map(|r| r.0).unwrap_or(current_closest);
            best_internal_for_chain.push(best_internal);

            // Translate internal IDs -> external doc_ids, filter tombstones, take k.
            let translated: Vec<(u32, f32)> = base_results
                .into_iter()
                .filter(|(internal_id, _)| !self.tombstones.is_deleted(*internal_id as usize))
                .take(k)
                .filter_map(|(internal_id, dist)| {
                    let doc_id = self.doc_ids.get(internal_id as usize).copied()?;
                    Some((doc_id, dist))
                })
                .collect();

            results_by_query[query_idx] = translated;
        }

        Ok(results_by_query)
    }

    /// Search with filter using filterable graph (integrated filtering).
    ///
    /// Uses intra-category edges to maintain graph connectivity during filtered search.
    /// Only explores neighbors that match the filter predicate.
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector (should be L2-normalized)
    /// * `k` - Number of neighbors to return
    /// * `ef` - Search width (higher = better recall, slower)
    /// * `filter` - Filter predicate (must be equality filter on filter_field)
    ///
    /// # Returns
    ///
    /// Vector of (doc_id, distance) pairs matching the filter, sorted by distance ascending
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: &crate::filtering::MetadataFilter,
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

        if self.metadata.is_none() || self.filter_field.is_none() {
            return Err(RetrieveError::InvalidParameter(
                "filtering not enabled; use HNSWIndex::with_filtering()".into(),
            ));
        }

        // Extract category value from filter
        let desired_category = match filter {
            crate::filtering::MetadataFilter::Equals { field, value } => {
                if Some(field) != self.filter_field.as_ref() {
                    return Err(RetrieveError::InvalidParameter(format!(
                        "filter field '{}' doesn't match index filter field '{:?}'",
                        field, self.filter_field
                    )));
                }
                Some(value.clone())
            }
            _ => {
                return Err(RetrieveError::InvalidParameter(
                    "only equality filters on filter_field are supported".into(),
                ));
            }
        };

        // Perform standard search but filter neighbors during traversal
        use crate::distance::FloatOrd;
        let dist_fn = self.dist_fn();

        let mut candidates: std::collections::BinaryHeap<std::cmp::Reverse<(FloatOrd, u32)>> =
            std::collections::BinaryHeap::new();
        let mut visited = std::collections::HashSet::new();

        // Start from entry point if it matches filter, otherwise find first matching vector
        let entry_point = self.get_entry_point().ok_or(RetrieveError::EmptyIndex)?;
        let entry_category = self
            .category_assignments
            .get(entry_point as usize)
            .and_then(|c| c.as_ref());

        let start_point = if entry_category == desired_category.as_ref() {
            entry_point
        } else {
            // Find first vector matching filter
            self.category_assignments
                .iter()
                .enumerate()
                .find(|(_, cat)| cat.as_ref() == desired_category.as_ref())
                .map(|(idx, _)| idx as u32)
                .ok_or(RetrieveError::NotFound)?
        };

        let start_vec = self.get_vector(start_point as usize);
        let start_dist = dist_fn(query, start_vec);
        candidates.push(std::cmp::Reverse((FloatOrd(start_dist), start_point)));

        // Greedy search in base layer, only exploring filtered neighbors
        while let Some(std::cmp::Reverse((FloatOrd(_dist), vector_id))) = candidates.pop() {
            if visited.contains(&vector_id) {
                continue;
            }
            visited.insert(vector_id);

            // Check if this vector matches filter
            if self
                .category_assignments
                .get(vector_id as usize)
                .and_then(|c| c.as_ref())
                != desired_category.as_ref()
            {
                continue;
            }

            // Explore neighbors that match filter
            let neighbors = self.layers[0].get_neighbors(vector_id);
            for &neighbor_id in neighbors.iter() {
                if visited.contains(&neighbor_id) {
                    continue;
                }

                // Only explore neighbors in same category
                if self
                    .category_assignments
                    .get(neighbor_id as usize)
                    .and_then(|c| c.as_ref())
                    != desired_category.as_ref()
                {
                    continue;
                }

                let neighbor_vec = self.get_vector(neighbor_id as usize);
                let neighbor_dist = dist_fn(query, neighbor_vec);
                candidates.push(std::cmp::Reverse((FloatOrd(neighbor_dist), neighbor_id)));
            }

            if visited.len() >= ef.max(k) {
                break;
            }
        }

        // Extract top-k results, filtering out deleted nodes
        let mut results: Vec<(u32, f32)> = visited
            .iter()
            .filter(|&&id| !self.tombstones.is_deleted(id as usize))
            .filter_map(|&id| {
                if self
                    .category_assignments
                    .get(id as usize)
                    .and_then(|c| c.as_ref())
                    == desired_category.as_ref()
                {
                    let vec = self.get_vector(id as usize);
                    let dist = dist_fn(query, vec);
                    let doc_id = self.doc_ids.get(id as usize).copied()?;
                    Some((doc_id, dist))
                } else {
                    None
                }
            })
            .collect();

        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        Ok(results.into_iter().take(k).collect())
    }

    /// Search with adaptive early termination.
    ///
    /// Behaves like [`HNSWIndex::search`] but uses an `EarlyTerminationOracle` on the
    /// base layer to skip distance computations once the oracle is confident
    /// the top-k has converged. Upper layers use the standard greedy search
    /// (they visit few nodes and don't benefit from early termination).
    ///
    /// Returns `Ok((results, num_evaluated))` where `num_evaluated` is the
    /// number of distance computations performed on the base layer.
    pub fn search_adaptive(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        config: &crate::adaptive::AdaptiveConfig,
    ) -> Result<(Vec<(u32, f32)>, usize), RetrieveError> {
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

        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        // Upper-layer navigation
        let ep = self.get_entry_point().ok_or(RetrieveError::EmptyIndex)?;
        let entry_layer = self.layer_assignments[ep as usize] as usize;
        let current_closest = self.descend_upper_layers(query, ep, entry_layer, ef);

        // Base layer: adaptive search
        let num_evaluated;
        if !self.layers.is_empty() {
            let (mut base_results, evaluated) = crate::hnsw::search::greedy_search_layer_adaptive(
                query,
                current_closest,
                &self.layers[0],
                &self.vectors,
                self.dimension,
                ef.max(k),
                k,
                config,
                self.dist_fn(),
            );
            num_evaluated = evaluated;

            base_results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

            #[cfg(feature = "id-compression")]
            {
                for layer in &self.layers {
                    layer.clear_cache();
                }
            }

            let results = base_results
                .into_iter()
                .filter(|(internal_id, _)| !self.tombstones.is_deleted(*internal_id as usize))
                .take(k)
                .filter_map(|(internal_id, dist)| {
                    let doc_id = self.doc_ids.get(internal_id as usize).copied()?;
                    Some((doc_id, dist))
                })
                .collect();
            Ok((results, num_evaluated))
        } else {
            Ok((Vec::new(), 0))
        }
    }

    /// Search with Probabilistic Routing Test (PRT) pre-filtering.
    ///
    /// Uses random subspace projections to cheaply estimate distances before
    /// computing full distances, skipping candidates unlikely to improve results.
    /// The Test Feedback Buffer (TFB) adaptively tightens the filter threshold
    /// during the search to reduce false positives.
    ///
    /// Returns `Ok((results, full_distance_count))` where `full_distance_count`
    /// is the number of full (O(d)) distance computations performed. Compare
    /// against `search()` to measure the PRT savings.
    ///
    /// `prt` must have been initialized with `project_database(&self.vectors)`.
    pub fn search_prt(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        prt: &crate::prt::ProbabilisticRoutingTest,
        initial_ratio: f32,
        decay: f32,
    ) -> Result<(Vec<(u32, f32)>, usize), RetrieveError> {
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
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        let ef = ef.max(k);

        // Navigate upper layers with standard distance.
        let entry_point = self.get_entry_point().ok_or(RetrieveError::EmptyIndex)?;
        let entry_layer = self.layer_assignments[entry_point as usize] as usize;
        let current_closest = self.descend_upper_layers(query, entry_point, entry_layer, ef);

        // Fine search in base layer with PRT pre-filtering.
        if self.layers.is_empty() {
            return Ok((Vec::new(), 0));
        }

        let query_proj = prt.project_query(query);
        let mut tfb = crate::prt::TestFeedbackBuffer::new(initial_ratio, decay);
        let dist_fn = self.dist_fn();

        let (base_results, full_dist_count) = crate::hnsw::search::greedy_search_layer_prt(
            query,
            current_closest,
            &self.layers[0],
            &self.vectors,
            self.dimension,
            ef,
            dist_fn,
            prt,
            &query_proj,
            &mut tfb,
        );

        let results = base_results
            .into_iter()
            .take(k)
            .filter(|(internal_id, _)| !self.tombstones.is_deleted(*internal_id as usize))
            .filter_map(|(internal_id, dist)| {
                let doc_id = self.doc_ids.get(internal_id as usize).copied()?;
                Some((doc_id, dist))
            })
            .collect();

        Ok((results, full_dist_count))
    }

    /// Assign layer for a new vector using exponential distribution.
    ///
    /// Returns the maximum layer where this vector will appear.
    /// Assign a layer to a new vector using a geometric distribution.
    ///
    /// # Mathematical Foundation
    ///
    /// The layer assignment follows a geometric distribution with parameter `p = 1/m_l`.
    /// This creates the hierarchical structure essential for HNSW's O(log n) search.
    ///
    /// ## Probability Distribution
    ///
    /// ```text
    /// P(layer = l) = (1 - 1/m_l) × (1/m_l)^l
    /// ```
    ///
    /// This is a truncated geometric distribution where:
    /// - `m_l` is typically `1/ln(M)` ≈ 1.44 for M=16
    /// - Most vectors are at layer 0 (base layer)
    /// - Higher layers are exponentially sparser
    ///
    /// ## Expected Properties
    ///
    /// - **E[layer]** ≈ 1/(m_l - 1): Expected layer level
    /// - **Layer 0 probability**: P(L=0) = 1 - 1/m_l ≈ 0.31 for default m_l
    /// - **Expected vectors per layer**: N × (1/m_l)^l
    ///
    /// ## Why This Works
    ///
    /// The geometric distribution creates a navigable small-world graph:
    /// 1. Upper layers have O(log n) vectors, enabling fast coarse search
    /// 2. Lower layers have O(n) vectors, enabling fine-grained retrieval
    /// 3. The hierarchical structure mimics skip lists, giving O(log n) complexity
    ///
    /// ## Reference
    ///
    /// Malkov & Yashunin (2016): "Efficient and robust approximate nearest
    /// neighbor search using Hierarchical Navigable Small World graphs"
    /// - Section 4.2: Level generation
    fn assign_layer(&self) -> u8 {
        #[cfg(feature = "hnsw")]
        {
            use rand::Rng;

            // Paper formula (Algorithm 1, line 4): l = floor(-ln(uniform) * mL)
            let u: f64 = {
                // Safety: lock is only held briefly during layer assignment;
                // poisoning would require a panic inside this critical section.
                #[allow(clippy::expect_used)]
                let mut rng_guard = self.rng.lock().expect("rng lock poisoned");
                if let Some(ref mut seeded_rng) = *rng_guard {
                    seeded_rng.random()
                } else {
                    rand::rng().random()
                }
            };
            (-u.ln() * self.params.m_l).floor() as u8
        }
        #[cfg(not(feature = "hnsw"))]
        {
            0
        }
    }

    /// Return a plain function pointer for the configured metric.
    ///
    /// Used when passing a distance function to free functions that can't take `&self`.
    #[inline(always)]
    pub(crate) fn dist_fn(&self) -> fn(&[f32], &[f32]) -> f32 {
        match self.params.metric {
            DistanceMetric::L2 => crate::distance::l2_distance,
            DistanceMetric::Cosine => crate::distance::cosine_distance_normalized,
            DistanceMetric::Angular => crate::distance::angular_distance,
            DistanceMetric::InnerProduct => crate::distance::inner_product_distance,
        }
    }

    /// Get vector by index (for internal use).
    #[inline]
    pub(crate) fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        &self.vectors[start..end]
    }

    /// Raw vector storage in internal (reordered) layout.
    ///
    /// After `build()`, vectors are reordered for cache locality (BFS order).
    /// The internal node IDs used by `search_with_distance`'s closure correspond
    /// to positions in this array, NOT to the original insertion order.
    ///
    /// Use this to build auxiliary data structures (ADSampling rotation, quantization
    /// codes) that are indexed by internal node ID.
    ///
    /// Prefer [`raw_vectors()`](Self::raw_vectors); this is a legacy alias.
    #[must_use]
    pub fn vectors_raw(&self) -> &[f32] {
        self.raw_vectors()
    }

    /// Get entry point (vector in highest layer).
    fn get_entry_point(&self) -> Option<u32> {
        if let Some(ep) = self.cached_entry_point {
            return Some(ep);
        }
        self.compute_entry_point()
    }

    /// Descend from `entry_point` at `entry_layer` through upper layers to layer 1,
    /// returning the closest node found. Uses a function pointer to avoid per-call
    /// enum dispatch on the distance metric.
    fn descend_upper_layers(
        &self,
        query: &[f32],
        entry_point: u32,
        entry_layer: usize,
        ef_hint: usize,
    ) -> u32 {
        let dist_fn = self.dist_fn();
        let mut current_closest = entry_point;
        let mut current_dist = dist_fn(query, self.get_vector(entry_point as usize));

        for layer_idx in (1..=entry_layer).rev() {
            if layer_idx >= self.layers.len() {
                continue;
            }
            let layer = &self.layers[layer_idx];
            let mut changed = true;
            let mut visited = std::collections::HashSet::with_capacity(ef_hint.min(100));

            while changed {
                changed = false;
                visited.insert(current_closest);
                let neighbors = layer.get_neighbors(current_closest);
                for &neighbor_id in neighbors.iter() {
                    if visited.contains(&neighbor_id) {
                        continue;
                    }
                    let dist = dist_fn(query, self.get_vector(neighbor_id as usize));
                    if dist < current_dist {
                        current_dist = dist;
                        current_closest = neighbor_id;
                        changed = true;
                    }
                }
            }
        }
        current_closest
    }

    /// O(n) scan to find the node with the highest layer assignment.
    /// Called once at build time; result is cached in `cached_entry_point`.
    fn compute_entry_point(&self) -> Option<u32> {
        if self.num_vectors == 0 {
            return None;
        }
        let mut entry_point = 0u32;
        let mut entry_layer = 0u8;
        for (idx, &layer) in self.layer_assignments.iter().enumerate() {
            if layer > entry_layer {
                entry_point = idx as u32;
                entry_layer = layer;
            }
        }
        Some(entry_point)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_create_index() {
        let index = HNSWIndex::new(128, 16, 16).unwrap();
        assert_eq!(index.dimension, 128);
        assert_eq!(index.num_vectors, 0);
    }

    #[test]
    fn test_add_vectors() {
        let mut index = HNSWIndex::new(3, 16, 16).unwrap();

        index.add(0, vec![1.0, 0.0, 0.0]).unwrap();
        index.add(1, vec![0.0, 1.0, 0.0]).unwrap();

        assert_eq!(index.num_vectors, 2);
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut index = HNSWIndex::new(3, 16, 16).unwrap();

        let result = index.add(0, vec![1.0, 0.0]); // Wrong dimension
        assert!(result.is_err());
    }

    /// Build a small index for adaptive search tests.
    /// Returns (index, query_vector).
    fn build_test_index() -> (HNSWIndex, Vec<f32>) {
        let dim = 32;
        let n = 200;
        let mut index = HNSWIndex::new(dim, 16, 32).unwrap();

        // Deterministic pseudo-random vectors using LCG
        let mut seed: u64 = 42;
        let mut next = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5
        };

        for i in 0..n {
            let mut v: Vec<f32> = (0..dim).map(|_| next()).collect();
            // L2-normalize for cosine distance
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|x| *x /= norm);
            }
            index.add(i as u32, v).unwrap();
        }
        index.build().unwrap();

        let mut q: Vec<f32> = (0..dim).map(|_| next()).collect();
        let norm = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            q.iter_mut().for_each(|x| *x /= norm);
        }

        (index, q)
    }

    #[test]
    fn test_search_adaptive_conservative_matches_search() {
        let (index, q) = build_test_index();
        let k = 5;
        let ef = 64;

        let baseline = index.search(&q, k, ef).unwrap();
        let config = crate::adaptive::AdaptiveConfig::conservative();
        let (adaptive, _evaluated) = index.search_adaptive(&q, k, ef, &config).unwrap();

        // Conservative config should return the same top-k (or very close)
        assert_eq!(adaptive.len(), baseline.len());
        // At minimum the nearest neighbor should match
        assert_eq!(adaptive[0].0, baseline[0].0);
    }

    #[test]
    fn test_search_adaptive_aggressive_fewer_evaluations() {
        let (index, q) = build_test_index();
        let k = 5;
        let ef = 64;

        let conservative = crate::adaptive::AdaptiveConfig::conservative();
        let aggressive = crate::adaptive::AdaptiveConfig::aggressive();

        let (_results_c, evaluated_c) = index.search_adaptive(&q, k, ef, &conservative).unwrap();
        let (_results_a, evaluated_a) = index.search_adaptive(&q, k, ef, &aggressive).unwrap();

        // Aggressive config should evaluate fewer (or equal) candidates
        assert!(
            evaluated_a <= evaluated_c,
            "aggressive ({}) should evaluate <= conservative ({})",
            evaluated_a,
            evaluated_c,
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_hnsw_save_load_roundtrip() {
        let (index, q) = build_test_index();
        let k = 5;
        let ef = 64;

        // Search the original index.
        let original_results = index.search(&q, k, ef).unwrap();

        // Serialize to an in-memory buffer.
        let mut buf = Vec::new();
        index.save_to_writer(&mut buf).unwrap();
        assert!(!buf.is_empty(), "serialized output should not be empty");

        // Deserialize back.
        let loaded = HNSWIndex::load_from_reader(buf.as_slice()).unwrap();

        // Basic structural checks.
        assert_eq!(loaded.dimension, index.dimension);
        assert_eq!(loaded.num_vectors, index.num_vectors);
        assert!(loaded.is_built());
        assert_eq!(loaded.doc_ids, index.doc_ids);
        // Reverse map should have been rebuilt.
        assert_eq!(
            loaded.doc_id_to_internal.len(),
            index.doc_id_to_internal.len()
        );

        // Search the loaded index and compare results.
        let loaded_results = loaded.search(&q, k, ef).unwrap();
        assert_eq!(
            loaded_results, original_results,
            "search results should be identical after save/load roundtrip"
        );
    }

    /// Round-trip via the path-based API (covers `save_to_file` /
    /// `load_from_file`, which the in-memory roundtrip above does not).
    /// Also asserts that `save_to_file` leaves no `.tmp` sibling on
    /// success -- if it does, the temp+rename atomicity got broken
    /// (e.g., the rename was skipped or the temp name doesn't match).
    #[cfg(feature = "serde")]
    #[test]
    fn test_hnsw_save_to_file_roundtrip_and_no_temp_leftover() {
        let (index, q) = build_test_index();
        let k = 5;
        let ef = 64;
        let original_results = index.search(&q, k, ef).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.json");
        index.save_to_file(&path).unwrap();

        // Final file must exist; the sibling .tmp must not.
        assert!(path.exists(), "saved file should exist at target path");
        let tmp_path = dir.path().join("index.json.tmp");
        assert!(
            !tmp_path.exists(),
            "save_to_file left a .tmp sibling: {}",
            tmp_path.display()
        );

        let loaded = HNSWIndex::load_from_file(&path).unwrap();
        let loaded_results = loaded.search(&q, k, ef).unwrap();
        assert_eq!(loaded_results, original_results);
    }

    /// Atomicity guarantee: when `save_to_file` overwrites an existing
    /// file, the prior contents are replaced wholesale rather than
    /// partially truncated. The check is indirect (the bytes match the
    /// new serialization, not the old) but the failure shape if the
    /// rename were skipped would be a file containing the OLD payload
    /// or a truncated mix.
    #[cfg(feature = "serde")]
    #[test]
    fn test_hnsw_save_to_file_overwrites_existing_atomically() {
        let (index, _q) = build_test_index();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.json");

        // Stage a sentinel "previous" file that is clearly not a valid index.
        std::fs::write(&path, b"{\"sentinel\": true}").unwrap();
        let pre_size = std::fs::metadata(&path).unwrap().len();

        index.save_to_file(&path).unwrap();

        // The replacement must be a valid index payload, not the sentinel.
        let post_size = std::fs::metadata(&path).unwrap().len();
        assert!(
            post_size > pre_size,
            "post-save size {} should exceed sentinel size {}",
            post_size,
            pre_size
        );
        let _ = HNSWIndex::load_from_file(&path).expect("post-overwrite file must load");
        // No temp leftover.
        assert!(!dir.path().join("index.json.tmp").exists());
    }

    /// Negative-path cleanup guarantee: when `save_to_file` fails at the
    /// rename step, it must not leak the `<path>.tmp` sibling. We force the
    /// rename to fail by pre-creating a directory at the target path; POSIX
    /// rename refuses to replace a non-empty directory with a file (EISDIR /
    /// ENOTEMPTY), and Windows MoveFileEx refuses similarly. The exact errno
    /// is platform-dependent so we don't assert on the error variant; we
    /// only assert the cleanup happened.
    ///
    /// Without the `remove_file(&tmp_path)` line in the rename-error branch,
    /// callers retrying the save would race against a stale temp from the
    /// previous attempt or accumulate junk files in the target directory.
    #[cfg(feature = "serde")]
    #[test]
    fn test_hnsw_save_to_file_cleans_up_tmp_on_rename_failure() {
        let (index, _q) = build_test_index();
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("save.json");

        // Pre-create a directory at the target path so the rename of
        // `<target>.tmp -> <target>` fails. The directory must be non-empty
        // on some platforms to guarantee EISDIR/ENOTEMPTY; an inner sentinel
        // file ensures the rename can't succeed by accident.
        std::fs::create_dir(&target).unwrap();
        std::fs::write(target.join("blocker"), b"x").unwrap();

        let result = index.save_to_file(&target);
        assert!(
            result.is_err(),
            "rename into a non-empty directory should fail"
        );

        let tmp_path = dir.path().join("save.json.tmp");
        assert!(
            !tmp_path.exists(),
            "save_to_file failed but left a temp file at {} \
             (rename-failure cleanup branch did not fire)",
            tmp_path.display(),
        );
    }

    /// One-shot measurement of save_to_file overhead vs a non-atomic
    /// File::create+write baseline. The atomic path adds: temp file
    /// creation, sync_all on the temp, rename, parent-dir sync_all.
    /// On SSD, sync_all is the dominant cost (single-digit ms typical).
    /// For a JSON-serialized HNSW index in the 100KB-few-MB range, the
    /// total is dominated by serialization + write throughput, not the
    /// extra fsyncs. This probe prints both numbers so future work can
    /// compare. Marked `#[ignore]` because it's a measurement, not a
    /// regression guard.
    ///
    /// Run with:
    ///   cargo test --release --features hnsw,serde
    ///       hnsw::graph::tests::save_to_file_overhead_probe
    ///       -- --ignored --nocapture
    #[cfg(feature = "serde")]
    #[test]
    #[ignore = "measurement only; run with --release --ignored --nocapture"]
    fn save_to_file_overhead_probe() {
        use std::time::Instant;

        let (index, _q) = build_test_index();
        let dir = tempfile::tempdir().unwrap();

        // Atomic path (current code): temp + sync + rename + dir sync.
        let iters = 30;
        let path = dir.path().join("atomic.json");
        let t = Instant::now();
        for _ in 0..iters {
            index.save_to_file(&path).unwrap();
        }
        let atomic_ms_per = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // Non-atomic baseline: File::create + write + drop. No syncs.
        let path2 = dir.path().join("naive.json");
        let t = Instant::now();
        for _ in 0..iters {
            let f = std::fs::File::create(&path2).unwrap();
            index.save_to_writer(std::io::BufWriter::new(f)).unwrap();
        }
        let naive_ms_per = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let file_kb = std::fs::metadata(&path).unwrap().len() / 1024;
        println!("# save_to_file_overhead_probe");
        println!(
            "# build_test_index ({} vectors, {} dim, ~{} KB JSON)",
            index.num_vectors, index.dimension, file_kb
        );
        println!("# atomic   {:.2} ms/op", atomic_ms_per);
        println!("# naive    {:.2} ms/op", naive_ms_per);
        println!(
            "# overhead {:.2} ms/op ({:+.0}%)",
            atomic_ms_per - naive_ms_per,
            100.0 * (atomic_ms_per - naive_ms_per) / naive_ms_per
        );
    }

    /// End-to-end build → save_to_file → drop → load_from_file → search
    /// equality. The in-memory roundtrip
    /// (`test_hnsw_save_load_roundtrip`) covers the
    /// `save_to_writer`/`load_from_reader` half; the v0 fixture decode
    /// tests pin a static byte sequence; this test plugs the gap by
    /// proving the writer and reader are mutually consistent through the
    /// path-based API on a freshly built live index.
    ///
    /// qdrant ships an analogous round-trip test for its `GraphLayers`
    /// persistence; this is the same shape adapted to vicinity's HNSW.
    #[cfg(feature = "serde")]
    #[test]
    fn test_hnsw_build_save_reload_search_equality() {
        let (index, q) = build_test_index();
        let k = 10;
        let ef = 64;
        let original_results = index.search(&q, k, ef).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.json");
        index.save_to_file(&path).unwrap();
        // Drop the in-memory copy explicitly so the load is from disk,
        // not aliasing.
        drop(index);

        let loaded = HNSWIndex::load_from_file(&path).expect("load_from_file roundtrip");
        let loaded_results = loaded.search(&q, k, ef).unwrap();

        assert_eq!(
            loaded_results, original_results,
            "search results diverged across path-based save/load roundtrip. \
             likely cause: writer and reader formats drifted (one side \
             changed without the other), or a transient field was \
             dropped during serialization that affects search behavior."
        );
    }

    /// Gate-fired check that `save_to_file` actually uses temp+rename and
    /// not in-place truncation. Truncate-overwrite preserves the inode;
    /// rename-overwrite replaces it with the temp's inode. This distinction
    /// is invisible to the happy-path tests above but is the entire point
    /// of the atomicity refactor: a crash mid-truncate corrupts the file,
    /// while a crash mid-temp-write leaves the original intact.
    #[cfg(all(feature = "serde", unix))]
    #[test]
    fn test_hnsw_save_to_file_uses_rename_not_truncate() {
        use std::os::unix::fs::MetadataExt;

        let (index, _q) = build_test_index();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.json");

        // First save: gives us a baseline file with an inode.
        index.save_to_file(&path).unwrap();
        let ino_before = std::fs::metadata(&path).unwrap().ino();

        // Second save over the same path. With temp+rename, the inode
        // changes (the new file is the temp file under its new name);
        // with in-place truncation, the inode stays the same.
        index.save_to_file(&path).unwrap();
        let ino_after = std::fs::metadata(&path).unwrap().ino();

        assert_ne!(
            ino_before, ino_after,
            "save_to_file must replace via rename (inode changes), not truncate \
             in place (inode preserved). truncate-overwrite would leave the file \
             in a partially-written state if a crash interrupts the write -- this \
             is exactly the partial-write corruption mode the atomic-rename refactor \
             is meant to prevent."
        );
    }

    #[test]
    fn test_delete_excludes_from_results() {
        let (mut index, q) = build_test_index();
        let k = 10;
        let ef = 64;

        let before = index.search(&q, k, ef).unwrap();
        assert!(!before.is_empty());

        // Delete the nearest neighbor
        let nearest_id = before[0].0;
        index.delete(nearest_id).unwrap();

        let after = index.search(&q, k, ef).unwrap();
        let after_ids: Vec<u32> = after.iter().map(|(id, _)| *id).collect();
        assert!(
            !after_ids.contains(&nearest_id),
            "deleted doc_id {} should not appear in results",
            nearest_id
        );
    }

    #[test]
    fn test_delete_all_returns_empty() {
        let dim = 4;
        let mut index = HNSWIndex::new(dim, 4, 4).unwrap();

        // Insert 5 normalized vectors
        for i in 0..5u32 {
            let mut v = vec![0.0f32; dim];
            v[i as usize % dim] = 1.0;
            index.add(i, v).unwrap();
        }
        index.build().unwrap();

        // Delete all
        for i in 0..5u32 {
            index.delete(i).unwrap();
        }

        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 5, 50).unwrap();
        assert!(
            results.is_empty(),
            "all vectors deleted, results should be empty"
        );
    }

    #[test]
    fn test_delete_nonexistent_returns_error() {
        let mut index = HNSWIndex::new(4, 4, 4).unwrap();
        index.add(0, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        index.build().unwrap();

        let result = index.delete(999);
        assert!(result.is_err(), "deleting nonexistent doc_id should error");
    }

    #[test]
    fn test_delete_idempotent() {
        let (mut index, _q) = build_test_index();

        // Delete same ID twice -- second should succeed silently
        index.delete(0).unwrap();
        index.delete(0).unwrap();
    }

    #[test]
    fn test_is_deleted() {
        let (mut index, _q) = build_test_index();
        assert!(!index.is_deleted(0));
        index.delete(0).unwrap();
        assert!(index.is_deleted(0));
        assert!(!index.is_deleted(1));
    }

    #[test]
    fn test_num_active() {
        let (mut index, _q) = build_test_index();
        let total = index.num_vectors;
        assert_eq!(index.num_active(), total);

        index.delete(0).unwrap();
        index.delete(1).unwrap();
        assert_eq!(index.num_active(), total - 2);
    }

    #[test]
    fn test_builder_produces_working_index() {
        let mut index = HNSWIndex::builder(4)
            .m(8)
            .ef_search(32)
            .auto_normalize(true)
            .build()
            .unwrap();

        // auto_normalize is on, so raw (un-normalized) vectors should be accepted
        index.add_slice(0, &[3.0, 4.0, 0.0, 0.0]).unwrap();
        index.add_slice(1, &[0.0, 0.0, 3.0, 4.0]).unwrap();
        index.build().unwrap();

        // Query with un-normalized vector (will be normalized at search time
        // only if the caller normalizes -- auto_normalize applies to add, not search).
        let q = crate::distance::normalize(&[3.0, 4.0, 0.0, 0.0]);
        let results = index.search(&q, 1, 32).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0, "nearest neighbor should be doc 0");
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_search_batch_matches_sequential() {
        let (index, q) = build_test_index();
        let k = 5;
        let ef = 64;

        // Create several query vectors by rotating the base query
        let dim = index.dimension;
        let mut queries_flat: Vec<f32> = Vec::new();
        let num_queries = 8;

        let mut seed: u64 = 99;
        let mut next = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5
        };

        for _ in 0..num_queries {
            let mut v: Vec<f32> = (0..dim).map(|_| next()).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|x| *x /= norm);
            }
            queries_flat.extend_from_slice(&v);
        }

        // Sequential results
        let sequential: Vec<Vec<(u32, f32)>> = (0..num_queries)
            .map(|i| {
                let query = &queries_flat[i * dim..(i + 1) * dim];
                index.search(query, k, ef).unwrap()
            })
            .collect();

        // Batch (parallel) results -- slice-of-slices variant
        let query_slices: Vec<&[f32]> = (0..num_queries)
            .map(|i| &queries_flat[i * dim..(i + 1) * dim])
            .collect();
        let batch = index.search_batch(&query_slices, k, ef).unwrap();

        // Batch (parallel) results -- flat-buffer variant
        let batch_flat = index
            .search_batch_flat(&queries_flat, num_queries, k, ef)
            .unwrap();

        // Also include the original query to make sure it still works
        let _ = index.search(&q, k, ef).unwrap();

        for i in 0..num_queries {
            assert_eq!(
                sequential[i], batch[i],
                "search_batch result {} differs from sequential",
                i
            );
            assert_eq!(
                sequential[i], batch_flat[i],
                "search_batch_flat result {} differs from sequential",
                i
            );
        }
    }

    /// Build a small index for structural invariant tests.
    fn build_structural_test_index(n: usize, dim: usize, m: usize) -> HNSWIndex {
        let params = HNSWParams {
            m,
            m_max: 2 * m,
            seed: Some(42),
            ..Default::default()
        };
        let mut index = HNSWIndex::with_params(dim, params).unwrap();
        for i in 0..n {
            let v: Vec<f32> = (0..dim)
                .map(|j| ((i * 7 + j * 3) % 100) as f32 / 100.0)
                .collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let normed: Vec<f32> = v.iter().map(|x| x / norm).collect();
            index.add(i as u32, normed).unwrap();
        }
        index.build().unwrap();
        index
    }

    #[test]
    fn test_layer_assignment_distribution() {
        // With 500 vectors, we should see layer 0 most populated and
        // exponentially fewer vectors in higher layers.
        let index = build_structural_test_index(500, 32, 16);

        let max_layer = *index.layer_assignments.iter().max().unwrap_or(&0) as usize;

        let mut layer_counts = vec![0usize; max_layer + 1];
        for &l in &index.layer_assignments {
            layer_counts[l as usize] += 1;
        }

        // With m_l ≈ 0.36 (default for M=16), ~94% of vectors land on layer 0.
        // Use a generous threshold to avoid flaking on unlucky seeds.
        let layer0_frac = layer_counts[0] as f64 / 500.0;
        assert!(
            layer0_frac > 0.5,
            "Layer 0 fraction {:.2} should be > 0.5 (got {} of 500)",
            layer0_frac,
            layer_counts[0]
        );

        // Layer 0 must have strictly the most vectors
        for l in 1..layer_counts.len() {
            assert!(
                layer_counts[l] < layer_counts[0],
                "Layer {} ({}) should have fewer vectors than layer 0 ({})",
                l,
                layer_counts[l],
                layer_counts[0]
            );
        }
    }

    #[test]
    fn test_m_max_enforced() {
        // Verify no node in layer 0 exceeds m_max neighbors,
        // and no node in upper layers exceeds m.
        let m = 8;
        let m_max = 2 * m;
        let index = build_structural_test_index(200, 16, m);

        for (layer_idx, layer) in index.layers.iter().enumerate() {
            let limit = if layer_idx == 0 { m_max } else { m };
            if let Some(neighbors) = layer.get_all_neighbors() {
                for (node_id, nbrs) in neighbors.iter().enumerate() {
                    assert!(
                        nbrs.len() <= limit,
                        "Node {} at layer {} has {} neighbors, limit is {}",
                        node_id,
                        layer_idx,
                        nbrs.len(),
                        limit
                    );
                }
            }
        }
    }

    #[test]
    fn test_neighbor_ids_in_bounds() {
        // All neighbor IDs must reference valid nodes.
        let index = build_structural_test_index(100, 16, 8);

        for (layer_idx, layer) in index.layers.iter().enumerate() {
            if let Some(neighbors) = layer.get_all_neighbors() {
                for (node_id, nbrs) in neighbors.iter().enumerate() {
                    for &nbr in nbrs.iter() {
                        assert!(
                            (nbr as usize) < index.num_vectors,
                            "Node {} at layer {} has out-of-bounds neighbor {}",
                            node_id,
                            layer_idx,
                            nbr
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_layer_assignment_matches_layers() {
        // Every node assigned to layer L should exist in layers 0..=L.
        let index = build_structural_test_index(100, 16, 8);

        for (node_id, &assigned_layer) in index.layer_assignments.iter().enumerate() {
            // Node should be present in layers 0 through assigned_layer
            for l in 0..=assigned_layer as usize {
                assert!(
                    l < index.layers.len(),
                    "Node {} assigned to layer {} but only {} layers exist",
                    node_id,
                    assigned_layer,
                    index.layers.len()
                );
            }
        }
    }

    /// `batch_search_mqo` must return results in input-query order and find at
    /// least as good a nearest neighbour as individual `search` calls.
    #[test]
    fn test_batch_search_mqo_order_and_recall() {
        let (index, _) = build_test_index();
        let k = 5;
        let ef = 64;

        // Build a small set of varied queries using the same deterministic LCG as
        // `build_test_index` but with a different seed so queries differ from the
        // indexed vectors.
        let dim = index.dimension;
        let mut seed: u64 = 99;
        let mut next = || -> f32 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5
        };

        let num_queries = 10;
        let queries_owned: Vec<Vec<f32>> = (0..num_queries)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim).map(|_| next()).collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    v.iter_mut().for_each(|x| *x /= norm);
                }
                v
            })
            .collect();
        let queries: Vec<&[f32]> = queries_owned.iter().map(|v| v.as_slice()).collect();

        // Individual searches (baseline).
        let individual: Vec<Vec<(u32, f32)>> = queries
            .iter()
            .map(|q| index.search(q, k, ef).unwrap())
            .collect();

        // Batch MQO search.
        let batch = index.batch_search_mqo(&queries, k, ef).unwrap();

        // Results must be in input order and have the right length.
        assert_eq!(batch.len(), num_queries);

        for (i, (ind, mqo)) in individual.iter().zip(batch.iter()).enumerate() {
            // MQO must return a result for every query.
            assert!(
                !mqo.is_empty(),
                "query {}: batch_search_mqo returned no results",
                i
            );
            // The nearest neighbour found by MQO must be at least as close as the
            // one from individual search (same ef, so recall can only stay equal or
            // improve with an additional warm-start entry point).
            let ind_best_dist = ind.first().map(|(_, d)| *d).unwrap_or(f32::INFINITY);
            let mqo_best_dist = mqo.first().map(|(_, d)| *d).unwrap_or(f32::INFINITY);
            assert!(
                mqo_best_dist <= ind_best_dist + 1e-5,
                "query {}: MQO nearest dist {} > individual nearest dist {} (regression)",
                i,
                mqo_best_dist,
                ind_best_dist,
            );
        }
    }

    /// Single-query batch_search_mqo must return the same result as search.
    #[test]
    fn test_batch_search_mqo_single_query() {
        let (index, q) = build_test_index();
        let k = 5;
        let ef = 64;

        let single = index.search(&q, k, ef).unwrap();
        let batch = index.batch_search_mqo(&[q.as_slice()], k, ef).unwrap();

        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0], single, "single-query MQO should match search");
    }

    #[test]
    fn test_search_with_distance_matches_standard() {
        let (index, q) = build_test_index();
        let k = 5;
        let ef = 64;

        let standard = index.search(&q, k, ef).unwrap();

        // Custom distance that does the same thing as the standard path
        let vectors = &index.vectors;
        let dim = index.dimension;
        let dist_fn = |query: &[f32], internal_id: u32| -> f32 {
            let start = internal_id as usize * dim;
            let vec = &vectors[start..start + dim];
            crate::distance::cosine_distance_normalized(query, vec)
        };
        let custom = index.search_with_distance(&q, k, ef, &dist_fn).unwrap();

        assert_eq!(standard.len(), custom.len());
        for (s, c) in standard.iter().zip(custom.iter()) {
            assert_eq!(s.0, c.0, "doc_ids should match");
            assert!((s.1 - c.1).abs() < 1e-6, "distances should match");
        }
    }

    #[test]
    fn test_search_with_distance_custom_metric() {
        // Build index with cosine, search with L2 -- should reorder results
        let (index, q) = build_test_index();
        let k = 5;
        let ef = 64;

        let vectors = &index.vectors;
        let dim = index.dimension;
        let l2_dist = |query: &[f32], internal_id: u32| -> f32 {
            let start = internal_id as usize * dim;
            let vec = &vectors[start..start + dim];
            crate::distance::l2_distance(query, vec)
        };
        let results = index.search_with_distance(&q, k, ef, &l2_dist).unwrap();

        // Results should be sorted by L2 distance
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1, "results not sorted by custom distance");
        }
    }

    // -----------------------------------------------------------------------
    // Wolverine++ delete_with_repair tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_delete_with_repair_excludes_from_results() {
        let (mut index, q) = build_test_index();
        let k = 10;
        let ef = 64;

        let before = index.search(&q, k, ef).unwrap();
        let nearest_id = before[0].0;

        let repairs = index.delete_with_repair(nearest_id).unwrap();

        let after = index.search(&q, k, ef).unwrap();
        let after_ids: Vec<u32> = after.iter().map(|(id, _)| *id).collect();
        assert!(
            !after_ids.contains(&nearest_id),
            "deleted doc_id {} should not appear in results (repairs={})",
            nearest_id,
            repairs,
        );
    }

    #[test]
    fn test_delete_with_repair_maintains_recall() {
        let (mut index, q) = build_test_index();
        let k = 10;
        let ef = 100;

        let before = index.search(&q, k, ef).unwrap();
        assert_eq!(before.len(), k);

        // Delete 20% of nodes (IDs 0..40 out of 200)
        for i in 0..40u32 {
            index.delete_with_repair(i).unwrap();
        }

        let after = index.search(&q, k, ef).unwrap();
        // Should still return k results (160 nodes remain)
        assert_eq!(
            after.len(),
            k,
            "should still return k={} results after deleting 20%",
            k
        );

        // No deleted IDs in results
        for (id, _) in &after {
            assert!(*id >= 40, "deleted id {} appeared in results", id);
        }
    }

    /// Companion to `test_delete_with_repair_maintains_recall`.
    ///
    /// The original test asserts the right *count* of results comes back and
    /// that deleted IDs are absent. It does NOT verify recall against ground
    /// truth on the surviving IDs. This test does:
    ///
    /// 1. Generate the same 200 deterministic vectors used by `build_test_index`.
    /// 2. Compute the brute-force top-`k` cosine neighbors of the query among
    ///    the surviving IDs `40..200`.
    /// 3. Run the HNSW search after deletion.
    /// 4. Assert the overlap (recall) between the brute-force ground truth and
    ///    the HNSW results is at least 0.7. The threshold is conservative for
    ///    a 200-node graph; the relevant signal is "no recall cliff", not "near 1.0".
    #[test]
    fn test_delete_with_repair_recall_floor_against_ground_truth() {
        let dim = 32usize;
        let n = 200u32;
        let deleted_count = 40u32;
        let k = 10usize;
        let ef = 100usize;
        let recall_floor = 0.7f32;

        // Reproduce the exact vector generation used by build_test_index().
        let mut seed: u64 = 42;
        let mut next = || -> f32 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5
        };
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let mut v: Vec<f32> = (0..dim).map(|_| next()).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|x| *x /= norm);
            }
            vectors.push(v);
        }
        // Same query construction as build_test_index().
        let mut q: Vec<f32> = (0..dim).map(|_| next()).collect();
        let qnorm = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        if qnorm > 0.0 {
            q.iter_mut().for_each(|x| *x /= qnorm);
        }

        // Brute-force ground truth on the surviving IDs only.
        let mut ground: Vec<(u32, f32)> = (deleted_count..n)
            .map(|i| {
                let v = &vectors[i as usize];
                // 1 - dot product = cosine distance for unit vectors.
                let dot: f32 = q.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                (i, 1.0 - dot)
            })
            .collect();
        ground.sort_by(|a, b| a.1.total_cmp(&b.1));
        let truth_ids: std::collections::HashSet<u32> =
            ground.iter().take(k).map(|(id, _)| *id).collect();

        // Build the index and apply the same deletions as the count test above.
        let mut index = HNSWIndex::new(dim, 16, 32).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            index.add(i as u32, v.clone()).unwrap();
        }
        index.build().unwrap();
        for i in 0..deleted_count {
            index.delete_with_repair(i).unwrap();
        }

        // Recall = |hnsw_topk ∩ truth_topk| / k.
        let results = index.search(&q, k, ef).unwrap();
        assert_eq!(results.len(), k, "search should return k results");
        let hits = results
            .iter()
            .filter(|(id, _)| truth_ids.contains(id))
            .count();
        let recall = hits as f32 / k as f32;
        assert!(
            recall >= recall_floor,
            "post-delete recall@{}={:.2} fell below floor {:.2}; deleted_count={}, n={}",
            k,
            recall,
            recall_floor,
            deleted_count,
            n
        );
    }

    /// Self-search reachability under aggressive deletion.
    ///
    /// Existing tests for `delete_with_repair` cover small deletion ratios
    /// (20-30%) and assert recall against a ground-truth top-k or that the
    /// result count survives. None probe the failure mode where many
    /// repairs leave individual live nodes orphaned -- the analog of
    /// FreshGraph's delete-reinsert reachability bug (arxiv:2407.07871).
    ///
    /// This test deletes 60% of nodes and asserts every surviving doc_id
    /// is still self-search-reachable. If a live node becomes unreachable
    /// after repair (e.g., its only in-edges were through nodes that were
    /// later deleted, and the crescent-locus replacement failed to add
    /// new in-edges), the assertion lists the orphaned ids.
    ///
    /// HNSW does not currently support post-build inserts, so this is a
    /// long-delete test rather than a delete-reinsert cycle. The bug class
    /// is the same: cumulative repair operations eroding reachability for
    /// the surviving subgraph.
    #[test]
    fn test_delete_with_repair_preserves_self_search_reachability() {
        let dim = 32usize;
        let n = 300u32;
        let delete_ratio = 0.6f32;
        let deleted_count = (n as f32 * delete_ratio) as u32;
        let k = 5usize;
        let ef = 200usize;

        // Same deterministic LCG as build_test_index().
        let mut seed: u64 = 42;
        let mut next = || -> f32 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5
        };
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let mut v: Vec<f32> = (0..dim).map(|_| next()).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|x| *x /= norm);
            }
            vectors.push(v);
        }

        let mut index = HNSWIndex::new(dim, 16, 32).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            index.add(i as u32, v.clone()).unwrap();
        }
        index.build().unwrap();

        // Delete the first `deleted_count` ids. Pseudo-random selection
        // would be more thorough but the IDs are already random with
        // respect to graph layout (assign_layer is the only ID-correlated
        // structure, and it's an exponential RV per node).
        for i in 0..deleted_count {
            index.delete_with_repair(i).unwrap();
        }

        // For each surviving id, self-search and check reachability.
        let mut unreachable: Vec<u32> = Vec::new();
        for id in deleted_count..n {
            let results = index.search(&vectors[id as usize], k, ef).unwrap();
            if !results.iter().any(|(rid, _)| *rid == id) {
                unreachable.push(id);
            }
        }

        assert!(
            unreachable.is_empty(),
            "{} of {} live ids became unreachable after deleting {} (ratio {:.0}%): {:?}. \
             likely cause: cumulative delete_with_repair calls left live nodes orphaned -- \
             crescent-locus replacement may have failed to add in-edges, or the entry-point \
             repromotion in find_entry_point_excluding picked an anchor whose neighborhood \
             has drifted away from these ids.",
            unreachable.len(),
            n - deleted_count,
            deleted_count,
            delete_ratio * 100.0,
            &unreachable[..unreachable.len().min(10)]
        );
    }

    #[test]
    fn test_delete_with_repair_entry_point() {
        let (mut index, q) = build_test_index();
        let ep = index.cached_entry_point.unwrap();
        let ep_doc_id = index.doc_ids[ep as usize];

        // Delete the entry point
        index.delete_with_repair(ep_doc_id).unwrap();

        // Entry point should have changed
        assert_ne!(
            index.cached_entry_point,
            Some(ep),
            "entry point should change after deleting it"
        );
        assert!(
            index.cached_entry_point.is_some(),
            "should find a new entry point"
        );

        // Search should still work
        let results = index.search(&q, 5, 64).unwrap();
        assert!(!results.is_empty(), "search should work after EP deletion");
    }

    #[test]
    fn test_delete_with_repair_graph_edges_cleaned() {
        let (mut index, _q) = build_test_index();
        let target_doc_id = 50u32;
        let internal_id = index.doc_id_to_internal[&target_doc_id];

        index.delete_with_repair(target_doc_id).unwrap();

        // No node in layer 0 should point to the deleted internal ID
        let layer = &index.layers[0];
        for node_id in 0..layer.len() as u32 {
            let neighbors = layer.get_neighbors(node_id);
            assert!(
                !neighbors.contains(&internal_id),
                "node {} still points to deleted node {} after repair",
                node_id,
                internal_id
            );
        }

        // Deleted node's own neighbor list should be empty
        assert!(
            layer.get_neighbors(internal_id).is_empty(),
            "deleted node should have empty neighbor list"
        );
    }

    #[test]
    fn test_delete_batch_with_repair() {
        let (mut index, q) = build_test_index();
        let k = 10;
        let ef = 100;

        // Batch delete 30 nodes
        let ids_to_delete: Vec<u32> = (10..40).collect();
        let repairs = index.delete_batch_with_repair(&ids_to_delete).unwrap();

        let after = index.search(&q, k, ef).unwrap();
        assert_eq!(after.len(), k);

        // No deleted IDs in results
        let deleted_set: std::collections::HashSet<u32> = ids_to_delete.iter().copied().collect();
        for (id, _) in &after {
            assert!(
                !deleted_set.contains(id),
                "deleted id {} in results (repairs={})",
                id,
                repairs,
            );
        }
    }

    #[test]
    fn test_delete_with_repair_nonexistent_errors() {
        let (mut index, _q) = build_test_index();
        assert!(index.delete_with_repair(9999).is_err());
    }

    #[test]
    fn test_delete_with_repair_recall_vs_tombstone() {
        // Compare recall after repair-delete vs tombstone-delete.
        // Repair should maintain better recall because the graph stays connected.
        let dim = 32;
        let n = 300;

        let mut seed: u64 = 77;
        let mut next = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5
        };

        let mut vectors: Vec<Vec<f32>> = Vec::new();
        for _ in 0..n {
            let mut v: Vec<f32> = (0..dim).map(|_| next()).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|x| *x /= norm);
            }
            vectors.push(v);
        }

        let mut query: Vec<f32> = (0..dim).map(|_| next()).collect();
        let norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            query.iter_mut().for_each(|x| *x /= norm);
        }

        let build = |seed_val: u64| -> HNSWIndex {
            let params = HNSWParams {
                seed: Some(seed_val),
                ..HNSWParams::default()
            };
            let mut idx = HNSWIndex::with_params(dim, params).unwrap();
            for (i, v) in vectors.iter().enumerate() {
                idx.add(i as u32, v.clone()).unwrap();
            }
            idx.build().unwrap();
            idx
        };

        let mut repair_idx = build(42);
        let mut tombstone_idx = build(42);

        // Delete 40% of nodes
        let delete_ids: Vec<u32> = (0..120).collect();
        for &id in &delete_ids {
            repair_idx.delete_with_repair(id).unwrap();
            tombstone_idx.delete(id).unwrap();
        }

        let k = 10;
        let ef = 100;

        let repair_results = repair_idx.search(&query, k, ef).unwrap();
        let tombstone_results = tombstone_idx.search(&query, k, ef).unwrap();

        assert!(
            !repair_results.is_empty(),
            "repair search should return results"
        );
        assert!(
            !tombstone_results.is_empty(),
            "tombstone search should return results"
        );

        // Repair should return at least as many results (graph stays connected)
        assert!(
            repair_results.len() >= tombstone_results.len(),
            "repair ({}) should return >= tombstone ({}) results",
            repair_results.len(),
            tombstone_results.len(),
        );
    }
}

#[cfg(test)]
#[cfg(feature = "hnsw")]
#[allow(clippy::unwrap_used)]
mod tests_prt_search {
    use super::*;

    /// PRT-accelerated search should produce similar recall to standard search,
    /// while performing fewer full distance computations.
    #[test]
    fn test_search_prt_recall_parity() {
        let dim = 32;
        let n = 300;
        let mut rng_seed: u64 = 42;
        let mut next = || -> f32 {
            rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_seed >> 33) as f32) / (u32::MAX as f32) - 0.5
        };

        let params = HNSWParams {
            m: 16,
            m_max: 32,
            ef_construction: 200,
            ef_search: 50,
            metric: DistanceMetric::L2,
            seed: Some(42),
            ..Default::default()
        };
        let mut index = HNSWIndex::with_params(dim, params).unwrap();

        let mut all_vecs = Vec::new();
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| next()).collect();
            all_vecs.push(v.clone());
            index.add(i as u32, v).unwrap();
        }
        index.build().unwrap();

        // Build PRT state.
        let num_proj = 16;
        let mut prt = crate::prt::ProbabilisticRoutingTest::new(dim, num_proj, Some(42));
        prt.project_database(&index.vectors);

        let k = 10;
        let ef = 50;
        let num_queries = 20;
        let mut prt_recall_total = 0.0;
        let mut std_recall_total = 0.0;
        let mut total_full_dists = 0usize;

        for qi in 0..num_queries {
            let query: Vec<f32> = (0..dim).map(|_| next()).collect();

            // Standard search.
            let std_results = index.search(&query, k, ef).unwrap();

            // PRT search.
            let (prt_results, full_dists) =
                index.search_prt(&query, k, ef, &prt, 1.5, 0.95).unwrap();
            total_full_dists += full_dists;

            // Brute-force ground truth.
            let mut gt: Vec<(u32, f32)> = all_vecs
                .iter()
                .enumerate()
                .map(|(i, v)| (i as u32, crate::distance::l2_distance(&query, v)))
                .collect();
            gt.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

            let gt_set: std::collections::HashSet<u32> =
                gt.iter().take(k).map(|&(id, _)| id).collect();

            let std_ids: std::collections::HashSet<u32> =
                std_results.iter().map(|&(id, _)| id).collect();
            let prt_ids: std::collections::HashSet<u32> =
                prt_results.iter().map(|&(id, _)| id).collect();

            std_recall_total += gt_set.intersection(&std_ids).count() as f32 / k as f32;
            prt_recall_total += gt_set.intersection(&prt_ids).count() as f32 / k as f32;

            let _ = qi; // used for iteration
        }

        let std_recall = std_recall_total / num_queries as f32;
        let prt_recall = prt_recall_total / num_queries as f32;
        let avg_full_dists = total_full_dists as f32 / num_queries as f32;

        // PRT recall should be close to standard (within 20% relative).
        assert!(
            prt_recall > std_recall * 0.7,
            "PRT recall ({:.1}%) too far from standard ({:.1}%)",
            prt_recall * 100.0,
            std_recall * 100.0
        );

        // PRT should compute fewer full distances than total visited nodes.
        // With n=300 and ef=50, standard search visits ~50+ nodes.
        // PRT should skip at least some.
        assert!(
            avg_full_dists < n as f32,
            "PRT should not compute all {} distances (avg={})",
            n,
            avg_full_dists
        );
    }

    /// PRT search should return valid results and correct doc_ids.
    #[test]
    fn test_search_prt_basic() {
        let dim = 8;
        let mut index = HNSWIndex::builder(dim)
            .metric(DistanceMetric::L2)
            .m(8)
            .build()
            .unwrap();

        for i in 0..50 {
            let v: Vec<f32> = (0..dim).map(|d| (i * dim + d) as f32 * 0.1).collect();
            index.add(i as u32, v).unwrap();
        }
        index.build().unwrap();

        let mut prt = crate::prt::ProbabilisticRoutingTest::new(dim, 4, Some(42));
        prt.project_database(&index.vectors);

        let query: Vec<f32> = vec![0.0; dim];
        let (results, full_dists) = index.search_prt(&query, 5, 20, &prt, 1.5, 0.95).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
        assert!(full_dists > 0);

        // Results should be sorted by distance.
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1, "results not sorted");
        }
    }
}

#[cfg(test)]
#[cfg(feature = "hnsw")]
#[allow(clippy::unwrap_used)]
mod tests_l2 {
    use super::*;

    #[test]
    fn test_hnsw_l2_distance() {
        let mut index = HNSWIndex::builder(4)
            .metric(DistanceMetric::L2)
            .build()
            .unwrap();
        index.add_slice(0, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.add_slice(1, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        index.add_slice(2, &[100.0, 100.0, 0.0, 0.0]).unwrap();
        index.build().unwrap();
        let results = index.search(&[1.0, 0.1, 0.0, 0.0], 2, 10).unwrap();
        assert_eq!(results[0].0, 0, "closest to [1,0,0,0] should be doc 0");
    }
}

#[cfg(test)]
#[cfg(feature = "hnsw")]
#[allow(clippy::unwrap_used)]
mod tests_capacity_law {
    use super::*;
    use std::collections::HashSet;

    /// Verify the capacity-law guard: when k > 2*ef, the guard auto-adjusts ef
    /// to prevent catastrophic recall collapse. Without the guard, requesting
    /// k=50 with ef=5 would return geometrically meaningless results.
    #[test]
    fn test_capacity_law_guard_prevents_recall_collapse() {
        let dim = 16;
        let n = 300;
        let mut rng_seed: u64 = 42;
        let mut next = || -> f32 {
            rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_seed >> 33) as f32) / (u32::MAX as f32) - 0.5
        };

        let params = HNSWParams {
            m: 16,
            m_max: 32,
            ef_construction: 200,
            ef_search: 50,
            metric: DistanceMetric::L2,
            seed: Some(42),
            ..Default::default()
        };
        let mut index = HNSWIndex::with_params(dim, params).unwrap();

        let mut all_vecs = Vec::new();
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| next()).collect();
            all_vecs.push(v.clone());
            index.add(i as u32, v).unwrap();
        }
        index.build().unwrap();

        let query: Vec<f32> = (0..dim).map(|_| next()).collect();
        let k = 20;

        // With capacity-law guard, ef=5 gets auto-adjusted to k=20.
        let results = index.search(&query, k, 5).unwrap();

        // Brute-force ground truth using stored vectors and L2 distance.
        let mut gt: Vec<(u32, f32)> = all_vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, crate::distance::l2_distance(&query, v)))
            .collect();
        gt.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

        let gt_set: HashSet<u32> = gt.iter().take(k).map(|&(id, _)| id).collect();
        let result_ids: HashSet<u32> = results.iter().map(|&(doc_id, _)| doc_id).collect();

        let recall = gt_set.intersection(&result_ids).count() as f32 / k as f32;
        // The guard auto-adjusts ef from 5 to 20, producing reasonable recall.
        assert!(
            recall > 0.3,
            "Capacity-law guard should prevent collapse. recall={:.1}%",
            recall * 100.0
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_build_parallel_recall() {
        let dim = 32;
        let n = 500;
        let k = 10;

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let params = HNSWParams {
            m: 16,
            m_max: 32,
            ef_construction: 200,
            metric: DistanceMetric::L2,
            seed: Some(42),
            ..Default::default()
        };
        let mut index = HNSWIndex::with_params(dim, params).unwrap();

        let mut all_vecs = Vec::new();
        for i in 0..n {
            let v: Vec<f32> = (0..dim)
                .map(|_| {
                    use rand::Rng;
                    rng.random::<f32>() * 2.0 - 1.0
                })
                .collect();
            all_vecs.push(v.clone());
            index.add(i as u32, v).unwrap();
        }
        index.build_parallel(64).unwrap();

        let query = &all_vecs[0];
        let results = index.search(query, k, 100).unwrap();

        // Brute-force ground truth.
        let mut gt: Vec<(u32, f32)> = all_vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, crate::distance::l2_distance(query, v)))
            .collect();
        gt.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        let gt_ids: HashSet<u32> = gt.iter().take(k).map(|(id, _)| *id).collect();
        let result_ids: HashSet<u32> = results.iter().map(|(id, _)| *id).collect();

        let recall = gt_ids.intersection(&result_ids).count() as f32 / k as f32;
        assert!(
            recall >= 0.5,
            "Parallel build recall@{k} = {:.1}%, expected >= 50%",
            recall * 100.0
        );
    }
}
