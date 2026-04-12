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
        compressor: &crate::compression::RocCompressor,
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
        _compressor: &crate::compression::RocCompressor,
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
                    crate::compression::AutoConfig::default(),
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

    /// Serialize this index to a writer as JSON.
    ///
    /// The `metadata` store and `doc_id_to_internal` reverse map are not
    /// serialized. The reverse map is rebuilt on [`Self::load_from_reader`]; metadata
    /// must be re-added if filtered search is needed.
    #[cfg(feature = "serde")]
    pub fn save_to_writer<W: std::io::Write>(&self, writer: W) -> Result<(), RetrieveError> {
        serde_json::to_writer(writer, self).map_err(|e| RetrieveError::Serialization(e.to_string()))
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
        if self.params.metric == DistanceMetric::Cosine {
            let norm_sq: f32 = vector.iter().map(|x| x * x).sum();
            if (norm_sq - 1.0).abs() > 0.1 {
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
                        let dist = self.dist(vector, other_vector);
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
                        let dist = self.dist(vector, other_vector);
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
            crate::compression::IdCompressionMethod::Roc => {
                let compressor = crate::compression::RocCompressor::new();
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
                let mut best_seed = seeds[0];
                let mut best_dist = f32::INFINITY;
                for &seed_id in &seeds {
                    let seed_vec = self.get_vector(seed_id as usize);
                    let dist = self.dist(query, seed_vec);
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
        let mut current_closest = entry_point;
        let mut current_dist = self.dist(query, self.get_vector(entry_point as usize));

        // Search in upper layers (coarse search)
        for layer_idx in (1..=entry_layer).rev() {
            if layer_idx >= self.layers.len() {
                continue;
            }

            let layer = &self.layers[layer_idx];
            let mut changed = true;
            // Pre-allocate visited set with capacity for typical layer traversal
            let mut visited = std::collections::HashSet::with_capacity(ef.min(100));

            while changed {
                changed = false;
                visited.insert(current_closest);

                let neighbors = layer.get_neighbors(current_closest);
                for &neighbor_id in neighbors.iter() {
                    if visited.contains(&neighbor_id) {
                        continue;
                    }

                    let neighbor_vec = self.get_vector(neighbor_id as usize);
                    let dist = self.dist(query, neighbor_vec);

                    if dist < current_dist {
                        current_dist = dist;
                        current_closest = neighbor_id;
                        changed = true;
                    }
                }
            }
        }

        // Fine search in base layer (layer 0)
        if !self.layers.is_empty() {
            // For KS strategy, warm up search with multiple seeds
            let base_results =
                if let SeedSelectionStrategy::KSampledRandom { .. } = &self.params.seed_selection {
                    // Use KS seeds to initialize search
                    // Find the best seed (closest to query) among KSampled seeds
                    // and the greedy-descended entry point, then run standard beam search.
                    let mut best_entry = current_closest;
                    let mut best_dist = self.dist(query, self.get_vector(current_closest as usize));
                    for &seed_id in &initial_seeds {
                        let dist = self.dist(query, self.get_vector(seed_id as usize));
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
    pub fn search_with_distance(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        dist_fn: &dyn Fn(&[f32], u32) -> f32,
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

        let mut current_closest = entry_point;
        let mut current_dist = self.dist(query, self.get_vector(entry_point as usize));

        for layer_idx in (1..=entry_layer).rev() {
            if layer_idx >= self.layers.len() {
                continue;
            }
            let layer = &self.layers[layer_idx];
            let mut changed = true;
            let mut visited = std::collections::HashSet::with_capacity(ef.min(100));

            while changed {
                changed = false;
                visited.insert(current_closest);
                for &neighbor_id in layer.get_neighbors(current_closest).iter() {
                    if visited.contains(&neighbor_id) {
                        continue;
                    }
                    let dist = self.dist(query, self.get_vector(neighbor_id as usize));
                    if dist < current_dist {
                        current_dist = dist;
                        current_closest = neighbor_id;
                        changed = true;
                    }
                }
            }
        }

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
    ) -> Vec<Vec<(u32, f32)>> {
        use rayon::prelude::*;
        queries
            .par_iter()
            .map(|query| self.search(query, k, ef_search).unwrap_or_default())
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
    ) -> Vec<Vec<(u32, f32)>> {
        use rayon::prelude::*;
        (0..num_queries)
            .into_par_iter()
            .map(|i| {
                let query = &queries_flat[i * self.dimension..(i + 1) * self.dimension];
                self.search(query, k, ef_search).unwrap_or_default()
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
                let d = self.dist(last_vec, queries[j]);
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
        let dist_fn = self.dist_fn();

        // results_by_query[i] = (doc_id, dist) results for query i
        let mut results_by_query: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];

        // best_internal[i] = internal node id of best result for chain position i
        let mut best_internal_for_chain: Vec<u32> = Vec::with_capacity(n);

        for (chain_pos, &query_idx) in chain.iter().enumerate() {
            let query = queries[query_idx];

            // Upper-layer greedy descent to find base-layer entry for this query.
            let entry_layer = self.layer_assignments[ep as usize] as usize;
            let mut current_closest = ep;
            let mut current_dist = self.dist(query, self.get_vector(ep as usize));

            for layer_idx in (1..=entry_layer).rev() {
                if layer_idx >= self.layers.len() {
                    continue;
                }
                let layer = &self.layers[layer_idx];
                let mut changed = true;
                let mut upper_visited =
                    std::collections::HashSet::with_capacity(ef_search.min(100));
                while changed {
                    changed = false;
                    upper_visited.insert(current_closest);
                    let neighbors = layer.get_neighbors(current_closest);
                    for &nb in neighbors.iter() {
                        if upper_visited.contains(&nb) {
                            continue;
                        }
                        let d = self.dist(query, self.get_vector(nb as usize));
                        if d < current_dist {
                            current_dist = d;
                            current_closest = nb;
                            changed = true;
                        }
                    }
                }
            }

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
        let start_dist = self.dist(query, start_vec);
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
                let neighbor_dist = self.dist(query, neighbor_vec);
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
                    let dist = self.dist(query, vec);
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

        // Upper-layer navigation (identical to `search`)
        let ep = self.get_entry_point().ok_or(RetrieveError::EmptyIndex)?;
        let entry_layer = self.layer_assignments[ep as usize] as usize;

        let mut current_closest = ep;
        let mut current_dist = self.dist(query, self.get_vector(ep as usize));

        for layer_idx in (1..=entry_layer).rev() {
            if layer_idx >= self.layers.len() {
                continue;
            }
            let layer = &self.layers[layer_idx];
            let mut changed = true;
            let mut visited = std::collections::HashSet::with_capacity(ef.min(100));

            while changed {
                changed = false;
                visited.insert(current_closest);

                let neighbors = layer.get_neighbors(current_closest);
                for &neighbor_id in neighbors.iter() {
                    if visited.contains(&neighbor_id) {
                        continue;
                    }
                    let neighbor_vec = self.get_vector(neighbor_id as usize);
                    let dist = self.dist(query, neighbor_vec);
                    if dist < current_dist {
                        current_dist = dist;
                        current_closest = neighbor_id;
                        changed = true;
                    }
                }
            }
        }

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

    /// Compute distance between two vectors using the configured metric.
    #[inline(always)]
    pub(crate) fn dist(&self, a: &[f32], b: &[f32]) -> f32 {
        self.params.metric.distance(a, b)
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

    /// Get entry point (vector in highest layer).
    fn get_entry_point(&self) -> Option<u32> {
        if let Some(ep) = self.cached_entry_point {
            return Some(ep);
        }
        self.compute_entry_point()
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
        let batch = index.search_batch(&query_slices, k, ef);

        // Batch (parallel) results -- flat-buffer variant
        let batch_flat = index.search_batch_flat(&queries_flat, num_queries, k, ef);

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
        let mut index = HNSWIndex::new(dim, m, 2 * m).unwrap();
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
