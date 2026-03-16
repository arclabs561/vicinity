//! HNSW graph structure and core types.

use crate::RetrieveError;

#[cfg(feature = "hnsw")]
use smallvec::SmallVec;

use std::collections::HashMap;

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

    /// Optional metadata store for filtering
    metadata: Option<crate::filtering::MetadataStore>,

    /// Field name for filtering (e.g., "category")
    filter_field: Option<String>,

    /// Category assignments: vector_idx -> category_id
    category_assignments: Vec<Option<u32>>,
}

/// Seed selection strategy for HNSW search initialization.
///
/// This controls how the search chooses its entrypoint(s). Different strategies can
/// behave better on different datasets and scale regimes; treat this as a tuning knob
/// and benchmark on your workload.
#[derive(Clone, Debug, Default, PartialEq)]
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

    /// Seed selection strategy (default: StackedNSW for large-scale)
    pub seed_selection: SeedSelectionStrategy,

    /// Neighborhood diversification strategy (default: RND for best performance)
    pub neighborhood_diversification: NeighborhoodDiversification,

    /// ID compression method (optional)
    #[cfg(feature = "id-compression")]
    pub id_compression: Option<crate::compression::IdCompressionMethod>,

    /// Minimum neighbor list size to compress (smaller lists use uncompressed storage)
    #[cfg(feature = "id-compression")]
    pub compression_threshold: usize,
}

impl Default for HNSWParams {
    fn default() -> Self {
        Self {
            m: 16,
            m_max: 16,
            m_l: 1.0 / 16.0_f64.ln(), // 1/ln(M), per Malkov & Yashunin 2018
            ef_construction: 200,
            ef_search: 50,
            seed_selection: SeedSelectionStrategy::default(),
            neighborhood_diversification: NeighborhoodDiversification::default(),
            #[cfg(feature = "id-compression")]
            id_compression: None,
            #[cfg(feature = "id-compression")]
            compression_threshold: 32, // Only compress if m >= 32 (per paper)
        }
    }
}

/// Storage for neighbor lists (compressed or uncompressed).
#[derive(Clone, Debug)]
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
struct CompressedNeighborList {
    data: Vec<u8>,
    num_neighbors: usize,
}

/// Graph layer containing neighbor lists for all vectors in that layer.
#[derive(Debug)]
pub(crate) struct Layer {
    storage: NeighborStorage,
    /// Cache for decompressed neighbors (temporary, cleared after use)
    #[cfg(feature = "id-compression")]
    decompressed_cache: std::sync::Mutex<std::collections::HashMap<u32, SmallVec<[u32; 16]>>>,
}

impl Layer {
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
            metadata: None,
            filter_field: None,
            category_assignments: Vec::new(),
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
            params,
            built: false,
            metadata: None,
            filter_field: None,
            category_assignments: Vec::new(),
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
            metadata: Some(crate::filtering::MetadataStore::new()),
            filter_field: Some(filter_field.into()),
            category_assignments: Vec::new(),
        })
    }

    /// Check if the index has been built and is ready for search.
    pub fn is_built(&self) -> bool {
        self.built
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
        // Validate vector/dimension alignment
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be > 0".into(),
            ));
        }
        if vectors.len() != num_vectors * dimension {
            return Err(RetrieveError::InvalidParameter(format!(
                "vectors.len() ({}) != num_vectors ({}) * dimension ({})",
                vectors.len(),
                num_vectors,
                dimension
            )));
        }

        // Validate doc_ids length
        if doc_ids.len() != num_vectors {
            return Err(RetrieveError::InvalidParameter(format!(
                "doc_ids.len() ({}) != num_vectors ({})",
                doc_ids.len(),
                num_vectors
            )));
        }

        // Validate layer_assignments length
        if layer_assignments.len() != num_vectors {
            return Err(RetrieveError::InvalidParameter(format!(
                "layer_assignments.len() ({}) != num_vectors ({})",
                layer_assignments.len(),
                num_vectors
            )));
        }

        // Build reverse map and check for duplicate doc_ids
        let mut doc_id_to_internal = HashMap::with_capacity(doc_ids.len());
        for (i, doc_id) in doc_ids.iter().copied().enumerate() {
            if doc_id_to_internal.insert(doc_id, i as u32).is_some() {
                return Err(RetrieveError::InvalidParameter(format!(
                    "duplicate doc_id: {}",
                    doc_id
                )));
            }
        }

        // Validate neighbor IDs are in bounds
        for (layer_idx, layer) in layers.iter().enumerate() {
            let layer_len = layer.len();
            for node in 0..layer_len {
                let neighbors = layer.get_neighbors(node as u32);
                for &neighbor in neighbors.iter() {
                    if neighbor as usize >= num_vectors {
                        return Err(RetrieveError::InvalidParameter(format!(
                            "layer {} node {} has out-of-bounds neighbor {} (num_vectors={})",
                            layer_idx, node, neighbor, num_vectors
                        )));
                    }
                }
            }
        }

        Ok(Self {
            vectors,
            doc_ids,
            doc_id_to_internal,
            dimension,
            num_vectors,
            layers,
            layer_assignments,
            params,
            built,
            metadata: None,
            filter_field: None,
            category_assignments: Vec::new(),
        })
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
    /// This is the most ergonomic entry-point for callers that already have `&[f32]`
    /// (e.g., from `ndarray`, `nalgebra`, or other tensor libraries).
    ///
    /// Notes:
    /// - The index stores vectors internally, so it must copy the slice into its own storage.
    /// - Vectors should be L2-normalized for cosine similarity.
    /// - The index must be built before searching.
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
                query_dim: self.dimension,
                doc_dim: vector.len(),
            });
        }

        debug_assert!(
            {
                let norm_sq: f32 = vector.iter().map(|x| x * x).sum();
                (norm_sq - 1.0).abs() < 0.1
            },
            "HNSW cosine distance requires L2-normalized vectors (got norm^2 = {})",
            vector.iter().map(|x| x * x).sum::<f32>()
        );

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
                .and_then(|m| m.get(field).copied());
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

        Ok(())
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
        let mut category_vectors: std::collections::HashMap<u32, Vec<u32>> =
            std::collections::HashMap::new();

        for (vector_idx, &category) in self.category_assignments.iter().enumerate() {
            if let Some(cat) = category {
                category_vectors
                    .entry(cat)
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
                        let dist =
                            crate::distance::cosine_distance_normalized(vector, other_vector);
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
                        let dist =
                            crate::distance::cosine_distance_normalized(vector, other_vector);
                        candidates.push((other_id, dist));
                    }
                }

                candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
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
                    let dist = crate::distance::cosine_distance_normalized(query, seed_vec);
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
        let mut current_dist = f32::INFINITY;

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
                    let dist = crate::distance::cosine_distance_normalized(query, neighbor_vec);

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
            let base_results = if let SeedSelectionStrategy::KSampledRandom { .. } =
                &self.params.seed_selection
            {
                // Use KS seeds to initialize search
                use crate::hnsw::search::SearchState;
                let mut state = SearchState::with_capacity(ef.max(k));

                // Add all KS seeds to candidate queue
                for &seed_id in &initial_seeds {
                    let seed_vec = self.get_vector(seed_id as usize);
                    let dist = crate::distance::cosine_distance_normalized(query, seed_vec);
                    state.add_candidate(seed_id, dist);
                }

                // Also add entry point neighbors
                let neighbors = self.layers[0].get_neighbors(current_closest);
                for &neighbor_id in neighbors.iter() {
                    let neighbor_vec = self.get_vector(neighbor_id as usize);
                    let dist = crate::distance::cosine_distance_normalized(query, neighbor_vec);
                    state.add_candidate(neighbor_id, dist);
                }

                // Continue search from candidates
                let mut results = Vec::new();
                while let Some(candidate) = state.pop_candidate() {
                    if results.len() >= ef.max(k) {
                        break;
                    }

                    results.push((candidate.id, candidate.distance));

                    // Explore neighbors
                    let neighbors = self.layers[0].get_neighbors(candidate.id);
                    for &neighbor_id in neighbors.iter() {
                        let neighbor_vec = self.get_vector(neighbor_id as usize);
                        let dist = crate::distance::cosine_distance_normalized(query, neighbor_vec);
                        state.add_candidate(neighbor_id, dist);
                    }
                }

                results
            } else {
                // Default: Use greedy search from entry point
                crate::hnsw::search::greedy_search_layer(
                    query,
                    current_closest,
                    &self.layers[0],
                    &self.vectors,
                    self.dimension,
                    ef.max(k),
                )
            };

            // `base_results` is not guaranteed to be sorted (it may be in exploration order).
            // Sort by distance first, then take top-k.
            let mut base_results = base_results;
            base_results.sort_by(|a, b| a.1.total_cmp(&b.1));
            let results: Vec<(u32, f32)> = base_results.into_iter().take(k).collect();

            // Clear decompression caches after search
            #[cfg(feature = "id-compression")]
            {
                for layer in &self.layers {
                    layer.clear_cache();
                }
            }

            // Convert internal IDs -> external doc_ids.
            let results = results
                .into_iter()
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

        // Extract category ID from filter
        let desired_category = match filter {
            crate::filtering::MetadataFilter::Equals { field, value } => {
                if Some(field) != self.filter_field.as_ref() {
                    return Err(RetrieveError::InvalidParameter(format!(
                        "filter field '{}' doesn't match index filter field '{:?}'",
                        field, self.filter_field
                    )));
                }
                Some(*value)
            }
            _ => {
                return Err(RetrieveError::InvalidParameter(
                    "only equality filters on filter_field are supported".into(),
                ));
            }
        };

        // Perform standard search but filter neighbors during traversal
        // Use min-heap for candidates with FloatOrd wrapper for f32 comparison
        #[derive(PartialEq)]
        struct FloatOrd(f32);
        impl Eq for FloatOrd {}
        impl PartialOrd for FloatOrd {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for FloatOrd {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.0
                    .partial_cmp(&other.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        }

        let mut candidates: std::collections::BinaryHeap<std::cmp::Reverse<(FloatOrd, u32)>> =
            std::collections::BinaryHeap::new();
        let mut visited = std::collections::HashSet::new();

        // Start from entry point if it matches filter, otherwise find first matching vector
        let entry_point = self.get_entry_point().ok_or(RetrieveError::EmptyIndex)?;
        let entry_category = self
            .category_assignments
            .get(entry_point as usize)
            .and_then(|&c| c);

        let start_point = if entry_category == desired_category {
            entry_point
        } else {
            // Find first vector matching filter
            self.category_assignments
                .iter()
                .enumerate()
                .find(|(_, &cat)| cat == desired_category)
                .map(|(idx, _)| idx as u32)
                .ok_or(RetrieveError::EmptyQuery)?
        };

        let start_vec = self.get_vector(start_point as usize);
        let start_dist = crate::distance::cosine_distance_normalized(query, start_vec);
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
                .and_then(|&c| c)
                != desired_category
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
                    .and_then(|&c| c)
                    != desired_category
                {
                    continue;
                }

                let neighbor_vec = self.get_vector(neighbor_id as usize);
                let neighbor_dist =
                    crate::distance::cosine_distance_normalized(query, neighbor_vec);
                candidates.push(std::cmp::Reverse((FloatOrd(neighbor_dist), neighbor_id)));
            }

            if visited.len() >= ef.max(k) {
                break;
            }
        }

        // Extract top-k results
        let mut results: Vec<(u32, f32)> = visited
            .iter()
            .filter_map(|&id| {
                if self.category_assignments.get(id as usize).and_then(|&c| c) == desired_category {
                    let vec = self.get_vector(id as usize);
                    let dist = crate::distance::cosine_distance_normalized(query, vec);
                    let doc_id = self.doc_ids.get(id as usize).copied()?;
                    Some((doc_id, dist))
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| a.1.total_cmp(&b.1));
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
        let mut current_dist = f32::INFINITY;

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
                    let dist = crate::distance::cosine_distance_normalized(query, neighbor_vec);
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
            );
            num_evaluated = evaluated;

            base_results.sort_by(|a, b| a.1.total_cmp(&b.1));

            #[cfg(feature = "id-compression")]
            {
                for layer in &self.layers {
                    layer.clear_cache();
                }
            }

            let results = base_results
                .into_iter()
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
            let mut rng = rand::rng();

            // Paper formula (Algorithm 1, line 4): l = floor(-ln(uniform) * mL)
            let u: f64 = rng.random();
            (-u.ln() * self.params.m_l).floor() as u8
        }
        #[cfg(not(feature = "hnsw"))]
        {
            0
        }
    }

    /// Get vector by index (for internal use).
    pub(crate) fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        &self.vectors[start..end]
    }

    /// Get entry point (vector in highest layer).
    fn get_entry_point(&self) -> Option<u32> {
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
}
