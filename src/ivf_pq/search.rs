//! IVF-PQ search implementation.

use super::opq::OptimizedProductQuantizer;
use super::pq::ProductQuantizer;
use crate::pq_simd::{adc_batch_dispatch, PackedLUT};
use crate::RetrieveError;
use serde::{Deserialize, Serialize};

/// Minimum candidates in a partition to use SIMD batch ADC.
/// Below this threshold, scalar per-candidate lookup is used.
const SIMD_BATCH_THRESHOLD: usize = 16;

// flat_table_to_nested removed: PackedLUT::from_flat skips the intermediate allocation.

/// Quantizer strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Quantizer {
    /// Standard Product Quantization.
    Product(ProductQuantizer),
    /// Optimized Product Quantization (with rotation).
    Optimized(OptimizedProductQuantizer),
}

impl Quantizer {
    /// Quantize a vector into PQ codes.
    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        match self {
            Self::Product(pq) => pq.quantize(vector),
            Self::Optimized(opq) => opq.quantize(vector),
        }
    }

    /// Compute an asymmetric distance computation (ADC) lookup table for a query.
    pub fn compute_adc_table(&self, query: &[f32]) -> Result<Vec<f32>, RetrieveError> {
        match self {
            Self::Product(pq) => pq.compute_adc_table(query),
            Self::Optimized(opq) => opq.approximate_distance_table(query),
        }
    }

    /// Compute approximate distance using a pre-computed ADC table and PQ codes.
    pub fn distance_with_table(&self, table: &[f32], codes: &[u8]) -> f32 {
        match self {
            Self::Product(pq) => pq.distance_with_table(table, codes),
            Self::Optimized(opq) => opq.distance_with_table(table, codes),
        }
    }
}

/// IVF-PQ index for memory-efficient approximate nearest neighbor search.
#[derive(Debug)]
pub struct IVFPQIndex {
    pub(crate) vectors: Vec<f32>,
    pub(crate) dimension: usize,
    pub(crate) num_vectors: usize,
    /// Maps insertion index → caller-provided doc_id.
    doc_ids: Vec<u32>,
    params: IVFPQParams,
    built: bool,

    // IVF components
    clusters: Vec<Cluster>,
    /// Flat centroid storage: `[c0_d0, c0_d1, ..., c1_d0, ...]` with stride = dimension.
    pub(crate) centroids: Vec<f32>,

    // PQ components
    pq: Option<Quantizer>,
    // Flattened codes: [vector_0_codes, vector_1_codes, ...]
    // Stride = num_codebooks
    pub(crate) quantized_codes: Vec<u8>,

    // Filtering support
    /// Metadata store: doc_id -> category_id mapping
    metadata: Option<crate::filtering::MetadataStore>,
    /// Field name for filtering (e.g., "category")
    filter_field: Option<String>,
}

/// IVF-PQ parameters.
#[derive(Clone, Debug)]
pub struct IVFPQParams {
    /// Number of clusters (inverted lists)
    pub num_clusters: usize,

    /// Number of clusters to search (nprobe)
    pub nprobe: usize,

    /// Product quantization: number of codebooks
    pub num_codebooks: usize,

    /// Product quantization: codebook size
    pub codebook_size: usize,

    /// Use Optimized Product Quantization (OPQ)
    pub use_opq: bool,

    /// ID compression method (optional)
    #[cfg(feature = "id-compression")]
    pub id_compression: Option<crate::compression::IdCompressionMethod>,

    /// Minimum cluster size to compress (smaller clusters use uncompressed storage)
    #[cfg(feature = "id-compression")]
    pub compression_threshold: usize,
}

impl Default for IVFPQParams {
    fn default() -> Self {
        Self {
            num_clusters: 1024,
            nprobe: 100,
            num_codebooks: 8,
            codebook_size: 256,
            use_opq: false,
            #[cfg(feature = "id-compression")]
            id_compression: None,
            #[cfg(feature = "id-compression")]
            compression_threshold: 100, // Only compress clusters with > 100 IDs
        }
    }
}

/// Storage for cluster IDs (compressed or uncompressed).
#[derive(Clone, Debug, Serialize, Deserialize)]
enum ClusterStorage {
    /// Uncompressed IDs (current implementation).
    Uncompressed(Vec<u32>),

    /// Compressed IDs using ROC.
    #[cfg(feature = "id-compression")]
    Compressed {
        data: Vec<u8>,
        num_ids: usize,
        universe_size: u32,
    },
}

/// Cluster (inverted list) containing vector indices.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Cluster {
    storage: ClusterStorage,
    /// Filter bitmask: set of category IDs present in this cluster
    /// Bit i is set if any vector in cluster has category i
    filter_bitmask: u64,
    /// Cache for decompressed IDs (temporary, cleared after use)
    #[cfg(feature = "id-compression")]
    #[serde(skip)]
    #[allow(dead_code)]
    decompressed_cache: Option<Vec<u32>>,
}

impl Cluster {
    /// Create uncompressed cluster.
    fn new(ids: Vec<u32>, filter_bitmask: u64) -> Self {
        Self {
            storage: ClusterStorage::Uncompressed(ids),
            filter_bitmask,
            #[cfg(feature = "id-compression")]
            decompressed_cache: None,
        }
    }

    /// Create compressed cluster.
    #[cfg(feature = "id-compression")]
    fn new_compressed(
        ids: Vec<u32>,
        filter_bitmask: u64,
        _compressor: &crate::compression::RocCompressor,
        universe_size: u32,
    ) -> Result<Self, crate::compression::CompressionError> {
        // Sort IDs (required for compression)
        let mut sorted_ids = ids;
        sorted_ids.sort();
        sorted_ids.dedup();

        // Compress (self-describing envelope)
        let compressed = crate::compression::compress_set_enveloped(
            &sorted_ids,
            universe_size,
            crate::compression::AutoConfig::default(),
        )?;

        Ok(Self {
            storage: ClusterStorage::Compressed {
                data: compressed,
                num_ids: sorted_ids.len(),
                universe_size,
            },
            filter_bitmask,
            decompressed_cache: None,
        })
    }

    /// Get IDs (decompress if needed).
    #[cfg(feature = "id-compression")]
    #[allow(dead_code)]
    fn get_ids(&mut self) -> Result<&[u32], crate::compression::CompressionError> {
        match &self.storage {
            ClusterStorage::Uncompressed(ids) => Ok(ids),
            ClusterStorage::Compressed {
                data,
                universe_size,
                ..
            } => {
                // Check cache first
                if let Some(ref cached) = self.decompressed_cache {
                    return Ok(cached);
                }

                // Decompress
                let (_choice, u2, decompressed) =
                    crate::compression::decompress_set_enveloped(data)?;
                if u2 != *universe_size {
                    return Err(crate::compression::CompressionError::DecompressionFailed(
                        "universe mismatch in envelope".to_string(),
                    ));
                }

                // Cache (will be cleared after search)
                self.decompressed_cache = Some(decompressed);
                // Safety: just assigned Some on the line above
                #[allow(clippy::unwrap_used)]
                Ok(self.decompressed_cache.as_ref().unwrap())
            }
        }
    }

    /// Get IDs (for immutable access, clones if compressed).
    fn get_ids_immut(&self) -> Vec<u32> {
        match &self.storage {
            ClusterStorage::Uncompressed(ids) => ids.clone(),
            #[cfg(feature = "id-compression")]
            ClusterStorage::Compressed {
                data,
                universe_size,
                ..
            } => {
                // For immutable access, we need to decompress (no caching)
                crate::compression::decompress_set_enveloped(data)
                    .map(|(_choice, u2, ids)| {
                        if u2 == *universe_size {
                            ids
                        } else {
                            Vec::new()
                        }
                    })
                    .unwrap_or_else(|_| Vec::new())
            }
        }
    }

    /// Get number of IDs.
    #[allow(dead_code)]
    fn len(&self) -> usize {
        match &self.storage {
            ClusterStorage::Uncompressed(ids) => ids.len(),
            #[cfg(feature = "id-compression")]
            ClusterStorage::Compressed { num_ids, .. } => *num_ids,
        }
    }

    /// Clear decompression cache (call after search).
    #[cfg(feature = "id-compression")]
    #[allow(dead_code)]
    fn clear_cache(&mut self) {
        self.decompressed_cache = None;
    }
}

impl IVFPQIndex {
    /// Set the number of clusters to probe during search.
    ///
    /// This can be changed after the index is built to sweep the
    /// recall/latency trade-off without rebuilding.
    pub fn set_nprobe(&mut self, nprobe: usize) {
        self.params.nprobe = nprobe;
    }

    /// Create a new IVF-PQ index.
    pub fn new(dimension: usize, params: IVFPQParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be > 0".into(),
            ));
        }

        Ok(Self {
            vectors: Vec::new(),
            dimension,
            num_vectors: 0,
            doc_ids: Vec::new(),
            params,
            built: false,
            clusters: Vec::new(),
            centroids: Vec::new(),
            pq: None,
            quantized_codes: Vec::new(),
            metadata: None,
            filter_field: None,
        })
    }

    /// Create a new IVF-PQ index with filtering support.
    ///
    /// # Arguments
    ///
    /// * `dimension` - Vector dimension
    /// * `params` - IVF-PQ parameters
    /// * `filter_field` - Field name for filtering (e.g., "category")
    pub fn with_filtering(
        dimension: usize,
        params: IVFPQParams,
        filter_field: impl Into<String>,
    ) -> Result<Self, RetrieveError> {
        Ok(Self {
            vectors: Vec::new(),
            dimension,
            num_vectors: 0,
            doc_ids: Vec::new(),
            params,
            built: false,
            clusters: Vec::new(),
            centroids: Vec::new(),
            pq: None,
            quantized_codes: Vec::new(),
            metadata: Some(crate::filtering::MetadataStore::new()),
            filter_field: Some(filter_field.into()),
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
                "filtering not enabled; use IVFPQIndex::with_filtering()".into(),
            ))
        }
    }

    /// Add a vector to the index.
    pub fn add(&mut self, _doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        self.add_slice(_doc_id, &vector)
    }

    /// Add a vector to the index from a borrowed slice.
    ///
    /// Notes:
    /// - The index stores vectors internally, so it must copy the slice into its own storage.
    /// - `doc_id` is preserved and returned in search results.
    /// - Vectors are L2-normalized on insertion (cosine similarity index).
    pub fn add_slice(&mut self, doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
        if self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot add vectors after index is built".into(),
            ));
        }

        if vector.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: vector.len(),
                doc_dim: self.dimension,
            });
        }

        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            self.vectors.extend(vector.iter().map(|x| x / norm));
        } else {
            self.vectors.extend_from_slice(vector);
        }
        self.doc_ids.push(doc_id);
        self.num_vectors += 1;
        Ok(())
    }

    /// Build the index.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }

        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        // Stage 1: k-means clustering for IVF
        let mut kmeans =
            crate::partitioning::kmeans::KMeans::new(self.dimension, self.params.num_clusters)?;
        kmeans.fit(&self.vectors, self.num_vectors)?;
        // Flatten centroids to contiguous storage for cache-friendly access.
        self.centroids = kmeans
            .centroids()
            .iter()
            .flat_map(|c| c.iter().copied())
            .collect();

        // Assign vectors to clusters
        let assignments = kmeans.assign_clusters(&self.vectors, self.num_vectors);

        // Build temporary clusters with IDs
        let mut temp_clusters: Vec<(Vec<u32>, u64)> =
            vec![(Vec::new(), 0); self.params.num_clusters];

        // Build clusters with filter bitmasks if filtering is enabled
        if let Some(ref metadata_store) = self.metadata {
            if let Some(ref field) = self.filter_field {
                for (vector_idx, &cluster_idx) in assignments.iter().enumerate() {
                    temp_clusters[cluster_idx].0.push(vector_idx as u32);

                    // Update cluster bitmask with category (look up by real doc_id)
                    let actual_doc_id = self.doc_ids[vector_idx];
                    if let Some(metadata) = metadata_store.get(actual_doc_id) {
                        if let Some(&category_id) = metadata.get(field) {
                            if category_id < 64 {
                                // Only support up to 64 categories (u64 bitmask)
                                temp_clusters[cluster_idx].1 |= 1u64 << category_id;
                            }
                        }
                    }
                }
            } else {
                // No filter field, just add vectors
                for (vector_idx, &cluster_idx) in assignments.iter().enumerate() {
                    temp_clusters[cluster_idx].0.push(vector_idx as u32);
                }
            }
        } else {
            // No metadata, just add vectors
            for (vector_idx, &cluster_idx) in assignments.iter().enumerate() {
                temp_clusters[cluster_idx].0.push(vector_idx as u32);
            }
        }

        // Convert to Cluster structs with optional compression
        self.clusters = temp_clusters
            .into_iter()
            .map(|(ids, bitmask)| {
                #[cfg(feature = "id-compression")]
                {
                    // Compress if enabled and cluster is large enough
                    if let Some(ref method) = self.params.id_compression {
                        if ids.len() >= self.params.compression_threshold {
                            match method {
                                crate::compression::IdCompressionMethod::Roc => {
                                    let compressor = crate::compression::RocCompressor::new();
                                    let universe_size = self.num_vectors as u32;
                                    // Clone ids for fallback case since new_compressed takes ownership
                                    let ids_clone = ids.clone();
                                    Cluster::new_compressed(
                                        ids,
                                        bitmask,
                                        &compressor,
                                        universe_size,
                                    )
                                    .unwrap_or_else(|_| Cluster::new(ids_clone, bitmask))
                                }
                                _ => Cluster::new(ids, bitmask), // Other methods not implemented yet
                            }
                        } else {
                            Cluster::new(ids, bitmask)
                        }
                    } else {
                        Cluster::new(ids, bitmask)
                    }
                }

                #[cfg(not(feature = "id-compression"))]
                {
                    Cluster::new(ids, bitmask)
                }
            })
            .collect();

        // Stage 2: Product Quantization on residual vectors
        // Compute residuals: vector[i] - centroid[assignment[i]]
        let mut residuals = Vec::with_capacity(self.num_vectors * self.dimension);
        for (i, &cluster_idx) in assignments.iter().enumerate() {
            let vec = self.get_vector(i);
            let centroid = self.get_centroid(cluster_idx);
            for (v, c) in vec.iter().zip(centroid.iter()) {
                residuals.push(v - c);
            }
        }

        // Train PQ or OPQ on residuals
        let pq: Quantizer = if self.params.use_opq {
            let mut opq = OptimizedProductQuantizer::new(
                self.dimension,
                self.params.num_codebooks,
                self.params.codebook_size,
            )?;
            opq.fit(&residuals, self.num_vectors, 10)?; // 10 iterations
            Quantizer::Optimized(opq)
        } else {
            let mut pq = ProductQuantizer::new(
                self.dimension,
                self.params.num_codebooks,
                self.params.codebook_size,
            )?;
            pq.fit(&residuals, self.num_vectors)?;
            Quantizer::Product(pq)
        };

        // Quantize residual vectors
        self.quantized_codes = Vec::with_capacity(self.num_vectors * self.params.num_codebooks);
        for i in 0..self.num_vectors {
            let residual = &residuals[i * self.dimension..(i + 1) * self.dimension];
            let codes = pq.quantize(residual);
            self.quantized_codes.extend_from_slice(&codes);
        }

        self.pq = Some(pq);
        self.built = true;
        Ok(())
    }

    /// Search for k nearest neighbors.
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

        let pq = self
            .pq
            .as_ref()
            .ok_or(RetrieveError::InvalidParameter("PQ not initialized".into()))?;

        // Normalize query (index operates on unit-length vectors)
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let query_normalized: Vec<f32> = if query_norm > 1e-10 {
            query.iter().map(|x| x / query_norm).collect()
        } else {
            query.to_vec()
        };
        let query = query_normalized.as_slice();

        // Find closest clusters (partial sort: O(C log nprobe) instead of O(C log C))
        let num_centroids = self.centroids.len() / self.dimension;
        let mut cluster_distances: Vec<(usize, f32)> = (0..num_centroids)
            .map(|idx| {
                let centroid = self.get_centroid(idx);
                let dist = crate::distance::cosine_distance_normalized(query, centroid);
                (idx, dist)
            })
            .collect();

        let nprobe = self.params.nprobe.min(cluster_distances.len());
        if nprobe < cluster_distances.len() {
            cluster_distances.select_nth_unstable_by(nprobe, |a, b| a.1.total_cmp(&b.1));
            cluster_distances.truncate(nprobe);
        }
        cluster_distances.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

        // Pre-allocate reusable buffers for the nprobe loop
        let mut candidates = Vec::new();
        let mut query_residual = vec![0.0f32; self.dimension];
        let mut codes_batch = Vec::new();

        for (cluster_idx, _) in &cluster_distances {
            let cluster = &self.clusters[*cluster_idx];
            let ids = cluster.get_ids_immut();

            // Compute query residual in-place (no allocation)
            let centroid = self.get_centroid(*cluster_idx);
            for (i, (q, c)) in query.iter().zip(centroid.iter()).enumerate() {
                query_residual[i] = q - c;
            }

            // Build ADC table from query residual
            let adc_table = pq.compute_adc_table(&query_residual)?;

            if ids.len() >= SIMD_BATCH_THRESHOLD {
                // Build PackedLUT directly from flat table (no intermediate Vec<Vec<f32>>)
                let packed_lut = PackedLUT::from_flat(
                    &adc_table,
                    self.params.num_codebooks,
                    self.params.codebook_size,
                );

                // SIMD batch path: gather codes into a reusable buffer
                let num_cb = self.params.num_codebooks;
                codes_batch.clear();
                codes_batch.reserve(ids.len() * num_cb);
                for &vector_idx in &ids {
                    let start = vector_idx as usize * num_cb;
                    codes_batch.extend_from_slice(&self.quantized_codes[start..start + num_cb]);
                }

                let distances = adc_batch_dispatch(&codes_batch, num_cb, &packed_lut);

                for (i, &vector_idx) in ids.iter().enumerate() {
                    let doc_id = self.doc_ids[vector_idx as usize];
                    candidates.push((doc_id, distances[i]));
                }
            } else {
                // Scalar fallback for small clusters
                for &vector_idx in &ids {
                    let start = vector_idx as usize * self.params.num_codebooks;
                    let end = start + self.params.num_codebooks;
                    let codes = &self.quantized_codes[start..end];

                    let dist = pq.distance_with_table(&adc_table, codes);
                    let doc_id = self.doc_ids[vector_idx as usize];
                    candidates.push((doc_id, dist));
                }
            }
        }

        // Sort and return top k
        candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        Ok(candidates.into_iter().take(k).collect())
    }

    /// Search with filter using cluster tagging (integrated filtering).
    ///
    /// Skips clusters that don't contain any vectors matching the filter,
    /// reducing search space and improving performance.
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector
    /// * `k` - Number of results
    /// * `filter` - Filter predicate (must be equality filter on filter_field)
    ///
    /// # Returns
    ///
    /// Vector of (doc_id, distance) pairs matching the filter
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
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

        // Extract category ID from filter (only supports equality on filter_field)
        let desired_category = match filter {
            crate::filtering::MetadataFilter::Equals { field, value } => {
                if Some(field) != self.filter_field.as_ref() {
                    return Err(RetrieveError::InvalidParameter(format!(
                        "filter field '{}' doesn't match index filter field '{:?}'",
                        field, self.filter_field
                    )));
                }
                if *value >= 64 {
                    return Err(RetrieveError::InvalidParameter(
                        "category ID must be < 64 for bitmask filtering".into(),
                    ));
                }
                *value
            }
            _ => {
                return Err(RetrieveError::InvalidParameter(
                    "only equality filters on filter_field are supported".into(),
                ));
            }
        };

        let filter_bit = 1u64 << desired_category;

        // Normalize query (index operates on unit-length vectors)
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let query_normalized: Vec<f32> = if query_norm > 1e-10 {
            query.iter().map(|x| x / query_norm).collect()
        } else {
            query.to_vec()
        };
        let query = query_normalized.as_slice();

        // Find closest clusters (partial sort)
        let num_centroids = self.centroids.len() / self.dimension;
        let mut cluster_distances: Vec<(usize, f32)> = (0..num_centroids)
            .map(|idx| {
                let centroid = self.get_centroid(idx);
                let dist = crate::distance::cosine_distance_normalized(query, centroid);
                (idx, dist)
            })
            .collect();

        let nprobe = self.params.nprobe.min(cluster_distances.len());
        if nprobe < cluster_distances.len() {
            cluster_distances.select_nth_unstable_by(nprobe, |a, b| a.1.total_cmp(&b.1));
            cluster_distances.truncate(nprobe);
        }
        cluster_distances.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

        // Search in top nprobe clusters, skipping those without matching vectors
        let mut candidates = Vec::new();
        let mut query_residual = vec![0.0f32; self.dimension];

        let pq = self
            .pq
            .as_ref()
            .ok_or(RetrieveError::InvalidParameter("PQ not initialized".into()))?;

        for (cluster_idx, _) in &cluster_distances {
            let cluster = &self.clusters[*cluster_idx];

            // Skip cluster if it doesn't contain any vectors matching the filter
            if (cluster.filter_bitmask & filter_bit) == 0 {
                continue;
            }

            // Compute query residual in-place
            let centroid = self.get_centroid(*cluster_idx);
            for (i, (q, c)) in query.iter().zip(centroid.iter()).enumerate() {
                query_residual[i] = q - c;
            }

            let adc_table = pq.compute_adc_table(&query_residual)?;

            // Search vectors in this cluster, filtering by metadata
            if let Some(ref metadata_store) = self.metadata {
                let ids = cluster.get_ids_immut();
                for &vector_idx in &ids {
                    let actual_doc_id = self.doc_ids[vector_idx as usize];
                    if metadata_store.matches(actual_doc_id, filter) {
                        let start = vector_idx as usize * self.params.num_codebooks;
                        let end = start + self.params.num_codebooks;
                        let codes = &self.quantized_codes[start..end];

                        let dist = pq.distance_with_table(&adc_table, codes);
                        candidates.push((actual_doc_id, dist));
                    }
                }
            } else {
                return Err(RetrieveError::InvalidParameter(
                    "metadata store not initialized".into(),
                ));
            }
        }

        // Sort and return top k
        candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        Ok(candidates.into_iter().take(k).collect())
    }

    /// Get vector from SoA storage.
    #[inline]
    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        &self.vectors[start..end]
    }

    /// Get centroid from flat storage.
    #[inline]
    fn get_centroid(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        &self.centroids[start..end]
    }
}
