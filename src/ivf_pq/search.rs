//! IVF-PQ search implementation.

use super::opq::OptimizedProductQuantizer;
use super::pq::ProductQuantizer;
use crate::pq_simd::{adc_batch_dispatch, PackedCodes4bit, PackedLUT};
use crate::RetrieveError;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Minimum candidates in a partition to use SIMD batch ADC.
/// Below this threshold, scalar per-candidate lookup is used.
const SIMD_BATCH_THRESHOLD: usize = 16;
const IVFPQ_FORMAT_VERSION: u32 = 1;
const IVFPQ_CLUSTER_MAGIC: &[u8; 8] = b"VICIVF1\0";

// flat_table_to_nested removed: PackedLUT::from_flat skips the intermediate allocation.

fn default_kmeans_max_iter() -> usize {
    100
}

#[derive(Clone, Copy, Debug)]
struct IVFPQTrainingConfig {
    sample_size: Option<usize>,
    kmeans_max_iter: usize,
}

impl Default for IVFPQTrainingConfig {
    fn default() -> Self {
        Self {
            sample_size: None,
            kmeans_max_iter: default_kmeans_max_iter(),
        }
    }
}

fn normalize_query(query: &[f32]) -> Vec<f32> {
    let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        query.iter().map(|x| x / norm).collect()
    } else {
        query.to_vec()
    }
}

fn copy_rows_by_index(vectors: &[f32], dimension: usize, indices: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(indices.len() * dimension);
    for &idx in indices {
        let start = idx * dimension;
        out.extend_from_slice(&vectors[start..start + dimension]);
    }
    out
}

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

    /// Compute ADC table into a caller-provided buffer (avoids allocation per cluster).
    pub fn compute_adc_table_into(
        &self,
        query: &[f32],
        table: &mut Vec<f32>,
    ) -> Result<(), RetrieveError> {
        match self {
            Self::Product(pq) => pq.compute_adc_table_into(query, table),
            Self::Optimized(opq) => {
                let t = opq.approximate_distance_table(query)?;
                table.clear();
                table.extend_from_slice(&t);
                Ok(())
            }
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

    /// HNSW index over centroids for O(log nlist) coarse lookup.
    /// Built automatically during `build()` when both `ivf_pq` and `hnsw` features are enabled.
    /// Falls back to brute-force centroid scan when `None`.
    #[cfg(feature = "hnsw")]
    coarse_quantizer: Option<crate::hnsw::HNSWIndex>,
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

    /// Random seed for deterministic IVF and PQ k-means training.
    pub seed: u64,

    /// ID compression method (optional)
    #[cfg(feature = "id-compression")]
    pub id_compression: Option<crate::compression::IdCompressionMethod>,

    /// Minimum cluster size to compress (smaller clusters use uncompressed storage)
    #[cfg(feature = "id-compression")]
    pub compression_threshold: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedIVFPQParams {
    num_clusters: usize,
    nprobe: usize,
    num_codebooks: usize,
    codebook_size: usize,
    use_opq: bool,
    seed: u64,
}

impl From<&IVFPQParams> for PersistedIVFPQParams {
    fn from(params: &IVFPQParams) -> Self {
        Self {
            num_clusters: params.num_clusters,
            nprobe: params.nprobe,
            num_codebooks: params.num_codebooks,
            codebook_size: params.codebook_size,
            use_opq: params.use_opq,
            seed: params.seed,
        }
    }
}

impl PersistedIVFPQParams {
    fn into_params(self) -> IVFPQParams {
        IVFPQParams {
            num_clusters: self.num_clusters,
            nprobe: self.nprobe,
            num_codebooks: self.num_codebooks,
            codebook_size: self.codebook_size,
            use_opq: self.use_opq,
            seed: self.seed,
            #[cfg(feature = "id-compression")]
            id_compression: None,
            #[cfg(feature = "id-compression")]
            compression_threshold: IVFPQParams::default().compression_threshold,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct IVFPQManifest {
    version: u32,
    dimension: usize,
    num_vectors: usize,
    num_centroids: usize,
    raw_vectors_present: bool,
    params: PersistedIVFPQParams,
    quantizer: Quantizer,
}

impl Default for IVFPQParams {
    fn default() -> Self {
        Self {
            num_clusters: 1024,
            nprobe: 100,
            num_codebooks: 8,
            codebook_size: 256,
            use_opq: false,
            seed: 42,
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
    /// Prepacked 4-bit FastScan codes in this cluster's ID order.
    ///
    /// Faiss stores FastScan codes in block layout instead of packing at query
    /// time. This cache follows that shape for 16-centroid PQ codebooks while
    /// leaving 8-bit and smaller-cluster paths unchanged.
    #[serde(skip)]
    fastscan_codes: Option<PackedCodes4bit>,
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
            fastscan_codes: None,
            #[cfg(feature = "id-compression")]
            decompressed_cache: None,
        }
    }

    /// Create compressed cluster.
    #[cfg(feature = "id-compression")]
    fn new_compressed(
        ids: Vec<u32>,
        filter_bitmask: u64,
        _compressor: &crate::compression::DeltaVarintCompressor,
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
            crate::compression::ChooseConfig::default(),
        )?;

        Ok(Self {
            storage: ClusterStorage::Compressed {
                data: compressed,
                num_ids: sorted_ids.len(),
                universe_size,
            },
            filter_bitmask,
            fastscan_codes: None,
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

    /// Get IDs as a borrowed slice (avoids cloning for the uncompressed case).
    fn get_ids_ref(&self) -> std::borrow::Cow<'_, [u32]> {
        match &self.storage {
            ClusterStorage::Uncompressed(ids) => std::borrow::Cow::Borrowed(ids),
            #[cfg(feature = "id-compression")]
            ClusterStorage::Compressed {
                data,
                universe_size,
                ..
            } => {
                // Compressed: must decompress (returns owned data)
                std::borrow::Cow::Owned(
                    crate::compression::decompress_set_enveloped(data)
                        .map(|(_choice, u2, ids)| {
                            if u2 == *universe_size {
                                ids
                            } else {
                                Vec::new()
                            }
                        })
                        .unwrap_or_else(|_| Vec::new()),
                )
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

    fn set_fastscan_codes(&mut self, codes: Option<PackedCodes4bit>) {
        self.fastscan_codes = codes;
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
            #[cfg(feature = "hnsw")]
            coarse_quantizer: None,
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
            #[cfg(feature = "hnsw")]
            coarse_quantizer: None,
        })
    }

    /// Add metadata for a document (required for filtering).
    ///
    /// Returns an error if the filter field's category ID is ≥ 64 (bitmask limit).
    pub fn add_metadata(
        &mut self,
        doc_id: u32,
        metadata: crate::filtering::DocumentMetadata,
    ) -> Result<(), RetrieveError> {
        if let Some(ref mut store) = self.metadata {
            // Validate category range early so callers get a clear error at insert time,
            // not silently discarded data at build time.
            if let Some(ref field) = self.filter_field {
                if let Some(category_val) = metadata.get(field) {
                    match category_val {
                        crate::filtering::MetadataValue::Int(n) if *n >= 0 && *n < 64 => {}
                        crate::filtering::MetadataValue::Int(n) => {
                            return Err(RetrieveError::InvalidParameter(format!(
                                "category ID {} exceeds bitmask limit of 63; \
                                 use an integer in 0..63",
                                n
                            )));
                        }
                        _ => {
                            return Err(RetrieveError::InvalidParameter(
                                "category ID must be an integer in 0..63 for bitmask filtering"
                                    .into(),
                            ));
                        }
                    }
                }
            }
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
        self.build_with_training_config(IVFPQTrainingConfig::default())
    }

    /// Build the index using a deterministic sample for IVF and PQ training.
    ///
    /// All inserted vectors are still assigned, quantized, and searchable. The
    /// sample only bounds k-means training cost for large datasets.
    pub fn build_with_training_sample(
        &mut self,
        training_sample_size: usize,
    ) -> Result<(), RetrieveError> {
        self.build_with_training_options(Some(training_sample_size), default_kmeans_max_iter())
    }

    /// Build the index with explicit IVF/PQ k-means training controls.
    ///
    /// `training_sample_size = None` trains on all inserted vectors, matching
    /// [`Self::build`]. `kmeans_max_iter` applies to both the coarse IVF
    /// k-means and the PQ codebook k-means calls.
    pub fn build_with_training_options(
        &mut self,
        training_sample_size: Option<usize>,
        kmeans_max_iter: usize,
    ) -> Result<(), RetrieveError> {
        self.build_with_training_config(IVFPQTrainingConfig {
            sample_size: training_sample_size,
            kmeans_max_iter,
        })
    }

    fn build_with_training_config(
        &mut self,
        training: IVFPQTrainingConfig,
    ) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }

        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        // PQ training fits `codebook_size` k-means centroids per subspace.
        // With fewer training vectors than `codebook_size`, k-means can only
        // produce `num_vectors` centroids; downstream LUT generation then
        // panics on a shape mismatch. Reject early with a clear message.
        if self.num_vectors < self.params.codebook_size {
            return Err(RetrieveError::InvalidParameter(format!(
                "IVF-PQ requires at least codebook_size training vectors \
                 (got {} vectors, codebook_size = {}). Add more vectors \
                 before build(), or lower codebook_size in IVFPQParams.",
                self.num_vectors, self.params.codebook_size
            )));
        }

        if training.kmeans_max_iter == 0 {
            return Err(RetrieveError::InvalidParameter(
                "kmeans_max_iter must be greater than 0".into(),
            ));
        }

        let training_indices = self.training_indices(training.sample_size)?;
        let training_count = training_indices.len();
        let min_training_count = self
            .params
            .num_clusters
            .min(self.num_vectors)
            .max(self.params.codebook_size);
        if training_count < min_training_count {
            return Err(RetrieveError::InvalidParameter(format!(
                "IVF-PQ training sample is too small (got {}, need at least {} \
                 for num_clusters={} and codebook_size={})",
                training_count,
                min_training_count,
                self.params.num_clusters,
                self.params.codebook_size
            )));
        }

        let sampled_training = if training_count == self.num_vectors {
            None
        } else {
            Some(copy_rows_by_index(
                &self.vectors,
                self.dimension,
                &training_indices,
            ))
        };
        let training_vectors = sampled_training.as_deref().unwrap_or(&self.vectors);

        // Stage 1: k-means clustering for IVF
        let mut kmeans =
            crate::partitioning::kmeans::KMeans::new(self.dimension, self.params.num_clusters)?
                .with_seed(self.params.seed)
                .with_max_iter(training.kmeans_max_iter);
        kmeans.fit(training_vectors, training_count)?;
        // Flatten centroids to contiguous storage for cache-friendly access.
        self.centroids = kmeans
            .centroids()
            .iter()
            .flat_map(|c| c.iter().copied())
            .collect();

        // Build HNSW coarse quantizer over centroids for fast lookup.
        #[cfg(feature = "hnsw")]
        {
            let num_centroids = self.centroids.len() / self.dimension;
            let mut hnsw = crate::hnsw::HNSWIndex::builder(self.dimension)
                .m(16)
                .ef_construction(200)
                .auto_normalize(true)
                .build()?;
            for i in 0..num_centroids {
                let centroid = self.get_centroid(i);
                hnsw.add_slice(i as u32, centroid)?;
            }
            hnsw.build()?;
            self.coarse_quantizer = Some(hnsw);
        }

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
                        if let Some(crate::filtering::MetadataValue::Int(n)) = metadata.get(field) {
                            // Category range validated at add_metadata time; skip silently
                            // if somehow out of range (defensive, should not occur).
                            let category_id = *n as u64;
                            if category_id < 64 {
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
                                crate::compression::IdCompressionMethod::DeltaVarint => {
                                    let compressor =
                                        crate::compression::DeltaVarintCompressor::new();
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
            let training_residuals = if training_count == self.num_vectors {
                None
            } else {
                Some(copy_rows_by_index(
                    &residuals,
                    self.dimension,
                    &training_indices,
                ))
            };
            let opq_training = training_residuals.as_deref().unwrap_or(&residuals);
            opq.fit(opq_training, training_count, 10)?; // 10 iterations
            Quantizer::Optimized(opq)
        } else {
            let mut pq = ProductQuantizer::new(
                self.dimension,
                self.params.num_codebooks,
                self.params.codebook_size,
            )?;
            let training_residuals = if training_count == self.num_vectors {
                None
            } else {
                Some(copy_rows_by_index(
                    &residuals,
                    self.dimension,
                    &training_indices,
                ))
            };
            let pq_training = training_residuals.as_deref().unwrap_or(&residuals);
            pq.fit_with_seed_and_max_iter(
                pq_training,
                training_count,
                Some(self.params.seed),
                training.kmeans_max_iter,
            )?;
            Quantizer::Product(pq)
        };

        // Quantize residual vectors
        self.quantized_codes = Vec::with_capacity(self.num_vectors * self.params.num_codebooks);
        for i in 0..self.num_vectors {
            let residual = &residuals[i * self.dimension..(i + 1) * self.dimension];
            let codes = pq.quantize(residual);
            self.quantized_codes.extend_from_slice(&codes);
        }
        self.build_fastscan_cache();

        self.pq = Some(pq);
        self.built = true;
        Ok(())
    }

    fn training_indices(
        &self,
        training_sample_size: Option<usize>,
    ) -> Result<Vec<usize>, RetrieveError> {
        let Some(sample_size) = training_sample_size else {
            return Ok((0..self.num_vectors).collect());
        };

        if sample_size == 0 {
            return Err(RetrieveError::InvalidParameter(
                "training_sample_size must be greater than 0".into(),
            ));
        }

        if sample_size >= self.num_vectors {
            return Ok((0..self.num_vectors).collect());
        }

        let mut indices: Vec<usize> = (0..self.num_vectors).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.params.seed);
        indices.shuffle(&mut rng);
        indices.truncate(sample_size);
        indices.sort_unstable();
        Ok(indices)
    }

    fn build_fastscan_cache(&mut self) {
        let num_cb = self.params.num_codebooks;
        if self.params.codebook_size != 16 {
            for cluster in &mut self.clusters {
                cluster.set_fastscan_codes(None);
            }
            return;
        }

        let quantized_codes = &self.quantized_codes;
        for cluster in &mut self.clusters {
            let ids = cluster.get_ids_ref();
            if ids.len() < SIMD_BATCH_THRESHOLD {
                cluster.set_fastscan_codes(None);
                continue;
            }

            let mut codes_batch = Vec::with_capacity(ids.len() * num_cb);
            for &vector_idx in ids.as_ref() {
                let start = vector_idx as usize * num_cb;
                codes_batch.extend_from_slice(&quantized_codes[start..start + num_cb]);
            }
            cluster.set_fastscan_codes(Some(PackedCodes4bit::pack(
                &codes_batch,
                ids.len(),
                num_cb,
            )));
        }
    }

    /// Drop raw f32 vectors after building to reduce memory usage.
    ///
    /// After `build()`, search uses only PQ codes and centroids -- raw vectors
    /// are no longer needed. Calling `compact()` frees ~`4 * dim * n` bytes.
    ///
    /// # Panics
    ///
    /// Panics if called before `build()`.
    pub fn compact(&mut self) {
        assert!(self.built, "compact() called before build()");
        self.vectors = Vec::new();
    }

    /// Save a built IVF-PQ index to a directory.
    ///
    /// The format stores large arrays as binary files and small metadata as
    /// `manifest.json`. Raw vectors are included only when still present; a
    /// compacted index reloads with approximate search available and
    /// `search_reranked()` unavailable.
    pub fn save_to_dir(&self, output_dir: impl AsRef<Path>) -> Result<(), RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot save unbuilt IVF-PQ index".into(),
            ));
        }
        if self.metadata.is_some() || self.filter_field.is_some() {
            return Err(RetrieveError::InvalidParameter(
                "IVF-PQ persistence does not yet include filter metadata".into(),
            ));
        }
        let pq = self
            .pq
            .clone()
            .ok_or_else(|| RetrieveError::InvalidParameter("missing IVF-PQ quantizer".into()))?;
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;

        let raw_vectors_present = self.vectors.len() == self.num_vectors * self.dimension;
        let manifest = IVFPQManifest {
            version: IVFPQ_FORMAT_VERSION,
            dimension: self.dimension,
            num_vectors: self.num_vectors,
            num_centroids: self.centroids.len() / self.dimension,
            raw_vectors_present,
            params: PersistedIVFPQParams::from(&self.params),
            quantizer: pq,
        };

        write_json_atomic(&output_dir.join("manifest.json"), &manifest)?;
        write_f32_atomic(&output_dir.join("centroids.bin"), &self.centroids)?;
        write_u32_atomic(&output_dir.join("doc_ids.bin"), &self.doc_ids)?;
        write_bytes_atomic(&output_dir.join("codes.bin"), &self.quantized_codes)?;
        write_clusters_atomic(&output_dir.join("clusters.bin"), &self.clusters)?;
        if raw_vectors_present {
            write_f32_atomic(&output_dir.join("raw_vectors.bin"), &self.vectors)?;
        }

        Ok(())
    }

    /// Load an IVF-PQ index saved by [`Self::save_to_dir`].
    pub fn load_from_dir(input_dir: impl AsRef<Path>) -> Result<Self, RetrieveError> {
        let input_dir = input_dir.as_ref();
        let manifest: IVFPQManifest = read_json(&input_dir.join("manifest.json"))?;
        if manifest.version != IVFPQ_FORMAT_VERSION {
            return Err(RetrieveError::FormatError(format!(
                "unsupported IVF-PQ format version {}",
                manifest.version
            )));
        }
        if manifest.dimension == 0 {
            return Err(RetrieveError::FormatError(
                "IVF-PQ manifest has zero dimension".into(),
            ));
        }
        if manifest.num_vectors == 0 {
            return Err(RetrieveError::FormatError(
                "IVF-PQ manifest has zero vectors".into(),
            ));
        }
        if manifest.num_centroids == 0 {
            return Err(RetrieveError::FormatError(
                "IVF-PQ manifest has zero centroids".into(),
            ));
        }

        let params = manifest.params.into_params();
        let mut index = Self::new(manifest.dimension, params)?;
        index.num_vectors = manifest.num_vectors;
        index.doc_ids = read_u32_exact(&input_dir.join("doc_ids.bin"), manifest.num_vectors)?;
        index.centroids = read_f32_exact(
            &input_dir.join("centroids.bin"),
            manifest.num_centroids * manifest.dimension,
        )?;
        index.quantized_codes = read_bytes_exact(
            &input_dir.join("codes.bin"),
            manifest.num_vectors * index.params.num_codebooks,
        )?;
        index.clusters = read_clusters(
            &input_dir.join("clusters.bin"),
            index.params.num_clusters,
            manifest.num_vectors,
        )?;
        index.vectors = if manifest.raw_vectors_present {
            read_f32_exact(
                &input_dir.join("raw_vectors.bin"),
                manifest.num_vectors * manifest.dimension,
            )?
        } else {
            Vec::new()
        };
        index.pq = Some(manifest.quantizer);
        index.built = true;
        index.build_fastscan_cache();

        Ok(index)
    }

    /// Search for k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        let candidates = self.search_approx_internal(query, k)?;
        Ok(candidates
            .into_iter()
            .map(|(vector_idx, dist)| (self.doc_ids[vector_idx as usize], dist))
            .collect())
    }

    /// Search with approximate IVF-PQ retrieval followed by exact f32 reranking.
    ///
    /// `candidate_pool` controls how many approximate candidates are reranked.
    /// Larger pools improve recall at the cost of exact distance work. This is
    /// unavailable after [`Self::compact`] because compaction drops the raw f32
    /// vectors needed for exact reranking.
    pub fn search_reranked(
        &self,
        query: &[f32],
        k: usize,
        candidate_pool: usize,
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

        if self.vectors.is_empty() {
            return Err(RetrieveError::InvalidParameter(
                "search_reranked unavailable after compact()".into(),
            ));
        }

        let pool = candidate_pool.max(k);
        let candidates = self.search_approx_internal(query, pool)?;
        let query_normalized = normalize_query(query);

        let mut reranked: Vec<(u32, f32)> = candidates
            .into_iter()
            .map(|(vector_idx, _approx_dist)| {
                let vector = self.get_vector(vector_idx as usize);
                let exact_dist =
                    crate::distance::cosine_distance_normalized(&query_normalized, vector);
                (self.doc_ids[vector_idx as usize], exact_dist)
            })
            .collect();
        reranked.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        reranked.truncate(k);
        Ok(reranked)
    }

    fn search_approx_internal(
        &self,
        query: &[f32],
        k: usize,
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

        let pq = self
            .pq
            .as_ref()
            .ok_or(RetrieveError::InvalidParameter("PQ not initialized".into()))?;

        // Normalize query (index operates on unit-length vectors)
        let query_normalized = normalize_query(query);
        let query = query_normalized.as_slice();

        // Find closest clusters
        let cluster_distances = self.find_nearest_centroids(query, self.params.nprobe);

        // Pre-allocate reusable buffers for the nprobe loop
        let mut candidates = Vec::new();
        let mut query_residual = vec![0.0f32; self.dimension];
        let mut codes_batch = Vec::new();
        let mut adc_table = Vec::new(); // Reused across clusters to avoid per-cluster allocation

        for (cluster_idx, _) in &cluster_distances {
            let cluster = &self.clusters[*cluster_idx];
            let ids = cluster.get_ids_ref();

            // Compute query residual in-place (no allocation)
            let centroid = self.get_centroid(*cluster_idx);
            for (i, (q, c)) in query.iter().zip(centroid.iter()).enumerate() {
                query_residual[i] = q - c;
            }

            // Build ADC table from query residual (reuses buffer)
            pq.compute_adc_table_into(&query_residual, &mut adc_table)?;

            if ids.len() >= SIMD_BATCH_THRESHOLD {
                let num_cb = self.params.num_codebooks;

                let distances = if self.params.codebook_size == 16 {
                    // FastScan path: 4-bit codes, SIMD shuffle-based lookup.
                    if let Some(packed) = cluster.fastscan_codes.as_ref() {
                        crate::pq_simd::fastscan_batch_flat(packed, &adc_table)
                    } else {
                        codes_batch.clear();
                        codes_batch.reserve(ids.len() * num_cb);
                        for &vector_idx in ids.as_ref() {
                            let start = vector_idx as usize * num_cb;
                            codes_batch
                                .extend_from_slice(&self.quantized_codes[start..start + num_cb]);
                        }
                        let packed = PackedCodes4bit::pack(&codes_batch, ids.len(), num_cb);
                        crate::pq_simd::fastscan_batch_flat(&packed, &adc_table)
                    }
                } else {
                    // Standard ADC batch path for larger codebooks
                    codes_batch.clear();
                    codes_batch.reserve(ids.len() * num_cb);
                    for &vector_idx in ids.as_ref() {
                        let start = vector_idx as usize * num_cb;
                        codes_batch.extend_from_slice(&self.quantized_codes[start..start + num_cb]);
                    }
                    let packed_lut = PackedLUT::from_flat(
                        &adc_table,
                        self.params.num_codebooks,
                        self.params.codebook_size,
                    );
                    adc_batch_dispatch(&codes_batch, num_cb, &packed_lut)
                };

                for (i, &vector_idx) in ids.iter().enumerate() {
                    candidates.push((vector_idx, distances[i]));
                }
            } else {
                // Scalar fallback for small clusters
                for &vector_idx in ids.as_ref() {
                    let start = vector_idx as usize * self.params.num_codebooks;
                    let end = start + self.params.num_codebooks;
                    let codes = &self.quantized_codes[start..end];

                    let dist = pq.distance_with_table(&adc_table, codes);
                    candidates.push((vector_idx, dist));
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

        // Extract category ID from filter (only supports integer equality on filter_field)
        let desired_category: u64 = match filter {
            crate::filtering::MetadataFilter::Equals { field, value } => {
                if Some(field) != self.filter_field.as_ref() {
                    return Err(RetrieveError::InvalidParameter(format!(
                        "filter field '{}' doesn't match index filter field '{:?}'",
                        field, self.filter_field
                    )));
                }
                match value {
                    crate::filtering::MetadataValue::Int(n) if *n >= 0 && *n < 64 => *n as u64,
                    crate::filtering::MetadataValue::Int(n) => {
                        return Err(RetrieveError::InvalidParameter(format!(
                            "category ID {} exceeds bitmask limit of 63",
                            n
                        )));
                    }
                    _ => {
                        return Err(RetrieveError::InvalidParameter(
                            "category ID must be an integer in 0..63 for bitmask filtering".into(),
                        ));
                    }
                }
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

        // Find closest clusters
        let cluster_distances = self.find_nearest_centroids(query, self.params.nprobe);

        // Search in top nprobe clusters, skipping those without matching vectors
        let mut candidates = Vec::new();
        let mut query_residual = vec![0.0f32; self.dimension];
        let mut adc_table = Vec::new();

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

            pq.compute_adc_table_into(&query_residual, &mut adc_table)?;

            // Search vectors in this cluster, filtering by metadata
            if let Some(ref metadata_store) = self.metadata {
                let ids = cluster.get_ids_ref();
                for &vector_idx in ids.as_ref() {
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

    /// Find the nprobe nearest centroid indices to `query`, sorted by distance.
    /// Uses HNSW coarse quantizer when available, brute-force otherwise.
    fn find_nearest_centroids(&self, query: &[f32], nprobe: usize) -> Vec<(usize, f32)> {
        #[cfg(feature = "hnsw")]
        if let Some(ref hnsw) = self.coarse_quantizer {
            let ef = nprobe * 2;
            if let Ok(results) = hnsw.search(query, nprobe, ef.max(nprobe)) {
                return results
                    .into_iter()
                    .map(|(id, d)| (id as usize, d))
                    .collect();
            }
        }

        let num_centroids = self.centroids.len() / self.dimension;
        let mut dists: Vec<(usize, f32)> = (0..num_centroids)
            .map(|idx| {
                let c = self.get_centroid(idx);
                (idx, crate::distance::cosine_distance_normalized(query, c))
            })
            .collect();
        let nprobe = nprobe.min(dists.len());
        if nprobe < dists.len() {
            dists.select_nth_unstable_by(nprobe, |a, b| a.1.total_cmp(&b.1));
            dists.truncate(nprobe);
        }
        dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        dists
    }
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

fn write_bytes_atomic(path: &Path, bytes: &[u8]) -> Result<(), RetrieveError> {
    write_atomic(path, |writer| writer.write_all(bytes))
}

fn write_clusters_atomic(path: &Path, clusters: &[Cluster]) -> Result<(), RetrieveError> {
    write_atomic(path, |writer| {
        writer.write_all(IVFPQ_CLUSTER_MAGIC)?;
        writer.write_all(&(clusters.len() as u64).to_le_bytes())?;
        for cluster in clusters {
            writer.write_all(&cluster.filter_bitmask.to_le_bytes())?;
            let ids = cluster.get_ids_ref();
            writer.write_all(&(ids.len() as u64).to_le_bytes())?;
            for id in ids.as_ref() {
                writer.write_all(&id.to_le_bytes())?;
            }
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
        .ok_or_else(|| RetrieveError::FormatError("f32 byte length overflow".into()))?;
    if bytes.len() != expected_bytes {
        return Err(RetrieveError::FormatError(format!(
            "{} size mismatch: expected {} bytes, got {}",
            path.display(),
            expected_bytes,
            bytes.len()
        )));
    }
    let mut values = Vec::with_capacity(expected_len);
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

fn read_u32_exact(path: &Path, expected_len: usize) -> Result<Vec<u32>, RetrieveError> {
    let bytes = std::fs::read(path)?;
    let expected_bytes = expected_len
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| RetrieveError::FormatError("u32 byte length overflow".into()))?;
    if bytes.len() != expected_bytes {
        return Err(RetrieveError::FormatError(format!(
            "{} size mismatch: expected {} bytes, got {}",
            path.display(),
            expected_bytes,
            bytes.len()
        )));
    }
    let mut values = Vec::with_capacity(expected_len);
    for chunk in bytes.chunks_exact(4) {
        values.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

fn read_bytes_exact(path: &Path, expected_len: usize) -> Result<Vec<u8>, RetrieveError> {
    let bytes = std::fs::read(path)?;
    if bytes.len() != expected_len {
        return Err(RetrieveError::FormatError(format!(
            "{} size mismatch: expected {} bytes, got {}",
            path.display(),
            expected_len,
            bytes.len()
        )));
    }
    Ok(bytes)
}

fn read_clusters(
    path: &Path,
    expected_clusters: usize,
    num_vectors: usize,
) -> Result<Vec<Cluster>, RetrieveError> {
    let mut reader = BufReader::new(std::fs::File::open(path)?);
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != IVFPQ_CLUSTER_MAGIC {
        return Err(RetrieveError::FormatError(
            "invalid IVF-PQ cluster file magic".into(),
        ));
    }
    let cluster_count = read_u64(&mut reader)? as usize;
    if cluster_count != expected_clusters {
        return Err(RetrieveError::FormatError(format!(
            "cluster count mismatch: expected {}, got {}",
            expected_clusters, cluster_count
        )));
    }

    let mut clusters = Vec::with_capacity(cluster_count);
    for _ in 0..cluster_count {
        let filter_bitmask = read_u64(&mut reader)?;
        let len = read_u64(&mut reader)? as usize;
        if len > num_vectors {
            return Err(RetrieveError::FormatError(format!(
                "cluster length {} exceeds vector count {}",
                len, num_vectors
            )));
        }
        let mut ids = Vec::with_capacity(len);
        for _ in 0..len {
            let id = read_u32(&mut reader)?;
            if id as usize >= num_vectors {
                return Err(RetrieveError::FormatError(format!(
                    "cluster id {} exceeds vector count {}",
                    id, num_vectors
                )));
            }
            ids.push(id);
        }
        clusters.push(Cluster::new(ids, filter_bitmask));
    }

    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        return Err(RetrieveError::FormatError(
            "trailing bytes in IVF-PQ cluster file".into(),
        ));
    }

    Ok(clusters)
}

fn read_u64(reader: &mut impl Read) -> Result<u64, RetrieveError> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_u32(reader: &mut impl Read) -> Result<u32, RetrieveError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// compact() drops raw vectors; search still returns results using PQ distances.
    #[test]
    fn compact_search_works() {
        let dim = 16;
        let n = 200;
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let params = IVFPQParams {
            num_clusters: 4,
            num_codebooks: 4,
            codebook_size: 16,
            nprobe: 4,
            ..IVFPQParams::default()
        };
        let mut index = IVFPQIndex::new(dim, params).unwrap();

        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.add(i as u32, v).unwrap();
        }
        index.build().unwrap();

        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let before = index.search(&query, 5).unwrap();

        index.compact();
        assert!(index.vectors.is_empty());

        let after = index.search(&query, 5).unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn reranked_search_uses_exact_distances() {
        let dim = 16;
        let n = 200;
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);

        let params = IVFPQParams {
            num_clusters: 4,
            num_codebooks: 4,
            codebook_size: 16,
            nprobe: 4,
            ..IVFPQParams::default()
        };
        let mut index = IVFPQIndex::new(dim, params).unwrap();

        let mut vectors = Vec::new();
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.add(i as u32, v.clone()).unwrap();
            vectors.push(v);
        }
        index.build().unwrap();

        let query_id = 17u32;
        let results = index
            .search_reranked(&vectors[query_id as usize], 1, n)
            .unwrap();
        assert_eq!(results[0].0, query_id);
        assert!(
            results[0].1.abs() < 1e-5,
            "self-query exact distance should be near zero, got {}",
            results[0].1
        );
    }

    #[test]
    fn search_and_rerank_return_external_doc_ids() {
        let dim = 16;
        let n = 200;
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);

        let params = IVFPQParams {
            num_clusters: 4,
            num_codebooks: 4,
            codebook_size: 16,
            nprobe: 4,
            ..IVFPQParams::default()
        };
        let mut index = IVFPQIndex::new(dim, params).unwrap();

        let mut vectors = Vec::new();
        let mut doc_ids = Vec::new();
        for i in 0..n {
            let doc_id = 10_000 + (i as u32 * 7);
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.add(doc_id, v.clone()).unwrap();
            vectors.push(v);
            doc_ids.push(doc_id);
        }
        index.build().unwrap();

        let query_idx = 17usize;
        let query = &vectors[query_idx];
        let search_results = index.search(query, 10).unwrap();
        assert!(search_results.iter().all(|(id, _)| doc_ids.contains(id)));

        let reranked = index.search_reranked(query, 1, n).unwrap();
        assert_eq!(reranked[0].0, doc_ids[query_idx]);
    }

    #[test]
    fn sub_4bit_codebook_search_uses_standard_adc() {
        let dim = 8;
        let n = 96;
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(10);

        let params = IVFPQParams {
            num_clusters: 4,
            num_codebooks: 4,
            codebook_size: 8,
            nprobe: 4,
            ..IVFPQParams::default()
        };
        let mut index = IVFPQIndex::new(dim, params).unwrap();

        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.add(1_000 + i as u32, v).unwrap();
        }
        index.build().unwrap();

        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let results = index.search(&query, 5).unwrap();
        assert_eq!(results.len(), 5);
        assert!(results
            .iter()
            .all(|(id, dist)| *id >= 1_000 && dist.is_finite()));
    }

    #[test]
    fn four_bit_builds_prepacked_fastscan_codes() {
        let dim = 16;
        let n = 200;
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);

        let params = IVFPQParams {
            num_clusters: 4,
            num_codebooks: 4,
            codebook_size: 16,
            nprobe: 4,
            ..IVFPQParams::default()
        };
        let mut index = IVFPQIndex::new(dim, params).unwrap();

        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.add(i as u32, v).unwrap();
        }
        index.build().unwrap();

        assert!(
            index
                .clusters
                .iter()
                .filter(|cluster| cluster.len() >= SIMD_BATCH_THRESHOLD)
                .all(|cluster| cluster.fastscan_codes.is_some()),
            "clusters large enough for batched 4-bit search should be prepacked"
        );
    }

    #[test]
    fn sampled_training_is_deterministic_with_seed() {
        let dim = 16;
        let n = 180;
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(15);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
            .collect();
        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();

        let build = || {
            let params = IVFPQParams {
                num_clusters: 8,
                num_codebooks: 4,
                codebook_size: 16,
                nprobe: 8,
                seed: 77,
                ..IVFPQParams::default()
            };
            let mut index = IVFPQIndex::new(dim, params).unwrap();
            for (i, vector) in vectors.iter().enumerate() {
                index.add_slice(i as u32, vector).unwrap();
            }
            index.build_with_training_sample(64).unwrap();
            #[cfg(feature = "hnsw")]
            {
                index.coarse_quantizer = None;
            }
            index
        };

        let a = build();
        let b = build();
        assert_eq!(a.centroids, b.centroids);
        assert_eq!(a.quantized_codes, b.quantized_codes);
        assert_eq!(a.search(&query, 10).unwrap(), b.search(&query, 10).unwrap());
    }

    #[test]
    fn sampled_training_indexes_all_inserted_vectors() {
        let dim = 16;
        let n = 180;
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(16);

        let params = IVFPQParams {
            num_clusters: 8,
            num_codebooks: 4,
            codebook_size: 16,
            nprobe: 8,
            seed: 88,
            ..IVFPQParams::default()
        };
        let mut index = IVFPQIndex::new(dim, params).unwrap();
        let mut vectors = Vec::new();
        for i in 0..n {
            let doc_id = 50_000 + i as u32;
            let vector: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.add(doc_id, vector.clone()).unwrap();
            vectors.push(vector);
        }
        index.build_with_training_sample(64).unwrap();

        let query_idx = n - 1;
        let results = index.search_reranked(&vectors[query_idx], 1, n).unwrap();
        assert_eq!(results[0].0, 50_000 + query_idx as u32);
    }

    #[test]
    fn save_load_sampled_training_preserves_search() {
        let dim = 16;
        let n = 180;
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(17);

        let params = IVFPQParams {
            num_clusters: 8,
            num_codebooks: 4,
            codebook_size: 16,
            nprobe: 8,
            seed: 99,
            ..IVFPQParams::default()
        };
        let mut index = IVFPQIndex::new(dim, params).unwrap();
        for i in 0..n {
            let vector: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.add(10_000 + i as u32, vector).unwrap();
        }
        index.build_with_training_options(Some(64), 5).unwrap();
        #[cfg(feature = "hnsw")]
        {
            index.coarse_quantizer = None;
        }

        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let approx_before = index.search(&query, 10).unwrap();
        let reranked_before = index.search_reranked(&query, 10, 80).unwrap();

        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let loaded = IVFPQIndex::load_from_dir(dir.path()).unwrap();

        assert_eq!(loaded.search(&query, 10).unwrap(), approx_before);
        assert_eq!(
            loaded.search_reranked(&query, 10, 80).unwrap(),
            reranked_before
        );
    }

    #[test]
    fn reranked_search_requires_raw_vectors() {
        let dim = 16;
        let n = 200;
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(8);

        let params = IVFPQParams {
            num_clusters: 4,
            num_codebooks: 4,
            codebook_size: 16,
            nprobe: 4,
            ..IVFPQParams::default()
        };
        let mut index = IVFPQIndex::new(dim, params).unwrap();

        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.add(i as u32, v).unwrap();
        }
        index.build().unwrap();
        index.compact();

        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let err = index.search_reranked(&query, 5, 50).unwrap_err();
        assert!(
            err.to_string().contains("search_reranked unavailable"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn save_load_roundtrip_preserves_search_and_rerank() {
        let dim = 16;
        let n = 240;
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);

        let params = IVFPQParams {
            num_clusters: 4,
            num_codebooks: 4,
            codebook_size: 16,
            nprobe: 4,
            seed: 123,
            ..IVFPQParams::default()
        };
        let mut index = IVFPQIndex::new(dim, params).unwrap();
        let mut doc_ids = Vec::new();
        for i in 0..n {
            let doc_id = 10_000 + i as u32;
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.add(doc_id, v).unwrap();
            doc_ids.push(doc_id);
        }
        index.build().unwrap();
        #[cfg(feature = "hnsw")]
        {
            index.coarse_quantizer = None;
        }

        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let approx_before = index.search(&query, 10).unwrap();
        let reranked_before = index.search_reranked(&query, 10, 80).unwrap();

        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let loaded = IVFPQIndex::load_from_dir(dir.path()).unwrap();

        assert_eq!(loaded.search(&query, 10).unwrap(), approx_before);
        assert_eq!(
            loaded.search_reranked(&query, 10, 80).unwrap(),
            reranked_before
        );
        assert!(loaded
            .clusters
            .iter()
            .filter(|cluster| cluster.len() >= SIMD_BATCH_THRESHOLD)
            .all(|cluster| cluster.fastscan_codes.is_some()));
        assert!(approx_before.iter().all(|(id, _)| doc_ids.contains(id)));
    }

    #[test]
    fn save_load_compacted_index_keeps_approximate_search_only() {
        let dim = 16;
        let n = 200;
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(12);

        let params = IVFPQParams {
            num_clusters: 4,
            num_codebooks: 4,
            codebook_size: 16,
            nprobe: 4,
            seed: 124,
            ..IVFPQParams::default()
        };
        let mut index = IVFPQIndex::new(dim, params).unwrap();
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.add(i as u32, v).unwrap();
        }
        index.build().unwrap();
        #[cfg(feature = "hnsw")]
        {
            index.coarse_quantizer = None;
        }
        index.compact();

        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let approx_before = index.search(&query, 10).unwrap();

        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let loaded = IVFPQIndex::load_from_dir(dir.path()).unwrap();

        assert_eq!(loaded.search(&query, 10).unwrap(), approx_before);
        let err = loaded.search_reranked(&query, 10, 80).unwrap_err();
        assert!(
            err.to_string().contains("search_reranked unavailable"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_rejects_future_manifest_version() {
        let dim = 16;
        let n = 200;
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(13);

        let params = IVFPQParams {
            num_clusters: 4,
            num_codebooks: 4,
            codebook_size: 16,
            nprobe: 4,
            ..IVFPQParams::default()
        };
        let mut index = IVFPQIndex::new(dim, params).unwrap();
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.add(i as u32, v).unwrap();
        }
        index.build().unwrap();

        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let manifest_path = dir.path().join("manifest.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        manifest["version"] = serde_json::json!(IVFPQ_FORMAT_VERSION + 1);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let err = IVFPQIndex::load_from_dir(dir.path()).unwrap_err();
        assert!(
            err.to_string()
                .contains("unsupported IVF-PQ format version"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_rejects_corrupt_cluster_magic() {
        let dim = 16;
        let n = 200;
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(14);

        let params = IVFPQParams {
            num_clusters: 4,
            num_codebooks: 4,
            codebook_size: 16,
            nprobe: 4,
            ..IVFPQParams::default()
        };
        let mut index = IVFPQIndex::new(dim, params).unwrap();
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.add(i as u32, v).unwrap();
        }
        index.build().unwrap();

        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        std::fs::write(dir.path().join("clusters.bin"), b"not ivfpq").unwrap();

        let err = IVFPQIndex::load_from_dir(dir.path()).unwrap_err();
        assert!(
            err.to_string()
                .contains("invalid IVF-PQ cluster file magic"),
            "unexpected error: {err}"
        );
    }
}
