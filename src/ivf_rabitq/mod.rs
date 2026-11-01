//! IVF-RaBitQ: Inverted File with Randomized Binary Quantization.
//!
//! Combines IVF partitioning with RaBitQ quantization for memory-efficient ANN
//! search. Unlike IVF-PQ, RaBitQ requires no codebook training, only a random
//! rotation matrix and per-vector correction factors.
//!
//! # Feature Flag
//!
//! ```toml
//! vicinity = { version = "0.10.5", features = ["ivf_rabitq"] }
//! ```
//!
//! # Quick Start
//!
//! ```ignore
//! use vicinity::ivf_rabitq::{IVFRaBitQIndex, IVFRaBitQParams};
//!
//! let params = IVFRaBitQParams {
//!     num_clusters: 256,
//!     nprobe: 10,
//!     ..Default::default()
//! };
//!
//! let mut index = IVFRaBitQIndex::new(128, params)?;
//! for (id, vec) in data {
//!     index.add(id, vec)?;
//! }
//! index.build()?;
//!
//! let results = index.search(&query, 10)?;
//! ```
//!
//! # Memory Comparison (d=128, n=1M)
//!
//! | Method | Bytes/vector | Total | Recall@10 |
//! |--------|-------------|-------|-----------|
//! | Raw f32 | 512 | 512 MB | 100% |
//! | IVF-PQ (M=8) | 8 | 8 MB | ~85% |
//! | IVF-RaBitQ (4-bit) | 72 | 72 MB | ~95% |
//! | IVF-RaBitQ (1-bit) | 24 | 24 MB | ~80% |
//!
//! RaBitQ byte counts are the compact encoding: `d * bits / 8` code bytes plus
//! two f32 correction factors. This implementation's in-memory index also keeps
//! per-dimension codes and other fields; `memory_usage()` reports the actual size.
//!
//! The 4-bit variant uses more memory than PQ but achieves higher recall because
//! RaBitQ preserves per-dimension information (vs PQ's subspace compression).
//! The 1-bit variant is competitive with PQ on memory while offering faster
//! distance computation via popcount.
//!
//! # How It Works
//!
//! 1. **IVF partitioning**: k-means clusters vectors into cells
//! 2. **Residual computation**: subtract cluster centroid from each vector
//! 3. **Rotation**: FHT/Kac random rotation ([`rotator::FhtKacRotator`])
//! 4. **RaBitQ quantization**: sign/extended bits + correction factors
//! 5. **Search**: probe nearest centroids, scan clusters with RaBitQ approximate
//!    distances, skip candidates the error bound excludes, exact-rerank survivors
//!
//! Built non-compacted indexes can be saved with `IVFRaBitQIndex::save_to_dir()`
//! and restored with `IVFRaBitQIndex::load_from_dir()`. The current format
//! reloads into memory and rebuilds edge codes from raw vectors.
//!
//! # Rotation
//!
//! The rotation is not optional decoration: RaBitQ's error analysis assumes a
//! uniformly random orthogonal transform, and constructing one dense
//! `D × D` matrix costs `O(D²)` per vector. This index uses the reference
//! library's default: the **FHT/Kac** rotator, four Rademacher sign passes, a
//! normalized fast Hadamard transform and a `π/4` Givens step over the padded
//! tail, at `O(D log D)` per vector with `4D` bits of state. The working
//! dimension is the input dimension rounded up to a multiple of 32, and that
//! padded dimension is persisted so a reload cannot silently change the
//! sampled transform.
//!
//! Data, centroids **and** queries all go through the same rotation.
//! [`IVFRaBitQIndex::dequantize`] un-rotates a stored code back into the
//! original coordinate system.
//!
//! [`RotationKind::Identity`] exists for ablation only. Without a rotation the
//! signs of the residual coordinates are correlated with the coordinate axes,
//! quantization error stops being isotropic, and recall drops sharply.
//!
//! # Distance and the error bound
//!
//! Distance is computed asymmetrically: query is exact f32, database vectors are
//! quantized. The correction factors (`f_add`, `f_rescale`) provide the
//! distance estimate and `f_error` its margin.
//!
//! Search uses that margin to skip candidates that cannot enter the result
//! heap: `lower_bound = max(0, estimate - f_error · ‖q − c‖)` and a candidate
//! is skipped when `lower_bound ≥` the heap's worst entry (FAISS
//! `RaBitQUtils.h::should_refine_candidate`). Survivors are reranked with exact
//! f32 distance when raw vectors are still resident.
//!
//! **The bound is probabilistic, not a guarantee.** `f_error` is RaBitQ's
//! `1.9·ε` margin under the random-rotation model
//! (FAISS `RaBitQUtils.cpp`, `kConstEpsilon = 1.9`). It bounds the estimator in
//! expectation over rotations, so a specific dataset can exceed it and a true
//! neighbor can be pruned. It is a speed/recall tradeoff, not a correctness
//! proof.
//!
//! # References
//!
//! - Gao, Long (2024). "RaBitQ: Quantizing High-Dimensional Vectors with a
//!   Theoretical Error Bound for Approximate Nearest Neighbor Search." SIGMOD 2024.
//! - Shi, Gao, Xia, Fehér, Long (2026). "GPU-Native Approximate Nearest Neighbor
//!   Search with IVF-RaBitQ: Fast Index Build and Search." arXiv:2602.23999.
//! - RaBitQ-Library `docs/docs/rabitq/rotator.md` (FHT/Kac default rotator,
//!   32-wide padding, `4D` bits of state).
//! - FAISS `faiss/impl/RaBitQUtils.cpp` (`kConstEpsilon = 1.9`),
//!   `faiss/impl/RaBitQUtils.h` (`should_refine_candidate` lower bound),
//!   `faiss/impl/RaBitQuantizer.cpp` (refine only when the bound can win).

pub mod rotator;

use crate::distance::FloatOrd;
use crate::RetrieveError;
use qntz::rabitq::{RaBitQConfig, RaBitQQuantizer};
use rotator::FhtKacRotator;
use serde::{Deserialize, Serialize};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

const IVFRABITQ_FORMAT_VERSION: u32 = 2;
const IVFRABITQ_CLUSTERS_MAGIC: &[u8; 8] = b"IVFRBQCL";
static IVFRABITQ_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Which rotation the index applies before quantizing residuals.
///
/// [`RotationKind::FhtKac`] is the reference default and the only supported
/// production setting. [`RotationKind::Identity`] exists so rotation can be
/// ablated against the same code path; RaBitQ's error model assumes a random
/// orthogonal rotation, so recall degrades without one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RotationKind {
    /// FHT/Kac structured rotation: `O(D log D)` per vector, `4D` bits of state.
    #[default]
    FhtKac,
    /// No rotation. Ablation only; see [`RotationKind::FhtKac`].
    Identity,
}

/// IVF-RaBitQ parameters.
#[derive(Clone, Debug)]
pub struct IVFRaBitQParams {
    /// Number of IVF clusters (inverted lists).
    pub num_clusters: usize,
    /// Number of clusters to probe during search.
    pub nprobe: usize,
    /// RaBitQ bits per dimension (1-8). Default: 4.
    pub total_bits: usize,
    /// Random seed for the rotation and for k-means initialization.
    pub seed: u64,
    /// Residual rotation. Default: [`RotationKind::FhtKac`].
    pub rotation: RotationKind,
    /// Skip candidates whose error-bound lower bound cannot beat the heap.
    ///
    /// Default: `true`. The bound is probabilistic (see the module docs), so
    /// pathological data can make it prune true neighbors; set this to `false`
    /// to rerank every candidate in the probed lists instead.
    pub use_error_bound: bool,
}

impl Default for IVFRaBitQParams {
    fn default() -> Self {
        Self {
            num_clusters: 256,
            nprobe: 10,
            total_bits: 4,
            seed: 42,
            rotation: RotationKind::FhtKac,
            use_error_bound: true,
        }
    }
}

/// Candidate accounting for one search, for tuning and for rotation ablations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IVFRaBitQSearchStats {
    /// Candidates scanned in the probed inverted lists.
    pub candidates_scanned: usize,
    /// Candidates skipped because the error-bound lower bound could not beat
    /// the heap's worst entry.
    pub pruned_by_bound: usize,
    /// Candidates reranked with exact f32 distance.
    pub exact_reranks: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PersistedIVFRaBitQParams {
    num_clusters: usize,
    nprobe: usize,
    total_bits: usize,
    seed: u64,
    rotation: RotationKind,
    use_error_bound: bool,
}

impl From<&IVFRaBitQParams> for PersistedIVFRaBitQParams {
    fn from(params: &IVFRaBitQParams) -> Self {
        Self {
            num_clusters: params.num_clusters,
            nprobe: params.nprobe,
            total_bits: params.total_bits,
            seed: params.seed,
            rotation: params.rotation,
            use_error_bound: params.use_error_bound,
        }
    }
}

impl PersistedIVFRaBitQParams {
    fn into_params(self) -> IVFRaBitQParams {
        IVFRaBitQParams {
            num_clusters: self.num_clusters,
            nprobe: self.nprobe,
            total_bits: self.total_bits,
            seed: self.seed,
            rotation: self.rotation,
            use_error_bound: self.use_error_bound,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct IVFRaBitQManifest {
    version: u32,
    dimension: usize,
    /// Working dimension the codes were built in. Persisted because the
    /// sampled FHT/Kac transform depends on it: reloading with a different
    /// padded dimension silently changes the rotation and therefore recall.
    padded_dim: usize,
    num_vectors: usize,
    params: PersistedIVFRaBitQParams,
}

/// A cluster (inverted list) storing RaBitQ-quantized residual vectors.
///
/// Vectors are quantized via qntz's edge API: each `v` is stored as an
/// [`EdgeQuantizedVector`](qntz::rabitq::EdgeQuantizedVector) whose "parent"
/// is the cluster centroid. At search time the caller adds the per-cluster
/// `||q - cluster_c||^2` to each edge term, yielding an absolute distance
/// that is consistent across clusters (and therefore rankable against other
/// probed clusters in a shared heap).
#[derive(Debug)]
struct Cluster {
    /// Indices into the global vector array (insertion order).
    vector_indices: Vec<u32>,
    /// Edge-quantized residuals (`v - cluster_c`) for each vector in this cluster.
    quantized: Vec<qntz::rabitq::EdgeQuantizedVector>,
}

/// IVF-RaBitQ index.
pub struct IVFRaBitQIndex {
    dimension: usize,
    params: IVFRaBitQParams,
    built: bool,
    compacted: bool,

    // Raw vectors (kept for build and exact reranking; drop with compact() to save memory)
    vectors: Vec<f32>,
    num_vectors: usize,
    doc_ids: Vec<u32>,

    // IVF components
    clusters: Vec<Cluster>,
    /// Flat centroid storage: `[c0_d0, c0_d1, ..., c1_d0, ...]`.
    centroids: Vec<f32>,

    // RaBitQ quantizer over the padded, pre-rotated residual space
    quantizer: RaBitQQuantizer,
    /// FHT/Kac rotator; `None` for [`RotationKind::Identity`].
    rotator: Option<FhtKacRotator>,
    /// Working dimension: the quantizer and every code operate in this space.
    padded_dim: usize,

    /// HNSW coarse quantizer over centroids for O(log k) lookup.
    #[cfg(feature = "hnsw")]
    coarse_quantizer: Option<crate::hnsw::HNSWIndex>,
}

impl IVFRaBitQIndex {
    /// Create a new IVF-RaBitQ index.
    pub fn new(dimension: usize, params: IVFRaBitQParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidConfig {
                field: "dimension",
                expected: "must be greater than 0",
            });
        }

        let config = RaBitQConfig {
            total_bits: params.total_bits,
            t_const: None,
        };
        let rotator = match params.rotation {
            RotationKind::FhtKac => Some(FhtKacRotator::new(dimension, params.seed)?),
            RotationKind::Identity => None,
        };
        let padded_dim = rotator
            .as_ref()
            .map_or(dimension, FhtKacRotator::padded_dim);
        let quantizer = RaBitQQuantizer::with_config(padded_dim, params.seed, config)
            .map_err(|e| RetrieveError::InvalidParameter(format!("RaBitQ config: {e}")))?;

        Ok(Self {
            dimension,
            params,
            built: false,
            compacted: false,
            vectors: Vec::new(),
            num_vectors: 0,
            doc_ids: Vec::new(),
            clusters: Vec::new(),
            centroids: Vec::new(),
            quantizer,
            rotator,
            padded_dim,
            #[cfg(feature = "hnsw")]
            coarse_quantizer: None,
        })
    }

    /// Working dimension of the rotated space.
    ///
    /// Equal to [`Self::dimension`] rounded up to a multiple of 32 under
    /// [`RotationKind::FhtKac`], and equal to [`Self::dimension`] under
    /// [`RotationKind::Identity`].
    pub fn padded_dimension(&self) -> usize {
        self.padded_dim
    }

    /// Rotate a vector from the input space into the working space.
    fn rotate(&self, vector: &[f32]) -> Result<Vec<f32>, RetrieveError> {
        match self.rotator {
            Some(ref rotator) => rotator.rotate(vector),
            None => Ok(vector.to_vec()),
        }
    }

    /// Invert [`Self::rotate`].
    fn unrotate(&self, rotated: &[f32]) -> Result<Vec<f32>, RetrieveError> {
        match self.rotator {
            Some(ref rotator) => rotator.unrotate(rotated),
            None => Ok(rotated.to_vec()),
        }
    }

    /// Set the number of clusters to probe during search.
    pub fn set_nprobe(&mut self, nprobe: usize) {
        self.params.nprobe = nprobe;
    }

    /// Add a vector to the index.
    pub fn add(&mut self, doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        self.add_slice(doc_id, &vector)
    }

    /// Add a vector from a borrowed slice.
    pub fn add_slice(&mut self, doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
        if self.built {
            return Err(RetrieveError::AlreadyBuilt {
                component: "IVF-RaBitQ index",
                operation: "add vectors",
            });
        }
        if vector.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: vector.len(),
                doc_dim: self.dimension,
            });
        }

        // L2-normalize on insertion (cosine index)
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

    /// Add a batch of vectors.
    pub fn add_batch(&mut self, doc_ids: &[u32], vectors: &[f32]) -> Result<(), RetrieveError> {
        let expected_len = checked_batch_len(doc_ids.len(), self.dimension)?;
        if vectors.len() != expected_len {
            return Err(RetrieveError::InvalidParameter(format!(
                "expected {} floats for {} vectors of dim {}, got {}",
                expected_len,
                doc_ids.len(),
                self.dimension,
                vectors.len()
            )));
        }
        for (i, &doc_id) in doc_ids.iter().enumerate() {
            let start = i * self.dimension;
            let end = start + self.dimension;
            self.add_slice(doc_id, &vectors[start..end])?;
        }
        Ok(())
    }

    /// Build the index: cluster vectors, quantize residuals.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        // Stage 1: k-means clustering. The seed makes both the k-means++ start
        // and (below) the rotation repeatable, so a seeded index is reproducible.
        let num_clusters = self.params.num_clusters.min(self.num_vectors);
        let mut kmeans = crate::partitioning::kmeans::KMeans::new(self.dimension, num_clusters)?
            .with_seed(self.params.seed);
        kmeans.fit(&self.vectors, self.num_vectors)?;

        self.centroids = kmeans
            .centroids()
            .iter()
            .flat_map(|c: &Vec<f32>| c.iter().copied())
            .collect();

        self.rebuild_coarse_quantizer()?;

        // Stage 2: assign vectors to clusters
        let assignments = kmeans.assign_clusters(&self.vectors, self.num_vectors);

        // Build per-cluster vector lists
        let mut cluster_indices: Vec<Vec<u32>> = vec![Vec::new(); num_clusters];
        for (vector_idx, &cluster_idx) in assignments.iter().enumerate() {
            cluster_indices[cluster_idx].push(vector_idx as u32);
        }

        self.rebuild_quantized_clusters(cluster_indices)?;

        self.built = true;
        Ok(())
    }

    #[cfg(feature = "hnsw")]
    fn rebuild_coarse_quantizer(&mut self) -> Result<(), RetrieveError> {
        let nc = self.centroids.len() / self.dimension;
        let mut hnsw = crate::hnsw::HNSWIndex::builder(self.dimension)
            .m(16)
            .ef_construction(200)
            .auto_normalize(true)
            .build()?;
        for i in 0..nc {
            let centroid = self.get_centroid(i);
            hnsw.add_slice(i as u32, centroid)?;
        }
        hnsw.build()?;
        self.coarse_quantizer = Some(hnsw);
        Ok(())
    }

    #[cfg(not(feature = "hnsw"))]
    fn rebuild_coarse_quantizer(&mut self) -> Result<(), RetrieveError> {
        Ok(())
    }

    fn rebuild_quantized_clusters(
        &mut self,
        cluster_indices: Vec<Vec<u32>>,
    ) -> Result<(), RetrieveError> {
        // Each cluster centroid plays the role of `parent` in VR; baking
        // `<R*cluster_c, xu_cb>` into the edge lets search form an absolute
        // `||q - v||^2` that is comparable across probed clusters.
        //
        // Data and centroids go through the same FHT/Kac rotation as the
        // query, so `R*v - R*c = R*(v - c)` holds exactly and the residual
        // handed to qntz is already rotated.
        self.clusters = Vec::with_capacity(cluster_indices.len());
        for (cluster_idx, indices) in cluster_indices.into_iter().enumerate() {
            let centroid = self.get_centroid(cluster_idx).to_vec();
            let centroid_rot = self.rotate(&centroid)?;
            let mut quantized = Vec::with_capacity(indices.len());

            for &vector_idx in &indices {
                let vec = self.get_vector(vector_idx as usize).to_vec();
                // R*(v - c) via O(d) subtraction of the pre-rotated views.
                let vec_rot = self.rotate(&vec)?;
                let residual_rot: Vec<f32> = vec_rot
                    .iter()
                    .zip(centroid_rot.iter())
                    .map(|(a, b)| a - b)
                    .collect();
                let edge = self
                    .quantizer
                    .quantize_edge_prerotated(&centroid_rot, &residual_rot)
                    .map_err(|e| {
                        RetrieveError::InvalidParameter(format!("RaBitQ quantize: {e}"))
                    })?;
                quantized.push(edge);
            }

            self.clusters.push(Cluster {
                vector_indices: indices,
                quantized,
            });
        }
        Ok(())
    }

    /// Drop raw f32 vectors after building to reduce memory usage.
    ///
    /// Calling `compact()` after `build()` drops the raw f32 vectors, reducing
    /// memory by ~4*dim bytes per vector. Search results will use approximate
    /// squared distances from quantized codes instead of exact cosine reranking.
    /// Negative estimates are reported as zero after ranking and selection.
    ///
    /// # Panics
    ///
    /// Panics if called before `build()`.
    pub fn compact(&mut self) {
        assert!(self.built, "compact() called before build()");
        self.vectors = Vec::new();
        self.compacted = true;
    }

    /// Save a built IVF-RaBitQ index to a directory.
    ///
    /// The current format stores normalized raw vectors and cluster membership,
    /// then rebuilds RaBitQ edge codes on load. Saving after [`Self::compact`]
    /// is rejected because `qntz::rabitq::EdgeQuantizedVector` intentionally
    /// cannot be reconstructed from fields outside `qntz`.
    pub fn save_to_dir(&self, output_dir: impl AsRef<Path>) -> Result<(), RetrieveError> {
        if !self.built {
            return Err(RetrieveError::NotBuilt {
                component: "IVF-RaBitQ index",
                operation: "save",
            });
        }
        let expected_vector_len = checked_batch_len(self.num_vectors, self.dimension)?;
        if self.compacted || self.vectors.len() != expected_vector_len {
            return Err(RetrieveError::Unsupported {
                feature: "saving a compacted IVF-RaBitQ index",
                hint: "save before compact(), or rebuild the index",
            });
        }

        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;
        let mut params = self.params.clone();
        params.num_clusters = self.clusters.len();
        let manifest = IVFRaBitQManifest {
            version: IVFRABITQ_FORMAT_VERSION,
            dimension: self.dimension,
            padded_dim: self.padded_dim,
            num_vectors: self.num_vectors,
            params: PersistedIVFRaBitQParams::from(&params),
        };

        write_json_atomic(&output_dir.join("manifest.json"), &manifest)?;
        write_f32_atomic(&output_dir.join("raw_vectors.bin"), &self.vectors)?;
        write_u32_atomic(&output_dir.join("doc_ids.bin"), &self.doc_ids)?;
        write_f32_atomic(&output_dir.join("centroids.bin"), &self.centroids)?;
        write_cluster_ids_atomic(&output_dir.join("clusters.bin"), &self.clusters)?;

        Ok(())
    }

    /// Load an IVF-RaBitQ index saved by [`Self::save_to_dir`].
    pub fn load_from_dir(input_dir: impl AsRef<Path>) -> Result<Self, RetrieveError> {
        let input_dir = input_dir.as_ref();
        let manifest: IVFRaBitQManifest = read_json(&input_dir.join("manifest.json"))?;
        validate_manifest(&manifest)?;
        let raw_vector_len = checked_len(manifest.num_vectors, manifest.dimension, "raw vector")?;
        let centroid_len =
            checked_len(manifest.params.num_clusters, manifest.dimension, "centroid")?;

        let params = manifest.params.into_params();
        let mut index = Self::new(manifest.dimension, params)?;
        if index.padded_dim != manifest.padded_dim {
            return Err(RetrieveError::FormatError(format!(
                "IVF-RaBitQ padded dimension mismatch: index was built in {} dimensions, this \
                 build would sample a {}-dimensional rotation; recall would silently change",
                manifest.padded_dim, index.padded_dim
            )));
        }
        index.num_vectors = manifest.num_vectors;
        index.vectors = read_f32_exact(&input_dir.join("raw_vectors.bin"), raw_vector_len)?;
        index.doc_ids = read_u32_exact(&input_dir.join("doc_ids.bin"), manifest.num_vectors)?;
        index.centroids = read_f32_exact(&input_dir.join("centroids.bin"), centroid_len)?;
        let cluster_indices = read_cluster_ids(
            &input_dir.join("clusters.bin"),
            index.params.num_clusters,
            manifest.num_vectors,
        )?;
        index.rebuild_coarse_quantizer()?;
        index.rebuild_quantized_clusters(cluster_indices)?;
        index.built = true;
        Ok(index)
    }

    /// Search for the k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.search_with_ef(query, k, self.params.nprobe)
    }

    /// Search with a custom nprobe value.
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.search_with_stats(query, k, nprobe)
            .map(|(results, _)| results)
    }

    /// Search with a custom nprobe value, returning candidate accounting.
    ///
    /// Each candidate is scored with one RaBitQ dot product. A candidate whose
    /// error-bound lower bound cannot beat the heap's worst entry is skipped
    /// outright; every other candidate is reranked with exact f32 distance when
    /// raw vectors are resident. See the module docs for why the bound is
    /// probabilistic rather than a guarantee.
    pub fn search_with_stats(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> Result<(Vec<(u32, f32)>, IVFRaBitQSearchStats), RetrieveError> {
        if !self.built {
            return Err(RetrieveError::NotBuilt {
                component: "IVF-RaBitQ index",
                operation: "search",
            });
        }
        if query.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.dimension,
            });
        }
        if k == 0 {
            return Ok((Vec::new(), IVFRaBitQSearchStats::default()));
        }

        // Normalize query
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let query_normalized: Vec<f32> = if query_norm > 1e-10 {
            query.iter().map(|x| x / query_norm).collect()
        } else {
            query.to_vec()
        };
        let query = query_normalized.as_slice();

        // Find nearest centroids
        let cluster_distances = self.find_nearest_centroids(query, nprobe);

        // Rotate the query once (O(d log d)), then pay O(d) per candidate.
        let rotated_query = self.rotate(query)?;

        let mut stats = IVFRaBitQSearchStats::default();
        // Max-heap of the current best `k`, so the worst entry is O(1) to read.
        let mut heap: std::collections::BinaryHeap<(FloatOrd, u32)> =
            std::collections::BinaryHeap::with_capacity(k);

        for (cluster_idx, _coarse_dist) in &cluster_distances {
            let cluster = &self.clusters[*cluster_idx];
            if cluster.vector_indices.is_empty() {
                continue;
            }

            // `rotated_query` is R*q, which is what
            // edge_distance_term_prerotated expects. Add ||q - cluster_c||^2 to
            // compose a true absolute distance comparable across probed
            // clusters.
            //
            // `_coarse_dist` from find_nearest_centroids is cosine distance
            // (or HNSW-metric distance), not guaranteed to be L2-squared, so
            // we recompute ||q - c||^2 directly here. One O(d) op per probed
            // cluster is cheap relative to the per-vector edge loop below, and
            // its square root is the query-side error factor `g_error`.
            let cluster_centroid = self.get_centroid(*cluster_idx);
            let q_c_sqr: f32 = query
                .iter()
                .zip(cluster_centroid.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            let g_error = q_c_sqr.sqrt();
            stats.candidates_scanned += cluster.quantized.len();

            for (i, edge) in cluster.quantized.iter().enumerate() {
                let edge_term =
                    RaBitQQuantizer::edge_distance_term_prerotated(&rotated_query, edge);
                let estimate = q_c_sqr + edge_term;

                // RaBitQ error bound: the estimate is off by at most
                // f_error * ||q - c||, so `lower_bound` is the best distance
                // this candidate could possibly have. Skipping when it cannot
                // beat the heap keeps ranking exact on survivors and drops the
                // rest (FAISS RaBitQUtils.h::should_refine_candidate).
                let bound_enabled = self.params.use_error_bound;
                let margin = if bound_enabled {
                    edge.quantized.f_error * g_error
                } else {
                    0.0
                };
                let lower_bound = (estimate - margin).max(0.0);
                if bound_enabled && heap.len() >= k {
                    if let Some(&(FloatOrd(worst), _)) = heap.peek() {
                        if lower_bound >= worst {
                            stats.pruned_by_bound += 1;
                            continue;
                        }
                    }
                }

                let vec_idx = cluster.vector_indices[i];
                let distance = if self.compacted {
                    estimate
                } else {
                    stats.exact_reranks += 1;
                    crate::distance::l2_distance_squared(query, self.get_vector(vec_idx as usize))
                };

                if heap.len() < k {
                    heap.push((FloatOrd(distance), vec_idx));
                } else if let Some(&(FloatOrd(worst), _)) = heap.peek() {
                    if distance < worst {
                        heap.pop();
                        heap.push((FloatOrd(distance), vec_idx));
                    }
                }
            }
        }

        // Exact mode ranks by squared L2 over unit-norm vectors, so cosine
        // distance is exactly half of it (`cosine_distance_normalized` returns
        // `1 - dot`). Ranking is unchanged; only the reported scale differs.
        // Compact mode keeps reporting the raw squared estimate, as documented
        // on `compact()`.
        let reported_scale = if self.compacted { 1.0 } else { 0.5 };
        let mut results: Vec<(u32, f32)> = heap
            .into_iter()
            .map(|(FloatOrd(distance), vec_idx)| {
                (self.doc_ids[vec_idx as usize], distance * reported_scale)
            })
            .collect();

        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        results.truncate(k);
        // Keep signed edge corrections and raw estimates throughout ranking.
        // Quantization error (compact mode) or cancellation in 1-dot (exact
        // mode) can make a final distance negative. Bound only the reported
        // distance after selection, preserving the raw estimates' ordering.
        for (_, dist) in &mut results {
            if *dist < 0.0 {
                *dist = 0.0;
            }
        }
        Ok((results, stats))
    }

    /// Decode one indexed vector back into the original coordinate system.
    ///
    /// Reconstructs the quantized residual in the rotated working space, adds
    /// the cluster centroid, un-rotates with the index's FHT/Kac transform, and
    /// truncates the padding. Works after [`Self::compact`] because only codes
    /// and centroids are needed.
    ///
    /// The result approximates the *normalized* vector that
    /// [`Self::add_slice`] stored, up to quantization error.
    pub fn dequantize(&self, vector_idx: usize) -> Result<Vec<f32>, RetrieveError> {
        if vector_idx >= self.num_vectors {
            return Err(RetrieveError::InvalidParameter(format!(
                "vector index {vector_idx} out of range (len {})",
                self.num_vectors
            )));
        }
        let (cluster_idx, slot) = self.locate(vector_idx)?;
        let code = &self.clusters[cluster_idx].quantized[slot].quantized;

        let mut rotated = vec![0.0f32; self.padded_dim];
        for (index, value) in rotated.iter_mut().enumerate() {
            // x_hat = delta * code + vl, the inverse of qntz's packed encoding.
            *value = code.delta * f32::from(code.codes[index]) + code.vl;
        }
        let centroid = self.get_centroid(cluster_idx).to_vec();
        let centroid_rot = self.rotate(&centroid)?;
        for (value, offset) in rotated.iter_mut().zip(centroid_rot.iter()) {
            *value += offset;
        }

        let restored = self.unrotate(&rotated)?;
        Ok(restored[..self.dimension].to_vec())
    }

    /// Locate a vector's cluster and slot. Cluster membership lists are built
    /// in ascending vector order, so this is a binary search per cluster.
    fn locate(&self, vector_idx: usize) -> Result<(usize, usize), RetrieveError> {
        let target = u32::try_from(vector_idx)
            .map_err(|_| RetrieveError::InvalidParameter("vector index does not fit u32".into()))?;
        for (cluster_idx, cluster) in self.clusters.iter().enumerate() {
            if let Ok(slot) = cluster.vector_indices.binary_search(&target) {
                return Ok((cluster_idx, slot));
            }
        }
        Err(RetrieveError::InvalidParameter(format!(
            "vector index {vector_idx} is not assigned to any cluster"
        )))
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.num_vectors
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    /// Memory usage breakdown for this index.
    pub fn memory_usage(&self) -> crate::memory::MemoryReport {
        let vectors_bytes = self.vectors.len() * std::mem::size_of::<f32>();

        let quantized_bytes: usize = self
            .clusters
            .iter()
            .flat_map(|c| &c.quantized)
            .map(|edge| {
                edge.quantized.binary_codes.len()
                    + edge.quantized.extended_codes.len()
                    + edge.quantized.codes.len() * std::mem::size_of::<u16>()
                    + std::mem::size_of::<f32>() // ip_parent_rot_codes per edge
            })
            .sum();

        let metadata_bytes = self.doc_ids.len() * std::mem::size_of::<u32>()
            + self.centroids.len() * std::mem::size_of::<f32>();

        crate::memory::MemoryReport {
            vectors_bytes,
            graph_bytes: 0,
            quantized_bytes,
            metadata_bytes,
        }
    }

    /// Get vector from flat storage.
    #[inline]
    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        &self.vectors[start..start + self.dimension]
    }

    /// Get centroid from flat storage.
    #[inline]
    fn get_centroid(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        &self.centroids[start..start + self.dimension]
    }

    /// Find nearest centroids. Uses HNSW coarse quantizer when available.
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

fn validate_manifest(manifest: &IVFRaBitQManifest) -> Result<(), RetrieveError> {
    if manifest.version != IVFRABITQ_FORMAT_VERSION {
        return Err(RetrieveError::FormatError(format!(
            "unsupported IVF-RaBitQ format version {}",
            manifest.version
        )));
    }
    if manifest.dimension == 0 {
        return Err(RetrieveError::FormatError(
            "IVF-RaBitQ manifest has zero dimension".into(),
        ));
    }
    if manifest.num_vectors == 0 {
        return Err(RetrieveError::FormatError(
            "IVF-RaBitQ manifest has zero vectors".into(),
        ));
    }
    if manifest.params.num_clusters == 0 {
        return Err(RetrieveError::FormatError(
            "IVF-RaBitQ manifest has zero clusters".into(),
        ));
    }
    if !(1..=8).contains(&manifest.params.total_bits) {
        return Err(RetrieveError::FormatError(format!(
            "IVF-RaBitQ manifest has invalid total_bits {}",
            manifest.params.total_bits
        )));
    }
    checked_len(manifest.num_vectors, manifest.dimension, "raw vector")?;
    checked_len(manifest.params.num_clusters, manifest.dimension, "centroid")?;
    Ok(())
}

fn checked_batch_len(vector_count: usize, dimension: usize) -> Result<usize, RetrieveError> {
    vector_count
        .checked_mul(dimension)
        .ok_or_else(|| RetrieveError::InvalidParameter("vector count overflows usize".into()))
}

fn checked_len(count: usize, dimension: usize, label: &str) -> Result<usize, RetrieveError> {
    count.checked_mul(dimension).ok_or_else(|| {
        RetrieveError::FormatError(format!("IVF-RaBitQ {label} length overflows usize"))
    })
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

fn write_cluster_ids_atomic(path: &Path, clusters: &[Cluster]) -> Result<(), RetrieveError> {
    write_atomic(path, |writer| {
        writer.write_all(IVFRABITQ_CLUSTERS_MAGIC)?;
        writer.write_all(&(clusters.len() as u64).to_le_bytes())?;
        for cluster in clusters {
            writer.write_all(&(cluster.vector_indices.len() as u64).to_le_bytes())?;
            for id in &cluster.vector_indices {
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
    let (tmp_path, file) = create_temp_file(path)?;
    let result = (|| {
        {
            let mut writer = BufWriter::new(file);
            write(&mut writer)?;
            writer.flush()?;
            writer.get_ref().sync_all()?;
        }
        std::fs::rename(&tmp_path, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
    }
    result.map_err(Into::into)
}

fn create_temp_file(path: &Path) -> std::io::Result<(std::path::PathBuf, std::fs::File)> {
    for _ in 0..100 {
        let sequence = IVFRABITQ_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let tmp_path = path.with_extension(format!("tmp-{}-{sequence}", std::process::id()));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)
        {
            Ok(file) => return Ok((tmp_path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "unable to create a unique snapshot temporary file",
    ))
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
    for chunk in bytes.as_chunks::<4>().0 {
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
    for chunk in bytes.as_chunks::<4>().0 {
        values.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

fn read_cluster_ids(
    path: &Path,
    expected_clusters: usize,
    num_vectors: usize,
) -> Result<Vec<Vec<u32>>, RetrieveError> {
    let mut reader = BufReader::new(std::fs::File::open(path)?);
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != IVFRABITQ_CLUSTERS_MAGIC {
        return Err(RetrieveError::FormatError(
            "invalid IVF-RaBitQ cluster file magic".into(),
        ));
    }
    let cluster_count = usize_from_u64(read_u64(&mut reader)?, "cluster count")?;
    if cluster_count != expected_clusters {
        return Err(RetrieveError::FormatError(format!(
            "cluster count mismatch: expected {}, got {}",
            expected_clusters, cluster_count
        )));
    }

    let mut seen = vec![false; num_vectors];
    let mut clusters = Vec::with_capacity(cluster_count);
    for _ in 0..cluster_count {
        let len = usize_from_u64(read_u64(&mut reader)?, "cluster length")?;
        if len > num_vectors {
            return Err(RetrieveError::FormatError(format!(
                "cluster length {} exceeds vector count {}",
                len, num_vectors
            )));
        }
        let mut ids = Vec::with_capacity(len);
        for _ in 0..len {
            let id = read_u32(&mut reader)?;
            let idx = usize::try_from(id).map_err(|_| {
                RetrieveError::FormatError(format!("cluster id {id} cannot fit usize"))
            })?;
            if idx >= num_vectors {
                return Err(RetrieveError::FormatError(format!(
                    "cluster id {} exceeds vector count {}",
                    id, num_vectors
                )));
            }
            if seen[idx] {
                return Err(RetrieveError::FormatError(format!(
                    "duplicate cluster id {}",
                    id
                )));
            }
            seen[idx] = true;
            ids.push(id);
        }
        clusters.push(ids);
    }

    if seen.iter().any(|present| !present) {
        return Err(RetrieveError::FormatError(
            "cluster memberships do not cover every vector".into(),
        ));
    }

    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        return Err(RetrieveError::FormatError(
            "trailing bytes in IVF-RaBitQ cluster file".into(),
        ));
    }

    Ok(clusters)
}

fn usize_from_u64(value: u64, label: &str) -> Result<usize, RetrieveError> {
    usize::try_from(value)
        .map_err(|_| RetrieveError::FormatError(format!("IVF-RaBitQ {label} cannot fit usize")))
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
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn make_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        // Simple deterministic pseudo-random vectors
        let mut rng = seed;
        (0..n * dim)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect()
    }

    fn write_manifest(dir: &Path, manifest: &IVFRaBitQManifest) {
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec_pretty(manifest).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn build_and_search_basic() {
        let dim = 32;
        let n = 200;
        let data = make_vectors(n, dim, 42);
        let doc_ids: Vec<u32> = (0..n as u32).collect();

        let params = IVFRaBitQParams {
            num_clusters: 8,
            nprobe: 4,
            total_bits: 4,
            seed: 42,
            ..IVFRaBitQParams::default()
        };
        let mut index = IVFRaBitQIndex::new(dim, params).unwrap();
        index.add_batch(&doc_ids, &data).unwrap();
        index.build().unwrap();

        let query = &data[0..dim]; // search for first vector
        let results = index.search(query, 5).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
        // First vector should be near the top (self-match after normalization)
        assert!(
            results.iter().any(|(id, _)| *id == 0),
            "expected doc_id 0 in results: {:?}",
            results
        );
    }

    #[test]
    fn search_zero_k_returns_empty() {
        let dim = 32;
        let n = 80;
        let data = make_vectors(n, dim, 44);
        let doc_ids: Vec<u32> = (0..n as u32).collect();

        let params = IVFRaBitQParams {
            num_clusters: 4,
            nprobe: 4,
            total_bits: 4,
            seed: 44,
            ..IVFRaBitQParams::default()
        };
        let mut index = IVFRaBitQIndex::new(dim, params).unwrap();
        index.add_batch(&doc_ids, &data).unwrap();
        index.build().unwrap();

        assert!(index.search(&data[0..dim], 0).unwrap().is_empty());
    }

    #[test]
    fn empty_index_returns_error() {
        let params = IVFRaBitQParams::default();
        let mut index = IVFRaBitQIndex::new(32, params).unwrap();
        assert!(index.build().is_err());
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let params = IVFRaBitQParams::default();
        let mut index = IVFRaBitQIndex::new(32, params).unwrap();
        assert!(index.add(0, vec![1.0; 64]).is_err());
    }

    #[test]
    fn binary_quantization_works() {
        let dim = 64;
        let n = 100;
        let data = make_vectors(n, dim, 99);
        let doc_ids: Vec<u32> = (0..n as u32).collect();

        let params = IVFRaBitQParams {
            num_clusters: 4,
            nprobe: 4,
            total_bits: 1, // binary only
            seed: 42,
            ..IVFRaBitQParams::default()
        };
        let mut index = IVFRaBitQIndex::new(dim, params).unwrap();
        index.add_batch(&doc_ids, &data).unwrap();
        index.build().unwrap();

        let results = index.search(&data[0..dim], 3).unwrap();
        assert!(!results.is_empty());
    }

    /// compact() drops raw vectors; search still returns results using quantized distances.
    #[test]
    fn compact_search_works() {
        let dim = 32;
        let n = 200;
        let data = make_vectors(n, dim, 42);
        let doc_ids: Vec<u32> = (0..n as u32).collect();

        let params = IVFRaBitQParams {
            num_clusters: 8,
            nprobe: 4,
            total_bits: 4,
            seed: 42,
            ..IVFRaBitQParams::default()
        };
        let mut index = IVFRaBitQIndex::new(dim, params).unwrap();
        index.add_batch(&doc_ids, &data).unwrap();
        index.build().unwrap();
        for compact in [false, true] {
            if compact {
                index.compact();
            }
            // Self queries exercise cancellation near zero in both exact
            // cosine reranking and the approximate squared-distance output.
            for query in data.chunks_exact(dim) {
                let results = index.search(query, 5).unwrap();
                assert_eq!(results.len(), 5);
                assert!(results.windows(2).all(|w| w[0].1 <= w[1].1));
                assert!(results.iter().any(|(_, dist)| *dist > 0.0));
                for &(id, dist) in &results {
                    assert!((id as usize) < n, "doc_id {id} out of range");
                    assert!(
                        dist.is_finite() && dist >= 0.0,
                        "invalid distance {dist}, compact={compact}"
                    );
                }
            }
        }
    }

    /// Self-search recall: searching for each vector should return itself.
    #[test]
    fn self_search_recall() {
        let dim = 32;
        let n = 100;
        let data = make_vectors(n, dim, 7);
        let doc_ids: Vec<u32> = (0..n as u32).collect();

        let params = IVFRaBitQParams {
            num_clusters: 4,
            nprobe: 4, // probe all clusters
            total_bits: 4,
            seed: 42,
            ..IVFRaBitQParams::default()
        };
        let mut index = IVFRaBitQIndex::new(dim, params).unwrap();
        index.add_batch(&doc_ids, &data).unwrap();
        index.build().unwrap();

        let mut hits = 0;
        for i in 0..n {
            let query = &data[i * dim..(i + 1) * dim];
            let results = index.search(query, 1).unwrap();
            if results.first().map(|(id, _)| *id) == Some(i as u32) {
                hits += 1;
            }
        }
        let recall = hits as f64 / n as f64;
        assert!(
            recall > 0.7,
            "self-search recall too low: {recall:.2} ({hits}/{n})"
        );
    }

    #[test]
    fn save_load_roundtrip_preserves_search() {
        let dim = 32;
        let n = 160;
        let data = make_vectors(n, dim, 101);
        let doc_ids: Vec<u32> = (0..n as u32).map(|i| 50_000 + i).collect();

        let params = IVFRaBitQParams {
            num_clusters: 8,
            nprobe: 4,
            total_bits: 4,
            seed: 101,
            ..IVFRaBitQParams::default()
        };
        let mut index = IVFRaBitQIndex::new(dim, params).unwrap();
        index.add_batch(&doc_ids, &data).unwrap();
        index.build().unwrap();

        let query = &data[3 * dim..4 * dim];
        let before = index.search(query, 10).unwrap();

        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let loaded = IVFRaBitQIndex::load_from_dir(dir.path()).unwrap();

        assert_eq!(loaded.search(query, 10).unwrap(), before);
    }

    #[test]
    fn atomic_writes_use_unique_temps_and_clean_failures() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("snapshot.bin");
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let first_path = path.clone();
        let first_barrier = barrier.clone();
        let first = std::thread::spawn(move || {
            first_barrier.wait();
            write_atomic(&first_path, |writer| writer.write_all(b"first"))
        });
        let second_path = path.clone();
        let second_barrier = barrier;
        let second = std::thread::spawn(move || {
            second_barrier.wait();
            write_atomic(&second_path, |writer| writer.write_all(b"second"))
        });
        assert!(first.join().unwrap().is_ok());
        assert!(second.join().unwrap().is_ok());
        assert!(matches!(
            std::fs::read(&path).unwrap().as_slice(),
            b"first" | b"second"
        ));

        let failed_path = dir.path().join("failed.bin");
        assert!(write_atomic(&failed_path, |_writer| {
            Err(std::io::Error::other("intentional test failure"))
        })
        .is_err());
        assert!(!dir
            .path()
            .read_dir()
            .unwrap()
            .filter_map(Result::ok)
            .any(|entry| entry
                .file_name()
                .to_string_lossy()
                .starts_with("failed.tmp-")));
    }

    #[test]
    fn load_rejects_truncated_raw_vectors() {
        let dim = 32;
        let n = 256;
        let data = make_vectors(n, dim, 105);
        let doc_ids: Vec<u32> = (0..n as u32).collect();

        let params = IVFRaBitQParams {
            num_clusters: 256,
            nprobe: 4,
            total_bits: 4,
            seed: 105,
            ..IVFRaBitQParams::default()
        };
        let mut index = IVFRaBitQIndex::new(dim, params).unwrap();
        index.add_batch(&doc_ids, &data).unwrap();
        index.build().unwrap();

        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let raw_vectors_path = dir.path().join("raw_vectors.bin");
        let raw_vectors = std::fs::read(&raw_vectors_path).unwrap();
        std::fs::write(&raw_vectors_path, &raw_vectors[..raw_vectors.len() - 1]).unwrap();

        let err = match IVFRaBitQIndex::load_from_dir(dir.path()) {
            Ok(_) => panic!("truncated raw vectors should fail"),
            Err(err) => err,
        };
        assert!(
            matches!(err, RetrieveError::FormatError(_)),
            "unexpected error: {err}"
        );
        assert!(
            err.to_string().contains("raw_vectors.bin size mismatch"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn save_rejects_compacted_index() {
        let dim = 32;
        let n = 100;
        let data = make_vectors(n, dim, 102);
        let doc_ids: Vec<u32> = (0..n as u32).collect();

        let params = IVFRaBitQParams {
            num_clusters: 4,
            nprobe: 4,
            total_bits: 4,
            seed: 102,
            ..IVFRaBitQParams::default()
        };
        let mut index = IVFRaBitQIndex::new(dim, params).unwrap();
        index.add_batch(&doc_ids, &data).unwrap();
        index.build().unwrap();
        index.compact();

        let dir = tempfile::tempdir().unwrap();
        let err = index.save_to_dir(dir.path()).unwrap_err();
        assert!(
            err.to_string().contains("compacted IVF-RaBitQ"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_rejects_corrupt_cluster_magic() {
        let dim = 32;
        let n = 100;
        let data = make_vectors(n, dim, 103);
        let doc_ids: Vec<u32> = (0..n as u32).collect();

        let params = IVFRaBitQParams {
            num_clusters: 4,
            nprobe: 4,
            total_bits: 4,
            seed: 103,
            ..IVFRaBitQParams::default()
        };
        let mut index = IVFRaBitQIndex::new(dim, params).unwrap();
        index.add_batch(&doc_ids, &data).unwrap();
        index.build().unwrap();

        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        std::fs::write(dir.path().join("clusters.bin"), b"not-rbq!").unwrap();

        let err = match IVFRaBitQIndex::load_from_dir(dir.path()) {
            Ok(_) => panic!("corrupt cluster magic should fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains("invalid IVF-RaBitQ cluster file magic"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_rejects_future_manifest_version() {
        let dim = 32;
        let n = 100;
        let data = make_vectors(n, dim, 104);
        let doc_ids: Vec<u32> = (0..n as u32).collect();

        let params = IVFRaBitQParams {
            num_clusters: 4,
            nprobe: 4,
            total_bits: 4,
            seed: 104,
            ..IVFRaBitQParams::default()
        };
        let mut index = IVFRaBitQIndex::new(dim, params).unwrap();
        index.add_batch(&doc_ids, &data).unwrap();
        index.build().unwrap();

        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let manifest_path = dir.path().join("manifest.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        manifest["version"] = serde_json::json!(IVFRABITQ_FORMAT_VERSION + 1);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let err = match IVFRaBitQIndex::load_from_dir(dir.path()) {
            Ok(_) => panic!("future manifest version should fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains("unsupported IVF-RaBitQ format version"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_rejects_zero_clusters_before_file_reads() {
        let dir = tempfile::tempdir().unwrap();
        write_manifest(
            dir.path(),
            &IVFRaBitQManifest {
                padded_dim: 64,
                version: IVFRABITQ_FORMAT_VERSION,
                dimension: 32,
                num_vectors: 10,
                params: PersistedIVFRaBitQParams {
                    rotation: RotationKind::FhtKac,
                    use_error_bound: true,
                    num_clusters: 0,
                    nprobe: 1,
                    total_bits: 4,
                    seed: 42,
                },
            },
        );

        let err = match IVFRaBitQIndex::load_from_dir(dir.path()) {
            Ok(_) => panic!("zero-cluster manifest should fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("zero clusters"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_rejects_invalid_total_bits_before_file_reads() {
        let dir = tempfile::tempdir().unwrap();
        write_manifest(
            dir.path(),
            &IVFRaBitQManifest {
                padded_dim: 64,
                version: IVFRABITQ_FORMAT_VERSION,
                dimension: 32,
                num_vectors: 10,
                params: PersistedIVFRaBitQParams {
                    rotation: RotationKind::FhtKac,
                    use_error_bound: true,
                    num_clusters: 4,
                    nprobe: 1,
                    total_bits: 0,
                    seed: 42,
                },
            },
        );

        let err = match IVFRaBitQIndex::load_from_dir(dir.path()) {
            Ok(_) => panic!("invalid-total-bits manifest should fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("invalid total_bits"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_rejects_raw_vector_length_overflow_before_file_reads() {
        let dir = tempfile::tempdir().unwrap();
        write_manifest(
            dir.path(),
            &IVFRaBitQManifest {
                padded_dim: 64,
                version: IVFRABITQ_FORMAT_VERSION,
                dimension: 2,
                num_vectors: usize::MAX,
                params: PersistedIVFRaBitQParams {
                    rotation: RotationKind::FhtKac,
                    use_error_bound: true,
                    num_clusters: 1,
                    nprobe: 1,
                    total_bits: 4,
                    seed: 42,
                },
            },
        );

        let err = match IVFRaBitQIndex::load_from_dir(dir.path()) {
            Ok(_) => panic!("overflowing raw-vector length should fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("raw vector length overflows"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_rejects_centroid_length_overflow_before_file_reads() {
        let dir = tempfile::tempdir().unwrap();
        write_manifest(
            dir.path(),
            &IVFRaBitQManifest {
                padded_dim: 64,
                version: IVFRABITQ_FORMAT_VERSION,
                dimension: 2,
                num_vectors: 1,
                params: PersistedIVFRaBitQParams {
                    rotation: RotationKind::FhtKac,
                    use_error_bound: true,
                    num_clusters: usize::MAX,
                    nprobe: 1,
                    total_bits: 4,
                    seed: 42,
                },
            },
        );

        let err = match IVFRaBitQIndex::load_from_dir(dir.path()) {
            Ok(_) => panic!("overflowing centroid length should fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("centroid length overflows"),
            "unexpected error: {err}"
        );
    }

    // ─── Rotation / bound regression tests ────────────────────────────────

    const SEEDED_DIM: usize = 96;
    const SEEDED_N: usize = 10_000;
    const SEEDED_CLUSTERS: usize = 64;
    const SEEDED_QUERIES: usize = 50;

    /// SplitMix64 fixture: same generator as the ablation example.
    struct TestRng(u64);

    impl TestRng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn uniform(&mut self) -> f32 {
            ((self.next_u64() >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        }
    }

    fn unit(mut v: Vec<f32>) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    /// Per-dimension scale decays geometrically, so a few coordinates dominate
    /// the norm. This is where a rotation earns its keep: without one, the
    /// quiet coordinates carry no sign information.
    fn anisotropic_set(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = TestRng::new(seed);
        let scales: Vec<f32> = (0..SEEDED_DIM)
            .map(|i| 1_000f32.powf(-(i as f32) / (SEEDED_DIM - 1) as f32))
            .collect();
        let mut flat = Vec::with_capacity(n * SEEDED_DIM);
        for _ in 0..n {
            let v: Vec<f32> = scales.iter().map(|s| s * rng.uniform()).collect();
            flat.extend(unit(v));
        }
        flat
    }

    fn clustered_set(n: usize, seed: u64, spread: f32) -> Vec<f32> {
        let mut rng = TestRng::new(seed);
        let centers: Vec<Vec<f32>> = (0..SEEDED_CLUSTERS)
            .map(|_| (0..SEEDED_DIM).map(|_| rng.uniform()).collect())
            .collect();
        let mut flat = Vec::with_capacity(n * SEEDED_DIM);
        for i in 0..n {
            let mut v = centers[i % SEEDED_CLUSTERS].clone();
            for value in &mut v {
                *value += spread * rng.uniform();
            }
            flat.extend(unit(v));
        }
        flat
    }

    fn build_seeded(data: &[f32], n: usize, rotation: RotationKind, seed: u64) -> IVFRaBitQIndex {
        let params = IVFRaBitQParams {
            num_clusters: SEEDED_CLUSTERS,
            nprobe: 8,
            total_bits: 4,
            seed,
            rotation,
            use_error_bound: true,
        };
        let mut index = IVFRaBitQIndex::new(SEEDED_DIM, params).unwrap();
        let ids: Vec<u32> = (0..n as u32).collect();
        index.add_batch(&ids, data).unwrap();
        index.build().unwrap();
        index
    }

    fn brute_force_top_k(data: &[f32], n: usize, query: &[f32], k: usize) -> Vec<u32> {
        let mut scored: Vec<(f32, u32)> = (0..n)
            .map(|i| {
                let v = &data[i * SEEDED_DIM..(i + 1) * SEEDED_DIM];
                let dot: f32 = query.iter().zip(v).map(|(a, b)| a * b).sum();
                (1.0 - dot, i as u32)
            })
            .collect();
        scored.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
        scored.truncate(k);
        scored.into_iter().map(|(_, id)| id).collect()
    }

    fn recall_at_k(
        index: &IVFRaBitQIndex,
        queries: &[f32],
        num_queries: usize,
        data: &[f32],
        n: usize,
        k: usize,
        nprobe: usize,
    ) -> f64 {
        let mut hits = 0usize;
        for q in 0..num_queries {
            let query = &queries[q * SEEDED_DIM..(q + 1) * SEEDED_DIM];
            let expected = brute_force_top_k(data, n, query, k);
            let got: Vec<u32> = index
                .search_with_ef(query, k, nprobe)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            hits += expected.iter().filter(|id| got.contains(id)).count();
        }
        hits as f64 / (num_queries * k) as f64
    }

    /// End-to-end recall floor on the seeded set.
    ///
    /// nprobe=32 of 64 lists, so the floor is dominated by quantization and
    /// pruning rather than by how many lists were probed. Measured: see the
    /// floor value below.
    #[test]
    fn recall_floor_on_seeded_set() {
        let data = anisotropic_set(SEEDED_N, 11);
        let queries = anisotropic_set(SEEDED_QUERIES, 12);

        let index = build_seeded(&data, SEEDED_N, RotationKind::FhtKac, 42);
        let recall = recall_at_k(&index, &queries, SEEDED_QUERIES, &data, SEEDED_N, 10, 32);
        assert!(
            recall >= 0.95,
            "recall@10 {recall:.3} below floor 0.95 on the seeded anisotropic set"
        );
    }

    /// Rotation is not cosmetic: on anisotropic data the estimate-only ranking
    /// loses a large amount of recall without it, which is the defect this
    /// module is meant to avoid.
    ///
    /// Compacted index (no exact rerank) and every list probed, so only the
    /// quantized estimate can move the number.
    #[test]
    fn rotation_lifts_estimate_only_recall() {
        let data = anisotropic_set(SEEDED_N, 11);
        let queries = anisotropic_set(20, 12);

        let mut rotated = build_seeded(&data, SEEDED_N, RotationKind::FhtKac, 42);
        rotated.compact();
        let mut plain = build_seeded(&data, SEEDED_N, RotationKind::Identity, 42);
        plain.compact();

        let rotated_recall =
            recall_at_k(&rotated, &queries, 20, &data, SEEDED_N, 10, SEEDED_CLUSTERS);
        let plain_recall = recall_at_k(&plain, &queries, 20, &data, SEEDED_N, 10, SEEDED_CLUSTERS);
        assert!(
            rotated_recall >= plain_recall + 0.05,
            "rotation did not help: FHT/Kac {rotated_recall:.3} vs none {plain_recall:.3}"
        );
    }

    /// The same seed reproduces the index bit for bit; a different seed
    /// samples a different transform, visible in decoded codes (end-to-end
    /// results can coincide because exact reranking masks code differences).
    #[test]
    fn seeded_build_is_reproducible_and_seed_sensitive() {
        let data = clustered_set(2_000, 21, 0.25);
        let query = data[..SEEDED_DIM].to_vec();

        let first = build_seeded(&data, 2_000, RotationKind::FhtKac, 7);
        let repeat = build_seeded(&data, 2_000, RotationKind::FhtKac, 7);
        let other = build_seeded(&data, 2_000, RotationKind::FhtKac, 8);

        assert_eq!(
            first.search(&query, 10).unwrap(),
            repeat.search(&query, 10).unwrap()
        );
        assert_eq!(first.dequantize(3).unwrap(), repeat.dequantize(3).unwrap());
        assert_ne!(first.dequantize(3).unwrap(), other.dequantize(3).unwrap());
    }

    /// Identity rotation is the ablation path: no padding, and it is recorded
    /// in the snapshot so a reload cannot silently swap the transform.
    #[test]
    fn identity_rotation_keeps_the_native_dimension() {
        let data = anisotropic_set(256, 31);
        let index = build_seeded(&data, 256, RotationKind::Identity, 42);
        assert_eq!(index.padded_dimension(), SEEDED_DIM);

        let rotated = build_seeded(&data, 256, RotationKind::FhtKac, 42);
        assert_eq!(rotated.padded_dimension(), SEEDED_DIM);
    }

    /// `dequantize` un-rotates: the decoded vector points the same way as the
    /// stored one, and decoding works without the raw vectors.
    #[test]
    fn dequantize_recovers_the_stored_direction() {
        let data = clustered_set(512, 41, 0.25);
        let mut index = build_seeded(&data, 512, RotationKind::FhtKac, 42);

        let mut min_cosine = 1.0f32;
        for id in [0usize, 7, 100, 511] {
            let original = &data[id * SEEDED_DIM..(id + 1) * SEEDED_DIM];
            let decoded = index.dequantize(id).unwrap();
            assert_eq!(decoded.len(), SEEDED_DIM);
            let dot: f32 = decoded.iter().zip(original).map(|(a, b)| a * b).sum();
            min_cosine = min_cosine.min(dot);
        }
        assert!(
            min_cosine > 0.9,
            "dequantized direction drifted: min cosine {min_cosine:.3}"
        );

        let before = index.dequantize(7).unwrap();
        index.compact();
        assert_eq!(index.dequantize(7).unwrap(), before);

        assert!(index.dequantize(512).is_err());
    }

    /// The 1.9-epsilon margin must actually bound the estimator on real codes.
    ///
    /// This is the evidence behind the pruning: if the margin were zero (a
    /// common failure when the ratio term collapses) the search would be
    /// pruning on a bare estimate. Measured coverage on this fixture is >0.95.
    #[test]
    fn error_bound_covers_observed_estimator_error() {
        let data = clustered_set(2_000, 51, 0.25);
        let index = build_seeded(&data, 2_000, RotationKind::FhtKac, 42);
        let queries = clustered_set(SEEDED_QUERIES, 52, 0.25);

        let mut covered = 0usize;
        let mut checked = 0usize;
        let mut saw_positive_margin = false;

        for q in 0..SEEDED_QUERIES {
            let query = &queries[q * SEEDED_DIM..(q + 1) * SEEDED_DIM];
            let rotated_query = index.rotate(query).unwrap();
            for cluster_idx in 0..SEEDED_CLUSTERS {
                let cluster = &index.clusters[cluster_idx];
                let centroid = index.get_centroid(cluster_idx);
                let q_c_sqr: f32 = query
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                let g_error = q_c_sqr.sqrt();

                for (slot, edge) in cluster.quantized.iter().enumerate() {
                    let estimate = q_c_sqr
                        + RaBitQQuantizer::edge_distance_term_prerotated(&rotated_query, edge);
                    let vec_idx = cluster.vector_indices[slot] as usize;
                    let vector = index.get_vector(vec_idx);
                    let exact: f32 = query
                        .iter()
                        .zip(vector.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    let margin = edge.quantized.f_error * g_error;
                    saw_positive_margin |= margin > 0.0;
                    if (estimate - exact).abs() <= margin {
                        covered += 1;
                    }
                    checked += 1;
                }
            }
        }

        assert!(saw_positive_margin, "error margin collapsed to zero");
        let coverage = covered as f64 / checked as f64;
        assert!(
            coverage >= 0.90,
            "1.9-epsilon margin covered only {coverage:.3} of estimator errors"
        );
    }

    /// The bound must not cost recall: pruning and exhaustive reranking of the
    /// same candidate set agree on the seeded data.
    #[test]
    fn error_bound_pruning_preserves_recall() {
        let data = clustered_set(2_000, 61, 0.25);
        let queries = clustered_set(20, 62, 0.25);

        let with_bound = build_seeded(&data, 2_000, RotationKind::FhtKac, 42);
        let mut without_bound = with_bound.params.clone();
        without_bound.use_error_bound = false;
        // Rebuild the same index with the bound disabled.
        let mut no_bound = IVFRaBitQIndex::new(SEEDED_DIM, without_bound).unwrap();
        let ids: Vec<u32> = (0..2_000u32).collect();
        no_bound.add_batch(&ids, &data).unwrap();
        no_bound.build().unwrap();

        let mut pruned_total = 0usize;
        for q in 0..20 {
            let query = &queries[q * SEEDED_DIM..(q + 1) * SEEDED_DIM];
            let (bounded, stats) = with_bound.search_with_stats(query, 10, 8).unwrap();
            let unbounded = no_bound.search_with_ef(query, 10, 8).unwrap();
            assert_eq!(bounded, unbounded, "bound changed results at query {q}");
            pruned_total += stats.pruned_by_bound;
            assert_eq!(
                stats.candidates_scanned,
                stats.pruned_by_bound + stats.exact_reranks,
                "candidate accounting does not add up"
            );
        }
        assert!(pruned_total > 0, "bound never pruned anything");
    }
}
