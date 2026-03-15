//! SCANN search implementation.

use crate::scann::partitioning::KMeans;
use crate::scann::quantization::AnisotropicQuantizer;
use crate::scann::reranking;
use crate::RetrieveError;

/// Anisotropic Vector Quantization with k-means Partitioning index.
#[derive(Debug)]
pub struct SCANNIndex {
    /// Full vectors (for re-ranking)
    pub(crate) vectors: Vec<f32>,
    pub(crate) dimension: usize,
    pub(crate) num_vectors: usize,
    params: SCANNParams,
    built: bool,

    // Partitioning
    partitions: Vec<Partition>,
    pub(crate) partition_centroids: Vec<Vec<f32>>,

    // Quantization
    quantizer: Option<AnisotropicQuantizer>,
}

/// Parameters for ScaNN index construction and search.
#[derive(Clone, Debug)]
pub struct SCANNParams {
    /// Number of k-means partitions for coarse quantization.
    pub num_partitions: usize,
    /// Number of candidates to re-rank with exact distances.
    pub num_reorder: usize,
    /// Number of PQ subspaces (M).
    pub num_codebooks: usize,
    /// Number of centroids per codebook (typically 256 for 8-bit codes).
    pub codebook_size: usize,
    /// Random seed for deterministic training (k-means + PQ codebooks).
    pub seed: u64,
}

impl Default for SCANNParams {
    fn default() -> Self {
        Self {
            num_partitions: 256,
            num_reorder: 100,
            num_codebooks: 16,
            codebook_size: 256,
            seed: 42,
        }
    }
}

/// Partition containing quantized codes and indices.
#[derive(Clone, Debug)]
struct Partition {
    /// Original indices of vectors in this partition
    vector_indices: Vec<u32>,
    /// Quantized codes for these vectors (flat layout)
    /// Layout: [vector_0_codes, vector_1_codes, ...]
    codes: Vec<u8>,
}

impl SCANNIndex {
    /// Create a new ScaNN index with the given vector dimension and parameters.
    pub fn new(dimension: usize, params: SCANNParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be > 0".into(),
            ));
        }
        Ok(Self {
            vectors: Vec::new(),
            dimension,
            num_vectors: 0,
            params,
            built: false,
            partitions: Vec::new(),
            partition_centroids: Vec::new(),
            quantizer: None,
        })
    }

    /// Add a vector to the index.
    pub fn add(&mut self, _doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        self.add_slice(_doc_id, &vector)
    }

    /// Add a vector to the index from a borrowed slice.
    ///
    /// Notes:
    /// - The index stores vectors internally, so it must copy the slice into its own storage.
    /// - ScaNN currently ignores `doc_id` and uses insertion order as the internal ID.
    pub fn add_slice(&mut self, _doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
        if self.built {
            return Err(RetrieveError::InvalidParameter(
                "index already built".into(),
            ));
        }
        if vector.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: self.dimension,
                doc_dim: vector.len(),
            });
        }
        self.vectors.extend_from_slice(vector);
        self.num_vectors += 1;
        Ok(())
    }

    /// Build the index (partitioning + quantization). Required before search.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        // 1. Train Partitioning (Coarse Quantizer)
        let mut kmeans =
            KMeans::new(self.dimension, self.params.num_partitions)?.with_seed(self.params.seed);
        kmeans.fit(&self.vectors, self.num_vectors)?;
        self.partition_centroids = kmeans.centroids().to_vec();

        // 2. Assign vectors to partitions and compute residuals
        let assignments = kmeans.assign_clusters(&self.vectors, self.num_vectors);
        let mut residuals = Vec::with_capacity(self.vectors.len());

        // Initialize partitions
        self.partitions = vec![
            Partition {
                vector_indices: Vec::new(),
                codes: Vec::new()
            };
            self.params.num_partitions
        ];

        for (i, &partition_idx) in assignments.iter().enumerate() {
            self.partitions[partition_idx].vector_indices.push(i as u32);

            let vec = self.get_vector(i);
            let centroid = &self.partition_centroids[partition_idx];

            // Compute residual: r = x - c
            for (x, c) in vec.iter().zip(centroid.iter()) {
                residuals.push(x - c);
            }
        }

        // 3. Train Quantizer on Residuals
        let mut quantizer = AnisotropicQuantizer::new(
            self.dimension,
            self.params.num_codebooks,
            self.params.codebook_size,
            self.params.seed,
        )?;
        quantizer.fit_residuals(&residuals, self.num_vectors)?;

        // 4. Quantize Residuals and Store
        // We re-compute residuals on the fly to keep code simple (or use the flat residuals vector)
        // But the flat residuals vector is ordered by input ID, not partition.
        // Let's iterate partitions to populate codes.

        for p_idx in 0..self.params.num_partitions {
            let centroid = &self.partition_centroids[p_idx];
            // Clone vector indices to avoid borrow conflict with self.get_vector()
            let vec_indices: Vec<u32> = self.partitions[p_idx].vector_indices.clone();

            let mut all_codes = Vec::with_capacity(vec_indices.len() * self.params.num_codebooks);
            for vec_idx in vec_indices {
                let vec = self.get_vector(vec_idx as usize);

                // Recompute residual
                let residual: Vec<f32> = vec
                    .iter()
                    .zip(centroid.iter())
                    .map(|(x, c)| x - c)
                    .collect();

                let codes = quantizer.quantize(&residual);
                all_codes.extend(codes);
            }

            self.partitions[p_idx].codes = all_codes;
        }

        self.quantizer = Some(quantizer);
        self.built = true;
        Ok(())
    }

    /// Search for the k nearest neighbors of the query vector.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "index must be built before search".into(),
            ));
        }

        let quantizer = self
            .quantizer
            .as_ref()
            .ok_or(RetrieveError::InvalidParameter(
                "quantizer not initialized".into(),
            ))?;

        // 1. Find top partitions
        // Compute dot product with all centroids
        let mut partition_scores: Vec<(usize, f32)> = self
            .partition_centroids
            .iter()
            .enumerate()
            .map(|(idx, c)| (idx, crate::simd::dot(query, c)))
            .collect();

        // Sort by score (descending for MIPS)
        partition_scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // Select top partitions
        let num_probe = (self.params.num_partitions / 10).clamp(1, 10);
        let lut = quantizer.build_lut(query); // Precompute LUT for residuals

        let mut candidates = Vec::new();

        // 2. Search within partitions
        for (p_idx, center_score) in partition_scores.iter().take(num_probe) {
            let partition = &self.partitions[*p_idx];
            let num_vectors = partition.vector_indices.len();
            let m = self.params.num_codebooks;

            for i in 0..num_vectors {
                // Reconstruct approximate score:
                // <q, x> ≈ <q, c> + <q, r>
                // We have <q, c> as center_score
                // <q, r> is approximated by LUT

                let mut residual_score = 0.0;
                let code_start = i * m;
                let codes = &partition.codes[code_start..code_start + m];

                for (subspace_idx, &code) in codes.iter().enumerate() {
                    residual_score += lut[subspace_idx][code as usize];
                }

                let approx_score = center_score + residual_score;
                candidates.push((partition.vector_indices[i], approx_score));
            }
        }

        // 3. Re-rank top candidates
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let top_candidates: Vec<(u32, f32)> = candidates
            .into_iter()
            .take(self.params.num_reorder.max(k))
            .collect();

        // Exact re-ranking
        let reranked = reranking::rerank(query, &top_candidates, &self.vectors, self.dimension, k);
        Ok(reranked)
    }

    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        &self.vectors[start..start + self.dimension]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::RetrieveError;

    #[test]
    fn test_create_index() {
        let params = SCANNParams {
            num_partitions: 2,
            num_reorder: 10,
            num_codebooks: 2,
            codebook_size: 256,
            seed: 42,
        };
        let index = SCANNIndex::new(4, params);
        assert!(index.is_ok());
        let index = index.unwrap();
        assert_eq!(index.dimension, 4);
        assert_eq!(index.num_vectors, 0);
    }

    #[test]
    fn test_add_and_search() {
        let params = SCANNParams {
            num_partitions: 2,
            num_reorder: 10,
            num_codebooks: 2,
            codebook_size: 256,
            seed: 42,
        };
        let mut index = SCANNIndex::new(4, params).unwrap();

        // Add 20 vectors (need enough for k-means partitioning)
        for i in 0..20u32 {
            let v = vec![i as f32, (i as f32) * 0.5, 1.0, 0.0];
            index.add(i, v).unwrap();
        }

        index.build().unwrap();

        let query = vec![0.0, 0.0, 1.0, 0.0];
        let results = index.search(&query, 3).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_zero_dimension_error() {
        let result = SCANNIndex::new(0, SCANNParams::default());
        assert!(result.is_err());
        match result.unwrap_err() {
            RetrieveError::InvalidParameter(_) => {}
            other => panic!("Expected InvalidParameter, got {:?}", other),
        }
    }
}
