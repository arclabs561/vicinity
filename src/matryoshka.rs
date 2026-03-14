//! Matryoshka Embedding Support
//!
//! Implements support for Matryoshka Representation Learning (MRL) embeddings,
//! which allow a single embedding model to produce representations at multiple
//! dimensionalities. The key insight is that the first d dimensions of a larger
//! embedding contain meaningful information at that lower dimensionality.
//!
//! References:
//! - Kusupati et al. (2022): "Matryoshka Representation Learning"
//! - Li et al. (2024): "2D Matryoshka Sentence Embeddings"
//!
//! Features:
//! - Truncation to any dimension <= original
//! - Adaptive dimension selection based on accuracy/speed tradeoff
//! - Cascaded search: coarse filtering at low dim, refinement at high dim

use crate::RetrieveError;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Configuration for Matryoshka embedding handling.
#[derive(Debug, Clone)]
pub struct MatryoshkaConfig {
    /// Full dimension of the embedding.
    pub full_dimension: usize,
    /// Supported truncation dimensions (must be sorted ascending).
    /// Common values: [64, 128, 256, 512, 768, 1024]
    pub supported_dimensions: Vec<usize>,
    /// Default dimension for search if not specified.
    pub default_dimension: usize,
    /// Whether to use cascaded search (coarse-to-fine).
    pub cascaded_search: bool,
    /// Expansion factor for coarse search candidates.
    pub cascade_expansion: usize,
}

impl Default for MatryoshkaConfig {
    fn default() -> Self {
        Self {
            full_dimension: 768,
            supported_dimensions: vec![64, 128, 256, 384, 512, 768],
            default_dimension: 256,
            cascaded_search: true,
            cascade_expansion: 4,
        }
    }
}

impl MatryoshkaConfig {
    /// Create config for OpenAI text-embedding-3 style models.
    pub fn openai_style() -> Self {
        Self {
            full_dimension: 3072,
            supported_dimensions: vec![256, 512, 1024, 1536, 3072],
            default_dimension: 1024,
            cascaded_search: true,
            cascade_expansion: 4,
        }
    }

    /// Create config for sentence-transformers style models.
    pub fn sentence_transformers() -> Self {
        Self {
            full_dimension: 768,
            supported_dimensions: vec![64, 128, 256, 384, 512, 768],
            default_dimension: 256,
            cascaded_search: true,
            cascade_expansion: 4,
        }
    }

    /// Create config for Cohere embed-v3 style models.
    pub fn cohere_style() -> Self {
        Self {
            full_dimension: 1024,
            supported_dimensions: vec![256, 512, 768, 1024],
            default_dimension: 512,
            cascaded_search: true,
            cascade_expansion: 4,
        }
    }

    /// Find the best supported dimension >= requested.
    pub fn find_dimension(&self, requested: usize) -> usize {
        for &dim in &self.supported_dimensions {
            if dim >= requested {
                return dim;
            }
        }
        self.full_dimension
    }

    /// Find the coarse dimension for cascaded search.
    pub fn coarse_dimension(&self) -> usize {
        self.supported_dimensions
            .first()
            .copied()
            .unwrap_or(self.full_dimension / 4)
    }
}

/// A Matryoshka embedding that can be truncated to different dimensions.
#[derive(Debug, Clone)]
pub struct MatryoshkaEmbedding {
    /// The full embedding vector.
    data: Vec<f32>,
    /// Configuration for this embedding type (stored for potential future use).
    #[allow(dead_code)]
    config: MatryoshkaConfig,
}

impl MatryoshkaEmbedding {
    /// Create a new Matryoshka embedding.
    pub fn new(data: Vec<f32>, config: MatryoshkaConfig) -> Result<Self, RetrieveError> {
        if data.len() != config.full_dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: config.full_dimension,
                doc_dim: data.len(),
            });
        }
        Ok(Self { data, config })
    }

    /// Get embedding truncated to specified dimension.
    #[inline]
    pub fn at_dimension(&self, dim: usize) -> &[f32] {
        let dim = dim.min(self.data.len());
        &self.data[..dim]
    }

    /// Get the full embedding.
    #[inline]
    pub fn full(&self) -> &[f32] {
        &self.data
    }

    /// Get dimension of full embedding.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.data.len()
    }

    /// Compute cosine similarity at a specific dimension.
    pub fn cosine_similarity_at(&self, other: &Self, dim: usize) -> f32 {
        let a = self.at_dimension(dim);
        let b = other.at_dimension(dim);
        cosine_similarity(a, b)
    }

    /// Compute L2 distance at a specific dimension.
    pub fn l2_distance_at(&self, other: &Self, dim: usize) -> f32 {
        let a = self.at_dimension(dim);
        let b = other.at_dimension(dim);
        l2_distance(a, b)
    }

    /// Compute inner product at a specific dimension.
    pub fn inner_product_at(&self, other: &Self, dim: usize) -> f32 {
        let a = self.at_dimension(dim);
        let b = other.at_dimension(dim);
        inner_product(a, b)
    }
}

/// Index that supports Matryoshka embeddings with cascaded search.
#[derive(Debug)]
pub struct MatryoshkaIndex {
    /// All embeddings stored at full dimension.
    embeddings: Vec<MatryoshkaEmbedding>,
    /// Document IDs corresponding to embeddings.
    doc_ids: Vec<u32>,
    /// Configuration.
    config: MatryoshkaConfig,
    /// Statistics.
    stats: MatryoshkaStats,
}

/// Statistics for Matryoshka index operations.
#[derive(Debug, Default, Clone)]
pub struct MatryoshkaStats {
    /// Total searches performed.
    pub total_searches: u64,
    /// Searches using cascaded approach.
    pub cascaded_searches: u64,
    /// Average coarse candidates per search.
    pub avg_coarse_candidates: f64,
    /// Distance computations saved by cascading.
    pub distance_computations_saved: u64,
}

/// Candidate for heap-based search.
#[derive(Debug, Clone)]
struct Candidate {
    id: u32,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: smaller distance = higher priority
        // Use total_cmp for IEEE 754 total ordering (NaN-safe)
        self.distance.total_cmp(&other.distance).reverse()
    }
}

impl MatryoshkaIndex {
    /// Create a new empty index.
    pub fn new(config: MatryoshkaConfig) -> Self {
        Self {
            embeddings: Vec::new(),
            doc_ids: Vec::new(),
            config,
            stats: MatryoshkaStats::default(),
        }
    }

    /// Add an embedding to the index.
    pub fn add(&mut self, doc_id: u32, embedding: Vec<f32>) -> Result<(), RetrieveError> {
        let emb = MatryoshkaEmbedding::new(embedding, self.config.clone())?;
        self.embeddings.push(emb);
        self.doc_ids.push(doc_id);
        Ok(())
    }

    /// Add multiple embeddings.
    pub fn add_batch(&mut self, items: Vec<(u32, Vec<f32>)>) -> Result<(), RetrieveError> {
        for (doc_id, embedding) in items {
            self.add(doc_id, embedding)?;
        }
        Ok(())
    }

    /// Search at a specific dimension (no cascading).
    pub fn search_at_dimension(
        &self,
        query: &[f32],
        k: usize,
        dimension: usize,
    ) -> Vec<(u32, f32)> {
        let query_slice = &query[..dimension.min(query.len())];

        let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k + 1);

        for (idx, emb) in self.embeddings.iter().enumerate() {
            let emb_slice = emb.at_dimension(dimension);
            let dist = l2_distance(query_slice, emb_slice);

            heap.push(Candidate {
                id: self.doc_ids[idx],
                distance: dist,
            });

            if heap.len() > k {
                heap.pop();
            }
        }

        let mut results: Vec<_> = heap.into_iter().map(|c| (c.id, c.distance)).collect();
        results.sort_by(|a, b| a.1.total_cmp(&b.1));
        results
    }

    /// Cascaded search: coarse at low dim, refine at high dim.
    pub fn search_cascaded(&mut self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        self.stats.total_searches += 1;
        self.stats.cascaded_searches += 1;

        let coarse_dim = self.config.coarse_dimension();
        let fine_dim = self.config.default_dimension;
        let coarse_k = k * self.config.cascade_expansion;

        // Phase 1: Coarse search at low dimension
        let coarse_results = self.search_at_dimension(query, coarse_k, coarse_dim);

        self.stats.avg_coarse_candidates = (self.stats.avg_coarse_candidates
            * (self.stats.cascaded_searches - 1) as f64
            + coarse_results.len() as f64)
            / self.stats.cascaded_searches as f64;

        // Phase 2: Refine at higher dimension
        let query_fine = &query[..fine_dim.min(query.len())];
        let mut refined: Vec<(u32, f32)> = coarse_results
            .iter()
            .filter_map(|(doc_id, _)| {
                let idx = self.doc_ids.iter().position(|&id| id == *doc_id)?;
                let emb = &self.embeddings[idx];
                let dist = l2_distance(query_fine, emb.at_dimension(fine_dim));
                Some((*doc_id, dist))
            })
            .collect();

        refined.sort_by(|a, b| a.1.total_cmp(&b.1));
        refined.truncate(k);

        // Compute savings
        let full_computations = self.embeddings.len() as u64;
        let actual_computations = self.embeddings.len() as u64 + coarse_results.len() as u64;
        if full_computations > actual_computations {
            self.stats.distance_computations_saved += full_computations - actual_computations;
        }

        refined
    }

    /// Search with automatic strategy selection.
    pub fn search(&mut self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        self.stats.total_searches += 1;

        if self.config.cascaded_search && self.embeddings.len() > 1000 {
            self.search_cascaded(query, k)
        } else {
            self.search_at_dimension(query, k, self.config.default_dimension)
        }
    }

    /// Get number of embeddings in index.
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Check if index is empty.
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Get statistics.
    pub fn stats(&self) -> &MatryoshkaStats {
        &self.stats
    }
}

/// Adaptive dimension selector based on query characteristics.
#[derive(Debug)]
pub struct AdaptiveDimensionSelector {
    /// Supported dimensions.
    dimensions: Vec<usize>,
    /// Accuracy estimates per dimension (from calibration).
    accuracy_estimates: Vec<f32>,
    /// Latency estimates per dimension (relative).
    latency_estimates: Vec<f32>,
    /// Target accuracy threshold.
    target_accuracy: f32,
}

impl AdaptiveDimensionSelector {
    /// Create a new selector with default estimates.
    pub fn new(dimensions: Vec<usize>) -> Self {
        let _n = dimensions.len();
        // Assume accuracy scales roughly with sqrt(dim)
        let accuracy_estimates: Vec<f32> = dimensions
            .iter()
            .map(|&d| (d as f32).sqrt() / (dimensions.last().copied().unwrap_or(1) as f32).sqrt())
            .collect();
        // Latency scales linearly with dimension
        let latency_estimates: Vec<f32> = dimensions
            .iter()
            .map(|&d| d as f32 / dimensions.last().copied().unwrap_or(1) as f32)
            .collect();

        Self {
            dimensions,
            accuracy_estimates,
            latency_estimates,
            target_accuracy: 0.95,
        }
    }

    /// Set target accuracy threshold.
    pub fn with_target_accuracy(mut self, accuracy: f32) -> Self {
        self.target_accuracy = accuracy;
        self
    }

    /// Update accuracy estimates from observed data.
    pub fn calibrate(&mut self, dimension_idx: usize, observed_accuracy: f32) {
        if dimension_idx < self.accuracy_estimates.len() {
            // Exponential moving average
            self.accuracy_estimates[dimension_idx] =
                0.9 * self.accuracy_estimates[dimension_idx] + 0.1 * observed_accuracy;
        }
    }

    /// Select the best dimension for given accuracy/speed preference.
    /// speed_preference: 0.0 = accuracy only, 1.0 = speed only
    pub fn select_dimension(&self, speed_preference: f32) -> usize {
        let mut best_dim = *self.dimensions.last().unwrap_or(&768);
        let mut best_score = f32::NEG_INFINITY;

        for (i, &dim) in self.dimensions.iter().enumerate() {
            let accuracy = self.accuracy_estimates.get(i).copied().unwrap_or(0.5);
            let latency = self.latency_estimates.get(i).copied().unwrap_or(1.0);

            // Skip if below accuracy threshold
            if accuracy < self.target_accuracy {
                continue;
            }

            // Score combines accuracy (want high) and latency (want low)
            let score = (1.0 - speed_preference) * accuracy - speed_preference * latency;

            if score > best_score {
                best_score = score;
                best_dim = dim;
            }
        }

        best_dim
    }

    /// Get the minimum dimension that meets accuracy threshold.
    pub fn minimum_dimension(&self) -> usize {
        for (i, &dim) in self.dimensions.iter().enumerate() {
            if self.accuracy_estimates.get(i).copied().unwrap_or(0.0) >= self.target_accuracy {
                return dim;
            }
        }
        *self.dimensions.last().unwrap_or(&768)
    }
}

// Delegate to crate-level distance functions (SIMD-accelerated).

#[inline]
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    crate::distance::l2_distance(a, b)
}

#[inline]
fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::dot(a, b)
}

#[inline]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::cosine(a, b)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn make_embedding(dim: usize, seed: u32) -> Vec<f32> {
        (0..dim)
            .map(|i| ((seed as f32 * 0.1 + i as f32) * 0.01).sin())
            .collect()
    }

    #[test]
    fn test_matryoshka_embedding_truncation() {
        let config = MatryoshkaConfig::default();
        let data = make_embedding(768, 42);
        let emb = MatryoshkaEmbedding::new(data.clone(), config).unwrap();

        assert_eq!(emb.at_dimension(64).len(), 64);
        assert_eq!(emb.at_dimension(256).len(), 256);
        assert_eq!(emb.at_dimension(768).len(), 768);
        assert_eq!(emb.at_dimension(1000).len(), 768); // Clamped
    }

    #[test]
    fn test_matryoshka_index_search() {
        let config = MatryoshkaConfig::default();
        let mut index = MatryoshkaIndex::new(config.clone());

        // Add some embeddings
        for i in 0..100 {
            let emb = make_embedding(768, i);
            index.add(i, emb).unwrap();
        }

        let query = make_embedding(768, 50);
        let results = index.search_at_dimension(&query, 5, 256);

        assert_eq!(results.len(), 5);
        // Results should be sorted by distance (ascending)
        for i in 1..results.len() {
            assert!(
                results[i - 1].1 <= results[i].1,
                "Results should be sorted by distance"
            );
        }
    }

    #[test]
    fn test_cascaded_search() {
        let config = MatryoshkaConfig {
            cascaded_search: true,
            ..Default::default()
        };
        let mut index = MatryoshkaIndex::new(config);

        for i in 0..2000 {
            let emb = make_embedding(768, i);
            index.add(i, emb).unwrap();
        }

        let query = make_embedding(768, 1000);
        let results = index.search_cascaded(&query, 10);

        assert_eq!(results.len(), 10);
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(
                results[i - 1].1 <= results[i].1,
                "Results should be sorted by distance"
            );
        }
        assert!(index.stats.cascaded_searches > 0);
    }

    #[test]
    fn test_dimension_selector() {
        let dims = vec![64, 128, 256, 512, 768];
        let selector = AdaptiveDimensionSelector::new(dims).with_target_accuracy(0.9);

        // High speed preference should pick smaller dimension
        let fast_dim = selector.select_dimension(0.8);
        let accurate_dim = selector.select_dimension(0.2);

        assert!(fast_dim <= accurate_dim);
    }

    #[test]
    fn test_config_presets() {
        let openai = MatryoshkaConfig::openai_style();
        assert_eq!(openai.full_dimension, 3072);

        let cohere = MatryoshkaConfig::cohere_style();
        assert_eq!(cohere.full_dimension, 1024);

        let st = MatryoshkaConfig::sentence_transformers();
        assert_eq!(st.full_dimension, 768);
    }
}
