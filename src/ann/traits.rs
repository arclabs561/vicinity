//! Unified traits for all ANN algorithms.

use crate::RetrieveError;

/// Unified trait for all ANN index implementations.
pub trait ANNIndex {
    /// Add a vector to the index.
    fn add(&mut self, doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError>;

    /// Add a vector to the index from a borrowed slice.
    ///
    /// Default implementation allocates a `Vec<f32>` and calls [`ANNIndex::add`].
    /// Implementations that store vectors in a flat buffer (SoA) should override
    /// this to avoid an intermediate allocation.
    fn add_slice(&mut self, doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
        self.add(doc_id, vector.to_vec())
    }

    /// Build the index (required before search).
    fn build(&mut self) -> Result<(), RetrieveError>;

    /// Search for k nearest neighbors.
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError>;

    /// Search with per-query effort parameter.
    ///
    /// `ef` controls the search effort (higher = better recall, slower).
    /// Default implementation ignores `ef` and delegates to [`search`](ANNIndex::search).
    /// Index types that support tunable search effort (HNSW, NSW, DiskANN) override this.
    fn search_ef(
        &self,
        query: &[f32],
        k: usize,
        _ef: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.search(query, k)
    }

    /// Get index size in bytes (approximate).
    fn size_bytes(&self) -> usize;

    /// Get index statistics.
    fn stats(&self) -> ANNStats;

    /// Get vector dimension.
    fn dimension(&self) -> usize;

    /// Get number of vectors.
    fn num_vectors(&self) -> usize;
}

/// Statistics about an ANN index.
#[derive(Debug, Clone)]
pub struct ANNStats {
    /// Number of vectors in the index.
    pub num_vectors: usize,
    /// Dimensionality of each vector.
    pub dimension: usize,
    /// Approximate memory usage in bytes.
    pub size_bytes: usize,
    /// Name of the ANN algorithm (e.g. "HNSW", "DiskANN").
    pub algorithm: &'static str,
}

// Implement ANNIndex for HNSW
#[cfg(feature = "hnsw")]
impl ANNIndex for crate::hnsw::HNSWIndex {
    fn add(&mut self, doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        self.add(doc_id, vector)
    }

    fn add_slice(&mut self, doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
        self.add_slice(doc_id, vector)
    }

    fn build(&mut self) -> Result<(), RetrieveError> {
        self.build()
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.search(query, k, self.params.ef_search)
    }

    fn search_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.search(query, k, ef)
    }

    fn size_bytes(&self) -> usize {
        // Approximate: vectors + graph structure
        self.vectors.len() * std::mem::size_of::<f32>()
            + self
                .layers
                .iter()
                .map(|l| l.len() * std::mem::size_of::<u32>())
                .sum::<usize>()
    }

    fn stats(&self) -> ANNStats {
        ANNStats {
            num_vectors: self.num_vectors,
            dimension: self.dimension,
            size_bytes: self.size_bytes(),
            algorithm: "HNSW",
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn num_vectors(&self) -> usize {
        self.num_vectors
    }
}

// Implement ANNIndex for Anisotropic Vector Quantization with k-means Partitioning
// (Technical name; vendor name: SCANN/ScaNN)
#[cfg(feature = "scann")]
impl ANNIndex for crate::scann::search::SCANNIndex {
    fn add(&mut self, doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        self.add(doc_id, vector)
    }

    fn add_slice(&mut self, doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
        self.add_slice(doc_id, vector)
    }

    fn build(&mut self) -> Result<(), RetrieveError> {
        self.build()
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.search(query, k)
    }

    fn size_bytes(&self) -> usize {
        self.vectors.len() * std::mem::size_of::<f32>()
            + self
                .partition_centroids
                .iter()
                .map(|c| c.len() * std::mem::size_of::<f32>())
                .sum::<usize>()
    }

    fn stats(&self) -> ANNStats {
        ANNStats {
            num_vectors: self.num_vectors,
            dimension: self.dimension,
            size_bytes: self.size_bytes(),
            algorithm: "AnisotropicVQ-kmeans", // Technical name (vendor: SCANN)
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn num_vectors(&self) -> usize {
        self.num_vectors
    }
}

// Implement ANNIndex for IVF-PQ
#[cfg(feature = "ivf_pq")]
impl ANNIndex for crate::ivf_pq::search::IVFPQIndex {
    fn add(&mut self, doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        self.add(doc_id, vector)
    }

    fn add_slice(&mut self, doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
        self.add_slice(doc_id, vector)
    }

    fn build(&mut self) -> Result<(), RetrieveError> {
        self.build()
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.search(query, k)
    }

    fn size_bytes(&self) -> usize {
        self.vectors.len() * std::mem::size_of::<f32>()
            + self
                .centroids
                .iter()
                .map(|c| c.len() * std::mem::size_of::<f32>())
                .sum::<usize>()
            + self.quantized_codes.len() * std::mem::size_of::<u8>()
    }

    fn stats(&self) -> ANNStats {
        ANNStats {
            num_vectors: self.num_vectors,
            dimension: self.dimension,
            size_bytes: self.size_bytes(),
            algorithm: "IVF-PQ",
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn num_vectors(&self) -> usize {
        self.num_vectors
    }
}

// Implement ANNIndex for KD-Tree
#[cfg(feature = "kdtree")]
impl ANNIndex for crate::classic::trees::kdtree::KDTreeIndex {
    fn add(&mut self, doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        self.add(doc_id, vector)
    }

    fn build(&mut self) -> Result<(), RetrieveError> {
        self.build()
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.search(query, k)
    }

    fn size_bytes(&self) -> usize {
        self.vectors.len() * std::mem::size_of::<f32>()
    }

    fn stats(&self) -> ANNStats {
        ANNStats {
            num_vectors: self.num_vectors,
            dimension: self.dimension,
            size_bytes: self.size_bytes(),
            algorithm: "KD-Tree",
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn num_vectors(&self) -> usize {
        self.num_vectors
    }
}

// Implement ANNIndex for Ball Tree
#[cfg(feature = "balltree")]
impl ANNIndex for crate::classic::trees::balltree::BallTreeIndex {
    fn add(&mut self, doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        self.add(doc_id, vector)
    }

    fn build(&mut self) -> Result<(), RetrieveError> {
        self.build()
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.search(query, k)
    }

    fn size_bytes(&self) -> usize {
        self.vectors.len() * std::mem::size_of::<f32>()
    }

    fn stats(&self) -> ANNStats {
        ANNStats {
            num_vectors: self.num_vectors,
            dimension: self.dimension,
            size_bytes: self.size_bytes(),
            algorithm: "Ball-Tree",
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn num_vectors(&self) -> usize {
        self.num_vectors
    }
}

// Implement ANNIndex for K-Means Tree
#[cfg(feature = "kmeans_tree")]
impl ANNIndex for crate::classic::trees::kmeans_tree::KMeansTreeIndex {
    fn add(&mut self, doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        self.add(doc_id, vector)
    }

    fn build(&mut self) -> Result<(), RetrieveError> {
        self.build()
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.search(query, k)
    }

    fn size_bytes(&self) -> usize {
        self.vectors.len() * std::mem::size_of::<f32>()
    }

    fn stats(&self) -> ANNStats {
        ANNStats {
            num_vectors: self.num_vectors,
            dimension: self.dimension,
            size_bytes: self.size_bytes(),
            algorithm: "K-Means-Tree",
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn num_vectors(&self) -> usize {
        self.num_vectors
    }
}

// Implement ANNIndex for Random Projection Tree
#[cfg(feature = "rptree")]
impl ANNIndex for crate::classic::trees::random_projection::RPTreeIndex {
    fn add(&mut self, doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        self.add(doc_id, vector)
    }

    fn build(&mut self) -> Result<(), RetrieveError> {
        self.build()
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.search(query, k)
    }

    fn size_bytes(&self) -> usize {
        self.vectors.len() * std::mem::size_of::<f32>()
    }

    fn stats(&self) -> ANNStats {
        ANNStats {
            num_vectors: self.num_vectors,
            dimension: self.dimension,
            size_bytes: self.size_bytes(),
            algorithm: "RP-Tree",
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn num_vectors(&self) -> usize {
        self.num_vectors
    }
}

// Implement ANNIndex for OPT-SNG
#[cfg(feature = "sng")]
impl ANNIndex for crate::sng::SNGIndex {
    fn add(&mut self, doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        self.add(doc_id, vector)
    }

    fn build(&mut self) -> Result<(), RetrieveError> {
        self.build()
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.search(query, k)
    }

    fn size_bytes(&self) -> usize {
        self.vectors.len() * std::mem::size_of::<f32>()
            + self
                .neighbors
                .iter()
                .map(|n| n.len() * std::mem::size_of::<u32>())
                .sum::<usize>()
    }

    fn stats(&self) -> ANNStats {
        ANNStats {
            num_vectors: self.num_vectors,
            dimension: self.dimension,
            size_bytes: self.size_bytes(),
            algorithm: "OPT-SNG",
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn num_vectors(&self) -> usize {
        self.num_vectors
    }
}

// Implement ANNIndex for DiskANN
#[cfg(feature = "diskann")]
impl ANNIndex for crate::diskann::graph::DiskANNIndex {
    fn add(&mut self, doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        self.add(doc_id, vector)
    }

    fn add_slice(&mut self, doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
        self.add_slice(doc_id, vector)
    }

    fn build(&mut self) -> Result<(), RetrieveError> {
        self.build()
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.search(query, k, self.ef_search())
    }

    fn search_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.search(query, k, ef)
    }

    fn size_bytes(&self) -> usize {
        self.size_bytes()
    }

    fn stats(&self) -> ANNStats {
        ANNStats {
            num_vectors: self.num_vectors(),
            dimension: self.dimension(),
            size_bytes: self.size_bytes(),
            algorithm: "DiskANN",
        }
    }

    fn dimension(&self) -> usize {
        self.dimension()
    }

    fn num_vectors(&self) -> usize {
        self.num_vectors()
    }
}

// NOTE: Annoy (Random Projection Tree Forest) impl removed - module not yet implemented.
// The rp_forest module provides similar functionality via RPForestIndex.

// Implement ANNIndex for Flat NSW
#[cfg(feature = "nsw")]
impl ANNIndex for crate::nsw::NSWIndex {
    fn add(&mut self, doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        self.add(doc_id, vector)
    }

    fn add_slice(&mut self, doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
        self.add_slice(doc_id, vector)
    }

    fn build(&mut self) -> Result<(), RetrieveError> {
        self.build()
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.search(query, k, self.params.ef_search)
    }

    fn search_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.search(query, k, ef)
    }

    fn size_bytes(&self) -> usize {
        self.vectors.len() * std::mem::size_of::<f32>()
            + self
                .neighbors
                .iter()
                .map(|n| n.len() * std::mem::size_of::<u32>())
                .sum::<usize>()
    }

    fn stats(&self) -> ANNStats {
        ANNStats {
            num_vectors: self.num_vectors,
            dimension: self.dimension,
            size_bytes: self.size_bytes(),
            algorithm: "NSW",
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn num_vectors(&self) -> usize {
        self.num_vectors
    }
}
