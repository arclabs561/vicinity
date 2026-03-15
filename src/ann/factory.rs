//! Index factory for creating ANN indexes from string descriptions.
//!
//! Inspired by Faiss's `index_factory` pattern, this module provides a simple
//! string-based API for creating different ANN index types.
//!
//! # Usage
//!
//! ```rust,no_run
//! use vicinity::ann::{index_factory, ANNIndex};
//!
//! fn main() -> Result<(), vicinity::RetrieveError> {
//!     // Create HNSW index (requires "hnsw" feature)
//!     let mut index = index_factory(128, "HNSW32")?;
//!     index.add_slice(0, &[0.1; 128])?;
//!     index.build()?;
//!     let results = index.search(&[0.1; 128], 10)?;
//!     Ok(())
//! }
//! ```
//!
//! # Supported Index Types
//!
//! - `"HNSW{m}"` - Hierarchical Navigable Small World (e.g., "HNSW32")
//! - `"NSW{m}"` - Flat Navigable Small World (e.g., "NSW32")
//! - `"IVF{n},PQ{m}"` - Inverted File Index with Product Quantization (e.g., "IVF1024,PQ8")
//! - `"SCANN{n}"` - Anisotropic Vector Quantization with k-means (e.g., "SCANN256")
//!
//! **Note:** Tree-based methods (KD-Tree, Ball Tree, K-Means Tree, Random Projection Tree) are not
//! supported via the factory pattern due to complex parameter structures. Create them directly
//! using their respective constructors.
//!
//! # Future Support
//!
//! - `"PCA{d},..."` - PCA preprocessing (e.g., "PCA64,IVF1024,PQ8")
//! - Composite indexes with preprocessing pipelines

use crate::ann::ANNIndex;
use crate::RetrieveError;

// Make sure AnyANNIndex is visible
#[cfg(any(
    feature = "hnsw",
    feature = "nsw",
    feature = "ivf_pq",
    feature = "scann",
))]
pub use self::AnyANNIndex as ANNIndexEnum;

/// Type-erased ANN index container.
///
/// This enum allows storing different index types in a single variable,
/// enabling polymorphic usage through the `ANNIndex` trait.
#[derive(Debug)]
pub enum AnyANNIndex {
    /// Hierarchical Navigable Small World index.
    #[cfg(feature = "hnsw")]
    HNSW(crate::hnsw::HNSWIndex),

    /// Flat Navigable Small World index.
    #[cfg(feature = "nsw")]
    NSW(crate::nsw::NSWIndex),

    /// Inverted File Index with Product Quantization.
    #[cfg(feature = "ivf_pq")]
    IVFPQ(crate::ivf_pq::IVFPQIndex),

    /// Anisotropic vector quantization with k-means partitioning.
    #[cfg(feature = "scann")]
    SCANN(crate::scann::search::SCANNIndex),

    /// Hierarchical k-means tree index.
    #[cfg(feature = "kmeans_tree")]
    KMeansTree(crate::classic::trees::kmeans_tree::KMeansTreeIndex),
}

impl ANNIndex for AnyANNIndex {
    fn add(&mut self, doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        match self {
            #[cfg(feature = "hnsw")]
            AnyANNIndex::HNSW(idx) => idx.add(doc_id, vector),

            #[cfg(feature = "nsw")]
            AnyANNIndex::NSW(idx) => idx.add(doc_id, vector),

            #[cfg(feature = "ivf_pq")]
            AnyANNIndex::IVFPQ(idx) => idx.add(doc_id, vector),

            #[cfg(feature = "scann")]
            AnyANNIndex::SCANN(idx) => idx.add(doc_id, vector),

            #[cfg(feature = "kmeans_tree")]
            AnyANNIndex::KMeansTree(idx) => idx.add(doc_id, vector),
        }
    }

    fn add_slice(&mut self, doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
        match self {
            #[cfg(feature = "hnsw")]
            AnyANNIndex::HNSW(idx) => idx.add_slice(doc_id, vector),

            #[cfg(feature = "nsw")]
            AnyANNIndex::NSW(idx) => idx.add_slice(doc_id, vector),

            #[cfg(feature = "ivf_pq")]
            AnyANNIndex::IVFPQ(idx) => idx.add_slice(doc_id, vector),

            #[cfg(feature = "scann")]
            AnyANNIndex::SCANN(idx) => idx.add_slice(doc_id, vector),

            #[cfg(feature = "kmeans_tree")]
            AnyANNIndex::KMeansTree(idx) => idx.add_slice(doc_id, vector),
        }
    }

    fn build(&mut self) -> Result<(), RetrieveError> {
        match self {
            #[cfg(feature = "hnsw")]
            AnyANNIndex::HNSW(idx) => idx.build(),

            #[cfg(feature = "nsw")]
            AnyANNIndex::NSW(idx) => idx.build(),

            #[cfg(feature = "ivf_pq")]
            AnyANNIndex::IVFPQ(idx) => idx.build(),

            #[cfg(feature = "scann")]
            AnyANNIndex::SCANN(idx) => idx.build(),

            #[cfg(feature = "kmeans_tree")]
            AnyANNIndex::KMeansTree(idx) => idx.build(),
        }
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        match self {
            #[cfg(feature = "hnsw")]
            AnyANNIndex::HNSW(idx) => idx.search(query, k, idx.params.ef_search),

            #[cfg(feature = "nsw")]
            AnyANNIndex::NSW(idx) => idx.search(query, k, idx.params.ef_search),

            #[cfg(feature = "ivf_pq")]
            AnyANNIndex::IVFPQ(idx) => idx.search(query, k),

            #[cfg(feature = "scann")]
            AnyANNIndex::SCANN(idx) => idx.search(query, k),

            #[cfg(feature = "kmeans_tree")]
            AnyANNIndex::KMeansTree(idx) => idx.search(query, k),
        }
    }

    fn search_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        match self {
            #[cfg(feature = "hnsw")]
            AnyANNIndex::HNSW(idx) => idx.search(query, k, ef),

            #[cfg(feature = "nsw")]
            AnyANNIndex::NSW(idx) => idx.search(query, k, ef),

            // IVF-PQ, ScaNN, KMeansTree don't have ef_search -- delegate to search()
            #[cfg(feature = "ivf_pq")]
            AnyANNIndex::IVFPQ(idx) => idx.search(query, k),

            #[cfg(feature = "scann")]
            AnyANNIndex::SCANN(idx) => idx.search(query, k),

            #[cfg(feature = "kmeans_tree")]
            AnyANNIndex::KMeansTree(idx) => idx.search(query, k),
        }
    }

    fn size_bytes(&self) -> usize {
        match self {
            #[cfg(feature = "hnsw")]
            AnyANNIndex::HNSW(idx) => idx.size_bytes(),

            #[cfg(feature = "nsw")]
            AnyANNIndex::NSW(idx) => idx.size_bytes(),

            #[cfg(feature = "ivf_pq")]
            AnyANNIndex::IVFPQ(idx) => idx.size_bytes(),

            #[cfg(feature = "scann")]
            AnyANNIndex::SCANN(idx) => idx.size_bytes(),

            #[cfg(feature = "kmeans_tree")]
            AnyANNIndex::KMeansTree(idx) => idx.size_bytes(),
        }
    }

    fn stats(&self) -> crate::ann::ANNStats {
        match self {
            #[cfg(feature = "hnsw")]
            AnyANNIndex::HNSW(idx) => idx.stats(),

            #[cfg(feature = "nsw")]
            AnyANNIndex::NSW(idx) => idx.stats(),

            #[cfg(feature = "ivf_pq")]
            AnyANNIndex::IVFPQ(idx) => idx.stats(),

            #[cfg(feature = "scann")]
            AnyANNIndex::SCANN(idx) => idx.stats(),

            #[cfg(feature = "kmeans_tree")]
            AnyANNIndex::KMeansTree(idx) => idx.stats(),
        }
    }

    fn dimension(&self) -> usize {
        match self {
            #[cfg(feature = "hnsw")]
            AnyANNIndex::HNSW(idx) => idx.dimension(),

            #[cfg(feature = "nsw")]
            AnyANNIndex::NSW(idx) => idx.dimension(),

            #[cfg(feature = "ivf_pq")]
            AnyANNIndex::IVFPQ(idx) => idx.dimension(),

            #[cfg(feature = "scann")]
            AnyANNIndex::SCANN(idx) => idx.dimension(),

            #[cfg(feature = "kmeans_tree")]
            AnyANNIndex::KMeansTree(idx) => idx.dimension(),
        }
    }

    fn num_vectors(&self) -> usize {
        match self {
            #[cfg(feature = "hnsw")]
            AnyANNIndex::HNSW(idx) => idx.num_vectors(),

            #[cfg(feature = "nsw")]
            AnyANNIndex::NSW(idx) => idx.num_vectors(),

            #[cfg(feature = "ivf_pq")]
            AnyANNIndex::IVFPQ(idx) => idx.num_vectors(),

            #[cfg(feature = "scann")]
            AnyANNIndex::SCANN(idx) => idx.num_vectors(),

            #[cfg(feature = "kmeans_tree")]
            AnyANNIndex::KMeansTree(idx) => idx.num_vectors(),
        }
    }
}

/// Create an ANN index from a factory string.
///
/// The factory string describes the index type and parameters in a simple format.
/// This is inspired by Faiss's `index_factory` pattern.
///
/// # Supported Formats
///
/// - `"HNSW{m}"` - HNSW with m connections (e.g., "HNSW32")
/// - `"NSW{m}"` - Flat NSW with m connections (e.g., "NSW32")
/// - `"IVF{n},PQ{m}"` - IVF-PQ with n clusters and m codebooks (e.g., "IVF1024,PQ8")
/// - `"SCANN{n}"` - SCANN with n partitions (e.g., "SCANN256")
///
/// # Examples
///
/// ```rust,no_run
/// use vicinity::ann::factory::index_factory;
/// use vicinity::ann::ANNIndex;
///
/// fn main() -> Result<(), vicinity::RetrieveError> {
///     let mut index = index_factory(128, "HNSW32")?;
///     let v0 = vec![0.1; 128];
///     index.add_slice(0, &v0)?;
///     index.build()?;
///     let results = index.search(&[0.15; 128], 10)?;
///     Ok(())
/// }
/// ```
///
/// # Errors
///
/// Returns `RetrieveError` if:
/// - The factory string is invalid or unsupported
/// - Required features are not enabled
/// - Parameters are invalid (e.g., dimension = 0, m = 0)
/// - Dimension mismatch between factory and vectors
pub fn index_factory(dimension: usize, factory_string: &str) -> Result<AnyANNIndex, RetrieveError> {
    // Validate dimension
    if dimension == 0 {
        return Err(RetrieveError::Other(
            "Dimension must be greater than 0".to_string(),
        ));
    }

    let factory_string = factory_string.trim();

    // Validate empty string
    if factory_string.is_empty() {
        return Err(RetrieveError::Other(
            "Factory string cannot be empty".to_string(),
        ));
    }

    // Parse HNSW: "HNSW{m}" or "HNSW{m},{m_max}"
    if let Some(rest) = factory_string.strip_prefix("HNSW") {
        #[cfg(not(feature = "hnsw"))]
        {
            let _ = rest; // Suppress unused warning
            return Err(RetrieveError::Other(
                "HNSW feature not enabled. Add 'hnsw' feature to Cargo.toml".to_string(),
            ));
        }

        #[cfg(feature = "hnsw")]
        {
            if rest.is_empty() {
                return Err(RetrieveError::Other(
                    "HNSW format: HNSW{m} or HNSW{m},{m_max}".to_string(),
                ));
            }

            let parts: Vec<&str> = rest.split(',').collect();

            let m = parts[0].parse::<usize>().map_err(|_| {
                RetrieveError::Other(format!(
                    "Invalid HNSW parameter: '{}'. Expected number.",
                    parts[0]
                ))
            })?;

            if m == 0 {
                return Err(RetrieveError::Other(
                    "HNSW m parameter must be greater than 0".to_string(),
                ));
            }

            let m_max = if parts.len() > 1 {
                let m_max_val = parts[1].parse::<usize>().map_err(|_| {
                    RetrieveError::Other(format!(
                        "Invalid HNSW m_max parameter: '{}'. Expected number.",
                        parts[1]
                    ))
                })?;
                if m_max_val == 0 {
                    return Err(RetrieveError::Other(
                        "HNSW m_max parameter must be greater than 0".to_string(),
                    ));
                }
                m_max_val
            } else {
                m // Default m_max = m
            };

            let index = crate::hnsw::HNSWIndex::new(dimension, m, m_max)?;
            return Ok(AnyANNIndex::HNSW(index));
        }
    }

    // Parse NSW: "NSW{m}"
    if let Some(rest) = factory_string.strip_prefix("NSW") {
        #[cfg(not(feature = "nsw"))]
        {
            let _ = rest; // Suppress unused warning
            return Err(RetrieveError::Other(
                "NSW feature not enabled. Add 'nsw' feature to Cargo.toml".to_string(),
            ));
        }

        #[cfg(feature = "nsw")]
        {
            if rest.is_empty() {
                return Err(RetrieveError::Other("NSW format: NSW{m}".to_string()));
            }

            let m = rest.parse::<usize>().map_err(|_| {
                RetrieveError::Other(format!(
                    "Invalid NSW parameter: '{}'. Expected number.",
                    rest
                ))
            })?;

            if m == 0 {
                return Err(RetrieveError::Other(
                    "NSW m parameter must be greater than 0".to_string(),
                ));
            }

            let index = crate::nsw::NSWIndex::new(dimension, m, m)?;
            return Ok(AnyANNIndex::NSW(index));
        }
    }

    // Parse IVF-PQ: "IVF{n},PQ{m}" or "IVF{n},PQ{m}x{b}" (m codebooks, b bits)
    if factory_string.starts_with("IVF") {
        #[cfg(not(feature = "ivf_pq"))]
        return Err(RetrieveError::Other(
            "IVF-PQ feature not enabled. Add 'ivf_pq' feature to Cargo.toml".to_string(),
        ));

        #[cfg(feature = "ivf_pq")]
        {
            let parts: Vec<&str> = factory_string.split(',').collect();
            if parts.len() < 2 {
                return Err(RetrieveError::Other(
                    "IVF-PQ format: IVF{n},PQ{m} or IVF{n},PQ{m}x{b}".to_string(),
                ));
            }

            // Parse IVF{n}
            let ivf_part = parts[0].trim();
            if !ivf_part.starts_with("IVF") {
                return Err(RetrieveError::Other(format!(
                    "Invalid IVF format: '{}'. Expected IVF{{n}}.",
                    ivf_part
                )));
            }

            if ivf_part.len() == 3 {
                return Err(RetrieveError::Other(
                    "IVF format: IVF{n} where n is number of clusters".to_string(),
                ));
            }

            let num_clusters = ivf_part[3..].parse::<usize>().map_err(|_| {
                RetrieveError::Other(format!(
                    "Invalid IVF cluster count: '{}'. Expected number.",
                    &ivf_part[3..]
                ))
            })?;

            if num_clusters == 0 {
                return Err(RetrieveError::Other(
                    "IVF num_clusters must be greater than 0".to_string(),
                ));
            }

            // Parse PQ{m} or PQ{m}x{b}
            let pq_part = parts[1].trim();
            if !pq_part.starts_with("PQ") {
                return Err(RetrieveError::Other(format!(
                    "Invalid PQ format: '{}'. Expected PQ{{m}} or PQ{{m}}x{{b}}.",
                    pq_part
                )));
            }

            if pq_part.len() == 2 {
                return Err(RetrieveError::Other(
                    "PQ format: PQ{m} or PQ{m}x{b}".to_string(),
                ));
            }

            let pq_rest = &pq_part[2..];
            let (num_codebooks, codebook_size) = if pq_rest.contains('x') {
                let pq_parts: Vec<&str> = pq_rest.split('x').collect();
                if pq_parts.len() != 2 {
                    return Err(RetrieveError::Other(format!(
                        "Invalid PQ format: '{}'. Expected PQ{{m}}x{{b}}.",
                        pq_part
                    )));
                }
                let num_codebooks = pq_parts[0].trim().parse::<usize>().map_err(|_| {
                    RetrieveError::Other(format!(
                        "Invalid PQ codebook count: '{}'. Expected number.",
                        pq_parts[0]
                    ))
                })?;
                if num_codebooks == 0 {
                    return Err(RetrieveError::Other(
                        "PQ num_codebooks must be greater than 0".to_string(),
                    ));
                }
                let bits = pq_parts[1].trim().parse::<usize>().map_err(|_| {
                    RetrieveError::Other(format!(
                        "Invalid PQ bits: '{}'. Expected number.",
                        pq_parts[1]
                    ))
                })?;
                if bits > 16 {
                    return Err(RetrieveError::Other(format!(
                        "PQ bits ({}) exceeds maximum (16)",
                        bits
                    )));
                }
                let codebook_size = 1 << bits; // 2^bits
                (num_codebooks, codebook_size)
            } else {
                // Default: PQ8 means 8 codebooks, 256 size (8 bits)
                let num_codebooks = pq_rest.trim().parse::<usize>().map_err(|_| {
                    RetrieveError::Other(format!(
                        "Invalid PQ codebook count: '{}'. Expected number.",
                        pq_rest
                    ))
                })?;
                if num_codebooks == 0 {
                    return Err(RetrieveError::Other(
                        "PQ num_codebooks must be greater than 0".to_string(),
                    ));
                }
                (num_codebooks, 256) // Default 8 bits = 256
            };

            // Validate dimension is divisible by num_codebooks for PQ
            if dimension % num_codebooks != 0 {
                return Err(RetrieveError::Other(format!(
                    "Dimension ({}) must be divisible by num_codebooks ({}) for PQ",
                    dimension, num_codebooks
                )));
            }

            use crate::ivf_pq::IVFPQParams;
            let params = IVFPQParams {
                num_clusters,
                nprobe: (num_clusters / 10).clamp(1, 100), // Default nprobe
                num_codebooks,
                codebook_size,
                use_opq: false,
                #[cfg(feature = "id-compression")]
                id_compression: None,
                #[cfg(feature = "id-compression")]
                compression_threshold: 100,
            };

            let index = crate::ivf_pq::IVFPQIndex::new(dimension, params)?;
            return Ok(AnyANNIndex::IVFPQ(index));
        }
    }

    // Parse SCANN: "SCANN{n}"
    if let Some(rest) = factory_string.strip_prefix("SCANN") {
        #[cfg(not(feature = "scann"))]
        {
            let _ = rest; // Suppress unused warning
            return Err(RetrieveError::Other(
                "SCANN feature not enabled. Add 'scann' feature to Cargo.toml".to_string(),
            ));
        }

        #[cfg(feature = "scann")]
        {
            if rest.is_empty() {
                return Err(RetrieveError::Other("SCANN format: SCANN{n}".to_string()));
            }

            let num_partitions = rest.trim().parse::<usize>().map_err(|_| {
                RetrieveError::Other(format!(
                    "Invalid SCANN parameter: '{}'. Expected number.",
                    rest
                ))
            })?;

            if num_partitions == 0 {
                return Err(RetrieveError::Other(
                    "SCANN num_partitions must be greater than 0".to_string(),
                ));
            }

            use crate::scann::search::SCANNParams;
            let params = SCANNParams {
                num_partitions,
                num_reorder: 100,   // Default reranking count
                num_codebooks: 8,   // Default: 8 subspaces
                codebook_size: 256, // Default: 8-bit quantization (256 codewords)
                seed: 42,
            };

            let index = crate::scann::search::SCANNIndex::new(dimension, params)?;
            return Ok(AnyANNIndex::SCANN(index));
        }
    }

    Err(RetrieveError::Other(format!(
        "Unsupported index factory string: '{}'. Supported: HNSW{{m}}, NSW{{m}}, IVF{{n}},PQ{{m}}, SCANN{{n}}",
        factory_string
    )))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_factory() {
        #[cfg(feature = "hnsw")]
        {
            // Valid cases
            let index = index_factory(128, "HNSW32");
            assert!(index.is_ok());

            let index = index_factory(128, "HNSW16,32");
            assert!(index.is_ok());

            // Edge cases
            let index = index_factory(0, "HNSW32");
            assert!(index.is_err());

            let index = index_factory(128, "HNSW0");
            assert!(index.is_err());

            let index = index_factory(128, "HNSW");
            assert!(index.is_err());

            let index = index_factory(128, "HNSWabc");
            assert!(index.is_err());
        }
    }

    #[test]
    fn test_nsw_factory() {
        #[cfg(feature = "nsw")]
        {
            let index = index_factory(128, "NSW32");
            assert!(index.is_ok());

            // Edge cases
            let index = index_factory(128, "NSW0");
            assert!(index.is_err());

            let index = index_factory(128, "NSW");
            assert!(index.is_err());
        }
    }

    #[test]
    fn test_ivf_pq_factory() {
        #[cfg(feature = "ivf_pq")]
        {
            // Valid cases
            let index = index_factory(128, "IVF1024,PQ8");
            assert!(index.is_ok());

            let index = index_factory(128, "IVF1024,PQ8x8");
            assert!(index.is_ok());

            // Edge cases
            let index = index_factory(128, "IVF0,PQ8");
            assert!(index.is_err());

            let index = index_factory(128, "IVF1024,PQ0");
            assert!(index.is_err());

            let index = index_factory(128, "IVF,PQ8");
            assert!(index.is_err());

            let index = index_factory(128, "IVF1024,PQ");
            assert!(index.is_err());

            let index = index_factory(128, "IVF1024");
            assert!(index.is_err());

            // Dimension not divisible by codebooks
            let index = index_factory(100, "IVF1024,PQ8"); // 100 % 8 != 0
            assert!(index.is_err());
        }
    }

    #[test]
    fn test_scann_factory() {
        #[cfg(feature = "scann")]
        {
            let index = index_factory(128, "SCANN256");
            assert!(index.is_ok());

            // Edge cases
            let index = index_factory(128, "SCANN0");
            assert!(index.is_err());

            let index = index_factory(128, "SCANN");
            assert!(index.is_err());
        }
    }

    #[test]
    fn test_invalid_factory() {
        // Invalid index type
        let result = index_factory(128, "Invalid");
        assert!(result.is_err());

        // Empty string
        let result = index_factory(128, "");
        assert!(result.is_err());

        // Whitespace only
        let result = index_factory(128, "   ");
        assert!(result.is_err());

        // Zero dimension
        let result = index_factory(0, "HNSW32");
        assert!(result.is_err());
    }

    #[test]
    fn test_factory_usage() {
        #[cfg(feature = "hnsw")]
        {
            let mut index = index_factory(128, "HNSW32").unwrap();

            // Add vectors
            for i in 0..10 {
                let vec = vec![0.1; 128];
                assert!(index.add(i, vec).is_ok());
            }

            // Build
            assert!(index.build().is_ok());

            // Search
            let query = vec![0.15; 128];
            let results = index.search(&query, 5);
            assert!(results.is_ok());
            let results = results.unwrap();
            assert!(!results.is_empty());
        }
    }

    #[test]
    fn test_factory_whitespace_handling() {
        #[cfg(feature = "hnsw")]
        {
            // Should handle whitespace
            let index1 = index_factory(128, "HNSW32");
            let index2 = index_factory(128, "  HNSW32  ");
            assert_eq!(index1.is_ok(), index2.is_ok());
        }
    }
}
