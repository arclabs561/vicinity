//! LEMUR: Late-interaction retrieval via learned single-vector embeddings.
//!
//! Converts multi-vector (ColBERT-style) MaxSim retrieval into single-vector
//! MIPS by training a small MLP to approximate MaxSim contributions.
//!
//! # Status: inference-only stub
//!
//! Three things are missing before this module produces useful results on
//! a real workload:
//!
//! 1. **No in-tree training path.** `LemurIndex::new` requires a pre-trained
//!    [`LemurEncoder`]; there is no training loop, no loss computation, and
//!    no integration with any ML framework. Random encoder weights produce
//!    random retrieval. Bring your own externally-trained encoder weights
//!    (the hidden `LemurEncoder::random` constructor exists only for tests
//!    and explicit `lemur-fixtures` builds, not for production indexing).
//! 2. **Mean-pool substitutes for OLS.** Document weight vectors are computed
//!    as the mean of `psi`-encoded tokens, not the full OLS solve described
//!    in the paper (which requires a shared feature matrix Z across all
//!    documents). This is a known approximation-quality regression on top
//!    of the training gap.
//! 3. **Brute-force MIPS.** Search scans all document weights linearly. For
//!    any corpus above a few thousand documents, wire an HNSW index over the
//!    weight vectors externally.
//!
//! In short: this module is a working inference scaffold that needs an
//! external encoder, an external HNSW for scale, and either the OLS solve
//! or acceptance of the mean-pool approximation. Treat the API as a
//! reference implementation, not a turn-key index.
//!
//! # Algorithm
//!
//! 1. **Offline**: Train a two-layer MLP `psi(x) = LayerNorm(GELU(W'x + b))`
//!    that maps token embeddings to a latent space.
//! 2. **Indexing**: For each document, compute a single weight vector. Current
//!    code mean-pools `psi`-encoded tokens; the full OLS solve is the target
//!    implementation once the shared feature matrix is available.
//! 3. **Search**: Encode query tokens through `psi`, pool (sum), then MIPS
//!    against document weight vectors. Rerank top candidates with exact MaxSim.
//!
//! # References
//!
//! - Kulkarni et al. (2025). "LEMUR: Improving Single-Vector Retrieval with
//!   Learned Multi-Vector Retrieval." arXiv 2601.21853.

mod model;

pub use model::LemurEncoder;

use crate::RetrieveError;

/// LEMUR index for single-vector retrieval with MaxSim reranking.
pub struct LemurIndex {
    /// Dimension of the input token embeddings.
    input_dim: usize,
    /// Hidden dimension of the MLP (d').
    hidden_dim: usize,
    /// The MLP encoder (psi).
    encoder: LemurEncoder,
    /// Per-document weight vectors, each of length hidden_dim.
    doc_weights: Vec<Vec<f32>>,
    /// Per-document token sets for exact MaxSim reranking.
    /// Stored flat: doc_tokens[i] is a Vec of token embeddings for document i.
    doc_tokens: Vec<Vec<Vec<f32>>>,
    /// Document IDs.
    doc_ids: Vec<u32>,
}

impl LemurIndex {
    /// Create a LEMUR index from pre-trained encoder weights.
    ///
    /// # Arguments
    /// * `encoder` - Pre-trained MLP encoder
    pub fn new(encoder: LemurEncoder) -> Self {
        let input_dim = encoder.input_dim();
        let hidden_dim = encoder.hidden_dim();
        Self {
            input_dim,
            hidden_dim,
            encoder,
            doc_weights: Vec::new(),
            doc_tokens: Vec::new(),
            doc_ids: Vec::new(),
        }
    }

    /// Add a document with its token embeddings.
    ///
    /// Computes the single-vector MIPS embedding by mean-pooling psi-encoded tokens.
    pub fn add_document(
        &mut self,
        doc_id: u32,
        tokens: Vec<Vec<f32>>,
    ) -> Result<(), RetrieveError> {
        if tokens.is_empty() {
            return Err(RetrieveError::InvalidParameter(
                "document must have at least one token".into(),
            ));
        }
        if tokens[0].len() != self.input_dim {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: tokens[0].len(),
                doc_dim: self.input_dim,
            });
        }

        // Encode tokens through psi
        let encoded: Vec<Vec<f32>> = tokens.iter().map(|t| self.encoder.forward(t)).collect();

        // Compute document weight vector: mean of encoded tokens.
        // (Simplified from OLS -- full OLS requires a shared feature matrix Z.)
        let n = encoded.len() as f32;
        let mut weight = vec![0.0f32; self.hidden_dim];
        for enc in &encoded {
            for (w, e) in weight.iter_mut().zip(enc.iter()) {
                *w += e;
            }
        }
        for w in &mut weight {
            *w /= n;
        }

        self.doc_weights.push(weight);
        self.doc_tokens.push(tokens);
        self.doc_ids.push(doc_id);
        Ok(())
    }

    /// Search: encode query tokens, MIPS against doc weights, rerank with MaxSim.
    pub fn search(
        &self,
        query_tokens: &[Vec<f32>],
        k: usize,
        rerank_pool: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if query_tokens.is_empty() {
            return Err(RetrieveError::EmptyQuery);
        }

        // Encode and pool query
        let query_vec = self.encode_query(query_tokens);

        // MIPS: brute-force over doc weights
        let pool = rerank_pool.max(k);
        let mut scores: Vec<(usize, f32)> = self
            .doc_weights
            .iter()
            .enumerate()
            .map(|(i, w)| (i, crate::simd::dot(&query_vec, w)))
            .collect();
        scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1)); // descending (MIPS)
        scores.truncate(pool);

        // Rerank with exact MaxSim
        let mut reranked: Vec<(u32, f32)> = scores
            .into_iter()
            .map(|(idx, _mips_score)| {
                let maxsim = self.maxsim(query_tokens, &self.doc_tokens[idx]);
                (self.doc_ids[idx], maxsim)
            })
            .collect();
        reranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1)); // descending
        reranked.truncate(k);
        Ok(reranked)
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    // ── internal ──────────────────────────────────────────────────────────

    /// Encode and pool query tokens: sum of psi(x_i).
    fn encode_query(&self, tokens: &[Vec<f32>]) -> Vec<f32> {
        let mut pooled = vec![0.0f32; self.hidden_dim];
        for token in tokens {
            let enc = self.encoder.forward(token);
            for (p, e) in pooled.iter_mut().zip(enc.iter()) {
                *p += e;
            }
        }
        pooled
    }

    /// Exact MaxSim: for each query token, find max dot product with any doc token.
    fn maxsim(&self, query_tokens: &[Vec<f32>], doc_tokens: &[Vec<f32>]) -> f32 {
        let mut total = 0.0f32;
        for qt in query_tokens {
            let mut max_sim = f32::NEG_INFINITY;
            for dt in doc_tokens {
                let sim = crate::simd::dot(qt, dt);
                if sim > max_sim {
                    max_sim = sim;
                }
            }
            total += max_sim;
        }
        total
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, deprecated)]
mod tests {
    use super::*;

    #[test]
    fn add_and_search() {
        let enc = LemurEncoder::random(32, 64, 42);
        let mut index = LemurIndex::new(enc);

        // Add 10 documents, each with 5 tokens
        for doc_id in 0..10u32 {
            let tokens: Vec<Vec<f32>> = (0..5)
                .map(|t| {
                    let seed = (doc_id as f32 * 0.1 + t as f32 * 0.01) * 0.1;
                    vec![seed; 32]
                })
                .collect();
            index.add_document(doc_id, tokens).unwrap();
        }
        assert_eq!(index.len(), 10);

        // Search with the first document's tokens (should find itself)
        let query_tokens: Vec<Vec<f32>> = (0..5).map(|t| vec![t as f32 * 0.01; 32]).collect();
        let results = index.search(&query_tokens, 5, 10).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn empty_query_returns_error() {
        let enc = LemurEncoder::random(32, 64, 42);
        let index = LemurIndex::new(enc);
        let empty: Vec<Vec<f32>> = vec![];
        assert!(index.search(&empty, 5, 10).is_err());
    }
}
