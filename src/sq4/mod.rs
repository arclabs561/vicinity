//! SQ4: 4-bit scalar quantization for flat-scan ANN.
//!
//! Quantizes each vector dimension to 4 bits (16 levels) using per-dimension
//! min/max scaling. Two codes are packed per byte (low nibble, high nibble),
//! giving 8x compression over f32.
//!
//! Search is asymmetric: the query stays in full float32 while database vectors
//! are decoded on-the-fly from 4-bit codes for distance computation. A rerank
//! pass with exact distance refines the top candidates.
//!
//! # When to Use
//!
//! - Memory-constrained scenarios where SQ8 (4x compression) is not enough
//! - As a first-stage filter before exact reranking
//! - Datasets where dimension distributions are relatively uniform (per-dim
//!   scaling handles heterogeneous distributions)
//!
//! For higher recall, prefer SQ8 (`rp_quant`). For more compression, use
//! binary quantization (`binary_index`).

use crate::distance::cosine_distance_normalized;
use crate::RetrieveError;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// SQ4 index parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SQ4Params {
    /// Candidate multiplier for re-ranking: fetch `k * rerank_factor` candidates
    /// from quantized scan, then re-rank with exact distance. Default: 10.
    pub rerank_factor: usize,
}

impl Default for SQ4Params {
    fn default() -> Self {
        Self { rerank_factor: 10 }
    }
}

/// Flat-scan index with 4-bit scalar quantization.
pub struct SQ4Index {
    dimension: usize,
    params: SQ4Params,
    built: bool,

    /// Original vectors for re-ranking, stored flat (n * d).
    vectors: Vec<f32>,
    num_vectors: usize,
    doc_ids: Vec<u32>,

    /// Packed 4-bit codes, `n * ceil(d/2)` bytes.
    /// Two dimensions per byte: dims[2i] in low nibble, dims[2i+1] in high nibble.
    codes: Vec<u8>,

    /// Per-dimension minimum values (length d).
    mins: Vec<f32>,

    /// Per-dimension inverse scale: `15.0 / (max - min)` (length d).
    /// Used for encoding. For decoding: `val = min + code * (max - min) / 15`.
    inv_scales: Vec<f32>,

    /// Per-dimension step size: `(max - min) / 15` (length d).
    /// Used for decoding during search.
    steps: Vec<f32>,
}

impl SQ4Index {
    /// Create a new SQ4 index.
    pub fn new(dimension: usize, params: SQ4Params) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be > 0".into(),
            ));
        }
        if params.rerank_factor == 0 {
            return Err(RetrieveError::InvalidParameter(
                "rerank_factor must be > 0".into(),
            ));
        }
        Ok(Self {
            dimension,
            params,
            built: false,
            vectors: Vec::new(),
            num_vectors: 0,
            doc_ids: Vec::new(),
            codes: Vec::new(),
            mins: Vec::new(),
            inv_scales: Vec::new(),
            steps: Vec::new(),
        })
    }

    /// Add a vector. Input is L2-normalized internally.
    pub fn add_slice(&mut self, doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
        if self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot add after build".into(),
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

    /// Batch add vectors.
    pub fn add_batch(&mut self, ids: &[u32], vectors: &[f32]) -> Result<(), RetrieveError> {
        if vectors.len() != ids.len() * self.dimension {
            return Err(RetrieveError::InvalidParameter(
                "vectors.len() must equal ids.len() * dimension".into(),
            ));
        }
        for (i, &id) in ids.iter().enumerate() {
            self.add_slice(id, &vectors[i * self.dimension..(i + 1) * self.dimension])?;
        }
        Ok(())
    }

    /// Build the index: compute per-dimension statistics and quantize all vectors.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        let d = self.dimension;
        let n = self.num_vectors;

        // Compute per-dimension min and max.
        let mut mins = vec![f32::INFINITY; d];
        let mut maxs = vec![f32::NEG_INFINITY; d];
        for i in 0..n {
            let v = &self.vectors[i * d..(i + 1) * d];
            for (j, &val) in v.iter().enumerate() {
                if val < mins[j] {
                    mins[j] = val;
                }
                if val > maxs[j] {
                    maxs[j] = val;
                }
            }
        }

        let mut inv_scales = vec![0.0f32; d];
        let mut steps = vec![0.0f32; d];
        for j in 0..d {
            let range = maxs[j] - mins[j];
            if range > 1e-10 {
                inv_scales[j] = 15.0 / range;
                steps[j] = range / 15.0;
            }
        }

        // Quantize and pack: two 4-bit codes per byte.
        let bytes_per_vec = d.div_ceil(2);
        let mut codes = vec![0u8; n * bytes_per_vec];
        for i in 0..n {
            let v = &self.vectors[i * d..(i + 1) * d];
            let c = &mut codes[i * bytes_per_vec..(i + 1) * bytes_per_vec];
            pack_vector(v, &mins, &inv_scales, c);
        }

        self.mins = mins;
        self.inv_scales = inv_scales;
        self.steps = steps;
        self.codes = codes;
        self.built = true;
        Ok(())
    }

    /// Save a built SQ4 index to a directory snapshot.
    ///
    /// The persisted vectors are already normalized by `add_slice`. Codes and
    /// per-dimension scales are rebuilt on load because they are deterministic
    /// derived state.
    pub fn save_to_dir(&self, output_dir: impl AsRef<Path>) -> Result<(), RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter("index not built".into()));
        }
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;
        let manifest = Sq4SnapshotManifest {
            magic: *b"VICSQ401",
            version: 1,
            dimension: self.dimension,
            num_vectors: self.num_vectors,
            params: self.params.clone(),
        };
        crate::graph_snapshot::write_json_atomic(&output_dir.join("manifest.json"), &manifest)?;
        crate::graph_snapshot::write_f32_atomic(&output_dir.join("vectors.bin"), &self.vectors)?;
        crate::graph_snapshot::write_u32_atomic(&output_dir.join("doc_ids.bin"), &self.doc_ids)?;
        Ok(())
    }

    /// Load an SQ4 index from a directory snapshot.
    pub fn load_from_dir(input_dir: impl AsRef<Path>) -> Result<Self, RetrieveError> {
        let input_dir = input_dir.as_ref();
        let manifest: Sq4SnapshotManifest =
            crate::graph_snapshot::read_json(&input_dir.join("manifest.json"))?;
        if manifest.magic != *b"VICSQ401" {
            return Err(RetrieveError::FormatError(
                "invalid SQ4 snapshot magic".into(),
            ));
        }
        if manifest.version != 1 {
            return Err(RetrieveError::FormatError(format!(
                "unsupported SQ4 snapshot version {}",
                manifest.version
            )));
        }
        if manifest.dimension == 0 {
            return Err(RetrieveError::FormatError(
                "SQ4 snapshot has zero dimension".into(),
            ));
        }

        let vector_len = manifest
            .num_vectors
            .checked_mul(manifest.dimension)
            .ok_or_else(|| RetrieveError::FormatError("SQ4 vector length overflow".into()))?;
        let vectors =
            crate::graph_snapshot::read_f32_exact(&input_dir.join("vectors.bin"), vector_len)?;
        let doc_ids = crate::graph_snapshot::read_u32_exact(
            &input_dir.join("doc_ids.bin"),
            manifest.num_vectors,
        )?;

        let mut index = Self::new(manifest.dimension, manifest.params)?;
        index.vectors = vectors;
        index.doc_ids = doc_ids;
        index.num_vectors = manifest.num_vectors;
        index.build()?;
        Ok(index)
    }

    /// Search for the `k` nearest neighbors.
    ///
    /// Performs asymmetric distance computation (float query vs 4-bit codes),
    /// then re-ranks top candidates with exact cosine distance.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "index must be built before search".into(),
            ));
        }
        if query.is_empty() {
            return Err(RetrieveError::EmptyQuery);
        }
        if query.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.dimension,
            });
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        let d = self.dimension;
        let n = self.num_vectors;
        let bytes_per_vec = d.div_ceil(2);

        // Normalize query.
        let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let query_norm: Vec<f32> = if norm > 1e-10 {
            query.iter().map(|x| x / norm).collect()
        } else {
            query.to_vec()
        };

        // Asymmetric L2 scan: decode 4-bit codes on the fly, compute distance to float query.
        let candidates = k * self.params.rerank_factor;
        let candidates = candidates.min(n);

        let mut dists: Vec<(usize, f32)> = Vec::with_capacity(n);
        for i in 0..n {
            let code = &self.codes[i * bytes_per_vec..(i + 1) * bytes_per_vec];
            let dist = asymmetric_l2_sq4(&query_norm, code, &self.mins, &self.steps, d);
            dists.push((i, dist));
        }

        // Partial sort to get top candidates.
        if candidates < n {
            dists.select_nth_unstable_by(candidates, |a, b| a.1.total_cmp(&b.1));
            dists.truncate(candidates);
        }

        // Re-rank with exact cosine distance.
        let mut results: Vec<(u32, f32)> = dists
            .iter()
            .map(|&(idx, _)| {
                let v = &self.vectors[idx * d..(idx + 1) * d];
                let dist = cosine_distance_normalized(&query_norm, v);
                (self.doc_ids[idx], dist)
            })
            .collect();

        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        results.truncate(k);
        Ok(results)
    }

    /// Number of indexed vectors.
    #[must_use]
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    /// Memory used by quantized codes in bytes.
    #[must_use]
    pub fn code_memory(&self) -> usize {
        self.codes.len()
    }

    /// Estimated heap memory used by this index.
    #[must_use]
    pub fn memory_usage(&self) -> crate::memory::MemoryReport {
        let f32_bytes = std::mem::size_of::<f32>();
        let vectors_bytes = self.vectors.capacity() * f32_bytes;
        let quantized_bytes = self.codes.capacity();
        let metadata_bytes = self.doc_ids.capacity() * std::mem::size_of::<u32>()
            + self.mins.capacity() * f32_bytes
            + self.inv_scales.capacity() * f32_bytes
            + self.steps.capacity() * f32_bytes;

        crate::memory::MemoryReport {
            vectors_bytes,
            graph_bytes: 0,
            quantized_bytes,
            metadata_bytes,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Sq4SnapshotManifest {
    magic: [u8; 8],
    version: u32,
    dimension: usize,
    num_vectors: usize,
    params: SQ4Params,
}

/// Pack a float vector into 4-bit codes, two per byte.
/// Low nibble = even dimension, high nibble = odd dimension.
pub(crate) fn pack_vector(v: &[f32], mins: &[f32], inv_scales: &[f32], out: &mut [u8]) {
    let d = v.len();
    let pairs = d / 2;
    for p in 0..pairs {
        let lo = quantize_4bit(v[2 * p], mins[2 * p], inv_scales[2 * p]);
        let hi = quantize_4bit(v[2 * p + 1], mins[2 * p + 1], inv_scales[2 * p + 1]);
        out[p] = lo | (hi << 4);
    }
    // Handle odd dimension.
    if !d.is_multiple_of(2) {
        out[pairs] = quantize_4bit(v[d - 1], mins[d - 1], inv_scales[d - 1]);
    }
}

/// Quantize a single float to 4 bits [0, 15].
#[inline]
pub(crate) fn quantize_4bit(val: f32, min: f32, inv_scale: f32) -> u8 {
    let q = ((val - min) * inv_scale + 0.5) as i32; // round to nearest
    q.clamp(0, 15) as u8
}

/// Asymmetric L2 distance: float query vs 4-bit packed codes.
/// Decodes on the fly and accumulates squared differences.
fn asymmetric_l2_sq4(query: &[f32], code: &[u8], mins: &[f32], steps: &[f32], dim: usize) -> f32 {
    let mut sum = 0.0f32;
    let pairs = dim / 2;

    for p in 0..pairs {
        let byte = code[p];
        let lo = (byte & 0x0F) as f32;
        let hi = (byte >> 4) as f32;

        let decoded_lo = mins[2 * p] + lo * steps[2 * p];
        let decoded_hi = mins[2 * p + 1] + hi * steps[2 * p + 1];

        let d0 = query[2 * p] - decoded_lo;
        let d1 = query[2 * p + 1] - decoded_hi;
        sum += d0 * d0 + d1 * d1;
    }

    // Handle odd dimension.
    if !dim.is_multiple_of(2) {
        let lo = (code[pairs] & 0x0F) as f32;
        let decoded = mins[dim - 1] + lo * steps[dim - 1];
        let d = query[dim - 1] - decoded;
        sum += d * d;
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        let d = 8;
        let v: Vec<f32> = vec![0.0, 0.5, 0.25, 0.75, 1.0, 0.1, 0.9, 0.6];
        let mins = vec![0.0f32; d];
        let inv_scales = vec![15.0f32; d]; // range [0, 1]
        let steps = vec![1.0 / 15.0; d];

        let mut packed = vec![0u8; d / 2];
        pack_vector(&v, &mins, &inv_scales, &mut packed);

        // Decode and check each value is close (within 1/15 = 0.067).
        for p in 0..d / 2 {
            let lo = (packed[p] & 0x0F) as f32;
            let hi = (packed[p] >> 4) as f32;
            let decoded_lo = mins[2 * p] + lo * steps[2 * p];
            let decoded_hi = mins[2 * p + 1] + hi * steps[2 * p + 1];
            assert!(
                (decoded_lo - v[2 * p]).abs() < 0.07,
                "dim {}: {} vs {}",
                2 * p,
                decoded_lo,
                v[2 * p]
            );
            assert!(
                (decoded_hi - v[2 * p + 1]).abs() < 0.07,
                "dim {}: {} vs {}",
                2 * p + 1,
                decoded_hi,
                v[2 * p + 1]
            );
        }
    }

    #[test]
    fn odd_dimension() {
        let d = 5;
        let v: Vec<f32> = vec![0.0, 0.5, 0.25, 0.75, 1.0];
        let mins = vec![0.0f32; d];
        let inv_scales = vec![15.0f32; d];

        let mut packed = vec![0u8; d.div_ceil(2)];
        pack_vector(&v, &mins, &inv_scales, &mut packed);

        // Last byte should only have low nibble set.
        assert_eq!(packed[2] & 0xF0, 0);
        assert_eq!(packed[2] & 0x0F, 15); // 1.0 * 15 = 15
    }

    #[test]
    fn build_and_search() {
        let d = 32;
        let n = 100;

        let mut index = SQ4Index::new(d, SQ4Params::default()).unwrap();

        // Generate simple normalized vectors.
        for i in 0..n {
            let v: Vec<f32> = (0..d).map(|j| ((i * d + j) as f32) * 0.01).collect();
            index.add_slice(i as u32, &v).unwrap();
        }
        index.build().unwrap();

        // Search for the first vector.
        let query: Vec<f32> = (0..d).map(|j| (j as f32) * 0.01).collect();
        let results = index.search(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
        // Self should be the closest (doc_id = 0).
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn memory_usage_reports_owned_buffers_after_build() {
        let d = 16;
        let n = 32;
        let mut index = SQ4Index::new(d, SQ4Params::default()).unwrap();

        for i in 0..n {
            let v: Vec<f32> = (0..d).map(|j| ((i + j) as f32) * 0.01).collect();
            index.add_slice(i as u32, &v).unwrap();
        }
        index.build().unwrap();

        let report = index.memory_usage();
        assert!(report.vectors_bytes >= n * d * std::mem::size_of::<f32>());
        assert!(report.quantized_bytes >= n * d.div_ceil(2));
        assert!(report.metadata_bytes >= n * std::mem::size_of::<u32>());
        assert!(report.total() >= report.vectors_bytes + report.quantized_bytes);
    }

    #[test]
    fn save_load_roundtrip_preserves_search() {
        let d = 32;
        let n = 80;
        let mut index = SQ4Index::new(d, SQ4Params::default()).unwrap();
        for i in 0..n {
            let v: Vec<f32> = (0..d).map(|j| ((i + j) as f32) * 0.001).collect();
            index.add_slice(10_000 + i as u32, &v).unwrap();
        }
        index.build().unwrap();

        let query: Vec<f32> = (0..d).map(|j| (j as f32) * 0.001).collect();
        let before = index.search(&query, 8).unwrap();
        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let loaded = SQ4Index::load_from_dir(dir.path()).unwrap();
        let after = loaded.search(&query, 8).unwrap();

        assert_eq!(after, before);
        assert!(after.iter().all(|(id, _)| *id >= 10_000));
    }

    #[test]
    fn compression_ratio() {
        let d = 128;
        let n = 1000;
        let mut index = SQ4Index::new(d, SQ4Params::default()).unwrap();

        for i in 0..n {
            let v: Vec<f32> = (0..d).map(|j| ((i + j) as f32) * 0.001).collect();
            index.add_slice(i as u32, &v).unwrap();
        }
        index.build().unwrap();

        let float_bytes = n * d * 4;
        let code_bytes = index.code_memory();
        let ratio = float_bytes as f64 / code_bytes as f64;

        // 4-bit = d/2 bytes per vector vs d*4 bytes = 8x compression.
        assert!(ratio > 7.5, "expected ~8x compression, got {ratio:.1}x");
    }
}
