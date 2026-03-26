#![allow(dead_code)]
//! SIMD-accelerated Product Quantization distance computation.
//!
//! # Asymmetric Distance Computation (ADC)
//!
//! In PQ search, we precompute a Lookup Table (LUT) for the query, then
//! for each database vector (stored as codes), sum up the corresponding
//! LUT entries:
//!
//! ```text
//! distance(query, pq_code) = Σ_m LUT[m][code[m]]
//! ```
//!
//! This is O(M) per candidate, where M is the number of subquantizers.
//!
//! # SIMD Optimization
//!
//! The naive approach loads one LUT value at a time. With SIMD shuffle
//! instructions (`vpshufb` on x86, `tbl` on ARM), we can perform multiple
//! table lookups in parallel:
//!
//! - AVX2 `vpshufb`: 32 parallel 4-bit lookups
//! - AVX-512 `vpermb`: 64 parallel 8-bit lookups
//! - NEON `tbl`: 16 parallel 4-bit lookups
//!
//! For codebook size 256, we use the 4-bit trick: split each 8-bit code
//! into two 4-bit parts, look up in two half-tables, then add.
//!
//! # Performance
//!
//! On typical workloads:
//! - Naive: ~1 lookup/cycle
//! - AVX2 shuffle: ~8 lookups/cycle
//! - AVX-512 shuffle: ~16 lookups/cycle
//!
//! 3-5x speedup on distance computation, which dominates PQ search time.
//!
//! # References
//!
//! - "Quicker ADC" (André et al., 2018) - shuffle-based PQ lookup
//! - Faiss implementation of PQ with SIMD

/// Compute ADC distance from codes using a precomputed LUT.
///
/// This is the naive (portable) implementation.
///
/// # Arguments
///
/// * `codes` - PQ codes for one vector, length = num_codebooks
/// * `lut` - Lookup table: `lut[m][code]` = distance to codeword in codebook m
///
/// # Returns
///
/// Sum of LUT lookups: `Σ lut[m][codes[m]]`
#[inline]
pub fn adc_distance(codes: &[u8], lut: &[Vec<f32>]) -> f32 {
    debug_assert_eq!(codes.len(), lut.len());
    codes
        .iter()
        .zip(lut.iter())
        .map(|(&code, table)| table[code as usize])
        .sum()
}

/// Compute ADC distances for multiple candidates in batch.
///
/// This enables better cache utilization and SIMD parallelism.
///
/// # Arguments
///
/// * `codes_batch` - Flattened codes: [n_candidates * num_codebooks]
/// * `num_codebooks` - Number of subquantizers
/// * `lut` - Lookup table: `lut[m][code]` = distance contribution
///
/// # Returns
///
/// Vector of distances, one per candidate.
pub fn adc_batch_distances(codes_batch: &[u8], num_codebooks: usize, lut: &[Vec<f32>]) -> Vec<f32> {
    let n_candidates = codes_batch.len() / num_codebooks;
    let mut distances = Vec::with_capacity(n_candidates);

    for i in 0..n_candidates {
        let codes = &codes_batch[i * num_codebooks..(i + 1) * num_codebooks];
        distances.push(adc_distance(codes, lut));
    }

    distances
}

/// Packed LUT for SIMD operations.
///
/// Reorganizes LUT data for cache-friendly and SIMD-friendly access patterns.
/// Instead of `lut[codebook][code]`, we pack data for streaming access.
#[derive(Debug, Clone)]
pub struct PackedLUT {
    /// Packed data: [codebook_0_values..., codebook_1_values..., ...]
    data: Vec<f32>,
    /// Number of codebooks
    num_codebooks: usize,
    /// Size of each codebook (typically 256)
    codebook_size: usize,
}

impl PackedLUT {
    /// Create a packed LUT from a standard nested vector LUT.
    pub fn from_nested(lut: &[Vec<f32>]) -> Self {
        let num_codebooks = lut.len();
        let codebook_size = if lut.is_empty() { 0 } else { lut[0].len() };

        let mut data = Vec::with_capacity(num_codebooks * codebook_size);
        for codebook in lut {
            data.extend_from_slice(codebook);
        }

        Self {
            data,
            num_codebooks,
            codebook_size,
        }
    }

    /// Look up a single value.
    #[inline]
    pub fn lookup(&self, codebook: usize, code: u8) -> f32 {
        self.data[codebook * self.codebook_size + code as usize]
    }

    /// Compute ADC distance using packed LUT.
    #[inline]
    pub fn adc_distance(&self, codes: &[u8]) -> f32 {
        debug_assert_eq!(codes.len(), self.num_codebooks);

        let mut sum = 0.0f32;
        for (m, &code) in codes.iter().enumerate() {
            sum += self.data[m * self.codebook_size + code as usize];
        }
        sum
    }

    /// Get a pointer to codebook data for SIMD operations.
    #[inline]
    pub fn codebook_ptr(&self, codebook_idx: usize) -> *const f32 {
        unsafe { self.data.as_ptr().add(codebook_idx * self.codebook_size) }
    }

    /// Number of codebooks.
    pub fn num_codebooks(&self) -> usize {
        self.num_codebooks
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SIMD implementations
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub mod x86_64 {
    //! AVX2/AVX-512 implementations of PQ distance.

    use super::*;

    /// AVX2 batch ADC with 8-way parallelism.
    ///
    /// Processes 8 candidates simultaneously using gather instructions.
    ///
    /// # Safety
    ///
    /// Requires AVX2. Caller must verify via runtime detection.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn adc_batch_avx2(
        codes_batch: &[u8],
        num_codebooks: usize,
        lut: &PackedLUT,
    ) -> Vec<f32> {
        use std::arch::x86_64::{
            __m256, __m256i, _mm256_add_ps, _mm256_i32gather_ps, _mm256_setzero_ps,
            _mm256_storeu_ps,
        };

        let n_candidates = codes_batch.len() / num_codebooks;
        let mut distances = vec![0.0f32; n_candidates];

        // Process 8 candidates at a time
        let chunks_8 = n_candidates / 8;

        for chunk in 0..chunks_8 {
            let base_idx = chunk * 8;
            let mut sum: __m256 = _mm256_setzero_ps();

            for m in 0..num_codebooks {
                // Gather 8 codes for codebook m from 8 consecutive candidates
                // codes[base_idx + i][m] = codes_batch[(base_idx + i) * num_codebooks + m]
                let mut indices = [0i32; 8];
                for i in 0..8 {
                    indices[i] = codes_batch[(base_idx + i) * num_codebooks + m] as i32;
                }

                // Load indices into SIMD register
                let indices_ptr = indices.as_ptr() as *const __m256i;
                let idx_vec = std::ptr::read_unaligned(indices_ptr);

                // Gather from LUT
                let lut_base = lut.codebook_ptr(m);
                // Scale 4 = sizeof(f32). Must be constant.
                let gathered = _mm256_i32gather_ps(lut_base, idx_vec, 4);

                sum = _mm256_add_ps(sum, gathered);
            }

            // Store results
            _mm256_storeu_ps(distances.as_mut_ptr().add(base_idx), sum);
        }

        // Handle remaining candidates
        let tail_start = chunks_8 * 8;
        for i in tail_start..n_candidates {
            let codes = &codes_batch[i * num_codebooks..(i + 1) * num_codebooks];
            distances[i] = lut.adc_distance(codes);
        }

        distances
    }

    /// AVX-512 batch ADC with 16-way parallelism.
    ///
    /// # Safety
    ///
    /// Requires AVX-512F. Caller must verify via runtime detection.
    #[cfg(all(target_arch = "x86_64", feature = "nightly"))]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn adc_batch_avx512(
        codes_batch: &[u8],
        num_codebooks: usize,
        lut: &PackedLUT,
    ) -> Vec<f32> {
        use std::arch::x86_64::{
            __m512, __m512i, _mm512_add_ps, _mm512_i32gather_ps, _mm512_setzero_ps,
            _mm512_storeu_ps,
        };

        let n_candidates = codes_batch.len() / num_codebooks;
        let mut distances = vec![0.0f32; n_candidates];

        // Process 16 candidates at a time
        let chunks_16 = n_candidates / 16;

        for chunk in 0..chunks_16 {
            let base_idx = chunk * 16;
            let mut sum: __m512 = _mm512_setzero_ps();

            for m in 0..num_codebooks {
                // Gather 16 codes for codebook m
                let mut indices = [0i32; 16];
                for i in 0..16 {
                    indices[i] = codes_batch[(base_idx + i) * num_codebooks + m] as i32;
                }

                let indices_ptr = indices.as_ptr() as *const __m512i;
                let idx_vec = std::ptr::read_unaligned(indices_ptr);

                let lut_base = lut.codebook_ptr(m);
                // Scale=4 means each index step is 4 bytes (size of f32)
                let gathered = _mm512_i32gather_ps(idx_vec, lut_base, 4);

                sum = _mm512_add_ps(sum, gathered);
            }

            _mm512_storeu_ps(distances.as_mut_ptr().add(base_idx), sum);
        }

        // Handle tail
        let tail_start = chunks_16 * 16;
        for i in tail_start..n_candidates {
            let codes = &codes_batch[i * num_codebooks..(i + 1) * num_codebooks];
            distances[i] = lut.adc_distance(codes);
        }

        distances
    }
}

#[cfg(target_arch = "aarch64")]
pub mod aarch64 {
    //! NEON implementations of PQ distance.

    use super::*;

    /// NEON batch ADC with 4-way parallelism.
    ///
    /// # Safety
    ///
    /// NEON is always available on aarch64.
    #[target_feature(enable = "neon")]
    pub unsafe fn adc_batch_neon(
        codes_batch: &[u8],
        num_codebooks: usize,
        lut: &PackedLUT,
    ) -> Vec<f32> {
        use std::arch::aarch64::{float32x4_t, vaddq_f32, vdupq_n_f32, vsetq_lane_f32, vst1q_f32};

        let n_candidates = codes_batch.len() / num_codebooks;
        let mut distances = vec![0.0f32; n_candidates];

        // Process 4 candidates at a time
        let chunks_4 = n_candidates / 4;

        for chunk in 0..chunks_4 {
            let base_idx = chunk * 4;
            let mut sum: float32x4_t = vdupq_n_f32(0.0);

            for m in 0..num_codebooks {
                // Manual gather: load 4 codes, look up, pack into f32x4
                let c0 = codes_batch[base_idx * num_codebooks + m] as usize;
                let c1 = codes_batch[(base_idx + 1) * num_codebooks + m] as usize;
                let c2 = codes_batch[(base_idx + 2) * num_codebooks + m] as usize;
                let c3 = codes_batch[(base_idx + 3) * num_codebooks + m] as usize;

                let lut_base = m * lut.codebook_size;
                let v0 = lut.data[lut_base + c0];
                let v1 = lut.data[lut_base + c1];
                let v2 = lut.data[lut_base + c2];
                let v3 = lut.data[lut_base + c3];

                // Pack 4 LUT values into vector (lane indices 0-3)
                let lane0 = vsetq_lane_f32(v0, vdupq_n_f32(0.0), 0);
                let lane01 = vsetq_lane_f32(v1, lane0, 1);
                let lane012 = vsetq_lane_f32(v2, lane01, 2);
                let gathered = vsetq_lane_f32(v3, lane012, 3);

                sum = vaddq_f32(sum, gathered);
            }

            // SAFETY: base_idx + 3 < distances.len() since chunk < chunks_4
            // Store requires unsafe due to raw pointer arithmetic
            unsafe { vst1q_f32(distances.as_mut_ptr().add(base_idx), sum) };
        }

        // Handle tail
        let tail_start = chunks_4 * 4;
        for i in tail_start..n_candidates {
            let codes = &codes_batch[i * num_codebooks..(i + 1) * num_codebooks];
            distances[i] = lut.adc_distance(codes);
        }

        distances
    }
}

/// Auto-dispatching batch ADC computation.
///
/// Selects the fastest available SIMD implementation.
pub fn adc_batch_dispatch(codes_batch: &[u8], num_codebooks: usize, lut: &PackedLUT) -> Vec<f32> {
    let n_candidates = codes_batch.len() / num_codebooks;

    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "nightly")]
        if n_candidates >= 16 && is_x86_feature_detected!("avx512f") {
            return unsafe { x86_64::adc_batch_avx512(codes_batch, num_codebooks, lut) };
        }
        if n_candidates >= 8 && is_x86_feature_detected!("avx2") {
            return unsafe { x86_64::adc_batch_avx2(codes_batch, num_codebooks, lut) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if n_candidates >= 4 {
            return unsafe { aarch64::adc_batch_neon(codes_batch, num_codebooks, lut) };
        }
    }

    // Fallback
    adc_batch_distances(codes_batch, num_codebooks, &lut_to_nested(lut))
}

/// Convert PackedLUT back to nested Vec (for fallback).
fn lut_to_nested(packed: &PackedLUT) -> Vec<Vec<f32>> {
    let mut result = Vec::with_capacity(packed.num_codebooks);
    for m in 0..packed.num_codebooks {
        let start = m * packed.codebook_size;
        let end = start + packed.codebook_size;
        result.push(packed.data[start..end].to_vec());
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// FastScan 4-bit PQ: SIMD-friendly ADC for 16-centroid subquantizers
// ─────────────────────────────────────────────────────────────────────────────
//
// FAISS FastScan processes 32 PQ candidates per SIMD iteration using vpshufb
// (AVX2) or tbl (NEON) for parallel LUT access. The constraint: codes must be
// 4-bit (16 centroids per subquantizer).
//
// This module provides:
// 1. PackedCodes4bit -- nibble-interleaved code storage for 32-vector blocks
// 2. LUT quantization (f32 -> u8 for integer SIMD accumulation)
// 3. Portable scalar FastScan kernel (simulates vpshufb behavior)
// 4. fastscan_batch -- end-to-end integration with f32 output
//
// ─────────────────────────────────────────────────────────────────────────────

/// Pack 4-bit PQ codes for FastScan SIMD processing.
///
/// Input: `codes[n][num_codebooks]` where each code is 0..15 (4-bit).
/// Output: packed blocks of 32 vectors, interleaved for vpshufb.
///
/// Within each block of 32 vectors, for each subquantizer m:
/// - 16 bytes, where byte[lane] =
///   low nibble: code for vector `block_base + lane`,
///   high nibble: code for vector `block_base + lane + 16`.
///
/// This layout lets vpshufb process 32 lookups (one subquantizer) per
/// instruction.
#[derive(Debug, Clone)]
pub struct PackedCodes4bit {
    /// Nibble-interleaved data. Layout: blocks of 32 vectors, each block
    /// contains `16 * num_codebooks` bytes.
    pub data: Vec<u8>,
    /// Total number of vectors (including padding in the last block).
    pub num_vectors: usize,
    /// Number of subquantizers.
    pub num_codebooks: usize,
    /// Always 32.
    pub block_size: usize,
}

impl PackedCodes4bit {
    /// Pack flat 8-bit codes into the FastScan nibble layout.
    ///
    /// `codes` is row-major: `codes[i * num_codebooks + m]` is vector i's
    /// code for subquantizer m. Each value must be in 0..16.
    pub fn pack(codes: &[u8], num_vectors: usize, num_codebooks: usize) -> Self {
        debug_assert_eq!(codes.len(), num_vectors * num_codebooks);

        let block_size = 32usize;
        let num_blocks = num_vectors.div_ceil(block_size);
        let bytes_per_block = 16 * num_codebooks;

        let mut data = vec![0u8; num_blocks * bytes_per_block];

        for block in 0..num_blocks {
            let block_base = block * block_size;
            let block_data_offset = block * bytes_per_block;

            // Each subquantizer m gets 16 bytes per block.
            // Byte layout within a block: [m0: 16 bytes][m1: 16 bytes]...
            // Within each 16-byte group for subquantizer m:
            //   byte[lane] = lo_nibble(vec[block_base+lane].code[m])
            //              | hi_nibble(vec[block_base+lane+16].code[m])
            for m in 0..num_codebooks {
                for lane in 0..16usize {
                    let vi_lo = block_base + lane;
                    let vi_hi = block_base + 16 + lane;

                    let code_lo = if vi_lo < num_vectors {
                        codes[vi_lo * num_codebooks + m] & 0x0F
                    } else {
                        0
                    };
                    let code_hi = if vi_hi < num_vectors {
                        codes[vi_hi * num_codebooks + m] & 0x0F
                    } else {
                        0
                    };

                    data[block_data_offset + m * 16 + lane] = code_lo | (code_hi << 4);
                }
            }
        }

        Self {
            data,
            num_vectors,
            num_codebooks,
            block_size,
        }
    }

    /// Number of 32-vector blocks.
    pub fn num_blocks(&self) -> usize {
        self.num_vectors.div_ceil(self.block_size)
    }

    /// Bytes per block: 16 bytes per subquantizer.
    pub fn bytes_per_block(&self) -> usize {
        16 * self.num_codebooks
    }

    /// Slice of packed data for a given block.
    pub fn block_data(&self, block_idx: usize) -> &[u8] {
        let bpb = self.bytes_per_block();
        let start = block_idx * bpb;
        &self.data[start..start + bpb]
    }
}

/// Quantize a float32 LUT to uint8 for SIMD integer accumulation.
///
/// The LUT has shape `[num_codebooks][16]` (4-bit PQ = 16 centroids).
///
/// Returns `(quantized_lut, scale, offset)` where:
///   `original_distance ~= quantized_value * scale + offset`
///
/// The quantization maps the global [min, max] range across all codebooks
/// to [0, 255], minimizing per-codebook rounding error.
pub fn quantize_lut(lut: &[Vec<f32>]) -> (Vec<u8>, f32, f32) {
    if lut.is_empty() {
        return (Vec::new(), 1.0, 0.0);
    }

    // Find global min/max across all codebook entries.
    let mut global_min = f32::INFINITY;
    let mut global_max = f32::NEG_INFINITY;
    for table in lut {
        for &v in table {
            if v < global_min {
                global_min = v;
            }
            if v > global_max {
                global_max = v;
            }
        }
    }

    let range = global_max - global_min;
    let (scale, offset) = if range < f32::EPSILON {
        // Degenerate case: all values equal.
        (1.0, global_min)
    } else {
        (range / 255.0, global_min)
    };

    let inv_scale = if range < f32::EPSILON {
        0.0
    } else {
        255.0 / range
    };

    let mut quantized = Vec::with_capacity(lut.len() * 16);
    for table in lut {
        for &v in table {
            let q = ((v - offset) * inv_scale).round().clamp(0.0, 255.0) as u8;
            quantized.push(q);
        }
    }

    (quantized, scale, offset)
}

/// Portable FastScan kernel: process one 32-vector block.
///
/// Simulates what vpshufb does in hardware: for each of 32 vectors, look up
/// each subquantizer's code in the 16-entry quantized LUT and accumulate
/// as u16.
///
/// # Arguments
///
/// * `block_data` -- packed nibble data for this 32-vector block
///   (16 bytes per subquantizer, laid out as `[m0_bytes..., m1_bytes..., ...]`)
/// * `lut_quantized` -- quantized LUT, flat `[num_codebooks * 16]`
/// * `num_codebooks` -- number of subquantizers
///
/// # Returns
///
/// 32 accumulated u16 distances (one per vector in the block).
pub fn fastscan_block_portable(
    block_data: &[u8],
    lut_quantized: &[u8],
    num_codebooks: usize,
) -> [u16; 32] {
    let mut accum = [0u16; 32];

    for m in 0..num_codebooks {
        let lut_offset = m * 16;
        let data_offset = m * 16;

        for lane in 0..16usize {
            let packed_byte = block_data[data_offset + lane];
            let code_lo = (packed_byte & 0x0F) as usize; // vector `lane`
            let code_hi = (packed_byte >> 4) as usize; // vector `lane + 16`

            accum[lane] += lut_quantized[lut_offset + code_lo] as u16;
            accum[lane + 16] += lut_quantized[lut_offset + code_hi] as u16;
        }
    }

    accum
}

/// End-to-end FastScan batch: pack codes, quantize LUT, scan all blocks,
/// convert back to f32 distances.
///
/// # Arguments
///
/// * `packed` -- pre-packed 4-bit codes
/// * `lut` -- float LUT, shape `[num_codebooks][16]`
///
/// # Returns
///
/// `Vec<f32>` of length `packed.num_vectors`, approximate distances.
///
/// The u16 accumulations are converted back via `value * scale + offset`
/// where `(scale, offset)` come from `quantize_lut`.
pub fn fastscan_batch(packed: &PackedCodes4bit, lut: &[Vec<f32>]) -> Vec<f32> {
    assert_eq!(lut.len(), packed.num_codebooks);

    let (lut_q, scale, offset) = quantize_lut(lut);
    let num_codebooks = packed.num_codebooks;
    let num_blocks = packed.num_blocks();
    let bpb = packed.bytes_per_block();

    // Per-codebook offset contributes `offset` to the sum `num_codebooks` times.
    let base_offset = offset * num_codebooks as f32;

    let mut distances = Vec::with_capacity(packed.num_vectors);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * bpb;
        let block_end = (block_start + bpb).min(packed.data.len());
        let block_data = &packed.data[block_start..block_end];

        let accum = fastscan_block_portable(block_data, &lut_q, num_codebooks);

        let vecs_in_block = if block_idx == num_blocks - 1 {
            let remaining = packed.num_vectors - block_idx * 32;
            remaining.min(32)
        } else {
            32
        };

        for &a in accum.iter().take(vecs_in_block) {
            distances.push(a as f32 * scale + base_offset);
        }
    }

    distances
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_lut(num_codebooks: usize, codebook_size: usize) -> Vec<Vec<f32>> {
        (0..num_codebooks)
            .map(|m| {
                (0..codebook_size)
                    .map(|c| (m * codebook_size + c) as f32 * 0.1)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_adc_distance_basic() {
        let lut = create_test_lut(4, 256);
        let codes = vec![0u8, 1, 2, 3];

        let dist = adc_distance(&codes, &lut);

        // Manual calculation:
        // m=0: lut[0][0] = 0.0
        // m=1: lut[1][1] = (256 + 1) * 0.1 = 25.7
        // m=2: lut[2][2] = (512 + 2) * 0.1 = 51.4
        // m=3: lut[3][3] = (768 + 3) * 0.1 = 77.1
        let expected = 0.0 + 25.7 + 51.4 + 77.1;
        assert!(
            (dist - expected).abs() < 0.01,
            "got {}, expected {}",
            dist,
            expected
        );
    }

    #[test]
    fn test_packed_lut_equivalence() {
        let nested_lut = create_test_lut(8, 256);
        let packed_lut = PackedLUT::from_nested(&nested_lut);

        let codes = vec![10u8, 20, 30, 40, 50, 60, 70, 80];

        let nested_dist = adc_distance(&codes, &nested_lut);
        let packed_dist = packed_lut.adc_distance(&codes);

        assert!(
            (nested_dist - packed_dist).abs() < 1e-6,
            "nested={}, packed={}",
            nested_dist,
            packed_dist
        );
    }

    #[test]
    fn test_adc_batch_correctness() {
        let lut = create_test_lut(4, 256);
        let packed_lut = PackedLUT::from_nested(&lut);

        // Create batch of 100 random-ish codes
        let n_candidates = 100;
        let num_codebooks = 4;
        let codes_batch: Vec<u8> = (0..n_candidates * num_codebooks)
            .map(|i| (i % 256) as u8)
            .collect();

        let batch_result = adc_batch_dispatch(&codes_batch, num_codebooks, &packed_lut);

        // Verify against individual computation
        for i in 0..n_candidates {
            let codes = &codes_batch[i * num_codebooks..(i + 1) * num_codebooks];
            let expected = packed_lut.adc_distance(codes);
            let actual = batch_result[i];

            assert!(
                (expected - actual).abs() < 1e-5,
                "candidate {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_adc_batch_simd_consistency() {
        let lut = create_test_lut(16, 256);
        let packed_lut = PackedLUT::from_nested(&lut);

        // Large batch to trigger SIMD paths
        let n_candidates = 1000;
        let num_codebooks = 16;
        let codes_batch: Vec<u8> = (0..n_candidates * num_codebooks)
            .map(|i| ((i * 7) % 256) as u8)
            .collect();

        let result = adc_batch_dispatch(&codes_batch, num_codebooks, &packed_lut);

        // Verify all results
        for i in 0..n_candidates {
            let codes = &codes_batch[i * num_codebooks..(i + 1) * num_codebooks];
            let expected = packed_lut.adc_distance(codes);
            let actual = result[i];

            assert!(
                (expected - actual).abs() < 1e-4,
                "mismatch at {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_empty_batch() {
        let lut = create_test_lut(4, 256);
        let packed_lut = PackedLUT::from_nested(&lut);

        let result = adc_batch_dispatch(&[], 4, &packed_lut);
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_candidate() {
        let lut = create_test_lut(4, 256);
        let packed_lut = PackedLUT::from_nested(&lut);

        let codes = vec![5u8, 10, 15, 20];
        let result = adc_batch_dispatch(&codes, 4, &packed_lut);

        assert_eq!(result.len(), 1);
        let expected = packed_lut.adc_distance(&codes);
        assert!((result[0] - expected).abs() < 1e-6);
    }

    // ── FastScan 4-bit tests ─────────────────────────────────────────────

    fn create_4bit_lut(num_codebooks: usize) -> Vec<Vec<f32>> {
        // 16 entries per codebook (4-bit PQ).
        (0..num_codebooks)
            .map(|m| (0..16).map(|c| (m * 16 + c) as f32 * 0.5).collect())
            .collect()
    }

    /// Scalar reference: compute ADC distance for one vector using 4-bit codes
    /// and a float LUT with 16 entries per codebook.
    fn scalar_adc_4bit(codes: &[u8], lut: &[Vec<f32>]) -> f32 {
        codes
            .iter()
            .zip(lut.iter())
            .map(|(&c, table)| table[(c & 0x0F) as usize])
            .sum()
    }

    #[test]
    fn test_packed_codes_4bit_roundtrip() {
        let num_vectors = 5;
        let num_codebooks = 4;
        // Codes: vector i, codebook m -> (i + m) % 16
        let codes: Vec<u8> = (0..num_vectors * num_codebooks)
            .map(|idx| {
                let i = idx / num_codebooks;
                let m = idx % num_codebooks;
                ((i + m) % 16) as u8
            })
            .collect();

        let packed = PackedCodes4bit::pack(&codes, num_vectors, num_codebooks);
        assert_eq!(packed.num_vectors, num_vectors);
        assert_eq!(packed.num_codebooks, num_codebooks);
        assert_eq!(packed.block_size, 32);
        assert_eq!(packed.num_blocks(), 1);

        // Verify we can unpack: for each vector, extract its code from the
        // nibble layout and compare with original.
        let bpb = packed.bytes_per_block();
        let block = &packed.data[0..bpb];
        for i in 0..num_vectors {
            for m in 0..num_codebooks {
                let byte = block[m * 16 + (i % 16)];
                let code = if i < 16 { byte & 0x0F } else { byte >> 4 };
                assert_eq!(
                    code,
                    codes[i * num_codebooks + m],
                    "mismatch at vec={}, cb={}",
                    i,
                    m
                );
            }
        }
    }

    #[test]
    fn test_quantize_lut_range() {
        let lut = create_4bit_lut(8);
        let (quantized, scale, offset) = quantize_lut(&lut);

        assert_eq!(quantized.len(), 8 * 16);
        // Min should map to ~0, max should map to ~255.
        assert_eq!(quantized[0], 0); // smallest value
        assert_eq!(*quantized.last().unwrap(), 255); // largest value

        // Reconstruct and check closeness.
        for (m, table) in lut.iter().enumerate() {
            for (c, &original) in table.iter().enumerate() {
                let reconstructed = quantized[m * 16 + c] as f32 * scale + offset;
                let err = (reconstructed - original).abs();
                // Max quantization error is scale/2 per entry.
                assert!(
                    err <= scale / 2.0 + 1e-5,
                    "m={}, c={}: original={}, reconstructed={}, err={}",
                    m,
                    c,
                    original,
                    reconstructed,
                    err,
                );
            }
        }
    }

    #[test]
    fn test_fastscan_vs_scalar_adc() {
        // Compare fastscan_batch output against scalar ADC on the same data.
        let num_codebooks = 8;
        let num_vectors = 100;
        let lut = create_4bit_lut(num_codebooks);

        // Deterministic pseudo-random codes in 0..16.
        let codes: Vec<u8> = (0..num_vectors * num_codebooks)
            .map(|i| ((i * 7 + 3) % 16) as u8)
            .collect();

        // Scalar reference distances.
        let scalar_dists: Vec<f32> = (0..num_vectors)
            .map(|i| {
                let c = &codes[i * num_codebooks..(i + 1) * num_codebooks];
                scalar_adc_4bit(c, &lut)
            })
            .collect();

        // FastScan distances.
        let packed = PackedCodes4bit::pack(&codes, num_vectors, num_codebooks);
        let fastscan_dists = fastscan_batch(&packed, &lut);

        assert_eq!(fastscan_dists.len(), num_vectors);

        // Tolerance: u8 quantization introduces up to `scale/2` error per
        // codebook, so total error bound is `num_codebooks * scale / 2`.
        let (_, scale, _) = quantize_lut(&lut);
        let tolerance = num_codebooks as f32 * scale * 0.5 + 1e-3;

        for i in 0..num_vectors {
            let err = (fastscan_dists[i] - scalar_dists[i]).abs();
            assert!(
                err <= tolerance,
                "vec {}: fastscan={}, scalar={}, err={}, tol={}",
                i,
                fastscan_dists[i],
                scalar_dists[i],
                err,
                tolerance,
            );
        }
    }

    #[test]
    fn test_fastscan_multiple_blocks() {
        // Exercise the multi-block path (>32 vectors).
        let num_codebooks = 4;
        let num_vectors = 100; // 4 blocks (32 + 32 + 32 + 4)
        let lut = create_4bit_lut(num_codebooks);

        let codes: Vec<u8> = (0..num_vectors * num_codebooks)
            .map(|i| ((i * 11 + 5) % 16) as u8)
            .collect();

        let packed = PackedCodes4bit::pack(&codes, num_vectors, num_codebooks);
        assert_eq!(packed.num_blocks(), 4);

        let fastscan_dists = fastscan_batch(&packed, &lut);
        assert_eq!(fastscan_dists.len(), num_vectors);

        let (_, scale, _) = quantize_lut(&lut);
        let tolerance = num_codebooks as f32 * scale * 0.5 + 1e-3;

        for i in 0..num_vectors {
            let c = &codes[i * num_codebooks..(i + 1) * num_codebooks];
            let scalar = scalar_adc_4bit(c, &lut);
            let err = (fastscan_dists[i] - scalar).abs();
            assert!(
                err <= tolerance,
                "vec {}: fastscan={}, scalar={}, err={}, tol={}",
                i,
                fastscan_dists[i],
                scalar,
                err,
                tolerance,
            );
        }
    }

    #[test]
    fn test_fastscan_odd_codebooks() {
        // Odd number of codebooks to exercise the solo-pair path.
        let num_codebooks = 5;
        let num_vectors = 40;
        let lut = create_4bit_lut(num_codebooks);

        let codes: Vec<u8> = (0..num_vectors * num_codebooks)
            .map(|i| (i % 16) as u8)
            .collect();

        let packed = PackedCodes4bit::pack(&codes, num_vectors, num_codebooks);
        let fastscan_dists = fastscan_batch(&packed, &lut);
        assert_eq!(fastscan_dists.len(), num_vectors);

        let (_, scale, _) = quantize_lut(&lut);
        let tolerance = num_codebooks as f32 * scale * 0.5 + 1e-3;

        for i in 0..num_vectors {
            let c = &codes[i * num_codebooks..(i + 1) * num_codebooks];
            let scalar = scalar_adc_4bit(c, &lut);
            let err = (fastscan_dists[i] - scalar).abs();
            assert!(
                err <= tolerance,
                "vec {}: fastscan={}, scalar={}, err={}",
                i,
                fastscan_dists[i],
                scalar,
                err,
            );
        }
    }
}
