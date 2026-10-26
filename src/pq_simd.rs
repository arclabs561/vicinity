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
//! # Kernels
//!
//! This module has no `unsafe` code.
//!
//! - 8-bit ADC (`codebook_size = 256`): scores four candidates per step from
//!   `[f32; 256]` table rows, so `u8` codes index without bounds checks and
//!   the four sums run in parallel. On arm64, f32 gathers use scalar loads.
//! - 4-bit FastScan (`codebook_size = 16`): packs 32 candidates per block.
//!   On aarch64 NEON `tbl` does the 16-way byte lookup through `pulp`;
//!   elsewhere a portable scalar kernel does.
//!
//! `codebook_size = 256` is the normal high-recall path used by current
//! GloVe-25 benchmarks. `codebook_size = 16` enables the FastScan layout but
//! trades recall headroom for speed and compactness.
//!
//! # Performance
//!
//! See `docs/benchmark-results.md` for measured rows; performance depends on
//! codebook shape, architecture, and whether the index has prepacked cluster
//! codes.
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
    assert_adc_batch_shape(codes_batch, num_codebooks);
    assert_eq!(
        lut.len(),
        num_codebooks,
        "lut codebook count {} does not match num_codebooks {}",
        lut.len(),
        num_codebooks
    );

    let n_candidates = codes_batch.len() / num_codebooks;
    let mut distances = Vec::with_capacity(n_candidates);

    for i in 0..n_candidates {
        let codes = &codes_batch[i * num_codebooks..(i + 1) * num_codebooks];
        distances.push(adc_distance(codes, lut));
    }

    distances
}

/// Compute ADC distances into a caller-owned buffer.
pub fn adc_batch_distances_into<L: PackedLUTData>(
    codes_batch: &[u8],
    num_codebooks: usize,
    lut: &L,
    distances: &mut Vec<f32>,
) {
    assert_packed_adc_batch_shape(codes_batch, num_codebooks, lut);

    let n_candidates = codes_batch.len() / num_codebooks;
    distances.clear();
    distances.resize(n_candidates, 0.0);

    for i in 0..n_candidates {
        let codes = &codes_batch[i * num_codebooks..(i + 1) * num_codebooks];
        distances[i] = lut.adc_distance(codes);
    }
}

#[inline]
fn assert_adc_batch_shape(codes_batch: &[u8], num_codebooks: usize) {
    assert!(num_codebooks > 0, "num_codebooks must be non-zero");
    assert_eq!(
        codes_batch.len() % num_codebooks,
        0,
        "codes_batch length {} is not a multiple of num_codebooks {}",
        codes_batch.len(),
        num_codebooks
    );
}

#[inline]
fn assert_packed_adc_batch_shape<L: PackedLUTData>(
    codes_batch: &[u8],
    num_codebooks: usize,
    lut: &L,
) {
    assert_adc_batch_shape(codes_batch, num_codebooks);
    assert_eq!(
        lut.num_codebooks(),
        num_codebooks,
        "packed LUT codebook count {} does not match num_codebooks {}",
        lut.num_codebooks(),
        num_codebooks
    );
    let expected_len = match lut.num_codebooks().checked_mul(lut.codebook_size()) {
        Some(value) => value,
        None => panic!("packed LUT shape overflows usize"),
    };
    assert!(
        lut.data().len() >= expected_len,
        "packed LUT length {} is smaller than required length {}",
        lut.data().len(),
        expected_len
    );
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

    /// Create a packed LUT directly from a flat ADC table, avoiding
    /// the intermediate `Vec<Vec<f32>>` allocation of `from_nested`.
    pub fn from_flat(table: &[f32], num_codebooks: usize, codebook_size: usize) -> Self {
        debug_assert_eq!(table.len(), num_codebooks * codebook_size);
        Self {
            data: table.to_vec(),
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

    /// Number of codebooks.
    pub fn num_codebooks(&self) -> usize {
        self.num_codebooks
    }
}

/// Borrowed view over a flat ADC table.
///
/// Standard 8-bit IVF-PQ search already has a flat table for each probed
/// cluster. Borrowing it avoids allocating and copying a `PackedLUT` before
/// every batch dispatch.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PackedLUTRef<'a> {
    data: &'a [f32],
    num_codebooks: usize,
    codebook_size: usize,
}

impl<'a> PackedLUTRef<'a> {
    pub(crate) fn from_flat(table: &'a [f32], num_codebooks: usize, codebook_size: usize) -> Self {
        debug_assert_eq!(table.len(), num_codebooks * codebook_size);
        Self {
            data: table,
            num_codebooks,
            codebook_size,
        }
    }
}

pub(crate) trait PackedLUTData {
    fn data(&self) -> &[f32];
    fn num_codebooks(&self) -> usize;
    fn codebook_size(&self) -> usize;

    #[inline]
    fn adc_distance(&self, codes: &[u8]) -> f32 {
        debug_assert_eq!(codes.len(), self.num_codebooks());

        let mut sum = 0.0f32;
        for (m, &code) in codes.iter().enumerate() {
            sum += self.data()[m * self.codebook_size() + code as usize];
        }
        sum
    }
}

impl PackedLUTData for PackedLUT {
    #[inline]
    fn data(&self) -> &[f32] {
        &self.data
    }

    #[inline]
    fn num_codebooks(&self) -> usize {
        self.num_codebooks
    }

    #[inline]
    fn codebook_size(&self) -> usize {
        self.codebook_size
    }
}

impl PackedLUTData for PackedLUTRef<'_> {
    #[inline]
    fn data(&self) -> &[f32] {
        self.data
    }

    #[inline]
    fn num_codebooks(&self) -> usize {
        self.num_codebooks
    }

    #[inline]
    fn codebook_size(&self) -> usize {
        self.codebook_size
    }
}

/// Auto-dispatching batch ADC computation.
pub fn adc_batch_dispatch<L: PackedLUTData>(
    codes_batch: &[u8],
    num_codebooks: usize,
    lut: &L,
) -> Vec<f32> {
    let mut distances = Vec::new();
    adc_batch_dispatch_into(codes_batch, num_codebooks, lut, &mut distances);
    distances
}

/// Auto-dispatching batch ADC computation into a caller-owned buffer.
///
/// 256-entry codebooks (8-bit codes) take a kernel that scores four
/// candidates at once from fixed-size table rows, so a `u8` code indexes a
/// `[f32; 256]` row without a bounds check and the four sums are independent.
/// Other codebook sizes use the per-candidate loop. Both sum each candidate's
/// codebooks in order, so they return identical results.
pub fn adc_batch_dispatch_into<L: PackedLUTData>(
    codes_batch: &[u8],
    num_codebooks: usize,
    lut: &L,
    distances: &mut Vec<f32>,
) {
    assert_packed_adc_batch_shape(codes_batch, num_codebooks, lut);

    if lut.codebook_size() == 256 {
        adc_batch_256_into(codes_batch, num_codebooks, lut.data(), distances);
    } else {
        adc_batch_distances_into(codes_batch, num_codebooks, lut, distances);
    }
}

/// Batch ADC for flat 256-entry LUTs, four candidates per step.
fn adc_batch_256_into(
    codes_batch: &[u8],
    num_codebooks: usize,
    lut_data: &[f32],
    distances: &mut Vec<f32>,
) {
    let n_candidates = codes_batch.len() / num_codebooks;
    distances.clear();
    distances.resize(n_candidates, 0.0);

    let (tables, _) = lut_data.as_chunks::<256>();
    let tables = &tables[..num_codebooks];

    let (out_quads, out_remainder) = distances.as_chunks_mut::<4>();
    let mut code_quads = codes_batch.chunks_exact(4 * num_codebooks);
    for (out, quad) in out_quads.iter_mut().zip(&mut code_quads) {
        let (r0, rest) = quad.split_at(num_codebooks);
        let (r1, rest) = rest.split_at(num_codebooks);
        let (r2, r3) = rest.split_at(num_codebooks);
        let mut sums = [0.0f32; 4];
        for ((((table, &c0), &c1), &c2), &c3) in tables.iter().zip(r0).zip(r1).zip(r2).zip(r3) {
            sums[0] += table[c0 as usize];
            sums[1] += table[c1 as usize];
            sums[2] += table[c2 as usize];
            sums[3] += table[c3 as usize];
        }
        out.copy_from_slice(&sums);
    }

    for (out, codes) in out_remainder
        .iter_mut()
        .zip(code_quads.remainder().chunks_exact(num_codebooks))
    {
        let mut sum = 0.0f32;
        for (table, &code) in tables.iter().zip(codes) {
            sum += table[code as usize];
        }
        *out = sum;
    }
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
// 3. Portable scalar FastScan kernel (simulates vpshufb behavior) and an
//    aarch64 NEON `tbl` kernel
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

#[inline]
fn fastscan_bytes_per_block(num_codebooks: usize) -> usize {
    match num_codebooks.checked_mul(16) {
        Some(value) => value,
        None => panic!("fastscan codebook count overflows usize"),
    }
}

impl PackedCodes4bit {
    /// Pack flat 8-bit codes into the FastScan nibble layout.
    ///
    /// `codes` is row-major: `codes[i * num_codebooks + m]` is vector i's
    /// code for subquantizer m. Each value must be in 0..16.
    pub fn pack(codes: &[u8], num_vectors: usize, num_codebooks: usize) -> Self {
        let expected_codes_len = match num_vectors.checked_mul(num_codebooks) {
            Some(value) => value,
            None => panic!("packed code shape overflows usize"),
        };
        assert_eq!(
            codes.len(),
            expected_codes_len,
            "codes length {} does not match num_vectors {} * num_codebooks {}",
            codes.len(),
            num_vectors,
            num_codebooks
        );

        let block_size = 32usize;
        let num_blocks = num_vectors.div_ceil(block_size);
        let bytes_per_block = fastscan_bytes_per_block(num_codebooks);
        let total_bytes = match num_blocks.checked_mul(bytes_per_block) {
            Some(value) => value,
            None => panic!("packed FastScan storage length overflows usize"),
        };

        let mut data = vec![0u8; total_bytes];

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
        fastscan_bytes_per_block(self.num_codebooks)
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

/// Quantize a flat 4-bit FastScan LUT without building nested vectors.
///
/// `lut` must contain `num_codebooks * 16` entries laid out as
/// `[codebook_0_values..., codebook_1_values..., ...]`.
pub fn quantize_lut_flat(lut: &[f32], num_codebooks: usize) -> (Vec<u8>, f32, f32) {
    debug_assert_eq!(lut.len(), num_codebooks * 16);
    if lut.is_empty() {
        return (Vec::new(), 1.0, 0.0);
    }

    let mut global_min = f32::INFINITY;
    let mut global_max = f32::NEG_INFINITY;
    for &value in lut {
        if value < global_min {
            global_min = value;
        }
        if value > global_max {
            global_max = value;
        }
    }

    let range = global_max - global_min;
    let (scale, offset) = if range < f32::EPSILON {
        (1.0, global_min)
    } else {
        (range / 255.0, global_min)
    };
    let inv_scale = if range < f32::EPSILON {
        0.0
    } else {
        255.0 / range
    };

    let quantized = lut
        .iter()
        .map(|&value| ((value - offset) * inv_scale).round().clamp(0.0, 255.0) as u8)
        .collect();

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
    let required_len = fastscan_bytes_per_block(num_codebooks);
    assert!(block_data.len() >= required_len);
    assert!(lut_quantized.len() >= required_len);

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

/// FastScan block kernel for aarch64: NEON `tbl` performs the 16-way byte
/// lookup for 16 lanes at once. The intrinsics go through `pulp`'s
/// capability token, so this crate needs no `unsafe`.
#[cfg(target_arch = "aarch64")]
fn fastscan_block_neon(
    simd: pulp::aarch64::Neon,
    block_data: &[u8],
    lut_quantized: &[u8],
    num_codebooks: usize,
) -> [u16; 32] {
    use std::arch::aarch64::{uint16x8_t, uint8x16_t};

    let required_len = fastscan_bytes_per_block(num_codebooks);
    let (codes, _) = block_data[..required_len].as_chunks::<16>();
    let (tables, _) = lut_quantized[..required_len].as_chunks::<16>();

    let neon = simd.neon;
    let low_mask = neon.vdupq_n_u8(0x0f);
    let mut low_0_7: uint16x8_t = neon.vdupq_n_u16(0);
    let mut low_8_15: uint16x8_t = neon.vdupq_n_u16(0);
    let mut high_0_7: uint16x8_t = neon.vdupq_n_u16(0);
    let mut high_8_15: uint16x8_t = neon.vdupq_n_u16(0);

    for (packed, table) in codes.iter().zip(tables) {
        let packed: uint8x16_t = pulp::cast(*packed);
        let table: uint8x16_t = pulp::cast(*table);
        let low_vals = neon.vqtbl1q_u8(table, neon.vandq_u8(packed, low_mask));
        let high_vals = neon.vqtbl1q_u8(table, neon.vshrq_n_u8::<4>(packed));

        low_0_7 = neon.vaddw_u8(low_0_7, neon.vget_low_u8(low_vals));
        low_8_15 = neon.vaddw_u8(low_8_15, neon.vget_high_u8(low_vals));
        high_0_7 = neon.vaddw_u8(high_0_7, neon.vget_low_u8(high_vals));
        high_8_15 = neon.vaddw_u8(high_8_15, neon.vget_high_u8(high_vals));
    }

    pulp::cast::<[uint16x8_t; 4], [u16; 32]>([low_0_7, low_8_15, high_0_7, high_8_15])
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
    fastscan_batch_quantized(packed, &lut_q, scale, offset)
}

/// End-to-end FastScan batch using a flat `[num_codebooks * 16]` LUT.
pub fn fastscan_batch_flat(packed: &PackedCodes4bit, lut: &[f32]) -> Vec<f32> {
    let (lut_q, scale, offset) = quantize_lut_flat(lut, packed.num_codebooks);
    fastscan_batch_quantized(packed, &lut_q, scale, offset)
}

fn fastscan_batch_quantized(
    packed: &PackedCodes4bit,
    lut_q: &[u8],
    scale: f32,
    offset: f32,
) -> Vec<f32> {
    let num_codebooks = packed.num_codebooks;
    let num_blocks = packed.num_blocks();
    let bpb = packed.bytes_per_block();

    // Per-codebook offset contributes `offset` to the sum `num_codebooks` times.
    let base_offset = offset * num_codebooks as f32;

    let mut distances = Vec::with_capacity(packed.num_vectors);
    #[cfg(target_arch = "aarch64")]
    let neon = pulp::aarch64::Neon::try_new();

    for block_idx in 0..num_blocks {
        let block_start = block_idx * bpb;
        let block_end = (block_start + bpb).min(packed.data.len());
        let block_data = &packed.data[block_start..block_end];

        #[cfg(target_arch = "aarch64")]
        let accum = match neon {
            Some(simd) => fastscan_block_neon(simd, block_data, lut_q, num_codebooks),
            None => fastscan_block_portable(block_data, lut_q, num_codebooks),
        };
        #[cfg(not(target_arch = "aarch64"))]
        let accum = fastscan_block_portable(block_data, lut_q, num_codebooks);

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
    fn test_borrowed_lut_dispatch_matches_owned_lut_and_scalar_oracle() {
        let nested_lut = create_test_lut(5, 256);
        let flat_lut: Vec<f32> = nested_lut.iter().flatten().copied().collect();
        let owned_lut = PackedLUT::from_flat(&flat_lut, 5, 256);
        let borrowed_lut = PackedLUTRef::from_flat(&flat_lut, 5, 256);
        let n_candidates = 257;
        let num_codebooks = 5;
        let codes_batch: Vec<u8> = (0..n_candidates * num_codebooks)
            .map(|i| ((i * 37) % 256) as u8)
            .collect();

        let owned = adc_batch_dispatch(&codes_batch, num_codebooks, &owned_lut);
        let borrowed = adc_batch_dispatch(&codes_batch, num_codebooks, &borrowed_lut);

        assert_eq!(borrowed, owned);
        for i in 0..n_candidates {
            let codes = &codes_batch[i * num_codebooks..(i + 1) * num_codebooks];
            let scalar = adc_distance(codes, &nested_lut);
            assert!(
                (borrowed[i] - scalar).abs() < 1e-4,
                "candidate {}: borrowed={}, scalar={}",
                i,
                borrowed[i],
                scalar,
            );
        }
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
    #[should_panic(expected = "num_codebooks must be non-zero")]
    fn test_adc_batch_rejects_zero_codebooks() {
        let lut = create_test_lut(1, 256);
        let packed_lut = PackedLUT::from_nested(&lut);

        let _ = adc_batch_dispatch(&[], 0, &packed_lut);
    }

    #[test]
    #[should_panic(expected = "is not a multiple of num_codebooks")]
    fn test_adc_batch_rejects_partial_candidate_row() {
        let lut = create_test_lut(2, 256);
        let packed_lut = PackedLUT::from_nested(&lut);

        let _ = adc_batch_dispatch(&[1, 2, 3], 2, &packed_lut);
    }

    #[test]
    #[should_panic(expected = "does not match num_codebooks")]
    fn test_adc_batch_rejects_lut_codebook_mismatch() {
        let lut = create_test_lut(3, 256);
        let packed_lut = PackedLUT::from_nested(&lut);

        let _ = adc_batch_dispatch(&[1, 2, 3, 4], 4, &packed_lut);
    }

    #[test]
    fn test_non_256_lut_dispatch_matches_scalar_oracle() {
        let lut = create_test_lut(3, 16);
        let packed_lut = PackedLUT::from_nested(&lut);
        let n_candidates = 37;
        let num_codebooks = 3;
        let codes_batch: Vec<u8> = (0..n_candidates * num_codebooks)
            .map(|i| ((i * 7 + 3) % 16) as u8)
            .collect();

        let batch_result = adc_batch_dispatch(&codes_batch, num_codebooks, &packed_lut);

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
    #[should_panic(expected = "does not match num_vectors")]
    fn test_packed_codes_4bit_rejects_partial_rows() {
        let codes = [0u8, 1, 2];

        let _ = PackedCodes4bit::pack(&codes, 2, 2);
    }

    #[test]
    #[should_panic]
    fn test_fastscan_block_rejects_short_lut() {
        let block_data = [0u8; 16];
        let lut_quantized = [0u8; 15];

        let _ = fastscan_block_portable(&block_data, &lut_quantized, 1);
    }

    /// Naive reference: sums one candidate's codebooks in order.
    fn naive_adc(codes: &[u8], lut: &[f32], codebook_size: usize) -> f32 {
        let mut sum = 0.0f32;
        for (m, &code) in codes.iter().enumerate() {
            sum += lut[m * codebook_size + code as usize];
        }
        sum
    }

    /// The batch kernels must equal the naive reference exactly for every
    /// candidate, including the ones after the last full group of four. Every
    /// LUT entry is positive, so a dropped remainder (left at 0.0) fails.
    #[test]
    fn adc_batch_matches_naive_reference_at_odd_sizes() {
        for codebook_size in [256usize, 16] {
            for num_codebooks in [1usize, 3, 5, 25] {
                let lut: Vec<f32> = (0..num_codebooks * codebook_size)
                    .map(|i| 0.25 + ((i * 7919) % 1009) as f32 * 0.003)
                    .collect();
                let packed = PackedLUTRef::from_flat(&lut, num_codebooks, codebook_size);
                for n_candidates in [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 17, 33, 257] {
                    let codes: Vec<u8> = (0..n_candidates * num_codebooks)
                        .map(|i| ((i * 131 + 17) % codebook_size) as u8)
                        .collect();
                    let got = adc_batch_dispatch(&codes, num_codebooks, &packed);
                    let want: Vec<f32> = codes
                        .chunks_exact(num_codebooks)
                        .map(|row| naive_adc(row, &lut, codebook_size))
                        .collect();
                    assert_eq!(
                        got, want,
                        "codebook_size={codebook_size} m={num_codebooks} n={n_candidates}"
                    );
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_fastscan_block_matches_portable() {
        let simd = pulp::aarch64::Neon::try_new().expect("aarch64 has NEON");
        for num_codebooks in [1usize, 2, 7, 25, 64] {
            let num_vectors = 32;
            let codes: Vec<u8> = (0..num_vectors * num_codebooks)
                .map(|i| {
                    let candidate = i / num_codebooks;
                    let codebook = i % num_codebooks;
                    // Distinguish both 8-lane groups and the low/high nibbles.
                    ((candidate * 11 + candidate / 8 + codebook * 7 + 3) % 16) as u8
                })
                .collect();
            let packed = PackedCodes4bit::pack(&codes, num_vectors, num_codebooks);
            let lut: Vec<f32> = (0..num_codebooks * 16)
                .map(|i| ((i * 17 + 5) % 251) as f32 * 0.01)
                .collect();
            let (lut_q, _, _) = quantize_lut_flat(&lut, num_codebooks);

            let portable = fastscan_block_portable(packed.block_data(0), &lut_q, num_codebooks);
            let neon = fastscan_block_neon(simd, packed.block_data(0), &lut_q, num_codebooks);
            assert_eq!(neon, portable, "num_codebooks={num_codebooks}");
        }
    }

    #[test]
    fn test_quantize_lut_range() {
        let lut = create_4bit_lut(8);
        let (quantized, scale, offset) = quantize_lut(&lut);

        assert_eq!(quantized.len(), 8 * 16);
        // Min should map to ~0, max should map to ~255.
        assert_eq!(quantized[0], 0); // smallest value
        assert_eq!(quantized[quantized.len() - 1], 255); // largest value

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
    fn test_flat_fastscan_matches_nested() {
        let num_vectors = 64;
        let num_codebooks = 5;
        let lut = create_4bit_lut(num_codebooks);
        let flat_lut: Vec<f32> = lut.iter().flat_map(|table| table.iter().copied()).collect();
        let codes: Vec<u8> = (0..num_vectors * num_codebooks)
            .map(|i| (i % 16) as u8)
            .collect();
        let packed = PackedCodes4bit::pack(&codes, num_vectors, num_codebooks);

        let nested = fastscan_batch(&packed, &lut);
        let flat = fastscan_batch_flat(&packed, &flat_lut);

        assert_eq!(nested, flat);
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
