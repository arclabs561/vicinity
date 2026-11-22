//! SAQ (Segmented Adaptive Quantization) implementation.
//!
//! SAQ-style segmented adaptive quantization helper with direct dimension
//! segmentation and variance-weighted bit allocation.
//!
//! Treat this as experimental until local quantization-error, encoding
//! throughput, and downstream recall rows justify it against the other
//! quantizers in this crate. This is not yet a full SAQ paper reproduction:
//! PCA projection and dynamic-programming bit allocation are reserved future
//! work. Per-segment codebooks are trained with deterministic spherical k-means.
//!
//! # References
//!
//! - Li et al. (2026): "SAQ: Pushing the Limits of Vector Quantization through
//!   Code Adjustment and Dimension Segmentation" - <https://arxiv.org/abs/2509.12086>

use crate::RetrieveError;
use crate::distance::cosine_distance_normalized;

/// SAQ quantizer with per-segment codebooks.
pub struct SAQQuantizer {
    dimension: usize,
    num_segments: usize,
    bits_per_segment: Vec<usize>,        // Bit allocation per segment
    codebooks: Vec<Vec<Vec<f32>>>,       // [segment][codeword][dimension]
    segment_bounds: Vec<(usize, usize)>, // (start, end) for each segment
}

impl SAQQuantizer {
    /// Create new SAQ quantizer.
    pub fn new(
        dimension: usize,
        num_segments: usize,
        total_bits: usize,
    ) -> Result<Self, RetrieveError> {
        if dimension == 0 || num_segments == 0 || total_bits == 0 {
            return Err(RetrieveError::InvalidConfig {
                field: "dimension, num_segments, total_bits",
                expected: "must all be greater than 0",
            });
        }

        if !dimension.is_multiple_of(num_segments) {
            return Err(RetrieveError::InvalidConfig {
                field: "dimension",
                expected: "must be divisible by num_segments",
            });
        }

        // Initial bit allocation (will be optimized)
        // Codes are u8. Preserve the remainder when dividing the usable budget.
        let budget = total_bits.min(num_segments.saturating_mul(8));
        let bits_per_segment = (0..num_segments)
            .map(|i| budget / num_segments + usize::from(i < budget % num_segments))
            .collect();

        // Segment bounds
        let segment_dim = dimension / num_segments;
        let mut segment_bounds = Vec::new();
        for i in 0..num_segments {
            segment_bounds.push((i * segment_dim, (i + 1) * segment_dim));
        }

        Ok(Self {
            dimension,
            num_segments,
            bits_per_segment,
            codebooks: Vec::new(),
            segment_bounds,
        })
    }

    /// Train per-segment codebooks under the configured bit budget.
    pub fn fit(&mut self, vectors: &[f32], num_vectors: usize) -> Result<(), RetrieveError> {
        if num_vectors == 0 || self.dimension.checked_mul(num_vectors) != Some(vectors.len()) {
            return Err(RetrieveError::InvalidConfig {
                field: "training vectors",
                expected: "nonempty buffer matching dimension times num_vectors",
            });
        }
        self.optimize_segmentation(vectors)?;

        // Stage 3: Train codebooks for each segment
        self.train_codebooks(vectors, num_vectors)?;

        Ok(())
    }

    /// Allocate bits to the fixed segments in proportion to their variance.
    fn optimize_segmentation(&mut self, vectors: &[f32]) -> Result<(), RetrieveError> {
        let total_bits: usize = self.bits_per_segment.iter().sum();

        // Welford's update measures variation across observations for each coordinate.
        // All segments have the same length and sample count, so sums of squared
        // deviations give the same proportions as their mean variances.
        let mut means = vec![0.0_f64; self.dimension];
        let mut segment_variances = vec![0.0_f64; self.num_segments];
        let width = self.dimension / self.num_segments;
        for (row, vector) in vectors.chunks_exact(self.dimension).enumerate() {
            for (column, (&value, mean)) in vector.iter().zip(&mut means).enumerate() {
                let value = f64::from(value);
                if !value.is_finite() {
                    return Err(RetrieveError::InvalidConfig {
                        field: "training vectors",
                        expected: "finite coordinates",
                    });
                }
                let delta = value - *mean;
                *mean += delta / (row + 1) as f64;
                segment_variances[column / width] += delta * (value - *mean);
            }
        }
        let total_variance: f64 = segment_variances.iter().sum();
        if total_variance == 0.0 {
            for (i, bits) in self.bits_per_segment.iter_mut().enumerate() {
                *bits = total_bits / self.num_segments
                    + usize::from(i < total_bits % self.num_segments);
            }
        } else {
            // Each segment has eight possible one-bit increments. Choose the
            // increments furthest below the segment's quota, breaking ties by index.
            let mut increments = Vec::with_capacity(self.num_segments * 8);
            for (index, variance) in segment_variances.iter().enumerate() {
                let quota = variance / total_variance * total_bits as f64;
                for used in 0..8 {
                    increments.push((quota - used as f64, index));
                }
            }
            increments.sort_unstable_by(|(a, i), (b, j)| b.total_cmp(a).then(i.cmp(j)));
            self.bits_per_segment.fill(0);
            for (_, index) in increments.into_iter().take(total_bits) {
                self.bits_per_segment[index] += 1;
            }
        }

        Ok(())
    }

    /// Train codebooks for each segment.
    fn train_codebooks(
        &mut self,
        vectors: &[f32],
        num_vectors: usize,
    ) -> Result<(), RetrieveError> {
        self.codebooks = Vec::new();

        for (segment_idx, (start, end)) in self.segment_bounds.iter().enumerate() {
            let segment_dim = end - start;
            let codebook_size = 2usize.pow(self.bits_per_segment[segment_idx].min(8) as u32);

            // Extract subvectors for this segment
            let mut subvectors = Vec::new();
            for i in 0..num_vectors {
                let vec = get_vector(vectors, self.dimension, i);
                subvectors.push(vec[*start..*end].to_vec());
            }

            // Spherical k-means (codewords are unit vectors, assignment by cosine
            // distance, matching `quantize`). Deterministic farthest-point init:
            // start from the first subvector, then repeatedly add the subvector
            // farthest from every codeword chosen so far.
            let normalized: Vec<Vec<f32>> = subvectors
                .iter()
                .map(|sub| crate::distance::normalize(sub))
                .collect();
            let mut codebook: Vec<Vec<f32>> = Vec::with_capacity(codebook_size);
            if let Some(first) = normalized.first() {
                codebook.push(first.clone());
                let mut nearest: Vec<f32> = normalized
                    .iter()
                    .map(|v| cosine_distance_normalized(v, first))
                    .collect();
                while codebook.len() < codebook_size {
                    let (far, _) =
                        nearest
                            .iter()
                            .enumerate()
                            .fold((0, f32::NEG_INFINITY), |best, (i, &d)| {
                                if d > best.1 { (i, d) } else { best }
                            });
                    let next = normalized[far].clone();
                    for (d, v) in nearest.iter_mut().zip(normalized.iter()) {
                        *d = d.min(cosine_distance_normalized(v, &next));
                    }
                    codebook.push(next);
                }
            } else {
                codebook = vec![vec![0.0; segment_dim]; codebook_size];
            }

            const KMEANS_ITERS: usize = 20;
            for _ in 0..KMEANS_ITERS {
                let mut sums = vec![vec![0.0f32; segment_dim]; codebook_size];
                let mut counts = vec![0usize; codebook_size];
                for sub in &subvectors {
                    let mut best = 0;
                    let mut best_dist = f32::INFINITY;
                    for (code, codeword) in codebook.iter().enumerate() {
                        let dist = cosine_distance_normalized(sub, codeword);
                        if dist < best_dist {
                            best_dist = dist;
                            best = code;
                        }
                    }
                    counts[best] += 1;
                    for (s, &x) in sums[best].iter_mut().zip(sub.iter()) {
                        *s += x;
                    }
                }
                let mut changed = false;
                for (code, sum) in sums.iter().enumerate() {
                    // Empty clusters keep their previous codeword.
                    if counts[code] == 0 {
                        continue;
                    }
                    let updated = crate::distance::normalize(sum);
                    if updated != codebook[code] {
                        codebook[code] = updated;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }

            self.codebooks.push(codebook);
        }

        Ok(())
    }

    /// Select the closest codeword in each segment. Ties use the first index.
    pub fn quantize(&self, vector: &[f32]) -> Vec<Vec<u8>> {
        let mut codes = Vec::new();

        for (segment_idx, (start, end)) in self.segment_bounds.iter().enumerate() {
            let segment = &vector[*start..*end];
            let codebook = &self.codebooks[segment_idx];

            // Find closest codeword
            let mut best_code = 0u8;
            let mut best_dist = f32::INFINITY;

            for (code, codeword) in codebook.iter().enumerate() {
                let dist = cosine_distance_normalized(segment, codeword);
                if dist < best_dist {
                    best_dist = dist;
                    best_code = code.min(255) as u8;
                }
            }

            codes.push(vec![best_code]);
        }

        codes
    }

    /// Compute approximate distance using quantized codes.
    pub fn approximate_distance(&self, query: &[f32], codes: &[Vec<u8>]) -> f32 {
        let mut total_dist = 0.0;

        for (segment_idx, (start, end)) in self.segment_bounds.iter().enumerate() {
            if let Some(code_vec) = codes.get(segment_idx) {
                if let Some(&code) = code_vec.first() {
                    let query_segment = &query[*start..*end];
                    if (code as usize) < self.codebooks[segment_idx].len() {
                        let codeword = &self.codebooks[segment_idx][code as usize];
                        total_dist += cosine_distance_normalized(query_segment, codeword);
                    }
                }
            }
        }

        total_dist
    }
}

/// Get vector from SoA storage.
fn get_vector(vectors: &[f32], dimension: usize, idx: usize) -> &[f32] {
    let start = idx * dimension;
    let end = start + dimension;
    &vectors[start..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// L2-normalize a vector in place, returning the original norm.
    fn normalize(v: &mut [f32]) -> f32 {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        norm
    }

    /// Build a set of L2-normalized training vectors (SoA layout).
    fn make_training_data(num_vectors: usize, dim: usize, seed: u64) -> Vec<f32> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(seed);
        let mut data = Vec::with_capacity(num_vectors * dim);
        for _ in 0..num_vectors {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
            normalize(&mut v);
            data.extend_from_slice(&v);
        }
        data
    }

    /// Two segments sharing 4 bits; variance-based allocation gives each segment
    /// at least 1 bit (2 codewords). Every training
    /// segment points near +e or -e, so trained codebooks must contain both
    /// directions and every vector must quantize to a codeword within a small
    /// cosine distance. Random codewords miss them. Fitting twice on the same
    /// data must give the same codebooks.
    #[test]
    fn saq_fit_trains_codebooks_on_the_data() {
        let dim = 8;
        let mut data = Vec::new();
        for i in 0..40usize {
            for seg in 0..2usize {
                let sign = if (i + seg) % 2 == 0 { 1.0 } else { -1.0 };
                let jitter = (i as f32 * 0.37).fract() * 0.1;
                let mut s = [sign, jitter, -jitter * 0.5, jitter * 0.25];
                normalize(&mut s);
                data.extend_from_slice(&s);
            }
        }
        let n = data.len() / dim;

        let mut q = SAQQuantizer::new(dim, 2, 4).unwrap();
        q.fit(&data, n).unwrap();
        for i in 0..n {
            let v = &data[i * dim..(i + 1) * dim];
            let codes = q.quantize(v);
            for (seg, (start, end)) in q.segment_bounds.iter().enumerate() {
                let codeword = &q.codebooks[seg][codes[seg][0] as usize];
                let d = cosine_distance_normalized(&v[*start..*end], codeword);
                assert!(d < 0.05, "vector {i} segment {seg}: cosine distance {d}");
            }
        }

        let mut again = SAQQuantizer::new(dim, 2, 4).unwrap();
        again.fit(&data, n).unwrap();
        assert_eq!(q.codebooks, again.codebooks);
    }

    #[test]
    fn allocation_preserves_remainder_and_never_exceeds_the_budget_after_refits() {
        let mut q = SAQQuantizer::new(6, 3, 5).unwrap();
        assert_eq!(q.bits_per_segment.iter().sum::<usize>(), 5);
        for _ in 0..3 {
            q.fit(
                &[
                    0.0, 0.0, -1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0,
                ],
                2,
            )
            .unwrap();
            assert_eq!(q.bits_per_segment.iter().sum::<usize>(), 5);
            assert_eq!(q.bits_per_segment[0], 0);
        }
        assert!(q.fit(&[], 0).is_err());
        assert!(q.fit(&[0.0], 1).is_err());
    }

    #[test]
    fn scalar_segments_track_variation_across_rows_and_reset_on_constant_data() {
        let mut q = SAQQuantizer::new(2, 2, 4).unwrap();
        q.fit(&[100.0, -1.0, 100.0, 1.0], 2).unwrap();
        assert_eq!(q.bits_per_segment, vec![0, 4]);
        assert!(q.fit(&[0.0, f32::NAN], 1).is_err());
        assert_eq!(q.bits_per_segment, vec![0, 4]);
        q.fit(&[100.0, 1.0, 100.0, 1.0], 2).unwrap();
        assert_eq!(q.bits_per_segment, vec![2, 2]);
    }

    #[test]
    fn saq_new_valid_params() {
        let q =
            SAQQuantizer::new(16, 4, 8).expect("SAQQuantizer::new must succeed for valid params");
        assert_eq!(q.dimension, 16);
        assert_eq!(q.num_segments, 4);
        assert_eq!(q.segment_bounds.len(), 4);
    }

    #[test]
    fn saq_new_rejects_zero_params() {
        assert!(SAQQuantizer::new(0, 4, 8).is_err());
        assert!(SAQQuantizer::new(16, 0, 8).is_err());
        assert!(SAQQuantizer::new(16, 4, 0).is_err());
    }

    #[test]
    fn saq_new_rejects_indivisible_dimension() {
        // 15 is not divisible by 4
        assert!(SAQQuantizer::new(15, 4, 8).is_err());
    }

    #[test]
    fn saq_encode_decode_roundtrip_finite() {
        let dim = 16;
        let num_segments = 4;
        let total_bits = 16;
        let num_train = 50;

        let data = make_training_data(num_train, dim, 42);
        let mut quantizer = SAQQuantizer::new(dim, num_segments, total_bits).unwrap();
        quantizer.fit(&data, num_train).unwrap();

        // Quantize each training vector and verify the approximate distance
        // is finite and non-negative. (The codebooks are random centroids,
        // not proper k-means clusters, so we cannot guarantee tight bounds.)
        for i in 0..num_train {
            let vec = get_vector(&data, dim, i);
            let codes = quantizer.quantize(vec);
            assert_eq!(
                codes.len(),
                num_segments,
                "code count must equal num_segments"
            );
            let self_dist = quantizer.approximate_distance(vec, &codes);
            assert!(
                self_dist.is_finite() && self_dist >= 0.0,
                "Self-distance must be finite and non-negative, got {} for vector {}",
                self_dist,
                i
            );
        }
    }

    #[test]
    fn saq_approximate_distance_closer_for_similar_vectors() {
        let dim = 8;
        let num_segments = 2;
        let total_bits = 8;
        let num_train = 30;

        let data = make_training_data(num_train, dim, 77);
        let mut quantizer = SAQQuantizer::new(dim, num_segments, total_bits).unwrap();
        quantizer.fit(&data, num_train).unwrap();

        // Take first vector as query, quantize all, check that self-distance
        // is <= distance to a random other vector (on average).
        let query = get_vector(&data, dim, 0);
        let self_codes = quantizer.quantize(query);
        let self_dist = quantizer.approximate_distance(query, &self_codes);

        let mut other_dists = Vec::new();
        for i in 1..num_train {
            let v = get_vector(&data, dim, i);
            let codes = quantizer.quantize(v);
            other_dists.push(quantizer.approximate_distance(query, &codes));
        }
        let avg_other: f32 = other_dists.iter().sum::<f32>() / other_dists.len() as f32;
        // Self-distance should generally be less than or equal to average distance
        // to other vectors.
        assert!(
            self_dist <= avg_other + 0.5,
            "Self-distance {} should be <= avg other distance {} (with margin)",
            self_dist,
            avg_other
        );
    }

    #[test]
    fn saq_single_vector() {
        let dim = 8;
        let num_segments = 2;
        let total_bits = 4;

        let mut v = vec![0.5, -0.3, 0.1, 0.7, -0.2, 0.4, -0.6, 0.8];
        normalize(&mut v);
        let data = v.clone();

        let mut quantizer = SAQQuantizer::new(dim, num_segments, total_bits).unwrap();
        quantizer.fit(&data, 1).unwrap();

        let codes = quantizer.quantize(&v);
        assert_eq!(codes.len(), num_segments);
        for code_vec in &codes {
            assert!(!code_vec.is_empty());
        }
    }

    #[test]
    fn saq_segment_bounds_correct() {
        let q = SAQQuantizer::new(12, 3, 6).unwrap();
        assert_eq!(q.segment_bounds, vec![(0, 4), (4, 8), (8, 12)]);
    }

    #[test]
    fn cosine_distance_fn_identical_vectors() {
        let v = vec![0.6, 0.8]; // already L2-normalized (0.36+0.64=1)
        let d = cosine_distance_normalized(&v, &v);
        assert!(
            d.abs() < 1e-5,
            "cosine distance to self should be ~0, got {}",
            d
        );
    }

    #[test]
    fn cosine_distance_fn_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let d = cosine_distance_normalized(&a, &b);
        // Cosine distance for opposite vectors = 1 - (-1) = 2
        assert!(
            (d - 2.0).abs() < 1e-5,
            "cosine distance for opposite vectors should be ~2.0, got {}",
            d
        );
    }

    #[test]
    fn get_vector_fn() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(get_vector(&data, 3, 0), &[1.0, 2.0, 3.0]);
        assert_eq!(get_vector(&data, 3, 1), &[4.0, 5.0, 6.0]);
        assert_eq!(get_vector(&data, 2, 2), &[5.0, 6.0]);
    }
}
