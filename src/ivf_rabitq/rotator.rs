//! FHT/Kac random rotation: the RaBitQ reference default rotator.
//!
//! RaBitQ's error analysis assumes each vector is multiplied by a uniformly
//! random orthogonal matrix `R` before sign thresholding. Sampling and
//! applying a dense `D × D` matrix costs `O(D²)` work per vector and `O(D²)`
//! storage. The reference library instead uses the *FHT/Kac* structured
//! approximation, which is exactly orthogonal, costs `O(D log D)` work per
//! vector, and needs `4D` bits of sign state.
//!
//! Each of four Kac-walk rounds applies:
//!
//! 1. a Rademacher sign flip (one bit per coordinate per round);
//! 2. a normalized fast Hadamard transform on the first or last
//!    `trunc_dim` coordinates, alternating by round;
//! 3. a Givens rotation by `θ = π/4` pairing coordinate `i` with
//!    `i + padded_dim / 2` — skipped when `trunc_dim == padded_dim`,
//!    matching the reference's degenerate branch.
//!
//! `trunc_dim` is the largest power of two not exceeding the input
//! dimension; `padded_dim` is the input dimension rounded up to a multiple of
//! 32 (and at least [`MIN_PADDED_DIM`]). The transform is orthogonal, so
//! un-rotating is exact up to floating-point rounding: quantized residuals can
//! be decoded back into the original coordinate system.
//!
//! # References
//!
//! - RaBitQ-Library, `docs/docs/rabitq/rotator.md` (FFHT + Kac's walk).
//! - Andoni, Indyk, Laarhoven, Razensheyn, Schmidt (2015). "Practical and
//!   Optimal Feature Hashing with Tensors, Anisotropy and Sparsity."
//! - Gao, Long (2024). "RaBitQ: Quantizing High-Dimensional Vectors with a
//!   Theoretical Error Bound for Approximate Nearest Neighbor Search."

use crate::RetrieveError;

/// Number of sign-flip / Hadamard / Givens rounds in the Kac walk.
const KAC_ROUNDS: usize = 4;

/// Smallest working dimension. The reference rotator supports input
/// dimensions from 64 upward; smaller inputs are padded up to this floor.
const MIN_PADDED_DIM: usize = 64;

/// Working dimensions are rounded up to a multiple of this, so the padded
/// ranges are byte (and SIMD) aligned.
const PAD_ALIGN: usize = 32;

/// FHT/Kac random rotator.
///
/// Deterministic in `(dimension, seed)`: the same pair always yields the same
/// transform, and different seeds yield different sign sequences.
#[derive(Clone, Debug)]
pub struct FhtKacRotator {
    input_dim: usize,
    padded_dim: usize,
    trunc_dim: usize,
    /// `KAC_ROUNDS * padded_dim` Rademacher signs, packed one bit per
    /// coordinate (`4D` bits total).
    signs: Vec<u64>,
}

impl FhtKacRotator {
    /// Build a rotator for `dimension`-dimensional inputs.
    pub fn new(dimension: usize, seed: u64) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "rotation dimension must be > 0".into(),
            ));
        }

        let padded_dim = padded_dim_for(dimension);
        let trunc_dim = largest_power_of_two_at_most(dimension);
        let bit_words = KAC_ROUNDS * padded_dim / 64;
        let mut rng = SplitMix64::new(seed);
        let signs = (0..bit_words).map(|_| rng.next_u64()).collect();

        Ok(Self {
            input_dim: dimension,
            padded_dim,
            trunc_dim,
            signs,
        })
    }

    /// Input (un-padded) dimension this rotator accepts.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Working dimension of rotated vectors. Equal to the input dimension
    /// rounded up to a multiple of 32.
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// Rotate `input` into a freshly allocated padded vector.
    pub fn rotate(&self, input: &[f32]) -> Result<Vec<f32>, RetrieveError> {
        let mut out = vec![0.0f32; self.padded_dim];
        self.rotate_into(input, &mut out)?;
        Ok(out)
    }

    /// Rotate `input` into `out`, which must have length
    /// [`Self::padded_dim`].
    ///
    /// Input coordinates beyond `input_dim` are zero-padded before the
    /// transform, so the tail carries rotated energy rather than zeros.
    pub fn rotate_into(&self, input: &[f32], out: &mut [f32]) -> Result<(), RetrieveError> {
        if input.len() != self.input_dim {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: input.len(),
                doc_dim: self.input_dim,
            });
        }
        if out.len() != self.padded_dim {
            return Err(RetrieveError::InvalidParameter(format!(
                "rotation output must have length {}, got {}",
                self.padded_dim,
                out.len()
            )));
        }

        out.fill(0.0);
        out[..self.input_dim].copy_from_slice(input);
        self.transform_in_place(out);
        Ok(())
    }

    /// Invert [`Self::rotate`]: map a rotated padded vector back to the
    /// original coordinate system.
    ///
    /// The result has length [`Self::padded_dim`]; callers that know the
    /// input dimension take the leading `input_dim` coordinates.
    pub fn unrotate(&self, rotated: &[f32]) -> Result<Vec<f32>, RetrieveError> {
        if rotated.len() != self.padded_dim {
            return Err(RetrieveError::InvalidParameter(format!(
                "rotated vector must have length {}, got {}",
                self.padded_dim,
                rotated.len()
            )));
        }

        let mut out = rotated.to_vec();
        self.transform_inverse_in_place(&mut out);
        Ok(out)
    }

    /// Apply the four Kac rounds to a padded working vector.
    fn transform_in_place(&self, values: &mut [f32]) {
        debug_assert_eq!(values.len(), self.padded_dim);
        for round in 0..KAC_ROUNDS {
            self.flip_signs(round, values);
            self.hadamard_block(round, values);
            if self.trunc_dim != self.padded_dim {
                givens_pi_over_4(values, false);
            }
        }
    }

    /// Undo [`Self::transform_in_place`].
    fn transform_inverse_in_place(&self, values: &mut [f32]) {
        debug_assert_eq!(values.len(), self.padded_dim);
        for round in (0..KAC_ROUNDS).rev() {
            if self.trunc_dim != self.padded_dim {
                givens_pi_over_4(values, true);
            }
            self.hadamard_block(round, values);
            self.flip_signs(round, values);
        }
    }

    /// Whether the round's sign bit at `index` is set.
    #[inline]
    fn sign_is_negative(&self, round: usize, index: usize) -> bool {
        let bit = round * self.padded_dim + index;
        (self.signs[bit / 64] >> (bit % 64)) & 1 == 1
    }

    fn flip_signs(&self, round: usize, values: &mut [f32]) {
        for (index, value) in values.iter_mut().enumerate() {
            if self.sign_is_negative(round, index) {
                *value = -*value;
            }
        }
    }

    /// Normalized Hadamard transform over the first or last `trunc_dim`
    /// coordinates, alternating by round.
    fn hadamard_block(&self, round: usize, values: &mut [f32]) {
        let start = if round % 2 == 0 {
            0
        } else {
            self.padded_dim - self.trunc_dim
        };
        fht_normalized(&mut values[start..start + self.trunc_dim]);
    }
}

/// Working dimension used for `dimension`-dimensional inputs.
///
/// Kept separate from [`FhtKacRotator::new`] so persistence can validate a
/// saved padded dimension against the current padding rule.
pub fn padded_dim_for(dimension: usize) -> usize {
    dimension.max(MIN_PADDED_DIM).next_multiple_of(PAD_ALIGN)
}

fn largest_power_of_two_at_most(value: usize) -> usize {
    debug_assert!(value > 0);
    let mut power = 1;
    while power * 2 <= value {
        power *= 2;
    }
    power
}

/// In-place normalized fast Hadamard transform over a power-of-two slice.
///
/// The normalized Sylvester Hadamard matrix is symmetric and orthogonal, so
/// this transform is its own inverse.
fn fht_normalized(values: &mut [f32]) {
    debug_assert!(values.len().is_power_of_two());
    let n = values.len();
    let mut half = 1;
    while half < n {
        let step = half * 2;
        let mut block = 0;
        while block < n {
            for offset in 0..half {
                let a = values[block + offset];
                let b = values[block + offset + half];
                values[block + offset] = a + b;
                values[block + offset + half] = a - b;
            }
            block += step;
        }
        half = step;
    }

    let scale = 1.0 / (n as f32).sqrt();
    for value in values.iter_mut() {
        *value *= scale;
    }
}

/// Givens rotation by `±π/4` pairing the two halves of `values`.
///
/// `inverse` rotates by `-π/4`, undoing the forward pass.
fn givens_pi_over_4(values: &mut [f32], inverse: bool) {
    const INV_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let half = values.len() / 2;
    let sin = if inverse { -INV_SQRT_2 } else { INV_SQRT_2 };
    let cos = INV_SQRT_2;

    for index in 0..half {
        let a = values[index];
        let b = values[index + half];
        values[index] = a * cos - b * sin;
        values[index + half] = a * sin + b * cos;
    }
}

/// SplitMix64: small, explicitly specified generator so the sign sequence is
/// reproducible across platforms and crate versions.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn dot(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum()
    }

    fn test_vector(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..dim)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    /// The full padded transform matrix must satisfy `R Rᵀ = I`.
    ///
    /// Build the matrix column by column from every padded basis vector, so
    /// the padded tail (`>= input_dim`) is covered too.
    #[test]
    fn rotation_matrix_is_orthogonal() {
        for dim in [64, 96, 100, 128, 300] {
            let rotator = FhtKacRotator::new(dim, 1234).unwrap();
            let padded = rotator.padded_dim();
            let mut matrix = vec![0.0f64; padded * padded];

            for column in 0..padded {
                let mut basis = vec![0.0f32; padded];
                basis[column] = 1.0;
                rotator.transform_in_place(&mut basis);
                for row in 0..padded {
                    matrix[row * padded + column] = f64::from(basis[row]);
                }
            }

            // R Rᵀ must equal I; accumulated error stays well below 1e-5.
            for i in 0..padded {
                for j in 0..=i {
                    let mut acc = 0.0f64;
                    for k in 0..padded {
                        acc += matrix[i * padded + k] * matrix[j * padded + k];
                    }
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (acc - expected).abs() <= 1e-5,
                        "dim={dim} (RR^T)[{i}][{j}] = {acc}, expected {expected}"
                    );
                }
            }
        }
    }

    /// Rotation preserves inner products and norms for padded inputs.
    #[test]
    fn rotation_preserves_inner_products() {
        let dim = 96;
        let rotator = FhtKacRotator::new(dim, 7).unwrap();
        let a = test_vector(dim, 1);
        let b = test_vector(dim, 2);

        let ra = rotator.rotate(&a).unwrap();
        let rb = rotator.rotate(&b).unwrap();

        let norm_a = dot(&a, &a).sqrt();
        let norm_ra = dot(&ra, &ra).sqrt();
        assert!((norm_a - norm_ra).abs() <= 1e-5 * norm_a.max(1.0));

        let inner = dot(&a, &b);
        let rotated_inner = dot(&ra, &rb);
        assert!((inner - rotated_inner).abs() <= 1e-5 * inner.abs().max(1.0));
    }

    /// Un-rotating restores the original coordinates, including the padded
    /// tail returning to exactly zero.
    #[test]
    fn unrotate_inverts_rotation() {
        for dim in [64, 96, 100, 300] {
            let rotator = FhtKacRotator::new(dim, 99).unwrap();
            let input = test_vector(dim, 3);
            let rotated = rotator.rotate(&input).unwrap();
            let restored = rotator.unrotate(&rotated).unwrap();

            for (index, &expected) in input.iter().enumerate() {
                assert!(
                    (restored[index] - expected).abs() <= 1e-4,
                    "dim={dim} coordinate {index}: {} vs {expected}",
                    restored[index]
                );
            }
            for &tail in &restored[dim..] {
                assert!(tail.abs() <= 1e-4, "dim={dim} padded tail = {tail}");
            }
        }
    }

    #[test]
    fn rotation_is_deterministic_for_a_seed() {
        let dim = 96;
        let first = FhtKacRotator::new(dim, 42).unwrap();
        let repeated = FhtKacRotator::new(dim, 42).unwrap();
        let other = FhtKacRotator::new(dim, 43).unwrap();

        let input = test_vector(dim, 5);
        assert_eq!(
            first.rotate(&input).unwrap(),
            repeated.rotate(&input).unwrap()
        );
        assert_ne!(first.rotate(&input).unwrap(), other.rotate(&input).unwrap());
    }

    #[test]
    fn padded_dimension_follows_the_reference_alignment() {
        assert_eq!(padded_dim_for(64), 64);
        assert_eq!(padded_dim_for(96), 96);
        assert_eq!(padded_dim_for(769), 800);
        assert_eq!(padded_dim_for(32), 64);

        // trunc == padded: the reference skips the Givens step here.
        let square = FhtKacRotator::new(64, 0).unwrap();
        assert_eq!(square.trunc_dim, 64);
        assert_eq!(square.padded_dim(), 64);

        // trunc < padded: four rounds with FFHT + Givens.
        let padded = FhtKacRotator::new(96, 0).unwrap();
        assert_eq!(padded.trunc_dim, 64);
        assert_eq!(padded.padded_dim(), 96);
    }

    #[test]
    fn zero_dimension_is_rejected() {
        assert!(FhtKacRotator::new(0, 1).is_err());
    }

    #[test]
    fn wrong_lengths_are_rejected() {
        let rotator = FhtKacRotator::new(96, 1).unwrap();
        assert!(rotator.rotate(&[0.0; 95]).is_err());
        assert!(rotator.unrotate(&[0.0; 95]).is_err());
        let mut short = vec![0.0; 64];
        assert!(rotator.rotate_into(&[0.0; 96], &mut short).is_err());
    }
}
