//! Vector operations with SIMD acceleration.
//!
//! Multiple backends are supported via feature flags:
//!
//! | Feature | Backend | Performance | Notes |
//! |---------|---------|-------------|-------|
//! | `innr` (default) | innr crate | 4-8x | Pure Rust, good baseline |
//! | `simsimd` | SimSIMD | Up to 200x | C bindings, best on modern CPUs |
//! | (none) | Portable | 1x | Fallback, no dependencies |
//!
//! # Feature Priority
//!
//! If multiple features are enabled, priority is: `simsimd` > `innr` > fallback
//!
//! # Usage
//!
//! For normalized embeddings, prefer `dot()` over `cosine()`.
//!
//! ```rust
//! use vicinity::simd::{dot, cosine, norm};
//!
//! let a = [1.0_f32, 0.0, 0.0];
//! let b = [0.707, 0.707, 0.0];
//!
//! let d = dot(&a, &b);
//! let c = cosine(&a, &b);
//! let n = norm(&a);
//! ```

// Priority 1: SimSIMD (fastest, but requires C bindings)
#[cfg(feature = "simsimd")]
mod simsimd_backend {
    use simsimd::SpatialSimilarity;

    /// Dot product using SimSIMD.
    #[inline]
    #[must_use]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        f32::dot(a, b)
            .map(|v| v as f32)
            .unwrap_or_else(|| super::fallback::dot(a, b))
    }

    /// Alias for `dot`.
    #[inline]
    #[must_use]
    pub fn dot_portable(a: &[f32], b: &[f32]) -> f32 {
        super::fallback::dot(a, b)
    }

    /// L2 norm of a vector.
    #[inline]
    #[must_use]
    pub fn norm(v: &[f32]) -> f32 {
        dot(v, v).sqrt()
    }

    /// Cosine similarity using SimSIMD.
    ///
    /// Note: SimSIMD's `cosine` returns **distance** (1 - cos_sim),
    /// so we convert to similarity here.
    #[inline]
    #[must_use]
    pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
        f32::cosine(a, b)
            .map(|v| 1.0 - v as f32)
            .unwrap_or_else(|| super::fallback::cosine(a, b))
    }

    /// L2 (Euclidean) distance using SimSIMD.
    #[inline]
    #[must_use]
    pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        // SimSIMD returns squared distance
        f32::sqeuclidean(a, b)
            .map(|d| (d as f32).sqrt())
            .unwrap_or_else(|| super::fallback::l2_distance(a, b))
    }

    /// L2 distance squared using SimSIMD.
    #[inline]
    #[must_use]
    pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
        f32::sqeuclidean(a, b)
            .map(|v| v as f32)
            .unwrap_or_else(|| super::fallback::l2_distance_squared(a, b))
    }
}

#[cfg(feature = "simsimd")]
pub use simsimd_backend::*;

// Priority 2: innr (pure Rust SIMD)
#[cfg(all(feature = "innr", not(feature = "simsimd")))]
pub use innr::{cosine, dot, dot_portable, l2_distance, l2_distance_squared, norm};

// Priority 3: Portable fallback (no dependencies)
#[cfg(not(any(feature = "innr", feature = "simsimd")))]
pub use fallback::*;

// Fallback implementation (only compiled when no SIMD backend is enabled)
#[cfg(not(any(feature = "innr", feature = "simsimd")))]
mod fallback {
    //! Portable fallback implementations when innr is not available.

    const NORM_EPSILON: f32 = 1e-9;

    /// Dot product of two vectors (portable implementation).
    #[inline]
    #[must_use]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Alias for `dot` (portable implementation).
    #[inline]
    #[must_use]
    pub fn dot_portable(a: &[f32], b: &[f32]) -> f32 {
        dot(a, b)
    }

    /// L2 norm of a vector.
    #[inline]
    #[must_use]
    pub fn norm(v: &[f32]) -> f32 {
        dot(v, v).sqrt()
    }

    /// Cosine similarity between two vectors.
    #[inline]
    #[must_use]
    pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let d = dot(a, b);
        let na = norm(a);
        let nb = norm(b);
        if na > NORM_EPSILON && nb > NORM_EPSILON {
            d / (na * nb)
        } else {
            0.0
        }
    }

    /// L2 (Euclidean) distance between two vectors.
    #[inline]
    #[must_use]
    pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        l2_distance_squared(a, b).sqrt()
    }

    /// L2 distance squared (faster when only comparing distances).
    #[inline]
    #[must_use]
    pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sparse operations (always local, innr doesn't provide these by default)
// ─────────────────────────────────────────────────────────────────────────────

/// Sparse dot product using sorted index arrays.
///
/// Computes the inner product of two sparse vectors represented as
/// parallel arrays of indices and values. Indices must be sorted.
#[inline]
#[must_use]
pub fn sparse_dot(a_indices: &[u32], a_values: &[f32], b_indices: &[u32], b_values: &[f32]) -> f32 {
    sparse_dot_portable(a_indices, a_values, b_indices, b_values)
}

/// Portable implementation of sparse dot product.
#[inline]
#[must_use]
pub fn sparse_dot_portable(
    a_indices: &[u32],
    a_values: &[f32],
    b_indices: &[u32],
    b_values: &[f32],
) -> f32 {
    let mut i = 0;
    let mut j = 0;
    let mut result = 0.0;

    while i < a_indices.len() && j < b_indices.len() {
        if a_indices[i] < b_indices[j] {
            i += 1;
        } else if a_indices[i] > b_indices[j] {
            j += 1;
        } else {
            result += a_values[i] * b_values[j];
            i += 1;
            j += 1;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_basic() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        let result = dot(&a, &b);
        assert!((result - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_norm() {
        let v = [3.0_f32, 4.0];
        assert!((norm(&v) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        assert!(cosine(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_l2_distance() {
        let a = [0.0_f32, 0.0];
        let b = [3.0_f32, 4.0];
        assert!((l2_distance(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_dot() {
        let a_idx = [0, 2, 5];
        let a_val = [1.0, 2.0, 3.0];
        let b_idx = [1, 2, 5];
        let b_val = [1.0, 4.0, 2.0];
        // Matches at indices 2 (2.0*4.0=8.0) and 5 (3.0*2.0=6.0)
        let result = sparse_dot(&a_idx, &a_val, &b_idx, &b_val);
        assert!((result - 14.0).abs() < 1e-6);
    }
}
