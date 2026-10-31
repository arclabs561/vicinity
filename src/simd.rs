//! Vector operations with SIMD acceleration via innr.
//!
//! innr provides runtime-dispatched SIMD (AVX-512, AVX2+FMA, NEON)
//! with 4-way unrolled accumulators. Fallback when `innr` feature
//! is disabled uses portable scalar code.
//!
//! This module is `pub(crate)` -- use [`crate::distance`] for the public API.
//!
//! # Internal usage
//!
//! ```ignore
//! use crate::simd::{dot, cosine, norm};
//!
//! let a = [1.0_f32, 0.0, 0.0];
//! let b = [0.707, 0.707, 0.0];
//!
//! let d = dot(&a, &b);
//! let c = cosine(&a, &b);
//! let n = norm(&a);
//! ```

// The backend is always reached through a `pub use` below, so enabling
// `distance-stats` can swap in counting wrappers without touching call sites.
#[cfg(feature = "innr")]
mod backend {
    #[allow(unused_imports)] // Subsets used depending on feature combination
    pub use innr::{cosine, dot, l2_distance, l2_distance_squared, norm};
}

#[cfg(not(feature = "innr"))]
mod backend {
    pub use super::fallback::*;
}

/// Counting wrappers for the `distance_stats` feature.
///
/// One distance computation is one vector-pair comparison, so each wrapper
/// records exactly one. `norm` is intentionally absent: a norm is not a
/// distance. The fallback backend is wrapped too, so the count is the same
/// with and without SIMD.
#[cfg(feature = "distance-stats")]
mod counted {
    #[inline]
    #[must_use]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        crate::distance_stats::note_distance();
        super::backend::dot(a, b)
    }

    /// Cosine similarity. Counted once even though it reads both vectors'
    /// norms internally.
    #[inline]
    #[must_use]
    pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
        crate::distance_stats::note_distance();
        super::backend::cosine(a, b)
    }

    #[inline]
    #[must_use]
    pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        crate::distance_stats::note_distance();
        super::backend::l2_distance(a, b)
    }

    #[inline]
    #[must_use]
    pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
        crate::distance_stats::note_distance();
        super::backend::l2_distance_squared(a, b)
    }

    pub use super::backend::norm;
}

#[cfg(feature = "distance-stats")]
pub use counted::*;

#[cfg(not(feature = "distance-stats"))]
pub use backend::*;

#[cfg(not(feature = "innr"))]
mod fallback {
    const NORM_EPSILON: f32 = 1e-9;

    /// Dot product of two f32 slices.
    #[inline]
    #[must_use]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// L2 norm of a f32 slice.
    #[inline]
    #[must_use]
    pub fn norm(v: &[f32]) -> f32 {
        dot(v, v).sqrt()
    }

    /// Cosine similarity between two f32 slices.
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

    /// L2 (Euclidean) distance between two f32 slices.
    #[inline]
    #[must_use]
    pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        l2_distance_squared(a, b).sqrt()
    }

    /// Squared L2 distance between two f32 slices.
    #[inline]
    #[must_use]
    pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }
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
}
