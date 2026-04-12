//! Vector operations with SIMD acceleration via innr.
//!
//! innr provides runtime-dispatched SIMD (AVX-512, AVX2+FMA, NEON)
//! with 4-way unrolled accumulators. Fallback when `innr` feature
//! is disabled uses portable scalar code.
//!
//! # Usage
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

#[cfg(feature = "innr")]
pub use innr::{cosine, dot, l2_distance, l2_distance_squared, norm};

#[cfg(not(feature = "innr"))]
pub use fallback::*;

#[cfg(not(feature = "innr"))]
mod fallback {
    const NORM_EPSILON: f32 = 1e-9;

    #[inline]
    #[must_use]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[inline]
    #[must_use]
    pub fn norm(v: &[f32]) -> f32 {
        dot(v, v).sqrt()
    }

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

    #[inline]
    #[must_use]
    pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        l2_distance_squared(a, b).sqrt()
    }

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
