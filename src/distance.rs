//! Distance metrics for dense vectors.
//!
//! `vicinity` has multiple ANN algorithms, and not all of them currently support all metrics.
//! This module provides a single, shared definition of common *dense* distance metrics
//! and basic helper functions.
//!
//! ## Important nuance
//!
//! Some implementations (notably HNSW) use dot-product-based cosine distance for speed and
//! therefore expect inputs to be **L2-normalized**. In contrast, [`cosine_distance`]
//! here is defined as `1 - cos(a,b)` and computes norms when needed.

use crate::simd;

/// Distance metric for dense vectors.
///
/// This is primarily used for evaluation utilities and as a common vocabulary in docs.
/// Individual index implementations may hard-code a metric today.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance.
    L2,
    /// Cosine distance `1 - cos(a,b)`.
    Cosine,
    /// Angular distance `arccos(cos(a,b)) / pi`, in `[0,1]`.
    Angular,
    /// Inner product distance `-dot(a,b)` (for maximum inner product search).
    InnerProduct,
}

impl DistanceMetric {
    /// Compute distance between two vectors.
    ///
    /// If dimensions mismatch, this returns `f32::INFINITY` (so it is never selected as a
    /// nearest neighbor).
    #[inline]
    #[must_use]
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            DistanceMetric::L2 => l2_distance(a, b),
            DistanceMetric::Cosine => cosine_distance(a, b),
            DistanceMetric::Angular => angular_distance(a, b),
            DistanceMetric::InnerProduct => inner_product_distance(a, b),
        }
    }
}

/// L2 (Euclidean) distance.
#[inline]
#[must_use]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    simd::l2_distance(a, b)
}

/// Cosine distance `1 - cos(a,b)`.
///
/// This computes cosine similarity (including norms), so it does **not** require pre-normalized
/// vectors.
#[inline]
#[must_use]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    1.0 - simd::cosine(a, b).clamp(-1.0, 1.0)
}

/// Cosine distance for **L2-normalized** vectors.
///
/// This is equivalent to `1 - dot(a, b)` when both vectors are normalized. It is faster
/// than [`cosine_distance`] but returns nonsense if inputs are not normalized.
#[inline]
#[must_use]
pub fn cosine_distance_normalized(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    1.0 - simd::dot(a, b)
}

/// Angular distance `arccos(cos(a,b)) / pi`, in `[0,1]`.
#[inline]
#[must_use]
pub fn angular_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    let cos_sim = simd::cosine(a, b).clamp(-1.0, 1.0);
    cos_sim.acos() / std::f32::consts::PI
}

/// Inner product distance (negative dot product).
#[inline]
#[must_use]
pub fn inner_product_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    -simd::dot(a, b)
}

/// Normalize a vector to unit L2 norm.
#[inline]
#[must_use]
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let n = simd::norm(v);
    if n < 1e-10 {
        return vec![0.0; v.len()];
    }
    v.iter().map(|x| x / n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_distance_is_zero_for_identical() {
        let a = [1.0_f32, 2.0, 3.0];
        let d = cosine_distance(&a, &a);
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn cosine_distance_normalized_matches_dot() {
        let a = normalize(&[3.0_f32, 4.0]);
        let b = normalize(&[6.0_f32, 8.0]);
        let d1 = cosine_distance(&a, &b);
        let d2 = cosine_distance_normalized(&a, &b);
        assert!((d1 - d2).abs() < 1e-6);
    }
}
