//! Optional per-run count of distance computations (`distance-stats` feature).
//!
//! Time is a bad unit for comparing ANN indexes across machines, and the ANN
//! benchmarking literature reports distance computations instead: "the number
//! of distance computations performed can indicate how effective an approach
//! is at reducing the amount of work" (Aumüller & Ceccarello, 2023). This
//! module makes that count observable.
//!
//! The counter is a single relaxed `AtomicU64`. It is compiled out unless the
//! non-default `distance-stats` feature is enabled: the hooks are
//! `#[cfg]`-gated out of the distance code entirely, and the two public
//! functions stay as empty `#[inline(always)]` stubs, so the default build
//! keeps the distance loops free of atomics and branches.
//!
//! # What is counted
//!
//! - One increment per vector-pair distance in `crate::simd`, the leaf that
//!   every dense metric goes through: `dot`, `cosine`, `l2_distance`,
//!   `l2_distance_squared`. `norm` is not a distance and is not counted.
//!   Because [`crate::distance`] delegates to those functions, the public
//!   metrics ([`crate::distance::cosine_distance`], [`crate::distance::l2_distance`],
//!   ...) are counted without a second hook.
//! - One increment per code scanned by IVF-PQ's ADC loop, where the distance is
//!   a table lookup inside the quantizer rather than a `simd` call.
//!
//! # What is not counted
//!
//! - Quantized scans of other families (IVF-AVQ, RaBitQ, binary, SQ4/SQ8, LSH)
//!   score codes through their quantizer crates; their exact rerank *is*
//!   counted.
//! - User-supplied closures (`search_with_distance`) and ground-truth scans
//!   computed by the benchmark harness itself.
//! - Build-time distance work is counted like query work; reset before the
//!   timed loop if you only want query cost.
//!
//! The counter is process-global, so with a parallel search (`parallel`
//! feature, rayon) the total is the sum across threads; per-thread
//! attribution is not provided.

#[cfg(feature = "distance-stats")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "distance-stats")]
static DISTANCE_COMPUTATIONS: AtomicU64 = AtomicU64::new(0);

/// Distance computations counted since the last [`reset_distance_count`].
///
/// Always `0` when the `distance-stats` feature is disabled.
#[cfg(feature = "distance-stats")]
#[must_use]
pub fn take_distance_count() -> u64 {
    DISTANCE_COMPUTATIONS.load(Ordering::Relaxed)
}

/// Distance computations counted since the last [`reset_distance_count`].
///
/// Always `0` when the `distance-stats` feature is disabled.
#[cfg(not(feature = "distance-stats"))]
#[inline(always)]
#[must_use]
pub fn take_distance_count() -> u64 {
    0
}

/// Reset the distance counter to zero.
///
/// Call before each measurement point; the counter is process-global, so any
/// other search running concurrently also contributes.
#[cfg(feature = "distance-stats")]
pub fn reset_distance_count() {
    DISTANCE_COMPUTATIONS.store(0, Ordering::Relaxed);
}

/// Reset the distance counter to zero.
///
/// A no-op when the `distance-stats` feature is disabled.
#[cfg(not(feature = "distance-stats"))]
#[inline(always)]
pub fn reset_distance_count() {}

/// Record one distance computation. Compiled to nothing without the feature.
#[cfg(feature = "distance-stats")]
#[inline]
pub(crate) fn note_distance() {
    DISTANCE_COMPUTATIONS.fetch_add(1, Ordering::Relaxed);
}

/// Record one distance computation. Compiled to nothing without the feature.
///
/// Nothing calls this when the feature is off -- the counting wrappers in
/// [`crate::simd`] are not built either -- which is the point: the default
/// build has no hook to optimise away.
#[cfg(not(feature = "distance-stats"))]
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn note_distance() {}

/// Record `n` distance computations. Compiled to nothing without the feature.
///
/// Batch form for scans that compute many distances per iteration; this keeps
/// the atomic traffic to one increment per cluster rather than one per vector.
///
/// Only quantized IVF-PQ scans use it, so it is dead code in builds without
/// `ivf_pq`.
#[cfg(feature = "distance-stats")]
#[allow(dead_code)]
#[inline]
pub(crate) fn note_distances(n: u64) {
    if n > 0 {
        DISTANCE_COMPUTATIONS.fetch_add(n, Ordering::Relaxed);
    }
}

/// Record `n` distance computations. Compiled to nothing without the feature.
#[cfg(not(feature = "distance-stats"))]
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn note_distances(_n: u64) {}

#[cfg(all(test, feature = "distance-stats"))]
mod tests {
    use super::*;

    // The counter is process-global and `cargo test` runs tests on several
    // threads, so a test running here can see increments from another test's
    // distances. These assertions are therefore one-sided (at least this many,
    // never fewer). Exact accounting lives in `tests/distance_stats.rs`, which
    // is a process of its own.
    #[test]
    fn counting_hook_adds_up() {
        let before = take_distance_count();
        note_distance();
        note_distance();
        note_distances(3);
        let after = take_distance_count();
        assert!(after >= before + 5, "counter went {before} -> {after}");
    }

    #[test]
    fn counted_functions_are_visible_through_distance() {
        let before = take_distance_count();
        let _ = crate::distance::l2_distance(&[0.0, 0.0], &[3.0, 4.0]);
        let _ = crate::distance::cosine_distance_normalized(&[1.0, 0.0], &[0.0, 1.0]);
        let _ = crate::distance::inner_product_distance(&[1.0, 0.0], &[0.0, 1.0]);
        let _ = crate::distance::normalize(&[3.0, 4.0]);
        let after = take_distance_count();
        // Three distances plus any concurrent test traffic; `normalize` adds none.
        assert!(after >= before + 3, "counter went {before} -> {after}");
    }
}
