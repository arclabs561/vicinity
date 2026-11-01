//! Exact accounting for the optional distance counter.
//!
//! Deliberately one test in one binary: the counter is process-global, so a
//! second test thread would break the exact assertions. Run with
//! `cargo test --features distance-stats --test distance_stats`.

#![cfg(feature = "distance-stats")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use vicinity::distance::{self, DistanceMetric};
use vicinity::distance_stats::{reset_distance_count, take_distance_count};

#[test]
fn counts_distance_computations_per_metric() {
    let a = [0.6_f32, 0.8];
    let b = [0.8_f32, 0.6];

    // Dimension mismatch is rejected before any distance is computed.
    reset_distance_count();
    assert!(distance::cosine_distance(&a, &b[..1]).is_infinite());
    assert_eq!(take_distance_count(), 0, "mismatch must not be counted");

    let _ = distance::l2_distance(&a, &b);
    assert_eq!(take_distance_count(), 1, "l2_distance counts once");

    let _ = distance::l2_distance_squared(&a, &b);
    assert_eq!(take_distance_count(), 2, "l2_distance_squared counts once");

    let _ = distance::cosine_distance(&a, &b);
    assert_eq!(take_distance_count(), 3, "cosine_distance counts once");

    let _ = distance::cosine_distance_normalized(&a, &b);
    assert_eq!(take_distance_count(), 4, "normalized cosine counts once");

    let _ = distance::inner_product_distance(&a, &b);
    assert_eq!(take_distance_count(), 5, "inner product counts once");

    let _ = distance::angular_distance(&a, &b);
    assert_eq!(take_distance_count(), 6, "angular distance counts once");

    let _ = distance::normalize(&a);
    assert_eq!(take_distance_count(), 6, "normalize is not a distance");

    let _ = DistanceMetric::L2.distance(&a, &b);
    assert_eq!(
        take_distance_count(),
        7,
        "DistanceMetric dispatches counted"
    );

    reset_distance_count();
    assert_eq!(take_distance_count(), 0, "reset zeroes the counter");
}
