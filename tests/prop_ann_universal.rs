//! Universal ANN property tests across all algorithms.
//!
//! The suite needs the graph and IVF feature set listed in the gate below. The
//! gate lives on the module declaration instead of a file-level `#![cfg(...)]`,
//! so a partial feature set still compiles and runs a real check rather than
//! reporting success having compared nothing.

#[cfg(all(
    feature = "hnsw",
    feature = "nsw",
    feature = "diskann",
    feature = "emg",
    feature = "finger",
    feature = "fresh_graph",
    feature = "ivf_rabitq",
    feature = "nsg",
    feature = "pipnn",
    feature = "sng",
    feature = "vamana"
))]
#[path = "prop_ann_universal/tests_feature_gated.rs"]
mod tests_feature_gated;

#[cfg(not(all(
    feature = "hnsw",
    feature = "nsw",
    feature = "diskann",
    feature = "emg",
    feature = "finger",
    feature = "fresh_graph",
    feature = "ivf_rabitq",
    feature = "nsg",
    feature = "pipnn",
    feature = "sng",
    feature = "vamana"
)))]
#[test]
fn prop_ann_universal_suite_is_not_silently_empty() {
    // Always-true invariant, the first property these tests assert: ranking by
    // the ungated distance kernel yields ascending distances and unique ids.
    let query = [0.0f32, 0.0];
    let corpus = [[2.0f32, 0.0], [0.5f32, 0.0], [1.0f32, 0.0]];
    let mut ranked: Vec<(u32, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(id, v)| (id as u32, vicinity::distance::l2_distance(&query, v)))
        .collect();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    assert_eq!(ranked, vec![(1, 0.5), (2, 1.0), (0, 2.0)]);
    assert!(
        ranked.windows(2).all(|w| w[0].1 <= w[1].1),
        "results must be sorted by ascending distance"
    );

    eprintln!(
        "tests/prop_ann_universal.rs: enable hnsw,nsw,diskann,emg,finger,fresh_graph,ivf_rabitq,nsg,pipnn,sng,vamana to run the gated property suite"
    );
}
