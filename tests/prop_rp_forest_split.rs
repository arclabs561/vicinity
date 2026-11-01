//! The two-means split must not lose a neighbour.
//!
//! Two properties, both about the split rule rather than the traversal budget:
//!
//! 1. **Partition completeness.** With a budget of `n` the traversal can reach
//!    every leaf, so its answer must agree with exhaustive search. If a split
//!    dropped or duplicated an id, that id would never be scored at any budget.
//! 2. **No exclusion.** Every true top-`k` id must appear in the result. A
//!    bound that was optimistic would park a subtree holding a true neighbour
//!    behind subtrees that cannot beat it, and a finite budget would drop it.
//!
//! The node-level proof that `far_child_dot_bound` covers its subtree lives in
//! the unit tests next to the function (it needs the private tree shape); this
//! file checks the same claim from the outside, through the public API.
//!
//! Run with:
//! ```text
//! cargo test --release --features rptree --test prop_rp_forest_split
//! ```

#![cfg(feature = "rptree")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vicinity::classic::trees::rp_forest::{RPTreeParams, RpForestIndex, RpForestParams};

/// Dimensions spanning the leaf-size derivation (which starts at `2`) through
/// the shape the parity harness measures.
const DIMS: &[usize] = &[2, 3, 5, 7, 12, 16, 24, 32, 48, 96];

fn dataset(n: usize, dim: usize, seed: u64, scaled: bool) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let scale = if scaled {
                0.2 + 1.8 * rng.random::<f32>()
            } else {
                1.0
            };
            let mut v: Vec<f32> = (0..dim)
                .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
                .collect();
            if !scaled {
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                for x in &mut v {
                    *x /= norm;
                }
            }
            v
        })
        .collect()
}

/// Exact top-`k` by descending cosine similarity, accumulated in `f64` so the
/// ground truth does not depend on the index's `f32` arithmetic.
fn brute_force(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(f64, u32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dot: f64 = v
                .iter()
                .zip(query)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            (dot, i as u32)
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    scored.into_iter().take(k).map(|(_, id)| id).collect()
}

fn build(vectors: &[Vec<f32>], dim: usize, trees: usize) -> RpForestIndex {
    let params = RpForestParams {
        num_trees: trees,
        // `0` = the dimension-derived leaf size, i.e. the default rule.
        tree_params: RPTreeParams { max_leaf_size: 0 },
    };
    let mut index = RpForestIndex::new(dim, params).expect("new");
    for (i, v) in vectors.iter().enumerate() {
        index.add(i as u32, v.clone()).expect("add");
    }
    index.build().expect("build");
    index
}

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 32,
        ..ProptestConfig::default()
    })]

    /// Full budget is exhaustive: the split partition, the leaf rule and the
    /// traversal must together reach every vector, on unit and non-unit data.
    #[test]
    fn full_budget_matches_brute_force(
        dim in prop::sample::select(DIMS.to_vec()),
        n in 40usize..400,
        k in 1usize..12,
        seed in any::<u64>(),
        scaled in any::<bool>(),
    ) {
        let vectors = dataset(n, dim, seed, scaled);
        let queries = dataset(6, dim, seed ^ 0xC0FF_EE00, scaled);
        let index = build(&vectors, dim, 4);

        for query in &queries {
            let approx: Vec<u32> = index
                .search_with_budget(query, k, n)
                .expect("search")
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let exact = brute_force(&vectors, query, k);

            prop_assert_eq!(approx.len(), k, "returned {} of {} requested", approx.len(), k);
            let mut seen = approx.clone();
            seen.sort_unstable();
            seen.dedup();
            prop_assert_eq!(seen.len(), approx.len(), "duplicate ids in result");

            // The sharp claim: a true neighbour is never excluded.
            for id in &exact {
                prop_assert!(
                    approx.contains(id),
                    "dim={} n={} k={} scaled={}: true neighbour {} missing from {:?}",
                    dim, n, k, scaled, id, approx
                );
            }
        }
    }

    /// A finite budget must not return anything that is not a real vector, and
    /// results must stay ordered and unique: the cheap invariants that catch a
    /// split which invents or misfiles ids.
    #[test]
    fn budgeted_search_stays_well_formed(
        dim in prop::sample::select(DIMS.to_vec()),
        n in 60usize..300,
        k in 1usize..10,
        budget in 1usize..300,
        seed in any::<u64>(),
    ) {
        let vectors = dataset(n, dim, seed, false);
        let index = build(&vectors, dim, 3);
        let query = &dataset(1, dim, seed ^ 0x51ED, false)[0];

        let results = index.search_with_budget(query, k, budget).expect("search");
        prop_assert!(results.len() <= k);
        for (id, dist) in &results {
            prop_assert!((*id as usize) < n, "id {} out of range", id);
            prop_assert!(dist.is_finite(), "non-finite distance {}", dist);
            // Cosine distance of the indexed vectors, so it cannot be below the
            // exact minimum by more than `f32` noise.
            let exact = 1.0 - vectors[*id as usize]
                .iter()
                .zip(query)
                .map(|(a, b)| a * b)
                .sum::<f32>();
            prop_assert!((dist - exact).abs() < 1e-4, "{} != {}", dist, exact);
        }
        for w in results.windows(2) {
            prop_assert!(w[0].1 <= w[1].1, "results not sorted");
            prop_assert!(w[0].0 != w[1].0, "duplicate id");
        }
    }
}
