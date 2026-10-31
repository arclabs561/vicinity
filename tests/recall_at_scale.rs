//! Recall@10 at N=5000, D=128 for NSG, FINGER and FreshGraph.
//!
//! The existing suite runs at N=300, D=32; the bug hunt reports 0.05-0.10 for
//! these families, so the difference has to be either scale/dimension or stale.

use std::collections::HashSet;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vicinity::distance::{cosine_distance, normalize};

const N: usize = 5_000;
const DIM: usize = 128;
const NQ: usize = 100;
const K: usize = 10;

fn dataset(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            normalize(
                &(0..DIM)
                    .map(|_| rng.random::<f32>() - 0.5)
                    .collect::<Vec<f32>>(),
            )
        })
        .collect()
}

fn ground_truth(query: &[f32], db: &[Vec<f32>]) -> Vec<u32> {
    let mut scored: Vec<(u32, f32)> = db
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, cosine_distance(query, v)))
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
    scored.into_iter().take(K).map(|(i, _)| i).collect()
}

fn recall(
    db: &[Vec<f32>],
    queries: &[Vec<f32>],
    search: impl Fn(&[f32]) -> Vec<(u32, f32)>,
) -> f32 {
    let mut total = 0.0;
    for q in queries {
        let expected: HashSet<u32> = ground_truth(q, db).into_iter().collect();
        let got: HashSet<u32> = search(q).into_iter().take(K).map(|(i, _)| i).collect();
        total += expected.intersection(&got).count() as f32 / K as f32;
    }
    total / queries.len() as f32
}

#[test]
fn recall_at_scale() {
    let db = dataset(N, 42);
    let queries = dataset(NQ, 1230);

    #[cfg(feature = "nsg")]
    {
        use vicinity::nsg::{NsgIndex, NsgParams};
        let mut ix = NsgIndex::new(DIM, NsgParams::default()).unwrap();
        for (i, v) in db.iter().enumerate() {
            ix.add_slice(i as u32, v).unwrap();
        }
        ix.build().unwrap();
        for ef in [32usize, 128, 256] {
            let r = recall(&db, &queries, |q| ix.search_with_ef(q, K, ef).unwrap());
            println!("nsg          ef={ef:3} recall@10={r:.3}");
        }
    }

    #[cfg(feature = "finger")]
    {
        use vicinity::finger::{FingerIndex, FingerParams};
        let mut ix = FingerIndex::new(DIM, FingerParams::default()).unwrap();
        for (i, v) in db.iter().enumerate() {
            ix.add_slice(i as u32, v).unwrap();
        }
        ix.build().unwrap();
        for ef in [32usize, 128, 256] {
            let r = recall(&db, &queries, |q| ix.search_with_ef(q, K, ef).unwrap());
            println!("finger       ef={ef:3} recall@10={r:.3}");
        }
    }

    #[cfg(feature = "fresh_graph")]
    {
        use vicinity::fresh_graph::{FreshGraphIndex, FreshGraphParams};
        let mut ix = FreshGraphIndex::new(DIM, FreshGraphParams::default()).unwrap();
        for (i, v) in db.iter().enumerate() {
            ix.add_slice(i as u32, v).unwrap();
        }
        ix.build().unwrap();
        for ef in [32usize, 128, 256] {
            let r = recall(&db, &queries, |q| ix.search_with_ef(q, K, ef).unwrap());
            println!("fresh_graph  ef={ef:3} recall@10={r:.3}");
        }
    }
}
