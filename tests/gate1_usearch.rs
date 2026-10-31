//! Gate 1 of the breadth kill test: HNSW at matched recall against `usearch`.
//!
//! Synthetic, seeded, one machine. Reports QPS at recall@10 for vicinity's
//! HNSW and for usearch at several search widths, so the matched-recall point
//! can be compared directly.

use std::collections::HashSet;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vicinity::distance::{cosine_distance, normalize};

const N: usize = 10_000;
const DIM: usize = 96;
const NQ: usize = 200;
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

fn ground_truth(query: &[f32], db: &[Vec<f32>]) -> HashSet<u32> {
    let mut scored: Vec<(u32, f32)> = db
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, cosine_distance(query, v)))
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
    scored.into_iter().take(K).map(|(i, _)| i).collect()
}

fn measure(
    name: &str,
    truth: &[HashSet<u32>],
    queries: &[Vec<f32>],
    search: impl Fn(&[f32]) -> Vec<u32>,
) {
    let mut total_recall = 0.0;
    let started = Instant::now();
    for (q, expected) in queries.iter().zip(truth) {
        let got: HashSet<u32> = search(q).into_iter().take(K).collect();
        total_recall += expected.intersection(&got).count() as f32 / K as f32;
    }
    let elapsed = started.elapsed().as_secs_f64();
    let qps = queries.len() as f64 / elapsed;
    println!(
        "{name:28} recall@10={:.3} qps={qps:.0}",
        total_recall / queries.len() as f32
    );
}

#[test]
fn gate1_hnsw_against_usearch() {
    let db = dataset(N, 11);
    let queries = dataset(NQ, 12);
    // Ground truth is computed once, outside every timed region.
    let truth: Vec<HashSet<u32>> = queries.iter().map(|q| ground_truth(q, &db)).collect();

    use vicinity::hnsw::HNSWIndex;
    let mut hnsw = HNSWIndex::new(DIM, 16, 32).unwrap();
    for (i, v) in db.iter().enumerate() {
        hnsw.add(i as u32, v.clone()).unwrap();
    }
    hnsw.build().unwrap();
    for ef in [64usize, 128, 256] {
        measure(&format!("vicinity hnsw ef={ef}"), &truth, &queries, |q| {
            hnsw.search(q, K, ef)
                .unwrap()
                .into_iter()
                .map(|(i, _)| i)
                .collect()
        });
    }

    let options = usearch::IndexOptions {
        dimensions: DIM,
        metric: usearch::MetricKind::Cos,
        connectivity: 16,
        expansion_add: 128,
        expansion_search: 128,
        ..Default::default()
    };
    let index = usearch::Index::new(&options).unwrap();
    index.reserve(N).unwrap();
    for (i, v) in db.iter().enumerate() {
        index.add(i as u64, v).unwrap();
    }
    for ef in [64usize, 128, 256] {
        index.change_expansion_search(ef);
        measure(&format!("usearch ef={ef}"), &truth, &queries, |q| {
            let matches = index.search(q, K).unwrap();
            matches.keys.iter().map(|k| *k as u32).collect()
        });
    }
}
