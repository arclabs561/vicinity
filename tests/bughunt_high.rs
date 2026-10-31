//! Regression tests for five High-severity graph/tree search bugs found in
//! the October 2026 bug hunt (copied from the hunt's `tests/bughunt4.rs`):
//!
//! 1. NSG, FINGER, FreshGraph and FilteredGraph pruned a node's candidates
//!    with the node itself in the list (`nsg_recall_*`, `finger_recall`,
//!    `fresh_graph_recall`, `filtered_graph_recall`).
//! 2. `ensure_connectivity` proved undirected connectivity instead of
//!    reachability from the entry point
//!    (`nsg_graph_has_no_self_loops_and_is_reachable`).
//! 3. Ball tree compared an L2 bound with a cosine best-so-far
//!    (`balltree_recall`).
//! 4. SNG search stopped after popping k nodes (`sng_recall`,
//!    `sng_k1_matches_top1_of_k10`).
//! 5. FreshGraph `insert` of a live id left two live copies
//!    (`fresh_graph_delete_after_reinsert_removes_id`).
//!
//! Ground truth is brute-force cosine distance over L2-normalized data, so the
//! L2 and cosine rankings agree for every index regardless of its metric.
#![allow(
    dead_code,
    unused_imports,
    clippy::needless_range_loop,
    clippy::unwrap_used,
    clippy::expect_used
)]

use std::collections::{HashMap, HashSet};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }
    fn gauss(&mut self) -> f32 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos()
    }
}

fn normalize(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

/// `n` points around 20 Gaussian cluster centers, L2-normalized.
fn clustered(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut r = Rng(seed);
    let centers: Vec<Vec<f32>> = (0..20)
        .map(|_| (0..d).map(|_| r.gauss()).collect())
        .collect();
    (0..n)
        .map(|i| {
            let mut v: Vec<f32> = centers[i % 20]
                .iter()
                .map(|c| c + 0.35 * r.gauss())
                .collect();
            normalize(&mut v);
            v
        })
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn brute(data: &[Vec<f32>], q: &[f32], k: usize) -> Vec<u32> {
    let mut qn = q.to_vec();
    normalize(&mut qn);
    let mut v: Vec<(u32, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, x)| (i as u32, 1.0 - dot(&qn, x)))
        .collect();
    v.sort_by(|a, b| a.1.total_cmp(&b.1));
    v.into_iter().take(k).map(|(i, _)| i).collect()
}

fn recall_of(got: &[(u32, f32)], want: &[u32]) -> f64 {
    let g: HashSet<u32> = got.iter().map(|x| x.0).collect();
    want.iter().filter(|w| g.contains(w)).count() as f64 / want.len() as f64
}

const N: usize = 2000;
const D: usize = 32;
const K: usize = 10;

struct Eval {
    recall: f64,
    self_hit: f64,
    short: usize,
}

/// Mean recall@K over 100 held-out queries; fraction of 200 stored vectors
/// whose own id is the top-1 hit; number of queries that got < K results.
fn eval<F: Fn(&[f32], usize) -> Vec<(u32, f32)>>(data: &[Vec<f32>], search: F) -> Eval {
    let queries = clustered(100, D, 991);
    let mut rsum = 0.0;
    let mut short = 0;
    for q in &queries {
        let got = search(q, K);
        if got.len() < K {
            short += 1;
        }
        rsum += recall_of(&got, &brute(data, q, K));
    }
    let mut hits = 0;
    let mut tried = 0;
    for i in (0..data.len()).step_by(data.len() / 200) {
        tried += 1;
        let got = search(&data[i], 1);
        if got.first().map(|x| x.0) == Some(i as u32) {
            hits += 1;
        }
    }
    Eval {
        recall: rsum / queries.len() as f64,
        self_hit: hits as f64 / tried as f64,
        short,
    }
}

fn report(name: &str, e: &Eval) {
    eprintln!(
        "{name}: recall@{K}={:.3} self_hit={:.3} short_results={}",
        e.recall, e.self_hit, e.short
    );
}

// ─── NSG ──────────────────────────────────────────────────────────────────

#[cfg(feature = "nsg")]
mod nsg {
    use super::*;
    use vicinity::nsg::{NsgIndex, NsgParams};

    fn build(n: usize) -> (Vec<Vec<f32>>, NsgIndex) {
        let data = clustered(n, D, 7);
        let mut idx = NsgIndex::new(D, NsgParams::default()).unwrap();
        for (i, v) in data.iter().enumerate() {
            idx.add_slice(i as u32, v).unwrap();
        }
        idx.build().unwrap();
        (data, idx)
    }

    /// NSG paper (Fu et al. 2019, Alg. 2) prunes candidates of node p that
    /// exclude p itself. Here the beam search for p returns p at distance 0,
    /// mrng_prune keeps it first, and then every other candidate c fails
    /// `d(p,c) >= d(p,c)`; out-lists collapse to self-loops.
    #[test]
    fn nsg_recall_small_bruteforce_knn() {
        let (data, idx) = build(800);
        let e = eval(&data, |q, k| idx.search(q, k).unwrap());
        report("nsg n=800", &e);
        assert!(e.recall >= 0.9, "NSG recall@10 {:.3} < 0.9", e.recall);
        assert!(e.self_hit >= 0.99, "NSG self-hit {:.3} < 0.99", e.self_hit);
    }

    #[test]
    fn nsg_recall_nndescent() {
        let (data, idx) = build(N);
        let e = eval(&data, |q, k| idx.search(q, k).unwrap());
        report("nsg n=2000", &e);
        assert!(e.recall >= 0.9, "NSG recall@10 {:.3} < 0.9", e.recall);
        assert!(e.self_hit >= 0.99, "NSG self-hit {:.3} < 0.99", e.self_hit);
    }
}

macro_rules! graph_recall_test {
    ($feat:literal, $name:ident, $ty:path, $params:expr, $search:ident) => {
        #[cfg(feature = $feat)]
        #[test]
        fn $name() {
            let data = clustered(N, D, 7);
            let mut idx = <$ty>::new(D, $params).unwrap();
            for (i, v) in data.iter().enumerate() {
                idx.add(i as u32, v.clone()).unwrap();
            }
            idx.build().unwrap();
            let e = eval(&data, |q, k| idx.$search(q, k).unwrap());
            report(stringify!($name), &e);
            assert!(
                e.recall >= 0.9,
                "{} recall {:.3}",
                stringify!($name),
                e.recall
            );
            assert!(
                e.self_hit >= 0.99,
                "{} self-hit {:.3}",
                stringify!($name),
                e.self_hit
            );
        }
    };
}

graph_recall_test!(
    "finger",
    finger_recall,
    vicinity::finger::FingerIndex,
    vicinity::finger::FingerParams::default(),
    search
);
graph_recall_test!(
    "sng",
    sng_recall,
    vicinity::sng::SNGIndex,
    vicinity::sng::SNGParams::default(),
    search
);
graph_recall_test!(
    "fresh_graph",
    fresh_graph_recall,
    vicinity::fresh_graph::FreshGraphIndex,
    vicinity::fresh_graph::FreshGraphParams::default(),
    search
);
graph_recall_test!(
    "balltree",
    balltree_recall,
    vicinity::classic::trees::balltree::BallTreeIndex,
    vicinity::classic::trees::balltree::BallTreeParams::default(),
    search
);

/// Same check for FilteredGraph's unfiltered build path, which shared the
/// self-candidate prune with the indexes above (no attributes attached).
#[cfg(feature = "filtered_graph")]
#[test]
fn filtered_graph_recall() {
    use vicinity::filtered_graph::{FilteredGraphIndex, FilteredGraphParams};
    let data = clustered(N, D, 7);
    let mut idx = FilteredGraphIndex::new(D, FilteredGraphParams::default()).unwrap();
    for (i, v) in data.iter().enumerate() {
        idx.add(i as u32, v.clone(), HashMap::new()).unwrap();
    }
    idx.build().unwrap();
    let e = eval(&data, |q, k| idx.search(q, k).unwrap());
    report("filtered_graph_recall", &e);
    assert!(e.recall >= 0.9, "filtered_graph recall {:.3}", e.recall);
    assert!(
        e.self_hit >= 0.99,
        "filtered_graph self-hit {:.3}",
        e.self_hit
    );
}

#[cfg(feature = "fresh_graph")]
mod fresh_graph {
    use super::*;
    use vicinity::fresh_graph::{FreshGraphIndex, FreshGraphParams};

    fn build(data: &[Vec<f32>]) -> FreshGraphIndex {
        let mut idx = FreshGraphIndex::new(D, FreshGraphParams::default()).unwrap();
        for (i, v) in data.iter().enumerate() {
            idx.add_slice(i as u32, v).unwrap();
        }
        idx.build().unwrap();
        idx
    }

    /// `insert` with a doc_id that is already live (an update) creates a second
    /// live copy; `delete(id)` then tombstones only the first, so the id stays
    /// searchable after it was deleted.
    #[test]
    fn fresh_graph_delete_after_reinsert_removes_id() {
        let data = clustered(300, D, 7);
        let mut idx = build(&data);
        let newv = clustered(1, D, 4242).remove(0);
        idx.insert(5, &newv).unwrap();
        assert!(idx.delete(5).unwrap());
        let hits: Vec<u32> = idx
            .search(&newv, 20)
            .unwrap()
            .into_iter()
            .chain(idx.search(&data[5], 20).unwrap())
            .map(|x| x.0)
            .collect();
        assert!(
            !hits.contains(&5),
            "doc 5 still returned after delete: {hits:?}"
        );
    }
}

// ─── Graph structure via save_to_dir (magic | u64 n | (u64 len, u32 ids)*) ──

fn read_saved_neighbors(path: &std::path::Path) -> Vec<Vec<u32>> {
    let b = std::fs::read(path).unwrap();
    let mut p = 8;
    let rd64 = |p: &mut usize| {
        let v = u64::from_le_bytes(b[*p..*p + 8].try_into().unwrap());
        *p += 8;
        v as usize
    };
    let n = rd64(&mut p);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let len = rd64(&mut p);
        let mut l = Vec::with_capacity(len);
        for _ in 0..len {
            l.push(u32::from_le_bytes(b[p..p + 4].try_into().unwrap()));
            p += 4;
        }
        out.push(l);
    }
    out
}

fn reachable_from(nbrs: &[Vec<u32>], start: usize) -> usize {
    let mut seen = vec![false; nbrs.len()];
    let mut stack = vec![start];
    seen[start] = true;
    let mut c = 1;
    while let Some(u) = stack.pop() {
        for &v in &nbrs[u] {
            if !seen[v as usize] {
                seen[v as usize] = true;
                c += 1;
                stack.push(v as usize);
            }
        }
    }
    c
}

fn mean_degree(nbrs: &[Vec<u32>]) -> f64 {
    nbrs.iter().map(|l| l.len()).sum::<usize>() as f64 / nbrs.len() as f64
}

#[cfg(feature = "nsg")]
mod nsg_structure {
    use super::*;
    use vicinity::nsg::{NsgIndex, NsgParams};

    /// After build, no node should list itself as a neighbor, and every node
    /// must be reachable from the navigating node along directed edges (NSG
    /// paper Alg. 2 step "DFS from the navigating node ... link unreached").
    #[test]
    fn nsg_graph_has_no_self_loops_and_is_reachable() {
        for n in [800usize, N] {
            let data = clustered(n, D, 7);
            let mut idx = NsgIndex::new(D, NsgParams::default()).unwrap();
            for (i, v) in data.iter().enumerate() {
                idx.add_slice(i as u32, v).unwrap();
            }
            idx.build().unwrap();
            let dir = tempfile::tempdir().unwrap();
            idx.save_to_dir(dir.path()).unwrap();
            let nb = read_saved_neighbors(&dir.path().join("neighbors.bin"));
            let manifest: serde_json::Value =
                serde_json::from_slice(&std::fs::read(dir.path().join("manifest.json")).unwrap())
                    .unwrap();
            let medoid = manifest["medoid"].as_u64().unwrap() as usize;
            let self_loops = nb
                .iter()
                .enumerate()
                .filter(|(i, l)| l.contains(&(*i as u32)))
                .count();
            let only_self = nb
                .iter()
                .enumerate()
                .filter(|(i, l)| l.len() == 1 && l[0] == *i as u32)
                .count();
            let reach = reachable_from(&nb, medoid);
            eprintln!(
                "nsg n={n}: self_loops={self_loops} only_self={only_self} mean_deg={:.2} reachable={reach}/{n}",
                mean_degree(&nb)
            );
            assert_eq!(self_loops, 0, "n={n}: {self_loops} nodes list themselves");
            assert_eq!(reach, n, "n={n}: only {reach} nodes reachable from medoid");
        }
    }
}

/// SNG: a k=1 query returns the best of the first 100 vectors (the entry scan)
/// because the loop breaks once `results.len() >= k`, before expanding any
/// neighbor. Compare with the top-1 of a k=10 query.
#[cfg(feature = "sng")]
#[test]
fn sng_k1_matches_top1_of_k10() {
    use vicinity::sng::{SNGIndex, SNGParams};
    let data = clustered(600, D, 7);
    let mut idx = SNGIndex::new(D, SNGParams::default()).unwrap();
    for (i, v) in data.iter().enumerate() {
        idx.add(i as u32, v.clone()).unwrap();
    }
    idx.build().unwrap();
    let mut differ = 0;
    for i in (0..600).step_by(6) {
        let a = idx.search(&data[i], 1).unwrap();
        let b = idx.search(&data[i], 10).unwrap();
        if a.first().map(|x| x.0) != b.first().map(|x| x.0) {
            differ += 1;
        }
    }
    assert_eq!(
        differ, 0,
        "{differ}/100 top-1 results differ between k=1 and k=10"
    );
}
