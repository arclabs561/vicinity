//! Regression tests for search, filtering, persistence and LID estimator bugs:
//! distance-sign assumptions in pruning rules (ACORN, SparseMIPS, PRT), search
//! widths capped at k (InPlaceIndex, DEG, k-means tree), FilteredGraph's
//! attribute ordering, HNSW metadata added after vectors, re-entrant HNSW
//! search, WAL reopen, and TwoNN normalization and censoring.
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

fn queries() -> Vec<Vec<f32>> {
    clustered(100, D, 991)
}

fn mean_recall<F: Fn(&[f32], usize) -> Vec<(u32, f32)>>(data: &[Vec<f32>], search: F) -> f64 {
    let qs = queries();
    qs.iter()
        .map(|q| recall_of(&search(q, K), &brute(data, q, K)))
        .sum::<f64>()
        / qs.len() as f64
}

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

fn write_saved_neighbors(path: &std::path::Path, magic: &[u8; 8], nbrs: &[Vec<u32>]) {
    let mut b = magic.to_vec();
    b.extend_from_slice(&(nbrs.len() as u64).to_le_bytes());
    for l in nbrs {
        b.extend_from_slice(&(l.len() as u64).to_le_bytes());
        for &x in l {
            b.extend_from_slice(&x.to_le_bytes());
        }
    }
    std::fs::write(path, b).unwrap();
}

#[cfg(feature = "hnsw")]
mod acorn_sign {
    use super::*;
    use vicinity::hnsw::{acorn_search, AcornConfig, FnFilter};

    /// `acorn_search` takes any distance closure. Its frontier admission
    /// (`dist < worst * 2.0`) and stop rule (`current > worst * 1.5`) are
    /// multiplicative, which only works for non-negative distances. With the
    /// crate's own inner-product convention (`-dot`, negative for similar
    /// vectors) `worst * 2.0` is *below* every candidate, so the frontier stops
    /// growing as soon as k results exist. Same graph, same ranking (data is
    /// unit-norm, so `-dot` and `1 - dot` order identically): recall should
    /// not depend on adding a constant to the distance.
    #[test]
    fn acorn_recall_independent_of_distance_offset() {
        let n = 1500;
        let data = clustered(n, D, 7);
        // Exact 12-NN graph (by cosine), symmetric-ish, so search quality is
        // about the search routine only.
        let nbrs: Vec<Vec<u32>> = (0..n)
            .map(|i| {
                let mut v: Vec<(u32, f32)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (j as u32, 1.0 - dot(&data[i], &data[j])))
                    .collect();
                v.sort_by(|a, b| a.1.total_cmp(&b.1));
                let mut l: Vec<u32> = v.into_iter().take(12).map(|x| x.0).collect();
                // plus 4 random long-range edges so the graph is connected
                let mut r = Rng(1000 + i as u64);
                for _ in 0..4 {
                    l.push((r.next() % n as u64) as u32);
                }
                l
            })
            .collect();
        let filter = FnFilter(|id: u32| id.is_multiple_of(2));
        let cfg = AcornConfig::default();
        let queries = clustered(60, D, 991);
        let (mut r_cos, mut r_ip) = (0.0, 0.0);
        for q in &queries {
            let mut qn = q.clone();
            normalize(&mut qn);
            let mut all: Vec<(u32, f32)> = (0..n)
                .filter(|i| i.is_multiple_of(2))
                .map(|i| (i as u32, 1.0 - dot(&qn, &data[i])))
                .collect();
            all.sort_by(|a, b| a.1.total_cmp(&b.1));
            let want: Vec<u32> = all.iter().take(K).map(|x| x.0).collect();
            let cos = acorn_search(
                K,
                &cfg,
                &filter,
                |u| nbrs[u as usize].clone(),
                |u| 1.0 - dot(&qn, &data[u as usize]),
                0,
            )
            .unwrap();
            let ip = acorn_search(
                K,
                &cfg,
                &filter,
                |u| nbrs[u as usize].clone(),
                |u| -dot(&qn, &data[u as usize]),
                0,
            )
            .unwrap();
            r_cos += recall_of(&cos, &want);
            r_ip += recall_of(&ip, &want);
        }
        let (r_cos, r_ip) = (r_cos / 60.0, r_ip / 60.0);
        eprintln!("acorn recall: 1-dot {r_cos:.3}, -dot {r_ip:.3}");
        assert!(
            r_ip >= r_cos - 0.05,
            "ACORN recall with -dot {r_ip:.3} vs 1-dot {r_cos:.3}"
        );
    }
}

#[cfg(feature = "hnsw")]
mod inplace {
    use super::*;
    use vicinity::hnsw::{InPlaceConfig, InPlaceIndex};

    fn eval_live(idx: &InPlaceIndex, data: &[Vec<f32>], live: &[bool], ids: &[u32]) -> f64 {
        // InPlaceIndex is Euclidean; data are unit vectors so rankings agree.
        let queries = clustered(100, D, 991);
        let mut s = 0.0;
        for q in &queries {
            let mut all: Vec<(u32, f32)> = data
                .iter()
                .enumerate()
                .filter(|(i, _)| live[*i])
                .map(|(i, v)| {
                    (
                        ids[i],
                        v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum(),
                    )
                })
                .collect();
            all.sort_by(|a, b| a.1.total_cmp(&b.1));
            let want: Vec<u32> = all.iter().take(K).map(|x| x.0).collect();
            s += recall_of(&idx.search(q, K).unwrap(), &want);
        }
        s / queries.len() as f64
    }

    /// Search keeps a candidate only `if candidates.len() < beam_width`, so
    /// once the frontier holds beam_width entries every newly found neighbor
    /// is discarded regardless of distance (standard beam search keeps the
    /// best beam_width). Recall vs exact on the live set.
    #[test]
    fn inplace_recall_after_inserts_and_deletes() {
        let data = clustered(N, D, 7);
        let mut idx = InPlaceIndex::new(D, InPlaceConfig::default());
        let ids: Vec<u32> = data
            .iter()
            .map(|v| idx.insert(v.clone()).unwrap())
            .collect();
        let mut live = vec![true; N];
        let r0 = eval_live(&idx, &data, &live, &ids);
        let mut r = Rng(5);
        let mut del = 0;
        while del < N * 3 / 10 {
            let i = (r.next() % N as u64) as usize;
            if live[i] {
                idx.delete(ids[i]).unwrap();
                live[i] = false;
                del += 1;
            }
        }
        let r1 = eval_live(&idx, &data, &live, &ids);
        eprintln!("inplace: recall after inserts {r0:.3}, after 30% deletes {r1:.3}");
        assert!(r0 >= 0.9 && r1 >= 0.9, "recall {r0:.3} / {r1:.3}");
    }
}

#[cfg(feature = "hnsw")]
#[test]
fn deg_recall() {
    use vicinity::hnsw::{DEGConfig, DEGIndex};
    let data = clustered(N, D, 7);
    let mut idx = DEGIndex::new(D, DEGConfig::default());
    for v in &data {
        idx.add(v.clone()).unwrap();
    }
    idx.build().unwrap();
    let e = eval(&data, |q, k| idx.search(q, k).unwrap());
    report("deg", &e);
    assert!(
        e.recall >= 0.9 && e.self_hit >= 0.99,
        "deg recall {:.3} self {:.3}",
        e.recall,
        e.self_hit
    );
}

/// DEG search breaks once `visited.len() >= config.ef_search`, and `visited`
/// counts every scored neighbor (≈ degree per expansion), so a search
/// expands only a handful of nodes. Raising ef should restore recall.
#[cfg(feature = "hnsw")]
#[test]
fn deg_recall_vs_ef() {
    use vicinity::hnsw::{DEGConfig, DEGIndex};
    let data = clustered(N, D, 7);
    let mut idx = DEGIndex::new(D, DEGConfig::default());
    for v in &data {
        idx.add(v.clone()).unwrap();
    }
    idx.build().unwrap();
    let e100 = eval(&data, |q, k| idx.search_with_ef(q, k, 100).unwrap());
    let e2000 = eval(&data, |q, k| idx.search_with_ef(q, k, 2000).unwrap());
    eprintln!(
        "deg: ef=100 recall {:.3} self {:.3}; ef=2000 recall {:.3} self {:.3}",
        e100.recall, e100.self_hit, e2000.recall, e2000.self_hit
    );
    assert!(
        e100.recall >= 0.9,
        "ef=100 recall {:.3} (ef=2000: {:.3})",
        e100.recall,
        e2000.recall
    );
}

#[cfg(feature = "filtered_graph")]
mod filtered_graph {
    use super::*;
    use vicinity::filtered_graph::{
        AttrValue, Filter, FilteredGraphIndex, FilteredGraphParams, Predicate,
    };

    /// An `Eq` on an attribute whose values mix Int and Float (common when
    /// attributes come from JSON) binary-searches the inverted list, so the
    /// list must be sorted by a total order.
    #[test]
    fn filtered_graph_eq_with_mixed_numeric_types() {
        let data = clustered(40, 8, 3);
        let mut idx = FilteredGraphIndex::new(8, FilteredGraphParams::default()).unwrap();
        for (i, v) in data.iter().enumerate() {
            let mut a = HashMap::new();
            let val = if i.is_multiple_of(2) {
                AttrValue::Int(i as i64)
            } else {
                AttrValue::Float(i as f64 + 0.5)
            };
            a.insert("x".to_string(), val);
            idx.add(i as u32, v.clone(), a).unwrap();
        }
        idx.build().unwrap();
        let mut missing = Vec::new();
        for i in (0..40).step_by(2) {
            let f = Filter::Clause(Predicate::Eq("x".into(), AttrValue::Int(i as i64)));
            let got = idx.search_filtered(&data[i], 5, &f).unwrap();
            if !got.iter().any(|(id, _)| *id == i as u32) {
                missing.push(i);
            }
        }
        assert!(missing.is_empty(), "Eq(x, Int(i)) missed docs {missing:?}");
    }
}

#[cfg(feature = "hnsw")]
mod hnsw_filter {
    use super::*;
    use vicinity::filtering::{DocumentMetadata, MetadataFilter, MetadataValue};
    use vicinity::hnsw::HNSWIndex;

    /// Metadata attached with `add_metadata` after `add` (the order the
    /// `add_metadata` docs do not forbid, and the order ACORN supports) is
    /// invisible to `search_with_filter`: the category is snapshotted at add
    /// time, so the query errors with NotFound.
    #[test]
    fn hnsw_search_with_filter_sees_metadata_added_after_vectors() {
        let data = clustered(200, 8, 5);
        let mut idx = HNSWIndex::with_filtering(8, 16, 32, "group").unwrap();
        for (i, v) in data.iter().enumerate() {
            idx.add_slice(i as u32, v).unwrap();
        }
        for i in 0..data.len() {
            let mut m = DocumentMetadata::new();
            m.insert("group".into(), MetadataValue::from((i % 2) as u32));
            idx.add_metadata(i as u32, m).unwrap();
        }
        idx.build().unwrap();
        let f = MetadataFilter::equals("group", 0_u32);
        let r = idx.search_with_filter(&data[0], 5, 50, &f);
        assert!(
            matches!(&r, Ok(v) if !v.is_empty()),
            "search_with_filter after add_metadata: {r:?}"
        );
    }
}

#[cfg(feature = "kmeans_tree")]
mod kmeans_tree {
    use super::*;
    use vicinity::classic::trees::kmeans_tree::{KMeansTreeIndex, KMeansTreeParams};

    fn build(n: usize, params: KMeansTreeParams) -> (Vec<Vec<f32>>, KMeansTreeIndex) {
        let data = clustered(n, D, 7);
        let mut idx = KMeansTreeIndex::new(D, params).unwrap();
        for (i, v) in data.iter().enumerate() {
            idx.add(i as u32, v.clone()).unwrap();
        }
        idx.build().unwrap();
        (data, idx)
    }

    /// `search` used to descend to exactly one leaf
    /// (branch_budget = 1). Any internal node with more than max_leaf_size=50
    /// points is split into num_clusters=16 children, so a 51..800-point node
    /// produces leaves of 3..50 points and the query's single leaf often holds
    /// fewer than k. FLANN's hierarchical k-means search (Muja & Lowe 2009,
    /// Sec. 3.2 / 2014 TPAMI Alg. 4) keeps a priority queue of unexplored
    /// branches and continues until `checks` points have been examined, so it
    /// always returns k when n >= k.
    #[test]
    fn kmeans_tree_default_search_returns_k() {
        let (data, idx) = build(N, KMeansTreeParams::default());
        let mut short = 0;
        let mut min_len = usize::MAX;
        for q in queries() {
            let got = idx.search(&q, K).unwrap();
            min_len = min_len.min(got.len());
            if got.len() < K {
                short += 1;
            }
        }
        let r1 = mean_recall(&data, |q, k| idx.search(q, k).unwrap());
        let r8 = mean_recall(&data, |q, k| idx.search_with_leaf_budget(q, k, 8).unwrap());
        let r32 = mean_recall(&data, |q, k| idx.search_with_leaf_budget(q, k, 32).unwrap());
        eprintln!(
            "kmeans_tree: {short}/100 queries got < {K} (min {min_len}); recall default {r1:.3}, leaf_budget 8 {r8:.3}, 32 {r32:.3}"
        );
        assert_eq!(
            short, 0,
            "{short}/100 queries returned fewer than k={K} results (min {min_len}) with n={N}"
        );
    }

    /// Minimal: 60 points, defaults. The root is split into 16 clusters of
    /// ~4 points; k=10 returns ~4.
    #[test]
    fn kmeans_tree_small_index_returns_k() {
        let (data, idx) = build(60, KMeansTreeParams::default());
        let got = idx.search(&data[0], 10).unwrap();
        assert_eq!(
            got.len(),
            10,
            "n=60, k=10, got {} results: {got:?}",
            got.len()
        );
    }
}

#[cfg(feature = "sparse_mips")]
mod sparse_mips {
    use super::*;
    use vicinity::sparse_mips::{SparseMipsIndex, SparseMipsParams, SparseVector};

    /// Distances are `-dot` (negative for similar vectors), so a stop rule of
    /// `current > 1.5 * worst` fires below the worst candidate. Isolation: a
    /// path graph 0-1-...-(n-1), entry 0, where dot(query, v_j) = 10 + 0.1 j
    /// rises along the path. Every greedy step improves, so a best-first
    /// search with any ef reaches node n-1.
    #[test]
    fn sparse_mips_search_walks_improving_path() {
        let n = 40u32;
        let mut idx = SparseMipsIndex::new(SparseMipsParams {
            ef_search: 4,
            ..Default::default()
        });
        for j in 0..n {
            idx.add(
                j,
                SparseVector {
                    indices: vec![0, 1 + j],
                    values: vec![10.0 + 0.1 * j as f32, 1.0],
                },
            )
            .unwrap();
        }
        idx.build().unwrap();
        let dir = tempfile::tempdir().unwrap();
        idx.save_to_dir(dir.path()).unwrap();
        let path: Vec<Vec<u32>> = (0..n)
            .map(|j| {
                let mut l = Vec::new();
                if j > 0 {
                    l.push(j - 1)
                }
                if j + 1 < n {
                    l.push(j + 1)
                }
                l
            })
            .collect();
        write_saved_neighbors(&dir.path().join("neighbors.bin"), b"SPMIPSG1", &path);
        let mp = dir.path().join("manifest.json");
        let mut man: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&mp).unwrap()).unwrap();
        man["entry_point"] = 0.into();
        std::fs::write(&mp, serde_json::to_vec(&man).unwrap()).unwrap();
        let loaded = SparseMipsIndex::load_from_dir(dir.path()).unwrap();
        let q = SparseVector {
            indices: vec![0],
            values: vec![1.0],
        };
        let got = loaded.search(&q, 1).unwrap();
        assert_eq!(
            got[0].0,
            n - 1,
            "best is {} (dot {:.1}); got {got:?}",
            n - 1,
            10.0 + 0.1 * (n - 1) as f32
        );
    }
}

#[cfg(feature = "hnsw")]
mod prt {
    use super::*;
    use vicinity::hnsw::{HNSWIndex, HNSWParams};
    use vicinity::prt::ProbabilisticRoutingTest;
    use vicinity::DistanceMetric;

    fn run(metric: DistanceMetric, scale: f32) -> (f64, f64, usize) {
        let base = clustered(N, D, 7);
        let data: Vec<Vec<f32>> = base
            .iter()
            .map(|v| v.iter().map(|x| x * scale).collect())
            .collect();
        let params = HNSWParams {
            m: 16,
            m_max: 32,
            ef_construction: 200,
            ef_search: 50,
            metric,
            seed: Some(42),
            ..Default::default()
        };
        let mut idx = HNSWIndex::with_params(D, params).unwrap();
        for (i, v) in data.iter().enumerate() {
            idx.add_slice(i as u32, v).unwrap();
        }
        idx.build().unwrap();
        let mut prt = ProbabilisticRoutingTest::new(D, 16, Some(42));
        prt.project_database(idx.raw_vectors());
        let qs: Vec<Vec<f32>> = queries()
            .iter()
            .map(|v| v.iter().map(|x| x * scale).collect())
            .collect();
        let (mut rs, mut rp, mut full) = (0.0, 0.0, 0usize);
        for q in &qs {
            let t = brute(&base, q, K); // cosine truth on unit data == L2/IP truth
            rs += recall_of(&idx.search(q, K, 50).unwrap(), &t);
            let (r, f) = idx.search_prt(q, K, 50, &prt, 1.5, 0.95).unwrap();
            full += f;
            rp += recall_of(&r, &t);
        }
        (rs / qs.len() as f64, rp / qs.len() as f64, full / qs.len())
    }

    /// The PRT threshold compares a projected squared L2 distance with the
    /// metric's distance; both must be in the same units, so scaling the data
    /// (same L2 ranking) must not change recall.
    #[test]
    fn prt_search_is_scale_invariant_for_l2() {
        let (s1, p1, f1) = run(DistanceMetric::L2, 1.0);
        let (s30, p30, f30) = run(DistanceMetric::L2, 30.0);
        eprintln!("prt L2: scale 1 plain {s1:.3} prt {p1:.3} ({f1} full dists); scale 30 plain {s30:.3} prt {p30:.3} ({f30} full dists)");
        assert!((p30 - p1).abs() < 0.05, "PRT recall changes with data scale: {p1:.3} at x1 vs {p30:.3} at x30 (plain HNSW {s1:.3}/{s30:.3})");
    }

    /// Inner-product distances are negative; a non-negative projected
    /// distance compared with them would skip every neighbor once the beam is
    /// full.
    #[test]
    fn prt_search_inner_product_recall() {
        let (s, p, f) = run(DistanceMetric::InnerProduct, 1.0);
        eprintln!("prt IP: plain {s:.3} prt {p:.3} ({f} full dists)");
        assert!(
            p >= s - 0.05,
            "PRT recall {p:.3} vs plain {s:.3} under InnerProduct"
        );
    }
}

#[cfg(feature = "hnsw")]
mod reentrancy {
    use super::*;
    use vicinity::hnsw::HNSWIndex;

    /// `with_visited_set` keeps the thread-local
    /// visited set mutably borrowed while it runs the caller's distance
    /// closure. A `search_with_distance` closure that itself runs any HNSW
    /// search on the same thread (e.g. scoring against a second index)
    /// panics with "already borrowed".
    #[test]
    fn search_with_distance_closure_may_search_another_index() {
        let data = clustered(300, D, 7);
        let build = |seed_off: usize| {
            let mut idx = HNSWIndex::new(D, 8, 16).unwrap();
            for (i, v) in data.iter().enumerate().skip(seed_off) {
                idx.add_slice(i as u32, v).unwrap();
            }
            idx.build().unwrap();
            idx
        };
        let a = build(0);
        let b = build(1);
        let q = data[5].clone();
        let raw = a.raw_vectors().to_vec();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            a.search_with_distance(&q, 5, 20, &|q: &[f32], id: u32| {
                let _ = b.search(q, 1, 10);
                let v = &raw[id as usize * D..(id as usize + 1) * D];
                1.0 - dot(q, v)
            })
        }));
        assert!(
            r.is_ok(),
            "search_with_distance panicked when its closure ran another search"
        );
    }
}

#[cfg(feature = "persistence")]
mod wal {
    use std::sync::Arc;
    use vicinity::persistence::directory::{Directory, FsDirectory};
    use vicinity::persistence::wal::{WalEntry, WalReader, WalWriter};

    /// After a restart, `WalWriter::open` continues the existing WAL (a
    /// writer from `WalWriter::new` refuses appends there), and
    /// `flush_and_sync` makes entries durable.
    #[test]
    fn wal_writer_can_append_after_reopen() {
        let tmp = tempfile::tempdir().unwrap();
        let dir: Arc<dyn Directory> = Arc::new(FsDirectory::new(tmp.path()).unwrap());
        {
            let mut w = WalWriter::new(dir.clone());
            w.append(WalEntry::AddSegment {
                segment_id: 1,
                doc_count: 3,
            })
            .unwrap();
            w.flush_and_sync().unwrap();
        }
        // "restart"
        let mut w2 = WalWriter::open(dir.clone()).unwrap();
        let r = w2.append(WalEntry::AddSegment {
            segment_id: 2,
            doc_count: 4,
        });
        let _ = w2.flush_and_sync();
        let replayed = WalReader::new(dir).replay().map(|v| v.len());
        assert!(
            r.is_ok(),
            "append after reopen failed: {:?}; replay {:?}",
            r.err(),
            replayed
        );
        assert_eq!(replayed.unwrap(), 2);
    }
}

#[cfg(feature = "hnsw")]
mod twonn {
    use super::*;
    use vicinity::lid::{estimate_twonn, estimate_twonn_with_ci};

    /// Exact Pareto(1, d) ratios mu = U^(-1/d): the TwoNN model itself.
    fn pareto(d: f32, n: usize, seed: u64) -> Vec<f32> {
        let mut r = Rng(seed);
        (0..n).map(|_| r.unit().powf(-1.0 / d)).collect()
    }

    /// TwoNN's linear fit must normalise the empirical CDF by the full count,
    /// not the *trimmed* count
    /// (F = i / n_kept) and the comment says this matches scikit-dimension;
    /// skdim/id/_TwoNN.py uses `Femp = np.arange(1, 1 + n_keep) / N` (full N).
    /// With the default discard_fraction 0.1, F reaches ~1 at the kept tail
    /// and the through-origin fit overestimates d by ~45%.
    #[test]
    fn twonn_linear_recovers_dimension() {
        for d in [2.0f32, 5.0] {
            let mu = pareto(d, 20_000, 11);
            let est = estimate_twonn(&mu, 0.1);
            let est0 = estimate_twonn(&mu, 0.0);
            eprintln!("twonn linear d={d}: discard 0.1 -> {est:.3}, discard 0 -> {est0:.3}");
            assert!(
                (est - d).abs() / d < 0.1,
                "d={d}: estimate {est:.3} with discard 0.1 (discard 0: {est0:.3})"
            );
        }
    }

    /// The MLE must not compute d = n / sum(log mu) over the ratios left after
    /// dropping the largest 10%: the dropped tail is ignored instead of
    /// treated as censored (skdim 'mle': (k-1)/(sum log mu_(i) + (N-k) log mu_(k))),
    /// so the estimate is biased up and the 95% CI excludes the true d.
    #[test]
    fn twonn_mle_ci_covers_true_dimension() {
        let d = 5.0f32;
        let mu = pareto(d, 20_000, 12);
        let r = estimate_twonn_with_ci(&mu, 0.1);
        eprintln!(
            "twonn mle d=5: {:.3} CI [{:.3}, {:.3}] n={}",
            r.dimension, r.ci_lower, r.ci_upper, r.n_samples
        );
        assert!(
            r.ci_lower <= d && d <= r.ci_upper,
            "95% CI [{:.3}, {:.3}] excludes d=5 (estimate {:.3})",
            r.ci_lower,
            r.ci_upper,
            r.dimension
        );
    }
}
