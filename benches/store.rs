//! Benchmarks for the `store` feature (segstore-backed updatable HNSW index).
//!
//! Run: `cargo bench --features store --bench store`. Without the feature the
//! harness is an empty no-op so the target still compiles. Measures build
//! throughput, warm query latency (per-segment HNSW cached), and the cold
//! "rebuild every segment" cost -- the cost a delete that clears the whole cache
//! incurs, which the targeted-invalidation delete avoids (one segment instead).

// Benches legitimately unwrap on setup; the workspace lints deny
// clippy::unwrap_used, so opt this bench target out.
#![allow(clippy::unwrap_used)]

#[cfg(not(feature = "store"))]
fn main() {}

#[cfg(feature = "store")]
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
#[cfg(feature = "store")]
use durability::Directory;
#[cfg(feature = "store")]
use std::sync::Arc;

#[cfg(feature = "store")]
const N: usize = 8_000;
#[cfg(feature = "store")]
const DIM: usize = 64;
#[cfg(feature = "store")]
const FLUSH: usize = 1_000; // ~8 segments

#[cfg(feature = "store")]
fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[cfg(feature = "store")]
fn vec(state: &mut u64) -> Vec<f32> {
    (0..DIM)
        .map(|_| (xorshift(state) % 2000) as f32 / 1000.0 - 1.0)
        .collect()
}

#[cfg(feature = "store")]
fn fresh_store(warm: bool) -> (vicinity::store::UpdatableIndex, Vec<f32>) {
    use durability::MemoryDirectory;
    let mut s = 0x1234_5678_9abc_def0u64;
    let mut store =
        vicinity::store::UpdatableIndex::open(MemoryDirectory::arc(), FLUSH, DIM, 16, 32).unwrap();
    for i in 0..N {
        store.add(i as u32, &vec(&mut s)).unwrap();
    }
    store.checkpoint().unwrap();
    let q = vec(&mut s);
    if warm {
        let _ = store.search(&q, 10, 64);
    }
    (store, q)
}

#[cfg(feature = "store")]
fn benches(c: &mut Criterion) {
    let mut g = c.benchmark_group("store");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function("build", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = fresh_store(false);
            },
            BatchSize::SmallInput,
        )
    });

    let (warm, q) = fresh_store(true);
    g.bench_function("search_warm", |b| b.iter(|| warm.search(&q, 10, 64)));

    g.finish();
}

/// Build a checkpointed corpus into a fresh in-memory directory (sidecars
/// persisted), returning the directory and a query vector.
#[cfg(feature = "store")]
fn build_dir() -> (Arc<dyn Directory>, Vec<f32>) {
    build_dir_with_params(16, 32)
}

#[cfg(feature = "store")]
fn build_dir_with_params(m: usize, m_max: usize) -> (Arc<dyn Directory>, Vec<f32>) {
    use durability::MemoryDirectory;
    let dir: Arc<dyn Directory> = MemoryDirectory::arc();
    let mut s = 0x1234_5678_9abc_def0u64;
    let mut store =
        vicinity::store::UpdatableIndex::open(dir.clone(), FLUSH, DIM, m, m_max).unwrap();
    for i in 0..N {
        store.add(i as u32, &vec(&mut s)).unwrap();
    }
    store.checkpoint().unwrap();
    let q = vec(&mut s);
    (dir, q)
}

/// Remove the persisted HNSW sidecars so a reopen rebuilds every segment.
#[cfg(feature = "store")]
fn delete_sidecars(dir: &Arc<dyn Directory>) {
    for name in dir.list_dir("").unwrap_or_default() {
        if name.starts_with("segstore.idx.") {
            let _ = dir.delete(&name);
        }
    }
}

/// Replace persisted HNSW sidecars with unreadable bytes so reopen observes a
/// sidecar file, rejects it, then rebuilds and overwrites it.
#[cfg(feature = "store")]
fn corrupt_sidecars(dir: &Arc<dyn Directory>) {
    for name in dir.list_dir("").unwrap_or_default() {
        if name.starts_with("segstore.idx.") {
            dir.atomic_write(&name, b"not-a-vicinity-hnsw-sidecar")
                .unwrap();
        }
    }
}

/// The headline restart contrast: the first search after reopening a persisted
/// corpus, loading each per-segment HNSW from its sidecar vs rebuilding it from
/// the raw vectors. Same corpus, same query; the only difference is whether the
/// sidecars are present and valid. `load` reopens one fixed directory (loading
/// never writes); each rebuild/stale/corrupt sample gets a fresh directory so
/// write-through re-persist can't turn later samples into loads.
#[cfg(feature = "store")]
fn reopen(c: &mut Criterion) {
    let mut g = c.benchmark_group("store_reopen");
    g.throughput(Throughput::Elements(N as u64));
    g.sample_size(20); // each sample reopens; rebuild also rebuilds the corpus in setup

    let (dir_load, q) = build_dir();
    g.bench_function("first_search_load", |b| {
        b.iter_batched(
            || dir_load.clone(),
            |d| {
                let s = vicinity::store::UpdatableIndex::open(d, FLUSH, DIM, 16, 32).unwrap();
                s.search(&q, 10, 64)
            },
            BatchSize::SmallInput,
        )
    });

    g.bench_function("first_search_rebuild", |b| {
        b.iter_batched(
            || {
                let (d, _) = build_dir();
                delete_sidecars(&d);
                d
            },
            |d| {
                let s = vicinity::store::UpdatableIndex::open(d, FLUSH, DIM, 16, 32).unwrap();
                s.search(&q, 10, 64)
            },
            BatchSize::PerIteration,
        )
    });

    g.bench_function("first_search_stale_recipe", |b| {
        b.iter_batched(
            || {
                let (d, _) = build_dir_with_params(8, 16);
                d
            },
            |d| {
                let s = vicinity::store::UpdatableIndex::open(d, FLUSH, DIM, 16, 32).unwrap();
                s.search(&q, 10, 64)
            },
            BatchSize::PerIteration,
        )
    });

    g.bench_function("first_search_corrupt_sidecar", |b| {
        b.iter_batched(
            || {
                let (d, _) = build_dir();
                corrupt_sidecars(&d);
                d
            },
            |d| {
                let s = vicinity::store::UpdatableIndex::open(d, FLUSH, DIM, 16, 32).unwrap();
                s.search(&q, 10, 64)
            },
            BatchSize::PerIteration,
        )
    });
    g.finish();
}

#[cfg(feature = "store")]
fn ingest_fs(c: &mut Criterion) {
    // The extend() win is invisible on MemoryDirectory (flush is free); on a real
    // filesystem the per-item WAL flush is the cost extend amortizes into one batch
    // sync. add-per-item vs extend over the same vectors.
    use durability::FsDirectory;
    let mut g = c.benchmark_group("ingest_fs");
    let n = 4_000usize;
    g.throughput(Throughput::Elements(n as u64));
    let mk = |tag: &str| {
        let mut p = std::env::temp_dir();
        p.push(format!("vicinity-bench-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    };
    g.bench_function("add", |b| {
        b.iter_batched(
            || mk("add"),
            |p| {
                let mut s = 0x1234_5678_9abc_def0u64;
                let mut store = vicinity::store::UpdatableIndex::open(
                    FsDirectory::arc(&p).unwrap(),
                    FLUSH,
                    DIM,
                    16,
                    32,
                )
                .unwrap();
                for i in 0..n {
                    store.add(i as u32, &vec(&mut s)).unwrap();
                }
                let _ = std::fs::remove_dir_all(&p);
            },
            BatchSize::PerIteration,
        )
    });
    g.bench_function("extend", |b| {
        b.iter_batched(
            || mk("extend"),
            |p| {
                let mut s = 0x1234_5678_9abc_def0u64;
                let mut store = vicinity::store::UpdatableIndex::open(
                    FsDirectory::arc(&p).unwrap(),
                    FLUSH,
                    DIM,
                    16,
                    32,
                )
                .unwrap();
                store
                    .extend((0..n).map(|i| (i as u32, vec(&mut s))))
                    .unwrap();
                let _ = std::fs::remove_dir_all(&p);
            },
            BatchSize::PerIteration,
        )
    });
    g.finish();
}

#[cfg(feature = "store")]
criterion_group!(g, benches, ingest_fs, reopen);
#[cfg(feature = "store")]
criterion_main!(g);
