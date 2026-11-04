//! Large-scale tier harness: beyond 10M points, where the dataset does not fit
//! comfortably in RAM and exact ground truth cannot be recomputed on every run.
//!
//! This is a sibling of `examples/all_families.rs`, not a mode of it. At the
//! 10k protocol the whole base set is a `Vec<Vec<f32>>`, ground truth is one
//! exact scan per domain, and every family is measured. Above 10M that design
//! does not survive:
//!
//! - the dataset is generated **to disk in bounded chunks** and then mapped
//!   (`memmap2`), so resident memory reflects the index rather than a second
//!   in-heap copy of the corpus;
//! - ground truth covers a **sampled query set** (200-1000 queries) and is
//!   computed once by a **parallel shard scan** (`rayon`, k-way merge), then
//!   cached under a key derived from the dataset identity and the query bytes;
//! - every phase has a **guardrail** instead of discovering the problem as an
//!   OOM, a full volume or a timeout, and every refusal names the resource:
//!   RAM (measured KiB/point times `--points` against the host's memory), disk
//!   (the volume's free GiB against the corpus plus the selected indexes),
//!   time (`--budget-seconds` against the phase estimate), and `--max-points`
//!   as the declared ceiling. Memory is checked first because a larger budget
//!   cannot move it.
//!
//! Two families are wired, selected with `--families`:
//!
//! - `hnsw` — built with `HNSWIndex::add_slice` from the mapped rows and
//!   searched from the same in-RAM object with `HNSWIndex::search`.
//! - `diskann` — built with `DiskANNIndex::add_slice` + `build_parallel` from
//!   the same mapped rows, persisted with `DiskANNIndex::save`, then the build
//!   object is **dropped** and the timed queries run through
//!   `DiskANNSearcher`, the block-cache reader over `graph.index` and
//!   `vectors.bin`. Its p50/QPS therefore include positional file reads and its
//!   search RSS is bounded by `--diskann-cache-mib`, not by the corpus. At a
//!   tier where the index still fits in RAM this measures the mechanism rather
//!   than disk pressure, and the run prints the cache budget and hit rate so
//!   the reader can see which it was.
//!
//! Neither family receives a `Vec<Vec<f32>>` corpus: every build streams
//! borrowed rows from `DatasetFile`, so resident memory after an unmapped build
//! is the index rather than the index plus a second copy of the base set.
//!
//! DiskANN scores squared L2 (`simd::l2_distance_squared`). Every stored row
//! and query in these domains is L2-normalized, and squared L2 is then a
//! monotone transform of cosine, so ranking — and recall against a cosine
//! ground truth — is the same measurement.
//!
//! The families that remain unwired (NSG, Vamana, IVF-PQ as a streamed corpus,
//! RaBitQ at a fixed memory budget, streaming/updatable indexes) are named in
//! `benchmarks/README.md` with the specific API or resource that stops each
//! one; the reasons are re-checked against `src/` rather than assumed.
//!
//! # On-disk dataset
//!
//! ```text
//! magic   : 8 bytes  b"VICDS01\0"
//! dim     : u32 LE
//! count   : u64 LE
//! data    : count * dim * f32 LE, row-major
//! ```
//!
//! Generation is resumable. Because each row's RNG is seeded from
//! `(dataset_seed, row_index)`, continuing at row `r` produces byte-identical
//! output to an uninterrupted run. A partially written final row is truncated
//! to the last complete row before continuing.
//!
//! # Ground-truth cache
//!
//! `gt/{domain}-n{count}-d{dim}-k{k}-q{queries}-h{query_hash:016x}.gt`, where
//! `query_hash` is FNV-1a 64 over the little-endian bytes of every query
//! component. The file repeats the key in its header and a mismatch is an
//! error, not a silent reuse.
//!
//! # Usage
//!
//! ```bash
//! # 100k: always affordable, and the tier where correctness is verified
//! # against a brute-force f32 and f64 reference.
//! cargo run --release --example large_scale --features "large-scale,hnsw,diskann" -- \
//!   --points 100k
//!
//! # 1M: requires a larger time budget and ~1.3 GiB of dataset + both indexes.
//! cargo run --release --example large_scale --features "large-scale,hnsw,diskann" -- \
//!   --points 1m --budget-seconds 7200 --json /tmp/large-1m.jsonl
//!
//! # Compute (or reuse) ground truth only, then exit. This is how a 100M run
//! # avoids repeating the scan for every family invocation.
//! cargo run --release --example large_scale --features "large-scale,hnsw,diskann" -- \
//!   --points 100m --max-points 100m --gt-only
//!
//! # Built-in checks: resumable generation and cached ground truth.
//! cargo run --release --example large_scale --features "large-scale,hnsw,diskann" -- \
//!   --self-test
//! ```
//!
//! `--features distance-stats` adds `dists_per_query` (compiled out otherwise,
//! and emitted as `null`).

// The library keeps `#![forbid(unsafe_code)]`. This example opts out at the
// crate root because `memmap2` maps a file only through an `unsafe` call; the
// call is confined to `DatasetFile::open` with its invariants documented.
#![allow(unsafe_code)]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

type Res<T> = Result<T, String>;

/// Dataset file magic. The trailing NUL keeps it eight bytes wide.
const MAGIC: [u8; 8] = *b"VICDS01\0";
const GT_MAGIC: [u8; 8] = *b"VICGT01\0";
/// `magic + dim (u32) + count (u64)`.
const DATA_HEADER_BYTES: u64 = 8 + 4 + 8;
/// `magic + dim (u32) + count, k, queries, query_hash (u64 x4)`.
const GT_HEADER_BYTES: usize = 8 + 4 + 8 + 8 + 8 + 8;

/// Rows generated per write. Bounds generation memory to `block * dim * 4`
/// bytes regardless of `--points`.
const GENERATION_BLOCK_ROWS: usize = 8192;
/// Base seed for the synthetic domains, matching the sibling harness's order of
/// magnitude but a distinct constant so the two harnesses never share a file.
const SEED: u64 = 0x4C41_5247_4553_4341; // "LARGESCA"
const QUERY_SALT: u64 = 0x5155_4552_5953_0001;
const HNSW_SEED: u64 = 42;
/// Queries timed with the exact scan for `inverse_ratio`. Exact p50 barely
/// moves across 16 queries, and 10M x 16 is already seconds.
const EXACT_TIMED_QUERIES: usize = 16;
const MIN_GT_QUERIES: usize = 200;
const MAX_GT_QUERIES: usize = 1000;

fn main() {
    match run() {
        Ok(()) => {}
        Err(message) => {
            eprintln!("large_scale: {message}");
            std::process::exit(2);
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration and guardrails
// ---------------------------------------------------------------------------

/// A family this harness can build from mapped rows and measure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    Hnsw,
    Diskann,
}

impl Family {
    fn parse(text: &str) -> Res<Self> {
        match text {
            "hnsw" => Ok(Self::Hnsw),
            "diskann" => Ok(Self::Diskann),
            other => Err(format!(
                "unknown family `{other}`; families wired at this tier: hnsw,diskann"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Hnsw => "hnsw",
            Self::Diskann => "diskann",
        }
    }
}

struct Args {
    points: u64,
    dim: usize,
    k: usize,
    gt_queries: usize,
    ef: Vec<usize>,
    families: Vec<Family>,
    /// Block-cache budget for the file-backed DiskANN searcher. `0` disables
    /// the cache, which makes every vector read a positional read.
    diskann_cache_mib: usize,
    domain: Domain,
    max_points: u64,
    budget_seconds: f64,
    gt_only: bool,
    /// Generate the dataset and stop. Used by `scripts/prepare_data.py`, which
    /// writes the checksum manifest without needing a second generator.
    prepare_only: bool,
    /// `None` = verify when the tier is small enough to afford it.
    verify_gt: Option<bool>,
    reuse_gt: bool,
    persist: bool,
    /// Vectors per parallel construction batch. Smaller batches see more of the
    /// graph while inserting and cost build time; 4096 is the library default
    /// and loses recall when a batch is a large share of the set.
    batch: usize,
    /// Build the graph as `K` independently built shards, one resident at a
    /// time, instead of one graph. Bounds the build peak to one shard, which is
    /// the only way a tier above this host's RAM is buildable locally.
    shards: usize,
    /// Partitions to probe per query in sharded mode (nearest centroid first).
    nprobe: Vec<usize>,
    json: Option<PathBuf>,
    self_test: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            points: 100_000,
            dim: 96,
            k: 10,
            gt_queries: 500,
            ef: vec![50, 100, 200],
            families: vec![Family::Hnsw, Family::Diskann],
            diskann_cache_mib: 64,
            domain: Domain::Uniform,
            // A deliberate ceiling: 20M fits this host's RAM (42 GiB at the
            // measured 2.09 KiB/point) and its disk, while 100M already needs an
            // explicit opt-in. It does not fit the default hour: a 20M build is
            // ~55 min at the measured 0.12-0.16 ms/point, so `--budget-seconds`
            // refuses it until the budget is raised deliberately.
            max_points: 20_000_000,
            budget_seconds: 3600.0,
            gt_only: false,
            prepare_only: false,
            verify_gt: None,
            reuse_gt: true,
            persist: true,
            batch: 4096,
            shards: 1,
            nprobe: vec![1, 2, 4],
            json: None,
            self_test: false,
        }
    }
}

fn run() -> Res<()> {
    let args = parse_args()?;
    if args.self_test {
        return self_test();
    }
    validate(&args)?;

    let root = data_root();
    let ctx = DomainContext::new(args.domain, SEED, args.dim);
    let dataset_path = dataset_path(&root, args.domain, args.points, args.dim);
    let queries = generate_queries(&ctx, args.gt_queries);
    let query_hash = query_hash(&queries);
    let gt_path = gt_path(
        &root,
        args.domain,
        args.points,
        args.dim,
        args.k,
        args.gt_queries,
        query_hash,
    );

    println!("# large_scale");
    println!();
    println!(
        "domain={} n={} dim={} k={} queries={} ef={:?} batch={} seed=0x{SEED:016X}",
        args.domain.name(),
        args.points,
        args.dim,
        args.k,
        args.gt_queries,
        args.ef,
        effective_batch(&args)
    );
    let family_names: Vec<&str> = args.families.iter().map(|family| family.name()).collect();
    println!(
        "families={} diskann-cache-mib={} distance-stats={}",
        family_names.join(","),
        args.diskann_cache_mib,
        cfg!(feature = "distance-stats")
    );
    println!("data root: {}", root.display());
    println!(
        "dataset: {} (memmap2; {} MiB mapped)",
        dataset_path.display(),
        bytes_to_mib(
            args.points
                .saturating_mul(args.dim as u64)
                .saturating_mul(4)
        )
    );
    println!(
        "ground truth: {} (query hash 0x{query_hash:016X})",
        gt_path.display()
    );
    println!();

    // Generation. Re-running resumes from the last complete row; the estimate
    // only counts the rows still missing.
    let completed = dataset_rows_on_disk(&dataset_path, args.points, args.dim)?;
    let estimate = estimate(&args, args.points - completed, gt_path.exists());
    println!(
        "estimated wall clock ~{:.0}s (generation {:.0}s, ground truth {:.0}s, hnsw build {:.0}s, \
         diskann build {:.0}s, persist {:.0}s) against --budget-seconds {:.0}s",
        estimate.total,
        estimate.generation,
        estimate.ground_truth,
        estimate.hnsw_build,
        estimate.diskann_build,
        estimate.persist,
        args.budget_seconds
    );
    println!(
        "note: query sweeps are not in the estimate. Each sweep stops at the budget, checked          per query, and rows for completed families are written as they land."
    );
    if let Some(refusal) = memory_refusal(&args) {
        return Err(refusal);
    }
    // `--prepare-only` writes the corpus and stops before any build, so the
    // build estimate cannot make it refuse.
    if estimate.total > args.budget_seconds && !args.prepare_only {
        return Err(format!(
            "refusing: time, not memory or disk. The estimated {:.0}s (generation {:.0}s, ground \
             truth {:.0}s, builds {:.0}s, persist {:.0}s) exceeds --budget-seconds {:.0}s. Raise \
             the budget deliberately, lower --points, or lower --gt-queries (currently {}).",
            estimate.total,
            estimate.generation,
            estimate.ground_truth,
            estimate.hnsw_build + estimate.diskann_build,
            estimate.persist,
            args.budget_seconds,
            args.gt_queries
        ));
    }

    let started = Instant::now();
    let generation = ensure_dataset(&ctx, &dataset_path, args.points, args.dim, &args.families)?;
    println!(
        "dataset {}: {} rows written, {} rows reused at {}, {:.1} MiB",
        if generation.generated > 0 {
            "ready"
        } else {
            "reused"
        },
        generation.generated,
        generation.resumed_from,
        dataset_path.display(),
        bytes_to_mib(generation.bytes)
    );
    println!();

    if args.prepare_only {
        // Stop before ground truth: preparing data must not depend on the query
        // set. The checksum manifest is written by scripts/prepare_data.py so
        // the format's provenance lives in one place.
        println!(
            "prepared: {} ({:.1} MiB, {} rows, dim {})",
            dataset_path.display(),
            bytes_to_mib(
                std::fs::metadata(&dataset_path)
                    .map(|m| m.len())
                    .unwrap_or(0)
            ),
            args.points,
            args.dim
        );
        return Ok(());
    }

    // Ground truth: cached across invocations, computed by a parallel shard
    // scan on a miss.
    let dataset = DatasetFile::open(&dataset_path)?;
    let (truth, cache_state) =
        load_or_compute_ground_truth(&dataset, &queries, args.k, &gt_path, args.reuse_gt)?;
    match cache_state {
        CacheState::Hit => println!("ground truth: cache hit — scan not repeated"),
        CacheState::Computed => println!("ground truth: computed and cached"),
        CacheState::Recomputed => println!("ground truth: --regenerate-gt, recomputed and cached"),
    }

    let verify = args
        .verify_gt
        .unwrap_or(args.points <= 1_000_000 && !args.gt_only);
    if verify {
        verify_ground_truth(&dataset, &queries, &truth, args.k)?;
    }
    drop(dataset);

    if args.gt_only {
        println!("--gt-only: stopping before any index is built");
        return Ok(());
    }

    if args.shards > 1 {
        return run_sharded(dataset_path.as_path(), &queries, &truth, &args);
    }

    let elapsed = started.elapsed().as_secs_f64();
    if elapsed > args.budget_seconds {
        return Err(format!(
            "stopped after ground truth: {elapsed:.0}s already exceeds --budget-seconds {:.0}s",
            args.budget_seconds
        ));
    }

    let dataset = DatasetFile::open(&dataset_path)?;
    let exact_p50 = exact_p50_us(&dataset, &queries, args.k);
    drop(dataset);
    println!("exact linear-scan p50: {exact_p50:.1} us/query");
    println!();

    let mut rows: Vec<Row> = Vec::new();
    // Rows are flushed after every family, not only at the end: a tier run can
    // hit the wall-clock budget inside a file-backed sweep, and the work already
    // done must survive that. The per-query deadline is what stops a single
    // sweep from consuming the whole budget between two of these flushes.
    let json_path = match &args.json {
        Some(path) if !path.as_os_str().is_empty() => path.clone(),
        _ => root.join("rows").join(default_rows_name(&args)),
    };
    let deadline = started + std::time::Duration::from_secs_f64(args.budget_seconds);
    const BUDGET_HIT: &str =
        "stopping at --budget-seconds; rows for the families already measured are written";

    let mut index_disk_mib_total = 0.0f64;
    let batch = effective_batch(&args);
    let exact_timed = exact_p50.max(f64::MIN_POSITIVE);
    let hnsw_path = index_path(&root, args.domain, args.points, args.dim, batch);
    let diskann_dir = diskann_index_dir(&root, args.domain, args.points, args.dim, batch);

    'families: for family in &args.families {
        match family {
            Family::Hnsw => {
                let rss_before = rss_mib();
                let build_start = Instant::now();
                let index = build_hnsw(&dataset_path, args.points, args.dim, batch)?;
                let build_s = build_start.elapsed().as_secs_f64();
                let rss_after = rss_mib();
                println!(
                    "hnsw build: {build_s:.1}s; RSS before {:.1} MiB, after {:.1} MiB (dataset \
                     unmapped before both samples)",
                    rss_before.unwrap_or(f64::NAN),
                    rss_after.unwrap_or(f64::NAN),
                );
                let (index_disk_mib, persist_s) = if args.persist {
                    let persist_start = Instant::now();
                    let bytes = persist_index(&index, &hnsw_path)?;
                    let persist_s = persist_start.elapsed().as_secs_f64();
                    let mib = bytes_to_mib(bytes);
                    println!(
                        "hnsw index persisted: {} ({mib:.1} MiB) in {persist_s:.1}s",
                        hnsw_path.display()
                    );
                    (mib, persist_s)
                } else {
                    println!(
                        "hnsw index not persisted (--no-persist); resident-size estimate {:.1} \
                         MiB (not an on-disk measurement)",
                        bytes_to_mib(index.memory_usage().total() as u64)
                    );
                    // No file was written, so the on-disk size is unmeasured.
                    // Emitting the resident estimate under this column would
                    // read as a disk measurement.
                    (f64::NAN, 0.0)
                };
                let report = BuildReport {
                    build_s,
                    mem_mib: bytes_to_mib(index.memory_usage().total() as u64),
                    rss_mib: rss_after.unwrap_or(f64::NAN),
                    index_disk_mib,
                    persist_s,
                };
                if report.index_disk_mib.is_finite() {
                    index_disk_mib_total += report.index_disk_mib;
                }
                for &ef in &args.ef {
                    if let Some(refusal) = budget_refusal(&args, started, family, ef) {
                        println!("{refusal}");
                        break 'families;
                    }
                    let Some(stats) = evaluate(
                        |query, k, ef| index.search(query, k, ef).map_err(string_err),
                        &queries,
                        &truth,
                        args.k,
                        ef,
                        Some(deadline),
                    ) else {
                        println!("{BUDGET_HIT}");
                        break 'families;
                    };
                    push_rows(
                        &mut rows,
                        family.name(),
                        "ef_search",
                        ef,
                        stats,
                        &report,
                        &args,
                        verify,
                        &gt_path,
                        exact_timed,
                    );
                }
            }
            Family::Diskann => {
                // The build is in RAM (no family here can stream a graph to
                // disk); the measured queries then run from files, with RSS
                // bounded by the block cache rather than by the corpus.
                let rss_before = rss_mib();
                let build_start = Instant::now();
                let index = build_diskann(&dataset_path, args.points, args.dim, batch)?;
                let build_s = build_start.elapsed().as_secs_f64();
                let rss_after = rss_mib();
                println!(
                    "diskann build: {build_s:.1}s; RSS before {:.1} MiB, after {:.1} MiB (in-RAM \
                     Vamana construction from the mapped corpus; the build object is dropped \
                     before any query runs)",
                    rss_before.unwrap_or(f64::NAN),
                    rss_after.unwrap_or(f64::NAN),
                );
                ensure_index_space(&diskann_dir, estimate_diskann_bytes(&args))?;
                let persist_start = Instant::now();
                let bytes = persist_diskann(&index, &diskann_dir)?;
                let persist_s = persist_start.elapsed().as_secs_f64();
                let index_disk_mib = bytes_to_mib(bytes);
                println!(
                    "diskann index persisted: {} ({index_disk_mib:.1} MiB) in {persist_s:.1}s",
                    diskann_dir.display()
                );
                let report = BuildReport {
                    build_s,
                    mem_mib: bytes_to_mib(index.size_bytes() as u64),
                    rss_mib: rss_after.unwrap_or(f64::NAN),
                    index_disk_mib,
                    persist_s,
                };
                index_disk_mib_total += index_disk_mib;

                // Same graph and the same build: one sweep against the resident
                // object, one against the files. Publishing only the file row
                // would read as "DiskANN is slow" when what is slow is the
                // storage path.
                let mut in_ram_recall: Vec<(usize, f64)> = Vec::new();
                for &ef in &args.ef {
                    if let Some(refusal) = budget_refusal(&args, started, family, ef) {
                        println!("{refusal}");
                        break 'families;
                    }
                    let Some(stats) = evaluate(
                        |query, k, ef| index.search(query, k, ef).map_err(string_err),
                        &queries,
                        &truth,
                        args.k,
                        ef,
                        Some(deadline),
                    ) else {
                        println!("{BUDGET_HIT}");
                        break 'families;
                    };
                    in_ram_recall.push((ef, stats.recall));
                    push_rows(
                        &mut rows,
                        family.name(),
                        "ef_search",
                        ef,
                        stats,
                        &report,
                        &args,
                        verify,
                        &gt_path,
                        exact_timed,
                    );
                }
                drop(index);

                let cache_bytes = args.diskann_cache_mib.saturating_mul(1 << 20);
                let mut searcher = vicinity::diskann::DiskANNSearcher::load_with_cache(
                    &diskann_dir,
                    vicinity::FileCacheConfig::with_budget(cache_bytes),
                )
                .map_err(string_err)?;
                let search_rss = rss_mib();
                println!(
                    "diskann file-backed search: block cache {} MiB, RSS after open {:.1} MiB, \
                     index files {index_disk_mib:.1} MiB ({} the cache budget)",
                    args.diskann_cache_mib,
                    search_rss.unwrap_or(f64::NAN),
                    if (cache_bytes as u64) < bytes {
                        "exceed"
                    } else {
                        "fit in"
                    }
                );
                let file_report = BuildReport {
                    rss_mib: search_rss.unwrap_or(f64::NAN),
                    ..report
                };
                let first_file_row = rows.len();
                for &ef in &args.ef {
                    if let Some(refusal) = budget_refusal(&args, started, family, ef) {
                        println!("{refusal}");
                        break 'families;
                    }
                    let Some(stats) = evaluate(
                        |query, k, ef| searcher.search(query, k, ef).map_err(string_err),
                        &queries,
                        &truth,
                        args.k,
                        ef,
                        Some(deadline),
                    ) else {
                        println!("{BUDGET_HIT}");
                        break 'families;
                    };
                    if let Some((_, resident)) =
                        in_ram_recall.iter().find(|(value, _)| *value == ef)
                    {
                        println!(
                            "diskann ef_search={ef}: file-backed recall {:.3} vs resident {:.3}",
                            stats.recall, resident
                        );
                    }
                    push_rows(
                        &mut rows,
                        "diskann-file",
                        "ef_search",
                        ef,
                        stats,
                        &file_report,
                        &args,
                        verify,
                        &gt_path,
                        exact_timed,
                    );
                }
                // Opening the searcher reuses pages the build object no longer
                // references; the kernel returns them lazily, so the RSS right
                // after open is the build's high-water mark, not what the
                // queries ran in. Publish the steady-state sample.
                if let Some(rss) = rss_mib() {
                    for row in rows.iter_mut().skip(first_file_row) {
                        row.rss_mib = rss;
                    }
                }
                let cache = searcher.cache_stats();
                println!(
                    "diskann block cache after the sweep: hit rate {:.1}% ({} hits / {} misses), \
                     resident {:.1} of {:.1} MiB; RSS after search {:.1} MiB",
                    cache.hit_rate() * 100.0,
                    cache.hits,
                    cache.misses,
                    bytes_to_mib(cache.resident_bytes as u64),
                    bytes_to_mib(cache.budget_bytes as u64),
                    rss_mib().unwrap_or(f64::NAN),
                );
            }
        }
        write_jsonl(&json_path, &rows, &args, &gt_path, dataset_path.as_path())?;
        println!(
            "rows written: {} ({} family/families)",
            rows.len(),
            args.families.len()
        );
    }

    print_rows(&rows, args.points, args.dim, args.gt_queries);
    write_jsonl(&json_path, &rows, &args, &gt_path, dataset_path.as_path())?;
    println!(
        "wall clock: {:.1}s; total dataset+index on disk: {:.1} MiB",
        started.elapsed().as_secs_f64(),
        bytes_to_mib(generation.bytes) + index_disk_mib_total
    );
    Ok(())
}

fn effective_batch(args: &Args) -> usize {
    // A parallel batch is inserted without its members seeing each other, so a
    // batch that is a large share of the set builds a poor graph. 4096 is the
    // library default and right at 100k+, but a 10k tier would otherwise get a
    // 40%-of-set first batch and lose recall. Cap at one eighth of the tier.
    args.batch.min((args.points / 8).max(1) as usize)
}

fn validate(args: &Args) -> Res<()> {
    if args.points == 0 {
        return Err("--points must be greater than zero".to_string());
    }
    if args.points > u64::from(u32::MAX) {
        return Err(format!(
            "refusing: --points {} exceeds the u32 id space ({}); the ground-truth cache stores \
             ids as u32",
            args.points,
            u32::MAX
        ));
    }
    if args.points > args.max_points {
        return Err(format!(
            "refusing: --points {} exceeds --max-points {}. The ceiling is a guardrail, not an \
             estimate: raise --max-points explicitly when the host has the disk and RAM for the \
             tier (see the ladder in benchmarks/README.md).",
            args.points, args.max_points
        ));
    }
    if args.dim == 0 || args.dim > 4096 {
        return Err(format!("--dim must be in 1..=4096 (got {})", args.dim));
    }
    if args.k == 0 || args.k as u64 >= args.points {
        return Err(format!(
            "--k must be in 1..points (got k={} for n={})",
            args.k, args.points
        ));
    }
    if !(MIN_GT_QUERIES..=MAX_GT_QUERIES).contains(&args.gt_queries) {
        return Err(format!(
            "--gt-queries must be in {MIN_GT_QUERIES}..={MAX_GT_QUERIES} (got {}); a smaller set \
             does not measure a tail, a larger one makes the independent scan the run",
            args.gt_queries
        ));
    }
    if args.ef.is_empty() || args.ef.iter().any(|&ef| ef == 0 || ef > 16_384) {
        return Err("--ef must be a non-empty comma-separated list in 1..=16384".to_string());
    }
    if args.families.contains(&Family::Diskann) && !args.persist {
        return Err(
            "--no-persist cannot measure diskann: the measurement is the file-backed \
                    searcher (save, drop the build object, then query through the block cache), \
                    so there are no files to search. Drop diskann from --families or drop \
                    --no-persist."
                .to_string(),
        );
    }
    if !(args.budget_seconds.is_finite() && args.budget_seconds > 0.0) {
        return Err("--budget-seconds must be a positive finite number".to_string());
    }
    Ok(())
}

fn parse_args() -> Res<Args> {
    let mut args = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut points_set = false;
    let mut index = 0usize;
    while index < argv.len() {
        let flag = argv[index].clone();
        index += 1;
        match flag.as_str() {
            "--points" | "--tier" | "-n" => {
                let value = take_value(&argv, &mut index, &flag)?;
                args.points = parse_count(&value)?;
                points_set = true;
            }
            "--dim" | "-d" => args.dim = parse_usize(&take_value(&argv, &mut index, &flag)?)?,
            "--k" => args.k = parse_usize(&take_value(&argv, &mut index, &flag)?)?,
            "--gt-queries" | "-q" => {
                args.gt_queries = parse_usize(&take_value(&argv, &mut index, &flag)?)?;
            }
            "--ef" => {
                let value = take_value(&argv, &mut index, &flag)?;
                args.ef = value
                    .split(',')
                    .map(str::trim)
                    .filter(|token| !token.is_empty())
                    .map(parse_usize)
                    .collect::<Res<Vec<usize>>>()?;
            }
            "--domain" => {
                let value = take_value(&argv, &mut index, &flag)?;
                args.domain = Domain::parse(&value)?;
            }
            "--shards" => {
                args.shards = parse_usize(&take_value(&argv, &mut index, &flag)?)?;
            }
            "--nprobe" => {
                args.nprobe = take_value(&argv, &mut index, &flag)?
                    .split(',')
                    .map(|value| {
                        value
                            .trim()
                            .parse::<usize>()
                            .map_err(|err| format!("--nprobe: {err}"))
                    })
                    .collect::<Result<Vec<usize>, _>>()?;
            }
            "--families" => {
                let value = take_value(&argv, &mut index, &flag)?;
                let families = value
                    .split(',')
                    .map(str::trim)
                    .filter(|token| !token.is_empty())
                    .map(Family::parse)
                    .collect::<Res<Vec<Family>>>()?;
                if families.is_empty() {
                    return Err("--families must name at least one family".to_string());
                }
                args.families = families;
            }
            "--diskann-cache-mib" => {
                args.diskann_cache_mib = parse_usize(&take_value(&argv, &mut index, &flag)?)?;
            }
            "--max-points" => {
                args.max_points = parse_count(&take_value(&argv, &mut index, &flag)?)?;
            }
            "--budget-seconds" => {
                let value = take_value(&argv, &mut index, &flag)?;
                args.budget_seconds = value
                    .parse::<f64>()
                    .map_err(|_| format!("invalid --budget-seconds `{value}`"))?;
            }
            "--json" => {
                // Optional path: a following non-flag token is the target.
                args.json = match argv.get(index) {
                    Some(next) if !next.starts_with("--") => {
                        index += 1;
                        Some(PathBuf::from(next))
                    }
                    _ => Some(PathBuf::new()),
                };
            }
            "--out" => args.json = Some(PathBuf::from(take_value(&argv, &mut index, &flag)?)),
            "--gt-only" => args.gt_only = true,
            "--prepare-only" => args.prepare_only = true,
            "--verify-gt" => args.verify_gt = Some(true),
            "--no-verify-gt" => args.verify_gt = Some(false),
            "--regenerate-gt" => args.reuse_gt = false,
            "--no-persist" => args.persist = false,
            "--batch" => {
                args.batch = parse_usize(&take_value(&argv, &mut index, &flag)?)?;
                if args.batch == 0 {
                    return Err("--batch must be greater than zero".to_string());
                }
            }
            "--self-test" => args.self_test = true,
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other if !other.starts_with('-') && !points_set => {
                args.points = parse_count(other)?;
                points_set = true;
            }
            other => return Err(format!("unknown argument `{other}`; try --help")),
        }
    }
    Ok(args)
}

fn take_value(argv: &[String], index: &mut usize, flag: &str) -> Res<String> {
    let value = argv
        .get(*index)
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))?;
    *index += 1;
    Ok(value)
}

fn parse_usize(text: &str) -> Res<usize> {
    text.trim()
        .parse::<usize>()
        .map_err(|_| format!("invalid number `{text}`"))
}

/// Accepts `100000`, `100_000`, `100k`, `1m`, `1b` (case-insensitive).
fn parse_count(text: &str) -> Res<u64> {
    let cleaned: String = text.trim().replace('_', "");
    let (digits, multiplier) = match cleaned.chars().last() {
        Some('k' | 'K') => (&cleaned[..cleaned.len() - 1], 1_000u64),
        Some('m' | 'M') => (&cleaned[..cleaned.len() - 1], 1_000_000),
        Some('b' | 'B') => (&cleaned[..cleaned.len() - 1], 1_000_000_000),
        _ => (cleaned.as_str(), 1),
    };
    let value = digits
        .parse::<u64>()
        .map_err(|_| format!("invalid count `{text}`"))?;
    value
        .checked_mul(multiplier)
        .ok_or_else(|| format!("count `{text}` overflows u64"))
}

fn print_usage() {
    println!(
        "large_scale — tier harness above 10M points\n\
         \n\
         Options:\n\
           --points <n|100k|1m>   base vectors (default 100000)\n\
           --domain <uniform|clustered|neardup>  synthetic statistics (default uniform)\n\
           --dim <d>              width (default 96)\n\
           --k <k>                neighbours (default 10)\n\
           --gt-queries <q>       sampled query set, 200..=1000 (default 500)\n\
           --ef <a,b,c>           ef_search sweep (default 50,100,200)\n\
           --families <a,b>       families to build and measure: hnsw,diskann\n\
                                  (default both)\n\
           --diskann-cache-mib <n>  block cache for the file-backed diskann\n\
                                  searcher; 0 disables it (default 64)\n\
           --max-points <n>       hard ceiling (default 20000000)\n\
           --budget-seconds <s>   refuse an estimate above this (default 3600)\n\
           --gt-only              compute/reuse ground truth, then stop\n\
           --prepare-only         write the dataset, then stop (scripts/prepare_data.py)\n\
           --shards K             build K shards, one resident at a time (default 1)\n\
           --nprobe a,b,c         shards probed per query in sharded mode\n\
           --regenerate-gt        ignore an existing cache and recompute\n\
           --verify-gt / --no-verify-gt  force/disable the f32+f64 cross-check\n\
           --no-persist           skip the postcard index snapshot (hnsw only;\n\
                                   refused with diskann in --families)\n\
           --batch <n>            parallel construction batch (default 4096; smaller\n\
                                  batches see more of the graph and raise recall)\n\
           --json [path]          write JSONL rows (default under <root>/rows/)\n\
           --self-test            resumable-generation and cache checks, then exit\n\
         \n\
         ANN_DATA_ROOT overrides the data root; otherwise $REBUILDABLE_ROOT/vicinity-data,\n\
         otherwise a temp directory."
    );
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

fn data_root() -> PathBuf {
    if let Ok(value) = std::env::var("ANN_DATA_ROOT") {
        if !value.is_empty() {
            return PathBuf::from(value);
        }
    }
    if let Ok(value) = std::env::var("REBUILDABLE_ROOT") {
        if !value.is_empty() {
            return Path::new(&value).join("vicinity-data");
        }
    }
    std::env::temp_dir().join("vicinity-data")
}

fn dataset_path(root: &Path, domain: Domain, count: u64, dim: usize) -> PathBuf {
    root.join("datasets")
        .join(format!("{}-n{count}-d{dim}.bin", domain.name()))
}

fn gt_path(
    root: &Path,
    domain: Domain,
    count: u64,
    dim: usize,
    k: usize,
    queries: usize,
    hash: u64,
) -> PathBuf {
    root.join("gt").join(format!(
        "{}-n{count}-d{dim}-k{k}-q{queries}-h{hash:016x}.gt",
        domain.name()
    ))
}

fn index_path(root: &Path, domain: Domain, count: u64, dim: usize, batch: usize) -> PathBuf {
    root.join("indexes").join(format!(
        "{}-n{count}-d{dim}-m16-efc200-b{batch}.postcard",
        domain.name()
    ))
}

/// `save` publishes a generation under this directory; the searcher opens the
/// one named by `CURRENT`.
fn diskann_index_dir(root: &Path, domain: Domain, count: u64, dim: usize, batch: usize) -> PathBuf {
    root.join("indexes").join(format!(
        "{}-n{count}-d{dim}-m32-efc100-b{batch}-diskann/",
        domain.name()
    ))
}

fn default_rows_name(args: &Args) -> String {
    // The family set is in the name so a narrower run cannot silently replace
    // an artifact that carried more rows.
    let families: Vec<&str> = args.families.iter().map(|family| family.name()).collect();
    format!(
        "large-scale-{}-{}-{}-d{}-k{}.jsonl",
        args.domain.name(),
        tier_label(args.points),
        families.join("+"),
        args.dim,
        args.k
    )
}

fn tier_label(points: u64) -> String {
    match points {
        0..=999 => points.to_string(),
        1_000..=999_999 => format!("{}k", points / 1_000),
        1_000_000..=999_999_999 => format!("{}m", points / 1_000_000),
        _ => format!("{}b", points / 1_000_000_000),
    }
}

// ---------------------------------------------------------------------------
// Domains: the sibling harness's statistics, streamed one row at a time
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Domain {
    Uniform,
    Clustered,
    Neardup,
}

impl Domain {
    fn parse(text: &str) -> Res<Self> {
        match text {
            "uniform" => Ok(Self::Uniform),
            "clustered" => Ok(Self::Clustered),
            "neardup" => Ok(Self::Neardup),
            other => Err(format!(
                "unknown domain `{other}`; known domains: uniform,clustered,neardup"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Clustered => "clustered",
            Self::Neardup => "neardup",
        }
    }

    fn seed(self) -> u64 {
        match self {
            Self::Uniform => SEED,
            Self::Clustered => SEED ^ 0xC1_0000,
            Self::Neardup => SEED ^ 0x0D_0000,
        }
    }
}

const CLUSTER_CENTRES: usize = 16;
const CLUSTER_SIGMA: f32 = 0.25;
const NEARDUP_SEEDS: usize = 64;
const NEARDUP_SIGMA: f32 = 0.02;
const NEARDUP_COPY_PCT: u32 = 90;

/// The small, seed-derived structure a domain needs while streaming rows.
/// Uniform has none; clustered keeps 16 centres; neardup keeps 64 seed
/// documents. All of it is independent of the point count, so memory stays
/// bounded.
struct DomainContext {
    domain: Domain,
    dim: usize,
    seed: u64,
    centres: Vec<Vec<f32>>,
    documents: Vec<Vec<f32>>,
}

impl DomainContext {
    fn new(domain: Domain, seed: u64, dim: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(domain.seed() ^ seed.rotate_left(17));
        let centres = if domain == Domain::Clustered {
            (0..CLUSTER_CENTRES)
                .map(|_| unit_vector(&mut rng, dim))
                .collect()
        } else {
            Vec::new()
        };
        let documents = if domain == Domain::Neardup {
            (0..NEARDUP_SEEDS)
                .map(|_| unit_vector(&mut rng, dim))
                .collect()
        } else {
            Vec::new()
        };
        Self {
            domain,
            dim,
            seed: domain.seed(),
            centres,
            documents,
        }
    }

    /// Deterministic row `index`, independent of chunk boundaries so a resumed
    /// generation is byte-identical to an uninterrupted one.
    fn row_at(&self, index: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(
            self.seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(7),
        );
        self.sample_one(index, &mut rng)
    }

    fn sample_one(&self, index: u64, rng: &mut StdRng) -> Vec<f32> {
        match self.domain {
            Domain::Uniform => normalize(sample_raw(rng, self.dim)),
            Domain::Clustered => {
                let centre = &self.centres[index as usize % self.centres.len()];
                let noise = sample_raw(rng, self.dim);
                let mixed: Vec<f32> = centre
                    .iter()
                    .zip(noise.iter())
                    .map(|(c, n)| c + CLUSTER_SIGMA * n)
                    .collect();
                normalize(mixed)
            }
            Domain::Neardup => {
                if rng.random_range(0..100) < NEARDUP_COPY_PCT {
                    let document = &self.documents[rng.random_range(0..self.documents.len())];
                    let jitter = unit_vector(rng, self.dim);
                    let mixed: Vec<f32> = document
                        .iter()
                        .zip(jitter.iter())
                        .map(|(d, j)| d + NEARDUP_SIGMA * j)
                        .collect();
                    normalize(mixed)
                } else {
                    unit_vector(rng, self.dim)
                }
            }
        }
    }
}

fn generate_queries(ctx: &DomainContext, count: usize) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(ctx.seed ^ QUERY_SALT);
    (0..count as u64)
        .map(|index| ctx.sample_one(index, &mut rng))
        .collect()
}

fn sample_raw(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect()
}

fn unit_vector(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    normalize(sample_raw(rng, dim))
}

fn normalize(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

// ---------------------------------------------------------------------------
// Dataset file: chunked, resumable generation and memory-mapped reads
// ---------------------------------------------------------------------------

struct Generation {
    generated: u64,
    resumed_from: u64,
    bytes: u64,
}

struct DataHeader {
    dim: usize,
    count: u64,
}

fn dataset_bytes(count: u64, dim: usize) -> Res<u64> {
    count
        .checked_mul(dim as u64)
        .and_then(|elements| elements.checked_mul(4))
        .and_then(|data| data.checked_add(DATA_HEADER_BYTES))
        .ok_or_else(|| format!("dataset size overflows u64 for n={count}, dim={dim}"))
}

/// Complete rows already on disk, or an error when the header contradicts the
/// requested dataset.
fn dataset_rows_on_disk(path: &Path, count: u64, dim: usize) -> Res<u64> {
    if !path.exists() {
        return Ok(0);
    }
    let length = std::fs::metadata(path)
        .map_err(|err| format!("cannot stat {}: {err}", path.display()))?
        .len();
    if length < DATA_HEADER_BYTES {
        return Ok(0);
    }
    let header = read_data_header(path)?;
    if header.dim != dim || header.count != count {
        return Err(format!(
            "{} holds dim={} count={}, but this run asks for dim={dim} count={count}. \
             Delete it or choose a different --points/--dim (the file name is not a guarantee).",
            path.display(),
            header.dim,
            header.count
        ));
    }
    Ok(((length - DATA_HEADER_BYTES) / (dim as u64 * 4)).min(count))
}

fn read_data_header(path: &Path) -> Res<DataHeader> {
    let mut file =
        File::open(path).map_err(|err| format!("cannot open {}: {err}", path.display()))?;
    let mut raw = [0u8; DATA_HEADER_BYTES as usize];
    file.read_exact(&mut raw)
        .map_err(|err| format!("cannot read {} header: {err}", path.display()))?;
    if raw[0..8] != MAGIC {
        return Err(format!(
            "{} has magic {:?}, expected {:?}; not a vicinity large-scale dataset",
            path.display(),
            &raw[0..8],
            MAGIC
        ));
    }
    let dim = u32::from_le_bytes(
        raw[8..12]
            .try_into()
            .map_err(|_| "header dim slice".to_string())?,
    ) as usize;
    let count = u64::from_le_bytes(
        raw[12..20]
            .try_into()
            .map_err(|_| "header count slice".to_string())?,
    );
    Ok(DataHeader { dim, count })
}

fn ensure_dataset(
    ctx: &DomainContext,
    path: &Path,
    count: u64,
    dim: usize,
    families: &[Family],
) -> Res<Generation> {
    let total = dataset_bytes(count, dim)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|err| format!("cannot create {}: {err}", parent.display()))?;
    }

    let mut resumed_from = 0u64;
    if path.exists() {
        let length = std::fs::metadata(path)
            .map_err(|err| format!("cannot stat {}: {err}", path.display()))?
            .len();
        if length >= DATA_HEADER_BYTES {
            let header = read_data_header(path)?;
            if header.dim != dim || header.count != count {
                return Err(format!(
                    "{} holds dim={} count={}, but this run asks for dim={dim} count={count}",
                    path.display(),
                    header.dim,
                    header.count
                ));
            }
            if length > total {
                return Err(format!(
                    "{} is {} bytes, larger than the {total} bytes implied by dim={dim} \
                     count={count}; refusing to truncate a file that may be something else",
                    path.display(),
                    length
                ));
            }
            let stride = dim as u64 * 4;
            resumed_from = (length - DATA_HEADER_BYTES) / stride;
            // Drop a partially written trailing row; per-row seeding makes the
            // rewrite identical to the interrupted bytes.
            let keep = DATA_HEADER_BYTES + resumed_from * stride;
            if keep != length {
                let file = OpenOptions::new()
                    .write(true)
                    .open(path)
                    .map_err(|err| format!("cannot open {} to truncate: {err}", path.display()))?;
                file.set_len(keep)
                    .map_err(|err| format!("cannot truncate {}: {err}", path.display()))?;
            }
        }
    }

    let remaining = count.saturating_sub(resumed_from);
    let remaining_bytes = remaining * dim as u64 * 4;
    if let Some(free) = available_bytes(path) {
        // Each index is roughly vectors + graph edges: 1.26x the dataset for
        // HNSW (46.3 MiB per 36.6 MiB measured at 100k) and 1.35x for DiskANN
        // (49.6 MiB). Require headroom rather than filling the volume to the
        // last byte.
        let index_estimate = index_disk_estimate(families, total);
        let needed = remaining_bytes.saturating_add(index_estimate);
        let headroom = 512 * 1024 * 1024;
        if free < needed.saturating_add(headroom) {
            let names: Vec<&str> = families.iter().map(|family| family.name()).collect();
            return Err(format!(
                "refusing: disk, not time or memory. {} has {:.2} GiB free; this tier needs \
                 {:.2} GiB more of dataset plus about {:.2} GiB for the {} index files ({}), \
                 {:.2} GiB short of the {:.2} GiB free requirement.",
                path.parent().unwrap_or(path).display(),
                bytes_to_mib(free) / 1024.0,
                bytes_to_mib(remaining_bytes) / 1024.0,
                bytes_to_mib(index_estimate) / 1024.0,
                names.join("+"),
                if names.len() > 1 {
                    "one per family"
                } else {
                    "one index"
                },
                bytes_to_mib(needed.saturating_add(headroom).saturating_sub(free)) / 1024.0,
                bytes_to_mib(needed.saturating_add(headroom)) / 1024.0,
            ));
        }
    }

    if remaining == 0 {
        return Ok(Generation {
            generated: 0,
            resumed_from,
            bytes: total,
        });
    }

    let mut file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path)
        .map_err(|err| format!("cannot open {} for generation: {err}", path.display()))?;

    if resumed_from == 0 {
        // Fresh header. A zero-length or short file is rewritten from scratch.
        file.set_len(0)
            .map_err(|err| format!("cannot reset {}: {err}", path.display()))?;
        file.seek(SeekFrom::Start(0))
            .map_err(|err| format!("cannot seek {}: {err}", path.display()))?;
        let mut header = Vec::with_capacity(DATA_HEADER_BYTES as usize);
        header.extend_from_slice(&MAGIC);
        header.extend_from_slice(&(dim as u32).to_le_bytes());
        header.extend_from_slice(&count.to_le_bytes());
        file.write_all(&header)
            .map_err(|err| format!("cannot write {} header: {err}", path.display()))?;
    } else {
        file.seek(SeekFrom::Start(
            DATA_HEADER_BYTES + resumed_from * dim as u64 * 4,
        ))
        .map_err(|err| format!("cannot seek {}: {err}", path.display()))?;
    }

    let mut writer = BufWriter::with_capacity(1 << 20, file);
    let mut row = resumed_from;
    while row < count {
        let block = ((count - row) as usize).min(GENERATION_BLOCK_ROWS);
        let mut buffer: Vec<u8> = Vec::with_capacity(block * dim * 4);
        for offset in 0..block {
            let vector = ctx.row_at(row + offset as u64);
            debug_assert_eq!(vector.len(), dim);
            for value in &vector {
                buffer.extend_from_slice(&value.to_le_bytes());
            }
        }
        writer
            .write_all(&buffer)
            .map_err(|err| format!("cannot write {}: {err}", path.display()))?;
        row += block as u64;
        if row.is_multiple_of(GENERATION_BLOCK_ROWS as u64 * 64) && row < count {
            println!("  generated {row}/{count} rows");
        }
    }
    writer
        .flush()
        .map_err(|err| format!("cannot flush {}: {err}", path.display()))?;
    let file = writer
        .into_inner()
        .map_err(|err| format!("cannot finish {}: {err}", path.display()))?;
    file.sync_data()
        .map_err(|err| format!("cannot sync {}: {err}", path.display()))?;

    let bytes = std::fs::metadata(path)
        .map_err(|err| format!("cannot stat {} after generation: {err}", path.display()))?
        .len();
    Ok(Generation {
        generated: remaining,
        resumed_from,
        bytes,
    })
}

/// Bytes the selected families' index files need for a full dataset, from the
/// measured 100k ratio: 46.3 MiB of HNSW and 49.6 MiB of DiskANN per 36.6 MiB
/// of dataset, i.e. 1.26x and 1.35x.
fn index_disk_estimate(families: &[Family], dataset_bytes: u64) -> u64 {
    families
        .iter()
        .map(|family| match family {
            Family::Hnsw => dataset_bytes.saturating_mul(126) / 100,
            Family::Diskann => dataset_bytes.saturating_mul(135) / 100,
        })
        .fold(0u64, |sum, bytes| sum.saturating_add(bytes))
}

/// Free bytes on the volume containing `path`, or `None` when `df` cannot say.
///
/// `path` need not exist: the nearest existing ancestor is what `df` is asked
/// about, so the check can run before the index directory is created.
fn available_bytes(path: &Path) -> Option<u64> {
    let mut target = path;
    while !target.exists() {
        target = target.parent()?;
    }
    let output = std::process::Command::new("df")
        .arg("-k")
        .arg(target)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.lines().rfind(|line| !line.trim().is_empty())?;
    let fields: Vec<&str> = line.split_whitespace().collect();
    let available_kib = fields.get(3)?.parse::<u64>().ok()?;
    available_kib.checked_mul(1024)
}

/// A read-only memory map of a dataset file. Rows are borrowed, never copied:
/// the index owns its vector storage, so RSS after an unmapped build is the
/// index rather than the corpus.
struct DatasetFile {
    mmap: memmap2::Mmap,
    dim: usize,
    count: u64,
}

impl DatasetFile {
    fn open(path: &Path) -> Res<Self> {
        let header = read_data_header(path)?;
        let file =
            File::open(path).map_err(|err| format!("cannot open {}: {err}", path.display()))?;
        let expected = dataset_bytes(header.count, header.dim)?;
        if expected > usize::MAX as u64 {
            return Err(format!(
                "{} needs {expected} bytes, more than this platform can map ({} bytes)",
                path.display(),
                usize::MAX
            ));
        }
        let length = file
            .metadata()
            .map_err(|err| format!("cannot stat {}: {err}", path.display()))?
            .len();
        if length != expected {
            return Err(format!(
                "{} is {length} bytes, expected {expected} for dim={} count={}; generation may \
                 still be running or the file is truncated",
                path.display(),
                header.dim,
                header.count
            ));
        }
        // SAFETY: the file is opened read-only and not mutated while mapped. The
        // header ends at byte 20, which is 4-byte aligned, and every row is a
        // whole number of f32 values, so `row()` only creates aligned slices.
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
            .map_err(|err| format!("cannot mmap {}: {err}", path.display()))?;
        if !cfg!(target_endian = "little") {
            return Err(
                "the dataset format is little-endian; big-endian hosts are unsupported".to_string(),
            );
        }
        if mmap.len() != expected as usize {
            return Err(format!(
                "mapped {} bytes of {}, expected {expected}",
                mmap.len(),
                path.display()
            ));
        }
        Ok(Self {
            mmap,
            dim: header.dim,
            count: header.count,
        })
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn count(&self) -> u64 {
        self.count
    }

    fn row(&self, index: u64) -> &[f32] {
        assert!(index < self.count, "row {index} out of bounds");
        let stride = self.dim * 4;
        let start = DATA_HEADER_BYTES as usize + index as usize * stride;
        let bytes = &self.mmap[start..start + stride];
        let pointer = bytes.as_ptr().cast::<f32>();
        // SAFETY: `mmap` is page-aligned, the header is 20 bytes (4-aligned),
        // and `stride` is a multiple of 4, so `pointer` is aligned for f32 and
        // `bytes` covers exactly `dim` initialized values.
        unsafe { std::slice::from_raw_parts(pointer, self.dim) }
    }
}

// ---------------------------------------------------------------------------
// Ground truth: sampled queries, parallel shards, cached by key
// ---------------------------------------------------------------------------

enum CacheState {
    Hit,
    Computed,
    Recomputed,
}

/// Fixed-capacity top-k, kept sorted by descending score. `k` is 10 in the
/// protocol, so a linear insertion is cheaper than a heap. Scores are f64 so
/// the independent f64 reference is not narrowed back to f32 while ranking.
struct TopK {
    k: usize,
    items: Vec<(f64, u32)>,
}

impl TopK {
    fn new(k: usize) -> Self {
        Self {
            k,
            items: Vec::with_capacity(k + 1),
        }
    }

    fn push(&mut self, score: f64, id: u32) {
        if self.items.len() == self.k {
            if let Some(&(worst, _)) = self.items.last() {
                if score <= worst {
                    return;
                }
            }
        }
        let position = self.items.partition_point(|&(value, _)| value > score);
        self.items.insert(position, (score, id));
        if self.items.len() > self.k {
            self.items.pop();
        }
    }
}

fn query_hash(queries: &[Vec<f32>]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64; // FNV-1a 64 offset basis
    for query in queries {
        for value in query {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    hash
}

fn load_or_compute_ground_truth(
    dataset: &DatasetFile,
    queries: &[Vec<f32>],
    k: usize,
    path: &Path,
    reuse: bool,
) -> Res<(Vec<Vec<u32>>, CacheState)> {
    let hash = query_hash(queries);
    let present = path.exists();
    if reuse && present {
        match load_gt_cache(path, dataset.dim(), dataset.count(), k, queries.len(), hash) {
            Ok(truth) => return Ok((truth, CacheState::Hit)),
            Err(message) => {
                return Err(format!(
                    "{message}. Delete the file to recompute, or pass --regenerate-gt"
                ));
            }
        }
    }
    let truth = compute_ground_truth(dataset, queries, k);
    save_gt_cache(path, &truth, dataset.dim(), dataset.count(), k, hash)?;
    // `present` is sampled before the save so "recomputed" means "a cache was
    // there and this run ignored it", not "the save just created one".
    Ok((
        truth,
        if present {
            CacheState::Recomputed
        } else {
            CacheState::Computed
        },
    ))
}

fn compute_ground_truth(dataset: &DatasetFile, queries: &[Vec<f32>], k: usize) -> Vec<Vec<u32>> {
    use rayon::prelude::*;

    let threads = rayon::current_num_threads().max(1);
    let shard_count = (threads * 4).max(8) as u64;
    let rows = dataset.count();
    let shard_rows = rows.div_ceil(shard_count).max(1);
    let ranges: Vec<(u64, u64)> = (0..shard_count)
        .map(|shard| {
            let low = (shard * shard_rows).min(rows);
            let high = ((shard + 1) * shard_rows).min(rows);
            (low, high)
        })
        .filter(|(low, high)| low < high)
        .collect();

    println!(
        "ground truth scan: {} queries x {} rows in {} shards on {} threads",
        queries.len(),
        rows,
        ranges.len(),
        threads
    );

    let scan_start = Instant::now();
    let partials: Vec<Vec<TopK>> = ranges
        .par_iter()
        .map(|&(low, high)| {
            let mut best: Vec<TopK> = (0..queries.len()).map(|_| TopK::new(k)).collect();
            for row_index in low..high {
                let row = dataset.row(row_index);
                for (query, top) in queries.iter().zip(best.iter_mut()) {
                    let distance = vicinity::distance::cosine_distance_normalized(query, row);
                    top.push(f64::from(-distance), row_index as u32);
                }
            }
            best
        })
        .collect();

    // k-way merge: each shard contributes a sorted top-k per query.
    let merged: Vec<Vec<u32>> = (0..queries.len())
        .map(|query_index| {
            let mut items: Vec<(f64, u32)> = Vec::with_capacity(k * partials.len());
            for partial in &partials {
                items.extend_from_slice(&partial[query_index].items);
            }
            items.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            items.truncate(k);
            items.into_iter().map(|(_, id)| id).collect()
        })
        .collect();
    println!(
        "ground truth scan: {:.1}s ({:.1} M candidate comparisons/s)",
        scan_start.elapsed().as_secs_f64(),
        (rows as f64 * queries.len() as f64 / scan_start.elapsed().as_secs_f64()) / 1e6
    );
    merged
}

fn load_gt_cache(
    path: &Path,
    dim: usize,
    count: u64,
    k: usize,
    queries: usize,
    hash: u64,
) -> Res<Vec<Vec<u32>>> {
    let mut file =
        File::open(path).map_err(|err| format!("cannot open {}: {err}", path.display()))?;
    let mut raw = vec![0u8; GT_HEADER_BYTES];
    file.read_exact(&mut raw)
        .map_err(|err| format!("cannot read {} header: {err}", path.display()))?;
    if raw[0..8] != GT_MAGIC {
        return Err(format!(
            "{} has magic {:?}, expected {:?}",
            path.display(),
            &raw[0..8],
            GT_MAGIC
        ));
    }
    let cached_dim = u32::from_le_bytes(raw[8..12].try_into().unwrap_or([0; 4])) as usize;
    let cached_count = u64::from_le_bytes(raw[12..20].try_into().unwrap_or([0; 8]));
    let cached_k = u64::from_le_bytes(raw[20..28].try_into().unwrap_or([0; 8])) as usize;
    let cached_queries = u64::from_le_bytes(raw[28..36].try_into().unwrap_or([0; 8])) as usize;
    let cached_hash = u64::from_le_bytes(raw[36..44].try_into().unwrap_or([0; 8]));
    if cached_dim != dim
        || cached_count != count
        || cached_k != k
        || cached_queries != queries
        || cached_hash != hash
    {
        return Err(format!(
            "{} is a stale ground-truth cache: dim {cached_dim}/{dim}, count \
             {cached_count}/{count}, k {cached_k}/{k}, queries {cached_queries}/{queries}, \
             hash {cached_hash:016x}/{hash:016x}",
            path.display()
        ));
    }

    let expected = queries
        .checked_mul(k)
        .and_then(|ids| ids.checked_mul(4))
        .ok_or_else(|| "ground-truth cache size overflows usize".to_string())?;
    let mut data = vec![0u8; expected];
    file.read_exact(&mut data)
        .map_err(|err| format!("cannot read {} ids: {err}", path.display()))?;
    let (words, _rest) = data.as_chunks::<4>();
    let truth = words
        .iter()
        .map(|chunk| u32::from_le_bytes(*chunk))
        .collect::<Vec<u32>>()
        .chunks(k)
        .map(<[u32]>::to_vec)
        .collect();
    Ok(truth)
}

fn save_gt_cache(
    path: &Path,
    truth: &[Vec<u32>],
    dim: usize,
    count: u64,
    k: usize,
    hash: u64,
) -> Res<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|err| format!("cannot create {}: {err}", parent.display()))?;
    }
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    let mut file = File::create(&temporary)
        .map_err(|err| format!("cannot create {}: {err}", temporary.display()))?;
    let mut header = Vec::with_capacity(GT_HEADER_BYTES);
    header.extend_from_slice(&GT_MAGIC);
    header.extend_from_slice(&(dim as u32).to_le_bytes());
    header.extend_from_slice(&count.to_le_bytes());
    header.extend_from_slice(&(k as u64).to_le_bytes());
    header.extend_from_slice(&(truth.len() as u64).to_le_bytes());
    header.extend_from_slice(&hash.to_le_bytes());
    file.write_all(&header)
        .map_err(|err| format!("cannot write {}: {err}", temporary.display()))?;
    for ids in truth {
        for id in ids {
            file.write_all(&id.to_le_bytes())
                .map_err(|err| format!("cannot write {}: {err}", temporary.display()))?;
        }
    }
    file.sync_all()
        .map_err(|err| format!("cannot sync {}: {err}", temporary.display()))?;
    std::fs::rename(&temporary, path).map_err(|err| {
        format!(
            "cannot rename {} to {}: {err}",
            temporary.display(),
            path.display()
        )
    })?;
    Ok(())
}

/// Cross-check the cached ground truth against two independent brute-force
/// references over a small query sample: scalar f32 accumulation (a different
/// code path from the SIMD shard scan) and f64 accumulation.
fn verify_ground_truth(
    dataset: &DatasetFile,
    queries: &[Vec<f32>],
    truth: &[Vec<u32>],
    k: usize,
) -> Res<()> {
    let sample = queries.len().min(8);
    let mut f32_overlap = 0usize;
    let mut f64_overlap = 0usize;
    let mut total = 0usize;
    let mut first: Option<(Vec<(u32, f64)>, Vec<(u32, f64)>)> = None;

    for query_index in 0..sample {
        let query = &queries[query_index];
        let mut f32_top = TopK::new(k);
        let mut f64_top = TopK::new(k);
        for row_index in 0..dataset.count() {
            let row = dataset.row(row_index);
            let mut sum32 = 0.0f32;
            let mut sum64 = 0.0f64;
            for (q, r) in query.iter().zip(row.iter()) {
                sum32 += q * r;
                sum64 += f64::from(*q) * f64::from(*r);
            }
            f32_top.push(f64::from(sum32), row_index as u32);
            f64_top.push(sum64, row_index as u32);
        }
        let f32_ids: Vec<u32> = f32_top.items.iter().map(|&(_, id)| id).collect();
        let f64_ids: Vec<u32> = f64_top.items.iter().map(|&(_, id)| id).collect();
        f32_overlap += overlap(&f32_ids, &truth[query_index]);
        f64_overlap += overlap(&f64_ids, &truth[query_index]);
        total += k;
        if first.is_none() {
            first = Some((
                f32_top.items.iter().map(|&(s, id)| (id, s)).collect(),
                f64_top.items.iter().map(|&(s, id)| (id, s)).collect(),
            ));
        }
    }

    let f32_rate = f32_overlap as f64 / total as f64;
    let f64_rate = f64_overlap as f64 / total as f64;
    println!(
        "gt verify: {sample} sampled queries, mean top-{k} overlap with cached truth \
         f32 reference {f32_rate:.3}, f64 reference {f64_rate:.3}"
    );
    if let Some((f32_items, f64_items)) = &first {
        println!("  query 0 f32 brute force: {f32_items:?}");
        println!("  query 0 f64 brute force: {f64_items:?}");
        println!("  query 0 cached truth:    {:?}", truth[0]);
    }
    if f32_rate < 0.98 || f64_rate < 0.98 {
        return Err(format!(
            "ground truth failed verification: f32 overlap {f32_rate:.3}, f64 overlap \
             {f64_rate:.3}; expected >= 0.98"
        ));
    }
    println!();
    Ok(())
}

fn overlap(ids: &[u32], truth: &[u32]) -> usize {
    ids.iter().filter(|id| truth.contains(id)).count()
}

fn brute_top_k(dataset: &DatasetFile, query: &[f32], k: usize) -> Vec<u32> {
    let mut top = TopK::new(k);
    for row_index in 0..dataset.count() {
        let distance =
            vicinity::distance::cosine_distance_normalized(query, dataset.row(row_index));
        top.push(f64::from(-distance), row_index as u32);
    }
    top.items.into_iter().map(|(_, id)| id).collect()
}

/// Exact-scan p50 over a bounded query sample, timed like a family query so
/// `1/ratio` is time-based (ann-benchmarks' definition).
fn exact_p50_us(dataset: &DatasetFile, queries: &[Vec<f32>], k: usize) -> f64 {
    for query in queries.iter().take(2) {
        let _ = brute_top_k(dataset, query, k);
    }
    let mut latencies: Vec<f64> = queries
        .iter()
        .take(EXACT_TIMED_QUERIES)
        .map(|query| {
            let start = Instant::now();
            let _ = brute_top_k(dataset, query, k);
            start.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    percentile(&mut latencies, 50.0)
}

// ---------------------------------------------------------------------------
// Index build, persistence and measurement
// ---------------------------------------------------------------------------

fn build_hnsw(path: &Path, count: u64, dim: usize, batch: usize) -> Res<vicinity::hnsw::HNSWIndex> {
    use vicinity::distance::DistanceMetric;
    use vicinity::hnsw::{HNSWIndex, HNSWParams};

    let dataset = DatasetFile::open(path)?;
    if dataset.count() != count || dataset.dim() != dim {
        return Err(format!(
            "dataset opened as dim={} count={}, expected dim={dim} count={count}",
            dataset.dim(),
            dataset.count()
        ));
    }
    let params = HNSWParams {
        metric: DistanceMetric::Cosine,
        seed: Some(HNSW_SEED),
        // Upper layers hold 1/16, 1/256 ... of the nodes, but the default
        // layout allocates a full-width neighbour slot for every node in every
        // layer. Freezing them into eligible-node CSR is the storage change
        // this flag measures; ANN_COMPACT_UPPER=1 turns it on.
        #[cfg(feature = "compact-hnsw")]
        compact_upper_layers: std::env::var_os("ANN_COMPACT_UPPER").is_some(),
        ..HNSWParams::default()
    };
    let mut index = HNSWIndex::with_params(dim, params).map_err(string_err)?;
    for row_index in 0..count {
        index
            .add_slice(row_index as u32, dataset.row(row_index))
            .map_err(string_err)?;
    }
    index.build_parallel(batch).map_err(string_err)?;
    Ok(index)
}

fn persist_index(index: &vicinity::hnsw::HNSWIndex, path: &Path) -> Res<u64> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|err| format!("cannot create {}: {err}", parent.display()))?;
    }
    let file =
        File::create(path).map_err(|err| format!("cannot create {}: {err}", path.display()))?;
    let writer = BufWriter::with_capacity(1 << 20, file);
    let mut writer = postcard::to_io(index, writer)
        .map_err(|err| format!("cannot serialize index to {}: {err}", path.display()))?;
    writer
        .flush()
        .map_err(|err| format!("cannot flush {}: {err}", path.display()))?;
    let file = writer
        .into_inner()
        .map_err(|err| format!("cannot finish {}: {err}", path.display()))?;
    file.sync_data()
        .map_err(|err| format!("cannot sync {}: {err}", path.display()))?;
    let bytes = std::fs::metadata(path)
        .map_err(|err| format!("cannot stat {}: {err}", path.display()))?
        .len();
    Ok(bytes)
}

/// DiskANN's builder is a strict superset of Vamana's: it is the same two-pass
/// Vamana construction, and `DiskANNIndex` is the only type in the crate that
/// both consumes borrowed slices and serializes to a layout a file-backed
/// searcher can read (`save` → `DiskANNSearcher`). `VamanaIndex` has no
/// `add_slice`, so it cannot ingest a mapped row without allocating a copy per
/// row, and it has no file-backed searcher at all.
fn build_diskann(
    path: &Path,
    count: u64,
    dim: usize,
    batch: usize,
) -> Res<vicinity::diskann::DiskANNIndex> {
    use vicinity::diskann::{DiskANNIndex, DiskANNParams};

    let dataset = DatasetFile::open(path)?;
    if dataset.count() != count || dataset.dim() != dim {
        return Err(format!(
            "dataset opened as dim={} count={}, expected dim={dim} count={count}",
            dataset.dim(),
            dataset.count()
        ));
    }
    let params = DiskANNParams {
        seed: Some(HNSW_SEED),
        ..DiskANNParams::default()
    };
    let mut index = DiskANNIndex::new(dim, params).map_err(string_err)?;
    for row_index in 0..count {
        index
            .add_slice(row_index as u32, dataset.row(row_index))
            .map_err(string_err)?;
    }
    index.build_parallel(batch).map_err(string_err)?;
    Ok(index)
}

/// Squared L2 between a row and centroid `c` of a flat centroid array.
fn centroid_distance(row: &[f32], centroids: &[f32], dim: usize, c: usize) -> f32 {
    row.iter()
        .zip(&centroids[c * dim..(c + 1) * dim])
        .map(|(a, b)| (a - b) * (a - b))
        .sum()
}

/// Shard ids for the `nprobe` nearest centroids, nearest first.
fn nearest_centroids(query: &[f32], centroids: &[f32], dim: usize, nprobe: usize) -> Vec<usize> {
    let shards = centroids.len() / dim;
    let mut all: Vec<(f32, usize)> = (0..shards)
        .map(|c| (centroid_distance(query, centroids, dim, c), c))
        .collect();
    all.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    all.into_iter()
        .take(nprobe.min(shards))
        .map(|(_, c)| c)
        .collect()
}

/// k-means on a bounded sample: k-means++ seeding, four Lloyd iterations.
fn train_centroids(sample: &[Vec<f32>], dim: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        state
    };
    let mut centres: Vec<Vec<f32>> = Vec::with_capacity(k);
    if sample.is_empty() {
        return vec![0.0; k * dim];
    }
    centres.push(sample[(next() >> 33) as usize % sample.len()].clone());
    while centres.len() < k {
        let mut best = (0usize, f32::NEG_INFINITY);
        for _ in 0..64 {
            let candidate = (next() >> 33) as usize % sample.len();
            let nearest = centres
                .iter()
                .map(|c| {
                    sample[candidate]
                        .iter()
                        .zip(c)
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f32>()
                })
                .fold(f32::INFINITY, f32::min);
            if nearest > best.1 {
                best = (candidate, nearest);
            }
        }
        centres.push(sample[best.0].clone());
    }

    let mut flat: Vec<f32> = centres.concat();
    for _ in 0..4 {
        let mut sums = vec![0f64; k * dim];
        let mut counts = vec![0usize; k];
        for point in sample {
            let mut best = (f32::INFINITY, 0usize);
            for c in 0..k {
                let d: f32 = point
                    .iter()
                    .zip(&flat[c * dim..(c + 1) * dim])
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                if d < best.0 {
                    best = (d, c);
                }
            }
            counts[best.1] += 1;
            for d in 0..dim {
                sums[best.1 * dim + d] += f64::from(point[d]);
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                continue;
            }
            for d in 0..dim {
                flat[c * dim + d] = (sums[c * dim + d] / counts[c] as f64) as f32;
            }
        }
    }
    flat
}

/// Build `K` shards, one resident at a time, and query `nprobe` of them.
///
/// The whole point is the peak: one shard of `n / K` rows instead of one graph
/// of `n`, which is what makes a tier above this host's RAM buildable locally.
/// Shards are built sequentially; a query's probe set is known from the
/// centroids before the first build, so each shard's results are merged into a
/// per-query accumulator and the shard is dropped immediately.
fn run_sharded(path: &Path, queries: &[Vec<f32>], truth: &[Vec<u32>], args: &Args) -> Res<()> {
    use vicinity::distance::DistanceMetric;
    use vicinity::hnsw::{HNSWIndex, HNSWParams};

    let dataset = DatasetFile::open(path)?;
    let count = dataset.count();
    let dim = dataset.dim();
    let shards = args.shards;
    if shards > u16::MAX as usize {
        return Err(format!(
            "--shards {shards} exceeds the u16 assignment limit"
        ));
    }
    let nprobes: Vec<usize> = if args.nprobe.is_empty() {
        vec![1]
    } else {
        args.nprobe.clone()
    };
    println!(
        "sharded build: {count} rows into {shards} shards, dim {dim}, k={}",
        args.k
    );

    // Centroids from a stride sample: deterministic, bounded, and independent of
    // the query set.
    let target = 20_000u64.min(count);
    let stride = (count / target).max(1);
    let mut sample: Vec<Vec<f32>> = Vec::new();
    let mut row = 0u64;
    while row < count && sample.len() < target as usize {
        sample.push(dataset.row(row).to_vec());
        row += stride;
    }
    let centroids = train_centroids(&sample, dim, shards, 0x5EED_1000);
    drop(sample);

    let mut assignment: Vec<u16> = Vec::with_capacity(count as usize);
    for id in 0..count {
        let mut best = (f32::INFINITY, 0usize);
        for c in 0..shards {
            let d = centroid_distance(dataset.row(id), &centroids, dim, c);
            if d < best.0 {
                best = (d, c);
            }
        }
        assignment.push(best.1 as u16);
    }
    let mut sizes = vec![0usize; shards];
    for &shard in &assignment {
        sizes[shard as usize] += 1;
    }
    println!(
        "shard sizes: min {}, max {}, mean {}",
        sizes.iter().min().copied().unwrap_or(0),
        sizes.iter().max().copied().unwrap_or(0),
        count as usize / shards
    );

    // Probe sets are fixed before the first build.
    let probes: Vec<Vec<Vec<usize>>> = nprobes
        .iter()
        .map(|&np| {
            queries
                .iter()
                .map(|q| nearest_centroids(q, &centroids, dim, np))
                .collect()
        })
        .collect();

    // accumulators[p][e][q] = merged top-k across probed shards; timings[p][e][q]
    // sums each shard's search time, which is what a parallel probe would cost.
    let mut accum: Vec<Vec<Vec<Vec<(u32, f32)>>>> = nprobes
        .iter()
        .map(|_| vec![vec![Vec::new(); queries.len()]; args.ef.len()])
        .collect();
    let mut timings: Vec<Vec<Vec<f64>>> = nprobes
        .iter()
        .map(|_| vec![vec![0.0; queries.len()]; args.ef.len()])
        .collect();

    let params = HNSWParams {
        m: 16,
        ef_construction: 200,
        metric: DistanceMetric::L2,
        seed: Some(HNSW_SEED),
        ..HNSWParams::default()
    };

    let build_start = Instant::now();
    let mut max_shard_build = 0.0f64;
    let mut max_shard_rss = 0.0f64;
    for (shard, &shard_size) in sizes.iter().enumerate() {
        let shard_start = Instant::now();
        let rss_before = rss_mib();
        let mut graph = HNSWIndex::with_params(dim, params.clone()).map_err(string_err)?;
        for id in 0..count {
            if assignment[id as usize] as usize == shard {
                graph
                    .add_slice(id as u32, dataset.row(id))
                    .map_err(string_err)?;
            }
        }
        graph.build_parallel(args.batch).map_err(string_err)?;
        let build_s = shard_start.elapsed().as_secs_f64();
        max_shard_build = max_shard_build.max(build_s);
        if let (Some(before), Some(after)) = (rss_before, rss_mib()) {
            max_shard_rss = max_shard_rss.max(after - before);
        }
        println!("shard {shard}: {shard_size} rows, build {build_s:.1}s");

        for (p, _) in nprobes.iter().enumerate() {
            for (e, &ef) in args.ef.iter().enumerate() {
                for (q, query) in queries.iter().enumerate() {
                    if !probes[p][q].contains(&shard) {
                        continue;
                    }
                    let started = Instant::now();
                    let found = graph.search(query, args.k, ef).map_err(string_err)?;
                    timings[p][e][q] += started.elapsed().as_secs_f64() * 1e6;
                    let acc = &mut accum[p][e][q];
                    acc.extend(found);
                    acc.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                    acc.truncate(args.k);
                }
            }
        }
        drop(graph);
    }
    let total_build = build_start.elapsed().as_secs_f64();
    println!(
        "sharded build done: {total_build:.0}s total, max shard {max_shard_build:.0}s, \
         peak shard rss +{max_shard_rss:.0} MiB"
    );

    let mut rows = String::new();
    let mut table = String::new();
    for (p, &np) in nprobes.iter().enumerate() {
        for (e, &ef) in args.ef.iter().enumerate() {
            let mut hits = 0usize;
            let mut latencies: Vec<f64> = Vec::with_capacity(queries.len());
            for (q, want) in truth.iter().enumerate() {
                hits += accum[p][e][q]
                    .iter()
                    .filter(|(id, _)| want.iter().take(args.k).any(|w| w == id))
                    .count();
                latencies.push(timings[p][e][q]);
            }
            latencies.sort_by(|a, b| a.total_cmp(b));
            let recall = hits as f64 / (queries.len() * args.k) as f64;
            let p50 = latencies[latencies.len() / 2];
            table.push_str(&format!(
                "| hnsw-sharded | nprobe={np} ef={ef} | {recall:.3} | {p50:.1} | {} |\n",
                sizes.iter().max().copied().unwrap_or(0)
            ));
            rows.push_str(&format!(
                "{{\"family\":\"hnsw-sharded\",\"storage_mode\":\"resident\",\
                 \"dataset\":\"{}@{}m\",\"knob\":\"nprobe\",\"knob_value\":{np},\
                 \"recall\":{recall:.4},\"p50_us\":{p50:.1},\"build_s\":{max_shard_build:.1},\
                 \"shards\":{shards},\"ef\":{ef},\"n\":{count},\"dim\":{dim}}}\n",
                args.domain.name(),
                count / 1_000_000
            ));
        }
    }
    println!();
    println!("| family | knob | recall@k | p50 us | largest shard |");
    println!("| --- | --- | ---: | ---: | ---: |");
    print!("{table}");

    if let Some(path) = &args.json {
        let meta = format!(
            "{{\"_meta\":true,\"schema\":1,\"harness\":\"large_scale_sharded\",\
             \"n\":{count},\"dim\":{dim},\"k\":{},\"queries\":{},\"dataset\":\"{}\",\
             \"tier\":\"{}\",\"shards\":{shards},\"seed\":\"0x{HNSW_SEED:016X}\",\
             \"distance_stats\":false,\"partition\":\"kmeans on a stride sample\"}}\n",
            args.k,
            queries.len(),
            args.domain.name(),
            tier_label(count)
        );
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|err| format!("cannot create {}: {err}", parent.display()))?;
            }
        }
        std::fs::write(path, meta + &rows)
            .map_err(|err| format!("cannot write {}: {err}", path.display()))?;
        println!(
            "wrote {} sharded rows to {}",
            nprobes.len() * args.ef.len(),
            path.display()
        );
    }
    Ok(())
}

/// Save into a fresh directory and return the bytes the searcher will read.
/// The old tree is removed first so the column is one index, not a stack of
/// generations from earlier runs.
fn persist_diskann(index: &vicinity::diskann::DiskANNIndex, dir: &Path) -> Res<u64> {
    if dir.exists() {
        std::fs::remove_dir_all(dir)
            .map_err(|err| format!("cannot clear {}: {err}", dir.display()))?;
    }
    std::fs::create_dir_all(dir)
        .map_err(|err| format!("cannot create {}: {err}", dir.display()))?;
    index.save(dir).map_err(string_err)?;
    dir_bytes(dir)
}

fn dir_bytes(dir: &Path) -> Res<u64> {
    let mut total = 0u64;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let entries = std::fs::read_dir(&current)
            .map_err(|err| format!("cannot read {}: {err}", current.display()))?;
        for entry in entries {
            let entry = entry.map_err(|err| format!("cannot read {}: {err}", current.display()))?;
            let metadata = entry
                .metadata()
                .map_err(|err| format!("cannot stat {}: {err}", entry.path().display()))?;
            if metadata.is_dir() {
                stack.push(entry.path());
            } else {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    Ok(total)
}

/// Refuse before writing when the volume cannot hold the index. The message
/// names the resource (free bytes), not just the refusal.
fn ensure_index_space(dir: &Path, needed: u64) -> Res<()> {
    const HEADROOM: u64 = 512 * 1024 * 1024;
    let Some(free) = available_bytes(dir) else {
        return Ok(());
    };
    if free < needed.saturating_add(HEADROOM) {
        return Err(format!(
            "refusing: disk, not time or memory. {} has {:.2} GiB free, the diskann index needs \
             about {:.2} GiB plus {:.2} GiB headroom: {:.2} GiB short. Free space, lower \
             --points, or drop diskann from --families.",
            dir.display(),
            bytes_to_mib(free) / 1024.0,
            bytes_to_mib(needed) / 1024.0,
            bytes_to_mib(HEADROOM) / 1024.0,
            bytes_to_mib(needed.saturating_add(HEADROOM) - free) / 1024.0,
        ));
    }
    Ok(())
}

/// `vectors.bin` plus the adjacency pages and ids: `dim * 4` bytes per vector,
/// `m * 4 + 8` bytes per node.
fn estimate_diskann_bytes(args: &Args) -> u64 {
    const M: u64 = 32;
    args.points.saturating_mul(args.dim as u64 * 4 + M * 4 + 8)
}

/// Measured peak RSS per point for each family's build at 1M (2.09 KiB HNSW,
/// 1.21 KiB DiskANN). The corpus is a file mapping and its clean pages are
/// evictable, so it is not added to the build peak.
fn build_peak_kib_per_point(family: Family) -> f64 {
    match family {
        Family::Hnsw => 2.09,
        Family::Diskann => 1.21,
    }
}

fn physical_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        String::from_utf8_lossy(&output.stdout).trim().parse().ok()
    }
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = text.lines().find(|line| line.starts_with("MemTotal:"))?;
        let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        kib.checked_mul(1024)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

/// Refuse when the largest selected build cannot fit in RAM. This is checked
/// before `--budget-seconds` because memory is the wall a longer budget cannot
/// move.
fn memory_refusal(args: &Args) -> Option<String> {
    // `--gt-only` stops before any index exists; only the scan's RSS matters and
    // that is bounded by the query count, not by the tier.
    if args.gt_only {
        return None;
    }
    let physical = physical_memory_bytes()?;
    let mut largest = args.families.first().copied()?;
    for family in &args.families {
        if build_peak_kib_per_point(*family) > build_peak_kib_per_point(largest) {
            largest = *family;
        }
    }
    let per_point = build_peak_kib_per_point(largest);
    // Sharded builds hold one shard at a time, so the peak is the shard's size,
    // not the tier's. Refusing 100M because the single graph would need 199 GiB
    // is wrong when the run is 4 shards of 25M.
    let resident_rows = if args.shards > 1 {
        args.points.div_ceil(args.shards as u64)
    } else {
        args.points
    };
    let needed = resident_rows as f64 * per_point * 1024.0;
    let usable = physical as f64 * 0.85;
    if needed <= usable {
        return None;
    }
    Some(format!(
        "refusing: memory, not time or disk. The {} build peaks at about {:.0} GiB \
         ({per_point:.2} KiB/point measured at 1M, {} row(s) resident) and this host has \
         {:.0} GiB, of which the guardrail allows {:.0} GiB; the largest tier that fits is \
         about {:.0}M points. Lower --points, raise --shards, or select diskann only \
         ({:.2} KiB/point).",
        largest.name(),
        bytes_to_mib(needed as u64) / 1024.0,
        resident_rows,
        bytes_to_mib(physical) / 1024.0,
        bytes_to_mib(usable as u64) / 1024.0,
        usable / (per_point * 1024.0) / 1e6,
        build_peak_kib_per_point(Family::Diskann),
    ))
}

/// `None` while the budget still affords the next point; `Some(message)` names
/// the resource that ran out when it does not.
fn budget_refusal(args: &Args, started: Instant, family: &Family, ef: usize) -> Option<String> {
    let spent = started.elapsed().as_secs_f64();
    if spent <= args.budget_seconds {
        return None;
    }
    Some(format!(
        "budget exhausted: time, not memory or disk. {spent:.0}s of --budget-seconds {:.0}s is \
         already spent before {} ef_search={ef}; reporting the rows measured so far. Raise \
         --budget-seconds deliberately or lower --points/--ef.",
        args.budget_seconds,
        family.name(),
    ))
}

struct Stats {
    recall: f64,
    p50: f64,
    p95: f64,
    p99: f64,
    qps: f64,
    dists_per_query: f64,
    errors: usize,
    first_error: Option<String>,
}

/// One measurement point over the shared query set and cached truth.
///
/// `search` is the family's own call — `HNSWIndex::search` for the in-RAM
/// graph, `DiskANNSearcher::search` for the file-backed one — so the closure
/// keeps each family's real call signature (`&self` vs `&mut self`) out of the
/// measurement code. Five untimed warm-up queries run first; with a block
/// cache that is what makes the timed rows steady-state rather than cold.
/// Run the query sweep for one `ef`, or stop at the wall-clock budget.
///
/// The budget is checked per query rather than per `ef`: at 10M a single sweep
/// of a file-backed index is I/O bound and can outlast the whole budget, and
/// stopping between sweeps means the operator waits hours to learn it was too
/// slow. A sweep that stops early returns `None`, so no partial-recall row is
/// ever written.
fn evaluate<F>(
    mut search: F,
    queries: &[Vec<f32>],
    truth: &[Vec<u32>],
    k: usize,
    ef: usize,
    deadline: Option<Instant>,
) -> Option<Stats>
where
    F: FnMut(&[f32], usize, usize) -> Result<Vec<(u32, f32)>, String>,
{
    for query in queries.iter().take(5) {
        let _ = search(query, k, ef);
    }

    let mut latencies = Vec::with_capacity(queries.len());
    let mut hits = 0usize;
    let mut errors = 0usize;
    let mut first_error = None;
    vicinity::distance_stats::reset_distance_count();
    let loop_start = Instant::now();
    for (query, expected) in queries.iter().zip(truth.iter()) {
        if deadline.is_some_and(|limit| Instant::now() > limit) {
            return None;
        }
        let start = Instant::now();
        match search(query, k, ef) {
            Ok(results) => {
                hits += results
                    .iter()
                    .take(k)
                    .filter(|(id, _)| expected.iter().take(k).any(|t| t == id))
                    .count();
            }
            Err(error) => {
                errors += 1;
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
        latencies.push(start.elapsed().as_secs_f64() * 1e6);
    }
    let total_s = loop_start.elapsed().as_secs_f64();
    let dists = vicinity::distance_stats::take_distance_count();
    let denominator = (queries.len() * k) as f64;
    Some(Stats {
        recall: hits as f64 / denominator,
        p50: percentile(&mut latencies.clone(), 50.0),
        p95: percentile(&mut latencies.clone(), 95.0),
        p99: percentile(&mut latencies, 99.0),
        qps: queries.len() as f64 / total_s.max(f64::EPSILON),
        // A number only when the counter exists; when the feature is off the
        // cell stays null rather than claiming "0 distances" (which would be a
        // false claim that every query was free).
        dists_per_query: if cfg!(feature = "distance-stats") {
            dists as f64 / queries.len() as f64
        } else {
            f64::NAN
        },
        errors,
        first_error,
    })
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

struct Row {
    family: &'static str,
    /// Heap or file cache. `diskann-file` is a different row, not a naming
    /// convention a reader has to decode.
    storage_mode: &'static str,
    dataset: String,
    knob: &'static str,
    knob_value: usize,
    recall: f64,
    ratio: f64,
    p50: f64,
    p95: f64,
    p99: f64,
    qps: f64,
    dists_per_query: f64,
    build_s: f64,
    mem_mib: f64,
    rss_mib: f64,
    index_disk_mib: f64,
    persist_s: f64,
    n: u64,
    dim: usize,
    gt_verified: bool,
    gt_cache: String,
    errors: usize,
    first_error: Option<String>,
}

/// Where a build put its measurements. One per family per run; every row from
/// that family carries the same values.
struct BuildReport {
    build_s: f64,
    mem_mib: f64,
    rss_mib: f64,
    index_disk_mib: f64,
    persist_s: f64,
}

#[allow(clippy::too_many_arguments)]
fn push_rows(
    rows: &mut Vec<Row>,
    family: &'static str,
    knob: &'static str,
    knob_value: usize,
    stats: Stats,
    report: &BuildReport,
    args: &Args,
    verify: bool,
    gt_path: &Path,
    exact_timed: f64,
) {
    rows.push(Row {
        family,
        storage_mode: if family.ends_with("-file") {
            "file"
        } else {
            "resident"
        },
        dataset: format!("{}@{}", args.domain.name(), tier_label(args.points)),
        knob,
        knob_value,
        recall: stats.recall,
        ratio: exact_timed / stats.p50.max(f64::MIN_POSITIVE),
        p50: stats.p50,
        p95: stats.p95,
        p99: stats.p99,
        qps: stats.qps,
        dists_per_query: stats.dists_per_query,
        build_s: report.build_s,
        mem_mib: report.mem_mib,
        rss_mib: report.rss_mib,
        index_disk_mib: report.index_disk_mib,
        persist_s: report.persist_s,
        n: args.points,
        dim: args.dim,
        gt_verified: verify,
        gt_cache: gt_path.display().to_string(),
        errors: stats.errors,
        first_error: stats.first_error,
    });
}

fn print_rows(rows: &[Row], count: u64, dim: usize, queries: usize) {
    println!("## large_scale: n={count} dim={dim} queries={queries}");
    println!();
    println!(
        "| family | knob | recall@10 | 1/ratio@10 | p50 us | p95 us | p99 us | QPS | build s | \
         index mem MiB | RSS at query MiB | index disk MiB | dists/q |"
    );
    println!(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | \
         ---: |"
    );
    for row in rows {
        let disk = if row.index_disk_mib.is_finite() {
            format!("{:.1}", row.index_disk_mib)
        } else {
            "-".to_string()
        };
        println!(
            "| {} | {}={} | {:.3} | {:.2} | {:.1} | {:.1} | {:.1} | {:.0} | {:.2} | {:.1} | \
             {:.1} | {} | {} |",
            row.family,
            row.knob,
            row.knob_value,
            row.recall,
            row.ratio,
            row.p50,
            row.p95,
            row.p99,
            row.qps,
            row.build_s,
            row.mem_mib,
            row.rss_mib,
            disk,
            if cfg!(feature = "distance-stats") {
                format!("{:.0}", row.dists_per_query)
            } else {
                "-".to_string()
            }
        );
        if row.errors > 0 {
            println!(
                "| ^ {} query errors: {} | | | | | | | | | | | | |",
                row.errors,
                row.first_error.as_deref().unwrap_or("unknown")
            );
        }
    }
    println!();
}

fn write_jsonl(path: &Path, rows: &[Row], args: &Args, gt_cache: &Path, dataset: &Path) -> Res<()> {
    use std::fmt::Write as _;

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|err| format!("cannot create {}: {err}", parent.display()))?;
        }
    }
    let mut out = String::new();
    let families: Vec<&str> = args.families.iter().map(|family| family.name()).collect();
    // Provenance the ratchet reads: the timing gate needs host, cores and the
    // one-minute load to decide whether a timing comparison means anything.
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let load1 = std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|text| text.split_whitespace().next()?.parse::<f64>().ok());
    let host = std::env::var("RUNNER_OS")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| std::env::consts::OS.to_string());
    let _ = writeln!(
        out,
        "{{\"_meta\":true,\"schema\":1,\"harness\":\"large_scale\",\"host\":\"{host}\",\
         \"cores\":{cores},\"load1\":{},\"n\":{},\"dim\":{},\
         \"k\":{},\"queries\":{},\"dataset\":\"{}\",\"tier\":\"{}\",\"seed\":\"0x{SEED:016X}\",\
         \"mmap\":true,\"large_scale\":true,\"batch\":{},\"families\":\"{}\",\
         \"diskann_cache_mib\":{},\"distance_stats\":{},\"compact_upper\":{},\"max_points\":{},\
         \"budget_seconds\":{},\"vicinity\":\"{}\",\"gt_cache\":\"{}\",\"data_file\":\"{}\"}}",
        match load1 {
            Some(value) => format!("{value:.2}"),
            None => "null".to_string(),
        },
        args.points,
        args.dim,
        args.k,
        args.gt_queries,
        args.domain.name(),
        tier_label(args.points),
        effective_batch(args),
        families.join(","),
        args.diskann_cache_mib,
        cfg!(feature = "distance-stats"),
        cfg!(feature = "compact-hnsw") && std::env::var_os("ANN_COMPACT_UPPER").is_some(),
        args.max_points,
        json_num(args.budget_seconds, 1),
        env!("CARGO_PKG_VERSION"),
        json_escape(&gt_cache.display().to_string()),
        json_escape(&dataset.display().to_string()),
    );
    for row in rows {
        let _ = writeln!(
            out,
            "{{\"family\":\"{}\",\"storage_mode\":\"{}\",\"dataset\":\"{}\",\"knob\":\"{}\",\"knob_value\":{},\
             \"recall_at_10\":{},\"recall\":{},\"inverse_ratio\":{},\"ratio\":{},\
             \"p50_us\":{},\"p95_us\":{},\"p99_us\":{},\"qps\":{},\"dists_per_query\":{},\
             \"build_s\":{},\"mem_mib\":{},\"rss_mib\":{},\"index_disk_mib\":{},\"persist_s\":{},\
             \"n\":{},\"dim\":{},\"gt_verified\":{},\"gt_cache\":\"{}\"}}",
            row.family,
            row.storage_mode,
            row.dataset,
            row.knob,
            row.knob_value,
            json_num(row.recall, 4),
            json_num(row.recall, 4),
            json_num(row.ratio, 4),
            json_num(row.ratio, 4),
            json_num(row.p50, 1),
            json_num(row.p95, 1),
            json_num(row.p99, 1),
            json_num(row.qps, 1),
            json_num(row.dists_per_query, 1),
            json_num(row.build_s, 4),
            json_num(row.mem_mib, 1),
            json_num(row.rss_mib, 1),
            json_num(row.index_disk_mib, 1),
            json_num(row.persist_s, 3),
            row.n,
            row.dim,
            row.gt_verified,
            json_escape(&row.gt_cache),
        );
    }
    std::fs::write(path, out).map_err(|err| format!("cannot write {}: {err}", path.display()))?;
    println!(
        "wrote {} rows to {} (scripts/benefits.py {})",
        rows.len(),
        path.display(),
        path.display()
    );
    Ok(())
}

fn json_num(value: f64, precision: usize) -> String {
    if value.is_finite() {
        format!("{value:.precision$}")
    } else {
        "null".to_string()
    }
}

fn json_escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

fn percentile(samples: &mut [f64], p: f64) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    samples.sort_by(f64::total_cmp);
    let rank = ((p / 100.0) * samples.len() as f64).ceil() as usize;
    samples[rank.saturating_sub(1).min(samples.len() - 1)]
}

fn rss_mib() -> Option<f64> {
    let pid = std::process::id().to_string();
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let kib: f64 = text.trim().parse().ok()?;
    Some(kib / 1024.0)
}

fn bytes_to_mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn string_err<E: std::fmt::Display>(error: E) -> String {
    error.to_string()
}

// ---------------------------------------------------------------------------
// Estimate: what the guardrail compares against --budget-seconds
// ---------------------------------------------------------------------------

struct Estimate {
    generation: f64,
    ground_truth: f64,
    hnsw_build: f64,
    diskann_build: f64,
    persist: f64,
    total: f64,
}

fn estimate(args: &Args, remaining_rows: u64, gt_cached: bool) -> Estimate {
    // Conservative coefficients measured on a 16-core arm64 workstation. The
    // 100M corpus was written at ~640 MB/s, so generation at 50e6 components/s
    // (200 MB/s) is ~3x conservative; HNSW construction measured 1.2-1.6e-4
    // s/point and DiskANN's two-pass Vamana 1.2-1.3e-4 s/point at 100k-3M, and
    // both coefficients below sit above those so a loaded host still fits.
    let generation = (remaining_rows as f64 * args.dim as f64) / 50e6;
    let ground_truth = if gt_cached {
        0.0
    } else {
        // Measured at 100M: 500 queries x 100M rows in 45.8 s = 1092 M
        // 96-dim comparisons/s across 16 threads. 1.0e11 is that rate with
        // ~10% of headroom.
        (args.gt_queries as f64 * args.points as f64 * args.dim as f64) / 1.0e11
    };
    // `--gt-only` stops before any index exists, so it must not be refused for
    // build time it will never spend.
    let builds = !args.gt_only;
    let hnsw_build = if builds && args.families.contains(&Family::Hnsw) {
        args.points as f64 * 1.8e-4
    } else {
        0.0
    };
    let diskann_build = if builds && args.families.contains(&Family::Diskann) {
        args.points as f64 * 1.6e-4
    } else {
        0.0
    };
    let persist_per_family = (args.points as f64 * args.dim as f64 * 4.0) / 300e6;
    let persist = if args.persist && builds {
        persist_per_family * args.families.len() as f64
    } else {
        0.0
    };
    let build = hnsw_build + diskann_build;
    let total = generation + ground_truth + build + persist;
    Estimate {
        generation,
        ground_truth,
        hnsw_build,
        diskann_build,
        persist,
        total,
    }
}

// ---------------------------------------------------------------------------
// Self-test: the two hard claims, checked without a large run
// ---------------------------------------------------------------------------

/// 1. Truncating a generated dataset and regenerating restores it byte for
///    byte (per-row seeding is what makes resume safe).
/// 2. A cached ground truth round-trips and matches brute force on both f32
///    and f64 accumulation.
fn self_test() -> Res<()> {
    println!("# large_scale self-test");
    let root = data_root().join("selftest");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root)
        .map_err(|err| format!("cannot create {}: {err}", root.display()))?;

    let domain = Domain::Neardup;
    let dim = 8;
    let count = 4096u64;
    let k = 10;
    let query_count = 200;
    let ctx = DomainContext::new(domain, SEED, dim);
    let path = dataset_path(&root, domain, count, dim);

    let families = [Family::Hnsw, Family::Diskann];
    let first = ensure_dataset(&ctx, &path, count, dim, &families)?;
    if first.generated != count || first.resumed_from != 0 {
        return Err(format!(
            "self-test: fresh generation reported {} rows from {}",
            first.generated, first.resumed_from
        ));
    }
    let complete = std::fs::read(&path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let stride = dim as u64 * 4;
    let truncated = DATA_HEADER_BYTES + (count - 37) * stride + 11; // partial row too
    let file = OpenOptions::new()
        .write(true)
        .open(&path)
        .map_err(|err| format!("open {}: {err}", path.display()))?;
    file.set_len(truncated)
        .map_err(|err| format!("truncate {}: {err}", path.display()))?;
    drop(file);

    let resumed = ensure_dataset(&ctx, &path, count, dim, &families)?;
    if resumed.resumed_from != count - 37 || resumed.generated != 37 {
        return Err(format!(
            "self-test: resume reported {} rows from {}",
            resumed.generated, resumed.resumed_from
        ));
    }
    let restored = std::fs::read(&path).map_err(|err| format!("read {}: {err}", path.display()))?;
    if restored != complete {
        return Err("self-test: resumed dataset differs from the uninterrupted bytes".to_string());
    }
    println!(
        "generation resume: PASS (truncated to {} rows, restored byte-identically)",
        count - 37
    );

    let dataset = DatasetFile::open(&path)?;
    let queries = generate_queries(&ctx, query_count);
    let hash = query_hash(&queries);
    let cache = gt_path(&root, domain, count, dim, k, query_count, hash);
    let (truth, state) = load_or_compute_ground_truth(&dataset, &queries, k, &cache, true)?;
    if !matches!(state, CacheState::Computed) {
        return Err("self-test: first ground-truth pass was not a computation".to_string());
    }
    let (cached, state) = load_or_compute_ground_truth(&dataset, &queries, k, &cache, true)?;
    if !matches!(state, CacheState::Hit) || cached != truth {
        return Err("self-test: cache hit did not reproduce the computed truth".to_string());
    }
    let mut mismatches = 0usize;
    for (query_index, query) in queries.iter().take(4).enumerate() {
        let reference = brute_top_k(&dataset, query, k);
        if reference != cached[query_index] {
            mismatches += 1;
        }
    }
    verify_ground_truth(&dataset, &queries, &cached, k)?;
    println!(
        "ground-truth cache: PASS (computed, hit, {mismatches} brute-force row(s) differed on 4)"
    );
    println!("self-test: PASS");
    Ok(())
}
