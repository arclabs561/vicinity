//! One protocol, every index family in the crate.
//!
//! This harness puts each feature-gated index family behind a single trait and
//! runs the same dataset, the same ground truth, the same metric and the same
//! query-timing loop for all of them, so the results are comparable and
//! rankable. It is deliberately not the `ann_benchmark` runner: no HDF5
//! datasets, no external baselines, no per-algorithm CLI flags. Build and run:
//!
//! ```bash
//! cargo run --release --example all_families --features hnsw,nsw,nsg,finger,\
//! fresh_graph,filtered_graph,vamana,diskann,ivf_pq,ivf_avq,ivf_rabitq,emg,\
//! curator,lsh,pipnn,rp_quant,sq4,binary_index,sng,range_filtered,kdtree,\
//! balltree,kmeans_tree,rptree
//! ```
//!
//! # Protocol
//!
//! - `n = 10_000` base vectors, `dim = 96`, `k = 10`, `200` held-out queries.
//! - Two datasets: `uniform` (i.i.d. uniform in `[-1, 1]^96`) and `clustered`
//!   (16 uniform centres, `sigma = 0.25` noise). Both are L2-normalised, so
//!   cosine, angular and L2 ranking agree and every family can be fed the same
//!   vectors even though some are cosine-only (Vamana, EMG, RaBitQ) and some
//!   rank by L2 (IVF-PQ, k-d tree).
//! - Ground truth: exact top-10 by inner product on the normalised base set,
//!   computed once per dataset before any index is built or timed.
//! - Per query: recall@10, and one wall-clock sample. `p50/p95/p99` are over
//!   those per-query samples; `QPS` is queries divided by the whole timed loop.
//!   Five untimed warm-up queries run before the loop.
//! - `dists/q` is distance computations per query, counted by
//!   [`vicinity::distance_stats`] between the end of the warm-up and the end of
//!   the timed loop. It is the hardware-independent cost signal the ANN
//!   benchmarking literature reports (Aumüller & Ceccarello, 2023): wall clock
//!   moves with the machine, distance computations do not. The column only has
//!   numbers when the runner is built with `--features distance-stats`;
//!   otherwise it reads `-` and the cost of the counting is not paid.
//! - `1/ratio@10` is the time-based definition used by ann-benchmarks: exact
//!   linear-scan p50 divided by the family's p50. Brute force is timed in the
//!   same process, single-threaded. Higher is faster than exact search.
//! - Build wall-clock time excludes dataset generation and ground truth.
//!
//! # Parameter sweeps
//!
//! A family exposes either a search-time knob (a per-call `ef`, `nprobe`, ...)
//! or only a construction-time knob. Search-time families are built once per
//! dataset and swept in place. Construction-time families are rebuilt for each
//! sweep point; their `build s` column is the build that produced that row.
//! Families with no search-time knob at all (`sng`) contribute one row.
//!
//! A family whose build or query fails is reported with the error instead of
//! being dropped silently. Pass a family name as the first argument to rerun
//! just that family (`cargo run --release --example all_families -- ivf_pq`).
//!
//! # JSONL output
//!
//! `--json` replaces the markdown report with one JSON object per row on
//! stdout (or in `--out <path>`), for the performance ratchet
//! (`scripts/ratchet.py`) and for diffing two runs. `scripts/ratchet.py`
//! consumes this format; `benchmarks/README.md` documents the contract. The
//! first line is a `_meta` object with the protocol constants, toolchain,
//! and host, so a comparison can refuse to mix hosts. Failures are emitted as
//! `_failure` objects instead of being folded into the report.

// Sweep-value constants are dead when the family that reads them is not
// compiled in (`--features hnsw` and nothing else), which is a normal way to
// run this harness; the alternative is a `cfg` on every constant.
#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

use std::time::Instant;

/// Recall is compared at three decimals so a 0.99958 and a 0.99999 tie and the
/// faster row wins; below that, differences are noise at 200 queries.
fn recall3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

const N: usize = 10_000;
const DIM: usize = 96;
const K: usize = 10;
const N_QUERIES: usize = 200;
const CENTRES: usize = 16;
const SEED: u64 = 0x4641_4d49_4c59; // "FAMILY"

const EF_VALUES: &[usize] = &[10, 20, 50, 100, 200];
const NPROBE_VALUES: &[usize] = &[1, 2, 4, 8, 16, 32];
const CURATOR_EF_VALUES: &[usize] = &[50, 100, 200, 400, 800];
const RERANK_VALUES: &[usize] = &[1, 2, 5, 10, 20];
const PROBE_VALUES: &[usize] = &[1, 2, 4, 8, 16];
const LEAF_VALUES: &[usize] = &[5, 10, 25, 50, 100];
const BRANCH_VALUES: &[usize] = &[1, 2, 4, 8, 16];
const NO_KNOB: &[usize] = &[0];

/// A built index that can answer one query.
trait Searcher {
    /// `knob` is the family's swept search parameter. Families without one
    /// ignore it.
    fn search(&mut self, query: &[f32], k: usize, knob: usize) -> Result<Vec<(u32, f32)>, String>;
}

/// Adapts a closure to [`Searcher`] so each family needs one builder function
/// instead of its own struct.
struct FnIndex<F>(F)
where
    F: FnMut(&[f32], usize, usize) -> Result<Vec<(u32, f32)>, String>;

impl<F> Searcher for FnIndex<F>
where
    F: FnMut(&[f32], usize, usize) -> Result<Vec<(u32, f32)>, String>,
{
    fn search(&mut self, query: &[f32], k: usize, knob: usize) -> Result<Vec<(u32, f32)>, String> {
        (self.0)(query, k, knob)
    }
}

type BuildFn = fn(&[Vec<f32>], usize, usize) -> Result<Box<dyn Searcher>, String>;

struct Family {
    name: &'static str,
    knob: &'static str,
    values: &'static [usize],
    /// `true` when the knob only affects construction, so each sweep point
    /// needs a fresh index.
    rebuild: bool,
    /// Fixed construction configuration and search path taken.
    note: &'static str,
    build: BuildFn,
}

struct Row {
    dataset: &'static str,
    family: &'static str,
    knob: &'static str,
    knob_value: usize,
    recall: f64,
    ratio: f64,
    p50: f64,
    p95: f64,
    p99: f64,
    qps: f64,
    /// Distance computations per query (0 when `distance-stats` is off).
    dists_per_query: f64,
    build_s: f64,
    /// Resident-set growth across the build, in MiB. `None` when the RSS probe
    /// is unavailable, which must not read as a measurement of zero.
    mem_mib: Option<f64>,
    errors: usize,
    first_error: Option<String>,
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

struct Dataset {
    name: &'static str,
    base: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
    truth: Vec<Vec<u32>>,
    exact_p50_us: f64,
}

struct Skipped {
    family: &'static str,
    reason: &'static str,
}

/// Command-line options: a family filter, and how to report.
struct Options {
    /// Positional filter: `all_families ivf_pq` reruns one family.
    only: Option<String>,
    /// Emit JSONL instead of the markdown report.
    json: bool,
    /// Where JSONL goes; stdout when absent.
    out: Option<String>,
}

fn parse_options() -> Result<Options, String> {
    let mut opts = Options {
        only: None,
        json: false,
        out: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => opts.json = true,
            "--out" => {
                opts.out = Some(args.next().ok_or("--out needs a path")?);
            }
            "--help" | "-h" => {
                println!(
                    "usage: all_families [FAMILY] [--json] [--out PATH]\n\n\
                     Runs every enabled family on the same protocol. Without --json it\n\
                     prints markdown; with --json it prints one JSON object per row."
                );
                std::process::exit(0);
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown flag `{other}` (try --help)"));
            }
            other => {
                if opts.only.is_some() {
                    return Err(format!("only one family filter is accepted, got `{other}`"));
                }
                opts.only = Some(other.to_string());
            }
        }
    }
    Ok(opts)
}

fn main() {
    let opts = match parse_options() {
        Ok(opts) => opts,
        Err(message) => {
            eprintln!("all_families: {message}");
            std::process::exit(2);
        }
    };
    let mut datasets = build_datasets();
    let families: Vec<Family> = families()
        .into_iter()
        .filter(|f| opts.only.as_deref().is_none_or(|name| name == f.name))
        .collect();
    let mut rows: Vec<Row> = Vec::new();
    let mut failures: Vec<(String, String)> = Vec::new();

    if !opts.json {
        println!("# All-family ANN benchmark");
        println!();
        println!(
            "n={N} dim={DIM} k={K} queries={N_QUERIES} seed=0x{SEED:016X} \
             datasets=uniform,clustered(16 centres) metric=cosine (L2-normalised)"
        );
        println!();

        for dataset in &mut datasets {
            println!(
                "exact linear scan p50 on `{}`: {:.1} us/query",
                dataset.name, dataset.exact_p50_us
            );
        }
        println!();
        println!(
            "1/ratio@10 = exact-scan p50 / family p50 (time-based, as in ann-benchmarks); \
             QPS = queries / total timed-loop seconds; p50/p95/p99 are per-query wall clock."
        );
        println!();
    }

    for dataset in &datasets {
        for family in &families {
            if opts.json {
                // The JSONL goes to stdout (or a file) and only lands at the
                // end; a full sweep takes minutes, so narrate on stderr.
                eprintln!(
                    "all_families: building and measuring {}/{}",
                    family.name, dataset.name
                );
            }
            match run_family(family, dataset) {
                Ok(mut produced) => rows.append(&mut produced),
                Err(reason) => {
                    if !failures.iter().any(|(name, _)| name == family.name) {
                        failures.push((family.name.to_string(), reason));
                    }
                }
            }
        }
    }

    if opts.json {
        emit_jsonl(&rows, &failures, &datasets, opts.out.as_deref());
        return;
    }

    print_rows(&rows);
    print_ranking(&rows);
    print_at_recall(&rows, 0.95);
    print_sweeps(&families);
    print_skipped(&failures);
}

// ---------------------------------------------------------------------------
// Protocol: datasets, ground truth, timing
// ---------------------------------------------------------------------------

fn build_datasets() -> Vec<Dataset> {
    let uniform = make_dataset("uniform", SEED, None);
    let clustered = make_dataset("clustered", SEED ^ 0xC1_0000, Some(0.25));
    vec![uniform, clustered]
}

fn make_dataset(name: &'static str, seed: u64, spread: Option<f32>) -> Dataset {
    use rand::{rngs::StdRng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);
    let centres = sample_centres(&mut rng, spread.is_some());
    let base = sample_points(&mut rng, &centres, spread, N);
    let queries = sample_points(&mut rng, &centres, spread, N_QUERIES);
    let truth: Vec<Vec<u32>> = queries.iter().map(|q| exact_top_k(&base, q, K)).collect();

    // Time the exact scan the same way queries are timed below, so `1/ratio`
    // uses a like-for-like p50.
    for q in queries.iter().take(5) {
        let _ = exact_top_k(&base, q, K);
    }
    let mut lat = Vec::with_capacity(queries.len());
    for q in &queries {
        let t = Instant::now();
        let _ = exact_top_k(&base, q, K);
        lat.push(t.elapsed().as_secs_f64() * 1e6);
    }

    Dataset {
        name,
        base,
        queries,
        truth,
        exact_p50_us: percentile(&mut lat.clone(), 50.0),
    }
}

fn sample_centres(rng: &mut rand::rngs::StdRng, needed: bool) -> Vec<Vec<f32>> {
    if !needed {
        return Vec::new();
    }
    (0..CENTRES).map(|_| normalize(sample_raw(rng))).collect()
}

fn sample_points(
    rng: &mut rand::rngs::StdRng,
    centres: &[Vec<f32>],
    spread: Option<f32>,
    count: usize,
) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            let noise = sample_raw(rng);
            match spread {
                None => normalize(noise),
                Some(sigma) => {
                    let centre = &centres[i % centres.len()];
                    let mixed: Vec<f32> = centre
                        .iter()
                        .zip(noise.iter())
                        .map(|(c, n)| c + sigma * n)
                        .collect();
                    normalize(mixed)
                }
            }
        })
        .collect()
}

fn sample_raw(rng: &mut rand::rngs::StdRng) -> Vec<f32> {
    use rand::Rng;
    (0..DIM).map(|_| rng.random_range(-1.0f32..1.0)).collect()
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Exact top-k by inner product. This is both the ground truth and the
/// brute-force baseline whose p50 defines `1/ratio`.
fn exact_top_k(base: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = base
        .iter()
        .enumerate()
        .map(|(i, v)| (dot(v, query), i as u32))
        .collect();
    if k < scored.len() {
        scored.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
        scored.truncate(k);
    }
    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    scored.into_iter().map(|(_, id)| id).collect()
}

fn evaluate(index: &mut dyn Searcher, dataset: &Dataset, knob: usize) -> Stats {
    for query in dataset.queries.iter().take(5) {
        let _ = index.search(query, K, knob);
    }

    // Warm-up is excluded from both the timing and the distance count, so the
    // two are measured over exactly the same calls.
    vicinity::distance_stats::reset_distance_count();

    let mut lat = Vec::with_capacity(dataset.queries.len());
    let mut hits = 0usize;
    let mut errors = 0usize;
    let mut first_error = None;

    let loop_start = Instant::now();
    for (query, truth) in dataset.queries.iter().zip(dataset.truth.iter()) {
        let start = Instant::now();
        match index.search(query, K, knob) {
            Ok(results) => {
                hits += intersection_count(&results, truth, K);
            }
            Err(err) => {
                errors += 1;
                if first_error.is_none() {
                    first_error = Some(err);
                }
            }
        }
        lat.push(start.elapsed().as_secs_f64() * 1e6);
    }
    let total_s = loop_start.elapsed().as_secs_f64();
    let dists = vicinity::distance_stats::take_distance_count();

    let denominator = (dataset.queries.len() * K) as f64;
    Stats {
        recall: hits as f64 / denominator,
        p50: percentile(&mut lat.clone(), 50.0),
        p95: percentile(&mut lat.clone(), 95.0),
        p99: percentile(&mut lat.clone(), 99.0),
        qps: dataset.queries.len() as f64 / total_s.max(f64::EPSILON),
        dists_per_query: if cfg!(feature = "distance-stats") {
            dists as f64 / dataset.queries.len() as f64
        } else {
            0.0
        },
        errors,
        first_error,
    }
}

/// `dists/q` cell: a number under `distance-stats`, `-` otherwise (the counter
/// reports 0 when it is compiled out, which would read as a real measurement).
fn dists_cell(row: &Row) -> String {
    if cfg!(feature = "distance-stats") {
        format!("{:.0}", row.dists_per_query)
    } else {
        "-".to_string()
    }
}

/// `mem MiB` cell: `-` when the RSS probe returned nothing.
fn mem_cell(row: &Row) -> String {
    match row.mem_mib {
        Some(mib) => format!("{mib:.1}"),
        None => "-".to_string(),
    }
}

// ---------------------------------------------------------------------------
// JSONL output (consumed by `scripts/ratchet.py`)
// ---------------------------------------------------------------------------

/// Escape a string for a JSON string literal. Every string written here is an
/// identifier or an error message, so this only needs the mandatory escapes.
fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// A finite `f64` as JSON, `null` for NaN/inf so the document stays valid.
fn json_number(value: f64, decimals: usize) -> String {
    if value.is_finite() {
        format!("{value:.decimals$}")
    } else {
        "null".to_string()
    }
}

/// First line of `<tool> --version`, or `unknown` when the tool is missing.
fn tool_version(tool: &str) -> String {
    std::process::Command::new(tool)
        .arg("--version")
        .output()
        .ok()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// One-minute load average: `/proc/loadavg` on Linux, `sysctl` on macOS.
///
/// Wall clock on a loaded host measures the scheduler, not the index. The
/// ratchet reads this to decide whether a timing gate means anything.
fn one_minute_load() -> Option<f64> {
    if let Ok(text) = std::fs::read_to_string("/proc/loadavg") {
        if let Some(first) = text.split_whitespace().next() {
            if let Ok(value) = first.parse() {
                return Some(value);
            }
        }
    }
    let output = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    text.split(|ch: char| !(ch.is_ascii_digit() || ch == '.'))
        .find(|token| !token.is_empty())
        .and_then(|token| token.parse().ok())
}

/// Host fields that let a consumer refuse to compare timings across machines.
fn host_fields() -> (String, String) {
    let arch = std::env::consts::ARCH.to_string();
    let os = std::env::consts::OS.to_string();
    let kernel = std::process::Command::new("uname")
        .arg("-r")
        .output()
        .ok()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    (format!("{arch}-{os}"), kernel)
}

/// Write one JSON object per measured row, preceded by a `_meta` line.
///
/// The field names are the contract with `scripts/ratchet.py`:
/// `recall_at_10` and `inverse_ratio` are the `recall@10` / `1/ratio@10`
/// columns, and `dists_per_query` is `null` unless the harness was built with
/// `--features distance-stats`.
fn emit_jsonl(
    rows: &[Row],
    failures: &[(String, String)],
    datasets: &[Dataset],
    out: Option<&str>,
) {
    let (host, kernel) = host_fields();
    let mut buf = String::new();

    let exact: Vec<String> = datasets
        .iter()
        .map(|d| {
            format!(
                "\"{}\":{}",
                json_escape(d.name),
                json_number(d.exact_p50_us, 3)
            )
        })
        .collect();
    let cores = std::thread::available_parallelism()
        .ok()
        .map(std::num::NonZeroUsize::get);
    buf.push_str(&format!(
        "{{\"_meta\":true,\"schema\":1,\"harness\":\"all_families\",\"n\":{N},\"dim\":{DIM},\
         \"k\":{K},\"queries\":{N_QUERIES},\"seed\":\"0x{SEED:016X}\",\"host\":\"{}\",\
         \"kernel\":\"{}\",\"rustc\":\"{}\",\"vicinity\":\"{}\",\"target\":\"{}\",\
         \"cores\":{},\"load1\":{},\"distance_stats\":{},\"exact_p50_us\":{{{}}}}}\n",
        json_escape(&host),
        json_escape(&kernel),
        json_escape(&tool_version("rustc")),
        env!("CARGO_PKG_VERSION"),
        json_escape(std::env::consts::ARCH),
        cores.map_or_else(|| "null".to_string(), |n| n.to_string()),
        one_minute_load().map_or_else(|| "null".to_string(), |load| format!("{load:.2}")),
        cfg!(feature = "distance-stats"),
        exact.join(","),
    ));

    for row in rows {
        let dists = if cfg!(feature = "distance-stats") {
            json_number(row.dists_per_query, 3)
        } else {
            "null".to_string()
        };
        let mem = match row.mem_mib {
            Some(mib) => json_number(mib, 3),
            None => "null".to_string(),
        };
        buf.push_str(&format!(
            "{{\"family\":\"{}\",\"dataset\":\"{}\",\"knob\":\"{}\",\"knob_value\":{},\
             \"recall_at_10\":{},\"inverse_ratio\":{},\"p50_us\":{},\"p95_us\":{},\
             \"p99_us\":{},\"qps\":{},\"build_s\":{},\"mem_mib\":{},\"dists_per_query\":{}}}",
            json_escape(row.family),
            json_escape(row.dataset),
            json_escape(row.knob),
            row.knob_value,
            json_number(row.recall, 6),
            json_number(row.ratio, 6),
            json_number(row.p50, 3),
            json_number(row.p95, 3),
            json_number(row.p99, 3),
            json_number(row.qps, 3),
            json_number(row.build_s, 6),
            mem,
            dists,
        ));
        match &row.first_error {
            None => buf.push('\n'),
            Some(err) => buf.push_str(&format!(
                ",\"query_errors\":{},\"first_error\":\"{}\"}}\n",
                row.errors,
                json_escape(err)
            )),
        }
    }

    for (family, reason) in failures {
        buf.push_str(&format!(
            "{{\"_failure\":true,\"family\":\"{}\",\"error\":\"{}\"}}\n",
            json_escape(family),
            json_escape(reason)
        ));
    }

    match out {
        Some(path) => {
            if let Err(err) = std::fs::write(path, buf) {
                eprintln!("all_families: cannot write {path}: {err}");
                std::process::exit(2);
            }
        }
        None => print!("{buf}"),
    }
}

fn intersection_count(results: &[(u32, f32)], truth: &[u32], k: usize) -> usize {
    results
        .iter()
        .take(k)
        .filter(|(id, _)| truth.iter().take(k).any(|t| t == id))
        .count()
}

fn percentile(samples: &mut [f64], p: f64) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    samples.sort_by(f64::total_cmp);
    let rank = ((p / 100.0) * samples.len() as f64).ceil() as usize;
    samples[rank.saturating_sub(1).min(samples.len() - 1)]
}

/// Resident-set size of this process in MiB, measured with `ps` so it works on
/// macOS and Linux without a libc dependency.
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

fn run_family(family: &Family, dataset: &Dataset) -> Result<Vec<Row>, String> {
    let mut rows = Vec::new();

    if family.rebuild {
        for &knob in family.values {
            let rss_before = rss_mib();
            let build_start = Instant::now();
            let mut index = (family.build)(&dataset.base, DIM, knob)
                .map_err(|e| format!("build failed at {}={knob}: {e}", family.knob))?;
            let build_s = build_start.elapsed().as_secs_f64();
            let mem_mib = match (rss_before, rss_mib()) {
                (Some(before), Some(after)) => Some((after - before).max(0.0)),
                _ => None,
            };
            rows.push(row_for(
                family,
                dataset,
                knob,
                build_s,
                mem_mib,
                &mut *index,
            ));
        }
    } else {
        let rss_before = rss_mib();
        let build_start = Instant::now();
        let mut index = (family.build)(&dataset.base, DIM, family.values[0])
            .map_err(|e| format!("build failed: {e}"))?;
        let build_s = build_start.elapsed().as_secs_f64();
        let mem_mib = match (rss_before, rss_mib()) {
            (Some(before), Some(after)) => Some((after - before).max(0.0)),
            _ => None,
        };
        for &knob in family.values {
            rows.push(row_for(
                family,
                dataset,
                knob,
                build_s,
                mem_mib,
                &mut *index,
            ));
        }
    }

    Ok(rows)
}

fn row_for(
    family: &Family,
    dataset: &Dataset,
    knob: usize,
    build_s: f64,
    mem_mib: Option<f64>,
    index: &mut dyn Searcher,
) -> Row {
    let stats = evaluate(index, dataset, knob);
    Row {
        dataset: dataset.name,
        family: family.name,
        knob: family.knob,
        knob_value: knob,
        recall: stats.recall,
        ratio: dataset.exact_p50_us / stats.p50.max(f64::MIN_POSITIVE),
        p50: stats.p50,
        p95: stats.p95,
        p99: stats.p99,
        qps: stats.qps,
        dists_per_query: stats.dists_per_query,
        build_s,
        mem_mib,
        errors: stats.errors,
        first_error: stats.first_error,
    }
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

fn print_rows(rows: &[Row]) {
    for dataset in ["uniform", "clustered"] {
        println!("## {dataset}");
        println!();
        println!(
            "| family | knob | recall@10 | 1/ratio@10 | p50 us | dists/q | p95 us | p99 us | QPS | build s | mem MiB |"
        );
        println!("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
        for row in rows.iter().filter(|r| r.dataset == dataset) {
            println!(
                "| {} | {}={} | {:.3} | {:.2} | {:.1} | {} | {:.1} | {:.1} | {:.0} | {:.2} | {} |",
                row.family,
                row.knob,
                row.knob_value,
                row.recall,
                row.ratio,
                row.p50,
                dists_cell(row),
                row.p95,
                row.p99,
                row.qps,
                row.build_s,
                mem_cell(row)
            );
            if row.errors > 0 {
                println!(
                    "| ^ {} query errors: {} | | | | | | | | | | |",
                    row.errors,
                    row.first_error.as_deref().unwrap_or("unknown")
                );
            }
        }
        println!();
    }
}

/// One line per family: the best operating point by recall, tie-broken by p50.
fn print_ranking(rows: &[Row]) {
    for dataset in ["uniform", "clustered"] {
        println!("## {dataset} ranking (best recall, tie-break p50)");
        println!();
        println!("| rank | family | at | recall@10 | p50 us | build s |");
        println!("| ---: | --- | --- | ---: | ---: | ---: |");
        let mut best: Vec<&Row> = Vec::new();
        for row in rows.iter().filter(|r| r.dataset == dataset) {
            match best
                .iter_mut()
                .find(|existing| existing.family == row.family)
            {
                Some(existing) => {
                    let (row_r, old_r) = (recall3(row.recall), recall3(existing.recall));
                    if row_r > old_r || (row_r == old_r && row.p50 < existing.p50) {
                        *existing = row;
                    }
                }
                None => best.push(row),
            }
        }
        best.sort_by(|a, b| {
            recall3(b.recall)
                .total_cmp(&recall3(a.recall))
                .then_with(|| a.p50.total_cmp(&b.p50))
        });
        for (i, row) in best.iter().enumerate() {
            println!(
                "| {} | {} | {}={} | {:.3} | {:.1} | {:.2} |",
                i + 1,
                row.family,
                row.knob,
                row.knob_value,
                row.recall,
                row.p50,
                row.build_s
            );
        }
        println!();
    }
}

/// Fastest operating point per family subject to `recall >= target`. Families
/// that never reach the target contribute their highest-recall row, marked `*`,
/// so the gap is visible instead of silently dropping them.
fn print_at_recall(rows: &[Row], target: f64) {
    for dataset in ["uniform", "clustered"] {
        println!(
            "## {dataset} at recall@10 >= {target:.2} (fastest p50; `*` = target not reached)"
        );
        println!();
        println!("| family | at | recall@10 | p50 us | QPS | 1/ratio@10 |");
        println!("| --- | --- | ---: | ---: | ---: | ---: |");
        let mut chosen: Vec<(bool, &Row)> = Vec::new();
        for row in rows.iter().filter(|r| r.dataset == dataset) {
            match chosen.iter_mut().find(|(_, c)| c.family == row.family) {
                Some((ok, current)) => {
                    let row_ok = recall3(row.recall) >= target;
                    if (row_ok && !*ok)
                        || (row_ok == *ok
                            && ((row_ok && row.p50 < current.p50)
                                || (!row_ok && recall3(row.recall) > recall3(current.recall))))
                    {
                        *ok = row_ok;
                        *current = row;
                    }
                }
                None => chosen.push((recall3(row.recall) >= target, row)),
            }
        }
        chosen.sort_by(|a, b| a.1.p50.total_cmp(&b.1.p50));
        for (ok, row) in chosen {
            println!(
                "| {}{} | {}={} | {:.3} | {:.1} | {:.0} | {:.2} |",
                row.family,
                if ok { "" } else { "*" },
                row.knob,
                row.knob_value,
                row.recall,
                row.p50,
                row.qps,
                row.ratio
            );
        }
        println!();
    }
}

fn print_sweeps(families: &[Family]) {
    println!("## Sweeps");
    println!();
    for family in families {
        if family.values == NO_KNOB {
            println!(
                "- `{}`: no search-time knob; one row at defaults. {}",
                family.name, family.note
            );
        } else if family.rebuild {
            println!(
                "- `{}`: `{}` swept {:?}, rebuilt per point. {}",
                family.name, family.knob, family.values, family.note
            );
        } else {
            println!(
                "- `{}`: `{}` swept {:?} on one build. {}",
                family.name, family.knob, family.values, family.note
            );
        }
    }
    println!();
}

fn print_skipped(failures: &[(String, String)]) {
    println!("## Not covered");
    println!();
    println!("| family | why |");
    println!("| --- | --- |");
    for entry in SKIPPED {
        println!("| {} | {} |", entry.family, entry.reason);
    }
    for (family, reason) in failures {
        println!("| {} | runtime: {} |", family, reason);
    }
    println!();
}

const SKIPPED: &[Skipped] = &[
    Skipped {
        family: "evoc",
        reason: "clustering wrapper: exposes fit_predict/cluster_layers, no k-NN query API",
    },
    Skipped {
        family: "lemur",
        reason: "needs externally trained LemurEncoder weights and multi-vector (token) documents; in-tree random weights are test-only",
    },
    Skipped {
        family: "sparse_mips",
        reason: "sparse-only input (SparseVector); dense 96-d vectors are fully dense, so the sparse path is not the family's workload",
    },
];

// ---------------------------------------------------------------------------
// Family builders
// ---------------------------------------------------------------------------

#[allow(clippy::vec_init_then_push)]
fn families() -> Vec<Family> {
    let mut families = Vec::new();

    #[cfg(feature = "hnsw")]
    families.push(Family {
        name: "hnsw",
        knob: "ef_search",
        values: EF_VALUES,
        rebuild: false,
        note: "m=16, ef_construction=200, cosine; ef_search per call",
        build: build_hnsw,
    });

    #[cfg(feature = "nsw")]
    families.push(Family {
        name: "nsw",
        knob: "ef_search",
        values: EF_VALUES,
        rebuild: false,
        note: "m=m_max=16, ef_construction=100; ef_search per call",
        build: build_nsw,
    });

    #[cfg(feature = "nsg")]
    families.push(Family {
        name: "nsg",
        knob: "ef_search",
        values: EF_VALUES,
        rebuild: false,
        note: "R=32, L=64, K=32; search_with_ef",
        build: build_nsg,
    });

    #[cfg(feature = "finger")]
    families.push(Family {
        name: "finger",
        knob: "ef_search",
        values: EF_VALUES,
        rebuild: false,
        note: "degree 32, ef_construction 200, alpha 1.2; search_with_ef",
        build: build_finger,
    });

    #[cfg(feature = "fresh_graph")]
    families.push(Family {
        name: "fresh_graph",
        knob: "ef_search",
        values: EF_VALUES,
        rebuild: false,
        note: "degree 32, ef_construction 200, alpha 1.2; search_with_ef",
        build: build_fresh_graph,
    });

    #[cfg(feature = "filtered_graph")]
    families.push(Family {
        name: "filtered_graph",
        knob: "ef_search",
        values: EF_VALUES,
        rebuild: false,
        note: "degree 32, ef_construction 200; attributes empty (unfiltered path); search_with_ef",
        build: build_filtered_graph,
    });

    #[cfg(feature = "vamana")]
    families.push(Family {
        name: "vamana",
        knob: "ef_search",
        values: EF_VALUES,
        rebuild: false,
        note: "max_degree 64, alpha 1.3, ef_construction 200, seed 42; ef_search per call",
        build: build_vamana,
    });

    #[cfg(feature = "diskann")]
    families.push(Family {
        name: "diskann",
        knob: "ef_search",
        values: EF_VALUES,
        rebuild: false,
        note: "in-memory DiskANNIndex: m=32, ef_construction=100, alpha 1.2, seed 42; ef_search per call",
        build: build_diskann,
    });

    #[cfg(feature = "emg")]
    families.push(Family {
        name: "emg",
        knob: "ef_search",
        values: EF_VALUES,
        rebuild: false,
        note: "degree 32, candidate_size 64, scale_t 32, 2 iterations; search_with_ef",
        build: build_emg,
    });

    #[cfg(feature = "pipnn")]
    families.push(Family {
        name: "pipnn",
        knob: "ef_search",
        values: EF_VALUES,
        rebuild: false,
        note: "leaf 2048, degree 32, 12 hash bits, final RobustPrune; search_with_ef",
        build: build_pipnn,
    });

    #[cfg(feature = "ivf_pq")]
    families.push(Family {
        name: "ivf_pq",
        knob: "nprobe",
        values: NPROBE_VALUES,
        rebuild: false,
        note: "100 clusters (sqrt n), M=16 (16 B/vector, 24x vs 384 B raw), codebook 256; nprobe per call, search_reranked pool 100",
        build: build_ivf_pq,
    });

    #[cfg(feature = "ivf_avq")]
    families.push(Family {
        name: "ivf_avq",
        knob: "nprobe",
        values: NPROBE_VALUES,
        rebuild: false,
        note: "100 partitions (sqrt n), M=16, codebook 256, reorder 100; nprobe per call",
        build: build_ivf_avq,
    });

    #[cfg(feature = "ivf_rabitq")]
    families.push(Family {
        name: "ivf_rabitq",
        knob: "nprobe",
        values: NPROBE_VALUES,
        rebuild: false,
        note: "100 clusters (sqrt n), 4 bits/dim; nprobe per call",
        build: build_ivf_rabitq,
    });

    #[cfg(feature = "lsh")]
    families.push(Family {
        name: "lsh",
        knob: "num_probes",
        values: PROBE_VALUES,
        rebuild: true,
        note: "8 tables, seed 42; num_probes at build (multiprobe width)",
        build: build_lsh,
    });

    #[cfg(feature = "rp_quant")]
    families.push(Family {
        name: "rp_quant",
        knob: "rerank_factor",
        values: RERANK_VALUES,
        rebuild: true,
        note: "64-d projection, seed 42; rerank_factor at build",
        build: build_rp_quant,
    });

    #[cfg(feature = "sq4")]
    families.push(Family {
        name: "sq4",
        knob: "rerank_factor",
        values: RERANK_VALUES,
        rebuild: true,
        note: "4-bit codes, exact rerank; rerank_factor at build",
        build: build_sq4,
    });

    #[cfg(feature = "binary_index")]
    families.push(Family {
        name: "binary_index",
        knob: "rerank_factor",
        values: RERANK_VALUES,
        rebuild: true,
        note: "1 bit/dim after rotation, exact rerank; rerank_factor at build",
        build: build_binary_index,
    });

    #[cfg(feature = "sng")]
    families.push(Family {
        name: "sng",
        knob: "none",
        values: NO_KNOB,
        rebuild: false,
        note: "auto-optimised max_degree; index has no search-time knob",
        build: build_sng,
    });

    #[cfg(feature = "curator")]
    families.push(Family {
        name: "curator",
        knob: "ef_search",
        values: CURATOR_EF_VALUES,
        rebuild: true,
        note: "branching 16, leaf 128, beam 4; ef_search at build; unfiltered search",
        build: build_curator,
    });

    #[cfg(feature = "range_filtered")]
    families.push(Family {
        name: "range_filtered",
        knob: "ef_search",
        values: EF_VALUES,
        rebuild: true,
        note: "HNSW m=16, ef_construction=200; ef_search at build; unfiltered range",
        build: build_range_filtered,
    });

    #[cfg(feature = "kdtree")]
    families.push(Family {
        name: "kdtree",
        knob: "max_leaf_size",
        values: LEAF_VALUES,
        rebuild: true,
        note: "max_leaf_size at build; rejects dim > 50",
        build: build_kdtree,
    });

    #[cfg(feature = "balltree")]
    families.push(Family {
        name: "balltree",
        knob: "max_leaf_size",
        values: LEAF_VALUES,
        rebuild: true,
        note: "max_leaf_size at build; branch-and-bound pruning",
        build: build_balltree,
    });

    #[cfg(feature = "kmeans_tree")]
    families.push(Family {
        name: "kmeans_tree",
        knob: "branch_budget",
        values: BRANCH_VALUES,
        rebuild: false,
        note: "16 clusters, leaf 50; branch_budget per call",
        build: build_kmeans_tree,
    });

    #[cfg(feature = "rptree")]
    families.push(Family {
        name: "rptree",
        knob: "max_leaf_size",
        values: LEAF_VALUES,
        rebuild: true,
        note: "max_leaf_size at build; search visits both children (no pruning)",
        build: build_rptree,
    });

    #[cfg(feature = "rptree")]
    families.push(Family {
        name: "rp_forest",
        knob: "max_leaf_size",
        values: LEAF_VALUES,
        rebuild: true,
        note: "10 trees; max_leaf_size at build",
        build: build_rp_forest,
    });

    families
}

fn to_string_err<E: std::fmt::Display>(err: E) -> String {
    err.to_string()
}

#[cfg(feature = "hnsw")]
fn build_hnsw(vectors: &[Vec<f32>], dim: usize, _knob: usize) -> Result<Box<dyn Searcher>, String> {
    use vicinity::distance::DistanceMetric;
    use vicinity::hnsw::{HNSWIndex, HNSWParams};

    // Layer assignment draws from the RNG. A fixed seed is what makes the
    // recall column comparable between runs; the builder does not expose it.
    let params = HNSWParams {
        m: 16,
        ef_construction: 200,
        metric: DistanceMetric::Cosine,
        seed: Some(42),
        ..Default::default()
    };
    let mut index = HNSWIndex::with_params(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add_slice(i as u32, vector).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, ef| {
        index.search(query, k, ef).map_err(to_string_err)
    })))
}

#[cfg(feature = "nsw")]
fn build_nsw(vectors: &[Vec<f32>], dim: usize, _knob: usize) -> Result<Box<dyn Searcher>, String> {
    use vicinity::nsw::{NSWIndex, NSWParams};

    let mut index = NSWIndex::with_params(dim, NSWParams::default()).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, ef| {
        index.search(query, k, ef).map_err(to_string_err)
    })))
}

#[cfg(feature = "nsg")]
fn build_nsg(vectors: &[Vec<f32>], dim: usize, _knob: usize) -> Result<Box<dyn Searcher>, String> {
    use vicinity::nsg::{NsgIndex, NsgParams};

    let mut index = NsgIndex::new(dim, NsgParams::default()).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, ef| {
        index.search_with_ef(query, k, ef).map_err(to_string_err)
    })))
}

#[cfg(feature = "finger")]
fn build_finger(
    vectors: &[Vec<f32>],
    dim: usize,
    _knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::finger::{FingerIndex, FingerParams};

    let mut index = FingerIndex::new(dim, FingerParams::default()).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, ef| {
        index.search_with_ef(query, k, ef).map_err(to_string_err)
    })))
}

#[cfg(feature = "fresh_graph")]
fn build_fresh_graph(
    vectors: &[Vec<f32>],
    dim: usize,
    _knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::fresh_graph::{FreshGraphIndex, FreshGraphParams};

    let mut index =
        FreshGraphIndex::new(dim, FreshGraphParams::default()).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, ef| {
        index.search_with_ef(query, k, ef).map_err(to_string_err)
    })))
}

#[cfg(feature = "filtered_graph")]
fn build_filtered_graph(
    vectors: &[Vec<f32>],
    dim: usize,
    _knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use std::collections::HashMap;
    use vicinity::filtered_graph::{FilteredGraphIndex, FilteredGraphParams};

    let mut index =
        FilteredGraphIndex::new(dim, FilteredGraphParams::default()).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index
            .add_slice(i as u32, vector, HashMap::new())
            .map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, ef| {
        index.search_with_ef(query, k, ef).map_err(to_string_err)
    })))
}

#[cfg(feature = "vamana")]
fn build_vamana(
    vectors: &[Vec<f32>],
    dim: usize,
    _knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::vamana::{VamanaIndex, VamanaParams};

    let params = VamanaParams {
        seed: Some(42),
        ..VamanaParams::default()
    };
    let mut index = VamanaIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, ef| {
        index.search(query, k, ef).map_err(to_string_err)
    })))
}

#[cfg(feature = "diskann")]
fn build_diskann(
    vectors: &[Vec<f32>],
    dim: usize,
    _knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::diskann::{DiskANNIndex, DiskANNParams};

    let params = DiskANNParams {
        seed: Some(42),
        ..DiskANNParams::default()
    };
    let mut index = DiskANNIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add_slice(i as u32, vector).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, ef| {
        index.search(query, k, ef).map_err(to_string_err)
    })))
}

#[cfg(feature = "emg")]
fn build_emg(vectors: &[Vec<f32>], dim: usize, _knob: usize) -> Result<Box<dyn Searcher>, String> {
    use vicinity::emg::{EmgIndex, EmgParams};

    let mut index = EmgIndex::new(dim, EmgParams::default()).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, ef| {
        index.search_with_ef(query, k, ef).map_err(to_string_err)
    })))
}

#[cfg(feature = "pipnn")]
fn build_pipnn(
    vectors: &[Vec<f32>],
    dim: usize,
    _knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::pipnn::{PipnnIndex, PipnnParams};

    let mut index = PipnnIndex::new(dim, PipnnParams::default()).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, ef| {
        index.search_with_ef(query, k, ef).map_err(to_string_err)
    })))
}

#[cfg(feature = "ivf_pq")]
fn build_ivf_pq(
    vectors: &[Vec<f32>],
    dim: usize,
    _knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::ivf_pq::{IVFPQIndex, IVFPQParams};

    let params = IVFPQParams {
        num_clusters: 100,
        nprobe: 4,
        num_codebooks: 16,
        codebook_size: 256,
        seed: 42,
        ..IVFPQParams::default()
    };
    let mut index = IVFPQIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add_slice(i as u32, vector).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    // `search_reranked` with an exact-rerank pool is the documented IVF-PQ
    // path; plain `search` returns PQ-approximated distances and ranks poorly.
    Ok(Box::new(FnIndex(move |query, k, nprobe| {
        index.set_nprobe(nprobe);
        index.search_reranked(query, k, 100).map_err(to_string_err)
    })))
}

#[cfg(feature = "ivf_avq")]
fn build_ivf_avq(
    vectors: &[Vec<f32>],
    dim: usize,
    _knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::ivf_avq::{IVFAVQIndex, IVFAVQParams};

    let params = IVFAVQParams {
        num_partitions: 100,
        nprobe: 4,
        num_reorder: 100,
        num_codebooks: 16,
        codebook_size: 256,
        seed: 42,
    };
    let mut index = IVFAVQIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add_slice(i as u32, vector).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, nprobe| {
        index.set_nprobe(nprobe);
        index.search(query, k).map_err(to_string_err)
    })))
}

#[cfg(feature = "ivf_rabitq")]
fn build_ivf_rabitq(
    vectors: &[Vec<f32>],
    dim: usize,
    _knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::ivf_rabitq::{IVFRaBitQIndex, IVFRaBitQParams};

    let params = IVFRaBitQParams {
        num_clusters: 100,
        nprobe: 4,
        total_bits: 4,
        seed: 42,
    };
    let mut index = IVFRaBitQIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add_slice(i as u32, vector).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, nprobe| {
        index
            .search_with_ef(query, k, nprobe)
            .map_err(to_string_err)
    })))
}

#[cfg(feature = "lsh")]
fn build_lsh(vectors: &[Vec<f32>], dim: usize, knob: usize) -> Result<Box<dyn Searcher>, String> {
    use vicinity::lsh::{CrossPolytopeLSHIndex, LSHParams};

    let params = LSHParams {
        num_tables: 8,
        num_probes: knob,
        seed: Some(42),
    };
    let mut index = CrossPolytopeLSHIndex::new(dim, params).map_err(to_string_err)?;
    let flat: Vec<f32> = vectors.iter().flatten().copied().collect();
    index.add_vectors(&flat).map_err(to_string_err)?;
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, _| {
        index.search(query, k).map_err(to_string_err)
    })))
}

#[cfg(feature = "rp_quant")]
fn build_rp_quant(
    vectors: &[Vec<f32>],
    dim: usize,
    knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::rp_quant::{RpQuantIndex, RpQuantParams};

    let params = RpQuantParams {
        projected_dim: 64,
        rerank_factor: knob,
        seed: 42,
    };
    let mut index = RpQuantIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add_slice(i as u32, vector).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, _| {
        index.search(query, k).map_err(to_string_err)
    })))
}

#[cfg(feature = "sq4")]
fn build_sq4(vectors: &[Vec<f32>], dim: usize, knob: usize) -> Result<Box<dyn Searcher>, String> {
    use vicinity::sq4::{SQ4Index, SQ4Params};

    let params = SQ4Params {
        rerank_factor: knob,
    };
    let mut index = SQ4Index::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add_slice(i as u32, vector).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, _| {
        index.search(query, k).map_err(to_string_err)
    })))
}

#[cfg(feature = "binary_index")]
fn build_binary_index(
    vectors: &[Vec<f32>],
    dim: usize,
    knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::binary_index::{BinaryFlatIndex, BinaryFlatParams};

    let params = BinaryFlatParams {
        projected_dim: dim,
        rerank_factor: knob,
        seed: 42,
    };
    let mut index = BinaryFlatIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add_slice(i as u32, vector).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, _| {
        index.search(query, k).map_err(to_string_err)
    })))
}

#[cfg(feature = "sng")]
fn build_sng(vectors: &[Vec<f32>], dim: usize, _knob: usize) -> Result<Box<dyn Searcher>, String> {
    use vicinity::sng::{SNGIndex, SNGParams};

    let mut index = SNGIndex::new(dim, SNGParams::default()).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, _| {
        index.search(query, k).map_err(to_string_err)
    })))
}

#[cfg(feature = "curator")]
fn build_curator(
    vectors: &[Vec<f32>],
    dim: usize,
    knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::curator::{CuratorIndex, CuratorParams};

    let params = CuratorParams {
        ef_search: knob,
        ..CuratorParams::default()
    };
    let mut index = CuratorIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index
            .add(i as u32, vector.clone(), Vec::new())
            .map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    // Unfiltered search so the row is comparable with the other families; the
    // label-filtered path is a different workload.
    Ok(Box::new(FnIndex(move |query, k, _| {
        index.search(query, k).map_err(to_string_err)
    })))
}

#[cfg(feature = "range_filtered")]
fn build_range_filtered(
    vectors: &[Vec<f32>],
    dim: usize,
    knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::range_filtered::{RangeFilteredIndex, RangeFilteredParams};

    let params = RangeFilteredParams {
        ef_search: knob,
        ..RangeFilteredParams::default()
    };
    let mut index = RangeFilteredIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index
            .add(i as u32, vector.clone(), i as f64)
            .map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    // `search` is the unfiltered path; the attribute range is metadata only.
    Ok(Box::new(FnIndex(move |query, k, _| {
        index.search(query, k).map_err(to_string_err)
    })))
}

#[cfg(feature = "kdtree")]
fn build_kdtree(
    vectors: &[Vec<f32>],
    dim: usize,
    knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::classic::trees::kdtree::{KDTreeIndex, KDTreeParams};

    let params = KDTreeParams {
        max_leaf_size: knob,
        ..KDTreeParams::default()
    };
    let mut index = KDTreeIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, _| {
        index.search(query, k).map_err(to_string_err)
    })))
}

#[cfg(feature = "balltree")]
fn build_balltree(
    vectors: &[Vec<f32>],
    dim: usize,
    knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::classic::trees::balltree::{BallTreeIndex, BallTreeParams};

    let params = BallTreeParams {
        max_leaf_size: knob,
        ..BallTreeParams::default()
    };
    let mut index = BallTreeIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, _| {
        index.search(query, k).map_err(to_string_err)
    })))
}

#[cfg(feature = "kmeans_tree")]
fn build_kmeans_tree(
    vectors: &[Vec<f32>],
    dim: usize,
    _knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::classic::trees::kmeans_tree::{KMeansTreeIndex, KMeansTreeParams};

    let mut index =
        KMeansTreeIndex::new(dim, KMeansTreeParams::default()).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, budget| {
        index
            .search_with_branch_budget(query, k, budget)
            .map_err(to_string_err)
    })))
}

#[cfg(feature = "rptree")]
fn build_rptree(
    vectors: &[Vec<f32>],
    dim: usize,
    knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::classic::trees::random_projection::{RPTreeIndex, RPTreeParams};

    let params = RPTreeParams {
        max_leaf_size: knob,
        ..RPTreeParams::default()
    };
    let mut index = RPTreeIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, _| {
        index.search(query, k).map_err(to_string_err)
    })))
}

#[cfg(feature = "rptree")]
fn build_rp_forest(
    vectors: &[Vec<f32>],
    dim: usize,
    knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::classic::trees::rp_forest::{RPTreeParams, RpForestIndex, RpForestParams};

    let params = RpForestParams {
        num_trees: 10,
        tree_params: RPTreeParams {
            max_leaf_size: knob,
        },
    };
    let mut index = RpForestIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, _| {
        index.search(query, k).map_err(to_string_err)
    })))
}
