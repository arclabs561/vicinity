# Benchmarks: the performance ratchet

"Raise the bar as we go" only means something if a lower bar fails. This
directory holds the measured baseline and the policy that compares new runs
against it.

- `baseline.jsonl` — one committed run of `examples/all_families.rs`, produced
  by `--json`. It is the bar.
- `hnswlib-parity-100k.jsonl`, `hnswlib-parity-1m.jsonl` — the same row schema,
  emitted by `tools/parity-rows`: `vicinity` HNSW against the vendored C++
  hnswlib (and USearch at 100k) on clustered16, build-only timing. Separate from
  the ratchet baseline; see `tools/parity-rows/README.md` for the protocol.
- `../scripts/ratchet.py` — runs the harness (or reads a JSONL produced
  elsewhere), compares row by row, and exits non-zero when a gated metric
  regresses past its tolerance.

## Preparing data

Every dataset behind a row is produced by `scripts/prepare_data.py`; it records a
sha256 manifest and `verify` recomputes it. See
[`docs/data-preparation.md`](../docs/data-preparation.md).

## Protocol

The harness is the single protocol every family is measured on:

- `n = 10_000` base vectors, `dim = 96`, `k = 10`, `200` held-out queries.
- Seven domains, each standing in for a class of real workload and each
  generated deterministically from a fixed seed (see *Domains* below). The two
  the committed baseline covers are `uniform` and `clustered` (16 centres,
  `sigma = 0.25`, width 96); the others vary in width, so a row's `dataset`
  names the domain rather than assuming one width for all of them.
- Ground truth is an exact top-10 scan per dataset, computed before any index is
  built or timed.
- `p50/p95/p99` are per-query wall clock; `qps` is queries over the whole timed
  loop; five untimed warm-up queries run first.
- `1/ratio@10` is exact-scan p50 divided by the family p50, the time-based
  definition ann-benchmarks uses.
- `dists_per_query` is the count from `vicinity::distance_stats`, enabled with
  `--features distance-stats`. It is `null` when the feature is off and the
  ratchet does not gate it then. `scripts/ratchet.py` always enables it, so the
  baseline and every compared run carry the same counting overhead and their
  timings stay comparable.
- `build_s` and `mem_mib` describe the build that produced the row; `mem_mib`
  is resident-set growth measured with `ps`, and is `null` when unavailable.

## The JSONL row

The first line of a run is a `_meta` object (protocol constants, harness and
host versions, `distance_stats`). Every following line is one row:

| field | column | notes |
| --- | --- | --- |
| `family` | family | index family |
| `dataset` | dataset | domain name, e.g. `uniform` or `clustered` |
| `knob` | knob | swept search parameter |
| `knob_value` | knob | its value for this row |
| `recall_at_10` | recall@10 | overlap with exact top-10 |
| `inverse_ratio` | 1/ratio@10 | exact-scan p50 / family p50 |
| `p50_us` | p50 us | per-query wall clock |
| `p95_us`, `p99_us` | p95, p99 us | per-query wall clock |
| `qps` | QPS | queries / timed-loop seconds |
| `build_s` | build s | build that produced the row |
| `mem_mib` | mem MiB | RSS growth across the build; `null` if unavailable |
| `dists_per_query` | dists/q | `null` without `--features distance-stats` |

The row key is `(family, dataset, knob, knob_value)`.

## What is gated, and by how much

A run fails when any of these move further than the stated tolerance:

| metric | direction | tolerance | noise floor |
| --- | --- | --- | --- |
| `recall@10` | higher is better | drop greater than **0.005 absolute** | — |
| `p50` | lower is better | rise greater than **15%** | +10 µs |
| `p99` | lower is better | rise greater than **25%** | +50 µs |
| `dists_per_query` | lower is better | rise greater than **20%** | +5 |
| `mem_mib` | lower is better | rise greater than **15%** | +2 MiB |
| `build_s` | lower is better | rise greater than **25%** | +0.01 s |

A relative tolerance is only meaningful above the measurement's noise floor: a
4 µs → 8 µs move is a 100% "regression" that is really one scheduling decision
on a 200-query loop, and a 0.0 → 7.5 MiB RSS move is page-reuse granularity. A
row whose absolute change is below the floor is not gated on that metric. A
baseline value of zero cannot support a ratio at all and is reported as
unmeasured instead of failing on any positive value.

Every (family, dataset, knob, knob value) row in the baseline is compared, so a
sweep point cannot quietly get worse while another improves.

One run is not a measurement of a tail. `--runs 3` (or repeated `--current`
files) takes the per-row median before comparing, which keeps a regression that
moves every run and drops a one-run scheduling spike. The invariants the gate
depends on are fixed: `all_families` builds `hnsw` with `SEED`
(`0x4641_4D49_4C59`) and `ivf_rabitq`'s k-means uses `params.seed`, so recall is
a property of the code rather than of the entropy pool.

That sentence used to claim a seed the harness did not pass: `build_hnsw` called
`HNSWIndex::builder(..)` with no seed, and the builder had no setter to pass one
with, so every run built a different graph. Two runs of the same binary over the
same data differed in 23 of 49 recall values, with spreads up to 0.010 — twice
the 0.005 gate, which was therefore racing the RNG it claimed not to depend on.
Both are fixed; the same two runs now differ in 0 of 49. Rows committed before
the fix are one valid draw, not the graph a re-run will build.

Improvements are printed in a table and never fail the run. A row present in the
baseline but missing from the run is a failure by default: a family silently
dropping out is a coverage regression the metric table cannot show. `--family`
and `--allow-subset` downgrade that to a warning for partial runs.

`recall@10` is the only gate that catches a correctness-shaped speedup. The
tolerances are one-sided allowances for measurement noise, not targets: a change
that buys a genuine improvement should re-baseline, not hide behind them.

## Why timings are schedule-only

The `ratchet` job runs on a schedule and on `workflow_dispatch`, never on pull
requests. `cargo test` remains the PR gate.

Timings on shared CI runners move with the CPU generation, the runner image, the
hypervisor neighbours, and frequency scaling. A 15% p50 budget is generous for
one machine and noisy across machines, so gating PRs on it would fail diffs that
changed nothing. Recall and `dists_per_query` are hardware-independent, but a
job that gates them is still a periodic job: it answers "did this week's `main`
get slower", not "is this commit slow".

The committed baseline records its host (`host`, `kernel`, `rustc`, `cores`,
`load1` in the `_meta` line). When the current run's host differs, the ratchet
prints a warning and the CI invocation passes `--advisory-timing`, which demotes
the timing and memory gates to advisories while recall and `dists_per_query`
keep failing the job. Re-recording the baseline on the runner image removes that
caveat.

Load is checked too: if either the baseline or the current run was recorded at
more than `--max-load-ratio` (default: 1.0) load per core, the ratchet says so
and treats the timing and memory gates as advisory for that comparison. A
machine running two other 700%-CPU jobs cannot answer "is this index slower",
and a ratchet that pretends otherwise is a random number generator with an exit
code. Re-record the baseline on an idle host, or pass `--max-load-ratio 0` to
insist.

## Running it

```bash
# Compare the working tree against benchmarks/baseline.jsonl.
python3 scripts/ratchet.py

# The reliable form when timing gates should actually mean something:
# three runs, per-row median, on an otherwise idle machine.
python3 scripts/ratchet.py --runs 3

# One family, while iterating (missing rows are warnings here).
python3 scripts/ratchet.py --family hnsw

# Compare two recorded runs without rebuilding.
python3 scripts/ratchet.py --current /tmp/rows.jsonl

# Produce the rows yourself, for inspection or for an artifact.
cargo run --release --example all_families -- \
  --features hnsw,nsw,nsg,finger,fresh_graph,filtered_graph,vamana,diskann,ivf_pq,ivf_avq,ivf_rabitq,emg,curator,lsh,pipnn,rp_quant,sq4,binary_index,sng,range_filtered,kdtree,balltree,kmeans_tree,rptree,distance-stats \
  -- --json --out /tmp/rows.jsonl
```

Tolerances can be overridden per run (`--recall-tolerance`,
`--p50-tolerance`, `--p99-tolerance`, `--dists-tolerance`,
`--mem-tolerance`, `--build-tolerance`), but the committed values above are the
policy. A looser tolerance in CI is a policy change and belongs in a commit
message, not in a workflow flag.

## Raising the bar

`--update` rewrites `benchmarks/baseline.jsonl` from the run and prints the
deltas it encodes:

```bash
python3 scripts/ratchet.py --update --save-current /tmp/rows.jsonl
```

Then commit the baseline **with the deltas in the commit message**:

```
bench: re-baseline after contiguous-leaf HNSW build

  hnsw / uniform / ef_search=100     p50   91.2 -> 78.4 us  (-14.0%)
  hnsw / uniform / ef_search=100     build  1.24 -> 0.98 s  (-21.0%)
  hnsw / clustered / ef_search=100   recall 0.9980 -> 0.9980 (0.0%)

Ratchet: scripts/ratchet.py --update
```

A baseline update without its delta is not reviewable: the diff of a JSONL file
does not say whether the bar moved because a change was faster, because a
tolerance was loosened, or because the machine changed. State the reason.

## Re-recording the baseline on the CI runner

The `ratchet` job uploads the exact rows it compared as the `ratchet-rows`
artifact (30 days). To move the baseline onto the runner's hardware:

1. Run the `ratchet` workflow from `workflow_dispatch`.
2. Download the `ratchet-rows` artifact from that run.
3. `python3 scripts/ratchet.py --update --current ratchet-rows.jsonl`
4. Commit it with the deltas, as above, and drop `--advisory-timing` from
   `.github/workflows/ratchet.yml` once the host matches.

## What this does not cover

- Absolute recall floors and correctness. Those stay in `cargo test` and the
  `recall-regression` job, which run on every PR.
- Per-query latency distributions beyond p99, and tail behaviour under
  concurrency. The harness is single-threaded by design.
- Families without a k-NN query path (`evoc`, `lemur`, `sparse_mips`); the
  harness lists them under "Not covered" with the reason.

## The benefit profile

The harness also answers a second question: what each method is *for*, rather
than which one wins one table.

- A rows file — one JSON object per measurement (family, storage_mode, dataset,
  knob, recall, p50/p95/p99, qps, build_s, mem_mib, dists_per_query, and
  capability flags `exact`, `supports_update`, `supports_filter`).
  `storage_mode` is `resident` or `file`, so a file-backed row is identified by
  the row rather than by its family name. It is generated, not committed;
  `baseline.jsonl` is the one rows file that is.
- `scripts/benefits.py` — turns those rows into a per-method *benefit profile*:
  the best row per family, the leader on each axis, the Pareto front on
  (recall, latency), and a one-line "wins when" statement.

## Why a benefit profile

A single recall/QPS number cannot represent what a method is for. The tree family
is exact and cheap to build but slow; IVF is cheap to build and needs training;
a graph has the best recall per query but no exactness; the filtered graph is the
only one that composes with predicates. Ranking methods by one axis hides the
reason to keep any of them.

The schema records the dimensions a method can win on:

| Field | Meaning | Axis |
| --- | --- | --- |
| `recall` | recall@k at this operating point | quality |
| `p50_us`, `p95_us`, `p99_us` | latency percentiles | speed, tail |
| `qps` | throughput | speed |
| `build_s` | construction wall clock | cost |
| `mem_mib` | resident growth across build | memory |
| `dists_per_query` | distance computations per query | work |
| `exact` | exact result guarantee | correctness |
| `supports_update` | insert/delete without rebuild | operation |
| `supports_filter` | predicate composes with search | operation |

## Usage

```sh
python3 scripts/benefits.py /tmp/rows.jsonl
```

Output is the table above: per-family rows, axis leaders, Pareto front, and the
"wins when" sentence that should appear in the README for each family.

## Producing rows without external datasets

`cargo run --release --example all_families` measures every family on one
protocol and can emit rows in the shape above. Structure is a variable, not a
constant: seven synthetic domains stand in for seven kinds of real workload, and
each one's hardness is measured at run time rather than implied by its generator.

| Domain | Width | Stands in for |
| --- | ---: | --- |
| `uniform` | 96 | baseline: uniform random points, no cluster structure |
| `clustered` | 96 | 16 centres, `sigma = 0.25`: a few strong clusters |
| `text` | 96 | Zipf-weighted 200-topic mixture, anisotropic per topic: word/document embeddings |
| `image` | 96 | 64 parents x 16 children at a small offset: deep-embedding locality |
| `sparse` | 768 | ~2% nonzero, queries share terms with a document: lexical vectors |
| `neardup` | 96 | 90% jitter around 64 seed documents: near-duplicate-heavy corpora |
| `lowdim` | 8 | points on a random 3-D subspace: intrinsically low-dimensional data |

Two numbers are printed beside each domain because one is not enough:

- **LID**: Levina-Bickel maximum-likelihood local intrinsic dimensionality at
  `k = 10`, median with p10-p90 over 200 sampled points. This is the difficulty of
  the *inside* of a neighbourhood.
- **Separation**: median distance to a random base point divided by the distance
  to the 10th neighbour. Near 1.0 means neighbours are indistinguishable from the
  background; large means a query's cluster is easy to *reach*.

They disagree on purpose: inside a near-duplicate cloud LID stays high (the cloud
is still locally high-dimensional) while separation is enormous, and recall
follows separation. Reporting only LID would call that domain hard and be wrong.

| Environment variable | Effect |
| --- | --- |
| `ANN_DATASETS` | Comma-separated domains to run, e.g. `text,image`; unset runs all seven |
| `ANN_FVECS` | Path to a `.fvecs` file (int32 dim then dim float32 per row, little-endian); replaces the synthetic domains. The first 200 rows are also the queries, so recall@10 has a 0.1 self-match floor |
| `ANN_JSONL` | Path to write JSONL rows in the shape above, ready for `scripts/benefits.py` |

Loading an angular dataset (GloVe, deep-image) reproduces the ann-benchmarks
ranking; rows are L2-normalised, so a Euclidean dataset such as SIFT is ranked by
cosine instead of L2.

## Above 10M: the large-scale tier

`examples/all_families.rs` holds the corpus in memory, computes exact ground
truth for every domain on every run, and measures every family. None of that
survives at 10M+. `examples/large_scale.rs` is a deliberate sibling rather than a
mode of it: the dataset is generated **to disk in bounded chunks** and mapped
(`memmap2`), ground truth covers a **sampled query set** (200-1000 queries) and is
computed once by a **parallel shard scan** (`rayon`, k-way merge) then cached, and
every phase has a guardrail that refuses instead of OOMing or running for hours.
A refusal always names the resource that ran out — the ceiling, time, or the
volume with its free/needed GiB — never just "too big".

```sh
# 100k: affordable anywhere, and the tier where ground truth is cross-checked.
cargo run --release --example large_scale --features "large-scale,hnsw,diskann" -- \
  --points 100k --ef 50,100,200,400,800,1600

# 1M: the tier where uniform recall collapses (see below).
cargo run --release --example large_scale --features "large-scale,hnsw,diskann" -- \
  --points 1m --budget-seconds 7200 --ef 200,1600,3200,4800,6400

# Ground truth only, so a 100M run does not rescan for every family invocation.
cargo run --release --example large_scale --features "large-scale,hnsw,diskann" -- \
  --points 100m --max-points 100m --gt-only

# Resumable generation and cache round-trip, checked without a large run.
cargo run --release --example large_scale --features "large-scale,hnsw,diskann" -- --self-test
```

Data lives under `ANN_DATA_ROOT`, else `$REBUILDABLE_ROOT/vicinity-data`, else a
temp directory. The dataset format is self-describing (`b"VICDS01\0"`, `dim: u32`,
`count: u64`, then row-major `f32`), generation resumes by row index, and the
ground-truth cache is keyed by `(domain, count, dim, k, query count, FNV-1a of
the query bytes)`. A key mismatch in the file header is an error, never a silent
reuse. `--verify-gt` cross-checks a sample against both an f32 and an f64 brute
force scan and prints the top-10 overlap; it is on by default.

### Which families can run here

A family is usable at this tier only if `add_slice` lets it ingest a borrowed row
from the mapping, the persisted form can be searched without the build object,
and the measurement uses the same cached exact ground truth as every other row.
The four candidates were checked against `src/`, not assumed:

| family | streams mapped rows | file-backed search | wired | why not |
| --- | --- | --- | --- | --- |
| `hnsw` | `HNSWIndex::add_slice` | no — `HNSWIndex::search` is in-RAM | yes (`hnsw`) | — |
| `diskann` | `DiskANNIndex::add_slice` + `build_parallel` | yes — `save`, then `DiskANNSearcher` over `graph.index`/`vectors.bin` through a block cache | yes (`diskann`, `diskann-file`) | — |
| `nsg` | `NsgIndex::add_slice` | no | no | in-RAM only. Its one build path (`NsgIndex::build`) runs a pool-64 beam search per node with no parallel variant and keeps the full f32 corpus, so the row repeats HNSW's memory story at 2.41 s per 10k clustered (`baseline.jsonl`) — the 1M build would be ~4 min of single-threaded work for no axis HNSW lacks. |
| `vamana` | no — `VamanaIndex::add(id, Vec<f32>)` only | no | no | cannot take a mapped row without a per-row heap copy, and `DiskANNIndex` is Vamana construction plus persistence, so the family is already covered by the `diskann` rows. |
| `ivf_pq` | `IVFPQIndex::add_slice` | `IVFPQFileSearcher` | no | `add_slice` appends the full f32 vector to `self.vectors` and `build()` trains k-means over that copy, so the build holds a corpus-sized second copy — the one thing this tier must avoid. `compact()` drops it, but then `search_reranked()` is unavailable, and the k-means sample/iteration settings have no tier evidence; the RaBitQ entry below is blocked by the same shape. |

`diskann` is measured twice from **one** build. `diskann` is
`DiskANNIndex::search` against the resident object; `diskann-file` is
`DiskANNSearcher::search` against the files after that object is dropped, reading
through a block cache (`--diskann-cache-mib`, default 64 MiB, `0` disables it).
Both rows share `build_s`, `index mem` and `index disk` and differ only in the
search path, so the storage cost is a column rather than an argument. Their
recall is identical on every row measured here — the check that the saved layout
round-trips — and the run prints that comparison per `ef`.

DiskANN scores squared L2 (`simd::l2_distance_squared`). Every stored row and
query in these domains is L2-normalized, and squared L2 is then a monotone
transform of cosine, so the ranking — and therefore recall against the cosine
ground truth — is the same measurement.

### Columns at this tier

The row schema is unchanged from the 10k protocol; three columns are filled
differently because one run holds more than one index:

- `mem_mib` is the index's own report of its resident structure
  (`HNSWIndex::memory_usage().total()`, `DiskANNIndex::size_bytes()`), not a
  process RSS delta. With two indexes alive in one process an RSS delta cannot be
  attributed to either: an earlier version of this table reported `0.0` for the
  second family because the allocator reused the first family's freed pages.
- `rss_mib` is process RSS **when the timed queries ran**: after the build for the
  in-RAM families, and after the query sweep for `diskann-file` (the kernel
  returns the freed build pages lazily, so the sample right after opening the
  searcher is the build's high-water mark, not what the queries ran in).
- `dists_per_query` is still `null` when `distance-stats` is off: a number is only
  emitted when the counter exists, and `0` would be a false claim that queries
  were free. The counter costs several-fold on build (45.2 s against 5.6 s for
  the same 100k clustered HNSW build in adjacent runs, so read it as an order of
  magnitude), so `dists/q` is measured in a **separate** `distance-stats`
  invocation and its timings are not quoted below.

### The tier ladder

`d = 96`, `f32`, 500 sampled queries, persisted indexes, on a 16-core / 128 GiB
aarch64 workstation. The 2026-11-01 rows were taken on a **shared** host: load
average 12-39 during the quoted runs and up to 130 during an earlier pass. Recall,
`dists/q` and disk bytes are load-independent; p50/QPS are not, and the same
exact-scan computation at 1M measured 8.6 ms, 12.2 ms and 26.8 ms in three runs
an hour apart (a 3.1x spread, not a code change). The 10M-1B rows are extrapolated from the
measured 1M rows and say so.

| Tier | Dataset | HNSW index | DiskANN index | Peak RSS | Wall clock | Proves | Cannot prove |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 10k | 3.7 MiB | 4.4 MiB | 5.0 MiB | ~40 MiB measured | ~17 s measured | correctness, ranking, JSONL schema, resume/cache round-trip, two families and two search paths on one protocol | anything about memory or disk pressure |
| 100k | 36.6 MiB measured | 46.3 MiB measured | 49.6 MiB measured | 375 MiB measured | 32-49 s measured | the index starts to dwarf the corpus; the ef/recall curve is stable; file-backed search costs 1.4-2.1x the resident p50 at the same ef | anything about a corpus that does not fit in RAM |
| 1M | 366 MiB measured | 470 MiB measured | 496 MiB measured | 2.9 GiB measured | 196 s (hnsw) / 282 s (both) measured | build cost and RSS become the interesting numbers; uniform recall collapses; the file path's miss rate becomes visible (81%) | that recall is tier-invariant; it is not (below) |
| 3M | 1.1 GiB measured | 1.4 GiB measured | ~1.5 GiB est. | 6.9 GiB measured | 390 s measured (hnsw) | the collapse continues: 0.971 (100k) -> 0.649 (1M) -> 0.425 (3M) at the same ef=800; recall still crosses 0.95, at ~ef 8000 | DiskANN at 3M: not run (another ~1.5 GiB index and ~7 min of build for a family that is already 0.6 recall behind HNSW on clustered at 1M) |
| 10M | 3.66 GiB measured | 4942 MiB measured | 4959 MiB measured | 14.5 GiB measured | 3763 s measured (both families) | the first tier where both indexes exceed a 64 MiB cache: the file path costs 9.6x at `ef = 3200`, and recall on `uniform` falls to 0.071 at `ef = 200` | behaviour once the index exceeds RAM; 100M still needs sharding |
| 100M | 35.8 GiB measured (written here) | ~45 GiB est. | ~48 GiB est. | ~199 GiB est. | ~4 h est. | only on a host with >199 GiB RAM; the 500-query exact scan itself takes 45.8 s measured | single-machine behaviour above 1B |
| 1B | 358 GiB est. | ~450 GiB est. | ~480 GiB est. | ~2.0 TiB est. | ~1.7 days est., if it fit | only with an out-of-core build (none exists here); ids fit u32 (1B < 2^32) | that any family in this crate can be built at this tier on one machine |

The disk column is not the whole story: the dataset and both indexes need about
**129 GiB at 100M** and **1.26 TB at 1B**. The harness checks that before
writing (and refused, when the volume was at 55 GiB free, with `... 35.76 GiB
more of dataset plus about 93.3 GiB for the hnsw+diskann index files ... 70.2
GiB short`). Free space on this volume moved between 55 and 174 GiB during the
measurement window, and the 35.8 GiB 100M corpus was then written for real and
scanned in 45.8 s, so disk is a host-state blocker, not a structural one.

The budget is checked per query as well as per phase, and rows are written after
every family. A 10M DiskANN run showed why: the builds landed on schedule (HNSW
1731 s against an 1800 s estimate, DiskANN 1792 s against 1626 s), and then the
file-backed sweep spent hours in `pread` because a 4.9 GiB index does not fit a
64 MiB block cache. Query sweeps are deliberately absent from the estimate; they
are bounded by the per-query deadline instead, and a run that stops at the budget
keeps every row it finished.

RAM is structural, and the harness checks it before the time budget because a
longer budget cannot move it. 100M refuses like this on this host:

```text
refusing: memory, not time or disk. The hnsw build peaks at about 199 GiB
(2.09 KiB/point measured at 1M) and this host has 128 GiB, of which the guardrail
allows 109 GiB; the largest tier that fits is about 55M points. Lower --points,
or select diskann only (1.21 KiB/point).
```

`--max-points` (default 20M) and `--budget-seconds` refuse the same way for their
own resource; a 20M run fits RAM but not the default hour, so the time refusal
names the build.

**Recall@10 is not tier-invariant.** At a fixed `ef = 200`, `uniform` falls
0.981 (10k) → 0.705 (100k) → 0.294 (1M) → 0.071 (10M), because i.i.d. uniform points in 96
dimensions have LID ≈ 96 and their neighbour distances concentrate; the exact
top-10 becomes an increasingly arbitrary sample of a large near-tie set.
`clustered` also falls (0.991 at 100k → 0.807 at 1M) because each of the 16
clusters grows from 6.3k to 63k points while the within-cluster noise stays
full-dimensional. Compare tiers at a **fixed recall target** (report the `ef`
required) or compare `dists_per_query`, not raw recall at one `ef`.

The construction batch is capped at `n/8`. The library default of 4096 is right
at 100k+ but would be 40% of a 10k set; that measured 0.924 instead of 0.981
recall at `ef = 200`, so `--batch` is an upper bound, not a fixed size.

### dists/q at 100k, `clustered`

`distance-stats` build, `benchmarks/large-scale-100k-dists.jsonl`. Same build,
same cached ground truth as the timing rows; only the counter is new.

| family | knob | recall@10 | dists/q |
| --- | ---: | ---: | ---: |
| hnsw | ef=50 | 0.838 | 1411 |
| hnsw | ef=100 | 0.946 | 2268 |
| hnsw | ef=200 | 0.991 | 3442 |
| hnsw | ef=800 | 0.998 | 5782 |
| diskann | ef=50 | 0.144 | 1231 |
| diskann | ef=100 | 0.227 | 2039 |
| diskann | ef=200 | 0.378 | 3418 |
| diskann | ef=800 | 0.832 | 8370 |
| diskann-file | ef=50 | 0.144 | 1231 |
| diskann-file | ef=100 | 0.227 | 2039 |
| diskann-file | ef=200 | 0.378 | 3418 |
| diskann-file | ef=800 | 0.832 | 8370 |

Two things this shows that timing cannot. HNSW and DiskANN do nearly the same
number of distance computations at the same `ef` (1411 vs 1231 at `ef = 50`, 3418
vs 3442 at `ef = 200`); what separates them on `clustered` is recall per distance,
not work per query — on `clustered` DiskANN needs `ef = 1600` where HNSW needs
`ef = 200` to pass 0.95. And `diskann` and `diskann-file` count identically at every `ef`, which is
the second check that the file path performs the same search as the resident one.

### Measured, 2026-11-01, `clustered` and `uniform` at 100k

Rows in [`large-scale-100k.jsonl`](large-scale-100k.jsonl) and
[`large-scale-100k-uniform.jsonl`](large-scale-100k-uniform.jsonl); the tables
below are the `ef = 50/200/800/1600` excerpt of the six-point sweep, and the
JSONL carries all 18 rows per domain. At 100k `hnsw` builds in 5.6 s and
`diskann` in 12.2 s (2.2x), both from
the same mapped corpus, with one sampled ground truth for every row. Dataset 36.6 MiB, HNSW 46.3 MiB, DiskANN 49.6 MiB.

`clustered`, exact linear-scan p50 1990.9 µs/query:

| family | knob | recall@10 | 1/ratio@10 | p50 µs | p95 | p99 | QPS | build s | index mem MiB | RSS at query MiB | index MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hnsw | ef=50 | 0.838 | 16.21 | 122.8 | 184.7 | 228.1 | 7829 | 5.62 | 108.9 | 246.3 | 46.3 |
| hnsw | ef=200 | 0.991 | 6.57 | 303.0 | 379.6 | 1022.0 | 2413 | 5.62 | 108.9 | 246.3 | 46.3 |
| hnsw | ef=800 | 0.998 | 3.20 | 622.4 | 738.0 | 800.7 | 1438 | 5.62 | 108.9 | 246.3 | 46.3 |
| hnsw | ef=1600 | 0.998 | 2.40 | 829.3 | 1018.0 | 8231.8 | 940 | 5.62 | 108.9 | 246.3 | 46.3 |
| diskann | ef=50 | 0.144 | 11.14 | 178.7 | 817.3 | 1910.7 | 3339 | 12.22 | 50.7 | 301.3 | 49.6 |
| diskann | ef=200 | 0.378 | 3.71 | 536.8 | 1731.2 | 2897.3 | 1350 | 12.22 | 50.7 | 301.3 | 49.6 |
| diskann | ef=800 | 0.832 | 0.95 | 2087.2 | 4905.0 | 9268.6 | 403 | 12.22 | 50.7 | 301.3 | 49.6 |
| diskann | ef=1600 | 0.982 | 0.69 | 2871.8 | 6591.9 | 10837.0 | 303 | 12.22 | 50.7 | 301.3 | 49.6 |
| diskann-file | ef=50 | 0.144 | 6.35 | 313.4 | 759.3 | 1927.7 | 2589 | 12.22 | 50.7 | 266.5 | 49.6 |
| diskann-file | ef=200 | 0.378 | 2.28 | 873.0 | 3606.0 | 6528.8 | 775 | 12.22 | 50.7 | 266.5 | 49.6 |
| diskann-file | ef=800 | 0.832 | 0.66 | 3023.3 | 7915.3 | 18020.1 | 266 | 12.22 | 50.7 | 266.5 | 49.6 |
| diskann-file | ef=1600 | 0.982 | 0.50 | 3992.5 | 9805.4 | 13236.6 | 214 | 12.22 | 50.7 | 266.5 | 49.6 |

`uniform`, exact linear-scan p50 2261.4 µs/query:

| family | knob | recall@10 | 1/ratio@10 | p50 µs | p95 | p99 | QPS | build s | index mem MiB | RSS at query MiB | index MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hnsw | ef=50 | 0.339 | 14.37 | 157.4 | 208.6 | 234.7 | 6226 | 6.14 | 108.9 | 276.8 | 46.4 |
| hnsw | ef=200 | 0.705 | 2.79 | 811.7 | 1695.2 | 2464.8 | 1030 | 6.14 | 108.9 | 276.8 | 46.4 |
| hnsw | ef=800 | 0.971 | 0.84 | 2678.8 | 4908.5 | 7460.5 | 326 | 6.14 | 108.9 | 276.8 | 46.4 |
| hnsw | ef=1600 | 0.994 | 0.46 | 4932.8 | 10118.3 | 12814.5 | 179 | 6.14 | 108.9 | 276.8 | 46.4 |
| diskann | ef=50 | 0.331 | 9.38 | 241.1 | 308.3 | 386.7 | 3537 | 15.39 | 50.7 | 220.1 | 49.6 |
| diskann | ef=200 | 0.702 | 2.46 | 918.9 | 1188.9 | 5172.4 | 916 | 15.39 | 50.7 | 220.1 | 49.6 |
| diskann | ef=800 | 0.967 | 0.67 | 3391.2 | 4700.5 | 10555.2 | 271 | 15.39 | 50.7 | 220.1 | 49.6 |
| diskann | ef=1600 | 0.996 | 0.38 | 5989.7 | 11848.1 | 23426.6 | 146 | 15.39 | 50.7 | 220.1 | 49.6 |
| diskann-file | ef=50 | 0.331 | 4.38 | 515.8 | 1078.0 | 2538.0 | 1593 | 15.39 | 50.7 | 137.9 | 49.6 |
| diskann-file | ef=200 | 0.702 | 1.31 | 1731.9 | 2160.3 | 2937.8 | 547 | 15.39 | 50.7 | 137.9 | 49.6 |
| diskann-file | ef=800 | 0.967 | 0.39 | 5873.9 | 13823.9 | 22034.4 | 145 | 15.39 | 50.7 | 137.9 | 49.6 |
| diskann-file | ef=1600 | 0.996 | 0.25 | 9164.4 | 15937.2 | 27489.5 | 98 | 15.39 | 50.7 | 137.9 | 49.6 |

The families match on `uniform` (0.705 vs 0.702 at `ef = 200`; 0.971 vs 0.967 at
`ef = 800`) and diverge sharply on `clustered` (at `ef = 200`: 0.991 vs 0.378; at
`ef = 800`: 0.998 vs 0.832) — DiskANN's two-pass Vamana graph at `m = 32`,
`ef_construction = 100` is not finding the cluster-local neighbours that HNSW's
`m = 16`, `ef_construction = 200` finds. That is the family's construction
quality at these defaults, visible in the 10k `baseline.jsonl` rows too
(`diskann` 0.755 vs `hnsw` 1.000 at `ef = 200`, clustered), not an artifact of
the tier wiring.

The file-backed path costs 1.4-2.1x the resident p50 at 100k with the whole
49.6 MiB index inside a 64 MiB block cache (hit rate 99.9-100%), i.e. this is the
cost of the block cache and the graph/vector split, not of the device.

### Measured, 2026-11-01, 1M

Rows in [`large-scale-1m-uniform.jsonl`](large-scale-1m-uniform.jsonl) and
[`large-scale-1m-clustered.jsonl`](large-scale-1m-clustered.jsonl). Dataset
366 MiB, HNSW 470 MiB, DiskANN 496 MiB.

`uniform` (HNSW only), exact linear-scan p50 12169.7 µs/query:

| family | knob | recall@10 | 1/ratio@10 | p50 µs | p95 | p99 | QPS | dists/q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hnsw | ef=200 | 0.294 | 13.20 | 922.1 | 1157.5 | 1322.1 | 1064 | — |
| hnsw | ef=1600 | 0.830 | 2.10 | 5804.6 | 6921.3 | 7264.5 | 171 | — |
| hnsw | ef=3200 | 0.942 | 1.08 | 11274.7 | 12904.4 | 13786.2 | 89 | — |
| hnsw | ef=4800 | 0.973 | 0.69 | 17674.6 | 23284.3 | 47411.1 | 54 | — |
| hnsw | ef=6400 | 0.986 | 0.57 | 21517.3 | 25506.4 | 26723.6 | 46 | — |

`clustered` (both families), exact linear-scan p50 7465.8 µs/query:

| family | knob | recall@10 | 1/ratio@10 | p50 µs | p95 | p99 | QPS | index mem MiB | RSS at query MiB | index MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hnsw | ef=200 | 0.807 | 16.96 | 440.2 | 589.5 | 792.8 | 2163 | 1231.8 | 2089.8 | 469.5 |
| hnsw | ef=800 | 0.969 | 5.29 | 1412.4 | 1993.1 | 3113.8 | 665 | 1231.8 | 2089.8 | 469.5 |
| hnsw | ef=1600 | 0.990 | 3.09 | 2417.3 | 3708.7 | 5858.8 | 381 | 1231.8 | 2089.8 | 469.5 |
| diskann | ef=200 | 0.197 | 7.96 | 937.6 | 1775.0 | 2326.3 | 947 | 507.4 | 1211.5 | 495.9 |
| diskann | ef=800 | 0.568 | 2.05 | 3648.8 | 7628.8 | 9053.1 | 234 | 507.4 | 1211.5 | 495.9 |
| diskann | ef=1600 | 0.777 | 1.02 | 7328.9 | 13103.7 | 16375.1 | 127 | 507.4 | 1211.5 | 495.9 |
| diskann-file | ef=200 | 0.197 | 3.81 | 1957.5 | 9629.9 | 14277.6 | 297 | 507.4 | 891.8 | 495.9 |
| diskann-file | ef=800 | 0.568 | 0.62 | 12017.5 | 30010.0 | 37218.7 | 74 | 507.4 | 891.8 | 495.9 |
| diskann-file | ef=1600 | 0.777 | 0.41 | 18375.2 | 35022.1 | 41109.3 | 51 | 507.4 | 891.8 | 495.9 |

At 1M the 496 MiB DiskANN index no longer fits the 64 MiB block cache: hit rate
drops to 81.1% (5.47M misses over the sweep), and `diskann-file` costs 2.1x the
resident p50 at `ef = 200` and 2.5-3.3x at `ef = 800/1600`. DiskANN's in-RAM
index is also 2.4x smaller than HNSW's at the same `n` (507 vs 1232 MiB) because
`m = 32` on a single layer stores fewer edges than `m = 32` per layer of a
hierarchy; it pays for that with recall on `clustered`.

### Where uniform recall collapses: a tuning gap or a limit?

Measured on `uniform` at 1M in one process, so every row shares one exact-scan
reference (12169.7 µs) and one build. `ef = 200` → 0.294 (the recorded collapse),
`ef = 4800` is the first point at or above 0.95, and it costs **17674.6 µs, 1.45x
the exact scan**: 54 QPS against the brute-force scan's 82. Restoring recall is
possible; restoring it *profitably* is not.

| ef | recall@10 | p50 µs | QPS | vs exact scan | vs ef=200 p50 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 0.294 | 922.1 | 1064 | 13.2x faster | 1.0x |
| 1600 | 0.830 | 5804.6 | 171 | 2.1x faster | 6.3x |
| 3200 | 0.942 | 11274.7 | 89 | 1.08x faster | 12.2x |
| 4800 | 0.973 | 17674.6 | 54 | 1.45x slower | 19.2x |
| 6400 | 0.986 | 21517.3 | 46 | 1.77x slower | 23.3x |

So the answer is not "raise `ef`". At 1M on this domain the graph crosses from
"faster than brute force with unusable recall" to "usable recall and slower than
brute force" between `ef = 3200` and `ef = 4800`, and the crossover is where the
two curves have already met: `ef = 3200` is 1.08x faster than the exact scan at
0.942 recall, and the next step buys 0.031 recall for 1.6x the p50. The honest
reading is that i.i.d. uniform 96-dimensional data at 1M is a domain where a
graph has nothing left to exploit; on `clustered` at 1M the same build reaches
0.969 at `ef = 800` for 1412 µs, 5.3x faster than the exact scan, so this is a
property of the domain at this tier, not of the index.

That also sets the tier comparison rule: equal-`ef` rows are not comparable
across tiers, equal-recall rows are, and the cost axis needs `dists/q` (measured
at 100k above; it is a function of `ef` and the graph, and the 1M rows carry
`null` because a counter build at 1M costs ~10x the build for a number that is
already known at 100k).

The same sweep at 3M (`benchmarks/large-scale-3m-uniform.jsonl`, HNSW only, exact
scan 21729.4 µs) reproduces the shape one tier further out, with the crossover in
the same place:

| ef | recall@10 | p50 µs | QPS | vs exact scan |
| ---: | ---: | ---: | ---: | ---: |
| 800 | 0.425 | 2235.4 | 444 | 9.7x faster |
| 3200 | 0.801 | 8769.8 | 111 | 2.5x faster |
| 6400 | 0.926 | 16374.7 | 60 | 1.33x faster |
| 16384 | 0.986 | 38711.9 | 26 | 1.78x slower |

At a fixed `ef = 800` the same domain reads 0.971 (100k) → 0.649 (1M) → 0.425
(3M), so the collapse is monotone with `n`; and at both 1M and 3M the `ef` that
restores ≥ 0.95 lands where the graph is already slower than the exact scan.
Raising `ef` is a tuning knob that works; it is not a way to keep the graph ahead
of brute force on this domain at this tier.

### What is still not measured, and the exact blocker

Nothing here says "future work". Each row names the API or the resource that
stops it and the number that would clear it.

| capability | blocker | what would unblock it |
| --- | --- | --- |
| SSD-resident DiskANN **under device pressure** | The build is in RAM: `DiskANNIndex::add_slice` appends to a `Vec<f32>` and `build_parallel` keeps the whole adjacency, so there is no out-of-core construction. At 1M the 496 MiB index is larger than the 64 MiB block cache (81% hit rate, 5.47M misses) but still fits in the OS page cache, so most miss traffic is served from memory, not the device. | An index larger than the page cache the host can hold (~100 GiB), i.e. ≈200M points (index ≈97 GiB), which needs ≈530 GiB of RAM to construct and 166 GiB of disk. Neither exists here, and no build path avoids the RAM. |
| RaBitQ at a fixed memory budget | `IVFRaBitQIndex::add_slice` exists, but `build()` keeps the full f32 corpus and `compact()` — which drops it — makes `save_to_dir` fail: quantized edges cannot be reconstructed from outside `qntz`, so a compacted index cannot be persisted or reloaded. The build peak is the corpus, not the codes. | A persisted code format that survives `compact()`, or an out-of-core build. At 10M `d = 96` the codes are 48 B/point = 480 MiB against 3.6 GiB for the f32 build peak — a 7.7x gap that is currently unmeasurable because neither end can be held to a budget. |
| Streaming / updatable indexes | `streaming::IndexOps` is implemented only for `InPlaceIndex`/`MappedInPlaceIndex` (both in-RAM, both `insert(id, Vec<f32>)`), and the protocol has no mutation workload: a static corpus has no insert/delete rate to measure. | A mutation trace or a rate parameter (e.g. 1% upchurn of a 10M corpus = 100k updates) plus an `IndexOps` impl whose insert takes `&[f32]`. |
| 100M | RAM: 2.09 KiB/point measured at 1M gives 199 GiB for the HNSW build against 128 GiB installed, and the harness refuses with that number (ceiling ≈55M points). Disk is not the blocker any more: the 35.8 GiB corpus was generated on this volume and the 500-query exact scan over 100M rows took 45.8 s (1092 M comparisons/s). | >199 GiB of RAM, or a family whose build is out-of-core. Time is ≈4 h at the measured 1.2-1.6e-4 s/point; the 20M default already estimates 6899 s against a 3600 s budget. |
| 1B | Disk: dataset 358 GiB + both indexes ≈934 GiB = 1.26 TB against a 7.3 TB volume with 174 GiB free. RAM: 2.09 KiB/point × 1B = 2.0 TiB for the build. Ids still fit (`u32` holds 1B). | 1.26 TB of fast disk and 2.0+ TiB of RAM, or an out-of-core builder plus a file-backed ground-truth scan. The scan itself is the cheap part: at the measured 1092 M rows/s it is ≈8 min. |

## External libraries, one protocol

Rows in [`hnswlib-parity-100k.jsonl`](hnswlib-parity-100k.jsonl) and
[`hnswlib-parity-1m.jsonl`](hnswlib-parity-1m.jsonl). hnswlib is vendored at
`ca426729609b3221563047ceb269cc1213e0d376` (Apache-2.0) through
`tools/hnswlib-sys`; USearch is the crates.io release driven by
`tools/parity-rows`. Both sides see the same generated corpus, one build each,
`M = 16`, `ef_construction = 200`, `k = 10`, one thread, and an exact top-10 scan
as ground truth. `build_s` is build only, with no serialisation on either side.

`clustered`, 96-d, 100k vectors, 500 queries:

| ef | vicinity recall | vicinity p50 µs | hnswlib recall | hnswlib p50 µs | USearch recall | USearch p50 µs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.450 | 25 | 0.374 | 36 | 0.385 | 31 |
| 20 | 0.615 | 38 | 0.544 | 54 | 0.552 | 50 |
| 50 | 0.836 | 71 | 0.787 | 104 | 0.793 | 106 |
| 100 | 0.953 | 116 | 0.923 | 185 | 0.924 | 182 |
| 200 | 0.993 | 192 | 0.985 | 342 | 0.985 | 316 |
| 400 | 0.999 | 264 | 0.997 | 563 | 0.997 | 544 |

Build: vicinity 24.7 s, hnswlib 29.0 s, USearch 22.8 s.

`clustered`, 96-d, 1M vectors, 200 queries (USearch skipped to stay inside the
30-minute budget):

| ef | vicinity recall | vicinity p50 µs | hnswlib recall | hnswlib p50 µs |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.185 | 39 | 0.157 | 62 |
| 20 | 0.300 | 53 | 0.250 | 98 |
| 50 | 0.480 | 119 | 0.435 | 197 |
| 100 | 0.651 | 250 | 0.608 | 354 |
| 200 | 0.801 | 390 | 0.754 | 682 |
| 400 | 0.914 | 708 | 0.882 | 1270 |

Build: vicinity 554.6 s, hnswlib 735.2 s.

The recall gap is **0.03 to 0.08 absolute in vicinity's favour through most of
both sweeps**, shrinking to 0.0016 at the top of the 100k sweep (`ef = 400`: 0.9988
against 0.9972), where both graphs are near the ceiling. The p50 gap grows
with `ef` (1.4x at `ef = 10`, 2.1x at `ef = 400`). Two caveats keep this from being a clean win:

- **This hnswlib commit has no NEON path.** On Apple Silicon it runs scalar
  while both Rust sides use NEON, so part of the latency gap is vectorisation,
  not graph quality. Recall is unaffected by that.
- **One run per row on one host.** The 100k tier showed ~10% p50 drift between
  runs, which is why the p50 column is reported and not QPS. Recall is stable to
  3 decimal places across repeats.

`mem_mib` and `dists_per_query` are null on both files. Sequential builds in one
process make RSS deltas meaningless (`hnswlib` measured 0.0 MiB on a probe run),
and vicinity's distance counter would add atomics to one side only.

## Rules

- A family is only listed once it has a measured row in a run.
- The committed artifact is `baseline.jsonl` (the ratchet's bar);
  anything else is generated with `--out` and named where it lands.
- A capability flag is `true` only if a test exercises it, not because the type
  has a method.
- The Pareto front is computed on (recall, p50) at the family's best recall; a
  method may be dominated there and still lead on memory or build cost, which is
  exactly why the axis leaders are printed separately.
