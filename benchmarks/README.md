# Benchmarks: the performance ratchet

"Raise the bar as we go" only means something if a lower bar fails. This
directory holds the measured baseline and the policy that compares new runs
against it.

- `baseline.jsonl` — one committed run of `examples/all_families.rs`, produced
  by `--json`. It is the bar.
- `../scripts/ratchet.py` — runs the harness (or reads a JSONL produced
  elsewhere), compares row by row, and exits non-zero when a gated metric
  regresses past its tolerance.

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
depends on are fixed: `hnsw` builds with `seed = Some(42)` and
`ivf_rabitq`'s k-means uses `params.seed`, so recall is a property of the code
rather than of the entropy pool.

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

- A rows file — one JSON object per measurement (family, dataset, knob, recall,
  p50/p95/p99, qps, build_s, mem_mib, dists_per_query, and capability flags
  `exact`, `supports_update`, `supports_filter`). It is generated, not committed;
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
every phase has a guardrail (`--max-points`, `--budget-seconds`) that refuses
instead of OOMing or running for hours.

```sh
# 100k: affordable anywhere, and the tier where ground truth is cross-checked.
cargo run --release --example large_scale --features "large-scale,hnsw" -- --points 100k

# Ground truth only, so a 100M run does not rescan for every family invocation.
cargo run --release --example large_scale --features "large-scale,hnsw" -- \
  --points 100m --max-points 100m --gt-only

# Resumable generation and cache round-trip, checked without a large run.
cargo run --release --example large_scale --features "large-scale,hnsw" -- --self-test
```

Data lives under `ANN_DATA_ROOT`, else `$REBUILDABLE_ROOT/vicinity-data`, else a
temp directory. The dataset format is self-describing (`b"VICDS01\0"`, `dim: u32`,
`count: u64`, then row-major `f32`), generation resumes by row index, and the
ground-truth cache key is `(domain, count, dim, k, query count, FNV-1a of the
query bytes)` — a key mismatch in the file header is an error, never a silent
reuse. `--verify-gt` cross-checks a sample against both an f32 and an f64 brute
force scan and prints the top-10 overlap; it is on by default.

### The tier ladder

| Tier | Proves | Rough cost |
| --- | --- | --- |
| 10k | correctness and ranking across every family | seconds |
| 100k | the same families still hold at a size where the index dwarfs the corpus | seconds to a minute |
| 1M | build cost and RSS become the interesting numbers | minutes |
| 10M | the 10k ranking is not a small-n artifact; recall loss appears | tens of minutes |
| 100M | memory layout and disk-resident behaviour decide the result | hours |
| 1B | only with an SSD-resident index; not attempted here | days |

What each tier cannot prove: 10k-1M say nothing about a corpus that does not fit
in RAM, and 100M still says nothing about a single-machine index above 1B. A
family that wins at 10k and loses at 100M has not been measured at 10k to a
useful depth.

Families that are only meaningful above 100M — SSD-resident DiskANN, RaBitQ at a
fixed memory budget, streaming/updatable indexes — are **not measured by this
harness yet**. Only HNSW is wired, because it is the family whose build and
search path can consume a mapped, streamed corpus (`add_slice`). Claiming a
large-scale ranking for the others would be a claim about code that cannot run
at that tier.

### Measured, 2026-11-01, `clustered` at 100k

`n = 100_000`, `d = 96`, `k = 10`, 500 sampled queries, HNSW `m = 16`,
`ef_construction = 200`, 16 cores, macOS aarch64. Rows in
[`large-scale-100k.jsonl`](large-scale-100k.jsonl).

| family | knob | recall@10 | 1/ratio@10 | p50 µs | p95 | p99 | QPS | build s | RSS MiB | index MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hnsw | ef=50 | 0.838 | 6.67 | 112.1 | 134.2 | 165.3 | 8848 | 2.48 | 300.7 | 46.3 |
| hnsw | ef=100 | 0.946 | 4.79 | 156.3 | 205.5 | 264.6 | 6345 | 2.48 | 300.7 | 46.3 |
| hnsw | ef=200 | 0.991 | 2.98 | 251.1 | 305.2 | 360.2 | 3995 | 2.48 | 300.7 | 46.3 |

Exact linear-scan p50 at this size was 747.9 µs/query, so `1/ratio` is the speed
of the exact scan over the ANN query: higher is better. Dataset 36.6 MiB, index
46.3 MiB, whole run 3.1 s wall clock after the build was warm.

## Rules

- A family is only listed once it has a measured row in a run.
- The committed artifact is `baseline.jsonl` (the ratchet's bar);
  anything else is generated with `--out` and named where it lands.
- A capability flag is `true` only if a test exercises it, not because the type
  has a method.
- The Pareto front is computed on (recall, p50) at the family's best recall; a
  method may be dominated there and still lead on memory or build cost, which is
  exactly why the axis leaders are printed separately.
