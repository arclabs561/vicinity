# Benchmark atlas

Status: current, 2026-11-01. Every figure is generated from a
committed rows file by [`scripts/plot_atlas.py`](../scripts/plot_atlas.py);
none is hand-drawn, and each carries its source and protocol in the image.

The atlas exists because four different questions hide behind "which ANN is
fastest": *at what recall*, *at what cost*, *on what data*, and *with what
storage*. A single recall/QPS curve answers the first and silently assumes the
other three.

What the atlas does *not* do is say what the field has that these families do
not. That is [research-frontier.md](research-frontier.md): the surveys this
protocol should be measured against, the methods absent here, and the five 2026
developments most likely to reorder a table like this one.

## Protocol

| | |
| --- | --- |
| Queries | 200 (10k synthetic), 500 sampled (large-scale), 10k held-out (GloVe) |
| Ground truth | exact top-10 scan per dataset, computed before any index is built |
| Metric | recall@10; `1/ratio@10` = exact-scan p50 ÷ family p50 |
| Latency | per-query wall clock, p50/p95/p99; QPS over the whole timed loop |
| Work | `dists_per_query` from `vicinity::distance_stats` (separate build) |
| Cost | `build_s`, resident index bytes, on-disk index bytes, `persist_s` |
| Threads | one; every number here is single-query, single-thread |

Rows are committed under [`benchmarks/`](../benchmarks/) and
[`docs/`](.). The 10k synthetic matrix is `benchmarks/baseline.jsonl`; the
large-scale tiers are `benchmarks/large-scale-*.jsonl`; the per-dataset runs are
`docs/*.jsonl`.

A **tier** is one rung of the corpus-size ladder (10k, 100k, 1M, 3M measured;
10M, 100M, 1B estimated), and it is a controlled comparison rather than a bigger
version of the rung below it. Within a tier, the generator, `d=96`, the metric,
the query count and the exact ground truth are fixed; only `n` moves. Between
tiers everything is held constant except `n` as well, which is the point: a
difference between the 100k and 1M curves is attributable to scale and nothing
else. It also means a tier result is not transportable. `ef` is an absolute
beam width, not a fraction of `n`, so the same `ef=200` that scores 0.981 at 10k
scores 0.294 at 1M on uniform data; a plot that joins tiers without saying so
invites exactly that misreading. Every figure here names its tier in the caption
and plots each tier as its own series.

## Compared with ann-benchmarks

[ANN-Benchmarks](https://ann-benchmarks.com/) is the reference this work is
measured against, and its principles are worth stating because they constrain
what a comparable plot may claim: one saturated CPU and single queries by
default, pre-split train/test with 100-neighbour ground truth, hyperparameters
swept and non-frontier points dropped, datasets that fit in RAM, and index builds
kept below "several hours". It is now unmaintained and points to
[VIBE](https://github.com/erikbern/vibe); billion-scale work belongs to
[big-ann-benchmarks](https://big-ann-benchmarks.com/) instead.

| | ann-benchmarks | this atlas |
| --- | --- | --- |
| Axis | recall vs QPS, log x | recall vs QPS, plus latency tails, build, memory, disk, dists/q |
| Frontier | hyperparameter sweep, dominated points dropped | same, per family, from committed rows |
| Datasets | 15 fixed HDF5 sets, 25–960 d, up to 1M | those plus seven synthetic domains and tiers to 3M |
| Ground truth | precomputed 100-NN, `k = 10` plots | exact per run, verified against f32 *and* f64 |
| Storage | in-memory only by construction | storage mode is a column and a figure |
| Regression gate | none | ratchet with per-metric tolerances (`benchmarks/README.md`) |
| Threading | single CPU enforced | single-thread, stated, not enforced |

The important difference is the last two rows. ann-benchmarks measures a
library's in-memory speed; it cannot tell you what happens when the index no
longer fits, and it has no notion of a result that got slower since last week.
Both are questions with answers here.

## Figures

### 1–2. Every family at 10k

[uniform](plots/atlas/pareto-uniform-10k.svg) ·
[clustered](plots/atlas/pareto-clustered-10k.svg)

Twenty-four families on one 10k, 96-d dataset, one line per family along its own
frontier. The spread is the point: at recall 0.95 on `clustered`, `hnsw` and
`vamana` sit near 10⁴ QPS, the exact trees (`balltree`, `kdtree`) fall below 10³,
and the quantised families trade 5–15 points of recall for their speed. On
`uniform` the same families compress into a much narrower band. The domain decides how much room there is, not the algorithm.

Read the shapes, not the ordering: a family whose frontier is flat is saturated
(more `ef` buys nothing), and one whose frontier is vertical is latency-bound by
per-query overhead rather than search work.

### 3. Recall against the knob

[recall-vs-ef-10k.svg](plots/atlas/recall-vs-ef-10k.svg)

The same rows with `ef_search` on x. Three behaviours are visible: monotone
families that keep climbing (`hnsw`, `vamana`, `nsg`), families that plateau
early (the quantised ones, where reranking caps recall), and families that are
exactly 1.0 at every `ef` because they are exact (`balltree`, `kdtree`, and
`brute`, which is the reference and not a competitor).

A plateau is not a failure; it is the family telling you its recall ceiling at
that code size. Sweeping `ef` past it measures nothing.

### 4. The cost axes

[cost-axes-10k.svg](plots/atlas/cost-axes-10k.svg)

Build seconds and resident bytes against the best recall each family reaches on
`clustered`. This is the figure that kills one-number rankings: `balltree`
reaches recall 1.000 with the smallest resident index and a sub-second build, and
is still the slowest at query time; `hnsw` needs ~2 s and ~20 MiB to reach 1.000.

### 5. Work per query

[dists-vs-recall-100k.svg](plots/atlas/dists-vs-recall-100k.svg)

Distance computations per query at 100k, from a `distance-stats` build. HNSW and
DiskANN do nearly the same work at the same `ef` (1411 vs 1231 at `ef = 50`;
3418 vs 3442 at `ef = 200`) while reaching different recall. What separates them is recall *per distance*, not distance work. `diskann` and `diskann-file`
count identically at every point, which is the check that the file path performs
the same search as the resident one.

### 6. Tiers are not comparable at equal `ef`

[tiers-recall-vs-ef-uniform.svg](plots/atlas/tiers-recall-vs-ef-uniform.svg)

Uniform 96-d at 100k, 1M and 3M on one axis. At a fixed `ef = 200` recall falls
0.981 → 0.705 → 0.294 across 10k → 100k → 1M; at 3M the same `ef = 800` point
that scored 0.971 at 100k scores 0.425. i.i.d. uniform data in 96 dimensions has
LID ≈ 96 and neighbour distances concentrate, so the exact top-10 becomes an
arbitrary sample of a large near-tie set. Compare tiers at a fixed recall target,
never at a fixed knob.

### 7. Cost grows superlinearly; recall does not

[tier-memory-and-build.svg](plots/atlas/tier-memory-and-build.svg)

Index memory and build time against `n` for HNSW and DiskANN. Both are log-log
and close to linear in `n`: 470 MiB and 109 s at 1M for HNSW, 1.43 GiB and 353 s
at 3M. That is why the 100M tier is a RAM question rather than a time question.
At 1M, DiskANN is 2.4× smaller resident than HNSW (507 vs 1232 MiB) and pays for
it in recall on clustered data.

### 8. Storage is a measured axis

[storage-resident-vs-file.svg](plots/atlas/storage-resident-vs-file.svg)

Left: the same DiskANN index queried resident and from its files at 100k and 1M.
At 100k the 49.6 MiB index fits the 64 MiB block cache (99.9–100% hit) and the
file path costs 1.4–2.1× the resident p50. That is the cache's cost, not the device's. At 1M the 496 MiB index no longer fits (81.1% hit, 5.47 M misses) and
the file path costs 2.1× at `ef = 200`, 2.5–3.3× at `ef = 800/1600`.
Right: a fixed-recall comparison from `docs/diskann-storage-latency.csv`, three
runs per storage mode.

### 9. Real datasets: GloVe-25, SIFT-128, GIST-960

[GloVe-25](plots/atlas/real-glove-25-angular.svg) ·
[SIFT-128](plots/atlas/real-sift-128-euclidean.svg) ·
[GIST-960](plots/atlas/real-gist-960-euclidean.svg)

The fixed public sets, where the ranking is the one ann-benchmarks reports and
the synthetic domains are not in play.

GloVe-25 (1.18M × 25): `hnsw-m16` holds recall 0.95 at 18.1k QPS, and even the
exact `kdtree` holds it at 50 QPS. That is 360× slower, and exact, which at 25
dimensions is a real option. SIFT-128: `hnsw-m16` at 9.5k QPS for 0.95, `sq8u`
at 6.1k, `vamana` at 5.2k. GIST-960 is the opposite regime: the best row is
`hnsw-m16` at 0.977 recall and 330 QPS, and two families never get above 0.65
recall at any speed. Width is the variable that decides whether the exact option
is viable.

### 10. Seed variance

[seed-variance.svg](plots/atlas/seed-variance.svg)

HNSW's level assignment is randomised; `docs/hnsw-seed-recall.csv` records three
seeds across four `ef` values. The spread is small (0.003–0.010 recall) and is
plotted rather than assumed away: a 0.005 regression gate smaller than the seed
spread would be measuring the RNG. The gate is set at 0.005 because that is the
spread on the same fixture.

Since then the harness seeds its build, so a published row is a fixed graph
rather than one draw: two runs of the same binary went from differing in 23 of
49 recall values to differing in none. This figure stays the record of how far a
*different* seed would move the same index, which is what a reader comparing
against an independent run needs to know.

### 11. What is actually measured

[metric-coverage-10k.svg](plots/atlas/metric-coverage-10k.svg)

Rows carrying each metric, per family. The gaps are honest: `dists_per_query`
exists only where a `distance-stats` build has been run, and the columns that are
empty across a whole row are families whose row set predates a field. An empty
cell is "not measured", never "zero".

### 12. The layout behind every memory number

[layout-upper-layers.svg](plots/atlas/layout-upper-layers.svg)

The same 1M HNSW index built twice: default layout, and with upper layers frozen
into eligible-node CSR. Recall is identical at `ef` 50/200/800, p50 is within 1%,
and the index is **1231.8 MiB against 549.8 MiB** (−55%). The memory was never
data: every layer allocated a 136-byte inline neighbour slot for every node, and
only a few percent of nodes live above layer 0.

Two things this figure is here to prevent. A memory claim measured by process
RSS would have shown *no change* (2111 MiB either way), because the freed
allocations return to the allocator rather than the OS. And a claim that the
resident size is inherent to HNSW is false: the serialized form of the same graph
is 470 MiB, which is what the compact layout approaches.
### 13. External libraries, with repeats

[external-repeats-fashion20k.svg](plots/atlas/external-repeats-fashion20k.svg)

The only result set here with external libraries and repeats on the same corpus
and query set: Fashion-MNIST at 20k x 784, k = 10, three runs per algorithm.
`hnsw` recall 0.984 ± 0.020 at ~17.0k QPS, `usearch` 0.986 ± 0.018 at ~6.4k,
`hnsw_rs` 0.969 ± 0.022 at ~4.3k, `brute` 1.000 at 433. Index footprint 72, 128
and 68 MiB.

Two cautions are visible in the error bars: repeated QPS on this host spreads by
±45%, so the *ordering* on speed is indicative while the recall ordering is
stable, and the heap figures are library-reported estimates rather than measured
allocations. The right panel pairs each algorithm's heap estimate with its
serialized size, which is the only place in the repo where both exist for
external code.

### 14. One family across two datasets

[single-family-datasets.svg](plots/atlas/single-family-datasets.svg)

SQ8U is the only family measured on GloVe-100 and Fashion-MNIST, so this is a
width comparison rather than an algorithm comparison: the same 4-bit traversal
reaches 0.922 at 874 QPS on 1.18M x 100, and 1.000 at 1172 QPS on 60k x 784. The
larger, narrower corpus is the harder one even at a tenth of the points, which is
the recall-versus-LID effect the synthetic domains also show.

### 15. hnswlib and USearch, same bytes at 100k and 1M

[competitor-parity-hnswlib.svg](plots/atlas/competitor-parity-hnswlib.svg)

The first row set where every column is the same protocol: same generated
clustered 96-d corpus, same `M = 16`, `ef_construction = 200`, `k = 10`, one
thread, exact top-10 ground truth, and `build_s` without serialisation on either
side. Rows in
[`hnswlib-parity-100k.jsonl`](../benchmarks/hnswlib-parity-100k.jsonl) and
[`hnswlib-parity-1m.jsonl`](../benchmarks/hnswlib-parity-1m.jsonl).

vicinity leads at every `ef` on both tiers: recall 0.450 vs 0.374 at `ef = 10`
and 0.999 vs 0.997 at `ef = 400` at 100k, with p50 25 vs 36 µs and 264 vs 563 µs;
at 1M, 0.185 vs 0.157 through 0.914 vs 0.882, with p50 39 vs 62 µs through 708 vs
1270 µs. Build: 24.7 vs 29.0 s at 100k, 554.6 vs 735.2 s at 1M. USearch sits
between them on recall at 100k (0.385 to 0.997) with its own 22.8 s build.

The recall gap is 0.03 to 0.08 absolute; the p50 gap is not purely graph quality,
because the pinned hnswlib commit has no NEON path and runs scalar on Apple
Silicon while both Rust sides use NEON. Read the figure for shape, not for a
speed claim, and read the rows for the numbers.
### 16. In-place construction versus HNSW

Rows in
[`inplace-construction-2k-20k.jsonl`](../benchmarks/inplace-construction-2k-20k.jsonl),
generated by `examples/inplace_construction.rs`. Both families see the same
corpus, the same exact top-10 truth and the same search budget; `InPlaceIndex`
uses its own defaults (`max_degree = 32`, construction beam 128) and HNSW uses
`m = 16`, `ef_construction = 200`.

| tier | family | search beam | recall@10 | p50 µs | build µs/insert |
| --- | --- | ---: | ---: | ---: | ---: |
| 2k x 16 | inplace | 64 | 1.000 | 49 | 67 |
| 2k x 16 | hnsw | 64 | 1.000 | 17 | 90 |
| 2k x 96 | inplace | 64 | 1.000 | 70 | 90 |
| 2k x 96 | hnsw | 64 | 1.000 | 27 | 149 |
| 20k x 16 | inplace | 64 | 1.000 | 63 | 116 |
| 20k x 16 | hnsw | 64 | 1.000 | 22 | 118 |
| 20k x 32 | inplace | 64 | 1.000 | 69 | 142 |
| 20k x 32 | hnsw | 64 | 0.997 | 24 | 146 |
| 20k x 96 | inplace | 64 | 0.913 | 100 | 173 |
| 20k x 96 | inplace | 128 | 0.963 | 160 | 173 |
| 20k x 96 | hnsw | 64 | 0.943 | 39 | 239 |

The in-place builder is competitive on build cost and recall through 32
dimensions. At 96 dimensions the build is 27% cheaper. Recall trails at a
matched search budget: 0.913 against 0.943. Its own default beam of 128 reaches
0.963, but a query then costs 160 µs against HNSW's 39 µs. The lead is bought,
not found. This family's case is build cost and mutability, not query latency.

## Nuances we have paid for

1. **Equal `ef` is not equal recall.** Only equal-recall rows compare across
   tiers, families or builds.
2. **Recall can be recovered at a loss.** At 1M uniform, `ef = 4800` reaches 0.973
   but costs 1.45× the exact scan it replaces; the graph crosses from "fast and
   unusable" to "usable and slower than brute force" between `ef = 3200` and
   `ef = 4800`.
3. **`1/ratio` is time-based.** The denominator is the exact-scan p50, measured in
   the same process, so a machine that is slow at scanning appears to make every
   index look better.
4. **QPS over a loop is load-sensitive; p50 is not.** The same exact scan measured
   8.6 ms, 12.2 ms and 26.8 ms in three runs an hour apart on a shared host. Quote
   latencies with the host, and never compare QPS across machines.
5. **Distance counts need their own run.** The counter costs several-fold on build
   (45.2 s against 5.6 s for the same 100k HNSW build), so `dists/q` timings are
   not comparable with timing-row timings.
6. **Build peak ≠ steady state.** A permutation step used to copy the corpus
   (+0.5 KiB/point at 1M) and upper HNSW layers allocate fixed-width slots for
   every node (+~0.7 KiB/point). The number in a row is a peak.
7. **RSS cannot be attributed with two indexes alive.** Per-family `mem_mib` is
   the index's own accounting; `rss_mib` is the process, and with two families in
   one process the allocator reuses freed pages (an earlier table reported 0.0 for
   the second family).
8. **Serialized size ≠ resident size.** HNSW is 470 MiB on disk and 1232 MiB in
   RAM at 1M. Comparing the two columns is comparing a format with a layout.
9. **Cache state is part of the result.** Warm-after-build and cold differ by
   more than most algorithm choices; every row records `cache_state`.
10. **Query depth must match.** Recall@10 and recall@100 are different
    measurements; a ten-result search reports `recall_100: null`, not a copy.
11. **Filtered search needs a selectivity sweep.** One selectivity is not a
    filtered benchmark: post-filtering is perfect at 20% and 0.158 at 0.1%,
    while the inverted-list path stays at 1.000. See `tests/filtered_recall.rs`.
12. **Exactness changes the ranking.** Trees are exact and dominated on latency;
    they lead on memory and build cost, which is why the atlas keeps a cost
    figure rather than a single leaderboard.
13. **Storage mode changes the answer more than most algorithm choices.** The
    same DiskANN index is 1.4–3.3× slower from disk depending on whether it fits
    the block cache.
14. **A driver's warm-up is part of the protocol.** usearch's
    `change_expansion_search` is free after the first search on a fresh index;
    the first search costs ~3×. Any comparison that starts cold measures the
    library's init path, not its search.
15. **Ground truth must be verified, not trusted.** The large-scale harness
    cross-checks its cached truth against f32 and f64 brute force and prints the
    top-10 overlap; a truth file whose key does not match is an error, never a
    silent reuse.
16. **A build-time beam that stops on a node count is not a beam.** Stopping the
    insert search after `beam_width * 2` visited nodes spent the budget on
    whichever heap branch popped first. With a neighbour cap that also dropped
    better edges once full, `InPlaceIndex` built at 0.572 recall against an exact
    scan (n=2000, d=32, `max_degree` 16) while HNSW on the same corpus scored
    1.000. Distance-bounded stop plus worst-edge replacement: 0.988. The insert
    path has its own recall test now, because a test that only checks for a
    non-empty result is not a construction test.

## What has no plot, and why

- **Filtered selectivity curves.** Measured (`tests/filtered_recall.rs`, three
  strategies × six selectivities × two domains) but not persisted as rows, so
  there is nothing to plot from. The next step is a JSONL emitter for that test.
- **Churn and update workloads.** The registry has `*_churn` selections, but the
  unified harness has no mutation workload, so no row carries an update rate.
- **`dists_per_query` above 100k.** A counter build at 1M costs ~10× the normal
  build, so it is measured at 100k where the curve has the same shape.
- **Families with no k-NN path.** `evoc` (clustering) and `lemur` (inference
  scaffold) are listed in the harness's "not covered" table with the reason.
- **Energy per query.** Not measured at all, and the 2026 survey puts it in the
  first tier at 10M+.

## Reproducing

```sh
# 10k matrix, 24 families, two domains
cargo run --release --example all_families --features "hnsw,nsw,nsg,finger,fresh_graph,filtered_graph,vamana,diskann,ivf_pq,ivf_avq,ivf_rabitq,emg,curator,lsh,pipnn,rp_quant,sq4,binary_index,sng,range_filtered,kdtree,balltree,kmeans_tree,rptree" -- --json --out /tmp/rows.jsonl

# tiers
cargo run --release --example large_scale --features "large-scale,hnsw,diskann" -- --points 1m --budget-seconds 7200

# figures
uv run --with matplotlib --with numpy python scripts/plot_atlas.py
```

The 10k matrix takes 338 s measured on 16 cores for two domains and 24 families;
the 1M tier takes 196 s (HNSW, uniform) to 282 s (HNSW + DiskANN, clustered). `benchmarks/README.md` documents the ratchet that turns these rows
into a regression gate.
