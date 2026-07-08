# Benchmark Results

Historical benchmark tables for standard ANN datasets.

The checked-in `docs/*.jsonl` files are legacy result artifacts from this older
schema. They do not include `_meta`, `storage_mode`, `cache_state`,
`load_time_s`, or `index_bytes`, so use them only as historical in-memory
context. Storage-aware coverage should be generated with the current harness
commands below.

Current benchmark runs should use `examples/ann_benchmark.rs`, which writes one
`_meta` JSON line with the dataset, metric, actual `rustc --version`, MSRV,
crate version, compiled feature list, train/query limits, and measured query
count, followed by one JSON line per measurement with build time, RSS, storage
mode, cache state, and p50/p95/p99 latency. Persisted, file, mmap, and
segmented-store rows also report `load_time_s` and `index_bytes` when the
runner can measure them. Use
`--resume` to skip completed rows and `--fresh` to recreate the result file.
Resume checks require explicit `storage_mode` on current in-memory rows and on
snapshot/file/mmap/segmented rows, so legacy rows without storage context do not
silently satisfy current requests.

QPS below is sequential single-query throughput (queries / wall-clock seconds)
from older single-run `--release` measurements on Apple Silicon. The exact CPU
model, thread count, frequency, and thermal state were not recorded for these
older rows, so treat absolute QPS as historical context. Re-run the harness on
your target machine before comparing against external papers.

Dataset difficulty should be reported alongside recall/QPS when the comparison
crosses datasets or query subsets. Candidate metadata includes local intrinsic
dimensionality (LID) for query difficulty
([Aumuller and Ceccarello, 2019](https://arxiv.org/abs/1907.07387)), relative
contrast or nearest-neighbor margin for separability
([He, Kumar, and Chang, 2012](https://arxiv.org/abs/1206.6411)), norm
distribution, duplicate rate, hubness, cluster or posting-list imbalance, and
whether the query set is in-distribution or OOD. Use
`scripts/profile_ann_dataset.py` for the first sampled profile pass over
converted `VEC1`/`NBR1` datasets, including sampled coarse-partition imbalance.
It also reports optional `query_splits` for generated drift, filtered, topic,
and difficulty-label files when those sidecars exist. These fields are dataset
metadata, not algorithm result rows.

Recommended current commands:

```bash
# Full single-query curve with latency tails.
cargo run --example ann_benchmark --release --features hnsw,ivf_pq,ivf_avq -- \
  data/ann-benchmarks/glove-25-angular \
  --algo hnsw --algo ivfpq --algo ivf_avq --json --fresh

# Bounded probe for fast iteration. The query limit is recorded in `_meta`, and
# `--resume` will not mix these rows with full-query runs.
cargo run --example ann_benchmark --release --features hnsw -- \
  data/ann-benchmarks/glove-25-angular --algo hnsw \
  --ef-search 10,20,30,40,50 --max-queries 1000 --json --fresh

# Bounded corpus probe for algorithms whose full-index build is expensive.
# Both the indexed-vector cap and the query cap are recorded in `_meta`, and
# recall is recomputed against the capped corpus before measurement.
cargo run --example ann_benchmark --release --features kdtree,balltree,rptree,kmeans_tree -- \
  data/ann-benchmarks/glove-25-angular \
  --algo balltree --algo rp_forest --algo kmeans_tree \
  --max-train 5000 --max-queries 200 --snapshot-load --json --fresh

# HNSW single-query plus parallel-query throughput.
RAYON_NUM_THREADS=4 cargo run --example ann_benchmark --release --features hnsw,parallel -- \
  data/ann-benchmarks/glove-25-angular --algo hnsw --batch --json --fresh

# HNSW build, save, reload, and post-load search rows.
cargo run --example ann_benchmark --release --features hnsw,serde -- \
  data/ann-benchmarks/glove-25-angular --algo hnsw --snapshot-load --json --fresh

# IVF-PQ with exact reranking pools.
# Add `--snapshot-load` when validating save/load equivalence and rerank
# persistence; it doubles the emitted IVF-PQ rows.
cargo run --example ann_benchmark --release --features ivf_pq,hnsw -- \
  data/ann-benchmarks/glove-25-angular --algo ivfpq \
  --pq-clusters 1024 --pq-codebooks 25 --pq-codebook-size 256 \
  --pq-training-sample-size 100000 --pq-kmeans-max-iter 20 \
  --pq-nprobes 16,32,64,128,256 --pq-rerank-pools 500,5000,20000 --json --fresh

# DiskANN in-memory graph search plus file and mmap search from the same build.
cargo run --example ann_benchmark --release --features hnsw,diskann -- \
  data/ann-benchmarks/glove-25-angular --algo diskann --json --fresh

# Classic tree baselines. These are comparison rows, not optimization targets.
# `--snapshot-load` adds persisted-and-reopened rows; search is still in-memory
# after load, unlike DiskANN's file and mmap rows.
cargo run --example ann_benchmark --release --features kdtree,balltree,rptree,kmeans_tree -- \
  data/ann-benchmarks/glove-25-angular \
  --algo kdtree --algo balltree --algo rptree --algo rp_forest --algo kmeans_tree \
  --tree-leaf-sizes 10,50 --rp-num-trees 10,20 --kmeans-clusters 8,16 \
  --snapshot-load --json --fresh

# FreshGraph delete/insert churn, scored against a live active-set oracle.
cargo run --example ann_benchmark --release --features hnsw,fresh_graph -- \
  data/ann-benchmarks/glove-25-angular --algo fresh_graph_churn --json --fresh

# IP-DiskANN-style in-place graph search and delete/insert churn.
cargo run --example ann_benchmark --release --features hnsw -- \
  data/ann-benchmarks/glove-25-angular \
  --algo inplace --algo inplace_churn --json --fresh

# LSM-tiered streaming churn, scored against a live active-set oracle.
cargo run --example ann_benchmark --release --features hnsw -- \
  data/ann-benchmarks/glove-25-angular --algo lsm_churn --json --fresh

# Filtered-search selectivity sweep. This is synthetic and writes its own JSONL.
cargo run --example acorn_selectivity --release \
  --features hnsw,filtered_graph,range_filtered,curator -- --json --fresh
uv run scripts/summarize_selectivity_results.py \
  data/ann-benchmarks/results/acorn-selectivity-*.jsonl

# Broad dense-dataset coverage sweep. This emits recall/QPS/latency rows for
# the implemented single-vector search families that can consume dense vectors.
FEATURES=hnsw,nsw,vamana,diskann,ivf_pq,ivf_avq,ivf_rabitq,emg,nsg,pipnn,sng
FEATURES=$FEATURES,finger,fresh_graph,filtered_graph,curator,range_filtered
FEATURES=$FEATURES,rp_quant,binary_index,sq4,sq8,lsh,rptree,kdtree,balltree,kmeans_tree
cargo run --example ann_benchmark --release \
  --features "$FEATURES" \
  data/ann-benchmarks/glove-25-angular \
  --algo hnsw --algo nsw --algo vamana --algo diskann \
  --algo ivfpq --algo ivf_avq --algo ivf_rabitq \
  --algo emg --algo nsg --algo dual_branch --algo deg --algo pipnn --algo sng --algo finger \
  --algo fresh_graph --algo filtered_graph --algo curator --algo range_filtered \
  --algo rp_quant --algo binary_index --algo sq4 --algo sq4u --algo sq8u \
  --algo symphony_qg --algo symphony_qg_vr --algo adsampling --algo lsh --algo hnsw_prt \
  --algo brute --algo kdtree --algo balltree --algo rptree --algo rp_forest --algo kmeans_tree \
  --pq-training-sample-size 100000 --pq-kmeans-max-iter 20 --json --fresh

# Dataset shape and difficulty profile. This uses memmaps and sampled exact
# distances, including neighbor gaps, hubness, LID, coordinate dispersion, and
# coarse partition imbalance.
uv run scripts/profile_ann_dataset.py data/ann-benchmarks/glove-25-angular \
  --sample-train 4096 --sample-queries 1000 --pair-samples 20000

uv run scripts/summarize_dataset_profiles.py /tmp/vicinity-dataset-profiles/*.json
```

These commands intentionally separate low-recall, high-throughput operating
points from high-recall runs. External comparisons should use a recall/QPS
curve, not only the `Recall@10 = 100%` row.

## Current Validation Notes

On 2026-07-06, a full GloVe-25 HNSW M=16 run with
`ef_construction=500` confirmed the expected lower-recall throughput behavior:

| ef_search | Recall@10 | QPS | p95 us |
| --- | --- | --- | --- |
| 10 | 63.1% | 87,800 | 15.8 |
| 20 | 76.0% | 57,726 | 23.2 |
| 30 | 82.5% | 44,230 | 29.2 |
| 40 | 86.2% | 35,939 | 35.7 |
| 50 | 88.8% | 27,950 | 48.5 |

This validates the basic Perplexity finding: the old 100% recall row is not the
right operating point for QPS expectations. The 95% recall point still requires
a sweep above `ef_search=50` for this build configuration.

The same current harness also validated HNSW snapshot-loaded search on the full
1.18M-vector corpus with a 1,000-query cap:

```bash
cargo run --example ann_benchmark --release --features hnsw,serde -- \
  data/ann-benchmarks/glove-25-angular \
  --algo hnsw --ef-search 50 --max-queries 1000 --snapshot-load \
  --json --fresh --results data/ann-benchmarks/results/glove-25-storage-current.jsonl
```

| Storage mode | Recall@10 | QPS | p50 us | p95 us | p99 us | Load s | Index bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| in_memory | 88.17% | 30,773.3 | 31.8 | 43.3 | 54.4 | n/a | n/a |
| snapshot_loaded | 88.17% | 30,583.7 | 32.2 | 43.4 | 52.8 | 1.6409 | 512,450,333 |

This is a persistence parity row, not a fixed-recall target row. Snapshot load
restored the same recall and near-identical warm-cache search throughput at this
operating point; the 95% recall point still needs a higher-`ef_search` sweep.

A capped HNSW storage sweep on 2026-07-07 measured the lower-recall operating
points on 50,000 indexed GloVe-25 vectors and 1,000 queries. Recall was
recomputed against the capped corpus, so these rows are storage and shape
evidence rather than full-corpus comparison rows.

```bash
cargo run --release --example ann_benchmark --no-default-features --features hnsw,serde -- \
  data/ann-benchmarks/glove-25-angular --algo hnsw \
  --ef-search 10,20,50,75,100 --max-train 50000 --max-queries 1000 \
  --snapshot-load --json \
  --results data/ann-benchmarks/results/glove-25-hnsw-storage-capped-20260707.jsonl
```

| ef_search | Storage mode | Recall@10 | QPS | p50 us | p95 us | p99 us | Load s | Index bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | in_memory | 73.85% | 207,883.5 | 4.4 | 8.5 | 10.7 | n/a | n/a |
| 10 | snapshot_loaded | 73.85% | 200,651.0 | 4.5 | 8.9 | 11.7 | 0.0640 | 20,337,704 |
| 20 | in_memory | 86.11% | 133,506.3 | 7.2 | 10.5 | 12.5 | n/a | n/a |
| 20 | snapshot_loaded | 86.11% | 130,378.6 | 7.5 | 10.3 | 12.8 | 0.0640 | 20,337,704 |
| 50 | in_memory | 94.97% | 67,846.2 | 14.5 | 18.2 | 20.7 | n/a | n/a |
| 50 | snapshot_loaded | 94.97% | 67,983.8 | 14.6 | 18.5 | 20.8 | 0.0640 | 20,337,704 |
| 75 | in_memory | 97.40% | 43,355.7 | 21.8 | 34.3 | 41.8 | n/a | n/a |
| 75 | snapshot_loaded | 97.40% | 50,448.6 | 19.6 | 24.6 | 27.6 | 0.0640 | 20,337,704 |
| 100 | in_memory | 98.47% | 37,267.6 | 26.5 | 33.1 | 38.9 | n/a | n/a |
| 100 | snapshot_loaded | 98.47% | 37,545.5 | 26.5 | 32.8 | 37.8 | 0.0640 | 20,337,704 |

The important storage result is parity: snapshot-loaded HNSW has the same recall
and comparable warm-cache QPS as the freshly built heap index. The important
QPS result is the recall curve: on this capped corpus, HNSW is above 200K QPS at
low recall and around 68K QPS just below 95% recall. Full GloVe-25 still needs
the corresponding higher-`ef_search` snapshot sweep.

A full-train HNSW storage sweep on 2026-07-07 indexed all 1,183,514
GloVe-25 vectors and capped only queries at 1,000. This is the stronger
full-corpus fixed-recall check for the Perplexity HNSW target.

```bash
cargo run --release --example ann_benchmark --no-default-features --features hnsw,serde -- \
  data/ann-benchmarks/glove-25-angular --algo hnsw \
  --ef-search 75,100,150,200 --max-queries 1000 \
  --snapshot-load --json \
  --results data/ann-benchmarks/results/glove-25-hnsw-fulltrain-storage-20260707.jsonl
```

| ef_search | Storage mode | Recall@10 | QPS | p50 us | p95 us | p99 us | Load s | Index bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 75 | in_memory | 91.72% | 20,385.7 | 47.9 | 67.7 | 81.9 | n/a | n/a |
| 75 | snapshot_loaded | 91.72% | 20,247.9 | 47.8 | 69.9 | 80.7 | 1.6267 | 512,409,052 |
| 100 | in_memory | 93.91% | 16,604.1 | 57.8 | 83.5 | 93.4 | n/a | n/a |
| 100 | snapshot_loaded | 93.91% | 18,681.4 | 53.5 | 67.1 | 77.8 | 1.6267 | 512,409,052 |
| 150 | in_memory | 96.17% | 11,870.8 | 82.8 | 116.3 | 127.0 | n/a | n/a |
| 150 | snapshot_loaded | 96.17% | 11,826.4 | 82.2 | 118.5 | 131.1 | 1.6267 | 512,409,052 |
| 200 | in_memory | 97.44% | 10,260.5 | 98.5 | 120.7 | 132.8 | n/a | n/a |
| 200 | snapshot_loaded | 97.44% | 10,240.8 | 97.8 | 121.5 | 140.0 | 1.6267 | 512,409,052 |

This confirms the lower-recall shape but narrows the HNSW performance gap:
with `ef_construction=200`, the full-corpus 95% recall operating point is
between `ef_search=100` and `ef_search=150`, and the first measured point above
95% recall is about 11.9K QPS. Snapshot-loaded search again preserves recall
and warm-cache QPS, so the remaining HNSW gap is a search/layout/tuning problem,
not a persistence problem.

A follow-up `samply` profile of the synthetic `hnsw_search_only/ef/200`
Criterion target saved `target/profiles/hnsw_search_ef200_20260707.json`.
The profile captured 10,130 main-thread leaf samples, but exported address
labels rather than fully symbolized Rust frames. Resolving address ranges with
`nm` showed this split:

| Range | Leaf samples | Share |
| --- | ---: | ---: |
| `vicinity::hnsw::search::flush_batch` | 4,074 | 40.2% |
| `innr::dense::dot` | 3,034 | 30.0% |
| `vicinity::hnsw::search::greedy_search_layer` | 2,360 | 23.3% |
| other address ranges | 528 | 5.2% |
| `vicinity::distance::cosine_distance_normalized` | 134 | 1.3% |

This makes further allocator-only work low priority. The next HNSW search pass
should focus on the batch/heap update structure around `flush_batch`, distance
kernel dispatch/inlining, and graph/vector layout locality. Future `samply`
runs should rebuild the bench with richer debuginfo or a sidecar dSYM before
claiming source-line percentages.

A repeat profile on 2026-07-08 rebuilt the same benchmark with frame pointers
and debug info in an isolated target directory:

```bash
RUSTFLAGS="-C force-frame-pointers=yes -C debuginfo=2" \
  CARGO_TARGET_DIR=/tmp/vicinity-hnsw-symbol-profile \
  CARGO_INCREMENTAL=0 RUSTC_WRAPPER= \
  samply record --save-only \
  -o /tmp/vicinity-hnsw-ef200-symbols-20260708.json.gz -- \
  cargo bench --bench hnsw_search --features hnsw -- \
  hnsw_search_only/ef/200 --profile-time 10
```

The exported profile still used address labels for the benchmark thread, but
manual `nm` range symbolication against the benchmark binary mapped 13,424
main-thread leaf samples to these function buckets:

| Bucket | Leaf samples | Share |
| --- | ---: | ---: |
| `innr::dense::dot` | 4,876 | 36.3% |
| `vicinity::hnsw::search::flush_batch` | 2,963 | 22.1% |
| `vicinity::hnsw::search::greedy_search_layer` | 2,818 | 21.0% |
| `BinaryHeap::pop` | 1,788 | 13.3% |
| other | 472 | 3.5% |
| fixture/build work | 295 | 2.2% |
| `vicinity::distance::cosine_distance_normalized` | 201 | 1.5% |

This improves confidence in the earlier profile without giving source-line
precision. The next HNSW perf attempt should be a heap/frontier-structure or
candidate-processing experiment with the same ef=10/50/100/200 controls. Do
not widen unsafe for this path; the dense kernel is already inside innr's safe
API boundary.

A later same-day profile used `CARGO_PROFILE_BENCH_DEBUG=1`, frame pointers,
and an explicit `dsymutil` pass against the benchmark executable saved by
`samply`:

```bash
CARGO_PROFILE_BENCH_DEBUG=1 RUSTFLAGS="-C force-frame-pointers=yes" \
  CARGO_TARGET_DIR=/tmp/vicinity-hnsw-profile-target CARGO_INCREMENTAL=0 \
  RUSTC_WRAPPER= samply record --save-only \
  -o /tmp/vicinity-hnsw-ef200-current-20260708.json.gz -- \
  cargo bench --bench hnsw_search --features hnsw,benchmark -- \
  hnsw_search_only/ef/200 --profile-time 10
```

That trace captured 14,069 benchmark-thread samples. Mapping the address
labels through the generated dSYM gave this leaf-sample split:

| Bucket | Leaf samples | Share |
| --- | ---: | ---: |
| `innr::dense::dot` | 5,314 | 37.8% |
| `vicinity::hnsw::search::greedy_search_layer` | 3,016 | 21.4% |
| `vicinity::hnsw::search::flush_batch` | 2,602 | 18.5% |
| `vicinity::hnsw::search::insert_result_if_accepted` | 2,000 | 14.2% |
| sort/setup/other | 918 | 6.5% |
| `vicinity::distance::cosine_distance_normalized` | 219 | 1.6% |

The new sample again argues against new local unsafe in HNSW. The largest leaf
bucket is already inside `innr`'s safe dense kernel, while the next buckets are
safe search/frontier bookkeeping. Future HNSW work should first test candidate
and result heap structure, visited-set behavior, and graph/vector layout
locality. If an `innr` helper is added, the measured shape should be a safe
row-indexed scorer over `query + flat_vectors + dim + ids -> output`.

### Graph Prefetch Hint Removal

The graph-search prefetch helper was the only product unsafe outside the PQ
SIMD kernels. It was also only a hint, so the safe negative control was to make
the helper a no-op while leaving every call site in place. HNSW search-only
Criterion controls on Apple Silicon used the same target directory before and
after the change:

```bash
CARGO_TARGET_DIR=/tmp/vicinity-prefetch-bench-target CARGO_INCREMENTAL=0 \
  RUSTC_WRAPPER= cargo bench --bench hnsw_search \
  --features hnsw,benchmark -- hnsw_search_only \
  --measurement-time 3 --warm-up-time 1 --sample-size 20

CARGO_TARGET_DIR=/tmp/vicinity-prefetch-bench-target CARGO_INCREMENTAL=0 \
  RUSTC_WRAPPER= cargo bench --bench hnsw_search \
  --features hnsw,benchmark -- hnsw_search_mmax32 \
  --measurement-time 3 --warm-up-time 1 --sample-size 20
```

For the normal `hnsw_search_only` group, replacing the architecture-specific
hint with a no-op measured:

| Row | Criterion result versus prefetch helper |
| --- | --- |
| `ef=10` | no change, -6.02% mean time |
| `ef=50` | improved, -8.14% mean time |
| `ef=100` | no change, -8.86% mean time |
| `ef=200` | improved, -17.54% mean time |

The denser `m_max=32` negative control was run in the opposite order: first
with the no-op helper, then with the prefetch helper restored. The restored
prefetch helper regressed `ef=10`, `ef=50`, and `ef=200`, and was within noise
at `ef=100`, so the no-op was the kept version:

| Row | Restored-prefetch result versus no-op baseline |
| --- | --- |
| `ef=10` | regressed, +10.69% mean time |
| `ef=50` | regressed, +8.86% mean time |
| `ef=100` | no change, +4.67% mean time |
| `ef=200` | regressed, +7.48% mean time |

DiskANN controls used the same before/after method on the in-memory direct
caller and one direct-file row:

```bash
CARGO_TARGET_DIR=/tmp/vicinity-prefetch-diskann-target CARGO_INCREMENTAL=0 \
  RUSTC_WRAPPER= cargo bench --bench diskann_search \
  --features diskann,benchmark -- diskann_search_only/memory_ef75 \
  --measurement-time 2 --warm-up-time 1 --sample-size 10

CARGO_TARGET_DIR=/tmp/vicinity-prefetch-diskann-target CARGO_INCREMENTAL=0 \
  RUSTC_WRAPPER= cargo bench --bench diskann_search \
  --features diskann,benchmark -- diskann_search_only/file_ef75 \
  --measurement-time 2 --warm-up-time 1 --sample-size 10
```

The safe no-op helper improved both rows:

| Row | Prefetch helper | No-op helper | Criterion change |
| --- | ---: | ---: | --- |
| `memory_ef75` | 10.531 ms / 100 queries | 9.5739 ms / 100 queries | -9.10% mean time |
| `file_ef75` | 96.019 ms / 100 queries | 87.592 ms / 100 queries | -8.78% mean time |

The kept change removes the prefetch unsafe surface while preserving the helper
as a safe compatibility hook. If future storage-layout work revisits prefetch,
measure it per workload and storage mode before adding any unsafe back.

The first `flush_batch` follow-up cached the current worst result distance for
every beam width. It improved high-ef rows but regressed `ef=10`, so the kept
change uses the cached-worst loop only for `ef >= 64`. The thresholded
Criterion run measured:

| ef_search | Time per 100 queries | Throughput |
| --- | ---: | ---: |
| 10 | 482.65 us | 207.19K queries/s |
| 50 | 1.8696 ms | 53.49K queries/s |
| 100 | 3.6628 ms | 27.30K queries/s |
| 200 | 6.9941 ms | 14.30K queries/s |

Against the immediately preceding cached-worst trial, the thresholded version
kept small beams out of the regressed path and improved `ef=200` by about 2.7%.
The change is still a micro-optimization; validate against a full-corpus
fixed-recall run before treating it as the main HNSW gap closure.

On 2026-07-07, a bounded DiskANN storage probe used 50,000 indexed GloVe-25
vectors and 1,000 queries. Recall was recomputed against the capped corpus, so
these rows validate the storage path and cache-state reporting but are not
full-dataset comparison rows.

```bash
cargo run --example ann_benchmark --release --features hnsw,diskann -- \
  data/ann-benchmarks/glove-25-angular \
  --algo diskann --ef-search 50 --max-train 50000 --max-queries 1000 \
  --json --fresh --results /tmp/vicinity-diskann-storage-capped.jsonl
```

| Storage mode | Recall@10 | QPS | p50 us | p95 us | p99 us | Load s | Index bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| in_memory | 92.17% | 32,095.2 | 30.8 | 42.3 | 52.7 | n/a | n/a |
| file | 92.17% | 1,251.3 | 790.3 | 1,012.0 | 1,159.9 | 0.0002 | 8,600,242 |
| mmap | 92.17% | 13,646.6 | 72.1 | 100.1 | 122.1 | 0.0002 | 8,600,242 |

The file and mmap rows visited the same graph work on average
(`avg_visited_nodes=565.97`, `avg_graph_reads=55.24`,
`avg_vector_reads=565.97`). The gap is therefore storage-access cost on this
warm-cache capped run: heap search is fastest, mmap is much closer to heap than
plain file reads, and direct file reads remain dominated by per-node vector
access.

A replicated capped-corpus DiskANN sweep then measured the recall/QPS curve
three times with 50,000 indexed vectors and 1,000 queries:

```bash
for r in 1 2 3; do
  CARGO_TARGET_DIR=/tmp/vicinity-ann-target \
    cargo run --example ann_benchmark --release --features hnsw,diskann -- \
    data/ann-benchmarks/glove-25-angular --algo diskann \
    --ef-search 50,75,100,150,200 --max-train 50000 --max-queries 1000 \
    --json --fresh --results /tmp/vicinity-diskann-glove50k-ef-curve-r${r}.jsonl
done
```

Median rows:

| Storage | ef | Recall@10 | QPS | p95 us | p99 us | Vector reads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| in_memory | 50 | 92.17% | 32,981.2 | 42.4 | 50.2 | n/a |
| in_memory | 75 | 95.16% | 24,069.8 | 56.9 | 67.6 | n/a |
| in_memory | 100 | 96.52% | 19,547.2 | 69.2 | 78.8 | n/a |
| mmap | 50 | 92.17% | 14,118.0 | 95.3 | 114.7 | 565.97 |
| mmap | 75 | 95.16% | 10,717.8 | 129.0 | 145.3 | 755.90 |
| mmap | 100 | 96.52% | 8,021.6 | 178.4 | 200.5 | 936.39 |
| file | 50 | 92.17% | 1,310.4 | 960.0 | 1,103.9 | 565.97 |
| file | 75 | 95.16% | 943.3 | 1,295.7 | 1,424.7 | 755.90 |
| file | 100 | 96.52% | 722.6 | 1,652.8 | 1,762.1 | 936.39 |

The first measured 95%+ recall operating point is `ef_search=75`, not 50.
Future DiskANN profiles should therefore target `ef=75` when validating fixed
recall, with `ef=50` kept only as a lower-recall throughput control.

The corresponding full-corpus DiskANN storage run used all 1,183,514
GloVe-25 vectors and 500 queries:

```bash
cargo run --release --example ann_benchmark --no-default-features --features hnsw,diskann -- \
  data/ann-benchmarks/glove-25-angular --algo diskann \
  --ef-search 75 --max-queries 500 --json --fresh \
  --results data/ann-benchmarks/results/glove-25-diskann-fulltrain-storage-ef75-20260707.jsonl
```

| Storage mode | Recall@10 | QPS | p50 us | p95 us | p99 us | Load s | Index bytes | Vector reads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| in_memory | 87.60% | 12,579.5 | 77.4 | 112.6 | 145.2 | n/a | n/a | n/a |
| file | 87.60% | 1,662.4 | 585.1 | 868.2 | 1,037.9 | 0.0014 | 203,564,652 | 928.74 |
| mmap | 87.60% | 5,646.0 | 172.7 | 259.1 | 339.2 | 0.0008 | 203,564,652 | 928.74 |

The next full-corpus point raised `ef_search` to 150:

```bash
cargo run --release --example ann_benchmark --no-default-features --features diskann,persistence -- \
  data/ann-benchmarks/glove-25-angular --algo diskann \
  --ef-search 150 --max-queries 500 --json --fresh \
  --results data/ann-benchmarks/results/glove-25-diskann-fulltrain-storage-ef150-20260707.jsonl
```

| Storage mode | Recall@10 | QPS | p50 us | p95 us | p99 us | Load s | Index bytes | Vector reads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| in_memory | 93.18% | 6,680.2 | 148.5 | 218.0 | 254.3 | n/a | n/a | n/a |
| file | 93.18% | 1,024.7 | 955.0 | 1,366.7 | 1,559.8 | 0.0011 | 203,564,652 | 1,561.10 |
| mmap | 93.18% | 3,287.8 | 293.9 | 462.3 | 516.4 | 0.0008 | 203,564,652 | 1,561.10 |

The first full-corpus DiskANN point above 95% recall was `ef_search=250`, with
`ef_search=200` kept as the near-miss control:

```bash
cargo run --release --example ann_benchmark --no-default-features --features diskann,persistence -- \
  data/ann-benchmarks/glove-25-angular --algo diskann \
  --ef-search 200,250 --max-queries 500 --json --fresh \
  --results data/ann-benchmarks/results/glove-25-diskann-fulltrain-storage-ef200-250-20260707.jsonl
```

| ef_search | Storage mode | Recall@10 | QPS | p50 us | p95 us | p99 us | Load s | Index bytes | Vector reads |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | in_memory | 94.96% | 5,279.2 | 189.0 | 262.5 | 301.4 | n/a | n/a | n/a |
| 200 | file | 94.96% | 814.5 | 1,242.9 | 1,697.6 | 2,025.2 | 0.0012 | 203,564,652 | 1,963.19 |
| 200 | mmap | 94.96% | 2,577.2 | 391.0 | 571.8 | 668.5 | 0.0009 | 203,564,652 | 1,963.19 |
| 250 | in_memory | 95.72% | 4,134.1 | 240.2 | 348.0 | 413.6 | n/a | n/a | n/a |
| 250 | file | 95.72% | 658.7 | 1,532.3 | 2,087.8 | 2,289.6 | 0.0012 | 203,564,652 | 2,343.51 |
| 250 | mmap | 95.72% | 2,382.1 | 421.2 | 591.1 | 651.6 | 0.0009 | 203,564,652 | 2,343.51 |

These full-corpus rows confirm the storage shape at scale: direct-file search
performs many small graph/vector reads, mmap is much closer to heap search, and
page/co-location work remains a storage-layout problem rather than a benchmark
harness problem.

The same local current-results directory now covers the observed standard
storage rows for current benchmark scopes:

```bash
uv run scripts/summarize_ann_results.py data/ann-benchmarks/results/*.jsonl \
  --current-schema-only --expect-observed-standard-storage \
  --missing-only --recall-floor 0.95
```

The missing-only output is empty for observed current scopes. Historical rows
without scope metadata still summarize as measurements, but no longer seed
storage-coverage expectations. This only proves row coverage for observed
families. It does not prove that every algorithm family has been run, it does
not promote capped rows to full-dataset results, and it does not turn below-95%
rows into fixed-recall evidence. Use `--expect-standard-storage` when auditing
an intended full algorithm/storage matrix.

A sampled dataset profile of local GloVe-25-angular on 2026-07-07 used:

```bash
uv run scripts/profile_ann_dataset.py data/ann-benchmarks/glove-25-angular \
  --sample-train 4096 --sample-queries 1000 --pair-samples 20000 \
  --output /tmp/vicinity-glove25-profile.json
```

| Field | Value |
| --- | ---: |
| Shape | 1,183,514 train x 25 dims; 10,000 queries; ground-truth k=100 |
| Train/query median norm | 1.0 / 1.0 |
| Exact duplicate fraction, sampled train | 0.0 |
| Pair-distance median, sampled train | 0.806 |
| Nearest-neighbor distance median | 0.098 |
| Top-2 gap median | 0.0089 |
| LID MLE median | 8.28 |
| Sampled relative contrast median | 8.11 |
| Ground-truth top-10 hubness | Gini 0.925; nonzero fraction 0.079 |

This supports the current reading of GloVe-25: the vectors are normalized and
pair distances are broadly dispersed, but many queries have close first/second
neighbors and the ground-truth top-10 set is hub-concentrated. Low-recall QPS can
therefore be high while 95%+ recall remains sensitive to graph traversal,
reranking, and locality.

A lighter cross-dataset pass used `--sample-train 2048 --sample-queries 500
--pair-samples 5000 --coarse-clusters 64 --coarse-iters 8` on every locally
converted standard dataset, then rendered with
`scripts/summarize_dataset_profiles.py`.

| Dataset | Metric | Train | Dim | Pair p50 | NN p50 | Top-2 gap p50 | LID p50 | Contrast p50 | Hub Gini | Coarse Gini |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deep-image-96-angular | cosine | 9,990,000 | 96 | 0.950 | 0.155 | 0.00767 | 13.2 | 5.93 | 0.990 | 0.255 |
| fashion-mnist-784-euclidean | l2 | 60,000 | 784 | 2.98e3 | 887 | 39.8 | 15.7 | 3.25 | 0.631 | 0.302 |
| gist-960-euclidean | l2 | 1,000,000 | 960 | 1.87 | 1.07 | 0.0131 | 52.2 | 1.70 | 0.991 | 0.474 |
| glove-100-angular | cosine | 1,183,514 | 100 | 0.871 | 0.326 | 0.0149 | 16.2 | 2.66 | 0.929 | 0.322 |
| glove-200-angular | cosine | 1,183,514 | 200 | 0.931 | 0.463 | 0.0188 | 21.8 | 2.00 | 0.931 | 0.363 |
| glove-25-angular | cosine | 1,183,514 | 25 | 0.804 | 0.0965 | 0.00777 | 8.40 | 7.85 | 0.925 | 0.241 |
| glove-50-angular | cosine | 1,183,514 | 50 | 0.851 | 0.204 | 0.0153 | 11.7 | 4.18 | 0.927 | 0.265 |
| mnist-784-euclidean | l2 | 60,000 | 784 | 2.61e3 | 1.09e3 | 43.4 | 14.0 | 2.40 | 0.553 | 0.221 |
| nytimes-256-angular | cosine | 290,000 | 256 | 0.986 | 0.360 | 0.0232 | 10.4 | 2.48 | 0.796 | 0.174 |
| sift-128-euclidean | l2 | 1,000,000 | 128 | 562 | 198 | 5.74 | 22.8 | 2.68 | 0.915 | 0.271 |

This table is not a substitute for benchmark rows, but it gives a cheap sanity
check before interpreting them. For example, GIST's high sampled LID, hub Gini,
and coarse-partition Gini imply that high-recall graph and quantized-search rows
should be evaluated separately from the lower-dimensional GloVe-25 curve.

Additional capped storage rows from 2026-07-07:

| Workload | Cap | Storage row | Recall@10 | QPS | Notes |
| --- | --- | --- | ---: | ---: | --- |
| IVF-PQ approximate | 50K train / 500 query | in_memory | 92.70% | 12,299.0 | `nprobe=32`, below 95% on this cap |
| IVF-PQ approximate | 50K train / 500 query | snapshot_loaded | 92.70% | 22,161.7 | persisted PQ-code sidecars rebuild scan caches on load |
| IVF-PQ approximate | 50K train / 500 query | file | 92.70% | 12,205.3 | list-local PQ codes avoid the old file-path penalty |
| IVF-PQ approximate | 50K train / 500 query | mmap | 92.70% | 19,933.1 | mmap remains faster than direct file reads |
| IVF-PQ approximate | 50K train / 500 query | in_memory | 96.64% | 11,375.8 | `nprobe=64`, first capped 95%+ row |
| IVF-PQ approximate | 50K train / 500 query | snapshot_loaded | 96.64% | 14,555.3 | persisted row at fixed recall |
| IVF-PQ approximate | 50K train / 500 query | file | 96.64% | 8,632.5 | file row at fixed recall |
| IVF-PQ approximate | 50K train / 500 query | mmap | 96.64% | 14,150.9 | mmap row at fixed recall |
| IVF-PQ rerank | 50K train / 500 query | in_memory | 93.22% | 11,159.4 | `rerank_pool=500`, still below 95% |
| IVF-PQ rerank | 50K train / 500 query | file | 93.22% | 2,018.2 | exact rerank still reads raw vectors by vector ID |
| IVF-PQ rerank | 50K train / 500 query | mmap | 93.22% | 13,549.5 | raw-vector locality is the next storage issue |
| IVF-PQ rerank | 50K train / 500 query | in_memory | 97.42% | 9,852.2 | `nprobe=64`, `rerank_pool=500` |
| IVF-PQ rerank | 50K train / 500 query | file | 97.42% | 2,150.6 | fixed-recall file rerank remains raw-vector bound |
| IVF-PQ rerank | 50K train / 500 query | mmap | 97.42% | 10,645.8 | mmap avoids most direct-file rerank cost |
| Store | 50K train / 1K query | segmented_store | 99.97% | 5,914.5 | warm after checkpoint, `index_bytes=13,838,119` |
| KD-tree | 5K train / 200 query | in_memory | 100.00% | 26,407.9 | capped low-dimensional baseline |
| KD-tree | 5K train / 200 query | snapshot_loaded | 100.00% | 26,685.8 | `load_time_s=0.0072`, `index_bytes=2,932,965` |
| Ball tree | 5K train / 200 query | in_memory | 98.55% | 18,261.0 | capped low-dimensional baseline |
| Ball tree | 5K train / 200 query | snapshot_loaded | 98.55% | 17,828.4 | `load_time_s=0.0140`, `index_bytes=5,714,434` |
| RP-tree | 5K train / 200 query | in_memory | 100.00% | 9,575.0 | current implementation traverses both subtrees |
| RP-tree | 5K train / 200 query | snapshot_loaded | 100.00% | 9,712.8 | `load_time_s=0.0089`, `index_bytes=3,597,682` |
| RP-forest | 5K train / 200 query | in_memory | 15.10% | 420,948.1 | fast, low-recall baseline |
| RP-forest | 5K train / 200 query | snapshot_loaded | 15.10% | 413,360.6 | `load_time_s=0.0219`, `index_bytes=8,890,185` |
| K-means tree | 5K train / 200 query | in_memory | 20.75% | 994,609.2 | fast, low-recall baseline |
| K-means tree | 5K train / 200 query | snapshot_loaded | 20.75% | 1,438,435.0 | `load_time_s=0.0089`, `index_bytes=3,638,525` |
| KD-tree | 50K train / 1K query | in_memory | 100.00% | 2,629.4 | exact capped baseline |
| KD-tree | 50K train / 1K query | snapshot_loaded | 100.00% | 2,700.6 | `load_time_s=0.0820`, `index_bytes=33,547,928` |
| Ball tree | 50K train / 1K query | in_memory | 95.02% | 2,791.2 | `build_time_s=21.84` |
| Ball tree | 50K train / 1K query | snapshot_loaded | 95.02% | 2,766.3 | `load_time_s=0.1645`, `index_bytes=67,328,391` |
| RP-tree | 50K train / 1K query | in_memory | 100.00% | 795.4 | exact capped baseline, current search visits both subtrees |
| RP-tree | 50K train / 1K query | snapshot_loaded | 100.00% | 798.3 | `load_time_s=0.1181`, `index_bytes=47,593,620` |
| RP-forest | 50K train / 1K query | in_memory | 24.77% | 91,630.2 | `num_trees=10`, fast low-recall baseline |
| RP-forest | 50K train / 1K query | snapshot_loaded | 24.77% | 85,595.4 | `load_time_s=0.6667`, `index_bytes=283,750,228` |
| K-means tree | 50K train / 1K query | in_memory | 17.24% | 1,063,262.0 | `num_clusters=16`, fast low-recall baseline |
| K-means tree | 50K train / 1K query | snapshot_loaded | 17.24% | 1,007,105.1 | `load_time_s=0.0971`, `index_bytes=38,734,017` |

Follow-up 50K classical sweep, using 500 queries and broader tree settings:

```bash
cargo run --example ann_benchmark --release --features kdtree,balltree,rptree,kmeans_tree -- \
  data/ann-benchmarks/glove-25-angular \
  --algo kdtree --algo balltree --algo rptree --algo rp_forest --algo kmeans_tree \
  --tree-leaf-sizes 10,50 --tree-depths 16,32 --rp-num-trees 10,20,50 \
  --kmeans-clusters 16,64 --kmeans-leaf-sizes 50,200 \
  --kmeans-depths 10,20 --kmeans-iters 10 \
  --max-train 50000 --max-queries 500 --snapshot-load --json --fresh \
  --results data/ann-benchmarks/results/glove-25-classical-50k-sweep-20260707.jsonl
```

| Algorithm | Best 50K row | Recall@10 | QPS | Notes |
| --- | --- | ---: | ---: | --- |
| KD-tree | leaf 50, depth 16 | 100.00% | 4,616.6 | high-recall baseline, still far below HNSW at the same 50K cap |
| Ball tree | leaf 50, depth 32 | 99.94% | 4,761.7 | fastest 95%+ row in the sweep |
| RP-tree | leaf 50, depth 32 | 100.00% | 1,023.6 | reaches recall by traversing enough of the tree to be slow |
| RP-forest | 50 trees, leaf 50 | 85.22% | 7,588.5 | improved from the 10-tree row but still has no 95% point |
| K-means tree | 16 clusters, leaf 200, depth 10 | 23.88% | 525,365.7 | remains a low-recall speed row under these knobs |

The sweep gives classical methods better coverage without changing the earlier
conclusion. KD-tree, ball tree, and RP-tree can serve as bounded high-recall
baselines. RP-forest and K-means tree need much larger search
budgets or reranking to become high-recall methods, at which point their QPS
advantage is expected to narrow.

A targeted 50K follow-up gave RP-forest and K-means tree those larger budgets
without reranking:

```bash
CARGO_TARGET_DIR=/tmp/vicinity-rp-forest-sweep CARGO_INCREMENTAL=0 RUSTC_WRAPPER= \
cargo run --release --example ann_benchmark --no-default-features --features rptree -- \
  data/ann-benchmarks/glove-25-angular --algo rp_forest \
  --max-train 50000 --max-queries 500 \
  --tree-leaf-sizes 100,200,500 --rp-num-trees 50,100,200 \
  --json --results /tmp/vicinity-rp-forest-sweep.jsonl

CARGO_TARGET_DIR=/tmp/vicinity-kmeans-branch-sweep CARGO_INCREMENTAL=0 RUSTC_WRAPPER= \
cargo run --release --example ann_benchmark --no-default-features --features kmeans_tree -- \
  data/ann-benchmarks/glove-25-angular --algo kmeans_tree \
  --max-train 50000 --max-queries 500 --kmeans-clusters 8 \
  --kmeans-leaf-sizes 200,500,1000 --kmeans-depths 10 \
  --kmeans-iters 10 --kmeans-search-branches 1,2,4,8 \
  --json --results /tmp/vicinity-kmeans-branch-sweep.jsonl
```

| Algorithm | Best 95%+ row | Recall@10 | QPS | Index bytes | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| RP-forest | 50 trees, leaf 200 | 97.54% | 3,990.4 | 21,294,476 | clears 95% by expanding the candidate budget |
| RP-forest | 200 trees, leaf 100 | 100.00% | 1,651.0 | 82,971,216 | exact on the 50K cap, but loses most of the speed advantage |
| K-means tree | 8 clusters, leaf 200, branch 8 | 100.00% | 367.4 | 8,469,516 | broad branch budget makes it effectively high-recall but slow |
| K-means tree | 8 clusters, leaf 1000, branch 4 | 92.10% | 1,244.8 | 8,042,124 | best near-high-recall speed row, still below 95% |

Matching snapshot-load spot checks for the best 95%+ rows preserved recall and
warm-cache QPS:

```bash
CARGO_TARGET_DIR=/tmp/vicinity-rp-forest-snapshot-target CARGO_INCREMENTAL=0 RUSTC_WRAPPER= \
cargo run --release --example ann_benchmark --no-default-features --features rptree,serde -- \
  data/ann-benchmarks/glove-25-angular --algo rp_forest \
  --max-train 50000 --max-queries 500 \
  --tree-leaf-sizes 200 --rp-num-trees 50 \
  --snapshot-load --json \
  --results data/ann-benchmarks/results/vicinity-rp-forest-snapshot-20260708.jsonl --fresh

CARGO_TARGET_DIR=/tmp/vicinity-kmeans-snapshot-target CARGO_INCREMENTAL=0 RUSTC_WRAPPER= \
cargo run --release --example ann_benchmark --no-default-features --features kmeans_tree,serde -- \
  data/ann-benchmarks/glove-25-angular --algo kmeans_tree \
  --max-train 50000 --max-queries 500 --kmeans-clusters 8 \
  --kmeans-leaf-sizes 200 --kmeans-depths 10 \
  --kmeans-iters 10 --kmeans-search-branches 8 \
  --snapshot-load --json \
  --results data/ann-benchmarks/results/vicinity-kmeans-snapshot-20260708.jsonl --fresh
```

| Algorithm | Storage mode | Recall@10 | QPS | Load s | Index bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| RP-forest | in_memory | 97.54% | 3,898.1 | n/a | 21,294,476 |
| RP-forest | snapshot_loaded | 97.54% | 3,897.7 | 0.5247 | 230,692,380 |
| K-means tree | in_memory | 100.00% | 440.6 | n/a | 8,469,516 |
| K-means tree | snapshot_loaded | 100.00% | 445.3 | 0.1552 | 30,286,231 |

This changes the classical read slightly: RP-forest is not capped below 95%
recall, but reaching that band costs enough tree and leaf budget that it sits
well below graph methods at the same 50K cap. K-means tree is controllable via
branch budget, but its 95%+ behavior is a baseline, not a competitive target.

On 2026-07-08, the in-memory benchmark rows for NSW, Vamana, NSG, SNG, EMG,
PiPNN, FINGER, and FreshGraph were wired to heap `index_bytes` from explicit
`memory_usage()` APIs rather than serialized snapshot sizes. A tiny JSON smoke
validated the emitted schema only, using 200 GloVe-25 vectors and 5 queries:

```bash
cargo run --example ann_benchmark --no-default-features --features nsw,vamana -- \
  data/ann-benchmarks/glove-25-angular --algo nsw --algo vamana \
  --ef-search 10 --max-train 200 --max-queries 5 --json --fresh \
  --results /tmp/vicinity-index-bytes-smoke.jsonl
```

Smoke rows included `storage_mode=in_memory`, `cache_state=warm_after_build`,
and positive `index_bytes` (`nsw=33,600`, `vamana=72,000`). Treat this as row
coverage evidence, not performance evidence.

The same row-coverage pass now includes the classical tree family. A bounded
schema smoke used 200 GloVe-25 vectors and 5 queries:

```bash
cargo run --example ann_benchmark --no-default-features --features kdtree,balltree,rptree,kmeans_tree -- \
  data/ann-benchmarks/glove-25-angular \
  --algo kdtree --algo balltree --algo rptree --algo rp_forest --algo kmeans_tree \
  --max-train 200 --max-queries 5 --tree-leaf-sizes 10 --tree-depths 8 \
  --rp-num-trees 3 --kmeans-clusters 4 --kmeans-leaf-sizes 20 \
  --kmeans-depths 4 --kmeans-iters 2 --json --fresh \
  --results /tmp/vicinity-classic-index-bytes-smoke.jsonl
```

The in-memory rows for KD-tree, ball tree, RP-tree, RP-forest, and K-means
tree emitted positive heap-estimated `index_bytes` (`29,408`, `36,260`,
`33,500`, `46,072`, and `38,688` bytes respectively). Treat these as schema
coverage numbers only; the workload is intentionally tiny.

The same in-memory `index_bytes` coverage now extends to the filtered and
update-heavy rows that own heap-resident structures. A bounded schema smoke used
200 GloVe-25 vectors and 5 queries:

```bash
cargo run --example ann_benchmark --no-default-features --features filtered_graph,curator,range_filtered,hnsw -- \
  data/ann-benchmarks/glove-25-angular \
  --algo filtered_graph --algo curator --algo range_filtered --algo inplace \
  --max-train 200 --max-queries 5 --ef-search 10 --json --fresh \
  --results /tmp/vicinity-filtered-index-bytes-smoke.jsonl
```

The in-memory rows emitted positive heap-estimated `index_bytes`
(`FilteredGraph=39,488`, `Curator=40,804`, `RangeFiltered=105,640`, and
`InPlace=68,768` bytes respectively). This only validates row shape and memory
accounting. Filter selectivity curves and churn fixed-recall sweeps are still
the publishable evaluation path.

A bounded filtered-search selectivity sweep now exercises the same synthetic
workload across ACORN, FilteredGraph, RangeFiltered, and Curator:

```bash
cargo run --release --example acorn_selectivity --no-default-features \
  --features hnsw,filtered_graph,range_filtered,curator -- \
  --n 3000 --queries 200 --k 10 --neighbors 32 --json --fresh \
  --results data/ann-benchmarks/results/acorn-selectivity-n3000-d32-q200-20260708.jsonl

uv run scripts/summarize_selectivity_results.py \
  data/ann-benchmarks/results/acorn-selectivity-n3000-d32-q200-20260708.jsonl
```

| Algorithm | Selectivity | Recall@10 | QPS | p95 latency | Mean returned | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ACORN | 1% | 56.35% | 6,279.9 | 226.0 us | 10.0 | returns full top-k but default traversal misses the exact filtered neighbors |
| ACORN | 50% | 69.45% | 10,345.9 | 112.0 us | 10.0 | needs fixed-recall tuning before any QPS target claim |
| Curator | 1% | 100.00% | 133,299.5 | 8.5 us | 10.0 | synthetic workload fits Curator's candidate path well |
| Curator | 50% | 100.00% | 18,410.0 | 57.0 us | 10.0 | QPS falls with broader candidate sets |
| FilteredGraph | 1% | 100.00% | 94,578.0 | 11.5 us | 10.0 | low-selectivity row is fast and exact on this fixture |
| FilteredGraph | 50% | 82.50% | 94.6 | 11,300.0 us | 10.0 | high-selectivity row collapses to a slow path |
| RangeFiltered | 1% | 3.50% | 3,809.6 | 269.0 us | 0.3 | under-returns at low selectivity |
| RangeFiltered | 50% | 100.00% | 5,048.7 | 242.5 us | 10.0 | clears recall only once the range admits enough candidates |

This sweep is diagnostic, not a publishable ACORN comparison. The useful finding
is that selectivity, returned-count, and latency-tail fields expose different
failure modes: ACORN is returning enough candidates but needs traversal tuning,
FilteredGraph has a sharp slow-path transition, and RangeFiltered visibly
under-returns at narrow ranges.

Two ACORN-only follow-up rows used the same 3K-vector, 200-query workload after
`acorn_selectivity` gained explicit tuning flags:

```bash
cargo run --release --example acorn_selectivity --no-default-features \
  --features hnsw -- \
  --n 3000 --queries 200 --k 10 --neighbors 32 \
  --ef-search 800 --acorn-max-two-hop-neighbors 128 --json --fresh \
  --results data/ann-benchmarks/results/acorn-selectivity-n3000-d32-q200-ef800-hop128-20260708.jsonl
```

| ACORN config | Selectivity | Recall@10 | QPS | p95 latency | Mean returned | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ef=200`, 32 two-hop | 1% | 56.35% | 6,279.9 | 226.0 us | 10.0 | default diagnostic row |
| `ef=400`, 64 two-hop | 1% | 63.60% | 7,252.3 | 156.8 us | 10.0 | still below 95% |
| `ef=800`, 128 two-hop | 1% | 93.55% | 3,816.6 | 286.0 us | 10.0 | close, but still below fixed-recall floor |
| `ef=200`, 32 two-hop | 2% | 47.50% | 9,804.4 | 120.0 us | 10.0 | default diagnostic row |
| `ef=400`, 64 two-hop | 2% | 69.00% | 7,321.3 | 149.2 us | 10.0 | improved but not enough |
| `ef=800`, 128 two-hop | 2% | 97.00% | 3,983.1 | 257.3 us | 10.0 | clears 95% with moderate QPS |
| `ef=800`, 128 two-hop | 5% | 98.90% | 3,942.7 | 261.5 us | 10.0 | clears 95% |
| `ef=800`, 128 two-hop | 50% | 99.85% | 3,895.4 | 264.3 us | 10.0 | clears 95% |

This supports a selectivity-gated filtered-search plan. ACORN can be tuned into
a useful moderate-selectivity path, but below 2% selectivity the benchmark still
points toward a pre-filtered exact/fallback candidate path rather than more
graph traversal.

The example now emits a `selectivity_acorn` row that uses the same tuned ACORN
path above a configurable threshold and an exact scan over pre-filtered matching
IDs below it:

```bash
cargo run --release --example acorn_selectivity --no-default-features \
  --features hnsw -- \
  --n 3000 --queries 200 --k 10 --neighbors 32 \
  --ef-search 800 --acorn-max-two-hop-neighbors 128 \
  --fallback-selectivity-threshold 0.02 --json --fresh \
  --results data/ann-benchmarks/results/acorn-selectivity-n3000-d32-q200-ef800-hop128-fallback0p0200-20260708.jsonl
```

| Algorithm | Selectivity | Recall@10 | QPS | p95 latency | 2-hop nodes | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ACORN | 1% | 93.80% | 3,770.6 | 272.0 us | 454,465 | graph traversal still misses the fixed-recall floor |
| `selectivity_acorn` | 1% | 100.00% | 2,012,436.9 | 0.5 us | 0 | exact over 30 pre-filtered IDs; useful as a policy check, not a production QPS claim |
| `selectivity_acorn` | 2% | 96.45% | 3,767.4 | 272.1 us | 454,113 | stays on tuned ACORN because the threshold is `< 0.02` |
| `selectivity_acorn` | 50% | 99.85% | 3,083.5 | 339.7 us | 430,710 | stays on tuned ACORN |

The ACORN search loop now has a dedicated Criterion target:

```bash
CARGO_TARGET_DIR=/tmp/vicinity-acorn-visited-bench CARGO_INCREMENTAL=0 \
  RUSTC_WRAPPER= cargo bench --bench acorn_search --no-default-features \
  --features hnsw -- acorn_search_only --sample-size 20 \
  --warm-up-time 1 --measurement-time 3 --save-baseline acorn_visited_before
```

A `samply` profile of `acorn_search_only/selectivity_0.10` saved
`/tmp/vicinity-acorn-search-s0p10-20260708.json.gz`. The benchmark thread had
9,568 samples. After resolving the hottest addresses with `atos`, the largest
leaf buckets were `hashbrown::HashMap::insert` and
`hashbrown::raw::RawTable::reserve_rehash`; `innr::dense::dot` was only about
1-2% of leaf samples. That makes this filtered-search shape a safe
data-structure target, not an unsafe SIMD target.

Pre-sizing ACORN's per-query visited set to the same order as the traversal cap
(`ef_search * 3`, plus two-hop and result slack, bounded at 1M entries) improved
the same saved-baseline comparison:

| Workload | Before | After | Criterion change |
| --- | ---: | ---: | --- |
| `acorn_search_only/selectivity_0.50` | 35.980 ms / 128 queries | 27.386 ms / 128 queries | 23.9% faster, p < 0.05 |
| `acorn_search_only/selectivity_0.10` | 33.455 ms / 128 queries | 25.969 ms / 128 queries | 22.5% faster, p < 0.05 |
| `acorn_search_only/selectivity_0.02` | 33.563 ms / 128 queries | 24.982 ms / 128 queries | 24.0% faster, p < 0.05 |

The benchmark counters stayed in the same regime after the change: roughly
127-139 two-hop invocations per query, 2.1K-2.3K two-hop nodes examined per
query, and 10 returned results per query.

A follow-up `samply` profile of the same row after preallocation saved
`/tmp/vicinity-acorn-search-s0p10-postprealloc-20260708.json.gz`; the benchmark
thread had 10,532 samples. The largest resolved leaf buckets were still
`hashbrown::HashMap::insert`, while `reserve_rehash` dropped out of the top
resolved addresses. Generalizing the neighbor callback from `Vec<u32>` to
`AsRef<[u32]>` removes forced cloning for in-memory HNSW and slice-backed
benchmarks, but did not produce a statistically significant Criterion change:

| Workload | Before | After | Criterion result |
| --- | ---: | ---: | --- |
| `acorn_search_only/selectivity_0.50` | 28.881 ms / 128 queries | 28.182 ms / 128 queries | no significant change, p = 0.58 |
| `acorn_search_only/selectivity_0.10` | 27.101 ms / 128 queries | 25.572 ms / 128 queries | no significant change, p = 0.26 |
| `acorn_search_only/selectivity_0.02` | 26.748 ms / 128 queries | 24.698 ms / 128 queries | no significant change, p = 0.78 |

The next ACORN change reused HNSW's safe dense generation-counter visited
tracker for contiguous node-ID graphs. The public default ACORN function stays
sparse; `HNSWIndex::search_acorn_with_stats` and the benchmark use the
node-count path, which opts into dense tracking only when `node_count <= 1M` and
`node_count <= visited_capacity_hint * 64`. That keeps large in-memory or
file-backed indexes from zeroing a full node-count array when the expected
visited set is small.

Compared with the same `acorn_borrow_before` baseline, the dense visited path
improved the dedicated search-only rows:

| Workload | Sparse visited | Dense visited | Criterion change |
| --- | ---: | ---: | --- |
| `acorn_search_only/selectivity_0.50` | 28.881 ms / 128 queries | 10.164 ms / 128 queries | 64.9% faster, p < 0.05 |
| `acorn_search_only/selectivity_0.10` | 27.101 ms / 128 queries | 8.8395 ms / 128 queries | 67.1% faster, p < 0.05 |
| `acorn_search_only/selectivity_0.02` | 26.748 ms / 128 queries | 8.5102 ms / 128 queries | 68.2% faster, p < 0.05 |

A post-dense profile saved
`/tmp/vicinity-acorn-search-s0p10-postdense-20260708.json.gz`. The benchmark
thread had 10,180 samples. The former `HashMap::insert` hot bucket disappeared;
the top resolved addresses were inlined ACORN loop work at
`benches/acorn_search.rs:87`, with `innr::dense::dot` visible again. The next
ACORN pass should use a more line-attributed profile or targeted loop
microbenchmarks before changing the search loop further.

Full-train IVF-PQ storage sweep from the same day, using all 1,183,514
GloVe-25 vectors and 500 queries:

```bash
cargo run --release --example ann_benchmark --no-default-features --features ivf_pq,hnsw,persistence -- \
  data/ann-benchmarks/glove-25-angular --algo ivfpq \
  --pq-clusters 1024 --pq-codebooks 25 --pq-codebook-size 256 \
  --pq-training-sample-size 100000 --pq-kmeans-max-iter 20 \
  --pq-nprobes 32,64 --pq-rerank-pools 500 --max-queries 500 \
  --snapshot-load --json \
  --results data/ann-benchmarks/results/glove-25-ivfpq-fulltrain-storage-20260707.jsonl
```

| Workload | Storage row | Recall@10 | QPS | p95 latency | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| IVF-PQ `nprobe=32` | in_memory | 95.42% | 2,640.0 | 522.2 us | first full-corpus 95%+ row in this sweep |
| IVF-PQ `nprobe=32` | snapshot_loaded | 95.42% | 3,134.7 | 440.8 us | `load_time_s=0.0648`, `index_bytes=187,254,901` |
| IVF-PQ `nprobe=32` | file | 95.42% | 2,552.8 | 535.0 us | direct-file PQ-code path stays in the target band |
| IVF-PQ `nprobe=32` | mmap | 95.42% | 2,722.3 | 500.6 us | mmap is near heap at this setting |
| IVF-PQ `nprobe=32`, rerank 500 | in_memory | 96.58% | 2,514.2 | 538.0 us | exact rerank improves recall without a large heap penalty |
| IVF-PQ `nprobe=32`, rerank 500 | snapshot_loaded | 96.58% | 2,570.4 | 532.1 us | persisted heap row stays comparable |
| IVF-PQ `nprobe=32`, rerank 500 | file | 96.58% | 1,453.4 | 833.0 us | still raw-vector-read bound, but no longer a 69-QPS class failure |
| IVF-PQ `nprobe=32`, rerank 500 | mmap | 96.58% | 2,533.9 | 524.3 us | mmap avoids most direct-file raw-vector cost |
| IVF-PQ `nprobe=64` | in_memory | 97.48% | 1,330.6 | 972.2 us | higher recall costs about 2x QPS |
| IVF-PQ `nprobe=64` | snapshot_loaded | 97.48% | 1,399.5 | 943.4 us | persisted row stays comparable |
| IVF-PQ `nprobe=64` | file | 97.48% | 1,281.5 | 986.2 us | direct-file approximate path remains near heap |
| IVF-PQ `nprobe=64` | mmap | 97.48% | 1,386.2 | 930.3 us | mmap remains near heap |
| IVF-PQ `nprobe=64`, rerank 500 | in_memory | 98.82% | 1,293.2 | 1,000.1 us | high-recall rerank row |
| IVF-PQ `nprobe=64`, rerank 500 | snapshot_loaded | 98.82% | 1,326.3 | 973.7 us | persisted row stays comparable |
| IVF-PQ `nprobe=64`, rerank 500 | file | 98.82% | 890.4 | 1,355.5 us | remaining direct-file locality gap |
| IVF-PQ `nprobe=64`, rerank 500 | mmap | 98.82% | 1,314.0 | 969.8 us | mmap row stays near heap |

The current-schema storage coverage check for this result scope reports no
missing observed rows:

```bash
uv run scripts/summarize_ann_results.py data/ann-benchmarks/results/*.jsonl \
  --current-schema-only --expect-observed-standard-storage \
  --only-dataset 'glove-25-angular[queries=500]' \
  --missing-only --json
```

Output:

```json
[]
```

The classical rows are useful storage and API coverage, not full-scale ANN
recommendations. The next IVF-PQ file-rerank work should avoid duplicating
full-precision vectors and instead profile candidate order, read batching,
page/cache behavior, or a raw-vector layout change that replaces the old layout
without growing snapshots.

## Profiling Ledger

These are incremental profiling findings from the current optimization pass.
They are workload-specific and should be re-run before making release claims.

### Build-Path Profiling Note

During the HNSW tombstone-persistence verification pass, a local
`hnsw,persistence` test compile appeared stuck for several minutes. A macOS
`sample` capture of the live `rustc` process showed the main thread almost
entirely in `readdir` while scanning the dependency search path
(`target/debug/deps`), not in type checking or code generation. The compile had
actually finished shortly after the sample; rerunning the same test with the
built artifact took 0.41s to start and the test itself completed immediately.

Practical takeaways for future profiling/verification runs:

- use a dedicated `CARGO_TARGET_DIR=/tmp/vicinity-...` for profile targets or
  heavy feature-matrix checks when the shared `target/debug/deps` directory has
  grown large;
- if a Rust compile looks idle, sample `rustc` before assuming source-level
  complexity or a cargo lock;
- keep final gates reproducible by recording any `RUSTC_WRAPPER=`,
  `CARGO_INCREMENTAL=0`, and `CARGO_TARGET_DIR=...` overrides.

### HNSW Search Loop

Search-only Criterion benchmark:

```bash
CARGO_TARGET_DIR=/tmp/vicinity-hnsw-batch-target \
  cargo bench --bench hnsw_search --features hnsw -- \
  hnsw_search_only/ef/50 --measurement-time 3 --warm-up-time 1 --sample-size 20
```

Baseline with 4-neighbor distance batches:

| Workload | Time | Throughput |
| --- | --- | --- |
| `hnsw_search_only/ef/50` | 2.0073 ms / 100 queries | 49.819 Kelem/s |

Increasing the internal distance batch to 8 improved the same benchmark:

| Workload | Time | Throughput | Criterion change |
| --- | --- | --- | --- |
| `hnsw_search_only/ef/50` | 1.9511 ms / 100 queries | 51.252 Kelem/s | 4.6% faster, p < 0.05 |

The supporting `samply` profile was recorded with:

```bash
CARGO_TARGET_DIR=/tmp/vicinity-hnsw-profile-target \
  samply record --save-only -o /tmp/vicinity-hnsw-search-20260706.json.gz -- \
  cargo bench --bench hnsw_search --features hnsw -- \
  hnsw_search_only/ef/50 --profile-time 15
```

Symbolized top samples showed `cosine_distance_normalized` at about 36.5%
inclusive time and `innr::dense::dot` as the largest leaf bucket at about
23.1%. `greedy_search_layer` and `flush_batch` were also hot. This supports
keeping `innr` for dense distance kernels and optimizing HNSW graph traversal
around the kernel rather than replacing the kernel in `vicinity`.

The `hnsw_search` benchmark now also reports heap allocation counts per query.
On the 10K-vector, 128-d fixture, search allocates about 7 times per query, with
bytes scaling with `ef`:

| ef_search | Alloc calls/query | Alloc bytes/query | Time / 100 queries |
| --- | --- | --- | --- |
| 10 | 6.9 | 813.6 | 476.42 us |
| 50 | 7.0 | 3,796.0 | 1.9362 ms |
| 100 | 7.0 | 7,556.0 | 3.7768 ms |
| 200 | 7.0 | 13,156.0 | 7.3105 ms |

This makes heap traffic visible for future work, but the counts are bounded and
do not yet justify a heap-first rewrite without a profile showing allocator time
or result-heap maintenance as the bottleneck.

A follow-up negative control tested replacing the normal HNSW base-layer
function-pointer distance dispatch with metric-specialized generic wrappers at
the `search` and MQO API boundary:

```bash
cargo bench --bench hnsw_search --features hnsw,benchmark -- \
  hnsw_search_only --measurement-time 3 --warm-up-time 1 --sample-size 20
```

Criterion reported no useful win and clear high-ef regressions against the
stored baseline:

| Workload | Criterion mean change | Decision |
| --- | ---: | --- |
| `hnsw_search_only/ef/10` | +0.45% time | no change |
| `hnsw_search_only/ef/50` | +5.84% time | rejected |
| `hnsw_search_only/ef/100` | +5.60% time | rejected |
| `hnsw_search_only/ef/200` | +10.17% time | rejected |

The patch was not committed. If distance dispatch is revisited, use binary
inspection or a narrower distance-kernel entrypoint rather than generic wrappers
around the whole graph traversal.

A follow-up binary inspection pass on the current code confirmed that the
function-pointer concern is real, but also narrows where it appears. The command
used:

```bash
CARGO_TARGET_DIR=/tmp/vicinity-asm-target CARGO_INCREMENTAL=0 RUSTC_WRAPPER= \
  cargo asm --lib --features hnsw --rust --context=1 greedy_search_layer 1 \
  > /tmp/vicinity-hnsw-greedy-search-layer.asm
```

The saved assembly has 16,562 lines. The relevant excerpt is in the
`flush_batch` symbol, where the unrolled batched-distance loop lowers
`dists[i] = dist_fn(query, vec)` to an indirect `blr x7` call. That validates
the dispatch hypothesis at the machine-code level, but the rejected benchmark
above says the next experiment should not be another whole-search generic
wrapper. The next candidate should isolate either `flush_batch` or the
distance-kernel entrypoint and keep the same ef=10/50/100/200 negative-control
rows.

That narrower candidate was tested next: `flush_batch` was made generic over
the distance function, then the normal `HNSWIndex::search` and MQO base-layer
calls were routed through metric-specialized entrypoints. Construction and
custom-distance search kept the existing function-pointer API. The candidate
reduced one allocation per query in the synthetic bench but still regressed the
search loop:

```bash
# Baseline, before applying the candidate patch.
CARGO_TARGET_DIR=/tmp/vicinity-hnsw-metric-bench CARGO_INCREMENTAL=0 \
  RUSTC_WRAPPER= cargo bench --bench hnsw_search --features hnsw -- \
  hnsw_search_only --measurement-time 3 --warm-up-time 1 --sample-size 20 \
  --save-baseline hnsw_metric_before

# Candidate, compared against the saved baseline.
CARGO_TARGET_DIR=/tmp/vicinity-hnsw-metric-bench CARGO_INCREMENTAL=0 \
  RUSTC_WRAPPER= cargo bench --bench hnsw_search --features hnsw -- \
  hnsw_search_only --measurement-time 3 --warm-up-time 1 --sample-size 20 \
  --baseline hnsw_metric_before
```

| Workload | Baseline mean | Candidate mean | Criterion change | Decision |
| --- | ---: | ---: | ---: | --- |
| `hnsw_search_only/ef/10` | 482.50 us | 508.52 us | +5.27% time | rejected |
| `hnsw_search_only/ef/50` | 1.8545 ms | 1.9316 ms | +4.51% time | rejected |
| `hnsw_search_only/ef/100` | 3.6265 ms | 3.7774 ms | +3.41% time | rejected |
| `hnsw_search_only/ef/200` | 7.1294 ms | 7.1557 ms | +2.01% time | no useful win |

The candidate patch was reverted. This makes the current dispatch evidence
more specific: the indirect `blr` exists in `flush_batch`, but monomorphizing
the standard-search entrypoints does not help this workload. Further HNSW
search work should prioritize heap-update structure, visited/result layout, and
vector/neighbor locality before another dispatch rewrite.

A still narrower safe-Rust variant specialized only the normalized-cosine
base-layer search path and left the generic function-pointer path in place for
all other metrics. It was also rejected. This adds evidence that local
monomorphization of the common cosine row is not worth keeping for this
workload unless a future profile changes the hotspot shape.

```bash
# Baseline, before applying the cosine-only candidate patch.
CARGO_TARGET_DIR=/tmp/vicinity-hnsw-dispatch-target CARGO_INCREMENTAL=0 \
  RUSTC_WRAPPER= cargo bench --bench hnsw_search --no-default-features \
  --features hnsw -- hnsw_search_ --sample-size 20 --warm-up-time 1 \
  --measurement-time 3 --save-baseline dispatch_baseline

# Candidate, compared against the saved baseline.
CARGO_TARGET_DIR=/tmp/vicinity-hnsw-dispatch-target CARGO_INCREMENTAL=0 \
  RUSTC_WRAPPER= cargo bench --bench hnsw_search --no-default-features \
  --features hnsw -- hnsw_search_ --sample-size 20 --warm-up-time 1 \
  --measurement-time 3 --baseline dispatch_baseline
```

| Workload | Baseline mean | Candidate mean | Criterion change | Decision |
| --- | ---: | ---: | ---: | --- |
| `hnsw_search_only/ef/10` | 464.52 us | 518.83 us | +13.21% time | rejected |
| `hnsw_search_only/ef/50` | 1.9015 ms | 1.9572 ms | +2.21% time | rejected |
| `hnsw_search_only/ef/100` | 3.5769 ms | 3.6563 ms | +2.16% time | rejected |
| `hnsw_search_only/ef/200` | 6.9958 ms | 7.0166 ms | no significant change | rejected |
| `hnsw_search_mmax32/ef/10` | 826.72 us | 829.33 us | within noise threshold | rejected |
| `hnsw_search_mmax32/ef/50` | 3.0185 ms | 2.9781 ms | no significant change | rejected |
| `hnsw_search_mmax32/ef/100` | 5.5988 ms | 5.5184 ms | -1.61% time | rejected |
| `hnsw_search_mmax32/ef/200` | 10.626 ms | 10.600 ms | within noise threshold | rejected |

The code change was reverted. The next useful HNSW perf target is still
candidate/frontier structure or data locality, not another dispatch rewrite.

A measurement-only `distance_dispatch` Criterion group was added next to keep
future dispatch work grounded before touching HNSW again. The benchmark compares
eight normalized candidate distances through a direct
`cosine_distance_normalized` call versus a black-boxed
`fn(&[f32], &[f32]) -> f32` call at 25, 128, and 960 dimensions.

```bash
CARGO_TARGET_DIR=/tmp/vicinity-distance-dispatch-target CARGO_INCREMENTAL=0 \
  RUSTC_WRAPPER= cargo bench --bench distance --no-default-features -- \
  distance_dispatch --sample-size 30 --warm-up-time 1 --measurement-time 3

CARGO_TARGET_DIR=/tmp/vicinity-distance-dispatch-innr-target \
  CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo bench --bench distance \
  --no-default-features --features innr -- distance_dispatch \
  --sample-size 30 --warm-up-time 1 --measurement-time 3
```

| Feature set | Dim | Direct mean | Function-pointer mean | Delta |
| --- | ---: | ---: | ---: | ---: |
| scalar fallback | 25 | 65.665 ns | 66.840 ns | +1.8% |
| scalar fallback | 128 | 459.86 ns | 514.47 ns | +11.9% |
| scalar fallback | 960 | 6.7344 us | 6.9067 us | +2.6% |
| `innr` | 25 | 35.014 ns | 44.305 ns | +26.5% |
| `innr` | 128 | 101.15 ns | 110.35 ns | +9.1% |
| `innr` | 960 | 666.73 ns | 656.26 ns | -1.6% |

This confirms that function-pointer dispatch can matter in the low-dimensional
kernel loop, especially with the `innr` path HNSW normally uses. It does not
reverse the production-search result above: every HNSW dispatch rewrite tried so
far lost or failed to clear the 5% keep threshold on the real search rows.
Future dispatch work should start from this bench plus the `hnsw_search_`
ef=10/50/100/200 and `m_max=32` controls, not from a broad hot-path rewrite.

The next safe heap-update experiment replaced the result-heap push-then-pop
sequence with top replacement through `BinaryHeap::peek_mut`. It targets the
`BinaryHeap::pop` bucket from the symbolized profile without changing distance
kernels, graph layout, or unsafe code.

```bash
# Baseline before applying the heap replacement.
CARGO_TARGET_DIR=/tmp/vicinity-hnsw-heapreplace-bench CARGO_INCREMENTAL=0 \
  RUSTC_WRAPPER= cargo bench --bench hnsw_search --features hnsw -- \
  hnsw_search_only --sample-size 20 --warm-up-time 1 --measurement-time 3

# Candidate after applying the heap replacement, same target dir and flags.
CARGO_TARGET_DIR=/tmp/vicinity-hnsw-heapreplace-bench CARGO_INCREMENTAL=0 \
  RUSTC_WRAPPER= cargo bench --bench hnsw_search --features hnsw -- \
  hnsw_search_only --sample-size 20 --warm-up-time 1 --measurement-time 3
```

| Workload | Baseline mean | Candidate mean | Criterion change | Decision |
| --- | ---: | ---: | ---: | --- |
| `hnsw_search_only/ef/10` | 487.04 us | 479.59 us | -2.57% time | kept |
| `hnsw_search_only/ef/50` | 1.9160 ms | 1.9090 ms | no significant change | kept |
| `hnsw_search_only/ef/100` | 3.6831 ms | 3.5919 ms | -2.86% time | kept |
| `hnsw_search_only/ef/200` | 7.0050 ms | 6.7698 ms | -3.93% time | kept |

Correctness checks:

```bash
CARGO_TARGET_DIR=/tmp/vicinity-hnsw-heapreplace-tests CARGO_INCREMENTAL=0 \
  RUSTC_WRAPPER= cargo test --no-default-features --features hnsw --lib hnsw

CARGO_TARGET_DIR=/tmp/vicinity-hnsw-heapreplace-tests CARGO_INCREMENTAL=0 \
  RUSTC_WRAPPER= cargo test --no-default-features --features hnsw \
  --test search_with_distance_parity
```

Both passed. `cross_algorithm_consistency` was also compiled with only `hnsw`,
but that integration test is gated on `hnsw,nsw,diskann,sng`, so it emitted
zero tests in that feature set.

A follow-up finalization experiment tried to remove the intermediate top-k
`Vec` allocation in `HNSWIndex::search` by chaining `take(k)` directly into
the tombstone/doc-id conversion. The intended invariant was unchanged
semantics: `greedy_search_layer` output stayed sorted, `take(k)` still happened
before tombstone filtering, and external doc-id mapping was unchanged.

```bash
CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo bench --bench hnsw_search \
  --features hnsw -- hnsw_search_only --sample-size 20 --warm-up-time 1 \
  --measurement-time 3 --save-baseline finalize_before

CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo bench --bench hnsw_search \
  --features hnsw -- hnsw_search_only --sample-size 20 --warm-up-time 1 \
  --measurement-time 3 --baseline finalize_before
```

The allocation diagnostics did not move: all rows stayed at 5 allocation calls
per query. Criterion also reported regressions on the lower-ef rows, so the
experiment was reverted.

| Workload | Baseline mean | Candidate mean | Criterion change | Decision |
| --- | ---: | ---: | ---: | --- |
| `hnsw_search_only/ef/10` | 686.02 us | 787.98 us | +25.71% time | rejected |
| `hnsw_search_only/ef/50` | 2.7448 ms | 3.5912 ms | +44.96% time | rejected |
| `hnsw_search_only/ef/100` | 5.8881 ms | 6.6302 ms | +13.68% time | rejected |
| `hnsw_search_only/ef/200` | 10.601 ms | 12.422 ms | no significant change | rejected |

Another finalization experiment tried to drain the result heap with repeated
`BinaryHeap::pop()` calls followed by `reverse()`, instead of `drain()` plus
`sort_unstable_by`. The intended invariant was unchanged sorted ascending
output while preserving the thread-local result heap allocation. It looked
plausible because the heap already contains ordered structure, but it regressed
every normal HNSW row and was reverted before the denser control rows finished.

```bash
CARGO_TARGET_DIR=/tmp/vicinity-hnsw-drain-target CARGO_INCREMENTAL=0 \
  RUSTC_WRAPPER= cargo bench --bench hnsw_search --features hnsw,benchmark -- \
  --save-baseline before_drain

CARGO_TARGET_DIR=/tmp/vicinity-hnsw-drain-target CARGO_INCREMENTAL=0 \
  RUSTC_WRAPPER= cargo bench --bench hnsw_search --features hnsw,benchmark -- \
  --baseline before_drain
```

| Workload | Baseline mean | Candidate mean | Criterion change | Decision |
| --- | ---: | ---: | ---: | --- |
| `hnsw_search_only/ef/10` | 896.40 us | 1.0351 ms | +13.29% time | rejected |
| `hnsw_search_only/ef/50` | 3.2261 ms | 3.8112 ms | +18.13% time | rejected |
| `hnsw_search_only/ef/100` | 6.3245 ms | 7.1811 ms | +13.54% time | rejected |
| `hnsw_search_only/ef/200` | 11.442 ms | 13.003 ms | +13.65% time | rejected |

Conclusion: keep the current `drain()` plus `sort_unstable_by` finalization for
the standard HNSW search path. The next heap work should target candidate/result
maintenance during traversal, not heap teardown after traversal.

A second follow-up swept `DISTANCE_BATCH_SIZE` from 8 to 16 to test whether
larger independent distance batches help the `flush_batch` hotspot. It was
measured against both default `m_max=16` and denser `m_max=32` graph rows:

```bash
CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo bench --bench hnsw_search \
  --features hnsw -- hnsw_search_ --sample-size 20 --warm-up-time 1 \
  --measurement-time 3 --save-baseline batch8

CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo bench --bench hnsw_search \
  --features hnsw -- hnsw_search_ --sample-size 20 --warm-up-time 1 \
  --measurement-time 3 --baseline batch8
```

The default graph rows were statistically neutral, but the denser graph
negative controls regressed at `ef=10` and `ef=200`, so batch size 16 was
reverted. The next batch experiment should be dimension-aware or workload-aware
rather than a global constant change.

| Workload | Batch-8 run | Batch-16 run | Criterion change | Decision |
| --- | ---: | ---: | ---: | --- |
| `hnsw_search_only/ef/10` | 885.89 us | 839.79 us | no significant change | rejected |
| `hnsw_search_only/ef/50` | 3.2292 ms | 3.1887 ms | no significant change | rejected |
| `hnsw_search_only/ef/100` | 6.3094 ms | 5.9090 ms | no significant change | rejected |
| `hnsw_search_only/ef/200` | 11.390 ms | 11.830 ms | no significant change | rejected |
| `hnsw_search_mmax32/ef/10` | 1.4080 ms | 1.7395 ms | +9.50% time | rejected |
| `hnsw_search_mmax32/ef/50` | 5.0563 ms | 5.2606 ms | no significant change | rejected |
| `hnsw_search_mmax32/ef/100` | 9.1639 ms | 7.8941 ms | no significant change | rejected |
| `hnsw_search_mmax32/ef/200` | 17.116 ms | 17.993 ms | +21.71% time | rejected |

A frontier-pruning follow-up then targeted the remaining `BinaryHeap::pop`
bucket. The kept variant prunes stale frontier candidates only for `ef >= 64`,
only every 64 popped candidates, and keeps candidates with distance equal to the
current worst result to preserve the strict HNSW stopping condition. An earlier
32-pop interval was rejected because the densest `m_max=32,ef=200` negative
control regressed.

```bash
CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo bench --bench hnsw_search \
  --features hnsw -- hnsw_search_ --sample-size 20 --warm-up-time 1 \
  --measurement-time 3 --baseline batch8
```

| Workload | Batch-8 run | Frontier-prune run | Criterion change | Decision |
| --- | ---: | ---: | ---: | --- |
| `hnsw_search_only/ef/10` | 885.89 us | 838.68 us | no significant change | kept |
| `hnsw_search_only/ef/50` | 3.2292 ms | 2.8731 ms | no significant change | kept |
| `hnsw_search_only/ef/100` | 6.3094 ms | 5.3339 ms | -8.19% time | kept |
| `hnsw_search_only/ef/200` | 11.390 ms | 10.254 ms | -8.97% time | kept |
| `hnsw_search_mmax32/ef/10` | 1.4080 ms | 1.2406 ms | -12.59% time | kept |
| `hnsw_search_mmax32/ef/50` | 5.0563 ms | 4.7936 ms | -11.33% time | kept |
| `hnsw_search_mmax32/ef/100` | 9.1639 ms | 8.5566 ms | no significant change | kept |
| `hnsw_search_mmax32/ef/200` | 17.116 ms | 16.373 ms | no significant change | kept |

Correctness checks:

```bash
cargo test --no-default-features --features hnsw --lib hnsw
cargo test --no-default-features --features hnsw --test search_with_distance_parity
cargo test --no-default-features --features hnsw --test regression_known_bugs
cargo test --no-default-features --features hnsw --test hnsw_e2e
cargo clippy --no-default-features --features hnsw --lib -- -D warnings
```

These passed: 99 HNSW lib tests plus 1 ignored measurement-only test, 2
search-with-distance parity tests, 5 known-regression tests, 13 HNSW e2e tests,
and the HNSW clippy slice.

### Dense Distance Kernel

Direct Criterion measurements on the 128-d L2 kernel:

| Feature set | Time | Throughput |
| --- | --- | --- |
| `--features innr` | 8.382 ns | 15.271 Gelem/s |
| `--no-default-features` | 52.846 ns | 2.422 Gelem/s |

Binary inspection of the benchmark artifact confirmed that the `innr` path
lowers to an aarch64 NEON loop with paired vector loads and `fmla.4s`
accumulation. The actionable conclusion is to use `innr` for full-vector dense
distance and spend `vicinity` work on search layout, pruning, and storage.

A follow-up read of `innr` found that its safe dense API already covers the
current `vicinity::simd` boundary (`dot`, `cosine`, `l2_distance`,
`l2_distance_squared`, and `norm`). Its `VerticalBatch` API is useful when a
candidate set is already materialized in a dense columnar batch, but HNSW
neighbor traversal visits graph-adjacent vectors by scattered ids. Packing each
small neighbor batch into `VerticalBatch` inside `flush_batch` would add
allocation/transposition to the hot loop, so it is not a justified HNSW change
without a profile showing candidate locality or a storage layout that amortizes
the packing. New unsafe SIMD in `vicinity` should remain limited to local
layout-specific kernels such as PQ code/LUT scans; dense full-vector kernels
should continue to enter through innr's safe API.

The narrow future innr helper to test is a safe row-indexed scorer that writes
one distance or dot product per caller-provided vector id into caller-owned
output. That matches HNSW/DiskANN row-major storage better than `VerticalBatch`.
Do not add it speculatively: first compare against current `flush_batch` at
several dimensions and graph densities, then run end-to-end HNSW, DiskANN, and
IVF rerank/centroid controls. The global HNSW batch-size-16 experiment above
already regressed denser-graph controls, so a future batch/scoring helper needs
to be dimension- or workload-aware.

### DiskANN File And Mmap Search

The search-only DiskANN benchmark isolates in-memory, file, and mmap search
from construction:

```bash
cargo bench --bench diskann_search --no-default-features --features diskann,benchmark -- \
  diskann_search_only --measurement-time 3 --warm-up-time 1 --sample-size 20
```

Baseline at 5,000 vectors, 64 dimensions, 100 queries, `ef=50`:

| Row | Time | Throughput |
| --- | --- | --- |
| `memory_ef50` | 7.7394 ms / 100 queries | 12.921 Kelem/s |
| `file_ef50` | 147.37 ms / 100 queries | 678.55 elem/s |
| `mmap_ef50` | 19.290 ms / 100 queries | 5.1840 Kelem/s |

A reusable neighbor-buffer experiment avoided allocating a `Vec<u32>` per graph
record read, but it did not improve the measured path:

| Row | Criterion change |
| --- | --- |
| `memory_ef50` | no change, -0.18% mean time |
| `file_ef50` | regressed, +1.70% mean time, p < 0.05 |
| `mmap_ef50` | no change, -0.16% mean time |

That experiment was rejected. DiskANN storage work should next target record
layout, syscall count, vector decoding, and graph/vector co-location rather
than neighbor-list allocation alone. The `ann_benchmark` DiskANN file and mmap
rows now emit average graph reads, vector reads, logical bytes, visited nodes,
and retained candidates so storage changes can be compared against query work.

A later callback-style neighbor visitor removed the owned neighbor vector from
the search loop entirely, but regressed the same search-only bench:

| Row | Criterion change |
| --- | --- |
| `memory_ef50` | no change, +0.85% mean time |
| `file_ef50` | regressed, +5.07% mean time, p < 0.05 |
| `mmap_ef50` | regressed, +5.24% mean time, p < 0.05 |

That experiment was also rejected. The measured path is not currently limited
by the neighbor-list allocation enough to justify callback overhead.

Removing the extra mmap vector-byte copy in `DiskANNSearcher::read_vector`
improved the same benchmark while leaving in-memory search statistically flat:

| Row | After | Change vs baseline |
| --- | --- | --- |
| `memory_ef50` | 7.7063 ms / 100 queries | no change versus baseline |
| `file_ef50` | 141.38 ms / 100 queries | about 4.1% faster |
| `mmap_ef50` | 13.338 ms / 100 queries | about 30.9% faster |

The fixed-recall file row was then profiled with Criterion's profile mode:

```bash
CARGO_TARGET_DIR=/tmp/vicinity-diskann-debug-profile-target \
  CARGO_PROFILE_BENCH_DEBUG=1 \
  RUSTFLAGS='-C force-frame-pointers=yes' \
  samply record --save-only \
  -o /tmp/vicinity-diskann-file-ef75-debug-10s.json.gz -- \
  cargo bench --bench diskann_search --no-default-features \
  --features diskann,benchmark -- diskann_search_only/file_ef75 --profile-time 10
```

Two setup lessons came out of this profile. A warmed target is required; a cold
target records compiler and `sccache` samples. `CARGO_PROFILE_BENCH_DEBUG=1`
plus a generated dSYM is also required on macOS before `atos` can resolve Rust
frames from the saved profile. With those in place, the benchmark thread still
reported address-only frames in the JSON, but manual symbolication worked by
adding the Mach-O `__TEXT` base address (`0x100000000`) before calling `atos`.

The useful profile finding was storage-shaped: `file_ef75` spent 9,390 of
12,467 benchmark-thread leaf samples in `libsystem_kernel.dylib`. The Rust-side
inclusive stack ran through `DiskANNSearcher::search_with_diagnostics`,
`DiskGraphReader::get_neighbors`, and `DiskANNSearcher::read_vector`. That
matched the code: the direct-file graph reader was doing one seek, one 4-byte
degree read, and then one 4-byte read per neighbor.

Batching each node's neighbor IDs into one reusable-buffer `read_exact` cut the
direct-file rows without changing mmap or heap search:

| Row | Before | After | Change |
| --- | ---: | ---: | ---: |
| `file_ef50` | 141.38 ms / 100 queries | 82.718 ms / 100 queries | about 41.5% faster |
| `file_ef75` | 199.28 ms / 100 queries | 112.57 ms / 100 queries | about 43.5% faster |

Control rows after the change:

| Row | Time / 100 queries | Notes |
| --- | ---: | --- |
| `memory_ef50` | 7.8148 ms | within prior noise |
| `memory_ef75` | 10.550 ms | fixed-recall heap control |
| `mmap_ef50` | 13.280 ms | within prior noise |
| `mmap_ef75` | 17.551 ms | fixed-recall mmap control |

The next file-backed profile target was cursor movement. Switching direct-file
graph and vector reads from `seek` plus `read_exact` to positional reads cut the
same file rows again while keeping heap and mmap controls flat:

| Row | Buffered-read row | Positional-read row | Change |
| --- | ---: | ---: | ---: |
| `file_ef50` | 82.718 ms / 100 queries | 54.506 ms / 100 queries | about 34.1% faster |
| `file_ef75` | 112.57 ms / 100 queries | 72.892 ms / 100 queries | about 35.2% faster |

The cumulative direct-file improvement from the pre-profile baseline is:

| Row | Original | Current | Change |
| --- | ---: | ---: | ---: |
| `file_ef50` | 141.38 ms / 100 queries | 54.506 ms / 100 queries | about 61.4% faster |
| `file_ef75` | 199.28 ms / 100 queries | 72.892 ms / 100 queries | about 63.4% faster |

After the full-corpus run found `ef_search=250` as the first 95%+ recall
DiskANN point, the search-only benchmark added an `ef250` row and profiled the
direct-file path:

```bash
samply record --unstable-presymbolicate --save-only \
  -o /tmp/vicinity-diskann-file-ef250-presym-20260707.json.gz -- \
  cargo bench --bench diskann_search --no-default-features --features diskann,benchmark -- \
  diskann_search_only/file_ef250 --profile-time 10
```

The saved profile was not useful for source-line claims because the raw profile
still contained address-heavy frames. Source inspection did find that the
file/mmap searcher still used a per-query `HashSet` for visited nodes while the
in-memory DiskANN, Vamana, NSW, and HNSW search paths already use dense
generation-counter visited sets. Moving the same pattern into
`DiskANNSearcher` improved both fixed-recall storage rows:

| Row | Before | After | Criterion change |
| --- | ---: | ---: | ---: |
| `file_ef250` | 151.43 ms / 100 queries | 140.45 ms / 100 queries | 7.25% faster mean time, p < 0.05 |
| `mmap_ef250` | 39.478 ms / 100 queries | 25.338 ms / 100 queries | 35.82% faster mean time, p < 0.05 |

The next DiskANN storage step is still record layout: the full-corpus
fixed-recall file row performs about 2,343 vector reads/query, and current
disk-resident graph literature points toward page-sized records that co-locate
vectors, neighbor IDs, and often compressed neighbor vectors.

An experimental `benchmark`-feature-only page layout then wrote one 4KB-aligned
`nodes.page` record per node: external id, vector, and neighbor IDs in one
record. The implementation keeps the format out of the normal `diskann` API and
adds page-specific diagnostics (`page_reads`, `page_bytes`) so graph/vector
file counters are not overloaded. Unit tests enforce that records are page
aligned, page file/mmap search matches heap search, and each visited node
causes one page read. Short Criterion smoke command:

```bash
cargo bench --bench diskann_search --no-default-features \
  --features diskann,benchmark -- diskann_search_only \
  --sample-size 10 --warm-up-time 0.1 --measurement-time 0.1
```

Same-run smoke results rejected this first 4KB-per-node layout as a promoted
path. It is useful as a harness for future compressed/coalesced page layouts,
but not as a replacement for the current file or mmap searcher:

| Row | Current layout | 4KB page layout | Verdict |
| --- | ---: | ---: | --- |
| `file_ef50` | 48.838 ms / 100 queries | 72.252 ms / 100 queries | slower |
| `mmap_ef50` | 9.034 ms / 100 queries | 17.317 ms / 100 queries | slower |
| `file_ef75` | 67.815 ms / 100 queries | 98.139 ms / 100 queries | slower |
| `mmap_ef75` | 12.366 ms / 100 queries | 23.334 ms / 100 queries | slower |
| `file_ef250` | 140.60 ms / 100 queries | 200.89 ms / 100 queries | slower |
| `mmap_ef250` | 25.557 ms / 100 queries | 48.759 ms / 100 queries | slower |

The next page-layout attempt should reduce page bloat before another
full-corpus run: compressed vectors, multiple low-dimensional nodes per page, or
a hot-node cache/prefetch layer. A literal one-node-one-page layout multiplies
the full GloVe-25 page file to roughly 4.8 GB, so it mainly measures page-cache
pressure at this dimensionality.

### HNSW Search Heap Pressure

The `hnsw_search` benchmark has an allocation counter around the normal
`HNSWIndex::search` path. On the 10K-vector, 128-dimension search-only bench,
the main search path was still allocating fresh candidate and result heaps for
each query even though the visited set already uses thread-local reuse.

Reusing the candidate/result heaps through a thread-local scratch buffer in
`greedy_search_layer` reduced allocation pressure and gave a small but measured
QPS gain:

| Row | Before | After |
| --- | ---: | ---: |
| `ef=50` allocation calls | 7.0/query | 5.0/query |
| `ef=50` allocation bytes | 3,796 bytes/query | 1,388 bytes/query |
| `hnsw_search_only/ef/50` | 1.9566 ms / 100 queries | 1.8877 ms / 100 queries |

Criterion reported the `ef=50` row about 3.9% faster. This is useful but not a
substitute for the bigger GloVe-25 fixed-recall work: HNSW still needs
full-corpus lower-recall sweeps and cache-layout profiling.

A follow-up changed HNSW's internal neighbor list alias from
`SmallVec<[u32; 16]>` to `SmallVec<[u32; 32]>`, matching the default
base-layer `m_max=32`. The guard
`neighbor_list_holds_default_base_degree_inline` verifies 32 neighbors stay
inline.

The search-only benchmark now includes a separate `hnsw_search_mmax32` group so
default-degree rows can be measured without changing the historical
`hnsw_search_only` fixture. A short same-binary Criterion run used:

```bash
CARGO_TARGET_DIR=/tmp/vicinity-hnsw-mmax32-bench CARGO_INCREMENTAL=0 \
  RUSTC_WRAPPER= cargo bench --bench hnsw_search --features hnsw -- \
  hnsw_search_ --sample-size 15 --warm-up-time 0.5 --measurement-time 1
```

| Workload | Time / 100 queries | Throughput | Alloc bytes/query |
| --- | ---: | ---: | ---: |
| `m=16,m_max=16,ef=10` | 459.36 us | 217.69K queries/s | 348 |
| `m=16,m_max=16,ef=50` | 1.8611 ms | 53.73K queries/s | 1,388 |
| `m=16,m_max=16,ef=100` | 3.5796 ms | 27.94K queries/s | 2,748 |
| `m=16,m_max=16,ef=200` | 6.9111 ms | 14.47K queries/s | 3,548 |
| `m=16,m_max=32,ef=10` | 789.85 us | 126.61K queries/s | 260 |
| `m=16,m_max=32,ef=50` | 3.0304 ms | 33.00K queries/s | 1,060 |
| `m=16,m_max=32,ef=100` | 5.5578 ms | 17.99K queries/s | 2,100 |
| `m=16,m_max=32,ef=200` | 10.658 ms | 9.38K queries/s | 2,900 |

This is a graph-topology result, not a before/after measurement of the
SmallVec capacity change. The `m_max=32` graph exposes more base-layer
neighbors and is slower on this 10K-vector synthetic fixture; the inline
capacity change still avoids heap spills for that default-degree topology.

### IVF-PQ Search Loop

The `ivfpq_search` benchmark's profiled counters show the remaining hot path is
ADC-table construction and scan dispatch, not final result selection:

| Shape | ADC table | ADC dispatch | Finalizer | Allocations |
| --- | --- | --- | --- | --- |
| `m25_one_dim_nprobe32_k10` | 97.315 us/query | 21.485 us/query | 2.568 us/query | 14 calls/query, 50.2 KB/query |
| `m5_runner_default_nprobe32_k10` | 46.726 us/query | 5.085 us/query | 2.511 us/query | 13 calls/query, 29.1 KB/query |

A bounded top-k heap experiment regressed GloVe-25 IVF-PQ at the same recall:

| Row | Before | Bounded top-k |
| --- | --- | --- |
| IVF-PQ `nprobe=32` | 1,759.6 QPS, 95.42% recall | 1,598.7 QPS, 95.42% recall |
| IVF-PQ `nprobe=32`, rerank 500 | 1,710.0 QPS, 96.58% recall | 1,418.6 QPS, 96.58% recall |

That experiment was rejected. The next IVF-PQ work should focus on the ADC
table path, codebook shape, and SIMD scan layout.

A file-searcher scratch-buffer reuse experiment was also rejected. It reused
centroid-distance, residual, code-batch, distance-batch, and ADC-table buffers
inside `IVFPQFileSearcher::search_approx_internal`, but the short Criterion run
regressed the rows that matter most:

| Row | Criterion change |
| --- | ---: |
| `m25_one_dim_file_nprobe32_k10` | about 10.7% faster |
| `m25_one_dim_file_nprobe32_rerank500_k10` | about 5.3% slower |
| `m25_one_dim_mmap_nprobe32_rerank500_k10` | about 4.1% slower |
| `m5_runner_default_file_nprobe32_rerank500_k10` | about 3.9% slower |
| `m5_runner_default_mmap_nprobe32_rerank500_k10` | within noise |

Command:

```bash
cargo bench --bench ivfpq_search --no-default-features \
  --features ivf_pq,persistence,benchmark -- \
  --sample-size 10 --warm-up-time 0.1 --measurement-time 0.1
```

The regression suggests the per-query allocation cleanup is not the current
file-rerank bottleneck. Keep the next attempt focused on ADC-table construction,
scan layout, or file-read planning.

The DiskANN positional-read helper also applies to IVF-PQ file-backed search:
`IVFPQFileSearcher` used to seek and then read for every byte-slice fetch. Using
the same positional-read helper moved the targeted file rerank rows without
touching mmap or heap paths:

| Row | Current time / 100 queries | Criterion change |
| --- | ---: | ---: |
| `m25_one_dim_file_nprobe32_rerank500_k10` | 24.410 ms | about 36.6% faster |
| `m5_runner_default_file_nprobe32_rerank500_k10` | 27.069 ms | about 33.0% faster |

The remaining file-backed IVF-PQ locality problem is layout, not cursor
movement: exact rerank still reads raw full-precision vectors by vector ID.
IVF-PQ file and mmap rerank benchmark rows now report this directly via
`avg_vector_reads`, `avg_vector_bytes`, and `avg_retained_candidates`.

A 5K-vector smoke row verifies the diagnostic fields on the benchmark output:

```bash
cargo run --release --example ann_benchmark --no-default-features --features ivf_pq,persistence -- \
  data/ann-benchmarks/glove-25-angular --algo ivfpq \
  --pq-clusters 64 --pq-codebooks 5 --pq-codebook-size 32 \
  --pq-training-sample-size 2000 --pq-kmeans-max-iter 3 \
  --pq-nprobes 4 --pq-rerank-pools 20 \
  --max-train 5000 --max-queries 20 --snapshot-load --json --fresh \
  --results /tmp/vicinity-ivfpq-diagnostics-smoke.jsonl
```

The file and mmap rerank rows both report `avg_vector_reads=20.00`,
`avg_vector_bytes=2000.00`, and `avg_retained_candidates=20.00`, which matches
`rerank_pool=20` on 25-d `f32` vectors.

An external read-only implementation review found the same shape in other
systems: FAISS on-disk IVF and SPANN/SPFresh organize work around posting-list
locality, while Lance keeps row IDs beside PQ codes and treats row-id stability
as a format invariant. A compatibility-preserving experiment added optional
`list_raw_offsets.bin` and `list_raw_vectors.bin` sidecars while retaining
`raw_vectors.bin`. The full-train fixed-recall row rejected that shape: it
duplicated raw vectors, grew the snapshot, and did not improve direct-file
rerank.

| Row | Before | With duplicate list-raw sidecar | Verdict |
| --- | ---: | ---: | --- |
| IVF-PQ `nprobe=32`, file rerank 500 | 1,453.4 QPS, 187.3 MB | 1,449.4 QPS, 305.6 MB | Rejected |
| IVF-PQ `nprobe=32`, mmap rerank 500 | 2,533.9 QPS, 187.3 MB | 2,301.8 QPS, 305.6 MB | Rejected |

Command for the rejected run:

```bash
cargo run --release --example ann_benchmark --no-default-features --features ivf_pq,persistence -- \
  data/ann-benchmarks/glove-25-angular --algo ivfpq \
  --pq-clusters 1024 --pq-codebooks 25 --pq-codebook-size 256 \
  --pq-training-sample-size 100000 --pq-kmeans-max-iter 20 \
  --pq-nprobes 32 --pq-rerank-pools 500 --max-queries 500 \
  --snapshot-load --json --fresh \
  --results /tmp/vicinity-ivfpq-list-raw-fulltrain.jsonl
```

## Benchmark Coverage

The dense `ann_benchmark` runner covers the implemented single-vector ANN
families that accept dense `Vec<f32>` input:

| Area | Algorithms / rows | Eval path |
| --- | --- | --- |
| Graph search | HNSW, NSW, Vamana, DiskANN, NSG, SNG, EMG, DualBranch, DEG, PiPNN, FINGER, FreshGraph | `ann_benchmark --algo ...`; HNSW, NSW, Vamana, NSG, SNG, EMG, PiPNN, FINGER, and FreshGraph support `--snapshot-load`; DEG caps dense runs at 10,000 indexed vectors because construction is O(n^2) |
| IVF / quantized search | IVF-PQ, IVF-PQ rerank, IVF-AVQ, IVF-RaBitQ, RP-Quant, BinaryIndex, SQ4 | `ann_benchmark --algo ...`; IVF-PQ supports sampled training; these rows support `--snapshot-load` where the index is saved, reopened, and then searched from memory |
| Quantized HNSW accelerators | SQ4U, SQ8U, SymphonyQG, SymphonyQG-VR, ADSampling, HNSW-PRT | `ann_benchmark --algo ...`; SQ4U, SQ8U, SymphonyQG, and SymphonyQG-VR support `--snapshot-load` when their persistence features are compiled |
| Filtering | FilteredGraph, RangeFiltered, Curator, ACORN | dense rows via `ann_benchmark` are labeled `filter_mode=none`; FilteredGraph, RangeFiltered, and Curator support `--snapshot-load`; selectivity curves live in `acorn_selectivity` |
| Classical baselines | KD-tree, Ball tree, RP-tree, RP-forest, K-means tree, brute force | `ann_benchmark --algo ...`; tree rows can add `--snapshot-load` |
| Streaming / updates | FreshGraph churn, in-place graph, in-place churn, LSM churn | `ann_benchmark --algo fresh_graph_churn --algo inplace --algo inplace_churn --algo lsm_churn` |

This table is runner capability, not proof that each row has a fresh
current-schema measurement in this file. Current-schema fixed-recall coverage is
still concentrated on HNSW, DiskANN, IVF-PQ/rerank, and segmented store. Graph
alternatives, quantized HNSW variants, non-PQ quantized indexes, filtered rows,
classical tree baselines, and churn workloads still need fresh fixed-recall
rows with `storage_mode`, `cache_state`, latency percentiles, and index-size
fields before they should be compared as measured results.

Not every implemented module should produce a dense ANN row:

| Module | Why it is separate | Honest eval direction |
| --- | --- | --- |
| SparseMIPS | Requires sparse vectors, not dense ann-benchmarks `f32` arrays | Use the SPV1 smoke harness for plumbing; add SPLADE/BM25 data before publishing QPS/recall |
| EVoC | Clustering wrapper, not nearest-neighbor search | Report clustering metrics such as NMI/ARI on labeled clustering datasets |
| LEMUR | Inference scaffold that requires externally trained encoder weights | Evaluate on multi-vector retrieval datasets with MaxSim ground truth |
| SAQ / quantization helpers | Quantizers, not standalone indexes | Report quantization error, encoding throughput, and downstream recall when attached to an index |

Persistence and storage-mode benchmark rows follow `docs/persistence.md`.
Methods with both in-memory and file-backed search should report separate
`storage_mode=in_memory`, `storage_mode=file`, and, when implemented,
`storage_mode=mmap` rows. File and mmap rows should also report `load_time_s`
and `index_bytes` when the runner opens a saved index.

The long-term target is not only faster heap-resident search. In-memory QPS is
the easiest number to improve and the least representative for datasets that do
not fit in RAM. For large datasets, benchmark and implementation work should
prefer layouts that search persisted bytes directly: DiskANN graph/vector files,
mmap graph pages, IVF posting lists, segmented stores, and cold/warm page-cache
measurements. Snapshot-loaded rows remain useful correctness and reload
baselines, but they are not a substitute for file or mmap search rows.

### IVF-PQ sampled-training diagnostic (2026-07-06)

Command:

```bash
cargo run --example ann_benchmark --release --features ivf_pq,hnsw -- \
  data/ann-benchmarks/glove-25-angular --algo ivfpq \
  --pq-clusters 1024 --pq-codebooks 25 --pq-codebook-size 256 \
  --pq-training-sample-size 100000 --pq-kmeans-max-iter 20 \
  --pq-nprobes 16,32,64 --pq-rerank-pools 500,5000 \
  --results /tmp/vicinity-ivfpq-glove25-cb25-sampled.jsonl --fresh --json
```

Build: 71.37s, RSS 643,904 KB, `rustc 1.95.0`. These rows are in-memory,
warm-after-build search on GloVe-25.

Rows marked `*` were re-run after the IVF-PQ candidate finalizer switched from
full sorting to `select_nth_unstable_by` plus sorting the retained prefix
(`dfbf176`). That targeted re-run used the same dataset and IVF-PQ shape, with
build time 72.13s and RSS 644,320 KB.

Rows marked `**` add the direct 1-D ADC table path used by the GloVe-25
`cb=25` shape. That targeted re-run used the same dataset and IVF-PQ shape,
with build time 72.25s and RSS 644,240 KB.

| nprobe | Rerank pool | Recall@10 | QPS | p95 us |
|--------|-------------|-----------|-----|--------|
| 16 | none | 91.60% | 1,034.5 | 1,399.9 |
| 16 | 500 | 92.43% | 1,016.6 | 1,409.6 |
| 16 | 5,000 | 92.43% | 826.5 | 1,657.0 |
| 32 | none** | 95.64% | 811.7 | 1,684.2 |
| 32 | 500** | 96.69% | 796.3 | 1,706.9 |
| 32 | 5,000 | 96.69% | 471.7 | 2,797.4 |
| 64 | none | 97.68% | 266.9 | 4,760.1 |
| 64 | 500 | 98.91% | 265.2 | 4,783.3 |
| 64 | 5,000 | 98.91% | 252.3 | 4,963.0 |

Interpretation: sampled training is not the recall blocker when the PQ shape
has enough capacity. The earlier `cb=5, codebook_size=16` recipe was a
memory-stress configuration and saturates at low recall on GloVe-25. With
`cb=25, codebook_size=256`, recall crosses 95% at `nprobe=32`. The remaining
gap versus FAISS-style IVF-PQ targets is QPS, so the next optimization target is
the ADC/FastScan scan path; larger rerank pools are not helping this shape.

Search-only profiling on the same day showed that the standard 8-bit ADC path
was allocating and copying a packed LUT for every probed cluster even though the
ADC table was already flat. Borrowing that table in the dispatch path improved
the Criterion `ivfpq_search_only` benchmark on Apple Silicon:

| Shape | Before | After | Change |
|-------|--------|-------|--------|
| `m25_one_dim_nprobe32_k10` | 15.25 ms / 100 queries | 13.91 ms / 100 queries | 8.8% faster |
| `m25_one_dim_nprobe32_rerank500_k10` | 16.96 ms / 100 queries | 14.88 ms / 100 queries | 12.2% faster |
| `m5_runner_default_nprobe32_k10` | 10.02 ms / 100 queries | 10.01 ms / 100 queries | no significant change |
| `m5_runner_default_nprobe32_rerank500_k10` | 11.16 ms / 100 queries | 10.54 ms / 100 queries | 5.6% faster |

Command:

```bash
cargo bench --bench ivfpq_search --no-default-features \
  --features ivf_pq,benchmark -- \
  --sample-size 10 --warm-up-time 0.1 --measurement-time 0.1
```

A follow-up small-subvector ADC-table path avoids repeated generic L2 calls for
PQ subvector dimensions up to 8. This matters for the default GloVe-25
`m=5` shape, where each subvector has only 5 dimensions. The same Criterion
command measured:

| Shape | Before small-dim table path | After small-dim table path | Change |
|-------|-----------------------------|----------------------------|--------|
| `m5_runner_default_nprobe32_k10` | 10.01 ms / 100 queries | 7.22 ms / 100 queries | 27.9% faster |
| `m5_runner_default_nprobe32_rerank500_k10` | 10.54 ms / 100 queries | 8.04 ms / 100 queries | 23.7% faster |

With the `innr` feature enabled, the same path measured 6.96 ms / 100 queries
for `m5_runner_default_nprobe32_k10`. Before the small-dimension specialization,
enabling `innr` regressed this shape to 13.47 ms / 100 queries because the hot
loop repeatedly called a general dense-vector L2 function below `innr`'s SIMD
threshold. The split is now: `innr` for full-vector dense kernels, local IVF-PQ
kernels for tiny-subvector ADC-table construction and lookup-table scan.

The next profile showed the standard 8-bit ADC path was still copying PQ codes
from vector-order storage into a cluster-order scan buffer for every probed
cluster. Prepacking those cluster-order code buffers at build/load time removed
that search-time copy (`code_copy=0.000us/query`) and improved the same
Criterion bench:

| Shape | Before cluster-code cache | After cluster-code cache | Change |
|-------|---------------------------|--------------------------|--------|
| `m25_one_dim_nprobe32_k10` | 13.91 ms / 100 queries | 12.78 ms / 100 queries | 8.1% faster |
| `m25_one_dim_nprobe32_rerank500_k10` | 14.88 ms / 100 queries | 14.01 ms / 100 queries | 5.8% faster |
| `m5_runner_default_nprobe32_k10` | 7.22 ms / 100 queries | 5.94 ms / 100 queries | 17.7% faster |
| `m5_runner_default_nprobe32_rerank500_k10` | 8.04 ms / 100 queries | 6.98 ms / 100 queries | 13.2% faster |

A bounded 500-query GloVe-25 run at the R@10≈0.95 operating point moved from
820.7 QPS after the earlier ADC-table work to 1,759.6 QPS after the
cluster-code cache:

```bash
cargo run --no-default-features --features ivf_pq,innr,serde --release \
  --example ann_benchmark -- data/ann-benchmarks/glove-25-angular \
  --algo ivfpq --pq-clusters 1024 --pq-codebooks 25 \
  --pq-codebook-size 256 --pq-training-sample-size 100000 \
  --pq-kmeans-max-iter 20 --pq-nprobes 32 --pq-rerank-pools 500 \
  --max-queries 500 --json --fresh
```

| Algorithm | Recall@10 | QPS | p95 latency |
|-----------|-----------|-----|-------------|
| IVF-PQ `nprobe=32` | 95.42% | 1,759.6 | 759.8 us |
| IVF-PQ `nprobe=32`, rerank 500 | 96.58% | 1,710.0 | 775.7 us |

The bench target now also reports allocation counts for each profiled
`search_profiled()` call. Reusing the standard ADC distance buffer across
probed clusters reduced heap activity, but did not materially change
search-only timing. That means heap allocation was real, but not the remaining
primary bottleneck for these shapes:

| Shape | Before distance-buffer reuse | After distance-buffer reuse |
|-------|------------------------------|-----------------------------|
| `m25_one_dim_nprobe32_k10` | 48 alloc calls/query, 105.4 KB/query | 18 alloc calls/query, 96.4 KB/query |
| `m5_runner_default_nprobe32_k10` | 48 alloc calls/query, 84.9 KB/query | 18 alloc calls/query, 75.9 KB/query |

Preallocating the query candidate buffer from the probed cluster sizes cut the
remaining search allocations again:

| Shape | After distance-buffer reuse | After candidate preallocation |
|-------|-----------------------------|-------------------------------|
| `m25_one_dim_nprobe32_k10` | 18 alloc calls/query, 96.4 KB/query | 8 alloc calls/query, 50.8 KB/query |
| `m5_runner_default_nprobe32_k10` | 18 alloc calls/query, 75.9 KB/query | 8 alloc calls/query, 30.3 KB/query |

On 2026-07-07, binary inspection with `cargo asm --lib --no-default-features
--features "ivf_pq benchmark"
vicinity::ivf_pq::pq::ProductQuantizer::compute_adc_table_into` showed the
one-dimensional ADC-table path still went through `Vec::push` growth checks for
each table entry. Rewriting only that one-dimensional path to size the table
once and fill mutable chunks moved the same short Criterion smoke as follows:

| Shape | Before | After | ADC table before | ADC table after |
|-------|--------|-------|------------------|-----------------|
| `m25_one_dim_nprobe32_k10` | 13.08 ms / 100 queries | 4.51 ms / 100 queries | 99.466 us/query | 17.513 us/query |
| `m25_one_dim_nprobe32_rerank500_k10` | 13.56 ms / 100 queries | 5.28 ms / 100 queries | 98.020 us/query | 17.503 us/query |

The broader variant that changed all ADC-table paths was rejected: it regressed
`m5_runner_default_nprobe32_k10` by about 3.8% in the same smoke. The kept
change is intentionally limited to `subvector_dim == 1`, which is the GloVe-25
`cb=25` shape where the QPS gap matters most.

The bounded 500-query GloVe-25 macro run at the same R@10≈0.95 operating point
then moved from 1,759.6 QPS to 2,149.7 QPS for approximate search, and from
1,710.0 QPS to 2,105.5 QPS with rerank pool 500:

| Algorithm | Recall@10 | QPS | p95 latency |
|-----------|-----------|-----|-------------|
| IVF-PQ `nprobe=32` | 95.42% | 2,149.7 | 648.2 us |
| IVF-PQ `nprobe=32`, rerank 500 | 96.58% | 2,105.5 | 662.2 us |

The next binary-inspection pass looked at
`vicinity::pq_simd::adc_batch_dispatch_into`. On Apple Silicon, the compiler
vectorized the generic NEON dispatcher four candidates at a time, but the
inner loop still carried bounds-check bailout blocks for LUT gathers. A
specialized aarch64 path for the common 8-bit PQ case (`codebook_size=256`)
checks the LUT shape once and then gathers through raw pointers inside the
NEON loop. The short Criterion probe moved as follows:

| Shape | Before 8-bit NEON specialization | After 8-bit NEON specialization | ADC dispatch before | ADC dispatch after |
|-------|----------------------------------|---------------------------------|---------------------|--------------------|
| `m25_one_dim_nprobe32_k10` | 4.60 ms / 100 queries | 3.81 ms / 100 queries | 20.783 us/query | 13.484 us/query |
| `m25_one_dim_nprobe32_rerank500_k10` | 5.61 ms / 100 queries | 4.74 ms / 100 queries | 21.459 us/query | 13.841 us/query |
| `m5_runner_default_nprobe32_k10` | 5.94 ms / 100 queries | 5.84 ms / 100 queries | 5.319 us/query | 3.429 us/query |
| `m5_runner_default_nprobe32_rerank500_k10` | 6.97 ms / 100 queries | 6.63 ms / 100 queries | 5.135 us/query | 3.376 us/query |

The same bounded 500-query GloVe-25 macro run then moved from 2,149.7 QPS to
2,941.4 QPS for approximate search, and from 2,105.5 QPS to 2,806.2 QPS with
rerank pool 500:

| Algorithm | Recall@10 | QPS | p95 latency |
|-----------|-----------|-----|-------------|
| IVF-PQ `nprobe=32` | 95.42% | 2,941.4 | 470.6 us |
| IVF-PQ `nprobe=32`, rerank 500 | 96.58% | 2,806.2 | 488.5 us |

The separate 4-bit FastScan path (`codebook_size=16`) was then checked because
the source comments still described a shuffle/table-lookup kernel while the
implemented block scanner was portable scalar. The baseline and after run used
the same saved Criterion baseline:

```bash
cargo bench --bench pq_simd -- pq_fastscan_lut_shape/flat_lut \
  --measurement-time 3 --warm-up-time 1 --sample-size 20 \
  --save-baseline before_fastscan_neon

cargo bench --bench pq_simd -- pq_fastscan_lut_shape/flat_lut \
  --measurement-time 3 --warm-up-time 1 --sample-size 20 \
  --baseline before_fastscan_neon
```

Adding an aarch64 NEON `tbl` block kernel moved the 1,024-vector flat-LUT
FastScan microbench from 3.3636 us to 934.32 ns:

| Shape | Before | After | Criterion change |
|-------|-------:|------:|------------------|
| `pq_fastscan_lut_shape/flat_lut` | 3.3636 us | 934.32 ns | 72.25% lower mean time |

This is a real FastScan-kernel fix, but it does not change the primary
GloVe-25 fixed-recall row above: that row uses `codebook_size=256` and the
standard 8-bit ADC path. The associated tests include a direct NEON-vs-portable
block parity check and the existing 4-bit IVF-PQ search tests.

The storage-mode Criterion probe compares the freshly built heap index,
snapshot-loaded heap index, plain file searcher, and mmap searcher on the same
20K-vector synthetic shape used for short profiling. Throughput is queries per
second; each sample processes 100 queries. Adding optional list-contiguous
`list_codes.bin` plus `list_offsets.bin` sidecars removes the per-query
vector-order gather for file and mmap searchers while preserving fallback for
old snapshots.

| Shape | Heap after | Snapshot after | File before | File after | Mmap before | Mmap after |
|-------|-----------:|---------------:|------------:|-----------:|------------:|-----------:|
| `m25_one_dim_nprobe32_k10` | 26.4K | 26.3K | 675 | 16.7K | 17.4K | 25.9K |
| `m25_one_dim_nprobe32_rerank500_k10` | 21.2K | 21.4K | 556 | 2.61K | 13.6K | 18.7K |
| `m5_runner_default_nprobe32_k10` | 17.8K | 17.7K | 689 | 12.4K | 12.4K | 17.9K |
| `m5_runner_default_nprobe32_rerank500_k10` | 15.3K | 15.2K | 555 | 2.40K | 10.1K | 13.9K |

Interpretation: `load_from_dir()` is effectively at heap speed. List-contiguous
PQ-code sidecars fix the largest file/mmap approximate-search penalty: file
search improves by roughly 18-25x and mmap becomes close to heap search. Rerank
file mode still lags because exact reranking reads raw vectors in candidate
order from `raw_vectors.bin`. Sorting file-mode rerank reads by vector index did
not materially change the `m25_one_dim` row and improved the `m5_runner_default`
row by about 2.5%, so it is a small cleanup rather than the full rerank fix. A
later duplicate list-raw sidecar experiment was also rejected on full-train
rows, so the next file-rerank attempt needs a different storage design or read
plan.

## Legacy GloVe-25 (1.18M vectors, 25-d, angular distance)

Ground truth: brute-force k-NN on L2-normalized vectors (angular ≡ cosine for unit vectors).

### Summary

| Algorithm | Best Recall@10 | QPS at best | Notes |
|-----------|---------------|-------------|-------|
| HNSW (M=16) | 100.0% | 2,857 | Default choice |
| HNSW (M=32) | 100.0% | 2,017 | Higher memory, marginal recall gain |
| Vamana | 100.0% | 1,177 | Slow build (~2000s) |
| DiskANN | 100.0% | 1,029 | Vamana-family graph; legacy in-memory row |
| SQ4U | 99.9% | 1,056 | 4-bit quantized HNSW; ~3x slower than plain HNSW at d=25 |
| NSW | 99.2% | 1,288 | |
| IVF-PQ (cb=25) | 98.7% | 69 | 25 codebooks on 25-d (1-d subspaces) |
| IVF-AVQ | 90.9% | 194 | ScaNN-style anisotropic VQ |
| RP-Forest | 58.5% | 4,221 | Fast build, moderate recall |
| IVF-PQ (cb=5) | 45.1% | 262 | 5 codebooks on 25-d (too coarse) |
| KD-Tree | 100.0% | 22 | Classical low-dimensional baseline; too slow at 1M+ |
| Brute | 100.0% | 42 | Exact baseline |

SQ4U and SymphonyQG are designed for high-dimensional data. At d=25, the quantization
overhead exceeds the savings from cheaper distance computation. SymphonyQG produces
<0.3% recall at d=25 (RaBitQ error is O(1/sqrt(d))) and is excluded.

### HNSW (M=16, ef_construction=200)

| ef_search | Recall@10 | QPS |
|-----------|-----------|-----|
| 10 | 77.8% | 57,747 |
| 20 | 87.9% | 35,138 |
| 50 | 96.1% | 17,035 |
| 100 | 98.8% | 9,857 |
| 200 | 99.8% | 5,463 |
| 400 | 100.0% | 2,907 |

### HNSW (M=32, ef_construction=200)

| ef_search | Recall@10 | QPS |
|-----------|-----------|-----|
| 10 | 79.7% | 44,927 |
| 20 | 90.1% | 28,098 |
| 50 | 97.1% | 13,098 |
| 100 | 99.2% | 7,377 |
| 200 | 99.8% | 4,019 |
| 400 | 100.0% | 2,017 |

### Vamana (R=64, alpha=1.3, ef_construction=200)

| ef_search | Recall@10 | QPS |
|-----------|-----------|-----|
| 10 | 88.5% | 18,282 |
| 20 | 94.7% | 11,222 |
| 50 | 98.7% | 5,599 |
| 100 | 99.7% | 3,213 |
| 200 | 100.0% | 1,821 |
| 400 | 100.0% | 1,177 |

### NSW (M=16)

| ef_search | Recall@10 | QPS |
|-----------|-----------|-----|
| 10 | 79.1% | 43,447 |
| 20 | 88.2% | 26,043 |
| 50 | 95.4% | 9,939 |
| 100 | 97.7% | 4,813 |
| 200 | 98.8% | 2,322 |
| 400 | 99.2% | 1,288 |

### SQ4U (M=16, 4-bit quantized traversal + exact rerank)

| ef_search | Recall@10 | QPS |
|-----------|-----------|-----|
| 10 | 56.7% | 17,824 |
| 20 | 77.5% | 11,782 |
| 50 | 93.3% | 5,852 |
| 100 | 98.1% | 3,409 |
| 200 | 99.6% | 1,927 |
| 400 | 99.9% | 1,056 |

### DiskANN (R=64, alpha=1.3)

| ef_search | Recall@10 | QPS |
|-----------|-----------|-----|
| 10 | 88.7% | 17,555 |
| 20 | 94.8% | 10,936 |
| 50 | 98.8% | 5,505 |
| 100 | 99.7% | 3,183 |
| 200 | 100.0% | 1,817 |
| 400 | 100.0% | 1,029 |

### IVF-PQ (1024 clusters, 25 codebooks)

| nprobe | Recall@10 | QPS |
|--------|-----------|-----|
| 4 | 81.7% | 1,200 |
| 8 | 90.1% | 682 |
| 16 | 94.9% | 370 |
| 32 | 97.2% | 198 |
| 64 | 98.3% | 110 |
| 128 | 98.6% | 73 |
| 256 | 98.7% | 69 |

### IVF-AVQ (512 partitions, 5 codebooks)

| nprobe | Recall@10 | QPS |
|--------|-----------|-----|
| 4 | 47.1% | 3,478 |
| 8 | 59.1% | 2,155 |
| 16 | 72.7% | 1,267 |
| 32 | 82.2% | 709 |
| 64 | 87.3% | 389 |
| 128 | 90.3% | 206 |
| 256 | 90.9% | 194 |

## GIST-960 (1M vectors, 960-d, L2)

Ground truth: brute-force L2 k-NN.

### Summary

| Algorithm | Best Recall@10 | QPS at best | Notes |
|-----------|---------------|-------------|-------|
| HNSW (M=16) | 97.7% | 330 | |
| SQ4U | 95.9% | 175 | ~2x slower than plain HNSW even at d=960 |

SQ4U does not outperform plain HNSW at d=960. The reranking pass (exact f32 distance
on the candidate pool) dominates, negating the savings from quantized graph traversal.

### HNSW (M=16, ef_construction=200)

| ef_search | Recall@10 | QPS |
|-----------|-----------|-----|
| 10 | 48.3% | 2,905 |
| 20 | 62.5% | 3,175 |
| 50 | 78.3% | 1,729 |
| 100 | 87.9% | 1,047 |
| 200 | 94.0% | 593 |
| 400 | 97.7% | 330 |

### SQ4U (M=16, 4-bit quantized traversal + exact rerank)

| ef_search | Recall@10 | QPS |
|-----------|-----------|-----|
| 10 | 33.8% | 2,536 |
| 20 | 49.8% | 1,800 |
| 50 | 69.4% | 989 |
| 100 | 81.7% | 569 |
| 200 | 90.4% | 317 |
| 400 | 95.9% | 175 |

## Caveats

- GloVe-25 rankings differ from high-dimensional data. SQ4U and SymphonyQG
  are designed for d >= 128 but underperform at all tested dimensions.
- Build times and QPS are single-run wall-clock on a lightly loaded machine.
- GIST-960 numbers were collected with two benchmark processes sharing CPU
  (both equally affected; ratios are valid, absolute QPS are ~50% lower than
  solo runs would produce).
- IVF-PQ with 5 codebooks on 25-d is a known misconfiguration (too coarse).
- Some algorithms from prior runs (EMG, PiPNN, NSG, IVF-RaBitQ, etc.) are not
  yet re-benchmarked with the current optimized codebase.
