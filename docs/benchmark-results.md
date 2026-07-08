# Benchmark Results

Historical benchmark tables for standard ANN datasets.

The checked-in `docs/*.jsonl` files are legacy result artifacts from this older
schema. They do not include `_meta`, `storage_mode`, `cache_state`,
`load_time_s`, or `index_bytes`, so use them only as historical in-memory
context. Storage-aware coverage should be generated with the current harness
commands below.

Current benchmark runs should use `examples/ann_benchmark.rs`, which writes one
`_meta` JSON line with the dataset, metric, actual `rustc --version`, MSRV, and
crate version, query limit, and measured query count, followed by one JSON line
per measurement with build time, RSS, storage mode, cache state, and
p50/p95/p99 latency. Persisted, file, mmap, and segmented-store rows also
report `load_time_s` and `index_bytes` when the runner can measure them. Use
`--resume` to skip completed rows and `--fresh` to recreate the result file.

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
# distances, so it is cheap enough to run before comparing benchmark curves.
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

### DiskANN File And Mmap Search

The search-only DiskANN benchmark isolates in-memory, file, and mmap search
from construction:

```bash
cargo bench --bench diskann_search --no-default-features --features diskann -- \
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
  --features diskann -- diskann_search_only/file_ef75 --profile-time 10
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
  cargo bench --bench diskann_search --no-default-features --features diskann -- \
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

Not every implemented module should produce a dense ANN row:

| Module | Why it is separate | Honest eval direction |
| --- | --- | --- |
| SparseMIPS | Requires sparse vectors, not dense ann-benchmarks `f32` arrays | Add a SPLADE/BM25 sparse dataset harness before reporting QPS/recall |
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
