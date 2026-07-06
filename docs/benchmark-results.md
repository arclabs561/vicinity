# Benchmark Results

Historical benchmark tables for standard ANN datasets.

Current benchmark runs should use `examples/ann_benchmark.rs`, which writes one
`_meta` JSON line with the dataset, metric, actual `rustc --version`, MSRV, and
crate version, followed by one JSON line per measurement with build time, RSS,
storage mode, cache state, and p50/p95/p99 latency. Use `--resume` to skip
completed rows and `--fresh` to recreate the result file.

QPS below is sequential single-query throughput (queries / wall-clock seconds)
from older single-run `--release` measurements on Apple Silicon. The exact CPU
model, thread count, frequency, and thermal state were not recorded for these
older rows, so treat absolute QPS as historical context. Re-run the harness on
your target machine before comparing against external papers.

Recommended current commands:

```bash
# Full single-query curve with latency tails.
cargo run --example ann_benchmark --release --features hnsw,ivf_pq,ivf_avq -- \
  data/ann-benchmarks/glove-25-angular \
  --algo hnsw --algo ivfpq --algo ivf_avq --json --fresh

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
```

These commands intentionally separate low-recall, high-throughput operating
points from high-recall runs. External comparisons should use a recall/QPS
curve, not only the `Recall@10 = 100%` row.

Persistence and storage-mode benchmark rows follow `docs/persistence.md`.
Methods with both in-memory and file-backed search should report separate
`storage_mode=in_memory`, `storage_mode=file`, and, when implemented,
`storage_mode=mmap` rows. File and mmap rows should also report `load_time_s`
and `index_bytes` when the runner opens a saved index.

## GloVe-25 (1.18M vectors, 25-d, angular distance)

Ground truth: brute-force k-NN on L2-normalized vectors (angular ≡ cosine for unit vectors).

### Summary

| Algorithm | Best Recall@10 | QPS at best | Notes |
|-----------|---------------|-------------|-------|
| HNSW (M=16) | 100.0% | 2,857 | Default choice |
| HNSW (M=32) | 100.0% | 2,017 | Higher memory, marginal recall gain |
| Vamana | 100.0% | 1,177 | Slow build (~2000s) |
| DiskANN | 100.0% | 1,029 | Vamana + I/O layout |
| SQ4U | 99.9% | 1,056 | 4-bit quantized HNSW; ~3x slower than plain HNSW at d=25 |
| NSW | 99.2% | 1,288 | |
| IVF-PQ (cb=25) | 98.7% | 69 | 25 codebooks on 25-d (1-d subspaces) |
| IVF-AVQ | 90.9% | 194 | ScaNN-style anisotropic VQ |
| RP-Forest | 58.5% | 4,221 | Fast build, moderate recall |
| IVF-PQ (cb=5) | 45.1% | 262 | 5 codebooks on 25-d (too coarse) |
| KD-Tree | 100.0% | 22 | Exact; too slow at 1M+ |
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
