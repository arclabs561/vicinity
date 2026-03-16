# Benchmark Results

Machine: Apple Silicon (Darwin 25.3.0), single-threaded, `--release`.
SIMD: `innr` (pure Rust SIMD, default feature).

## GloVe-25 (1.2M vectors, 25 dims, angular/cosine)

Dataset: ann-benchmarks.com `glove-25-angular`.
Ground truth: brute-force cosine k-NN on L2-normalized vectors.

### HNSW (M=16, ef_construction=200)

Build: 270s (4,377 vectors/sec)

| ef_search | Recall@10 | Latency | QPS |
|-----------|-----------|---------|-----|
| 10 | 58.0% | 689us | 1,451 |
| 20 | 70.4% | 709us | 1,411 |
| 50 | 83.1% | 744us | 1,344 |
| 100 | 89.9% | 809us | 1,236 |
| 200 | 94.3% | 962us | 1,040 |
| 400 | 96.8% | 1,231us | 812 |

### HNSW (M=32, ef_construction=200)

Build: 505s (2,343 vectors/sec)

| ef_search | Recall@10 | Latency | QPS |
|-----------|-----------|---------|-----|
| 10 | 72.8% | 706us | 1,416 |
| 20 | 83.1% | 729us | 1,372 |
| 50 | 92.1% | 786us | 1,273 |
| 100 | 96.2% | 886us | 1,128 |
| 200 | 98.3% | 1,060us | 943 |
| 400 | 99.2% | 1,433us | 698 |

### Context

- **hnswlib (C++)**: ~95% recall @ ~5K QPS on same dataset (AVX2, optimized C).
  vicinity is ~4-5x slower, expected for pure Rust without hand-tuned intrinsics.
- **SimSIMD feature**: enabling `simsimd` instead of `innr` should improve distance
  computation throughput significantly (up to 200x for the distance kernel), though
  the overall improvement depends on the graph traversal overhead ratio.

### Synthetic (03_quick_benchmark, 10K vectors, 384 dims)

| ef_search | Recall@10 | Latency | QPS |
|-----------|-----------|---------|-----|
| 20 | 85.7% | 45us | 22,202 |
| 50 | 95.6% | 72us | 13,942 |
| 100 | 98.8% | 100us | 9,994 |
| 200 | 99.6% | 137us | 7,280 |
