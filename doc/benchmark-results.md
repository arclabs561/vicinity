# Benchmark Results

Machine: Apple Silicon (Darwin 25.3.0), single-threaded, `--release`.
SIMD: `innr` (pure Rust SIMD, default feature).

## GloVe-25 (1.2M vectors, 25 dims, angular/cosine)

Dataset: ann-benchmarks.com `glove-25-angular`.
Ground truth: brute-force cosine k-NN on L2-normalized vectors.

### HNSW (M=16, m_max=32, ef_construction=200)

| ef_search | Recall@10 | QPS |
|-----------|-----------|-----|
| 10 | 63.0% | 1,496 |
| 20 | 75.8% | 1,473 |
| 50 | 88.4% | 1,409 |
| 100 | 94.3% | 1,326 |
| 200 | 97.6% | 1,189 |
| 400 | 99.1% | 992 |

Previous results (m_max=16, before fix): 58-97% recall. The m_max=32
correction (paper's m_max0 = 2*M) improved recall by 3-5 percentage
points across all ef values.

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
  computation throughput (not yet benchmarked for vicinity; improvement depends on
  the graph traversal overhead ratio vs distance computation time).

### Synthetic (03_quick_benchmark, 10K vectors, 384 dims)

| ef_search | Recall@10 | Latency | QPS |
|-----------|-----------|---------|-----|
| 20 | 85.7% | 45us | 22,202 |
| 50 | 95.6% | 72us | 13,942 |
| 100 | 98.8% | 100us | 9,994 |
| 200 | 99.6% | 137us | 7,280 |
