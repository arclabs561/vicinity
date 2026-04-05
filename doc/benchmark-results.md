# Benchmark Results

Machine: Apple Silicon (M-series), single-threaded, `--release`.
Dataset: GloVe-25 (1.18M vectors, 25-d, cosine / angular), from ann-benchmarks.com.
Ground truth: brute-force cosine k-NN on L2-normalized vectors.
SIMD: `innr` (pure Rust SIMD, default feature).

## GloVe-25 — Graph indexes

### HNSW (M=16, m_max=32, ef_construction=200)

Build: ~372s

| ef_search | Recall@10 | QPS |
|-----------|-----------|-----|
| 10 | 77.8% | 1,480 |
| 20 | 88.0% | 1,456 |
| 50 | 96.0% | 1,392 |
| 100 | 98.8% | 1,297 |
| 200 | 99.8% | 1,152 |
| 400 | 100.0% | 959 |

### HNSW (M=32, m_max=64, ef_construction=200)

Build: ~1159s

| ef_search | Recall@10 | QPS |
|-----------|-----------|-----|
| 10 | 87.4% | 1,461 |
| 20 | 94.1% | 1,427 |
| 50 | 98.6% | 1,334 |
| 100 | 99.6% | 1,214 |
| 200 | 99.9% | 1,044 |
| 400 | 100.0% | 831 |

### NSW (M=16, ef_construction=32)

Build: fast (no hierarchy)

| ef_search | Recall@10 | QPS |
|-----------|-----------|-----|
| 10 | 66.7% | 18,444 |
| 20 | 79.0% | 11,972 |
| 50 | 90.1% | 6,380 |
| 100 | 95.2% | 3,930 |
| 200 | 97.9% | 2,290 |
| 400 | 99.2% | 1,288 |

### Vamana (R=64, α=1.3, ef_construction=200)

Build: ~1853s

| ef_search | Recall@10 | QPS |
|-----------|-----------|-----|
| 10 | 86.4% | 12,763 |
| 20 | 93.1% | 7,962 |
| 50 | 97.7% | 4,213 |
| 100 | 99.1% | 2,401 |
| 200 | 99.7% | 1,361 |
| 400 | 99.9% | 751 |

Vamana at ef=10 reaches 86.4% recall at 12,763 QPS — ~8.7× faster than HNSW M=32
(which achieves 87.4% recall at 1,461 QPS). Trade-off: HNSW has a flatter latency
curve (less degradation at lower ef). Vamana's recall ceiling matches HNSW at ef≥200.

## GloVe-25 — Partition-based indexes

### ScaNN (512 partitions, 5 codebooks, reorder=500)

Build: ~708s

| nprobe | Recall@10 | QPS |
|--------|-----------|-----|
| 4 | 77.5% | 5,714 |
| 8 | 84.7% | 3,054 |
| 16 | 88.5% | 1,559 |
| 32 | 90.3% | 792 |
| 64 | 90.8% | 396 |
| 128 | 90.9% | 194 |
| 256 | 90.9% | 87 |

Recall ceiling of ~91% is a known limitation of PQ-based re-ranking on 25-d data:
the PQ residual quantization error is significant relative to the vector dimension.

### IVF-PQ (1024 clusters, 5 codebooks, 256 codebook size)

| nprobe | Recall@10 | QPS |
|--------|-----------|-----|
| 4 | 41.3% | 8,526 |
| 8 | 43.4% | 4,208 |
| 16 | 44.4% | 2,216 |
| 32 | 44.8% | 1,094 |
| 64 | 45.0% | 546 |
| 128 | 45.0% | 273 |
| 256 | 45.0% | 135 |

Recall caps at ~45% on this dataset. Same PQ dimension issue as ScaNN: 5 codebooks
over 25 dims = 5-d subspaces, which is marginal. Increasing codebooks (e.g., 25) or
clusters would improve recall at the cost of build time and memory.

## Brute Force

Exact k-NN via exhaustive cosine search: 100% recall @ 42 QPS. Baseline.

## Context

- **hnswlib (C++)**: ~95% recall @ ~5K QPS on same dataset (AVX2, optimized C++).
  vicinity is ~3-4× slower on graph traversal — expected for pure Rust without
  hand-tuned AVX2. The `simsimd` feature is not yet benchmarked here.
- **Vamana vs HNSW**: at the same recall level (~87%), Vamana is ~8.7× faster.
  Vamana's search is a single greedy beam (no hierarchy traversal), which amortizes
  better at low ef. HNSW has a smaller QPS range across the ef sweep.
- **NSW speed**: NSW's flat graph search is significantly faster than HNSW at the same
  ef, consistent with Munyampirwa et al. (2024) (arXiv:2412.01940). The recall ceiling
  is ~1-2 pp lower than HNSW at the same ef.
