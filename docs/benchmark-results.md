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

### DiskANN (R=64, α=1.3, ef_construction=200)

Build: ~1853s (same as Vamana; DiskANN is Vamana + disk I/O layout)

| ef_search | Recall@10 | QPS |
|-----------|-----------|-----|
| 10 | 86.5% | 10,874 |
| 20 | 94.2% | 6,883 |
| 50 | 98.7% | 3,400 |
| 100 | 99.7% | 1,904 |
| 200 | 100.0% | 1,044 |
| 400 | 100.0% | 582 |

DiskANN and Vamana use the same graph (Vamana construction). DiskANN adds a disk
layout for datasets larger than available RAM; on this in-memory benchmark the graph
is fully resident, so QPS is slightly lower than Vamana (~15%) due to the disk I/O
abstraction layer. Recall trajectory is essentially identical.

## GloVe-25 — Partition-based indexes

### IVF-AVQ (512 partitions, 5 codebooks, reorder=500)

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

### IVF-PQ (1024 clusters, 5 codebooks)

Build: ~1096s

| nprobe | Recall@10 | QPS |
|--------|-----------|-----|
| 4 | 41.3% | 7,238 |
| 8 | 43.4% | 3,589 |
| 16 | 44.5% | 1,811 |
| 32 | 44.9% | 937 |
| 64 | 45.0% | 451 |
| 128 | 45.1% | 240 |
| 256 | 45.1% | 106 |

Recall caps at ~45%: 5 codebooks over 25 dims = 5-d subspaces, which is too coarse
for 25-d data. The ceiling is not a bug — it is inherent to quantization granularity.

### IVF-PQ (1024 clusters, 25 codebooks — 1-d subspaces, equivalent to SQ8)

Build: ~5520s

| nprobe | Recall@10 | QPS |
|--------|-----------|-----|
| 4 | 76.3% | 2,920 |
| 8 | 86.2% | 1,484 |
| 16 | 92.6% | 738 |
| 32 | 96.2% | 369 |
| 64 | 97.9% | 188 |
| 128 | 98.6% | 99 |

With 25 codebooks (1-d subspaces, equivalent to SQ8 scalar quantization), recall
reaches 98.6% at the cost of build time (~5× longer) and memory (~5× more for codes).
This validates that the 45% cb5 ceiling was entirely due to quantization granularity,
not a structural limitation of IVF-PQ.

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
- **DiskANN vs Vamana**: same recall trajectory (both use the Vamana graph); DiskANN
  is ~15% slower QPS due to the disk I/O layout abstraction. On datasets that fit in
  RAM, Vamana is the better choice; DiskANN's advantage is for datasets > available RAM.
