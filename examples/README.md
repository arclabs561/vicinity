# Examples

Run from the repository root. Use `--release` for benchmark and numeric
examples unless a command says otherwise.

## Core

Synthetic-data examples that run without downloads.

| Example | Lines | Covers |
|---------|-------|-----------------|
| `01_basic_search` | 62 | Minimal HNSW: add vectors, search |
| `02_measure_recall` | 91 | How to validate an ANN index |
| `03_quick_benchmark` | 345 | Benchmark with bundled data (no downloads) |
| `05_normalization_matters` | 145 | Cosine input contract and `auto_normalize(true)` |

```sh
cargo run --example 01_basic_search --release
cargo run --example 02_measure_recall --release
cargo run --example 03_quick_benchmark --release                       # bench: 10K x 384
VICINITY_DATASET=quick cargo run --example 03_quick_benchmark --release     # CI: 2K x 128
cargo run --example 05_normalization_matters --release
```

## Synthetic Workflows

Synthetic workloads for algorithm behavior and parameter effects.

| Example | Lines | Algorithm | Covers |
|---------|-------|-----------|---------|
| `semantic_search_demo` | 334 | HNSW | Document search with categories |
| `acorn_selectivity` | 802 | ACORN | Filtered-search selectivity sweep |
| `ivf_pq_demo` | 306 | IVF-PQ | Compressed inverted-file search |
| `lid_demo` | 344 | LID | Intrinsic dimensionality estimation |
| `rabitq_demo` | 261 | RaBitQ | Randomized binary quantization |
| `sparse_mips_benchmark` | 191 | SparseMIPS | Sparse MIPS smoke benchmark over SPV1 data |
| `symphonyqg_demo` | 246 | HNSW + RaBitQ | Quantized graph traversal pattern |

```sh
cargo run --example semantic_search_demo --release
cargo run --example acorn_selectivity --release --features hnsw
cargo run --example acorn_selectivity --release --features hnsw,filtered_graph,range_filtered,curator -- \
  --json --fresh
cargo run --example acorn_selectivity --release --features hnsw,filtered_graph,range_filtered,curator -- \
  --json --fresh --ef-search 400 --acorn-max-two-hop-neighbors 64 --fallback-selectivity-threshold 0.02
uv run scripts/summarize_selectivity_results.py data/ann-benchmarks/results/acorn-selectivity-*.jsonl
cargo run --example ivf_pq_demo --release --features ivf_pq
cargo run --example lid_demo --release
cargo run --example rabitq_demo --release --features "rabitq,hnsw,quantization"
uv run scripts/generate_sparse_mips_smoke_data.py data/sparse-mips/smoke
cargo run --example sparse_mips_benchmark --release --features sparse_mips -- \
  data/sparse-mips/smoke
cargo run --example symphonyqg_demo --release --features "hnsw,rabitq,quantization"
```

## Benchmarks (Real Data)

Compare against standard ANN benchmark datasets from [ann-benchmarks.com](http://ann-benchmarks.com/).

### Bundled Data (No Downloads)

| Dataset | Vectors | Dims | Size | Difficulty |
|---------|---------|------|------|------------|
| `quick` | 2K | 128 | ~1MB | Easy (CI) |
| `bench` | 10K | 384 | ~16MB | Medium |
| `hard` | 10K | 768 | ~31MB | Stress case: topics, duplicates, hard-tail queries |

Difficulty progression based on He et al. "On the Difficulty of Nearest Neighbor Search" (ICML 2012):
- **quick**: Well-separated clusters, standard queries. Reaches 99%+ recall.
- **bench**: Moderate overlap, adversarial queries. Reaches ~93% at ef=200.
- **hard**: Anisotropic topic mixture, near-duplicates, and a small hard query tail. Lower recall at the same ef.

```sh
cargo run --example 03_quick_benchmark --release                      # bench (default)
VICINITY_DATASET=quick cargo run --example 03_quick_benchmark --release    # CI
VICINITY_DATASET=hard cargo run --example 03_quick_benchmark --release     # stress test
```

### Real ANN Benchmark Datasets

| Example | Dataset | Vectors | Dims | Distance | Size |
|---------|---------|---------|------|----------|------|
| `glove_benchmark` | GloVe-25 | 1.18M | 25 | Angular | 121MB |
| `sift_benchmark` | SIFT-128 | 1M | 128 | Euclidean | 501MB |
| `hnsw_benchmark` | Synthetic | config | config | config | - |

Both have synthetic fallbacks if data isn't available.

```sh
# Real datasets (requires download)
cargo run --example glove_benchmark --release -- --full
cargo run --example sift_benchmark --release --features hnsw
```

### Standard ANN Benchmark Datasets

For benchmark-compatible runs, download from [ann-benchmarks.com](http://ann-benchmarks.com/):

| Dataset | Dims | Use | Notes |
|---------|------|----------|-----|
| **GloVe-25** | 25 | Quick iteration | Smallest, fast downloads |
| **GloVe-50** | 50 | Dimensionality sweep | Lower-mid text dimension |
| **GloVe-100** | 100 | Text embeddings | Common word embedding dim |
| **GloVe-200** | 200 | Quantization checks | Better for quantized variants |
| **SIFT-128** | 128 | Euclidean baseline | Standard image features |
| **NYTimes-256** | 256 | Text embeddings | Higher-dimensional text dataset |
| **Fashion-MNIST** | 784 | High-dim | Tests curse of dimensionality |
| **GIST-960** | 960 | High-dim stress test | Larger dense descriptors |
| **Deep Image** | 96 | Large-scale graph/IVF | 10M learned image vectors |

Many embedding workloads use 768-3072 dims. The ann-benchmarks datasets
are smaller, but they remain useful for repeatable comparisons.

```sh
# Standard JSONL benchmark output
cargo run --example ann_benchmark --release --features hnsw,ivf_pq,ivf_avq -- \
  data/ann-benchmarks/glove-25-angular --algo hnsw --algo ivfpq --algo ivf_avq --json

# HNSW persisted snapshot-load rows
cargo run --example ann_benchmark --release --features hnsw,serde -- \
  data/ann-benchmarks/glove-25-angular --algo hnsw --snapshot-load --json

# DiskANN in-memory, file, and mmap rows from the same build
cargo run --example ann_benchmark --release --features hnsw,diskann -- \
  data/ann-benchmarks/glove-25-angular --algo diskann --json

# IVF-PQ cluster/codebook/rerank sweep
# Add --snapshot-load to also measure saved-and-reopened raw and reranked rows.
cargo run --example ann_benchmark --release --features hnsw,ivf_pq -- \
  data/ann-benchmarks/glove-25-angular --algo ivfpq \
  --pq-clusters 1024 --pq-codebooks 25 --pq-codebook-size 256 \
  --pq-training-sample-size 100000 --pq-kmeans-max-iter 20 \
  --pq-nprobes 16,32,64,128,256 --pq-rerank-pools 500,5000,20000 --json

# FreshGraph delete/insert churn, scored against a live active-set oracle
cargo run --example ann_benchmark --release --features hnsw,fresh_graph -- \
  data/ann-benchmarks/glove-25-angular --algo fresh_graph_churn --json

# IP-DiskANN-style in-place graph search and delete/insert churn
cargo run --example ann_benchmark --release --features hnsw -- \
  data/ann-benchmarks/glove-25-angular --algo inplace --algo inplace_churn --json

# LSM-tiered streaming churn, scored against a live active-set oracle
cargo run --example ann_benchmark --release --features hnsw -- \
  data/ann-benchmarks/glove-25-angular --algo lsm_churn --json

# Classic tree baselines for comparison
cargo run --example ann_benchmark --release --features kdtree,balltree,rptree,kmeans_tree -- \
  data/ann-benchmarks/glove-25-angular \
  --algo kdtree --algo balltree --algo rptree --algo rp_forest --algo kmeans_tree \
  --tree-leaf-sizes 10,50 --rp-num-trees 10,20 --kmeans-clusters 8,16 \
  --snapshot-load --json

# Lower-recall HNSW accelerator comparison
cargo run --example ann_benchmark --release --features hnsw,sq4,sq8,ivf_rabitq,finger -- \
  data/ann-benchmarks/glove-25-angular \
  --algo hnsw --algo adsampling --algo hnsw_prt --algo sq4u --algo sq8u --algo symphony_qg --json
```

At 25 dimensions, ADSampling, FINGER, SymphonyQG, and SQ4U are diagnostic
comparisons rather than optimization targets. Re-run on GloVe-100+ or GIST-960
before treating those rows as architecture guidance.

## Research Variants

Graph-based ANN variants and clustering examples.

| Example | Algorithm | Paper |
|---------|-----------|-------|
| `dual_branch_demo` | Dual-Branch HNSW | LID-based insertion |
| `dual_branch_hnsw_demo` | Dual-Branch variant | Skip bridges |
| `evoc_demo` | EVōC | Hierarchical clustering |

Read the linked paper before treating output as a recommendation.

## Choosing an Algorithm

```
Do you have < 10K vectors?
 └─> Brute force (no index needed)

Do you need streaming inserts/deletes?
 └─> In-place graph or LSM-tiered streaming rows in `ann_benchmark`

Are you memory-constrained (> 1M vectors)?
 └─> IVF-PQ (see ivf_pq_demo)

Default choice:
 └─> HNSW (see 01_basic_search, semantic_search_demo)
```

## Running All Examples

```sh
# Quick smoke test of all algorithms
for ex in 01_basic_search 02_measure_recall semantic_search_demo; do
    cargo run --example $ex --release
done
```
