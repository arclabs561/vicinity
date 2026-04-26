# Examples

Organized by learning path and use case.

## Quick Start (Toy Examples)

Start here. These work immediately with synthetic data.

| Example | Lines | What It Teaches |
|---------|-------|-----------------|
| `01_basic_search` | 63 | Minimal HNSW: add vectors, search |
| `02_measure_recall` | 91 | How to validate an ANN index |
| `03_quick_benchmark` | 190 | Benchmark with bundled data (no downloads) |

```sh
cargo run --example 01_basic_search --release
cargo run --example 02_measure_recall --release
cargo run --example 03_quick_benchmark --release                       # bench: 10K x 384
VICINITY_DATASET=quick cargo run --example 03_quick_benchmark --release     # CI: 2K x 128
```

## Educational (Motivated Toy)

Realistic scenarios with synthetic data. Demonstrate when/why to use each algorithm.

| Example | Lines | Algorithm | Teaches |
|---------|-------|-----------|---------|
| `semantic_search_demo` | 334 | HNSW | Document search with categories |
| `ivf_pq_demo` | 321 | IVF-PQ | Billion-scale with compression |
| `lid_demo` | 342 | LID | Intrinsic dimensionality estimation |
| `lid_outlier_detection` | 186 | LID | Anomaly detection via LID |
| `rabitq_demo` | 294 | RaBitQ | Randomized binary quantization |

```sh
cargo run --example semantic_search_demo --release
cargo run --example ivf_pq_demo --release --features ivf_pq
```

## Benchmarks (Real Data)

Compare against standard ANN benchmark datasets from [ann-benchmarks.com](http://ann-benchmarks.com/).

### Bundled Data (No Downloads)

| Dataset | Vectors | Dims | Size | Difficulty |
|---------|---------|------|------|------------|
| `quick` | 2K | 128 | ~1MB | Easy (CI) |
| `bench` | 10K | 384 | ~16MB | Medium |
| `hard` | 10K | 768 | ~31MB | Hard (realistic: topics + duplicates + hard-tail queries) |

Difficulty progression based on He et al. "On the Difficulty of Nearest Neighbor Search" (ICML 2012):
- **quick**: Well-separated clusters, standard queries. Reaches 99%+ recall.
- **bench**: Moderate overlap, adversarial queries. Reaches ~93% at ef=200.
- **hard**: Anisotropic topic mixture + near-duplicates + a small hard query tail. Expect lower recall at the same ef.

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

For serious benchmarking, download from [ann-benchmarks.com](http://ann-benchmarks.com/):

| Dataset | Dims | Best For | Why |
|---------|------|----------|-----|
| **GloVe-25** | 25 | Quick iteration | Smallest, fast downloads |
| **GloVe-100** | 100 | Realistic text | Common word embedding dim |
| **SIFT-128** | 128 | Euclidean baseline | Standard image features |
| **NYTimes-256** | 256 | Text embeddings | Closer to modern dims |
| **Fashion-MNIST** | 784 | High-dim | Tests curse of dimensionality |
| **GIST-960** | 960 | Stress test | Near modern embedding dims |

Modern embedding models (OpenAI, Cohere) use 768-3072 dims. The ann-benchmarks
datasets are smaller but still useful for algorithm comparison.

## Advanced (Research Implementations)

Recent research algorithms. Useful for understanding ongoing work in graph-based ANN.

| Example | Algorithm | Paper |
|---------|-----------|-------|
| `dual_branch_demo` | Dual-Branch HNSW | LID-based insertion |
| `dual_branch_hnsw_demo` | Dual-Branch variant | Skip bridges |
| `evoc_demo` | EVōC | Hierarchical clustering |

These are more complex and require reading the accompanying paper.

## Choosing an Algorithm

```
Do you have < 10K vectors?
 └─> Brute force (no index needed)

Do you need streaming inserts with theoretical guarantees?
 └─> Hash/LSH-style approaches (locality-sensitive hashing)

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
