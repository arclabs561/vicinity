# Dataset Guide

Which datasets to use for benchmarking and evaluation.

## Quick Reference

| Use Case | Dataset | Size | Why |
|----------|---------|------|-----|
| **CI tests** | Bundled `quick_*` | ~1MB | 2K x 128, easy, instant |
| **Quick iteration** | Bundled `bench_*` | ~16MB | 10K x 384, medium, adversarial queries |
| **Stress test** | Bundled `hard_*` | ~31MB | 10K x 768, hard, never reaches 90% recall |
| **Standard benchmark** | SIFT-128 | 501MB | Industry standard, Euclidean |
| **Text embeddings** | GloVe-100 | 463MB | Word vectors, Angular |
| **High-dimensional** | GIST-960 | 3.6GB | Stress test, near modern dims |
| **Modern embeddings** | Generate from fastembed | varies | Match your production dims |

## Standard ANN Benchmark Datasets

From [ann-benchmarks.com](http://ann-benchmarks.com/). All include train/test split and ground truth.

### Recommended for Development

| Dataset | Dims | Vectors | Distance | Size | Notes |
|---------|------|---------|----------|------|-------|
| GloVe-25 | 25 | 1.18M | Angular | 121MB | Fastest download |
| GloVe-100 | 100 | 1.18M | Angular | 463MB | Good balance |
| SIFT-128 | 128 | 1M | Euclidean | 501MB | Standard benchmark |
| NYTimes-256 | 256 | 290K | Angular | 301MB | Text-like dims |

### Stress Testing

| Dataset | Dims | Vectors | Distance | Size | Notes |
|---------|------|---------|----------|------|-------|
| Fashion-MNIST | 784 | 60K | Euclidean | 217MB | High-dim images |
| GIST-960 | 960 | 1M | Euclidean | 3.6GB | Near modern dims |
| DEEP1B | 96 | 10M | Angular | 3.6GB | Large scale |

### Download

```sh
# GloVe (recommended starting point)
curl -o data/glove-100-angular.hdf5 http://ann-benchmarks.com/glove-100-angular.hdf5

# SIFT (Euclidean benchmark)
curl -o data/sift-128-euclidean.hdf5 http://ann-benchmarks.com/sift-128-euclidean.hdf5

# NYTimes (text-like)
curl -o data/nytimes-256-angular.hdf5 http://ann-benchmarks.com/nytimes-256-angular.hdf5
```

## Modern Embedding Dimensions

Standard benchmark datasets have lower dimensions than modern embedding models:

| Model | Dimensions | Notes |
|-------|------------|-------|
| OpenAI text-embedding-3-small | 1536 | Can reduce to 512-1024 |
| OpenAI text-embedding-3-large | 3072 | |
| Cohere embed-v3 | 1024 | |
| BGE-base | 768 | |
| all-MiniLM-L6-v2 | 384 | Efficient |
| GTE-small | 384 | |

For production benchmarking at these dimensions, generate your own dataset:

```rust
// Generate embeddings with fastembed (or any embedding model)
use fastembed::{EmbeddingModel, TextEmbedding};

let model = TextEmbedding::try_new(Default::default())?;
let texts = load_your_corpus(); // your actual data
let embeddings = model.embed(texts, None)?;
// Save and use for benchmarking
```

## Synthetic vs Real Data

**Synthetic (bundled):**
- Clustered Gaussian vectors
- Ground truth computed exactly
- Good for: algorithm correctness, quick iteration, CI/CD

**Real (ann-benchmarks):**
- Actual word/image embeddings
- Ground truth from brute force
- Good for: performance comparison, publishing results

**Your data:**
- Matches production characteristics
- Ground truth: sample + brute force
- Good for: production decisions

## Dataset Format

### Binary (our format)

```
VEC1 (4 bytes) + n (u32) + dim (u32) + data (n * dim * f32)
```

Simple, fast to load, no dependencies.

### HDF5 (ann-benchmarks)

Standard format with train/test/neighbors groups. Requires hdf5 feature.

```rust
// Enable HDF5 support
// Cargo.toml: vicinity = { features = ["hdf5"] }
```

## Recommendations by Task

### Algorithm Development
1. Start with bundled `data/sample/bench_*` (10K x 384)
2. Graduate to GloVe-100 or SIFT-128
3. Stress test with GIST-960

### Production Evaluation
1. Generate embeddings from your actual corpus
2. Sample 10K queries from real usage
3. Compute ground truth on sample

### Publishing Results
1. Use standard ann-benchmarks datasets
2. Report recall@k vs QPS curves
3. Include build time and memory usage
