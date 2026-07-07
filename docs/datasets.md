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
| **Quantization checks** | GloVe-200 | 918MB | Higher dimension, Angular |
| **High-dimensional** | GIST-960 | 3.6GB | Stress test, near current embedding dims |
| **Production embeddings** | Generate from fastembed | varies | Match your production dims |

## Standard ANN Benchmark Datasets

From [ann-benchmarks.com](http://ann-benchmarks.com/). All include train/test split and ground truth.

### Recommended for Development

| Dataset | Dims | Vectors | Distance | Size | Notes |
|---------|------|---------|----------|------|-------|
| GloVe-25 | 25 | 1.18M | Angular | 121MB | Fastest download |
| GloVe-50 | 50 | 1.18M | Angular | 235MB | Lower-mid text dimension |
| GloVe-100 | 100 | 1.18M | Angular | 463MB | Good balance |
| GloVe-200 | 200 | 1.18M | Angular | 918MB | Better for quantized variants |
| SIFT-128 | 128 | 1M | Euclidean | 501MB | Standard benchmark |
| NYTimes-256 | 256 | 290K | Angular | 301MB | Text-like dims |

These datasets are small enough for normal in-memory sweeps on a development
machine. Use them to compare algorithm families at fixed recall and to validate
resume behavior, latency tails, and build-time reporting.

### Stress Testing

| Dataset | Dims | Vectors | Distance | Size | Notes |
|---------|------|---------|----------|------|-------|
| Fashion-MNIST | 784 | 60K | Euclidean | 217MB | High-dim images |
| GIST-960 | 960 | 1M | Euclidean | 3.6GB | Near current embedding dims |
| Deep Image | 96 | 10M | Angular | 3.6GB | Large scale |

Stress datasets should be reported with explicit storage modes. An in-memory
row answers a different question from a file or mmap row: in-memory QPS measures
the search algorithm and memory layout after data is resident, while file/mmap
rows include open/load behavior and can depend on the OS page cache. Do not
collapse these rows into a single leaderboard number.

### Download and Convert

```sh
# List available datasets
uv run scripts/download_ann_benchmarks.py --list

# GloVe (recommended starting point)
uv run scripts/download_ann_benchmarks.py glove-100-angular

# SIFT (Euclidean benchmark)
uv run scripts/download_ann_benchmarks.py sift-128-euclidean

# Verify every configured source HDF5 without converting yet
uv run scripts/download_ann_benchmarks.py --all --download-only
```

The script writes `data/ann-benchmarks/<dataset>/{train,test,neighbors}.bin`
plus `dataset.json`. Re-running it is idempotent: existing converted files are
reused only when their headers, byte lengths, and manifest match the current
conversion settings. Cached or downloaded HDF5 files are checked against the
expected byte size; datasets with a verified source hash are also checked against
the pinned SHA-256. Use `--force` to rebuild the `.bin` files from the cached
HDF5, or `--redownload` to replace the cached HDF5 before conversion. If you
already have converted `.bin` files from an older checkout but no manifest, use
`--adopt-existing` to write `dataset.json` after validating the cached HDF5 size,
source hash when pinned, and binary headers.
Use `--all` to apply the same idempotent fetch or conversion flow to every
configured dataset.

## Current Embedding Dimensions

Standard benchmark datasets have lower dimensions than current embedding models:

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

Standard format with train/test/neighbors groups. vicinity does not include
an HDF5 reader; convert downloaded `.hdf5` files to the binary format first
with `scripts/download_ann_benchmarks.py`:

```sh
uv run scripts/download_ann_benchmarks.py glove-25-angular
# writes data/ann-benchmarks/glove-25-angular/{train,test,neighbors}.bin

uv run scripts/download_ann_benchmarks.py glove-25-angular --force
# rebuilds the converted binary files from the cached HDF5

uv run scripts/download_ann_benchmarks.py glove-25-angular --adopt-existing
# writes dataset.json for legacy converted files without reconverting
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
