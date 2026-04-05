# vicinity

[![crates.io](https://img.shields.io/crates/v/vicinity.svg)](https://crates.io/crates/vicinity)
[![docs.rs](https://docs.rs/vicinity/badge.svg)](https://docs.rs/vicinity)

Approximate nearest-neighbor search.

## Install

Each algorithm is a separate feature. Enable what you need:

```toml
[dependencies]
vicinity = { version = "0.3", features = ["hnsw"] }          # graph index
# vicinity = { version = "0.3", features = ["ivf_pq"] }      # compressed index
# vicinity = { version = "0.3", features = ["nsw"] }         # flat graph
```

## Usage

### HNSW

High recall, in-memory. Best default choice.

```rust
use vicinity::hnsw::HNSWIndex;

let mut index = HNSWIndex::builder(128).m(16).ef_search(50).build()?;
index.add_slice(0, &[0.1; 128])?;
index.add_slice(1, &[0.2; 128])?;
index.build()?;

let results = index.search(&[0.1; 128], 5, 50)?;
// results: Vec<(doc_id, distance)>
```

### IVF-PQ

Compressed index. 32–64× less memory than HNSW, lower recall. Use for datasets that don't fit in RAM.

```rust
use vicinity::ivf_pq::{IVFPQIndex, IVFPQParams};

let params = IVFPQParams { num_clusters: 256, num_codebooks: 8, nprobe: 16, ..Default::default() };
let mut index = IVFPQIndex::new(128, params)?;
index.add_slice(0, &[0.1; 128])?;
index.add_slice(1, &[0.2; 128])?;
index.build()?;

let results = index.search(&[0.1; 128], 5)?;
```

## Benchmark

GloVe-25 (1.18M vectors, 25-d, cosine), Apple Silicon, single-threaded:

<p align="center">
  <img src="docs/plots/algorithm_comparison_glove-25-angular.png" width="680" alt="Recall vs QPS on GloVe-25" />
</p>

Full numbers in [`docs/benchmark-results.md`](docs/benchmark-results.md).

## Algorithms

Each algorithm has a named feature flag:

| Algorithm | Feature | Notes |
|-----------|---------|-------|
| HNSW | `hnsw` (default) | Best recall/QPS balance for in-memory search up to ~100M vectors |
| NSW | `nsw` | ~10× faster search than HNSW at the same ef; 1–2 pp lower recall ceiling |
| IVF-PQ | `ivf_pq` | ~25× less memory than HNSW; recall depends on codebooks — use num_codebooks ≥ dim/5 |
| Vamana | `vamana` | ~8.7× faster search than HNSW at same recall; higher build time than HNSW |
| DiskANN | `diskann` | Vamana + disk I/O layout; suited for datasets > available RAM |
| IVF-AVQ | `ivf_avq` | Anisotropic VQ + reranking; optimized for inner product search (MIPS) |
| SNG | `sng` | O(n²) construction; seconds at n=10K, hours at n=100K — not for large datasets |
| DEG | `hnsw` | Density-adaptive edge count; O(n²) construction — same scale limits as SNG |
| KD-Tree | `kdtree` | Exact; fast for d ≤ 20, recall degrades sharply above d=30 |
| Ball Tree | `balltree` | Exact; slightly better than KD-Tree for d=20–50 |
| RP-Forest | `rptree` | Approximate; fast build, moderate recall; good for high-d data |
| K-Means Tree | `kmeans_tree` | Hierarchical clustering index; suited for clustered or categorical data |

Quantization: PQ, RaBitQ, SQ8 (feature: `quantization`).

See [docs.rs](https://docs.rs/vicinity) for the full API.

## License

MIT OR Apache-2.0
