# vicinity

[![crates.io](https://img.shields.io/crates/v/vicinity.svg)](https://crates.io/crates/vicinity)
[![docs.rs](https://docs.rs/vicinity/badge.svg)](https://docs.rs/vicinity)

Approximate nearest-neighbor search.

## Install

Each algorithm is a separate feature. Enable what you need:

```toml
[dependencies]
vicinity = { version = "0.4", features = ["hnsw"] }          # graph index
# vicinity = { version = "0.4", features = ["ivf_pq"] }      # compressed index
# vicinity = { version = "0.4", features = ["nsw"] }         # flat graph
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

## Persistence

Save and load indexes with the `serde` feature:

```toml
[dependencies]
vicinity = { version = "0.4", features = ["hnsw", "serde"] }
```

```rust
// Save
index.save_to_file("my_index.json")?;

// Load
let index = HNSWIndex::load_from_file("my_index.json")?;
```

See [`examples/06_save_and_load.rs`](examples/06_save_and_load.rs) for a full example.

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
| HNSW | `hnsw` (default) | Best recall/QPS balance for in-memory search |
| SQ4U | `hnsw` + `sq4` | HNSW with 4-bit quantized graph traversal + exact rerank; benefits high-d data |
| SymphonyQG | `hnsw` + `ivf_rabitq` | HNSW with RaBitQ quantized graph traversal; cheap approximate beam search + exact rerank |
| NSW | `nsw` | Flat small-world graph; competitive with HNSW on high-d data |
| Vamana | `vamana` | DiskANN-style robust pruning; fast search, higher build time |
| NSG | `nsg` | Monotonic RNG pruning; 50K cap due to O(n) ensure_connectivity |
| EMG | `emg` | Multi-scale graph with alpha-pruning |
| FINGER | `finger` | Projection-based distance lower bounds for search pruning |
| PiPNN | `pipnn` | Partition-then-refine with HashPrune; reduces I/O during build |
| FreshGraph | `fresh_graph` | Streaming insert/delete with tombstones |
| IVF-PQ | `ivf_pq` | Compressed index; 32-64x less memory, lower recall |
| IVF-AVQ | `ivf_avq` | Anisotropic VQ + reranking; inner product search |
| IVF-RaBitQ | `ivf_rabitq` | RaBitQ binary quantization; provable error bounds |
| RpQuant | `rp_quant` | Random projection + scalar quantization |
| BinaryFlat | `binary_index` | 1-bit quantization + full-precision rerank |
| Curator | `curator` | K-means tree with per-label Bloom filters; low-selectivity filtered search |
| FilteredGraph | `filtered_graph` | Predicate-filtered graph search (AND/OR metadata filters) |
| ACORN | `hnsw` | Filtered HNSW search with subgraph sampling (SIGMOD 2024) |
| ESG | `esg` | Range-filtered search over numeric attributes |
| SparseMIPS | `sparse_mips` | Graph index for sparse vectors (SPLADE/BM25) |
| LEMUR | `lemur` | Late-interaction retrieval (multi-vector MIPS); inference-only |
| DiskANN | `diskann` | Vamana + SSD I/O layout; experimental |
| SNG | `sng` | Small navigable graph; O(n^2) construction |
| DEG | `hnsw` | Density-adaptive edge budgets (submodule of hnsw); O(n^2) |
| KD-Tree | `kdtree` | Exact NN; fast for d <= 20 |
| Ball Tree | `balltree` | Exact NN; slightly better than KD-Tree for d=20-50 |
| RP-Forest | `rptree` | Approximate; fast build, moderate recall |

Quantization: RaBitQ, SAQ (`quantization` feature, via `qntz` crate). PQ is part of `ivf_pq`.

See [docs.rs](https://docs.rs/vicinity) for the full API.

## Documentation

- **[User guide](docs/GUIDE.md)** -- quick start, distance metrics, LID, common pitfalls
- **[Benchmarks](docs/benchmark-results.md)** -- recall/QPS tables across datasets
- **[ANN landscape](docs/landscape.md)** -- algorithmic principles, math foundations, research context
- **[References](docs/references.md)** -- bibliography for every algorithm in the crate

## License

MIT OR Apache-2.0
