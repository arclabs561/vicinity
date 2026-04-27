# vicinity

[![crates.io](https://img.shields.io/crates/v/vicinity.svg)](https://crates.io/crates/vicinity)
[![docs.rs](https://docs.rs/vicinity/badge.svg)](https://docs.rs/vicinity)

Approximate nearest-neighbor search. Each algorithm is a separate
feature flag. Depend on only what you use.

Default distance is cosine; most graph indices L2-normalize on insert
and operate on the cosine-equivalent unit sphere (so angular and inner
product over normalized vectors fall out for free). True L2 distance is
natively wired only in `ivf_avq` (which targets MIPS). The
`DistanceMetric` enum in `distance.rs` is consumed by evaluation
utilities and brute-force comparison; per-index metric selection is
algorithm-specific.

## Which index?

```
Which index?
├── General purpose: HNSW (default)
├── Memory constrained: HNSW + RaBitQ (SymphonyQG) or IVF-PQ
├── Disk-backed / large scale: DiskANN (experimental; file-based, mmap planned)
├── Streaming insert/delete: FreshGraph (per-op latency) or LsmIndex (write throughput)
├── Filtered search: ACORN (metadata filters) or Curator (label filters)
├── Batch/static: IVF-PQ or IVF-AVQ
└── Sparse vectors: SparseMIPS
```

For high-dimensional data (d ≥ 256), prefer SQ4U or SymphonyQG over plain HNSW. Quantized
graph traversal reduces distance computation cost. At low dimensions (d ≤ 25), plain HNSW
wins; quantization overhead outweighs savings.

## Install

Each algorithm is a separate feature. Enable what you need:

```toml
[dependencies]
vicinity = { version = "0.7", features = ["hnsw"] }          # graph index
# vicinity = { version = "0.7", features = ["ivf_pq"] }      # compressed index
# vicinity = { version = "0.7", features = ["nsw"] }         # flat graph
```

## Usage

### HNSW

High recall, in-memory. Default index.

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
vicinity = { version = "0.7", features = ["hnsw", "serde"] }
```

```rust
// Save
index.save_to_file("my_index.json")?;

// Load
let index = HNSWIndex::load_from_file("my_index.json")?;
```

See [`examples/06_save_and_load.rs`](examples/06_save_and_load.rs) for a full example.

## Benchmark

GloVe-25 (1.18M vectors, 25-d, angular distance), Apple Silicon, single-threaded:

<p align="center">
  <img src="docs/plots/algorithm_comparison_glove-25-angular.png" width="680" alt="Recall vs QPS on GloVe-25" />
</p>

Summary at best recall per algorithm:

| Algorithm | Recall@10 | QPS |
|-----------|-----------|-----|
| HNSW (M=16) | 100.0% | 2,857 |
| Vamana | 100.0% | 1,177 |
| DiskANN | 100.0% | 1,029 |
| NSW (M=16) | 99.2% | 1,288 |
| IVF-PQ | 98.7% | 69 |
| IVF-AVQ | 90.9% | 194 |
| RP-Forest | 58.5% | 4,221 |

Full numbers and SIFT-128 results in [`docs/benchmark-results.md`](docs/benchmark-results.md).

## Supported Algorithms

Each algorithm has a named feature flag:

| Algorithm | Feature | Notes |
|-----------|---------|-------|
| HNSW | `hnsw` (default) | Best recall/QPS balance for in-memory search |
| SQ4U | `hnsw` + `sq4` | HNSW with 4-bit quantized graph traversal + exact rerank; benefits high-d data |
| SymphonyQG | `hnsw` + `ivf_rabitq` | HNSW with RaBitQ quantized graph traversal; cheap approximate beam search + exact rerank |
| NSW | `nsw` | Flat small-world graph; competitive with HNSW on high-d data |
| Vamana | `vamana` | DiskANN-style robust pruning; fast search, higher build time |
| NSG | `nsg` | Monotonic RNG pruning; build slows above ~50K vectors due to O(n) ensure_connectivity |
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
| LEMUR | `lemur` | Late-interaction MIPS; needs externally-provided encoder weights (no in-tree training); mean-pool used in place of OLS |
| LSH | `lsh` | Cross-Polytope LSH (Andoni et al. 2015); single hash table + multiprobe |
| LsmIndex | `hnsw` | LSM-tree tiered HNSW for streaming insert/delete/update workloads |
| DiskANN | `diskann` | Vamana + SSD I/O layout; experimental |
| SNG | `sng` | OPT-SNG (auto-tuned sparse neighborhood graph); sub-quadratic build per arXiv:2509.15531 |
| DEG | `hnsw` | Density-adaptive edge budgets (submodule of hnsw); O(n^2) |
| KD-Tree | `kdtree` | Exact NN; fast for d <= 20 (experimental) |
| Ball Tree | `balltree` | Exact NN; slightly better than KD-Tree for d=20-50 (experimental) |
| RP-Forest | `rptree` | Approximate; fast build, moderate recall (experimental) |

Quantization: RaBitQ, SAQ (`quantization` feature, via `qntz` crate). PQ is part of `ivf_pq`.

See [docs.rs](https://docs.rs/vicinity) for the full API.

## Documentation

- [User guide](docs/GUIDE.md): quick start, distance metrics, LID, common pitfalls
- [Benchmarks](docs/benchmark-results.md): recall/QPS tables across datasets
- [ANN landscape](docs/landscape.md): algorithmic principles, math foundations, research context
- [References](docs/references.md): bibliography for every algorithm in the crate

## License

MIT OR Apache-2.0
