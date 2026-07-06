# vicinity

[![crates.io](https://img.shields.io/crates/v/vicinity.svg)](https://crates.io/crates/vicinity)
[![docs.rs](https://docs.rs/vicinity/badge.svg)](https://docs.rs/vicinity)
[![PyPI](https://img.shields.io/pypi/v/pyvicinity.svg)](https://pypi.org/project/pyvicinity/)

Approximate nearest-neighbor search. Each algorithm is a separate
feature flag. Depend on only what you use. Rust crate `vicinity`;
Python bindings ship as `pyvicinity` on PyPI.

HNSW defaults to cosine distance and expects unit-norm vectors unless
`auto_normalize(true)` is set on the builder. It also supports L2, angular, and
inner-product distance through `DistanceMetric`. Other indexes expose their own
metric choices; check the specific algorithm before assuming every metric is
available.

## Which index?

```
Which index?
├── General purpose: HNSW (default)
├── Memory constrained: IVF-PQ compressed in-memory search
├── File-backed search: DiskANN (experimental; mmap/page layout planned)
├── Streaming insert/delete: FreshGraph (per-op latency) or LsmIndex (write throughput)
├── Filtered search: ACORN (metadata filters) or Curator (label filters)
├── Batch/static: IVF-PQ or IVF-AVQ
└── Sparse vectors: SparseMIPS
```

Start with plain HNSW and benchmark alternatives on your dataset. SQ4U and
SymphonyQG are available for quantized graph traversal, but current measurements
do not show a win over plain HNSW on the published GloVe-25 or GIST-960 runs.

## Install

Requires Rust 1.89+. Each algorithm is a separate feature; enable what you need:

```toml
[dependencies]
vicinity = { version = "0.10.5", features = ["hnsw"] }          # graph index
# vicinity = { version = "0.10.5", features = ["ivf_pq"] }      # compressed index
# vicinity = { version = "0.10.5", features = ["nsw"] }         # flat graph
```

## Usage

### HNSW

High recall, in-memory. Default index.

```rust
use vicinity::hnsw::HNSWIndex;

let mut index = HNSWIndex::builder(128)
    .m(16)
    .ef_search(50)
    .auto_normalize(true) // cosine (default) requires unit-norm vectors
    .build()?;
index.add_slice(0, &[0.1; 128])?;
index.add_slice(1, &[0.2; 128])?;
index.build()?;

let results = index.search(&[0.1; 128], 5, 50)?;
// results: Vec<(doc_id, distance)>; distance in [0, 2], lower is closer.
```

### IVF-PQ

Compressed in-memory index. 32–64× less vector payload than HNSW, lower recall.
Use when raw vectors dominate RAM; use DiskANN for file-backed search.

```rust
use vicinity::ivf_pq::{IVFPQIndex, IVFPQParams};

// IVF-PQ trains a quantizer on `build()`; aim for ≳ codebook_size × num_codebooks
// training vectors (defaults: codebook_size=256, num_codebooks=8 → ≳ 2048).
let params = IVFPQParams { num_clusters: 1024, num_codebooks: 8, nprobe: 16, ..Default::default() };
let mut index = IVFPQIndex::new(128, params)?;
for (id, vec) in dataset.iter().enumerate() {
    index.add_slice(id as u32, vec)?;
}
index.build()?;

let results = index.search(&query, 5)?;
let reranked = index.search_reranked(&query, 5, 200)?;
```

`search()` uses compressed PQ distances and still works after `compact()`.
`search_reranked()` keeps raw vectors and reranks an approximate candidate pool
with exact cosine distance.

See [`examples/ivf_pq_demo.rs`](examples/ivf_pq_demo.rs) for a runnable end-to-end example.

### Python (pyvicinity)

The Python bindings expose HNSW search over NumPy `float32` arrays. Wheels
built from this source tree use abi3 for CPython 3.10+. The published
[`pyvicinity`](https://pypi.org/project/pyvicinity/) package may lag this
repository.

```bash
pip install pyvicinity
```

```python
import numpy as np
from pyvicinity import HNSWIndex, DistanceMetric

# embeddings: any (n, 384) float32 array (sentence-transformers, openai, etc.)
embeddings = np.random.default_rng(0).standard_normal((10_000, 384), dtype=np.float32)

index = HNSWIndex(
    dim=384,
    metric=DistanceMetric.Cosine,
    auto_normalize=True,  # normalizes inserts and queries
    seed=42,
)
index.add_items(embeddings)
index.build()

# single query: top-10 neighbors. Distances in [0, 2]; lower means more similar.
ids, dists = index.search(embeddings[0], k=10)

# batch: same shape, one row per query.
queries = embeddings[:32]
batch_ids, batch_dists = index.batch_search(queries, k=10)  # (32, 10) int64
```

Runnable examples (in the [source repo](https://github.com/arclabs561/vicinity),
under [`examples/python/`](examples/python/), not shipped with the wheel):

- `01_text_similarity.py`: semantic search over text with sentence-transformers
- `02_batch_and_recall.py`: recall@10 vs `ef_search` sweep
- `03_ann_benchmarks_harness.py`: drop-in `ann-benchmarks` / VIBE wrapper

The bindings ship hand-written `.pyi` stubs (`py.typed`) and are verified
in CI by `mypy.stubtest`.

## Persistence

Save and load indexes with the `serde` feature:

```toml
[dependencies]
vicinity = { version = "0.10.5", features = ["hnsw", "serde"] }
```

```rust
// Save
index.save_to_file("my_index.json")?;

// Load
let index = HNSWIndex::load_from_file("my_index.json")?;
```

See [`examples/06_save_and_load.rs`](examples/06_save_and_load.rs) for a full example.

A binary segment format (smaller on disk, with WAL support) is available under the
`persistence` feature via `HNSWSegmentWriter` / `HNSWSegmentReader`. Segments carry a
magic header and format version: files written by 0.6.x still load via a legacy
fallback, files from a newer format version are rejected with an error, and corrupt
or truncated input returns an error rather than panicking or decoding garbage.

The `store` feature adds `store::UpdatableIndex`, an updatable, durable
*multi-segment* ANN index backed by [`segstore`](https://crates.io/crates/segstore):
incremental add/delete, a write-ahead log, checkpoint, compaction, and crash
recovery. Each segment is a cached HNSW and can be persisted as a sidecar, so a
restart loads unchanged segment graphs instead of rebuilding them. For read-only
serving, `store::SnapshotIndex` opens the checkpoint manifest and queries
sidecars first, decoding a source-vector segment only when its sidecar is
missing or must be rebuilt. Segments are searched and merged; this is the
deliberate alternative to a single evolving graph, and like any HNSW the merged
result is approximate. Fully out-of-core ANN search is the DiskANN/page-layout
track, not this wrapper. Opt-in; the default build does not depend on segstore. See
[`examples/updatable_store.rs`](examples/updatable_store.rs) for add, delete,
checkpoint, reopen, and search through this path.
For measurement, `cargo run --release --features store --example store_reopen_diagnostics`
prints the first snapshot-search cost with persisted HNSW sidecars present
versus after deleting those sidecars and forcing source-segment rebuilds.

## Benchmark

The benchmark harness writes JSONL rows with recall, QPS, build time, RSS, and
p50/p95/p99 latency:

```bash
cargo run --example ann_benchmark --release --features hnsw,ivf_pq,ivf_avq -- \
  data/ann-benchmarks/glove-25-angular \
  --algo hnsw --algo ivfpq --algo ivf_avq --json --fresh
```

Historical GloVe-25 results (1.18M vectors, 25-d, angular distance):

<p align="center">
  <img src="docs/plots/algorithm_comparison_glove-25-angular.png" width="680" alt="Recall vs QPS on GloVe-25" />
</p>

Summary at best recall per algorithm from older single-threaded runs:

| Algorithm | Recall@10 | QPS |
|-----------|-----------|-----|
| HNSW (M=16) | 100.0% | 2,857 |
| Vamana | 100.0% | 1,177 |
| DiskANN | 100.0% | 1,029 |
| NSW (M=16) | 99.2% | 1,288 |
| IVF-PQ | 98.7% | 69 |
| IVF-AVQ | 90.9% | 194 |
| RP-Forest | 58.5% | 4,221 |

The IVF-PQ row above predates the reranked search path and is not the expected
ceiling. Selected benchmark tables and current run commands are in
[`docs/benchmark-results.md`](docs/benchmark-results.md); raw SIFT-128 records
are in [`docs/sift-128-euclidean.jsonl`](docs/sift-128-euclidean.jsonl).

## Algorithms

Each algorithm has a named feature flag:

| Algorithm | Feature | Notes |
|-----------|---------|-------|
| HNSW | `hnsw` (default) | Best recall/QPS balance for in-memory search |
| SQ4U | `hnsw` + `sq4` | HNSW with 4-bit quantized graph traversal + exact rerank; experimental, benchmark before defaulting |
| SymphonyQG | `hnsw` + `ivf_rabitq` | HNSW with RaBitQ quantized graph traversal; experimental, benchmark before defaulting |
| NSW | `nsw` | Flat small-world graph; competitive with HNSW on high-d data |
| Vamana | `vamana` | DiskANN-style robust pruning; fast search, higher build time |
| NSG | `nsg` | Monotonic RNG pruning; build slows above ~50K vectors due to O(n) ensure_connectivity |
| EMG | `emg` | Multi-scale graph with alpha-pruning |
| FINGER | `finger` | Projection-based distance lower bounds for search pruning |
| PiPNN | `pipnn` | Partition-then-refine with HashPrune; reduces I/O during build |
| FreshGraph | `fresh_graph` | Streaming insert/delete with tombstones |
| IVF-PQ | `ivf_pq` | Compressed index with optional exact reranking |
| IVF-AVQ | `ivf_avq` | Anisotropic VQ + reranking; inner product search |
| IVF-RaBitQ | `ivf_rabitq` | RaBitQ binary quantization; provable error bounds |
| RpQuant | `rp_quant` | Random projection + scalar quantization |
| BinaryFlat | `binary_index` | 1-bit quantization + full-precision rerank |
| Curator | `curator` | K-means tree with per-label Bloom filters; low-selectivity filtered search |
| FilteredGraph | `filtered_graph` | Predicate-filtered graph search (AND/OR metadata filters) |
| ACORN | `hnsw` | Filtered HNSW search with two-hop expansion (SIGMOD 2024) |
| RangeFiltered | `range_filtered` | HNSW + attribute-range post-filter (renamed from `esg` in 0.8.0) |
| SparseMIPS | `sparse_mips` | Graph index for sparse vectors (SPLADE/BM25) |
| LEMUR | `lemur` | Late-interaction MIPS; needs externally-provided encoder weights (no in-tree training); mean-pool used in place of OLS |
| LSH | `lsh` | Cross-Polytope LSH (Andoni et al. 2015); single hash table + multiprobe |
| LsmIndex | `hnsw` | LSM-tree tiered HNSW for streaming insert/delete/update workloads |
| DiskANN | `diskann` | Vamana + SSD I/O layout; experimental |
| SNG | `sng` | OPT-SNG (auto-tuned sparse neighborhood graph); sub-quadratic build per arXiv:2509.15531 |
| DEG | `hnsw` | Density-adaptive edge budgets (in-house experimental variant; no benchmark) |
| KD-Tree | `kdtree` | Tree baseline for low-dimensional angular data (experimental) |
| Ball Tree | `balltree` | Tree baseline for low-dimensional angular data (experimental) |
| RP-Tree | `rptree` | Single random-projection tree baseline (experimental) |
| RP-Forest | `rptree` | Approximate tree baseline; fast build, lower recall (experimental) |
| K-Means Tree | `kmeans_tree` | Hierarchical clustering tree baseline (experimental) |

Quantization features are split by use: the public IVF-RaBitQ and
`hnsw::SymphonyQGIndex` types use `ivf_rabitq`; standalone
`vicinity::quantization` re-exports need `quantization` plus `rabitq` or `saq`.
The standalone RaBitQ and SymphonyQG examples use the latter path. PQ is part of
`ivf_pq`.

### Experimental status

Algorithms tagged `experimental` are reachable from the public API but
have not yet cleared the bar to be recommended defaults. Each has a
specific gap that, once closed, would promote it:

- **DiskANN**: file-based save/load and `DiskANNSearcher` work, and
  `search_with_diagnostics` reports graph reads, vector reads, logical
  bytes read, and visited-node counts. Promote when the file-backed path
  has mmap/page-layout measurements and recall@10 stays competitive on a
  1M-vector dataset.
- **DEG**: in-house density-adaptive variant of HNSW. No published
  benchmark vs plain HNSW. Promote when a head-to-head shows a recall
  or QPS win on at least two ann-benchmarks datasets.
- **KD-Tree, Ball Tree**: exact NN, niche use case (d ≤ 20-50). Stable
  but not heavily optimized. Promote when a representative workload
  motivates SIMD'd distance kernels and parallel build.
- **RP-Forest**: fast build, moderate recall (~58% on GloVe-25 per the
  benchmark table above). Promote when a seed-selection or projection
  improvement closes the recall gap to NSW (~99%) at the same QPS.
- **LEMUR**: late-interaction MIPS; ships an inference-only skeleton
  that requires externally-provided encoder weights and uses mean-pool
  in place of the paper's OLS fit. Promote when in-tree training lands.
- **SQ4U and SymphonyQG**: quantized graph traversal with exact rerank.
  Promote when they beat plain HNSW on recall/QPS on at least two published
  benchmark datasets; the current published runs do not show that yet.

See [docs.rs](https://docs.rs/vicinity) for the full API.

## Documentation

- [User guide](docs/GUIDE.md): quick start, distance metrics, LID, common pitfalls
- [Benchmarks](docs/benchmark-results.md): recall/QPS tables across datasets
- [ANN landscape](docs/landscape.md): algorithmic principles, math foundations, research context
- [References](docs/references.md): bibliography for every algorithm in the crate

## License

MIT OR Apache-2.0
