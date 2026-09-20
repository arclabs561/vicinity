# vicinity

[![crates.io](https://img.shields.io/crates/v/vicinity.svg)](https://crates.io/crates/vicinity)
[![docs.rs](https://docs.rs/vicinity/badge.svg)](https://docs.rs/vicinity)

Approximate nearest-neighbor search.

`vicinity` provides Rust indexes and Python bindings for vector search.

## Rust

```toml
[dependencies]
vicinity = { version = "0.11.1", features = ["hnsw"] }
```

HNSW is the default in-memory index for dense vectors. For cosine distance,
pass unit-norm vectors or set `auto_normalize(true)`.

```rust
use vicinity::hnsw::HNSWIndex;

fn main() -> vicinity::Result<()> {
    let mut index = HNSWIndex::builder(2)
        .ef_search(50)
        .auto_normalize(true)
        .build()?;

    index.add_slice(7, &[1.0, 0.0])?;
    index.add_slice(8, &[0.8, 0.2])?;
    index.add_slice(9, &[0.0, 1.0])?;
    index.build()?;

    let results = index.search(&[1.0, 0.0], 2, 50)?;
    for (id, distance) in results {
        println!("{id}: {distance:.4}");
    }
    Ok(())
}
```

```text
7: 0.0000
8: 0.0299
```

Results contain the IDs supplied on insert and distances; lower is closer.
Use `distance::DistanceMetric` to select L2, angular, or inner-product distance.

## Python

The Python package is named `pyvicinity` because the bare `vicinity` name is
held by an unrelated PyPI project. The published wrapper exposes HNSW; the
repository also contains IVF-PQ bindings that remain outside the published
wheel until their benchmark and persistence contracts are settled.

```bash
pip install pyvicinity
```

```python
import numpy as np
from pyvicinity import DistanceMetric, HNSWIndex

vectors = np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)
index = HNSWIndex(dim=2, metric=DistanceMetric.Cosine, auto_normalize=True, seed=42)
index.add_items(vectors)
index.build()

ids, distances = index.search(vectors[0], k=2)
print(ids.tolist())
```

```text
[0, 1]
```

[`examples/python/02_batch_and_recall.py`](examples/python/02_batch_and_recall.py)
shows batch search and recall measurement.

## Feature and dependency footprints

Choose the smallest feature set that matches the workload. Optional algorithm
families do not belong in the default build, and `--all-features` is intended
for development and comparison rather than a production dependency profile.

| Use case | Cargo features | Adds |
| --- | --- | --- |
| Minimal library | `--no-default-features` | Core `smallvec`, `rand`, and `thiserror`. |
| Default HNSW | *(default)* | `hnsw` plus SIMD distance kernels from `innr`. |
| JSON snapshots | `hnsw,serde` | `serde` and `serde_json`. |
| Binary persistence/mmap | `persistence` | `postcard` and `durability`; this is restart/file support, not automatically crash-safe generation publication. |
| Segmented mutable store | `store` | `segstore` plus persistence dependencies and HNSW. |
| IVF-PQ/OPQ | `ivf_pq` | `clump`, `nalgebra`, and serialization support. OPQ uses the linear-algebra path. |
| Parallel batch search | `parallel` | `rayon`. |
| Python extension | `python` | PyO3 stable ABI (`abi3-py310`), NumPy, HNSW, IVF-PQ, and persistence. The current wheel contract targets CPython 3.10+ and does not cover free-threaded CPython builds. |
| WASM experiment | `--no-default-features` plus the target recipe | `getrandom` uses the `wasm_js` backend on `wasm32-unknown-unknown`; file/mmap and persistence support need separate validation. |

Rust consumers should treat feature flags as part of the build contract. Python
callers should benchmark end-to-end NumPy conversion, GIL detachment, and batch
overhead, not only the native search loop. See the
[Python wrapper competition plan](docs/design/python-wrapper-competition.md) and
the reproducible `scripts/benchmark_python_wrapper.py` harness.

## Indexes and persistence

| Need | Use | Feature |
| --- | --- | --- |
| Dense vectors in memory | HNSW | `hnsw` (default) |
| Lower vector memory use | IVF-PQ, with recall tradeoffs | `ivf_pq` |
| HNSW JSON save/load | `save_to_file` / `load_from_file` | `serde` |
| HNSW binary segments | `persistence::hnsw` | `persistence` |

IVF-PQ's `search()` uses compressed PQ distances. `search_reranked()` retains
raw vectors and reranks candidates with exact cosine distance. See the runnable
[`examples/ivf_pq_demo.rs`](examples/ivf_pq_demo.rs). Other indexes are
feature-gated; their status and tradeoffs are in the
[algorithm catalog](docs/algorithms.md).

## Benchmarks

Selected HNSW measurements use the full corpus, 1,000 queries, and the median
of three isolated runs on an Apple M3 Max with Rust 1.97.1. QPS measures
sequential single-query throughput; index sizes are heap estimates.

| Dataset | Vectors | `ef_search` | Recall@10 | QPS | Index size |
| --- | ---: | ---: | ---: | ---: | ---: |
| GloVe-100 cosine | 1,183,514 | 1600 | 91.51% | 742 | 1.50 GB |
| SIFT-128 L2 | 1,000,000 | 200 | 98.20% | 3,836 | 1.38 GB |

Results depend on the dataset and search settings. See
[benchmark results](docs/benchmark-results.md#current-full-corpus-compressed-search-comparison)
for parameters, repeat spread, compressed indexes, and reproduction commands.

## Limits

Search is approximate. Increase `ef_search` for higher HNSW recall at the cost
of query time. Build the index before searching; the default HNSW index does
not accept new vectors after build. DiskANN and several other indexes are
experimental; see the [algorithm catalog](docs/algorithms.md) before choosing them.

## Documentation

[User guide](docs/GUIDE.md) · [API](https://docs.rs/vicinity) ·
[Algorithms](docs/algorithms.md) · [Datasets](docs/datasets.md) ·
[References](docs/references.md)

## License

MIT OR Apache-2.0
