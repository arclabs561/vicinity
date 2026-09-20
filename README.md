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

The published Python package, `pyvicinity` 0.8.0, exposes HNSW.

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
shows batch search and recall measurement. The repository also contains
IVF-PQ Python bindings; these are not yet published on PyPI.

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
