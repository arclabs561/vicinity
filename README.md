# vicinity

Approximate nearest-neighbour search in Rust, with Python bindings.

HNSW is the default. Behind feature flags sit compression families (IVF-PQ,
IVF-AVQ, RaBitQ), graph builders (Vamana, DiskANN), the NSW line (NSG, NSW, SNG),
random-projection trees, ball and k-d trees, LSH, and binary or 4-bit codes.

Two paths cover what plain HNSW cannot. A filtered graph answers predicate
queries without leaving the graph, and an in-place index takes updates without a
rebuild.

One protocol measures all of them at the same sizes, so comparisons are between
families rather than between benchmarks.

```toml
[dependencies]
vicinity = { version = "0.11.2", features = ["hnsw", "serde"] }   # serde adds save_to_file/load_from_file
```

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

Results carry the IDs you supplied and the distance; lower is closer.
`distance::DistanceMetric` selects L2, angular or inner product. Cosine assumes
unit-norm vectors unless you set `auto_normalize(true)`.

## What it does that the usual answer does not

**Filtered search stays on the graph.** `hnsw::search_acorn` expands two hops
through a predicate instead of filtering after the fact, and
`filtered_graph::search_filtered` picks between a beam and an exact scan of the
inverted list depending on selectivity.

On 20k random points at 20% selectivity, recall@10 is 1.000 for a pre-filtered
exact scan (91 µs), 1.000 for post-filtering an HNSW beam (602 µs), 1.000 through
the filtered graph and 0.583 for the ACORN-style two-hop walk (2.1 ms).

The gap opens at the other end. At 0.1% selectivity post-filtering collapses to
**0.158** recall while the inverted-list path still returns 1.000.
`tests/filtered_recall.rs` prints the whole sweep.

**Updates do not force a rebuild.** `InPlaceIndex` inserts and deletes in place;
`HNSWIndex::delete` leaves tombstones that are excluded from results and repaired
on demand. Deleting 10% of a 5k index used to cost most of the recall because
tombstones consumed the `ef` budget. That is fixed: at `ef=10` the same sweep
returns 0.978 instead of 0.090.

**It reads from disk if you want.** `diskann::DiskANNSearcher::load` queries
`graph.index` and `vectors.bin` through a block cache instead of holding the
index in the heap. At 1M points that costs 2.1–3.3× the resident p50 latency with
an 81% cache hit rate.

## Measured

`examples/large_scale.rs`, 16-core aarch64, macOS, 2026-11-01. Corpus generated
to disk and memory-mapped, ground truth exact over 500 sampled queries, all rows
committed under [`benchmarks/`](benchmarks/).

| n | domain | ef | recall@10 | p50 | build | index on disk | peak RSS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100k | clustered | 200 | 0.991 | 303 µs | 5.6 s | 46 MiB | 246 MiB |
| 1M | clustered | 200 | 0.807 | 440 µs | 109 s | 470 MiB | 2.0 GiB |
| 1M | uniform | 200 | 0.294 | 922 µs | 164 s | 470 MiB | 2.0 GiB |
| 3M | uniform | 6400 | 0.926 | 16.4 ms | 353 s | 1.4 GiB | 5.8 GiB |

Distance counts are in [`benchmarks/large-scale-100k-dists.jsonl`](benchmarks/large-scale-100k-dists.jsonl):
1411 / 2268 / 3442 / 5782 at ef 50 / 100 / 200 / 800 on 100k clustered.

Full-corpus GloVe-100/200 and GloVe-25 numbers, with parameters and repeat
spread, are in [benchmark results](docs/benchmark-results.md).

## What it costs

| at d=96, m=16 | per point |
| --- | ---: |
| vectors, f32 | 384 B |
| layer-0 adjacency, 32 neighbours | ~136 B |
| ids, reverse map, level | ~17 B |
| **steady state** | **~0.5 KiB** |
| measured build peak at 1M | ~2.0 KiB |

Two consequences. First, the build is in memory: there is no out-of-core
construction, so a 100M index would need roughly 199 GiB of RAM at this layout
and refuses rather than swapping. DiskANN and SPANN solve that by keeping
compressed vectors in RAM and the graph on SSD, and by building the graph in
shards; neither path is implemented here. Second, the build peak is not all data: a
1M build costs ~2.0 KiB/point at its peak, and the reorder that permutes the index
for locality used to account for ~0.5 KiB of that by copying the corpus. It
permutes in place now, which is a transient either way, not a requirement.

## Limits

Recall is not tier-invariant, and this is the number most likely to be quoted
wrong. On uniform random data at `ef=200`, recall@10 is 0.981 at 10k, 0.705 at
100k and 0.294 at 1M. The recall is recoverable: `ef=4800` reaches 0.973 at 1M.
But at 1M that is 1.45× **slower** than the exact scan it is meant to replace.
Uniform data in 96 dimensions is close to the worst case; clustered data holds
0.991 → 0.807 over the same range.

DiskANN, several tree and quantisation families are experimental: they are
measured on the same protocol but their persistence and update paths are not
covered to the same depth. The Python wheel exposes HNSW only; IVF-PQ bindings
exist in the repository and stay out of the wheel until their persistence
contract settles. No GPU path: graph traversal does not port without a second,
CAGRA-shaped index.

## Python

The package is `pyvicinity` because `vicinity` on PyPI is an unrelated project.

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

[`examples/python/02_batch_and_recall.py`](examples/python/02_batch_and_recall.py)
shows batch search and recall measurement.

## Features

Features are the build contract: an algorithm family you do not enable is not
compiled. `--all-features` is for comparison work, not for a shipped profile.

| Use case | Features | Adds |
| --- | --- | --- |
| Minimal | `--no-default-features` | `smallvec`, `rand`, `thiserror` |
| HNSW | *(default)* | `hnsw`, SIMD distance kernels from `innr` |
| JSON snapshots | `hnsw,serde` | `save_to_file` / `load_from_file` |
| Binary segments | `persistence` | `postcard` + `durability`; restart support, generation publication is separate |
| Segmented mutable store | `store` | `segstore`, persistence, HNSW |
| IVF-PQ / OPQ | `ivf_pq` | `clump`, `nalgebra` |
| Parallel batch search | `parallel` | `rayon` |
| Python | `python` | PyO3 `abi3-py310`, NumPy, HNSW, IVF-PQ, persistence |
| Large-scale harness | `large-scale` | `memmap2`; the example maps files, the library never does |

Where each path lives:

| Need | Use | Feature |
| --- | --- | --- |
| Dense vectors in memory | `HNSWIndex` | `hnsw` |
| Lower memory, recall tradeoff | `IVFPQIndex` | `ivf_pq` |
| Updates without rebuild | `InPlaceIndex` | `hnsw` |
| Query from disk | `DiskANNSearcher` | `diskann` |
| Filtered search | `search_acorn`, `search_filtered` | `hnsw`, `filtered_graph` |

Other families and their status are in the [algorithm catalogue](docs/algorithms.md).

## Status

Latest release: 0.11.2 (2026-10-31, crates.io). MSRV 1.94. The crate's own code
is `#![forbid(unsafe_code)]`; dependencies including Pulp and the benchmark
allocator are not, so this is not a transitive guarantee.

## Documentation

[User guide](docs/GUIDE.md) · [API](https://docs.rs/vicinity) ·
[Algorithms](docs/algorithms.md) · [Datasets](docs/datasets.md) ·
[References](docs/references.md)

## License

MIT OR Apache-2.0
