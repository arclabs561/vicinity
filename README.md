# vicinity

[![crates.io](https://img.shields.io/crates/v/vicinity.svg)](https://crates.io/crates/vicinity)
[![docs.rs](https://docs.rs/vicinity/badge.svg)](https://docs.rs/vicinity)

Nearest-neighbor search.

```toml
[dependencies]
# Pick the algorithm(s) you need:
vicinity = { version = "0.3", features = ["hnsw"] }
# vicinity = { version = "0.3", features = ["ivf_pq"] }
# vicinity = { version = "0.3", features = ["hnsw", "ivf_pq", "quantization"] }
```

**HNSW** — graph index, high recall, in-memory:

```rust
use vicinity::hnsw::HNSWIndex;

let mut index = HNSWIndex::builder(128).m(16).ef_search(50).build()?;
index.add_slice(0, &[0.1; 128])?;
index.add_slice(1, &[0.2; 128])?;
index.build()?;

let results = index.search(&[0.1; 128], 5, 50)?;
```

**IVF-PQ** — compressed index, lower memory, larger datasets (`features = ["ivf_pq"]`):

```rust
use vicinity::ivf_pq::{IVFPQIndex, IVFPQParams};

let params = IVFPQParams { num_clusters: 64, num_codebooks: 8, nprobe: 8, ..Default::default() };
let mut index = IVFPQIndex::new(128, params)?;
index.add_slice(0, &[0.1; 128])?;
index.add_slice(1, &[0.2; 128])?;
index.build()?;

let results = index.search(&[0.1; 128], 5)?;
```

## Benchmark

HNSW (M=16) on GloVe-25 (1.18M vectors, 25-d, cosine), Apple Silicon, single-threaded:

| ef_search | Recall@10 | QPS   |
|-----------|-----------|-------|
| 10        | 63.0%     | 1,496 |
| 50        | 88.4%     | 1,409 |
| 100       | 94.3%     | 1,326 |
| 200       | 97.6%     | 1,189 |
| 400       | 99.1%     | 992   |

<p align="center">
  <img src="doc/plots/algorithm_comparison_glove-25-final.png" width="680" alt="HNSW recall vs QPS on GloVe-25" />
</p>

Context: hnswlib (C++, AVX2) achieves ~95% recall at ~5K QPS on the same dataset. The gap is graph traversal overhead, not distance computation. Full numbers and M=32 results in [`doc/benchmark-results.md`](doc/benchmark-results.md).

## Algorithms

Stable: HNSW, NSW, IVF-PQ, PQ, RaBitQ, SQ8.

Experimental (behind feature flags): Vamana/DiskANN, SNG, DEG, ScaNN, KD-Tree, Ball Tree, RP-Forest, K-Means Tree.

See [docs.rs](https://docs.rs/vicinity) for the full API.

## License

MIT OR Apache-2.0
