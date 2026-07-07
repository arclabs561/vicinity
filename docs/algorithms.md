# Algorithm Catalog

Start with `hnsw::HNSWIndex` unless a workload gives you a reason not to. This
page lists the other public indexes and the feature flags that expose them.

## Feature Flags

| Algorithm | Feature | Notes |
| --- | --- | --- |
| HNSW | `hnsw` (default) | Default in-memory index |
| NSW | `nsw` | Flat small-world graph |
| Vamana | `vamana` | DiskANN-style graph construction |
| DiskANN | `diskann` | Vamana plus file and mmap search paths; experimental |
| In-place HNSW | `hnsw` | In-place insert/delete graph variant |
| LSM HNSW | `hnsw` | Tiered HNSW for streaming writes (`streaming::lsm`) |
| Dual-branch HNSW | `hnsw` | Extra bridge edges for high-LID regions |
| DEG | `hnsw` | Density-adaptive HNSW variant |
| LID estimation | `hnsw` | Diagnostic/helper for local intrinsic dimensionality; not a search index |
| Adaptive HNSW | `hnsw` | DARTH-style early termination via `search_adaptive` |
| AD-sampling HNSW | `hnsw` | Adaptive distance-evaluation helper for HNSW |
| HNSW-PRT | `hnsw` | HNSW with random-projection tree candidate filtering |
| HNSW repair/tombstones | `hnsw` | Deletion repair and tombstone internals |
| NSG | `nsg` | Monotonic RNG pruning; build slows above roughly 50K vectors due to O(n) connectivity repair |
| SNG | `sng` | OPT-SNG-style sparse neighborhood graph |
| EMG | `emg` | Multi-scale graph with alpha pruning |
| FINGER | `finger` | Projection-based lower bounds for search pruning |
| PiPNN | `pipnn` | Partition-then-refine search |
| FreshGraph | `fresh_graph` | Streaming insert/delete with tombstones |
| Updatable store | `store` | Segmented HNSW plus WAL/checkpoint lifecycle |
| IVF-PQ | `ivf_pq` | Compressed inverted-file search with optional exact reranking |
| IVF-AVQ | `ivf_avq` | Anisotropic vector quantization with reranking |
| IVF-RaBitQ | `ivf_rabitq` | RaBitQ binary quantization |
| RpQuant | `rp_quant` | Random projection plus scalar quantization |
| BinaryFlat | `binary_index` | One-bit quantization with full-precision rerank |
| SQ4 | `sq4` | Standalone 4-bit scalar quantized flat index |
| SQ4U | `hnsw` + `sq4` | HNSW with 4-bit quantized traversal and exact rerank |
| SQ8U | `hnsw` + `sq8` | HNSW with 8-bit quantized traversal |
| SymphonyQG | `hnsw` + `ivf_rabitq` | HNSW with RaBitQ inside graph traversal |
| SymphonyQG-VR | `hnsw` + `ivf_rabitq` | Vertex-relative SymphonyQG graph traversal |
| Curator | `curator` | K-means tree with per-label Bloom filters |
| FilteredGraph | `filtered_graph` | Predicate-filtered graph search |
| ACORN | `hnsw` | Filtered HNSW search with two-hop expansion |
| RangeFiltered | `range_filtered` | HNSW plus attribute-range post-filter |
| SparseMIPS | `sparse_mips` | Sparse vector graph index for SPLADE/BM25-style vectors |
| LEMUR | `lemur` | Late-interaction MIPS scaffold; requires external encoder weights |
| LSH | `lsh` | Cross-polytope LSH |
| KD-tree | `kdtree` | Low-dimensional cosine tree baseline |
| Ball tree | `balltree` | Low-dimensional cosine tree baseline |
| RP-tree | `rptree` | Single random-projection tree |
| RP-forest | `rptree` | Approximate projection-tree baseline |
| K-means tree | `kmeans_tree` | Hierarchical clustering tree baseline |
| EVoC | `evoc` | Clustering wrapper, not nearest-neighbor search |
| Brute force | benchmark runner | Exact ground-truth baseline, not an index type |

Quantization feature names are split by use. Public IVF-RaBitQ and
`hnsw::SymphonyQGIndex` types use `ivf_rabitq`; standalone quantizer re-exports
need `quantization` plus `rabitq` or `saq`. PQ is part of `ivf_pq`.

## Recommended Defaults

| Workload | Start with | Try next |
| --- | --- | --- |
| Small corpus (<10K vectors) | Brute force | HNSW when scale or latency requires |
| Dense vectors that fit in memory | HNSW | NSW or Vamana |
| Raw vectors dominate RAM | HNSW, then IVF-PQ | IVF-PQ with reranking |
| Frequent writes/deletes | `store::UpdatableIndex` or FreshGraph | LSM HNSW |
| Metadata filters | HNSW with post-filtering | ACORN, Curator, or FilteredGraph |
| Sparse learned retrieval | SparseMIPS | Workload-specific sparse baseline |
| File-backed graph search | DiskANN | Benchmark mmap/file rows before serving |
| File-backed compressed search | IVF-PQ file or mmap searcher | Use persisted PQ-code snapshots |

## Experimental Status

These APIs are reachable but are not recommended defaults yet.

- **Graph alternatives (NSW, Vamana, NSG, SNG, EMG, FINGER, PiPNN)**: promote
  when a head-to-head benchmark shows a recall/QPS/build-time win over HNSW on a
  documented workload. Vamana is the closest default candidate; the others need
  fresh rows before they should guide users.
- **DiskANN**: file save/load, mmap search, and search diagnostics exist.
  Promote when mmap/page-layout measurements stay competitive on a 1M-vector
  dataset, and when the file-backed row is clearly separated from in-memory
  Vamana in the benchmark docs.
- **Streaming and updates (FreshGraph, in-place HNSW, LSM HNSW, repair,
  tombstones, and `store::UpdatableIndex`)**: promote per workload, not as a
  global default. Each needs churn rows that report active-set recall, update
  throughput, query latency, and storage residency.
- **Filtered search (HNSW post-filtering, ACORN, Curator, FilteredGraph, and
  RangeFiltered)**: promote from selectivity sweeps. Report recall/QPS over at
  least low, middle, and high selectivity instead of a single QPS number.
- **Compressed inverted files (IVF-PQ, IVF-AVQ, IVF-RaBitQ, RpQuant,
  BinaryFlat, SQ4)**: promote per memory budget. IVF-PQ is the current
  compressed default candidate; the others need recall/QPS/storage rows on the
  datasets where their quantization assumptions apply.
- **Quantized HNSW traversal (SQ4U, SQ8U, SymphonyQG, and SymphonyQG-VR)**:
  promote when they beat plain HNSW on recall/QPS on at least two published
  datasets. Low-dimensional GloVe-25 alone is not enough.
- **LID estimation, Adaptive HNSW, AD-sampling, and HNSW-PRT**: treat these as
  diagnostics or search helpers, not standalone index recommendations. Promote
  them when they improve the same built HNSW index at fixed recall, not just raw
  QPS, and include a fallback row where the heuristic is disabled.
- **Classical trees (KD-tree, ball tree, RP-tree, RP-forest, and K-means tree)**:
  treat as first-class baselines. Promote only for the dimensionality band where
  they win, and keep exact KD/ball rows separate from approximate tree rows.
- **LSH**: keep as a legacy dense-vector baseline until a workload needs its
  false-positive/false-negative tuning properties.
- **SparseMIPS and LEMUR**: evaluate on sparse or late-interaction datasets,
  not dense GloVe rows. LEMUR also needs in-tree training or a reproducible
  model-loading path before it can be recommended.
- **EVoC**: clustering support, not nearest-neighbor search. Keep it out of
  ANN recommendations unless the workload is explicitly clustering-first.
- **Brute force**: exact oracle for correctness and recall measurement. It is
  not an ANN target, but it should stay in benchmark output as the ground-truth
  floor.

Detailed benchmark rows are in [`benchmark-results.md`](benchmark-results.md).
