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
| AD-sampling HNSW | `hnsw` | Adaptive distance-evaluation helper for HNSW |
| HNSW-PRT | `hnsw` | HNSW with random-projection tree candidate filtering |
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
| KD-tree | `kdtree` | Exact low-dimensional baseline |
| Ball tree | `balltree` | Exact low-dimensional baseline |
| RP-tree | `rptree` | Single random-projection tree |
| RP-forest | `rptree` | Approximate projection-tree baseline |
| K-means tree | `kmeans_tree` | Hierarchical clustering tree baseline |
| EVoC | `evoc` | Clustering wrapper, not nearest-neighbor search |

Quantization feature names are split by use. Public IVF-RaBitQ and
`hnsw::SymphonyQGIndex` types use `ivf_rabitq`; standalone quantizer re-exports
need `quantization` plus `rabitq` or `saq`. PQ is part of `ivf_pq`.

## Recommended Defaults

| Workload | Start with | Try next |
| --- | --- | --- |
| Dense vectors that fit in memory | HNSW | NSW or Vamana |
| Raw vectors dominate RAM | HNSW, then IVF-PQ | IVF-PQ with reranking |
| Frequent writes/deletes | `store::UpdatableIndex` or FreshGraph | LSM HNSW |
| Metadata filters | HNSW with post-filtering | ACORN, Curator, or FilteredGraph |
| Sparse learned retrieval | SparseMIPS | Workload-specific sparse baseline |
| File-backed search | DiskANN | Benchmark mmap/file rows before serving |

## Experimental Status

These APIs are reachable but are not recommended defaults yet.

- **DiskANN**: file save/load and search diagnostics exist. Promote when
  mmap/page-layout measurements stay competitive on a 1M-vector dataset.
- **DualBranch, DEG, AD-sampling, HNSW-PRT, and in-place HNSW**: HNSW-family
  variants. Promote only after head-to-head recall/QPS/storage rows show a
  workload-specific win over plain HNSW.
- **KD-tree and ball tree**: exact low-dimensional baselines. Promote only for a
  documented low-dimensional workload.
- **RP-forest**: fast build, lower recall on the published GloVe-25 run. Promote
  if recall closes the gap to NSW at similar QPS.
- **LEMUR**: inference scaffold that requires external encoder weights. Promote
  when in-tree training or a reproducible model-loading path exists.
- **SQ4U, SQ8U, SymphonyQG, and SymphonyQG-VR**: quantized graph traversal with
  exact rerank. Promote when they beat plain HNSW on recall/QPS on at least two
  published datasets.

Detailed benchmark rows are in [`benchmark-results.md`](benchmark-results.md).
