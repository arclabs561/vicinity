# Persistence

This document defines how each index family should persist and search saved
data. The goal is to keep storage choices matched to the index structure:
segmented HNSW, DiskANN graph pages, and IVF posting lists need different file
layouts.

## Lower Layers

`durability` is the shared persistence substrate. Use it for:

- atomic file replacement
- directory abstraction
- write-ahead logs
- checkpoints and recovery
- read-only mmap with access-pattern hints

`segstore` is narrower. Use it when the natural unit is an immutable segment
plus tombstones, compaction, and optional per-segment sidecar indexes. It is a
good fit for `store::UpdatableIndex` because that type is explicitly a
multi-segment HNSW index. It is not a universal ANN storage layer.

Do not add another lower-layer crate until at least two index families share the
same code. Keep the first shared pieces internal to `vicinity`, then extract
only after the interface has two real consumers.

## Storage Model

Storage is not a binary choice between "in memory" and "on disk". Treat each
row below as a different implementation and benchmark contract:

| Mode | Meaning | Typical fit |
| --- | --- | --- |
| Heap-resident index | Vectors, graph, codes, and candidate buffers are owned by the process. | Small to medium datasets and algorithm upper-bound measurements. |
| Snapshot reload | The index is serialized, loaded, then searched from process memory. | Format compatibility, restart cost, and post-load parity. |
| Read-only mmap | Persisted bytes are mapped and paged by the OS. Query latency depends on page-cache state and access locality. | IVF posting lists, graph pages, and vector payloads whose layout is already query-friendly. |
| File reader | Query code issues explicit file reads. This makes I/O batching, prefetch, and page layout visible. | DiskANN-style graph search where random reads dominate latency. |
| Segmented durable store | Mutations append to a WAL and seal immutable segments; search merges per-segment indexes. | Updatable HNSW-like stores with tombstones and compaction. |
| Disk-resident navigation | A compressed or centroid-level navigation structure stays memory-resident while full vectors or postings live on SSD. | Datasets larger than RAM where tail latency is bounded by page reads. |
| Object-storage backed cache | Object storage is the durable source of truth; SSD and memory are caches. Query planning minimizes remote round trips. | Large multi-tenant or cold/warm workloads. |

This spectrum is why `durability`, `segstore`, and algorithm-specific file
layouts should coexist. `durability` owns byte durability and mmap. `segstore`
owns append/checkpoint/compact semantics for immutable segments. DiskANN and
IVF-family indexes still need layouts shaped around their access patterns:
co-located graph pages for DiskANN, posting-list pages for IVF, and centroid or
compressed-vector navigation for cold data.

Production systems use the same separation. Faiss exposes mmap-backed on-disk IVF
inverted lists with list prefetching. DiskANN-style systems keep compressed
navigation data and hot-node caches in memory while graph/vector payloads live
on SSD. SPFresh/HFresh-style designs keep a centroid index in memory and scan
disk postings. Object-storage systems add another tier: object storage is the
durable source of truth, while SSD and memory are caches whose warm/cold state
must be part of the benchmark row.

## Capability Matrix

| Index family | Save/load | Memory search | File search | Mmap search | Updates | Storage direction |
| --- | --- | --- | --- | --- | --- | --- |
| HNSW | JSON via `serde`; binary segments via `persistence` | Yes | No | No | Tombstones and repair in memory; `store` for durable segments | Keep JSON and binary segment paths. HNSW snapshots preserve tombstone flags for restart parity. Use `store` when the workload needs WAL-backed segmented durability, checkpoint/reopen, and compaction. |
| `store::UpdatableIndex` | Open/checkpoint/reopen via `segstore` + HNSW sidecars | Segment sidecars loaded into memory | No | No | Add/delete/compact/checkpoint | Keep on `segstore`; this is the segmented-HNSW path. Benchmark both live post-checkpoint search and reopened `SnapshotIndex` search when validating restart behavior. |
| DiskANN | Yes, graph + vector files | Yes | Yes | Yes | Build-once | Save writes the current graph/vector files directly. File and mmap searchers read those files, with mmap using `durability`. Page/co-location layout remains next. Do not route through `segstore`. |
| NSW / SNG / Vamana / NSG / FINGER / PiPNN / EMG / LSH | Yes, directory format | Yes | No | No | Build-once | Persists the built in-memory graph state and restores it directly. This is snapshot-memory persistence, not file-backed search. |
| DualBranch / DEG | Yes, JSON via `serde` | Yes | No | No | Build-once | These are HNSW-family experimental variants. Their benchmark snapshot rows require the `serde` feature and reload into memory. DEG dense benchmark rows cap indexed vectors at 10,000 because construction is O(n^2). |
| HNSW quantized variants | Yes, for SQ4/SQ8 and non-compacted SymphonyQG | Yes | No | No | Build-once | SQ variants persist the underlying HNSW and rebuild quantization. SymphonyQG persists the underlying HNSW and RaBitQ manifest, then rebuilds quantized state on load. SymphonyQG-VR compacted snapshots are rejected because current search still needs raw parent vectors. |
| HNSW query accelerators | No separate accelerator snapshot | Yes | No | No | Derived from HNSW | ADSampling and PRT state are derived from a built HNSW's reordered raw vectors. Persist the base HNSW first; add accelerator snapshots only if rebuild cost shows up in benchmark rows. |
| IVF-PQ | Yes, directory format | Yes | Yes | Yes, with `persistence` | Build-once | `load_from_dir` rebuilds an in-memory snapshot, including optional filter field and document metadata. `IVFPQFileSearcher` reads persisted PQ codes and optional raw vectors from files or mmap. PQ-code sidecars are list-contiguous; exact rerank still reads optional raw vectors by vector ID. |
| IVF-AVQ | Yes, directory format | Yes | Yes | Yes, with `persistence` | Build-once | `load_from_dir` rebuilds an in-memory snapshot. `IVFAVQFileSearcher` reads partition IDs, AVQ codes, and raw rerank vectors directly from the saved files or read-only mmap. |
| IVF-RaBitQ | Yes, directory format for non-compacted indexes | Yes | No | No | Build-once | Persists raw vectors, centroids, and cluster membership, then rebuilds RaBitQ edge codes on load. Manifest validation rejects impossible dimensions, cluster counts, bit widths, and byte lengths before reading payload files. |
| FreshGraph / in-place graph | Yes | Yes | No | No | Insert/delete/compact | FreshGraph uses a snapshot directory with tombstones and inbound counts. `InPlaceIndex` and `MappedInPlaceIndex` use `serde`-gated file snapshots that preserve free slots and external-ID maps. WAL/checkpoint durability remains separate from `segstore`. |
| Curator | Yes, directory format | Yes | No | No | Build-once | Persists normalized vectors, doc IDs, labels, and parameters, then rebuilds the tree on load. |
| Range-filtered graph | Yes, directory format | Yes | No | No | Build-once | Persists normalized vectors and sorted attributes, then rebuilds HNSW on load. |
| Filtered graph | Yes, directory format | Yes | No | No | Build-once | Persists normalized vectors, graph neighbors, medoid, and inverted filter payloads. |
| BinaryFlat / RP-Quant | Yes, directory format | Yes | No | No | Build-once | Persists full-precision vectors and params, then rebuilds quantizer or projection-derived payloads on load. |
| SparseMIPS | Yes, directory format | Yes | No | No | Build-once | Persists sparse vectors in CSR-style offsets, indices, and values files plus the built graph. Current persistence coverage is API-level; dense `ann_benchmark` datasets skip SparseMIPS, while the SPV1 smoke harness covers benchmark plumbing until real SPLADE/BM25 data exists. |
| streaming LSM | Yes, directory snapshot | Yes | No | No | Insert/delete/compact | Snapshot persists L0, compacted levels, tombstones, config, and counters, then rebuilds compacted-level HNSW graphs on load when `hnsw` is enabled. This is restart persistence, not WAL/checkpoint durability. |
| LEMUR | No | Yes | No | No | Inference scaffold | LEMUR requires externally trained encoder weights and still scans document weights linearly. Add model loading/training and a real multi-vector benchmark before adding persistence. |
| EVoC | No | No ANN search row | No | No | Clustering | EVoC is a clustering wrapper, not a nearest-neighbor index. Evaluate it with clustering metrics and add persistence only if clustered outputs become a durable API. |
| Classic trees | Yes, directory format | Yes | No | No | Build-once | KD-tree, Ball tree, K-means tree, RP-tree, and RP-forest persist built trees and preserve external doc IDs. |

## Required Persistence Tests

Each saved format must have tests for:

- successful save/load search parity on a small deterministic corpus
- corrupt magic/version rejection
- truncated file rejection
- future-version rejection
- dimension and count bounds before allocation
- doc-id preservation
- storage-mode benchmark rows where file or mmap search exists

For approximate indexes, parity means the saved index returns the same result
set as the in-memory index at the same parameters on a deterministic fixture.
For file-backed searchers, recall should also be measured against ground truth.

## Implementation Order

1. Keep `durability` and `segstore` as the only lower-layer crates. Add no new
   crate until shared storage code has two consumers.
2. Finish DiskANN storage modes: current direct file save, file search, and mmap
   graph/vector readers are present. A `benchmark`-feature page-layout harness
   exists, but the literal one-node-per-4KB-page shape is slower on the current
   5K-vector smoke bench and is not promoted.
3. Keep IVF-PQ file-backed search covered by benchmark rows, then improve the
   raw-vector rerank locality for file and mmap search. The current format
   persists:
   - manifest with format version, metric, dimensions, counts, and parameters
   - centroids
   - PQ codebooks
   - cluster doc IDs and original vector-ID-ordered PQ codes
   - list offsets and list-contiguous PQ codes for file/mmap scans
   - 4-bit packed FastScan blocks rebuilt on memory-snapshot load
   - optional raw normalized vectors when exact rerank should survive reload
   - optional filter field and document metadata for snapshot-loaded filtered search
   `IVFPQFileSearcher` scans saved list-local PQ codes through file I/O or mmap.
   The next storage layout work is raw-vector locality for rerank, so reranking
   does not page through full-precision vectors in vector-ID order.
4. Keep IVF-AVQ file and mmap rows covered by benchmark output. Current direct
   search opens the saved partition and raw-vector files through normal file I/O
   or `durability` read-only mmap without adding unsafe code in `vicinity`.
5. Extend IVF-RaBitQ persistence to compacted indexes only after `qntz` exposes
   a safe serialized edge-code representation. Current non-compacted snapshots
   validate manifest sizes before reading raw payloads and deliberately rebuild
   qntz edge codes instead of reconstructing private edge-code fields.
6. Extend range-filtered persistence to file-backed search only if the filtered
   benchmark contract needs it; current load rebuilds an in-memory HNSW.
7. Keep ADSampling and PRT as derived HNSW accelerators until benchmark rows
   show rebuild cost matters. Persisting base HNSW plus raw vectors and
   tombstone flags is enough for correctness today.
8. Decide FreshGraph persistence separately. Its update model is not the same
   as segment append/compact, and forcing it through `segstore` would hide the
   in-place-update tradeoff.
9. Design streaming LSM durability separately from snapshots. The current
   snapshot path is useful for restart parity, but WAL/checkpoint recovery still
   needs level manifests, tombstone semantics, and compaction recovery.

## Benchmark Contract

Persistence work is not complete until the benchmark harness can distinguish:

- `storage_mode=in_memory`
- `storage_mode=snapshot_loaded`
- `storage_mode=file`
- `storage_mode=mmap`
- `storage_mode=segmented_store`

Rows must include recall, QPS, build time, p50/p95/p99 latency, cache state, and
RSS. In-memory rows may include `index_bytes` as the index's heap-owned size
estimate. Snapshot, file, and mmap rows should also include `load_time_s` and
`index_bytes` when the runner opens a saved index, where `index_bytes` is the
persisted or mapped artifact size. For a method with both memory and file-backed
search, the harness should emit both rows from the same built index when
possible.

For storage rows, also record enough context to avoid false comparisons:

- whether raw full-precision vectors are present for rerank
- how many queries the runner used to warm the index or OS page cache before
  timing
- whether reads go through mmap, buffered file I/O, direct I/O, or object
  storage/cache fetches
- how many bytes are heap-owned, mmap-backed, and persisted
- any cache budget or beam-width parameter that changes I/O parallelism

Storage rows are not interchangeable:

- `storage_mode=in_memory` measures the search path after vectors and graph
  structures are resident in process memory.
- `storage_mode=snapshot_loaded` measures an index saved to a snapshot,
  reopened, and then searched from process memory. It captures load time,
  serialized size, and post-load search equivalence; it is not an on-disk query
  path.
- `storage_mode=file` measures a searcher that reads persisted structures
  through normal file I/O. Report `load_time_s` separately from query latency.
- `storage_mode=mmap` measures a searcher opened over memory-mapped persisted
  structures. Treat OS page-cache state as part of the workload description,
  not as a hidden constant.
- `storage_mode=segmented_store` measures a durable multi-segment index whose
  WAL, checkpoint, source segments, and per-segment HNSW sidecars live in the
  lower storage layer while query-time search merges warm per-segment indexes.
  It is not equivalent to direct file or mmap search.

Large datasets that do not fit comfortably in RAM should be benchmarked as
storage workloads first. An in-memory row is still useful as an upper bound, but
it should not be compared directly to file or mmap rows.

## References

- [Faiss `OnDiskInvertedLists`](https://faiss.ai/cpp_api/file/OnDiskInvertedLists_8h.html):
  mmap-backed IVF lists with prefetch support.
- [DiskANN I/O optimization survey](https://arxiv.org/html/2602.21514v2):
  memory-resident navigation, hot-node cache, SSD graph/vector payloads, and
  page-level layout constraints.
- [SPFresh](https://arxiv.org/html/2410.14452v1) and
  [Weaviate HFresh](https://docs.weaviate.io/weaviate/concepts/vector-index#hfresh-index):
  centroid navigation over disk posting lists, with local rebalancing for
  updates in the SPFresh lineage.
- [Turbopuffer ANN v3](https://turbopuffer.com/blog/ann-v3): object storage as
  durable source of truth, SSD/memory as caches, and warm/cold query behavior
  as part of the API contract.
