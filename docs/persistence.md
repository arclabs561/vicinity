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

## Capability Matrix

| Index family | Save/load | Memory search | File search | Mmap search | Updates | Storage direction |
| --- | --- | --- | --- | --- | --- | --- |
| HNSW | JSON via `serde`; binary segments via `persistence` | Yes | No | No | Tombstones and repair; `store` for durable segments | Keep JSON and binary segment paths. Use `store` for durable segmented HNSW. |
| `store::UpdatableIndex` | Yes, via `segstore` + HNSW sidecars | Segment sidecars loaded into memory | No | No | Add/delete/compact/checkpoint | Keep on `segstore`; this is the segmented-HNSW path. |
| DiskANN | Yes, graph + vector files | Yes | Yes | Yes | Build-once | Mmap exists for current separate graph/vector files. Page/co-location layout remains next. Do not route through `segstore`. |
| NSW / SNG / Vamana / NSG / FINGER / PiPNN / EMG | Yes, directory format | Yes | No | No | Build-once | Persists the built in-memory graph state and restores it directly. This is snapshot-memory persistence, not file-backed search. |
| HNSW quantized variants | No | Yes | No | No | Build-once | Persist derived quantization payloads only after the in-memory search mode is benchmarked and promoted. |
| IVF-PQ | Yes, directory format | Yes | No | No | Build-once | Persists centroids, PQ codebooks, posting lists, codes, and optional raw vectors for rerank. |
| IVF-AVQ | Yes, directory format | Yes | No | No | Build-once | Persists centroids, AVQ codebooks, partitions, codes, and raw vectors for rerank. |
| IVF-RaBitQ | Yes, directory format for non-compacted indexes | Yes | No | No | Build-once | Persists raw vectors, centroids, and cluster membership, then rebuilds RaBitQ edge codes on load. |
| FreshGraph | Yes, snapshot directory format | Yes | No | No | Insert/delete/compact | Persists current graph state, tombstones, and inbound counts. WAL/checkpoint durability remains separate from `segstore`. |
| Curator | Yes, directory format | Yes | No | No | Build-once | Persists normalized vectors, doc IDs, labels, and parameters, then rebuilds the tree on load. |
| Range-filtered graph | Yes, directory format | Yes | No | No | Build-once | Persists normalized vectors and sorted attributes, then rebuilds HNSW on load. |
| Filtered graph | Yes, directory format | Yes | No | No | Build-once | Persists normalized vectors, graph neighbors, medoid, and inverted filter payloads. |
| BinaryFlat / RP-Quant | Yes, directory format | Yes | No | No | Build-once | Persists full-precision vectors and params, then rebuilds quantizer or projection-derived payloads on load. |
| SparseMIPS | Yes, directory format | Yes | No | No | Build-once | Persists sparse vectors in CSR-style offsets, indices, and values files plus the built graph. |
| streaming LSM / EVoC | No | Yes | No | No | Varies | Keep memory-only until each has a benchmark row and a concrete persistence consumer. |
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
2. Finish DiskANN storage modes: current file search, then mmap graph/vector
   readers, then page/co-location layout.
3. Extend IVF-PQ persistence from save/load to file-backed search. The current
   format persists:
   - manifest with format version, metric, dimensions, counts, and parameters
   - centroids
   - PQ codebooks
   - cluster doc IDs and PQ codes
   - 4-bit packed FastScan blocks rebuilt on load
   - optional raw normalized vectors when exact rerank should survive reload
4. Extend IVF-AVQ persistence from save/load to file-backed search.
5. Extend IVF-RaBitQ persistence to compacted indexes only after `qntz` exposes
   a safe serialized edge-code representation.
6. Extend range-filtered persistence to file-backed search only if the filtered
   benchmark contract needs it; current load rebuilds an in-memory HNSW.
7. Decide FreshGraph persistence separately. Its update model is not the same
   as segment append/compact, and forcing it through `segstore` would hide the
   in-place-update tradeoff.

## Benchmark Contract

Persistence work is not complete until the benchmark harness can distinguish:

- `storage_mode=in_memory`
- `storage_mode=file`
- `storage_mode=mmap`

Rows must include recall, QPS, build time, p50/p95/p99 latency, cache state, and
RSS. File and mmap rows should also include `load_time_s` and `index_bytes`
when the runner opens a saved index. For a method with both memory and
file-backed search, the harness should emit both rows from the same built index
when possible.

Storage rows are not interchangeable:

- `storage_mode=in_memory` measures the search path after vectors and graph
  structures are resident in process memory.
- `storage_mode=file` measures a searcher that reads persisted structures
  through normal file I/O. Report `load_time_s` separately from query latency.
- `storage_mode=mmap` measures a searcher opened over memory-mapped persisted
  structures. Treat OS page-cache state as part of the workload description,
  not as a hidden constant.

Large datasets that do not fit comfortably in RAM should be benchmarked as
storage workloads first. An in-memory row is still useful as an upper bound, but
it should not be compared directly to file or mmap rows.
