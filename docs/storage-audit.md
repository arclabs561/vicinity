# Storage audit: what each family does with bytes at rest

Status: audit, 2026-11-01, at `9a6f615`. Read alongside
[benchmark-atlas](benchmark-atlas.md) and
[benchmark results](benchmark-results.md).

A benchmark row that only ever measures an in-RAM index answers a question
nobody asks above a few million vectors. This audit records, per family, what is
written, what is read back, and whether a query can run **without the whole
index in RAM**. That last property, not recall, is what decides whether a family
is usable at 100M.

## The three storage properties

| Property | Meaning |
| --- | --- |
| **snapshot** | `save_*` / `load_*` exist and reload into RAM. Storage cost only. |
| **file search** | A `*Searcher` reads records from disk during the query (`pread` through a bounded block cache). The graph and vectors stay off-heap. |
| **build in RAM** | Construction materialises the corpus and the full adjacency in memory. Currently true of every family. |

`mmap` is deliberately absent: `DiskGraphReader::open_mmap` and
`IVFAVQFileSearcher::open_mmap` are deprecated aliases that call the positional
read path (`src/diskann/disk_io.rs:168`, `src/ivf_avq/search.rs:595`,
"vicinity no longer memory-maps index files"). Page-cache behaviour is therefore
ours to bound rather than the kernel's to guess, and the block cache is what a
row's latency actually reflects.

## Matrix

Evidence is the module named in each row; `save`/`load` means a public
`save_to_dir`/`load_from_dir` (or the JSON `save_to_file`) pair.

| Family | Snapshot | File search | Block cache | Notes |
| --- | --- | --- | --- | --- |
| `hnsw` | yes (`persistence::hnsw` segments, JSON) | **no** | n/a | In-RAM only; the largest resident footprint of any family (below). |
| `hnsw` in-place / `MappedInPlaceIndex` | yes (`save_to_file`) | no | n/a | `MappedInPlaceIndex` maps **external ids**, not memory. A naming trap for anyone reading the API. |
| `sq4u`, `sq8u`, `symphony_qg` | yes | no | n/a | Quantised traversal over codes held in RAM; raw vectors kept for rerank. |
| `diskann` | yes (generation dir) | **yes**: `DiskANNSearcher`, `DiskANNPageSearcher` | yes, 64 MiB default, 4 KiB blocks | Two layouts (plain record, page-aligned) × two read paths. |
| `ivf_pq` | yes | **yes**: `IVFPQFileSearcher` | yes | Codes and raw vectors on disk; centroids and `doc_ids` in RAM. |
| `ivf_avq` | yes | **yes**: `IVFAVQFileSearcher` | yes | Partitions and raw vectors on disk; centroids and `doc_ids` in RAM. |
| `ivf_rabitq` | yes | no | n/a | Two shapes from format v3: a live index writes vectors and rebuilds codes on load; a compacted index writes `codes.bin` and reloads compacted. The code file carries a checksum, because a flipped bit decodes to a different valid code. |
| `vamana`, `nsg`, `nsw`, `sng`, `emg`, `finger`, `pipnn`, `fresh_graph` | yes | no | n/a | Snapshot → heap. |
| `curator`, `filtered_graph`, `range_filtered`, `lsh`, `rp_quant`, `binary_index`, `sq4` | yes | no | n/a | Snapshot → heap. |
| `kdtree`, `balltree`, `rptree`, `rp_forest`, `kmeans_tree` | yes | no | n/a | Snapshot → heap. |
| `store` (segstore) | yes, with compaction | no | n/a | Segmented mutable store; compaction rewrites segments, no query-time file reader. |
| `streaming` (`IndexOps`) | via owned `Vec` inserts | no | n/a | Implemented for `InPlaceIndex`/`MappedInPlaceIndex` only. |
| `sparse_mips` | yes | no | n/a | Separate SPV1 harness. |
| `evoc`, `lemur` | none | none | n/a | Not k-NN query paths; excluded from every dense comparison. |

Three of twenty-odd families can answer a query without the index in RAM. Nine
of the ten graph families cannot, which is exactly the class where the
literature's billion-point results come from.

## What the numbers say about the layout

At 1M points, 96 dimensions, `m = 16` (`benchmarks/large-scale-1m-clustered.jsonl`):

| | HNSW | DiskANN |
| --- | ---: | ---: |
| index in RAM (`mem_mib`) | 1231.8 MiB | 507.4 MiB |
| index on disk (`index_disk_mib`) | 469.5 MiB | 495.9 MiB |
| corpus | 366.2 MiB | 366.2 MiB |

Two things follow.

**HNSW's serialized form is 2.6× smaller than its resident form.** The file holds
vectors once (366 MiB) plus edges at their actual degree (~68 MiB). The resident
form holds 136 B of inline neighbour slots *per node per layer*: a
`SmallVec<[u32; 32]>` allocated for every node in every layer
(`src/hnsw/construction.rs:297`, `Layer::new_uncompressed(vec![NeighborList::new(); n])`)
— plus a `doc_id → index` hash map and `Vec` growth slack. At 1M with the five
or six layers a 1M build produces, that is roughly 700–800 MiB of slots, of which
only the few percent of nodes assigned above layer 0 can ever hold an edge.

That is not an estimate any more. Building the same 1M index with
`compact_upper_layers` on (upper layers frozen into eligible-node CSR, edges at
their actual degree) and off, same seed, same corpus, same ground truth:

| `ef` | recall (off / on) | p50 µs (off / on) | index MiB (off / on) |
| ---: | --- | --- | --- |
| 50 | 0.4784 / 0.4784 | 124.0 / 125.3 | 1231.8 / **549.8** |
| 200 | 0.8074 / 0.8074 | 386.9 / 385.2 | 1231.8 / **549.8** |
| 800 | 0.9690 / 0.9690 | 1237.2 / 1239.5 | 1231.8 / **549.8** |

Identical recall at every point, p50 within 1%, build 55.4 s against 54.2 s, and
**55% less resident memory**: 0.55 KiB/point that was never data. Rows in
`benchmarks/large-scale-1m-hnsw-{layout,compact}.jsonl`.

Process RSS does not fall with it (2111 MiB either way), because the freed
per-node allocations go back to the allocator rather than to the OS. That is a
measurement trap: a change can halve the index and leave `ps` unchanged.

The same trick applies to layer 0, which the `compact-hnsw` feature deliberately
does not touch, and it is off by default even for the upper layers.

**DiskANN is the only family whose disk number is honest.** In RAM it is 507 MiB
for the same 1M, 1.21 KiB/point against HNSW's 2.09, because it stores one layer
at degree 32 rather than a hierarchy, and it pays for that with recall on
clustered data (0.197 against 0.807 at `ef = 200`). Its on-disk size is larger
than HNSW's because each record pads neighbours to `R`, the price of
positional reads.

## Findings

1. **Build is the memory ceiling, not search.** Every family's `add_slice`
   appends to a `Vec<f32>` / `Vec<Vec<f32>>` and the builders keep the full
   adjacency. The 100M refusal is arithmetic on that: 2.09 KiB/point measured at
   1M. File-backed search does not help, because by the time you can search from
   disk you have already had to build in RAM.

2. **The default layout of an HNSW index wastes ~60% of its resident memory** on fixed-width
   inline slots across layers (above). This is an implementation choice, not a
   property of HNSW. The same graph in CSR is what the serialized file already
   is.

3. **A file-backed HNSW is the storage win with the widest reach.**
   HNSW is the default family and the first row every benchmark measures, and
   it has no reader. The pieces exist: the segment writer already emits
   `vectors.bin` (SoA), `doc_ids.bin`, `layers.bin` (count-prefixed per-node
   lists) and `metadata.bin`, and `file_io::CachedFile` is a generic block cache
   over `pread`. What is missing is an offsets table for `layers.bin` (records
   are variable length) and a `HNSWSegmentSearcher` that uses it.

4. **The block cache is per-searcher and `&mut self`** (`src/file_io.rs:1-10`,
   "Every searcher owns its cache and searches through `&mut self`"). Parallel
   search therefore multiplies the cache by the thread count and cannot share
   a hot block; a 16-thread server with the default budget would hold 16 × 64 MiB
   of duplicated blocks, and serialising queries through one `&mut` would cap
   throughput at one core.

5. **RaBitQ could not be persisted after compaction** — fixed, and worth
   recording because the reason was widely mis-stated. It was not that the codes
   were unserialisable: `compact()` frees the f32 corpus, and the format was
   rebuild-on-load, so the only input it had was gone. Serialising the codes
   needed `qntz` to own the codec, since an `EdgeQuantizedVector` is only
   meaningful if its codes were built against a zero centroid and its
   `ip_parent_rot_codes` belongs to the parent used for the residual, and the
   type is `#[non_exhaustive]` with no public constructor. `qntz` 0.3.0 adds
   `to_bytes` / `from_bytes` on both types; `vicinity` format v3 writes
   `codes.bin` for a compacted index and reloads it compacted. The derived
   `codes` vector is recomputed on load rather than stored, and the file carries
   an FNV-1a checksum, because a flipped bit inside an edge code decodes to a
   different valid code and nothing else in the format would notice.

6. **No family compresses at rest.** Index files are raw `f32` vectors and `u32`
   ids. That is the right default for latency, but it means disk bytes ≈ corpus
   bytes × (1 + edges), and the 100M tier needs 129 GiB of dataset plus indexes
   for two families.

7. **Snapshot reload is not free and is not measured as a cost axis.** Every
   family reports `build_s`, but only the large-scale harness reports
   `persist_s` (0.418 s for 469.5 MiB at 1M). A restart in production pays both,
   and the two are conflated in every row that lacks `persist_s`.

8. **`storage_mode` exists in the older harness and is missing from the unified
   one.** `examples/ann_benchmark.rs` records `in_memory`, `snapshot_loaded`,
   `file`, `segmented_store`; `all_families` and `large_scale` rows carry no such
   column, so a reader cannot tell from a row whether it measured a file path.
   The 1M table above had to be read from the family name (`diskann-file`).

## Priorities

Ranked by coverage gained per unit of risk.

1. **Sparse upper layers, as an explicit build step rather than a default.**
   Measured at −55% resident memory with no recall or latency change (above).
   It cannot simply become the default: freezing a layer makes it read-only
   (`NeighborStorage::FrozenCsr` has no mutable path), so `delete_with_repair`
   and the batch repair both return `Unsupported`, and an index that is still
   being inserted into cannot use it. What it needs is a named call —
   `freeze_upper_layers()` after the last insert — and, if incremental use
   matters, a `thaw()` that rebuilds uncompressed lists. Saving is not the
   constraint: the writer walks node ids and emits an empty list for nodes absent
   from a layer, so a compacted segment is still dense by id on disk and reads
   back like any other (the segment searcher has a test for exactly that).
   Layer 0 is the larger half and is untouched by the flag.
2. **`HNSWSegmentSearcher`.** Done: offsets table plus a reader over the existing
   files, preferring an AoS vector sidecar. Turns the default family into a
   file-searchable one. Measured at 100k x 96: 39 µs resident against 104 µs
   file-backed at a 64 MiB cache (2.7x), 518 µs at 4 MiB (12.3x).
3. **`storage_mode` in every harness row.** Cheap, and without it the audit
   above is a manual reading of family names.
4. **Shared, thread-safe block cache.** Removes the N× cache multiplication and
   the single-writer bottleneck; needed before any parallel file-backed claim.
5. ~~**RaBitQ code persistence.**~~ Done. `qntz` 0.3.0 exposes the codec and
   `ivf_rabitq` format v3 writes `codes.bin`; the fixed-memory tier can now be
   written to disk. The next item for this family is a file-searchable reader,
   not a writer.
6. **Out-of-core build.** The only item that changes the 100M answer, and by far
   the largest: sharded construction plus cross-shard stitching, as DiskANN does
   with k-means partitions and a merge pass.

## Not covered

- No measurement of device-level I/O: the block cache hides the device, and at
  1M the index still fits in the OS page cache, so every "file" number above is a number
  from the OS page cache, with a user-space cache in front of it.
- No multi-process or concurrent-reader test.
- Durability of the snapshot writers (generation publication, crash consistency)
  is a separate audit; `persistence/generation.rs` and `store.rs` compaction are
  not reviewed here.
