# What the competing implementations do, and what their tests catch

Status: 2026-11-02. Read from source, not from documentation. Clones:

| library | commit | date |
| --- | --- | --- |
| hnswlib | `ca42672` | 2026-09-14 |
| usearch | `33d3ab6` | 2026-10-31 |
| annoy | `379f744` | 2025-10-29 |

The point of this page is not admiration. Each item below is either a technique
worth copying, a test invariant we do not currently assert, or a design choice
we should consciously reject.

## hnswlib

**Implementation notes.**

- `updatePoint(dataPoint, internalId, updateNeighborProbability)`
  (`hnswalg.h:1109`) overwrites the vector in place, then re-picks neighbours
  within one and two hops, accepting each new neighbour with a probability. It
  is an approximate repair rather than a rebuild: cheap, and it deliberately
  leaves some stale edges.
- `allow_replace_deleted` (`hnswalg.h:74`, `:899`) keeps a set of deleted
  internal ids and reuses them for new inserts, so a long-lived index can stay at
  a fixed size instead of accumulating tombstones.
- `resizeIndex` (`hnswalg.h:664`) grows the preallocated element arrays. We do
  not need this: our vectors live in `Vec` and grow, which is the same capability
  without the API.
- `BaseFilterFunctor` (`hnswlib.h:272`) is a per-search predicate, and
  `BaseSearchStopCondition` (`hnswlib.h:279`, implemented in
  `stop_condition.h`) is a pluggable stopping rule. Both are *interfaces*, so a
  caller can supply their own; our equivalents are fixed internal strategies.
- `num_deleted_` is an atomic and searches take a filter pointer, so search and
  delete can run concurrently.

**What its tests cover that ours do not.**

- `bindings_test_replace.py` and `bindings_test_stress_mt_replace.py`: replacing
  deleted slots, including under concurrent inserts.
- `bindings_test_resize.py`: capacity growth without losing the graph.
- `bindings_test_pickle.py`: save/load roundtrip across processes.
- `bindings_test_recall.py`: asserts a recall threshold, then saves and reloads
  the index, then asserts `recall_before == recall_after` exactly (`:98`). The
  saved file is the brute-force index rather than the graph, but the invariant is
  the right one: a reload must not change answers at all.

## usearch

**Implementation notes.**

- `reserve` (`index.hpp:1096`, `:1438`, `:1551`) preallocates capacity and
  reports failure rather than growing silently, which lets a server bound its
  memory exactly.
- Progress callbacks that can *cancel* a build (its Python tests assert both the
  firing and the cancellation path). A multi-minute build that cannot be stopped
  is a usability problem we have: `build()` either returns or does not.
- Tests exercise GIL release during add and search, and a
  `test_concurrent_access_serializes_safely` case: the API documents its
  concurrency contract and tests it.

## annoy

**Implementation notes.**

- `on_disk_build` (`annoylib.h:1012`) `mmap`s the node array into a file and
  grows it with `ftruncate` + remap as the forest is built. The index is never
  held entirely in anonymous memory, so the page cache bounds the working set.
  This is the technique behind our 100M blocker: our build peak is ~200 GiB
  because the corpus, the graph and the reorder are all anonymous allocations.
- `on_disk_build` also inserts the roots' copies at the end and remaps
  (`:1055-1070`), so the file is the index, not a serialization of it.

**What its tests cover that ours do not.**

- `threading_test.py`: concurrent search over a shared index.
- `on_disk_build_test.py`: the index works *from* the file it was built into.
- `accuracy_test.py`: per-metric accuracy floors.

## Capability matrix

| capability | hnswlib | usearch | annoy | vicinity |
| --- | --- | --- | --- | --- |
| build into a file, bounded heap | no | no | **yes** | **no** |
| replace vector for an existing id | **yes** (`updatePoint`) | `update` | no | **no** |
| reuse deleted slots | **yes** | no | no | **yes** (`InPlaceIndex`) |
| pluggable stop condition | **yes** (virtual) | no | no | partial (`adaptive`, `adsampling`) |
| per-search filter predicate | **yes** | yes | no | **yes** (`search_with_filter`, `search_acorn`) |
| cancellable build | no | **yes** | no | **no** |
| capacity reserve with failure | **yes** | **yes** | no | no (Vec growth) |
| typed errors | no (`Status` strings) | partial | no | **yes** |
| one crate, many families | no | no | no | **yes** |

## Second pass: the layouts, in bytes

The capability list above is the shallow difference. The deeper one is how each
implementation arranges memory, and it explains both our resident-memory number
and our load cost.

hnswlib allocates **one block per element**, sized from its own level
(`hnswalg.h:126-129`, `:151`):

| part | bytes at `d = 96`, `M = 16` |
| --- | ---: |
| layer-0 links (`maxM0 * 4 + 4`) | 132 |
| vector | 384 |
| label | 4 |
| upper layers, amortised (`M*4+4` per node above level 0, `m_L = 0.36`) | ~24 |
| **total** | **~544** |

We hold **1292 B/point** at 1M `d = 96` (committed row `mem_mib` 1231.8), and
the serialized form is 492 B/point. The gap is not the vector or the edges: it
is that every layer allocates a 136-byte inline neighbour slot for *every* node,
while hnswlib sizes the block from the node's own level. That is the same defect
the storage audit measured at −55% when upper layers were frozen, seen from the
other library's side.

Two consequences follow, and they are the same consequence:

1. **Loads.** hnswlib's file is its memory image: `loadIndex` reads the arrays and
   searches them, no reconstruction (`hnswalg.h:783`). Our segment loader reads
   dimension-major vectors and rebuilds AoS plus per-layer lists, which is why
   the audit lists snapshot reload as an unmeasured cost axis. A level-sized or
   CSR layout with a fixed record size could be mapped and used directly, which
   would make restart a page-fault rather than a rebuild.
2. **Writes.** hnswlib holds `std::vector<std::mutex> link_list_locks_` indexed
   by element (`hnswalg.h:49`, `:103`) plus an atomic size, so `addPoint` is
   safe from many threads. Our insert path takes `&mut self`; the streaming
   coordinator serialises, and the parallel feature batches a build rather than
   admitting concurrent writers. That is a design difference, not an oversight,
   but it is why a server in front of this crate needs its own writer lock.

One more technique is worth naming because it is unusual. Annoy reuses the vector
slot of small internal nodes to hold the list of descendant ids
(`annoylib.h:452-470`), with `K` computed from the node size. That trick saves
`f * sizeof(T)` per small node. Not worth copying here, since our tree families
do not store vectors at internal nodes at all.

### Neighbour selection is at parity, and worth proving

hnswlib's `getNeighborsByHeuristic2` (`hnswalg.h:475-507`) accepts a candidate
only if no already-selected neighbour is closer to it than it is to the query,
which is the relative-neighbourhood rule at `alpha = 1`, then optionally fills
the remainder from the pruned list (`keepPrunedConnections`). We implement the
same rule (`src/hnsw/construction.rs:27` `select_neighbors_rnd`), plus `alpha`
relaxation (`:171`) and maximum-oriented diversification (`:89`).

So we offer a superset, and have never shown that the base case reproduces the
reference exactly. A parity test would settle it: build one fixed graph with both
at `M = 16`, `ef_construction = 200`, then compare neighbour sets edge by edge.
That turns "we implement the same rule" into "the rules produce the same graph".
Right now this repository keeps making that claim without settling it.

## What to copy, in order

1. **Build into a file (annoy's `on_disk_build`).** It is the only listed
   technique that changes a scale answer rather than a convenience. Our 100M
   build needs ~200 GiB against 128 GiB installed; mapping the node array and
   growing it by remap moves that into page cache, which the kernel can evict.
   Lands in `large_scale`'s out-of-core item, not in `HNSWIndex::build`.
2. **Replace a vector in place (hnswlib's `updatePoint`).** We can delete and
   insert, but a caller with a mutable corpus and a fixed doc id has no way to
   say "this id now has a new vector". The probabilistic one-and-two-hop repair
   is the part worth copying: it bounds the work per update, which is what makes
   updates cheaper than rebuilds.
3. **Reuse deleted slots (hnswlib's `allow_replace_deleted`).** Checked after
   writing this list: `InPlaceIndex` already does it (`src/hnsw/inplace.rs:178`
   pops a free slot on insert, `:370` pushes one on delete, and the free list is
   validated on load at `:978`). What does not is the resident `HNSWIndex`, which
   is build-once by construction: `add_slice` refuses after `build()`, so its
   tombstones can only be reclaimed by `delete_with_repair`. That is a different
   life-cycle, not a missing free list, and the row above is now accurate about
   which family has which.
4. **A cancellable build (usearch).** Long builds are opaque today. A callback
   that returns "continue or stop" costs little and makes the 10M tier usable
   interactively.
5. **Pluggable stop condition (hnswlib).** We have `adaptive` and `adsampling`
   as internal strategies; exposing the choice is what the 2026 declarative
   recall work assumes a benchmark can do.

## What their tests assert and ours should

- **Recall as an assertion per metric**, not only as a benchmark row: hnswlib
  checks each space has an accuracy floor in CI. Our recall floors live in
  `tests/recall*` and in the ratchet; making one run per PR on a fixed fixture
  would catch a regression before a weekly job does.
- **Replace/resize roundtrips**: we have no test that an id keeps working after
  its vector changes, because we cannot change it.
- **Concurrent search over one index**: annoy's `threading_test.py` is 30 lines
  and would have caught the `&mut self` cache limitation that the storage audit
  documented from reading code. Closed: `concurrent_search_over_one_index_returns_identical_results`
  runs 8 threads x 32 searches over one shared `&HNSWIndex`.
- **Exact result equality across a reload on the segment path.** We do assert
  this for `deg`, `dual_branch`, `inplace`, `sq4u`, `sq8u` and `symphony_qg`,
  each with a `save_load_roundtrip_preserves_search` test. The segment writer,
  which is the path the store and the large-scale harness use, checked two top-1
  ids and the tombstone flags only. Closed:
  `segment_reload_returns_identical_results` compares every id and distance of 40
  queries x 10 results, through both `load_index` and `HNSWSegmentSearcher`.
  Both closures are in PR #98.

## Second corpus: FAISS, ScaNN, and the ANN servers

Read the same way, at pinned commits: FAISS `b80093c`, ScaNN `83f1e7b`, Qdrant
`222b2e4`, LanceDB `8ee7698` with the `lance` engine at `66054f1`, pgvector
`1d5ef0a`. Full reports, with file:line for every claim, are in
`.agents/reports/competitor-faiss-scann-20261102.md` and
`.agents/reports/competitor-servers-20261102.md`.

Three things they do that this crate does not, in order of what they change:

- **PQ 8-bit table path.** FAISS's FastScan is 4-bit only and its LUT is scaled
  with `a = min(255/span, (65535 - 0.5M)/sum_span)`
  (`faiss/utils/quantize_lut.cpp:63-73`); the headroom term is what keeps a
  uint16 accumulator from overflowing across `M` terms. We quantise a LUT and
  pack 4-bit already (`src/pq_simd.rs:515`); the missing piece is the overflow
  budget per `M`, which is arithmetic, not architecture.
- **PQ codes wider than 8 bits.** `code_size = (nbits*M + 7)/8` with `ksub =
  1 << nbits` (`faiss/impl/ProductQuantizer.cpp:73-75`). Accepting `nbits` above
  8 with bit-packed codes is a construction-time decision; rejecting unsupported
  widths mid-training is not.
- **Anisotropic noise shaping as a post-pass.** ScaNN's cost is
  `eta * parallel + perpendicular` with a closed-form `eta`
  (`scann/hashes/internal/stacked_quantizers.h:79-85`), and up to ten
  improvement rounds per block (`:196-210`). It runs on codes that already
  exist, so it is a flag on any IVF-PQ/AVQ index rather than a new family.

Two things we now do too, plus one correction to the earlier reading:

- **`ensure_topk_full`.** FAISS lifts the scan budget so a filter cannot leave
  the result short (`faiss/IndexIVF.cpp:597`); Lance re-runs the query exactly
  when the filtered path under-delivers (`hnsw/builder.rs:1734`). We returned
  fewer than `k` hits in that case until PR #100, which adds the exact
  fallback.
- **Filter masks are materialised.** Lance builds a mask of matching row addresses before
  touching the index (`lance/prefilter.rs:27-81`) and picks its path from that
  mask's cardinality. Qdrant is the one that estimates cardinality before
  materialising, using an Agresti-Coull interval over a 1000-id sample
  (`qdrant/.../dispatch.rs:161-170`). Our 10% cutoff is the same threshold Lance
  uses, so the threshold was never the gap; the missing fallback was.
- **Annoy and hnswlib both allow in-place vector replacement; we do not.** That
  remains the largest semantic difference, and it is a build-once design
  decision, not an oversight.

What not to copy from this group, because each is a documented trap: pgvector's
session-wide `hnsw.ef_search` / `iterative_scan` GUCs (two queries in one
statement cannot use different budgets), Lance's six overlapping probe knobs
where any `search_effort` silently disables `nprobes`, Qdrant's 24-variant
quantized-enum matrix, and pgvector's 10-heap-TID duplicate cap that silently
drops extra rows. ScaNN ships no C++ tests at all, so nothing about its numeric
kernels is evidence of correctness; only FAISS's and pgvector's assertions were
treated as test evidence here.

## What we should not copy

- `Status`/string errors. Our typed `RetrieveError` is a real advantage; the
  hnswlib issue tracker is full of callers parsing message text.
- Preallocated fixed-capacity arrays. Rust's `Vec` growth gives the same
  capability without a `resizeIndex` call, and without the failure mode where a
  forgotten resize aborts a build.
