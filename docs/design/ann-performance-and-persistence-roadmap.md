# ANN performance and persistence roadmap

Status: proposal
Scope: benchmark coverage, profiling, search performance, persistence and interchange
Grounded-in: `docs/benchmark-results.md`, `docs/persistence.md`, `docs/algorithms.md`,
`README.md`, the benchmark registry in `examples/ann_benchmark/support.rs`, and
commits `f19fca3`, `58f097b`, `8924696`
Review trigger: after the first representative-data sweep or before introducing
a new on-disk format, cache, or public persistence API

## Where we are

### Done

- The README is short and first-use oriented; deeper algorithm and benchmark
  detail lives in `docs/`.
- The dense benchmark harness has a registry-derived `--all-dense` selection and
  an opt-in `--require-complete` gate. CI runs it with all features, batch mode,
  snapshot reload, JSONL output, and a fresh result path.
- The current strict fixture produced 99 rows, 47 emitted labels, and five
  storage modes. This is registry coverage, not coverage of every public API.
- DiskANN now validates an independent exact L2 oracle, distinct/finite/sorted
  results, and neighbor-set parity across memory, file, mmap, page-file, and
  page-mmap paths. Its five-path rows are required by the benchmark completeness
  gate when `benchmark` is enabled.
- LSM restart snapshots reject duplicate tombstone IDs instead of silently
  collapsing malformed input. The malformed-snapshot regression is red against
  the old loader and green with the fix.
- The first DiskANN storage measurements and profiles are durable. At `ef=75` on
  a small warm-cache fixture, page layouts were slower than the existing
  separate-file and mmap paths. The direct-file profile is dominated by repeated
  `pread`/`read_vector` samples.
- Persistence documentation distinguishes JSONL/HDF5/possible NumPy
  interchange, algorithm-native layouts, `segstore` segmented durability, and
  the current multi-file snapshot publication limits.

### Verified open work

- The benchmark registry does not cover every public search shape. HNSW adaptive,
  flat-batch, selectivity-routing, and comparable filtered workloads remain
  unmeasured in the current strict harness. Confirmed by the harness/docs rather
  than inferred from a missing filename.
- The current Criterion suite has focused targets, not one quality-and-timing
  contract for every registered family. Existing targets include HNSW, ACORN,
  distance, PQ SIMD, IVF-PQ, recall, memory, scaling, ADSampling, store, and
  DiskANN.
- LEMUR still needs a trained-model/multi-vector workload. EVoC is clustering,
  not an ANN search row. SparseMIPS has a separate harness and needs real sparse
  benchmark data for a meaningful cross-family comparison.
- Multi-file snapshot replacement is not one atomic published generation across
  all index families. `segstore` WAL/checkpoint guarantees must not be implied for
  ordinary snapshot directories.
- No `.npy` Rust importer exists. Adding one is an interchange decision, not a
  prerequisite for making native search indexes fast.

## Ordering principles

1. Measure a user-paid operation before changing its implementation.
2. Keep exactness, recall, IDs, deletion visibility, and persistence parity as
   non-negotiable gates for speed work.
3. Separate build, open/reload, steady-state search, update, checkpoint, and
   compaction costs. A faster benchmark setup is not a faster query path.
4. Prefer reversible configuration, benchmark, and profiling changes before
   one-way file-format or public-API changes.
5. Do not add implementations merely to increase a count. A new method earns its
   place only when it has a clear consumer, a quality oracle, a persistence
   contract, and a benchmark that can compare it fairly.

## Phases

### 0. Freeze the evidence contract

Owner/consumer: benchmark maintainers and README readers.

Write a small machine-readable coverage matrix derived from the registry and
the implementation dispatch. For each selection, record: feature requirements,
metric assumptions, dataset kind, query API, exact-oracle availability, build
mode, search modes, update modes, persistence modes, and whether the row belongs
in the dense or sparse harness. Keep the registry as the source of truth; the
matrix must not become a second hand-maintained algorithm list.

Add negative fixtures for feature-disabled methods, metric-incompatible methods,
unknown selections, missing storage rows, and missing quality oracles. Preserve
the current strict fresh/resume semantics so stale JSONL cannot make a partial
run look complete.

Gate: every registry selection is classified as `covered`, `separate-harness`,
`not-ANN`, or `blocked-with-reason`; strict CI fails on an unclassified row.
This phase is reversible and should not change a library search algorithm.

### 1. Build the representative measurement grid

Owner/consumer: performance work and future maintainers comparing results.

For each covered family, choose at least one representative real dataset and a
small deterministic fixture. Record dimensionality, corpus/query counts,
metric, normalization, `k`, recall target, build seed, feature set, CPU, and
cache state. Use the small fixture for CI correctness and the real fixture for
performance decisions.

For every search family, measure at least:

- one normal/prunable query shape;
- one forced-exact or full-exploration control;
- at least two search-depth points around the intended recall target;
- warm-cache and cold/open-cost boundaries where storage is involved;
- serial and parallel modes when both are public options.

For mutable indexes, add insert/delete, checkpoint/reopen, compaction, and
post-reopen search rows. For compressed indexes, separate code-scan,
decompression, rerank, and raw-vector I/O costs where the implementation exposes
them. Criterion means are not latency percentiles; record p50/p95/p99 only when
the harness measures them consistently.

Gate: each family has an exact or explicitly justified quality oracle, three
repeat runs, a variance record, and a negative control. No optimization is kept
from one synthetic shape alone.

### 2. Profile before optimizing

Owner/consumer: the maintainer of the specific hot path.

Use release/bench binaries with symbols, frame pointers when needed, and
criterion `--profile-time` so setup and analysis do not dominate. Store raw
profiles outside the repository; commit only the concise attribution and the
reproduction command.

First targets, in dependency order:

1. DiskANN direct-file `pread`/`read_vector` behavior, with mmap and memory as
   controls. Test bounded read-ahead or coalescing only if the profile remains
   I/O dominated on a larger corpus.
2. HNSW frontier/heap updates and dense distance dispatch, using the existing
   `ef=10/50/100/200` and full-exploration controls. Do not widen unsafe or
   replace the safe `innr` kernel without a measured win.
3. IVF-PQ/AVQ code scan, ADC-table construction, rerank, allocation, and file
   versus mmap reads. Existing profiled counters should be aligned with the
   common benchmark output before changing kernels.
4. Store/LSM checkpoint, sidecar rebuild, WAL replay, and post-reopen query
   costs. Keep durability costs visible rather than hiding them in a search
   number.

Gate: a candidate has a named hotspot, a before measurement, an invariant test,
and a negative-control measurement. If two hypotheses fail for the same reason,
fix the measurement model before trying a third implementation change.

### 3. Optimize one path at a time

Owner/consumer: the path's benchmark and its production caller.

For each candidate, use a one-change experiment:

- state the hypothesis and expected metric;
- implement the smallest safe change;
- run exactness/recall/parity tests;
- repeat the baseline and controls with identical data and build flags;
- keep only a meaningful, repeatable improvement without a material memory or
  durability regression;
- record rejected experiments so the same attractive idea is not retried.

The first likely experiment is a bounded direct-file vector-read cache or
coalescing strategy. It must define memory limits, eviction behavior, thread
safety, and cold-cache semantics before implementation. The page layout is a
negative control in this decision: current warm-cache data does not justify
promoting it.

Gate: same-corpus median improves beyond measurement noise, recall and result
parity remain unchanged, and mmap/memory controls do not regress. This phase is
partially reversible; any persisted layout change is deferred to the fork below.

### 4. Decide persistence generation semantics before broadening formats

Owner/consumer: users reopening indexes after a crash, operators serving a
snapshot, and every native persistence implementation.

This is the first structural fork. Do not add a universal persistence layer or
claim crash-safe replacement until an ADR answers it.

**Fork A — How is a multi-file generation published?**

- **Recommended:** write a new generation directory, validate it, fsync files
  and directory, then atomically publish one manifest/current pointer. Keep old
  generations for recovery. This gives one publication point while allowing
  algorithm-native component files.
- **Per-file replacement:** smaller change, but readers can observe mixed
  generations after interruption and recovery is weaker.
- **Single-container format:** one publication unit, but it can impose copying,
  random-access, and migration costs on large graph/posting indexes.

The recommendation preserves native graph/posting-list layouts while making
publication explicit. It must be reconciled with `segstore` rather than
replacing its WAL/checkpoint contract.

Gate before implementation: ADR records generation identity, manifest schema,
fsync requirements, reader behavior during publication, garbage collection,
interruption recovery tests, and compatibility/version policy.

### 5. Decide interchange scope separately

Owner/consumer: users moving dense arrays into the library, not serving native
indexes.

**Fork B — Should `.npy` become a supported importer?**

- **Recommended later:** add a read-only dense-array importer only after a real
  consumer and dataset workflow need it. Validate shape, dtype, byte order,
  dimensions, IDs, file bounds, and reject object/pickle payloads; convert into
  the internal VEC1 path without changing native index persistence.
- **Keep HDF5 + JSONL/VEC1:** lowest surface area while covering benchmark and
  CLI workflows.
- **Add a broader interchange crate:** convenient for users, but adds dependency,
  format-security, and maintenance surface before demand is proven.

No `.npy` support should be implemented merely because it is familiar. Gate on
one named consumer, a bounded dependency/security review, and round-trip tests
against canonical NumPy-produced files.

### 6. Make docs and CI the release ratchet

Owner/consumer: outside readers and contributors.

Update README only for stable user-facing behavior. Keep experimental method
claims, benchmark tables, persistence matrices, and reproduction commands in
their focused docs. CI should run:

- strict all-dense registry coverage;
- separate sparse harness;
- focused Criterion smoke/quality controls for each benchmark target;
- persistence corruption, round-trip, reopen, and duplicate-record tests;
- Python plotting/data-tool tests and docs/link checks.

Gate: a fresh-clone reader can follow README links, run the smallest example,
understand which methods are experimental, and reproduce at least one benchmark
without relying on private paths or unstated machine settings.

## Backlog, not current phases

- Real sparse SPLADE/BM25-style data for SparseMIPS.
- Trained multi-vector LEMUR workload.
- Comparable filtered/selectivity datasets and public batch/adaptive API rows.
- Native generation publication ADR and interruption-injection harness.
- `.npy` importer evaluation after a named consumer appears.
- Additional algorithm implementations only when a concrete workload exposes a
  gap that existing methods cannot serve.

## Decision guardrail

Do not start Phase 3's direct-file cache/read-ahead implementation until Phase 2
has a larger representative DiskANN profile and the cache/coalescing contract is
written down. Do not start Phase 4 or Phase 5 format work until Fork A has an
accepted ADR. Do not start `.npy` work until Fork B has a named consumer and a
security/compatibility gate.
