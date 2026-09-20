# Production-readiness rubric

Status: proposal
Scope: every public index, search path, persistence mode, benchmark row, and
Python-wrapper surface in `vicinity`

Production-level is a per-implementation claim. Compilation, registry presence,
or one synthetic speed result is not sufficient.

## Gates

### Correctness

- deterministic tests for valid, empty, malformed, wrong-dimension, duplicate-ID,
  and boundary inputs;
- exact-oracle or documented approximate-quality checks across dimensions,
  corpus sizes, `k`, search breadth, and metric assumptions;
- stable external IDs, sorted finite scores, unique results, and explicit ties;
- update/delete/compaction visibility tests for mutable methods;
- feature-matrix coverage and mutation/counterexample checks for pruning,
  quantization, filtering, and early termination.

### Performance

- release benchmark with setup outside the timed closure;
- at least three repeats with workload, hardware, compiler, features, cache state,
  and seed recorded;
- exact/simple baseline plus negative control;
- latency distribution when tail latency matters;
- separate build, open/reload, search, update, checkpoint, compaction, memory,
  and allocation costs;
- profile-backed hotspot and retained rejected-experiment record.

### Persistence

- versioned format with dimensions, metric, IDs, parameters, and byte bounds;
- save/load parity plus corrupt, truncated, future-version, and duplicate-record
  rejection;
- clear distinction between restart snapshots, mmap/file query, WAL/checkpoint,
  and crash-safe generation publication;
- interruption and mixed-generation behavior documented before durability claims;
- compatibility and migration tests before public format changes.

### API and operations

- typed errors for malformed data and unsupported modes;
- bounded memory, file descriptors, threads, and query sizes;
- thread-safety, GIL behavior, zero-copy/ownership rules, and serial crossover
  tested for Python and Rust callers;
- public docs with runnable examples and explicit experimental limits;
- observability for quality, latency, storage, resource use, and failure mode.

### Release

- CI runs focused correctness and quality gates;
- README/catalog status matches actual support;
- result schemas include method, mode, metric, data scope, seed, features, and
  cache state;
- every promoted method has an owner, consumer, and review trigger.

## Promotion levels

| Level | Meaning |
| --- | --- |
| Experimental | Research/comparison path with honest limits; no production promise. |
| Benchmarked | Reproducible quality and performance evidence on named workloads. |
| Supported | Stable public API and persistence behavior for stated workloads. |
| Production-ready | Supported plus representative data, recovery, tail/resource measurements, Python-wrapper behavior, owner, and release review. |

Promotion is not automatic. A valuable method may remain Benchmarked or
Experimental until its data, persistence, and operational contract are strong.

## Current priorities

- HNSW has the strongest base, but adaptive, batch, MQO, filtered, and
  selectivity paths need representative workloads and Python coverage.
- DiskANN has five-path parity; page layouts are not production-ready from the
  current warm-cache measurements, and direct-file I/O remains the target.
- IVF-PQ/AVQ need representative quality, tail, memory, and Python-wrapper rows.
- `store`/`segstore` is the mutable durability path; ordinary snapshots do not
  inherit its WAL/checkpoint guarantees.
- Experimental graph/tree/quantized/filtered/churn, SparseMIPS, and LEMUR paths
  need a concrete consumer and workload before promotion.

## Workflow

For every method: define the consumer and non-goals; add a counterexample test;
establish release baselines; profile; optimize one bottleneck with controls; run
the feature matrix and full QA; then update README/catalog and promote only when
all gates are green. This rubric is the bar for “every implementation,” not a
claim that every current implementation has already met it.
