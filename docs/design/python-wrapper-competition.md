# Python-wrapper competition plan

Status: proposal
Evidence: `src/python.rs`, `pyvicinity/__init__.py`,
`pyvicinity/ann_benchmarks.py`, `pyvicinity/_core.pyi`, and `tests/test_python.py`.

## Current strengths

- HNSW and IVF-PQ are the stable Python-facing classes.
- Inputs use NumPy read-only views where possible and reject wrong dimensions or
  non-contiguous arrays explicitly.
- Native search/build work detaches from Python in the primary paths.
- HNSW and IVF-PQ expose single-query and batch-query APIs, persistence, and
  typed stubs. IVF-PQ also exposes file-backed search with mmap selection.
- `pyvicinity.ann_benchmarks` already adapts HNSW and IVF-PQ to ann-benchmarks,
  BigANN, and VIBE-style single/batch interfaces.

## Gaps to close before a superiority claim

1. **End-to-end benchmark matrix.** Measure Python single-query and batch
   latency, p50/p95/p99, throughput, build time, RSS, conversion/copy time, and
   native-only time on the same datasets and operating points as Rust. Compare
   against FAISS, hnswlib, USearch, and the relevant ANN-Benchmarks baselines.
2. **Wrapper breadth.** Decide whether DiskANN file/mmap, filtered/range search,
   mutable store/LSM, and compressed file search are supported Python products.
   If not, keep them explicitly Rust-only rather than implying parity.
3. **Zero-copy contract.** Test C/F-order arrays, dtype conversion, views with
   unusual strides, output ownership, and whether `np.ascontiguousarray` copies.
   Report conversion time separately from native search.
4. **Concurrency contract.** Stress detached searches from Python threads,
   concurrent batch calls, save/load during prohibited concurrent mutation, and
   free-threaded Python where supported. Document which objects are Send/Sync and
   which calls require external locking.
5. **Error and shape parity.** Maintain one table of Python exceptions and
   Rust errors for empty indexes, wrong dimensions, invalid `k`, invalid nprobe,
   malformed snapshots, NaNs, duplicate IDs, and insufficient result rows.
6. **Persistence and deployment.** Add Python tests for snapshot compatibility,
   mmap availability in wheels, path errors, version mismatch, and reopening
   after process restart. Keep native index format claims separate from NumPy or
   HDF5 data interchange.

## Promotion gate

The wrapper can claim a production-ready method only after the native method
passes the production rubric and the Python layer has matching correctness,
copy/ownership, GIL/concurrency, persistence, and end-to-end benchmark evidence.
One fast native loop is not a fast Python product if array conversion or GIL
coordination dominates the caller's operation.
