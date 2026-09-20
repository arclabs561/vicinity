# External gap research

Status: evidence packet, not an implementation decision
Research: Firecrawl public-source survey, 2026-09-20
Scope: methods, performance techniques, persistence, Python-wrapper quality, and
testing needed to compete with related implementations in Rust and Python.

The items below are candidates from public systems. “Likely gap” means the local
capability list did not establish the complete feature; each item still needs a
source audit and a local consumer before code is added.

## Priority candidates

| Priority | Candidate | Why it matters | Local next gate |
| --- | --- | --- | --- |
| P0 | Crash-safe generation publication plus WAL/replay tests | Save/load is not power-loss recovery. A torn graph, vector file, ID map, or segment must never appear as one committed index. | ADR for generation identity, manifest/current pointer, fsync ordering, replay, retention, and interruption injection. |
| P0 | IVF scalar quantization: SQ4/SQ6/SQ8/fp16 | Provides a simpler operating point between IVF-flat and heavy PQ with controllable bytes/vector and recall. | Verify no existing equivalent; compare against Faiss IVF-SQ on train/add/search/save-load with recall, QPS, RSS, and bytes/vector. |
| P0 | Validated IVF-PQ ADC and reranking | IVF-PQ presence alone does not prove fast LUT-based ADC, residual handling, SIMD scan, or ranking-preserving rerank. | Compare distances/rankings against Faiss; benchmark LUT construction, scan, decode, rerank, and precomputed-table variants. |
| P0 | Common Rust/Python benchmark matrix | Breadth without pinned operating points cannot prove superiority. | Three real datasets plus deterministic CI fixture; fixed CPU/thread/compiler; recall, p50/p95/p99, QPS, build, memory, disk, and Python overhead. |
| P0 | Python GIL, zero-copy, and concurrency contract | The wrapper must compete on end-to-end Python cost, not only native Rust loops. | Audit every binding method for `py.detach`, contiguous dtype/shape behavior, ownership, thread safety, and batch allocation; add wrapper benchmarks. |
| P1 | Filtered/range/selectivity quality matrix | Filtering changes graph navigability and work amplification; unconstrained recall hides production failures. | Exact constrained oracle at deterministic selectivities, including empty results, deleted IDs, p50/p95/p99, and candidate amplification. |
| P1 | Property/fuzz/mutation/concurrency gates | Mutable approximate indexes combine serialization, deletes, quantization, and concurrency failure modes. | Seeded fuzz/replay for insert/delete/search/save/load, malformed dimensions/NaNs, race checks, and mutation tests for tombstones/pruning. |
| P1 | OPQ/PCA transform composition | Learned transforms can improve PQ recall at equal code size, but persistence and train/add/query consistency are easy to get wrong. | Verify whether current OPQ is conventional and composable; add transform parity and persisted-transform tests before benchmarking. |
| P1 | Out-of-core read-ahead/coalescing | Direct-file DiskANN profiling shows `pread`/`read_vector` dominance; a simple cache already regressed. | Measure access locality and syscall counts on larger data; test bounded coalescing with mmap/memory negative controls. |
| P1 | Batch/GPU boundary benchmarks | Bulk query and training workloads can have a different winner than single-query CPU serving. | Keep GPU optional; benchmark host transfer, batch size, training, and end-to-end Python pipeline separately if a supported hardware consumer exists. |

## Public evidence

- [Faiss index families](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
  documents IVF-SQ4/SQ8/fp16, IVFADC/PQ, and refinement/reranking families.
- [Faiss implementation notes](https://github.com/facebookresearch/faiss/wiki/Implementation-notes)
  describes distance tables, precomputation, and PQ fast-scan considerations.
- [Faiss index I/O and tuning](https://github.com/facebookresearch/faiss/wiki/Index-IO%2C-cloning-and-hyper-parameter-tuning)
  warns that loading is not integrity validation and emphasizes held-out tuning.
- [Microsoft DiskANN](https://github.com/microsoft/DiskANN) documents disk-backed
  search, filters, pagination, and diversity-oriented query features.
- [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks) provides datasets,
  train/test splits, Dockerized runners, integrity tests, and recall/time plots.
- [Big ANN Benchmarks](https://big-ann-benchmarks.com/neurips21.html) separates
  hardware tracks and evaluates recall/time at large and out-of-core scale.
- [Qdrant storage concepts](https://qdrant.tech/documentation/concepts/storage/)
  and [snapshots](https://qdrant.tech/documentation/concepts/snapshots/) show
  explicit WAL/recovery and snapshot concerns in a production vector service.
- [PyO3 Python-from-Rust guide](https://pyo3.rs/main/python-from-rust) documents
  interpreter attachment, detaching native work, and free-threaded concerns.

## Deliberate non-conclusions

- These sources do not prove that a named feature is absent locally. The local
  implementation and consumer audit owns that claim.
- GPU support is not automatically a priority for a CPU-first Rust crate. It
  requires a supported hardware consumer and end-to-end transfer measurements.
- Adding more algorithms is not inherently progress. A candidate needs a real
  workload, quality oracle, persistence contract, Python behavior where relevant,
  and a benchmark that can beat or explain the incumbent.
- External benchmark numbers are not imported into local claims. Reproduce under
  matched datasets, metrics, compiler/features, hardware, and cache conditions.
