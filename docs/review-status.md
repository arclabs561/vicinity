# Review Status

This file tracks the current review queue for implementation coverage,
benchmarking, persistence, Python bindings, and performance work.

## Checked Gates

| Area | Status | Evidence |
| --- | --- | --- |
| Dataset fetch/generation repeatability | Passing | `uv run pytest tests/test_download_ann_benchmarks.py tests/test_generate_ann_smoke_data.py tests/test_generate_sample_data.py tests/test_generate_multiscale_data.py` |
| Benchmark resume/storage expectations | Passing | `cargo test --example ann_benchmark --no-default-features --features hnsw,ivf_pq,persistence,diskann,serde -- support::tests` |
| Python exposed API | Passing | `uv run maturin develop --release --features hnsw,python,parallel`; `uv run pytest tests/test_python.py`; `uv run python -m mypy.stubtest pyvicinity._core` |
| Algorithm recommendation docs | Updated | README and `docs/algorithms.md` distinguish brute force, in-memory, file-backed graph, and file-backed compressed search |

## Current Conclusions

- Dataset scripts are idempotent enough for the current benchmark workflow:
  they write atomically, verify HDF5 byte size, validate converted binary
  headers, use a manifest, and require `--force`, `--redownload`, or
  `--adopt-existing` for non-matching cached outputs. They also support
  `--download-only` for pinning and verifying large source HDF5 files before
  conversion, and `--all` for applying the same flow to every configured
  ann-benchmarks dataset.
- Dataset reproducibility is pinned for locally verified standard HDF5 files:
  SIFT-128, GloVe-25, GloVe-50, GloVe-100, GloVe-200, Fashion-MNIST, MNIST,
  NYTimes, GIST, and Deep Image now have expected SHA-256 values.
- The benchmark resume contract is storage-aware for the implemented storage
  modes. DiskANN requires memory, file, and mmap rows. IVF-PQ requires
  snapshot-loaded and file rows, plus mmap rows when the `persistence` feature
  is compiled. `store::UpdatableIndex` requires a `segmented_store` row. Other
  snapshot-capable families require `snapshot_loaded` rows when
  `--snapshot-load` is requested.
- Benchmark result coverage can be summarized from JSONL with
  `uv run scripts/summarize_ann_results.py data/ann-benchmarks/results/*.jsonl`.
  Add `--expect ALGORITHM:STORAGE` rows when reviewing an intended matrix, so
  missing measurements are visible instead of silently absent. Use
  `qps_at_recall_floor` for fixed-recall QPS claims; `best_qps` is the fastest
  row at any recall.
- Python intentionally exposes the stable core today: common HNSW construction,
  HNSW JSON save/load, IVF-PQ directory save/load, IVF-PQ file/mmap search, and
  parallel batch search in release wheels. It should not mirror every
  experimental Rust module until the Rust module has a clear recommendation or
  benchmark gate.
- The largest validated Perplexity finding so far is IVF-PQ: the old low-QPS
  result was an implementation and layout gap. Current work has improved the
  IVF-PQ search path and made file/mmap rows visible.
- DiskANN storage rows have been validated on a capped GloVe-25 corpus:
  in-memory, direct file, and mmap search now report comparable recall,
  latency tails, load time, index bytes, and file-read diagnostics. Full
  GloVe-25 rows and cold-cache storage studies remain open.
- A replicated capped DiskANN EF sweep found `ef_search=75` is the first
  measured 95%+ recall point on the 50K-vector GloVe-25 probe. Profile
  `ef=75` for fixed-recall storage work; keep `ef=50` as a lower-recall
  throughput control.
- Profiling `diskann_search_only/file_ef75` showed the direct-file row was
  dominated by kernel-side I/O samples. Batching file-backed graph neighbor
  reads moved `file_ef75` from 199.28 ms to 112.57 ms per 100 queries, with
  heap and mmap controls staying in family.
- The local ignored result set now has no missing rows for
  `scripts/summarize_ann_results.py --expect-standard-storage --only-dataset
  glove-25-angular --missing-only`. This is row coverage, not full-scale
  fixed-recall coverage: HNSW snapshot rows are still below 95% recall, and
  several rows use `--max-train` caps. Capped IVF-PQ storage rows now do cross
  95% recall at `nprobe=64`.

## Remaining Review Queue

| Priority | Area | Next review |
| --- | --- | --- |
| 1 | Storage-mode matrix | Verify every algorithm row in `docs/persistence.md` against public APIs and `ann_benchmark` support. Keep heap, snapshot-loaded heap, file, mmap, and segmented-store modes separate. |
| 2 | Benchmark coverage | The standard storage matrix has local current-schema rows for GloVe-25 when ignored `data/ann-benchmarks/results/*.jsonl` files are included. `ann_benchmark` records both `--max-train` and `--max-queries` in `_meta`, so bounded rows do not mix with full-dataset rows. Next review should promote selected capped rows to full-corpus runs and add fixed-recall sweeps where `qps_at_recall_floor` is still empty, especially HNSW snapshot. |
| 3 | CI benchmark smoke breadth | CI now runs cheap smoke rows for DiskANN file/mmap, Vamana, `store::UpdatableIndex`, filtered dense rows, FreshGraph, churn modes, and classical baselines. Keep adding rows when new implemented algorithms enter `ann_benchmark`. |
| 4 | Dataset source pinning | All configured ann-benchmarks HDF5 sources now have direct SHA-256 pins. Next review should decide whether stable mirrors are needed beyond `ann-benchmarks.com`. |
| 5 | Segmented-store benchmark row | Added `--algo store` with `storage_mode=segmented_store`; capped 50K GloVe-25 row reaches 99.97% recall at 5.9K QPS. Next review is dataset-scale comparison against HNSW, FreshGraph, in-place graph, and LSM churn. |
| 6 | File-backed raw-vector locality | IVF-PQ approximate file/mmap search is now list-contiguous for PQ codes, and capped fixed-recall rows show approximate file search no longer has the old large file-path penalty. Exact rerank still reads raw vectors by vector ID; at capped fixed recall, file rerank is much slower than mmap. Review whether batching, page layout, or a separate list-local raw-vector sidecar is the right next step. |
| 7 | DiskANN storage layout | Callback-based neighbor reading was measured and rejected. Batching file-backed graph neighbor reads was measured and kept. Next review should focus on full-scale rows, graph/vector page co-location, vector-read locality, mmap page behavior, and cold-cache reporting. |
| 8 | Classical methods | Corrupt-snapshot rejection now covers KD-tree, ball tree, RP-tree, RP-forest, and K-means tree, and docs no longer call KD/Ball exact. Capped benchmark rows now cover all five classical methods with heap plus snapshot-loaded storage metadata. Next review should decide which classical rows need full GloVe-25 runs, then revisit dimensionality and metric gates. |
| 9 | Filtered search | Review ACORN, FilteredGraph, RangeFiltered, and Curator with selectivity sweeps, not single dense-search rows. |
| 10 | Streaming/update workloads | Review FreshGraph, in-place HNSW, LSM HNSW, tombstones, and `store::UpdatableIndex` against active-set recall, update throughput, query latency, compaction, and storage residency. |
| 11 | Sparse and late-interaction harnesses | SparseMIPS needs a SPLADE/BM25-style sparse dataset harness. LEMUR needs training or reproducible model loading before storage or QPS rows matter. |
| 12 | Python policy | Decide which Rust APIs become Python APIs. Keep the default policy narrow unless an algorithm has stable benchmarks, persistence behavior, and examples. Rust-only gaps today: DiskANN, `store`, FreshGraph, filtered search/update APIs, and HNSW binary segments. |
| 13 | External research claims | Verify newer roadmap claims before implementation: Extended RaBitQ, VSAG layout tricks, IP-DiskANN, ACORN production behavior, PAG, SAQ, and ARM/SVE2 kernels. |
| 14 | Profiling depth | Add profile artifacts for the next actual performance change. Record baseline, profiler target, negative controls, before/after, and rejected hypotheses in `docs/benchmark-results.md`. |

## Guardrails

- Do not compare in-memory QPS directly to file or mmap QPS. Treat storage mode
  and cache state as part of the workload.
- Do not promote an experimental algorithm from a single GloVe-25 row. Require
  fixed-recall comparisons on the datasets where the algorithm's assumptions
  apply.
- Do not add another lower-layer storage crate until at least two index
  families need the same extracted interface.
- Do not broaden Python bindings to every Rust module by default. Python should
  expose stable workflows first.
