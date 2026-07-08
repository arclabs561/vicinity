# Review Status

This file tracks the current review queue for implementation coverage,
benchmarking, persistence, Python bindings, and performance work.

## Checked Gates

| Area | Status | Evidence |
| --- | --- | --- |
| Dataset fetch/generation repeatability | Passing | `uv run pytest tests/test_download_ann_benchmarks.py tests/test_generate_ann_smoke_data.py tests/test_generate_sample_data.py tests/test_generate_multiscale_data.py` |
| Dataset difficulty profiling | First pass | `uv run pytest tests/test_profile_ann_dataset.py`; `uv run scripts/profile_ann_dataset.py data/ann-benchmarks/glove-25-angular --sample-train 4096 --sample-queries 1000 --pair-samples 20000 --output /tmp/vicinity-glove25-profile.json` |
| Benchmark resume/storage expectations | Passing | `cargo test --example ann_benchmark --no-default-features --features hnsw,ivf_pq,persistence,diskann,serde -- support::tests` (25 tests, including dense SparseMIPS skip) |
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
  `--expect-observed-standard-storage` for mixed historical result directories,
  where an exhaustive all-family matrix would mostly report algorithms that
  were never part of that run. The summarizer separates capped scopes in the
  dataset label, for example `glove-25-angular[train=50000,queries=1000]`. Use
  `--current-schema-only` when reviewing modern storage rows in a directory that
  also contains legacy JSONL. Use `qps_at_recall_floor` for fixed-recall QPS
  claims; `best_qps` is the fastest row at any recall. The summary also reports
  `index_bytes` from the corresponding rows when the benchmark emitted it, so
  QPS and footprint can be reviewed together.
- Python intentionally exposes the stable core today: common HNSW construction,
  HNSW JSON save/load, IVF-PQ directory save/load, IVF-PQ file/mmap search, and
  parallel batch search in release wheels. It should not mirror every
  experimental Rust module until the Rust module has a clear recommendation or
  benchmark gate.
- The largest validated Perplexity finding so far is IVF-PQ: the old low-QPS
  result was an implementation and layout gap. A full-train GloVe-25 sweep now
  reaches 95.42% recall at 2,640 QPS in memory, 2,553 QPS from direct file
  search, and 2,722 QPS from mmap at `nprobe=32`. The `nprobe=32`,
  `rerank_pool=500` row reaches 96.58% recall at 2,514 QPS in memory and
  2,534 QPS from mmap; direct-file rerank is still slower at 1,453 QPS because
  exact rerank reads raw vectors by vector ID. A duplicate list-local raw-vector
  sidecar experiment was measured and rejected: file rerank stayed flat at
  1,449 QPS while the snapshot grew from 187.3 MB to 305.6 MB.
- DiskANN storage rows have been validated on capped and full GloVe-25 corpus
  runs: in-memory, direct file, and mmap search now report comparable recall,
  latency tails, load time, index bytes, and file-read diagnostics. Full-corpus
  `ef_search=250` is the first measured 95%+ recall point: 95.72% recall at
  4,134.1 QPS in memory, 658.7 QPS from direct file, and 2,382.1 QPS from mmap.
  Cold-cache storage studies remain open.
- A replicated capped DiskANN EF sweep found `ef_search=75` is the first
  measured 95%+ recall point on the 50K-vector GloVe-25 probe. Profile
  `ef=75` for fixed-recall storage work; keep `ef=50` as a lower-recall
  throughput control.
- Profiling `diskann_search_only/file_ef75` showed the direct-file row was
  dominated by kernel-side I/O samples. Batching file-backed graph neighbor
  reads and then switching direct-file graph/vector reads to positional reads
  moved `file_ef75` from 199.28 ms to 72.892 ms per 100 queries, with heap
  and mmap controls staying in family.
- The full-corpus DiskANN fixed-recall point moved the search-only profiling
  target to `ef=250`. Replacing the file/mmap searcher's per-query `HashSet`
  visited set with the dense generation-counter pattern already used by the
  in-memory graph paths improved `file_ef250` by 7.25% mean time and
  `mmap_ef250` by 35.82% mean time in Criterion.
- HNSW allocation profiling found fresh per-query candidate/result heap
  allocation in the normal `HNSWIndex::search` path. Thread-local heap reuse
  reduced the `ef=50` search-only row from 7.0 to 5.0 allocation calls/query
  and measured about 3.9% faster in Criterion.
- HNSW `samply` profiling on `hnsw_search_only/ef/200` shifted the next search
  target away from allocator-only work. Leaf samples were concentrated in
  `flush_batch` (~40%), `innr::dense::dot` (~30%), and
  `greedy_search_layer` (~23%). The profile exported address labels, so future
  source-line claims need a richer debuginfo/dSYM profile, but the current
  evidence points at batch/heap update structure, distance kernel
  dispatch/inlining, and graph/vector locality.
- A thresholded `flush_batch` follow-up now caches the worst result distance
  only for `ef >= 64`. The first all-ef version improved high-ef rows but
  regressed `ef=10`; the thresholded version avoids putting low-recall search
  on that path and measured 6.994 ms per 100 queries at `ef=200` on the
  synthetic HNSW bench, about 2.7% faster than the immediately preceding
  cached-worst trial.
- A capped HNSW storage sweep on 50K GloVe-25 vectors now shows
  snapshot-loaded search at the same recall and comparable warm-cache QPS as
  freshly built in-memory search. On that cap, HNSW reaches 207K QPS at 73.85%
  recall and about 68K QPS at 94.97% recall. Treat those as capped storage and
  curve-shape evidence, not full-corpus targets.
- A full-train HNSW storage sweep on all 1.18M GloVe-25 vectors with 1,000
  queries now covers the higher-`ef_search` fixed-recall band. With
  `ef_construction=200`, the first measured point above 95% recall is
  `ef_search=150` at 96.17% recall and about 11.9K QPS. Snapshot-loaded rows
  preserve recall and warm-cache QPS, so remaining HNSW work is search/layout
  tuning rather than persistence parity.
- The standard storage expectation matrix now covers all current benchmark
  families at the coarse algorithm/storage level, including graph snapshots,
  file/mmap searchers, segmented store, churn rows, and sparse placeholders.
  This is stricter than the earlier shortlist. Missing rows now identify the
  remaining all-family coverage work; measured rows still need separate
  fixed-recall review because several local rows use `--max-train` caps.
- A read-only storage matrix audit found no unsupported DiskANN or IVF-PQ
  file/mmap claims. Follow-up edits made HNSW tombstone persistence explicit,
  marked InPlace/MappedInPlace file snapshots as `serde`-gated, and taught
  resume that SparseMIPS is an intentional dense-harness skip until a sparse
  dataset harness exists.
- A subagent read-only review of the experimental-status docs found DiskANN,
  DEG, LEMUR, SQ4U/SymphonyQG, and classical coverage mostly accurate. Follow-up
  edits narrowed stale KD-tree exactness wording, marked old GloVe-25 rows as
  legacy context, clarified DiskANN's current file/mmap path versus target
  co-located pages, and aligned LEMUR docs with the current mean-pool
  implementation.
- A 50K-vector classical sweep with broader tree settings now separates the
  classical baselines more clearly. KD-tree, ball tree, and RP-tree reach
  high recall on the cap but remain much slower than HNSW at the same scale.
  RP-forest improves to 85.22% recall with 50 trees, and K-means tree remains
  below 25% recall, so neither has a measured 95% point under the current
  sweep.
- Dataset-level benchmark summaries now have a first sampled profiler:
  `scripts/profile_ann_dataset.py` reports shape, norm distribution, sampled
  pair-distance dispersion, exact duplicate rate on the sample, query
  nearest-neighbor margins, LID estimates, sampled relative contrast, and
  ground-truth hubness. Local GloVe-25 showed unit norms, zero sampled exact
  duplicates, median sampled pair distance 0.806, median top-2 gap 0.0089,
  median LID 8.28, median sampled relative contrast 8.11, and top-10 hubness
  Gini 0.925. A lighter pass now covers nine locally converted standard
  datasets; `deep-image-96-angular` was downloaded-only and still needs
  conversion before profiling. GIST stood out with sampled LID p50 52.2 and
  hubness Gini 0.991, which is a warning against treating the GloVe-25 curve as
  representative. Next profiler work is list/cluster imbalance and OOD split
  labels.

## Remaining Review Queue

| Priority | Area | Next review |
| --- | --- | --- |
| 1 | Storage-mode matrix | Main matrix reviewed against public APIs and `ann_benchmark` support. Next pass should compile broader feature combinations and add any new storage modes as algorithms graduate. Keep heap, snapshot-loaded heap, file, mmap, and segmented-store modes separate. |
| 2 | Benchmark coverage | The standard storage matrix now covers every current benchmark family at the coarse algorithm/storage level. `ann_benchmark` records both `--max-train` and `--max-queries` in `_meta`, so bounded rows do not mix with full-dataset rows. Mixed historical result directories can now use observed-only storage expectations and `--current-schema-only`. HNSW, IVF-PQ, and DiskANN now have full-train fixed-recall storage sweeps. Next review should add fixed-recall sweeps where `qps_at_recall_floor` is still empty. |
| 3 | CI benchmark smoke breadth | CI now runs cheap smoke rows for DiskANN file/mmap, Vamana, `store::UpdatableIndex`, filtered dense rows, FreshGraph, churn modes, and classical baselines. Keep adding rows when new implemented algorithms enter `ann_benchmark`. |
| 4 | Dataset source pinning | All configured ann-benchmarks HDF5 sources now have direct SHA-256 pins. Next review should decide whether stable mirrors are needed beyond `ann-benchmarks.com`. |
| 5 | Segmented-store benchmark row | Added `--algo store` with live `store` and reopened `store_snapshot` rows under `storage_mode=segmented_store`; capped 50K GloVe-25 live row reaches 99.97% recall at 5.9K QPS. Next review is dataset-scale comparison against HNSW, FreshGraph, in-place graph, and LSM churn. |
| 6 | File-backed raw-vector locality | IVF-PQ approximate file/mmap search is now list-contiguous for PQ codes, and full-train fixed-recall rows show approximate direct-file search stays in the low-thousands QPS band at 95%+ recall. Positional reads cut targeted file-rerank rows by about 33-37%, but exact rerank still reads raw vectors by vector ID; full-train direct-file rerank at `nprobe=32` is 1,453 QPS versus 2,514 QPS in memory. Benchmark JSON now reports rerank raw-vector reads and bytes. Optional duplicate list-local raw-vector sidecars were measured and rejected because they grew snapshots without improving file rerank. Next review should profile read batching, page/cache behavior, or a replacement raw-vector layout instead. |
| 7 | DiskANN storage layout | Callback-based neighbor reading was measured and rejected. Batching file-backed graph neighbor reads, positional direct-file reads, and dense visited tracking were measured and kept. Full-corpus ef=250 measures 95.72% recall at 4,134.1 QPS in memory, 658.7 QPS file, and 2,382.1 QPS mmap before the dense-visited patch, with 2,343.51 vector reads/query on file and mmap. Next review should focus on graph/vector page co-location, vector-read locality, mmap page behavior, and cold-cache reporting. |
| 8 | Classical methods | Corrupt-snapshot rejection now covers KD-tree, ball tree, RP-tree, RP-forest, and K-means tree, and docs no longer call KD/Ball exact. Capped benchmark rows now cover all five classical methods with heap plus snapshot-loaded storage metadata at 5K and 50K indexed vectors, including a broader 50K sweep. Next review should decide whether any full GloVe-25 classical run is worth the time, or whether the capped evidence is enough to keep them as bounded baselines rather than optimization targets. |
| 9 | Filtered search | Review ACORN, FilteredGraph, RangeFiltered, and Curator with selectivity sweeps, not single dense-search rows. |
| 10 | Streaming/update workloads | Review FreshGraph, in-place HNSW, LSM HNSW, tombstones, and `store::UpdatableIndex` against active-set recall, update throughput, query latency, compaction, and storage residency. |
| 11 | Sparse and late-interaction harnesses | SparseMIPS needs a SPLADE/BM25-style sparse dataset harness. LEMUR needs training or reproducible model loading before storage or QPS rows matter. |
| 12 | Python policy | Decide which Rust APIs become Python APIs. Keep the default policy narrow unless an algorithm has stable benchmarks, persistence behavior, and examples. Rust-only gaps today: DiskANN, `store`, FreshGraph, filtered search/update APIs, and HNSW binary segments. |
| 13 | LSH/sketch boundary | The `lsh` feature uses `sketchir` for cross-polytope hashing primitives. Keep `sketchir` focused on MinHash/SimHash/LSH sketches and durable sketch sidecars; keep vicinity focused on ANN storage, exact reranking, persistence modes, and fixed-recall benchmark rows. Benchmark sharing is useful, but PRT, RP-tree/RP-forest, SparseMIPS, and LEMUR should stay in vicinity unless their role becomes pure sketch generation. |
| 14 | External research claims | Verify newer roadmap claims before implementation: Extended RaBitQ, VSAG layout tricks, IP-DiskANN, ACORN production behavior, PAG, SAQ, and ARM/SVE2 kernels. |
| 15 | Dataset difficulty metadata | First sampled profile script exists for VEC1/NBR1 datasets, and local profiles now cover SIFT, GloVe-25/50/100/200, NYTimes, Fashion-MNIST, MNIST, and GIST. Next review should convert/profile Deep Image, then add partition/list imbalance for IVF-style indexes and explicit OOD split labels where datasets provide them. |
| 16 | Profiling depth | Add profile artifacts for the next actual performance change. Record baseline, profiler target, negative controls, before/after, and rejected hypotheses in `docs/benchmark-results.md`. |

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
