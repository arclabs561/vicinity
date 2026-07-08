# Review Status

This file tracks the current review queue for implementation coverage,
benchmarking, persistence, Python bindings, and performance work.

## Checked Gates

| Area | Status | Evidence |
| --- | --- | --- |
| Dataset fetch/generation repeatability | Passing | ANN smoke and multiscale manifests record payload SHA-256 plus byte counts; `uv run pytest tests/test_download_ann_benchmarks.py tests/test_generate_ann_smoke_data.py tests/test_generate_sample_data.py tests/test_generate_multiscale_data.py` |
| Dataset difficulty profiling | First pass | `uv run pytest tests/test_profile_ann_dataset.py`; `uv run scripts/profile_ann_dataset.py data/ann-benchmarks/glove-25-angular --sample-train 4096 --sample-queries 1000 --pair-samples 20000 --output /tmp/vicinity-glove25-profile.json`; `uv run pytest tests/test_summarize_ann_results.py`; `uv run python scripts/summarize_ann_results.py --json --current-schema-only --require-declared-index-bytes --profile-dir /tmp/vicinity-profile-kind-dir /tmp/vicinity-footprint-kind.jsonl` |
| Benchmark resume/storage expectations | Passing | `CARGO_TARGET_DIR=/tmp/vicinity-support-refactor-test CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test --example ann_benchmark --no-default-features --features hnsw,ivf_pq,persistence,diskann,serde,store,kdtree,balltree,rptree,kmeans_tree,lsh,filtered_graph,curator,range_filtered,fresh_graph,ivf_avq,ivf_rabitq,emg,nsg,nsw,sng,pipnn,vamana,finger,rp_quant,sparse_mips,binary_index,sq4,sq8 -- support::tests` (32 tests, including dense SparseMIPS skip, storage-mode resume checks, and legacy row rejection without storage mode); `CARGO_TARGET_DIR=/tmp/vicinity-support-refactor-clippy CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo clippy --example ann_benchmark --no-default-features --features hnsw,ivf_pq,persistence,diskann,serde,store,kdtree,balltree,rptree,kmeans_tree,lsh,filtered_graph,curator,range_filtered,fresh_graph,ivf_avq,ivf_rabitq,emg,nsg,nsw,sng,pipnn,vamana,finger,rp_quant,sparse_mips,binary_index,sq4,sq8 -- -D warnings` |
| Feature-matrix compilation | Passing | `CARGO_TARGET_DIR=/tmp/vicinity-feature-matrix-current CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo hack check --each-feature --no-dev-deps --exclude-features python` (44 Rust feature configs, including `persistence` and `store`; `python` remains covered by the PyO3-specific gate) |
| Graph prefetch unsafe surface | Removed | HNSW `hnsw_search_only` and `hnsw_search_mmax32` controls measured the architecture-specific prefetch hint as neutral or slower. DiskANN `memory_ef75` and `file_ef75` also improved after replacing it with a safe no-op helper. `RUSTFLAGS=-Dwarnings CARGO_TARGET_DIR=/tmp/vicinity-prefetch-target CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo check --no-default-features --features diskann,emg,finger,fresh_graph,hnsw,nsg,nsw,pipnn,sng,vamana`; `CARGO_TARGET_DIR=/tmp/vicinity-prefetch-target CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo clippy --no-default-features --features diskann,emg,finger,fresh_graph,hnsw,nsg,nsw,pipnn,sng,vamana --lib -- -D warnings` |
| PQ SIMD unsafe surface | Reduced | `CARGO_TARGET_DIR=/tmp/vicinity-pq-unsafe-target CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test --no-default-features --features ivf_pq --lib pq_simd::tests`; `CARGO_TARGET_DIR=/tmp/vicinity-pq-unsafe-target CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo clippy --no-default-features --features ivf_pq --lib -- -D warnings` |
| PQ SIMD unsafe boundary | Reduced | Architecture-specific dispatch wrappers own runtime feature checks and target-feature calls. Safe batch entrypoints now reject zero codebooks, partial candidate rows, mismatched LUT shapes, and overflowing LUT lengths before SIMD dispatch; x86 gather dispatch is limited to full 256-entry LUTs where every `u8` code is in range. FastScan packing and block lookup now validate checked byte counts before the NEON block kernel. The borrowed flat 256-entry LUT dispatch path is checked against the scalar nested ADC oracle, including tail candidates. `CARGO_TARGET_DIR=/tmp/vicinity-pq-simd-safety-target CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test --no-default-features --features ivf_pq --lib pq_simd::tests`; `CARGO_TARGET_DIR=/tmp/vicinity-pq-simd-safety-clippy CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo clippy --no-default-features --features ivf_pq --lib -- -D warnings` |
| Unsafe lint boundary | Passing | `unsafe_code = "deny"` and `unsafe_op_in_unsafe_fn = "deny"` are enabled at the crate level. Product unsafe is limited to architecture-specific PQ SIMD modules with safe dispatch wrappers and parity tests. Benchmarks keep allocation counters behind local `allow(unsafe_code)` scopes. |
| innr SIMD boundary | Keep | `vicinity::simd` already re-exports innr's safe dense full-vector kernels. innr batch APIs require a materialized columnar candidate batch, so they are not a measured fit for scattered HNSW neighbor traversal yet. The plausible future innr addition is a safe indexed-row scoring helper, gated on Criterion and end-to-end evidence. |
| IVF-PQ FastScan split | Resolved | The FastScan gate is intentional: `codebook_size = 16` uses the 4-bit packed block layout, while the main fixed-recall GloVe-25 path uses `codebook_size = 256` and the standard 8-bit ADC batch kernel. Tests cover both prepacked layouts. |
| SmallVec memory accounting | Passing | `smallvec_u32_bytes` now counts resident `Vec<SmallVec<_>>` element storage plus spilled buffers instead of only payload capacity. This fixes graph `index_bytes` undercounting for inline-capacity experiments. `CARGO_TARGET_DIR=/tmp/vicinity-memory-smallvec-test CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test --no-default-features --features hnsw --lib memory::tests` |
| Graph-family inline neighbors | Partial | A bounded GloVe-25 diagnostic compared 16-inline, blanket 32-inline, and trimmed graph-neighbor lists. The original run's byte deltas were payload-only and undercounted inline storage; corrected current 2K rows report Vamana 1,008,000 bytes, PiPNN 497,200 bytes, and EMG 546,200 bytes. NSG, FINGER, FilteredGraph, FreshGraph, SparseMIPS, NSW, and SNG stay 16-inline until larger workload-specific evidence justifies the memory cost. `CARGO_TARGET_DIR=/tmp/vicinity-graph-inline-check CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo check --no-default-features --features vamana,nsg,finger,pipnn,emg,filtered_graph,fresh_graph,sparse_mips,serde --lib`; `CARGO_TARGET_DIR=/tmp/vicinity-graph-inline-legacy-check CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo check --no-default-features --features nsw,sng,serde --lib`; diagnostic rows: `/tmp/vicinity-graph-inline-baseline.jsonl`, `/tmp/vicinity-graph-inline-after.jsonl`, `/tmp/vicinity-graph-inline-trimmed.jsonl`, `/tmp/vicinity-graph-inline-corrected.jsonl` |
| Python exposed API | Passing | `PYO3_PYTHON=/opt/homebrew/bin/python3.12 CARGO_TARGET_DIR=/tmp/vicinity-python-gate CARGO_INCREMENTAL=0 RUSTC_WRAPPER= uv run maturin develop --release --features hnsw,python,parallel`; `uv run --extra test pytest tests/test_python.py`; `uv run --extra test python -m mypy.stubtest pyvicinity._core`; `uv run ruff check pyproject.toml tests/test_python.py examples/python` |
| Algorithm recommendation docs | Updated | README and `docs/algorithms.md` distinguish brute force, in-memory, file-backed graph, and file-backed compressed search |
| Classical tree recall knobs | First pass | `CARGO_TARGET_DIR=/tmp/vicinity-rp-forest-sweep CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo run --release --example ann_benchmark --no-default-features --features rptree -- data/ann-benchmarks/glove-25-angular --algo rp_forest --max-train 50000 --max-queries 500 --tree-leaf-sizes 100,200,500 --rp-num-trees 50,100,200 --json --results /tmp/vicinity-rp-forest-sweep.jsonl`; `CARGO_TARGET_DIR=/tmp/vicinity-kmeans-branch-sweep CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo run --release --example ann_benchmark --no-default-features --features kmeans_tree -- data/ann-benchmarks/glove-25-angular --algo kmeans_tree --max-train 50000 --max-queries 500 --kmeans-clusters 8 --kmeans-leaf-sizes 200,500,1000 --kmeans-depths 10 --kmeans-iters 10 --kmeans-search-branches 1,2,4,8 --json --results /tmp/vicinity-kmeans-branch-sweep.jsonl`; `CARGO_TARGET_DIR=/tmp/vicinity-kmeans-branch-test CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test --no-default-features --features kmeans_tree --lib classic::trees::kmeans_tree::tests`; `CARGO_TARGET_DIR=/tmp/vicinity-kmeans-support-test-full CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test --example ann_benchmark --no-default-features --features kmeans_tree,serde -- support::tests` |
| RP-forest candidate allocation | Passing | `CARGO_TARGET_DIR=/tmp/vicinity-rpforest-candidate-baseline CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo run --release --example ann_benchmark --no-default-features --features rptree,serde -- data/ann-benchmarks/glove-25-angular --algo rp_forest --max-train 50000 --max-queries 500 --tree-leaf-sizes 200 --rp-num-trees 50 --snapshot-load --json --results /tmp/vicinity-rpforest-candidate-baseline.jsonl`; rejected sort/dedup trial in `/tmp/vicinity-rpforest-candidate-after.jsonl`; kept direct leaf-slice insertion with preallocated `HashSet` in `/tmp/vicinity-rpforest-candidate-after2.jsonl`; `CARGO_TARGET_DIR=/tmp/vicinity-rpforest-candidate-test2 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test --no-default-features --features rptree,serde --lib classic::trees::rp_forest` |
| IVF-AVQ direct file/mmap search | Passing | Existing file-backed gates: `CARGO_TARGET_DIR=/tmp/vicinity-ivfavq-file-test CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test --no-default-features --features ivf_avq --lib ivf_avq::search::tests`; `CARGO_TARGET_DIR=/tmp/vicinity-ivfavq-file-smoke-target CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo run --release --example ann_benchmark --no-default-features --features ivf_avq -- data/ann-benchmarks/glove-25-angular --algo ivf_avq --max-train 1024 --max-queries 50 --pq-nprobes 1,2 --snapshot-load --json --results /tmp/vicinity-ivfavq-file-smoke-20260708-rerun.jsonl`. Current mmap, reorder, and heap-size gates: `CARGO_TARGET_DIR=/tmp/vicinity-avq-reorder-lib-test CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test --no-default-features --features ivf_avq,persistence --lib ivf_avq::search`; `CARGO_TARGET_DIR=/tmp/vicinity-avq-reorder-support-test CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test --no-default-features --features ivf_avq,ivf_rabitq,persistence --example ann_benchmark support::tests`; `CARGO_TARGET_DIR=/tmp/vicinity-avq-reorder-clippy CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo clippy --no-default-features --features ivf_avq,ivf_rabitq,persistence --lib --example ann_benchmark -- -D warnings`; `CARGO_TARGET_DIR=/tmp/vicinity-avq-reorder-smoke CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo run --no-default-features --features ivf_avq,persistence --example ann_benchmark -- data/ann-benchmarks/glove-25-angular --algo ivf_avq --max-train 500 --max-queries 20 --pq-nprobes 1 --pq-rerank-pools 50,100 --snapshot-load --json --results /tmp/vicinity-avq-reorder-smoke.jsonl` emitted `in_memory`, `snapshot_loaded`, `file`, and `mmap` rows for both reorder settings with matching recall; `CARGO_TARGET_DIR=/tmp/vicinity-avq-memory-smoke CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo run --no-default-features --features ivf_avq --example ann_benchmark -- data/ann-benchmarks/glove-25-angular --algo ivf_avq --max-train 500 --max-queries 20 --pq-nprobes 1 --pq-rerank-pools 50 --json --results /tmp/vicinity-avq-memory-smoke.jsonl` emitted positive in-memory `index_bytes`. |
| IVF-RaBitQ snapshot format bounds | Passing | Non-compacted snapshots stay heap-reloaded and rebuild qntz edge codes instead of adding unsafe edge-field reconstruction. Manifest validation now rejects zero clusters, invalid bit widths, and overflowing raw-vector or centroid lengths before reading payload files, and zero-`k` search exits without scanning. `CARGO_TARGET_DIR=/tmp/vicinity-ivfrabitq-format-test CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test --no-default-features --features ivf_rabitq --lib ivf_rabitq::tests`; `CARGO_TARGET_DIR=/tmp/vicinity-ivfrabitq-format-clippy CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo clippy --no-default-features --features ivf_rabitq --lib -- -D warnings` |
| Shared graph snapshot helpers | Passing | `graph_snapshot` now centralizes repeated feature-gate lists and rejects on-disk neighbor counts, neighbor ids, and dense vector shape overflows with checked conversions before allocation or indexing. `CARGO_TARGET_DIR=/tmp/vicinity-graph-snapshot-check CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo check --no-default-features --features nsw,sng,vamana,nsg,finger,pipnn,emg,binary_index,rp_quant,sparse_mips,lsh,sq4 --lib`; `CARGO_TARGET_DIR=/tmp/vicinity-graph-snapshot-clippy CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo clippy --no-default-features --features nsw,sng,vamana,nsg,finger,pipnn,emg,binary_index,rp_quant,sparse_mips,lsh,sq4 --lib -- -D warnings` |

## Current Conclusions

- Dataset scripts are idempotent enough for the current benchmark workflow:
  they write atomically, verify HDF5 byte size, validate converted binary
  headers and payload sizes, record generated-output SHA-256 values in
  manifests where generation is deterministic, and require `--force`,
  `--redownload`, or `--adopt-existing` for non-matching cached outputs.
  They also support
  `--download-only` for pinning and verifying large source HDF5 files before
  conversion, and `--all` for applying the same flow to every configured
  ann-benchmarks dataset.
- Dataset reproducibility is pinned for locally verified standard HDF5 files:
  SIFT-128, GloVe-25, GloVe-50, GloVe-100, GloVe-200, Fashion-MNIST, MNIST,
  NYTimes, GIST, and Deep Image now have expected SHA-256 values.
- The benchmark resume contract is storage-aware for the implemented storage
  modes. DiskANN requires memory, file, and mmap rows. IVF-PQ and IVF-AVQ
  require snapshot-loaded and file rows, plus mmap rows when the `persistence`
  feature is compiled.
  `store::UpdatableIndex` requires a `segmented_store` row. Other snapshot-
  capable families require `snapshot_loaded` rows when `--snapshot-load` is
  requested.
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
  QPS and footprint can be reviewed together. With `--profile-dir`, exact
  dataset-profile matches now carry metric, shape, sampled distance dispersion,
  nearest-neighbor margin, LID, sampled contrast, hubness, coarse-partition
  imbalance, and query-split kinds into the same coverage rows.
- Python intentionally exposes the stable core today: common HNSW construction,
  HNSW JSON save/load, IVF-PQ directory save/load, IVF-PQ file/mmap search, and
  parallel batch search in release wheels. It should not mirror every
  experimental Rust module until the Rust module has a clear recommendation or
  benchmark gate.
- Local cargo builds of the `python` feature need a 3.10+ interpreter because
  `pyo3` is configured with `abi3-py310`. On macOS, `/usr/bin/python3` may be
  3.9; set `PYO3_PYTHON=/opt/homebrew/bin/python3.12` or run through the
  project `uv` environment when checking that feature.
- The largest validated Perplexity finding so far is IVF-PQ: the old low-QPS
  result was an implementation and layout gap. A full-train GloVe-25 sweep now
  reaches 95.42% recall at 2,640 QPS in memory, 2,553 QPS from direct file
  search, and 2,722 QPS from mmap at `nprobe=32`. The `nprobe=32`,
  `rerank_pool=500` row reaches 96.58% recall at 2,514 QPS in memory and
  2,534 QPS from mmap; direct-file rerank is still slower at 1,453 QPS because
  exact rerank reads raw vectors by vector ID. A duplicate list-local raw-vector
  sidecar experiment was measured and rejected: file rerank stayed flat at
  1,449 QPS while the snapshot grew from 187.3 MB to 305.6 MB. In-memory
  approximate and rerank rows now also report heap-estimated `index_bytes`;
  file and mmap rows keep reporting saved snapshot bytes.
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
- IVF-AVQ now has direct file and read-only mmap search paths over saved
  partition/code files and raw vectors for exact rerank. The capped smoke rows
  validate equal recall between in-memory, snapshot-loaded, direct-file, and
  mmap storage modes for the same nprobe and `num_reorder` setting. The
  benchmark runner now sweeps IVF-AVQ `num_reorder` through
  `--pq-rerank-pools`; on a 50K-vector, 500-query GloVe-25 cap, the first
  measured 95%+ row is `nprobe=64,num_reorder=500` at 99.22% recall and
  8,525.9 QPS. That result means the earlier AVQ recall gap was partly a
  hard-coded rerank-pool gap, not only an AVQ kernel problem. A full-train
  follow-up with the same `num_reorder=500` did not clear the fixed-recall bar,
  but the full-train storage sweep with larger reorder pools did: the fastest
  heap 95%+ row is `nprobe=20,num_reorder=5000` at 98.36% recall and 1,867.8
  QPS, the fastest direct-file 95%+ row is `nprobe=50,num_reorder=1000` at
  489.1 QPS, and the fastest mmap 95%+ row is `nprobe=20,num_reorder=5000` at
  1,059.4 QPS. In-memory AVQ rows now also report heap-estimated
  `index_bytes`; file and mmap rows keep reporting saved snapshot bytes.
- The full-corpus DiskANN fixed-recall point moved the search-only profiling
  target to `ef=250`. Replacing the file/mmap searcher's per-query `HashSet`
  visited set with the dense generation-counter pattern already used by the
  in-memory graph paths improved `file_ef250` by 7.25% mean time and
  `mmap_ef250` by 35.82% mean time in Criterion.
- HNSW allocation profiling found fresh per-query candidate/result heap
  allocation in the normal `HNSWIndex::search` path. Thread-local heap reuse
  reduced the `ef=50` search-only row from 7.0 to 5.0 allocation calls/query
  and measured about 3.9% faster in Criterion.
- HNSW's internal neighbor-list alias now holds 32 neighbors inline, matching
  the default base-layer `m_max=32`. The capacity guard passes, and
  `benches/hnsw_search.rs` now has a separate `hnsw_search_mmax32` group. A
  same-binary synthetic run measured `m=16,m_max=32` slower than the historical
  `m=16,m_max=16` fixture because the graph exposes more base-layer neighbors;
  treat this as topology evidence, not a capacity-regression signal.
- HNSW `samply` profiling on `hnsw_search_only/ef/200` shifted the next search
  target away from allocator-only work. Leaf samples were concentrated in
  `flush_batch` (~40%), `innr::dense::dot` (~30%), and
  `greedy_search_layer` (~23%). A repeat profile with frame pointers and
  debuginfo still exported address labels, but manual `nm` symbolication mapped
  13,424 benchmark-thread leaf samples to `innr::dense::dot` (36.3%),
  `flush_batch` (22.1%), `greedy_search_layer` (21.0%), and `BinaryHeap::pop`
  (13.3%). The current evidence points at heap/frontier structure,
  candidate-processing layout, and graph/vector locality, not allocator-only
  work or a broad dispatch rewrite.
- Unsafe remains a last resort for the current perf queue. Dense vector math
  should keep using innr's safe API. New unsafe belongs only in small, local,
  layout-specific kernels with parity tests and before/after profiles, as with
  the existing PQ SIMD wrappers.
- The graph prefetch helper is now a safe no-op. HNSW search-only controls and
  DiskANN `memory_ef75` plus `file_ef75` controls measured the
  architecture-specific hint as neutral or slower on this machine.
- If innr grows a helper for vicinity, prefer a safe row-indexed dense scoring
  API over exposing raw SIMD details. The evidence gate is current HNSW
  `flush_batch` versus dimension- or workload-aware batch choices at
  64/128/384/768 dimensions, plus end-to-end HNSW, DiskANN, and IVF
  rerank/centroid controls. A global batch size of 16 was measured and
  rejected on the 128-d HNSW search bench because denser-graph negative
  controls regressed.
- A direct generic-wrapper attempt for normal HNSW metric dispatch was measured
  and rejected. It left `ef=10` unchanged but regressed `ef=50`, `ef=100`, and
  `ef=200` by about 5.8%, 5.6%, and 10.2% respectively in the search-only
  Criterion target. Future dispatch work should start from binary inspection or
  a narrower distance-kernel boundary, not a traversal-wide generic wrapper.
- Replacing HNSW result-heap push-then-pop with `BinaryHeap::peek_mut`
  top replacement was measured and kept. The search-only synthetic bench
  improved `ef=10` by 2.57%, `ef=100` by 2.86%, and `ef=200` by 3.93%, with
  no significant `ef=50` change. HNSW lib tests and search-with-distance parity
  passed under `--no-default-features --features hnsw`.
- Draining the HNSW result heap with repeated `BinaryHeap::pop()` plus
  `reverse()` was measured and rejected. It preserved the intended ordering but
  regressed all normal search-only rows by about 13-18%, so the standard search
  path keeps `drain()` plus `sort_unstable_by`.
- Periodic HNSW frontier pruning was measured and kept with a conservative
  `ef >= 64`, 64-pop interval. It improved the 128-d search-only bench at
  default `ef=100` and `ef=200`, improved denser `m_max=32` rows at `ef=10`
  and `ef=50`, and stayed neutral on the remaining controls. The earlier
  32-pop interval was rejected because `m_max=32,ef=200` regressed.
- The newer Perplexity2/unfinished-risk note was written against an older
  snapshot and was re-checked against current `HEAD`. Its HNSW tombstone,
  IVF-PQ filter metadata, DiskANN rustdoc, HNSW `SmallVec` inline-capacity,
  SNG overclaim, unsupported compression fallback, and 4-bit NEON FastScan
  claims are resolved or mitigated. The broader graph-family inline-neighbor
  follow-up is partially measured: Vamana, PiPNN, and EMG now use 32-inline
  neighbor lists, while the blanket 32-inline trial was rejected for NSG,
  FINGER, FilteredGraph, and FreshGraph because it grew graph footprint by
  about one third under the original payload-only estimator. A follow-up fixed
  the estimator to include inline `SmallVec` element storage, so the kept
  Vamana/PiPNN/EMG rows still need full-corpus memory review before they become
  recommendations. Live follow-ups from that note are HNSW distance-dispatch
  profiling, trained/model-loaded LEMUR weights, and fixed-recall rows for
  experimental families. LEMUR's random-weight fixture constructor is now hidden
  behind test builds or the explicit `lemur-fixtures` feature. The
  `ivf_pq/search.rs` size item has been
  reduced without behavior changes by extracting manifest, cluster, and
  file-storage helpers; continue splitting only when a coherent private
  boundary appears.
- The 4-bit IVF-PQ FastScan path now has an aarch64 NEON `tbl` block kernel.
  The direct `pq_fastscan_lut_shape/flat_lut` microbench improved from
  3.3636 us to 934.32 ns, with parity covered against the portable block
  kernel. This is separate from the main GloVe-25 IVF-PQ row, which uses
  `codebook_size=256` and the standard 8-bit ADC path.
- IVF-PQ snapshot persistence now includes the optional filter field and
  document metadata. The format remains compatible with older v1 manifests that
  lack those fields, and tests cover filtered search after save/load.
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
  file/mmap claims. Follow-up edits made HNSW serde and binary-segment
  tombstones roundtrip, marked InPlace/MappedInPlace file snapshots as
  `serde`-gated, and taught resume that SparseMIPS is an intentional
  dense-harness skip until a sparse dataset harness exists.
- A subagent read-only review of the experimental-status docs found DiskANN,
  DEG, LEMUR, SQ4U/SymphonyQG, and classical coverage mostly accurate. Follow-up
  edits narrowed stale KD-tree exactness wording, marked old GloVe-25 rows as
  legacy context, clarified DiskANN's current file/mmap path versus target
  co-located pages, and aligned LEMUR docs with the current mean-pool
  implementation. SAQ docs now say the current helper uses direct segmentation
  and variance-weighted bit allocation; PCA projection, dynamic-programming
  allocation, and trained k-means codebooks remain future work.
- A 50K-vector classical sweep with broader tree settings now separates the
  classical baselines more clearly. KD-tree, ball tree, and RP-tree reach
  high recall on the cap but remain much slower than HNSW at the same scale.
  RP-forest reaches 97.54% recall at 3,990 QPS with 50 trees and leaf size
  200, so the earlier sub-95% result was mostly an undersized candidate budget.
  K-means tree now has an explicit per-node branch-budget search path:
  branch budget 4 reaches 92.10% recall at 1,245 QPS, while branch budget 8
  reaches 100% recall at 295-590 QPS on the same cap. A newer global
  leaf-budget path is a better fixed-recall point: leaf size 500 with
  `leaf_budget=48` reaches 95.92% recall at 5,622 QPS in memory and preserves
  95.92% recall at 5,216 QPS after snapshot load. Treat K-means tree as a
  controllable classical baseline; it now has a usable 95%+ row, but graph
  methods still lead the same cap.
- Dataset-level benchmark summaries now have a first sampled profiler:
  `scripts/profile_ann_dataset.py` reports shape, norm distribution, sampled
  pair-distance dispersion, exact duplicate rate on the sample, query
  nearest-neighbor margins, LID estimates, sampled relative contrast, and
  ground-truth hubness, and sampled coarse-partition imbalance. Local GloVe-25
  showed unit norms, zero sampled exact duplicates, median sampled pair distance
  0.806, median top-2 gap 0.0089, median LID 8.28, median sampled relative
  contrast 8.11, top-10 hubness Gini 0.925, and coarse-partition Gini 0.241. A
  lighter pass now covers all ten locally converted standard datasets. GIST
  stood out with sampled LID p50 52.2, hubness Gini 0.991, and coarse-partition
  Gini 0.474; Deep Image showed 9.99M vectors, sampled LID p50 13.2, hubness
  Gini 0.990, and coarse-partition Gini 0.255. These are warnings against
  treating the GloVe-25 curve as representative. The profiler also reports
  generated query splits when `test_drift.bin`, `test_filter.bin`, topic labels,
  or difficulty labels are present.
- Filtered-search selectivity sweeps now have a dedicated summarizer. Run
  `uv run scripts/summarize_selectivity_results.py` against
  `data/ann-benchmarks/results/acorn-selectivity-*.jsonl`. A 3K-vector,
  200-query synthetic sweep now verifies ACORN, FilteredGraph, RangeFiltered,
  and Curator rows across the selectivity curve and keeps target count, recall,
  QPS, latency tails, returned count, and ACORN two-hop counters visible. It is
  diagnostic only: default ACORN returns full top-k but misses exact filtered
  neighbors, FilteredGraph has a sharp high-selectivity slow path, and
  RangeFiltered under-returns at narrow ranges. Follow-up ACORN-only tuning on
  the same workload shows `ef_search=800` with 128 two-hop neighbors clears
  95% recall from 2% through 50% selectivity at about 3.8-4.0K QPS, but still
  reaches only about 93.6-93.8% recall at 1% selectivity. The new
  `selectivity_acorn` benchmark row uses tuned ACORN above a configurable
  threshold and exact search over pre-filtered IDs below it; at the 1% synthetic
  row it reaches 100% recall with zero two-hop traversal, confirming the policy
  shape before promoting it into public API behavior.
- A read-only external implementation scout found concrete prior-art patterns
  that match the local evidence: Qdrant-style selectivity-gated ACORN,
  Weaviate-style sparse visited sets, Qdrant/Lance-style safe prefetch
  abstraction, Faiss FastScan layout and deserialization validation, and
  DiskANN-style provider boundaries for disk graph/vector storage. The unsafe
  lesson is narrow: use private SIMD/layout boundaries with scalar parity and
  corrupt-file tests, not broad unsafe graph traversal rewrites.

## Remaining Review Queue

| Priority | Area | Next review |
| --- | --- | --- |
| 1 | Storage-mode matrix | Main matrix reviewed against public APIs and `ann_benchmark` support, and the resume matcher now has a broad feature-set test/clippy gate covering graph, IVF, storage, filter, streaming, sparse, quantized, and classical families together. Next pass should add any new storage modes as algorithms graduate. Keep heap, snapshot-loaded heap, file, mmap, and segmented-store modes separate. |
| 2 | Benchmark coverage | The standard storage matrix now covers every current benchmark family at the coarse algorithm/storage level. `ann_benchmark` records `--max-train`, `--max-queries`, `--warmup-queries`, and the compiled feature list in `_meta`, so bounded rows, cache warmup, and narrower binaries are auditable. Mixed historical result directories can now use observed-only storage expectations, `--current-schema-only`, `--footprint-contract-only`, and `--suppress-dominated-recall-gaps` for stale smoke/default rows that were superseded by fixed-recall sweeps at the same or larger train cap. HNSW, NSW, Vamana, NSG, SNG, EMG, PiPNN, FINGER, FreshGraph, KD-tree, ball tree, RP-tree, RP-forest, K-means tree, FilteredGraph, Curator, RangeFiltered, InPlace, DiskANN, IVF-PQ, IVF-AVQ, IVF-RaBitQ, RpQuant, BinaryFlat, SQ4, SQ4U, SQ8U, SymphonyQG, SymphonyQG-VR, LSH, and HNSW-PRT in-memory rows now report heap-estimated `index_bytes`. HNSW, IVF-PQ, DiskANN, and IVF-AVQ have full-train fixed-recall storage sweeps. Use `scripts/summarize_ann_results.py --recall-gap-only` to identify the next scoped fixed-recall sweeps where `qps_at_recall_floor` is still empty; qualifying summary rows now include the winning params. |
| 3 | CI benchmark smoke breadth | CI now runs cheap smoke rows for DiskANN file/mmap, IVF-PQ file/mmap, IVF-AVQ file/mmap, Vamana, `store::UpdatableIndex`, filtered dense rows, FreshGraph, churn modes, and classical baselines. Keep adding rows when new implemented algorithms enter `ann_benchmark`. |
| 4 | Dataset source pinning | All configured ann-benchmarks HDF5 sources now have direct SHA-256 pins. Next review should decide whether stable mirrors are needed beyond `ann-benchmarks.com`. |
| 5 | Segmented-store benchmark row | Added `--algo store` with live `store` and reopened `store_snapshot` rows under `storage_mode=segmented_store`; capped 50K GloVe-25 live row reaches 99.97% recall at 5.9K QPS. Next review is dataset-scale comparison against HNSW, FreshGraph, in-place graph, and LSM churn. |
| 6 | File-backed raw-vector locality | IVF-PQ approximate file/mmap search is now list-contiguous for PQ codes, and full-train fixed-recall rows show approximate direct-file search stays in the low-thousands QPS band at 95%+ recall. Positional reads cut targeted file-rerank rows by about 33-37%, but exact rerank still reads raw vectors by vector ID; full-train direct-file rerank at `nprobe=32` is 1,453 QPS versus 2,514 QPS in memory. Benchmark JSON now reports IVF-PQ probed-list/code-read counters and IVF-AVQ partition/code/raw-vector counters, plus rerank raw-vector reads and bytes. Optional duplicate list-local raw-vector sidecars were measured and rejected because they grew snapshots without improving file rerank. Next review should profile read batching, page/cache behavior, or a replacement raw-vector layout instead. |
| 7 | DiskANN storage layout | Callback-based neighbor reading was measured and rejected. Batching file-backed graph neighbor reads, positional direct-file reads, and dense visited tracking were measured and kept. Full-corpus ef=250 measures 95.72% recall at 4,134.1 QPS in memory, 658.7 QPS file, and 2,382.1 QPS mmap before the dense-visited patch, with 2,343.51 vector reads/query on file and mmap. Next review should focus on graph/vector page co-location, vector-read locality, mmap page behavior, and cold-cache reporting. |
| 8 | Classical methods | Corrupt-snapshot rejection now covers KD-tree, ball tree, RP-tree, RP-forest, K-means tree, and LSH, and docs no longer call KD/Ball exact. Capped benchmark rows now cover all five tree methods with in-memory heap `index_bytes` plus snapshot-loaded storage metadata at 5K and 50K indexed vectors; LSH and brute-force now also have current-schema 50K rows with LSH in-memory and snapshot-loaded footprint metadata. Broader 50K sweeps show RP-forest can clear 95% recall with larger tree/leaf budgets, K-means tree can clear 95% only with a broad branch budget that cuts QPS to a few hundred, and LSH can clear 95% on the capped row but remains a classical/storage baseline rather than a full-corpus target. Targeted snapshot-load rows now confirm parity for the best 50K high-recall RP-forest, K-means, and LSH configs. Next review should decide whether to run full GloVe-25 classical rows and whether K-means needs a better global beam policy before more tuning. |
| 9 | Filtered search | The selectivity JSONL summarizer exists and a 3K-vector, 200-query synthetic curve now exercises ACORN, FilteredGraph, RangeFiltered, Curator, and `selectivity_acorn` with target counts, latency tails, returned counts, and ACORN two-hop counters. FilteredGraph, Curator, and RangeFiltered dense rows also emit in-memory heap `index_bytes`, but the dense rows are still unfiltered rows labeled `filter_mode=none`. ACORN tuning now shows `ef_search=800`/128 two-hop clears 95% recall from 2% through 50% selectivity at about 11.8-13.6K QPS after dense visited tracking, but still misses at 1%; `selectivity_acorn` fixes the 1% row by exact search over pre-filtered IDs. A search-only profile showed ACORN was dominated by visited-set hash insertion and rehashing rather than dense distance math; bounded visited-set preallocation improved the dedicated Criterion rows by about 22-24%, and the contiguous node-count dense visited path improved them by about 65-68%. Borrowed neighbor callbacks remove forced cloning for in-memory HNSW, but did not show a significant QPS change on their own. Next review should validate the selectivity policy on larger and data-backed filter workloads before making public API defaults or QPS claims. |
| 10 | Streaming/update workloads | Churn rows for FreshGraph, in-place HNSW, and LSM HNSW use active-set ground truth and now expose update diagnostics as top-level JSON fields (`active_count`, `update_time_s`, `update_qps`, tombstone/free-slot/compaction counters) so summaries can carry them. InPlace dense rows now emit in-memory heap `index_bytes`. LSM now has snapshot restart persistence for levels, tombstones, config, and counters; the churn benchmark emits `snapshot_loaded` rows under `hnsw,serde`, and the strict summarizer expects them. WAL/checkpoint recovery remains a separate durability design. Next review should run larger fixed-recall churn sweeps and compare live versus snapshot-loaded LSM at non-smoke scale. |
| 11 | Sparse and late-interaction harnesses | SparseMIPS now canonicalizes unsorted sparse vectors by sorting dimensions and summing duplicate entries, and `scripts/generate_sparse_mips_smoke_data.py` plus `examples/sparse_mips_benchmark.rs` provide an idempotent SPV1/NBR1 smoke path. Publishable QPS/recall still needs a real SPLADE/BM25 sparse dataset harness. LEMUR needs training or reproducible model loading before storage or QPS rows matter. |
| 12 | Python policy | Settled for now: Python exposes stable workflows, not every Rust module. Keep HNSW construction/search/save-load, IVF-PQ snapshot/file/mmap search, and parallel batch search as the supported surface. Add a Rust module to Python only after it has stable benchmarks, persistence behavior, examples, and a clear user workflow. Rust-only gaps today: DiskANN, `store`, FreshGraph, filtered search/update APIs, and HNSW binary segments. |
| 13 | LSH/sketch boundary | The `lsh` feature uses `sketchir` for cross-polytope hashing primitives. Keep `sketchir` focused on MinHash/SimHash/LSH sketches and durable sketch sidecars; keep vicinity focused on ANN storage, exact reranking, persistence modes, and fixed-recall benchmark rows. Benchmark sharing is useful, but PRT, RP-tree/RP-forest, SparseMIPS, and LEMUR should stay in vicinity unless their role becomes pure sketch generation. |
| 14 | External research claims | New implementation-scouting evidence points to Qdrant's mmap/residency model, Weaviate sparse visited sets, Qdrant/Vespa selectivity-gated ACORN, Faiss FastScan layout validation, DiskANN provider boundaries, and Qdrant-style private SIMD kernels as the most actionable prior art. Still verify newer roadmap claims before implementation: Extended RaBitQ, VSAG layout tricks, IP-DiskANN, PAG, SAQ, and ARM/SVE2 kernels. Keep `innr` as the optional dense-distance SIMD dependency; use local `pq_simd` work for PQ-code/LUT kernels that `innr` does not cover. |
| 15 | Dataset difficulty metadata | First sampled profile script exists for VEC1/NBR1 datasets, and local profiles now cover SIFT, GloVe-25/50/100/200, Deep Image, NYTimes, Fashion-MNIST, MNIST, and GIST. Optional generated split labels are reported when present, and `scripts/summarize_dataset_profiles.py` renders profile JSONs into the docs table shape. `scripts/summarize_ann_results.py --profile-dir PATH` now joins exact profile metrics into ANN coverage rows while leaving capped dataset labels unlinked unless an exact profile exists. |
| 16 | Profiling depth | Runtime profiles now cover HNSW search, ACORN filtered search, DiskANN direct-file rows, IVF-PQ ADC/allocation paths, dataset difficulty, and a same-binary `m_max=16` versus `m_max=32` HNSW search-only comparison. The ledger also records a build-path sample where `rustc` stalled in `readdir` over a large `target/debug/deps`; use isolated `CARGO_TARGET_DIR` values for future profile targets. HNSW binary inspection confirmed an indirect `blr x7` in `flush_batch`, and the new `distance_dispatch` Criterion group shows function-pointer dispatch costs on low-dimensional `innr` kernels, but the broad, `flush_batch`, and cosine-only HNSW dispatch rewrites all regressed or missed the keep threshold. The latest plain-HNSW symbolized sample still puts the largest leaf bucket inside `innr::dense::dot`; the ACORN samples first put the largest buckets in `HashMap::insert` and `reserve_rehash`, then shifted after dense tracking to inlined ACORN loop work plus `innr::dense::dot`. The kept ACORN fix is a safe visited-tracker change, not unsafe SIMD. The graph prefetch experiment removed a product unsafe surface and improved or held controls, so do not add local HNSW unsafe before safe heap/frontier/layout experiments. Next actual performance change should still record baseline, profiler target, negative controls, before/after, and rejected hypotheses in `docs/benchmark-results.md`. |

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
- Do not add or widen `unsafe` for a performance idea until the safe path has
  been profiled and the candidate is measured against its negative controls.
  Keep necessary SIMD `unsafe` inside small dispatch/kernel wrappers with
  parity tests.
