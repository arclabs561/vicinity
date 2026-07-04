# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The 0.x
series is unstable: minor bumps may break the public API.

## [Unreleased]

### Added

- Added an `updatable_store` example covering add, checkpoint, delete, reopen,
  and search through the optional `store` feature.
- Added a `store_reopen_diagnostics` example that prints first snapshot-search
  cost with persisted HNSW sidecars present versus after deleting sidecars and
  forcing source-segment rebuilds.
- Added `store::SnapshotIndex`, a read-only checkpoint view that opens the
  segstore manifest and queries persisted HNSW sidecars before falling back to
  one source-vector segment when a sidecar is missing or stale.

### Changed

- `examples/dual_branch_demo.rs` now reports standard-HNSW recall against
  metric-matched brute-force ground truth and labels Dual-Branch recall changes
  as deltas rather than assumed improvements.
- `store::UpdatableIndex` now keys its in-memory per-segment HNSW cache by
  segstore's stable segment ids instead of `Arc` pointers, and prunes stale cache
  entries when compaction/reclaim changes the segment set.
- Store writer searches now build the temporary writer-buffer HNSW from the
  buffer slice instead of cloning buffered vectors first.
- The `store` feature now requires `segstore = "0.4.1"` for manifest-only
  snapshot reads.

### Fixed

- Corrected the RaBitQ example command and README quantization feature note to
  match the current feature split.
- `store::UpdatableIndex::{compact, compact_tiers, reclaim}` now persist sidecars
  for newly merged segments immediately after segstore checkpoints them, instead
  of waiting for the next search to rebuild and write the sidecar lazily.

## [0.10.5] - 2026-06-30

### Added

- `HNSWIndex::to_postcard` and `HNSWIndex::from_postcard` are now available
  with the `persistence` feature, so consumers can persist graph sidecars
  without enabling the full `store` feature.
- The `store` feature persists per-segment HNSW sidecars and loads them on
  restart when the build recipe and live id set still match.

### Changed

- The `store` feature now requires `segstore = "0.4"` for persisted per-segment
  index sidecars.

### Fixed

- Stale, corrupt, or recipe-mismatched sidecars are rejected and rebuilt instead
  of being used for search.

## [0.10.4] - 2026-06-28

### Added

- `store::UpdatableIndex::extend(vectors)`: bulk ingest that syncs the write-ahead
  log once per batch instead of once per vector (each vector is dimension-checked
  and L2-normalized before any is ingested). ~3.4x faster than a loop of `add` on a
  real filesystem (bench `ingest_fs`: 12.6ms vs 3.7ms / 4000 vectors).

### Changed

- The `store` feature now requires `segstore = "0.3"`; the internal `merge_segments`
  takes `&[&Segment]` (segstore 0.3's by-reference signature).

### Fixed

- The `store` bench opts out of `clippy::unwrap_used` (benches legitimately unwrap
  on setup), fixing a pre-existing CI failure.

## [0.10.3] - 2026-06-27

### Changed

- A `delete` now invalidates only the cached HNSW of the segment that holds the
  id, not the whole cache, so one delete no longer forces every segment to
  rebuild on the next query.

## [0.10.2] - 2026-06-27

### Added

- `store::UpdatableIndex::compact_tiers()`: one round of size-tiered compaction
  (merge similarly-sized segments), keeping segment count bounded without a full
  `compact()`.

## [0.10.1] - 2026-06-27

### Added

- `store::UpdatableIndex::reclaim(min_live_ratio)` and `space_amplification()`
  (via the new `Store::live_len`): cheap tombstone reclamation, merging only the
  delete-heavy segments instead of a full compaction.

## [0.10.0] - 2026-06-26

### Changed

- `store::UpdatableIndex` now targets segstore 0.2 and caches the per-segment HNSW
  by the segment's stable `Arc` identity, so a sealed add rebuilds only the one new
  segment instead of the whole corpus. Measured: a single add after 5k vectors took
  the next search from a full ~193ms rebuild to ~5ms (40x -> 1x). Deletes still
  invalidate the cache (a tombstone changes the live filter); they are far rarer
  than adds.

## [0.9.1] - 2026-06-26

### Fixed

- `store::UpdatableIndex` caches the per-segment HNSW indexes and rebuilds them
  only on mutation (add/delete/compact), instead of rebuilding every segment's
  graph on every query.
- `store::UpdatableIndex::add` now returns an error for a vector whose dimension
  does not match the index, rather than silently dropping it from every rebuild.
- Restored the `cdylib` crate-type so maturin builds the `pyvicinity` extension
  (it reported "No Cargo targets to build" without it).

### Changed

- `store` docs no longer claim an "exact" cross-segment merge. HNSW search is
  itself approximate, so the merged result is approximate.

## [0.9.0] - 2026-06-26

### Added

- Optional `store` feature: `store::UpdatableIndex`, an updatable, durable
  multi-segment ANN index backed by
  [`segstore`](https://crates.io/crates/segstore). A per-segment HNSW is searched
  and merged (exact top-k); vectors are L2-normalized on ingest. Incremental
  add/delete plus write-ahead log, checkpoint, compaction, and crash recovery.
  Opt-in; the default build does not depend on segstore.
- `cli` feature: a `vicinity` binary that builds an HNSW index from a JSONL file
  of dense vectors and runs k-NN queries.

## [0.8.2] - 2026-06-11

### Fixed

- Corrected `HNSWBuilder::m_max` and `HNSWIndex::new` parameter docs: `m_max`
  caps neighbors on the base layer (the paper's `M_max0`, default `2 * m`).
  The previous docs claimed it applied to non-zero layers and defaulted to
  `m`; construction has always used `m_max` for layer 0 and `m` above it.
- Corrected `HNSWIndex::search_with_distance` docs: the closure's
  `internal_id` indexes the BFS-reordered vector storage after `build()`,
  not insertion order. Parallel arrays must be built from `raw_vectors()`
  after `build()`.

### Added

- Documented contracts on `HNSWIndex::build` (BFS reorder of internal IDs,
  idempotency, RNG determinism via `HNSWParams::seed`), `search` (error
  variants, `k > n` returns fewer results, `ef < k` silently clamped to `k`),
  and `raw_vectors` (post-build reorder warning, previously only on the
  legacy `vectors_raw` alias).
- Documented the segment persistence compatibility contract on the
  `persistence` module and in the README: 0.6.x legacy fallback,
  future-version rejection, corrupt/truncated input returns errors rather
  than panicking.
- Tests pinning future-format-version rejection, garbage-byte JSON load,
  `k > n` and `k = 0` search results, exact error variants for dimension
  mismatch / add-after-build / empty build, and `build()` idempotency.

## [0.8.1] - 2026-04-30

### Fixed

- `HNSWIndex::search` and `batch_search_mqo` now also normalize the query
  when `auto_normalize=true` for cosine and angular metrics. Previously the
  flag normalized inserts only, so search queries silently produced
  meaningless distances under cosine (the dot-only fast path assumes both
  sides are unit-norm). The Python binding already did this in `prep_query`;
  the Rust API now matches.
- README HNSW snippet ran but raised `InvalidParameter` on cosine because
  the constant `[0.1; 128]` isn't unit-norm. Now uses `auto_normalize(true)`
  on the builder so the literal copy-paste works.
- README IVF-PQ snippet trained 256 clusters on 2 vectors and silently
  returned distance=0 for every result. Reframed as API-shape with a note
  about training-data scale plus a pointer to `examples/ivf_pq_demo.rs`.

### Added

- `pyvicinity` Python bindings on PyPI (abi3 wheel, CPython 3.9+).
  Ships hand-written `.pyi` stubs + `py.typed` (PEP 561), exposes
  `HNSWIndex` + `DistanceMetric`. Verified in CI by `mypy.stubtest`,
  `mypy --strict`, `ruff`, and 25 pytests.
- `examples/python/`: text similarity (sentence-transformers),
  recall@k sweep, and ann-benchmarks/VIBE drop-in wrapper.
- Sentinels `pyvicinity.MISSING_LABEL` (= -1) and
  `pyvicinity.MISSING_DISTANCE` (= +inf) for masking short rows in
  `batch_search` results, matching faiss's convention.
- `auto_normalize` constructor flag normalizes vectors on **both**
  insert and query (closes the hnswlib #592 class of correctness
  bug). Rejected at construction with `L2`/`InnerProduct` since those
  metrics don't have spherical semantics.
- CI gate on `publish-pypi` workflow: aborts unless the source SHA's
  `ci.yml` run is `success`.
- musllinux x86_64 + aarch64 wheel matrix in publish-pypi.

### Changed

- All GitHub Actions in `publish-pypi.yml` pinned to specific minor
  versions instead of mutable major tags.

## [0.8.0] - 2026-04-27

### Changed (breaking)

- Renamed `esg` module to `range_filtered`. The 0.7.x `esg` name implied
  fidelity to the partition-aware structure of Yang et al. 2025
  (arXiv:2504.04018); the shipped implementation is the paper's
  strawman baseline (HNSW + attribute-range post-filter), so the rename
  reflects what the module actually does. A paper-fidelity ESG variant
  is planned as a separate module.
  - `EsgIndex` -> `RangeFilteredIndex`
  - `EsgParams` -> `RangeFilteredParams`
  - feature `esg` -> `range_filtered`
  - module path `vicinity::esg::*` -> `vicinity::range_filtered::*`
  - `RangeFilteredParams::num_checkpoints` removed (was deprecated in
    0.7.3; the partition-aware path that consumed it was never
    implemented).
- `vicinity::spectral` module visibility downgraded from `pub` to
  `pub(crate)`. The Marchenko-Pastur helpers were reachable only via
  `ADSamplingState::new_auto`, which has no production consumer. The
  `rmt-spectral` feature and the `rmt` dep remain available for the
  internal `new_auto` path; the public wrappers `mp_lambda_plus` and
  `count_mp_outliers` are no longer accessible from outside the crate.

### Migration from 0.7.x

```diff
-vicinity = { version = "0.7", features = ["esg"] }
+vicinity = { version = "0.8", features = ["range_filtered"] }

-use vicinity::esg::{EsgIndex, EsgParams};
+use vicinity::range_filtered::{RangeFilteredIndex, RangeFilteredParams};

 let params = EsgParams {
-    num_checkpoints: 16,  // remove
     hnsw_m: 16,
     hnsw_ef_construction: 200,
     ef_search: 100,
 };
```

## [0.7.2] - 2026-04-27

### Changed

- `FreshGraphIndex::insert` no longer rebuilds the per-node `inbound_count`
  array from scratch on every call. The count is maintained as a struct
  field across `add_slice` / `insert` / `build` / `compact` and updated
  per neighbor-list mutation. Per-insert cost was
  `O(max_degree²) + O(n × max_degree)` and is now `O(max_degree²)` -
  flat in `n` once every node is at cap. Measured on Apple Silicon at
  dim=32, max_degree=16: ~120 µs/op at n=10K..50K vs an extrapolated
  ~920 µs/op pre-fix at n=50K (~7x speedup at scale). Both
  reachability invariants (I1 entry-point liveness, I2 in-edge
  survival under reverse-edge prune) survive the refactor.
- `HNSWIndex::save_to_file` is now atomic and durable. Writes to a
  sibling temp file, fsyncs the temp, atomically renames into place,
  then fsyncs the parent directory so the rename survives a crash on
  filesystems that journal data and metadata independently (XFS,
  some ext4 configs, overlayfs). A crash mid-write leaves the prior
  file intact rather than truncated. P-HNSW (MDPI 2025) flagged
  partial-write corruption as a recurring failure mode; this routine
  is the in-tree counterpart to
  `durability::storage::FsDirectory::atomic_write` (which is gated
  behind the `persistence` feature). Cost: ~13 ms/op (durability
  syncs) vs ~0.5 ms/op for the prior non-atomic path; trade-off is
  correctness, dominated by typical save cadence.
- `AcornConfig` docstring now names the selectivity regimes where
  2-hop expansion is load-bearing (2-20%), where it's overhead
  (>20%), and where it hits a recall floor (<2%, fall back to
  `selectivity_search` with `matching_ids`). Doc-only change;
  `selectivity_search`'s Low branch already routes correctly.

### Added

- `acorn_search_with_stats` returns an `AcornStats { two_hop_invocations,
  two_hop_nodes_examined }` struct alongside results, so regression
  tests can assert "the 2-hop branch fired N>=1 times" rather than
  relying on a noisier black-box recall delta. The existing
  `acorn_search` becomes a thin wrapper that discards stats; signature
  unchanged. `AcornStats` is `#[non_exhaustive]` so additional
  counters can be added without bumping major. The wrapper has no
  measurable overhead in release builds (Rust inlines the
  discard-stats indirection; measured 803 ns/op for both entry
  points).
- `tests/hnsw_integration_tests.rs::test_acorn_two_hop_branch_fires_at_sparse_selectivity`
  asserts the 2-hop branch fires at least once at ~2.5% selectivity.
  Verified to catch the gate-disabled regression by toggling
  `enable_two_hop=false` and confirming the assertion fires with
  `two_hop_invocations=0`.
- `src/hnsw/graph.rs::tests::test_delete_with_repair_preserves_self_search_reachability`
  asserts every surviving doc_id is self-search-reachable after
  deleting 60% of nodes via `delete_with_repair`. Existing tests
  stop at 20-30% deletion and check result count or top-k recall;
  this guards the failure mode where many cumulative repair calls
  leave individual live nodes orphaned.
- `src/fresh_graph/mod.rs::tests::build_post_state_every_non_entry_node_has_inbound_edge`
  is a static post-build connectivity guard: every non-entry node
  has at least one inbound edge. Adapted from qdrant's
  `test_graph_connectivity` shape, parameterized over five seeds.
  Catches build-time orphans the dynamic delete-reinsert cycle
  test wouldn't surface.
- `src/hnsw/graph.rs::tests::test_hnsw_save_to_file_uses_rename_not_truncate`
  is a Unix-only inode-based gate guard: truncate-overwrite preserves
  the inode, rename-overwrite replaces it. Verified to catch the
  regression by reverting `save_to_file` to non-atomic and confirming
  this test fails (left==right) while the happy-path roundtrip tests
  still pass.
- `src/hnsw/graph.rs::tests::test_hnsw_build_save_reload_search_equality`
  end-to-end build -> save_to_file -> drop -> load_from_file -> search-
  equality round-trip via the path-based API on a fresh live index.
  Plugs the gap between the in-memory roundtrip (already covered)
  and the v0 fixture decode tests (which pin a static byte sequence
  but don't exercise the writer).
- Three `#[ignore]`'d perf probes (`orphan_protection_scaling_probe`
  extended to n=50K; `save_to_file_overhead_probe`;
  `acorn_stats_overhead_probe`) for one-shot validation of the
  changes above. Same shape as the existing scaling probe; not run
  in CI.

### Dependencies

- No upstream dependency bumps. The session also added a
  `sync_parent_of_path` helper to `durability` (0.6.4) for the
  upstream concern that motivated `save_to_file`'s atomic refactor;
  vicinity's own `save_to_file` inlines the equivalent pattern
  rather than taking the helper as a dep, since `save_to_file` is
  `serde`-gated and `durability` is `persistence`-gated.

## [0.7.1] - 2026-04-26

### Added

- Real v0.6.2-written segment fixture under `tests/fixtures/v0_segment_dim8/`
  (~2.6 KB total) and a regression test
  (`real_v0_fixture_loads_and_searches_correctly`) that loads it via
  the v1 reader and asserts the search ordering matches the original
  v0.6.2 run. Guards the legacy v0 decode path against silent
  regression and documents that 0.6.x → 0.7.x persistence migration
  works on real data. The 0.7.0 test set only proved no-panic on
  random bytes; it did not prove correctness on a real legacy file.
- `tests/edge_cases.rs::search_returns_external_doc_ids_at_high_offset`
  pins the external-`doc_id` contract at `u32::MAX - 200` (catches a
  regression where the search path returns an internal slot index).
- `tests/edge_cases.rs::zero_query_does_not_panic_with_auto_normalize`
  pins the documented zero-query path: search returns finite distances
  or an explicit error, never a panic on the `0 / ||q||` normalization
  step.
- `src/hnsw/graph.rs::tests::test_delete_with_repair_recall_floor_against_ground_truth`
  is a recall-floor companion to the existing
  `test_delete_with_repair_maintains_recall`. The original asserts
  result count and absence-of-deleted-ids; this adds a
  recall-vs-brute-force ground-truth assertion at >= 0.7 after
  deleting 20% of nodes.
- `tests/hnsw_integration_tests.rs::test_acorn_low_selectivity_returns_valid_results`
  guards ACORN at ~2.5% predicate selectivity. An earlier draft
  compared `enable_two_hop` true vs false and asserted a positive
  recall gap; in measurement the gap flipped sign at sparse
  selectivity (2-hop adds candidates that displace better ones in a
  tight beam). Reframed as a sparse-predicate regression guard with a
  0.5 recall floor.
- `src/fresh_graph/mod.rs::tests::delete_reinsert_cycles_preserve_reachability`
  guards two FreshGraph invariants that together close the
  delete-reinsert unreachability failure mode of arxiv:2407.07871:
  (I1) `entry_point` always references a non-tombstoned internal id,
  and (I2) `insert`'s reverse-edge prune never evicts a node whose
  only remaining in-edge would be the one through the neighbor being
  pruned at. Runs 200 cycles on a 60-vector graph and asserts every
  live `doc_id` is self-search-reachable. The assertion message names
  the broken invariant when it fires.
- `src/fresh_graph/mod.rs::tests::orphan_protection_scaling_probe`
  (`#[ignore]`'d, run with `--release --nocapture`) measures per-insert
  cost of the orphan-protection rebuild as `n` grows. Sample on Apple
  Silicon, dim=32: 12 µs/op below max-degree saturation, 300+ µs/op
  once every node hits cap. Validates the rebuild-per-call cost is
  acceptable at FreshGraph's target n; informs whether incremental
  inbound-count maintenance is worth designing.

### Changed

- `DistanceMetric::InnerProduct` doc comment now states that the
  caller is responsible for L2-normalization. Un-normalized
  inner-product ranking is dominated by magnitude, which is rarely
  the intended retrieval behavior (Milvus discussion #32479).
- `FreshGraph::build` and `FreshGraph::insert` now share a single
  `add_reverse_edge_protected` helper. Previously the reverse-edge
  prune was duplicated between the two paths and only `insert` had
  orphan protection. The build path now applies the same I2 invariant
  symmetrically: the initial RNG-prune pass cannot evict a node whose
  only remaining in-edge would be the one being considered.

### Fixed

- FreshGraph delete-reinsert unreachability. Two paired changes:
  - `delete()` now repromotes `entry_point` when it tombstones the
    medoid. Previously the entry-point field was left pointing at the
    deleted internal index, so subsequent searches and inserts rooted
    their beam at a dead anchor. Repromotion prefers a live neighbor
    of the stale entry (preserves graph locality) and falls back to a
    linear scan over live nodes only if the entire neighborhood is
    tombstoned.
  - `insert()`'s reverse-edge prune now computes a per-insert
    inbound-degree count, marks any candidate whose count is `<= 1`
    as orphan-protected, and runs RNG-prune only over the unprotected
    remainder. Without this, repeated cycles can evict a long-existing
    in-edge from every list that referenced it, leaving its target
    unreachable even though the node was never deleted.
  - The two changes are paired because the entry-point fix alone
    leaves the same unreachable set on the regression test (5 of 60
    live ids after 200 cycles). The orphan-protection change is the
    load-bearing one; the entry-point fix is shipped because rooting
    search at a tombstoned anchor is incorrect on its own merits.

## [0.7.0] - 2026-04-26

### Added

- HNSW segment binary persistence now starts with a magic + version header
  (`HNSW_SEGMENT_MAGIC = b"VCNHNSW\x01"` + `FORMAT_VERSION: u32 = 1`).
  Mismatched magic returns `PersistenceError::Format` instead of silently
  decoding garbage; unsupported version numbers return a descriptive error.
  Files written by 0.6.x lack the magic — the load path falls back to a
  legacy v0 decoder, so existing persisted indices round-trip transparently.
- `tests/persistence_robustness.rs::segment_binary` module:
  `loading_corrupt_magic_returns_format_error` (sanity), and
  `proptest_one_byte_corruption_never_panics` (random byte flip in the
  metadata header always produces `Result::Err`, never a panic).

### Fixed

- `docs/GUIDE.md` outlier-detection example referenced a non-existent
  `LidEstimate.category` field. Switched to `LidStats::from_estimates(&est).categorize(lid)`,
  which is what `examples/lid_outlier_detection.rs` already does.
- Three rendered intra-doc-link errors in `src/lemur/model.rs` (`[hidden_dim]`
  was parsed as a link target by rustdoc). All shape annotations now use
  backticked code so they render literally.
- `docs/datasets.md` referenced a non-existent `hdf5` Cargo feature. Updated
  to point at `scripts/download_ann_benchmarks.py` (the actual conversion path).
- `lib.rs` recommendation table claimed DiskANN persistence was mmap-based;
  `src/diskann/disk_io.rs:95` documents it as planned. Now reads "file-based
  save/load; mmap planned".
- README and benchmark-results.md called the GloVe-25 dataset "cosine"; the
  ann-benchmarks dataset and the rendered plot are angular distance.
- README's NSG row claimed a 50K hard cap; the limit is empirical, not
  enforced. Reworded to "build slows above ~50K vectors".

### Changed

- `docs/GUIDE.md` quickstart now uses `HNSWIndex::builder` to match the README
  and the bulk of the examples. The direct `HNSWIndex::new(dim, m, m_max)`
  constructor is still exposed.
- README's Supported Algorithms table now surfaces LSH (cross-polytope) and
  `LsmIndex` (LSM-tree streaming HNSW), both previously implemented but
  undocumented.
- `KD-Tree`, `Ball Tree`, and `RP-Forest` rows are flagged "(experimental)"
  per `src/classic/mod.rs`.

## [0.6.2] - 2026-04-25

### Changed

- CI tightened to `-D warnings` for clippy across the per-feature matrix.
- Fixed six pre-existing clippy issues in production code (uncovered by the
  stricter CI).
- Test code is now allowed to use `unwrap`/`expect`/`needless_update` via
  scoped `#[cfg_attr(test, allow(...))]` in `lib.rs`.

[`6b92ae9`](https://github.com/arclabs561/vicinity/commit/6b92ae9) ·
[v0.6.1...v0.6.2](https://github.com/arclabs561/vicinity/compare/v0.6.1...v0.6.2)

## [0.6.1] - 2026-04-23

### Added

- `publish-pypi` GitHub Actions workflow with OIDC trusted publishing and a
  full wheel matrix.

### Changed

- Python package renamed `vicinity` → `pyvicinity` for PyPI registration.

### Fixed

- IVF-RaBitQ cross-cluster ranking via the qntz typed-edge API (corrected a
  cross-cluster comparability bug introduced when residuals are evaluated
  against different cluster centroids).

[v0.6.0...v0.6.1](https://github.com/arclabs561/vicinity/compare/v0.6.0...v0.6.1)

## [0.6.0] - 2026-04-22

### Added

- Vamana parallel build with batched rayon and deferred pruning. Measured
  9.5x speedup on SIFT-128 (35 min → 3.7 min) and 7.1x on GIST-960; default
  build batch is 4096.
- DiskANN parallel build (same batched-rayon pattern as Vamana).
- NSW parallel build.
- Batched distance computation in HNSW beam search (Faiss pattern). +10% QPS
  at `ef=100` on SIFT-128.
- SymphonyQG-VR (vertex-relative) variant with per-parent residual encoding,
  ported to qntz's type-safe edge API.

### Changed

- Module docs across the crate were updated to drop stale version references
  and machine-specific numbers from public doc comments.
- README opening trimmed to undersell tone (one-line description; no feature
  list in the tagline).

### Fixed

- SymphonyQG-VR cross-space bias: recall jumped from 55-86% to 99.9% at ef=400
  on the standard benchmark after correcting the cross-parent distance
  comparison.
- SymphonyQG-VR cross-parent distance comparability (precondition for the
  bias fix above).

### Removed

- Per-parent VR residual (kept the simpler global-rotation variant after the
  per-parent variant showed no recall benefit).

[v0.5.0...v0.6.0](https://github.com/arclabs561/vicinity/compare/v0.5.0...v0.6.0)

## [0.5.0] - 2026-04-12

Highlights from the 0.5 line: generation-counter visited-tracking in HNSW
search (replacing per-search HashSet allocation), devirtualized distance
dispatch, SymphonyQG search variant, and KD-Tree pruning improvements. See
`git log v0.4.0..v0.5.0` for the full commit list.

[v0.4.0...v0.5.0](https://github.com/arclabs561/vicinity/compare/v0.4.0...v0.5.0)

## [0.4.0] - 2026-04-05

Highlights: ADSampling integration, SQ4 (4-bit scalar quantization) module,
and dependency bumps for `innr`, `clump`, `sbits`, `rankops`. See
`git log v0.3.6..v0.4.0` for details.

[v0.3.6...v0.4.0](https://github.com/arclabs561/vicinity/compare/v0.3.6...v0.4.0)

## [0.3.x] - 2026-03 to 2026-04

The 0.3 line covered the initial public-API stabilization, the
`MetadataValue` enum and Range filter (breaking, in 0.3.5), and a series of
CI-greening commits. Earlier than 0.3 the project was pre-public; consult
`git log v0.3.0..` for full history.

[v0.3.0...v0.3.6](https://github.com/arclabs561/vicinity/compare/v0.3.0...v0.3.6)

[Unreleased]: https://github.com/arclabs561/vicinity/compare/v0.6.2...HEAD
