# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The 0.x
series is unstable: minor bumps may break the public API.

## [Unreleased]

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
