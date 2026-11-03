# parity-rows (benchmark-only)

Emits rows in the `examples/all_families.rs` JSONL schema so
`scripts/benefits.py` and `scripts/ratchet.py` can read them unmodified:

```text
{"family","dataset","knob","knob_value","recall","p50_us","p95_us","p99_us",
 "qps","dists_per_query","build_s","mem_mib","ratio","dim","lid"}
```

Sides: `hnsw` (`vicinity`), `hnswlib` (the vendored C++ reference), and `usearch`
(native SIMD control). All three build and search the same normalised vectors
with `M = 16`, `ef_construction = 200`, `k = 10`, one thread, and the same
`ef_search` sweep: 10, 20, 50, 100, 200, 400.

## Protocol

- Dataset: `clustered16`, the generator from
  `examples/all_families.rs::generate_clustered16` (16 centres,
  `sigma = 0.25`, dim 96, seed `0x46414D494C59 ^ 0xC10000`). Base points keep the
  same draw order, so the 100k/1M corpora extend the 10k harness corpus.
- Vectors are L2-normalised once at generation. `vicinity` uses
  `DistanceMetric::Cosine` (its dot fast path), hnswlib uses
  `InnerProductSpace`, USearch uses `MetricKind::IP`. All three rank by
  `1 - dot`, which equals cosine distance on unit vectors.
- Ground truth: exact top-10 by `f32` inner product on the same vectors,
  untimed, followed by a timed exact scan whose p50 feeds `ratio`.
- Each `ef` row runs five untimed warm-up queries, then one timed pass over the
  query set; recall and latency come from that pass (`all_families::evaluate`).
- `build_s` is build-only: no index serialisation for hnswlib/USearch, no
  snapshot write for `vicinity`.
- `dists_per_query` is `null` for every row: `vicinity`'s counter requires the
  `distance-stats` feature, whose atomics would slow exactly one side, and the
  C++ libraries expose no comparable count in this shim.
- `mem_mib` is `null` on every row. All three sides build in the same process so
  a later side reuses pages freed by an earlier index — at 300k this produced a
  measured `0.0` for hnswlib — which would read as a footprint rather than an
  allocator artefact. A comparable per-index footprint needs one process per
  side and is out of scope here.

## Run

```text
cargo run --release --manifest-path tools/parity-rows/Cargo.toml -- \
    benchmarks/hnswlib-parity-100k.jsonl 100k
cargo run --release --manifest-path tools/parity-rows/Cargo.toml -- \
    benchmarks/hnswlib-parity-1m.jsonl 1m
```

The second positional argument accepts `100000`, `100k`, or `1m`; the third
(optional) is the query count, default 500. The rows go to argv[1], a human
table to stderr.

`PARITY_SIDES=hnsw,hnswlib` restricts the run to the named sides (default:
all three). The committed 1M file uses that setting with 200 queries: the two
single-threaded builds already take about half an hour at 1M, and adding the
USearch build would push the run past the ~30-minute budget. The 100k file
carries all three sides with 500 queries.

## AArch64 caveat

hnswlib's vendored commit has no NEON kernels; on Apple Silicon it runs its
scalar distance loop while `vicinity` and USearch use NEON. Recall comparisons
are unaffected (same data, same `M`/`ef_construction`/`ef`), but the p50 ratio
between hnswlib and the Rust/SIMD sides is partly a vectorisation difference,
not only a graph-quality difference.

This crate lives outside `vicinity`'s manifest on purpose: it links the
unpublished `hnswlib-sys` shim, and a path dependency without a version in the
parent manifest blocks `cargo publish` outright.
