# Preparing data

Every row in `benchmarks/` was measured on a file this page describes how to
produce. One entry point, one manifest, no dataset committed to git.

```sh
uv run scripts/prepare_data.py list
uv run scripts/prepare_data.py synthetic --points 10m --dim 96
uv run scripts/prepare_data.py ann-benchmarks glove-25-angular
uv run scripts/prepare_data.py verify
```

The data root is `ANN_DATA_ROOT`, else `$REBUILDABLE_ROOT/vicinity-data`, else
`./data`. `manifest.json` in that root records, per dataset, the path, byte size,
sha256, row count, dimension and the command that produced it. `verify`
recomputes every checksum, so a truncated download fails loudly instead of
quietly benchmarking a smaller corpus.

## Synthetic

`examples/large_scale.rs` owns the generator for the tier domains; `prepare_data`
calls it with `--prepare-only`, which writes the corpus and stops before ground
truth. The file format is self-describing: `b"VICDS01\0"`, `dim: u32`,
`count: u64`, then row-major `f32`, so a reader can verify shape without a
sidecar.

| domain | generator | measured LID | stands in for |
| --- | --- | ---: | --- |
| `uniform` | i.i.d. uniform in [-1,1]^96 | 50.7 | the baseline where every method is worst |
| `clustered` | 16 uniform centres, sigma 0.25 | 41.3 | few strong clusters, mild LID |

The other five domains (`text`, `image`, `sparse`, `neardup`, `lowdim`) are
generated in memory by `examples/all_families.rs` at 10k only and are **not** yet
wired to the file format. `prepare_data.py list` says so rather than offering a
command that would fail.

## Public sets

`scripts/download_ann_benchmarks.py` fetches HDF5 and converts to `VEC1`/`NBR1`
with pinned sha256 per file; `prepare_data` calls it and records the result in
the manifest.

## Which benchmark is alive, and what replaced what

"ANN-Benchmarks is deprecated" is not true as of 2026-11-03. The repository
`erikbern/ann-benchmarks` is not archived, last pushed 2026-07-10, and its data
endpoints still serve (`glove-25-angular.hdf5` answered HTTP 200, 127 MB). So the
classic sets remain reproducible and we keep using them.

What changed is coverage, not the tool:

| benchmark | state | what it adds | what it cannot give us |
| --- | --- | --- | --- |
| ann-benchmarks | alive, data served | SIFT, GIST, GloVe, Fashion-MNIST, NYTimes, LastFM | no out-of-distribution queries, no modern embedding models, no int8/binary variants |
| VIBE (`vector-index-bench/vibe`, MIT) | active: 146 commits, last 2026-09-23; paper arXiv:2505.17810, JDCMLR 2026 | 11 in-distribution and 8 OOD datasets, quantised (int8/binary) variants, 22 algorithms, single-core protocol | its data pipeline runs embedding models in a GPU container, so "download the set" is not a one-command operation |
| big-ann-benchmarks | active competitions | billion-scale, plus filtered, sparse, streaming and OOD tracks | the billion-scale tracks assume more disk and RAM than one machine has |
| VectorDBBench (`zilliztech`) | active | end-to-end database comparison, insertion, filtered, full-text | measures deployments, so it belongs beside our server rows, not our index rows |

VIBE is the successor for the question "does this hold on modern embeddings and
on out-of-distribution queries", which is exactly the gap `docs/scale-and-parity-plan.md`
names as P5. It is not a drop-in replacement for the ann-benchmarks download
because its corpus is produced by running models, not by fetching a file.

## What each row file used

| rows | dataset | prepared by |
| --- | --- | --- |
| `benchmarks/baseline.jsonl` | 7 synthetic domains at 10k, in memory | inside `all_families` (no file) |
| `benchmarks/large-scale-100k-*`, `-1m-*`, `-3m-*`, `-10m-*` | `uniform@n` | `prepare_data.py synthetic --points n` |
| `benchmarks/hnswlib-parity-100k.jsonl`, `-1m.jsonl` | `clustered@n` | generated inside `tools/parity-rows` (clustered16, same draws as `all_families`) |
| `benchmarks/fashion20k-k10.jsonl` | Fashion-MNIST | `download_ann_benchmarks.py` |
| `docs/glove-25-angular.jsonl` and friends | public sets | `download_ann_benchmarks.py` |
| `benchmarks/qdrant-parity-100k.jsonl` | `uniform@100k`, same bytes | `scripts/qdrant_parity.py` dumps `corpus.f32` / `queries.f32` |

The parity and Qdrant rows use a *different* generator spelling for their corpus,
which is why their `dataset` field says `clustered@100k` rather than
`uniform@100k`. That is a naming inconsistency, not a protocol difference: each
row's `_meta` names the file it read.
