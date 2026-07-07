# Documentation

## For users

- [GUIDE.md](GUIDE.md): quick start, HNSW usage, distance metrics,
  LID, common pitfalls, worked examples.

## Benchmark data

- [benchmark-results.md](benchmark-results.md): recall/QPS tables
  and analysis across datasets.
- [datasets.md](datasets.md): ANN benchmark datasets, download
  instructions, and the format spec.

Benchmark results are stored as JSONL. `examples/ann_benchmark.rs` writes one
`_meta` line per run, then one measurement line per algorithm/config/storage
mode:

```jsonl
{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","metric":"cosine","result_schema":2,"rustc":"rustc ...","query_limit":10000,"queries":10000}}
{"algorithm":"hnsw","params":{"m":16,"ef_search":50},"storage_mode":"in_memory","cache_state":"warm_after_build","recall_at_10":0.95,"qps":5000.0,"p50_us":180.0,"p95_us":260.0,"p99_us":320.0,"rss_kb":123456}
```

Common result files:

- `glove-25-angular.jsonl`: GloVe-25 (1.18M, 25-d, angular distance)
- `gist-960-euclidean.jsonl`: GIST-960 (1M, 960-d, L2)
- `sift-128-euclidean.jsonl`: SIFT-128 (1M, 128-d, L2)

Use `storage_mode` and `cache_state` when comparing rows. `in_memory`,
`snapshot_loaded`, `file`, and `mmap` are different workloads. `--resume`
skips rows already present for the same dataset metadata and parameters;
`--fresh` starts a new result file.

Plots live in `plots/` and are regenerated from JSONL via
`scripts/plot_comparison.py`.

## Background

- [landscape.md](landscape.md): the ANN algorithmic landscape. Principles,
  algorithm families, mathematical foundations, and where the field is heading.
  Canonical reference updated as research evolves.
- [references.md](references.md): bibliography. Primary sources for
  every algorithm and technique referenced in the codebase.

## For contributors

- [TESTING.md](TESTING.md): test organization, feature gates, and how
  to run tests per module.
