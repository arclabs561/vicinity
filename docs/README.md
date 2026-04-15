# Documentation

## For users

- **[GUIDE.md](GUIDE.md)** -- Quick start, HNSW usage, distance metrics,
  LID, common pitfalls, worked examples.

## Benchmark data

- **[benchmark-results.md](benchmark-results.md)** -- Recall/QPS tables
  and analysis across datasets.
- **[datasets.md](datasets.md)** -- ANN benchmark datasets: what's
  available, how to download, format spec.

Benchmark results are stored as JSONL (one measurement per line):
- `glove-25-angular.jsonl` -- GloVe-25 (1.18M, 25-d, cosine)
- `gist-960-euclidean.jsonl` -- GIST-960 (1M, 960-d, L2)
- `sift-128-euclidean.jsonl` -- SIFT-128 (1M, 128-d, L2)

Each row: `{"algorithm":"name","recall_at_10":0.95,"qps":5000,"git_sha":"abc1234"}`.
The `git_sha` field tracks which code version produced the result. The
benchmark runner (`examples/glove_all_algos.rs`) auto-invalidates cached
results when the SHA changes.

Plots live in `plots/` and are regenerated from JSONL via
`scripts/plot_comparison.py`.

## Background

- **[landscape.md](landscape.md)** -- The ANN algorithmic landscape:
  principles, algorithm families, mathematical foundations, and where the
  field is heading. Canonical reference updated as research evolves.
- **[references.md](references.md)** -- Bibliography. Primary sources for
  every algorithm and technique referenced in the codebase.

## For contributors

- **[TESTING.md](TESTING.md)** -- Test organization, feature gates, how
  to run tests per module.
