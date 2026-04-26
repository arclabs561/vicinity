# Contributing to vicinity

Thanks for your interest. vicinity is an approximate-nearest-neighbor library: HNSW + custom-distance + quantization, with a focus on correctness on cosine and Euclidean distances at scale.

## Before you start

For non-trivial work (new index types, distance metrics, on-disk format changes), open an issue first to align on scope. Drive-by bug fixes and doc patches don't need an issue.

## Setup

- Rust toolchain: stable, MSRV `1.89`. Use `rustup` to manage.
- Optional: `cargo-nextest` for faster test runs.
- Optional: `just` — canonical recipes (`brew install just` or `cargo install just`).

```
just check    # fmt + clippy + tests
just test     # full test suite
```

## Style

- Direct, lowercase prose in commits. No marketing words ("powerful", "robust", "elegant"). No em-dashes in prose.
- Commit messages: `vicinity: short lowercase description`. One commit per logical change.
- `cargo fmt` and `cargo clippy --all-targets --all-features -- -D warnings` must pass before `git add`.

## Testing

- `just test` (or `cargo test --all-features`) runs the full suite.
- HNSW / NSW tests with fewer than ~10 nodes are flaky (graph connectivity is degenerate). Use 15-20+ nodes for deterministic test behavior.
- Distance metric tests must use vectors normalized for the metric being tested (L2-normalized for cosine, etc.). Un-normalized data changes search difficulty and makes regression numbers meaningless.

## Benchmarks

Benchmark scripts live under `examples/`. Recall numbers in the README correspond to GloVe-25 (1.18M) at the documented hyperparameters; don't ship a metric change without re-running on the same dataset.

## Pull requests

- Keep PRs scoped to one concern.
- Show before/after for behavior changes (especially recall@k or QPS deltas).
- Link the related issue.
- CI must be green before requesting review.

## License

Dual-licensed under MIT or Apache-2.0 at your option. By contributing you agree your contributions are licensed under both.
