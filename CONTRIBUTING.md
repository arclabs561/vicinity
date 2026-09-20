# Contributing to vicinity

Thanks for your interest. vicinity is an approximate-nearest-neighbor library: HNSW + custom-distance + quantization, with a focus on correctness on cosine and Euclidean distances at scale.

## Before you start

For non-trivial work (new index types, distance metrics, on-disk format changes), open an issue first to align on scope. Drive-by bug fixes and doc patches don't need an issue.

## Setup

- Rust toolchain: stable, MSRV `1.89`. Use `rustup` to manage.
- Optional: `cargo-nextest` for faster test runs.
- Optional: `just` — canonical recipes (`brew install just` or `cargo install just`).

```
just qa       # fmt + clippy + default-feature tests
just check    # compile all features
```

## Style

- Direct, lowercase prose in commits. No marketing words ("powerful", "robust", "elegant"). No em-dashes in prose.
- Commit messages: `vicinity: short lowercase description`. One commit per logical change.
- `cargo fmt` and `cargo clippy --all-targets --all-features -- -D warnings` must pass before `git add`.

## Testing

- `just test` runs the default-feature suite. Use `cargo test --all-features`
  for optional-feature tests; Python bindings require a Python 3.10+ interpreter.
- HNSW / NSW tests with fewer than ~10 nodes are flaky (graph connectivity is degenerate). Use 15-20+ nodes for deterministic test behavior.
- Distance metric tests must use vectors normalized for the metric being tested (L2-normalized for cosine, etc.). Un-normalized data changes search difficulty and makes regression numbers meaningless.

## Benchmarks

Benchmark runners live under `examples/`. Dataset sizes, parameters, and
measurement conditions are recorded in `docs/benchmark-results.md`. Re-run the
affected workload when changing a distance metric or search behavior.

## Pull requests

- Keep PRs scoped to one concern.
- Show before/after for behavior changes (especially recall@k or QPS deltas).
- Link the related issue.
- CI must be green before requesting review.

## License

Dual-licensed under MIT or Apache-2.0 at your option. By contributing you agree your contributions are licensed under both.
