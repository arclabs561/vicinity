# Testing Strategy

Our testing strategy is grounded in real issues from ANN libraries (hnswlib, faiss, usearch, and related ecosystems).

## Known Failure Modes (from real issues)

### 1. Neighbor Selection Parameter Bugs

**hnswlib #635, #606**: Using `M` instead of `M_max` for layer 0 results in fewer connections than configured.

**Our mitigation**: Property tests verify that layer 0 has expected connectivity:
```rust
#[test]
fn layer0_has_correct_connectivity() {
    // Verify layer 0 uses M_max, not M
}
```

### 2. Deletion/Streaming Bugs

**hnswlib #608, #626**: Use-after-free and search failures after deleting vectors.

**Our mitigation**: 
- Integration tests that add, delete, then search
- Property tests with random add/delete sequences

### 3. Normalization Bugs

**hnswlib #592**: Brute force search doesn't normalize query for cosine distance.

**Our mitigation**:
- Document the normalization requirement clearly
- Provide `normalize()` helper in examples
- Test that cosine distance returns expected values for known inputs

### 4. Integer Overflow

**faiss #4295**: `ntotal * M` overflow on 60M+ vectors with M=64.

**Our mitigation**:
- Use `usize` (64-bit on 64-bit platforms) for size calculations
- Property tests with boundary values

### 5. Quantization Issues

**usearch #405**: i8 quantization + inner product gives wrong results.

**Our mitigation**:
- Document metric compatibility for each quantization type
- Integration tests for ternary quantization recall

### 6. Small Dataset Performance

**hnswlib #618**: Suboptimal latency on small datasets (13K vectors).

**Our mitigation**:
- Document parameter guidance for different dataset sizes
- Examples show latency/recall tradeoffs

## Test Categories

### Unit Tests (`src/**/*_test.rs`)

Fast, isolated tests for individual functions.

### Integration Tests (`tests/`)

- `hnsw_e2e.rs`: Full index lifecycle (build, search, persist, reload)
- `edge_cases.rs`: Boundary conditions, empty inputs, single elements
- `property_tests.rs`: Randomized invariant checking (determinism, distance monotonicity)
- `regression_known_bugs.rs`: Regression tests for fixed bugs (Vamana doc_id, etc.)
- `robustness.rs`: Stress tests and adversarial inputs
- `cross_algorithm_consistency.rs`: Cross-algorithm recall agreement
- `correctness_regression.rs`: Recall floor enforcement
- `invariants.rs`: Structural invariants (graph connectivity, degree bounds)
- `persistence_robustness.rs`, `chaos_persistence.rs`, `diskann_persistence_test.rs`: Persistence layer tests
- `cross_crate_integration.rs`: Integration with `innr`, `qntz`, `clump`

### Benchmarks (`benches/`)

- `recall.rs`: Measures recall@k vs brute force
- `distance.rs`: SIMD dispatch performance
- `scaling.rs`: Performance vs dataset size

### Examples (`examples/`)

- Serve as smoke tests (`cargo run --example X`)
- Document expected outputs

## Recall Regression Detection

CI fails if recall@10 drops below 80% on the standard test:

```yaml
- name: Run recall benchmark
  run: |
    cargo run --release --example 02_measure_recall 2>&1 | tee recall_output.txt
    if ! grep -q "84\." recall_output.txt; then
      echo "Recall regression detected!"
      exit 1
    fi
```

## Adding New Tests

When adding tests, consider:

1. **What real-world bug does this catch?** Link to issue if possible.
2. **Is this a unit test or integration test?** Unit = one function, integration = multiple components.
3. **Should this be a property test?** If testing an invariant (e.g., "distance is always non-negative").

## Running Tests Locally

```bash
# Fast iteration (unit tests only)
cargo test --lib

# Full test suite
cargo test --no-default-features --features hnsw

# Property tests (slower, more thorough)
cargo test property_

# With coverage
cargo llvm-cov --no-default-features --features hnsw
```

## CI Matrix

| Job | Platform | Tests | Purpose |
|-----|----------|-------|---------|
| `test` | ubuntu-latest | Clippy + build + test | Primary validation |
| `test-arm` | macos-latest | Build + test | ARM/NEON code paths |
| `msrv` | ubuntu-latest | `cargo check` | MSRV 1.89 compatibility |
| `feature-matrix` | ubuntu-latest | `cargo-hack` each feature | Feature flag correctness |
| `cross-compile` | ubuntu-latest | x86_64 + aarch64 | Cross-target compilation |
| `recall-regression` | ubuntu-latest | Example | Recall floor (80% at ef=100) |
| `regression` | ubuntu-latest | `regression_known_bugs` | Known bug non-regression |
| `docs` | ubuntu-latest | `cargo doc` | Doc completeness |
