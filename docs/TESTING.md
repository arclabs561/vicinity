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
- `property_tests.rs`: Randomized invariant checking

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
| `test` | ubuntu-latest | All | Primary validation |
| `test-arm` | macos-latest | All | ARM/NEON paths |
| `msrv` | ubuntu-latest | Check | API stability |
| `recall-regression` | ubuntu-latest | Example | Quality guard |
| `docs` | ubuntu-latest | Doc build | Doc completeness |
