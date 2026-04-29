# vicinity -- task runner

default:
    @just --list

# ─── Standard ann-benchmarks suite ───────────────────────────────────────────

# Download an ann-benchmarks dataset (e.g. just download glove-25-angular)
download dataset:
    uv run scripts/download_ann_benchmarks.py {{dataset}}

# List available ann-benchmarks datasets
download-list:
    uv run scripts/download_ann_benchmarks.py --list

# Run ann-benchmark on a dataset with all compiled algorithms
ann dataset="data/ann-benchmarks/glove-25-angular":
    cargo run --example ann_benchmark --release --features hnsw,nsw -- {{dataset}}

# Run ann-benchmark with JSON output (for plotting)
ann-json dataset="data/ann-benchmarks/glove-25-angular":
    cargo run --example ann_benchmark --release --features hnsw,nsw -- {{dataset}} --json

# Full standard benchmark pipeline: download + run + plot
bench-standard:
    @echo "Downloading datasets (if needed)..."
    just download glove-25-angular || true
    just download sift-128-euclidean || true
    @echo "Running benchmarks..."
    mkdir -p data/ann-benchmarks/results
    just ann-json data/ann-benchmarks/glove-25-angular > data/ann-benchmarks/results/glove-25.jsonl || true
    just ann-json data/ann-benchmarks/sift-128-euclidean > data/ann-benchmarks/results/sift-128.jsonl || true
    @echo "Done. Results in data/ann-benchmarks/results/"

# ─── Rigorous benchmark (multi-run, CI, LID-stratified) ─────────────────────

# Generate synthetic multiscale data (S/M/L/B/T/P)
gen scale:
    uvx --with numpy python scripts/generate_multiscale_data.py --scale {{scale}}

gen-all:
    uvx --with numpy python scripts/generate_multiscale_data.py --scale all

# Run rigorous benchmark at a scale
rigorous scale:
    cargo run --example 04_rigorous_benchmark --release -- --scale {{scale}}

# Full rigorous suite
rigorous-all:
    just gen S && just rigorous S
    just gen M && just rigorous M
    just gen L && just rigorous L
    just plot

# Generate plots and report from rigorous benchmark data
plot:
    uvx --with numpy --with matplotlib python scripts/plot_pareto.py

# ─── Criterion microbenchmarks ───────────────────────────────────────────────

# Run all Criterion benchmarks
criterion:
    cargo bench

# Run a specific Criterion benchmark (e.g. just criterion-one hnsw)
criterion-one name:
    cargo bench --bench {{name}}

# ─── Development ─────────────────────────────────────────────────────────────

# Check all features compile
check:
    cargo check --all-features

# Run tests
test:
    cargo test

# Clippy (default features)
lint:
    cargo clippy --features hnsw -- -D warnings

# Format check
fmt:
    cargo fmt -- --check

# Full QA: fmt + lint + test
qa: fmt lint test

# ─── Python bindings (pyvicinity) ────────────────────────────────────────────

# Build + install the Python wheel into the local venv
py-build:
    .venv/bin/maturin develop --release

# Run the Python test suite (rebuilds wheel first)
py-test: py-build
    .venv/bin/python -m pytest tests/test_python.py -v

# Lint the Python sources with ruff
py-lint:
    .venv/bin/python -m ruff check pyvicinity tests/test_python.py

# Type-check a sample caller against the .pyi stubs
py-typecheck: py-build
    .venv/bin/python -m mypy --strict pyvicinity

# Verify the hand-written .pyi matches the compiled module (voyager pattern)
py-stubtest: py-build
    .venv/bin/python -m mypy.stubtest pyvicinity._core

# Full Python QA: build + lint + typecheck + stubtest + test
py-qa: py-build py-lint py-typecheck py-stubtest py-test
