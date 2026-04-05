#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "=== Generating S scale (10K) ==="
rm -rf data/multiscale/S
uvx --with numpy python scripts/generate_multiscale_data.py --scale S

echo ""
echo "=== Running S benchmark ==="
cargo run --example 04_rigorous_benchmark --release -- --scale S

echo ""
echo "=== Generating M scale (100K) ==="
rm -rf data/multiscale/M
uvx --with numpy python scripts/generate_multiscale_data.py --scale M

echo ""
echo "=== Running M benchmark ==="
cargo run --example 04_rigorous_benchmark --release -- --scale M

echo ""
echo "=== Generating L scale (1M) ==="
rm -rf data/multiscale/L
uvx --with numpy python scripts/generate_multiscale_data.py --scale L

echo ""
echo "=== Running L benchmark ==="
cargo run --example 04_rigorous_benchmark --release -- --scale L

echo ""
echo "=== Done! ==="
