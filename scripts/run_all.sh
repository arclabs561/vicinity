#!/bin/bash
set -e
cd "$(dirname "$0")/.."

# Ensure directories exist
mkdir -p data/multiscale

echo "========================================================"
echo "Starting Rigorous Multi-Scale Benchmark (S, M, L)"
echo "========================================================"
date

# Function to run a scale
run_scale() {
    local scale=$1
    echo ""
    echo "--------------------------------------------------------"
    echo "Processing Scale: $scale"
    echo "--------------------------------------------------------"
    
    echo "[1] Generating Data..."
    uvx --with numpy python scripts/generate_multiscale_data.py --scale "$scale"
    
    echo "[2] Running Benchmark..."
    cargo run --example 04_rigorous_benchmark --release -- --scale "$scale"
}

# Run scales
run_scale S
run_scale M
run_scale L

echo ""
echo "========================================================"
echo "Generating Plots"
echo "========================================================"
uvx --with numpy --with matplotlib python scripts/plot_pareto.py

echo ""
echo "Done."
date
