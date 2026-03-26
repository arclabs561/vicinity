//! Benchmark utilities for ANN evaluation.
//!
//! Provides metrics, dataset generation, and evaluation utilities for
//! measuring ANN index quality across multiple dimensions:
//!
//! - **Accuracy**: recall@k, precision@k, MRR
//! - **Speed**: QPS, latency percentiles
//! - **Memory**: bytes per vector, index overhead
//! - **Scaling**: performance vs dataset size, dimensionality
//!
//! # Standard Benchmark Datasets
//!
//! | Dataset | Size | Dim | Distance | Use Case |
//! |---------|------|-----|----------|----------|
//! | SIFT-1M | 1M | 128 | L2 | Image descriptors |
//! | GIST-1M | 1M | 960 | L2 | High-dimensional |
//! | GloVe-100 | 1.2M | 100 | Angular | Word embeddings |
//! | Fashion-MNIST | 60K | 784 | L2 | Small baseline |
//!
//! Reference: <https://ann-benchmarks.com/>

pub mod datasets;
pub mod evaluation;
pub mod memory;
pub mod metrics;

pub use datasets::{create_benchmark_dataset, create_clustered_dataset, Dataset};
pub use evaluation::{
    evaluate, generate_clustered_dataset, generate_normalized_clustered_dataset,
    generate_uniform_dataset, mrr, EvalDataset, EvalResults,
};
pub use memory::{IndexMemoryStats, MemoryTracker};
pub use metrics::{precision_at_k, recall_at_k};
