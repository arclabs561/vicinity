# vicinity User Guide

Approximate nearest neighbor search and local intrinsic dimensionality.

## Quick Start (5 minutes)

```rust
use vicinity::hnsw::HNSWIndex;

// 1. Create index
let mut index = HNSWIndex::new(768, 16, 32)?;

// 2. Add vectors
for (id, vec) in embeddings.iter().enumerate() {
    index.add(id as u32, vec.clone())?;
}
index.build()?;

// 3. Search
let results = index.search(&query, 10, 50)?;  // k=10, ef=50
for (id, distance) in results {
    println!("{}: {:.3}", id, 1.0 - distance);  // similarity
}
```

Run the example: `cargo run --example hnsw_benchmark --release`

---

## Contents

1. [HNSW Index](#1-hnsw-index)
2. [Distance Metrics](#2-distance-metrics)
3. [Local Intrinsic Dimensionality](#3-local-intrinsic-dimensionality)
4. [Common Pitfalls](#4-common-pitfalls)
5. [Examples](#5-examples)

---

## 1. HNSW Index

### What it does

HNSW builds a navigable graph for O(log N) approximate nearest neighbor queries.

### Parameters

| Parameter | What it controls | Default | Guidance |
|-----------|-----------------|---------|----------|
| `M` | Edges per node | 16 | Higher = better recall, more memory |
| `ef_construction` | Build-time search width | 200 | Higher = better graph quality, slower build |
| `ef_search` | Query-time search width | 50 | Higher = better recall, slower queries |

### Tuning ef_search

```
ef_search   Recall@10   Latency   Use case
----------------------------------------------
10          ~50%        <100us    Real-time filtering
50          ~85%        ~200us    Production search
100         ~92%        ~300us    High precision
200         ~97%        ~600us    Near-exact
```

### Dual-Branch HNSW (for difficult datasets)

If your dataset has outliers or varying density, standard HNSW may have poor recall on some queries. Dual-Branch HNSW addresses this:

```rust
use vicinity::hnsw::dual_branch::{DualBranchConfig, DualBranchHNSW};

let config = DualBranchConfig {
    m: 16,
    m_high_lid: 24,  // More connections for outliers
    ..Default::default()
};

let mut index = DualBranchHNSW::new(768, config);
index.add_vectors(&data)?;
index.build()?;
```

**When to use**: Recall varies significantly across queries; dataset has known outliers.

**Example**: `cargo run --example dual_branch_demo --release`

---

## 2. Distance Metrics

### Cosine vs L2

| Metric | Formula | When to use |
|--------|---------|-------------|
| Cosine | `1 - dot(a,b)/(‖a‖‖b‖)` | Text embeddings, normalized vectors |
| L2 | `√Σ(aᵢ-bᵢ)²` | Image features, absolute distances |

**Key insight**: For L2-normalized vectors, cosine and L2 are equivalent:
```
L2² = 2(1 - cosine_similarity)
```

### Normalizing vectors

```rust
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / norm).collect()
}
```

---

## 3. Local Intrinsic Dimensionality

LID measures local data complexity. Points in dense clusters have low LID; outliers have high LID.

### MLE Estimator

```
LID(x) = -k / Σ log(dᵢ/dₖ)
```

where d₁ ≤ d₂ ≤ ... ≤ dₖ are k-nearest neighbor distances.

```rust
use vicinity::lid::{estimate_lid, LidConfig};

let config = LidConfig { k: 20, ..Default::default() };
let estimate = estimate_lid(&distances, &config);
println!("LID: {:.1}", estimate.lid);
```

### Outlier Detection

```rust
use vicinity::lid::LidCategory;

// High LID = sparse region = potential outlier
if estimate.category == LidCategory::Sparse {
    println!("Point is in a sparse region");
}
```

**Example**: `cargo run --example lid_outlier_detection --release`

---

## 4. Common Pitfalls

### 1. Forgetting to normalize

**Symptom**: Poor recall despite good parameters.

**Fix**: Always normalize for cosine similarity:
```rust
let normalized = normalize(&embedding);
index.add(id, normalized)?;
```

### 2. Using L2 distance with unnormalized embeddings

**Symptom**: Search results dominated by magnitude, not direction.

**Fix**: Either normalize vectors or use cosine distance.

### 3. ef_search too low

**Symptom**: Low recall, especially on difficult queries.

**Fix**: Start with ef_search=100, measure recall, adjust.

### 4. M too low for high-dimensional data

**Symptom**: Recall degrades as dimension increases.

**Fix**: For dim > 512, use M=24-32 instead of M=16.

### 5. Not measuring recall

**Symptom**: Assuming the index "just works."

**Fix**: Always validate with a holdout set:
```rust
let ground_truth = brute_force_knn(&query, &data, k);
let approx = index.search(&query, k, ef)?;
let recall = intersection(&ground_truth, &approx).len() as f32 / k as f32;
```

### 6. Ignoring outliers

**Symptom**: Some queries have much worse recall than average.

**Fix**: Use LID to identify problematic points; consider Dual-Branch HNSW.

---

## 5. Examples

| Example | What it demonstrates | Run command |
|---------|---------------------|-------------|
| `hnsw_benchmark` | Basic HNSW usage, recall/latency | `cargo run --example hnsw_benchmark --release` |
| `semantic_search_demo` | Full search pipeline | `cargo run --example semantic_search_demo --release` |
| `dual_branch_demo` | LID-aware indexing | `cargo run --example dual_branch_demo --release` |
| `lid_demo` | LID estimation | `cargo run --example lid_demo --release` |
| `lid_outlier_detection` | Finding outliers | `cargo run --example lid_outlier_detection --release` |

---

## Further Reading

- [HNSW paper](https://arxiv.org/abs/1603.09320) (Malkov & Yashunin, 2018)
- [LID estimation](https://projecteuclid.org/journals/annals-of-statistics/volume-33/issue-1/Maximum-likelihood-estimation-of-intrinsic-dimension/10.1214/009053604000000689.full) (Levina & Bickel, 2004)
- [Dual-Branch HNSW](https://arxiv.org/abs/2501.13992) (2025)
