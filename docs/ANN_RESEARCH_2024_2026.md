# ANN Research Notes (2024-2026)

Practical insights from recent approximate nearest neighbor research.

## Key Papers Reviewed

| Paper | Year | Key Idea | Status |
|-------|------|----------|---------------|
| FreshDiskANN | 2021 | Tombstones + streaming merge for live updates | `hnsw::tombstones` |
| IP-DiskANN | 2025 | In-place updates without rebuild (see paper for reported trade-offs) | `hnsw::inplace` |
| RaBitQ | 2024 | 1-bit/dim quantization with O(1/sqrt(d)) error bound | Planned |
| HENN | 2025 | Epsilon-net navigation with theoretical guarantees | Research only |
| PEOs | 2024 | Probabilistic routing for graph-based ANNS | `hnsw::probabilistic_routing` |
| CleANN | 2025 | Real-time insertions via workload adaptation | Research only |
| DGAI | 2025 | Decoupled on-disk graph index for updates | Research only |
| Dual-Branch HNSW | 2025 | LID-based insertion with skip bridges | `hnsw::dual_branch` |
| DEG | 2025 | Dynamic edge navigation for bimodal data | `hnsw::deg` |

## Practical Ideas Implemented

### 1. Tombstone-Based Deletions (`hnsw/tombstones.rs`)

From FreshDiskANN: soft deletion via tombstones rather than immediate graph repair.

**When to use**: Deletion latency matters more than search overhead.

**Trade-off**: O(1) deletion cost, slight search overhead from filtering.

```rust
use vicinity::hnsw::TombstoneSet;

let mut tombstones = TombstoneSet::new(0.1); // 10% threshold
tombstones.delete(doc_id);

// Filter search results
let filtered = tombstones.filter_results(raw_results.into_iter());

// Check if compaction needed
if tombstones.should_compact(total_nodes) {
    // Trigger background rebuild
}
```

### 2. Probabilistic Edge Routing (existing: `hnsw/probabilistic_routing.rs`)

From Lu et al. (2024): probabilistically test edges to reduce wasted distance computations.

**Reported benefit**: higher throughput with minimal recall loss (exact numbers depend on dataset/parameters; see paper).

### 3. Dual-Branch with Skip Bridges (existing: `hnsw/dual_branch.rs`)

From arXiv 2501.13992: LID-based insertion strategy with skip bridges.

**Reported benefit**: improved recall on challenging/clustered regimes at similar latency (see arXiv:2501.13992 for measured deltas).

## Ideas for Future Implementation

### RaBitQ Quantization

1-bit per dimension with random rotation preprocessing. This is a 32× *storage* reduction versus `f32` vectors (1 bit vs 32 bits per component); error bounds and recall behavior are paper-dependent.

**Implementation sketch**:
```rust
// 1. Random orthogonal rotation (once per index)
let rotation = random_orthogonal_matrix(dim, seed);

// 2. Binary quantization
fn quantize(v: &[f32], rotation: &[Vec<f32>]) -> BitVec {
    let rotated = rotate(v, rotation);
    rotated.iter().map(|&x| x > 0.0).collect()
}

// 3. Distance via popcount
fn hamming_approx(a: &BitVec, b: &BitVec) -> u32 {
    (a ^ b).count_ones()
}
```

**Trade-off**: Need to store correction factors for accurate rescoring.

### Query-Aware Entry Points (GATE)

Extract hub nodes and learn query-specific entry point selection.

**When useful**: Multi-modal queries, cross-domain search.

### LSM-Style Tiering

From LSM-VEC: Recent updates in memory, periodic merge to disk graph.

**Architecture**:
```
[Memory Layer] - small dynamic graph, recent updates
     |
     v (periodic merge)
[Disk Layer 0] - recent merged data
     |
     v (compaction)
[Disk Layer 1] - older data
```

## Metrics to Track

| Metric | Target | Notes |
|--------|--------|-------|
| Recall@10 | (pick) | Choose per product need; report recall/latency curves |
| QPS | (measure) | Hardware- and dataset-dependent |
| Build time | (measure) | Report alongside recall/latency and memory |
| Memory | (measure) | Separate vector store vs graph/index overhead |
| Deletion latency | (measure) | Tombstones trade deletion cost vs search filtering/compaction |

## Integration Example

```rust
use vicinity::hnsw::{HNSWIndex, TombstoneSet};

// Build index
let mut index = HNSWIndex::new(128, 16, 200)?;
for (id, vec) in documents {
    index.add(id, vec)?;
}
index.build()?;

// Streaming deletions via tombstones
let mut tombstones = TombstoneSet::new(0.1);
tombstones.delete(stale_doc_id as usize);

// Search with tombstone filtering
let raw_results = index.search(&query, k * 2, ef)?; // Over-fetch
let filtered: Vec<_> = raw_results
    .into_iter()
    .filter(|(id, _)| !tombstones.is_deleted(*id as usize))
    .take(k)
    .collect();

// Periodic compaction
if tombstones.should_compact(index.len()) {
    // Rebuild index, excluding tombstoned nodes
}
```

## References

1. Singh et al. (2021). "FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search." `https://arxiv.org/abs/2105.09613`

2. Xu et al. (2025). "IP-DiskANN: In-Place Graph Index Updates for Streaming ANN." `https://arxiv.org/abs/2502.13826`

3. Gao et al. (2024). "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search." (SIGMOD 2024) `https://dl.acm.org/doi/10.1145/3626246.3653391`

4. Dehghankar & Asudeh (2025). "HENN: A Hierarchical Epsilon Net Navigation Graph for Approximate Nearest Neighbor Search." `https://arxiv.org/abs/2505.17368`

5. Lu, Xiao, Ishikawa (2024). "Probabilistic Routing for Graph-Based Approximate Nearest Neighbor Search." `https://arxiv.org/abs/2402.11354`

6. Xiao et al. (2024). "Enhancing HNSW Index for Real-Time Updates: Addressing Unreachable Points and Performance Degradation." `https://arxiv.org/abs/2407.07871`
