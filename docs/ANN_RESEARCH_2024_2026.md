# ANN Research Notes (2024-2026)

Practical insights from recent approximate nearest neighbor research.

## Key Papers Reviewed

| Paper | Year | Key Idea | Status |
|-------|------|----------|---------------|
| FreshDiskANN | 2021 | Tombstones + streaming merge for live updates | `hnsw::tombstones` |
| IP-DiskANN | 2025 | In-place updates without rebuild (see paper for reported trade-offs) | `hnsw::inplace` |
| RaBitQ | 2024 | 1-bit/dim quantization with O(1/sqrt(d)) error bound | `ivf_rabitq` |
| HENN | 2025 | Epsilon-net navigation with theoretical guarantees | Research only |
| PEOs | 2024 | Probabilistic routing for graph-based ANNS | `hnsw::probabilistic_routing` |
| CleANN | 2025 | Real-time insertions via workload adaptation | `cleann` |
| DGAI | 2025 | Decoupled on-disk graph index for updates | Research only |
| Dual-Branch HNSW | 2025 | LID-based insertion with skip bridges | `hnsw::dual_branch` |
| DEG | 2025 | Dynamic edge navigation for bimodal data | `hnsw::deg` |
| delta-EMG | 2025 | Error-bounded monotonic graph with occlusion pruning | `emg` |
| ESG | 2025 | Elastic subgraphs for range-filtered ANN | `esg` |
| PiPNN | 2026 | Partition + GEMM + HashPrune parallel construction | `pipnn` |
| Curator | 2026 | K-means tree for low-selectivity filtered search | `curator` |
| NSG | 2019 | Navigating Spreading-out Graph with MRNG pruning | `nsg` |
| LSM-VEC | 2025 | LSM-tree tiered streaming vector index | `streaming::lsm` |

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

### Query-Aware Entry Points (GATE)

Extract hub nodes and learn query-specific entry point selection.

**When useful**: Multi-modal queries, cross-domain search.

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

## 2025-2026 Algorithms

### Implemented

#### IVF-RaBitQ (`ivf_rabitq`)

IVF partitioning with RaBitQ quantization instead of Product Quantization. No codebook
training needed -- uses random orthogonal rotation + sign/extended bits + per-vector
correction factors (f_add, f_rescale).

**Key difference from IVF-PQ**: RaBitQ preserves per-dimension information while PQ
compresses via subspace codebooks. At 4 bits/dim, RaBitQ uses more memory than PQ (M=8)
but achieves higher recall. At 1 bit/dim, RaBitQ is competitive on memory and faster
(popcount arithmetic).

**Distance estimation**: Asymmetric (exact query, quantized database). The RaBitQ error
bound is $O(1/\sqrt{d})$ where d is dimension -- tighter than PQ's empirical bounds.

**Ref**: Gao et al. (2024), SIGMOD. Chen et al. (2026), arXiv:2602.23999.

#### Selectivity-Aware Filtered Search (`hnsw::filtered::selectivity_search`)

Three-regime filter strategy based on estimated selectivity (match ratio):

| Regime | Selectivity | Strategy |
|--------|-------------|----------|
| High | >10% | Standard ACORN 2-hop |
| Medium | 1-10% | ACORN with 4x ef_search, aggressive 2-hop |
| Low | <1% | Brute-force scan over pre-filtered matching IDs |

Selectivity is estimated by BFS-probing ~50 nodes from the entry point. The low-
selectivity regime requires the caller to pre-compute matching IDs (e.g., from an
inverted index on metadata). Without pre-filtered IDs, falls back to aggressive ACORN.

**Ref**: Curator (SIGMOD 2026, arXiv:2601.01291) for low-selectivity analysis. ESG
(arXiv:2504.04018) for range-filter regime analysis. FANNS survey (arXiv:2602.11443)
for the empirical finding that selectivity regime matters more than algorithm choice.

#### LSM-Tiered Streaming (`streaming::lsm::LsmIndex`)

LSM-tree architecture for write-heavy vector workloads:

```
L0 (buffer, brute-force) ──compaction──► L1 (HNSW) ──cascade──► L2 (HNSW) ──► ...
```

- **Write amplification**: O(log_T(N/B)) rewrites per vector, where T is size ratio
  (default 10), N total vectors, B buffer size. E.g., T=10, N=100M, B=10K → 4 rewrites.
- **Search**: Query each level independently, merge results. Cost: O(L * search_per_level).
- **Compaction trigger**: Size-tiered -- level i compacts into i+1 when
  `size_i >= T * size_{i+1}`.
- **Tombstones**: Global set filtered from all search results. Garbage-collected during
  compaction (excluded from merged graph). Cleared on `force_merge_all`.

**Ref**: Inspired by LSM-VEC (arXiv:2505.17152). O'Neil et al. (1996), "The Log-Structured
Merge-Tree."

#### delta-EMG (`emg`)

Single-layer graph with per-query $(1/\delta)$-approximation guarantee via occlusion-based
pruning. Adaptive per-edge delta for automatic degree balancing. Expected O(ln n) degree.

**Ref**: Yin et al. (2025), arXiv:2511.16921.

#### ESG (`esg`)

Range-filtered ANN via half-bounded interval decomposition. Any range [l,r] needs at most
2 HNSW subgraph searches (vs O(log N) in prior work). Elastic relaxation allows traversing
out-of-range nodes for navigation while excluding them from results.

**Ref**: Yang et al. (2025), arXiv:2504.04018.

#### PiPNN (`pipnn`)

Partition-based graph construction: Randomized Ball Carving → all-pairs distance within
cache-friendly leaves → HashPrune (SimHash reservoir) for directionally-diverse edge pruning.
History-independent: same result regardless of insertion order, enabling parallel construction.

**Ref**: Rubel et al. (2026), arXiv:2602.21247.

#### Curator (`curator`)

K-means tree with per-label sorted-ID buffers and Bloom filters for low-selectivity filtered
search (<5% match rate). Finds matching vectors by spatial containment, not graph traversal.
No vector duplication; ~4% memory overhead. Complement to graph indexes.

**Ref**: Jin et al. (2026), SIGMOD 2026. arXiv:2601.01291.

### Research Only (Not Yet Implemented)

| Paper | Year | Key Idea | Why It Matters |
|-------|------|----------|----------------|
| LEMUR | 2026 | Maps multi-vector documents (ColBERT) into single-vector ANN index | MaxSim-compatible ANN for late-interaction retrieval |
| CleANN | 2025 | Work-stealing concurrent insert/delete/query with consistency guarantees | Formal concurrent correctness for dynamic graphs |
| PAG | 2026 | Random projections on graph edges for better long-range routing | Simple retrofit onto existing Vamana graphs |
| DGAI | 2025 | Decoupled graph topology from vector storage | Right architecture for true disk-resident updates |
| SINDI | 2025 | Graph-based index for sparse vectors (SPLADE/BM25) | Bridges dense and sparse retrieval in one structure |
| PathFinder | 2025 | Conjunctive/disjunctive filter predicates in graph search | AND/OR trees of filter predicates |
| QMP | 2024 | Quantization meets projection -- avoids PQ table lookup | Simpler than PQ, competitive accuracy |

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
