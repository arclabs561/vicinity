# ANN Research Notes (2024-2026)

Practical insights from recent approximate nearest neighbor research.

## Key Papers Reviewed

| Paper | Year | Key Idea | Status |
|-------|------|----------|---------------|
| FreshDiskANN | 2021 | Tombstones + streaming merge for live updates | `hnsw::tombstones` |
| IP-DiskANN | 2025 | In-place updates without rebuild (see paper for reported trade-offs) | `hnsw::inplace` |
| RaBitQ | 2024 | 1-bit/dim quantization with O(1/sqrt(d)) error bound | `ivf_rabitq` |
| HENN | 2025 | Epsilon-net navigation with theoretical guarantees | Research only |
| PEOs | 2024 | Probabilistic routing for graph-based ANNS | Removed (research prototype) |
| CleANN | 2025 | Real-time insertions via workload adaptation | Removed (research prototype) |
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

### 2. Dual-Branch with Skip Bridges (existing: `hnsw/dual_branch.rs`)

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

#### SQ4U (`hnsw::sq4u`)

4-bit scalar quantized graph traversal for HNSW. Precomputes a [d x 16] lookup table
per query, replaces O(d) f32 distance with O(d/2) table lookups during beam search,
then reranks the candidate pool with exact f32 distance.

**Benchmark verdict (2026-04-15)**: SQ4U is ~3x slower than plain HNSW at d=25
(GloVe-25) and ~2x slower at d=960 (GIST-960). The reranking pass dominates --
computing exact f32 distance on the full candidate pool negates the savings from
quantized traversal. The approach would need to avoid reranking (use quantized
distances directly) or reduce the rerank pool significantly to break even.

**Comparison with PAG**: PAG's Probabilistic Routing Test (PRT) uses random
projections at o(d) cost and recycles false positives incrementally (TFB), avoiding
the need for a separate rerank pass. This is architecturally better than SQ4U's
two-stage approach.

**Status**: Experimental. Not recommended as a default algorithm.

#### LEMUR (`lemur`)

Inference-only late-interaction retrieval. Maps multi-vector documents (ColBERT-style)
into single-vector ANN index via mean-pooling (paper uses OLS via SVD). Brute-force
MIPS for now; needs HNSW-backed MIPS and proper OLS for production use.

**Ref**: Kulkarni et al. (2026). arXiv:2501.xxxxx.

### Research Only (Not Yet Implemented)

| Paper | Year | Key Idea | Why It Matters |
|-------|------|----------|----------------|
| CleANN | 2025 | Work-stealing concurrent insert/delete/query with consistency guarantees | Formal concurrent correctness for dynamic graphs |
| DGAI | 2025 | Decoupled graph topology from vector storage | Right architecture for true disk-resident updates |
| SINDI | 2025 | Graph-based index for sparse vectors (SPLADE/BM25) | Bridges dense and sparse retrieval in one structure |
| PathFinder | 2025 | Conjunctive/disjunctive filter predicates in graph search | AND/OR trees of filter predicates |
| QMP | 2024 | Quantization meets projection -- avoids PQ table lookup | Simpler than PQ, competitive accuracy |

### Deep Dives (Research Context)

#### PAG: Projection-Augmented Graph (Mar 2026)

The most significant recent advance. Unifies projection techniques as a first-class
building block of graph construction rather than a plug-in. Targets six demands:
high QPS, fast indexing, low memory, high-dim scalability, retrieval-size robustness,
and online insertion.

**Probabilistic Routing Test (PRT)**: Uses random projections to estimate angles
between vectors in subspaces, avoiding unnecessary exact distance computations during
search. Runs in o(d) time via AVX512-optimized projection lookups.

**Test Feedback Buffer (TFB)**: Recycles false positives from PRT by incrementally
tightening thresholds across search rounds, reusing intermediate computations that
prior methods (PEOs, KS2) would discard.

**Probabilistic Edge Selection (PES)**: A statistical test that detects useful incoming
edges outside the standard out-neighbor set, improving graph connectivity for hard
datasets at O(1) cost per candidate when combined with PRT.

**Results**: Up to 5x faster QPS than HNSW on modern embedding datasets
(OpenAI text-embedding-3-large at 1536/3072d, CLIP, DINOv2), using 20-40% of HNSW's
indexing time.

**Relevance to vicinity**: vicinity's FINGER module is conceptually adjacent (projection-
based search pruning on top of HNSW), but PAG integrates projections into construction
itself. A PAG-style retrofit onto the existing Vamana graph builder would be the
cleanest path. SQ4U's benchmark results confirm that two-stage quantized-traverse +
exact-rerank is the wrong architecture; PAG's incremental approach is better.

**Ref**: Ma et al. (2026). arXiv:2603.06660.

#### HNSW++ (Dual-Branch + LID + Skip Bridges)

Three structural changes to standard HNSW:

1. **LID-driven insertion**: Computes each node's Local Intrinsic Dimensionality via
   MLE over k-nearest neighbors. High-LID nodes (locally complex regions) are
   prioritized for upper layers, ensuring "hard" regions have better routing coverage.

2. **Dual-branch structure**: Two independent branches process nodes separately but
   converge at the base layer. Doubles the probability of finding a good entry region.

3. **Skip bridges**: If node e at layer l > 0 has LID(e) > T and d(e, q) < eps,
   search bypasses intermediate layers and jumps to layer 0. Reduces per-layer
   traversal by 20-50% with < 2% recall drop.

**Results**: Across 12 CV/NLP/recommendation datasets, LID-ordered insertion is the
single most impactful improvement. Skip bridges provide the largest latency reduction.

**Status in vicinity**: `hnsw::dual_branch` exists. Skip bridges not yet implemented.

**Ref**: Zhang et al. (2025). arXiv:2501.13992.

#### Wolverine: Deletion via MSP Repair (VLDB 2025)

First rigorous treatment of deletion as a monotonic-search-path repair problem.

**Key insight**: Naive "connect all in-neighbors to all out-neighbors" (DwFC) often
*worsens* recall because the new edges are long, violate the short-edge-priority rule,
and cause EdgeTrim to remove existing short edges. This creates a feedback loop of
degrading connectivity.

**Wolverine's approach**: For each out-neighbor of the deleted node, run kANN search
to find quality repair candidates satisfying: (1) close to the out-neighbor,
(2) far from the deleted node, (3) monotonic to many disrupted path endpoints.
The geometric locus is a crescent-shaped region -- intersection of two sphere
complements.

**Results**: Up to 11x better throughput than FreshDiskANN. Repair cost is
O(theta_d^2) per deletion (2-hop neighborhood), not O(n).

**Relevance to vicinity**: vicinity's tombstone-based deletion (`hnsw::tombstones`)
avoids graph repair entirely. Wolverine shows when real repair is needed
(high deletion rates where tombstone overhead degrades search).

**Ref**: Zheng et al. (2025). VLDB 2025.

#### MARGO: Graph Layout Optimization (VLDB 2025)

For disk-based indices, random I/O per graph hop (~100us) dominates. MARGO proves
optimal graph layout is NP-hard and provides a greedy approximation.

**Edge weight**: w(p, p*) = monotonic_reachability(p, p*) * reachability_to(p).
Captures the intuition that edges both easy to reach and leading to many destinations
are most frequently traversed.

**Two-stage**: Cluster the dataset, run greedy layout in parallel per cluster
(intra-cluster), then handle boundary vertices (inter-cluster).

**Results**: 5.5x faster layout optimization, 26.6% better search efficiency at
identical recall.

**Relevance to vicinity**: Relevant when vicinity adds disk-resident index support.
The edge weight formula directly uses SNG occlusion counts.

**Ref**: Zheng et al. (2025). VLDB 2025.

#### Extended-RaBitQ (2-7 bit)

Generalizes RaBitQ from 1-bit to arbitrary compression rates. At same rate,
dominates classical scalar quantization:

| Bits/dim | Recall (no rerank) | Compression |
|----------|--------------------|-------------|
| 1 | ~80% | 32x |
| 4 | ~90% | 8x |
| 5 | ~95% | 6.4x |
| 7 | ~99% | 4.6x |

**Relevance to vicinity**: Current `ivf_rabitq` is 1-bit. Extended-RaBitQ at 4-bit
could be a better graph traversal distance than SQ4U's scalar 4-bit (uses measure
concentration rather than per-dimension min/max scaling).

**Ref**: Chen et al. (2026). github.com/VectorDB-NTU/Extended-RaBitQ.

#### PCNN: Polar Code Nearest Neighbor (Amazon, 2025)

Reframes multiprobe LSH as an error-correcting decoding problem. Hashes into a
high-dimensional binary space (preserving distance fidelity), structures the bucket
space as a polar code, and uses list decoding for multiprobe queries.

**Key result**: A single hash table outperforms classical multiprobe LSH with
multiple tables. Eliminates LSH's historical disadvantage (needing many tables).

**Theoretical significance**: Opens a rich design space connecting coding theory
(LDPC, Reed-Muller, turbo codes) to similarity search. Suggests a "capacity theorem
for similarity search" may be possible.

**Ref**: Amazon Science (2025).

#### Quake: Adaptive Indexing (OSDI 2025)

Addresses the blind spot that all existing indices assume static, uniformly-accessed
data. Real workloads have skew, drift, and continuous insertions.

- Multi-level partitioning with dynamic split/merge based on cost model
- Recall estimation model that dynamically sets nprobe/beam width
- NUMA-aware intra-query parallelism

**Results**: On Wikipedia-12M (growing 1M to 12M): 28x lower search latency (parallel),
4.5-126x lower update latency vs HNSW/DiskANN/SVS/ScaNN.

**Ref**: Mohoney et al. (2025). OSDI 2025.

## Theoretical Foundations

### Monotonic Search Paths (MSPs)

The universal primitive underlying all graph-based ANN. A path P(v1, ..., vl) is
monotonic to query q iff d(vi, q) > d(vi+1, q) for all i. A graph is a Monotonic
Search Network (MSNET) iff every pair of nodes has at least one MSP.

Construction quality = how many MSPs exist (SNG occlusion count).
Search efficiency = MSP lengths (O(log n) if navigable).
Deletion difficulty = MSPs destroyed (Wolverine).
Disk layout quality = MSP edges per page (MARGO).
Quantization quality = whether approximate distances preserve MSP traversal.

### The SNG Occlusion Rule

For node p with candidate neighbor p', p' is occluded by already-selected neighbor
p* if d(p', p*) < d(p', p). Geometrically: p' is on the same side of the
perpendicular bisector as p*. This ensures edges radiate in diverse angular directions.

**Lemma (occlusion <-> monotonic reachability)**: In an SNG, a monotonic path
[p, p*, ..., p'] exists iff edge (p, p*) occludes edge (p, p'). The number of
vertices monotonically reachable from edge (p, p*) equals the edges it occludes + 1.

### Martingale Analysis of Graph Construction (Ma et al., 2026)

SNG pruning modeled as a submartingale. Each accepted neighbor occludes a geometrically
determined fraction of remaining candidates.

- **Fast phase** (> n^(2/3) candidates): constant fraction eliminated per step.
  Azuma's inequality gives concentration.
- **Plateau phase** (<= n^(2/3)): Wormald's Differential Equation Method converts
  discrete process to continuous ODE, proving convergence to O(ln n) degree.
- **Maximum out-degree**: O(n^(2/3+eps)) -- the transition point between phases.

### Capacity-Law Failure in HNSW (Jan 2026)

HNSW doesn't degrade gracefully -- it fails abruptly. At approximately
k ~ 2-3.5 * efSearch, search undergoes discontinuous breakdown: neighbor distances
explode, geometric structure collapses. Below the threshold, results are
geometrically meaningless, not merely "slightly worse."

**Implication**: For k neighbors, efSearch >= 2k is mandatory. This is a hard
boundary, not a tuning recommendation.

### Concentration of Measure

The mathematical engine behind RaBitQ, JL projections, and LSH bounds.

**Levy's isoperimetric inequality**: For A in S^(n-1) with sigma(A) >= 1/2,
sigma(A_eps) >= 1 - 2*exp(-n*eps^2/2). In 1000 dimensions, ~100% of the sphere
lies within O(1/sqrt(n)) of any half-measure set.

**Consequence for quantization**: Rounding each coordinate to +/-1/sqrt(D) gives
O(1/sqrt(D)) total error -- matching the Alon-Klartag information-theoretic lower
bound. Higher dimensions need fewer bits per dimension for the same accuracy.

### The Coding Theory Connection

PCNN shows ANN and channel decoding are structurally parallel: both solve "find the
nearest codeword" problems. The key open question: is there a "capacity theorem for
similarity search"? Given n points in d dimensions with intrinsic dimension d*, what
is the minimum bits-per-point of index space needed for (1+eps)-approximation in
O(log n) time?

### Circuit Complexity Barrier (ITCS 2025)

Proving superpolynomial lower bounds on k-NN representation complexity would require
a breakthrough in circuit complexity (P vs NP-adjacent). This means we may be
fundamentally unable to prove current methods are near-optimal -- but also that
dramatic algorithmic improvements may exist undiscovered.

### The Convergence

The field is collapsing the traditional taxonomy. The best modern systems compose
primitives from multiple families:
- PAG = graph + LSH-family projections + incremental threshold refinement
- IVF_RABITQ = IVF partitioning + SimHash-family quantization + SIMD popcount
- DiskANN = Vamana graph + PQ quantization + SSD-aware beam search
- Quake = adaptive IVF + cost-model maintenance + NUMA-parallel search
- HNSW++ = multi-layer graph + LID-based insertion + skip bridges

The "which family is best?" question is the wrong one. The answer is "all of them,
composed correctly."

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
