# ANN Algorithmic Landscape

How approximate nearest neighbor search works, why it works, and where
the field is heading. This document is canonical: updated as the
landscape evolves, not time-stamped.

For the bibliography, see [references.md](references.md).
For benchmark numbers, see [benchmark-results.md](benchmark-results.md).

---

## Principles

### The fundamental tradeoff

Exact nearest neighbor search in high dimensions is intractable. The
Delaunay graph (the perfect proximity structure) degenerates to a
complete graph in high dimensions; degree grows as $2^{\Theta(d)}$.
Every ANN method is an approximation strategy for avoiding this while
preserving enough navigational structure for greedy search to converge.

### Monotonic search paths

The universal primitive. A path $P(v_1, \ldots, v_l)$ is monotonic to
query $q$ iff $d(v_i, q) > d(v_{i+1}, q)$ for all $i$. A graph is a
Monotonic Search Network (MSNET) iff every pair of nodes has at least one
MSP connecting them. This guarantees greedy search converges without
backtracking.

Every design decision in graph-based ANN traces back to MSPs:

| Concern | MSP connection |
|---------|---------------|
| Construction quality | How many MSPs exist (SNG occlusion count) |
| Search efficiency | MSP lengths: $O(\log n)$ if the graph is navigable |
| Deletion difficulty | How many MSPs are destroyed (Wolverine) |
| Disk layout quality | How many MSP edges share a page (MARGO) |
| Quantization quality | Whether approximate distances preserve MSP traversal |

### The SNG occlusion rule

The construction rule underlying HNSW, NSG, DiskANN/Vamana, and EMG.
For node $p$ with candidate neighbor $p'$, $p'$ is **occluded** by
already-selected neighbor $p^*$ if $d(p', p^*) < d(p', p)$.
Geometrically: $p'$ lies on the same side of the perpendicular bisector
between $p$ and $p^*$ as $p^*$ does. Ensures edges radiate in diverse
angular directions rather than clustering.

**Key lemma**: A monotonic path $[p, p^*, \ldots, p']$ exists iff edge
$(p, p^*)$ occludes edge $(p, p')$. The number of vertices monotonically
reachable from $(p, p^*)$ equals the edges it occludes + 1. Construction
rule = search guarantee.

### Concentration of measure

The mathematical engine behind RaBitQ, JL projections, and LSH bounds.

**Levy's isoperimetric inequality**: For $A \subseteq S^{n-1}$ with
$\sigma(A) \geq 1/2$:

$$\sigma(A_\varepsilon) \geq 1 - 2e^{-n\varepsilon^2/2}$$

In 1000 dimensions, ~100% of the sphere lies within $O(1/\sqrt{n})$ of
any half-measure set. This is why a random unit vector's coordinates
concentrate around $\pm 1/\sqrt{D}$, and why rounding to those values
loses surprisingly little information.

**Consequence for quantization**: $O(1/\sqrt{D})$ total error from 1-bit
quantization, matching the Alon-Klartag information-theoretic lower
bound. Higher dimensions need *fewer* bits per dimension for the same
accuracy.

**The "free information" principle**: Concentration provides geometric
constraints that substitute for stored bits. RaBitQ's unbiased inner
product estimator exploits this: the cross-term $\langle \bar{o}, e_1
\rangle$ concentrates around 0, so setting it to zero yields an
estimator matching the lower bound without computing it.

### The Johnson-Lindenstrauss lemma

For any $\varepsilon \in (0,1)$ and $n$ points in $\mathbb{R}^d$, there
exists a map $f: \mathbb{R}^d \to \mathbb{R}^k$ with
$k = O(\varepsilon^{-2} \ln n)$ preserving all pairwise distances within
$(1 \pm \varepsilon)$. The target dimension depends only on $\ln n$ and
$\varepsilon$, not on $d$. This is why random projections work for
modern high-dimensional embeddings.

On manifolds with bounded curvature and reach, the target dimension
improves to $k = O(d^* \ln(n\tau\kappa/\varepsilon) / \varepsilon^2)$
where $d^*$ is intrinsic dimension, much smaller than $\ln n$ when
the data has low-dimensional structure.

### Capacity-law failure

HNSW doesn't degrade gracefully. It fails abruptly. At approximately
$k \approx 2\text{-}3.5 \times \text{efSearch}$, search undergoes
discontinuous breakdown: neighbor distances explode, geometric structure
collapses completely. Below the threshold, results are geometrically
meaningless, not merely "slightly worse."

**Implication**: $\text{efSearch} \geq 2k$ is mandatory, not a tuning
recommendation. This is a hard boundary.

---

## Algorithm Families

### Graph-based (HNSW, Vamana/DiskANN, NSG, EMG)

Proximity graph traversal via greedy search. Best QPS-recall tradeoff in
most regimes. Slow construction, high memory.

**HNSW**: Multi-layer skip-list-inspired graph. Layer assignment via
exponential distribution $P(\ell) \propto e^{-\ell/m_L}$ where
$m_L = 1/\ln(M)$. ~63% of nodes on layer 0 only, ~23% reach layer 1.
The dominant paradigm.

**Vamana/DiskANN**: Two-pass alpha-pruning for angular diversity. First
pass with tight $\alpha$ for local edges, second with relaxed $\alpha$
for long-range shortcuts. SSD layout: PQ codes in RAM for cheap distance
estimates, full vectors + graph on SSD.

**NSG**: MRNG pruning (monotonic relative neighborhood). $O(n)$
ensure-connectivity pass limits scalability.

**delta-EMG**: Single-layer graph with per-query $(1/\delta)$-approximation
guarantee via occlusion pruning. The $\delta$ parameter continuously
interpolates between strong navigability (small $\delta$, dense) and
efficiency (large $\delta$, sparse). Expected $O(\ln n)$ degree.

### Quantization-based (PQ, RaBitQ, ScaNN)

Compress vectors, rank by approximate distance. Fast indexing, low
memory. Generally lower QPS-recall than graphs alone.

**Product Quantization**: Decompose $\mathbb{R}^d$ into $M$ subspaces,
train k-means in each, represent vectors as $M$ centroid IDs. Quality
depends on space decomposition. OPQ jointly optimizes rotation +
codebooks; random rotation performs surprisingly well (concentration
balances variance across subspaces).

**RaBitQ**: 1-bit/dim via random rotation + hypercube projection.
32x compression. Error $O(1/\sqrt{D})$, provably optimal (matches
Alon-Klartag). Distance reduces to binary dot product + popcount
(single-cycle on modern CPUs via AVX512-VPOPCNTDQ or SVE).

**Extended-RaBitQ**: Generalizes to 2-7 bits/dim. 4-bit: ~90% recall
without reranking. 7-bit: ~99%. Dominates scalar quantization at every
compression rate.

**ScaNN/SOAR**: Anisotropic VQ for inner product. SOAR adds
Orthogonality-Amplified Residuals to reduce correlated search failures.

### Projection / LSH

Hash similar points to same buckets. Theoretical guarantees, simplicity.
Less competitive than graphs on modern data, but increasingly used as
a *primitive within* other methods.

**Cross-polytope LSH**: Theoretically optimal for angular distance.
Hash via random rotation (Fast Hadamard Transform, $O(d \ln d)$) then
select coordinate with maximum absolute value. Combined with multiprobe,
dominates hyperplane LSH.

**LSH sensitivity bounds**: Log-convexity of noise stability (Fourier
analysis on the Boolean hypercube) gives tight $\rho \geq 1/c$ for
data-independent LSH. Data-dependent LSH achieves $\rho \leq 1/(2c-1)$;
the gap quantifies the value of adaptation.

**PCNN**: Reframes multiprobe LSH as polar code decoding. Single hash
table outperforms multiple-table classical LSH. Opens a design space
connecting coding theory to similarity search.

**Modern role**: LSH is less a standalone index and more a building
block. PAG uses projection-based routing (SimHash-family). RaBitQ's
hypercube projection is structurally identical to SimHash.

### Hybrid / quantized graph

The convergence direction. Embed quantization or projection directly
into graph construction.

**SymphonyQG**: RaBitQ-style distance during graph traversal. Requires
$O(d^2)$ query rotation but avoids exact f32 distance for routing.

**SQ4U**: 4-bit scalar quantized traversal + exact rerank. Benchmarks
show ~2-3x *slower* than plain HNSW at all tested dimensions (25, 960).
Reranking negates the quantized traversal savings.

**PAG**: Projection-Augmented Graph. Probabilistic Routing Test (PRT)
uses random projections at $o(d)$ cost to filter candidates. Test
Feedback Buffer (TFB) recycles false positives incrementally.
Probabilistic Edge Selection (PES) detects useful incoming edges at
$O(1)$ cost. Up to 5x faster than HNSW on 1536-3072d embeddings.

### Filtered / dynamic

**ACORN**: Filtered HNSW via subgraph sampling (SIGMOD 2024). Selectivity
regime matters more than algorithm choice.

**RACORN-1**: ACORN-compatible low-selectivity fallback (arXiv 2026).
Repurposes filter-failing nodes as transient bridges when ACORN expansion
stalls, then switches to exact filtered scan at extreme low selectivity.
Treat this as an extension point after ACORN, not a replacement baseline.

**Curator**: K-means tree with per-label Bloom filters for low-selectivity
(<5%) filtered search. Complement to graph indexes.

**Quake**: Adaptive indexing for dynamic workloads. Multi-level
partitioning with cost-model-driven split/merge. NUMA-aware parallelism.
28x lower search latency on growing datasets.

**Wolverine**: Deletion via MSP repair. Key insight: naive "connect all"
repair *worsens* recall by introducing long edges that cause EdgeTrim to
remove short ones. Quality repair candidates must be close to the
out-neighbor, far from the deleted node, and monotonic to many endpoints.
Geometric locus: crescent-shaped region. 11x better than FreshDiskANN.

---

## Mathematical Foundations

### Martingale analysis of graph construction

Ma et al. (2026) model SNG pruning as a submartingale. Each accepted
neighbor occludes a fraction of remaining candidates.

- **Fast phase** ($> n^{2/3}$ candidates): Constant fraction eliminated
  per step. Azuma's inequality gives concentration around the expected
  trajectory.
- **Plateau phase** ($\leq n^{2/3}$): Wormald's Differential Equation
  Method converts discrete process to continuous ODE
  $z'(x) = -\alpha z(x)$, solution $z(x) = e^{-\alpha x}$.
- **Degree bound**: $O(\ln n)$ expected, $O(n^{2/3+\varepsilon})$ maximum.

The Wormald framework requires bounded one-step changes
$|Y_i(t+1) - Y_i(t)| \leq C$. The $O(n^{2/3})$ maximum degree is the
point where this condition breaks.

### The occlusion-navigability duality

The delta-EMG occlusion region for nodes $u, v$ with parameter $\delta$:

$$\text{Occ}_\delta(u, v) = \{x : d(x,u) < d(u,v) \text{ and } d^2(x,v) + 2\delta \cdot d(u,v) \cdot d(x,u) < d^2(u,v)\}$$

As $\delta \to 0$, the region expands (strong navigability, dense graph).
As $\delta \to 1$, it contracts (efficiency, sparse graph).

**Key lemma**: For $w \in \text{Occ}_\delta(u,v)$ and query $q$
satisfying $d(q,v) < \delta \cdot d(q,u)$: $d(q,w) < d(q,u)$.
Proof uses only triangle inequality and inner product geometry; no
distributional assumptions.

### Voronoi collapse in high dimensions

The Voronoi diagram of $n$ points in $\mathbb{R}^d$ has complexity
$\Theta(n^{\lceil d/2 \rceil})$, exponential in dimension. The
Delaunay graph (dual) degenerates to a complete graph for uniform points.
This is the fundamental reason the entire ANN field exists.

### LSH lower bounds via Fourier analysis

For hash function $h$ on $\{0,1\}^d$, the noise stability
$K_H(t) = \sum_S \mathbb{E}[\|\hat{h}(S)\|^2] e^{-t|S|}$
is log-convex in $t$ (each term $e^{-t|S|}$ is log-convex; nonneg
combination preserves log-convexity). Combined with $K_H(0) = 1$:

$$\rho = \frac{\ln(1/p)}{\ln(1/q)} \geq \frac{1}{c}$$

Tight for data-independent LSH. Data-dependent achieves $1/(2c-1)$.

### Complexity barriers

ITCS 2025: proving superpolynomial lower bounds on k-NN representation
complexity would require a circuit complexity breakthrough (P vs NP
-adjacent). We may be unable to prove current methods are near-optimal,
but this also means undiscovered improvements may exist.

---

## The Convergence

The traditional taxonomy (graph vs LSH vs quantization vs tree) is
dissolving. The best systems compose primitives from multiple families:

- **PAG** = graph + LSH-family projections + incremental thresholds
- **IVF-RaBitQ** = IVF partitioning + SimHash-family quantization + popcount
- **DiskANN** = Vamana graph + PQ quantization + SSD-aware beam search
- **Quake** = adaptive IVF + cost-model maintenance + NUMA parallelism
- **HNSW++** = multi-layer graph + LID-based insertion + skip bridges

The bottleneck is shifting from raw QPS on static benchmarks (increasingly
solved) to dynamic workloads, filtered/hybrid retrieval, construction
speed, and maintaining geometric fidelity under the capacity-law failure
mode.

Hardware co-design is the next multiplier: MLX unified memory + Neural
Accelerators, AVX512-VPOPCNTDQ for RaBitQ, GPU graph construction
(CAGRA), near-memory computing (ANSMET).

The deepest open question: can we prove that a graph built with SNG
occlusion, using RaBitQ-quantized distances ($O(1/\sqrt{D})$ error),
maintains $O(\ln n)$ navigability? Rigorous composition of construction
guarantees with quantization error bounds would unify the theory.
