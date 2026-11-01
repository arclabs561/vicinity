# Research frontier: what we do not implement, and which surveys set the frame

Status: 2026-11-01. Compiled from primary sources; every claim carries a URL.
Marked **[unverified]** where the primary page could not be read and the claim
rests on a secondary source. This page is the counterpart to the
[atlas](benchmark-atlas.md): the atlas says what *these* families do at a given
size, and this page says what the field does that these families cannot.

## Surveys worth being measured against

| Work | Year | Perspective it adds |
| --- | --- | --- |
| [ANN-Benchmarks](https://arxiv.org/abs/1807.05614) | 2018 | The measurement contract itself: distance-threshold recall, per-implementation rather than per-algorithm comparison, automatic sweeps. |
| [Graph ANNS survey and comparison](https://arxiv.org/abs/2101.12631) | 2021 | Decomposes a graph index into candidate acquisition, neighbour selection, seeding and routing, so a missing component is visible instead of hidden inside a whole-system score. |
| [Graph-based vector search: experimental evaluation](https://arxiv.org/abs/2502.05575) | 2025 | Re-runs the 2021 comparison up to 1B points, i.e. tests whether its conclusions survive scale. |
| [Filtered-DiskANN](https://doi.org/10.1145/3543507.3583552) | 2023 | Makes *selectivity* the governing variable for filtered search, and shows unfiltered graphs collapsing at 10⁻⁴ to 10⁻⁶. |
| [Survey of filtered ANNS](https://arxiv.org/abs/2505.06501) | 2025 | The first filtered taxonomy: formal problem statement, difficulty analysis by data/query relationship, evaluation metrics. |
| [Hardness-controlled filtered benchmark](https://arxiv.org/abs/2606.14193) | 2026 | Shows filtered benchmarks are confounded by uncontrolled predicate hardness, and supplies a generator that varies it deliberately. |
| [RaBitQ](https://arxiv.org/abs/2405.12497) | 2024 | Reframes compression as estimation theory: PQ-family codes have no error bound, D-bit codes do. |
| [VIBE](https://arxiv.org/abs/2505.17810) | 2025 | Modern embedding workloads, out-of-distribution queries, and per-query difficulty breakdowns, which aggregate curves hide. |
| [Big ANN: NeurIPS'23 results](https://arxiv.org/abs/2409.17424) | 2024 | Competition-grade definitions for the four hard variants: filtered, OOD, sparse, streaming. |
| [SISAP index challenge](https://sisap-challenges.github.io/2025/evaluation/) | 2023-25 | Hard resource envelopes (16 GB, 8 vCPU, 12 h) plus a *build* task, so deployability is measured rather than asymptotic curves. |
| [CANDOR-Bench](http://me.sage.org.ai/contents/research_papers/2026/2026_candor_bench_sigmod_2026.pdf) | 2026 | Continuous ingestion under concurrent query, where throughput turns out to be insertion-bound. |
| Aumüller and Ceccarello, *Recent approaches and trends in ANN search* | 2023 | The survey-of-benchmarks angle: which benchmark design choices bias conclusions. **[unverified]**: cited in VIBE's bibliography, primary page not read. |

Two corrections worth recording because they are easy to inherit by accident.
There is no "Ding et al." graph-ANN survey; the canonical graph survey is Wang et
al. 2021 above, with Azizi et al. 2025 as its experimental successor. And
`ann-benchmarks` is unmaintained and now points at VIBE; billion-scale work
belongs to [big-ann-benchmarks](https://big-ann-benchmarks.com/).

## What a recall-versus-QPS table misses

Each of these is a measurement the atlas does not yet make. They are listed in
the order they would change our published claims.

1. **Selectivity, not just presence, for filtered search.** Our filtered numbers
   are quoted at 20% and 0.1% selectivity on random predicates; published work
   uses real attributes and sweeps selectivity over orders of magnitude.
2. **Out-of-distribution queries.** VIBE splits in-distribution from OOD and
   reports them separately. A 3M tier on synthetic data says nothing about a
   query distribution that shifted.
3. **Per-query difficulty.** Aggregates hide the gap between the easiest and
   hardest 100 queries; a method can win the mean and lose the tail.
4. **Updates and churn.** No row here carries an update rate. Insertion and
   deletion cost, and recall after churn, are unmeasured.
5. **Concurrency.** Every row is one query on one thread. Tail latency under
   simultaneous readers is a different number and a different ranking.
6. **Build at a fixed budget.** SISAP bounds RAM, CPUs and wall clock; an
   unbounded build makes a large tier meaningless as a deployability claim.
7. **Energy and hardware.** Nothing here measures joules per query, and the GPU
   path does not exist.
8. **Metric semantics.** Cosine on un-normalised input, inner product, and
   L2-with-renormalisation are not interchangeable, and a benchmark that mixes
   them is not comparing methods.

## Methods not implemented

Grouped by what they add over the closest thing in this crate. One line each; the
URL is the primary source.

### Graph, GPU and parallel

- **CAGRA** ([2308.15136](https://arxiv.org/abs/2308.15136)): GPU graph
  *construction*, not just search.
- **SONG** ([ICDE'20](https://doi.org/10.1109/ICDE48307.2020.00094)): decouples
  traversal into batched stages, so it accelerates an existing extracted graph.
- **GGNN** ([1912.01059](https://arxiv.org/abs/1912.01059)): single-kernel GPU
  search with no host round-trips.
- **Glass** ([repo](https://github.com/zilliztech/pyglass)): HNSW with in-index
  SQ4U/SQ8U/PQ codes and hand-written kernels; top-3 at 95% recall in VIBE. No
  peer-reviewed paper found, so this is documented as software rather than as a
  publication.
- **FANNG** ([CVPR'16](https://doi.org/10.1109/CVPR.2016.616)): occlusion-rule
  pruning, a stronger RNG approximation than HNSW's.
- **Flat NSW** ([2412.01940](https://arxiv.org/abs/2412.01940)): the hierarchy
  removed; parity on median and tail latency with less build memory. A negative
  control for HNSW tuning rather than a replacement.

### Graph, dynamic and streaming

- **FreshDiskANN** ([2105.09613](https://arxiv.org/abs/2105.09613)): a small
  in-memory index merged into the on-disk graph.
- **OdinANN** ([FAST'26](https://www.usenix.org/conference/fast26/presentation/guo)):
  direct insert with no merge, and bounded latency fluctuation under concurrent
  insert.
- **SPFresh** ([SOSP'23](https://doi.org/10.1145/3600006.3613166)): in-place
  rebalancing that splits and reassigns only boundary vectors.
- **Vamana shard-merge build** ([DiskANN](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html)):
  k-means sharding plus a merge pass, which is how a billion points fit in 64 GB.

### Graph, filtered and range-filtered

- **FilteredVamana / StitchedVamana** ([WWW'23](https://doi.org/10.1145/3543507.3583552)):
  label-aware pruning and per-label stitched graphs.
- **ACORN** ([SIGMOD'24](https://arxiv.org/abs/2403.04871)): predicate-agnostic
  densified HNSW, which is what our `search_acorn` approximates.
- **CAPS** ([2308.15014](https://arxiv.org/abs/2308.15014)): partition-first
  filtered search, strongest at low selectivity where graphs are weakest.
- **iRangeGraph** ([SIGMOD'25](https://arxiv.org/abs/2409.02571)): a range graph
  improvised at query time from elemental graphs.
- **DIGRA** ([PDF](https://www1.se.cuhk.edu.hk/~swang/V3mod148-DIGRA.pdf)): a
  dynamic multi-way attribute tree fused with NSW graphs, so range filtering
  accepts updates, which the static variants do not. The DOI is unconfirmed; the
  author-hosted PDF is the verified source.

### Filtered search, 2026

The 2023-2025 line above is no longer the frontier. These are verified primary
results, all from 2026, and they change what a filtered benchmark should compare.

- **FAVOR** ([2605.07770](https://arxiv.org/abs/2605.07770)): an *exclusion
  distance* that pushes non-target vectors away instead of letting them occupy
  beams, plus a selectivity-driven switch between brute force and graph search.
  1.3× to 5× QPS at recall 0.95 under arbitrary predicates, which is the
  strongest filter-agnostic result published so far.
- **E2E** ([2602.06721](https://arxiv.org/abs/2602.06721)): predicts a filtered
  query's cost from early-probe attribute and distance statistics, then terminates
  it adaptively; 1.1× to 3.7× over fixed termination at recall 0.95.
- **EMA** ([2606.00734](https://arxiv.org/abs/2606.00734)): attaches *markers* to
  graph edges as conservative predicate summaries, so mixed numerical and
  categorical predicates work with zero marker-level false negatives, and the
  index takes dynamic updates. 1.68× to 12.25× over general filtered methods.
- **RNSG** ([2603.12913](https://arxiv.org/abs/2603.12913)): a range-aware
  relative neighbourhood graph, proved monotonic-searchable and structurally
  hereditary, so one graph answers every range instead of one graph per range.
- **RaG-Tree** ([2608.01255](https://arxiv.org/abs/2608.01255)): an R-tree over
  the attributes combined with HNSW over the vectors, for multi-attribute ranges.
- **REACH** ([2610.34109](https://arxiv.org/abs/2610.34109)): geo-filtered search
  on a single proximity graph, after showing that a disk filter breaks greedy
  routing on selective disks.
- **UNIFY** ([2412.02448](https://arxiv.org/abs/2412.02448)): a segmented inclusive
  graph built so that the subgraph for any combination of attribute segments is
  contained in the whole, which lets pre-, post- and hybrid filtering all run on
  one index instead of one index per range strategy.
- **BOA** ([2609.16175](https://arxiv.org/abs/2609.16175)): online beamwidth
  adaptation across a GPU query batch, exploiting the finding that more than half
  of queries are solved by a beam of 50 or less while a fixed-beam system pays 500
  for all of them; 7× to 12.5× throughput.
- **Interval-aware graph indexing** ([2606.11789](https://arxiv.org/abs/2606.11789)):
  the interval-valued attribute case, which is what a timestamp or price range
  actually is.

- **Curator** ([2601.01291](https://arxiv.org/abs/2601.01291)): a dual index for
  low-selectivity filters. Per-label indexes live inside one shared clustering
  tree, which is what bounds the memory overhead at about 4%, and complex
  predicates are served by building a temporary index on the fly. Up to 20.9×
  lower latency than a pre-filtering fallback at 5.5% more build time. This is
  the design our `curator` family follows.

### Dynamic graph indexes on GPU, 2026

- **ETALE** ([2607.02543](https://arxiv.org/abs/2607.02543)): evolving topology
  with lock-free execution, so a GPU graph accepts updates instead of being
  rebuilt.
- **GrAND** ([2608.21163](https://arxiv.org/abs/2608.21163)): GPU dynamic graph
  indexes for approximate nearest neighbour search.

### Guarantees and post-hoc tuning, 2026

- **HNSW with accuracy guarantees using graph spanners**
  ([2607.02338](https://arxiv.org/abs/2607.02338)): certifies an ordinary HNSW
  result with a distribution-free test and escalates to exact recovery when the
  certificate fails, bounding the true-neighbour distance via the graph's
  empirical stretch factor. This is the closest thing to "average-case speed,
  worst-case correctness" and it makes a recall-only benchmark incomplete.
- **RP-Tuning** ([2602.08097](https://arxiv.org/abs/2602.08097)): adjusts
  DiskANN's α-reachability parameter by re-pruning a built graph rather than
  rebuilding it, up to 43× faster tuning with the worst-case reachability
  guarantee preserved. A sweep over `α` is therefore no longer a sweep over builds,
  which changes what a build-cost axis means.

### Early termination

- **LAET** ([code](https://github.com/pkusys/Auncel)): learned per-query stopping.
- **DARTH** ([2505.19001](https://arxiv.org/abs/2505.19001)): declarative recall,
  predicting current recall mid-search.
- **ANNiE** ([PVLDB 19](https://www.vldb.org/pvldb/vol19/p3820-wang.pdf)):
  pre-search cost estimation plus a probabilistic recall guarantee.
- **Auncel** ([NSDI'23](https://www.usenix.org/static_files/nsdi23_slides_zhang-zili.pdf)):
  error-bound termination for IVF, with a bounded rather than average-case
  guarantee.

### Quantisation

- **TurboQuant** ([2504.19874](https://arxiv.org/abs/2504.19874)): data-oblivious,
  training-free quantisation, so it is streaming-safe.
- **LVQ** ([VLDB'23](https://doi.org/10.14778/3611479.3611537)): per-vector
  scaling tuned for graph traversal's random access pattern.
- **ScaNN anisotropic** ([1908.10396](https://arxiv.org/abs/1908.10396)): loss
  weighted along the query direction, plus 4-bit SIMD scoring.
- **OPQ / LOPQ** ([TPAMI'14](https://doi.org/10.1109/TPAMI.2013.240),
  [CVPR'14](https://doi.org/10.1109/CVPR.2014.298)): a learned rotation, then a
  per-cell one.
- **SymphonyQG** ([SIGMOD'25](https://doi.org/10.1145/3709730)): the graph built
  *for* FastScan rather than quantised afterwards. Partially present here as
  `hnsw::symphony_qg`.
- **LoRANN** ([NeurIPS'24](https://arxiv.org/abs/2410.18926)): a supervised
  low-rank bilinear scorer instead of PQ; wins below 90% recall and at high `d`.
- **RaBitQ-SSD** ([2610.02652](https://arxiv.org/abs/2610.02652)): split codes and
  pipelined I/O, carrying the 1-bit advantage from RAM to disk.

### Inverted and partition

- **SPANN** ([2111.08566](https://arxiv.org/abs/2111.08566)): three things our
  IVF lacks, namely balanced multi-constraint clustering to bound posting length,
  closure multi-assignment of boundary vectors, and query-aware posting pruning.
- **Seismic** ([SIGIR'24](https://arxiv.org/abs/2404.18812)): geometric block
  organisation for sparse retrieval, beating the Big-ANN'23 sparse winners.
- **Block-Max Pruning** ([2405.01117](https://arxiv.org/abs/2405.01117)): block
  bounds instead of per-term bounds, for SPLADE-scale sparse workloads.

### Multi-vector

- **MUVERA** ([NeurIPS'24](https://arxiv.org/abs/2405.19504)): reduces
  multi-vector search to single-vector MIPS, so an existing index can serve it.
  The cheapest of these to adopt.
- **PLAID** ([SIGIR'22](https://arxiv.org/abs/2205.09707)) and **ColBERTv2**
  ([NAACL'22](https://arxiv.org/abs/2112.01488)): the late-interaction
  representation and the pruned candidate lists over it.

## Five developments most likely to reorder a benchmark within a year

Ranked by the probability that including them changes the *order* of a
recall-versus-QPS table, not by novelty. Each carries what would falsify it.

1. **Per-query search effort stops being a global knob.** DARTH makes recall a
   query-time target ([2505.19001](https://arxiv.org/abs/2505.19001)), ANNiE
   estimates cost before searching
   ([PVLDB 19](https://www.vldb.org/pvldb/vol19/p3820-wang.pdf)), E2E predicts
   filtered-query cost from early-probe statistics
   ([2602.06721](https://arxiv.org/abs/2602.06721)), and BOA shows a GPU batch
   averaging a beam of 22 to 77 where a fixed-beam system pays 500
   ([2609.16175](https://arxiv.org/abs/2609.16175)). A benchmark that sweeps one
   global `ef` systematically under-rewards all of them. Falsified if their gains
   vanish at p99 once predictor overhead is counted under concurrency.
2. **Filtered search becomes a matrix, and the 2026 methods are filter-agnostic.**
   A 23-method taxonomy on real embeddings finds no universal winner
   ([2507.21989](https://arxiv.org/abs/2507.21989)); FAVOR reaches 1.3× to 5×
   QPS at recall 0.95 under arbitrary predicates
   ([2605.07770](https://arxiv.org/abs/2605.07770)); EMA handles mixed numeric
   and categorical predicates with updates
   ([2606.00734](https://arxiv.org/abs/2606.00734)); RNSG answers every range
   from one graph ([2603.12913](https://arxiv.org/abs/2603.12913)). Falsified if
   real filtered workloads collapse to one predicate family, leaving a single
   column.
3. **RaBitQ becomes a full stack: RAM, SSD, GPU, with theory.** RaBitQ
   ([SIGMOD'24](https://arxiv.org/abs/2405.12497)) to IVF-RaBitQ on GPU
   ([2602.23999](https://arxiv.org/abs/2602.23999)) to RaBitQ-SSD
   ([2610.02652](https://arxiv.org/abs/2610.02652)), with an independent
   re-evaluation ([2604.19528](https://arxiv.org/abs/2604.19528)). Falsified if a
   full-precision graph with a better layout beats 4-bit codes at 95% recall on
   `d ≥ 768` in the same RAM budget and build time.
4. **Search acquires certificates, so recall stops being the only axis.** HNSW
   with graph spanners certifies a result and escalates to exact recovery when
   the certificate fails ([2607.02338](https://arxiv.org/abs/2607.02338)), and
   RP-Tuning preserves DiskANN's worst-case reachability while re-pruning instead
   of rebuilding ([2602.08097](https://arxiv.org/abs/2602.08097)). A recall number
   becomes a *reported* property rather than the only one. Falsified if
   certification overhead makes it unusable below 95% recall.
5. **Streaming under concurrency becomes the default frame.** CANDOR-Bench finds
   throughput to be insertion-bound and that no algorithm wins on recall,
   throughput and update cost together
   ([CANDOR-Bench](http://me.sage.org.ai/contents/research_papers/2026/2026_candor_bench_sigmod_2026.pdf),
   and the earlier [CANDY](https://arxiv.org/abs/2406.19651)), and the GPU dynamic line
   (ETALE [2607.02543](https://arxiv.org/abs/2607.02543), GrAND
   [2608.21163](https://arxiv.org/abs/2608.21163)) now takes updates rather than
   rebuilding. Falsified if static-snapshot rankings predict streaming ones at a
   rank correlation above about 0.9.

## What we adopt first

1. **Selectivity and per-query difficulty in the filtered numbers**, because our
   filtered claims are the ones least comparable to published work today, and the
   2026 baselines to beat are FAVOR and EMA rather than ACORN.
2. **An out-of-distribution track**, because every current tier measures
   in-distribution synthetic data and VIBE shows that overstates all methods.
3. **A resource envelope on the large tiers**: fixed RAM, CPUs and wall clock, as
   SISAP does, so "3M tier" becomes a deployability claim rather than a
   self-reported number.
4. **A per-query effort axis**, because a single `ef` sweep cannot express what
   DARTH, ANNiE, E2E and BOA are solving, and would report them as worse than
   they are.

Two of these change what an existing figure means rather than adding a new one.
Once recall can be certified ([2607.02338](https://arxiv.org/abs/2607.02338)),
a recall-versus-QPS curve is a description of typical behaviour and not a claim
about any given query, which is worth stating wherever recall is quoted.

