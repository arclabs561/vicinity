# References (primary sources)

This page is a **curated bibliography** for the algorithms and phenomena referenced in `vicinity`.
It is intended to backstop claims in module docs and to give you a starting point for deeper reading.

## Small-world graph ANN (HNSW / NSW / NSG)

- Malkov, Yashunin (2016/2018). *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.* (HNSW)  
  `https://arxiv.org/abs/1603.09320`

- Malkov, Ponomarenko, Logvinov, Krylov (2014). *Approximate nearest neighbor algorithm based on navigable small world graphs.* (NSW)  
  `https://doi.org/10.1016/j.is.2013.10.006`

- Fu, Xiang, Wang, Huang (2017). *Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph (NSG).*  
  `https://arxiv.org/abs/1707.00143`

## “Hierarchy may not matter” (flat vs hierarchical)

- Munyampirwa et al. (2024). *Down with the Hierarchy: The “H” in HNSW Stands for “Hubs”.*  
  `https://arxiv.org/abs/2412.01940`

## SSD / out-of-core graph ANN (DiskANN / Vamana-style graphs)

- Subramanya, Devvrit, Simhadri, Krishnaswamy, Kadekodi, Bhattacharya, Srinivasa (2019). *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node.* (NeurIPS 2019)  
  `https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html`

## Partition + quantization (IVF / PQ / OPQ)

- Jégou, Douze, Schmid (2011). *Product Quantization for Nearest Neighbor Search.* (PQ / IVFADC)  
  `https://ieeexplore.ieee.org/document/5432202`

- Ge, He, Ke, Sun (2014). *Optimized Product Quantization.* (OPQ)  
  `https://arxiv.org/abs/1311.4055`

## MIPS / ScaNN (anisotropic / query-aware quantization)

- Guo et al. (2020). *Accelerating Large-Scale Inference with Anisotropic Vector Quantization.* (AVQ / ScaNN line)  
  `https://arxiv.org/abs/1908.10396`

## Filtering in graph ANN (predicate-aware search)

- Wang et al. (2024). *ACORN: Approximate Nearest Neighbor Search with Attribute Filtering.* (SIGMOD 2024)  
  `https://dl.acm.org/doi/10.1145/3626246.3653367`

## Probabilistic routing / learned navigation

- Lu, Xiao, Ishikawa (2024). *Probabilistic Routing for Graph-Based Approximate Nearest Neighbor Search.*  
  `https://arxiv.org/abs/2402.11354`

## Dataset difficulty, hubness, and distance concentration

- Radovanović, Nanopoulos, Ivanović (2010). *Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data.* (hubness)  
  `https://link.springer.com/chapter/10.1007/978-3-642-15880-3_28`

- Beyer, Goldstein, Ramakrishnan, Shaft (1999). *When Is “Nearest Neighbor” Meaningful?* (distance concentration)  
  `https://doi.org/10.1007/s007780050006`

## Graph-based ANN surveys and theory

- Wang, Xu, Yue, Wang (2021). *A Comprehensive Survey and Experimental Comparison of Graph-Based Approximate Nearest Neighbor Search.*
  `https://arxiv.org/abs/2101.12631`

- Lin, Zhao (2019). *Graph-based Nearest Neighbor Search: Promises and Failures.*
  `https://arxiv.org/abs/1904.02077`

- Prokhorenkova, Shekhovtsov (2019). *Graph-based Nearest Neighbor Search: From Practice to Theory.*
  `https://arxiv.org/abs/1907.00845`

- Laarhoven (2017). *Graph-based Time-Space Trade-offs for Approximate Near Neighbors.*
  `https://arxiv.org/abs/1712.03158`

## Construction improvements

- Yang et al. (2024). *Revisiting the Index Construction of Proximity Graph-Based Approximate Nearest Neighbor Search.* (alpha-pruning, 5.6x construction speedup)
  `https://arxiv.org/abs/2410.01231`

- Dehghankar, Asudeh (2025). *HENN: A Hierarchical Epsilon Net Navigation Graph for Approximate Nearest Neighbor Search.* (provable bounds)
  `https://arxiv.org/abs/2505.17368`

- Ponomarenko (2025). *Three Algorithms for Merging Hierarchical Navigable Small World Graphs.* (distributed indexing)
  `https://arxiv.org/abs/2505.16064`

## Routing and search

- Baranchuk, Persiyanov, Sinitsin, Babenko (2019). *Learning to Route in Similarity Graphs.* (ICML 2019, learned routing)
  `https://arxiv.org/abs/1905.10987`

## RaBitQ and quantization advances

- Gao, Long (2024). *RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound.* (SIGMOD 2024)
  `https://arxiv.org/abs/2405.12497`

- Gao et al. (2025). *Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space.* (multi-bit RaBitQ, SIGMOD 2025)
  `https://arxiv.org/abs/2409.09913`

## Hubness and intrinsic dimensionality

- Angiulli (2018). *On the Behavior of Intrinsically High-Dimensional Spaces: Distances, Direct and Reverse Nearest Neighbors, and Hubness.* (JMLR 18)

- Nguyen et al. (2025). *Dual-Branch HNSW Approach with Skip Bridges and LID-Driven Optimization.*
  `https://arxiv.org/abs/2501.13992`

## Streaming / dynamic updates

- Singh, Subramanya, Krishnaswamy, Simhadri (2021). *FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search.* (streaming insert/delete with merge)
  `https://arxiv.org/abs/2105.09613`

- Xu, Dobson Manohar, Bernstein, Chandramouli, Wen, Simhadri (2025). *In-Place Updates of a Graph Index for Streaming Approximate Nearest Neighbor Search.* (IP-DiskANN)
  `https://arxiv.org/abs/2502.13826`

- Liu et al. (2025). *Wolverine: Highly Efficient Monotonic Search Path Repair for Graph-Based ANN Index Updates.* (PVLDB 18; 11x deletion throughput via 2-hop in-edge repair)

- Yu et al. (2025). *A Topology-Aware Localized Update Strategy for Graph-Based ANN Index.* (batch updates without topology degradation)
  `https://arxiv.org/abs/2503.00402`

- Mohoney et al. (2024). *Incremental IVF Index Maintenance for Streaming Vector Search.* (Ada-IVF; incremental codebook updates)
  `https://arxiv.org/abs/2411.00970`

## Filtered / hybrid search

- Li et al. (2025). *Attribute Filtering in Approximate Nearest Neighbor Search: An In-depth Experimental Study.* (no single method dominates across selectivity ratios)
  `https://arxiv.org/abs/2508.16263`

- Xu et al. (2026). *JAG: Joint Attribute Graphs for Filtered Nearest Neighbor Search.* (single graph encoding similarity + predicates)
  `https://arxiv.org/abs/2602.10258`

## Error-bounded monotonic graphs

- Yin et al. (2025). *delta-EMG: Error-Bounded Monotonic Graph for Approximate Nearest Neighbor Search.* (provable per-query distance bounds via occlusion pruning)
  `https://arxiv.org/abs/2511.16921`

## Range-filtered ANN

- Yang, Li, Shen, Xiao, Wang (2025). *ESG: Elastic Graphs for Range-Filtering Approximate kNN Search.* (2 subgraph searches vs O(log N))
  `https://arxiv.org/abs/2504.04018`

## Low-selectivity filtered ANN

- Jin, Wu, Hu, Maggs, Yang, Zhang, Zhuo (2026). *Curator: Efficient Vector Search with Low-Selectivity Filters.* (SIGMOD 2026; k-means tree + per-label buffers)
  `https://arxiv.org/abs/2601.01291`

- Kim, Choe (2026). *RACORN-1: Adaptive Recall-Preserving Speedup for Low-Selectivity Filtered Vector Search.* (ACORN-1 extension with adaptive search fallback and exact fallback)
  `https://arxiv.org/abs/2607.00768`

## IVF-RaBitQ

- Chen et al. (2026). *IVF-RaBitQ: GPU-native IVF with RaBitQ quantization.*
  `https://arxiv.org/abs/2602.23999`

## Multi-vector retrieval (ColBERT-style)

- Kulkarni, Hrishikesh, Simhadri (2026). *LEMUR: Learned Multi-Vector Retrieval.* (MLP + OLS for MaxSim-compatible single-vector ANNS)
  `https://arxiv.org/abs/2601.21853`

## LSM-tree vector indexing

- Bai et al. (2025). *LSM-VEC: Streaming Vector Search via LSM-tree.* (Poly-LSM engine, delta/pivot entries)
  `https://arxiv.org/abs/2505.17152`

## Parallel graph construction

- Rubel, Wen, Dhulipala, Gottesbüren, Jayaram, Łącki (2026). *PiPNN: Ultra-Scalable Graph-Based Nearest Neighbor Indexing.* (overlapping partitions + GEMM + HashPrune; 11.6x faster than Vamana)
  `https://arxiv.org/abs/2602.21247`

## Quantization-graph fusion

- Gou, Gao, Xu, Long (2025). *SymphonyQG: Towards Symphonious Integration of Quantization and Graph for Approximate Nearest Neighbor Search.* (SIGMOD 2025; RaBitQ-style distance during graph traversal)

## SNG theory

- Ma et al. (2025). *Sparse Neighborhood Graph-Based Approximate Nearest Neighbor Search Revisited: Theoretical Analysis and Optimization.* (first rigorous SNG bounds)
  `https://arxiv.org/abs/2509.15531`

- Conway et al. (2025). *Efficiently Constructing Sparse Navigable Graphs.* (proves greedy-insert + alpha-pruning produces navigable graphs)
  `https://arxiv.org/abs/2507.13296`

## Robustness

- Hua et al. (2025). *Dynamically Detect and Fix Hardness for Efficient Approximate Nearest Neighbor Search.* (OOD query detection + adaptive effort)
  `https://arxiv.org/abs/2510.22316`

## Inverted multi-index

- Baranchuk, Babenko, Malkov (2018). *Revisiting the Inverted Indices for Billion-Scale Approximate Nearest Neighbors.* (improved IVF coarse quantization)
  `https://arxiv.org/abs/1802.02422`

## Projection-based distance bounds (FINGER / PAG)

- Chen, Wei-cheng, Yu, Dhillon, Hsieh (2022). *FINGER: Fast Inference for Graph-based Approximate Nearest Neighbor Search.* (edge-projection distance bounds; basis for `finger` module)
  `https://arxiv.org/abs/2206.11408`

- Ma et al. (2026). *PAG: Projection-Augmented Graph for ANN Search.* (PRT + TFB + PES; up to 5x HNSW QPS on high-dim embeddings)
  `https://arxiv.org/abs/2603.06660`

## Projection-quantization fusion (MRQ / QMP)

- Yang, Jing, Li, Wang (2024). *Quantization Meets Projection: A Happy Marriage for Approximate k-Nearest Neighbor Search.* (projection concentrates info in leading dims; basis for `rp_quant` module)
  `https://arxiv.org/abs/2411.06158`

## Rotation-based quantization

- Zandieh, Daliri, Hadian, Mirrokni (2025). *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* (orthogonal rotation + 1-bit QJL residual; near Shannon lower bound)
  `https://arxiv.org/abs/2504.19874`

## Declarative recall / early termination

- Chatzakis, Papakonstantinou, Palpanas (2025). *DARTH: Declarative Recall Through Early Termination for Approximate Nearest Neighbor Search.* (ML-driven recall predictor; 6-14x speedups)
  `https://arxiv.org/abs/2505.19001`

## Adaptive indexing

- Mohoney, Sarda, Tang, Chowdhury, Pacaci, Ilyas, Rekatsinas, Venkataraman (2025). *Quake: Adaptive Indexing for Vector Search.* (ML cost model for adaptive partitioning under data drift)
  `https://arxiv.org/abs/2506.03437`

## Adaptive entry point selection

- Oguri, Matsui (2024). *Theoretical and Empirical Analysis of Adaptive Entry Point Selection for Graph-based Approximate Nearest Neighbor Search.* (b-monotonic paths, B-MSNET)
  `https://arxiv.org/abs/2402.04713`

## NN-descent

- Dong, Moses, Li (2011). *Efficient k-nearest neighbor graph construction for generic similarity measures.* (NN-descent)
  `https://doi.org/10.1145/1963405.1963487`

## Extended quantization

- Chen et al. (2026). *Extended-RaBitQ: Arbitrary-rate vector quantization from 2-7 bits/dim.*
  `https://github.com/VectorDB-NTU/Extended-RaBitQ`

- Google Research (2025). *SOAR: Orthogonality-Amplified Residuals for reduced correlated search failures in ScaNN.*
  `https://research.google/blog/soar-new-algorithms-for-even-faster-vector-search-with-scann/`

## Coding theory + LSH

- Amazon Science (2025). *PCNN: Polar Code Nearest Neighbor -- multiprobe LSH via error-correcting code list decoding.*
  `https://www.amazon.science/publications/approximate-nearest-neighbor-search-through-modern-error-correcting-codes`

## Graph layout optimization

- Zheng et al. (2025). *MARGO: Graph Layout Optimization for Disk-Based ANN via Monotonic Reachability Weighting.* (VLDB 2025)

## Capacity-law failure

- PMC (2026). *Capacity-limited failure in HNSW: discontinuous breakdown at k ~ 2-3.5 * efSearch.* (ResNet-50 on STL-10)
  `https://pmc.ncbi.nlm.nih.gov/articles/PMC12942108/`

## Parallel / concurrent ANN

- ParlayANN (CMU, 2025). *Lock-free deterministic parallel graph-based ANN (DiskANN, HNSW, HCNNG, pyNNDescent). CLEANN-Tree: first linearizable concurrent k-NN structure.*
  `https://arxiv.org/abs/2603.06660`

## GPU graph ANN

- CAGRA, CUHNSW. *GPU graph construction and search; significantly outperforms 40-thread CPU HNSW.*
  `https://www.shimin-chen.com/papers/gpu-graph-anns-hardbdactive25.pdf`

## Practical benchmarking references

- Aumueller, Bernhardsson, Faithfull (2020). *ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms.* (Information Systems, 87)
  `http://ann-benchmarks.com/`

- Simhadri et al. (2024). *Results of the Big ANN: NeurIPS'23 Competition.* (billion-scale baselines)
  `https://arxiv.org/abs/2409.17424`
