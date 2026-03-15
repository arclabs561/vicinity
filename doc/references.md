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

## Practical benchmarking references

- Aumueller, Bernhardsson, Faithfull (2020). *ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms.* (Information Systems, 87)
  `http://ann-benchmarks.com/`

- Simhadri et al. (2024). *Results of the Big ANN: NeurIPS'23 Competition.* (billion-scale baselines)
  `https://arxiv.org/abs/2409.17424`

