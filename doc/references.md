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

## Practical benchmarking references

- Aumueller, Bernhardsson, Faithfull (2020). *ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms.* (Information Systems, 87)
  `http://ann-benchmarks.com/`

- Simhadri et al. (2024). *Results of the Big ANN: NeurIPS'23 Competition.* (billion-scale baselines)
  `https://arxiv.org/abs/2409.17424`

