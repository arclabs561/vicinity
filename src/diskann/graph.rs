//! DiskANN graph structure and Vamana construction.

use crate::RetrieveError;
use rand::seq::SliceRandom;
use rand::Rng;
use smallvec::SmallVec;
use std::collections::HashSet;
use std::path::Path;

/// DiskANN index for disk-based approximate nearest neighbor search.
///
/// Implements the Vamana graph construction algorithm:
/// 1. Random graph initialization
/// 2. Two-pass construction (alpha=1.0, then alpha>1.0)
/// 3. Robust pruning (alpha-pruning) to maintain long-range edges
pub struct DiskANNIndex {
    dimension: usize,
    params: DiskANNParams,
    built: bool,

    // Vectors stored in memory for build (would be on disk in prod)
    vectors: Vec<f32>,
    num_vectors: usize,

    /// External doc_ids aligned with internal indices
    doc_ids: Vec<u32>,

    // Graph structure (adjacency list)
    // Using SmallVec to optimize for typical degree M=16-32
    // Stored in memory for construction, serialized to disk later
    adj: Vec<SmallVec<[u32; 32]>>,

    // Entry point for search (medoid)
    start_node: u32,
}

impl DiskANNIndex {
    /// Vector dimensionality.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Number of vectors currently stored in the index.
    #[inline]
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    /// Default search width (`ef_search`) configured for this index.
    #[inline]
    pub fn ef_search(&self) -> usize {
        self.params.ef_search
    }

    /// Approximate memory usage in bytes (vectors + adjacency lists).
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.vectors.len() * std::mem::size_of::<f32>()
            + self
                .adj
                .iter()
                .map(|n| n.len() * std::mem::size_of::<u32>())
                .sum::<usize>()
    }

    /// Save the built index to disk.
    ///
    /// Saves:
    /// - Graph structure (adjacency list) using DiskGraphWriter
    /// - Vectors (flat binary format)
    /// - Metadata (JSON)
    pub fn save(&self, output_dir: &Path) -> Result<(), RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot save unbuilt index".into(),
            ));
        }

        if !output_dir.exists() {
            std::fs::create_dir_all(output_dir).map_err(|e| RetrieveError::Io(e.to_string()))?;
        }

        // 1. Save Vectors (vectors.bin)
        let vectors_path = output_dir.join("vectors.bin");
        let mut vectors_file =
            std::fs::File::create(&vectors_path).map_err(|e| RetrieveError::Io(e.to_string()))?;
        let vectors_bytes = unsafe {
            std::slice::from_raw_parts(
                self.vectors.as_ptr() as *const u8,
                self.vectors.len() * std::mem::size_of::<f32>(),
            )
        };
        use std::io::Write;
        vectors_file
            .write_all(vectors_bytes)
            .map_err(|e| RetrieveError::Io(e.to_string()))?;

        // 2. Save Graph (graph.index)
        let graph_path = output_dir.join("graph.index");
        // Convert persistence error to RetrieveError if needed, or handle unwraps
        // We'll define a simple wrapper
        let mut graph_writer = super::disk_io::DiskGraphWriter::new(
            &graph_path,
            self.num_vectors,
            self.params.m,
            self.start_node,
        )
        .map_err(|e| RetrieveError::Io(format!("failed to create graph writer: {}", e)))?;

        for neighbors in &self.adj {
            graph_writer
                .write_adjacency(neighbors)
                .map_err(|e| RetrieveError::Io(format!("failed to write adjacency: {}", e)))?;
        }
        graph_writer
            .flush()
            .map_err(|e| RetrieveError::Io(format!("failed to flush graph: {}", e)))?;

        // 3. Save Metadata (metadata.json)
        let metadata_path = output_dir.join("metadata.json");
        let metadata = serde_json::json!({
            "dimension": self.dimension,
            "num_vectors": self.num_vectors,
            "start_node": self.start_node,
            "params": {
                "m": self.params.m,
                "ef_construction": self.params.ef_construction,
                "alpha": self.params.alpha,
                "ef_search": self.params.ef_search
            }
        });
        let metadata_file =
            std::fs::File::create(&metadata_path).map_err(|e| RetrieveError::Io(e.to_string()))?;
        serde_json::to_writer_pretty(metadata_file, &metadata)
            .map_err(|e| RetrieveError::Serialization(e.to_string()))?; // Need to add Serialization error to RetrieveError

        Ok(())
    }
}

/// Disk-based searcher for DiskANN.
///
/// Operates on persisted index without loading the full graph into RAM.
pub struct DiskANNSearcher {
    dimension: usize,
    num_vectors: usize,
    start_node: u32,
    params: DiskANNParams,

    // Components
    graph_reader: super::disk_io::DiskGraphReader,
    vectors_file: std::fs::File, // Or mmap
                                 // Using simple file I/O for vectors for now, upgradable to mmap
}

impl DiskANNSearcher {
    /// Load searcher from index directory.
    pub fn load(index_dir: &Path) -> Result<Self, RetrieveError> {
        // 1. Load Metadata
        let metadata_path = index_dir.join("metadata.json");
        let metadata_file =
            std::fs::File::open(&metadata_path).map_err(|e| RetrieveError::Io(e.to_string()))?;
        let metadata: serde_json::Value = serde_json::from_reader(metadata_file)
            .map_err(|e| RetrieveError::Serialization(e.to_string()))?;

        let dimension = metadata["dimension"]
            .as_u64()
            .ok_or(RetrieveError::FormatError("Missing dimension".to_string()))?
            as usize;
        let num_vectors = metadata["num_vectors"]
            .as_u64()
            .ok_or(RetrieveError::FormatError(
                "Missing num_vectors".to_string(),
            ))? as usize;
        let start_node = metadata["start_node"]
            .as_u64()
            .ok_or(RetrieveError::FormatError("Missing start_node".to_string()))?
            as u32;

        let params_val = &metadata["params"];
        let params = DiskANNParams {
            m: params_val["m"].as_u64().unwrap_or(32) as usize,
            ef_construction: params_val["ef_construction"].as_u64().unwrap_or(100) as usize,
            alpha: params_val["alpha"].as_f64().unwrap_or(1.2) as f32,
            ef_search: params_val["ef_search"].as_u64().unwrap_or(100) as usize,
        };

        // 2. Open Graph
        let graph_path = index_dir.join("graph.index");
        let graph_reader = super::disk_io::DiskGraphReader::open(&graph_path)
            .map_err(|e| RetrieveError::Io(format!("failed to open graph: {}", e)))?;

        // 3. Open Vectors
        let vectors_path = index_dir.join("vectors.bin");
        let vectors_file =
            std::fs::File::open(&vectors_path).map_err(|e| RetrieveError::Io(e.to_string()))?;

        Ok(Self {
            dimension,
            num_vectors,
            start_node,
            params,
            graph_reader,
            vectors_file,
        })
    }

    /// Search for k nearest neighbors using disk-based graph.
    pub fn search(
        &mut self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        let ef = ef_search.max(k).max(self.params.ef_search);

        // Use greedy search similar to in-memory, but fetching neighbors from disk
        // Note: Performance will be limited by random I/O here without caching/prefetching
        // This is a functional baseline.

        let mut visited = HashSet::new();
        let mut retset: Vec<Candidate> = Vec::with_capacity(ef + 1);

        // Fetch start node vector
        let start_vec = self.get_vector(self.start_node)?;
        let start_dist = self.dist(query, &start_vec);

        retset.push(Candidate {
            id: self.start_node,
            dist: start_dist,
        });
        visited.insert(self.start_node);

        let mut current_idx = 0;

        while current_idx < retset.len() {
            retset.sort_by(|a, b| a.dist.total_cmp(&b.dist));

            if current_idx >= retset.len() {
                break;
            }

            let current = retset[current_idx];
            current_idx += 1;

            // Fetch neighbors from disk
            // TODO: Cache hot nodes (top levels of Vamana) in RAM
            let neighbors = self.graph_reader.get_neighbors(current.id)?;

            for neighbor in neighbors {
                if visited.contains(&neighbor) {
                    continue;
                }
                visited.insert(neighbor);

                // Fetch neighbor vector from disk
                let neighbor_vec = self.get_vector(neighbor)?;
                let dist = self.dist(query, &neighbor_vec);

                retset.push(Candidate { id: neighbor, dist });
            }

            // Keep top L
            retset.sort_by(|a, b| a.dist.total_cmp(&b.dist));
            if retset.len() > ef {
                retset.truncate(ef);
            }
        }

        Ok(retset.into_iter().take(k).map(|c| (c.id, c.dist)).collect())
    }

    fn get_vector(&mut self, idx: u32) -> Result<Vec<f32>, RetrieveError> {
        use std::io::{Read, Seek, SeekFrom};
        let offset = idx as u64 * self.dimension as u64 * 4;
        self.vectors_file
            .seek(SeekFrom::Start(offset))
            .map_err(|e| RetrieveError::Io(e.to_string()))?;

        let mut buffer = vec![0u8; self.dimension * 4];
        self.vectors_file
            .read_exact(&mut buffer)
            .map_err(|e| RetrieveError::Io(e.to_string()))?;

        let mut vec = Vec::with_capacity(self.dimension);
        for i in 0..self.dimension {
            let start = i * 4;
            let val = f32::from_le_bytes([
                buffer[start],
                buffer[start + 1],
                buffer[start + 2],
                buffer[start + 3],
            ]);
            vec.push(val);
        }
        Ok(vec)
    }

    fn dist(&self, a: &[f32], b: &[f32]) -> f32 {
        crate::simd::l2_distance_squared(a, b)
    }
}

/// DiskANN parameters.
#[derive(Clone, Debug)]
pub struct DiskANNParams {
    /// Maximum connections per node (R in paper)
    pub m: usize,

    /// Beam width for construction search (L in paper)
    pub ef_construction: usize,

    /// Alpha parameter for pruning (typically 1.2 - 1.4)
    pub alpha: f32,

    /// Search width
    pub ef_search: usize,
}

impl Default for DiskANNParams {
    fn default() -> Self {
        Self {
            m: 32,
            ef_construction: 100,
            alpha: 1.2,
            ef_search: 100,
        }
    }
}

/// Candidate for priority queues
#[derive(Clone, Copy, PartialEq)]
struct Candidate {
    id: u32,
    dist: f32,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: larger distance = higher priority (for results pruning)
        // Use total_cmp for IEEE 754 total ordering (NaN-safe, NaN > all)
        self.dist.total_cmp(&other.dist)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl DiskANNIndex {
    /// Create a new DiskANN index.
    pub fn new(dimension: usize, params: DiskANNParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            dimension,
            params,
            built: false,
            vectors: Vec::new(),
            num_vectors: 0,
            doc_ids: Vec::new(),
            adj: Vec::new(),
            start_node: 0,
        })
    }

    /// Add a vector to the index.
    pub fn add(&mut self, doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        self.add_slice(doc_id, &vector)
    }

    /// Add a vector to the index from a borrowed slice.
    ///
    /// Notes:
    /// - The index stores vectors internally, so it must copy the slice into its own storage.
    /// - `doc_id` is stored and mapped back in search results.
    pub fn add_slice(&mut self, doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
        if self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot add vectors after index is built".into(),
            ));
        }

        if vector.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: vector.len(),
                doc_dim: self.dimension,
            });
        }

        self.vectors.extend_from_slice(vector);
        self.doc_ids.push(doc_id);
        self.num_vectors += 1;
        self.adj.push(SmallVec::new());
        Ok(())
    }

    /// Build the index using Vamana construction.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }

        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        // 1. Initialize random graph (R-regular)
        self.initialize_random_graph();

        // 2. Compute medoid as start node
        self.start_node = self.compute_medoid();

        // 3. First pass: alpha = 1.0 (approximates RNG)
        // Helps build initial connectivity
        self.vamana_pass(1.0)?;

        // 4. Second pass: alpha = params.alpha (e.g. 1.2)
        // Adds long-range edges for small-world navigation
        self.vamana_pass(self.params.alpha)?;

        self.built = true;
        Ok(())
    }

    /// Initialize random R-regular graph.
    fn initialize_random_graph(&mut self) {
        let mut rng = rand::rng();
        let r = self.params.m;

        for i in 0..self.num_vectors {
            // Pick R random neighbors
            let mut neighbors: HashSet<u32> = HashSet::with_capacity(r);
            while neighbors.len() < r && neighbors.len() < self.num_vectors - 1 {
                let n = rng.random_range(0..self.num_vectors) as u32;
                if n != i as u32 {
                    neighbors.insert(n);
                }
            }
            self.adj[i] = neighbors.into_iter().collect();
        }
    }

    /// Compute geometric medoid of the dataset.
    fn compute_medoid(&self) -> u32 {
        // Approximate medoid by centroid of a sample
        // For simplicity in this implementation, just pick a random node if N is large,
        // or 0. A robust implementation would compute the true centroid.
        // Using 0 is a common valid simplification for prototype.
        0
    }

    /// Single pass of Vamana construction.
    fn vamana_pass(&mut self, alpha: f32) -> Result<(), RetrieveError> {
        // Random permutation of nodes
        let mut nodes: Vec<u32> = (0..self.num_vectors as u32).collect();
        nodes.shuffle(&mut rand::rng());

        for &i in &nodes {
            let query_vec = self.get_vector(i);

            // Greedy search to find candidates
            // We use the graph as it exists so far
            let (visited, _) =
                self.greedy_search(query_vec, self.params.ef_construction, self.start_node);

            // Candidate set V = visited nodes
            // Run RobustPrune on V to find new neighbors for i
            let new_neighbors = self.robust_prune(i, &visited, alpha, self.params.m);

            // Update graph: add directed edges
            self.adj[i as usize] = new_neighbors.into_iter().collect();

            // Note: In full DiskANN, we'd also add reverse edges to keep graph undirected/balanced,
            // but vanilla Vamana works well with directed edges refined this way.
            // For production, we'd enforce max degree on reverse updates.
        }

        Ok(())
    }

    /// RobustPrune (Alpha-Pruning) algorithm.
    ///
    /// Selects neighbors that are close to `node`, but also "orthogonal" to each other
    /// to ensure good coverage of the space.
    fn robust_prune(
        &self,
        node: u32,
        candidates: &[u32],
        alpha: f32,
        max_degree: usize,
    ) -> Vec<u32> {
        let node_vec = self.get_vector(node);

        // 1. Calculate distances to all candidates
        let mut candidates_with_dist: Vec<Candidate> = candidates
            .iter()
            .filter(|&&c| c != node) // distinct
            .map(|&c| Candidate {
                id: c,
                dist: self.dist(node_vec, self.get_vector(c)),
            })
            .collect();

        // Add current neighbors to candidate set (to refine them)
        for &neighbor in &self.adj[node as usize] {
            if !candidates.contains(&neighbor) {
                candidates_with_dist.push(Candidate {
                    id: neighbor,
                    dist: self.dist(node_vec, self.get_vector(neighbor)),
                });
            }
        }

        // 2. Sort by distance (ascending)
        candidates_with_dist.sort_by(|a, b| a.dist.total_cmp(&b.dist));

        // 3. Prune
        let mut new_neighbors: Vec<u32> = Vec::with_capacity(max_degree);

        // Remove duplicates if any
        candidates_with_dist.dedup_by(|a, b| a.id == b.id);

        for cand in candidates_with_dist {
            if new_neighbors.len() >= max_degree {
                break;
            }

            // Check if cand is reachable from any existing neighbor with shorter path
            // alpha parameter controls "shorter": distance(p*, p') <= alpha * distance(p, p')
            let mut prune = false;
            let cand_vec = self.get_vector(cand.id);

            for &existing_neighbor in &new_neighbors {
                let dist_existing_cand = self.dist(self.get_vector(existing_neighbor), cand_vec);

                // If existing neighbor is closer to candidate than node is (scaled by alpha),
                // then candidate is redundant (we can reach it via existing neighbor).
                if alpha * dist_existing_cand <= cand.dist {
                    prune = true;
                    break;
                }
            }

            if !prune {
                new_neighbors.push(cand.id);
            }
        }

        new_neighbors
    }

    /// Greedy search for construction and querying.
    ///
    /// Returns (visited_nodes, nearest_candidates).
    fn greedy_search(
        &self,
        query: &[f32],
        l_size: usize,
        start_node: u32,
    ) -> (Vec<u32>, Vec<Candidate>) {
        let mut visited = HashSet::new();
        // Note: We use retset Vec instead of BinaryHeap for simpler control over L closest

        // Use a max-heap for the working queue to easily pop the worst candidate
        // Wait, standard beam search keeps L closest.
        // Let's implement standard "iterate until convergence" greedy search.

        // Results set (L closest found so far) - sorted vector or binary heap
        // We'll use a vector and sort it, for simplicity in this proto.
        let mut retset: Vec<Candidate> = Vec::with_capacity(l_size + 1);

        let start_dist = self.dist(query, self.get_vector(start_node));
        retset.push(Candidate {
            id: start_node,
            dist: start_dist,
        });
        visited.insert(start_node);

        let mut current_idx = 0;

        while current_idx < retset.len() {
            // Find the closest unvisited node in retset
            // (In optimized impl, we iterate sorted retset)
            retset.sort_by(|a, b| a.dist.total_cmp(&b.dist));

            if current_idx >= retset.len() {
                break;
            }

            let current = retset[current_idx];
            current_idx += 1;

            // If closest unvisited is farther than our worst candidate (and list is full), stop?
            // Vamana doesn't strictly stop, it explores all neighbors.

            for &neighbor in &self.adj[current.id as usize] {
                if visited.contains(&neighbor) {
                    continue;
                }
                visited.insert(neighbor);

                let dist = self.dist(query, self.get_vector(neighbor));

                // Add to retset
                retset.push(Candidate { id: neighbor, dist });
            }

            // Keep only top L
            retset.sort_by(|a, b| a.dist.total_cmp(&b.dist));
            if retset.len() > l_size {
                retset.truncate(l_size);
            }
        }

        let ids: Vec<u32> = retset.iter().map(|c| c.id).collect();
        (ids, retset)
    }

    /// Search for k nearest neighbors.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "index must be built before search".into(),
            ));
        }

        if query.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.dimension,
            });
        }

        let ef = ef_search.max(k);
        let (_, candidates) = self.greedy_search(query, ef, self.start_node);

        // Return top k, mapping internal indices back to external doc_ids
        let result = candidates
            .into_iter()
            .take(k)
            .filter_map(|c| {
                let doc_id = self.doc_ids.get(c.id as usize).copied()?;
                Some((doc_id, c.dist))
            })
            .collect();

        Ok(result)
    }

    fn get_vector(&self, idx: u32) -> &[f32] {
        let start = idx as usize * self.dimension;
        &self.vectors[start..start + self.dimension]
    }

    // Euclidean distance (squared), using SIMD when available.
    fn dist(&self, a: &[f32], b: &[f32]) -> f32 {
        crate::simd::l2_distance_squared(a, b)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::error::RetrieveError;

    #[test]
    fn test_create_index() {
        let index = DiskANNIndex::new(4, DiskANNParams::default());
        assert!(index.is_ok());
        let index = index.unwrap();
        assert_eq!(index.dimension(), 4);
        assert_eq!(index.num_vectors(), 0);
    }

    #[test]
    fn test_add_and_search() {
        let params = DiskANNParams {
            m: 4,
            ef_construction: 20,
            alpha: 1.2,
            ef_search: 20,
        };
        let mut index = DiskANNIndex::new(4, params).unwrap();

        // Add 10 vectors
        for i in 0..10u32 {
            let v = vec![i as f32, (i as f32) * 0.5, 1.0, 0.0];
            index.add(i, v).unwrap();
        }

        index.build().unwrap();

        let query = vec![0.0, 0.0, 1.0, 0.0];
        let results = index.search(&query, 3, 20).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 3);
        // The closest vector should be doc_id 0 (vector [0, 0, 1, 0])
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_zero_dimension_error() {
        let result = DiskANNIndex::new(0, DiskANNParams::default());
        match result {
            Err(RetrieveError::InvalidParameter(_)) => {}
            Err(other) => panic!("Expected InvalidParameter, got {:?}", other),
            Ok(_) => panic!("Expected error for dimension 0"),
        }
    }
}
