//! IVF-AVQ search implementation.

use crate::ivf_avq::partitioning::KMeans;
use crate::ivf_avq::quantization::AnisotropicQuantizer;
use crate::ivf_avq::reranking;
use crate::RetrieveError;
#[cfg(feature = "persistence")]
use durability::mmap::{AccessPattern, MappedFile};
use serde::{Deserialize, Serialize};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

const IVFAVQ_FORMAT_VERSION: u32 = 1;
const IVFAVQ_PARTITION_MAGIC: &[u8; 8] = b"IVFAVQPT";

/// Anisotropic Vector Quantization with k-means Partitioning index.
#[derive(Debug)]
pub struct IVFAVQIndex {
    /// Full vectors (for re-ranking)
    pub(crate) vectors: Vec<f32>,
    pub(crate) dimension: usize,
    pub(crate) num_vectors: usize,
    /// Maps insertion index → caller-provided doc_id.
    doc_ids: Vec<u32>,
    params: IVFAVQParams,
    built: bool,

    // Partitioning
    partitions: Vec<Partition>,
    pub(crate) partition_centroids: Vec<Vec<f32>>,

    // Quantization
    quantizer: Option<AnisotropicQuantizer>,
}

/// File-backed searcher for a saved IVF-AVQ index.
pub struct IVFAVQFileSearcher {
    dimension: usize,
    num_vectors: usize,
    doc_ids: Vec<u32>,
    params: IVFAVQParams,
    partition_locations: Vec<PartitionLocation>,
    partition_centroids: Vec<Vec<f32>>,
    quantizer: AnisotropicQuantizer,
    partitions_storage: IVFAVQByteStorage,
    raw_vectors_storage: IVFAVQByteStorage,
    id_buf: Vec<u32>,
    id_byte_buf: Vec<u8>,
    code_buf: Vec<u8>,
    raw_byte_buf: Vec<u8>,
    vector_buf: Vec<f32>,
}

/// Per-query storage diagnostics for [`IVFAVQFileSearcher`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IVFAVQFileSearchDiagnostics {
    /// Number of IVF partitions probed.
    pub probed_lists: usize,
    /// Number of vectors scanned across probed partitions.
    pub scanned_vectors: usize,
    /// Number of logical partition payload reads.
    pub partition_reads: usize,
    /// Number of partition payload bytes read, including IDs and codes.
    pub partition_bytes: usize,
    /// Number of logical code payload reads.
    pub code_reads: usize,
    /// Number of quantized code bytes read.
    pub code_bytes: usize,
    /// Number of candidates retained for exact scoring.
    pub retained_candidates: usize,
    /// Number of full-precision vectors read for exact scoring.
    pub raw_vector_reads: usize,
    /// Number of raw vector bytes read for exact scoring.
    pub raw_vector_bytes: usize,
}

/// Parameters for IVF-AVQ index construction and search.
#[derive(Clone, Debug)]
pub struct IVFAVQParams {
    /// Number of k-means partitions for coarse quantization.
    pub num_partitions: usize,
    /// Number of partitions to probe during search (higher = better recall, slower).
    pub nprobe: usize,
    /// Number of candidates to re-rank with exact distances.
    pub num_reorder: usize,
    /// Number of PQ subspaces (M).
    pub num_codebooks: usize,
    /// Number of centroids per codebook (typically 256 for 8-bit codes).
    pub codebook_size: usize,
    /// Random seed for deterministic training (k-means + PQ codebooks).
    pub seed: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PersistedIVFAVQParams {
    num_partitions: usize,
    nprobe: usize,
    num_reorder: usize,
    num_codebooks: usize,
    codebook_size: usize,
    seed: u64,
}

impl PersistedIVFAVQParams {
    fn from_runtime(params: &IVFAVQParams, actual_codebook_size: usize) -> Self {
        Self {
            num_partitions: params.num_partitions,
            nprobe: params.nprobe,
            num_reorder: params.num_reorder,
            num_codebooks: params.num_codebooks,
            codebook_size: actual_codebook_size,
            seed: params.seed,
        }
    }

    fn into_params(self) -> IVFAVQParams {
        IVFAVQParams {
            num_partitions: self.num_partitions,
            nprobe: self.nprobe,
            num_reorder: self.num_reorder,
            num_codebooks: self.num_codebooks,
            codebook_size: self.codebook_size,
            seed: self.seed,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct IVFAVQManifest {
    version: u32,
    dimension: usize,
    num_vectors: usize,
    params: PersistedIVFAVQParams,
}

impl Default for IVFAVQParams {
    fn default() -> Self {
        Self {
            num_partitions: 256,
            nprobe: 20,
            num_reorder: 100,
            num_codebooks: 16,
            codebook_size: 256,
            seed: 42,
        }
    }
}

/// Partition containing quantized codes and indices.
#[derive(Clone, Debug)]
struct Partition {
    /// Original indices of vectors in this partition
    vector_indices: Vec<u32>,
    /// Quantized codes for these vectors (flat layout)
    /// Layout: [vector_0_codes, vector_1_codes, ...]
    codes: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
struct PartitionLocation {
    ids_len: usize,
    ids_offset: u64,
    codes_len: usize,
    codes_offset: u64,
}

enum IVFAVQByteStorage {
    File(std::fs::File),
    #[cfg(feature = "persistence")]
    Mmap(Box<MappedFile>),
}

impl IVFAVQIndex {
    /// Create a new IVF-AVQ index with the given vector dimension and parameters.
    pub fn new(dimension: usize, params: IVFAVQParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be > 0".into(),
            ));
        }
        Ok(Self {
            vectors: Vec::new(),
            dimension,
            num_vectors: 0,
            doc_ids: Vec::new(),
            params,
            built: false,
            partitions: Vec::new(),
            partition_centroids: Vec::new(),
            quantizer: None,
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
    /// - `doc_id` is preserved and returned in search results.
    pub fn add_slice(&mut self, doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
        if self.built {
            return Err(RetrieveError::InvalidParameter(
                "index already built".into(),
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
        Ok(())
    }

    /// Build the index (partitioning + quantization). Required before search.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        // 1. Train Partitioning (Coarse Quantizer)
        let mut kmeans =
            KMeans::new(self.dimension, self.params.num_partitions)?.with_seed(self.params.seed);
        kmeans.fit(&self.vectors, self.num_vectors)?;
        self.partition_centroids = kmeans.centroids().to_vec();

        // 2. Assign vectors to partitions and compute residuals
        let assignments = kmeans.assign_clusters(&self.vectors, self.num_vectors);
        let mut residuals = Vec::with_capacity(self.vectors.len());

        // Initialize partitions
        self.partitions = vec![
            Partition {
                vector_indices: Vec::new(),
                codes: Vec::new()
            };
            self.params.num_partitions
        ];

        for (i, &partition_idx) in assignments.iter().enumerate() {
            self.partitions[partition_idx].vector_indices.push(i as u32);

            let vec = self.get_vector(i);
            let centroid = &self.partition_centroids[partition_idx];

            // Compute residual: r = x - c
            for (x, c) in vec.iter().zip(centroid.iter()) {
                residuals.push(x - c);
            }
        }

        // 3. Train Quantizer on Residuals
        let mut quantizer = AnisotropicQuantizer::new(
            self.dimension,
            self.params.num_codebooks,
            self.params.codebook_size,
            self.params.seed,
        )?;
        quantizer.fit_residuals(&residuals, self.num_vectors)?;

        // 4. Quantize Residuals and Store
        // We re-compute residuals on the fly to keep code simple (or use the flat residuals vector)
        // But the flat residuals vector is ordered by input ID, not partition.
        // Let's iterate partitions to populate codes.

        for p_idx in 0..self.params.num_partitions {
            let centroid = &self.partition_centroids[p_idx];
            // Clone vector indices to avoid borrow conflict with self.get_vector()
            let vec_indices: Vec<u32> = self.partitions[p_idx].vector_indices.clone();

            let mut all_codes = Vec::with_capacity(vec_indices.len() * self.params.num_codebooks);
            for vec_idx in vec_indices {
                let vec = self.get_vector(vec_idx as usize);

                // Recompute residual
                let residual: Vec<f32> = vec
                    .iter()
                    .zip(centroid.iter())
                    .map(|(x, c)| x - c)
                    .collect();

                let codes = quantizer.quantize(&residual);
                all_codes.extend(codes);
            }

            self.partitions[p_idx].codes = all_codes;
        }

        self.quantizer = Some(quantizer);
        self.built = true;
        Ok(())
    }

    /// Save a built IVF-AVQ index to a directory.
    pub fn save_to_dir(&self, output_dir: impl AsRef<Path>) -> Result<(), RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot save unbuilt IVF-AVQ index".into(),
            ));
        }
        let quantizer = self
            .quantizer
            .as_ref()
            .ok_or_else(|| RetrieveError::InvalidParameter("missing IVF-AVQ quantizer".into()))?;
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;

        let manifest = IVFAVQManifest {
            version: IVFAVQ_FORMAT_VERSION,
            dimension: self.dimension,
            num_vectors: self.num_vectors,
            params: PersistedIVFAVQParams::from_runtime(&self.params, quantizer.codebook_size()),
        };

        write_json_atomic(&output_dir.join("manifest.json"), &manifest)?;
        write_f32_atomic(&output_dir.join("raw_vectors.bin"), &self.vectors)?;
        write_u32_atomic(&output_dir.join("doc_ids.bin"), &self.doc_ids)?;
        write_centroids_atomic(&output_dir.join("centroids.bin"), &self.partition_centroids)?;
        write_f32_atomic(&output_dir.join("codebooks.bin"), quantizer.codebooks())?;
        write_partitions_atomic(&output_dir.join("partitions.bin"), &self.partitions)?;

        Ok(())
    }

    /// Load an IVF-AVQ index saved by [`Self::save_to_dir`].
    pub fn load_from_dir(input_dir: impl AsRef<Path>) -> Result<Self, RetrieveError> {
        let input_dir = input_dir.as_ref();
        let manifest: IVFAVQManifest = read_json(&input_dir.join("manifest.json"))?;
        validate_manifest(&manifest)?;

        let params = manifest.params.into_params();
        let mut index = Self::new(manifest.dimension, params)?;
        index.num_vectors = manifest.num_vectors;
        let vector_len = checked_len(
            manifest.num_vectors,
            manifest.dimension,
            "IVF-AVQ vector length overflow",
        )?;
        index.vectors = read_f32_exact(&input_dir.join("raw_vectors.bin"), vector_len)?;
        index.doc_ids = read_u32_exact(&input_dir.join("doc_ids.bin"), manifest.num_vectors)?;
        index.partition_centroids = read_centroids(
            &input_dir.join("centroids.bin"),
            index.params.num_partitions,
            manifest.dimension,
        )?;
        let codebooks_len = index
            .params
            .num_codebooks
            .checked_mul(index.params.codebook_size)
            .and_then(|len| len.checked_mul(manifest.dimension / index.params.num_codebooks))
            .ok_or_else(|| RetrieveError::FormatError("AVQ codebook length overflow".into()))?;
        let codebooks = read_f32_exact(&input_dir.join("codebooks.bin"), codebooks_len)?;
        index.quantizer = Some(AnisotropicQuantizer::from_codebooks(
            manifest.dimension,
            index.params.num_codebooks,
            index.params.codebook_size,
            index.params.seed,
            codebooks,
        )?);
        index.partitions = read_partitions(
            &input_dir.join("partitions.bin"),
            index.params.num_partitions,
            manifest.num_vectors,
            index.params.num_codebooks,
        )?;
        index.built = true;
        Ok(index)
    }

    /// Search for the k nearest neighbors of the query vector.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
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
        if k == 0 {
            return Ok(Vec::new());
        }

        let quantizer = self
            .quantizer
            .as_ref()
            .ok_or(RetrieveError::InvalidParameter(
                "quantizer not initialized".into(),
            ))?;

        // 1. Find top partitions
        // Compute dot product with all centroids
        let mut partition_scores: Vec<(usize, f32)> = self
            .partition_centroids
            .iter()
            .enumerate()
            .map(|(idx, c)| (idx, crate::simd::dot(query, c)))
            .collect();

        // Sort by score (descending for MIPS)
        partition_scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // Select top partitions
        let num_probe = self.params.nprobe.min(self.params.num_partitions);
        let lut = quantizer.build_lut_flat(query); // Precompute flat LUT for residuals
        let codebook_size = quantizer.codebook_size();

        let mut candidates = Vec::new();

        // 2. Search within partitions
        for (p_idx, center_score) in partition_scores.iter().take(num_probe) {
            let partition = &self.partitions[*p_idx];
            let num_vectors = partition.vector_indices.len();
            let m = self.params.num_codebooks;

            for i in 0..num_vectors {
                // Reconstruct approximate score:
                // <q, x> ≈ <q, c> + <q, r>
                // <q, r> approximated by flat LUT (single array, no pointer chasing)

                let mut residual_score = 0.0;
                let code_start = i * m;
                let codes = &partition.codes[code_start..code_start + m];

                for (subspace_idx, &code) in codes.iter().enumerate() {
                    residual_score += lut[subspace_idx * codebook_size + code as usize];
                }

                let approx_score = center_score + residual_score;
                candidates.push((partition.vector_indices[i], approx_score));
            }
        }

        // 3. Re-rank top candidates (partial sort: O(n) instead of O(n log n))
        let num_reorder = self.params.num_reorder.max(k);
        let top_candidates: Vec<(u32, f32)> = if candidates.len() > num_reorder {
            // Partition so the top num_reorder are in candidates[..num_reorder]
            candidates.select_nth_unstable_by(num_reorder - 1, |a, b| b.1.total_cmp(&a.1));
            candidates.truncate(num_reorder);
            candidates
        } else {
            candidates
        };

        // Exact re-ranking (uses vector_idx to look up in self.vectors)
        let reranked = reranking::rerank(query, &top_candidates, &self.vectors, self.dimension, k);

        // Translate insertion-order indices to caller-provided doc_ids
        let results = reranked
            .into_iter()
            .map(|(vector_idx, dist)| (self.doc_ids[vector_idx as usize], dist))
            .collect();
        Ok(results)
    }

    /// Set the number of partitions probed during search.
    pub fn set_nprobe(&mut self, nprobe: usize) {
        self.params.nprobe = nprobe;
    }

    /// Set the number of candidates re-ranked with exact distances.
    pub fn set_num_reorder(&mut self, num_reorder: usize) {
        self.params.num_reorder = num_reorder;
    }

    /// Estimated heap memory used by this index.
    pub fn memory_usage(&self) -> crate::memory::MemoryReport {
        let vectors_bytes = self.vectors.capacity() * std::mem::size_of::<f32>();
        let quantized_bytes = self
            .quantizer
            .as_ref()
            .map(|quantizer| std::mem::size_of_val(quantizer.codebooks()))
            .unwrap_or(0)
            + self
                .partitions
                .iter()
                .map(|partition| partition.codes.capacity())
                .sum::<usize>();
        let metadata_bytes = self.doc_ids.capacity() * std::mem::size_of::<u32>()
            + self.partitions.capacity() * std::mem::size_of::<Partition>()
            + self
                .partitions
                .iter()
                .map(|partition| partition.vector_indices.capacity() * std::mem::size_of::<u32>())
                .sum::<usize>()
            + self.partition_centroids.capacity() * std::mem::size_of::<Vec<f32>>()
            + self
                .partition_centroids
                .iter()
                .map(|centroid| centroid.capacity() * std::mem::size_of::<f32>())
                .sum::<usize>();

        crate::memory::MemoryReport {
            vectors_bytes,
            graph_bytes: 0,
            quantized_bytes,
            metadata_bytes,
        }
    }

    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        &self.vectors[start..start + self.dimension]
    }
}

impl IVFAVQFileSearcher {
    /// Open an IVF-AVQ snapshot for direct file-backed search.
    pub fn open(input_dir: impl AsRef<Path>) -> Result<Self, RetrieveError> {
        Self::open_with_storage(input_dir.as_ref(), false)
    }

    /// Open an IVF-AVQ snapshot with read-only mmap-backed payloads.
    #[cfg(feature = "persistence")]
    pub fn open_mmap(input_dir: impl AsRef<Path>) -> Result<Self, RetrieveError> {
        Self::open_with_storage(input_dir.as_ref(), true)
    }

    fn open_with_storage(input_dir: &Path, mmap: bool) -> Result<Self, RetrieveError> {
        let manifest: IVFAVQManifest = read_json(&input_dir.join("manifest.json"))?;
        validate_manifest(&manifest)?;

        let params = manifest.params.clone().into_params();
        let doc_ids = read_u32_exact(&input_dir.join("doc_ids.bin"), manifest.num_vectors)?;
        let partition_centroids = read_centroids(
            &input_dir.join("centroids.bin"),
            params.num_partitions,
            manifest.dimension,
        )?;
        let codebooks_len = params
            .num_codebooks
            .checked_mul(params.codebook_size)
            .and_then(|len| len.checked_mul(manifest.dimension / params.num_codebooks))
            .ok_or_else(|| RetrieveError::FormatError("AVQ codebook length overflow".into()))?;
        let codebooks = read_f32_exact(&input_dir.join("codebooks.bin"), codebooks_len)?;
        let quantizer = AnisotropicQuantizer::from_codebooks(
            manifest.dimension,
            params.num_codebooks,
            params.codebook_size,
            params.seed,
            codebooks,
        )?;
        let partitions_path = input_dir.join("partitions.bin");
        let partition_locations = scan_partition_locations(
            &partitions_path,
            params.num_partitions,
            manifest.num_vectors,
            params.num_codebooks,
        )?;
        let expected_partition_bytes = file_len_usize(&partitions_path)?;
        let raw_vector_bytes = checked_byte_len(
            manifest.dimension,
            std::mem::size_of::<f32>(),
            "IVF-AVQ raw vector byte length overflow",
        )?;
        let raw_vectors_path = input_dir.join("raw_vectors.bin");
        let expected_raw_bytes = checked_byte_len(
            manifest.num_vectors,
            raw_vector_bytes,
            "IVF-AVQ raw vector file byte length overflow",
        )?;
        let partitions_storage =
            open_byte_storage(&partitions_path, expected_partition_bytes, mmap)?;
        let raw_vectors_storage = open_byte_storage(&raw_vectors_path, expected_raw_bytes, mmap)?;

        Ok(Self {
            dimension: manifest.dimension,
            num_vectors: manifest.num_vectors,
            doc_ids,
            params,
            partition_locations,
            partition_centroids,
            quantizer,
            partitions_storage,
            raw_vectors_storage,
            id_buf: Vec::new(),
            id_byte_buf: Vec::new(),
            code_buf: Vec::new(),
            raw_byte_buf: vec![0; raw_vector_bytes],
            vector_buf: vec![0.0; manifest.dimension],
        })
    }

    /// Search the saved index using file reads for partitions and exact rerank vectors.
    pub fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.search_with_diagnostics(query, k)
            .map(|(results, _diagnostics)| results)
    }

    /// Search the saved index and return storage diagnostics for the query.
    pub fn search_with_diagnostics(
        &mut self,
        query: &[f32],
        k: usize,
    ) -> Result<(Vec<(u32, f32)>, IVFAVQFileSearchDiagnostics), RetrieveError> {
        if query.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.dimension,
            });
        }
        if k == 0 {
            return Ok((Vec::new(), IVFAVQFileSearchDiagnostics::default()));
        }

        let mut partition_scores: Vec<(usize, f32)> = self
            .partition_centroids
            .iter()
            .enumerate()
            .map(|(idx, c)| (idx, crate::simd::dot(query, c)))
            .collect();
        partition_scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        let num_probe = self.params.nprobe.min(self.params.num_partitions);
        let mut diagnostics = IVFAVQFileSearchDiagnostics {
            probed_lists: num_probe,
            ..IVFAVQFileSearchDiagnostics::default()
        };
        let lut = self.quantizer.build_lut_flat(query);
        let codebook_size = self.quantizer.codebook_size();
        let mut candidates = Vec::new();

        for (p_idx, center_score) in partition_scores.iter().take(num_probe) {
            let location = self.partition_locations[*p_idx];
            diagnostics.scanned_vectors += location.ids_len;
            diagnostics.partition_reads += 1;
            let id_bytes_len = checked_byte_len(
                location.ids_len,
                std::mem::size_of::<u32>(),
                "IVF-AVQ diagnostic partition id byte length overflow",
            )?;
            diagnostics.partition_bytes += id_bytes_len + location.codes_len;
            if location.codes_len > 0 {
                diagnostics.code_reads += 1;
            }
            diagnostics.code_bytes += location.codes_len;
            self.read_partition(location)?;
            let m = self.params.num_codebooks;

            for (i, &vector_idx) in self.id_buf.iter().enumerate() {
                let code_start = i * m;
                let codes = &self.code_buf[code_start..code_start + m];
                let mut residual_score = 0.0;
                for (subspace_idx, &code) in codes.iter().enumerate() {
                    residual_score += lut[subspace_idx * codebook_size + code as usize];
                }
                candidates.push((vector_idx, center_score + residual_score));
            }
        }

        let num_reorder = self.params.num_reorder.max(k);
        if candidates.len() > num_reorder {
            candidates.select_nth_unstable_by(num_reorder - 1, |a, b| b.1.total_cmp(&a.1));
            candidates.truncate(num_reorder);
        }

        diagnostics.retained_candidates = candidates.len();
        diagnostics.raw_vector_reads = candidates.len();
        diagnostics.raw_vector_bytes = checked_byte_len(
            candidates.len(),
            self.raw_byte_buf.len(),
            "IVF-AVQ diagnostic raw-vector byte count overflow",
        )?;
        let mut reranked = Vec::with_capacity(candidates.len().min(k));
        for (vector_idx, _) in candidates {
            let exact_dist = self.read_exact_distance(query, vector_idx)?;
            reranked.push((vector_idx, exact_dist));
        }
        reranked.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        reranked.truncate(k);

        let results = reranked
            .into_iter()
            .map(|(vector_idx, dist)| {
                let doc_id = *self.doc_ids.get(vector_idx as usize).ok_or_else(|| {
                    RetrieveError::FormatError(format!(
                        "IVF-AVQ vector id {} exceeds doc id count {}",
                        vector_idx,
                        self.doc_ids.len()
                    ))
                })?;
                Ok((doc_id, dist))
            })
            .collect::<Result<Vec<_>, RetrieveError>>()?;
        Ok((results, diagnostics))
    }

    /// Set the number of partitions probed during search.
    pub fn set_nprobe(&mut self, nprobe: usize) {
        self.params.nprobe = nprobe;
    }

    /// Set the number of candidates re-ranked with exact distances.
    pub fn set_num_reorder(&mut self, num_reorder: usize) {
        self.params.num_reorder = num_reorder;
    }

    fn read_partition(&mut self, location: PartitionLocation) -> Result<(), RetrieveError> {
        let id_bytes_len = checked_byte_len(
            location.ids_len,
            std::mem::size_of::<u32>(),
            "IVF-AVQ partition id byte length overflow",
        )?;
        self.id_byte_buf.resize(id_bytes_len, 0);
        read_bytes_from_storage(
            &mut self.partitions_storage,
            location.ids_offset,
            &mut self.id_byte_buf,
        )?;
        self.id_buf.resize(location.ids_len, 0);
        for (slot, chunk) in self
            .id_buf
            .iter_mut()
            .zip(self.id_byte_buf.chunks_exact(std::mem::size_of::<u32>()))
        {
            *slot = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if *slot as usize >= self.num_vectors {
                return Err(RetrieveError::FormatError(format!(
                    "IVF-AVQ partition id {} exceeds vector count {}",
                    *slot, self.num_vectors
                )));
            }
        }

        self.code_buf.resize(location.codes_len, 0);
        read_bytes_from_storage(
            &mut self.partitions_storage,
            location.codes_offset,
            &mut self.code_buf,
        )?;
        for &code in &self.code_buf {
            if code as usize >= self.params.codebook_size {
                return Err(RetrieveError::FormatError(format!(
                    "IVF-AVQ partition code {} exceeds codebook size {}",
                    code, self.params.codebook_size
                )));
            }
        }
        Ok(())
    }

    fn read_exact_distance(
        &mut self,
        query: &[f32],
        vector_idx: u32,
    ) -> Result<f32, RetrieveError> {
        if vector_idx as usize >= self.num_vectors {
            return Err(RetrieveError::FormatError(format!(
                "IVF-AVQ vector id {} exceeds vector count {}",
                vector_idx, self.num_vectors
            )));
        }
        let vector_offset = checked_byte_len(
            vector_idx as usize,
            self.raw_byte_buf.len(),
            "IVF-AVQ raw vector offset overflow",
        )?;
        let vector_offset = u64::try_from(vector_offset)
            .map_err(|_| RetrieveError::FormatError("IVF-AVQ raw vector offset overflow".into()))?;
        read_bytes_from_storage(
            &mut self.raw_vectors_storage,
            vector_offset,
            &mut self.raw_byte_buf,
        )?;
        for (slot, chunk) in self
            .vector_buf
            .iter_mut()
            .zip(self.raw_byte_buf.chunks_exact(4))
        {
            *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Ok(crate::distance::cosine_distance_normalized(
            query,
            &self.vector_buf,
        ))
    }
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), RetrieveError> {
    write_atomic(path, |writer| {
        serde_json::to_writer_pretty(writer, value)
            .map_err(|e| std::io::Error::other(e.to_string()))
    })
}

fn write_f32_atomic(path: &Path, values: &[f32]) -> Result<(), RetrieveError> {
    write_atomic(path, |writer| {
        for value in values {
            writer.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    })
}

fn write_u32_atomic(path: &Path, values: &[u32]) -> Result<(), RetrieveError> {
    write_atomic(path, |writer| {
        for value in values {
            writer.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    })
}

fn write_centroids_atomic(path: &Path, centroids: &[Vec<f32>]) -> Result<(), RetrieveError> {
    write_atomic(path, |writer| {
        for centroid in centroids {
            for value in centroid {
                writer.write_all(&value.to_le_bytes())?;
            }
        }
        Ok(())
    })
}

fn write_partitions_atomic(path: &Path, partitions: &[Partition]) -> Result<(), RetrieveError> {
    write_atomic(path, |writer| {
        writer.write_all(IVFAVQ_PARTITION_MAGIC)?;
        writer.write_all(&(partitions.len() as u64).to_le_bytes())?;
        for partition in partitions {
            writer.write_all(&(partition.vector_indices.len() as u64).to_le_bytes())?;
            for id in &partition.vector_indices {
                writer.write_all(&id.to_le_bytes())?;
            }
            writer.write_all(&(partition.codes.len() as u64).to_le_bytes())?;
            writer.write_all(&partition.codes)?;
        }
        Ok(())
    })
}

fn write_atomic(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<std::fs::File>) -> std::io::Result<()>,
) -> Result<(), RetrieveError> {
    let tmp_path = path.with_extension("tmp");
    {
        let file = std::fs::File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);
        write(&mut writer)?;
        writer.flush()?;
    }
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, RetrieveError> {
    let file = std::fs::File::open(path)?;
    serde_json::from_reader(BufReader::new(file))
        .map_err(|e| RetrieveError::FormatError(e.to_string()))
}

fn read_f32_exact(path: &Path, expected_len: usize) -> Result<Vec<f32>, RetrieveError> {
    let bytes = std::fs::read(path)?;
    let expected_bytes = expected_len
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| RetrieveError::FormatError("f32 byte length overflow".into()))?;
    if bytes.len() != expected_bytes {
        return Err(RetrieveError::FormatError(format!(
            "{} size mismatch: expected {} bytes, got {}",
            path.display(),
            expected_bytes,
            bytes.len()
        )));
    }
    let mut values = Vec::with_capacity(expected_len);
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

fn read_u32_exact(path: &Path, expected_len: usize) -> Result<Vec<u32>, RetrieveError> {
    let bytes = std::fs::read(path)?;
    let expected_bytes = expected_len
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| RetrieveError::FormatError("u32 byte length overflow".into()))?;
    if bytes.len() != expected_bytes {
        return Err(RetrieveError::FormatError(format!(
            "{} size mismatch: expected {} bytes, got {}",
            path.display(),
            expected_bytes,
            bytes.len()
        )));
    }
    let mut values = Vec::with_capacity(expected_len);
    for chunk in bytes.chunks_exact(4) {
        values.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

fn read_centroids(
    path: &Path,
    num_partitions: usize,
    dimension: usize,
) -> Result<Vec<Vec<f32>>, RetrieveError> {
    let flat_len = checked_len(
        num_partitions,
        dimension,
        "IVF-AVQ centroid length overflow",
    )?;
    let flat = read_f32_exact(path, flat_len)?;
    Ok(flat
        .chunks_exact(dimension)
        .map(|chunk| chunk.to_vec())
        .collect())
}

fn validate_manifest(manifest: &IVFAVQManifest) -> Result<(), RetrieveError> {
    if manifest.version != IVFAVQ_FORMAT_VERSION {
        return Err(RetrieveError::FormatError(format!(
            "unsupported IVF-AVQ format version {}",
            manifest.version
        )));
    }
    if manifest.dimension == 0 {
        return Err(RetrieveError::FormatError(
            "IVF-AVQ manifest has zero dimension".into(),
        ));
    }
    if manifest.num_vectors == 0 {
        return Err(RetrieveError::FormatError(
            "IVF-AVQ manifest has zero vectors".into(),
        ));
    }
    if manifest.params.num_partitions == 0 {
        return Err(RetrieveError::FormatError(
            "IVF-AVQ manifest has zero partitions".into(),
        ));
    }
    if manifest.params.nprobe == 0 {
        return Err(RetrieveError::FormatError(
            "IVF-AVQ manifest has zero nprobe".into(),
        ));
    }
    if manifest.params.num_codebooks == 0 {
        return Err(RetrieveError::FormatError(
            "IVF-AVQ manifest has zero codebooks".into(),
        ));
    }
    if !manifest
        .dimension
        .is_multiple_of(manifest.params.num_codebooks)
    {
        return Err(RetrieveError::FormatError(format!(
            "IVF-AVQ dimension {} is not divisible by {} codebooks",
            manifest.dimension, manifest.params.num_codebooks
        )));
    }
    if manifest.params.codebook_size == 0 || manifest.params.codebook_size > 256 {
        return Err(RetrieveError::FormatError(format!(
            "IVF-AVQ codebook size {} is outside 1..=256",
            manifest.params.codebook_size
        )));
    }
    Ok(())
}

fn checked_len(lhs: usize, rhs: usize, message: &str) -> Result<usize, RetrieveError> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| RetrieveError::FormatError(message.into()))
}

fn checked_byte_len(lhs: usize, rhs: usize, message: &str) -> Result<usize, RetrieveError> {
    checked_len(lhs, rhs, message)
}

fn validate_file_size(path: &Path, expected_len: usize) -> Result<(), RetrieveError> {
    let actual_len = std::fs::metadata(path)?.len();
    let expected_len = u64::try_from(expected_len)
        .map_err(|_| RetrieveError::FormatError("IVF-AVQ expected file size overflow".into()))?;
    if actual_len != expected_len {
        return Err(RetrieveError::FormatError(format!(
            "{} size mismatch: expected {} bytes, got {}",
            path.display(),
            expected_len,
            actual_len
        )));
    }
    Ok(())
}

fn file_len_usize(path: &Path) -> Result<usize, RetrieveError> {
    let len = std::fs::metadata(path)?.len();
    usize::try_from(len).map_err(|_| {
        RetrieveError::FormatError(format!(
            "{} size {} exceeds addressable memory",
            path.display(),
            len
        ))
    })
}

fn open_byte_storage(
    path: &Path,
    expected_len: usize,
    mmap: bool,
) -> Result<IVFAVQByteStorage, RetrieveError> {
    validate_file_size(path, expected_len)?;

    #[cfg(feature = "persistence")]
    if mmap {
        let mapped = MappedFile::open(path, AccessPattern::Random).map_err(|e| {
            RetrieveError::Io(std::sync::Arc::new(std::io::Error::other(format!(
                "failed to mmap {}: {e}",
                path.display()
            ))))
        })?;
        if mapped.as_slice().len() != expected_len {
            return Err(RetrieveError::FormatError(format!(
                "{} mmap size mismatch: expected {} bytes, got {}",
                path.display(),
                expected_len,
                mapped.as_slice().len()
            )));
        }
        return Ok(IVFAVQByteStorage::Mmap(Box::new(mapped)));
    }

    let _ = mmap;
    Ok(IVFAVQByteStorage::File(std::fs::File::open(path)?))
}

fn read_bytes_from_storage(
    storage: &mut IVFAVQByteStorage,
    offset: u64,
    out: &mut [u8],
) -> Result<(), RetrieveError> {
    #[cfg(feature = "persistence")]
    let start = usize::try_from(offset)
        .map_err(|_| RetrieveError::FormatError("IVF-AVQ byte storage offset overflow".into()))?;
    #[cfg(feature = "persistence")]
    let end = start
        .checked_add(out.len())
        .ok_or_else(|| RetrieveError::FormatError("IVF-AVQ byte storage offset overflow".into()))?;

    match storage {
        IVFAVQByteStorage::File(file) => {
            crate::file_io::read_exact_at(file, offset, out)?;
        }
        #[cfg(feature = "persistence")]
        IVFAVQByteStorage::Mmap(mapped) => {
            let bytes = mapped.as_slice();
            if end > bytes.len() {
                return Err(RetrieveError::FormatError(format!(
                    "IVF-AVQ storage read out of bounds: end {} > len {}",
                    end,
                    bytes.len()
                )));
            }
            out.copy_from_slice(&bytes[start..end]);
        }
    }
    Ok(())
}

fn scan_partition_locations(
    path: &Path,
    expected_partitions: usize,
    num_vectors: usize,
    num_codebooks: usize,
) -> Result<Vec<PartitionLocation>, RetrieveError> {
    let mut reader = BufReader::new(std::fs::File::open(path)?);
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != IVFAVQ_PARTITION_MAGIC {
        return Err(RetrieveError::FormatError(
            "invalid IVF-AVQ partition file magic".into(),
        ));
    }
    let partition_count = read_u64(&mut reader)? as usize;
    if partition_count != expected_partitions {
        return Err(RetrieveError::FormatError(format!(
            "partition count mismatch: expected {}, got {}",
            expected_partitions, partition_count
        )));
    }

    let mut locations = Vec::with_capacity(partition_count);
    for _ in 0..partition_count {
        let ids_len = read_u64(&mut reader)? as usize;
        if ids_len > num_vectors {
            return Err(RetrieveError::FormatError(format!(
                "partition length {} exceeds vector count {}",
                ids_len, num_vectors
            )));
        }
        let ids_offset = reader.stream_position()?;
        for _ in 0..ids_len {
            let id = read_u32(&mut reader)?;
            if id as usize >= num_vectors {
                return Err(RetrieveError::FormatError(format!(
                    "partition id {} exceeds vector count {}",
                    id, num_vectors
                )));
            }
        }

        let codes_len = read_u64(&mut reader)? as usize;
        let expected_codes = ids_len
            .checked_mul(num_codebooks)
            .ok_or_else(|| RetrieveError::FormatError("partition code length overflow".into()))?;
        if codes_len != expected_codes {
            return Err(RetrieveError::FormatError(format!(
                "partition code length mismatch: expected {}, got {}",
                expected_codes, codes_len
            )));
        }
        let codes_offset = reader.stream_position()?;
        seek_forward(&mut reader, codes_len)?;

        locations.push(PartitionLocation {
            ids_len,
            ids_offset,
            codes_len,
            codes_offset,
        });
    }

    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        return Err(RetrieveError::FormatError(
            "trailing bytes in IVF-AVQ partition file".into(),
        ));
    }

    Ok(locations)
}

fn seek_forward(reader: &mut BufReader<std::fs::File>, bytes: usize) -> Result<(), RetrieveError> {
    let current = reader.stream_position()?;
    let bytes = u64::try_from(bytes)
        .map_err(|_| RetrieveError::FormatError("IVF-AVQ seek distance overflow".into()))?;
    let next = current
        .checked_add(bytes)
        .ok_or_else(|| RetrieveError::FormatError("IVF-AVQ seek offset overflow".into()))?;
    reader.seek(SeekFrom::Start(next))?;
    Ok(())
}

fn read_partitions(
    path: &Path,
    expected_partitions: usize,
    num_vectors: usize,
    num_codebooks: usize,
) -> Result<Vec<Partition>, RetrieveError> {
    let mut reader = BufReader::new(std::fs::File::open(path)?);
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != IVFAVQ_PARTITION_MAGIC {
        return Err(RetrieveError::FormatError(
            "invalid IVF-AVQ partition file magic".into(),
        ));
    }
    let partition_count = read_u64(&mut reader)? as usize;
    if partition_count != expected_partitions {
        return Err(RetrieveError::FormatError(format!(
            "partition count mismatch: expected {}, got {}",
            expected_partitions, partition_count
        )));
    }

    let mut partitions = Vec::with_capacity(partition_count);
    for _ in 0..partition_count {
        let ids_len = read_u64(&mut reader)? as usize;
        if ids_len > num_vectors {
            return Err(RetrieveError::FormatError(format!(
                "partition length {} exceeds vector count {}",
                ids_len, num_vectors
            )));
        }
        let mut vector_indices = Vec::with_capacity(ids_len);
        for _ in 0..ids_len {
            let id = read_u32(&mut reader)?;
            if id as usize >= num_vectors {
                return Err(RetrieveError::FormatError(format!(
                    "partition id {} exceeds vector count {}",
                    id, num_vectors
                )));
            }
            vector_indices.push(id);
        }

        let codes_len = read_u64(&mut reader)? as usize;
        let expected_codes = ids_len
            .checked_mul(num_codebooks)
            .ok_or_else(|| RetrieveError::FormatError("partition code length overflow".into()))?;
        if codes_len != expected_codes {
            return Err(RetrieveError::FormatError(format!(
                "partition code length mismatch: expected {}, got {}",
                expected_codes, codes_len
            )));
        }
        let mut codes = vec![0u8; codes_len];
        reader.read_exact(&mut codes)?;
        partitions.push(Partition {
            vector_indices,
            codes,
        });
    }

    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        return Err(RetrieveError::FormatError(
            "trailing bytes in IVF-AVQ partition file".into(),
        ));
    }

    Ok(partitions)
}

fn read_u64(reader: &mut impl Read) -> Result<u64, RetrieveError> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_u32(reader: &mut impl Read) -> Result<u32, RetrieveError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::RetrieveError;

    #[test]
    fn test_create_index() {
        let params = IVFAVQParams {
            num_partitions: 2,
            nprobe: 2,
            num_reorder: 10,
            num_codebooks: 2,
            codebook_size: 256,
            seed: 42,
        };
        let index =
            IVFAVQIndex::new(4, params).expect("IVFAVQIndex::new must succeed for valid params");
        assert_eq!(index.dimension, 4);
        assert_eq!(index.num_vectors, 0);
    }

    #[test]
    fn test_add_and_search() {
        let params = IVFAVQParams {
            num_partitions: 2,
            nprobe: 2,
            num_reorder: 10,
            num_codebooks: 2,
            codebook_size: 256,
            seed: 42,
        };
        let mut index = IVFAVQIndex::new(4, params).unwrap();

        // Add 20 vectors (need enough for k-means partitioning)
        for i in 0..20u32 {
            let v = vec![i as f32, (i as f32) * 0.5, 1.0, 0.0];
            index.add(i, v).unwrap();
        }

        index.build().unwrap();

        let query = vec![0.0, 0.0, 1.0, 0.0];
        let results = index.search(&query, 3).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 3);
    }

    #[test]
    fn memory_usage_reports_owned_buffers_after_build() {
        let params = IVFAVQParams {
            num_partitions: 2,
            nprobe: 2,
            num_reorder: 10,
            num_codebooks: 2,
            codebook_size: 16,
            seed: 42,
        };
        let mut index = IVFAVQIndex::new(4, params).unwrap();
        for i in 0..20u32 {
            index
                .add(i, vec![i as f32, (i as f32) * 0.5, 1.0, 0.0])
                .unwrap();
        }
        index.build().unwrap();

        let report = index.memory_usage();
        assert!(report.vectors_bytes >= 20 * 4 * std::mem::size_of::<f32>());
        assert!(report.quantized_bytes >= 20 * 2);
        assert!(report.metadata_bytes >= 20 * std::mem::size_of::<u32>());
        assert!(report.total() >= report.vectors_bytes + report.quantized_bytes);
    }

    #[test]
    fn search_rejects_dimension_mismatch() {
        let params = IVFAVQParams {
            num_partitions: 2,
            nprobe: 2,
            num_reorder: 10,
            num_codebooks: 2,
            codebook_size: 16,
            seed: 42,
        };
        let mut index = IVFAVQIndex::new(4, params).unwrap();
        for i in 0..20u32 {
            index.add(i, vec![i as f32, 0.0, 1.0, 0.0]).unwrap();
        }
        index.build().unwrap();

        let err = index.search(&[1.0, 0.0], 3).unwrap_err();
        match err {
            RetrieveError::DimensionMismatch { query_dim, doc_dim } => {
                assert_eq!(query_dim, 2);
                assert_eq!(doc_dim, 4);
            }
            other => panic!("expected dimension mismatch, got {other:?}"),
        }
    }

    #[test]
    fn search_zero_k_returns_empty() {
        let params = IVFAVQParams {
            num_partitions: 2,
            nprobe: 2,
            num_reorder: 10,
            num_codebooks: 2,
            codebook_size: 16,
            seed: 43,
        };
        let mut index = IVFAVQIndex::new(4, params).unwrap();
        for i in 0..20u32 {
            index.add(i, vec![i as f32, 0.0, 1.0, 0.0]).unwrap();
        }
        index.build().unwrap();

        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let mut file_searcher = IVFAVQFileSearcher::open(dir.path()).unwrap();

        assert!(index.search(&[0.0, 0.0, 1.0, 0.0], 0).unwrap().is_empty());
        assert!(file_searcher
            .search(&[0.0, 0.0, 1.0, 0.0], 0)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn test_zero_dimension_error() {
        let result = IVFAVQIndex::new(0, IVFAVQParams::default());
        assert!(result.is_err());
        match result.unwrap_err() {
            RetrieveError::InvalidParameter(_) => {}
            other => panic!("Expected InvalidParameter, got {:?}", other),
        }
    }

    #[test]
    fn save_load_roundtrip_preserves_search() {
        use rand::{Rng, SeedableRng};

        let dim = 8;
        let n = 96;
        let params = IVFAVQParams {
            num_partitions: 4,
            nprobe: 4,
            num_reorder: 32,
            num_codebooks: 4,
            codebook_size: 16,
            seed: 77,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(77);
        let mut index = IVFAVQIndex::new(dim, params).unwrap();
        for i in 0..n {
            let doc_id = 10_000 + i as u32;
            let vector: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.add(doc_id, vector).unwrap();
        }
        index.build().unwrap();

        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let before = index.search(&query, 10).unwrap();

        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let loaded = IVFAVQIndex::load_from_dir(dir.path()).unwrap();

        assert_eq!(loaded.search(&query, 10).unwrap(), before);
    }

    #[test]
    fn file_searcher_matches_snapshot_loaded_search() {
        use rand::{Rng, SeedableRng};

        let dim = 8;
        let n = 96;
        let params = IVFAVQParams {
            num_partitions: 4,
            nprobe: 4,
            num_reorder: 32,
            num_codebooks: 4,
            codebook_size: 16,
            seed: 80,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(80);
        let mut index = IVFAVQIndex::new(dim, params).unwrap();
        for i in 0..n {
            let doc_id = 20_000 + i as u32;
            let vector: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.add(doc_id, vector).unwrap();
        }
        index.build().unwrap();

        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let loaded = IVFAVQIndex::load_from_dir(dir.path()).unwrap();
        let mut file_searcher = IVFAVQFileSearcher::open(dir.path()).unwrap();

        assert_eq!(
            file_searcher.search(&query, 10).unwrap(),
            loaded.search(&query, 10).unwrap()
        );
    }

    #[test]
    fn file_searcher_diagnostics_report_partition_and_vector_reads() {
        use rand::{Rng, SeedableRng};

        let dim = 8;
        let n = 96;
        let params = IVFAVQParams {
            num_partitions: 4,
            nprobe: 3,
            num_reorder: 24,
            num_codebooks: 4,
            codebook_size: 16,
            seed: 82,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(82);
        let mut index = IVFAVQIndex::new(dim, params).unwrap();
        for i in 0..n {
            let doc_id = 40_000 + i as u32;
            let vector: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.add(doc_id, vector).unwrap();
        }
        index.build().unwrap();

        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let mut file_searcher = IVFAVQFileSearcher::open(dir.path()).unwrap();

        let plain = file_searcher.search(&query, 10).unwrap();
        let (with_diagnostics, diagnostics) =
            file_searcher.search_with_diagnostics(&query, 10).unwrap();

        assert_eq!(with_diagnostics, plain);
        assert_eq!(diagnostics.probed_lists, 3);
        assert_eq!(diagnostics.partition_reads, 3);
        assert!(diagnostics.scanned_vectors >= diagnostics.retained_candidates);
        assert_eq!(
            diagnostics.code_bytes,
            diagnostics.scanned_vectors * file_searcher.params.num_codebooks
        );
        assert!(diagnostics.partition_bytes >= diagnostics.code_bytes);
        assert_eq!(
            diagnostics.raw_vector_reads,
            diagnostics.retained_candidates
        );
        assert!(diagnostics.raw_vector_reads >= with_diagnostics.len());
        assert_eq!(
            diagnostics.raw_vector_bytes,
            diagnostics.raw_vector_reads * dim * std::mem::size_of::<f32>()
        );
    }

    #[cfg(feature = "persistence")]
    #[test]
    fn mmap_searcher_matches_snapshot_loaded_search() {
        use rand::{Rng, SeedableRng};

        let dim = 8;
        let n = 96;
        let params = IVFAVQParams {
            num_partitions: 4,
            nprobe: 4,
            num_reorder: 32,
            num_codebooks: 4,
            codebook_size: 16,
            seed: 81,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(81);
        let mut index = IVFAVQIndex::new(dim, params).unwrap();
        for i in 0..n {
            let doc_id = 30_000 + i as u32;
            let vector: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.add(doc_id, vector).unwrap();
        }
        index.build().unwrap();

        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let loaded = IVFAVQIndex::load_from_dir(dir.path()).unwrap();
        let mut mmap_searcher = IVFAVQFileSearcher::open_mmap(dir.path()).unwrap();

        assert_eq!(
            mmap_searcher.search(&query, 10).unwrap(),
            loaded.search(&query, 10).unwrap()
        );
    }

    #[test]
    fn load_rejects_future_manifest_version() {
        use rand::{Rng, SeedableRng};

        let dim = 8;
        let n = 96;
        let params = IVFAVQParams {
            num_partitions: 4,
            nprobe: 4,
            num_reorder: 32,
            num_codebooks: 4,
            codebook_size: 16,
            seed: 78,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(78);
        let mut index = IVFAVQIndex::new(dim, params).unwrap();
        for i in 0..n {
            let vector: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.add(i as u32, vector).unwrap();
        }
        index.build().unwrap();

        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let manifest_path = dir.path().join("manifest.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        manifest["version"] = serde_json::json!(IVFAVQ_FORMAT_VERSION + 1);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let err = IVFAVQIndex::load_from_dir(dir.path()).unwrap_err();
        assert!(
            err.to_string()
                .contains("unsupported IVF-AVQ format version"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_rejects_vector_length_overflow_before_allocation() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = serde_json::json!({
            "version": IVFAVQ_FORMAT_VERSION,
            "dimension": usize::MAX,
            "num_vectors": 2,
            "params": {
                "num_partitions": 1,
                "nprobe": 1,
                "num_reorder": 1,
                "num_codebooks": 1,
                "codebook_size": 1,
                "seed": 1
            }
        });
        std::fs::write(
            dir.path().join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let err = IVFAVQIndex::load_from_dir(dir.path()).unwrap_err();
        assert!(
            err.to_string().contains("IVF-AVQ vector length overflow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn read_centroids_rejects_length_overflow_before_io() {
        let err = read_centroids(Path::new("/definitely/not/present"), usize::MAX, 2).unwrap_err();
        assert!(
            err.to_string().contains("IVF-AVQ centroid length overflow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_rejects_corrupt_partition_magic() {
        use rand::{Rng, SeedableRng};

        let dim = 8;
        let n = 96;
        let params = IVFAVQParams {
            num_partitions: 4,
            nprobe: 4,
            num_reorder: 32,
            num_codebooks: 4,
            codebook_size: 16,
            seed: 79,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(79);
        let mut index = IVFAVQIndex::new(dim, params).unwrap();
        for i in 0..n {
            let vector: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.add(i as u32, vector).unwrap();
        }
        index.build().unwrap();

        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        std::fs::write(dir.path().join("partitions.bin"), b"not-avq!").unwrap();

        let err = IVFAVQIndex::load_from_dir(dir.path()).unwrap_err();
        assert!(
            err.to_string()
                .contains("invalid IVF-AVQ partition file magic"),
            "unexpected error: {err}"
        );
        let err = match IVFAVQFileSearcher::open(dir.path()) {
            Ok(_) => panic!("corrupt partition magic should reject file-backed IVF-AVQ search"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains("invalid IVF-AVQ partition file magic"),
            "unexpected error: {err}"
        );
    }
}
