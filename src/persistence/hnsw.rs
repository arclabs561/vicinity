//! HNSW graph persistence.
//!
//! Provides disk persistence for HNSW indexes, including:
//! - Graph structure serialization (layers, neighbors)
//! - Vector storage (reuses dense segment format)
//! - Layer assignments
//! - Tombstone flags
//! - Parameters

use crate::persistence::directory::Directory;
use crate::persistence::error::PersistenceResult;
use crate::persistence::format::{FORMAT_VERSION, HNSW_SEGMENT_MAGIC};
use std::io::{Read, Write};

#[cfg(feature = "hnsw")]
use crate::hnsw::graph::{Layer, NeighborList};
#[cfg(feature = "hnsw")]
use crate::hnsw::{HNSWIndex, HNSWParams, NeighborhoodDiversification, SeedSelectionStrategy};

/// Leading marker on a v3 params file. Without a prefix, a truncated file is
/// indistinguishable from the 24-byte v1 layout and loads as a Cosine index.
const PARAMS_MAGIC: &[u8; 4] = b"HNP3";

/// HNSW segment writer for graph persistence.
#[cfg(feature = "hnsw")]
pub struct HNSWSegmentWriter {
    directory: Box<dyn Directory>,
    segment_id: u64,
}

#[cfg(feature = "hnsw")]
impl HNSWSegmentWriter {
    /// Create a new HNSW segment writer.
    pub fn new(directory: Box<dyn Directory>, segment_id: u64) -> Self {
        Self {
            directory,
            segment_id,
        }
    }

    /// Seal a fully written temporary file into place.
    ///
    /// `flush()` only pushes bytes into the OS, and a rename only changes a
    /// directory entry: a crash between them can leave a correctly named file
    /// whose data blocks were never written, and a crash after the rename can
    /// lose the name itself. Both are synced here, through the directory backend
    /// so an in-memory backend stays a no-op instead of failing.
    fn seal(&self, tmp: &str, final_path: &str) -> PersistenceResult<()> {
        self.directory.durable_sync_file(tmp)?;
        self.directory.atomic_rename_durable(tmp, final_path)
    }

    /// Write an HNSW index to disk.
    ///
    /// Format:
    /// - `vectors.bin`: Vector data (SoA layout, same as dense segment)
    /// - `doc_ids.bin`: External doc_ids aligned with internal indices
    /// - `tombstones.bin`: One byte per internal id, `1` for soft-deleted
    /// - `layers.bin`: Graph layers (serialized neighbor lists)
    /// - `layer_assignments.bin`: Layer assignment for each vector
    /// - `params.bin`: HNSW parameters
    /// - `metadata.bin`: Index metadata (dimension, num_vectors, commit byte)
    ///
    /// The segment is marked uncommitted before any file is touched and
    /// committed only after the last seal, so an interrupted save is refused by
    /// both readers rather than loaded as a mixed generation.
    ///
    /// Both metadata writes go through `atomic_write`, which on a filesystem is
    /// durable: it syncs the temporary file, renames it, then syncs the parent
    /// directory. The order is therefore durable 0, then the seven sealed files,
    /// then durable 1. A reader that runs during a re-save sees an uncommitted
    /// segment and fails; writes must not race reads on the same segment id.
    pub fn write_hnsw_index(&mut self, index: &HNSWIndex) -> PersistenceResult<()> {
        // Mark the segment unusable before touching any of its files. A crash
        // between two seals then leaves a segment the loader refuses, rather
        // than the previous metadata paired with the new vectors.
        self.invalidate_segment(index)?;
        self.write_index_files(index)?;
        self.write_metadata(index, true)
    }

    /// Write every segment file except `metadata.bin`.
    ///
    /// Split out so the AoS sidecar can be sealed before the commit byte is
    /// set: `write_hnsw_index_aos` must not publish a commit that does not cover
    /// the file the searcher prefers.
    fn write_index_files(&mut self, index: &HNSWIndex) -> PersistenceResult<()> {
        let segment_dir = format!("segments/segment_hnsw_{}", self.segment_id);
        self.directory.create_dir_all(&segment_dir)?;

        // A re-save writes the SoA layout only. A sidecar left from an earlier
        // `write_hnsw_index_aos` would still be found by the file searcher,
        // which prefers it, so the searcher would serve the previous index's
        // vectors while `load_index` served the new ones.
        let stale_aos = format!("{segment_dir}/vectors_aos.bin");
        if self.directory.exists(&stale_aos) {
            self.directory.delete(&stale_aos)?;
        }

        // Write vectors in SoA layout (same as dense segment).
        //
        // NOTE: `HNSWIndex` stores vectors in-memory in AoS order (each vector contiguous),
        // but on disk we store SoA for cache/SIMD-friendly access and consistency with
        // `DenseSegmentWriter`.
        let vectors_path = format!("{}/vectors.bin", segment_dir);
        let vectors_path_tmp = format!("{vectors_path}.tmp");
        let mut vectors_file = self.directory.create_file(&vectors_path_tmp)?;
        for d in 0..index.dimension {
            for v_idx in 0..index.num_vectors {
                let aos_idx = v_idx * index.dimension + d;
                vectors_file.write_all(&index.vectors[aos_idx].to_le_bytes())?;
            }
        }
        vectors_file.flush()?;
        drop(vectors_file);
        self.seal(&vectors_path_tmp, &vectors_path)?;

        // Write doc_ids (external IDs aligned with internal insertion order).
        //
        // This is the critical "KeyedVectors separation" bit: it ensures searches
        // on a loaded index return the same external IDs that were originally added.
        let doc_ids_path = format!("{}/doc_ids.bin", segment_dir);
        let doc_ids_path_tmp = format!("{doc_ids_path}.tmp");
        let mut doc_ids_file = self.directory.create_file(&doc_ids_path_tmp)?;
        for &doc_id in &index.doc_ids {
            doc_ids_file.write_all(&doc_id.to_le_bytes())?;
        }
        doc_ids_file.flush()?;
        drop(doc_ids_file);
        self.seal(&doc_ids_path_tmp, &doc_ids_path)?;

        // Write tombstone flags (one byte per internal id).
        let tombstones_path = format!("{}/tombstones.bin", segment_dir);
        let tombstones_path_tmp = format!("{tombstones_path}.tmp");
        let mut tombstones_file = self.directory.create_file(&tombstones_path_tmp)?;
        tombstones_file.write_all(&index.tombstone_flags())?;
        tombstones_file.flush()?;
        drop(tombstones_file);
        self.seal(&tombstones_path_tmp, &tombstones_path)?;

        // Write layer assignments
        let assignments_path = format!("{}/layer_assignments.bin", segment_dir);
        let assignments_path_tmp = format!("{assignments_path}.tmp");
        let mut assignments_file = self.directory.create_file(&assignments_path_tmp)?;
        for &assignment in &index.layer_assignments {
            assignments_file.write_all(&[assignment])?;
        }
        assignments_file.flush()?;
        drop(assignments_file);
        self.seal(&assignments_path_tmp, &assignments_path)?;

        // Write graph layers
        let layers_path = format!("{}/layers.bin", segment_dir);
        let layers_path_tmp = format!("{layers_path}.tmp");
        let mut layers_file = self.directory.create_file(&layers_path_tmp)?;

        // Write number of layers
        layers_file.write_all(&(index.layers.len() as u32).to_le_bytes())?;

        // Write each layer
        for layer in &index.layers {
            // Write number of neighbor lists
            layers_file.write_all(&(layer.len() as u32).to_le_bytes())?;

            // Write each neighbor list
            for node in 0..layer.len() as u32 {
                let neighbor_list = layer.get_neighbors(node);
                // Write number of neighbors
                layers_file.write_all(&(neighbor_list.len() as u32).to_le_bytes())?;

                // Write neighbor IDs
                for &neighbor_id in neighbor_list.iter() {
                    layers_file.write_all(&neighbor_id.to_le_bytes())?;
                }
            }
        }
        layers_file.flush()?;
        drop(layers_file);
        self.seal(&layers_path_tmp, &layers_path)?;

        // Write parameters
        let params_path = format!("{}/params.bin", segment_dir);
        let params_path_tmp = format!("{params_path}.tmp");
        let mut params_file = self.directory.create_file(&params_path_tmp)?;
        params_file.write_all(PARAMS_MAGIC)?;
        params_file.write_all(&(index.params.m as u32).to_le_bytes())?;
        params_file.write_all(&(index.params.m_max as u32).to_le_bytes())?;
        params_file.write_all(&index.params.m_l.to_le_bytes())?;
        params_file.write_all(&(index.params.ef_construction as u32).to_le_bytes())?;
        params_file.write_all(&(index.params.ef_search as u32).to_le_bytes())?;
        // v2 fields: metric + auto_normalize (appended for backward compat)
        let metric_byte: u8 = match index.params.metric {
            crate::distance::DistanceMetric::L2 => 0,
            crate::distance::DistanceMetric::Cosine => 1,
            crate::distance::DistanceMetric::Angular => 2,
            crate::distance::DistanceMetric::InnerProduct => 3,
        };
        params_file.write_all(&[metric_byte])?;
        params_file.write_all(&[if index.params.auto_normalize { 1 } else { 0 }])?;
        #[cfg(feature = "compact-hnsw")]
        params_file.write_all(&[if index.params.compact_upper_layers {
            1
        } else {
            0
        }])?;
        params_file.flush()?;
        drop(params_file);
        self.seal(&params_path_tmp, &params_path)?;

        Ok(())
    }

    /// Write `metadata.bin` with `committed` in the v2 commit byte.
    ///
    /// Written with `atomic_write` rather than the temp+seal pair: the
    /// invalidation must replace the previous metadata before any data file
    /// changes, and the final commit must be one durable rename.
    ///
    /// Layout (v2):
    ///   [0..8]   HNSW_SEGMENT_MAGIC  (8 bytes, "VCNHNSW\x01")
    ///   [8..12]  FORMAT_VERSION      (u32 LE)
    ///   [12..16] dimension           (u32 LE)
    ///   [16..20] num_vectors         (u32 LE)
    ///   [20]     is_built            (u8 flag)
    ///   [21]     committed           (u8 flag; 1 means every file is sealed)
    ///
    /// v1 files have no commit byte. Legacy v0 (vicinity 0.6.x) has no magic or
    /// version either: the first 4 bytes are the dimension.
    fn write_metadata(&mut self, index: &HNSWIndex, committed: bool) -> PersistenceResult<()> {
        let segment_dir = format!("segments/segment_hnsw_{}", self.segment_id);
        let mut metadata = Vec::with_capacity(22);
        metadata.extend_from_slice(&HNSW_SEGMENT_MAGIC);
        metadata.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        metadata.extend_from_slice(&(index.dimension as u32).to_le_bytes());
        metadata.extend_from_slice(&(index.num_vectors as u32).to_le_bytes());
        metadata.push(u8::from(index.is_built()));
        metadata.push(u8::from(committed));
        self.directory
            .atomic_write(&format!("{segment_dir}/metadata.bin"), &metadata)
    }

    /// Replace the segment's metadata with an uncommitted header.
    fn invalidate_segment(&mut self, index: &HNSWIndex) -> PersistenceResult<()> {
        let segment_dir = format!("segments/segment_hnsw_{}", self.segment_id);
        self.directory.create_dir_all(&segment_dir)?;
        self.write_metadata(index, false)
    }

    /// Write an HNSW index plus the AoS vector sidecar.
    ///
    /// Calls [`Self::write_hnsw_index`] and additionally writes
    /// `vectors_aos.bin`, where each vector's `dimension` components are
    /// contiguous. [`HNSWSegmentSearcher`](crate::hnsw::segment_searcher::HNSWSegmentSearcher)
    /// prefers this sidecar: one AoS vector fits in a single cache block,
    /// while the dimension-major `vectors.bin` needs up to `dimension` block
    /// reads for the same vector.
    ///
    /// The sidecar duplicates the vector bytes (the same data as `vectors.bin`
    /// in a different order) in exchange for cache-friendly reads.
    pub fn write_hnsw_index_aos(&mut self, index: &HNSWIndex) -> PersistenceResult<()> {
        self.invalidate_segment(index)?;
        self.write_index_files(index)?;

        let segment_dir = format!("segments/segment_hnsw_{}", self.segment_id);
        let aos_path = format!("{}/vectors_aos.bin", segment_dir);
        let aos_path_tmp = format!("{aos_path}.tmp");
        let mut aos_file = self.directory.create_file(&aos_path_tmp)?;
        // `HNSWIndex::vectors` is already AoS in internal id order, which is
        // the order the searcher needs for layer neighbour ids.
        for value in &index.vectors {
            aos_file.write_all(&value.to_le_bytes())?;
        }
        aos_file.flush()?;
        drop(aos_file);
        self.seal(&aos_path_tmp, &aos_path)?;

        self.write_metadata(index, true)
    }
}

/// HNSW segment reader for loading graphs from disk.
#[cfg(feature = "hnsw")]
pub struct HNSWSegmentReader {
    directory: Box<dyn Directory>,
    segment_id: u64,
    dimension: usize,
    num_vectors: usize,
    params: HNSWParams,
    built: bool,
}

#[cfg(feature = "hnsw")]
impl HNSWSegmentReader {
    /// Load an HNSW segment from disk.
    ///
    /// Supports two on-disk layouts:
    ///
    /// - **v1** (vicinity 0.7+): `metadata.bin` starts with `HNSW_SEGMENT_MAGIC` (8 bytes)
    ///   followed by `FORMAT_VERSION` (u32 LE), then `dimension`, `num_vectors`, `is_built`.
    ///
    /// - **v0** (vicinity 0.6.x legacy): no magic/version prefix; the file starts directly
    ///   with `dimension` (u32 LE), `num_vectors` (u32 LE), `is_built` (u8).
    ///
    /// If the first 8 bytes match the magic but the version exceeds [`FORMAT_VERSION`], an
    /// error is returned immediately. If the first 8 bytes don't match the magic, v0 decode
    /// is attempted using the first 4 bytes as the dimension field. If that also produces
    /// unreasonable values, a `PersistenceError::Format` is returned.
    pub fn load(directory: Box<dyn Directory>, segment_id: u64) -> PersistenceResult<Self> {
        let segment_dir = format!("segments/segment_hnsw_{}", segment_id);

        // Load metadata -- detect magic to distinguish v1 from legacy v0.
        let metadata_path = format!("{}/metadata.bin", segment_dir);
        let mut metadata_file = directory.open_file(&metadata_path)?;

        // Read the first 8 bytes to check for the magic marker.
        let mut magic_buf = [0u8; 8];
        metadata_file.read_exact(&mut magic_buf)?;

        let (dimension, num_vectors, built) = if magic_buf == HNSW_SEGMENT_MAGIC {
            // v1 file: read FORMAT_VERSION, then the three metadata fields.
            let mut version_bytes = [0u8; 4];
            metadata_file.read_exact(&mut version_bytes)?;
            let version = u32::from_le_bytes(version_bytes);
            if version > FORMAT_VERSION {
                return Err(crate::persistence::error::PersistenceError::Format(
                    format!("unsupported format version {version}, expected {FORMAT_VERSION}"),
                ));
            }

            let mut dim_bytes = [0u8; 4];
            let mut num_vec_bytes = [0u8; 4];
            let mut built_byte = [0u8; 1];
            metadata_file.read_exact(&mut dim_bytes)?;
            metadata_file.read_exact(&mut num_vec_bytes)?;
            metadata_file.read_exact(&mut built_byte)?;
            // v2 appends a commit byte. Zero means the last save stopped between
            // two files, so the surviving files are not one generation.
            if version >= 2 {
                let mut commit_byte = [0u8; 1];
                metadata_file.read_exact(&mut commit_byte).map_err(|_| {
                    crate::persistence::error::PersistenceError::Format(
                        "v2 metadata.bin is missing its commit byte".into(),
                    )
                })?;
                if commit_byte[0] == 0 {
                    return Err(crate::persistence::error::PersistenceError::Format(
                        "segment write did not complete; the last save was interrupted between                          two files, so the segment mixes generations and must be rebuilt"
                            .into(),
                    ));
                }
            }
            (
                u32::from_le_bytes(dim_bytes) as usize,
                u32::from_le_bytes(num_vec_bytes) as usize,
                built_byte[0] != 0,
            )
        } else {
            // Possible v0 (legacy, no magic).
            //
            // v0 layout: dimension (u32 LE) | num_vectors (u32 LE) | is_built (u8).
            // The first 4 bytes of magic_buf are the dimension; bytes 4..8 are num_vectors.
            // We read one more byte for the is_built flag.
            //
            // NOTE: If this produces unreasonable values (caught below by the size guards),
            // the bytes were not a v0 header either: report the magic mismatch instead of
            // pretending the file claimed an absurd index size.
            let dim_bytes: [u8; 4] = [magic_buf[0], magic_buf[1], magic_buf[2], magic_buf[3]];
            let num_vec_bytes: [u8; 4] = [magic_buf[4], magic_buf[5], magic_buf[6], magic_buf[7]];
            let mut built_byte = [0u8; 1];
            if metadata_file.read_exact(&mut built_byte).is_err() {
                return Err(crate::persistence::error::PersistenceError::BadMagic {
                    expected: crate::RetrieveError::describe_magic(&HNSW_SEGMENT_MAGIC),
                    found: crate::RetrieveError::describe_magic(&magic_buf),
                });
            }
            (
                u32::from_le_bytes(dim_bytes) as usize,
                u32::from_le_bytes(num_vec_bytes) as usize,
                built_byte[0] != 0,
            )
        };

        // Guard against crafted files that claim enormous sizes.
        const MAX_VECTORS: usize = 100_000_000; // 100M vectors
        const MAX_DIMENSION: usize = 65_536; // 64K dimensions
        const MAX_ALLOC_BYTES: usize = 8 * 1024 * 1024 * 1024; // 8 GB

        if num_vectors > MAX_VECTORS || dimension > MAX_DIMENSION {
            // A v1 file that passes the magic check can legitimately hit this guard
            // only when its own header is corrupt; a file that failed the magic check
            // is simply not an HNSW segment.
            if magic_buf != HNSW_SEGMENT_MAGIC {
                return Err(crate::persistence::error::PersistenceError::BadMagic {
                    expected: crate::RetrieveError::describe_magic(&HNSW_SEGMENT_MAGIC),
                    found: crate::RetrieveError::describe_magic(&magic_buf),
                });
            }
            return Err(crate::persistence::error::PersistenceError::Format(
                format!(
                    "unreasonable index size: {} vectors x {} dimensions",
                    num_vectors, dimension
                ),
            ));
        }
        let alloc_size = num_vectors
            .checked_mul(dimension)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                crate::persistence::error::PersistenceError::Format(
                    "allocation size overflow".into(),
                )
            })?;
        if alloc_size > MAX_ALLOC_BYTES {
            return Err(crate::persistence::error::PersistenceError::Format(
                format!("index too large for memory: {} bytes", alloc_size),
            ));
        }

        // Load parameters. Read the whole file first: a truncation at exactly
        // the 24-byte v1 length used to look like a legacy file and silently
        // default the metric to Cosine.
        let params_path = format!("{}/params.bin", segment_dir);
        let params_bytes = {
            let mut file = directory.open_file(&params_path)?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;
            buffer
        };
        let prefixed = params_bytes.starts_with(PARAMS_MAGIC);
        let (body_len, bytes) = if prefixed {
            let body = params_bytes.len() - PARAMS_MAGIC.len();
            if !(26..=27).contains(&body) {
                return Err(crate::persistence::error::PersistenceError::Format(
                    format!(
                        "params.bin has the v3 marker but {body} bytes of body; expected 26 or 27"
                    ),
                ));
            }
            (body, &params_bytes[PARAMS_MAGIC.len()..])
        } else {
            if !(24..=27).contains(&params_bytes.len()) {
                return Err(crate::persistence::error::PersistenceError::Format(
                    format!(
                        "params.bin is {} bytes; expected 24-27 (legacy)",
                        params_bytes.len()
                    ),
                ));
            }
            (params_bytes.len(), &params_bytes[..])
        };
        let _ = body_len;
        // The length was checked above, but a recoverable error beats a panic in
        // a loader: a crafted file must never take the process down.
        let field = |at: usize, width: usize| -> PersistenceResult<&[u8]> {
            bytes.get(at..at + width).ok_or_else(|| {
                crate::persistence::error::PersistenceError::Format(format!(
                    "params.bin is {} bytes, too short for the field at offset {at}",
                    bytes.len()
                ))
            })
        };
        let m_bytes: [u8; 4] = field(0, 4)?.try_into().map_err(|_| {
            crate::persistence::error::PersistenceError::Format("params.bin m field".into())
        })?;
        let m_max_bytes: [u8; 4] = field(4, 4)?.try_into().map_err(|_| {
            crate::persistence::error::PersistenceError::Format("params.bin m_max field".into())
        })?;
        let m_l_bytes: [u8; 8] = field(8, 8)?.try_into().map_err(|_| {
            crate::persistence::error::PersistenceError::Format("params.bin m_l field".into())
        })?;
        let ef_construction_bytes: [u8; 4] = field(16, 4)?.try_into().map_err(|_| {
            crate::persistence::error::PersistenceError::Format(
                "params.bin ef_construction field".into(),
            )
        })?;
        let ef_search_bytes: [u8; 4] = field(20, 4)?.try_into().map_err(|_| {
            crate::persistence::error::PersistenceError::Format("params.bin ef_search field".into())
        })?;

        // v2 fields: metric + auto_normalize (optional, backward-compat with v1
        // files). Their absence is only accepted on a 24-byte body, i.e. a real
        // v1 file, never on a truncated v2 one.
        let metric = match bytes.get(24) {
            Some(byte) => match byte {
                0 => crate::distance::DistanceMetric::L2,
                1 => crate::distance::DistanceMetric::Cosine,
                2 => crate::distance::DistanceMetric::Angular,
                3 => crate::distance::DistanceMetric::InnerProduct,
                other => {
                    return Err(crate::persistence::error::PersistenceError::Format(
                        format!("unknown distance metric: {}", other),
                    ));
                }
            },
            // Legacy v1 file without metric byte: default to Cosine
            None => crate::distance::DistanceMetric::Cosine,
        };
        let auto_normalize = bytes.get(25).is_some_and(|byte| *byte != 0);

        #[cfg(feature = "compact-hnsw")]
        let compact_upper_layers = bytes.get(26).is_some_and(|byte| *byte != 0);

        let params = HNSWParams {
            m: u32::from_le_bytes(m_bytes) as usize,
            m_max: u32::from_le_bytes(m_max_bytes) as usize,
            m_l: f64::from_le_bytes(m_l_bytes),
            ef_construction: u32::from_le_bytes(ef_construction_bytes) as usize,
            ef_search: u32::from_le_bytes(ef_search_bytes) as usize,
            auto_normalize,
            seed_selection: SeedSelectionStrategy::default(),
            neighborhood_diversification: NeighborhoodDiversification::default(),
            seed: None,
            metric,
            #[cfg(feature = "compact-hnsw")]
            compact_upper_layers,
            #[cfg(feature = "id-compression")]
            id_compression: None,
            #[cfg(feature = "id-compression")]
            compression_threshold: 100,
        };

        Ok(Self {
            directory,
            segment_id,
            dimension,
            num_vectors,
            params,
            built,
        })
    }

    /// Reconstruct the HNSW index from disk.
    ///
    /// This loads all data structures into memory.
    pub fn load_index(&self) -> PersistenceResult<HNSWIndex> {
        let segment_dir = format!("segments/segment_hnsw_{}", self.segment_id);

        // Load vectors
        let vectors_path = format!("{}/vectors.bin", segment_dir);
        let mut vectors_file = self.directory.open_file(&vectors_path)?;
        // On disk: SoA layout. In memory for HNSWIndex: AoS layout.
        let mut vectors = vec![0f32; self.num_vectors * self.dimension];
        let mut value_bytes = [0u8; 4];
        for d in 0..self.dimension {
            for v_idx in 0..self.num_vectors {
                vectors_file.read_exact(&mut value_bytes)?;
                let value = f32::from_le_bytes(value_bytes);
                let aos_idx = v_idx * self.dimension + d;
                vectors[aos_idx] = value;
            }
        }

        // Load doc_ids (if present). Backward-compatible fallback to identity mapping.
        let doc_ids_path = format!("{}/doc_ids.bin", segment_dir);
        let doc_ids: Vec<u32> = if self.directory.exists(&doc_ids_path) {
            let mut doc_ids_file = self.directory.open_file(&doc_ids_path)?;
            let mut doc_ids = Vec::with_capacity(self.num_vectors);
            let mut id_bytes = [0u8; 4];
            for _ in 0..self.num_vectors {
                doc_ids_file.read_exact(&mut id_bytes)?;
                doc_ids.push(u32::from_le_bytes(id_bytes));
            }
            doc_ids
        } else {
            // Legacy segments (created before doc_id persistence existed)
            (0..self.num_vectors as u32).collect()
        };

        // Load tombstone flags if present. Legacy segments have no sidecar.
        let tombstones_path = format!("{}/tombstones.bin", segment_dir);
        let tombstone_flags = if self.directory.exists(&tombstones_path) {
            let mut tombstones_file = self.directory.open_file(&tombstones_path)?;
            let mut flags = vec![0u8; self.num_vectors];
            tombstones_file.read_exact(&mut flags)?;
            Some(flags)
        } else {
            None
        };

        // Load layer assignments
        let assignments_path = format!("{}/layer_assignments.bin", segment_dir);
        let mut assignments_file = self.directory.open_file(&assignments_path)?;
        let mut layer_assignments = vec![0u8; self.num_vectors];
        assignments_file.read_exact(&mut layer_assignments)?;

        // Load graph layers
        let layers_path = format!("{}/layers.bin", segment_dir);
        let mut layers_file = self.directory.open_file(&layers_path)?;

        let mut num_layers_bytes = [0u8; 4];
        layers_file.read_exact(&mut num_layers_bytes)?;
        let num_layers = u32::from_le_bytes(num_layers_bytes) as usize;

        // HNSW layers are logarithmic; even very large indexes rarely exceed ~40.
        const MAX_LAYERS: usize = 256;
        if num_layers > MAX_LAYERS {
            return Err(crate::persistence::error::PersistenceError::Format(
                format!("unreasonable layer count: {}", num_layers),
            ));
        }

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let mut num_lists_bytes = [0u8; 4];
            layers_file.read_exact(&mut num_lists_bytes)?;
            let num_lists = u32::from_le_bytes(num_lists_bytes) as usize;

            if num_lists > self.num_vectors {
                return Err(crate::persistence::error::PersistenceError::Format(
                    format!(
                        "layer has {} neighbor lists but index has {} vectors",
                        num_lists, self.num_vectors
                    ),
                ));
            }

            let mut neighbors_list = Vec::with_capacity(num_lists);
            for _ in 0..num_lists {
                let mut num_neighbors_bytes = [0u8; 4];
                layers_file.read_exact(&mut num_neighbors_bytes)?;
                let num_neighbors = u32::from_le_bytes(num_neighbors_bytes) as usize;

                // Each node can have at most m_max * 2 neighbors (bottom layer).
                // Use a generous ceiling to catch corruption without false positives.
                const MAX_NEIGHBORS: usize = 65_536;
                if num_neighbors > MAX_NEIGHBORS {
                    return Err(crate::persistence::error::PersistenceError::Format(
                        format!("unreasonable neighbor count: {}", num_neighbors),
                    ));
                }

                let mut neighbors = NeighborList::new();
                for _ in 0..num_neighbors {
                    let mut neighbor_bytes = [0u8; 4];
                    layers_file.read_exact(&mut neighbor_bytes)?;
                    neighbors.push(u32::from_le_bytes(neighbor_bytes));
                }
                neighbors_list.push(neighbors);
            }

            // Construct layer using crate-internal constructor
            layers.push(Layer::new_uncompressed(neighbors_list));
        }

        // Reconstruct index (validates structural invariants)
        let mut index = HNSWIndex::from_parts(
            vectors,
            self.dimension,
            self.num_vectors,
            layers,
            layer_assignments,
            self.params.clone(),
            self.built,
            doc_ids,
        )?;

        #[cfg(feature = "compact-hnsw")]
        if index.params.compact_upper_layers {
            index.freeze_upper_layers()?;
        }

        if let Some(flags) = tombstone_flags {
            index.restore_tombstone_flags(&flags)?;
        }

        Ok(index)
    }
}

#[cfg(test)]
#[cfg(feature = "hnsw")]
mod tests {
    use super::*;
    use crate::persistence::directory::{Directory, FsDirectory, MemoryDirectory};
    use crate::persistence::error::PersistenceError;

    fn write_metadata(mem: &MemoryDirectory, bytes: &[u8]) {
        mem.atomic_write("segments/segment_hnsw_0/metadata.bin", bytes)
            .expect("write metadata");
    }

    /// A filesystem directory whose `fsync` fails, standing in for a disk that
    /// cannot guarantee durability.
    struct UnsyncedDirectory {
        inner: FsDirectory,
    }

    impl Directory for UnsyncedDirectory {
        fn create_file(&self, path: &str) -> PersistenceResult<Box<dyn Write + Send>> {
            self.inner.create_file(path)
        }
        fn open_file(&self, path: &str) -> PersistenceResult<Box<dyn Read + Send>> {
            self.inner.open_file(path)
        }
        fn exists(&self, path: &str) -> bool {
            self.inner.exists(path)
        }
        fn delete(&self, path: &str) -> PersistenceResult<()> {
            self.inner.delete(path)
        }
        fn atomic_rename(&self, from: &str, to: &str) -> PersistenceResult<()> {
            self.inner.atomic_rename(from, to)
        }
        fn create_dir_all(&self, path: &str) -> PersistenceResult<()> {
            self.inner.create_dir_all(path)
        }
        fn list_dir(&self, path: &str) -> PersistenceResult<Vec<String>> {
            self.inner.list_dir(path)
        }
        fn append_file(&self, path: &str) -> PersistenceResult<Box<dyn Write + Send>> {
            self.inner.append_file(path)
        }
        fn atomic_write(&self, path: &str, data: &[u8]) -> PersistenceResult<()> {
            self.inner.atomic_write(path, data)
        }
        fn file_path(&self, path: &str) -> Option<std::path::PathBuf> {
            self.inner.file_path(path)
        }
        fn durable_sync_file(&self, _path: &str) -> PersistenceResult<()> {
            Err(PersistenceError::NotSupported(
                "test: fsync disabled".into(),
            ))
        }
    }

    #[test]
    fn write_refuses_to_publish_a_file_that_could_not_be_synced() {
        let mut index = HNSWIndex::new(4, 8, 8).unwrap();
        for i in 0..8u32 {
            let vector = crate::distance::normalize(
                &(0..4).map(|d| (i * 7 + d) as f32 / 8.0).collect::<Vec<_>>(),
            );
            index.add(i, vector).unwrap();
        }
        index.build().unwrap();

        let dir = tempfile::tempdir().unwrap();
        let mut writer = HNSWSegmentWriter::new(
            Box::new(UnsyncedDirectory {
                inner: FsDirectory::new(dir.path()).unwrap(),
            }),
            5,
        );
        let error = writer
            .write_hnsw_index(&index)
            .expect_err("a failed fsync must fail the write");

        assert!(
            matches!(error, PersistenceError::NotSupported(_)),
            "{error:?}"
        );
        assert!(
            !dir.path()
                .join("segments/segment_hnsw_5/vectors.bin")
                .exists(),
            "the temp file must not be renamed into place before it is durable"
        );
    }

    /// A filesystem directory whose `durable_sync_file` fails on the Nth call,
    /// so a save can be stopped between two files.
    struct SealFailsAfter {
        inner: FsDirectory,
        remaining: std::sync::atomic::AtomicUsize,
    }

    impl Directory for SealFailsAfter {
        fn create_file(&self, path: &str) -> PersistenceResult<Box<dyn Write + Send>> {
            self.inner.create_file(path)
        }
        fn open_file(&self, path: &str) -> PersistenceResult<Box<dyn Read + Send>> {
            self.inner.open_file(path)
        }
        fn exists(&self, path: &str) -> bool {
            self.inner.exists(path)
        }
        fn delete(&self, path: &str) -> PersistenceResult<()> {
            self.inner.delete(path)
        }
        fn atomic_rename(&self, from: &str, to: &str) -> PersistenceResult<()> {
            self.inner.atomic_rename(from, to)
        }
        fn create_dir_all(&self, path: &str) -> PersistenceResult<()> {
            self.inner.create_dir_all(path)
        }
        fn list_dir(&self, path: &str) -> PersistenceResult<Vec<String>> {
            self.inner.list_dir(path)
        }
        fn append_file(&self, path: &str) -> PersistenceResult<Box<dyn Write + Send>> {
            self.inner.append_file(path)
        }
        fn atomic_write(&self, path: &str, data: &[u8]) -> PersistenceResult<()> {
            self.inner.atomic_write(path, data)
        }
        fn file_path(&self, path: &str) -> Option<std::path::PathBuf> {
            self.inner.file_path(path)
        }
        fn durable_sync_file(&self, path: &str) -> PersistenceResult<()> {
            let remaining = self.remaining.load(std::sync::atomic::Ordering::Relaxed);
            if remaining == 0 {
                return Err(PersistenceError::NotSupported(
                    "test: fsync budget exhausted".into(),
                ));
            }
            self.remaining
                .store(remaining - 1, std::sync::atomic::Ordering::Relaxed);
            self.inner.durable_sync_file(path)
        }
    }

    /// A filesystem directory whose `atomic_write` fails on the Nth call, so a
    /// test can stop the writer at the invalidation write or at the commit.
    struct CommitFailsAfter {
        inner: FsDirectory,
        remaining: std::sync::atomic::AtomicUsize,
    }

    impl Directory for CommitFailsAfter {
        fn create_file(&self, path: &str) -> PersistenceResult<Box<dyn Write + Send>> {
            self.inner.create_file(path)
        }
        fn open_file(&self, path: &str) -> PersistenceResult<Box<dyn Read + Send>> {
            self.inner.open_file(path)
        }
        fn exists(&self, path: &str) -> bool {
            self.inner.exists(path)
        }
        fn delete(&self, path: &str) -> PersistenceResult<()> {
            self.inner.delete(path)
        }
        fn atomic_rename(&self, from: &str, to: &str) -> PersistenceResult<()> {
            self.inner.atomic_rename(from, to)
        }
        fn create_dir_all(&self, path: &str) -> PersistenceResult<()> {
            self.inner.create_dir_all(path)
        }
        fn list_dir(&self, path: &str) -> PersistenceResult<Vec<String>> {
            self.inner.list_dir(path)
        }
        fn append_file(&self, path: &str) -> PersistenceResult<Box<dyn Write + Send>> {
            self.inner.append_file(path)
        }
        fn atomic_write(&self, path: &str, data: &[u8]) -> PersistenceResult<()> {
            let remaining = self.remaining.load(std::sync::atomic::Ordering::Relaxed);
            if remaining == 0 {
                return Err(PersistenceError::NotSupported(
                    "test: atomic_write budget exhausted".into(),
                ));
            }
            self.remaining
                .store(remaining - 1, std::sync::atomic::Ordering::Relaxed);
            self.inner.atomic_write(path, data)
        }
        fn file_path(&self, path: &str) -> Option<std::path::PathBuf> {
            self.inner.file_path(path)
        }
        fn durable_sync_file(&self, path: &str) -> PersistenceResult<()> {
            self.inner.durable_sync_file(path)
        }
    }

    fn small_index(count: u32, dim: usize) -> HNSWIndex {
        let mut index = HNSWIndex::new(dim, 8, 8).unwrap();
        for id in 0..count {
            let vector = crate::distance::normalize(
                &(0..dim)
                    .map(|d| ((id as usize * 31 + d * 17) % 97) as f32 / 97.0 + 0.01)
                    .collect::<Vec<_>>(),
            );
            index.add(id, vector).unwrap();
        }
        index.build().unwrap();
        index
    }

    #[test]
    fn interrupted_resave_is_refused_instead_of_mixing_generations() {
        // A save that stops between two files must leave metadata marked
        // uncommitted. The loader then refuses the segment: with the old
        // metadata still valid it would pair the previous row count with the new
        // vectors, and `read_exact` would accept the new prefix silently.
        let dir = tempfile::tempdir().unwrap();
        let first = small_index(64, 8);
        HNSWSegmentWriter::new(Box::new(FsDirectory::new(dir.path()).unwrap()), 7)
            .write_hnsw_index(&first)
            .unwrap();

        let second = small_index(128, 8);
        let mut failing = HNSWSegmentWriter::new(
            Box::new(SealFailsAfter {
                inner: FsDirectory::new(dir.path()).unwrap(),
                remaining: std::sync::atomic::AtomicUsize::new(3),
            }),
            7,
        );
        assert!(
            failing.write_hnsw_index(&second).is_err(),
            "the third seal must fail"
        );

        // vectors.bin was sealed from `second` before the failure, so the old
        // file set is already mixed. Loading must error, not serve 64 vectors of
        // a 128-vector index.
        let error = HNSWSegmentReader::load(Box::new(FsDirectory::new(dir.path()).unwrap()), 7)
            .err()
            .expect("an interrupted save must not load");
        assert!(
            matches!(error, PersistenceError::Format(ref message)
                if message.contains("did not complete")),
            "expected the commit-byte error, got {error:?}"
        );
    }

    #[test]
    fn a_failed_commit_refuses_the_new_segment_and_keeps_the_old_one() {
        // Committing is the last step. If the commit write itself fails after
        // every file is sealed, the header must still read uncommitted: the new
        // files are complete, but nothing may bless them without a durable
        // commit. This is the failure mode a commit byte exists for.
        let dir = tempfile::tempdir().unwrap();
        let first = small_index(32, 8);
        HNSWSegmentWriter::new(Box::new(FsDirectory::new(dir.path()).unwrap()), 9)
            .write_hnsw_index(&first)
            .unwrap();

        let second = small_index(64, 8);
        let mut writer = HNSWSegmentWriter::new(
            Box::new(CommitFailsAfter {
                inner: FsDirectory::new(dir.path()).unwrap(),
                // one budgeted write: the invalidation succeeds, the commit fails
                remaining: std::sync::atomic::AtomicUsize::new(1),
            }),
            9,
        );
        assert!(writer.write_hnsw_index(&second).is_err());

        let error = HNSWSegmentReader::load(Box::new(FsDirectory::new(dir.path()).unwrap()), 9)
            .err()
            .expect("an uncommitted save must not load");
        assert!(
            matches!(error, PersistenceError::Format(ref message)
                if message.contains("did not complete")),
            "expected the commit-byte error, got {error:?}"
        );
    }

    #[test]
    fn a_failed_invalidation_leaves_the_previous_segment_readable() {
        // The invalidation is the first write. If it fails, nothing else has been
        // touched, so the previous generation must still load: a refused save
        // must not destroy a working segment.
        let dir = tempfile::tempdir().unwrap();
        let first = small_index(32, 8);
        HNSWSegmentWriter::new(Box::new(FsDirectory::new(dir.path()).unwrap()), 10)
            .write_hnsw_index(&first)
            .unwrap();

        let second = small_index(64, 8);
        let mut writer = HNSWSegmentWriter::new(
            Box::new(CommitFailsAfter {
                inner: FsDirectory::new(dir.path()).unwrap(),
                remaining: std::sync::atomic::AtomicUsize::new(0),
            }),
            10,
        );
        assert!(writer.write_hnsw_index(&second).is_err());

        let reader = HNSWSegmentReader::load(Box::new(FsDirectory::new(dir.path()).unwrap()), 10)
            .expect("the previous generation must still load");
        assert_eq!(reader.load_index().unwrap().num_vectors, 32);
    }

    #[test]
    fn v1_metadata_without_a_commit_byte_still_loads() {
        // Files written before the commit byte must keep loading: the byte is
        // only required when the header says v2.
        let dir = tempfile::tempdir().unwrap();
        let index = small_index(32, 8);
        HNSWSegmentWriter::new(Box::new(FsDirectory::new(dir.path()).unwrap()), 8)
            .write_hnsw_index(&index)
            .unwrap();

        let metadata_path = dir.path().join("segments/segment_hnsw_8/metadata.bin");
        let mut v1 = Vec::new();
        v1.extend_from_slice(&HNSW_SEGMENT_MAGIC);
        v1.extend_from_slice(&1u32.to_le_bytes());
        v1.extend_from_slice(&(index.dimension as u32).to_le_bytes());
        v1.extend_from_slice(&(index.num_vectors as u32).to_le_bytes());
        v1.push(u8::from(index.is_built()));
        std::fs::write(&metadata_path, v1).unwrap();

        let reader = HNSWSegmentReader::load(Box::new(FsDirectory::new(dir.path()).unwrap()), 8)
            .expect("a v1 header has no commit byte to check");
        assert_eq!(reader.load_index().unwrap().num_vectors, 32);
    }

    #[test]
    fn foreign_metadata_reports_bad_magic() {
        let mem = MemoryDirectory::new();
        // A JSON file (or any non-v0, non-HNSW file): the magic does not match and
        // the legacy-v0 decode does not produce a plausible header either.
        write_metadata(&mem, br#"{"vectors":[1.0,0.0],"dimension":2}"#);

        let error = HNSWSegmentReader::load(Box::new(mem.clone()), 0)
            .err()
            .expect("foreign metadata must fail");
        match error {
            PersistenceError::BadMagic { expected, found } => {
                assert!(expected.contains("56 43 4e 48 4e 53 57 01"), "{expected}");
                assert!(found.contains("vector"), "{found}");
            }
            other => panic!("expected BadMagic, got {other:?}"),
        }

        // The retrievable error keeps the same typed cause.
        let typed: crate::RetrieveError = HNSWSegmentReader::load(Box::new(mem), 0)
            .err()
            .expect("foreign metadata must fail")
            .into();
        assert!(
            matches!(typed, crate::RetrieveError::BadMagic { .. }),
            "expected RetrieveError::BadMagic, got {typed:?}"
        );
    }

    #[test]
    fn truncated_v0_header_reports_bad_magic() {
        let mem = MemoryDirectory::new();
        // Bad magic and too short for the v0 header byte.
        write_metadata(&mem, b"BADMAGIC");

        let error = HNSWSegmentReader::load(Box::new(mem), 0)
            .err()
            .expect("truncated v0 header must fail");
        assert!(
            matches!(error, PersistenceError::BadMagic { .. }),
            "expected BadMagic, got {error:?}"
        );
    }

    #[test]
    fn test_hnsw_segment_write_read() {
        let dim = 4;
        let mut index = HNSWIndex::new(dim, 8, 8).unwrap();

        // Two simple normalized vectors.
        let v0 = vec![1.0, 0.0, 0.0, 0.0];
        let v1 = vec![0.0, 1.0, 0.0, 0.0];

        index.add(42, v0.clone()).unwrap();
        index.add(7, v1.clone()).unwrap();
        index.build().unwrap();

        let mem = MemoryDirectory::new();

        let mut writer = HNSWSegmentWriter::new(Box::new(mem.clone()), 1);
        writer.write_hnsw_index(&index).unwrap();

        let reader = HNSWSegmentReader::load(Box::new(mem.clone()), 1).unwrap();
        let loaded = reader.load_index().unwrap();

        let r0 = loaded.search(&v0, 1, 50).unwrap();
        assert_eq!(r0[0].0, 42);

        let r1 = loaded.search(&v1, 1, 50).unwrap();
        assert_eq!(r1[0].0, 7);
    }

    /// A reload must not change answers, not even by one id.
    ///
    /// hnswlib asserts exactly this (`bindings_test_recall.py:98` compares
    /// recall before and after a reload). The other segment tests check top-1
    /// ids and tombstone flags, which would not notice a reordered or truncated
    /// result set. This one compares every id and distance of every result, both
    /// through `load_index` and through the file-backed searcher.
    #[test]
    fn segment_reload_returns_identical_results() {
        use crate::persistence::directory::FsDirectory;

        let dimension = 8;
        let params = HNSWParams {
            m: 8,
            m_max: 16,
            ef_construction: 64,
            ef_search: 50,
            metric: crate::distance::DistanceMetric::L2,
            seed: Some(11),
            ..Default::default()
        };
        let mut index = HNSWIndex::with_params(dimension, params).unwrap();
        for doc_id in 0..400u32 {
            let vector: Vec<f32> = (0..dimension)
                .map(|d| ((doc_id as usize * 41 + d * 29) % 211) as f32 / 211.0)
                .collect();
            index.add(doc_id, vector).unwrap();
        }
        index.build().unwrap();

        let queries: Vec<Vec<f32>> = (0..40u32)
            .map(|q| {
                (0..dimension)
                    .map(|d| ((q as usize * 73 + d * 19) % 197) as f32 / 197.0)
                    .collect()
            })
            .collect();
        let expected: Vec<_> = queries
            .iter()
            .map(|query| index.search(query, 10, 50).unwrap())
            .collect();

        let dir = tempfile::tempdir().unwrap();
        HNSWSegmentWriter::new(Box::new(FsDirectory::new(dir.path()).unwrap()), 11)
            .write_hnsw_index(&index)
            .unwrap();
        let segment = dir.path().join("segments/segment_hnsw_11");

        let resident = HNSWSegmentReader::load(Box::new(FsDirectory::new(dir.path()).unwrap()), 11)
            .unwrap()
            .load_index()
            .unwrap();
        let mut searcher = crate::hnsw::HNSWSegmentSearcher::open(
            &segment,
            crate::hnsw::FileCacheConfig::with_budget(1 << 20),
        )
        .unwrap();

        for (query, want) in queries.iter().zip(&expected) {
            assert_eq!(&resident.search(query, 10, 50).unwrap(), want);
            assert_eq!(&searcher.search(query, 10, 50).unwrap(), want);
        }
    }

    #[test]
    fn test_hnsw_segment_preserves_tombstones() {
        let dim = 4;
        let mut index = HNSWIndex::new(dim, 8, 8).unwrap();

        let v0 = vec![1.0, 0.0, 0.0, 0.0];
        let v1 = vec![0.0, 1.0, 0.0, 0.0];
        index.add(42, v0.clone()).unwrap();
        index.add(7, v1.clone()).unwrap();
        index.build().unwrap();
        index.delete(42).unwrap();

        let mem = MemoryDirectory::new();

        let mut writer = HNSWSegmentWriter::new(Box::new(mem.clone()), 2);
        writer.write_hnsw_index(&index).unwrap();

        let reader = HNSWSegmentReader::load(Box::new(mem.clone()), 2).unwrap();
        let loaded = reader.load_index().unwrap();

        assert!(loaded.is_deleted(42));
        assert!(!loaded.is_deleted(7));
        assert_eq!(loaded.num_active(), 1);

        let result_ids: Vec<_> = loaded
            .search(&v0, 2, 50)
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert!(
            !result_ids.contains(&42),
            "segment roundtrip resurrected deleted doc_id 42"
        );
        assert_eq!(result_ids, vec![7]);
    }

    #[cfg(feature = "compact-hnsw")]
    #[test]
    fn compact_hnsw_segment_roundtrip_preserves_layout_contract() {
        let params = HNSWParams {
            m: 8,
            m_max: 16,
            ef_construction: 64,
            ef_search: 64,
            seed: Some(42),
            metric: crate::distance::DistanceMetric::L2,
            compact_upper_layers: true,
            ..Default::default()
        };
        let mut index = HNSWIndex::with_params(8, params).unwrap();
        for id in 0..512_u32 {
            let vector = (0..8)
                .map(|dimension| ((id as usize * 17 + dimension * 31) % 101) as f32 / 101.0)
                .collect();
            index.add(id, vector).unwrap();
        }
        index.build().unwrap();
        let occupancy = index.layer_occupancy();
        let query = index.get_vector(137).to_vec();
        let expected = index.search(&query, 20, 64).unwrap();

        let mem = MemoryDirectory::new();
        HNSWSegmentWriter::new(Box::new(mem.clone()), 3)
            .write_hnsw_index(&index)
            .unwrap();
        let mut loaded = HNSWSegmentReader::load(Box::new(mem), 3)
            .unwrap()
            .load_index()
            .unwrap();

        assert!(loaded.params.compact_upper_layers);
        assert_eq!(loaded.layer_occupancy(), occupancy);
        assert_eq!(loaded.search(&query, 20, 64).unwrap(), expected);
        assert!(loaded.delete_with_repair(137).is_err());
        loaded.delete(137).unwrap();
        assert!(loaded
            .search(&query, 20, 64)
            .unwrap()
            .iter()
            .all(|&(doc_id, _)| doc_id != 137));
    }

    #[test]
    fn truncated_params_bin_is_rejected() {
        use crate::persistence::directory::FsDirectory;

        let dimension = 4;
        let mut index = HNSWIndex::with_params(
            dimension,
            HNSWParams {
                m: 8,
                m_max: 16,
                metric: crate::distance::DistanceMetric::L2,
                ..HNSWParams::default()
            },
        )
        .unwrap();
        for id in 0..16u32 {
            let vector = vec![id as f32, 1.0, 0.0, 0.0];
            index.add(id, vector).unwrap();
        }
        index.build().unwrap();

        let dir = tempfile::tempdir().unwrap();
        HNSWSegmentWriter::new(Box::new(FsDirectory::new(dir.path()).unwrap()), 21)
            .write_hnsw_index(&index)
            .unwrap();

        // A v2/v3 params file truncated at exactly the v1 body length used to
        // load as a Cosine index instead of failing.
        let params = dir.path().join("segments/segment_hnsw_21/params.bin");
        let bytes = std::fs::read(&params).unwrap();
        assert!(bytes.len() > 26, "writer should append a sentinel");
        std::fs::write(&params, &bytes[..24]).unwrap();

        let error = HNSWSegmentReader::load(Box::new(FsDirectory::new(dir.path()).unwrap()), 21)
            .err()
            .expect("a truncated params file must not load");
        let typed: crate::RetrieveError = error.into();
        assert!(
            matches!(typed, crate::RetrieveError::FormatError(_)),
            "expected a format error, got {typed:?}"
        );
    }

    #[test]
    fn resaving_without_the_aos_layout_removes_the_stale_sidecar() {
        use crate::persistence::directory::FsDirectory;

        let dimension = 8;
        let mut index = HNSWIndex::with_params(
            dimension,
            HNSWParams {
                m: 8,
                m_max: 16,
                metric: crate::distance::DistanceMetric::L2,
                ..HNSWParams::default()
            },
        )
        .unwrap();
        for id in 0..32u32 {
            let vector: Vec<f32> = (0..dimension).map(|d| (id as usize + d) as f32).collect();
            index.add(id, vector).unwrap();
        }
        index.build().unwrap();

        let dir = tempfile::tempdir().unwrap();
        let mut writer =
            HNSWSegmentWriter::new(Box::new(FsDirectory::new(dir.path()).unwrap()), 22);
        writer.write_hnsw_index_aos(&index).unwrap();
        // Same segment id, SoA layout only: the sidecar must not survive.
        writer.write_hnsw_index(&index).unwrap();

        let segment = dir.path().join("segments/segment_hnsw_22");
        let searcher = crate::hnsw::HNSWSegmentSearcher::open(
            &segment,
            crate::hnsw::FileCacheConfig::with_budget(1 << 20),
        )
        .unwrap();
        assert!(
            searcher.uses_soa_fallback(),
            "a stale AoS sidecar would make the searcher serve the old vectors"
        );
    }

    #[test]
    fn aos_sidecar_roundtrips_through_segment_searcher() {
        use crate::persistence::directory::FsDirectory;

        let dimension = 8;
        let params = HNSWParams {
            m: 8,
            m_max: 16,
            ef_construction: 64,
            ef_search: 50,
            metric: crate::distance::DistanceMetric::L2,
            seed: Some(7),
            ..Default::default()
        };
        let mut index = HNSWIndex::with_params(dimension, params).unwrap();
        for internal in 0..256u32 {
            let vector: Vec<f32> = (0..dimension)
                .map(|d| ((internal as usize * 31 + d * 17) % 97) as f32 / 97.0)
                .collect();
            index.add(internal * 3 + 1, vector).unwrap();
        }
        index.build().unwrap();
        let query = index.get_vector(37).to_vec();
        let expected = index.search(&query, 10, 40).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let mut writer = HNSWSegmentWriter::new(Box::new(FsDirectory::new(dir.path()).unwrap()), 9);
        writer.write_hnsw_index_aos(&index).unwrap();

        let segment = dir.path().join("segments/segment_hnsw_9");
        let mut searcher = crate::hnsw::HNSWSegmentSearcher::open(
            &segment,
            crate::hnsw::FileCacheConfig::with_budget(1 << 20),
        )
        .unwrap();
        assert!(!searcher.uses_soa_fallback());
        assert_eq!(searcher.search(&query, 10, 40).unwrap(), expected);
    }
}
