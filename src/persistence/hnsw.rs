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

    /// Write an HNSW index to disk.
    ///
    /// Format:
    /// - `vectors.bin`: Vector data (SoA layout, same as dense segment)
    /// - `doc_ids.bin`: External doc_ids aligned with internal indices
    /// - `tombstones.bin`: One byte per internal id, `1` for soft-deleted
    /// - `layers.bin`: Graph layers (serialized neighbor lists)
    /// - `layer_assignments.bin`: Layer assignment for each vector
    /// - `params.bin`: HNSW parameters
    /// - `metadata.bin`: Index metadata (dimension, num_vectors, etc.)
    pub fn write_hnsw_index(&mut self, index: &HNSWIndex) -> PersistenceResult<()> {
        let segment_dir = format!("segments/segment_hnsw_{}", self.segment_id);
        self.directory.create_dir_all(&segment_dir)?;

        // Write vectors in SoA layout (same as dense segment).
        //
        // NOTE: `HNSWIndex` stores vectors in-memory in AoS order (each vector contiguous),
        // but on disk we store SoA for cache/SIMD-friendly access and consistency with
        // `DenseSegmentWriter`.
        let vectors_path = format!("{}/vectors.bin", segment_dir);
        let mut vectors_file = self.directory.create_file(&vectors_path)?;
        for d in 0..index.dimension {
            for v_idx in 0..index.num_vectors {
                let aos_idx = v_idx * index.dimension + d;
                vectors_file.write_all(&index.vectors[aos_idx].to_le_bytes())?;
            }
        }
        vectors_file.flush()?;

        // Write doc_ids (external IDs aligned with internal insertion order).
        //
        // This is the critical "KeyedVectors separation" bit: it ensures searches
        // on a loaded index return the same external IDs that were originally added.
        let doc_ids_path = format!("{}/doc_ids.bin", segment_dir);
        let mut doc_ids_file = self.directory.create_file(&doc_ids_path)?;
        for &doc_id in &index.doc_ids {
            doc_ids_file.write_all(&doc_id.to_le_bytes())?;
        }
        doc_ids_file.flush()?;

        // Write tombstone flags (one byte per internal id).
        let tombstones_path = format!("{}/tombstones.bin", segment_dir);
        let mut tombstones_file = self.directory.create_file(&tombstones_path)?;
        tombstones_file.write_all(&index.tombstone_flags())?;
        tombstones_file.flush()?;

        // Write layer assignments
        let assignments_path = format!("{}/layer_assignments.bin", segment_dir);
        let mut assignments_file = self.directory.create_file(&assignments_path)?;
        for &assignment in &index.layer_assignments {
            assignments_file.write_all(&[assignment])?;
        }
        assignments_file.flush()?;

        // Write graph layers
        let layers_path = format!("{}/layers.bin", segment_dir);
        let mut layers_file = self.directory.create_file(&layers_path)?;

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

        // Write parameters
        let params_path = format!("{}/params.bin", segment_dir);
        let mut params_file = self.directory.create_file(&params_path)?;
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

        // Write metadata.
        //
        // Layout (v1):
        //   [0..8]   HNSW_SEGMENT_MAGIC  (8 bytes, "VCNHNSW\x01")
        //   [8..12]  FORMAT_VERSION      (u32 LE)
        //   [12..16] dimension           (u32 LE)
        //   [16..20] num_vectors         (u32 LE)
        //   [20]     is_built            (u8 flag)
        //
        // Legacy v0 (vicinity 0.6.x): no magic/version prefix; the first 4
        // bytes are the dimension u32.
        let metadata_path = format!("{}/metadata.bin", segment_dir);
        let mut metadata_file = self.directory.create_file(&metadata_path)?;
        metadata_file.write_all(&HNSW_SEGMENT_MAGIC)?;
        metadata_file.write_all(&FORMAT_VERSION.to_le_bytes())?;
        metadata_file.write_all(&(index.dimension as u32).to_le_bytes())?;
        metadata_file.write_all(&(index.num_vectors as u32).to_le_bytes())?;
        metadata_file.write_all(&[if index.is_built() { 1 } else { 0 }])?;
        metadata_file.flush()?;

        Ok(())
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
            // we propagate a Format error rather than guessing further.
            let dim_bytes: [u8; 4] = [magic_buf[0], magic_buf[1], magic_buf[2], magic_buf[3]];
            let num_vec_bytes: [u8; 4] = [magic_buf[4], magic_buf[5], magic_buf[6], magic_buf[7]];
            let mut built_byte = [0u8; 1];
            if metadata_file.read_exact(&mut built_byte).is_err() {
                return Err(crate::persistence::error::PersistenceError::Format(
                    "bad magic and legacy decode failed: file too short for v0 header".into(),
                ));
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

        // Load parameters
        let params_path = format!("{}/params.bin", segment_dir);
        let mut params_file = directory.open_file(&params_path)?;
        let mut m_bytes = [0u8; 4];
        let mut m_max_bytes = [0u8; 4];
        let mut m_l_bytes = [0u8; 8];
        let mut ef_construction_bytes = [0u8; 4];
        let mut ef_search_bytes = [0u8; 4];

        params_file.read_exact(&mut m_bytes)?;
        params_file.read_exact(&mut m_max_bytes)?;
        params_file.read_exact(&mut m_l_bytes)?;
        params_file.read_exact(&mut ef_construction_bytes)?;
        params_file.read_exact(&mut ef_search_bytes)?;

        // v2 fields: metric + auto_normalize (optional, backward-compat with v1 files)
        let mut metric_byte = [0u8; 1];
        let metric = if params_file.read_exact(&mut metric_byte).is_ok() {
            match metric_byte[0] {
                0 => crate::distance::DistanceMetric::L2,
                1 => crate::distance::DistanceMetric::Cosine,
                2 => crate::distance::DistanceMetric::Angular,
                3 => crate::distance::DistanceMetric::InnerProduct,
                other => {
                    return Err(crate::persistence::error::PersistenceError::Format(
                        format!("unknown distance metric: {}", other),
                    ));
                }
            }
        } else {
            // Legacy v1 file without metric byte: default to Cosine
            crate::distance::DistanceMetric::Cosine
        };

        let mut auto_norm_byte = [0u8; 1];
        let auto_normalize =
            params_file.read_exact(&mut auto_norm_byte).is_ok() && auto_norm_byte[0] != 0;

        #[cfg(feature = "compact-hnsw")]
        let compact_upper_layers = {
            let mut compact_byte = [0u8; 1];
            params_file.read_exact(&mut compact_byte).is_ok() && compact_byte[0] != 0
        };

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
    use crate::persistence::directory::MemoryDirectory;

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
}
