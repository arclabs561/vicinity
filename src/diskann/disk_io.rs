//! Disk I/O optimization for DiskANN.
//!
//! Handles sequential file access patterns and memory-mapped files.

use crate::persistence::error::{PersistenceError, PersistenceResult};
use crate::RetrieveError;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

// Magic bytes for DiskANN graph file: "DANN" + version 1
const GRAPH_MAGIC: &[u8; 8] = b"DANN\x00\x00\x00\x01";

/// Writer for DiskANN graph format.
///
/// Format:
/// - Header (64 bytes):
///   - Magic (8 bytes)
///   - Num nodes (8 bytes)
///   - Max degree (8 bytes)
///   - Start node (8 bytes)
///   - Padding (32 bytes)
/// - Nodes:
///   - For each node:
///     - Degree (4 bytes)
///     - Neighbors (max_degree * 4 bytes)
pub struct DiskGraphWriter {
    writer: BufWriter<File>,
    num_nodes: usize,
    max_degree: usize,
    start_node: u32,
}

impl DiskGraphWriter {
    /// Create a new graph writer.
    pub fn new(
        path: &Path,
        num_nodes: usize,
        max_degree: usize,
        start_node: u32,
    ) -> PersistenceResult<Self> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        writer.write_all(GRAPH_MAGIC)?;
        writer.write_all(&(num_nodes as u64).to_le_bytes())?;
        writer.write_all(&(max_degree as u64).to_le_bytes())?;
        writer.write_all(&(start_node as u64).to_le_bytes())?;
        writer.write_all(&[0u8; 32])?; // Padding

        Ok(Self {
            writer,
            num_nodes,
            max_degree,
            start_node,
        })
    }

    /// Write adjacency list for a node.
    pub fn write_adjacency(&mut self, neighbors: &[u32]) -> PersistenceResult<()> {
        if neighbors.len() > self.max_degree {
            return Err(PersistenceError::Serialization(format!(
                "Node degree {} exceeds max_degree {}",
                neighbors.len(),
                self.max_degree
            )));
        }

        // Write actual degree
        self.writer
            .write_all(&(neighbors.len() as u32).to_le_bytes())?;

        // Write neighbors
        for &neighbor in neighbors {
            self.writer.write_all(&neighbor.to_le_bytes())?;
        }

        // Write padding (zeros) to maintain fixed record size
        let padding_len = (self.max_degree - neighbors.len()) * 4;
        // Inefficient for large padding, but simple
        for _ in 0..padding_len {
            self.writer.write_all(&[0u8])?;
        }

        Ok(())
    }

    /// Finalize writing.
    pub fn flush(&mut self) -> PersistenceResult<()> {
        self.writer.flush()?;
        Ok(())
    }
}

/// Reader for DiskANN graph format.
///
/// Uses standard IO with seeking (pread equivalent) for simplicity and robustness.
/// Can be upgraded to mmap later.
pub struct DiskGraphReader {
    file: File,
    /// Total number of nodes in the graph.
    pub num_nodes: usize,
    /// Maximum out-degree per node.
    pub max_degree: usize,
    /// Entry point for graph search.
    pub start_node: u32,
    header_size: u64,
    record_size: u64,
}

impl DiskGraphReader {
    /// Open a graph file.
    pub fn open(path: &Path) -> PersistenceResult<Self> {
        let mut file = File::open(path)?;

        // Read header
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if &magic != GRAPH_MAGIC {
            return Err(PersistenceError::Format(
                "Invalid DiskANN graph file".to_string(),
            ));
        }

        let mut buf_u64 = [0u8; 8];

        file.read_exact(&mut buf_u64)?;
        let num_nodes = u64::from_le_bytes(buf_u64) as usize;

        file.read_exact(&mut buf_u64)?;
        let max_degree = u64::from_le_bytes(buf_u64) as usize;

        file.read_exact(&mut buf_u64)?;
        let start_node = u64::from_le_bytes(buf_u64) as u32;

        // Skip padding
        file.seek(SeekFrom::Current(32))?;

        // Guard against crafted files that claim enormous graph sizes.
        const MAX_NODES: usize = 100_000_000; // 100M nodes
        const MAX_DEGREE: usize = 65_536; // 64K degree

        if num_nodes > MAX_NODES {
            return Err(PersistenceError::Format(format!(
                "unreasonable node count: {}",
                num_nodes
            )));
        }
        if max_degree > MAX_DEGREE {
            return Err(PersistenceError::Format(format!(
                "unreasonable max degree: {}",
                max_degree
            )));
        }

        let header_size = 8 + 8 + 8 + 8 + 32;
        // record_size = 4 + max_degree * 4; check for overflow.
        let record_size = (max_degree as u64)
            .checked_mul(4)
            .and_then(|n| n.checked_add(4))
            .ok_or_else(|| PersistenceError::Format("record size overflow".into()))?;

        Ok(Self {
            file,
            num_nodes,
            max_degree,
            start_node,
            header_size,
            record_size,
        })
    }

    /// Read neighbors for a node.
    pub fn get_neighbors(&mut self, node_id: u32) -> Result<Vec<u32>, RetrieveError> {
        if node_id as usize >= self.num_nodes {
            return Err(RetrieveError::OutOfBounds(node_id as usize));
        }

        let offset = self.header_size + (node_id as u64 * self.record_size);

        // Seek to node record.
        // Safety: `&mut self` prevents concurrent calls. For parallel search,
        // create one DiskGraphReader per thread (each with its own file handle).
        self.file.seek(SeekFrom::Start(offset))?;

        // Read degree
        let mut degree_buf = [0u8; 4];
        self.file.read_exact(&mut degree_buf)?;
        let degree = u32::from_le_bytes(degree_buf) as usize;

        if degree > self.max_degree {
            return Err(RetrieveError::FormatError(
                "invalid node degree in graph file".into(),
            ));
        }

        // Read neighbors
        let mut neighbors = Vec::with_capacity(degree);
        let mut neighbor_buf = [0u8; 4];

        for _ in 0..degree {
            self.file.read_exact(&mut neighbor_buf)?;
            neighbors.push(u32::from_le_bytes(neighbor_buf));
        }

        Ok(neighbors)
    }
}
