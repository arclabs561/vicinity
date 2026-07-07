//! Disk I/O optimization for DiskANN.
//!
//! Handles sequential file access patterns and memory-mapped files.

use crate::persistence::error::{PersistenceError, PersistenceResult};
use crate::RetrieveError;
use durability::mmap::{AccessPattern, MappedFile};
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
    max_degree: usize,
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

        // num_nodes already written to header; not needed at runtime.
        let _ = num_nodes;

        Ok(Self { writer, max_degree })
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

enum DiskGraphStorage {
    File(File),
    Mmap(Box<MappedFile>),
}

pub(crate) fn read_exact_at(file: &mut File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    read_exact_at_impl(file, offset, buf)
}

#[cfg(unix)]
fn read_exact_at_impl(file: &mut File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind};
    use std::os::unix::fs::FileExt;

    let mut read = 0;
    while read < buf.len() {
        let n = file.read_at(&mut buf[read..], offset + read as u64)?;
        if n == 0 {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                "failed to fill buffer",
            ));
        }
        read += n;
    }
    Ok(())
}

#[cfg(windows)]
fn read_exact_at_impl(file: &mut File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind};
    use std::os::windows::fs::FileExt;

    let mut read = 0;
    while read < buf.len() {
        let n = file.seek_read(&mut buf[read..], offset + read as u64)?;
        if n == 0 {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                "failed to fill buffer",
            ));
        }
        read += n;
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn read_exact_at_impl(file: &mut File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(buf)
}

/// Reader for DiskANN graph format.
///
/// Uses standard IO by default and can be opened in mmap mode with
/// [`DiskGraphReader::open_mmap`]. The mmap path preserves the same on-disk
/// format and avoids per-record seek/read syscalls.
pub struct DiskGraphReader {
    storage: DiskGraphStorage,
    read_buf: Vec<u8>,
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
            storage: DiskGraphStorage::File(file),
            read_buf: Vec::new(),
            num_nodes,
            max_degree,
            start_node,
            header_size,
            record_size,
        })
    }

    /// Open a graph file through a read-only memory map.
    pub fn open_mmap(path: &Path) -> PersistenceResult<Self> {
        let mapped = MappedFile::open(path, AccessPattern::Random)
            .map_err(|e| PersistenceError::Io(std::io::Error::other(e.to_string())))?;
        let bytes = mapped.as_slice();
        if bytes.len() < 64 {
            return Err(PersistenceError::Format(
                "DiskANN graph file is too short".into(),
            ));
        }
        if &bytes[0..8] != GRAPH_MAGIC {
            return Err(PersistenceError::Format(
                "Invalid DiskANN graph file".to_string(),
            ));
        }

        let num_nodes = read_u64_at(bytes, 8)? as usize;
        let max_degree = read_u64_at(bytes, 16)? as usize;
        let start_node = read_u64_at(bytes, 24)? as u32;

        validate_header(num_nodes, max_degree)?;

        let header_size = 8 + 8 + 8 + 8 + 32;
        let record_size = record_size(max_degree)?;
        let expected_len = header_size as usize
            + (num_nodes as u64)
                .checked_mul(record_size)
                .ok_or_else(|| PersistenceError::Format("graph file size overflow".into()))?
                as usize;
        if bytes.len() < expected_len {
            return Err(PersistenceError::Format(format!(
                "truncated DiskANN graph file: expected at least {} bytes, got {}",
                expected_len,
                bytes.len()
            )));
        }

        Ok(Self {
            storage: DiskGraphStorage::Mmap(Box::new(mapped)),
            read_buf: Vec::new(),
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
        match &mut self.storage {
            DiskGraphStorage::File(file) => {
                Self::get_neighbors_file(file, offset, self.max_degree, &mut self.read_buf)
            }
            DiskGraphStorage::Mmap(mapped) => Self::get_neighbors_mmap(
                mapped.as_slice(),
                offset,
                self.max_degree,
                self.record_size,
            ),
        }
    }

    fn get_neighbors_file(
        file: &mut File,
        offset: u64,
        max_degree: usize,
        read_buf: &mut Vec<u8>,
    ) -> Result<Vec<u32>, RetrieveError> {
        // Safety: `&mut self` prevents concurrent calls. For parallel search,
        // create one DiskGraphReader per thread (each with its own file handle).
        let mut degree_buf = [0u8; 4];
        read_exact_at(file, offset, &mut degree_buf)?;
        let degree = u32::from_le_bytes(degree_buf) as usize;

        if degree > max_degree {
            return Err(RetrieveError::FormatError(
                "invalid node degree in graph file".into(),
            ));
        }

        let neighbor_bytes = degree
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| RetrieveError::FormatError("neighbor byte count overflow".into()))?;
        read_buf.resize(neighbor_bytes, 0);
        read_exact_at(file, offset + 4, read_buf)?;

        let mut neighbors = Vec::with_capacity(degree);
        for chunk in read_buf.chunks_exact(std::mem::size_of::<u32>()) {
            neighbors.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        Ok(neighbors)
    }

    fn get_neighbors_mmap(
        bytes: &[u8],
        offset: u64,
        max_degree: usize,
        record_size: u64,
    ) -> Result<Vec<u32>, RetrieveError> {
        let offset = usize::try_from(offset)
            .map_err(|_| RetrieveError::FormatError("graph offset overflow".into()))?;
        let record_end = offset
            .checked_add(record_size as usize)
            .ok_or_else(|| RetrieveError::FormatError("graph record end overflow".into()))?;
        if record_end > bytes.len() {
            return Err(RetrieveError::FormatError(
                "graph record extends past mapped file".into(),
            ));
        }

        let degree = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        if degree > max_degree {
            return Err(RetrieveError::FormatError(
                "invalid node degree in graph file".into(),
            ));
        }

        let neighbors_start = offset + 4;
        let neighbors_end = neighbors_start + degree * 4;
        if neighbors_end > record_end {
            return Err(RetrieveError::FormatError(
                "graph neighbor list extends past record".into(),
            ));
        }

        let mut neighbors = Vec::with_capacity(degree);
        for chunk in bytes[neighbors_start..neighbors_end].chunks_exact(4) {
            neighbors.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(neighbors)
    }
}

fn validate_header(num_nodes: usize, max_degree: usize) -> PersistenceResult<()> {
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
    Ok(())
}

fn record_size(max_degree: usize) -> PersistenceResult<u64> {
    (max_degree as u64)
        .checked_mul(4)
        .and_then(|n| n.checked_add(4))
        .ok_or_else(|| PersistenceError::Format("record size overflow".into()))
}

fn read_u64_at(bytes: &[u8], offset: usize) -> PersistenceResult<u64> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| PersistenceError::Format("offset overflow".into()))?;
    let chunk = bytes
        .get(offset..end)
        .ok_or_else(|| PersistenceError::Format("truncated DiskANN graph header".into()))?;
    Ok(u64::from_le_bytes([
        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
    ]))
}
