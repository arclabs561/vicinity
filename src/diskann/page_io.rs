//! Page-co-located DiskANN node records.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use durability::mmap::{AccessPattern, MappedFile};

use crate::persistence::error::{PersistenceError, PersistenceResult};
use crate::RetrieveError;

use super::graph::DiskANNSearchDiagnostics;

const PAGE_MAGIC: &[u8; 8] = b"DANP\x00\x00\x00\x01";
const PAGE_ALIGNMENT: usize = 4096;
const HEADER_SIZE: usize = PAGE_ALIGNMENT;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DiskPageNode {
    pub(crate) doc_id: u32,
    pub(crate) vector: Vec<f32>,
    pub(crate) neighbors: Vec<u32>,
}

pub(crate) struct DiskPageWriter {
    writer: BufWriter<File>,
    dimension: usize,
    max_degree: usize,
    nodes_written: usize,
    num_nodes: usize,
    scratch: Vec<u8>,
}

impl DiskPageWriter {
    pub(crate) fn create(
        path: &Path,
        num_nodes: usize,
        dimension: usize,
        max_degree: usize,
        start_node: u32,
    ) -> PersistenceResult<Self> {
        validate_header(num_nodes, dimension, max_degree)?;
        let record_size = record_size(dimension, max_degree)?;
        let mut writer = BufWriter::new(File::create(path)?);

        writer.write_all(PAGE_MAGIC)?;
        write_u64(&mut writer, num_nodes as u64)?;
        write_u64(&mut writer, dimension as u64)?;
        write_u64(&mut writer, max_degree as u64)?;
        write_u64(&mut writer, start_node as u64)?;
        write_u64(&mut writer, record_size as u64)?;
        writer.write_all(&[0u8; HEADER_SIZE - 48])?;

        Ok(Self {
            writer,
            dimension,
            max_degree,
            nodes_written: 0,
            num_nodes,
            scratch: vec![0; record_size],
        })
    }

    pub(crate) fn write_node(
        &mut self,
        doc_id: u32,
        vector: &[f32],
        neighbors: &[u32],
    ) -> PersistenceResult<()> {
        if self.nodes_written >= self.num_nodes {
            return Err(PersistenceError::Serialization(
                "too many DiskANN page records".into(),
            ));
        }
        if vector.len() != self.dimension {
            return Err(PersistenceError::Serialization(format!(
                "vector dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
            )));
        }
        if neighbors.len() > self.max_degree {
            return Err(PersistenceError::Serialization(format!(
                "node degree {} exceeds max_degree {}",
                neighbors.len(),
                self.max_degree
            )));
        }

        self.scratch.fill(0);
        self.scratch[0..4].copy_from_slice(&doc_id.to_le_bytes());
        self.scratch[4..8].copy_from_slice(&(neighbors.len() as u32).to_le_bytes());

        let mut offset = 8;
        for &value in vector {
            self.scratch[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            offset += 4;
        }
        for &neighbor in neighbors {
            self.scratch[offset..offset + 4].copy_from_slice(&neighbor.to_le_bytes());
            offset += 4;
        }

        self.writer.write_all(&self.scratch)?;
        self.nodes_written += 1;
        Ok(())
    }

    pub(crate) fn flush(&mut self) -> PersistenceResult<()> {
        if self.nodes_written != self.num_nodes {
            return Err(PersistenceError::Serialization(format!(
                "expected {} DiskANN page records, wrote {}",
                self.num_nodes, self.nodes_written
            )));
        }
        self.writer.flush()?;
        Ok(())
    }
}

/// Experimental searcher over page-co-located DiskANN node records.
///
/// This reads `nodes.page`, written by [`super::graph::DiskANNIndex::save_page_layout`].
/// It is separate from [`super::graph::DiskANNSearcher`] so the legacy
/// `graph.index` plus `vectors.bin` format remains stable while page-layout
/// experiments are measured.
pub struct DiskANNPageSearcher {
    reader: DiskPageReader,
    visited_marks: Vec<u8>,
    visited_generation: u8,
}

impl DiskANNPageSearcher {
    /// Load a page-layout searcher from `nodes.page` using positional file reads.
    pub fn load(index_dir: &Path) -> Result<Self, RetrieveError> {
        Self::load_with_storage(index_dir, false)
    }

    /// Load a page-layout searcher from `nodes.page` using read-only mmap.
    pub fn load_mmap(index_dir: &Path) -> Result<Self, RetrieveError> {
        Self::load_with_storage(index_dir, true)
    }

    fn load_with_storage(index_dir: &Path, mmap: bool) -> Result<Self, RetrieveError> {
        let page_path = index_dir.join("nodes.page");
        let reader = if mmap {
            DiskPageReader::open_mmap(&page_path)?
        } else {
            DiskPageReader::open(&page_path)?
        };
        let num_nodes = reader.num_nodes;
        Ok(Self {
            reader,
            visited_marks: vec![0; num_nodes],
            visited_generation: 1,
        })
    }

    /// Search for `k` approximate nearest neighbors.
    pub fn search(
        &mut self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.search_with_diagnostics(query, k, ef_search)
            .map(|(results, _)| results)
    }

    /// Search and return logical page-read diagnostics.
    pub fn search_with_diagnostics(
        &mut self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<(Vec<(u32, f32)>, DiskANNSearchDiagnostics), RetrieveError> {
        if query.len() != self.reader.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.reader.dimension,
            });
        }

        let ef = ef_search.max(k);
        let mut diagnostics = DiskANNSearchDiagnostics {
            ef_search: ef,
            ..DiskANNSearchDiagnostics::default()
        };

        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        self.reset_visited();
        let mut visited_count = 0usize;
        let mut frontier: BinaryHeap<Reverse<PageFrontierCandidate>> =
            BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<PageCandidate> = BinaryHeap::with_capacity(ef + 1);

        let start = self.reader.get_node(self.reader.start_node)?;
        diagnostics.page_reads += 1;
        let start_dist = crate::simd::l2_distance_squared(query, &start.vector);
        frontier.push(Reverse(PageFrontierCandidate {
            id: self.reader.start_node,
            dist: start_dist,
            neighbors: start.neighbors,
        }));
        results.push(PageCandidate {
            doc_id: start.doc_id,
            dist: start_dist,
        });
        self.insert_visited(self.reader.start_node)?;
        visited_count += 1;

        while let Some(Reverse(current)) = frontier.pop() {
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if current.dist >= worst.dist {
                        break;
                    }
                }
            }

            for neighbor in current.neighbors {
                if !self.insert_visited(neighbor)? {
                    continue;
                }
                visited_count += 1;

                let node = self.reader.get_node(neighbor)?;
                diagnostics.page_reads += 1;
                let dist = crate::simd::l2_distance_squared(query, &node.vector);
                frontier.push(Reverse(PageFrontierCandidate {
                    id: neighbor,
                    dist,
                    neighbors: node.neighbors,
                }));
                results.push(PageCandidate {
                    doc_id: node.doc_id,
                    dist,
                });
                if results.len() > ef {
                    results.pop();
                }
            }
        }

        diagnostics.visited_nodes = visited_count;
        diagnostics.retained_candidates = results.len();
        diagnostics.page_bytes = diagnostics.page_reads * self.reader.record_size;

        let mut result_vec = results.into_vec();
        result_vec.sort_unstable_by(|a, b| a.dist.total_cmp(&b.dist));
        let results = result_vec
            .into_iter()
            .map(|c| (c.doc_id, c.dist))
            .take(k)
            .collect();

        Ok((results, diagnostics))
    }

    fn reset_visited(&mut self) {
        if let Some(next) = self.visited_generation.checked_add(1) {
            self.visited_generation = next;
        } else {
            self.visited_marks.fill(0);
            self.visited_generation = 1;
        }
    }

    fn insert_visited(&mut self, node_id: u32) -> Result<bool, RetrieveError> {
        let idx = node_id as usize;
        let mark = self
            .visited_marks
            .get_mut(idx)
            .ok_or(RetrieveError::OutOfBounds(idx))?;
        if *mark == self.visited_generation {
            Ok(false)
        } else {
            *mark = self.visited_generation;
            Ok(true)
        }
    }
}

#[derive(PartialEq)]
struct PageFrontierCandidate {
    id: u32,
    dist: f32,
    neighbors: Vec<u32>,
}

impl Eq for PageFrontierCandidate {}

impl Ord for PageFrontierCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.total_cmp(&other.dist)
    }
}

impl PartialOrd for PageFrontierCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, PartialEq)]
struct PageCandidate {
    doc_id: u32,
    dist: f32,
}

impl Eq for PageCandidate {}

impl Ord for PageCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.total_cmp(&other.dist)
    }
}

impl PartialOrd for PageCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

enum PageStorage {
    File(File),
    Mmap(Box<MappedFile>),
}

pub(crate) struct DiskPageReader {
    storage: PageStorage,
    read_buf: Vec<u8>,
    vec_buf: Vec<f32>,
    pub(crate) num_nodes: usize,
    pub(crate) dimension: usize,
    pub(crate) max_degree: usize,
    pub(crate) start_node: u32,
    record_size: usize,
}

impl DiskPageReader {
    pub(crate) fn open(path: &Path) -> PersistenceResult<Self> {
        let mut file = File::open(path)?;
        let header = read_header_from_file(&mut file)?;
        Ok(Self {
            storage: PageStorage::File(file),
            read_buf: vec![0; header.record_size],
            vec_buf: vec![0.0; header.dimension],
            num_nodes: header.num_nodes,
            dimension: header.dimension,
            max_degree: header.max_degree,
            start_node: header.start_node,
            record_size: header.record_size,
        })
    }

    pub(crate) fn open_mmap(path: &Path) -> PersistenceResult<Self> {
        let mapped = MappedFile::open(path, AccessPattern::Random)
            .map_err(|e| PersistenceError::Io(std::io::Error::other(e.to_string())))?;
        let header = read_header_from_bytes(mapped.as_slice())?;
        validate_file_len(mapped.as_slice().len(), &header)?;
        Ok(Self {
            storage: PageStorage::Mmap(Box::new(mapped)),
            read_buf: Vec::new(),
            vec_buf: vec![0.0; header.dimension],
            num_nodes: header.num_nodes,
            dimension: header.dimension,
            max_degree: header.max_degree,
            start_node: header.start_node,
            record_size: header.record_size,
        })
    }

    pub(crate) fn get_node(&mut self, node_id: u32) -> Result<DiskPageNode, RetrieveError> {
        let idx = node_id as usize;
        if idx >= self.num_nodes {
            return Err(RetrieveError::OutOfBounds(idx));
        }
        let offset = HEADER_SIZE
            .checked_add(idx * self.record_size)
            .ok_or_else(|| RetrieveError::FormatError("page record offset overflow".into()))?;

        match &mut self.storage {
            PageStorage::File(file) => {
                crate::file_io::read_exact_at(file, offset as u64, &mut self.read_buf)?;
                decode_node(
                    &self.read_buf,
                    self.dimension,
                    self.max_degree,
                    &mut self.vec_buf,
                )
            }
            PageStorage::Mmap(mapped) => {
                let end = offset
                    .checked_add(self.record_size)
                    .ok_or_else(|| RetrieveError::FormatError("page record end overflow".into()))?;
                let bytes = mapped.as_slice();
                let record = bytes.get(offset..end).ok_or_else(|| {
                    RetrieveError::FormatError("page record extends past mapped file".into())
                })?;
                decode_node(record, self.dimension, self.max_degree, &mut self.vec_buf)
            }
        }
    }
}

struct PageHeader {
    num_nodes: usize,
    dimension: usize,
    max_degree: usize,
    start_node: u32,
    record_size: usize,
}

fn read_header_from_file(file: &mut File) -> PersistenceResult<PageHeader> {
    let mut header = [0u8; HEADER_SIZE];
    file.read_exact(&mut header)?;
    read_header_from_bytes(&header)
}

fn read_header_from_bytes(bytes: &[u8]) -> PersistenceResult<PageHeader> {
    if bytes.len() < HEADER_SIZE {
        return Err(PersistenceError::Format(
            "DiskANN page file is too short".into(),
        ));
    }
    if &bytes[0..8] != PAGE_MAGIC {
        return Err(PersistenceError::Format("Invalid DiskANN page file".into()));
    }

    let num_nodes = read_u64_at(bytes, 8)? as usize;
    let dimension = read_u64_at(bytes, 16)? as usize;
    let max_degree = read_u64_at(bytes, 24)? as usize;
    let start_node = read_u64_at(bytes, 32)? as u32;
    let stored_record_size = read_u64_at(bytes, 40)? as usize;
    validate_header(num_nodes, dimension, max_degree)?;
    let expected_record_size = record_size(dimension, max_degree)?;
    if stored_record_size != expected_record_size {
        return Err(PersistenceError::Format(format!(
            "record size mismatch: expected {}, got {}",
            expected_record_size, stored_record_size
        )));
    }
    Ok(PageHeader {
        num_nodes,
        dimension,
        max_degree,
        start_node,
        record_size: stored_record_size,
    })
}

fn validate_header(num_nodes: usize, dimension: usize, max_degree: usize) -> PersistenceResult<()> {
    const MAX_NODES: usize = 100_000_000;
    const MAX_DIMENSION: usize = 1_000_000;
    const MAX_DEGREE: usize = 65_536;

    if num_nodes > MAX_NODES {
        return Err(PersistenceError::Format(format!(
            "unreasonable node count: {num_nodes}"
        )));
    }
    if dimension == 0 || dimension > MAX_DIMENSION {
        return Err(PersistenceError::Format(format!(
            "unreasonable dimension: {dimension}"
        )));
    }
    if max_degree > MAX_DEGREE {
        return Err(PersistenceError::Format(format!(
            "unreasonable max degree: {max_degree}"
        )));
    }
    Ok(())
}

fn validate_file_len(len: usize, header: &PageHeader) -> PersistenceResult<()> {
    let expected_len = HEADER_SIZE
        .checked_add(
            header
                .num_nodes
                .checked_mul(header.record_size)
                .ok_or_else(|| PersistenceError::Format("page file size overflow".into()))?,
        )
        .ok_or_else(|| PersistenceError::Format("page file size overflow".into()))?;
    if len < expected_len {
        return Err(PersistenceError::Format(format!(
            "truncated DiskANN page file: expected at least {expected_len} bytes, got {len}"
        )));
    }
    Ok(())
}

fn record_size(dimension: usize, max_degree: usize) -> PersistenceResult<usize> {
    let payload =
        8usize
            .checked_add(dimension.checked_mul(4).ok_or_else(|| {
                PersistenceError::Format("page vector byte count overflow".into())
            })?)
            .and_then(|n| n.checked_add(max_degree.checked_mul(4)?))
            .ok_or_else(|| PersistenceError::Format("page record size overflow".into()))?;
    Ok(payload.next_multiple_of(PAGE_ALIGNMENT))
}

fn decode_node(
    record: &[u8],
    dimension: usize,
    max_degree: usize,
    vec_buf: &mut [f32],
) -> Result<DiskPageNode, RetrieveError> {
    let doc_id = read_u32_at_record(record, 0)?;
    let degree = read_u32_at_record(record, 4)? as usize;
    if degree > max_degree {
        return Err(RetrieveError::FormatError(
            "invalid node degree in DiskANN page file".into(),
        ));
    }

    let mut offset = 8;
    for value in vec_buf.iter_mut().take(dimension) {
        *value = f32::from_le_bytes([
            record[offset],
            record[offset + 1],
            record[offset + 2],
            record[offset + 3],
        ]);
        offset += 4;
    }
    let mut neighbors = Vec::with_capacity(degree);
    for _ in 0..degree {
        neighbors.push(read_u32_at_record(record, offset)?);
        offset += 4;
    }

    Ok(DiskPageNode {
        doc_id,
        vector: vec_buf.to_vec(),
        neighbors,
    })
}

fn read_u32_at_record(record: &[u8], offset: usize) -> Result<u32, RetrieveError> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| RetrieveError::FormatError("page record offset overflow".into()))?;
    let chunk = record
        .get(offset..end)
        .ok_or_else(|| RetrieveError::FormatError("truncated page record".into()))?;
    Ok(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
}

fn read_u64_at(bytes: &[u8], offset: usize) -> PersistenceResult<u64> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| PersistenceError::Format("header offset overflow".into()))?;
    let chunk = bytes
        .get(offset..end)
        .ok_or_else(|| PersistenceError::Format("truncated DiskANN page header".into()))?;
    Ok(u64::from_le_bytes([
        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
    ]))
}

fn write_u64(mut writer: impl Write, value: u64) -> PersistenceResult<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diskann::graph::{DiskANNIndex, DiskANNParams};

    fn generate_vectors(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32) - 0.5
        };

        (0..n).map(|_| (0..d).map(|_| next()).collect()).collect()
    }

    #[test]
    fn page_records_round_trip_file_and_mmap() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nodes.page");
        let mut writer = DiskPageWriter::create(&path, 2, 3, 4, 1).unwrap();
        writer.write_node(10, &[1.0, 2.0, 3.0], &[1, 2, 3]).unwrap();
        writer.write_node(11, &[4.0, 5.0, 6.0], &[0]).unwrap();
        writer.flush().unwrap();

        assert_eq!(
            std::fs::metadata(&path).unwrap().len() as usize,
            HEADER_SIZE + PAGE_ALIGNMENT * 2
        );
        assert_eq!(HEADER_SIZE % PAGE_ALIGNMENT, 0);

        let mut file_reader = DiskPageReader::open(&path).unwrap();
        assert_eq!(file_reader.num_nodes, 2);
        assert_eq!(file_reader.dimension, 3);
        assert_eq!(file_reader.max_degree, 4);
        assert_eq!(file_reader.start_node, 1);
        assert_eq!(
            file_reader.get_node(0).unwrap(),
            DiskPageNode {
                doc_id: 10,
                vector: vec![1.0, 2.0, 3.0],
                neighbors: vec![1, 2, 3],
            }
        );

        let mut mmap_reader = DiskPageReader::open_mmap(&path).unwrap();
        assert_eq!(
            mmap_reader.get_node(1).unwrap(),
            DiskPageNode {
                doc_id: 11,
                vector: vec![4.0, 5.0, 6.0],
                neighbors: vec![0],
            }
        );
    }

    #[test]
    fn page_writer_rejects_partial_flush() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nodes.page");
        let mut writer = DiskPageWriter::create(&path, 2, 3, 4, 0).unwrap();
        writer.write_node(10, &[1.0, 2.0, 3.0], &[1]).unwrap();

        let err = writer.flush().unwrap_err();
        assert!(err.to_string().contains("expected 2 DiskANN page records"));
    }

    #[test]
    fn page_writer_rejects_bad_degree() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nodes.page");
        let mut writer = DiskPageWriter::create(&path, 1, 3, 1, 0).unwrap();

        let err = writer
            .write_node(10, &[1.0, 2.0, 3.0], &[0, 1])
            .unwrap_err();
        assert!(err.to_string().contains("exceeds max_degree"));
    }

    #[test]
    fn page_search_matches_heap_and_reads_each_visited_node_once() {
        let n = 300;
        let d = 16;
        let k = 10;
        let ef = 32;
        let vectors = generate_vectors(n, d, 99);
        let queries = generate_vectors(6, d, 199);
        let params = DiskANNParams {
            m: 12,
            ef_construction: 40,
            alpha: 1.2,
            ef_search: ef,
            seed: Some(7),
        };

        let mut index = DiskANNIndex::new(d, params).unwrap();
        for (i, vector) in vectors.iter().enumerate() {
            index.add_slice(i as u32, vector).unwrap();
        }
        index.build().unwrap();

        let dir = tempfile::tempdir().unwrap();
        let index_path = dir.path().join("diskann_page");
        index.save_page_layout(&index_path).unwrap();
        let mut page_searcher = DiskANNPageSearcher::load(&index_path).unwrap();
        let mut page_mmap_searcher = DiskANNPageSearcher::load_mmap(&index_path).unwrap();

        for query in &queries {
            let expected = index.search(query, k, ef).unwrap();
            let (file_results, diagnostics) =
                page_searcher.search_with_diagnostics(query, k, ef).unwrap();
            assert_eq!(file_results, expected);
            assert_eq!(diagnostics.page_reads, diagnostics.visited_nodes);
            assert_eq!(
                diagnostics.page_bytes,
                diagnostics.page_reads * PAGE_ALIGNMENT
            );
            assert_eq!(diagnostics.graph_reads, 0);
            assert_eq!(diagnostics.vector_reads, 0);

            let mmap_results = page_mmap_searcher.search(query, k, ef).unwrap();
            assert_eq!(mmap_results, expected);
        }
    }
}
