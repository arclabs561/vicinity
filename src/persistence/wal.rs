//! Write-ahead log (WAL) for incremental updates.
//!
//! This module is intentionally a **thin shim** over `durability::walog`.
//! Rationale: `durability` is the canonical L3 persistence substrate for “segment + WAL +
//! checkpoint” systems. Keeping a second WAL implementation in `vicinity` is abstraction leakage:
//! it causes format drift (e.g., segment ordering, torn-tail handling) and duplicates tricky
//! recovery logic.

use crate::persistence::directory as vicinity_dir;
use crate::persistence::error::{PersistenceError, PersistenceResult};
use std::sync::Arc;

pub use crate::persistence::format::GraphWalEntry;
pub use durability::walog::{WalEntry, WalRecord, WalReplayMode};

fn to_durability_err(e: PersistenceError) -> durability::PersistenceError {
    match e {
        PersistenceError::Io(e) => durability::PersistenceError::Io(e),
        PersistenceError::Format(s) => durability::PersistenceError::Format(s),
        PersistenceError::Serialization(s) => durability::PersistenceError::Encode(s),
        PersistenceError::Deserialization(s) => durability::PersistenceError::Decode(s),
        PersistenceError::ChecksumMismatch { expected, actual } => {
            durability::PersistenceError::CrcMismatch { expected, actual }
        }
        PersistenceError::LockFailed { resource, reason } => {
            durability::PersistenceError::LockFailed { resource, reason }
        }
        PersistenceError::InvalidState(s) => durability::PersistenceError::InvalidState(s),
        PersistenceError::NotFound(s) => durability::PersistenceError::NotFound(s),
        PersistenceError::InvalidConfig(s) => durability::PersistenceError::InvalidConfig(s),
        PersistenceError::NotSupported(s) => durability::PersistenceError::NotSupported(s),
    }
}

#[derive(Clone)]
struct DirAdapter {
    inner: Arc<dyn vicinity_dir::Directory>,
}

impl durability::Directory for DirAdapter {
    fn create_file(&self, path: &str) -> durability::PersistenceResult<Box<dyn std::io::Write>> {
        self.inner.create_file(path).map_err(to_durability_err)
    }
    fn open_file(&self, path: &str) -> durability::PersistenceResult<Box<dyn std::io::Read>> {
        self.inner.open_file(path).map_err(to_durability_err)
    }
    fn exists(&self, path: &str) -> bool {
        self.inner.exists(path)
    }
    fn delete(&self, path: &str) -> durability::PersistenceResult<()> {
        self.inner.delete(path).map_err(to_durability_err)
    }
    fn atomic_rename(&self, from: &str, to: &str) -> durability::PersistenceResult<()> {
        self.inner
            .atomic_rename(from, to)
            .map_err(to_durability_err)
    }
    fn create_dir_all(&self, path: &str) -> durability::PersistenceResult<()> {
        self.inner.create_dir_all(path).map_err(to_durability_err)
    }
    fn list_dir(&self, path: &str) -> durability::PersistenceResult<Vec<String>> {
        self.inner.list_dir(path).map_err(to_durability_err)
    }
    fn append_file(&self, path: &str) -> durability::PersistenceResult<Box<dyn std::io::Write>> {
        self.inner.append_file(path).map_err(to_durability_err)
    }
    fn atomic_write(&self, path: &str, data: &[u8]) -> durability::PersistenceResult<()> {
        self.inner
            .atomic_write(path, data)
            .map_err(to_durability_err)
    }
    fn file_path(&self, path: &str) -> Option<std::path::PathBuf> {
        self.inner.file_path(path)
    }
}

/// WAL writer for appending entries.
///
/// Compatibility wrapper: keeps `vicinity::persistence::wal::WalWriter` stable while using the
/// canonical `durability::walog::WalWriter` implementation underneath.
pub struct WalWriter {
    inner: durability::walog::WalWriter<WalEntry>,
}

impl WalWriter {
    /// Create a new WAL writer.
    ///
    /// Uses the conservative durability posture: flush after each append.
    pub fn new(directory: impl Into<Arc<dyn vicinity_dir::Directory>>) -> Self {
        let inner_dir: Arc<dyn durability::Directory> = Arc::new(DirAdapter {
            inner: directory.into(),
        });
        Self {
            inner: durability::walog::WalWriter::new(inner_dir),
        }
    }

    /// Append an entry to the WAL, returning its assigned entry id.
    pub fn append(&mut self, entry: WalEntry) -> PersistenceResult<u64> {
        self.inner.append(&entry).map_err(PersistenceError::from)
    }

    /// Flush buffered bytes (if any).
    pub fn flush(&mut self) -> PersistenceResult<()> {
        self.inner.flush().map_err(PersistenceError::from)
    }
}

/// WAL reader for replaying entries.
pub struct WalReader {
    inner: durability::walog::WalReader<WalEntry>,
}

impl WalReader {
    /// Create a new WAL reader.
    pub fn new(directory: impl Into<Arc<dyn vicinity_dir::Directory>>) -> Self {
        let inner_dir: Arc<dyn durability::Directory> = Arc::new(DirAdapter {
            inner: directory.into(),
        });
        Self {
            inner: durability::walog::WalReader::new(inner_dir),
        }
    }

    /// Replay all WAL entries from disk (strict).
    pub fn replay(&self) -> PersistenceResult<Vec<WalRecord<WalEntry>>> {
        self.inner.replay().map_err(PersistenceError::from)
    }

    /// Best-effort replay: tolerate a truncated tail record in the final segment.
    pub fn replay_best_effort(&self) -> PersistenceResult<Vec<WalRecord<WalEntry>>> {
        self.inner
            .replay_best_effort()
            .map_err(PersistenceError::from)
    }
}

// ---------------------------------------------------------------------------
// Graph WAL: finer-grained entries for streaming HNSW updates
// ---------------------------------------------------------------------------

/// WAL writer for graph-level operations (HNSW node insert/delete/neighbor update).
///
/// Uses a separate WAL directory (`graph_wal/`) from the segment-lifecycle WAL (`wal/`).
/// Backed by the same `durability::walog` infrastructure, parameterized over `GraphWalEntry`.
pub struct GraphWalWriter {
    inner: durability::walog::WalWriter<GraphWalEntry>,
}

impl GraphWalWriter {
    /// Create a new graph WAL writer.
    ///
    /// Writes to `graph_wal/` within the provided directory.
    pub fn new(directory: impl Into<Arc<dyn vicinity_dir::Directory>>) -> Self {
        let inner_dir: Arc<dyn durability::Directory> = Arc::new(GraphDirAdapter {
            inner: DirAdapter {
                inner: directory.into(),
            },
        });
        Self {
            inner: durability::walog::WalWriter::new(inner_dir),
        }
    }

    /// Append a graph entry to the WAL, returning its assigned entry id.
    pub fn append(&mut self, entry: GraphWalEntry) -> PersistenceResult<u64> {
        self.inner.append(&entry).map_err(PersistenceError::from)
    }

    /// Flush buffered bytes (if any).
    pub fn flush(&mut self) -> PersistenceResult<()> {
        self.inner.flush().map_err(PersistenceError::from)
    }
}

/// WAL reader for graph-level operations.
///
/// Reads from `graph_wal/` within the provided directory.
pub struct GraphWalReader {
    inner: durability::walog::WalReader<GraphWalEntry>,
}

impl GraphWalReader {
    /// Create a new graph WAL reader.
    pub fn new(directory: impl Into<Arc<dyn vicinity_dir::Directory>>) -> Self {
        let inner_dir: Arc<dyn durability::Directory> = Arc::new(GraphDirAdapter {
            inner: DirAdapter {
                inner: directory.into(),
            },
        });
        Self {
            inner: durability::walog::WalReader::new(inner_dir),
        }
    }

    /// Replay all graph WAL entries from disk (strict).
    pub fn replay(&self) -> PersistenceResult<Vec<WalRecord<GraphWalEntry>>> {
        self.inner.replay().map_err(PersistenceError::from)
    }

    /// Best-effort replay: tolerate a truncated tail record in the final segment.
    pub fn replay_best_effort(&self) -> PersistenceResult<Vec<WalRecord<GraphWalEntry>>> {
        self.inner
            .replay_best_effort()
            .map_err(PersistenceError::from)
    }
}

/// Directory adapter that prefixes all paths with `graph_wal/` so that graph WAL files
/// live in a separate directory from the segment-lifecycle WAL.
///
/// `durability::walog` unconditionally uses `wal/` as its subdirectory. This adapter
/// remaps that to `graph_wal/` by rewriting paths.
#[derive(Clone)]
struct GraphDirAdapter {
    inner: DirAdapter,
}

impl durability::Directory for GraphDirAdapter {
    fn create_file(&self, path: &str) -> durability::PersistenceResult<Box<dyn std::io::Write>> {
        self.inner.create_file(&remap_graph_wal_path(path))
    }
    fn open_file(&self, path: &str) -> durability::PersistenceResult<Box<dyn std::io::Read>> {
        self.inner.open_file(&remap_graph_wal_path(path))
    }
    fn exists(&self, path: &str) -> bool {
        self.inner.exists(&remap_graph_wal_path(path))
    }
    fn delete(&self, path: &str) -> durability::PersistenceResult<()> {
        self.inner.delete(&remap_graph_wal_path(path))
    }
    fn atomic_rename(&self, from: &str, to: &str) -> durability::PersistenceResult<()> {
        self.inner
            .atomic_rename(&remap_graph_wal_path(from), &remap_graph_wal_path(to))
    }
    fn create_dir_all(&self, path: &str) -> durability::PersistenceResult<()> {
        self.inner.create_dir_all(&remap_graph_wal_path(path))
    }
    fn list_dir(&self, path: &str) -> durability::PersistenceResult<Vec<String>> {
        self.inner.list_dir(&remap_graph_wal_path(path))
    }
    fn append_file(&self, path: &str) -> durability::PersistenceResult<Box<dyn std::io::Write>> {
        self.inner.append_file(&remap_graph_wal_path(path))
    }
    fn atomic_write(&self, path: &str, data: &[u8]) -> durability::PersistenceResult<()> {
        self.inner.atomic_write(&remap_graph_wal_path(path), data)
    }
    fn file_path(&self, path: &str) -> Option<std::path::PathBuf> {
        self.inner.file_path(&remap_graph_wal_path(path))
    }
}

/// Rewrite `wal/...` paths to `graph_wal/...`.
fn remap_graph_wal_path(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("wal/") {
        format!("graph_wal/{}", rest)
    } else if path == "wal" {
        "graph_wal".to_string()
    } else {
        path.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::directory::MemoryDirectory;

    #[test]
    fn wal_write_read_roundtrip() {
        let dir: Arc<dyn vicinity_dir::Directory> = Arc::new(MemoryDirectory::new());
        let mut w = WalWriter::new(dir.clone());
        w.append(WalEntry::AddSegment {
            segment_id: 1,
            doc_count: 10,
        })
        .unwrap();
        w.append(WalEntry::DeleteDocuments {
            deletes: vec![(1, 5)],
        })
        .unwrap();
        w.flush().unwrap();

        let r = WalReader::new(dir);
        let entries = r.replay().unwrap();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn graph_wal_write_read_roundtrip() {
        let dir: Arc<dyn vicinity_dir::Directory> = Arc::new(MemoryDirectory::new());

        let mut gw = GraphWalWriter::new(dir.clone());
        gw.append(GraphWalEntry::InsertNode {
            doc_id: 42,
            level: 1,
            vector: vec![1.0, 2.0, 3.0],
            neighbors_per_level: vec![vec![10, 20], vec![30]],
        })
        .unwrap();
        gw.append(GraphWalEntry::UpdateNeighbors {
            node_id: 10,
            level: 0,
            neighbors: vec![42, 99],
        })
        .unwrap();
        gw.append(GraphWalEntry::DeleteNode { doc_id: 99 }).unwrap();
        gw.flush().unwrap();

        let gr = GraphWalReader::new(dir.clone());
        let entries = gr.replay().unwrap();
        assert_eq!(entries.len(), 3);

        // Verify first entry
        match &entries[0].payload {
            GraphWalEntry::InsertNode {
                doc_id,
                level,
                vector,
                ..
            } => {
                assert_eq!(*doc_id, 42);
                assert_eq!(*level, 1);
                assert_eq!(vector.len(), 3);
            }
            other => panic!("expected InsertNode, got {:?}", other),
        }

        // Verify graph WAL is isolated from segment WAL
        let segment_reader = WalReader::new(dir);
        let segment_entries = segment_reader.replay().unwrap();
        assert_eq!(
            segment_entries.len(),
            0,
            "graph WAL should not pollute segment WAL"
        );
    }
}
