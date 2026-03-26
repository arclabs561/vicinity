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
        // `durability` does not model distributed coordination errors; treat as unsupported.
        PersistenceError::Distributed(s) => durability::PersistenceError::NotSupported(s),
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
        self.inner.replay_best_effort().map_err(PersistenceError::from)
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
}
