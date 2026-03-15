//! Storage backends for `vicinity` persistence.
//!
//! `vicinity` exposes a `Directory` abstraction for all persistence modules to use.
//! When `feature = "persistence"` is enabled, this module delegates to the canonical
//! implementations in `durability::storage` and adapts error types.
//!
//! When `feature = "persistence"` is disabled, the same types exist but all operations
//! return `PersistenceError::NotSupported("persistence feature disabled")`.

use crate::persistence::error::{PersistenceError, PersistenceResult};
use std::io::{Read, Write};
use std::path::PathBuf;

/// Filesystem-like directory abstraction for `vicinity` persistence.
pub trait Directory: Send + Sync {
    /// Create a new file for writing, truncating if it exists.
    fn create_file(&self, path: &str) -> PersistenceResult<Box<dyn Write>>;
    /// Open an existing file for reading.
    fn open_file(&self, path: &str) -> PersistenceResult<Box<dyn Read>>;
    /// Check whether a file or directory exists at the given path.
    fn exists(&self, path: &str) -> bool;
    /// Delete a file.
    fn delete(&self, path: &str) -> PersistenceResult<()>;
    /// Atomically rename a file.
    fn atomic_rename(&self, from: &str, to: &str) -> PersistenceResult<()>;
    /// Create a directory and all parent directories.
    fn create_dir_all(&self, path: &str) -> PersistenceResult<()>;
    /// List entries in a directory.
    fn list_dir(&self, path: &str) -> PersistenceResult<Vec<String>>;
    /// Open a file for appending, creating it if it does not exist.
    fn append_file(&self, path: &str) -> PersistenceResult<Box<dyn Write>>;
    /// Write data atomically (write-to-temp then rename).
    fn atomic_write(&self, path: &str, data: &[u8]) -> PersistenceResult<()>;
    /// Return the underlying filesystem path, if backed by one.
    fn file_path(&self, path: &str) -> Option<PathBuf>;
}

/// Controls how often writers call `flush()` on their underlying `Write`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlushPolicy {
    /// Flush after every append call.
    PerAppend,
    /// Flush every N append calls.
    EveryN(usize),
    /// Caller is responsible for flushing.
    Manual,
}

impl Default for FlushPolicy {
    fn default() -> Self {
        Self::EveryN(64)
    }
}

fn disabled() -> PersistenceError {
    PersistenceError::NotSupported("persistence feature disabled".to_string())
}

// -----------------------------------------------------------------------------
// Stubs (feature disabled)
// -----------------------------------------------------------------------------

#[cfg(not(feature = "persistence"))]
#[derive(Debug, Default, Clone)]
/// In-memory directory stub (persistence feature disabled).
pub struct MemoryDirectory;

#[cfg(not(feature = "persistence"))]
impl MemoryDirectory {
    /// Create a new in-memory directory (no-op without persistence feature).
    pub fn new() -> Self {
        Self
    }
}

#[cfg(not(feature = "persistence"))]
impl Directory for MemoryDirectory {
    fn create_file(&self, _path: &str) -> PersistenceResult<Box<dyn Write>> {
        Err(disabled())
    }
    fn open_file(&self, _path: &str) -> PersistenceResult<Box<dyn Read>> {
        Err(disabled())
    }
    fn exists(&self, _path: &str) -> bool {
        false
    }
    fn delete(&self, _path: &str) -> PersistenceResult<()> {
        Err(disabled())
    }
    fn atomic_rename(&self, _from: &str, _to: &str) -> PersistenceResult<()> {
        Err(disabled())
    }
    fn create_dir_all(&self, _path: &str) -> PersistenceResult<()> {
        Err(disabled())
    }
    fn list_dir(&self, _path: &str) -> PersistenceResult<Vec<String>> {
        Err(disabled())
    }
    fn append_file(&self, _path: &str) -> PersistenceResult<Box<dyn Write>> {
        Err(disabled())
    }
    fn atomic_write(&self, _path: &str, _data: &[u8]) -> PersistenceResult<()> {
        Err(disabled())
    }
    fn file_path(&self, _path: &str) -> Option<PathBuf> {
        None
    }
}

#[cfg(not(feature = "persistence"))]
/// Filesystem-backed directory stub (persistence feature disabled).
#[derive(Debug, Default, Clone)]
pub struct FsDirectory;

#[cfg(not(feature = "persistence"))]
impl FsDirectory {
    /// Create a filesystem directory (fails without persistence feature).
    pub fn new(_root: impl Into<PathBuf>) -> PersistenceResult<Self> {
        Err(disabled())
    }
}

#[cfg(not(feature = "persistence"))]
impl Directory for FsDirectory {
    fn create_file(&self, _path: &str) -> PersistenceResult<Box<dyn Write>> {
        Err(disabled())
    }
    fn open_file(&self, _path: &str) -> PersistenceResult<Box<dyn Read>> {
        Err(disabled())
    }
    fn exists(&self, _path: &str) -> bool {
        false
    }
    fn delete(&self, _path: &str) -> PersistenceResult<()> {
        Err(disabled())
    }
    fn atomic_rename(&self, _from: &str, _to: &str) -> PersistenceResult<()> {
        Err(disabled())
    }
    fn create_dir_all(&self, _path: &str) -> PersistenceResult<()> {
        Err(disabled())
    }
    fn list_dir(&self, _path: &str) -> PersistenceResult<Vec<String>> {
        Err(disabled())
    }
    fn append_file(&self, _path: &str) -> PersistenceResult<Box<dyn Write>> {
        Err(disabled())
    }
    fn atomic_write(&self, _path: &str, _data: &[u8]) -> PersistenceResult<()> {
        Err(disabled())
    }
    fn file_path(&self, _path: &str) -> Option<PathBuf> {
        None
    }
}

// -----------------------------------------------------------------------------
// Enabled implementations (feature enabled)
// -----------------------------------------------------------------------------

#[cfg(feature = "persistence")]
mod enabled {
    use super::*;
    use durability::storage::Directory as DurabilityDirectory;

    #[derive(Clone, Default)]
    pub struct MemoryDirectory {
        inner: durability::storage::MemoryDirectory,
    }

    impl MemoryDirectory {
        pub fn new() -> Self {
            Self {
                inner: durability::storage::MemoryDirectory::new(),
            }
        }
    }

    impl Directory for MemoryDirectory {
        fn create_file(&self, path: &str) -> PersistenceResult<Box<dyn Write>> {
            self.inner.create_file(path).map_err(PersistenceError::from)
        }
        fn open_file(&self, path: &str) -> PersistenceResult<Box<dyn Read>> {
            self.inner.open_file(path).map_err(PersistenceError::from)
        }
        fn exists(&self, path: &str) -> bool {
            self.inner.exists(path)
        }
        fn delete(&self, path: &str) -> PersistenceResult<()> {
            self.inner.delete(path).map_err(PersistenceError::from)
        }
        fn atomic_rename(&self, from: &str, to: &str) -> PersistenceResult<()> {
            self.inner
                .atomic_rename(from, to)
                .map_err(PersistenceError::from)
        }
        fn create_dir_all(&self, path: &str) -> PersistenceResult<()> {
            self.inner
                .create_dir_all(path)
                .map_err(PersistenceError::from)
        }
        fn list_dir(&self, path: &str) -> PersistenceResult<Vec<String>> {
            self.inner.list_dir(path).map_err(PersistenceError::from)
        }
        fn append_file(&self, path: &str) -> PersistenceResult<Box<dyn Write>> {
            self.inner.append_file(path).map_err(PersistenceError::from)
        }
        fn atomic_write(&self, path: &str, data: &[u8]) -> PersistenceResult<()> {
            self.inner
                .atomic_write(path, data)
                .map_err(PersistenceError::from)
        }
        fn file_path(&self, path: &str) -> Option<PathBuf> {
            self.inner.file_path(path)
        }
    }

    pub struct FsDirectory {
        root: PathBuf,
        inner: durability::storage::FsDirectory,
    }

    impl FsDirectory {
        pub fn new(root: impl Into<PathBuf>) -> PersistenceResult<Self> {
            let root = root.into();
            let inner = durability::storage::FsDirectory::new(root.clone())
                .map_err(PersistenceError::from)?;
            Ok(Self { root, inner })
        }

        pub fn root(&self) -> &PathBuf {
            &self.root
        }
    }

    impl Directory for FsDirectory {
        fn create_file(&self, path: &str) -> PersistenceResult<Box<dyn Write>> {
            self.inner.create_file(path).map_err(PersistenceError::from)
        }
        fn open_file(&self, path: &str) -> PersistenceResult<Box<dyn Read>> {
            self.inner.open_file(path).map_err(PersistenceError::from)
        }
        fn exists(&self, path: &str) -> bool {
            self.inner.exists(path)
        }
        fn delete(&self, path: &str) -> PersistenceResult<()> {
            self.inner.delete(path).map_err(PersistenceError::from)
        }
        fn atomic_rename(&self, from: &str, to: &str) -> PersistenceResult<()> {
            self.inner
                .atomic_rename(from, to)
                .map_err(PersistenceError::from)
        }
        fn create_dir_all(&self, path: &str) -> PersistenceResult<()> {
            self.inner
                .create_dir_all(path)
                .map_err(PersistenceError::from)
        }
        fn list_dir(&self, path: &str) -> PersistenceResult<Vec<String>> {
            self.inner.list_dir(path).map_err(PersistenceError::from)
        }
        fn append_file(&self, path: &str) -> PersistenceResult<Box<dyn Write>> {
            self.inner.append_file(path).map_err(PersistenceError::from)
        }
        fn atomic_write(&self, path: &str, data: &[u8]) -> PersistenceResult<()> {
            self.inner
                .atomic_write(path, data)
                .map_err(PersistenceError::from)
        }
        fn file_path(&self, path: &str) -> Option<PathBuf> {
            self.inner.file_path(path)
        }
    }

    pub use FsDirectory as PublicFsDirectory;
    pub use MemoryDirectory as PublicMemoryDirectory;
}

#[cfg(feature = "persistence")]
pub use enabled::{PublicFsDirectory as FsDirectory, PublicMemoryDirectory as MemoryDirectory};
