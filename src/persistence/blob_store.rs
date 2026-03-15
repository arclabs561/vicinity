//! Blob storage for large metadata and content.
//!
//! Provides a trait-based interface for storing large blobs (document content,
//! images, etc.) separately from small metadata (pointers, IDs, flags).
//!
//! # Design Philosophy
//!
//! Wilson Lin's 3B search engine found success using **RocksDB's BlobDB** feature:
//! - Store small metadata/pointers in LSM tree (fast lookups)
//! - Store large blobs in separate log files (avoids write amplification)
//! - Critical for billion-scale systems
//!
//! This module provides:
//! 1. `BlobStore` trait for abstraction
//! 2. `FileBlobStore` implementation (simple, file-based)
//! 3. Future: `RocksBlobStore` implementation (production-scale)

use crate::persistence::error::{PersistenceError, PersistenceResult};
use std::path::{Path, PathBuf};

/// Trait for blob storage backends.
///
/// Blobs are arbitrary byte sequences stored by key.
/// Keys are typically document IDs, entity IDs, or content hashes.
pub trait BlobStore: Send + Sync {
    /// Store a blob with the given key.
    ///
    /// # Arguments
    /// * `key` - Unique identifier for the blob
    /// * `blob` - The blob data to store
    ///
    /// # Returns
    /// Error if storage fails (e.g., disk full, permission denied)
    fn put(&self, key: &[u8], blob: &[u8]) -> PersistenceResult<()>;

    /// Retrieve a blob by key.
    ///
    /// # Arguments
    /// * `key` - The blob identifier
    ///
    /// # Returns
    /// `Ok(Some(blob))` if found, `Ok(None)` if not found, `Err` on error
    fn get(&self, key: &[u8]) -> PersistenceResult<Option<Vec<u8>>>;

    /// Delete a blob by key.
    ///
    /// # Arguments
    /// * `key` - The blob identifier
    ///
    /// # Returns
    /// Error if deletion fails (e.g., permission denied)
    fn delete(&self, key: &[u8]) -> PersistenceResult<()>;

    /// Check if a blob exists.
    ///
    /// # Arguments
    /// * `key` - The blob identifier
    ///
    /// # Returns
    /// `Ok(true)` if exists, `Ok(false)` if not, `Err` on error
    fn exists(&self, key: &[u8]) -> PersistenceResult<bool> {
        self.get(key).map(|opt| opt.is_some())
    }
}

/// File-based blob storage implementation.
///
/// Stores blobs as individual files in a directory structure.
/// Key format: hex-encoded bytes (e.g., `a1b2c3d4...`)
///
/// # Performance
///
/// - Simple and reliable
// - Good for small to medium scale (< 100M blobs)
/// - Not optimal for very large scale (billions of blobs)
///
/// # Future Optimization
///
/// For billion-scale, use `RocksBlobStore` with RocksDB BlobDB:
/// - Better write amplification (blobs in separate log files)
/// - Better compaction performance
/// - Better concurrent access
pub struct FileBlobStore {
    base_path: PathBuf,
}

impl FileBlobStore {
    /// Create a new file-based blob store.
    ///
    /// # Arguments
    /// * `base_path` - Base directory for blob storage
    ///
    /// # Errors
    /// Returns error if directory cannot be created or is not writable
    pub fn new<P: AsRef<Path>>(base_path: P) -> PersistenceResult<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_path).map_err(PersistenceError::Io)?;

        Ok(Self { base_path })
    }

    /// Convert key bytes to file path.
    ///
    /// Uses hex encoding with two-level directory structure to avoid
    /// too many files in a single directory (e.g., `a1/b2/blob_a1b2c3d4...`).
    fn key_to_path(&self, key: &[u8]) -> PathBuf {
        let hex_key = hex::encode(key);

        // Use first 4 characters for two-level directory structure
        if hex_key.len() >= 4 {
            let dir1 = &hex_key[0..2];
            let dir2 = &hex_key[2..4];
            self.base_path.join(dir1).join(dir2).join(&hex_key)
        } else {
            // Fallback for very short keys
            self.base_path.join(&hex_key)
        }
    }
}

impl BlobStore for FileBlobStore {
    fn put(&self, key: &[u8], blob: &[u8]) -> PersistenceResult<()> {
        let path = self.key_to_path(key);

        // Create parent directories
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(PersistenceError::Io)?;
        }

        // Write blob atomically (write to temp file, then rename)
        let temp_path = path.with_extension(".tmp");
        std::fs::write(&temp_path, blob).map_err(PersistenceError::Io)?;

        std::fs::rename(&temp_path, &path).map_err(PersistenceError::Io)?;

        Ok(())
    }

    fn get(&self, key: &[u8]) -> PersistenceResult<Option<Vec<u8>>> {
        let path = self.key_to_path(key);

        match std::fs::read(&path) {
            Ok(blob) => Ok(Some(blob)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(PersistenceError::Io(e)),
        }
    }

    fn delete(&self, key: &[u8]) -> PersistenceResult<()> {
        let path = self.key_to_path(key);

        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()), // Already deleted
            Err(e) => Err(PersistenceError::Io(e)),
        }
    }
}

#[cfg(all(test, feature = "persistence"))]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_file_blob_store() {
        let temp_dir = TempDir::new().unwrap();
        let store = FileBlobStore::new(temp_dir.path()).unwrap();

        let key = b"test_key";
        let blob = b"test blob content";

        // Put
        store.put(key, blob).unwrap();

        // Get
        let retrieved = store.get(key).unwrap().unwrap();
        assert_eq!(retrieved, blob);

        // Exists
        assert!(store.exists(key).unwrap());

        // Delete
        store.delete(key).unwrap();
        assert!(!store.exists(key).unwrap());
        assert!(store.get(key).unwrap().is_none());
    }

    #[test]
    fn test_key_to_path_structure() {
        let temp_dir = TempDir::new().unwrap();
        let store = FileBlobStore::new(temp_dir.path()).unwrap();

        // Key bytes: 'a'=0x61, 'b'=0x62, etc. -> hex: "6162636465..."
        let key = b"abcdefghijklmnop"; // 16 bytes -> 32 hex chars
        let path = store.key_to_path(key);

        // Should have two-level directory structure based on hex encoding
        // 'ab' -> hex "6162", so dir1="61", dir2="62"
        assert!(path.to_string_lossy().contains("61/62"));
        assert!(path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("6162"));
    }
}
