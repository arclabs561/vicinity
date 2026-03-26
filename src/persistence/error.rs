//! Error types for persistence operations.

#[cfg(feature = "persistence")]
use durability as durability_crate;
use thiserror::Error;

/// Errors that can occur during persistence operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PersistenceError {
    /// I/O error (file operations, disk I/O)
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Format error (invalid magic bytes, version mismatch, corruption)
    #[error("format error: {0}")]
    Format(String),

    /// Serialization error (postcard, serde)
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Deserialization error
    #[error("deserialization error: {0}")]
    Deserialization(String),

    /// Checksum mismatch (data corruption detected)
    #[error("checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch {
        /// Expected CRC32 checksum.
        expected: u32,
        /// Actual CRC32 checksum read from data.
        actual: u32,
    },

    /// Lock acquisition failed (concurrent access conflict)
    #[error("failed to acquire lock on {resource}: {reason}")]
    LockFailed {
        /// The resource that could not be locked.
        resource: String,
        /// Why the lock could not be acquired.
        reason: String,
    },

    /// Invalid state (e.g., operation not allowed in current state)
    #[error("invalid state: {0}")]
    InvalidState(String),

    /// Resource not found (file, segment, etc.)
    #[error("resource not found: {0}")]
    NotFound(String),

    /// Invalid configuration
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Operation not supported
    #[error("operation not supported: {0}")]
    NotSupported(String),
}

#[cfg(feature = "persistence")]
impl From<postcard::Error> for PersistenceError {
    fn from(e: postcard::Error) -> Self {
        Self::Serialization(format!("postcard error: {}", e))
    }
}

impl From<crate::RetrieveError> for PersistenceError {
    fn from(e: crate::RetrieveError) -> Self {
        Self::Format(e.to_string())
    }
}

impl From<PersistenceError> for crate::RetrieveError {
    fn from(e: PersistenceError) -> Self {
        match e {
            PersistenceError::Io(e) => Self::Other(format!("persistence I/O: {e}")),
            PersistenceError::Format(s) => Self::FormatError(s),
            PersistenceError::Serialization(s) => Self::Serialization(s),
            PersistenceError::Deserialization(s) => Self::FormatError(s),
            _ => Self::Other(e.to_string()),
        }
    }
}

/// Result type for persistence operations.
pub type PersistenceResult<T> = Result<T, PersistenceError>;

#[cfg(feature = "persistence")]
impl From<durability_crate::PersistenceError> for PersistenceError {
    fn from(e: durability_crate::PersistenceError) -> Self {
        match e {
            durability_crate::PersistenceError::Io(e) => Self::Io(e),
            durability_crate::PersistenceError::Format(s) => Self::Format(s),
            durability_crate::PersistenceError::CrcMismatch { expected, actual } => {
                Self::ChecksumMismatch { expected, actual }
            }
            durability_crate::PersistenceError::Encode(s) => Self::Serialization(s),
            durability_crate::PersistenceError::Decode(s) => Self::Deserialization(s),
            durability_crate::PersistenceError::InvalidState(s) => Self::InvalidState(s),
            durability_crate::PersistenceError::InvalidConfig(s) => Self::InvalidConfig(s),
            durability_crate::PersistenceError::NotSupported(s) => Self::NotSupported(s),
            durability_crate::PersistenceError::LockFailed { resource, reason } => {
                Self::LockFailed { resource, reason }
            }
            durability_crate::PersistenceError::NotFound(s) => Self::NotFound(s),
            _ => Self::Format(format!("unknown durability error: {e}")),
        }
    }
}
