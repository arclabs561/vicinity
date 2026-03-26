//! Error types for vicinity.

use std::sync::Arc;

use thiserror::Error;

/// Errors that can occur during indexing/search operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RetrieveError {
    /// Empty query provided.
    #[error("query is empty")]
    EmptyQuery,

    /// Empty index (no documents indexed).
    #[error("index is empty")]
    EmptyIndex,

    /// Invalid parameter value.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// Dimension mismatch between query and documents.
    #[error("dimension mismatch: query has {query_dim} dimensions, document has {doc_dim}")]
    DimensionMismatch {
        /// Dimension of the query vector.
        query_dim: usize,
        /// Dimension of the document vector (or index).
        doc_dim: usize,
    },

    /// I/O error (preserves the original `std::io::Error` via `Arc` for `ErrorKind` access).
    #[error("I/O error: {0}")]
    Io(Arc<std::io::Error>),

    /// Out of bounds access
    #[error("index out of bounds: {0}")]
    OutOfBounds(usize),

    /// Format error
    #[error("format error: {0}")]
    FormatError(String),

    /// Serialization error
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Other error (for extensibility).
    #[error("{0}")]
    Other(String),
}

impl From<std::io::Error> for RetrieveError {
    fn from(err: std::io::Error) -> Self {
        RetrieveError::Io(Arc::new(err))
    }
}

#[cfg(feature = "qntz")]
impl From<qntz::VQuantError> for RetrieveError {
    fn from(err: qntz::VQuantError) -> Self {
        RetrieveError::Other(err.to_string())
    }
}

#[cfg(feature = "id-compression")]
impl From<crate::compression::CompressionError> for RetrieveError {
    fn from(err: crate::compression::CompressionError) -> Self {
        RetrieveError::Other(err.to_string())
    }
}

/// Result type alias for vicinity operations.
pub type Result<T> = std::result::Result<T, RetrieveError>;
