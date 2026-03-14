//! Error types for vicinity.

use thiserror::Error;

/// Errors that can occur during indexing/search operations.
#[derive(Debug, Clone, PartialEq, Error)]
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
    DimensionMismatch { query_dim: usize, doc_dim: usize },

    /// I/O error (wrapped)
    #[error("I/O error: {0}")]
    Io(String), // RetrieveError needs to be Clone, std::io::Error isn't. Store string representation.

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
        RetrieveError::Io(err.to_string())
    }
}

#[cfg(feature = "qntz")]
impl From<qntz::VQuantError> for RetrieveError {
    fn from(err: qntz::VQuantError) -> Self {
        RetrieveError::Other(err.to_string())
    }
}

/// Result type alias for vicinity operations.
pub type Result<T> = std::result::Result<T, RetrieveError>;
