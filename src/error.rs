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

    /// No results matched the query (e.g. filter excluded all candidates).
    #[error("no results found")]
    NotFound,

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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Payload alphabet safe to search for verbatim: no braces, no format
    /// specifiers, and no leading or trailing whitespace.
    fn payload() -> impl Strategy<Value = String> {
        "[A-Za-z0-9 _/.:+-]{1,40}"
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn invalid_parameter_display_keeps_payload(text in payload()) {
            let message = RetrieveError::InvalidParameter(text.clone()).to_string();
            prop_assert!(
                message.contains(&text),
                "Display of InvalidParameter lost its payload: {:?} does not contain {:?}",
                message,
                text
            );
        }

        #[test]
        fn dimension_mismatch_display_keeps_both_dimensions(
            query_dim in any::<usize>(),
            doc_dim in any::<usize>(),
        ) {
            let message = RetrieveError::DimensionMismatch { query_dim, doc_dim }.to_string();
            prop_assert!(
                message.contains(&query_dim.to_string()),
                "Display of DimensionMismatch lost query_dim={}: {:?}",
                query_dim,
                message
            );
            prop_assert!(
                message.contains(&doc_dim.to_string()),
                "Display of DimensionMismatch lost doc_dim={}: {:?}",
                doc_dim,
                message
            );
        }

        #[test]
        fn out_of_bounds_display_keeps_index(index in any::<usize>()) {
            let message = RetrieveError::OutOfBounds(index).to_string();
            prop_assert!(
                message.contains(&index.to_string()),
                "Display of OutOfBounds lost index={}: {:?}",
                index,
                message
            );
        }

        #[test]
        fn format_error_display_keeps_payload(text in payload()) {
            let message = RetrieveError::FormatError(text.clone()).to_string();
            prop_assert!(
                message.contains(&text),
                "Display of FormatError lost its payload: {:?}",
                message
            );
        }

        #[test]
        fn serialization_error_display_keeps_payload(text in payload()) {
            let message = RetrieveError::Serialization(text.clone()).to_string();
            prop_assert!(
                message.contains(&text),
                "Display of Serialization lost its payload: {:?}",
                message
            );
        }

        #[test]
        fn other_error_display_keeps_payload(text in payload()) {
            let message = RetrieveError::Other(text.clone()).to_string();
            prop_assert!(
                message.contains(&text),
                "Display of Other lost its payload: {:?}",
                message
            );
        }

        #[test]
        fn io_error_display_keeps_payload_and_kind(text in payload()) {
            let error = RetrieveError::Io(Arc::new(std::io::Error::other(text.clone())));
            let message = error.to_string();
            prop_assert!(
                message.contains(&text),
                "Display of Io lost the underlying message: {:?}",
                message
            );
            prop_assert!(
                matches!(&error, RetrieveError::Io(inner) if inner.kind() == std::io::ErrorKind::Other),
                "Io must preserve the underlying ErrorKind"
            );
        }
    }

    /// Fieldless variants have no payload to lose, so pin their exact text:
    /// a rename here silently changes user-visible output.
    #[test]
    fn fieldless_variants_display_exactly() {
        assert_eq!(RetrieveError::EmptyQuery.to_string(), "query is empty");
        assert_eq!(RetrieveError::EmptyIndex.to_string(), "index is empty");
        assert_eq!(RetrieveError::NotFound.to_string(), "no results found");
    }
}
