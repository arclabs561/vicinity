//! Error types for vicinity.

use std::path::PathBuf;
use std::sync::Arc;

use thiserror::Error;

/// Errors that can occur during indexing/search operations.
///
/// The typed variants below (`NotBuilt`, `AlreadyBuilt`, `InvalidConfig`,
/// `Unsupported`, `NonFiniteInput`, `IoAt`, `SerializationAt`, `BadMagic`) are
/// additive: the stringly `InvalidParameter`, `FormatError`, and `Other`
/// variants remain for call sites that have not been converted yet, so matching
/// on a specific cause is possible without breaking existing consumers.
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

    /// A configuration parameter has an unusable value.
    #[error("invalid config `{field}`: {expected}")]
    InvalidConfig {
        /// The configuration field (or accepted constructor argument) at fault.
        field: &'static str,
        /// What a usable value looks like, phrased as the violated requirement.
        expected: &'static str,
    },

    /// An operation required a built index or built subcomponent that is absent.
    #[error("{component} is not built; call build() before {operation}")]
    NotBuilt {
        /// Index or subcomponent that has not been built.
        component: &'static str,
        /// Operation that was attempted, e.g. `"search"` or `"save"`.
        operation: &'static str,
    },

    /// An operation was attempted on an index that was already built.
    #[error(
        "{component} is already built; {operation} is not allowed — use insert()/push for \
         post-build updates, or rebuild the index"
    )]
    AlreadyBuilt {
        /// Index or subcomponent that was already built.
        component: &'static str,
        /// Operation that was attempted, e.g. `"add vectors"` or `"build"`.
        operation: &'static str,
    },

    /// A requested feature or operation is not available for this index configuration.
    #[error("unsupported: {feature}; {hint}")]
    Unsupported {
        /// Feature or operation that is not supported.
        feature: &'static str,
        /// What to do instead or what to rebuild.
        hint: &'static str,
    },

    /// An input vector contained a NaN or infinite component.
    #[error(
        "non-finite input for doc_id {doc_id}: component {component} is {value}; every vector \
         component must be finite, because NaN and infinity propagate into every distance"
    )]
    NonFiniteInput {
        /// Document whose vector was rejected.
        doc_id: u64,
        /// Position of the offending component in the vector.
        component: usize,
        /// The offending value (NaN or ±infinity).
        value: f64,
    },

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

    /// I/O error with the path that was being read or written.
    #[error("I/O error at {path}: {source}; check that the path exists and is readable")]
    IoAt {
        /// Path involved in the failed I/O.
        path: PathBuf,
        /// Underlying I/O error.
        source: Arc<std::io::Error>,
    },

    /// Out of bounds access
    #[error("index out of bounds: {0}")]
    OutOfBounds(usize),

    /// Format error
    #[error("format error: {0}")]
    FormatError(String),

    /// Serialization error
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Serialization error with the path that was being decoded.
    #[error(
        "serialization error at {path}: {message}; the file is corrupt or was written with an \
         incompatible schema"
    )]
    SerializationAt {
        /// Path of the file that failed to decode.
        path: PathBuf,
        /// Decoder message.
        message: String,
    },

    /// The file's leading magic bytes do not match the expected format.
    #[error(
        "bad magic: expected {expected}, found {found}; the file is not a vicinity index or was \
         written by an incompatible encoder"
    )]
    BadMagic {
        /// Expected magic bytes, rendered as text.
        expected: String,
        /// Bytes actually found, rendered as text or hex.
        found: String,
    },

    /// Other error (for extensibility).
    #[error("{0}")]
    Other(String),
}

impl From<std::io::Error> for RetrieveError {
    fn from(err: std::io::Error) -> Self {
        RetrieveError::Io(Arc::new(err))
    }
}

impl RetrieveError {
    /// Render leading file bytes for [`RetrieveError::BadMagic`].
    ///
    /// Printable ASCII is shown as a quoted string so the message names the file
    /// type a human would recognize; anything else is hex-escaped.
    #[cfg_attr(not(feature = "persistence"), allow(dead_code))]
    pub(crate) fn describe_magic(bytes: &[u8]) -> String {
        if !bytes.is_empty()
            && bytes
                .iter()
                .all(|byte| byte.is_ascii_graphic() || *byte == b' ')
        {
            format!("{:?}", String::from_utf8_lossy(bytes))
        } else {
            let hex: Vec<String> = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
            format!("[{}]", hex.join(" "))
        }
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

    /// Every typed variant must say what failed, which parameter/field is at
    /// fault, and what the caller can do about it.
    #[test]
    fn typed_variants_display_actionable_fields() {
        let not_built = RetrieveError::NotBuilt {
            component: "HNSW index",
            operation: "search",
        };
        let message = not_built.to_string();
        assert!(message.contains("HNSW index"), "{message}");
        assert!(message.contains("not built"), "{message}");
        assert!(message.contains("call build()"), "{message}");
        assert!(message.contains("search"), "{message}");

        let already_built = RetrieveError::AlreadyBuilt {
            component: "HNSW index",
            operation: "add vectors",
        };
        let message = already_built.to_string();
        assert!(message.contains("already built"), "{message}");
        assert!(message.contains("insert()"), "{message}");
        assert!(message.contains("add vectors"), "{message}");

        let config = RetrieveError::InvalidConfig {
            field: "rerank_factor",
            expected: "must be greater than 0",
        };
        let message = config.to_string();
        assert!(message.contains("invalid config"), "{message}");
        assert!(message.contains("rerank_factor"), "{message}");
        assert!(message.contains("greater than 0"), "{message}");

        let unsupported = RetrieveError::Unsupported {
            feature: "filtered search",
            hint: "rebuild with HNSWIndex::with_filtering()",
        };
        let message = unsupported.to_string();
        assert!(message.contains("unsupported"), "{message}");
        assert!(message.contains("filtered search"), "{message}");
        assert!(message.contains("with_filtering()"), "{message}");

        let non_finite = RetrieveError::NonFiniteInput {
            doc_id: 7,
            component: 3,
            value: f64::from(f32::NAN),
        };
        let message = non_finite.to_string();
        assert!(message.contains("non-finite"), "{message}");
        assert!(message.contains("doc_id 7"), "{message}");
        assert!(message.contains("component 3"), "{message}");
        assert!(message.contains("NaN"), "{message}");
        assert!(message.contains("finite"), "{message}");

        let io = RetrieveError::IoAt {
            path: PathBuf::from("/tmp/index.json"),
            source: Arc::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "no such file",
            )),
        };
        let message = io.to_string();
        assert!(message.contains("/tmp/index.json"), "{message}");
        assert!(message.contains("no such file"), "{message}");
        assert!(message.contains("readable"), "{message}");

        let serialization = RetrieveError::SerializationAt {
            path: PathBuf::from("/tmp/index.json"),
            message: "expected value at line 1 column 1".into(),
        };
        let message = serialization.to_string();
        assert!(message.contains("/tmp/index.json"), "{message}");
        assert!(message.contains("expected value"), "{message}");
        assert!(message.contains("corrupt"), "{message}");

        let bad_magic = RetrieveError::BadMagic {
            expected: RetrieveError::describe_magic(b"VCNHNSW\x01"),
            found: RetrieveError::describe_magic(b"{\"vectors\""),
        };
        let message = bad_magic.to_string();
        assert!(message.contains("bad magic"), "{message}");
        assert!(message.contains("expected"), "{message}");
        assert!(message.contains("found"), "{message}");
        assert!(message.contains("{\\\"vectors\\\""), "{message}");
        assert!(message.contains("incompatible"), "{message}");
    }

    #[test]
    fn describe_magic_keeps_printable_text_readable() {
        assert_eq!(
            RetrieveError::describe_magic(b"{\"vectors\""),
            "\"{\\\"vectors\\\"\""
        );
        assert_eq!(
            RetrieveError::describe_magic(b"VCNHNSW\x01"),
            "[56 43 4e 48 4e 53 57 01]"
        );
        assert_eq!(RetrieveError::describe_magic(&[]), "[]");
    }

    #[test]
    fn io_at_preserves_error_kind() {
        let error = RetrieveError::IoAt {
            path: PathBuf::from("missing.bin"),
            source: Arc::new(std::io::Error::from(std::io::ErrorKind::PermissionDenied)),
        };
        match error {
            RetrieveError::IoAt { source, .. } => {
                assert_eq!(source.kind(), std::io::ErrorKind::PermissionDenied);
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }
}
