//! Shared helpers for modules that delegate to `clump`.

use crate::RetrieveError;

/// Convert a flat SoA buffer into AoS `Vec<Vec<f32>>`.
///
/// `flat` has length `num_vectors * dim`, laid out as consecutive `dim`-sized rows.
pub(crate) fn soa_to_aos(flat: &[f32], num_vectors: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..num_vectors)
        .map(|i| flat[i * dim..(i + 1) * dim].to_vec())
        .collect()
}

/// Map `clump::Error` to `RetrieveError`.
pub(crate) fn map_clump_error(e: clump::Error) -> RetrieveError {
    match e {
        clump::Error::EmptyInput => RetrieveError::InvalidParameter("Empty input".to_string()),
        clump::Error::InvalidParameter { name, message } => {
            RetrieveError::InvalidParameter(format!("{name}: {message}"))
        }
        clump::Error::InvalidClusterCount { requested, n_items } => {
            RetrieveError::InvalidParameter(format!(
                "invalid cluster count: requested {requested}, but dataset has {n_items} items"
            ))
        }
        clump::Error::DimensionMismatch { expected, found } => RetrieveError::InvalidParameter(
            format!("dimension mismatch: expected {expected}, found {found}"),
        ),
        clump::Error::ConstraintViolation(msg) => {
            RetrieveError::InvalidParameter(format!("constraint violation: {msg}"))
        }
        clump::Error::Other(msg) => RetrieveError::Other(msg),
    }
}
