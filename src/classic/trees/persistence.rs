use crate::RetrieveError;
use serde::{de::DeserializeOwned, Serialize};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

pub(crate) fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), RetrieveError> {
    let tmp_path = path.with_extension("tmp");
    {
        let file = std::fs::File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, value)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        writer.flush()?;
    }
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

pub(crate) fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, RetrieveError> {
    let file = std::fs::File::open(path)?;
    serde_json::from_reader(BufReader::new(file))
        .map_err(|e| RetrieveError::FormatError(e.to_string()))
}

pub(crate) fn validate_vector_shape(
    name: &str,
    dimension: usize,
    num_vectors: usize,
    vectors: &[f32],
    doc_ids: &[u32],
) -> Result<(), RetrieveError> {
    if dimension == 0 {
        return Err(RetrieveError::FormatError(format!(
            "{name} manifest has zero dimension"
        )));
    }
    if num_vectors == 0 {
        return Err(RetrieveError::FormatError(format!(
            "{name} manifest has zero vectors"
        )));
    }
    if vectors.len() != num_vectors * dimension {
        return Err(RetrieveError::FormatError(format!(
            "{name} vectors length {} does not match {} vectors of dimension {}",
            vectors.len(),
            num_vectors,
            dimension
        )));
    }
    if doc_ids.len() != num_vectors {
        return Err(RetrieveError::FormatError(format!(
            "{name} doc_ids length {} does not match vector count {}",
            doc_ids.len(),
            num_vectors
        )));
    }
    Ok(())
}
