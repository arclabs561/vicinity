use crate::RetrieveError;
use serde::{de::DeserializeOwned, Serialize};
use smallvec::SmallVec;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

pub(crate) fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), RetrieveError> {
    write_atomic(path, |writer| {
        serde_json::to_writer_pretty(writer, value)
            .map_err(|e| std::io::Error::other(e.to_string()))
    })
}

pub(crate) fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, RetrieveError> {
    let file = std::fs::File::open(path)?;
    serde_json::from_reader(BufReader::new(file))
        .map_err(|e| RetrieveError::FormatError(e.to_string()))
}

pub(crate) fn write_f32_atomic(path: &Path, values: &[f32]) -> Result<(), RetrieveError> {
    write_atomic(path, |writer| {
        for value in values {
            writer.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    })
}

pub(crate) fn write_u32_atomic(path: &Path, values: &[u32]) -> Result<(), RetrieveError> {
    write_atomic(path, |writer| {
        for value in values {
            writer.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    })
}

pub(crate) fn write_neighbors_atomic(
    path: &Path,
    magic: &[u8; 8],
    neighbors: &[SmallVec<[u32; 16]>],
) -> Result<(), RetrieveError> {
    write_atomic(path, |writer| {
        writer.write_all(magic)?;
        writer.write_all(&(neighbors.len() as u64).to_le_bytes())?;
        for list in neighbors {
            writer.write_all(&(list.len() as u64).to_le_bytes())?;
            for id in list {
                writer.write_all(&id.to_le_bytes())?;
            }
        }
        Ok(())
    })
}

fn write_atomic(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<std::fs::File>) -> std::io::Result<()>,
) -> Result<(), RetrieveError> {
    let tmp_path = path.with_extension("tmp");
    {
        let file = std::fs::File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);
        write(&mut writer)?;
        writer.flush()?;
    }
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

pub(crate) fn read_f32_exact(path: &Path, expected_len: usize) -> Result<Vec<f32>, RetrieveError> {
    let bytes = std::fs::read(path)?;
    let expected_bytes = expected_len
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| RetrieveError::FormatError("f32 byte length overflow".into()))?;
    if bytes.len() != expected_bytes {
        return Err(RetrieveError::FormatError(format!(
            "{} size mismatch: expected {} bytes, got {}",
            path.display(),
            expected_bytes,
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

pub(crate) fn read_u32_exact(path: &Path, expected_len: usize) -> Result<Vec<u32>, RetrieveError> {
    let bytes = std::fs::read(path)?;
    let expected_bytes = expected_len
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| RetrieveError::FormatError("u32 byte length overflow".into()))?;
    if bytes.len() != expected_bytes {
        return Err(RetrieveError::FormatError(format!(
            "{} size mismatch: expected {} bytes, got {}",
            path.display(),
            expected_bytes,
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

pub(crate) fn read_neighbors(
    path: &Path,
    magic: &[u8; 8],
    expected_nodes: usize,
) -> Result<Vec<SmallVec<[u32; 16]>>, RetrieveError> {
    let mut reader = BufReader::new(std::fs::File::open(path)?);
    let mut actual_magic = [0u8; 8];
    reader.read_exact(&mut actual_magic)?;
    if &actual_magic != magic {
        return Err(RetrieveError::FormatError(format!(
            "invalid graph neighbors magic in {}",
            path.display()
        )));
    }
    let count = read_u64(&mut reader)? as usize;
    if count != expected_nodes {
        return Err(RetrieveError::FormatError(format!(
            "neighbor list count {} does not match manifest count {}",
            count, expected_nodes
        )));
    }

    let mut neighbors = Vec::with_capacity(expected_nodes);
    for node in 0..expected_nodes {
        let len = read_u64(&mut reader)? as usize;
        let max_reasonable_degree = expected_nodes.saturating_mul(4).max(64);
        if len > max_reasonable_degree {
            return Err(RetrieveError::FormatError(format!(
                "node {node} has too many neighbors: {len}"
            )));
        }
        let mut list = SmallVec::<[u32; 16]>::new();
        for _ in 0..len {
            let id = read_u32(&mut reader)?;
            if id as usize >= expected_nodes {
                return Err(RetrieveError::FormatError(format!(
                    "neighbor id {id} exceeds vector count {expected_nodes}"
                )));
            }
            list.push(id);
        }
        neighbors.push(list);
    }

    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        return Err(RetrieveError::FormatError(
            "graph neighbors file has trailing bytes".into(),
        ));
    }
    Ok(neighbors)
}

pub(crate) fn validate_graph_shape(
    name: &str,
    dimension: usize,
    num_vectors: usize,
    vectors: &[f32],
    doc_ids: &[u32],
    neighbors: &[SmallVec<[u32; 16]>],
    entry: Option<u32>,
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
    if neighbors.len() != num_vectors {
        return Err(RetrieveError::FormatError(format!(
            "{name} neighbor list count {} does not match vector count {}",
            neighbors.len(),
            num_vectors
        )));
    }
    if let Some(entry) = entry {
        if entry as usize >= num_vectors {
            return Err(RetrieveError::FormatError(format!(
                "{name} entry node {entry} exceeds vector count {num_vectors}"
            )));
        }
    }
    Ok(())
}

fn read_u64(reader: &mut impl Read) -> Result<u64, RetrieveError> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_u32(reader: &mut impl Read) -> Result<u32, RetrieveError> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}
