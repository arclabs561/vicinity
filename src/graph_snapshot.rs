use crate::RetrieveError;
use serde::{de::DeserializeOwned, Serialize};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

macro_rules! cfg_graph_neighbors {
    ($($item:item)*) => {
        $(
            #[cfg(any(
                feature = "nsw",
                feature = "sng",
                feature = "vamana",
                feature = "nsg",
                feature = "finger",
                feature = "pipnn",
                feature = "emg",
                feature = "sparse_mips"
            ))]
            $item
        )*
    };
}

macro_rules! cfg_f32_payload {
    ($($item:item)*) => {
        $(
            #[cfg(any(
                feature = "nsw",
                feature = "sng",
                feature = "vamana",
                feature = "nsg",
                feature = "finger",
                feature = "pipnn",
                feature = "emg",
                feature = "binary_index",
                feature = "rp_quant",
                feature = "sparse_mips",
                feature = "lsh",
                feature = "sq4"
            ))]
            $item
        )*
    };
}

macro_rules! cfg_u32_payload {
    ($($item:item)*) => {
        $(
            #[cfg(any(
                feature = "nsw",
                feature = "sng",
                feature = "vamana",
                feature = "nsg",
                feature = "finger",
                feature = "pipnn",
                feature = "emg",
                feature = "binary_index",
                feature = "rp_quant",
                feature = "sparse_mips",
                feature = "sq4"
            ))]
            $item
        )*
    };
}

macro_rules! cfg_dense_graph_shape {
    ($($item:item)*) => {
        $(
            #[cfg(any(
                feature = "nsw",
                feature = "sng",
                feature = "vamana",
                feature = "nsg",
                feature = "finger",
                feature = "pipnn",
                feature = "emg"
            ))]
            $item
        )*
    };
}

cfg_graph_neighbors! {
    use smallvec::{Array, SmallVec};
    use std::io::Read;
}

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

cfg_f32_payload! {
    pub(crate) fn write_f32_atomic(path: &Path, values: &[f32]) -> Result<(), RetrieveError> {
        write_atomic(path, |writer| {
            for value in values {
                writer.write_all(&value.to_le_bytes())?;
            }
            Ok(())
        })
    }
}

cfg_u32_payload! {
    pub(crate) fn write_u32_atomic(path: &Path, values: &[u32]) -> Result<(), RetrieveError> {
        write_atomic(path, |writer| {
            for value in values {
                writer.write_all(&value.to_le_bytes())?;
            }
            Ok(())
        })
    }
}

#[cfg(feature = "sparse_mips")]
pub(crate) fn write_u64_atomic(path: &Path, values: &[u64]) -> Result<(), RetrieveError> {
    write_atomic(path, |writer| {
        for value in values {
            writer.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    })
}

cfg_graph_neighbors! {
    pub(crate) fn write_neighbors_atomic<A>(
        path: &Path,
        magic: &[u8; 8],
        neighbors: &[SmallVec<A>],
    ) -> Result<(), RetrieveError>
    where
        A: Array<Item = u32>,
    {
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
}

fn write_atomic(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<std::fs::File>) -> std::io::Result<()>,
) -> Result<(), RetrieveError> {
    let (tmp_path, file) = create_temp_file(path)?;
    let result: std::io::Result<()> = (|| {
        let mut writer = BufWriter::new(file);
        write(&mut writer)?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
    }
    result.map_err(Into::into)
}

fn create_temp_file(path: &Path) -> std::io::Result<(PathBuf, std::fs::File)> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "snapshot path has no filename",
        )
    })?;
    let process_id = std::process::id();
    loop {
        let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp_path = parent.join(format!(
            ".{}.tmp.{process_id}.{counter}",
            name.to_string_lossy()
        ));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)
        {
            Ok(file) => return Ok((tmp_path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
}

cfg_f32_payload! {
    pub(crate) fn read_f32_exact(
        path: &Path,
        expected_len: usize,
    ) -> Result<Vec<f32>, RetrieveError> {
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }
}

cfg_u32_payload! {
    pub(crate) fn read_u32_exact(
        path: &Path,
        expected_len: usize,
    ) -> Result<Vec<u32>, RetrieveError> {
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }
}

#[cfg(feature = "sparse_mips")]
pub(crate) fn read_u64_exact(path: &Path, expected_len: usize) -> Result<Vec<u64>, RetrieveError> {
    let bytes = std::fs::read(path)?;
    let expected_bytes = expected_len
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or_else(|| RetrieveError::FormatError("u64 byte length overflow".into()))?;
    if bytes.len() != expected_bytes {
        return Err(RetrieveError::FormatError(format!(
            "{} size mismatch: expected {} bytes, got {}",
            path.display(),
            expected_bytes,
            bytes.len()
        )));
    }
    Ok(bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| {
            u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect())
}

cfg_graph_neighbors! {
    pub(crate) fn read_neighbors<A>(
        path: &Path,
        magic: &[u8; 8],
        expected_nodes: usize,
    ) -> Result<Vec<SmallVec<A>>, RetrieveError>
    where
        A: Array<Item = u32>,
    {
        let mut reader = BufReader::new(std::fs::File::open(path)?);
        let mut actual_magic = [0u8; 8];
        reader.read_exact(&mut actual_magic)?;
        if &actual_magic != magic {
            return Err(RetrieveError::FormatError(format!(
                "invalid graph neighbors magic in {}",
                path.display()
            )));
        }
        let count = usize::try_from(read_one_u64(&mut reader)?).map_err(|_| {
            RetrieveError::FormatError("neighbor list count exceeds usize".into())
        })?;
        if count != expected_nodes {
            return Err(RetrieveError::FormatError(format!(
                "neighbor list count {} does not match manifest count {}",
                count, expected_nodes
            )));
        }

        let mut neighbors = Vec::with_capacity(expected_nodes);
        for node in 0..expected_nodes {
            let len = usize::try_from(read_one_u64(&mut reader)?).map_err(|_| {
                RetrieveError::FormatError(format!("node {node} neighbor count exceeds usize"))
            })?;
            let max_reasonable_degree = expected_nodes.saturating_mul(4).max(64);
            if len > max_reasonable_degree {
                return Err(RetrieveError::FormatError(format!(
                    "node {node} has too many neighbors: {len}"
                )));
            }
            let mut list = SmallVec::<A>::new();
            for _ in 0..len {
                let id = read_one_u32(&mut reader)?;
                let id_usize = usize::try_from(id).map_err(|_| {
                    RetrieveError::FormatError(format!("neighbor id {id} exceeds usize"))
                })?;
                if id_usize >= expected_nodes {
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
}

cfg_dense_graph_shape! {
    pub(crate) fn validate_graph_shape<A>(
        name: &str,
        dimension: usize,
        num_vectors: usize,
        vectors: &[f32],
        doc_ids: &[u32],
        neighbors: &[SmallVec<A>],
        entry: Option<u32>,
    ) -> Result<(), RetrieveError>
    where
        A: Array<Item = u32>,
    {
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
        let expected_vector_len = num_vectors.checked_mul(dimension).ok_or_else(|| {
            RetrieveError::FormatError(format!("{name} vector length overflow"))
        })?;
        if vectors.len() != expected_vector_len {
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
            let entry_usize = usize::try_from(entry).map_err(|_| {
                RetrieveError::FormatError(format!("{name} entry node {entry} exceeds usize"))
            })?;
            if entry_usize >= num_vectors {
                return Err(RetrieveError::FormatError(format!(
                    "{name} entry node {entry} exceeds vector count {num_vectors}"
                )));
            }
        }
        Ok(())
    }
}

cfg_graph_neighbors! {
    fn read_one_u64(reader: &mut impl Read) -> Result<u64, RetrieveError> {
        let mut bytes = [0u8; 8];
        reader.read_exact(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_one_u32(reader: &mut impl Read) -> Result<u32, RetrieveError> {
        let mut bytes = [0u8; 4];
        reader.read_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }
}
