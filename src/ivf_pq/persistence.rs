use super::cluster::Cluster;
use super::manifest::IVFPQManifest;
use crate::RetrieveError;
use serde::{Deserialize, Serialize};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

pub(super) const IVFPQ_FORMAT_VERSION: u32 = 1;
const IVFPQ_CLUSTER_MAGIC: &[u8; 8] = b"VICIVF1\0";

pub(super) fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), RetrieveError> {
    write_atomic(path, |writer| {
        serde_json::to_writer_pretty(writer, value)
            .map_err(|e| std::io::Error::other(e.to_string()))
    })
}

pub(super) fn write_f32_atomic(path: &Path, values: &[f32]) -> Result<(), RetrieveError> {
    write_atomic(path, |writer| {
        for value in values {
            writer.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    })
}

pub(super) fn write_u32_atomic(path: &Path, values: &[u32]) -> Result<(), RetrieveError> {
    write_atomic(path, |writer| {
        for value in values {
            writer.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    })
}

pub(super) fn write_u64_atomic(path: &Path, values: &[u64]) -> Result<(), RetrieveError> {
    write_atomic(path, |writer| {
        for value in values {
            writer.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    })
}

pub(super) fn write_bytes_atomic(path: &Path, bytes: &[u8]) -> Result<(), RetrieveError> {
    write_atomic(path, |writer| writer.write_all(bytes))
}

pub(super) fn write_clusters_atomic(
    path: &Path,
    clusters: &[Cluster],
) -> Result<(), RetrieveError> {
    write_atomic(path, |writer| {
        writer.write_all(IVFPQ_CLUSTER_MAGIC)?;
        writer.write_all(&(clusters.len() as u64).to_le_bytes())?;
        for cluster in clusters {
            writer.write_all(&cluster.filter_bitmask.to_le_bytes())?;
            let ids = cluster.get_ids_ref();
            writer.write_all(&(ids.len() as u64).to_le_bytes())?;
            for id in ids.as_ref() {
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

pub(super) fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, RetrieveError> {
    let file = std::fs::File::open(path)?;
    serde_json::from_reader(BufReader::new(file))
        .map_err(|e| RetrieveError::FormatError(e.to_string()))
}

pub(super) fn read_f32_exact(path: &Path, expected_len: usize) -> Result<Vec<f32>, RetrieveError> {
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
    let mut values = Vec::with_capacity(expected_len);
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

pub(super) fn read_u32_exact(path: &Path, expected_len: usize) -> Result<Vec<u32>, RetrieveError> {
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
    let mut values = Vec::with_capacity(expected_len);
    for chunk in bytes.chunks_exact(4) {
        values.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

pub(super) fn read_bytes_exact(path: &Path, expected_len: usize) -> Result<Vec<u8>, RetrieveError> {
    let bytes = std::fs::read(path)?;
    if bytes.len() != expected_len {
        return Err(RetrieveError::FormatError(format!(
            "{} size mismatch: expected {} bytes, got {}",
            path.display(),
            expected_len,
            bytes.len()
        )));
    }
    Ok(bytes)
}

pub(super) fn validate_manifest(manifest: &IVFPQManifest) -> Result<(), RetrieveError> {
    if manifest.version != IVFPQ_FORMAT_VERSION {
        return Err(RetrieveError::FormatError(format!(
            "unsupported IVF-PQ format version {}",
            manifest.version
        )));
    }
    if manifest.dimension == 0 {
        return Err(RetrieveError::FormatError(
            "IVF-PQ manifest has zero dimension".into(),
        ));
    }
    if manifest.num_vectors == 0 {
        return Err(RetrieveError::FormatError(
            "IVF-PQ manifest has zero vectors".into(),
        ));
    }
    if manifest.num_centroids == 0 {
        return Err(RetrieveError::FormatError(
            "IVF-PQ manifest has zero centroids".into(),
        ));
    }
    Ok(())
}

pub(super) fn read_clusters(
    path: &Path,
    expected_clusters: usize,
    num_vectors: usize,
) -> Result<Vec<Cluster>, RetrieveError> {
    let mut reader = BufReader::new(std::fs::File::open(path)?);
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != IVFPQ_CLUSTER_MAGIC {
        return Err(RetrieveError::FormatError(
            "invalid IVF-PQ cluster file magic".into(),
        ));
    }
    let cluster_count = read_u64(&mut reader)? as usize;
    if cluster_count != expected_clusters {
        return Err(RetrieveError::FormatError(format!(
            "cluster count mismatch: expected {}, got {}",
            expected_clusters, cluster_count
        )));
    }

    let mut clusters = Vec::with_capacity(cluster_count);
    for _ in 0..cluster_count {
        let filter_bitmask = read_u64(&mut reader)?;
        let len = read_u64(&mut reader)? as usize;
        if len > num_vectors {
            return Err(RetrieveError::FormatError(format!(
                "cluster length {} exceeds vector count {}",
                len, num_vectors
            )));
        }
        let mut ids = Vec::with_capacity(len);
        for _ in 0..len {
            let id = read_u32(&mut reader)?;
            if id as usize >= num_vectors {
                return Err(RetrieveError::FormatError(format!(
                    "cluster id {} exceeds vector count {}",
                    id, num_vectors
                )));
            }
            ids.push(id);
        }
        clusters.push(Cluster::new(ids, filter_bitmask));
    }

    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        return Err(RetrieveError::FormatError(
            "trailing bytes in IVF-PQ cluster file".into(),
        ));
    }

    Ok(clusters)
}

fn read_u64(reader: &mut impl Read) -> Result<u64, RetrieveError> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_u32(reader: &mut impl Read) -> Result<u32, RetrieveError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}
