use super::cluster::Cluster;
use crate::RetrieveError;
#[cfg(feature = "persistence")]
use durability::mmap::{AccessPattern, MappedFile};
use std::path::Path;

pub(super) enum IVFPQByteStorage {
    File(std::fs::File),
    #[cfg(feature = "persistence")]
    Mmap(Box<MappedFile>),
}

pub(super) struct IVFPQListCodeStorage {
    offsets: Vec<u64>,
    codes: IVFPQByteStorage,
}

pub(super) fn checked_len(lhs: usize, rhs: usize, message: &str) -> Result<usize, RetrieveError> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| RetrieveError::FormatError(message.into()))
}

pub(super) fn open_byte_storage(
    path: &Path,
    expected_len: usize,
    mmap: bool,
) -> Result<IVFPQByteStorage, RetrieveError> {
    let actual_len = std::fs::metadata(path)?.len() as usize;
    if actual_len != expected_len {
        return Err(RetrieveError::FormatError(format!(
            "{} size mismatch: expected {} bytes, got {}",
            path.display(),
            expected_len,
            actual_len
        )));
    }

    #[cfg(feature = "persistence")]
    if mmap {
        let mapped = MappedFile::open(path, AccessPattern::Random).map_err(|e| {
            RetrieveError::Io(std::sync::Arc::new(std::io::Error::other(format!(
                "failed to mmap {}: {e}",
                path.display()
            ))))
        })?;
        if mapped.as_slice().len() != expected_len {
            return Err(RetrieveError::FormatError(format!(
                "{} mmap size mismatch: expected {} bytes, got {}",
                path.display(),
                expected_len,
                mapped.as_slice().len()
            )));
        }
        return Ok(IVFPQByteStorage::Mmap(Box::new(mapped)));
    }

    let _ = mmap;
    Ok(IVFPQByteStorage::File(std::fs::File::open(path)?))
}

pub(super) fn build_list_codes(
    clusters: &[Cluster],
    quantized_codes: &[u8],
    num_codebooks: usize,
) -> Result<(Vec<u64>, Vec<u8>), RetrieveError> {
    let mut offsets = Vec::with_capacity(clusters.len() + 1);
    let mut codes = Vec::with_capacity(quantized_codes.len());
    offsets.push(0);
    for cluster in clusters {
        let ids = cluster.get_ids_ref();
        for &vector_idx in ids.as_ref() {
            let start = checked_len(
                vector_idx as usize,
                num_codebooks,
                "IVF-PQ list-code offset overflow",
            )?;
            let end = start.checked_add(num_codebooks).ok_or_else(|| {
                RetrieveError::FormatError("IVF-PQ list-code end overflow".into())
            })?;
            if end > quantized_codes.len() {
                return Err(RetrieveError::FormatError(format!(
                    "IVF-PQ cluster references code range {}..{} beyond {} bytes",
                    start,
                    end,
                    quantized_codes.len()
                )));
            }
            codes.extend_from_slice(&quantized_codes[start..end]);
        }
        offsets.push(codes.len() as u64);
    }
    Ok((offsets, codes))
}

pub(super) fn open_list_code_storage(
    input_dir: &Path,
    num_clusters: usize,
    expected_codes_len: usize,
    mmap: bool,
) -> Result<Option<IVFPQListCodeStorage>, RetrieveError> {
    let offsets_path = input_dir.join("list_offsets.bin");
    let codes_path = input_dir.join("list_codes.bin");
    let offsets_exists = offsets_path.exists();
    let codes_exists = codes_path.exists();
    if offsets_exists != codes_exists {
        return Err(RetrieveError::FormatError(
            "partial IVF-PQ list-code sidecar: expected both list_offsets.bin and list_codes.bin"
                .into(),
        ));
    }
    if !offsets_exists {
        return Ok(None);
    }

    let offsets = read_u64_exact(&offsets_path, num_clusters + 1)?;
    validate_list_code_offsets(&offsets, expected_codes_len)?;
    let codes = open_byte_storage(&codes_path, expected_codes_len, mmap)?;
    Ok(Some(IVFPQListCodeStorage { offsets, codes }))
}

pub(super) fn append_codes_for_ids(
    storage: &mut IVFPQByteStorage,
    out: &mut Vec<u8>,
    ids: &[u32],
    num_codebooks: usize,
) -> Result<(), RetrieveError> {
    out.clear();
    out.reserve(ids.len() * num_codebooks);
    for &vector_idx in ids {
        let old_len = out.len();
        out.resize(old_len + num_codebooks, 0);
        read_bytes_from_storage(
            storage,
            vector_idx as usize * num_codebooks,
            &mut out[old_len..old_len + num_codebooks],
        )?;
    }
    Ok(())
}

pub(super) fn read_list_codes_for_cluster(
    storage: &mut IVFPQListCodeStorage,
    cluster_idx: usize,
    num_ids: usize,
    num_codebooks: usize,
    out: &mut Vec<u8>,
) -> Result<(), RetrieveError> {
    let start = *storage.offsets.get(cluster_idx).ok_or_else(|| {
        RetrieveError::FormatError("IVF-PQ list-code cluster offset missing".into())
    })?;
    let end = *storage
        .offsets
        .get(cluster_idx + 1)
        .ok_or_else(|| RetrieveError::FormatError("IVF-PQ list-code cluster end missing".into()))?;
    let start = usize::try_from(start)
        .map_err(|_| RetrieveError::FormatError("IVF-PQ list-code start overflow".into()))?;
    let end = usize::try_from(end)
        .map_err(|_| RetrieveError::FormatError("IVF-PQ list-code end overflow".into()))?;
    let expected_len = checked_len(
        num_ids,
        num_codebooks,
        "IVF-PQ list-code cluster length overflow",
    )?;
    let actual_len = end
        .checked_sub(start)
        .ok_or_else(|| RetrieveError::FormatError("IVF-PQ list-code negative range".into()))?;
    if actual_len != expected_len {
        return Err(RetrieveError::FormatError(format!(
            "IVF-PQ list-code cluster has {} bytes, expected {}",
            actual_len, expected_len
        )));
    }
    out.resize(actual_len, 0);
    read_bytes_from_storage(&mut storage.codes, start, out)?;
    Ok(())
}

pub(super) fn read_code_from_storage<'a>(
    storage: &mut IVFPQByteStorage,
    out: &'a mut Vec<u8>,
    vector_idx: usize,
    num_codebooks: usize,
) -> Result<&'a [u8], RetrieveError> {
    out.resize(num_codebooks, 0);
    read_bytes_from_storage(storage, vector_idx * num_codebooks, out)?;
    Ok(out)
}

pub(super) fn read_vector_from_storage<'a>(
    storage: &mut IVFPQByteStorage,
    bytes: &mut [u8],
    out: &'a mut [f32],
    vector_idx: usize,
    dimension: usize,
) -> Result<&'a [f32], RetrieveError> {
    let byte_len = checked_len(
        dimension,
        std::mem::size_of::<f32>(),
        "IVF-PQ vector byte length overflow",
    )?;
    let offset = checked_len(vector_idx, byte_len, "IVF-PQ vector byte offset overflow")?;
    if bytes.len() != byte_len {
        return Err(RetrieveError::InvalidParameter(format!(
            "IVF-PQ vector byte buffer has {} bytes, expected {}",
            bytes.len(),
            byte_len
        )));
    }
    read_bytes_from_storage(storage, offset, bytes)?;
    for (value, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        *value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(out)
}

fn read_bytes_from_storage(
    storage: &mut IVFPQByteStorage,
    offset: usize,
    out: &mut [u8],
) -> Result<(), RetrieveError> {
    #[cfg(feature = "persistence")]
    let end = offset
        .checked_add(out.len())
        .ok_or_else(|| RetrieveError::FormatError("IVF-PQ byte offset overflow".into()))?;
    match storage {
        IVFPQByteStorage::File(file) => {
            crate::file_io::read_exact_at(file, offset as u64, out)?;
        }
        #[cfg(feature = "persistence")]
        IVFPQByteStorage::Mmap(mapped) => {
            let bytes = mapped.as_slice();
            if end > bytes.len() {
                return Err(RetrieveError::FormatError(format!(
                    "IVF-PQ storage read out of bounds: end {} > len {}",
                    end,
                    bytes.len()
                )));
            }
            out.copy_from_slice(&bytes[offset..end]);
        }
    }
    Ok(())
}

fn read_u64_exact(path: &Path, expected_len: usize) -> Result<Vec<u64>, RetrieveError> {
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
    let mut values = Vec::with_capacity(expected_len);
    for chunk in bytes.chunks_exact(8) {
        values.push(u64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]));
    }
    Ok(values)
}

fn validate_list_code_offsets(
    offsets: &[u64],
    expected_codes_len: usize,
) -> Result<(), RetrieveError> {
    if offsets.first().copied() != Some(0) {
        return Err(RetrieveError::FormatError(
            "IVF-PQ list-code offsets must start at zero".into(),
        ));
    }
    for pair in offsets.windows(2) {
        if pair[0] > pair[1] {
            return Err(RetrieveError::FormatError(
                "IVF-PQ list-code offsets must be nondecreasing".into(),
            ));
        }
    }
    if offsets.last().copied() != Some(expected_codes_len as u64) {
        return Err(RetrieveError::FormatError(format!(
            "IVF-PQ list-code offsets end at {}, expected {}",
            offsets.last().copied().unwrap_or_default(),
            expected_codes_len
        )));
    }
    Ok(())
}
