//! Checkpoint creation and loading.
//!
//! Checkpoints are full snapshots of the index state, enabling fast recovery
//! and efficient backups.

use crate::persistence::directory::Directory;
use crate::persistence::error::{PersistenceError, PersistenceResult};
use crate::persistence::format::CHECKPOINT_MAGIC;
use crc32fast::Hasher;
use std::io::{Read, Write};
use std::sync::Arc;

#[cfg(feature = "persistence")]
use postcard;

/// Checkpoint header.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CheckpointHeader {
    /// Magic bytes: b"CHKP"
    pub magic: [u8; 4],
    /// Format version
    pub format_version: u32,
    /// Last entry ID covered by checkpoint
    pub entry_id: u64,
    /// Number of segments
    pub segment_count: u32,
    /// Offset to segment list
    pub segment_list_offset: u64,
    /// Total document count
    pub doc_count: u64,
    /// Unix timestamp
    pub created_at: u64,
    /// CRC32 checksum of header
    pub checksum: u32,
}

impl CheckpointHeader {
    /// Size of checkpoint header in bytes.
    /// Calculated as: magic(4) + format_version(4) + entry_id(8) + segment_count(4) +
    /// segment_list_offset(8) + doc_count(8) + created_at(8) + checksum(4) = 48 bytes
    /// Note: This is the actual written size, not std::mem::size_of which includes padding.
    pub const SIZE: usize = 4 + 4 + 8 + 4 + 8 + 8 + 8 + 4; // 48 bytes

    /// Write header to a writer (little-endian).
    pub fn write<W: Write>(&self, writer: &mut W) -> PersistenceResult<()> {
        use byteorder::{LittleEndian, WriteBytesExt};

        writer.write_all(&self.magic)?;
        writer.write_u32::<LittleEndian>(self.format_version)?;
        writer.write_u64::<LittleEndian>(self.entry_id)?;
        writer.write_u32::<LittleEndian>(self.segment_count)?;
        writer.write_u64::<LittleEndian>(self.segment_list_offset)?;
        writer.write_u64::<LittleEndian>(self.doc_count)?;
        writer.write_u64::<LittleEndian>(self.created_at)?;
        writer.write_u32::<LittleEndian>(self.checksum)?;

        Ok(())
    }

    /// Read header from a reader (little-endian).
    pub fn read<R: Read>(reader: &mut R) -> PersistenceResult<Self> {
        use byteorder::{LittleEndian, ReadBytesExt};

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;

        if magic != CHECKPOINT_MAGIC {
            return Err(PersistenceError::Format(format!(
                "Invalid checkpoint magic bytes (expected: {:?}, actual: {:?})",
                CHECKPOINT_MAGIC, magic
            )));
        }

        let format_version = reader.read_u32::<LittleEndian>()?;
        let entry_id = reader.read_u64::<LittleEndian>()?;
        let segment_count = reader.read_u32::<LittleEndian>()?;
        let segment_list_offset = reader.read_u64::<LittleEndian>()?;
        let doc_count = reader.read_u64::<LittleEndian>()?;
        let created_at = reader.read_u64::<LittleEndian>()?;
        let checksum = reader.read_u32::<LittleEndian>()?;

        Ok(Self {
            magic,
            format_version,
            entry_id,
            segment_count,
            segment_list_offset,
            doc_count,
            created_at,
            checksum,
        })
    }

    /// Validate checksum of checkpoint header and segment list.
    ///
    /// Reads the segment list from the reader and verifies the checksum.
    /// The checksum is computed over: magic + format_version + entry_id + segment_count +
    /// segment_list_offset + doc_count + created_at + segment_list_bytes
    pub fn validate_checksum<R: Read>(&self, reader: &mut R) -> PersistenceResult<()> {
        // Read entire file to get segment list
        let mut all_data = Vec::new();
        reader.read_to_end(&mut all_data)?;

        // Extract segment list bytes
        let segment_list_start = self.segment_list_offset as usize;
        if segment_list_start > all_data.len() {
            return Err(PersistenceError::Format(format!(
                "Segment list offset beyond file size (expected < {}, actual: {})",
                all_data.len(),
                segment_list_start
            )));
        }
        let segment_list_bytes = &all_data[segment_list_start..];

        // Compute expected checksum (same as in create_checkpoint)
        let mut hasher = Hasher::new();
        hasher.update(&self.magic);
        hasher.update(&self.format_version.to_le_bytes());
        hasher.update(&self.entry_id.to_le_bytes());
        hasher.update(&self.segment_count.to_le_bytes());
        hasher.update(&self.segment_list_offset.to_le_bytes());
        hasher.update(&self.doc_count.to_le_bytes());
        hasher.update(&self.created_at.to_le_bytes());
        hasher.update(segment_list_bytes);
        let expected_checksum = hasher.finalize();

        if expected_checksum != self.checksum {
            return Err(PersistenceError::ChecksumMismatch {
                expected: self.checksum,
                actual: expected_checksum,
            });
        }

        Ok(())
    }
}

/// Segment metadata in checkpoint.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SegmentMetadata {
    /// Segment ID
    pub segment_id: u64,
    /// Relative path to segment directory
    pub path: String,
    /// Number of documents in segment
    pub doc_count: u32,
    /// Maximum document ID in segment
    pub max_doc_id: u32,
    /// Size of segment in bytes
    pub size_bytes: u64,
}

/// Checkpoint writer for creating checkpoints.
pub struct CheckpointWriter {
    directory: Arc<dyn Directory>,
}

impl CheckpointWriter {
    /// Create a new checkpoint writer.
    pub fn new(directory: Box<dyn Directory>) -> Self {
        Self {
            directory: Arc::<dyn Directory>::from(directory),
        }
    }

    /// Create a new checkpoint writer from an `Arc` directory.
    pub fn new_arc(directory: impl Into<Arc<dyn Directory>>) -> Self {
        Self {
            directory: directory.into(),
        }
    }

    /// Create a checkpoint from current index state.
    ///
    /// This copies all segment files and creates checkpoint metadata.
    pub fn create_checkpoint(
        &self,
        entry_id: u64,
        segments: &[SegmentMetadata],
    ) -> PersistenceResult<String> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let checkpoint_id = entry_id;
        let checkpoint_path = format!("checkpoints/checkpoint_{}.bin", checkpoint_id);

        self.directory.create_dir_all("checkpoints")?;

        // NOTE: compute timestamp once; it is part of the checksum and header.
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| PersistenceError::InvalidState(format!("system clock error: {e}")))?
            .as_secs();

        // Serialize segment list using postcard (format-stable, smaller size for long-term retention)
        let segment_list_bytes = postcard::to_allocvec(segments).map_err(|e| {
            PersistenceError::Serialization(format!(
                "Failed to serialize segment list with postcard: {}",
                e
            ))
        })?;

        // Compute checksum over header (excluding checksum field) + segment list
        let mut hasher = Hasher::new();
        hasher.update(&CHECKPOINT_MAGIC);
        hasher.update(&1u32.to_le_bytes()); // format_version
        hasher.update(&entry_id.to_le_bytes());
        hasher.update(&(segments.len() as u32).to_le_bytes());
        hasher.update(&(CheckpointHeader::SIZE as u64).to_le_bytes()); // segment_list_offset
        hasher.update(
            &segments
                .iter()
                .map(|s| s.doc_count as u64)
                .sum::<u64>()
                .to_le_bytes(),
        ); // doc_count
        hasher.update(&created_at.to_le_bytes()); // created_at
        hasher.update(&segment_list_bytes);
        let checksum = hasher.finalize();

        // Write header with computed checksum
        let header = CheckpointHeader {
            magic: CHECKPOINT_MAGIC,
            format_version: 1,
            entry_id,
            segment_count: segments.len() as u32,
            segment_list_offset: CheckpointHeader::SIZE as u64, // After header
            doc_count: segments.iter().map(|s| s.doc_count as u64).sum(),
            created_at,
            checksum,
        };

        // Serialize header + segment list
        let mut checkpoint_data = Vec::new();
        header.write(&mut checkpoint_data)?;
        checkpoint_data.extend_from_slice(&segment_list_bytes);

        // Use atomic write for crash safety
        self.directory
            .atomic_write(&checkpoint_path, &checkpoint_data)?;

        // Copy segment files to checkpoint directory
        // This ensures checkpoint is self-contained
        let checkpoint_segments_dir = format!("checkpoints/checkpoint_{}/segments", checkpoint_id);
        self.directory.create_dir_all(&checkpoint_segments_dir)?;

        for segment in segments {
            let source_segment_dir = &segment.path;
            let dest_segment_dir =
                format!("{}/segment_{}", checkpoint_segments_dir, segment.segment_id);

            // Copy all files in segment directory
            if self.directory.exists(source_segment_dir) {
                self.directory.create_dir_all(&dest_segment_dir)?;

                // List files in source segment
                let files = self.directory.list_dir(source_segment_dir)?;
                for file_name in files {
                    let source_file = format!("{}/{}", source_segment_dir, file_name);
                    let dest_file = format!("{}/{}", dest_segment_dir, file_name);

                    // Read source file
                    let mut source_reader = self.directory.open_file(&source_file)?;
                    let mut file_data = Vec::new();
                    source_reader.read_to_end(&mut file_data)?;

                    // Write to destination
                    self.directory.atomic_write(&dest_file, &file_data)?;
                }
            }
        }

        Ok(checkpoint_path)
    }
}

/// Checkpoint reader for loading checkpoints.
pub struct CheckpointReader {
    directory: Arc<dyn Directory>,
}

impl CheckpointReader {
    /// Create a new checkpoint reader.
    pub fn new(directory: Box<dyn Directory>) -> Self {
        Self {
            directory: Arc::<dyn Directory>::from(directory),
        }
    }

    /// Create a new checkpoint reader from an `Arc` directory.
    pub fn new_arc(directory: impl Into<Arc<dyn Directory>>) -> Self {
        Self {
            directory: directory.into(),
        }
    }

    /// Load a checkpoint from disk.
    pub fn load_checkpoint(&self, checkpoint_path: &str) -> PersistenceResult<CheckpointHeader> {
        let mut file = self.directory.open_file(checkpoint_path)?;
        let header = CheckpointHeader::read(&mut file)?;

        // Validate checksum
        let mut file_for_checksum = self.directory.open_file(checkpoint_path)?;
        header.validate_checksum(&mut file_for_checksum)?;

        Ok(header)
    }

    /// Load checkpoint header and segment metadata.
    pub fn load_checkpoint_with_segments(
        &self,
        checkpoint_path: &str,
    ) -> PersistenceResult<(CheckpointHeader, Vec<SegmentMetadata>)> {
        // Read entire file first (simpler than seeking with trait objects)
        let mut file = self.directory.open_file(checkpoint_path)?;
        let mut all_data = Vec::new();
        file.read_to_end(&mut all_data)?;

        // Parse header from beginning
        let mut header_reader = std::io::Cursor::new(&all_data);
        let header = CheckpointHeader::read(&mut header_reader)?;

        // Extract segment list bytes
        let segment_list_start = header.segment_list_offset as usize;
        if segment_list_start > all_data.len() {
            return Err(PersistenceError::Format(format!(
                "Segment list offset beyond file size (expected < {}, actual: {})",
                all_data.len(),
                segment_list_start
            )));
        }

        let segment_list_bytes = &all_data[segment_list_start..];
        let segments: Vec<SegmentMetadata> =
            postcard::from_bytes(segment_list_bytes).map_err(|e| {
                PersistenceError::Deserialization(format!(
                    "Failed to deserialize segment list: {}",
                    e
                ))
            })?;

        // Validate checksum
        let mut file_for_checksum = self.directory.open_file(checkpoint_path)?;
        header.validate_checksum(&mut file_for_checksum)?;

        Ok((header, segments))
    }

    /// List all available checkpoints.
    pub fn list_checkpoints(&self) -> PersistenceResult<Vec<String>> {
        if !self.directory.exists("checkpoints") {
            return Ok(Vec::new());
        }

        let mut checkpoints = self.directory.list_dir("checkpoints")?;
        checkpoints.retain(|f| f.ends_with(".bin"));
        checkpoints.sort();
        Ok(checkpoints)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use crate::persistence::directory::MemoryDirectory;

    #[test]
    fn test_checkpoint_header_roundtrip() {
        let mut buffer = Vec::new();
        let header = CheckpointHeader {
            magic: CHECKPOINT_MAGIC,
            format_version: 1,
            entry_id: 100,
            segment_count: 5,
            segment_list_offset: 64,
            doc_count: 1000,
            created_at: 1234567890,
            checksum: 42,
        };

        header.write(&mut buffer).unwrap();
        assert_eq!(buffer.len(), CheckpointHeader::SIZE);

        let mut reader = std::io::Cursor::new(&buffer);
        let read_header = CheckpointHeader::read(&mut reader).unwrap();

        assert_eq!(read_header.magic, header.magic);
        assert_eq!(read_header.entry_id, header.entry_id);
        assert_eq!(read_header.segment_count, header.segment_count);
    }
}
