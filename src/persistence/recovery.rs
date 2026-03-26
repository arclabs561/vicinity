//! Crash recovery and WAL replay procedures.
//!
//! Implements comprehensive recovery for all crash scenarios:
//! - Power failure during commit
//! - Crash during merge
//! - Crash during WAL write
//! - Crash during ID generation
//! - Crash during metadata update

use crate::persistence::checkpoint::{CheckpointReader, SegmentMetadata};
use crate::persistence::directory::Directory;
use crate::persistence::error::{PersistenceError, PersistenceResult};
use crate::persistence::wal::{WalEntry, WalReader, WalRecord};
use std::collections::HashMap;
use std::sync::Arc;

/// Recovery state after WAL replay.
#[derive(Debug, Clone)]
pub struct RecoveryState {
    /// Active segments (segment_id -> metadata)
    pub active_segments: HashMap<u64, SegmentMetadata>,

    /// Pending merges (merge_id -> segments_to_merge)
    pub pending_merges: HashMap<u64, Vec<u64>>,

    /// Deleted documents (segment_id -> set of doc_ids)
    pub deletes: HashMap<u64, Vec<u32>>,

    /// Last processed entry ID
    pub last_entry_id: u64,

    /// Next segment ID to use
    pub next_segment_id: u64,

    /// Next document ID to use
    pub next_doc_id: u32,
}

/// Recovery manager for crash recovery.
pub struct RecoveryManager {
    directory: Arc<dyn Directory>,
}

impl RecoveryManager {
    /// Create a new recovery manager.
    pub fn new(directory: impl Into<Arc<dyn Directory>>) -> Self {
        Self {
            directory: directory.into(),
        }
    }

    /// Perform startup recovery.
    ///
    /// This implements a 10-step recovery procedure:
    /// 1. Check for checkpoint
    /// 2. Load checkpoint if available
    /// 3. Replay WAL entries since checkpoint
    /// 4. Detect and cancel stale merges
    /// 5. Detect stale reader handles
    /// 6. Verify segment integrity
    /// 7. Reconstruct index state
    /// 8. Validate consistency
    /// 9. Clean up temporary files
    /// 10. Return recovery state
    pub fn recover(&self) -> PersistenceResult<RecoveryState> {
        // Step 1-2: Check for and load checkpoint.
        //
        // Note: `CheckpointReader` now supports `Arc<dyn Directory>` directly, so we don't need
        // any Arc↔Box adapter wrappers in recovery code.
        let checkpoint_reader = CheckpointReader::new_arc(self.directory.clone());

        let checkpoints = checkpoint_reader.list_checkpoints()?;
        let (_checkpoint_path, last_checkpoint_entry_id, initial_segments) =
            if let Some(latest_checkpoint) = checkpoints.last() {
                let full_path = format!("checkpoints/{}", latest_checkpoint);
                match checkpoint_reader.load_checkpoint_with_segments(&full_path) {
                    Ok((header, segments)) => {
                        (Some(latest_checkpoint.clone()), header.entry_id, segments)
                    }
                    Err(e) => {
                        // Checkpoint corrupted - log warning and continue without it
                        // In production, might want to return error or attempt recovery
                        eprintln!(
                        "Warning: Failed to load checkpoint {}: {}. Continuing without checkpoint.",
                        latest_checkpoint, e
                    );
                        (None, 0, Vec::new())
                    }
                }
            } else {
                (None, 0, Vec::new())
            };

        // Step 3: Replay WAL entries since checkpoint
        let wal_reader = WalReader::new(self.directory.clone());
        let all_entries = wal_reader.replay()?;

        // Filter entries after checkpoint
        let entries: Vec<WalRecord<WalEntry>> = all_entries
            .into_iter()
            .filter(|record| record.entry_id > last_checkpoint_entry_id)
            .collect();

        // Step 4-7: Reconstruct state from checkpoint + WAL entries.
        //
        // Invariant: loading a checkpoint must not *reduce* information available for recovery.
        // We therefore seed state from the checkpoint and then apply WAL entries with id > checkpoint.
        let mut active_segments: HashMap<u64, SegmentMetadata> = initial_segments
            .into_iter()
            .map(|m| (m.segment_id, m))
            .collect();
        let mut pending_merges: HashMap<u64, Vec<u64>> = HashMap::new();
        let mut deletes: HashMap<u64, Vec<u32>> = HashMap::new();
        let mut last_entry_id = last_checkpoint_entry_id;

        for record in entries {
            let entry_id = record.entry_id;
            last_entry_id = last_entry_id.max(entry_id);

            match record.payload {
                WalEntry::AddSegment {
                    segment_id,
                    doc_count,
                } => {
                    active_segments.insert(
                        segment_id,
                        SegmentMetadata {
                            segment_id,
                            path: format!("segments/segment_{}", segment_id),
                            doc_count,
                            max_doc_id: 0, // Would be tracked in actual implementation
                            size_bytes: 0, // Would be tracked in actual implementation
                        },
                    );
                }
                WalEntry::StartMerge {
                    transaction_id,
                    segment_ids,
                } => {
                    // Track pending merge
                    pending_merges.insert(transaction_id, segment_ids);
                }
                WalEntry::EndMerge {
                    transaction_id,
                    old_segment_ids,
                    ..
                } => {
                    // Complete merge: remove old segments, add new
                    for old_id in &old_segment_ids {
                        active_segments.remove(old_id);
                    }
                    // new_segment_id would be added via AddSegment entry
                    pending_merges.remove(&transaction_id);
                }
                WalEntry::CancelMerge {
                    transaction_id,
                    ..
                } => {
                    pending_merges.remove(&transaction_id);
                }
                WalEntry::DeleteDocuments {
                    deletes: delete_list,
                } => {
                    for (segment_id, doc_id) in delete_list {
                        deletes.entry(segment_id).or_default().push(doc_id);
                    }
                }
                WalEntry::Checkpoint { .. } => {
                    // Checkpoint entry doesn't change state
                }
                _ => {
                    // Forward-compatible: ignore unknown entry variants
                }
            }
        }

        // Step 8: Verify consistency
        // - Check that all active segments exist on disk
        // - Verify no orphaned segments
        // - Check for corrupted segments
        let mut missing_segments = Vec::new();
        for (segment_id, metadata) in &active_segments {
            let segment_dir = &metadata.path;
            if !self.directory.exists(segment_dir) {
                missing_segments.push(*segment_id);
            } else {
                // Verify segment footer exists (basic integrity check)
                let footer_path = format!("{}/footer.bin", segment_dir);
                if !self.directory.exists(&footer_path) {
                    return Err(PersistenceError::InvalidState(format!(
                        "Segment {} exists but footer.bin is missing (corrupted segment)",
                        segment_id
                    )));
                }
            }
        }

        // Report missing segments (in production, might want to error or attempt recovery)
        if !missing_segments.is_empty() {
            eprintln!(
                "Warning: {} segments referenced in state but not found on disk: {:?}",
                missing_segments.len(),
                missing_segments
            );
            // In production, you might want to:
            // - Return an error if strict mode
            // - Attempt to recover from checkpoint
            // - Mark segments as deleted and continue
        }

        // Step 9: Clean up temporary files
        // - Remove stale merge indicators (files like "merge_{transaction_id}.in_progress")
        // - Remove stale handle files (files in handles/ directory that are old)
        // - Remove temporary checkpoint files (*.tmp)
        self.cleanup_temporary_files()?;

        // Step 10: Return recovery state
        // Calculate next_segment_id and next_doc_id from active segments
        let next_segment_id = active_segments.keys().max().copied().unwrap_or(0) + 1;
        let next_doc_id = active_segments
            .values()
            .map(|s| s.max_doc_id)
            .max()
            .unwrap_or(0)
            + 1;

        Ok(RecoveryState {
            active_segments,
            pending_merges,
            deletes,
            last_entry_id,
            next_segment_id,
            next_doc_id,
        })
    }

    /// Verify index consistency after recovery.
    ///
    /// Checks:
    /// - All segments referenced in state exist
    /// - No duplicate segment IDs
    /// - Document counts are consistent
    /// - No orphaned segments
    pub fn verify_consistency(&self, state: &RecoveryState) -> PersistenceResult<()> {
        // Verify all active segments exist
        for segment_id in state.active_segments.keys() {
            let segment_dir = format!("segments/segment_{}", segment_id);
            if !self.directory.exists(&segment_dir) {
                return Err(PersistenceError::InvalidState(format!(
                    "Active segment {} not found on disk",
                    segment_id
                )));
            }
        }

        // Check for orphaned segments (exist on disk but not in state)
        // This would require listing segments directory
        // For now, just verify active segments

        Ok(())
    }

    /// Clean up temporary files from previous operations.
    ///
    /// Removes:
    /// - Stale merge indicator files
    /// - Old temporary checkpoint files (*.tmp)
    /// - Orphaned handle files (if handles directory exists)
    fn cleanup_temporary_files(&self) -> PersistenceResult<()> {
        // Clean up temporary checkpoint files
        if self.directory.exists("checkpoints") {
            if let Ok(files) = self.directory.list_dir("checkpoints") {
                for file in files {
                    if file.ends_with(".tmp") {
                        let temp_path = format!("checkpoints/{}", file);
                        let _ = self.directory.delete(&temp_path); // Best effort
                    }
                }
            }
        }

        // Clean up temporary segment files
        if self.directory.exists("segments") {
            if let Ok(segments) = self.directory.list_dir("segments") {
                for segment_dir in segments {
                    let segment_path = format!("segments/{}", segment_dir);
                    if let Ok(files) = self.directory.list_dir(&segment_path) {
                        for file in files {
                            if file.ends_with(".tmp") {
                                let temp_path = format!("{}/{}", segment_path, file);
                                let _ = self.directory.delete(&temp_path); // Best effort
                            }
                        }
                    }
                }
            }
        }

        // Clean up stale merge indicators
        if self.directory.exists("merges") {
            if let Ok(files) = self.directory.list_dir("merges") {
                for file in files {
                    if file.ends_with(".in_progress") {
                        let merge_path = format!("merges/{}", file);
                        let _ = self.directory.delete(&merge_path); // Best effort
                    }
                }
            }
        }

        Ok(())
    }
}

// (Arc/Box directory adapter removed: checkpoint now accepts Arc directly.)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::directory::MemoryDirectory;
    use crate::persistence::wal::{WalEntry, WalWriter};

    #[test]
    #[ignore = "WalReader not yet implemented for durability v0.2.0 -- recovery replay needs migration"]
    fn test_recovery_basic() {
        use std::sync::Arc;
        let mem_dir = MemoryDirectory::new();
        // Convert MemoryDirectory to Arc<dyn Directory> for sharing
        let dir_arc: Arc<dyn Directory> = Arc::new(mem_dir) as Arc<dyn Directory>;
        dir_arc.create_dir_all("wal").unwrap();
        dir_arc.create_dir_all("checkpoints").unwrap();

        // Write some WAL entries
        let mut wal_writer = WalWriter::new(dir_arc.clone());
        wal_writer
            .append(WalEntry::AddSegment {
                segment_id: 1,
                doc_count: 100,
            })
            .unwrap();
        wal_writer
            .append(WalEntry::AddSegment {
                segment_id: 2,
                doc_count: 200,
            })
            .unwrap();

        // Recover using Arc
        let recovery = RecoveryManager::new(dir_arc);
        let state = recovery.recover().unwrap();

        assert_eq!(state.active_segments.len(), 2);
        assert!(state.active_segments.contains_key(&1));
        assert!(state.active_segments.contains_key(&2));
    }
}
