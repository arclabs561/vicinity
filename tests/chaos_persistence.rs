#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(feature = "persistence")]

use std::sync::Arc;
use vicinity::persistence::directory::{Directory, MemoryDirectory};
use vicinity::persistence::wal::{WalEntry, WalWriter};

#[tokio::test]
#[ignore = "WalReader not yet implemented for durability v0.2.0"]
async fn test_crash_recovery_invariant() -> anyhow::Result<()> {
    let dir: Arc<dyn Directory> = Arc::new(MemoryDirectory::new());
    let mut writer = WalWriter::new(dir.clone());

    // 1. Record an incomplete operation in WAL
    writer.append(WalEntry::StartMerge {
        transaction_id: 100,
        segment_ids: vec![1, 2],
    })?;

    // Simulate "Crash" - writer is dropped, data is still in MemoryDirectory
    drop(writer);

    // 2. Replay WAL to recover state
    let reader = vicinity::persistence::wal::WalReader::new(dir.clone());
    let entries = reader.replay()?;

    assert_eq!(entries.len(), 1);
    match &entries[0].payload {
        WalEntry::StartMerge { transaction_id, .. } => {
            assert_eq!(*transaction_id, 100);
        }
        _ => panic!("Expected StartMerge entry"),
    }

    println!("Crash recovery verified via WAL replay.");
    Ok(())
}

