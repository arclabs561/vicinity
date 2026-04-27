#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(feature = "persistence")]

use std::sync::Arc;
use vicinity::persistence::directory::{Directory, MemoryDirectory};
use vicinity::persistence::wal::{WalEntry, WalWriter};

/// Verify that WAL entries survive a write+flush+drop cycle and can be replayed.
///
/// Previously `#[tokio::test] async fn`. The body uses only synchronous
/// APIs (`WalWriter::append`, `flush`, `WalReader::replay`); de-asyncing
/// drops the `tokio` dev-dep.
#[test]
fn wal_roundtrip_replays_written_entries() -> anyhow::Result<()> {
    let dir: Arc<dyn Directory> = Arc::new(MemoryDirectory::new());
    let mut writer = WalWriter::new(dir.clone());

    // Write an entry and flush (simulates data that hit disk before crash).
    writer.append(WalEntry::StartMerge {
        transaction_id: 100,
        segment_ids: vec![1, 2],
    })?;
    writer.flush()?;
    drop(writer);

    // Replay WAL to recover state
    let reader = vicinity::persistence::wal::WalReader::new(dir.clone());
    let entries = reader.replay()?;

    assert_eq!(entries.len(), 1, "expected 1 WAL entry after replay");
    match &entries[0].payload {
        WalEntry::StartMerge {
            transaction_id,
            segment_ids,
        } => {
            assert_eq!(*transaction_id, 100);
            assert_eq!(segment_ids, &[1, 2]);
        }
        other => panic!("Expected StartMerge entry, got {:?}", other),
    }

    Ok(())
}
