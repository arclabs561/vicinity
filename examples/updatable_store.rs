//! Durable, updatable multi-segment HNSW search.
//!
//! Run:
//! `cargo run --features store --example updatable_store`

use durability::MemoryDirectory;
use vicinity::store::UpdatableIndex;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = MemoryDirectory::arc();
    let mut index = UpdatableIndex::open(dir.clone(), 2, 2, 16, 32)?;

    index.add(0, &[1.0, 0.0])?;
    index.add(1, &[0.0, 1.0])?;
    index.add(2, &[0.7, 0.7])?;
    index.checkpoint()?;

    let query = [0.9, 0.1];
    let before_delete = index.search(&query, 3, 16);
    assert_eq!(before_delete.first().map(|(id, _)| *id), Some(0));

    index.delete(0)?;
    index.checkpoint()?;
    drop(index);

    let recovered = UpdatableIndex::open(dir, 2, 2, 16, 32)?;
    let after_reopen = recovered.search(&query, 3, 16);
    assert_eq!(after_reopen.first().map(|(id, _)| *id), Some(2));
    assert!(!after_reopen.iter().any(|(id, _)| *id == 0));

    println!("before delete:");
    for (id, distance) in before_delete {
        println!("  doc {id}: distance={distance:.4}");
    }
    println!("after reopen:");
    for (id, distance) in after_reopen {
        println!("  doc {id}: distance={distance:.4}");
    }

    Ok(())
}
