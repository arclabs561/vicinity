#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(feature = "persistence")]

use tempfile::tempdir;
use vicinity::persistence::locking::{FileLock, LockType};

#[test]
fn test_local_file_lock_invariant() {
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("local.lock");

    // Test that local locking still works
    let _lock1 = FileLock::acquire(&path, LockType::Exclusive).unwrap();

    // FileLock::acquire is non-blocking in our implementation (NB flag)
    // so a second acquisition should fail.
    let lock2 = FileLock::acquire(&path, LockType::Exclusive);
    assert!(
        lock2.is_err(),
        "Local exclusive lock should prevent second acquisition"
    );
}
