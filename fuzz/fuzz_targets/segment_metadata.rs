//! Fuzz the HNSW segment metadata reader.
//!
//! The reader is the crate's only parser of bytes it did not write, and the
//! segment can arrive from a crashed writer, a truncated copy, or a hostile
//! store. Every byte after the magic is attacker-controlled, so the contract is
//! "returns Err, never panics" — a panic in a store process is an outage.
//!
//! Run: `cargo +nightly fuzz run segment_metadata -- -max_total_time=60`

#![no_main]

use libfuzzer_sys::fuzz_target;
use vicinity::persistence::directory::{Directory, MemoryDirectory};
use vicinity::persistence::hnsw::HNSWSegmentReader;

fuzz_target!(|data: &[u8]| {
    let directory = MemoryDirectory::new();
    if directory
        .atomic_write("segments/segment_hnsw_0/metadata.bin", data)
        .is_err()
    {
        return;
    }
    // Either a loaded reader or a typed error; the harness only fails on a panic.
    let _ = HNSWSegmentReader::load(Box::new(directory), 0);
});
