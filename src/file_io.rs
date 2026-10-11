//! Positional file reads and the block cache used by file-backed searchers.
//!
//! File-backed searchers read fixed records (graph adjacency, vectors, page
//! records, IVF lists) at arbitrary offsets. One `pread` per record costs a
//! syscall each time, so [`CachedFile`] keeps recently read fixed-size blocks
//! in a bounded user-space cache with CLOCK (second-chance) eviction. Every
//! searcher owns its cache and searches through `&mut self`, so the cache
//! needs no locking.

use std::collections::HashMap;
use std::fs::File;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{Error, ErrorKind};

/// Default byte budget of a file-backed searcher's block cache: 64 MiB.
///
/// Large enough to hold small and medium indexes entirely, small enough that
/// one searcher per thread stays bounded. Set [`FileCacheConfig::budget_bytes`]
/// to the index's hot set (or its size) when memory allows.
pub const DEFAULT_FILE_CACHE_BYTES: usize = 64 << 20;

/// Default cache block size: 4 KiB, the DiskANN page-layout record alignment.
pub const DEFAULT_FILE_CACHE_BLOCK_SIZE: usize = 4096;

/// Block cache settings for file-backed searchers.
///
/// `budget_bytes` bounds the cached block bytes for one searcher; a searcher
/// that reads several files splits it across them in proportion to file size.
/// A budget smaller than one block disables the cache, and every read goes
/// straight to the file.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileCacheConfig {
    /// Maximum bytes of cached blocks. `0` disables the cache.
    pub budget_bytes: usize,
    /// Block size in bytes, rounded up to a power of two (minimum 512).
    pub block_size: usize,
}

impl Default for FileCacheConfig {
    fn default() -> Self {
        Self {
            budget_bytes: DEFAULT_FILE_CACHE_BYTES,
            block_size: DEFAULT_FILE_CACHE_BLOCK_SIZE,
        }
    }
}

impl FileCacheConfig {
    /// Default block size with the given byte budget.
    pub fn with_budget(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            ..Self::default()
        }
    }

    /// No caching: every read is a positional read.
    pub fn disabled() -> Self {
        Self::with_budget(0)
    }

    /// Split this budget across files in proportion to their sizes.
    pub(crate) fn split(self, file_lens: &[u64]) -> Vec<Self> {
        let total: u128 = file_lens.iter().map(|&len| len as u128).sum();
        file_lens
            .iter()
            .map(|&len| {
                let share = (self.budget_bytes as u128 * len as u128)
                    .checked_div(total)
                    .unwrap_or(0) as usize;
                Self {
                    budget_bytes: share,
                    ..self
                }
            })
            .collect()
    }
}

/// Block cache counters, summed over a searcher's files.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FileCacheStats {
    /// Block lookups served from the cache.
    pub hits: u64,
    /// Block lookups that read the file.
    pub misses: u64,
    /// Bytes of cached blocks currently held.
    pub resident_bytes: usize,
    /// Configured budget in bytes.
    pub budget_bytes: usize,
}

impl FileCacheStats {
    /// Fraction of block lookups served from the cache (0 when none happened).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    pub(crate) fn merge(self, other: Self) -> Self {
        Self {
            hits: self.hits + other.hits,
            misses: self.misses + other.misses,
            resident_bytes: self.resident_bytes + other.resident_bytes,
            budget_bytes: self.budget_bytes + other.budget_bytes,
        }
    }
}

/// Block numbers are dense integers; a multiplicative hash spreads them
/// without SipHash's per-lookup cost.
#[derive(Default)]
struct BlockHasher(u64);

impl Hasher for BlockHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0 ^ byte as u64).wrapping_mul(0x0100_0000_01b3);
        }
    }

    fn write_u64(&mut self, n: u64) {
        let h = n.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        self.0 = h ^ (h >> 32);
    }
}

const NO_BLOCK: u64 = u64::MAX;

/// A file with a bounded CLOCK cache of fixed-size blocks.
pub(crate) struct CachedFile {
    file: File,
    len: u64,
    block_size: usize,
    shift: u32,
    /// Maximum resident blocks; 0 means passthrough.
    capacity: usize,
    budget_bytes: usize,
    slots: HashMap<u64, usize, BuildHasherDefault<BlockHasher>>,
    slot_block: Vec<u64>,
    referenced: Vec<bool>,
    data: Vec<u8>,
    hand: usize,
    hits: u64,
    misses: u64,
}

impl CachedFile {
    pub(crate) fn new(file: File, config: FileCacheConfig) -> std::io::Result<Self> {
        let len = file.metadata()?.len();
        let block_size = config.block_size.max(512).next_power_of_two();
        let file_blocks = len.div_ceil(block_size as u64);
        let capacity = (config.budget_bytes / block_size).min(file_blocks as usize);
        Ok(Self {
            file,
            len,
            block_size,
            shift: block_size.trailing_zeros(),
            capacity,
            budget_bytes: config.budget_bytes,
            slots: HashMap::default(),
            slot_block: Vec::new(),
            referenced: Vec::new(),
            data: Vec::new(),
            hand: 0,
            hits: 0,
            misses: 0,
        })
    }

    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    pub(crate) fn stats(&self) -> FileCacheStats {
        FileCacheStats {
            hits: self.hits,
            misses: self.misses,
            resident_bytes: self.data.len(),
            budget_bytes: self.budget_bytes,
        }
    }

    /// Fill `buf` with the bytes at `offset`, failing with `UnexpectedEof`
    /// if the range extends past the end of the file.
    pub(crate) fn read_exact_at(&mut self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        // A read larger than half the cache would evict most of it; read it
        // directly instead. Empty reads succeed at any offset, as they do
        // uncached.
        if self.capacity == 0 || buf.is_empty() || buf.len() > self.capacity * self.block_size / 2 {
            return read_exact_at_impl(&self.file, offset, buf);
        }
        let end = offset
            .checked_add(buf.len() as u64)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "read offset overflow"))?;
        if end > self.len {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                "failed to fill buffer",
            ));
        }

        let mask = self.block_size as u64 - 1;
        let mut pos = offset;
        let mut filled = 0;
        while filled < buf.len() {
            let slot = self.slot_for(pos >> self.shift)?;
            let within = (pos & mask) as usize;
            let n = (self.block_size - within).min(buf.len() - filled);
            let start = slot * self.block_size + within;
            buf[filled..filled + n].copy_from_slice(&self.data[start..start + n]);
            filled += n;
            pos += n as u64;
        }
        Ok(())
    }

    fn slot_for(&mut self, block: u64) -> std::io::Result<usize> {
        if let Some(&slot) = self.slots.get(&block) {
            self.referenced[slot] = true;
            self.hits += 1;
            return Ok(slot);
        }
        self.misses += 1;

        let slot = if self.slot_block.len() < self.capacity {
            self.slot_block.push(NO_BLOCK);
            self.referenced.push(false);
            self.data.resize(self.data.len() + self.block_size, 0);
            self.slot_block.len() - 1
        } else {
            // CLOCK: skip (and clear) recently referenced slots.
            while self.referenced[self.hand] {
                self.referenced[self.hand] = false;
                self.hand = (self.hand + 1) % self.capacity;
            }
            let victim = self.hand;
            self.hand = (self.hand + 1) % self.capacity;
            self.slots.remove(&self.slot_block[victim]);
            self.slot_block[victim] = NO_BLOCK;
            victim
        };

        let block_start = block << self.shift;
        let n = (self.len - block_start).min(self.block_size as u64) as usize;
        let start = slot * self.block_size;
        read_exact_at_impl(&self.file, block_start, &mut self.data[start..start + n])?;
        self.slot_block[slot] = block;
        self.slots.insert(block, slot);
        Ok(slot)
    }
}

#[cfg(unix)]
fn read_exact_at_impl(file: &File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;

    let mut read = 0;
    while read < buf.len() {
        let n = file.read_at(&mut buf[read..], offset + read as u64)?;
        if n == 0 {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                "failed to fill buffer",
            ));
        }
        read += n;
    }
    Ok(())
}

#[cfg(windows)]
fn read_exact_at_impl(file: &File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;

    let mut read = 0;
    while read < buf.len() {
        let n = file.seek_read(&mut buf[read..], offset + read as u64)?;
        if n == 0 {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                "failed to fill buffer",
            ));
        }
        read += n;
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn read_exact_at_impl(mut file: &File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};

    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::io::Write;

    fn file_with(bytes: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.bin");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(bytes)
            .unwrap();
        (dir, path)
    }

    fn pattern(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 31 + i / 7) as u8).collect()
    }

    fn cached(path: &std::path::Path, budget: usize, block: usize) -> CachedFile {
        let config = FileCacheConfig {
            budget_bytes: budget,
            block_size: block,
        };
        CachedFile::new(File::open(path).unwrap(), config).unwrap()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Cached reads return the file's bytes for any in-range offset and
        /// length (block-crossing and tail ranges included), and fail like a
        /// direct read past the end. The budget is small so eviction happens.
        #[test]
        fn cached_reads_match_direct_reads(
            file_len in 1usize..20_000,
            budget_blocks in 1usize..8,
            reads in proptest::collection::vec((0u64..21_000, 0usize..3_000), 1..64),
        ) {
            let bytes = pattern(file_len);
            let (_dir, path) = file_with(&bytes);
            let direct = File::open(&path).unwrap();
            let mut cache = cached(&path, budget_blocks * 512, 512);
            for (offset, len) in reads {
                let mut got = vec![0u8; len];
                let mut want = vec![0u8; len];
                let got_res = cache.read_exact_at(offset, &mut got);
                let want_res = read_exact_at_impl(&direct, offset, &mut want);
                prop_assert_eq!(got_res.is_ok(), want_res.is_ok(), "offset {} len {}", offset, len);
                if want_res.is_ok() {
                    prop_assert_eq!(&got, &want);
                }
                prop_assert!(cache.stats().resident_bytes <= budget_blocks * 512);
            }
        }
    }

    #[test]
    fn read_crossing_block_boundary_and_file_tail() {
        let bytes = pattern(10_000);
        let (_dir, path) = file_with(&bytes);
        let mut cache = cached(&path, 64 * 1024, 4096);
        let mut buf = vec![0u8; 200];
        cache.read_exact_at(4000, &mut buf).unwrap();
        assert_eq!(buf, bytes[4000..4200]);
        let mut tail = vec![0u8; 100];
        cache.read_exact_at(9_900, &mut tail).unwrap();
        assert_eq!(tail, bytes[9_900..]);
        let err = cache.read_exact_at(9_950, &mut tail).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::UnexpectedEof);
    }

    #[test]
    fn resident_bytes_stay_within_budget_and_blocks_are_evicted() {
        let bytes = pattern(64 * 4096);
        let (_dir, path) = file_with(&bytes);
        let budget = 8 * 4096;
        let mut cache = cached(&path, budget, 4096);
        let mut buf = [0u8; 16];
        for round in 0..4u64 {
            for block in 0..64u64 {
                cache.read_exact_at(block * 4096 + round, &mut buf).unwrap();
                assert!(cache.stats().resident_bytes <= budget);
            }
        }
        let stats = cache.stats();
        assert_eq!(stats.resident_bytes, budget);
        // A cyclic scan over 64 blocks with room for 8 must keep missing:
        // every block is evicted before it is read again.
        assert_eq!(stats.misses, 4 * 64);

        // A hot block re-read immediately is a hit.
        cache.read_exact_at(5, &mut buf).unwrap();
        cache.read_exact_at(9, &mut buf).unwrap();
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn zero_budget_is_passthrough() {
        let bytes = pattern(9_000);
        let (_dir, path) = file_with(&bytes);
        let mut cache = cached(&path, 0, 4096);
        let mut buf = vec![0u8; 300];
        for offset in [0u64, 4000, 8_700] {
            cache.read_exact_at(offset, &mut buf).unwrap();
            assert_eq!(buf, bytes[offset as usize..offset as usize + 300]);
        }
        let stats = cache.stats();
        assert_eq!((stats.hits, stats.misses, stats.resident_bytes), (0, 0, 0));
    }

    #[test]
    fn budget_splits_in_proportion_to_file_size() {
        let parts = FileCacheConfig::with_budget(1000).split(&[300, 100, 0]);
        let budgets: Vec<usize> = parts.iter().map(|c| c.budget_bytes).collect();
        assert_eq!(budgets, vec![750, 250, 0]);
    }
}
