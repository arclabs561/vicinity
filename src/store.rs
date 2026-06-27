//! Updatable, durable multi-segment ANN index via `segstore` (the `store` feature).
//!
//! Enabled by the optional `store` feature. The base [`HNSWIndex`] is build-once;
//! this wraps the corpus in a segstore `SegmentedStore` so vectors can be added
//! and deleted incrementally with a write-ahead log + checkpoint + compaction,
//! and the index survives a restart.
//!
//! Design note: a segstore-backed ANN is *multi-segment* by construction, a
//! per-segment HNSW searched and merged. This is the deliberate alternative to a
//! single evolving graph (`fresh_graph` + FreshDiskANN consolidation): it trades
//! single-graph recall/latency at very large scale for native updatability and
//! durability, which fits a modest or frequently-churning corpus. The
//! cross-segment top-k merge is exact *given* exact per-segment top-k, but HNSW
//! search is itself approximate, so the merged result is approximate (as any HNSW
//! result is); the segmentation adds no exactness it didn't already have.
//!
//! Each per-segment HNSW is built over that segment's *live* vectors and
//! **cached**, rebuilt only when the index is mutated (an add that seals a
//! segment, a delete, or a compaction), not on every query. The small unflushed
//! buffer is built per query.
//!
//! Vectors are L2-normalized on ingest so the default cosine HNSW is well-formed.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

use durability::{Directory, PersistenceError, PersistenceResult};
use segstore::{SegmentedStore, Store};

use crate::distance;
use crate::hnsw::HNSWIndex;

/// segstore payload: items are dense vectors, a segment is a batch of source
/// vectors (a per-segment HNSW is built + cached from the live ones).
struct VectorBacking;

impl Store for VectorBacking {
    type Id = u32;
    type Item = Vec<f32>;
    type Segment = Vec<(u32, Vec<f32>)>;

    fn build_segment(&self, batch: &[(u32, Vec<f32>)]) -> Vec<(u32, Vec<f32>)> {
        batch.to_vec()
    }

    fn merge_segments(
        &self,
        segs: &[Vec<(u32, Vec<f32>)>],
        live: &dyn Fn(&u32) -> bool,
    ) -> Vec<(u32, Vec<f32>)> {
        segs.iter()
            .flatten()
            .filter(|(id, _)| live(id))
            .cloned()
            .collect()
    }

    fn segment_len(&self, seg: &Vec<(u32, Vec<f32>)>) -> usize {
        seg.len()
    }

    fn live_len(&self, seg: &Vec<(u32, Vec<f32>)>, live: &dyn Fn(&u32) -> bool) -> Option<usize> {
        Some(seg.iter().filter(|(id, _)| live(id)).count())
    }
}

/// Per-segment HNSW indexes keyed by the segment's stable `Arc` identity. Because
/// segstore keeps an unchanged segment's `Arc` across mutations, a sealed add only
/// builds the one new segment's HNSW (the rest are reused) instead of rebuilding
/// the whole corpus -- the dominant cost for an interactive add-then-search loop.
struct Cache {
    by_ptr: HashMap<usize, Option<HNSWIndex>>,
}

/// An updatable, durable multi-segment HNSW index.
pub struct UpdatableIndex {
    inner: SegmentedStore<VectorBacking>,
    dim: usize,
    m: usize,
    m_max: usize,
    cache: RefCell<Cache>,
}

impl UpdatableIndex {
    /// Open (or recover) an index under `dir` for `dim`-dimensional vectors, using
    /// HNSW parameters `m` / `m_max` per segment. Up to `flush_threshold` vectors
    /// are buffered before a new immutable segment is sealed.
    pub fn open(
        dir: Arc<dyn Directory>,
        flush_threshold: usize,
        dim: usize,
        m: usize,
        m_max: usize,
    ) -> PersistenceResult<Self> {
        Ok(Self {
            inner: SegmentedStore::open(dir, VectorBacking, flush_threshold)?,
            dim,
            m,
            m_max,
            cache: RefCell::new(Cache {
                by_ptr: HashMap::new(),
            }),
        })
    }

    /// Add (or re-add) a vector by id. The vector is L2-normalized on ingest.
    /// Returns an error if its dimension does not match the index, rather than
    /// silently dropping it from every per-segment rebuild.
    pub fn add(&mut self, id: u32, vector: &[f32]) -> PersistenceResult<()> {
        if vector.len() != self.dim {
            return Err(PersistenceError::InvalidConfig(format!(
                "vector dimension {} does not match index dimension {}",
                vector.len(),
                self.dim
            )));
        }
        // A sealed add introduces a new segment (a new Arc identity); existing
        // segments keep theirs, so the cache reuses them and builds only the new one.
        self.inner.add(id, distance::normalize(vector))?;
        Ok(())
    }

    /// Tombstone a vector.
    pub fn delete(&mut self, id: u32) -> PersistenceResult<()> {
        self.inner.delete(id)?;
        // A tombstone changes the live filter used to build every segment's HNSW,
        // so invalidate the cache (deletes are far rarer than adds).
        self.cache.borrow_mut().by_ptr.clear();
        Ok(())
    }

    /// Merge segments (dropping tombstoned vectors) and persist a checkpoint.
    pub fn compact(&mut self) -> PersistenceResult<()> {
        self.inner.compact()?;
        Ok(())
    }

    /// Persist a checkpoint without merging.
    pub fn checkpoint(&mut self) -> PersistenceResult<()> {
        self.inner.checkpoint()
    }

    /// Merge only the segments whose live ratio is below `min_live_ratio`,
    /// reclaiming tombstoned vectors -- the cheap alternative to a full
    /// [`compact`](Self::compact) when a few segments are delete-heavy.
    pub fn reclaim(&mut self, min_live_ratio: f64) -> PersistenceResult<()> {
        self.inner.reclaim_tombstones(min_live_ratio)?;
        Ok(())
    }

    /// Storage amplification: stored vectors divided by live vectors (`1.0` with
    /// no tombstones, higher as deletes accumulate).
    pub fn space_amplification(&self) -> Option<f64> {
        self.inner.space_amplification()
    }

    /// The `k` nearest neighbors of `query` (by cosine distance) over the live
    /// corpus, searched per segment with the given `ef` and merged.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        let q = distance::normalize(query);
        let mut cand: Vec<(u32, f32)> = Vec::new();
        {
            let segs = self.inner.segments();
            let mut cache = self.cache.borrow_mut();
            // Drop cached indexes for segments no longer present (post-compaction).
            let current: std::collections::HashSet<usize> =
                segs.iter().map(|a| Arc::as_ptr(a) as usize).collect();
            cache.by_ptr.retain(|key, _| current.contains(key));
            // Build only segments not already cached (i.e. new ones).
            for seg in segs {
                let key = Arc::as_ptr(seg) as usize;
                cache
                    .by_ptr
                    .entry(key)
                    .or_insert_with(|| self.build_live_index(&seg[..]));
            }
            for idx in cache.by_ptr.values().flatten() {
                cand.extend(idx.search(&q, k, ef).unwrap_or_default());
            }
        }
        // The small unflushed buffer is built per query.
        let buffered = self.inner.buffer().to_vec();
        if let Some(idx) = self.build_live_index(&buffered) {
            cand.extend(idx.search(&q, k, ef).unwrap_or_default());
        }
        // Lower cosine distance is nearer.
        cand.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        cand.truncate(k);
        cand
    }

    /// Build a per-segment HNSW over the live vectors of `batch` (None if empty
    /// or the build fails). Vectors are stored already-normalized.
    fn build_live_index(&self, batch: &[(u32, Vec<f32>)]) -> Option<HNSWIndex> {
        let mut idx = match HNSWIndex::new(self.dim, self.m, self.m_max) {
            Ok(i) => i,
            Err(_) => return None,
        };
        let mut any = false;
        for (id, v) in batch {
            if self.inner.is_live(id) && idx.add(*id, v.clone()).is_ok() {
                any = true;
            }
        }
        if !any || idx.build().is_err() {
            return None;
        }
        Some(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use durability::MemoryDirectory;

    #[test]
    fn add_delete_compact_recover_through_real_hnsw() {
        let dir = MemoryDirectory::arc();
        {
            let mut store = UpdatableIndex::open(dir.clone(), 2, 2, 16, 32).unwrap();
            store.add(0, &[1.0, 0.0]).unwrap();
            store.add(1, &[0.0, 1.0]).unwrap(); // flush
            store.add(2, &[0.7, 0.7]).unwrap(); // buffered

            // Query near the x-axis: doc 0 is nearest, then doc 2.
            let top: Vec<u32> = store
                .search(&[0.9, 0.1], 2, 16)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            assert_eq!(top.first(), Some(&0), "nearest to the x-axis is doc 0");
            // Second query (no mutation) must use the cache and stay correct.
            let again: Vec<u32> = store
                .search(&[0.9, 0.1], 2, 16)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            assert_eq!(again.first(), Some(&0), "cached query is stable");

            store.delete(0).unwrap();
            let top: Vec<u32> = store
                .search(&[0.9, 0.1], 1, 16)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            assert_eq!(top, vec![2], "after deleting 0, nearest is doc 2");

            store.compact().unwrap();
            assert_eq!(
                store.search(&[0.9, 0.1], 1, 16).first().map(|(id, _)| *id),
                Some(2)
            );
        }
        let store = UpdatableIndex::open(dir, 2, 2, 16, 32).unwrap();
        let top: Vec<u32> = store
            .search(&[0.9, 0.1], 1, 16)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(top, vec![2], "recovery preserves the search");
    }
}
