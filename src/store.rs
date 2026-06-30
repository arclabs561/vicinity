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
use std::collections::{HashMap, HashSet};
use std::io::Read;
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
        segs: &[&Vec<(u32, Vec<f32>)>],
        live: &dyn Fn(&u32) -> bool,
    ) -> Vec<(u32, Vec<f32>)> {
        segs.iter()
            .flat_map(|s| s.iter())
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

/// The `kind` tag for a persisted per-segment HNSW sidecar; segstore reserves the
/// file `segstore.idx.<seg_id>.hnsw` and garbage-collects it with its segment.
const INDEX_KIND: &str = "hnsw";

/// An updatable, durable multi-segment HNSW index.
pub struct UpdatableIndex {
    inner: SegmentedStore<VectorBacking>,
    dim: usize,
    m: usize,
    m_max: usize,
    cache: RefCell<Cache>,
    /// Segment ids whose on-disk HNSW sidecar is current, so a checkpoint this
    /// process re-persists only new segments. Pre-populated on open from the
    /// sidecars already on disk; the per-segment staleness guard still
    /// re-validates each before it is trusted.
    persisted: RefCell<HashSet<u64>>,
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
        let inner = SegmentedStore::open(dir, VectorBacking, flush_threshold)?;
        // A sidecar already on disk means that segment's HNSW need not be rebuilt:
        // record those ids so a checkpoint this process re-persists only genuinely
        // new segments. `load_sidecar` re-validates each one before it is trusted.
        let mut persisted = HashSet::new();
        for &id in inner.segment_ids() {
            if inner.dir().exists(&inner.index_name(id, INDEX_KIND)) {
                persisted.insert(id);
            }
        }
        Ok(Self {
            inner,
            dim,
            m,
            m_max,
            cache: RefCell::new(Cache {
                by_ptr: HashMap::new(),
            }),
            persisted: RefCell::new(persisted),
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

    /// Add (or re-add) many vectors, syncing the write-ahead log once for the whole
    /// batch instead of once per vector. Each vector is dimension-checked and
    /// L2-normalized before any is ingested (mirrors [`Self::add`]). This is the
    /// bulk-ingest path (the corpus-load phase): per-item WAL sync is the dominant
    /// cost on a real disk, so one sync per batch is several times faster than a
    /// loop of [`Self::add`].
    pub fn extend(
        &mut self,
        vectors: impl IntoIterator<Item = (u32, Vec<f32>)>,
    ) -> PersistenceResult<()> {
        let dim = self.dim;
        let normalized: Result<Vec<(u32, Vec<f32>)>, PersistenceError> = vectors
            .into_iter()
            .map(|(id, vector)| {
                if vector.len() != dim {
                    Err(PersistenceError::InvalidConfig(format!(
                        "vector dimension {} does not match index dimension {}",
                        vector.len(),
                        dim
                    )))
                } else {
                    Ok((id, distance::normalize(&vector)))
                }
            })
            .collect();
        self.inner.extend(normalized?)?;
        Ok(())
    }

    /// Tombstone a vector.
    pub fn delete(&mut self, id: u32) -> PersistenceResult<()> {
        self.inner.delete(id)?;
        // A tombstone only changes the live-set of the segment that holds `id`, so
        // invalidate just that segment's cached HNSW -- not the whole cache -- and
        // drop its now-stale sidecar so the next build re-persists over the live
        // set. (The `load_sidecar` guard would reject a stale sidecar anyway; this
        // just avoids a wasted load + rebuild on the next search.)
        let ids = self.inner.segment_ids();
        let mut cache = self.cache.borrow_mut();
        for (i, seg) in self.inner.segments().iter().enumerate() {
            if seg.iter().any(|(sid, _)| *sid == id) {
                cache.by_ptr.remove(&(Arc::as_ptr(seg) as usize));
                let seg_id = ids[i];
                self.persisted.borrow_mut().remove(&seg_id);
                let _ = self
                    .inner
                    .dir()
                    .delete(&self.inner.index_name(seg_id, INDEX_KIND));
            }
        }
        Ok(())
    }

    /// Merge segments (dropping tombstoned vectors) and persist a checkpoint.
    pub fn compact(&mut self) -> PersistenceResult<()> {
        self.inner.compact()?;
        Ok(())
    }

    /// Persist a checkpoint without merging, then persist a per-segment HNSW
    /// sidecar for every sealed segment that lacks a current one, so a restart
    /// loads each graph instead of rebuilding it. Incremental: only segments new
    /// since the last checkpoint are built (O(new), not O(corpus)).
    pub fn checkpoint(&mut self) -> PersistenceResult<()> {
        self.inner.checkpoint()?;
        self.persist_new_segments();
        Ok(())
    }

    /// Run one round of size-tiered compaction, merging similarly-sized segments
    /// so the segment count stays bounded without a full [`compact`](Self::compact).
    pub fn compact_tiers(&mut self) -> PersistenceResult<()> {
        self.inner.compact_tiers()?;
        Ok(())
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
            let ids = self.inner.segment_ids();
            let mut cache = self.cache.borrow_mut();
            // Drop cached indexes for segments no longer present (post-compaction).
            let current: HashSet<usize> = segs.iter().map(|a| Arc::as_ptr(a) as usize).collect();
            cache.by_ptr.retain(|key, _| current.contains(key));
            // For each segment not already cached, load its persisted HNSW sidecar
            // (the restart win) or build it over the live vectors and persist it for
            // next time. `segment_ids()[i]` is the stable id of `segments()[i]`.
            for (i, seg) in segs.iter().enumerate() {
                let key = Arc::as_ptr(seg) as usize;
                let seg_id = ids[i];
                cache
                    .by_ptr
                    .entry(key)
                    .or_insert_with(|| self.build_or_load(&seg[..], seg_id));
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

    /// Load segment `seg_id`'s persisted HNSW from its sidecar, or build it over
    /// the segment's live vectors and persist it (write-through) for next time.
    fn build_or_load(&self, seg: &[(u32, Vec<f32>)], seg_id: u64) -> Option<HNSWIndex> {
        if let Some(idx) = self.load_sidecar(seg, seg_id) {
            self.persisted.borrow_mut().insert(seg_id);
            return Some(idx);
        }
        let idx = self.build_live_index(seg)?;
        self.persist_sidecar(&idx, seg_id);
        Some(idx)
    }

    /// Load segment `seg_id`'s HNSW from its sidecar if one exists and is still
    /// valid for the segment's *current* live set. Returns None (forcing a
    /// rebuild) when the sidecar is absent, unreadable, or stale -- e.g. a crash
    /// left a sidecar written before a delete that has since tombstoned one of its
    /// ids. The guard compares the persisted ids against the segment's live ids,
    /// so a stale sidecar can never serve a deleted vector.
    fn load_sidecar(&self, seg: &[(u32, Vec<f32>)], seg_id: u64) -> Option<HNSWIndex> {
        let name = self.inner.index_name(seg_id, INDEX_KIND);
        if !self.inner.dir().exists(&name) {
            return None;
        }
        let mut bytes = Vec::new();
        self.inner
            .dir()
            .open_file(&name)
            .ok()?
            .read_to_end(&mut bytes)
            .ok()?;
        let idx = HNSWIndex::from_postcard(&bytes).ok()?;
        let live: HashSet<u32> = seg
            .iter()
            .map(|(id, _)| *id)
            .filter(|id| self.inner.is_live(id))
            .collect();
        let stored: HashSet<u32> = idx.doc_ids.iter().copied().collect();
        if stored == live {
            Some(idx)
        } else {
            None
        }
    }

    /// Persist a built per-segment HNSW as its sidecar (best-effort: a failed
    /// write leaves the in-memory index usable and simply re-persists next time).
    fn persist_sidecar(&self, idx: &HNSWIndex, seg_id: u64) {
        if let Ok(bytes) = idx.to_postcard() {
            if self
                .inner
                .dir()
                .atomic_write(&self.inner.index_name(seg_id, INDEX_KIND), &bytes)
                .is_ok()
            {
                self.persisted.borrow_mut().insert(seg_id);
            }
        }
    }

    /// Build + persist a sidecar for every sealed segment without a current one,
    /// first pruning ids for segments that compaction has removed. Incremental: a
    /// segment already persisted this process (or recovered with its sidecar on
    /// disk) is skipped, so this is O(new segments), not O(corpus).
    fn persist_new_segments(&self) {
        let ids = self.inner.segment_ids();
        let id_set: HashSet<u64> = ids.iter().copied().collect();
        self.persisted.borrow_mut().retain(|id| id_set.contains(id));
        for (i, seg) in self.inner.segments().iter().enumerate() {
            let seg_id = ids[i];
            if self.persisted.borrow().contains(&seg_id) {
                continue;
            }
            if let Some(idx) = self.build_live_index(&seg[..]) {
                self.persist_sidecar(&idx, seg_id);
            }
        }
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

    #[test]
    fn checkpoint_persists_sidecars_and_reopen_loads_them() {
        let dir = MemoryDirectory::arc();
        {
            let mut store = UpdatableIndex::open(dir.clone(), 4, 3, 16, 32).unwrap();
            for i in 0..12u32 {
                let a = i as f32;
                store.add(i, &[a.cos(), a.sin(), 1.0]).unwrap();
            }
            store.checkpoint().unwrap();
            // Every sealed segment carries a persisted HNSW sidecar on disk.
            let ids: Vec<u64> = store.inner.segment_ids().to_vec();
            assert!(
                !ids.is_empty(),
                "12 adds at flush 4 seal at least one segment"
            );
            for id in &ids {
                assert!(
                    store
                        .inner
                        .dir()
                        .exists(&store.inner.index_name(*id, INDEX_KIND)),
                    "segment {id} must have a persisted sidecar after checkpoint"
                );
            }
        }
        // Reopen: `open` recovers the persisted set from the sidecars and `search`
        // loads each graph instead of rebuilding it. Results must stay correct.
        let store = UpdatableIndex::open(dir, 4, 3, 16, 32).unwrap();
        assert!(
            !store.search(&[1.0, 0.0, 1.0], 1, 16).is_empty(),
            "search over loaded sidecars returns results"
        );
    }

    #[test]
    fn deleted_id_does_not_resurface_through_a_sidecar() {
        let dir = MemoryDirectory::arc();
        {
            let mut store = UpdatableIndex::open(dir.clone(), 2, 2, 16, 32).unwrap();
            store.add(0, &[1.0, 0.0]).unwrap();
            store.add(1, &[0.95, 0.05]).unwrap();
            store.add(2, &[0.0, 1.0]).unwrap();
            store.checkpoint().unwrap(); // sidecars written, including id 0's segment
            store.delete(0).unwrap(); // drops that sidecar + clears it from the persisted set
            store.checkpoint().unwrap(); // re-persists that segment over the live set (no id 0)
        }
        // Reopen and query id 0's old location: a stale sidecar must not revive it.
        let store = UpdatableIndex::open(dir, 2, 2, 16, 32).unwrap();
        let top: Vec<u32> = store
            .search(&[1.0, 0.0], 3, 16)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert!(
            !top.contains(&0),
            "deleted id 0 must not resurface from a persisted sidecar"
        );
        assert!(
            top.contains(&1),
            "nearest live vector to the x-axis is id 1"
        );
    }
}
