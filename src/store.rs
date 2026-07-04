//! Updatable, durable multi-segment ANN index via `segstore` (the `store` feature).
//!
//! Enabled by the optional `store` feature. The base [`crate::hnsw::HNSWIndex`]
//! is build-once;
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
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::sync::Arc;

use durability::{Directory, PersistenceError, PersistenceResult};
use segstore::{SegmentCatalog, SegmentedStore, Store};

use crate::distance;
use crate::hnsw::{HNSWIndex, HNSWParams};

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

/// Per-segment HNSW indexes keyed by segstore's stable segment id. A sealed add
/// creates one new segment id, so cached HNSW indexes for existing segments are
/// reused instead of rebuilding the whole corpus on the next query.
struct Cache {
    by_segment_id: HashMap<u64, Option<HNSWIndex>>,
}

/// The `kind` tag for a persisted per-segment HNSW sidecar; segstore reserves the
/// file `segstore.idx.<seg_id>.hnsw` and garbage-collects it with its segment.
const INDEX_KIND: &str = "hnsw";
const SIDECAR_MAGIC: &[u8; 8] = b"VICHNSW1";
const SIDECAR_VERSION: u32 = 1;

/// An updatable, durable multi-segment HNSW index.
pub struct UpdatableIndex {
    inner: SegmentedStore<VectorBacking>,
    dim: usize,
    m: usize,
    m_max: usize,
    sidecar_recipe: String,
    cache: RefCell<Cache>,
    /// Segment ids whose on-disk HNSW sidecar was validated or written in this
    /// process, so a checkpoint re-persists only genuinely missing/stale segments.
    persisted: RefCell<HashSet<u64>>,
}

/// A read-only checkpoint view that loads per-segment HNSW sidecars before
/// falling back to source vector segment payloads.
///
/// This is the restart/query path for larger stores whose built HNSW graphs
/// have already been persisted by [`UpdatableIndex::checkpoint`]. It opens the
/// segstore manifest without decoding source segments, then loads graph
/// sidecars. A sidecar that contains a tombstoned id is rebuilt from that one
/// segment before search; HNSW is approximate, so filtering deleted hits after a
/// truncated graph search is not enough to preserve recall.
pub struct SnapshotIndex {
    catalog: SegmentCatalog<u32>,
    dim: usize,
    m: usize,
    m_max: usize,
    sidecar_recipe: String,
    cache: RefCell<HashMap<u64, Option<HNSWIndex>>>,
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
        Ok(Self {
            inner,
            dim,
            m,
            m_max,
            sidecar_recipe: Self::make_sidecar_recipe(dim, m, m_max),
            cache: RefCell::new(Cache {
                by_segment_id: HashMap::new(),
            }),
            persisted: RefCell::new(HashSet::new()),
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
        // A sealed add introduces a new segment id; existing segment ids stay
        // stable, so the cache reuses them and builds only the new one.
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
                let seg_id = ids[i];
                cache.by_segment_id.remove(&seg_id);
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
        self.prune_cache_to_current_segments();
        self.persist_new_segments();
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
        let stats = self.inner.compact_tiers()?;
        if stats.merges > 0 {
            self.prune_cache_to_current_segments();
            self.persist_new_segments();
        }
        Ok(())
    }

    /// Merge only the segments whose live ratio is below `min_live_ratio`,
    /// reclaiming tombstoned vectors -- the cheap alternative to a full
    /// [`compact`](Self::compact) when a few segments are delete-heavy.
    pub fn reclaim(&mut self, min_live_ratio: f64) -> PersistenceResult<()> {
        let stats = self.inner.reclaim_tombstones(min_live_ratio)?;
        if stats.merges > 0 {
            self.prune_cache_to_current_segments();
            self.persist_new_segments();
        }
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
            // For each segment not already cached, load its persisted HNSW sidecar
            // (the restart win) or build it over the live vectors and persist it for
            // next time. `segment_ids()[i]` is the stable id of `segments()[i]`.
            for (i, seg) in segs.iter().enumerate() {
                let seg_id = ids[i];
                let index = cache
                    .by_segment_id
                    .entry(seg_id)
                    .or_insert_with(|| self.build_or_load(&seg[..], seg_id));
                if let Some(idx) = index {
                    cand.extend(idx.search(&q, k, ef).unwrap_or_default());
                }
            }
        }
        // The small unflushed buffer is built per query.
        let buffered = self.inner.buffer();
        if let Some(idx) = self.build_live_index(buffered) {
            cand.extend(idx.search(&q, k, ef).unwrap_or_default());
        }
        // Lower cosine distance is nearer.
        cand.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        cand.truncate(k);
        cand
    }

    fn prune_cache_to_current_segments(&self) {
        let current: HashSet<u64> = self.inner.segment_ids().iter().copied().collect();
        self.cache
            .borrow_mut()
            .by_segment_id
            .retain(|id, _| current.contains(id));
    }

    /// Build a per-segment HNSW over the live vectors of `batch` (None if empty
    /// or the build fails). Vectors are stored already-normalized.
    fn build_live_index(&self, batch: &[(u32, Vec<f32>)]) -> Option<HNSWIndex> {
        Self::build_live_index_from(self.dim, self.m, self.m_max, batch, &|id| {
            self.inner.is_live(id)
        })
    }

    fn build_live_index_from(
        dim: usize,
        m: usize,
        m_max: usize,
        batch: &[(u32, Vec<f32>)],
        live: &dyn Fn(&u32) -> bool,
    ) -> Option<HNSWIndex> {
        let mut idx = match HNSWIndex::new(dim, m, m_max) {
            Ok(i) => i,
            Err(_) => return None,
        };
        let mut any = false;
        for (id, v) in batch {
            if live(id) && idx.add(*id, v.clone()).is_ok() {
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
        let graph_bytes = self.decode_sidecar(&bytes)?;
        let idx = HNSWIndex::from_postcard(graph_bytes).ok()?;
        let mut live = HashSet::with_capacity(seg.len());
        for (id, _) in seg {
            if self.inner.is_live(id) {
                live.insert(*id);
            }
        }
        if idx.doc_ids.len() == live.len() && idx.doc_ids.iter().all(|id| live.contains(id)) {
            Some(idx)
        } else {
            None
        }
    }

    /// Persist a built per-segment HNSW as its sidecar (best-effort: a failed
    /// write leaves the in-memory index usable and simply re-persists next time).
    fn persist_sidecar(&self, idx: &HNSWIndex, seg_id: u64) {
        if let Ok(graph) = idx.to_postcard() {
            let Some(bytes) = self.encode_sidecar(&graph) else {
                return;
            };
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

    fn make_sidecar_recipe(dim: usize, m: usize, m_max: usize) -> String {
        let params = HNSWParams {
            m,
            m_max,
            ..Default::default()
        };
        format!(
            "vicinity-store-hnsw-v1;\
             dim={};m={};m_max={};m_l={:.17};ef_construction={};\
             metric={:?};normalization=store-l2-on-ingest-and-query;\
             seed_selection={:?};diversification={:?};seed={:?};\
            codec=postcard-hnsw-v1;id_compression={}",
            dim,
            params.m,
            params.m_max,
            params.m_l,
            params.ef_construction,
            params.metric,
            params.seed_selection,
            params.neighborhood_diversification,
            params.seed,
            cfg!(feature = "id-compression")
        )
    }

    fn encode_sidecar(&self, graph: &[u8]) -> Option<Vec<u8>> {
        Self::encode_sidecar_for_recipe(&self.sidecar_recipe, graph)
    }

    fn encode_sidecar_for_recipe(sidecar_recipe: &str, graph: &[u8]) -> Option<Vec<u8>> {
        let recipe = sidecar_recipe.as_bytes();
        let recipe_len = u32::try_from(recipe.len()).ok()?;
        let mut bytes = Vec::with_capacity(16 + recipe.len() + graph.len());
        bytes.extend_from_slice(SIDECAR_MAGIC);
        bytes.extend_from_slice(&SIDECAR_VERSION.to_le_bytes());
        bytes.extend_from_slice(&recipe_len.to_le_bytes());
        bytes.extend_from_slice(recipe);
        bytes.extend_from_slice(graph);
        Some(bytes)
    }

    fn decode_sidecar<'a>(&self, bytes: &'a [u8]) -> Option<&'a [u8]> {
        Self::decode_sidecar_for_recipe(&self.sidecar_recipe, bytes)
    }

    fn decode_sidecar_for_recipe<'a>(sidecar_recipe: &str, bytes: &'a [u8]) -> Option<&'a [u8]> {
        if bytes.len() < 16 {
            return None;
        }
        if &bytes[..8] != SIDECAR_MAGIC {
            return None;
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
        if version != SIDECAR_VERSION {
            return None;
        }
        let recipe_len = u32::from_le_bytes(bytes[12..16].try_into().ok()?) as usize;
        let recipe_start = 16usize;
        let recipe_end = recipe_start.checked_add(recipe_len)?;
        if bytes.len() < recipe_end {
            return None;
        }
        if &bytes[recipe_start..recipe_end] != sidecar_recipe.as_bytes() {
            return None;
        }
        Some(&bytes[recipe_end..])
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
            if self.load_sidecar(&seg[..], seg_id).is_some() {
                self.persisted.borrow_mut().insert(seg_id);
                continue;
            }
            if let Some(idx) = self.build_live_index(&seg[..]) {
                self.persist_sidecar(&idx, seg_id);
            }
        }
    }
}

impl SnapshotIndex {
    /// Open the last checkpoint under `dir` as a read-only ANN snapshot.
    ///
    /// WAL records after the last checkpoint are intentionally not visible;
    /// checkpoint before opening a snapshot when newly added vectors must be
    /// searchable through this path.
    pub fn open(
        dir: Arc<dyn Directory>,
        dim: usize,
        m: usize,
        m_max: usize,
    ) -> PersistenceResult<Self> {
        Ok(Self {
            catalog: SegmentCatalog::open(dir)?,
            dim,
            m,
            m_max,
            sidecar_recipe: UpdatableIndex::make_sidecar_recipe(dim, m, m_max),
            cache: RefCell::new(HashMap::new()),
        })
    }

    /// Number of checkpointed immutable segments in this snapshot.
    pub fn segment_count(&self) -> usize {
        self.catalog.segment_count()
    }

    /// Number of tombstoned document ids in this snapshot.
    pub fn tombstone_count(&self) -> usize {
        self.catalog.tombstone_count()
    }

    /// The `k` nearest neighbors of `query` over the live checkpointed corpus.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> PersistenceResult<Vec<(u32, f32)>> {
        if k == 0 {
            return Ok(Vec::new());
        }

        let q = distance::normalize(query);
        let mut cand = Vec::new();
        {
            let mut cache = self.cache.borrow_mut();
            let current: HashSet<u64> = self.catalog.segment_ids().iter().copied().collect();
            cache.retain(|seg_id, _| current.contains(seg_id));

            for &seg_id in self.catalog.segment_ids() {
                if let Entry::Vacant(entry) = cache.entry(seg_id) {
                    let index = self.build_or_load(seg_id)?;
                    entry.insert(index);
                }
                if let Some(Some(idx)) = cache.get(&seg_id) {
                    cand.extend(
                        idx.search(&q, k, ef)
                            .unwrap_or_default()
                            .into_iter()
                            .filter(|(id, _)| self.catalog.is_live(id)),
                    );
                }
            }
        }

        cand.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        cand.truncate(k);
        Ok(cand)
    }

    fn build_or_load(&self, seg_id: u64) -> PersistenceResult<Option<HNSWIndex>> {
        if let Some(index) = self.load_sidecar(seg_id) {
            return Ok(Some(index));
        }
        let segment: Vec<(u32, Vec<f32>)> = self.catalog.read_segment(seg_id)?;
        let index =
            UpdatableIndex::build_live_index_from(self.dim, self.m, self.m_max, &segment, &|id| {
                self.catalog.is_live(id)
            });
        if let Some(index) = &index {
            self.persist_sidecar(index, seg_id);
        }
        Ok(index)
    }

    fn load_sidecar(&self, seg_id: u64) -> Option<HNSWIndex> {
        let name = self.catalog.index_name(seg_id, INDEX_KIND);
        if !self.catalog.dir().exists(&name) {
            return None;
        }
        let mut bytes = Vec::new();
        self.catalog
            .dir()
            .open_file(&name)
            .ok()?
            .read_to_end(&mut bytes)
            .ok()?;
        let graph_bytes = UpdatableIndex::decode_sidecar_for_recipe(&self.sidecar_recipe, &bytes)?;
        let idx = HNSWIndex::from_postcard(graph_bytes).ok()?;
        if idx.doc_ids.iter().all(|id| self.catalog.is_live(id)) {
            Some(idx)
        } else {
            None
        }
    }

    fn persist_sidecar(&self, idx: &HNSWIndex, seg_id: u64) {
        if let Ok(graph) = idx.to_postcard() {
            let Some(bytes) =
                UpdatableIndex::encode_sidecar_for_recipe(&self.sidecar_recipe, &graph)
            else {
                return;
            };
            let _ = self
                .catalog
                .dir()
                .atomic_write(&self.catalog.index_name(seg_id, INDEX_KIND), &bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use durability::MemoryDirectory;
    use std::io::{Read, Write};
    use std::path::PathBuf;

    struct RecordingDirectory {
        inner: Arc<dyn Directory>,
        opened: Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl RecordingDirectory {
        fn wrap(
            inner: Arc<dyn Directory>,
        ) -> (Arc<dyn Directory>, Arc<std::sync::Mutex<Vec<String>>>) {
            let opened = Arc::new(std::sync::Mutex::new(Vec::new()));
            (
                Arc::new(Self {
                    inner,
                    opened: opened.clone(),
                }),
                opened,
            )
        }
    }

    impl Directory for RecordingDirectory {
        fn create_file(&self, path: &str) -> PersistenceResult<Box<dyn Write + Send>> {
            self.inner.create_file(path)
        }

        fn open_file(&self, path: &str) -> PersistenceResult<Box<dyn Read + Send>> {
            if let Ok(mut opened) = self.opened.lock() {
                opened.push(path.to_string());
            }
            self.inner.open_file(path)
        }

        fn exists(&self, path: &str) -> bool {
            self.inner.exists(path)
        }

        fn delete(&self, path: &str) -> PersistenceResult<()> {
            self.inner.delete(path)
        }

        fn atomic_rename(&self, from: &str, to: &str) -> PersistenceResult<()> {
            self.inner.atomic_rename(from, to)
        }

        fn create_dir_all(&self, path: &str) -> PersistenceResult<()> {
            self.inner.create_dir_all(path)
        }

        fn list_dir(&self, path: &str) -> PersistenceResult<Vec<String>> {
            self.inner.list_dir(path)
        }

        fn append_file(&self, path: &str) -> PersistenceResult<Box<dyn Write + Send>> {
            self.inner.append_file(path)
        }

        fn atomic_write(&self, path: &str, data: &[u8]) -> PersistenceResult<()> {
            self.inner.atomic_write(path, data)
        }

        fn file_path(&self, path: &str) -> Option<PathBuf> {
            self.inner.file_path(path)
        }
    }

    fn read_file(dir: &Arc<dyn Directory>, name: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        dir.open_file(name)
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        bytes
    }

    fn checkpointed_store(dir: Arc<dyn Directory>, m: usize, m_max: usize) -> (String, Vec<u8>) {
        let mut store = UpdatableIndex::open(dir, 4, 2, m, m_max).unwrap();
        for i in 0..12u32 {
            let angle = i as f32 * 0.37;
            store.add(i, &[angle.cos(), angle.sin()]).unwrap();
        }
        store.checkpoint().unwrap();
        let seg_id = store.inner.segment_ids()[0];
        let name = store.inner.index_name(seg_id, INDEX_KIND);
        let bytes = read_file(store.inner.dir(), &name);
        (name, bytes)
    }

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
    fn snapshot_index_queries_sidecars_without_opening_segment_payloads() {
        let dir = MemoryDirectory::arc();
        {
            let mut store = UpdatableIndex::open(dir.clone(), 2, 2, 16, 32).unwrap();
            store.add(0, &[1.0, 0.0]).unwrap();
            store.add(1, &[0.95, 0.05]).unwrap();
            store.add(2, &[0.0, 1.0]).unwrap();
            store.add(3, &[-1.0, 0.0]).unwrap();
            store.checkpoint().unwrap();
        }

        let (watched, opened) = RecordingDirectory::wrap(dir);
        let snapshot = SnapshotIndex::open(watched, 2, 16, 32).unwrap();
        assert_eq!(snapshot.segment_count(), 2);
        assert_eq!(snapshot.tombstone_count(), 0);
        let hits = snapshot.search(&[1.0, 0.0], 2, 16).unwrap();
        assert!(
            hits.iter().any(|(id, _)| *id == 0),
            "snapshot should search persisted HNSW sidecars"
        );

        let opened = opened.lock().unwrap().clone();
        assert!(
            opened.iter().any(|path| path.starts_with("segstore.idx.")),
            "snapshot should open persisted sidecars: {opened:?}"
        );
        assert!(
            !opened.iter().any(|path| path.starts_with("segstore.seg.")),
            "valid sidecars should avoid source segment payload reads: {opened:?}"
        );
    }

    #[test]
    fn snapshot_index_rebuilds_tombstoned_hnsw_sidecar_before_search() {
        let dir = MemoryDirectory::arc();
        let (name, stale_sidecar) = checkpointed_store(dir.clone(), 16, 32);
        {
            let mut store = UpdatableIndex::open(dir.clone(), 4, 2, 16, 32).unwrap();
            store.delete(0).unwrap();
            store.checkpoint().unwrap();
            store
                .inner
                .dir()
                .atomic_write(&name, &stale_sidecar)
                .unwrap();
        }

        let (watched, opened) = RecordingDirectory::wrap(dir);
        let snapshot = SnapshotIndex::open(watched, 2, 16, 32).unwrap();
        assert_eq!(snapshot.tombstone_count(), 1);
        let hits = snapshot.search(&[1.0, 0.0], 3, 16).unwrap();
        assert!(
            !hits.iter().any(|(id, _)| *id == 0),
            "deleted id must not be served by a stale HNSW sidecar"
        );
        assert!(!hits.is_empty(), "rebuilt segment should keep live hits");

        let opened = opened.lock().unwrap().clone();
        assert!(
            opened.iter().any(|path| path.starts_with("segstore.idx.")),
            "snapshot should inspect the stale sidecar first: {opened:?}"
        );
        assert!(
            opened.iter().any(|path| path.starts_with("segstore.seg.")),
            "stale HNSW sidecars should be rebuilt from the source segment: {opened:?}"
        );
    }

    #[test]
    fn snapshot_index_rebuilds_missing_sidecar_from_one_segment() {
        let dir = MemoryDirectory::arc();
        let (name, _) = checkpointed_store(dir.clone(), 16, 32);
        dir.delete(&name).unwrap();

        let (watched, opened) = RecordingDirectory::wrap(dir.clone());
        let snapshot = SnapshotIndex::open(watched, 2, 16, 32).unwrap();
        let hits = snapshot.search(&[1.0, 0.0], 3, 16).unwrap();
        assert!(
            !hits.is_empty(),
            "missing sidecar should rebuild enough index to search"
        );
        assert!(
            dir.exists(&name),
            "snapshot fallback should persist the rebuilt sidecar"
        );

        let opened = opened.lock().unwrap().clone();
        assert!(
            opened.iter().any(|path| path.starts_with("segstore.seg.")),
            "missing sidecar should fall back to one source segment read: {opened:?}"
        );
    }

    #[test]
    fn compact_persists_sidecar_and_prunes_cached_indexes() {
        let dir = MemoryDirectory::arc();
        let mut store = UpdatableIndex::open(dir, 2, 2, 16, 32).unwrap();
        store.add(0, &[1.0, 0.0]).unwrap();
        store.add(1, &[0.95, 0.05]).unwrap();
        store.add(2, &[0.0, 1.0]).unwrap();
        store.add(3, &[-1.0, 0.0]).unwrap();

        let before_ids = store.inner.segment_ids().to_vec();
        assert!(
            before_ids.len() >= 2,
            "test setup should create multiple sealed segments"
        );
        let _ = store.search(&[1.0, 0.0], 2, 16);
        assert_eq!(
            store.cache.borrow().by_segment_id.len(),
            before_ids.len(),
            "warm query should cache each sealed segment"
        );

        store.compact().unwrap();

        let after_ids = store.inner.segment_ids().to_vec();
        assert_eq!(
            after_ids.len(),
            1,
            "compact should merge the sealed segments"
        );
        assert!(
            store
                .inner
                .dir()
                .exists(&store.inner.index_name(after_ids[0], INDEX_KIND)),
            "merged segment should have a sidecar immediately after compact"
        );
        assert!(
            store
                .cache
                .borrow()
                .by_segment_id
                .keys()
                .all(|id| after_ids.contains(id)),
            "cache should not retain indexes for compacted-away segment ids"
        );
    }

    #[test]
    fn hnsw_sidecar_recipe_mismatch_rebuilds() {
        let dir = MemoryDirectory::arc();
        let (name, before) = checkpointed_store(dir.clone(), 16, 32);
        assert_eq!(
            &before[..SIDECAR_MAGIC.len()],
            SIDECAR_MAGIC,
            "new sidecars carry the vicinity HNSW envelope"
        );

        let store = UpdatableIndex::open(dir.clone(), 4, 2, 8, 16).unwrap();
        let seg_id = store.inner.segment_ids()[0];
        assert!(
            store
                .load_sidecar(&store.inner.segments()[0][..], seg_id)
                .is_none(),
            "sidecar built with m=16/m_max=32 must not load under m=8/m_max=16"
        );
        assert!(
            !store.search(&[1.0, 0.0], 1, 16).is_empty(),
            "mismatched sidecar falls back to rebuild"
        );

        let after = read_file(store.inner.dir(), &name);
        assert_ne!(before, after, "rebuild overwrites the stale-recipe sidecar");
        assert!(
            store
                .load_sidecar(&store.inner.segments()[0][..], seg_id)
                .is_some(),
            "rebuilt sidecar now matches the current recipe"
        );
    }

    #[test]
    fn hnsw_sidecar_envelope_rejects_corrupt_headers() {
        let store = UpdatableIndex::open(MemoryDirectory::arc(), 4, 2, 16, 32).unwrap();
        let graph = b"graph-bytes";
        let bytes = store.encode_sidecar(graph).unwrap();
        assert_eq!(store.decode_sidecar(&bytes), Some(graph.as_slice()));

        assert!(store.decode_sidecar(&bytes[..8]).is_none());

        let mut bad_magic = bytes.clone();
        bad_magic[0] ^= 0xFF;
        assert!(store.decode_sidecar(&bad_magic).is_none());

        let mut bad_version = bytes.clone();
        bad_version[8..12].copy_from_slice(&(SIDECAR_VERSION + 1).to_le_bytes());
        assert!(store.decode_sidecar(&bad_version).is_none());

        let mut bad_recipe_len = bytes.clone();
        bad_recipe_len[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(store.decode_sidecar(&bad_recipe_len).is_none());

        let mut bad_recipe = bytes.clone();
        bad_recipe[16] ^= 0x01;
        assert!(store.decode_sidecar(&bad_recipe).is_none());
    }

    #[test]
    fn hnsw_sidecar_invalid_graph_payload_rebuilds() {
        let dir = MemoryDirectory::arc();
        let (name, _) = checkpointed_store(dir.clone(), 16, 32);
        {
            let store = UpdatableIndex::open(dir.clone(), 4, 2, 16, 32).unwrap();
            let corrupt = store.encode_sidecar(b"not-a-postcard-hnsw-graph").unwrap();
            store.inner.dir().atomic_write(&name, &corrupt).unwrap();
        }

        let store = UpdatableIndex::open(dir.clone(), 4, 2, 16, 32).unwrap();
        let seg_id = store.inner.segment_ids()[0];
        assert!(
            store
                .load_sidecar(&store.inner.segments()[0][..], seg_id)
                .is_none(),
            "valid envelope with invalid graph bytes is rejected"
        );
        assert!(
            !store.search(&[1.0, 0.0], 1, 16).is_empty(),
            "invalid graph payload falls back to rebuild"
        );
        assert!(
            store
                .load_sidecar(&store.inner.segments()[0][..], seg_id)
                .is_some(),
            "rebuilt sidecar loads after the fallback"
        );
    }

    #[test]
    fn hnsw_sidecar_query_ef_does_not_invalidate_recipe() {
        let dir = MemoryDirectory::arc();
        let (name, before) = checkpointed_store(dir.clone(), 16, 32);

        let store = UpdatableIndex::open(dir.clone(), 4, 2, 16, 32).unwrap();
        let seg_id = store.inner.segment_ids()[0];
        assert!(
            store
                .load_sidecar(&store.inner.segments()[0][..], seg_id)
                .is_some(),
            "sidecar loads before any query"
        );

        assert!(!store.search(&[1.0, 0.0], 3, 8).is_empty());
        assert!(!store.search(&[1.0, 0.0], 3, 64).is_empty());

        assert_eq!(
            read_file(&dir, &name),
            before,
            "query-time ef is not part of the sidecar recipe and must not rewrite it"
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

    #[test]
    fn checkpoint_after_replayed_delete_rewrites_stale_sidecar() {
        let dir = MemoryDirectory::arc();
        let (name, stale_bytes) = {
            let mut store = UpdatableIndex::open(dir.clone(), 2, 2, 16, 32).unwrap();
            store.add(0, &[1.0, 0.0]).unwrap();
            store.add(1, &[0.95, 0.05]).unwrap();
            store.add(2, &[0.0, 1.0]).unwrap();
            store.checkpoint().unwrap();

            let seg_id = store.inner.segment_ids()[0];
            let name = store.inner.index_name(seg_id, INDEX_KIND);
            let bytes = read_file(store.inner.dir(), &name);

            // Simulate a crash after the delete is durably logged but before
            // `UpdatableIndex::delete` removes the now-stale sidecar.
            store.inner.delete(0).unwrap();
            (name, bytes)
        };

        let mut store = UpdatableIndex::open(dir.clone(), 2, 2, 16, 32).unwrap();
        let seg_id = store.inner.segment_ids()[0];
        assert!(
            store
                .load_sidecar(&store.inner.segments()[0][..], seg_id)
                .is_none(),
            "replayed tombstone must make the old sidecar stale"
        );

        store.checkpoint().unwrap();

        let rewritten = read_file(&dir, &name);
        assert_ne!(
            rewritten, stale_bytes,
            "checkpoint should rewrite stale sidecars even before search"
        );
        let idx = store
            .load_sidecar(&store.inner.segments()[0][..], seg_id)
            .expect("rewritten sidecar should be valid");
        assert!(
            !idx.doc_ids.contains(&0),
            "rewritten sidecar must exclude the replayed delete"
        );
        assert!(
            idx.doc_ids.contains(&1),
            "rewritten sidecar should keep live ids from the segment"
        );
    }
}
