//! Benchmark-only wrapper around [hnswlib](https://github.com/nmslib/hnswlib),
//! the reference C++ HNSW implementation.
//!
//! Vendored source and provenance: `tools/hnswlib-sys/vendor/hnswlib/` and
//! `tools/hnswlib-sys/README.md`. This crate exists so the comparison in
//! `tools/parity-rows` measures the real C++ implementation rather than a Rust
//! re-implementation of it.
//!
//! Scope: `hnswlib::HierarchicalNSW<float>`, built level by level through
//! `addPointNoExceptions`, searched through
//! `searchKnnCloserFirstNoExceptions`. Both are hnswlib's non-throwing entry
//! points; the shim additionally catches anything that still escapes.
//!
//! Metric convention: [`Metric::L2`] is squared Euclidean. [`Metric::InnerProduct`]
//! is `1 - dot(q, v)`; hnswlib has no cosine space, so callers who want cosine
//! ordering must L2-normalise their vectors first.

use std::ffi::{c_char, c_int, c_void, CStr};

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// Squared Euclidean distance (`hnswlib::L2Space`).
    L2 = 0,
    /// `1 - dot(q, v)` (`hnswlib::InnerProductSpace`). Normalise first for
    /// cosine ordering.
    InnerProduct = 1,
}

extern "C" {
    fn hnswlib_new(
        dim: usize,
        m: usize,
        ef_construction: usize,
        metric: c_int,
        max_elements: usize,
    ) -> *mut c_void;
    fn hnswlib_free(handle: *mut c_void);
    fn hnswlib_add(handle: *mut c_void, id: u64, vector: *const f32) -> c_int;
    fn hnswlib_set_ef(handle: *mut c_void, ef: usize) -> c_int;
    fn hnswlib_search(
        handle: *mut c_void,
        query: *const f32,
        k: usize,
        out_ids: *mut u64,
        out_dists: *mut f32,
    ) -> c_int;
    fn hnswlib_size(handle: *mut c_void) -> usize;
    fn hnswlib_last_error() -> *const c_char;
}

/// An hnswlib `HierarchicalNSW<float>` index.
pub struct HnswlibIndex {
    raw: *mut c_void,
    dim: usize,
}

impl HnswlibIndex {
    /// Create an index with capacity for `max_elements` vectors.
    ///
    /// hnswlib allocates its level-0 arena up front and growing it
    /// (`resizeIndex`) reallocates and copies the whole graph, so the corpus
    /// size is a constructor argument here rather than a growth path.
    ///
    /// Returns `None` when the shim rejects the configuration or allocation
    /// fails; [`HnswlibIndex::last_error`] then carries the reason.
    #[must_use]
    pub fn new(
        dim: usize,
        m: usize,
        ef_construction: usize,
        metric: Metric,
        max_elements: usize,
    ) -> Option<Self> {
        // SAFETY: plain FFI constructor; the shim validates the arguments.
        let raw = unsafe { hnswlib_new(dim, m, ef_construction, metric as c_int, max_elements) };
        if raw.is_null() {
            None
        } else {
            Some(Self { raw, dim })
        }
    }

    /// Add one vector under `id`, in insertion order.
    ///
    /// hnswlib has no separate build step: each `add` links the point into the
    /// graph as it arrives, so this call is the whole build.
    ///
    /// # Errors
    /// Returns hnswlib's error message when the point is rejected (for example,
    /// capacity exhausted or a duplicate label).
    pub fn add(&mut self, id: u64, vector: &[f32]) -> Result<(), String> {
        assert_eq!(vector.len(), self.dim, "vector dimension mismatch");
        // SAFETY: `raw` is live and `vector` holds `dim` elements.
        let rc = unsafe { hnswlib_add(self.raw, id, vector.as_ptr()) };
        if rc == 0 {
            Ok(())
        } else {
            Err(Self::last_error())
        }
    }

    /// Set the search beam width for subsequent queries.
    ///
    /// # Errors
    /// Returns hnswlib's error message if the shim rejects the call.
    pub fn set_ef(&mut self, ef: usize) -> Result<(), String> {
        // SAFETY: `raw` is a live index owned by `self`.
        let rc = unsafe { hnswlib_set_ef(self.raw, ef) };
        if rc == 0 {
            Ok(())
        } else {
            Err(Self::last_error())
        }
    }

    /// Search for the `k` nearest neighbours, closest first.
    ///
    /// Returns `(label, distance)` pairs with hnswlib's raw distance (`1 - dot`
    /// on unit vectors for [`Metric::InnerProduct`]).
    ///
    /// # Errors
    /// Returns hnswlib's error message when the search fails.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>, String> {
        assert_eq!(query.len(), self.dim, "query dimension mismatch");
        let mut ids = vec![0u64; k];
        let mut dists = vec![0f32; k];
        // SAFETY: both output buffers hold `k` elements and `raw` is live.
        let written = unsafe {
            hnswlib_search(
                self.raw,
                query.as_ptr(),
                k,
                ids.as_mut_ptr(),
                dists.as_mut_ptr(),
            )
        };
        if written < 0 {
            return Err(Self::last_error());
        }
        let written = usize::try_from(written).unwrap_or(0).min(k);
        ids.truncate(written);
        dists.truncate(written);
        Ok(ids.into_iter().zip(dists).collect())
    }

    /// Number of vectors currently in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        // SAFETY: `raw` is a live index owned by `self`.
        unsafe { hnswlib_size(self.raw) }
    }

    /// Whether the index holds no vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Last error recorded on this thread by the shim.
    ///
    /// hnswlib's error channel is per-thread, so this is only meaningful on the
    /// thread that made the failing call.
    #[must_use]
    pub fn last_error() -> String {
        // SAFETY: the shim returns a pointer to a thread-local `std::string`
        // that lives until the next shim call on this thread.
        unsafe {
            let ptr = hnswlib_last_error();
            if ptr.is_null() {
                String::new()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        }
    }
}

impl Drop for HnswlibIndex {
    fn drop(&mut self) {
        // SAFETY: `raw` came from `hnswlib_new` and is freed exactly once.
        unsafe { hnswlib_free(self.raw) };
    }
}

// SAFETY: the index is heap-allocated and self-contained; `search` never mutates
// through `&self`, and hnswlib's per-search scratch is allocated inside the
// call.
unsafe impl Send for HnswlibIndex {}
