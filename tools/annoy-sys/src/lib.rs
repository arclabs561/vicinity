//! Benchmark-only wrapper around Spotify's Annoy C++ library (v1.17.3).
//!
//! Vendored source and provenance: `tools/annoy-sys/vendor/` and
//! `tools/annoy-sys/README.md`. This crate exists so the comparison in
//! `tests/annoy_parity.rs` measures the real C++ implementation rather than a
//! Rust re-implementation of it.
//!
//! Scope: `AnnoyIndex<int32_t, float, Angular>` with the single-threaded build
//! policy.

use std::os::raw::{c_int, c_void};

extern "C" {
    fn annoy_new(dimension: c_int) -> *mut c_void;
    fn annoy_free(handle: *mut c_void);
    fn annoy_set_seed(handle: *mut c_void, seed: u32);
    fn annoy_leaf_k(handle: *mut c_void) -> i32;
    fn annoy_add_item(handle: *mut c_void, item: i32, v: *const f32) -> c_int;
    fn annoy_build(handle: *mut c_void, trees: c_int) -> c_int;
    fn annoy_search(
        handle: *mut c_void,
        q: *const f32,
        k: c_int,
        search_k: c_int,
        out_ids: *mut i32,
        out_dists: *mut f32,
    ) -> c_int;
}

/// Angular-distance Annoy index over `f32` vectors.
pub struct AnnoyIndex {
    raw: *mut c_void,
    dim: usize,
}

impl AnnoyIndex {
    /// Create an index for `dim`-dimensional vectors.
    ///
    /// # Panics
    /// Panics if `dim` is zero or does not fit in `c_int`.
    pub fn new(dim: usize) -> Self {
        let dim_i32 = c_int::try_from(dim).expect("dim must fit in c_int");
        assert!(dim_i32 > 0, "dim must be > 0");
        // SAFETY: FFI constructor with a validated dimension.
        let raw = unsafe { annoy_new(dim_i32) };
        assert!(!raw.is_null(), "annoy_new returned null");
        Self { raw, dim }
    }

    /// Seed Annoy's rng (controls the two-means sampling).
    pub fn set_seed(&mut self, seed: u32) {
        // SAFETY: `raw` is a live index owned by `self`.
        unsafe { annoy_set_seed(self.raw, seed) };
    }

    /// Annoy's effective leaf threshold, which the C++ library derives from the
    /// node layout for the given dimension.
    #[must_use]
    pub fn leaf_k(&self) -> usize {
        // SAFETY: `raw` is a live index owned by `self`.
        let k = unsafe { annoy_leaf_k(self.raw) };
        usize::try_from(k).unwrap_or(0)
    }

    /// Add one vector. Annoy's Angular metric normalizes it internally.
    pub fn add_item(&mut self, id: u32, v: &[f32]) -> bool {
        assert_eq!(v.len(), self.dim, "vector dimension mismatch");
        let id = i32::try_from(id).expect("id must fit in i32");
        // SAFETY: `v` has `dim` elements, matching the index dimension.
        unsafe { annoy_add_item(self.raw, id, v.as_ptr()) == 1 }
    }

    /// Build `trees` trees with a single build thread.
    pub fn build(&mut self, trees: usize) -> bool {
        let trees = c_int::try_from(trees).expect("tree count must fit in c_int");
        // SAFETY: `raw` is a live, unbuilt index owned by `self`.
        unsafe { annoy_build(self.raw, trees) == 1 }
    }

    /// Search for the `k` nearest neighbors.
    ///
    /// `search_k == -1` selects Annoy's default budget, `k * trees`. The result
    /// is `(id, angular_distance)` in ascending distance order, with at most `k`
    /// entries.
    pub fn search(&self, q: &[f32], k: usize, search_k: i32) -> Vec<(u32, f32)> {
        assert_eq!(q.len(), self.dim, "query dimension mismatch");
        let mut ids = vec![0i32; k];
        let mut dists = vec![0f32; k];
        // SAFETY: both output buffers hold `k` elements.
        let n = unsafe {
            annoy_search(
                self.raw,
                q.as_ptr(),
                c_int::try_from(k).expect("k must fit in c_int"),
                search_k,
                ids.as_mut_ptr(),
                dists.as_mut_ptr(),
            )
        };
        let n = usize::try_from(n.max(0)).unwrap_or(0).min(k);
        ids.truncate(n);
        dists.truncate(n);
        ids.into_iter()
            .zip(dists)
            .map(|(id, d)| (u32::try_from(id).unwrap_or_default(), d))
            .collect()
    }
}

impl Drop for AnnoyIndex {
    fn drop(&mut self) {
        // SAFETY: `raw` came from `annoy_new` and is dropped exactly once.
        unsafe { annoy_free(self.raw) };
    }
}

// SAFETY: the underlying C++ index is heap-allocated and self-contained; it is
// never mutated through `&self` by the safe API.
unsafe impl Send for AnnoyIndex {}
