# hnswlib-sys (benchmark-only)

A thin `extern "C"` shim over [hnswlib](https://github.com/nmslib/hnswlib),
the reference C++ HNSW implementation, used by `tools/parity-rows` to measure
the real library rather than a Rust re-implementation of it.

- Upstream: `nmslib/hnswlib`
- Pinned commit: `ca426729609b3221563047ceb269cc1213e0d376`
  (recorded in `vendor/hnswlib/COMMIT`)
- Vendored files: every header under `vendor/hnswlib/` (unmodified, header-only)
- License: Apache-2.0 (upstream), same as this crate
- Instantiation: `hnswlib::HierarchicalNSW<float>` with
  `hnswlib::L2Space` or `hnswlib::InnerProductSpace`
- Compiler: Apple clang, `-std=c++14 -O3`; no system packages beyond the Xcode
  Command Line Tools (libc++ is the system C++ runtime)

## Cosine needs normalised input

hnswlib has **no cosine space**. For cosine ordering, L2-normalise the vectors
once and use `Metric::InnerProduct` (`hnswlib::InnerProductSpace`), whose
distance is `1 - dot(q, v)`. On unit vectors that is exactly `1 - cos(q, v)`,
so the ordering and the numeric distance match a cosine index. The harness does
this normalisation once, before either side sees the data.

## Differences from the minimal surface

- `hnswlib_new` also takes `max_elements`. hnswlib allocates its level-0 arena
  up front, and growing it (`resizeIndex`) reallocates and copies the whole
  graph. The harness always knows the corpus size, so the capacity is passed at
  construction rather than treated as a growth path.
- Search uses `searchKnnCloserFirstNoExceptions`, and insertion uses
  `addPointNoExceptions`. Both are hnswlib's non-throwing entry points; the
  shim still wraps every body in `try/catch (...)` and reports failures as
  negative counts or `0`/null, so no exception can cross the FFI boundary.
- Labels are `size_t` in hnswlib and `uint64_t` in the shim. `u64` is lossless
  for every id this harness produces (base offsets below 2^32).

## AArch64 caveat

This hnswlib commit vectorises only on x86 (`USE_SSE`, `USE_AVX`,
`USE_AVX512`); there is no NEON path. On Apple Silicon it compiles the scalar
fallback while `vicinity` and USearch use NEON kernels. Per-distance cost is
therefore not like-for-like on this host, and the harness reports the ratio
rather than hiding it behind a single headline QPS.

This crate is a **dev-dependency only** of `tools/parity-rows`. It is never
built for the parent crate's library target.

The vendored upstream headers are load-bearing for the shim.
`~/.gitignore_global` ignores `vendor/`, so they must be added explicitly
(`git add -f tools/hnswlib-sys/vendor`).
