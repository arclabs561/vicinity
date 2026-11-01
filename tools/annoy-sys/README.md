# annoy-sys (benchmark-only)

A thin `extern "C"` shim over Spotify's [Annoy](https://github.com/spotify/annoy),
used by `tests/annoy_parity.rs` to measure the real C++ implementation instead of
a Rust port of it.

- Upstream: `spotify/annoy`, tag `v1.17.3`
- Vendored files: `vendor/annoylib.h`, `vendor/kissrandom.h` (unmodified; Annoy is
  header-only, there is no `annoylib.cpp` in this version)
- License: Apache-2.0 (upstream), same as this crate
- Instantiation: `AnnoyIndex<int32_t, float, Angular, Kiss32Random>`,
  `AnnoyIndexSingleThreadedBuildPolicy`, build with `n_threads = 1`
- Compiler: Apple clang, `-std=c++11`; no system packages beyond the Xcode
  Command Line Tools (libc++ is the system C++ runtime)

Why single-threaded: the vector crate's tree builds are single-threaded, so
multi-threaded Annoy builds would not be a like-for-like build-time comparison.

This crate is a **dev-dependency only**, behind the `annoy-bench` feature of the
parent crate. It is never built for the default library target.

# The two vendored upstream headers are load-bearing for the bench shim;
# ~/.gitignore_global ignores `vendor/`, so add them explicitly.
