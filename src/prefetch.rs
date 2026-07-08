//! Compatibility hook for graph-search prefetch sites.

/// Compatibility no-op for former software prefetch call sites.
///
/// The architecture-specific hint was measured as neutral or slower on the
/// Apple Silicon HNSW search workloads that dominate the current profiles. Keep
/// the wrapper so graph-family call sites can be retested without widening the
/// unsafe surface.
#[inline(always)]
#[allow(unused_variables)]
pub(crate) fn prefetch_read_data<T>(ptr: *const T) {}
