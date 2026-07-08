//! Memory reporting types for index introspection.

/// Memory usage breakdown for an index.
#[derive(Debug, Clone)]
pub struct MemoryReport {
    /// Raw vector data (f32 arrays).
    pub vectors_bytes: usize,
    /// Graph structure (neighbor lists, layers).
    pub graph_bytes: usize,
    /// Quantized codes (PQ codes, RaBitQ codes, etc.)
    pub quantized_bytes: usize,
    /// Other metadata (doc IDs, layer assignments, centroids, etc.)
    pub metadata_bytes: usize,
}

impl MemoryReport {
    /// Total memory across all categories.
    pub fn total(&self) -> usize {
        self.vectors_bytes + self.graph_bytes + self.quantized_bytes + self.metadata_bytes
    }
}

#[cfg(any(
    feature = "emg",
    feature = "finger",
    feature = "fresh_graph",
    feature = "hnsw",
    feature = "nsg",
    feature = "nsw",
    feature = "pipnn",
    feature = "sng",
    feature = "vamana"
))]
pub(crate) fn smallvec_u32_bytes<A>(lists: &[smallvec::SmallVec<A>]) -> usize
where
    A: smallvec::Array<Item = u32>,
{
    lists
        .iter()
        .map(|list| list.capacity() * std::mem::size_of::<u32>())
        .sum()
}
