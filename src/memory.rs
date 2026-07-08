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
    feature = "diskann",
    feature = "finger",
    feature = "filtered_graph",
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
    let list_storage_bytes = lists
        .len()
        .saturating_mul(std::mem::size_of::<smallvec::SmallVec<A>>());
    let spilled_bytes = lists
        .iter()
        .filter(|list| list.spilled())
        .map(|list| list.capacity() * std::mem::size_of::<u32>())
        .sum::<usize>();
    list_storage_bytes.saturating_add(spilled_bytes)
}

#[cfg(all(
    test,
    any(
        feature = "emg",
        feature = "finger",
        feature = "filtered_graph",
        feature = "fresh_graph",
        feature = "hnsw",
        feature = "nsg",
        feature = "nsw",
        feature = "pipnn",
        feature = "sng",
        feature = "vamana"
    )
))]
mod tests {
    use smallvec::SmallVec;

    #[test]
    fn smallvec_u32_bytes_counts_inline_storage() {
        let mut lists: Vec<SmallVec<[u32; 16]>> = Vec::with_capacity(4);
        lists.push([1, 2, 3].into_iter().collect());
        lists.push((0..16).collect());

        assert_eq!(
            super::smallvec_u32_bytes(&lists),
            lists.len() * std::mem::size_of::<SmallVec<[u32; 16]>>()
        );
    }

    #[test]
    fn smallvec_u32_bytes_counts_spilled_storage_once() {
        let mut list: SmallVec<[u32; 2]> = SmallVec::new();
        list.extend(0..3);
        assert!(list.spilled());
        let spilled_capacity = list.capacity();

        let mut lists = Vec::with_capacity(3);
        lists.push(list);

        assert_eq!(
            super::smallvec_u32_bytes(&lists),
            lists.len() * std::mem::size_of::<SmallVec<[u32; 2]>>()
                + spilled_capacity * std::mem::size_of::<u32>()
        );
    }
}
