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
