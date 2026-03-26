//! Statistics from applying streaming updates.

/// Statistics from applying updates.
#[derive(Debug, Clone, Default)]
pub struct UpdateStats {
    /// Number of insert operations applied.
    pub inserts_applied: usize,
    /// Number of delete operations applied.
    pub deletes_applied: usize,
    /// Number of update operations applied.
    pub updates_applied: usize,
    /// Number of operations that failed.
    pub errors: usize,
    /// Time spent applying updates (microseconds).
    pub duration_us: u64,
}

impl UpdateStats {
    /// Accumulate counters from another stats snapshot.
    pub fn merge(&mut self, other: &UpdateStats) {
        self.inserts_applied += other.inserts_applied;
        self.deletes_applied += other.deletes_applied;
        self.updates_applied += other.updates_applied;
        self.errors += other.errors;
        self.duration_us += other.duration_us;
    }
}
