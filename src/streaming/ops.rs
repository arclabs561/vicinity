//! Update operations for streaming indices.

/// A single update operation.
#[derive(Debug, Clone)]
pub enum UpdateOp {
    /// Insert a new vector.
    Insert {
        /// Identifier for the new vector.
        id: u32,
        /// The vector data to insert.
        vector: Vec<f32>,
    },
    /// Delete an existing vector.
    Delete {
        /// Identifier of the vector to delete.
        id: u32,
    },
    /// Update (atomic delete + insert).
    Update {
        /// Identifier of the vector to update.
        id: u32,
        /// Replacement vector data.
        vector: Vec<f32>,
    },
}

impl UpdateOp {
    /// Get the ID affected by this operation.
    pub fn id(&self) -> u32 {
        match self {
            UpdateOp::Insert { id, .. } => *id,
            UpdateOp::Delete { id } => *id,
            UpdateOp::Update { id, .. } => *id,
        }
    }

    /// Check if this is a delete operation.
    pub fn is_delete(&self) -> bool {
        matches!(self, UpdateOp::Delete { .. })
    }

    /// Check if this operation adds data (insert or update).
    pub fn adds_data(&self) -> bool {
        matches!(self, UpdateOp::Insert { .. } | UpdateOp::Update { .. })
    }
}

/// A batch of update operations.
#[derive(Debug, Clone, Default)]
pub struct UpdateBatch {
    ops: Vec<UpdateOp>,
}

impl UpdateBatch {
    /// Create an empty batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a batch pre-allocated for `capacity` operations.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            ops: Vec::with_capacity(capacity),
        }
    }

    /// Append an operation to the batch.
    pub fn push(&mut self, op: UpdateOp) {
        self.ops.push(op);
    }

    /// Number of operations in the batch.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Check whether the batch contains no operations.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Iterate over operations in submission order.
    pub fn iter(&self) -> impl Iterator<Item = &UpdateOp> {
        self.ops.iter()
    }
}

impl IntoIterator for UpdateBatch {
    type Item = UpdateOp;
    type IntoIter = std::vec::IntoIter<UpdateOp>;

    fn into_iter(self) -> Self::IntoIter {
        self.ops.into_iter()
    }
}

impl UpdateBatch {
    /// Count inserts in batch.
    pub fn insert_count(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, UpdateOp::Insert { .. }))
            .count()
    }

    /// Count deletes in batch.
    pub fn delete_count(&self) -> usize {
        self.ops.iter().filter(|op| op.is_delete()).count()
    }

    /// Count updates in batch.
    pub fn update_count(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, UpdateOp::Update { .. }))
            .count()
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch() {
        let mut batch = UpdateBatch::new();
        batch.push(UpdateOp::Insert {
            id: 0,
            vector: vec![1.0],
        });
        batch.push(UpdateOp::Delete { id: 1 });
        batch.push(UpdateOp::Update {
            id: 2,
            vector: vec![2.0],
        });

        assert_eq!(batch.len(), 3);
        assert_eq!(batch.insert_count(), 1);
        assert_eq!(batch.delete_count(), 1);
        assert_eq!(batch.update_count(), 1);
    }
}
