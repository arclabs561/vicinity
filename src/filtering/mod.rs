//! Filtering and lightweight faceting support for vector search.
//!
//! Filtered ANN traversal is an index concern (it changes how graph search
//! proceeds), not a pipeline concern.

use std::collections::HashMap;

/// Filter predicate for metadata-based filtering.
#[derive(Clone, Debug)]
pub enum MetadataFilter {
    /// Equality filter: field must equal value.
    Equals {
        /// Metadata field name to match on.
        field: String,
        /// Required value for the field.
        value: u32,
    },
    /// Multiple equality filters (AND logic)
    And(Vec<MetadataFilter>),
    /// Multiple equality filters (OR logic)
    Or(Vec<MetadataFilter>),
}

impl MetadataFilter {
    /// Create an equality filter for a field and value.
    pub fn equals(field: impl Into<String>, value: u32) -> Self {
        Self::Equals {
            field: field.into(),
            value,
        }
    }

    /// Check whether the given document metadata satisfies this predicate.
    pub fn matches(&self, metadata: &DocumentMetadata) -> bool {
        match self {
            Self::Equals { field, value } => metadata.get(field).is_some_and(|&v| v == *value),
            Self::And(predicates) => predicates.iter().all(|p| p.matches(metadata)),
            Self::Or(predicates) => predicates.iter().any(|p| p.matches(metadata)),
        }
    }
}

/// Document metadata storage.
pub type DocumentMetadata = HashMap<String, u32>;

/// Metadata storage for a collection of documents.
#[derive(Debug)]
pub struct MetadataStore {
    metadata: HashMap<u32, DocumentMetadata>,
}

impl MetadataStore {
    /// Create an empty metadata store.
    pub fn new() -> Self {
        Self {
            metadata: HashMap::new(),
        }
    }

    /// Insert metadata for a document, replacing any existing entry.
    pub fn add(&mut self, doc_id: u32, metadata: DocumentMetadata) {
        self.metadata.insert(doc_id, metadata);
    }

    /// Retrieve metadata for a document, if present.
    pub fn get(&self, doc_id: u32) -> Option<&DocumentMetadata> {
        self.metadata.get(&doc_id)
    }

    /// Check whether a document's metadata satisfies the given filter.
    pub fn matches(&self, doc_id: u32, filter: &MetadataFilter) -> bool {
        self.metadata
            .get(&doc_id)
            .is_some_and(|metadata| filter.matches(metadata))
    }

    /// Estimate the fraction of documents that match the filter (0.0 to 1.0).
    ///
    /// Returns `None` if the store is empty.
    pub fn estimate_selectivity(&self, filter: &MetadataFilter) -> Option<f32> {
        if self.metadata.is_empty() {
            return None;
        }

        let matching = self
            .metadata
            .iter()
            .filter(|(_, metadata)| filter.matches(metadata))
            .count();

        Some(matching as f32 / self.metadata.len() as f32)
    }

    /// Return all distinct values for a metadata field, sorted ascending.
    pub fn get_all_values(&self, field: &str) -> Vec<u32> {
        let mut values: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for metadata in self.metadata.values() {
            if let Some(&value) = metadata.get(field) {
                values.insert(value);
            }
        }
        let mut result: Vec<u32> = values.into_iter().collect();
        result.sort();
        result
    }

    /// Return `(value, count)` pairs for a field, sorted descending by count.
    pub fn get_value_counts(&self, field: &str) -> Vec<(u32, usize)> {
        let mut counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for metadata in self.metadata.values() {
            if let Some(&value) = metadata.get(field) {
                *counts.entry(value).or_insert(0) += 1;
            }
        }
        let mut result: Vec<(u32, usize)> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }

    /// Like [`get_value_counts`](Self::get_value_counts), but only considers documents matching the filter.
    pub fn get_value_counts_filtered(
        &self,
        field: &str,
        filter: &MetadataFilter,
    ) -> Vec<(u32, usize)> {
        let mut counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for metadata in self.metadata.values() {
            if filter.matches(metadata) {
                if let Some(&value) = metadata.get(field) {
                    *counts.entry(value).or_insert(0) += 1;
                }
            }
        }
        let mut result: Vec<(u32, usize)> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }
}

impl Default for MetadataStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metadata() -> DocumentMetadata {
        let mut m = DocumentMetadata::new();
        m.insert("color".to_string(), 1);
        m.insert("size".to_string(), 42);
        m
    }

    // --- MetadataFilter ---

    #[test]
    fn equals_matches_correct_field_value() {
        let meta = sample_metadata();
        let pred = MetadataFilter::equals("color", 1);
        assert!(pred.matches(&meta));
    }

    #[test]
    fn equals_rejects_wrong_value() {
        let meta = sample_metadata();
        let pred = MetadataFilter::equals("color", 99);
        assert!(!pred.matches(&meta));
    }

    #[test]
    fn equals_rejects_missing_field() {
        let meta = sample_metadata();
        let pred = MetadataFilter::equals("weight", 1);
        assert!(!pred.matches(&meta));
    }

    #[test]
    fn and_all_true() {
        let meta = sample_metadata();
        let pred = MetadataFilter::And(vec![
            MetadataFilter::equals("color", 1),
            MetadataFilter::equals("size", 42),
        ]);
        assert!(pred.matches(&meta));
    }

    #[test]
    fn and_one_false() {
        let meta = sample_metadata();
        let pred = MetadataFilter::And(vec![
            MetadataFilter::equals("color", 1),
            MetadataFilter::equals("size", 99),
        ]);
        assert!(!pred.matches(&meta));
    }

    #[test]
    fn and_empty_is_vacuously_true() {
        let meta = sample_metadata();
        let pred = MetadataFilter::And(vec![]);
        assert!(pred.matches(&meta));
    }

    #[test]
    fn or_one_true() {
        let meta = sample_metadata();
        let pred = MetadataFilter::Or(vec![
            MetadataFilter::equals("color", 99),
            MetadataFilter::equals("size", 42),
        ]);
        assert!(pred.matches(&meta));
    }

    #[test]
    fn or_none_true() {
        let meta = sample_metadata();
        let pred = MetadataFilter::Or(vec![
            MetadataFilter::equals("color", 99),
            MetadataFilter::equals("size", 99),
        ]);
        assert!(!pred.matches(&meta));
    }

    #[test]
    fn or_empty_is_false() {
        let meta = sample_metadata();
        let pred = MetadataFilter::Or(vec![]);
        assert!(!pred.matches(&meta));
    }

    // --- MetadataStore ---

    #[test]
    fn metadata_store_add_get_roundtrip() {
        let mut store = MetadataStore::new();
        let meta = sample_metadata();
        store.add(0, meta.clone());
        let retrieved = store.get(0).unwrap();
        assert_eq!(retrieved.get("color"), Some(&1));
        assert_eq!(retrieved.get("size"), Some(&42));
    }

    #[test]
    fn metadata_store_get_missing_returns_none() {
        let store = MetadataStore::new();
        assert!(store.get(999).is_none());
    }

    #[test]
    fn metadata_store_matches_delegates_to_predicate() {
        let mut store = MetadataStore::new();
        store.add(0, sample_metadata());
        assert!(store.matches(0, &MetadataFilter::equals("color", 1)));
        assert!(!store.matches(0, &MetadataFilter::equals("color", 99)));
        assert!(!store.matches(999, &MetadataFilter::equals("color", 1)));
    }

    #[test]
    fn estimate_selectivity_empty_store_returns_none() {
        let store = MetadataStore::new();
        assert!(store
            .estimate_selectivity(&MetadataFilter::equals("x", 1))
            .is_none());
    }

    #[test]
    fn estimate_selectivity_all_match() {
        let mut store = MetadataStore::new();
        for i in 0..10 {
            let mut m = DocumentMetadata::new();
            m.insert("x".to_string(), 1);
            store.add(i, m);
        }
        let sel = store
            .estimate_selectivity(&MetadataFilter::equals("x", 1))
            .unwrap();
        assert!((sel - 1.0).abs() < 1e-6);
    }

    #[test]
    fn estimate_selectivity_half_match() {
        let mut store = MetadataStore::new();
        for i in 0..10 {
            let mut m = DocumentMetadata::new();
            m.insert("x".to_string(), if i < 5 { 1 } else { 2 });
            store.add(i, m);
        }
        let sel = store
            .estimate_selectivity(&MetadataFilter::equals("x", 1))
            .unwrap();
        assert!((sel - 0.5).abs() < 1e-6);
    }

    #[test]
    fn estimate_selectivity_none_match() {
        let mut store = MetadataStore::new();
        let mut m = DocumentMetadata::new();
        m.insert("x".to_string(), 1);
        store.add(0, m);
        let sel = store
            .estimate_selectivity(&MetadataFilter::equals("x", 99))
            .unwrap();
        assert!((sel - 0.0).abs() < 1e-6);
    }

    #[test]
    fn get_all_values_sorted() {
        let mut store = MetadataStore::new();
        for (i, val) in [5, 3, 1, 3, 5].iter().enumerate() {
            let mut m = DocumentMetadata::new();
            m.insert("x".to_string(), *val);
            store.add(i as u32, m);
        }
        assert_eq!(store.get_all_values("x"), vec![1, 3, 5]);
    }

    #[test]
    fn get_value_counts_descending() {
        let mut store = MetadataStore::new();
        for (i, val) in [1, 2, 2, 3, 3, 3].iter().enumerate() {
            let mut m = DocumentMetadata::new();
            m.insert("x".to_string(), *val);
            store.add(i as u32, m);
        }
        let counts = store.get_value_counts("x");
        // Descending by count
        assert_eq!(counts[0], (3, 3));
        assert_eq!(counts[1], (2, 2));
        assert_eq!(counts[2], (1, 1));
    }
}
