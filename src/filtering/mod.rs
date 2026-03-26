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
}
