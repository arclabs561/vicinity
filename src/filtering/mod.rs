//! Filtering and lightweight faceting support for vector search.
//!
//! Filtered ANN traversal is an index concern (it changes how graph search
//! proceeds), not a pipeline concern.

pub mod inline;

use crate::error::RetrieveError;
use std::collections::HashMap;

pub use inline::{FilterStrategy, FilterStrategySelector, InlineFilter, InlineFilterConfig};

/// Filter predicate for metadata-based filtering.
#[derive(Clone, Debug)]
pub enum FilterPredicate {
    /// Equality filter: field must equal value
    Equals { field: String, value: u32 },
    /// Multiple equality filters (AND logic)
    And(Vec<FilterPredicate>),
    /// Multiple equality filters (OR logic)
    Or(Vec<FilterPredicate>),
}

impl FilterPredicate {
    pub fn equals(field: impl Into<String>, value: u32) -> Self {
        Self::Equals {
            field: field.into(),
            value,
        }
    }

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
    pub fn new() -> Self {
        Self {
            metadata: HashMap::new(),
        }
    }

    pub fn add(&mut self, doc_id: u32, metadata: DocumentMetadata) {
        self.metadata.insert(doc_id, metadata);
    }

    pub fn get(&self, doc_id: u32) -> Option<&DocumentMetadata> {
        self.metadata.get(&doc_id)
    }

    pub fn matches(&self, doc_id: u32, filter: &FilterPredicate) -> bool {
        self.metadata
            .get(&doc_id)
            .is_some_and(|metadata| filter.matches(metadata))
    }

    pub fn estimate_selectivity(&self, filter: &FilterPredicate) -> Option<f32> {
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

    pub fn get_value_counts_filtered(
        &self,
        field: &str,
        filter: &FilterPredicate,
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

    // --- FilterPredicate ---

    #[test]
    fn equals_matches_correct_field_value() {
        let meta = sample_metadata();
        let pred = FilterPredicate::equals("color", 1);
        assert!(pred.matches(&meta));
    }

    #[test]
    fn equals_rejects_wrong_value() {
        let meta = sample_metadata();
        let pred = FilterPredicate::equals("color", 99);
        assert!(!pred.matches(&meta));
    }

    #[test]
    fn equals_rejects_missing_field() {
        let meta = sample_metadata();
        let pred = FilterPredicate::equals("weight", 1);
        assert!(!pred.matches(&meta));
    }

    #[test]
    fn and_all_true() {
        let meta = sample_metadata();
        let pred = FilterPredicate::And(vec![
            FilterPredicate::equals("color", 1),
            FilterPredicate::equals("size", 42),
        ]);
        assert!(pred.matches(&meta));
    }

    #[test]
    fn and_one_false() {
        let meta = sample_metadata();
        let pred = FilterPredicate::And(vec![
            FilterPredicate::equals("color", 1),
            FilterPredicate::equals("size", 99),
        ]);
        assert!(!pred.matches(&meta));
    }

    #[test]
    fn and_empty_is_vacuously_true() {
        let meta = sample_metadata();
        let pred = FilterPredicate::And(vec![]);
        assert!(pred.matches(&meta));
    }

    #[test]
    fn or_one_true() {
        let meta = sample_metadata();
        let pred = FilterPredicate::Or(vec![
            FilterPredicate::equals("color", 99),
            FilterPredicate::equals("size", 42),
        ]);
        assert!(pred.matches(&meta));
    }

    #[test]
    fn or_none_true() {
        let meta = sample_metadata();
        let pred = FilterPredicate::Or(vec![
            FilterPredicate::equals("color", 99),
            FilterPredicate::equals("size", 99),
        ]);
        assert!(!pred.matches(&meta));
    }

    #[test]
    fn or_empty_is_false() {
        let meta = sample_metadata();
        let pred = FilterPredicate::Or(vec![]);
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
        assert!(store.matches(0, &FilterPredicate::equals("color", 1)));
        assert!(!store.matches(0, &FilterPredicate::equals("color", 99)));
        assert!(!store.matches(999, &FilterPredicate::equals("color", 1)));
    }

    #[test]
    fn estimate_selectivity_empty_store_returns_none() {
        let store = MetadataStore::new();
        assert!(store
            .estimate_selectivity(&FilterPredicate::equals("x", 1))
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
            .estimate_selectivity(&FilterPredicate::equals("x", 1))
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
            .estimate_selectivity(&FilterPredicate::equals("x", 1))
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
            .estimate_selectivity(&FilterPredicate::equals("x", 99))
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

    // --- fusion ---

    #[test]
    fn augment_embedding_appends_one_hot() {
        let emb = vec![0.1, 0.2, 0.3];
        let aug = fusion::augment_embedding(&emb, 1, 3, 0.5).unwrap();
        assert_eq!(aug.len(), 6);
        assert_eq!(&aug[..3], &[0.1, 0.2, 0.3]);
        assert_eq!(&aug[3..], &[0.0, 0.5, 0.0]);
    }

    #[test]
    fn augment_embedding_out_of_range_category() {
        let emb = vec![0.1];
        assert!(fusion::augment_embedding(&emb, 5, 3, 1.0).is_err());
    }

    #[test]
    fn extract_original_strips_augmentation() {
        let aug = vec![0.1, 0.2, 0.3, 0.0, 1.0, 0.0];
        assert_eq!(fusion::extract_original(&aug, 3), vec![0.1, 0.2, 0.3]);
    }
}

pub mod fusion {
    use super::*;

    pub fn augment_embedding(
        embedding: &[f32],
        category_id: u32,
        num_categories: usize,
        weight: f32,
    ) -> Result<Vec<f32>, RetrieveError> {
        if category_id as usize >= num_categories {
            return Err(RetrieveError::InvalidParameter(format!(
                "Category ID {} >= num_categories {}",
                category_id, num_categories
            )));
        }

        let mut augmented = Vec::with_capacity(embedding.len() + num_categories);
        augmented.extend_from_slice(embedding);

        for i in 0..num_categories {
            if i == category_id as usize {
                augmented.push(weight);
            } else {
                augmented.push(0.0);
            }
        }

        Ok(augmented)
    }

    pub fn augment_query(
        query: &[f32],
        desired_category: u32,
        num_categories: usize,
        weight: f32,
    ) -> Result<Vec<f32>, RetrieveError> {
        augment_embedding(query, desired_category, num_categories, weight)
    }

    pub fn extract_original(augmented: &[f32], original_dim: usize) -> Vec<f32> {
        augmented[..original_dim].to_vec()
    }
}
