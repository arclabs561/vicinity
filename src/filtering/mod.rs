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
