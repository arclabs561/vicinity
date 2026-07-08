//! Filtering and lightweight faceting support for vector search.
//!
//! Filtered ANN traversal is an index concern (it changes how graph search
//! proceeds), not a pipeline concern.

use std::collections::HashMap;

/// A typed metadata value.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    any(feature = "serde", feature = "ivf_pq"),
    derive(serde::Deserialize, serde::Serialize)
)]
pub enum MetadataValue {
    /// 64-bit integer.
    Int(i64),
    /// 64-bit float.
    Float(f64),
    /// UTF-8 string.
    Str(String),
    /// Boolean.
    Bool(bool),
}

impl Eq for MetadataValue {}

// Hash via bit-pattern for floats so MetadataValue can be used as a HashMap key.
// NaN values hash equal to each other (same bit pattern = same hash),
// which is acceptable for the category-filtering use case.
impl std::hash::Hash for MetadataValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Int(v) => v.hash(state),
            Self::Float(v) => v.to_bits().hash(state),
            Self::Str(v) => v.hash(state),
            Self::Bool(v) => v.hash(state),
        }
    }
}

impl PartialOrd for MetadataValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => a.partial_cmp(b),
            (Self::Float(a), Self::Float(b)) => a.partial_cmp(b),
            (Self::Str(a), Self::Str(b)) => a.partial_cmp(b),
            (Self::Bool(a), Self::Bool(b)) => a.partial_cmp(b),
            // Cross-type comparisons are not ordered.
            _ => None,
        }
    }
}

impl From<i64> for MetadataValue {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}

impl From<i32> for MetadataValue {
    fn from(v: i32) -> Self {
        Self::Int(v as i64)
    }
}

impl From<u32> for MetadataValue {
    fn from(v: u32) -> Self {
        Self::Int(v as i64)
    }
}

impl From<f64> for MetadataValue {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}

impl From<f32> for MetadataValue {
    fn from(v: f32) -> Self {
        Self::Float(v as f64)
    }
}

impl From<String> for MetadataValue {
    fn from(v: String) -> Self {
        Self::Str(v)
    }
}

impl From<&str> for MetadataValue {
    fn from(v: &str) -> Self {
        Self::Str(v.to_string())
    }
}

impl From<bool> for MetadataValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

/// Filter predicate for metadata-based filtering.
#[derive(Clone, Debug)]
pub enum MetadataFilter {
    /// Equality filter: field must equal value.
    Equals {
        /// Metadata field name to match on.
        field: String,
        /// Required value for the field.
        value: MetadataValue,
    },
    /// Range filter: field value must fall within [min, max] (both inclusive, both optional).
    Range {
        /// Metadata field name to match on.
        field: String,
        /// Lower bound (inclusive). `None` means no lower bound.
        min: Option<MetadataValue>,
        /// Upper bound (inclusive). `None` means no upper bound.
        max: Option<MetadataValue>,
    },
    /// All child predicates must match (AND logic).
    And(Vec<MetadataFilter>),
    /// At least one child predicate must match (OR logic).
    Or(Vec<MetadataFilter>),
}

impl MetadataFilter {
    /// Create an equality filter for a field and value.
    pub fn equals(field: impl Into<String>, value: impl Into<MetadataValue>) -> Self {
        Self::Equals {
            field: field.into(),
            value: value.into(),
        }
    }

    /// Create a range filter for a field. Both bounds are inclusive and optional.
    pub fn range(
        field: impl Into<String>,
        min: impl Into<Option<MetadataValue>>,
        max: impl Into<Option<MetadataValue>>,
    ) -> Self {
        Self::Range {
            field: field.into(),
            min: min.into(),
            max: max.into(),
        }
    }

    /// Check whether the given document metadata satisfies this predicate.
    pub fn matches(&self, metadata: &DocumentMetadata) -> bool {
        match self {
            Self::Equals { field, value } => metadata.get(field).is_some_and(|v| v == value),
            Self::Range { field, min, max } => {
                let Some(v) = metadata.get(field) else {
                    return false;
                };
                if let Some(lo) = min {
                    if v.partial_cmp(lo)
                        .is_none_or(|o| o == std::cmp::Ordering::Less)
                    {
                        return false;
                    }
                }
                if let Some(hi) = max {
                    if v.partial_cmp(hi)
                        .is_none_or(|o| o == std::cmp::Ordering::Greater)
                    {
                        return false;
                    }
                }
                true
            }
            Self::And(predicates) => predicates.iter().all(|p| p.matches(metadata)),
            Self::Or(predicates) => predicates.iter().any(|p| p.matches(metadata)),
        }
    }
}

/// Document metadata storage: a map from field name to [`MetadataValue`].
pub type DocumentMetadata = HashMap<String, MetadataValue>;

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

    /// Iterate over all stored metadata entries.
    #[cfg(feature = "ivf_pq")]
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&u32, &DocumentMetadata)> {
        self.metadata.iter()
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
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn sample_metadata() -> DocumentMetadata {
        let mut m = DocumentMetadata::new();
        m.insert("color".to_string(), MetadataValue::Str("red".to_string()));
        m.insert("size".to_string(), MetadataValue::Int(42));
        m.insert("score".to_string(), MetadataValue::Float(0.9));
        m.insert("active".to_string(), MetadataValue::Bool(true));
        m
    }

    // --- MetadataValue From impls ---

    #[test]
    fn from_integer_types() {
        assert_eq!(MetadataValue::from(1i32), MetadataValue::Int(1));
        assert_eq!(MetadataValue::from(1u32), MetadataValue::Int(1));
        assert_eq!(MetadataValue::from(1i64), MetadataValue::Int(1));
    }

    #[test]
    fn from_float_types() {
        assert_eq!(MetadataValue::from(1.0f32), MetadataValue::Float(1.0));
        assert_eq!(MetadataValue::from(1.0f64), MetadataValue::Float(1.0));
    }

    #[test]
    fn from_str_types() {
        assert_eq!(
            MetadataValue::from("hello"),
            MetadataValue::Str("hello".to_string())
        );
        assert_eq!(
            MetadataValue::from("hello".to_string()),
            MetadataValue::Str("hello".to_string())
        );
    }

    #[test]
    fn from_bool() {
        assert_eq!(MetadataValue::from(true), MetadataValue::Bool(true));
    }

    // --- MetadataFilter::Equals ---

    #[test]
    fn equals_matches_string_field() {
        let meta = sample_metadata();
        assert!(MetadataFilter::equals("color", "red").matches(&meta));
    }

    #[test]
    fn equals_matches_int_field() {
        let meta = sample_metadata();
        assert!(MetadataFilter::equals("size", 42i64).matches(&meta));
    }

    #[test]
    fn equals_rejects_wrong_value() {
        let meta = sample_metadata();
        assert!(!MetadataFilter::equals("color", "blue").matches(&meta));
    }

    #[test]
    fn equals_rejects_missing_field() {
        let meta = sample_metadata();
        assert!(!MetadataFilter::equals("weight", 1i64).matches(&meta));
    }

    // --- MetadataFilter::Range ---

    #[test]
    fn range_within_bounds() {
        let meta = sample_metadata(); // size = Int(42)
        let pred = MetadataFilter::range(
            "size",
            Some(MetadataValue::Int(10)),
            Some(MetadataValue::Int(100)),
        );
        assert!(pred.matches(&meta));
    }

    #[test]
    fn range_at_lower_bound_inclusive() {
        let meta = sample_metadata();
        let pred = MetadataFilter::range("size", Some(MetadataValue::Int(42)), None);
        assert!(pred.matches(&meta));
    }

    #[test]
    fn range_at_upper_bound_inclusive() {
        let meta = sample_metadata();
        let pred = MetadataFilter::range("size", None, Some(MetadataValue::Int(42)));
        assert!(pred.matches(&meta));
    }

    #[test]
    fn range_below_lower_bound() {
        let meta = sample_metadata();
        let pred = MetadataFilter::range("size", Some(MetadataValue::Int(50)), None);
        assert!(!pred.matches(&meta));
    }

    #[test]
    fn range_above_upper_bound() {
        let meta = sample_metadata();
        let pred = MetadataFilter::range("size", None, Some(MetadataValue::Int(10)));
        assert!(!pred.matches(&meta));
    }

    #[test]
    fn range_missing_field_is_false() {
        let meta = sample_metadata();
        let pred = MetadataFilter::range("weight", None, None);
        assert!(!pred.matches(&meta));
    }

    #[test]
    fn range_float_within_bounds() {
        let meta = sample_metadata(); // score = Float(0.9)
        let pred = MetadataFilter::range(
            "score",
            Some(MetadataValue::Float(0.5)),
            Some(MetadataValue::Float(1.0)),
        );
        assert!(pred.matches(&meta));
    }

    // --- MetadataFilter::And / Or ---

    #[test]
    fn and_all_true() {
        let meta = sample_metadata();
        let pred = MetadataFilter::And(vec![
            MetadataFilter::equals("color", "red"),
            MetadataFilter::equals("size", 42i64),
        ]);
        assert!(pred.matches(&meta));
    }

    #[test]
    fn and_one_false() {
        let meta = sample_metadata();
        let pred = MetadataFilter::And(vec![
            MetadataFilter::equals("color", "red"),
            MetadataFilter::equals("size", 99i64),
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
            MetadataFilter::equals("color", "blue"),
            MetadataFilter::equals("size", 42i64),
        ]);
        assert!(pred.matches(&meta));
    }

    #[test]
    fn or_none_true() {
        let meta = sample_metadata();
        let pred = MetadataFilter::Or(vec![
            MetadataFilter::equals("color", "blue"),
            MetadataFilter::equals("size", 99i64),
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
        assert_eq!(
            retrieved.get("color"),
            Some(&MetadataValue::Str("red".to_string()))
        );
        assert_eq!(retrieved.get("size"), Some(&MetadataValue::Int(42)));
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
        assert!(store.matches(0, &MetadataFilter::equals("color", "red")));
        assert!(!store.matches(0, &MetadataFilter::equals("color", "blue")));
        assert!(!store.matches(999, &MetadataFilter::equals("color", "red")));
    }
}
