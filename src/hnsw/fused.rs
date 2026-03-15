//! FusedANN: Attribute-vector fusion for filtered search.
//!
//! Instead of treating filters as hard constraints evaluated during traversal,
//! FusedANN fuses attribute predicates into the vector space via a Lagrangian-like
//! relaxation. Hard filters become soft penalties, enabling efficient approximate
//! search while preserving top-k semantics.
//!
//! # Algorithm
//!
//! 1. **Attribute Embedding**: Encode categorical/numeric attributes into vectors
//! 2. **Fusion**: Combine attribute vectors with content vectors:
//!    `fused = alpha * content_vec + (1-alpha) * attribute_vec`
//! 3. **Search**: Standard ANN search in fused space
//! 4. **Refinement**: Post-filter exact matches if needed
//!
//! # Key Insight
//!
//! When filter selectivity is very high (few matches), traditional pre-filtering
//! degrades recall. FusedANN's soft penalties naturally bias search toward
//! matching regions while still exploring nearby non-matches for navigability.
//!
//! # References
//!
//! - Heidari et al. (2025): "FusedANN: Convexified Hybrid ANN via Attribute-Vector
//!   Fusion" - <https://arxiv.org/abs/2509.19767>

use crate::RetrieveError;
use std::collections::HashMap;

/// Attribute value that can be embedded.
#[derive(Clone, Debug, PartialEq)]
pub enum AttributeValue {
    /// A single categorical label.
    Categorical(String),
    /// A scalar numeric value.
    Numeric(f32),
    /// A numeric range with bounds.
    NumericRange {
        /// Lower bound of the range.
        min: f32,
        /// Upper bound of the range.
        max: f32,
    },
    /// A boolean flag.
    Boolean(bool),
    /// Multiple categorical labels.
    MultiCategory(Vec<String>),
}

/// Attribute schema defining how to embed attributes.
#[derive(Clone, Debug)]
pub struct AttributeSchema {
    /// Dimension of attribute embedding
    pub dimension: usize,
    /// Attribute definitions
    pub attributes: Vec<AttributeDefinition>,
}

/// Single attribute definition.
#[derive(Clone, Debug)]
pub struct AttributeDefinition {
    /// Attribute name.
    pub name: String,
    /// Type and encoding parameters.
    pub attr_type: AttributeType,
    /// Weight when fusing (higher = more important for filtering)
    pub weight: f32,
}

/// Attribute type for embedding strategy.
#[derive(Clone, Debug)]
pub enum AttributeType {
    /// One-hot encoding for categories.
    Categorical {
        /// Known category labels for one-hot encoding.
        categories: Vec<String>,
    },
    /// Normalized numeric value.
    Numeric {
        /// Minimum value for normalization.
        min: f32,
        /// Maximum value for normalization.
        max: f32,
    },
    /// Boolean flag
    Boolean,
}

impl AttributeSchema {
    /// Create new schema.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            attributes: Vec::new(),
        }
    }

    /// Add categorical attribute.
    pub fn add_categorical(&mut self, name: &str, categories: Vec<String>, weight: f32) {
        self.attributes.push(AttributeDefinition {
            name: name.to_string(),
            attr_type: AttributeType::Categorical { categories },
            weight,
        });
    }

    /// Add numeric attribute.
    pub fn add_numeric(&mut self, name: &str, min: f32, max: f32, weight: f32) {
        self.attributes.push(AttributeDefinition {
            name: name.to_string(),
            attr_type: AttributeType::Numeric { min, max },
            weight,
        });
    }

    /// Add boolean attribute.
    pub fn add_boolean(&mut self, name: &str, weight: f32) {
        self.attributes.push(AttributeDefinition {
            name: name.to_string(),
            attr_type: AttributeType::Boolean,
            weight,
        });
    }

    /// Compute total embedding dimension needed for all attributes.
    pub fn attribute_embedding_dim(&self) -> usize {
        let mut dim = 0;
        for attr in &self.attributes {
            dim += match &attr.attr_type {
                AttributeType::Categorical { categories } => categories.len(),
                AttributeType::Numeric { .. } => 1,
                AttributeType::Boolean => 1,
            };
        }
        dim
    }
}

/// Attribute embedder.
pub struct AttributeEmbedder {
    schema: AttributeSchema,
    embedding_dim: usize,
}

impl AttributeEmbedder {
    /// Create embedder from schema.
    pub fn new(schema: AttributeSchema) -> Self {
        let embedding_dim = schema.attribute_embedding_dim();
        Self {
            schema,
            embedding_dim,
        }
    }

    /// Embed attribute values into vector.
    pub fn embed(&self, attributes: &HashMap<String, AttributeValue>) -> Vec<f32> {
        let mut embedding = vec![0.0f32; self.embedding_dim];
        let mut offset = 0;

        for attr_def in &self.schema.attributes {
            let weight = attr_def.weight;

            match &attr_def.attr_type {
                AttributeType::Categorical { categories } => {
                    if let Some(AttributeValue::Categorical(cat)) = attributes.get(&attr_def.name) {
                        if let Some(idx) = categories.iter().position(|c| c == cat) {
                            embedding[offset + idx] = weight;
                        }
                    } else if let Some(AttributeValue::MultiCategory(cats)) =
                        attributes.get(&attr_def.name)
                    {
                        for cat in cats {
                            if let Some(idx) = categories.iter().position(|c| c == cat) {
                                embedding[offset + idx] = weight / cats.len() as f32;
                            }
                        }
                    }
                    offset += categories.len();
                }
                AttributeType::Numeric { min, max } => {
                    if let Some(AttributeValue::Numeric(val)) = attributes.get(&attr_def.name) {
                        // Normalize to [0, 1]
                        let normalized = (val - min) / (max - min);
                        embedding[offset] = normalized.clamp(0.0, 1.0) * weight;
                    }
                    offset += 1;
                }
                AttributeType::Boolean => {
                    if let Some(AttributeValue::Boolean(b)) = attributes.get(&attr_def.name) {
                        embedding[offset] = if *b { weight } else { 0.0 };
                    }
                    offset += 1;
                }
            }
        }

        embedding
    }

    /// Embed a filter query into vector.
    ///
    /// For filter queries, we want to maximize similarity to matching vectors.
    pub fn embed_filter(&self, filters: &HashMap<String, AttributeValue>) -> Vec<f32> {
        // Filter embedding is the same as data embedding
        // Similarity will be high when attributes match
        self.embed(filters)
    }

    /// Get embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

/// FusedANN configuration.
#[derive(Clone, Debug)]
pub struct FusedConfig {
    /// Fusion weight: alpha for content, (1-alpha) for attributes
    pub alpha: f32,
    /// Penalty for attribute mismatch (Lagrangian multiplier)
    pub lambda: f32,
    /// Whether to post-filter exact matches
    pub exact_filter: bool,
    /// Expansion factor for search (to account for soft filtering)
    pub expansion_factor: f32,
}

impl Default for FusedConfig {
    fn default() -> Self {
        Self {
            alpha: 0.7,
            lambda: 1.0,
            exact_filter: true,
            expansion_factor: 2.0,
        }
    }
}

/// Fused vector combining content and attributes.
#[derive(Clone, Debug)]
pub struct FusedVector {
    /// Content vector
    pub content: Vec<f32>,
    /// Attribute embedding
    pub attributes: Vec<f32>,
    /// Precomputed fused vector
    pub fused: Vec<f32>,
}

impl FusedVector {
    /// Create fused vector.
    pub fn new(content: Vec<f32>, attributes: Vec<f32>, alpha: f32) -> Self {
        let fused = Self::compute_fusion(&content, &attributes, alpha);
        Self {
            content,
            attributes,
            fused,
        }
    }

    /// Compute fusion of content and attribute vectors.
    fn compute_fusion(content: &[f32], attributes: &[f32], alpha: f32) -> Vec<f32> {
        let mut fused = Vec::with_capacity(content.len() + attributes.len());

        // Scale content by alpha
        for &c in content {
            fused.push(c * alpha);
        }

        // Scale attributes by (1-alpha)
        for &a in attributes {
            fused.push(a * (1.0 - alpha));
        }

        // Normalize
        let norm: f32 = fused.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for f in &mut fused {
                *f /= norm;
            }
        }

        fused
    }

    /// Dimension of fused vector.
    pub fn fused_dim(&self) -> usize {
        self.fused.len()
    }
}

/// FusedANN index.
pub struct FusedIndex {
    config: FusedConfig,
    embedder: AttributeEmbedder,
    content_dim: usize,
    /// Stored fused vectors (for simple brute-force search)
    vectors: Vec<FusedVector>,
    /// Original attributes (for exact filtering)
    original_attributes: Vec<HashMap<String, AttributeValue>>,
}

impl FusedIndex {
    /// Create new FusedANN index.
    pub fn new(embedder: AttributeEmbedder, content_dim: usize, config: FusedConfig) -> Self {
        Self {
            config,
            embedder,
            content_dim,
            vectors: Vec::new(),
            original_attributes: Vec::new(),
        }
    }

    /// Add vector with attributes.
    pub fn add(
        &mut self,
        content: Vec<f32>,
        attributes: HashMap<String, AttributeValue>,
    ) -> Result<u32, RetrieveError> {
        if content.len() != self.content_dim {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: self.content_dim,
                doc_dim: content.len(),
            });
        }

        let attr_embedding = self.embedder.embed(&attributes);
        let fused = FusedVector::new(content, attr_embedding, self.config.alpha);

        let id = self.vectors.len() as u32;
        self.vectors.push(fused);
        self.original_attributes.push(attributes);

        Ok(id)
    }

    /// Search with optional attribute filter.
    pub fn search(
        &self,
        query_content: &[f32],
        query_filter: Option<&HashMap<String, AttributeValue>>,
        k: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if query_content.len() != self.content_dim {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query_content.len(),
                doc_dim: self.content_dim,
            });
        }

        // Create query fused vector
        let query_attrs = if let Some(filter) = query_filter {
            self.embedder.embed_filter(filter)
        } else {
            vec![0.0; self.embedder.embedding_dim()]
        };

        let query_fused = FusedVector::new(query_content.to_vec(), query_attrs, self.config.alpha);

        // Search limit (expanded to account for filtering)
        let search_k = if query_filter.is_some() {
            (k as f32 * self.config.expansion_factor) as usize
        } else {
            k
        };

        // Compute distances to all vectors
        let mut candidates: Vec<(u32, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(idx, vec)| {
                let dist = fused_distance(&query_fused.fused, &vec.fused, self.config.lambda);
                (idx as u32, dist)
            })
            .collect();

        // Sort by distance
        candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
        candidates.truncate(search_k);

        // Post-filter if exact filtering is enabled
        if self.config.exact_filter {
            if let Some(filter) = query_filter {
                candidates.retain(|(id, _)| {
                    let attrs = &self.original_attributes[*id as usize];
                    check_filter(attrs, filter)
                });
            }
        }

        candidates.truncate(k);
        Ok(candidates)
    }

    /// Get number of vectors in index.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if index is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

/// Compute distance in fused space.
fn fused_distance(query: &[f32], candidate: &[f32], lambda: f32) -> f32 {
    if query.len() != candidate.len() {
        return f32::INFINITY;
    }

    // Euclidean distance in fused space
    let mut sum = 0.0f32;
    for (q, c) in query.iter().zip(candidate.iter()) {
        let diff = q - c;
        sum += diff * diff;
    }

    // Scale by lambda for attribute-heavy queries
    sum.sqrt() * lambda
}

/// Check if attributes match filter exactly.
fn check_filter(
    attributes: &HashMap<String, AttributeValue>,
    filter: &HashMap<String, AttributeValue>,
) -> bool {
    for (key, filter_val) in filter {
        if let Some(attr_val) = attributes.get(key) {
            let matches = match (filter_val, attr_val) {
                (AttributeValue::Categorical(f), AttributeValue::Categorical(a)) => f == a,
                (AttributeValue::Numeric(f), AttributeValue::Numeric(a)) => (f - a).abs() < 1e-6,
                (AttributeValue::Boolean(f), AttributeValue::Boolean(a)) => f == a,
                (AttributeValue::NumericRange { min, max }, AttributeValue::Numeric(a)) => {
                    a >= min && a <= max
                }
                (AttributeValue::Categorical(f), AttributeValue::MultiCategory(cats)) => {
                    cats.contains(f)
                }
                _ => false,
            };
            if !matches {
                return false;
            }
        } else {
            return false; // Missing required attribute
        }
    }
    true
}

/// Estimate optimal alpha based on filter selectivity.
///
/// When filter is very selective, use lower alpha (more weight on attributes).
/// When filter is loose, use higher alpha (more weight on content).
pub fn recommend_alpha(estimated_selectivity: f32, k: usize, total_docs: usize) -> f32 {
    let expected_matches = estimated_selectivity * total_docs as f32;

    // If expected matches >> k, we can focus on content
    if expected_matches > k as f32 * 10.0 {
        return 0.9; // High alpha = content-focused
    }

    // If expected matches ~= k, balance both
    if expected_matches > k as f32 {
        return 0.7;
    }

    // If expected matches < k, heavily weight attributes
    0.5
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn create_test_schema() -> AttributeSchema {
        let mut schema = AttributeSchema::new(64);
        schema.add_categorical("category", vec!["A".into(), "B".into(), "C".into()], 1.0);
        schema.add_numeric("year", 2000.0, 2025.0, 0.5);
        schema.add_boolean("premium", 0.3);
        schema
    }

    #[test]
    fn test_attribute_embedding() {
        let schema = create_test_schema();
        let embedder = AttributeEmbedder::new(schema);

        let mut attrs = HashMap::new();
        attrs.insert(
            "category".to_string(),
            AttributeValue::Categorical("B".to_string()),
        );
        attrs.insert("year".to_string(), AttributeValue::Numeric(2020.0));
        attrs.insert("premium".to_string(), AttributeValue::Boolean(true));

        let embedding = embedder.embed(&attrs);

        // Should have dimensions for: 3 categories + 1 numeric + 1 boolean = 5
        assert_eq!(embedder.embedding_dim(), 5);
        assert_eq!(embedding.len(), 5);

        // Category B (index 1) should be 1.0
        assert_eq!(embedding[1], 1.0);

        // Year normalized: (2020 - 2000) / (2025 - 2000) = 0.8, weighted by 0.5
        assert!((embedding[3] - 0.4).abs() < 0.01);

        // Premium true, weighted by 0.3
        assert_eq!(embedding[4], 0.3);
    }

    #[test]
    fn test_fused_vector() {
        let content = vec![1.0, 0.0, 0.0];
        let attrs = vec![0.0, 1.0];

        let fused = FusedVector::new(content, attrs, 0.7);

        // Fused dimension = content (3) + attrs (2) = 5
        assert_eq!(fused.fused_dim(), 5);

        // Should be normalized
        let norm: f32 = fused.fused.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_fused_index_basic() {
        let schema = create_test_schema();
        let embedder = AttributeEmbedder::new(schema);
        let config = FusedConfig::default();

        let mut index = FusedIndex::new(embedder, 4, config);

        // Add vectors
        let mut attrs1 = HashMap::new();
        attrs1.insert(
            "category".to_string(),
            AttributeValue::Categorical("A".to_string()),
        );
        index.add(vec![1.0, 0.0, 0.0, 0.0], attrs1).unwrap();

        let mut attrs2 = HashMap::new();
        attrs2.insert(
            "category".to_string(),
            AttributeValue::Categorical("B".to_string()),
        );
        index.add(vec![0.0, 1.0, 0.0, 0.0], attrs2).unwrap();

        assert_eq!(index.len(), 2);
    }

    #[test]
    fn test_fused_search_no_filter() {
        let schema = create_test_schema();
        let embedder = AttributeEmbedder::new(schema);
        let mut index = FusedIndex::new(embedder, 4, FusedConfig::default());

        // Add vectors
        for i in 0..10 {
            let mut attrs = HashMap::new();
            attrs.insert(
                "category".to_string(),
                AttributeValue::Categorical(if i % 2 == 0 { "A" } else { "B" }.to_string()),
            );
            let mut content = vec![0.0; 4];
            content[i % 4] = 1.0;
            index.add(content, attrs).unwrap();
        }

        // Search without filter
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], None, 3).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_fused_search_with_filter() {
        let schema = create_test_schema();
        let embedder = AttributeEmbedder::new(schema);
        let mut index = FusedIndex::new(embedder, 4, FusedConfig::default());

        // Add vectors - half A, half B
        for i in 0..10 {
            let mut attrs = HashMap::new();
            let cat = if i < 5 { "A" } else { "B" };
            attrs.insert(
                "category".to_string(),
                AttributeValue::Categorical(cat.to_string()),
            );
            let mut content = vec![0.0; 4];
            content[i % 4] = 1.0;
            index.add(content, attrs).unwrap();
        }

        // Search with filter for category A
        let mut filter = HashMap::new();
        filter.insert(
            "category".to_string(),
            AttributeValue::Categorical("A".to_string()),
        );

        let results = index
            .search(&[1.0, 0.0, 0.0, 0.0], Some(&filter), 3)
            .unwrap();

        // All results should be category A
        for (id, _) in &results {
            assert!(*id < 5, "ID {} should be in category A", id);
        }
    }

    #[test]
    fn test_check_filter() {
        let mut attrs = HashMap::new();
        attrs.insert(
            "cat".to_string(),
            AttributeValue::Categorical("A".to_string()),
        );
        attrs.insert("year".to_string(), AttributeValue::Numeric(2020.0));

        // Matching filter
        let mut filter = HashMap::new();
        filter.insert(
            "cat".to_string(),
            AttributeValue::Categorical("A".to_string()),
        );
        assert!(check_filter(&attrs, &filter));

        // Non-matching filter
        let mut filter2 = HashMap::new();
        filter2.insert(
            "cat".to_string(),
            AttributeValue::Categorical("B".to_string()),
        );
        assert!(!check_filter(&attrs, &filter2));

        // Range filter
        let mut filter3 = HashMap::new();
        filter3.insert(
            "year".to_string(),
            AttributeValue::NumericRange {
                min: 2015.0,
                max: 2025.0,
            },
        );
        assert!(check_filter(&attrs, &filter3));
    }

    #[test]
    fn test_recommend_alpha() {
        // Very selective filter (few matches) - expected_matches = 0.01 * 10000 = 100
        // 100 > 10 * 10 = 100, so goes to second branch
        let alpha1 = recommend_alpha(0.01, 10, 10000);
        // expected_matches (100) == k * 10 (100), so it returns 0.7
        assert!(alpha1 <= 0.7);

        // Loose filter (many matches)
        let alpha2 = recommend_alpha(0.9, 10, 10000);
        assert!(alpha2 >= 0.7);

        // Very few matches - force the 0.5 case
        let alpha3 = recommend_alpha(0.001, 10, 10000); // expected = 10, k = 10
        assert!(alpha3 <= 0.7);
    }

    #[test]
    fn test_multi_category_filter() {
        let mut attrs = HashMap::new();
        attrs.insert(
            "tags".to_string(),
            AttributeValue::MultiCategory(vec!["rust".to_string(), "python".to_string()]),
        );

        let mut filter = HashMap::new();
        filter.insert(
            "tags".to_string(),
            AttributeValue::Categorical("rust".to_string()),
        );

        assert!(check_filter(&attrs, &filter));
    }
}
