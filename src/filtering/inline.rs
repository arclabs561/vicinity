//! Inline filtering strategies for vector search.

use super::{FilterPredicate, MetadataStore};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterStrategy {
    PreFilter,
    PostFilter,
    Inline,
}

#[derive(Clone, Debug)]
pub struct InlineFilterConfig {
    pub post_filter_oversearch: f32,
    pub inline_max_candidates: usize,
    pub pre_filter_threshold: f32,
    pub post_filter_threshold: f32,
}

impl Default for InlineFilterConfig {
    fn default() -> Self {
        Self {
            post_filter_oversearch: 4.0,
            inline_max_candidates: 10_000,
            pre_filter_threshold: 0.05,
            post_filter_threshold: 0.5,
        }
    }
}

pub struct FilterStrategySelector {
    config: InlineFilterConfig,
}

impl FilterStrategySelector {
    pub fn new() -> Self {
        Self {
            config: InlineFilterConfig::default(),
        }
    }

    pub fn with_config(config: InlineFilterConfig) -> Self {
        Self { config }
    }

    pub fn select(
        &self,
        estimated_selectivity: f32,
        k: usize,
        total_docs: usize,
    ) -> FilterStrategy {
        if estimated_selectivity < self.config.pre_filter_threshold {
            return FilterStrategy::PreFilter;
        }
        if estimated_selectivity > self.config.post_filter_threshold {
            return FilterStrategy::PostFilter;
        }

        let expected_matches = (total_docs as f32 * estimated_selectivity) as usize;
        if k * 10 < expected_matches {
            return FilterStrategy::PostFilter;
        }

        FilterStrategy::Inline
    }

    pub fn estimate_selectivity(
        &self,
        filter: &FilterPredicate,
        metadata: &MetadataStore,
        total_docs: usize,
    ) -> f32 {
        if total_docs == 0 {
            return 1.0;
        }
        estimate_predicate_selectivity(filter, metadata, total_docs)
    }
}

impl Default for FilterStrategySelector {
    fn default() -> Self {
        Self::new()
    }
}

fn estimate_predicate_selectivity(
    predicate: &FilterPredicate,
    metadata: &MetadataStore,
    total_docs: usize,
) -> f32 {
    match predicate {
        FilterPredicate::Equals { field, value } => {
            let counts = metadata.get_value_counts(field);
            let matching = counts
                .iter()
                .find(|(v, _)| *v == *value)
                .map(|(_, count)| *count)
                .unwrap_or(0);
            matching as f32 / total_docs as f32
        }
        FilterPredicate::And(predicates) => predicates.iter().fold(1.0, |acc, p| {
            acc * estimate_predicate_selectivity(p, metadata, total_docs)
        }),
        FilterPredicate::Or(predicates) => {
            if predicates.is_empty() {
                return 0.0;
            }
            let non_selectivity = predicates.iter().fold(1.0, |acc, p| {
                acc * (1.0 - estimate_predicate_selectivity(p, metadata, total_docs))
            });
            1.0 - non_selectivity
        }
    }
}

pub struct InlineFilter<'a> {
    predicate: &'a FilterPredicate,
    metadata: &'a MetadataStore,
    evaluated: usize,
    passed: usize,
}

impl<'a> InlineFilter<'a> {
    pub fn new(predicate: &'a FilterPredicate, metadata: &'a MetadataStore) -> Self {
        Self {
            predicate,
            metadata,
            evaluated: 0,
            passed: 0,
        }
    }

    pub fn matches(&mut self, doc_id: u32) -> bool {
        self.evaluated += 1;
        let result = self.metadata.matches(doc_id, self.predicate);
        if result {
            self.passed += 1;
        }
        result
    }

    pub fn observed_selectivity(&self) -> f32 {
        if self.evaluated == 0 {
            1.0
        } else {
            self.passed as f32 / self.evaluated as f32
        }
    }

    pub fn stats(&self) -> InlineFilterStats {
        InlineFilterStats {
            evaluated: self.evaluated,
            passed: self.passed,
            selectivity: self.observed_selectivity(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct InlineFilterStats {
    pub evaluated: usize,
    pub passed: usize,
    pub selectivity: f32,
}

pub fn post_filter_results(
    results: Vec<(u32, f32)>,
    predicate: &FilterPredicate,
    metadata: &MetadataStore,
    k: usize,
) -> Vec<(u32, f32)> {
    results
        .into_iter()
        .filter(|(id, _)| metadata.matches(*id, predicate))
        .take(k)
        .collect()
}

pub fn calculate_oversearch_k(
    k: usize,
    estimated_selectivity: f32,
    oversearch_factor: f32,
) -> usize {
    if estimated_selectivity <= 0.0 {
        return k * 100;
    }
    let base_oversearch = (k as f32 / estimated_selectivity).ceil() as usize;
    let with_factor = (base_oversearch as f32 * oversearch_factor).ceil() as usize;
    with_factor.max(k).min(k * 100)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_store() -> MetadataStore {
        let mut store = MetadataStore::new();
        for i in 0..100 {
            let mut m = HashMap::new();
            m.insert("color".to_string(), i % 5); // 5 categories, 20 each
            m.insert("size".to_string(), i % 2); // 2 categories, 50 each
            store.add(i, m);
        }
        store
    }

    // --- FilterStrategySelector ---

    #[test]
    fn selector_low_selectivity_prefilter() {
        let sel = FilterStrategySelector::new();
        // selectivity 0.01 < pre_filter_threshold 0.05
        assert_eq!(sel.select(0.01, 10, 10000), FilterStrategy::PreFilter);
    }

    #[test]
    fn selector_high_selectivity_postfilter() {
        let sel = FilterStrategySelector::new();
        // selectivity 0.8 > post_filter_threshold 0.5
        assert_eq!(sel.select(0.8, 10, 10000), FilterStrategy::PostFilter);
    }

    #[test]
    fn selector_mid_selectivity_inline() {
        let sel = FilterStrategySelector::new();
        // selectivity 0.1: between thresholds, k*10=100 not < expected_matches=1000
        // Actually k*10=100 < 1000, so PostFilter
        // Let's pick k=200, expected=1000, k*10=2000 >= 1000 -> Inline
        assert_eq!(sel.select(0.1, 200, 10000), FilterStrategy::Inline);
    }

    #[test]
    fn selector_estimate_selectivity_zero_docs() {
        let sel = FilterStrategySelector::new();
        let store = MetadataStore::new();
        let pred = FilterPredicate::equals("x", 1);
        let s = sel.estimate_selectivity(&pred, &store, 0);
        assert!((s - 1.0).abs() < 1e-6, "zero docs -> selectivity 1.0");
    }

    #[test]
    fn selector_estimate_selectivity_equals() {
        let sel = FilterStrategySelector::new();
        let store = make_store();
        // color=0 matches 20 out of 100
        let pred = FilterPredicate::equals("color", 0);
        let s = sel.estimate_selectivity(&pred, &store, 100);
        assert!(
            (s - 0.2).abs() < 1e-6,
            "expected selectivity ~0.2, got {}",
            s
        );
    }

    #[test]
    fn selector_estimate_selectivity_and() {
        let sel = FilterStrategySelector::new();
        let store = make_store();
        // AND(color=0, size=0): independence assumption -> 0.2 * 0.5 = 0.1
        let pred = FilterPredicate::And(vec![
            FilterPredicate::equals("color", 0),
            FilterPredicate::equals("size", 0),
        ]);
        let s = sel.estimate_selectivity(&pred, &store, 100);
        assert!(
            (s - 0.1).abs() < 1e-6,
            "expected AND selectivity ~0.1, got {}",
            s
        );
    }

    #[test]
    fn selector_estimate_selectivity_or() {
        let sel = FilterStrategySelector::new();
        let store = make_store();
        // OR(color=0, color=1): P(A or B) = 1 - (1-0.2)*(1-0.2) = 0.36
        let pred = FilterPredicate::Or(vec![
            FilterPredicate::equals("color", 0),
            FilterPredicate::equals("color", 1),
        ]);
        let s = sel.estimate_selectivity(&pred, &store, 100);
        assert!(
            (s - 0.36).abs() < 1e-6,
            "expected OR selectivity ~0.36, got {}",
            s
        );
    }

    #[test]
    fn selector_estimate_selectivity_or_empty() {
        let sel = FilterStrategySelector::new();
        let store = make_store();
        let pred = FilterPredicate::Or(vec![]);
        let s = sel.estimate_selectivity(&pred, &store, 100);
        assert!((s - 0.0).abs() < 1e-6);
    }

    // --- InlineFilter ---

    #[test]
    fn inline_filter_tracks_selectivity() {
        let store = make_store();
        let pred = FilterPredicate::equals("color", 0);
        let mut filter = InlineFilter::new(&pred, &store);

        // Evaluate all 100 docs
        let mut passed_count = 0;
        for i in 0..100 {
            if filter.matches(i) {
                passed_count += 1;
            }
        }

        let stats = filter.stats();
        assert_eq!(stats.evaluated, 100);
        assert_eq!(stats.passed, 20);
        assert_eq!(passed_count, 20);
        assert!((stats.selectivity - 0.2).abs() < 1e-6);
    }

    #[test]
    fn inline_filter_no_evaluations_selectivity_one() {
        let store = MetadataStore::new();
        let pred = FilterPredicate::equals("x", 1);
        let filter = InlineFilter::new(&pred, &store);
        assert!((filter.observed_selectivity() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn inline_filter_missing_doc_does_not_match() {
        let store = MetadataStore::new();
        let pred = FilterPredicate::equals("x", 1);
        let mut filter = InlineFilter::new(&pred, &store);
        assert!(!filter.matches(999));
        assert_eq!(filter.stats().passed, 0);
        assert_eq!(filter.stats().evaluated, 1);
    }

    // --- post_filter_results ---

    #[test]
    fn post_filter_keeps_matching_results() {
        let store = make_store();
        let pred = FilterPredicate::equals("color", 0);
        let results: Vec<(u32, f32)> = (0..20).map(|i| (i, i as f32 * 0.1)).collect();
        let filtered = post_filter_results(results, &pred, &store, 5);
        // Only ids where color=0 (i.e. i%5==0: 0, 5, 10, 15) pass, take 5
        assert!(filtered.len() <= 5);
        for (id, _) in &filtered {
            assert_eq!(id % 5, 0, "doc {} should have color=0", id);
        }
    }

    // --- calculate_oversearch_k ---

    #[test]
    fn oversearch_k_zero_selectivity() {
        assert_eq!(calculate_oversearch_k(10, 0.0, 2.0), 1000);
    }

    #[test]
    fn oversearch_k_full_selectivity() {
        let result = calculate_oversearch_k(10, 1.0, 1.0);
        assert_eq!(result, 10);
    }

    #[test]
    fn oversearch_k_bounded_by_max() {
        // Very low selectivity -> capped at k*100
        let result = calculate_oversearch_k(10, 0.001, 4.0);
        assert_eq!(result, 1000);
    }

    // --- InlineFilterConfig ---

    #[test]
    fn default_config_values() {
        let cfg = InlineFilterConfig::default();
        assert!((cfg.post_filter_oversearch - 4.0).abs() < 1e-6);
        assert_eq!(cfg.inline_max_candidates, 10_000);
        assert!((cfg.pre_filter_threshold - 0.05).abs() < 1e-6);
        assert!((cfg.post_filter_threshold - 0.5).abs() < 1e-6);
    }

    #[test]
    fn custom_config_thresholds() {
        let cfg = InlineFilterConfig {
            pre_filter_threshold: 0.1,
            post_filter_threshold: 0.9,
            ..Default::default()
        };
        let sel = FilterStrategySelector::with_config(cfg);
        // 0.2 is between 0.1 and 0.9, large k -> inline
        assert_eq!(sel.select(0.2, 500, 10000), FilterStrategy::Inline);
    }
}
