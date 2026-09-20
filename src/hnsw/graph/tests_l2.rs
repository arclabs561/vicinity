use super::*;

#[test]
fn auto_normalize_preserves_magnitude_for_l2_and_inner_product() {
    for metric in [DistanceMetric::L2, DistanceMetric::InnerProduct] {
        let mut index = HNSWIndex::builder(2)
            .metric(metric)
            .auto_normalize(true)
            .build()
            .unwrap();
        let vectors = [(7, [10.0, 0.0]), (8, [1.0, 1.0]), (9, [0.0, 3.0])];
        for (id, vector) in &vectors {
            index.add_slice(*id, vector).unwrap();
        }
        index.build().unwrap();
        let query = [9.0, 0.0];
        let results = index.search(&query, 3, 10).unwrap();
        let mut expected: Vec<_> = vectors
            .iter()
            .map(|(id, vector)| (*id, metric.distance(&query, vector)))
            .collect();
        expected.sort_by(|a, b| a.1.total_cmp(&b.1));
        assert_eq!(results.len(), expected.len());
        for (actual, expected) in results.iter().zip(&expected) {
            assert_eq!(actual.0, expected.0, "metric: {metric:?}");
            assert!(
                (actual.1 - expected.1).abs() < 1e-5,
                "{metric:?}: {actual:?} != {expected:?}"
            );
        }
    }
}

#[test]
fn test_hnsw_l2_distance() {
    let mut index = HNSWIndex::builder(4)
        .metric(DistanceMetric::L2)
        .build()
        .unwrap();
    index.add_slice(0, &[1.0, 0.0, 0.0, 0.0]).unwrap();
    index.add_slice(1, &[0.0, 1.0, 0.0, 0.0]).unwrap();
    index.add_slice(2, &[100.0, 100.0, 0.0, 0.0]).unwrap();
    index.build().unwrap();
    let results = index.search(&[1.0, 0.1, 0.0, 0.0], 2, 10).unwrap();
    assert_eq!(results[0].0, 0, "closest to [1,0,0,0] should be doc 0");
}
