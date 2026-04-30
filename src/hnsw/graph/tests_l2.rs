use super::*;

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
