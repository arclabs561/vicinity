//! Find the nearest documents in a requested collection.
//!
//! Run with `cargo run --release --example filtered_neighbors --features filtered_graph`.

use std::collections::HashMap;
use vicinity::filtered_graph::{AttrValue, Filter, FilteredGraphIndex, Predicate};

fn main() -> vicinity::Result<()> {
    let mut index = FilteredGraphIndex::new(2, Default::default())?;
    for i in 0..37 {
        index.add_slice(
            1000 + i,
            &[1.0, i as f32 / 100.0],
            HashMap::from([("collection".into(), AttrValue::Int(1))]),
        )?;
    }
    for doc_id in [90, 10, 50] {
        index.add_slice(
            doc_id,
            &[0.0, 1.0],
            HashMap::from([("collection".into(), AttrValue::Int(2))]),
        )?;
    }
    index.build()?;

    let query = [1.0, 0.0];
    let nearest = index.search(&query, 2)?;
    println!("Unfiltered top 2: {nearest:?}");
    for (collection, k) in [(2, 2), (2, 5), (3, 5)] {
        let predicate = Filter::Clause(Predicate::Eq(
            "collection".into(),
            AttrValue::Int(collection),
        ));
        let hits = index.search_filtered(&query, k, &predicate)?;
        let ids: Vec<u32> = hits.into_iter().map(|(doc_id, _)| doc_id).collect();
        println!("collection={collection}, k={k}: {ids:?}");
    }
    Ok(())
}
