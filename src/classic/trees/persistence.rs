use crate::RetrieveError;
use serde::{de::DeserializeOwned, Serialize};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

pub(crate) fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), RetrieveError> {
    let tmp_path = path.with_extension("tmp");
    {
        let file = std::fs::File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, value)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        writer.flush()?;
    }
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

pub(crate) fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, RetrieveError> {
    let file = std::fs::File::open(path)?;
    serde_json::from_reader(BufReader::new(file))
        .map_err(|e| RetrieveError::FormatError(e.to_string()))
}

pub(crate) fn validate_vector_shape(
    name: &str,
    dimension: usize,
    num_vectors: usize,
    vectors: &[f32],
    doc_ids: &[u32],
) -> Result<(), RetrieveError> {
    if dimension == 0 {
        return Err(RetrieveError::FormatError(format!(
            "{name} manifest has zero dimension"
        )));
    }
    if num_vectors == 0 {
        return Err(RetrieveError::FormatError(format!(
            "{name} manifest has zero vectors"
        )));
    }
    if vectors.len() != num_vectors * dimension {
        return Err(RetrieveError::FormatError(format!(
            "{name} vectors length {} does not match {} vectors of dimension {}",
            vectors.len(),
            num_vectors,
            dimension
        )));
    }
    if doc_ids.len() != num_vectors {
        return Err(RetrieveError::FormatError(format!(
            "{name} doc_ids length {} does not match vector count {}",
            doc_ids.len(),
            num_vectors
        )));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::path::Path;

    type SnapshotBuilder = fn(&Path) -> Result<(), RetrieveError>;
    type SnapshotLoader = fn(&Path) -> Result<(), RetrieveError>;

    struct Case {
        build: SnapshotBuilder,
        load: SnapshotLoader,
    }

    #[cfg(feature = "kdtree")]
    fn build_kdtree(dir: &Path) -> Result<(), RetrieveError> {
        use crate::classic::trees::kdtree::{KDTreeIndex, KDTreeParams};

        let mut index = KDTreeIndex::new(3, KDTreeParams::default())?;
        for i in 0..16u32 {
            index.add(1000 + i, vec![i as f32, (i * 2) as f32, 1.0])?;
        }
        index.build()?;
        index.save_to_dir(dir)
    }

    #[cfg(feature = "kdtree")]
    fn load_kdtree(dir: &Path) -> Result<(), RetrieveError> {
        crate::classic::trees::kdtree::KDTreeIndex::load_from_dir(dir).map(|_| ())
    }

    #[cfg(feature = "balltree")]
    fn build_balltree(dir: &Path) -> Result<(), RetrieveError> {
        use crate::classic::trees::balltree::{BallTreeIndex, BallTreeParams};

        let mut index = BallTreeIndex::new(3, BallTreeParams::default())?;
        for i in 0..16u32 {
            index.add(2000 + i, vec![i as f32, (i * 2) as f32, 1.0])?;
        }
        index.build()?;
        index.save_to_dir(dir)
    }

    #[cfg(feature = "balltree")]
    fn load_balltree(dir: &Path) -> Result<(), RetrieveError> {
        crate::classic::trees::balltree::BallTreeIndex::load_from_dir(dir).map(|_| ())
    }

    #[cfg(feature = "rptree")]
    fn build_rptree(dir: &Path) -> Result<(), RetrieveError> {
        use crate::classic::trees::random_projection::{RPTreeIndex, RPTreeParams};

        let mut index = RPTreeIndex::new(4, RPTreeParams::default())?;
        for i in 0..32u32 {
            let mut v = vec![0.0f32; 4];
            v[(i as usize) % 4] = 1.0;
            index.add(3000 + i, v)?;
        }
        index.build()?;
        index.save_to_dir(dir)
    }

    #[cfg(feature = "rptree")]
    fn load_rptree(dir: &Path) -> Result<(), RetrieveError> {
        crate::classic::trees::random_projection::RPTreeIndex::load_from_dir(dir).map(|_| ())
    }

    #[cfg(feature = "rptree")]
    fn build_rp_forest(dir: &Path) -> Result<(), RetrieveError> {
        use crate::classic::trees::rp_forest::{RPTreeParams, RpForestIndex, RpForestParams};

        let params = RpForestParams {
            num_trees: 3,
            tree_params: RPTreeParams { max_leaf_size: 4 },
        };
        let mut index = RpForestIndex::new(4, params)?;
        for i in 0..32u32 {
            let mut v = vec![0.0f32; 4];
            v[(i as usize) % 4] = 1.0;
            index.add(4000 + i, v)?;
        }
        index.build()?;
        index.save_to_dir(dir)
    }

    #[cfg(feature = "rptree")]
    fn load_rp_forest(dir: &Path) -> Result<(), RetrieveError> {
        crate::classic::trees::rp_forest::RpForestIndex::load_from_dir(dir).map(|_| ())
    }

    #[cfg(feature = "kmeans_tree")]
    fn build_kmeans_tree(dir: &Path) -> Result<(), RetrieveError> {
        use crate::classic::trees::kmeans_tree::{KMeansTreeIndex, KMeansTreeParams};

        let mut index = KMeansTreeIndex::new(3, KMeansTreeParams::default())?;
        for i in 0..32u32 {
            index.add(5000 + i, vec![i as f32, (i * 2) as f32, 1.0])?;
        }
        index.build()?;
        index.save_to_dir(dir)
    }

    #[cfg(feature = "kmeans_tree")]
    fn load_kmeans_tree(dir: &Path) -> Result<(), RetrieveError> {
        crate::classic::trees::kmeans_tree::KMeansTreeIndex::load_from_dir(dir).map(|_| ())
    }

    fn cases() -> Vec<Case> {
        let mut cases = Vec::new();
        #[cfg(feature = "kdtree")]
        cases.push(Case {
            build: build_kdtree,
            load: load_kdtree,
        });
        #[cfg(feature = "balltree")]
        cases.push(Case {
            build: build_balltree,
            load: load_balltree,
        });
        #[cfg(feature = "rptree")]
        {
            cases.push(Case {
                build: build_rptree,
                load: load_rptree,
            });
            cases.push(Case {
                build: build_rp_forest,
                load: load_rp_forest,
            });
        }
        #[cfg(feature = "kmeans_tree")]
        cases.push(Case {
            build: build_kmeans_tree,
            load: load_kmeans_tree,
        });
        cases
    }

    fn mutate_snapshot(dir: &Path, mutator: impl FnOnce(&mut Value)) {
        let path = dir.join("index.json");
        let mut value: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        mutator(&mut value);
        std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    }

    fn assert_load_fails(case: &Case, mutator: impl FnOnce(&Path)) {
        let dir = tempfile::tempdir().unwrap();
        (case.build)(dir.path()).unwrap();
        mutator(dir.path());
        assert!(
            (case.load)(dir.path()).is_err(),
            "corrupt classical tree snapshot loaded successfully"
        );
    }

    #[test]
    fn classical_snapshots_reject_bad_version_type() {
        for case in cases() {
            assert_load_fails(&case, |dir| {
                mutate_snapshot(dir, |v| v["version"] = "bad".into())
            });
        }
    }

    #[test]
    fn classical_snapshots_reject_future_version() {
        for case in cases() {
            assert_load_fails(&case, |dir| {
                mutate_snapshot(dir, |v| v["version"] = 999u32.into())
            });
        }
    }

    #[test]
    fn classical_snapshots_reject_truncated_json() {
        for case in cases() {
            assert_load_fails(&case, |dir| {
                std::fs::write(dir.join("index.json"), br#"{"version":"#).unwrap();
            });
        }
    }

    #[test]
    fn classical_snapshots_reject_zero_dimension() {
        for case in cases() {
            assert_load_fails(&case, |dir| {
                mutate_snapshot(dir, |v| v["index"]["dimension"] = 0u32.into())
            });
        }
    }

    #[test]
    fn classical_snapshots_reject_bad_vector_count() {
        for case in cases() {
            assert_load_fails(&case, |dir| {
                mutate_snapshot(dir, |v| {
                    v["index"]["vectors"].as_array_mut().unwrap().pop();
                });
            });
        }
    }

    #[test]
    fn classical_snapshots_reject_bad_doc_id_count() {
        for case in cases() {
            assert_load_fails(&case, |dir| {
                mutate_snapshot(dir, |v| {
                    v["index"]["doc_ids"].as_array_mut().unwrap().pop();
                });
            });
        }
    }
}
