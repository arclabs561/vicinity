//! Opt-in coverage checks using the same expected rows as resume.

use super::support::{load_completed_results_for_run, request_completed, Config};

pub(super) fn validate_config(cfg: &Config) -> Result<(), String> {
    if !cfg.require_complete {
        return Ok(());
    }
    if !cfg.json {
        return Err("--require-complete requires --json".into());
    }
    if cfg.batch && !cfg!(feature = "parallel") {
        return Err("--require-complete --batch requires the parallel feature".into());
    }
    if cfg.snapshot_load && !cfg!(feature = "serde") {
        return Err("--require-complete --snapshot-load requires the serde feature".into());
    }
    if !cfg.resume && cfg.results_path.exists() {
        return Err("--require-complete needs a new results path, --fresh, or --resume".into());
    }
    Ok(())
}

pub(super) fn verify_results(
    cfg: &Config,
    dim: usize,
    train_len: usize,
    test_len: usize,
) -> Result<(), String> {
    if !cfg.require_complete {
        return Ok(());
    }
    let completed = load_completed_results_for_run(
        &cfg.results_path,
        &cfg.data_dir,
        cfg.max_train,
        cfg.max_queries,
        cfg.warmup_queries,
        cfg.search_k,
        cfg.seed,
        cfg.repeat,
    );
    let missing: Vec<_> = cfg
        .algos
        .iter()
        .filter(|algo| {
            !completed.has_matching_meta
                || algo.as_str() == "sparse_mips"
                || !request_completed(&completed, algo, cfg, dim, train_len, test_len)
        })
        .map(String::as_str)
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "incomplete benchmark result coverage: {} (check features, metric, storage modes, and dataset kind)",
            missing.join(", ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_coverage_rejects_text_output_and_stale_nonresume_files() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = Config {
            require_complete: true,
            results_path: dir.path().join("results.jsonl"),
            ..Default::default()
        };
        assert!(validate_config(&cfg).unwrap_err().contains("--json"));
        cfg.json = true;
        assert!(validate_config(&cfg).is_ok());
        std::fs::write(&cfg.results_path, "stale").unwrap();
        assert!(validate_config(&cfg)
            .unwrap_err()
            .contains("new results path"));
        cfg.resume = true;
        assert!(validate_config(&cfg).is_ok());
        assert!(verify_results(&cfg, 8, 64, 12).is_err());
    }
}
