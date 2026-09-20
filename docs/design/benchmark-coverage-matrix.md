# Benchmark coverage matrix

Status: current snapshot
Source of truth: `examples/ann_benchmark/support.rs::ALGORITHM_OPTIONS`
Validated against: `examples/ann_benchmark.rs` dispatch, `Cargo.toml` benchmark
entries, `docs/benchmark-results.md`, and `docs/persistence.md`.

This is a coverage map, not a performance ranking. A method is comparable only
when its metric, dataset shape, quality oracle, and persistence mode agree.

## Registry accounting

The registry contains 42 selections. `--all-dense` selects the 41 non-sparse
selections. Strict CI uses `--all-features --json --batch --snapshot-load
--require-complete`; it fails on missing, stale, or metric-incompatible rows.

| Classification | Count | Runner | Meaning |
| --- | ---: | --- | --- |
| Dense registry selections | 41 | `ann_benchmark --all-dense` | Registered dense dispatch rows, including baselines and update/churn rows. |
| Sparse selection | 1 | `sparse_mips_benchmark` | Separate SPV1 sparse-vector harness; not a dense substitute. |
| Non-ANN registry selections | 0 | n/a | EVoC and LEMUR are not registry rows: EVoC is clustering and LEMUR remains an inference scaffold. |
| Unclassified registry selections | 0 | n/a | Every current registry selection is assigned to one of the two harnesses. |

## Dense registry rows

All names below are selected by `--all-dense` and participate in the strict
result gate. “Snapshot” means a restart/reload row where supported; it does not
mean direct on-disk query support.

| Selection | Family / workload | Storage or special path |
| --- | --- | --- |
| `external_hnsw_rs`, `external_usearch` | External HNSW/USearch baselines | In-memory; external seed control unavailable. |
| `hnsw`, `hnsw_parallel`, `nsw` | HNSW-family graph search | In-memory plus serde snapshot where enabled. |
| `ivfpq` | IVF-PQ | In-memory, snapshot, direct file, and mmap paths. |
| `ivf_avq` | IVF-AVQ | In-memory, snapshot, direct file, and mmap paths. |
| `ivf_rabitq` | IVF-RaBitQ | In-memory/snapshot path; no direct file row. |
| `rp_quant`, `binary_index`, `sq4`, `sq4u`, `sq8u` | Quantized/projected HNSW-family rows | Snapshot or rebuild-derived state; exact mode is selection-specific. |
| `emg`, `nsg`, `dual_branch`, `deg`, `pipnn`, `sng`, `vamana` | Experimental graph families | Feature-gated build/search rows; DEG construction is capped because it is quadratic. |
| `diskann` | DiskANN in-memory | Heap-resident graph search. |
| `diskann_file`, `diskann_mmap` | DiskANN separate-file paths | Positional reads or read-only mapped graph/vector files. |
| `diskann_page_file`, `diskann_page_mmap` | DiskANN page-layout paths | Benchmark-feature experimental page readers. |
| `symphony_qg`, `symphony_qg_vr`, `finger` | Quantized/accelerated graph variants | Snapshot/rebuild-derived state; current compacted-VR restrictions remain. |
| `fresh_graph`, `fresh_graph_churn` | Mutable fresh graph | Snapshot/reload and churn; not a WAL-equivalent durability claim. |
| `store`, `store_snapshot` | Segmented HNSW store | `segstore` mutable path plus reopened snapshot row. |
| `inplace`, `inplace_churn`, `lsm_churn` | Mutable/update paths | Restart snapshots and churn; update persistence is distinct from `segstore`. |
| `filtered_graph`, `curator`, `range_filtered` | Filtered graph/tree paths | Snapshot/reload; representative filter distributions remain needed. |
| `adsampling`, `hnsw_prt` | HNSW-derived query accelerators | Derived from persisted base HNSW; no independent persistence format. |
| `lsh` | Locality-sensitive hashing | In-memory/snapshot path according to enabled feature. |
| `kdtree`, `balltree`, `rptree`, `rp_forest`, `kmeans_tree` | Classic tree baselines | Tree snapshot/rebuild paths; leaf/depth parameters remain visible. |
| `brute` | Exact baseline | In-memory exact scan and canonical quality control. |

The grouped table is documentation convenience only. The registry names remain
the authoritative row keys; grouped prose must not imply identical parameters
or persistence behavior.

## What this does not prove

- It does not cover every public HNSW query shape: adaptive, flat-batch,
  selectivity-routing, and comparable filtered distributions need workloads and
  quality oracles.
- It does not make rows comparable across sparse, clustering, multi-vector,
  filtered, or mutable workloads; those need their own data and metrics.
- It does not prove every storage mode is fast or durable. DiskANN has explicit
  five-path parity checks; other families need equivalent path-specific gates.
- The tiny CI fixture proves dispatch, validity, storage-row emission, and
  resume completeness. Representative data and repeated release measurements
  are still required for optimization decisions.

## Next coverage gates

1. Add representative HNSW batch/adaptive/selectivity and filtered workloads.
2. Add real sparse data to SPV1 and a trained multi-vector workload before
   promoting SparseMIPS or LEMUR into cross-family comparisons.
3. Give every new public query path an exact oracle or documented approximate
   quality contract before adding speed rows.
4. Keep persistence mode, open/reload cost, cache state, and update durability
   separate in every result schema.
