# Benchmark Results

Historical benchmark tables for standard ANN datasets.

The checked-in `docs/*.jsonl` files are legacy result artifacts from this older
schema. They do not include `_meta`, `storage_mode`, `cache_state`,
`load_time_s`, or `index_bytes`, so use them only as historical in-memory
context. Storage-aware coverage should be generated with the current harness
commands below.

Current benchmark runs should use `examples/ann_benchmark.rs`, which writes one
`_meta` JSON line with the dataset, metric, actual `rustc --version`, MSRV, and
crate version, query limit, and measured query count, followed by one JSON line
per measurement with build time, RSS, storage mode, cache state, and
p50/p95/p99 latency. Use `--resume` to skip completed rows and `--fresh` to
recreate the result file.

QPS below is sequential single-query throughput (queries / wall-clock seconds)
from older single-run `--release` measurements on Apple Silicon. The exact CPU
model, thread count, frequency, and thermal state were not recorded for these
older rows, so treat absolute QPS as historical context. Re-run the harness on
your target machine before comparing against external papers.

Recommended current commands:

```bash
# Full single-query curve with latency tails.
cargo run --example ann_benchmark --release --features hnsw,ivf_pq,ivf_avq -- \
  data/ann-benchmarks/glove-25-angular \
  --algo hnsw --algo ivfpq --algo ivf_avq --json --fresh

# Bounded probe for fast iteration. The query limit is recorded in `_meta`, and
# `--resume` will not mix these rows with full-query runs.
cargo run --example ann_benchmark --release --features hnsw -- \
  data/ann-benchmarks/glove-25-angular --algo hnsw \
  --ef-search 10,20,30,40,50 --max-queries 1000 --json --fresh

# Bounded corpus probe for algorithms whose full-index build is expensive.
# Both the indexed-vector cap and the query cap are recorded in `_meta`, and
# recall is recomputed against the capped corpus before measurement.
cargo run --example ann_benchmark --release --features kdtree,balltree,rptree,kmeans_tree -- \
  data/ann-benchmarks/glove-25-angular \
  --algo balltree --algo rp_forest --algo kmeans_tree \
  --max-train 5000 --max-queries 200 --snapshot-load --json --fresh

# HNSW single-query plus parallel-query throughput.
RAYON_NUM_THREADS=4 cargo run --example ann_benchmark --release --features hnsw,parallel -- \
  data/ann-benchmarks/glove-25-angular --algo hnsw --batch --json --fresh

# HNSW build, save, reload, and post-load search rows.
cargo run --example ann_benchmark --release --features hnsw,serde -- \
  data/ann-benchmarks/glove-25-angular --algo hnsw --snapshot-load --json --fresh

# IVF-PQ with exact reranking pools.
# Add `--snapshot-load` when validating save/load equivalence and rerank
# persistence; it doubles the emitted IVF-PQ rows.
cargo run --example ann_benchmark --release --features ivf_pq,hnsw -- \
  data/ann-benchmarks/glove-25-angular --algo ivfpq \
  --pq-clusters 1024 --pq-codebooks 25 --pq-codebook-size 256 \
  --pq-training-sample-size 100000 --pq-kmeans-max-iter 20 \
  --pq-nprobes 16,32,64,128,256 --pq-rerank-pools 500,5000,20000 --json --fresh

# DiskANN in-memory graph search plus file and mmap search from the same build.
cargo run --example ann_benchmark --release --features hnsw,diskann -- \
  data/ann-benchmarks/glove-25-angular --algo diskann --json --fresh

# Classic tree baselines. These are comparison rows, not optimization targets.
# `--snapshot-load` adds persisted-and-reopened rows; search is still in-memory
# after load, unlike DiskANN's file and mmap rows.
cargo run --example ann_benchmark --release --features kdtree,balltree,rptree,kmeans_tree -- \
  data/ann-benchmarks/glove-25-angular \
  --algo kdtree --algo balltree --algo rptree --algo rp_forest --algo kmeans_tree \
  --tree-leaf-sizes 10,50 --rp-num-trees 10,20 --kmeans-clusters 8,16 \
  --snapshot-load --json --fresh

# FreshGraph delete/insert churn, scored against a live active-set oracle.
cargo run --example ann_benchmark --release --features hnsw,fresh_graph -- \
  data/ann-benchmarks/glove-25-angular --algo fresh_graph_churn --json --fresh

# IP-DiskANN-style in-place graph search and delete/insert churn.
cargo run --example ann_benchmark --release --features hnsw -- \
  data/ann-benchmarks/glove-25-angular \
  --algo inplace --algo inplace_churn --json --fresh

# LSM-tiered streaming churn, scored against a live active-set oracle.
cargo run --example ann_benchmark --release --features hnsw -- \
  data/ann-benchmarks/glove-25-angular --algo lsm_churn --json --fresh

# Filtered-search selectivity sweep. This is synthetic and writes its own JSONL.
cargo run --example acorn_selectivity --release \
  --features hnsw,filtered_graph,range_filtered,curator -- --json --fresh

# Broad dense-dataset coverage sweep. This emits recall/QPS/latency rows for
# the implemented single-vector search families that can consume dense vectors.
FEATURES=hnsw,nsw,vamana,diskann,ivf_pq,ivf_avq,ivf_rabitq,emg,nsg,pipnn,sng
FEATURES=$FEATURES,finger,fresh_graph,filtered_graph,curator,range_filtered
FEATURES=$FEATURES,rp_quant,binary_index,sq4,sq8,lsh,rptree,kdtree,balltree,kmeans_tree
cargo run --example ann_benchmark --release \
  --features "$FEATURES" \
  data/ann-benchmarks/glove-25-angular \
  --algo hnsw --algo nsw --algo vamana --algo diskann \
  --algo ivfpq --algo ivf_avq --algo ivf_rabitq \
  --algo emg --algo nsg --algo dual_branch --algo deg --algo pipnn --algo sng --algo finger \
  --algo fresh_graph --algo filtered_graph --algo curator --algo range_filtered \
  --algo rp_quant --algo binary_index --algo sq4 --algo sq4u --algo sq8u \
  --algo symphony_qg --algo symphony_qg_vr --algo adsampling --algo lsh --algo hnsw_prt \
  --algo brute --algo kdtree --algo balltree --algo rptree --algo rp_forest --algo kmeans_tree \
  --pq-training-sample-size 100000 --pq-kmeans-max-iter 20 --json --fresh
```

These commands intentionally separate low-recall, high-throughput operating
points from high-recall runs. External comparisons should use a recall/QPS
curve, not only the `Recall@10 = 100%` row.

## Current Validation Notes

On 2026-07-06, a full GloVe-25 HNSW M=16 run with
`ef_construction=500` confirmed the expected lower-recall throughput behavior:

| ef_search | Recall@10 | QPS | p95 us |
| --- | --- | --- | --- |
| 10 | 63.1% | 87,800 | 15.8 |
| 20 | 76.0% | 57,726 | 23.2 |
| 30 | 82.5% | 44,230 | 29.2 |
| 40 | 86.2% | 35,939 | 35.7 |
| 50 | 88.8% | 27,950 | 48.5 |

This validates the basic Perplexity finding: the old 100% recall row is not the
right operating point for QPS expectations. The 95% recall point still requires
a sweep above `ef_search=50` for this build configuration.

## Profiling Ledger

These are incremental profiling findings from the current optimization pass.
They are workload-specific and should be re-run before making release claims.

### HNSW Search Loop

Search-only Criterion benchmark:

```bash
CARGO_TARGET_DIR=/tmp/vicinity-hnsw-batch-target \
  cargo bench --bench hnsw_search --features hnsw -- \
  hnsw_search_only/ef/50 --measurement-time 3 --warm-up-time 1 --sample-size 20
```

Baseline with 4-neighbor distance batches:

| Workload | Time | Throughput |
| --- | --- | --- |
| `hnsw_search_only/ef/50` | 2.0073 ms / 100 queries | 49.819 Kelem/s |

Increasing the internal distance batch to 8 improved the same benchmark:

| Workload | Time | Throughput | Criterion change |
| --- | --- | --- | --- |
| `hnsw_search_only/ef/50` | 1.9511 ms / 100 queries | 51.252 Kelem/s | 4.6% faster, p < 0.05 |

The supporting `samply` profile was recorded with:

```bash
CARGO_TARGET_DIR=/tmp/vicinity-hnsw-profile-target \
  samply record --save-only -o /tmp/vicinity-hnsw-search-20260706.json.gz -- \
  cargo bench --bench hnsw_search --features hnsw -- \
  hnsw_search_only/ef/50 --profile-time 15
```

Symbolized top samples showed `cosine_distance_normalized` at about 36.5%
inclusive time and `innr::dense::dot` as the largest leaf bucket at about
23.1%. `greedy_search_layer` and `flush_batch` were also hot. This supports
keeping `innr` for dense distance kernels and optimizing HNSW graph traversal
around the kernel rather than replacing the kernel in `vicinity`.

The `hnsw_search` benchmark now also reports heap allocation counts per query.
On the 10K-vector, 128-d fixture, search allocates about 7 times per query, with
bytes scaling with `ef`:

| ef_search | Alloc calls/query | Alloc bytes/query | Time / 100 queries |
| --- | --- | --- | --- |
| 10 | 6.9 | 813.6 | 476.42 us |
| 50 | 7.0 | 3,796.0 | 1.9362 ms |
| 100 | 7.0 | 7,556.0 | 3.7768 ms |
| 200 | 7.0 | 13,156.0 | 7.3105 ms |

This makes heap traffic visible for future work, but the counts are bounded and
do not yet justify a heap-first rewrite without a profile showing allocator time
or result-heap maintenance as the bottleneck.

### Dense Distance Kernel

Direct Criterion measurements on the 128-d L2 kernel:

| Feature set | Time | Throughput |
| --- | --- | --- |
| `--features innr` | 8.382 ns | 15.271 Gelem/s |
| `--no-default-features` | 52.846 ns | 2.422 Gelem/s |

Binary inspection of the benchmark artifact confirmed that the `innr` path
lowers to an aarch64 NEON loop with paired vector loads and `fmla.4s`
accumulation. The actionable conclusion is to use `innr` for full-vector dense
distance and spend `vicinity` work on search layout, pruning, and storage.

### DiskANN File And Mmap Search

The search-only DiskANN benchmark isolates in-memory, file, and mmap search
from construction:

```bash
cargo bench --bench diskann_search --no-default-features --features diskann -- \
  diskann_search_only --measurement-time 3 --warm-up-time 1 --sample-size 20
```

Baseline at 5,000 vectors, 64 dimensions, 100 queries, `ef=50`:

| Row | Time | Throughput |
| --- | --- | --- |
| `memory_ef50` | 7.7394 ms / 100 queries | 12.921 Kelem/s |
| `file_ef50` | 147.37 ms / 100 queries | 678.55 elem/s |
| `mmap_ef50` | 19.290 ms / 100 queries | 5.1840 Kelem/s |

A reusable neighbor-buffer experiment avoided allocating a `Vec<u32>` per graph
record read, but it did not improve the measured path:

| Row | Criterion change |
| --- | --- |
| `memory_ef50` | no change, -0.18% mean time |
| `file_ef50` | regressed, +1.70% mean time, p < 0.05 |
| `mmap_ef50` | no change, -0.16% mean time |

That experiment was rejected. DiskANN storage work should next target record
layout, syscall count, vector decoding, and graph/vector co-location rather
than neighbor-list allocation alone. The `ann_benchmark` DiskANN file and mmap
rows now emit average graph reads, vector reads, logical bytes, visited nodes,
and retained candidates so storage changes can be compared against query work.

A later callback-style neighbor visitor removed the owned neighbor vector from
the search loop entirely, but regressed the same search-only bench:

| Row | Criterion change |
| --- | --- |
| `memory_ef50` | no change, +0.85% mean time |
| `file_ef50` | regressed, +5.07% mean time, p < 0.05 |
| `mmap_ef50` | regressed, +5.24% mean time, p < 0.05 |

That experiment was also rejected. The measured path is not currently limited
by the neighbor-list allocation enough to justify callback overhead.

Removing the extra mmap vector-byte copy in `DiskANNSearcher::read_vector`
improved the same benchmark while leaving in-memory search statistically flat:

| Row | After | Change vs baseline |
| --- | --- | --- |
| `memory_ef50` | 7.7063 ms / 100 queries | no change versus baseline |
| `file_ef50` | 141.38 ms / 100 queries | about 4.1% faster |
| `mmap_ef50` | 13.338 ms / 100 queries | about 30.9% faster |

### IVF-PQ Search Loop

The `ivfpq_search` benchmark's profiled counters show the remaining hot path is
ADC-table construction and scan dispatch, not final result selection:

| Shape | ADC table | ADC dispatch | Finalizer | Allocations |
| --- | --- | --- | --- | --- |
| `m25_one_dim_nprobe32_k10` | 97.315 us/query | 21.485 us/query | 2.568 us/query | 14 calls/query, 50.2 KB/query |
| `m5_runner_default_nprobe32_k10` | 46.726 us/query | 5.085 us/query | 2.511 us/query | 13 calls/query, 29.1 KB/query |

A bounded top-k heap experiment regressed GloVe-25 IVF-PQ at the same recall:

| Row | Before | Bounded top-k |
| --- | --- | --- |
| IVF-PQ `nprobe=32` | 1,759.6 QPS, 95.42% recall | 1,598.7 QPS, 95.42% recall |
| IVF-PQ `nprobe=32`, rerank 500 | 1,710.0 QPS, 96.58% recall | 1,418.6 QPS, 96.58% recall |

That experiment was rejected. The next IVF-PQ work should focus on the ADC
table path, codebook shape, and SIMD scan layout.

## Benchmark Coverage

The dense `ann_benchmark` runner covers the implemented single-vector ANN
families that accept dense `Vec<f32>` input:

| Area | Algorithms / rows | Eval path |
| --- | --- | --- |
| Graph search | HNSW, NSW, Vamana, DiskANN, NSG, SNG, EMG, DualBranch, DEG, PiPNN, FINGER, FreshGraph | `ann_benchmark --algo ...`; HNSW, NSW, Vamana, NSG, SNG, EMG, PiPNN, FINGER, and FreshGraph support `--snapshot-load`; DEG caps dense runs at 10,000 indexed vectors because construction is O(n^2) |
| IVF / quantized search | IVF-PQ, IVF-PQ rerank, IVF-AVQ, IVF-RaBitQ, RP-Quant, BinaryIndex, SQ4 | `ann_benchmark --algo ...`; IVF-PQ supports sampled training; these rows support `--snapshot-load` where the index is saved, reopened, and then searched from memory |
| Quantized HNSW accelerators | SQ4U, SQ8U, SymphonyQG, SymphonyQG-VR, ADSampling, HNSW-PRT | `ann_benchmark --algo ...`; SQ4U, SQ8U, SymphonyQG, and SymphonyQG-VR support `--snapshot-load` when their persistence features are compiled |
| Filtering | FilteredGraph, RangeFiltered, Curator, ACORN | dense rows via `ann_benchmark` are labeled `filter_mode=none`; FilteredGraph, RangeFiltered, and Curator support `--snapshot-load`; selectivity curves live in `acorn_selectivity` |
| Classical baselines | KD-tree, Ball tree, RP-tree, RP-forest, K-means tree, brute force | `ann_benchmark --algo ...`; tree rows can add `--snapshot-load` |
| Streaming / updates | FreshGraph churn, in-place graph, in-place churn, LSM churn | `ann_benchmark --algo fresh_graph_churn --algo inplace --algo inplace_churn --algo lsm_churn` |

Not every implemented module should produce a dense ANN row:

| Module | Why it is separate | Honest eval direction |
| --- | --- | --- |
| SparseMIPS | Requires sparse vectors, not dense ann-benchmarks `f32` arrays | Add a SPLADE/BM25 sparse dataset harness before reporting QPS/recall |
| EVoC | Clustering wrapper, not nearest-neighbor search | Report clustering metrics such as NMI/ARI on labeled clustering datasets |
| LEMUR | Inference scaffold that requires externally trained encoder weights | Evaluate on multi-vector retrieval datasets with MaxSim ground truth |
| SAQ / quantization helpers | Quantizers, not standalone indexes | Report quantization error, encoding throughput, and downstream recall when attached to an index |

Persistence and storage-mode benchmark rows follow `docs/persistence.md`.
Methods with both in-memory and file-backed search should report separate
`storage_mode=in_memory`, `storage_mode=file`, and, when implemented,
`storage_mode=mmap` rows. File and mmap rows should also report `load_time_s`
and `index_bytes` when the runner opens a saved index.

The long-term target is not only faster heap-resident search. In-memory QPS is
the easiest number to improve and the least representative for datasets that do
not fit in RAM. For large datasets, benchmark and implementation work should
prefer layouts that search persisted bytes directly: DiskANN graph/vector files,
mmap graph pages, IVF posting lists, segmented stores, and cold/warm page-cache
measurements. Snapshot-loaded rows remain useful correctness and reload
baselines, but they are not a substitute for file or mmap search rows.

### IVF-PQ sampled-training diagnostic (2026-07-06)

Command:

```bash
cargo run --example ann_benchmark --release --features ivf_pq,hnsw -- \
  data/ann-benchmarks/glove-25-angular --algo ivfpq \
  --pq-clusters 1024 --pq-codebooks 25 --pq-codebook-size 256 \
  --pq-training-sample-size 100000 --pq-kmeans-max-iter 20 \
  --pq-nprobes 16,32,64 --pq-rerank-pools 500,5000 \
  --results /tmp/vicinity-ivfpq-glove25-cb25-sampled.jsonl --fresh --json
```

Build: 71.37s, RSS 643,904 KB, `rustc 1.95.0`. These rows are in-memory,
warm-after-build search on GloVe-25.

Rows marked `*` were re-run after the IVF-PQ candidate finalizer switched from
full sorting to `select_nth_unstable_by` plus sorting the retained prefix
(`dfbf176`). That targeted re-run used the same dataset and IVF-PQ shape, with
build time 72.13s and RSS 644,320 KB.

Rows marked `**` add the direct 1-D ADC table path used by the GloVe-25
`cb=25` shape. That targeted re-run used the same dataset and IVF-PQ shape,
with build time 72.25s and RSS 644,240 KB.

| nprobe | Rerank pool | Recall@10 | QPS | p95 us |
|--------|-------------|-----------|-----|--------|
| 16 | none | 91.60% | 1,034.5 | 1,399.9 |
| 16 | 500 | 92.43% | 1,016.6 | 1,409.6 |
| 16 | 5,000 | 92.43% | 826.5 | 1,657.0 |
| 32 | none** | 95.64% | 811.7 | 1,684.2 |
| 32 | 500** | 96.69% | 796.3 | 1,706.9 |
| 32 | 5,000 | 96.69% | 471.7 | 2,797.4 |
| 64 | none | 97.68% | 266.9 | 4,760.1 |
| 64 | 500 | 98.91% | 265.2 | 4,783.3 |
| 64 | 5,000 | 98.91% | 252.3 | 4,963.0 |

Interpretation: sampled training is not the recall blocker when the PQ shape
has enough capacity. The earlier `cb=5, codebook_size=16` recipe was a
memory-stress configuration and saturates at low recall on GloVe-25. With
`cb=25, codebook_size=256`, recall crosses 95% at `nprobe=32`. The remaining
gap versus FAISS-style IVF-PQ targets is QPS, so the next optimization target is
the ADC/FastScan scan path; larger rerank pools are not helping this shape.

Search-only profiling on the same day showed that the standard 8-bit ADC path
was allocating and copying a packed LUT for every probed cluster even though the
ADC table was already flat. Borrowing that table in the dispatch path improved
the Criterion `ivfpq_search_only` benchmark on Apple Silicon:

| Shape | Before | After | Change |
|-------|--------|-------|--------|
| `m25_one_dim_nprobe32_k10` | 15.25 ms / 100 queries | 13.91 ms / 100 queries | 8.8% faster |
| `m25_one_dim_nprobe32_rerank500_k10` | 16.96 ms / 100 queries | 14.88 ms / 100 queries | 12.2% faster |
| `m5_runner_default_nprobe32_k10` | 10.02 ms / 100 queries | 10.01 ms / 100 queries | no significant change |
| `m5_runner_default_nprobe32_rerank500_k10` | 11.16 ms / 100 queries | 10.54 ms / 100 queries | 5.6% faster |

Command:

```bash
cargo bench --bench ivfpq_search --no-default-features \
  --features ivf_pq,benchmark -- \
  --sample-size 10 --warm-up-time 0.1 --measurement-time 0.1
```

A follow-up small-subvector ADC-table path avoids repeated generic L2 calls for
PQ subvector dimensions up to 8. This matters for the default GloVe-25
`m=5` shape, where each subvector has only 5 dimensions. The same Criterion
command measured:

| Shape | Before small-dim table path | After small-dim table path | Change |
|-------|-----------------------------|----------------------------|--------|
| `m5_runner_default_nprobe32_k10` | 10.01 ms / 100 queries | 7.22 ms / 100 queries | 27.9% faster |
| `m5_runner_default_nprobe32_rerank500_k10` | 10.54 ms / 100 queries | 8.04 ms / 100 queries | 23.7% faster |

With the `innr` feature enabled, the same path measured 6.96 ms / 100 queries
for `m5_runner_default_nprobe32_k10`. Before the small-dimension specialization,
enabling `innr` regressed this shape to 13.47 ms / 100 queries because the hot
loop repeatedly called a general dense-vector L2 function below `innr`'s SIMD
threshold. The split is now: `innr` for full-vector dense kernels, local IVF-PQ
kernels for tiny-subvector ADC-table construction and lookup-table scan.

The next profile showed the standard 8-bit ADC path was still copying PQ codes
from vector-order storage into a cluster-order scan buffer for every probed
cluster. Prepacking those cluster-order code buffers at build/load time removed
that search-time copy (`code_copy=0.000us/query`) and improved the same
Criterion bench:

| Shape | Before cluster-code cache | After cluster-code cache | Change |
|-------|---------------------------|--------------------------|--------|
| `m25_one_dim_nprobe32_k10` | 13.91 ms / 100 queries | 12.78 ms / 100 queries | 8.1% faster |
| `m25_one_dim_nprobe32_rerank500_k10` | 14.88 ms / 100 queries | 14.01 ms / 100 queries | 5.8% faster |
| `m5_runner_default_nprobe32_k10` | 7.22 ms / 100 queries | 5.94 ms / 100 queries | 17.7% faster |
| `m5_runner_default_nprobe32_rerank500_k10` | 8.04 ms / 100 queries | 6.98 ms / 100 queries | 13.2% faster |

A bounded 500-query GloVe-25 run at the R@10≈0.95 operating point moved from
820.7 QPS after the earlier ADC-table work to 1,759.6 QPS after the
cluster-code cache:

```bash
cargo run --no-default-features --features ivf_pq,innr,serde --release \
  --example ann_benchmark -- data/ann-benchmarks/glove-25-angular \
  --algo ivfpq --pq-clusters 1024 --pq-codebooks 25 \
  --pq-codebook-size 256 --pq-training-sample-size 100000 \
  --pq-kmeans-max-iter 20 --pq-nprobes 32 --pq-rerank-pools 500 \
  --max-queries 500 --json --fresh
```

| Algorithm | Recall@10 | QPS | p95 latency |
|-----------|-----------|-----|-------------|
| IVF-PQ `nprobe=32` | 95.42% | 1,759.6 | 759.8 us |
| IVF-PQ `nprobe=32`, rerank 500 | 96.58% | 1,710.0 | 775.7 us |

The bench target now also reports allocation counts for each profiled
`search_profiled()` call. Reusing the standard ADC distance buffer across
probed clusters reduced heap activity, but did not materially change
search-only timing. That means heap allocation was real, but not the remaining
primary bottleneck for these shapes:

| Shape | Before distance-buffer reuse | After distance-buffer reuse |
|-------|------------------------------|-----------------------------|
| `m25_one_dim_nprobe32_k10` | 48 alloc calls/query, 105.4 KB/query | 18 alloc calls/query, 96.4 KB/query |
| `m5_runner_default_nprobe32_k10` | 48 alloc calls/query, 84.9 KB/query | 18 alloc calls/query, 75.9 KB/query |

Preallocating the query candidate buffer from the probed cluster sizes cut the
remaining search allocations again:

| Shape | After distance-buffer reuse | After candidate preallocation |
|-------|-----------------------------|-------------------------------|
| `m25_one_dim_nprobe32_k10` | 18 alloc calls/query, 96.4 KB/query | 8 alloc calls/query, 50.8 KB/query |
| `m5_runner_default_nprobe32_k10` | 18 alloc calls/query, 75.9 KB/query | 8 alloc calls/query, 30.3 KB/query |

On 2026-07-07, binary inspection with `cargo asm --lib --no-default-features
--features "ivf_pq benchmark"
vicinity::ivf_pq::pq::ProductQuantizer::compute_adc_table_into` showed the
one-dimensional ADC-table path still went through `Vec::push` growth checks for
each table entry. Rewriting only that one-dimensional path to size the table
once and fill mutable chunks moved the same short Criterion smoke as follows:

| Shape | Before | After | ADC table before | ADC table after |
|-------|--------|-------|------------------|-----------------|
| `m25_one_dim_nprobe32_k10` | 13.08 ms / 100 queries | 4.51 ms / 100 queries | 99.466 us/query | 17.513 us/query |
| `m25_one_dim_nprobe32_rerank500_k10` | 13.56 ms / 100 queries | 5.28 ms / 100 queries | 98.020 us/query | 17.503 us/query |

The broader variant that changed all ADC-table paths was rejected: it regressed
`m5_runner_default_nprobe32_k10` by about 3.8% in the same smoke. The kept
change is intentionally limited to `subvector_dim == 1`, which is the GloVe-25
`cb=25` shape where the QPS gap matters most.

The bounded 500-query GloVe-25 macro run at the same R@10≈0.95 operating point
then moved from 1,759.6 QPS to 2,149.7 QPS for approximate search, and from
1,710.0 QPS to 2,105.5 QPS with rerank pool 500:

| Algorithm | Recall@10 | QPS | p95 latency |
|-----------|-----------|-----|-------------|
| IVF-PQ `nprobe=32` | 95.42% | 2,149.7 | 648.2 us |
| IVF-PQ `nprobe=32`, rerank 500 | 96.58% | 2,105.5 | 662.2 us |

The next binary-inspection pass looked at
`vicinity::pq_simd::adc_batch_dispatch_into`. On Apple Silicon, the compiler
vectorized the generic NEON dispatcher four candidates at a time, but the
inner loop still carried bounds-check bailout blocks for LUT gathers. A
specialized aarch64 path for the common 8-bit PQ case (`codebook_size=256`)
checks the LUT shape once and then gathers through raw pointers inside the
NEON loop. The short Criterion probe moved as follows:

| Shape | Before 8-bit NEON specialization | After 8-bit NEON specialization | ADC dispatch before | ADC dispatch after |
|-------|----------------------------------|---------------------------------|---------------------|--------------------|
| `m25_one_dim_nprobe32_k10` | 4.60 ms / 100 queries | 3.81 ms / 100 queries | 20.783 us/query | 13.484 us/query |
| `m25_one_dim_nprobe32_rerank500_k10` | 5.61 ms / 100 queries | 4.74 ms / 100 queries | 21.459 us/query | 13.841 us/query |
| `m5_runner_default_nprobe32_k10` | 5.94 ms / 100 queries | 5.84 ms / 100 queries | 5.319 us/query | 3.429 us/query |
| `m5_runner_default_nprobe32_rerank500_k10` | 6.97 ms / 100 queries | 6.63 ms / 100 queries | 5.135 us/query | 3.376 us/query |

The same bounded 500-query GloVe-25 macro run then moved from 2,149.7 QPS to
2,941.4 QPS for approximate search, and from 2,105.5 QPS to 2,806.2 QPS with
rerank pool 500:

| Algorithm | Recall@10 | QPS | p95 latency |
|-----------|-----------|-----|-------------|
| IVF-PQ `nprobe=32` | 95.42% | 2,941.4 | 470.6 us |
| IVF-PQ `nprobe=32`, rerank 500 | 96.58% | 2,806.2 | 488.5 us |

The storage-mode Criterion probe compares the freshly built heap index,
snapshot-loaded heap index, plain file searcher, and mmap searcher on the same
20K-vector synthetic shape used for short profiling. Throughput is queries per
second; each sample processes 100 queries. Adding optional list-contiguous
`list_codes.bin` plus `list_offsets.bin` sidecars removes the per-query
vector-order gather for file and mmap searchers while preserving fallback for
old snapshots.

| Shape | Heap after | Snapshot after | File before | File after | Mmap before | Mmap after |
|-------|-----------:|---------------:|------------:|-----------:|------------:|-----------:|
| `m25_one_dim_nprobe32_k10` | 26.4K | 26.3K | 675 | 16.7K | 17.4K | 25.9K |
| `m25_one_dim_nprobe32_rerank500_k10` | 21.2K | 21.4K | 556 | 2.61K | 13.6K | 18.7K |
| `m5_runner_default_nprobe32_k10` | 17.8K | 17.7K | 689 | 12.4K | 12.4K | 17.9K |
| `m5_runner_default_nprobe32_rerank500_k10` | 15.3K | 15.2K | 555 | 2.40K | 10.1K | 13.9K |

Interpretation: `load_from_dir()` is effectively at heap speed. List-contiguous
PQ-code sidecars fix the largest file/mmap approximate-search penalty: file
search improves by roughly 18-25x and mmap becomes close to heap search. Rerank
file mode still lags because exact reranking reads raw vectors in candidate
order from `raw_vectors.bin`; raw-vector locality is now the next storage-mode
target. Sorting file-mode rerank reads by vector index did not materially change
the `m25_one_dim` row and improved the `m5_runner_default` row by about 2.5%, so
it is a small cleanup rather than the full rerank fix.

## GloVe-25 (1.18M vectors, 25-d, angular distance)

Ground truth: brute-force k-NN on L2-normalized vectors (angular ≡ cosine for unit vectors).

### Summary

| Algorithm | Best Recall@10 | QPS at best | Notes |
|-----------|---------------|-------------|-------|
| HNSW (M=16) | 100.0% | 2,857 | Default choice |
| HNSW (M=32) | 100.0% | 2,017 | Higher memory, marginal recall gain |
| Vamana | 100.0% | 1,177 | Slow build (~2000s) |
| DiskANN | 100.0% | 1,029 | Vamana + I/O layout |
| SQ4U | 99.9% | 1,056 | 4-bit quantized HNSW; ~3x slower than plain HNSW at d=25 |
| NSW | 99.2% | 1,288 | |
| IVF-PQ (cb=25) | 98.7% | 69 | 25 codebooks on 25-d (1-d subspaces) |
| IVF-AVQ | 90.9% | 194 | ScaNN-style anisotropic VQ |
| RP-Forest | 58.5% | 4,221 | Fast build, moderate recall |
| IVF-PQ (cb=5) | 45.1% | 262 | 5 codebooks on 25-d (too coarse) |
| KD-Tree | 100.0% | 22 | Exact; too slow at 1M+ |
| Brute | 100.0% | 42 | Exact baseline |

SQ4U and SymphonyQG are designed for high-dimensional data. At d=25, the quantization
overhead exceeds the savings from cheaper distance computation. SymphonyQG produces
<0.3% recall at d=25 (RaBitQ error is O(1/sqrt(d))) and is excluded.

### HNSW (M=16, ef_construction=200)

| ef_search | Recall@10 | QPS |
|-----------|-----------|-----|
| 10 | 77.8% | 57,747 |
| 20 | 87.9% | 35,138 |
| 50 | 96.1% | 17,035 |
| 100 | 98.8% | 9,857 |
| 200 | 99.8% | 5,463 |
| 400 | 100.0% | 2,907 |

### HNSW (M=32, ef_construction=200)

| ef_search | Recall@10 | QPS |
|-----------|-----------|-----|
| 10 | 79.7% | 44,927 |
| 20 | 90.1% | 28,098 |
| 50 | 97.1% | 13,098 |
| 100 | 99.2% | 7,377 |
| 200 | 99.8% | 4,019 |
| 400 | 100.0% | 2,017 |

### Vamana (R=64, alpha=1.3, ef_construction=200)

| ef_search | Recall@10 | QPS |
|-----------|-----------|-----|
| 10 | 88.5% | 18,282 |
| 20 | 94.7% | 11,222 |
| 50 | 98.7% | 5,599 |
| 100 | 99.7% | 3,213 |
| 200 | 100.0% | 1,821 |
| 400 | 100.0% | 1,177 |

### NSW (M=16)

| ef_search | Recall@10 | QPS |
|-----------|-----------|-----|
| 10 | 79.1% | 43,447 |
| 20 | 88.2% | 26,043 |
| 50 | 95.4% | 9,939 |
| 100 | 97.7% | 4,813 |
| 200 | 98.8% | 2,322 |
| 400 | 99.2% | 1,288 |

### SQ4U (M=16, 4-bit quantized traversal + exact rerank)

| ef_search | Recall@10 | QPS |
|-----------|-----------|-----|
| 10 | 56.7% | 17,824 |
| 20 | 77.5% | 11,782 |
| 50 | 93.3% | 5,852 |
| 100 | 98.1% | 3,409 |
| 200 | 99.6% | 1,927 |
| 400 | 99.9% | 1,056 |

### DiskANN (R=64, alpha=1.3)

| ef_search | Recall@10 | QPS |
|-----------|-----------|-----|
| 10 | 88.7% | 17,555 |
| 20 | 94.8% | 10,936 |
| 50 | 98.8% | 5,505 |
| 100 | 99.7% | 3,183 |
| 200 | 100.0% | 1,817 |
| 400 | 100.0% | 1,029 |

### IVF-PQ (1024 clusters, 25 codebooks)

| nprobe | Recall@10 | QPS |
|--------|-----------|-----|
| 4 | 81.7% | 1,200 |
| 8 | 90.1% | 682 |
| 16 | 94.9% | 370 |
| 32 | 97.2% | 198 |
| 64 | 98.3% | 110 |
| 128 | 98.6% | 73 |
| 256 | 98.7% | 69 |

### IVF-AVQ (512 partitions, 5 codebooks)

| nprobe | Recall@10 | QPS |
|--------|-----------|-----|
| 4 | 47.1% | 3,478 |
| 8 | 59.1% | 2,155 |
| 16 | 72.7% | 1,267 |
| 32 | 82.2% | 709 |
| 64 | 87.3% | 389 |
| 128 | 90.3% | 206 |
| 256 | 90.9% | 194 |

## GIST-960 (1M vectors, 960-d, L2)

Ground truth: brute-force L2 k-NN.

### Summary

| Algorithm | Best Recall@10 | QPS at best | Notes |
|-----------|---------------|-------------|-------|
| HNSW (M=16) | 97.7% | 330 | |
| SQ4U | 95.9% | 175 | ~2x slower than plain HNSW even at d=960 |

SQ4U does not outperform plain HNSW at d=960. The reranking pass (exact f32 distance
on the candidate pool) dominates, negating the savings from quantized graph traversal.

### HNSW (M=16, ef_construction=200)

| ef_search | Recall@10 | QPS |
|-----------|-----------|-----|
| 10 | 48.3% | 2,905 |
| 20 | 62.5% | 3,175 |
| 50 | 78.3% | 1,729 |
| 100 | 87.9% | 1,047 |
| 200 | 94.0% | 593 |
| 400 | 97.7% | 330 |

### SQ4U (M=16, 4-bit quantized traversal + exact rerank)

| ef_search | Recall@10 | QPS |
|-----------|-----------|-----|
| 10 | 33.8% | 2,536 |
| 20 | 49.8% | 1,800 |
| 50 | 69.4% | 989 |
| 100 | 81.7% | 569 |
| 200 | 90.4% | 317 |
| 400 | 95.9% | 175 |

## Caveats

- GloVe-25 rankings differ from high-dimensional data. SQ4U and SymphonyQG
  are designed for d >= 128 but underperform at all tested dimensions.
- Build times and QPS are single-run wall-clock on a lightly loaded machine.
- GIST-960 numbers were collected with two benchmark processes sharing CPU
  (both equally affected; ratios are valid, absolute QPS are ~50% lower than
  solo runs would produce).
- IVF-PQ with 5 codebooks on 25-d is a known misconfiguration (too coarse).
- Some algorithms from prior runs (EMG, PiPNN, NSG, IVF-RaBitQ, etc.) are not
  yet re-benchmarked with the current optimized codebase.
