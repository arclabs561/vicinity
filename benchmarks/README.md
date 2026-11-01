# Benchmarks and benefit profiles

Two artifacts live here:

- `rows.jsonl` — one JSON object per measurement (family, dataset, knob, recall,
  p50/p95/p99, qps, build_s, mem_mib, dists_per_query, and capability flags
  `exact`, `supports_update`, `supports_filter`).
- `scripts/benefits.py` — turns those rows into a per-method *benefit profile*:
  the best row per family, the leader on each axis, the Pareto front on
  (recall, latency), and a one-line "wins when" statement.

## Why a benefit profile

A single recall/QPS number cannot represent what a method is for. The tree family
is exact and cheap to build but slow; IVF is cheap to build and needs training;
a graph has the best recall per query but no exactness; the filtered graph is the
only one that composes with predicates. Ranking methods by one axis hides the
reason to keep any of them.

The schema records the dimensions a method can win on:

| Field | Meaning | Axis |
| --- | --- | --- |
| `recall` | recall@k at this operating point | quality |
| `p50_us`, `p95_us`, `p99_us` | latency percentiles | speed, tail |
| `qps` | throughput | speed |
| `build_s` | construction wall clock | cost |
| `mem_mib` | resident growth across build | memory |
| `dists_per_query` | distance computations per query | work |
| `exact` | exact result guarantee | correctness |
| `supports_update` | insert/delete without rebuild | operation |
| `supports_filter` | predicate composes with search | operation |

## Usage

```sh
python3 scripts/benefits.py benchmarks/rows.jsonl
```

Output is the table above: per-family rows, axis leaders, Pareto front, and the
"wins when" sentence that should appear in the README for each family.

## Producing rows without external datasets

`cargo run --release --example all_families` measures every family on one
protocol and can emit rows in the shape above. Structure is a variable, not a
constant: seven synthetic domains stand in for seven kinds of real workload, and
each one's hardness is measured at run time rather than implied by its generator.

| Domain | Width | Stands in for |
| --- | ---: | --- |
| `uniform` | 96 | baseline: uniform random points, no cluster structure |
| `clustered16` | 96 | 16 centres, `sigma = 0.25`: a few strong clusters |
| `text` | 96 | Zipf-weighted 200-topic mixture, anisotropic per topic: word/document embeddings |
| `image` | 96 | 64 parents x 16 children at a small offset: deep-embedding locality |
| `sparse` | 768 | ~2% nonzero, queries share terms with a document: lexical vectors |
| `neardup` | 96 | 90% jitter around 64 seed documents: near-duplicate-heavy corpora |
| `lowdim` | 8 | points on a random 3-D subspace: intrinsically low-dimensional data |

Two numbers are printed beside each domain because one is not enough:

- **LID**: Levina-Bickel maximum-likelihood local intrinsic dimensionality at
  `k = 10`, median with p10-p90 over 200 sampled points. This is the difficulty of
  the *inside* of a neighbourhood.
- **Separation**: median distance to a random base point divided by the distance
  to the 10th neighbour. Near 1.0 means neighbours are indistinguishable from the
  background; large means a query's cluster is easy to *reach*.

They disagree on purpose: inside a near-duplicate cloud LID stays high (the cloud
is still locally high-dimensional) while separation is enormous, and recall
follows separation. Reporting only LID would call that domain hard and be wrong.

| Environment variable | Effect |
| --- | --- |
| `ANN_DATASETS` | Comma-separated domains to run, e.g. `text,image`; unset runs all seven |
| `ANN_FVECS` | Path to a `.fvecs` file (int32 dim then dim float32 per row, little-endian); replaces the synthetic domains. The first 200 rows are also the queries, so recall@10 has a 0.1 self-match floor |
| `ANN_JSONL` | Path to write JSONL rows in the shape above, ready for `scripts/benefits.py` |

Loading an angular dataset (GloVe, deep-image) reproduces the ann-benchmarks
ranking; rows are L2-normalised, so a Euclidean dataset such as SIFT is ranked by
cosine instead of L2.

## Rules

- A family is only listed once it has a measured row in `rows.jsonl`.
- A capability flag is `true` only if a test exercises it, not because the type
  has a method.
- The Pareto front is computed on (recall, p50) at the family's best recall; a
  method may be dominated there and still lead on memory or build cost, which is
  exactly why the axis leaders are printed separately.
