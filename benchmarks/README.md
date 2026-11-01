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

## Rules

- A family is only listed once it has a measured row in `rows.jsonl`.
- A capability flag is `true` only if a test exercises it, not because the type
  has a method.
- The Pareto front is computed on (recall, p50) at the family's best recall; a
  method may be dominated there and still lead on memory or build cost, which is
  exactly why the axis leaders are printed separately.
