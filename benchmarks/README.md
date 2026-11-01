# Benchmarks: the performance ratchet

"Raise the bar as we go" only means something if a lower bar fails. This
directory holds the measured baseline and the policy that compares new runs
against it.

- `baseline.jsonl` — one committed run of `examples/all_families.rs`, produced
  by `--json`. It is the bar.
- `../scripts/ratchet.py` — runs the harness (or reads a JSONL produced
  elsewhere), compares row by row, and exits non-zero when a gated metric
  regresses past its tolerance.

## Protocol

The harness is the single protocol every family is measured on:

- `n = 10_000` base vectors, `dim = 96`, `k = 10`, `200` held-out queries.
- Two distributions: `uniform` and `clustered` (16 centres, `sigma = 0.25`).
  Both are L2-normalised, so every family can rank the same vectors.
- Ground truth is an exact top-10 scan per dataset, computed before any index is
  built or timed.
- `p50/p95/p99` are per-query wall clock; `qps` is queries over the whole timed
  loop; five untimed warm-up queries run first.
- `1/ratio@10` is exact-scan p50 divided by the family p50, the time-based
  definition ann-benchmarks uses.
- `dists_per_query` is the count from `vicinity::distance_stats`, enabled with
  `--features distance-stats`. It is `null` when the feature is off and the
  ratchet does not gate it then. `scripts/ratchet.py` always enables it, so the
  baseline and every compared run carry the same counting overhead and their
  timings stay comparable.
- `build_s` and `mem_mib` describe the build that produced the row; `mem_mib`
  is resident-set growth measured with `ps`, and is `null` when unavailable.

## The JSONL row

The first line of a run is a `_meta` object (protocol constants, harness and
host versions, `distance_stats`). Every following line is one row:

| field | column | notes |
| --- | --- | --- |
| `family` | family | index family |
| `dataset` | dataset | `uniform` or `clustered` |
| `knob` | knob | swept search parameter |
| `knob_value` | knob | its value for this row |
| `recall_at_10` | recall@10 | overlap with exact top-10 |
| `inverse_ratio` | 1/ratio@10 | exact-scan p50 / family p50 |
| `p50_us` | p50 us | per-query wall clock |
| `p95_us`, `p99_us` | p95, p99 us | per-query wall clock |
| `qps` | QPS | queries / timed-loop seconds |
| `build_s` | build s | build that produced the row |
| `mem_mib` | mem MiB | RSS growth across the build; `null` if unavailable |
| `dists_per_query` | dists/q | `null` without `--features distance-stats` |

The row key is `(family, dataset, knob, knob_value)`.

## What is gated, and by how much

A run fails when any of these move further than the stated tolerance:

| metric | direction | tolerance | noise floor |
| --- | --- | --- | --- |
| `recall@10` | higher is better | drop greater than **0.005 absolute** | — |
| `p50` | lower is better | rise greater than **15%** | +10 µs |
| `p99` | lower is better | rise greater than **25%** | +50 µs |
| `dists_per_query` | lower is better | rise greater than **20%** | +5 |
| `mem_mib` | lower is better | rise greater than **15%** | +2 MiB |
| `build_s` | lower is better | rise greater than **25%** | +0.01 s |

A relative tolerance is only meaningful above the measurement's noise floor: a
4 µs → 8 µs move is a 100% "regression" that is really one scheduling decision
on a 200-query loop, and a 0.0 → 7.5 MiB RSS move is page-reuse granularity. A
row whose absolute change is below the floor is not gated on that metric. A
baseline value of zero cannot support a ratio at all and is reported as
unmeasured instead of failing on any positive value.

Every (family, dataset, knob, knob value) row in the baseline is compared, so a
sweep point cannot quietly get worse while another improves.

One run is not a measurement of a tail. `--runs 3` (or repeated `--current`
files) takes the per-row median before comparing, which keeps a regression that
moves every run and drops a one-run scheduling spike. The invariants the gate
depends on are fixed: `hnsw` builds with `seed = Some(42)` and
`ivf_rabitq`'s k-means uses `params.seed`, so recall is a property of the code
rather than of the entropy pool.

Improvements are printed in a table and never fail the run. A row present in the
baseline but missing from the run is a failure by default: a family silently
dropping out is a coverage regression the metric table cannot show. `--family`
and `--allow-subset` downgrade that to a warning for partial runs.

`recall@10` is the only gate that catches a correctness-shaped speedup. The
tolerances are one-sided allowances for measurement noise, not targets: a change
that buys a genuine improvement should re-baseline, not hide behind them.

## Why timings are schedule-only

The `ratchet` job runs on a schedule and on `workflow_dispatch`, never on pull
requests. `cargo test` remains the PR gate.

Timings on shared CI runners move with the CPU generation, the runner image, the
hypervisor neighbours, and frequency scaling. A 15% p50 budget is generous for
one machine and noisy across machines, so gating PRs on it would fail diffs that
changed nothing. Recall and `dists_per_query` are hardware-independent, but a
job that gates them is still a periodic job: it answers "did this week's `main`
get slower", not "is this commit slow".

The committed baseline records its host (`host`, `kernel`, `rustc`, `cores`,
`load1` in the `_meta` line). When the current run's host differs, the ratchet
prints a warning and the CI invocation passes `--advisory-timing`, which demotes
the timing and memory gates to advisories while recall and `dists_per_query`
keep failing the job. Re-recording the baseline on the runner image removes that
caveat.

Load is checked too: if either the baseline or the current run was recorded at
more than `--max-load-ratio` (default: 1.0) load per core, the ratchet says so
and treats the timing and memory gates as advisory for that comparison. A
machine running two other 700%-CPU jobs cannot answer "is this index slower",
and a ratchet that pretends otherwise is a random number generator with an exit
code. Re-record the baseline on an idle host, or pass `--max-load-ratio 0` to
insist.

## Running it

```bash
# Compare the working tree against benchmarks/baseline.jsonl.
python3 scripts/ratchet.py

# The reliable form when timing gates should actually mean something:
# three runs, per-row median, on an otherwise idle machine.
python3 scripts/ratchet.py --runs 3

# One family, while iterating (missing rows are warnings here).
python3 scripts/ratchet.py --family hnsw

# Compare two recorded runs without rebuilding.
python3 scripts/ratchet.py --current /tmp/rows.jsonl

# Produce the rows yourself, for inspection or for an artifact.
cargo run --release --example all_families -- \
  --features hnsw,nsw,nsg,finger,fresh_graph,filtered_graph,vamana,diskann,ivf_pq,ivf_avq,ivf_rabitq,emg,curator,lsh,pipnn,rp_quant,sq4,binary_index,sng,range_filtered,kdtree,balltree,kmeans_tree,rptree,distance-stats \
  -- --json --out /tmp/rows.jsonl
```

Tolerances can be overridden per run (`--recall-tolerance`,
`--p50-tolerance`, `--p99-tolerance`, `--dists-tolerance`,
`--mem-tolerance`, `--build-tolerance`), but the committed values above are the
policy. A looser tolerance in CI is a policy change and belongs in a commit
message, not in a workflow flag.

## Raising the bar

`--update` rewrites `benchmarks/baseline.jsonl` from the run and prints the
deltas it encodes:

```bash
python3 scripts/ratchet.py --update --save-current /tmp/rows.jsonl
```

Then commit the baseline **with the deltas in the commit message**:

```
bench: re-baseline after contiguous-leaf HNSW build

  hnsw / uniform / ef_search=100     p50   91.2 -> 78.4 us  (-14.0%)
  hnsw / uniform / ef_search=100     build  1.24 -> 0.98 s  (-21.0%)
  hnsw / clustered / ef_search=100   recall 0.9980 -> 0.9980 (0.0%)

Ratchet: scripts/ratchet.py --update
```

A baseline update without its delta is not reviewable: the diff of a JSONL file
does not say whether the bar moved because a change was faster, because a
tolerance was loosened, or because the machine changed. State the reason.

## Re-recording the baseline on the CI runner

The `ratchet` job uploads the exact rows it compared as the `ratchet-rows`
artifact (30 days). To move the baseline onto the runner's hardware:

1. Run the `ratchet` workflow from `workflow_dispatch`.
2. Download the `ratchet-rows` artifact from that run.
3. `python3 scripts/ratchet.py --update --current ratchet-rows.jsonl`
4. Commit it with the deltas, as above, and drop `--advisory-timing` from
   `.github/workflows/ratchet.yml` once the host matches.

## What this does not cover

- Absolute recall floors and correctness. Those stay in `cargo test` and the
  `recall-regression` job, which run on every PR.
- Per-query latency distributions beyond p99, and tail behaviour under
  concurrency. The harness is single-threaded by design.
- Families without a k-NN query path (`evoc`, `lemur`, `sparse_mips`); the
  harness lists them under "Not covered" with the reason.
