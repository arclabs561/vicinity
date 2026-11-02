# qdrant-parity

Same-corpus comparison between an in-process `vicinity` HNSW and a Qdrant
server, so the two are not separated by a generator that is only meant to agree.

Two halves, one corpus:

- `scripts/qdrant_parity.py` (repository root) generates 100k x 96 unit vectors
  and 200 queries from a fixed LCG, loads them into a local Qdrant container,
  and times end-to-end HTTP searches across an `hnsw_ef` sweep.
- this crate reads the same two `.f32` files, builds `vicinity`'s HNSW with the
  same `m = 16` and `ef_construction = 200`, and times the same sweep in process.

## Running it

```bash
# 1. a container on loopback
docker run -d --rm --name vic-qdrant -p 6333:6333 qdrant/qdrant:latest

# 2. server rows, and the corpus files the other half needs
python3 scripts/qdrant_parity.py benchmarks/qdrant-parity-100k.jsonl

# 3. in-process rows on the same bytes
PARITY_DIM=96 cargo run --release --manifest-path tools/qdrant-parity/Cargo.toml \
    -- corpus.f32 queries.f32 benchmarks/qdrant-parity-100k-vicinity.jsonl
```

The Python half writes `corpus.f32` and `queries.f32` next to its output path in
the current directory; keep them together.

## Reading the rows

Every row carries `category`: `server` for Qdrant, `embedded` for this crate.
**Never diff one against the other as an index comparison.** The server row
includes the client protocol and a loopback hop, which is what using a server
costs and therefore belongs in the number, but it is not the index's search
time. Qdrant's p50 is nearly flat across `ef` (1.4 ms at `ef = 10` to 2.6 ms at
`ef = 400`) while the in-process p50 scales with it (19 µs to 400 µs), which is
the transport floor showing through.

## What the committed rows say

`benchmarks/qdrant-parity-100k.jsonl` and `...-vicinity.jsonl`, on 100k x 96
uniform, cosine, k = 10:

| ef | vicinity recall | Qdrant recall (3 runs) | vicinity p50 | Qdrant p50 (3 runs) |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.1795 | 0.1730 (0.155–0.181) | 19 µs | 1531 µs (1138–1698) |
| 100 | 0.6265 | 0.6165 (0.614–0.627) | 107 µs | 1610 µs (1341–2025) |
| 400 | 0.8850 | 0.8800 (0.879–0.883) | 400 µs | 2626 µs (2006–2776) |

Equal recall within one or two results out of two thousand, and a latency gap
that is mostly transport. The Qdrant column is a median of three runs with the
range in brackets, because a single server run is not a measurement: its own
recall moved 0.026 at `ef = 10` and its p50 moved 49% between runs on an
unloaded host.

## Caveats

- One container, one node, default optimizers, no replication, no payload
  filters. This measures a deployment nobody would ship as-is.
- Qdrant's own numbers moved between runs because segment optimization changes
  the index and its posting lists between runs. The committed rows are the median
  of three runs, with `recall_min`/`recall_max` and `p50_min`/`p50_max` recorded
  in each row so the spread travels with the number. `vicinity`'s half is seeded
  and deterministic, so a re-run of that half reproduces exactly.
- Uniform 96-dimensional data is close to the worst case for graph recall; both
  sides score low at small `ef` for that reason, not because either is broken.
