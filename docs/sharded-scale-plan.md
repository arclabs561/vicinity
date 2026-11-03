# Sharding past 10M

Status: 2026-11-03. Written because the tier ladder stops at 10M for a reason that
is not disk and not time: a single HNSW build holds the corpus, the graph and the
reorder at once, which measures 2.09 KiB/point. 100M needs about 199 GiB and the
host has 128 GiB.

This page is the plan and the first measurement, not a claim that 100M works.

## What sharding has to buy

| tier | single build | with 16 shards | still blocked by |
| --- | ---: | ---: | --- |
| 10M | ~21 GiB | 1.3 GiB per shard | nothing, 10M already fits |
| 100M | ~199 GiB | 12.4 GiB per shard | total index 45 GiB: query must be file-backed |
| 1B | ~2.0 TiB | 124 GiB per shard | index 450 GiB: query must be file-backed, and shard selection must not scan all shards |

Sharding bounds the *build* peak to one shard. It does not bound the *query*
working set, which is why sharding and `HNSWSegmentSearcher` are two halves of one
answer: the shard is built in RAM, written as a segment, and queried from files.

## The pieces that already exist

- `HNSWSegmentWriter` / `HNSWSegmentSearcher`: a built HNSW index can be written
  and searched from disk with a bounded block cache. Measured at 100k x 96, from
  `docs/storage-audit.md`: 104 µs at a 64 MiB cache (96% hit) against 39-42 µs
  resident.
- `examples/large_scale.rs` streams the corpus from a memory map, so the corpus
  is never a second copy.
- `HNSWIndex::add_slice` accepts a borrowed row, so a shard build can ingest
  directly from the mapping.

## The missing pieces

1. **A partitioner.** k-means over a sample, or a cheaper random projection
   split. The partitioner decides both build balance and query recall: with
   `nprobe = 1`, recall is bounded by how often the true neighbours land in the
   same partition as the query's nearest centroid.
2. **A shard router.** Query the `nprobe` nearest shard centroids, merge their
   results by distance. This is where the recall/latency knob lives.
3. **A harness that persists per-shard segments and reports the peak of one
   shard, not of the run.** Without that separation the measurement would report
   the sum of shard builds and prove nothing.

## Recall model, stated before measuring

For `nprobe = 1`, recall is at most the probability that all `k` true neighbours
share the query's nearest partition. That probability falls as the number of
shards grows and as the data becomes less clustered. Uniform 96-d data is the
worst case and the one already known to be hard: recall at a fixed `ef` collapses
with `n` on that domain. A sharded index on uniform data should therefore be
expected to need `nprobe` well above 1, and the honest way to report it is a
`nprobe` sweep, not a single number.

## First measurement

`examples/sharded_tier.rs` builds `K` partitions with a k-means pass, writes each
shard as a segment, and queries `nprobe` of them from files. It reports, per
`nprobe` and `ef`, recall against an exact scan, p50, and the peak RSS of a single
shard build. Rows go to `benchmarks/sharded-<n>.jsonl`.

The comparison that matters is against the monolithic row at the same tier and
the same memory budget, not against nothing.

## Order of work

1. Prototype at 100k and 1M with `K = 16`: does `nprobe = 2` reach the monolithic
   recall at comparable p50? Measured, committed as rows.
2. If yes, run 100M with 16 shards and a bounded per-shard build. The gate is a
   completed build inside 128 GiB with the peak reported.
3. 1B needs one more thing: shard selection that does not read every centroid's
   vector per query is fine (450 KiB of centroids), but the *ground truth* for a
   1B evaluation is a full scan of 358 GiB. That is the real 1B cost, not the
   graph: a sampled evaluation or a shard-local truth is the only affordable
   option, and any 1B recall claim has to say which one it used.
