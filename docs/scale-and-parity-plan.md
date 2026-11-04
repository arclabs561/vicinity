# Scale and parity plan

Status: 2026-11-02. This is the gap list behind "is vicinity better", and the
order in which the gaps are worth closing. It is deliberately about evidence and
reach, not about adding a twenty-fifth family.

Every claim here is bounded by what the repository can currently show: one
external comparison (`docs/fashion20k-k10.jsonl`, 20k x 784, three repeats) and
self-comparison everywhere else.

## What we cannot currently answer

| question | why not |
| --- | --- |
| Is our HNSW faster than hnswlib? | Rows exist on `clustered16` at 100k and 1M (`benchmarks/hnswlib-parity-*.jsonl`), and against FAISS on the `uniform` corpus (`benchmarks/faiss-parity-*.jsonl`). Neither is a claim about every dataset. |
| Is it faster than FAISS-IVF / Glass / ScaNN at 1M+? | FAISS HNSW and IVF-PQ now have rows at 100k and 1M; FAISS IVF-PQ has no vicinity counterpart at a tier (our `ivf_pq` is not wired). Glass and ScaNN have no binding. |
| Does it hold at 10M and 100M? | 100M is extrapolated: ~200 GiB build peak against 128 GiB installed, so no run exists. |
| Is it competitive when queries carry predicates? | Measured only against our own families; the 2026 baselines (FAVOR, EMA, RNSG) were never run. |
| What happens under updates or concurrency? | No row carries an update rate or a concurrent-reader latency. |
| Does recall hold on out-of-distribution queries? | Every tier is in-distribution synthetic. |
| Can a billion points be served? | No out-of-core build, and ids at 1B only just fit `u32`. |

## The competitive set, in three categories

"Better than other libs" is three different questions, and one table cannot
answer them. Keep them separate and label every row with the category it is in.

**A. Embedded index libraries** — same calling convention as this crate: a
library you link and query in-process. `hnswlib`, FAISS, ScaNN, usearch, Glass,
Annoy, NGT. This is the only category where a latency comparison is
apples-to-apples, because both sides do the same in-process work with no
serialisation or network hop.

**B. Single-node servers and embedded databases** — Qdrant, Milvus, LanceDB,
pgvector, Weaviate, Vespa. These are what a user actually chooses between, but
they answer a different question: end-to-end latency including the client
protocol, plus filtering, persistence, restart, and tenancy. A row measured
against one of these describes a deployment, not an index, and must say so.

**C. Managed services** — Pinecone and equivalents. Excluded: no reproducible
single-node deployment, no control over hardware, no stable version to pin. Say
so rather than measuring a moving target.

The honest split of what each category can and cannot show:

| category | can show | cannot show |
| --- | --- | --- |
| A | kernel and graph quality at equal recall and memory | anything about operations |
| B | whether someone should use this instead of Qdrant for their data | whether the index itself is better |
| C | nothing reproducible | — |

Where this crate is likely behind and should say so: multi-tenancy, replication
and failover, authentication and quotas, hybrid dense+sparse fusion, on-line
schema and payload indexes, and the operational surface of a server. Where it is
plausibly ahead: in-process latency, memory shape, a typed Rust API, and one
crate covering graph, IVF, quantised and tree families rather than a server that
wraps one index. A plan that claims the first group is a plan nobody believes.

## P0a. Embedded libraries on the same protocol

hnswlib is the reference implementation the field quotes, and the only external
library this repo has never measured. Until that row exists, "faster than other
libraries" means "faster than the two we happened to bind".

- **Shim**: `tools/hnswlib-sys`, vendored header plus `cc`, following
  `tools/annoy-sys` exactly (own workspace root, `publish = false`, excluded
  from the package). The existing pattern exists precisely to keep a C++
  dependency out of the published manifest.
- **Harness**: `tools/parity-rows`, a sibling of `tools/annoy-parity` that
  builds both sides on the same vectors, the same `k`, the same `ef` sweep, one
  thread, and reports build seconds as well as query latency.
- **Protocol**: identical to `all_families` — exact top-10 ground truth, 500
  sampled queries at 100k and above, reported as recall against p50.
- Bind **FAISS** (`ivf_pq` at least) and **ScaNN** (via its Python package) on
  the same harness, because HNSW-only parity leaves the quantised families
  uncompared. FAISS HNSW and IVF-PQ are bound (`scripts/faiss_parity.py`, rows
  at 100k and 1M); the local counterpart for IVF-PQ is still missing, so that
  half of the table is one-sided and labelled as such.
- **Gate**: a committed row per library per tier at 100k, 1M and (if it fits)
  3M, with the same `M`, `ef_construction` and `ef_search` sweep and build time
  included. No row, no claim.

Cost: two to three days. Risk: none to the library; the shims are excluded from
the package.

## P0b. Servers on the same corpus

Run each server as a single container on this machine, load the same corpus with
the same ids, and query with the same vectors and `k`. Docker and 128 GiB are
available here, so 100k and 1M fit; 10M is a stretch for the heavier ones.

- **Start with Qdrant**, because it exposes HNSW parameters directly
  (`m`, `ef_construct`, `hnsw_ef`) and therefore allows a like-for-like sweep
  rather than a default-vs-default comparison.
- Then **LanceDB** and **pgvector** (the two embedded-ish options a Rust user
  would weigh), then Milvus if the first three are informative.
- Record: ingest wall clock, end-to-end p50/p99 at fixed recall, resident memory
  after load, on-disk bytes, restart-to-ready seconds, and the filtered matrix
  from P3.
- **Report the client protocol in the row.** A loopback REST or gRPC call is
  part of using a server; subtracting it silently would make the comparison look
  better and mean less.
- Label every such row `category = server` so it is never diffed against an
  in-process row.

Cost: two days for Qdrant, one more for the rest. Risk: the
services will beat us on features and possibly on filtered recall; that is the
result the plan exists to find, not a reason to skip it.

## P1. Every tier, measured rather than extrapolated

The tier ladder currently stops at 3M real rows with 10M, 100M and 1B
extrapolated. All three need to become measurements, and the blocker is
different at each step.

- **10M** fits today (~26 GiB build peak) but no row exists. Gate: a committed
  `large-scale-10m-*.jsonl` from the tier harness.
- **100M is a memory wall, not a disk wall.** The build holds the corpus, the
  graph and the reorder at once. Two changes make it fit: shard the build
  (k-means partitions, per-shard graph, edge-union merge, as DiskANN does), and
  stop holding the full `f32` corpus during construction by quantising at
  insert time. Gate: a 100M build inside 128 GiB, with the peak reported.
- **1B** needs sharding plus the file-backed path for query, since no single
  machine holds 1B resident. Gate: a design note with a measured 100M shard
  build and a stated extrapolation, or a real run on a bigger host.

Cost: 100M sharding is the largest single item in this plan, roughly a week.
The honest alternative is to publish 10M and say 100M needs sharding.

## P2. Memory shape

Resident memory is 2.6x the serialized form at 1M, and half of that is a layout
defect rather than data.

- **Compact upper layers by default** without changing the snapshot format
  (measured: -55% resident, identical recall). Currently opt-in and unreadable
  by the new segment searcher.
- **Layer-0 CSR**, the other half of the defect.
- **A shared, thread-safe block cache.** Every file-backed searcher owns its
  cache and takes `&mut self`, so parallel search multiplies the budget by the
  thread count. Gate: a 16-thread run where resident memory does not scale with
  the thread count.

Cost: 3-4 days total. This is the cheapest real win on the list.

## P3. Filtered search as a matrix, not an anecdote

The filtered claim is currently "our filtered graph beats our own post-filter",
which is not a comparison.

- Sweep **selectivity over orders of magnitude** on real attributes, not random
  predicates.
- Cover the **filter types** the 2025 benchmark uses (equality, range, set
  membership, and their conjunctions).
- Bind at least **one 2026 baseline** (FAVOR or EMA) on the same harness. Both
  are filter-agnostic and both publish numbers at recall 0.95, so a row is
  possible without reimplementing them.
- Gate: a selectivity x filter-type matrix with recall and p50, committed as
  rows, for our families and at least one external.

Cost: 3 days, most of it the external binding.

## P4. Dynamic and concurrent

- Insert and delete **rates**, with recall after churn measured, not just
  asserted.
- **Tail latency under N concurrent readers**, which is the number a server
  actually cares about and which no current row reports.
- Gate: rows with `update_qps`, `recall_after_churn` and p99 under 1/4/16
  threads.

Cost: 2 days if the block cache is shared first (P2), otherwise the concurrency
run measures a cache multiplied by threads.

## P5. Out-of-distribution and multi-vector

- An **OOD track**: queries drawn from a different distribution than the corpus,
  reported separately, because VIBE showed aggregates overstate every method.
- **Multi-vector** (MUVERA-style reduction) is the cheapest way to serve
  late-interaction retrieval with an existing single-vector index; without it,
  that workload is out of scope.

Cost: OOD is 1 day once there is a real Embeddings dataset; multi-vector is a
week if it is wanted at all.

## P6. GPU — a decision, not a backlog item

Graph traversal does not port without a second, CAGRA-shaped index: a
different construction algorithm, different memory layout, different failure
modes. The decision gate is a parity row: if the P0 hnswlib comparison shows we
are within a small factor on CPU, a GPU track buys little for the effort; if it
shows a large gap, the gap is a CPU-kernel problem first. Recommend: no GPU work
before P0-P2 are complete.

## Which of these actually matters

Ordered by expected effect on "is it better":

1. **P0a** — without an hnswlib row there is no claim to make at all.
2. **P2** — the memory win is measured, cheap, and affects every deployment.
   **P0b** joins it: one Qdrant row is what a reader evaluating this against a
   server needs, and it is two days.
3. **P1 at 10M** — one day, and it turns the top real tier from 3M to 10M.
4. **P3** — the differentiator we advertise is the least evidenced.
5. **P4** — what a server operator asks first.
6. **P1 at 100M** — a week, and it changes the answer to "can I use this at
   scale" from no to yes.
7. **P5, P6** — real, but not before the six above.

## What this plan does not promise

Nothing here closes the gap by adding families. The gap is evidence and reach:
measurements against the libraries people already use, and the ability to run at
sizes people actually have.
