import json, math, platform, sys, time, urllib.request

BASE = "http://127.0.0.1:6333"
DIM, N, NQ, K = 96, 100_000, 200, 10
SEED = 0x46414D494C59
EFS = [10, 20, 50, 100, 200, 400]

def lcg(seed):
    s = seed | 1
    while True:
        s = (s * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
        yield (s >> 33) / (1 << 31) - 1.0

def vectors(count, offset):
    gen = lcg(SEED ^ offset)
    out = []
    for _ in range(count):
        v = [next(gen) for _ in range(DIM)]
        nrm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / nrm for x in v])
    return out

def req(method, path, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else {}

def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else None
    corpus = vectors(N, 0)
    queries = vectors(NQ, 0x9E37)

    # Dump the exact bytes both halves read, so the in-process side is measured
    # on the same vectors rather than on a Python-side regenerated guess.
    import array
    for name, data in (("corpus", corpus), ("queries", queries)):
        with open(f"{name}.f32", "wb") as handle:
            array.array("f", [x for v in data for x in v]).tofile(handle)
        print(f"wrote {name}.f32", file=sys.stderr)

    req("DELETE", "/collections/vic_parity")
    req("PUT", "/collections/vic_parity", {
        "vectors": {"size": DIM, "distance": "Cosine"},
        "hnsw_config": {"m": 16, "ef_construct": 200},
        "optimizers_config": {"default_segment_number": 2},
    })

    t0 = time.time()
    batch = 2000
    for i in range(0, N, batch):
        chunk = corpus[i:i + batch]
        req("PUT", "/collections/vic_parity/points?wait=false", {
            "points": [{"id": i + j, "vector": v} for j, v in enumerate(chunk)]
        })
    # Force indexing to finish before timing queries.
    while True:
        info = req("GET", "/collections/vic_parity")["result"]
        if info["status"] in ("green",) and info.get("points_count") == N:
            break
        time.sleep(1)
    ingest = time.time() - t0

    # exact top-k by cosine on normalized vectors (dot product)
    gt = []
    for q in queries:
        scored = sorted(((sum(a * b for a, b in zip(q, v)), i) for i, v in enumerate(corpus)),
                        reverse=True)[:K]
        gt.append({i for _, i in scored})

    rows = []
    for ef in EFS:
        lat, hits = [], 0
        for qi, q in enumerate(queries):
            body = {"vector": q, "limit": K, "with_payload": False,
                    "params": {"hnsw_ef": ef, "exact": False}}
            t = time.time()
            res = req("POST", "/collections/vic_parity/points/search", body)
            lat.append((time.time() - t) * 1e6)
            hits += len({p["id"] for p in res["result"]} & gt[qi])
        lat.sort()
        rows.append({"family": "qdrant", "dataset": f"uniform@{N}", "knob": "ef_search",
                     "knob_value": ef, "recall": hits / (NQ * K),
                     "recall_at_10": hits / (NQ * K),
                     "p50_us": round(lat[len(lat) // 2], 1),
                     "p95_us": round(lat[int(len(lat) * 0.95)], 1),
                     "p99_us": round(lat[int(len(lat) * 0.99)], 1),
                     "qps": round(NQ / (sum(lat) / 1e6), 1),
                     "build_s": round(ingest, 2), "dim": DIM, "category": "server"})
        print(json.dumps(rows[-1]), flush=True)

    if out_path:
        # Provenance first, including the two facts that make these numbers
        # different from an embedded row: latency carries one HTTP round trip
        # per query, and recall is against this script's exact scan.
        meta = {
            "_meta": True,
            "schema": 1,
            "harness": "qdrant_parity",
            "host": platform.node(),
            "n": N,
            "dim": DIM,
            "k": K,
            "queries": NQ,
            "ef_sweep": EFS,
            "exact_queries": NQ,
            "latency": "client-side per query, includes one HTTP round trip",
            "server": "qdrant, hnsw_ef per request, exact=false",
            "vicinity": "measured separately in qdrant-parity-100k-vicinity.jsonl",
        }
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta, sort_keys=True) + "\n")
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"wrote {len(rows)} rows to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
