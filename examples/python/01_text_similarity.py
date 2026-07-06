# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pyvicinity",
#   "numpy",
#   "sentence-transformers>=3.0",
# ]
# ///
"""Semantic search over a small text corpus -- the canonical ANN use case.

Embeds 24 short sentences with a tiny pretrained model (all-MiniLM-L6-v2,
~22MB), indexes them with HNSW, and runs a few real queries. First run
will download the model; subsequent runs use the cache.

Run with:

    uv run examples/python/01_text_similarity.py
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from pyvicinity import DistanceMetric, HNSWIndex

CORPUS = [
    # animals
    "Border collies are widely considered the smartest dog breed.",
    "House cats descend from African wildcats domesticated ~10,000 years ago.",
    "Octopuses can solve puzzles and open childproof jars.",
    "Honeybees communicate flower locations via the waggle dance.",
    "Crows pass tools and grudges across generations.",
    # food
    "Sourdough bread relies on wild yeast and lactic-acid bacteria.",
    "Espresso is brewed under ~9 bar of pressure for ~25 seconds.",
    "Tomatoes are botanically fruits but legally vegetables in the US.",
    "Miso paste ferments soybeans with the koji mold Aspergillus oryzae.",
    "Saffron is harvested by hand from crocus flowers.",
    # space
    "The Voyager 1 probe entered interstellar space in 2012.",
    "A neutron star can spin hundreds of times per second.",
    "Jupiter's Great Red Spot has been raging for at least 400 years.",
    "The Sun loses 4 million tons of mass per second to fusion.",
    "Saturn would float if you could find a bathtub big enough.",
    # programming
    "Rust's borrow checker prevents data races at compile time.",
    "Python's GIL serializes bytecode execution per interpreter.",
    "HNSW builds a hierarchical small-world graph for ANN search.",
    "Vector databases store dense embeddings indexed by ANN.",
    "Trusted publishing on PyPI removes long-lived API tokens.",
    # geography
    "The Mariana Trench reaches almost 11 km below sea level.",
    "Antarctica is the largest desert on Earth by area.",
    "Iceland sits on the Mid-Atlantic Ridge and is splitting in two.",
    "Mount Everest grows ~4 mm taller each year.",
]


def main() -> None:
    print("loading model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(CORPUS, convert_to_numpy=True).astype(np.float32)
    n, dim = embeddings.shape
    print(f"embedded {n} sentences -> dim={dim}")

    index = HNSWIndex(
        dim=dim,
        metric=DistanceMetric.Cosine,
        auto_normalize=True,
        seed=0,
    )
    index.add_items(embeddings)
    index.build()

    queries = [
        "Which animal is unusually clever?",
        "How do you brew good coffee?",
        "Tell me something about deep oceans.",
        "What's the deal with HNSW?",
    ]
    query_vecs = model.encode(queries, convert_to_numpy=True).astype(np.float32)

    for q, qv in zip(queries, query_vecs, strict=True):
        print(f"\n>> {q}")
        ids, dists = index.search(qv, k=3)
        for rank, (i, d) in enumerate(
            zip(ids.tolist(), dists.tolist(), strict=True), 1
        ):
            print(f"  {rank}. ({d:+.3f}) {CORPUS[i]}")


if __name__ == "__main__":
    main()
