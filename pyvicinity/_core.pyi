"""Type stubs for the native ``pyvicinity._core`` extension module.

Hand-written because PyO3 doesn't generate stubs. Verified against the
compiled module by ``mypy.stubtest`` in CI -- keep in sync with
``src/python.rs`` or stubtest will fail.
"""

from __future__ import annotations

from os import PathLike
from typing import ClassVar, final

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MISSING_DISTANCE",
    "MISSING_LABEL",
    "DistanceMetric",
    "HNSWIndex",
    "IVFPQFileSearcher",
    "IVFPQIndex",
    "__version__",
]

__version__: str

MISSING_LABEL: int
"""Sentinel value padded into ``batch_search`` ID rows shorter than ``k``.

Equal to ``-1``, matching faiss's convention. Mask with::

    valid = ids[ids != pyvicinity.MISSING_LABEL]
"""

MISSING_DISTANCE: float
"""Sentinel value padded into ``batch_search`` distance rows shorter than ``k``.

Equal to ``math.inf``."""

@final
class DistanceMetric:
    """Distance metric for vector comparison.

    Not a true ``enum.Enum`` (PyO3 emits a plain class with ``int``-comparable
    members), but works the same way for ``==`` and pattern matching.
    """

    L2: ClassVar[DistanceMetric]
    """Euclidean (L2) distance."""

    Cosine: ClassVar[DistanceMetric]
    """Cosine distance: ``1 - cos(a, b)``. Inputs are expected to be
    L2-normalized; pass ``auto_normalize=True`` to the index to handle
    raw vectors on both insert and query."""

    Angular: ClassVar[DistanceMetric]
    """Angular distance: ``arccos(cos(a, b)) / pi``, in ``[0, 1]``.
    Computes norms internally; raw vectors are fine."""

    InnerProduct: ClassVar[DistanceMetric]
    """Inner-product distance: ``-dot(a, b)`` (for MIPS).
    Not normalized; query magnitude affects ranking by design."""

    __hash__: None  # type: ignore[assignment]  # eq_int enums are unhashable
    def __eq__(self, value: object, /) -> bool: ...
    def __ne__(self, value: object, /) -> bool: ...
    def __int__(self) -> int: ...

@final
class HNSWIndex:
    """HNSW index for approximate nearest-neighbor search.

    Example::

        import numpy as np
        from pyvicinity import HNSWIndex, DistanceMetric

        vectors = np.random.randn(10_000, 128).astype(np.float32)
        index = HNSWIndex(dim=128, metric=DistanceMetric.Cosine, auto_normalize=True)
        index.add_items(vectors)
        index.build()
        ids, dists = index.search(vectors[0], k=10)
    """

    def __new__(
        cls,
        dim: int,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        metric: DistanceMetric = ...,
        auto_normalize: bool = False,
        seed: int | None = None,
    ) -> HNSWIndex: ...
    def add_items(
        self,
        vectors: NDArray[np.float32],
        ids: NDArray[np.int64] | None = None,
    ) -> None:
        """Add a batch of vectors.

        Args:
            vectors: 2-D ``(n, dim)`` float32 array, C-contiguous.
            ids: Optional 1-D int64 array of length n. Each value must be
                in ``[0, 2**32)`` (vicinity stores IDs as u32 internally).
                If omitted, sequential IDs are assigned starting at the
                current ``len(index)``.
        """

    def build(self) -> None:
        """Finalize the index. Must be called before any search."""

    def set_ef_search(self, ef: int) -> None:
        """Set the default ``ef_search`` parameter for subsequent queries."""

    def save(self, path: str | PathLike[str]) -> None:
        """Save this index to a JSON snapshot file."""

    @staticmethod
    def load(path: str | PathLike[str]) -> HNSWIndex:
        """Load an index from a JSON snapshot file written by :meth:`save`."""

    def search(
        self,
        query: NDArray[np.float32],
        k: int,
        ef: int | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Search for the k nearest neighbors of one query vector.

        Args:
            query: 1-D ``(dim,)`` float32 array.
            k: Number of neighbors to return.
            ef: Search width. Defaults to ``self.ef_search``.

        Returns:
            ``(ids, distances)``: 1-D arrays of length at most k.
            Shorter than k only when the index has fewer than k vectors.
        """

    def batch_search(
        self,
        queries: NDArray[np.float32],
        k: int,
        ef: int | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Search for the k nearest neighbors of each query.

        Args:
            queries: 2-D ``(nq, dim)`` float32 array, C-contiguous.
            k: Number of neighbors per query.
            ef: Search width. Defaults to ``self.ef_search``.

        Returns:
            ``(ids, distances)``: 2-D arrays of shape ``(nq, k)``. Rows
            with fewer than k results are padded with :data:`MISSING_LABEL`
            and :data:`MISSING_DISTANCE` so the result is rectangular.
        """

    @property
    def num_vectors(self) -> int:
        """Number of vectors currently in the index."""

    @property
    def dimension(self) -> int:
        """Vector dimension."""

    @property
    def metric(self) -> DistanceMetric:
        """Distance metric this index was built with."""

    @property
    def auto_normalize(self) -> bool:
        """Whether inserts and queries are L2-normalized internally."""

    @property
    def m(self) -> int:
        """Max connections per node (the ``M`` HNSW parameter)."""

    @property
    def ef_construction(self) -> int:
        """Search width during construction."""

    @property
    def ef_search(self) -> int:
        """Default search width for queries."""

    @property
    def memory_usage_bytes(self) -> int:
        """Estimated memory used by this index, in bytes."""

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class IVFPQIndex:
    """IVF-PQ index for compressed approximate nearest-neighbor search."""

    def __new__(
        cls,
        dim: int,
        num_clusters: int = 256,
        num_codebooks: int | None = None,
        codebook_size: int = 256,
        nprobe: int = 1,
        use_opq: bool = False,
        seed: int = 42,
    ) -> IVFPQIndex: ...
    def add_items(
        self,
        vectors: NDArray[np.float32],
        ids: NDArray[np.int64] | None = None,
    ) -> None:
        """Add a batch of dense vectors."""

    def build(
        self,
        training_sample_size: int | None = None,
        kmeans_max_iter: int = 100,
    ) -> None:
        """Finalize the index."""

    def set_nprobe(self, nprobe: int) -> None:
        """Set the default number of IVF clusters scanned per query."""

    def compact(self) -> None:
        """Drop raw f32 vectors. Approximate search still works."""

    def save(self, path: str | PathLike[str]) -> None:
        """Save this index to a directory snapshot."""

    @staticmethod
    def load(path: str | PathLike[str]) -> IVFPQIndex:
        """Load an index from a directory snapshot written by :meth:`save`."""

    def search(
        self,
        query: NDArray[np.float32],
        k: int,
        nprobe: int | None = None,
        rerank_pool: int | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Search one query vector."""

    def batch_search(
        self,
        queries: NDArray[np.float32],
        k: int,
        nprobe: int | None = None,
        rerank_pool: int | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Search a batch of query vectors."""

    @property
    def num_vectors(self) -> int:
        """Number of vectors currently in the index."""

    @property
    def dimension(self) -> int:
        """Vector dimension."""

    @property
    def num_clusters(self) -> int:
        """Number of IVF coarse clusters."""

    @property
    def num_codebooks(self) -> int:
        """Number of PQ codebooks."""

    @property
    def codebook_size(self) -> int:
        """Number of centroids per PQ codebook."""

    @property
    def nprobe(self) -> int:
        """Default IVF clusters scanned per query."""

    @property
    def use_opq(self) -> bool:
        """Whether OPQ rotation is enabled."""

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class IVFPQFileSearcher:
    """File-backed IVF-PQ searcher for saved index directories."""

    @staticmethod
    def load(path: str | PathLike[str], mmap: bool = False) -> IVFPQFileSearcher:
        """Load a file-backed searcher from a directory written by
        :meth:`IVFPQIndex.save`.

        ``mmap=True`` uses read-only memory maps and requires a build with
        the Rust ``persistence`` feature. PyPI wheels enable it.
        """

    def set_nprobe(self, nprobe: int) -> None:
        """Set the default number of IVF clusters scanned per query."""

    def search(
        self,
        query: NDArray[np.float32],
        k: int,
        nprobe: int | None = None,
        rerank_pool: int | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Search one query vector."""

    def batch_search(
        self,
        queries: NDArray[np.float32],
        k: int,
        nprobe: int | None = None,
        rerank_pool: int | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Search a batch of query vectors."""

    @property
    def num_vectors(self) -> int:
        """Number of vectors currently in the index."""

    @property
    def dimension(self) -> int:
        """Vector dimension."""

    @property
    def num_clusters(self) -> int:
        """Number of IVF coarse clusters."""

    @property
    def num_codebooks(self) -> int:
        """Number of PQ codebooks."""

    @property
    def codebook_size(self) -> int:
        """Number of centroids per PQ codebook."""

    @property
    def nprobe(self) -> int:
        """Default IVF clusters scanned per query."""

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
