"""FAISS index manager for biometric embeddings.

Wraps a ``faiss.IndexFlatIP`` (exact inner-product search).  Because all vectors
are L2-normalized before insertion, inner product equals cosine similarity in
``[-1, 1]``.  Each ``add`` returns the row offset, which is used as the stable
``*_faiss_id`` persisted in SQLite (we never delete, so offsets are stable).

Indices are persisted to disk (``.bin``) after every write so a container
restart restores all enrolled vectors.  A threading lock guards mutation since
FAISS objects are not safe for concurrent writers.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FaissIndexManager:
    """Manages a single L2-normalized inner-product FAISS index on disk."""

    def __init__(self, dim: int, index_path: str | Path, name: str = "index"):
        self._dim = dim
        self._path = Path(index_path)
        self._name = name
        self._lock = threading.Lock()
        self._index = self._load_or_create()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def _load_or_create(self) -> "faiss.Index":
        if self._path.exists():
            index = faiss.read_index(str(self._path))
            if index.d != self._dim:
                raise ValueError(
                    f"[FAISS:{self._name}] On-disk index dim {index.d} != configured "
                    f"dim {self._dim} ({self._path}). Delete the stale index or fix config."
                )
            logger.info(
                "[FAISS:%s] Loaded %d vector(s) from %s", self._name, index.ntotal, self._path
            )
            return index
        logger.info("[FAISS:%s] Creating new IndexFlatIP(dim=%d)", self._name, self._dim)
        return faiss.IndexFlatIP(self._dim)

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._path))

    # ── helpers ──────────────────────────────────────────────────────────────

    def _prepare(self, vector: np.ndarray) -> np.ndarray:
        """Validate, cast to float32, reshape to (1, dim), and L2-normalize."""
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if vec.shape[1] != self._dim:
            raise ValueError(
                f"[FAISS:{self._name}] Expected dim {self._dim}, got {vec.shape[1]}"
            )
        faiss.normalize_L2(vec)
        return vec

    # ── public API ─────────────────────────────────────────────────────────--

    def add(self, vector: np.ndarray) -> int:
        """Add an embedding and return its FAISS offset (the ``*_faiss_id``)."""
        with self._lock:
            vec = self._prepare(vector)
            faiss_id = int(self._index.ntotal)
            self._index.add(vec)
            self._persist()
            logger.debug("[FAISS:%s] Added vector at offset %d", self._name, faiss_id)
            return faiss_id

    def search(self, vector: np.ndarray, k: int = 1) -> tuple[int, float]:
        """Return ``(offset, similarity)`` for the nearest neighbour.

        Returns ``(-1, 0.0)`` when the index is empty.
        """
        with self._lock:
            if self._index.ntotal == 0:
                return -1, 0.0
            vec = self._prepare(vector)
            scores, idxs = self._index.search(vec, k)
            return int(idxs[0][0]), float(scores[0][0])

    @property
    def size(self) -> int:
        return int(self._index.ntotal)
