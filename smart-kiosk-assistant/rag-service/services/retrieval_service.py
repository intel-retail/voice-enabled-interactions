"""Retrieval: similarity search against the vector store, returning typed
``RetrievalRecord`` instances filtered by score threshold and top-k.

When a reranker is configured, ``fetch_k`` documents are pulled by ANN, then
re-scored with a cross-encoder, and the top-k by rerank score are returned.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

from utils.latency_store import retrieval_latency

from .types import RetrievalRecord


logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
        self,
        vectorstore_provider: Callable[[], object],
        top_k: int,
        fetch_k: int,
        score_threshold: float | None,
        reranker=None,
        rerank_score_threshold: float | None = None,
    ) -> None:
        self._vectorstore_provider = vectorstore_provider
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.score_threshold = score_threshold
        self.rerank_score_threshold = rerank_score_threshold
        self._reranker = reranker

    @property
    def vectorstore(self):
        return self._vectorstore_provider()

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievalRecord]:
        desired_k = top_k or self.top_k
        ann_k = max(desired_k, self.fetch_k)

        t_ann0 = time.monotonic()
        docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=ann_k)
        t_ann_ms = (time.monotonic() - t_ann0) * 1000

        candidates: list[tuple[object, float | None]] = []
        for document, score in docs_with_scores:
            if (
                self.score_threshold is not None
                and score is not None
                and score > self.score_threshold
            ):
                continue
            candidates.append((document, score))

        t_rr_ms = 0.0
        if self._reranker is not None and len(candidates) > 1:
            passages = [doc.page_content for doc, _ in candidates]
            t_rr0 = time.monotonic()
            rerank_scores = self._reranker.score(question, passages)
            t_rr_ms = (time.monotonic() - t_rr0) * 1000
            order = sorted(range(len(candidates)), key=lambda i: rerank_scores[i], reverse=True)
            reordered: list[tuple[object, float | None]] = []
            dropped = 0
            for i in order[:desired_k]:
                rr_score = float(rerank_scores[i])
                # The cross-encoder emits a relevance logit: strongly negative
                # means "this passage does not answer the query". Without this
                # floor a no-match passage is still handed to the LLM as
                # authoritative context, which is how an unrelated section gets
                # reported as fact. Filtering here adds no compute — the scores
                # are already known.
                if (
                    self.rerank_score_threshold is not None
                    and rr_score < self.rerank_score_threshold
                ):
                    dropped += 1
                    continue
                doc, _ann_score = candidates[i]
                reordered.append((doc, rr_score))
            logger.info(
                "[RETRIEVAL] q=%r ann_k=%d ann=%.1fms rerank=%.1fms top_k=%d top_score=%.4f "
                "dropped_below_threshold=%d",
                question[:80], ann_k, t_ann_ms, t_rr_ms, len(reordered),
                rerank_scores[order[0]] if order else 0.0, dropped,
            )
            if dropped and not reordered:
                logger.warning(
                    "[RETRIEVAL] All %d candidate(s) scored below rerank threshold %.2f for q=%r "
                    "— returning no context",
                    dropped, self.rerank_score_threshold, question[:80],
                )
            candidates = reordered
        else:
            candidates = candidates[:desired_k]
            logger.info(
                "[RETRIEVAL] q=%r ann_k=%d ann=%.1fms top_k=%d (no rerank)",
                question[:80], ann_k, t_ann_ms, len(candidates),
            )

        records: list[RetrievalRecord] = []
        for document, score in candidates:
            records.append(
                RetrievalRecord(
                    source=str(document.metadata.get("source", "context")),
                    content=document.page_content,
                    score=float(score) if score is not None else None,
                    metadata=document.metadata,
                )
            )
        retrieval_latency.record(t_ann_ms + t_rr_ms)
        return records

