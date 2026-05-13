from __future__ import annotations

import logging

import torch
from sentence_transformers import SentenceTransformer

from utils.config_loader import config
from utils.ensure_model import resolve_embedding_model_source


logger = logging.getLogger(__name__)


class EmbeddingComponent:
    def __init__(self) -> None:
        embedding_cfg = config.models.embedding
        source = resolve_embedding_model_source()
        normalize = bool(getattr(embedding_cfg, "normalize_embeddings", True))
        device = getattr(embedding_cfg, "device", None)
        if isinstance(device, str):
            normalized_device = device.strip().lower()
            if normalized_device in {"gpu", "cuda"}:
                device = "cuda"
            elif normalized_device in {"cpu", "auto", ""}:
                device = normalized_device or None

        logger.info("Loading embedding model %s", source)
        self.model = SentenceTransformer(source)
        if device:
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested for embeddings but no NVIDIA driver/GPU was detected; falling back to CPU")
                device = "cpu"
            self.model = self.model.to(device)
        self.normalize = normalize

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding.tolist()
