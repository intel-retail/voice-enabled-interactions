"""IdentityService — orchestration layer.

Owns the challenge provider, the SQLite loyalty-profile repository, and the FAISS
index managers (face + voice).  Routers call this service only; they never touch
SQLite, FAISS, or inference directly.

Phase 3 status: storage layer (SQLite + FAISS) is wired and ``challenge`` /
``stats`` are functional.  ``verify`` and ``register`` remain gated on the
inference pipeline (Phases 4–6) since they need OpenVINO embeddings.
"""

from __future__ import annotations

import logging

from identity_core.challenge import ChallengeProvider
from identity_core.config import Settings
from identity_core.inference.factory import build_engines
from identity_core.models import (
    ChallengeResponse,
    RegisterRequest,
    RegisterResponse,
    StatsResponse,
    VerifyRequest,
    VerifyResponse,
)
from identity_core.persistence.db import get_db, init_db
from identity_core.persistence.faiss_index import FaissIndexManager
from identity_core.persistence.repository import SqliteProfileRepository

logger = logging.getLogger(__name__)


class IdentityService:
    """Coordinates biometric enrolment and verification."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._db_path = settings.db_path
        self._challenge = ChallengeProvider(settings.prompts)

        # ── Storage layer (Phase 3) ──────────────────────────────────────────
        self._face_index = FaissIndexManager(
            dim=settings.face_embedding_dim,
            index_path=settings.face_index_path,
            name="face",
        )
        self._voice_index = FaissIndexManager(
            dim=settings.voice_embedding_dim,
            index_path=settings.voice_index_path,
            name="voice",
        )

        # ── Inference layer (Phase 4) ────────────────────────────────────────
        # Engines are built from the model files mounted at settings.models_dir.
        # If any model is missing they degrade to None and the service still
        # serves /health, /challenge and /stats (inference_ready=False).
        self._engines = build_engines(settings)
        self._inference_ready = self._engines.ready

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def init_storage(self) -> None:
        """Bootstrap the shared SQLite schema (idempotent)."""
        await init_db(self._db_path)
        logger.info(
            "[IDENTITY] Storage ready — face_index=%d voice_index=%d",
            self._face_index.size,
            self._voice_index.size,
        )

    # ── Challenge ────────────────────────────────────────────────────────────

    def get_challenge(self) -> ChallengeResponse:
        challenge_id, prompt_text = self._challenge.next_challenge()
        logger.debug("[IDENTITY] Issued challenge %s", challenge_id)
        return ChallengeResponse(challenge_id=challenge_id, prompt_text=prompt_text)

    # ── Stats ────────────────────────────────────────────────────────────────

    async def get_stats(self) -> StatsResponse:
        async with get_db(self._db_path) as db:
            repo = SqliteProfileRepository(db)
            profiles = await repo.count()
        return StatsResponse(
            profiles=profiles,
            face_index_size=self._face_index.size,
            voice_index_size=self._voice_index.size,
            inference_ready=self._inference_ready,
        )

    # ── Verify ───────────────────────────────────────────────────────────────

    async def verify(self, request: VerifyRequest) -> VerifyResponse:
        """Run multimodal verification.  Both modalities are required."""
        if not request.image_base64 or not request.audio_base64:
            return VerifyResponse(
                verified=False,
                reason="Both face (image) and voice (audio) are required.",
            )
        if not self._inference_ready:
            return VerifyResponse(
                verified=False,
                reason="Inference engines unavailable (models not loaded).",
            )
        # Engines are loaded (Phase 4).  The decode → embed → FAISS search →
        # fusion pipeline is wired in Phases 5–6; until then report not-ready
        # rather than raising so the endpoint stays well-behaved.
        # Phases 5–6 will:
        #   1. decode image/audio
        #   2. self._engines.face.embed(...) / self._engines.voice.embed(...)
        #   3. FAISS search both indices
        #   4. fuse: 0.6*face + 0.4*voice vs combined_threshold
        #   5. load profile from SQLite, return payload
        return VerifyResponse(
            verified=False,
            reason="Verification pipeline pending (Phase 5–6).",
        )

    # ── Register (admin / bootstrap) ─────────────────────────────────────────

    async def register(self, request: RegisterRequest) -> RegisterResponse:
        if not self._inference_ready:
            return RegisterResponse(
                user_id=request.user_id,
                registered=False,
                reason="Inference engines unavailable (models not loaded).",
            )
        # Enrolment pipeline (embed → FAISS add → SQLite insert) lands in Phase 5.
        return RegisterResponse(
            user_id=request.user_id,
            registered=False,
            reason="Registration pipeline pending (Phase 5).",
        )
