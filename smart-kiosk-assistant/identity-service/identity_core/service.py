"""IdentityService — orchestration layer.

Owns the challenge provider, the SQLite loyalty-profile repository, and the FAISS
index managers (face + voice).  Routers call this service only; they never touch
SQLite, FAISS, or inference directly.

Phase 3 status: storage layer (SQLite + FAISS) is wired and ``challenge`` /
``stats`` are functional.  ``verify`` and ``register`` remain gated on the
inference pipeline (Phases 4–6) since they need OpenVINO embeddings.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from identity_core.challenge import ChallengeProvider
from identity_core.config import Settings
from identity_core.inference.factory import build_engines
from identity_core.media import MediaDecodeError, decode_image_bgr, decode_wav
from identity_core.models import (
    ChallengeResponse,
    LoyaltyProfile,
    RegisterRequest,
    RegisterResponse,
    StatsResponse,
    VerifyRequest,
    VerifyResponse,
)
from identity_core.persistence.db import get_db, init_db
from identity_core.persistence.faiss_index import FaissIndexManager
from identity_core.persistence.repository import ProfileRecord, SqliteProfileRepository

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
        """Run biometric verification over the provided modalities.

        At least one biometric (image or audio) must be supplied.  Each modality
        is embedded and searched against its FAISS index; the nearest offset is
        resolved back to a loyalty profile.  When both modalities are present
        the similarities are fused (``0.6*face + 0.4*voice``) against the
        combined threshold; a single modality is compared to its own threshold.
        A FAISS hit with no backing SQLite row is logged and treated as no
        match.  No-match is a normal outcome — it never raises.
        """
        if not self._inference_ready:
            return VerifyResponse(
                verified=False,
                reason="Inference engines unavailable (models not loaded).",
            )
        if not request.image_base64 and not request.audio_base64:
            return VerifyResponse(
                verified=False,
                reason="At least one biometric (image or audio) is required.",
            )

        face_sim: float | None = None
        voice_sim: float | None = None
        face_record = None
        voice_record = None

        async with get_db(self._db_path) as db:
            repo = SqliteProfileRepository(db)
            try:
                if request.image_base64:
                    face_vector = self._engines.face.embed(
                        decode_image_bgr(request.image_base64)
                    )
                    if face_vector is None:
                        return VerifyResponse(
                            verified=False,
                            reason="No face detected in the provided image.",
                        )
                    offset, sim = self._face_index.search(face_vector)
                    if offset >= 0:
                        face_sim = sim
                        face_record = await repo.get_by_face_faiss_id(offset)
                        if face_record is None:
                            logger.warning(
                                "[IDENTITY] Face FAISS hit offset=%d has no profile "
                                "row (orphan); treating as no match",
                                offset,
                            )
                if request.audio_base64:
                    waveform, sample_rate = decode_wav(request.audio_base64)
                    voice_vector = self._engines.voice.embed(waveform, sample_rate)
                    offset, sim = self._voice_index.search(voice_vector)
                    if offset >= 0:
                        voice_sim = sim
                        voice_record = await repo.get_by_voice_faiss_id(offset)
                        if voice_record is None:
                            logger.warning(
                                "[IDENTITY] Voice FAISS hit offset=%d has no profile "
                                "row (orphan); treating as no match",
                                offset,
                            )
            except MediaDecodeError as exc:
                return VerifyResponse(
                    verified=False,
                    reason=f"Could not decode biometric media: {exc}",
                )

        def _ok(record, f_sim, v_sim, fused) -> VerifyResponse:
            logger.info(
                "[IDENTITY] Verified user_id=%s face=%s voice=%s fused=%s",
                record.user_id,
                f_sim,
                v_sim,
                fused,
            )
            return VerifyResponse(
                verified=True,
                user_id=record.user_id,
                profile=LoyaltyProfile(
                    user_id=record.user_id,
                    name=record.name,
                    favorites=record.favorites,
                    restrictions=record.restrictions,
                ),
                face_similarity=f_sim,
                voice_similarity=v_sim,
                fused_score=fused,
            )

        settings = self._settings

        # Both modalities present → weighted fusion (requires the same profile).
        if request.image_base64 and request.audio_base64:
            if (
                face_record is not None
                and voice_record is not None
                and face_record.user_id == voice_record.user_id
            ):
                fused = (
                    settings.fusion_face_weight * face_sim
                    + settings.fusion_voice_weight * voice_sim
                )
                if fused >= settings.combined_threshold:
                    return _ok(face_record, face_sim, voice_sim, fused)
                logger.info(
                    "[IDENTITY] Verify REJECTED user_id=%s face=%s voice=%s "
                    "fused=%s < combined_threshold=%s",
                    face_record.user_id,
                    face_sim,
                    voice_sim,
                    fused,
                    settings.combined_threshold,
                )
                return VerifyResponse(
                    verified=False,
                    face_similarity=face_sim,
                    voice_similarity=voice_sim,
                    fused_score=fused,
                    reason="No matching profile above the combined threshold.",
                )
            logger.info(
                "[IDENTITY] Verify REJECTED face_user=%s voice_user=%s "
                "face_sim=%s voice_sim=%s (modality mismatch or no hit)",
                face_record.user_id if face_record else None,
                voice_record.user_id if voice_record else None,
                face_sim,
                voice_sim,
            )
            return VerifyResponse(
                verified=False,
                face_similarity=face_sim,
                voice_similarity=voice_sim,
                reason="Face and voice did not match the same profile.",
            )

        # Single modality → compare against that modality's own threshold.
        if request.image_base64:
            if face_record is not None and face_sim >= settings.face_threshold:
                return _ok(face_record, face_sim, None, None)
            logger.info(
                "[IDENTITY] Verify REJECTED (face-only) user=%s face_sim=%s "
                "< face_threshold=%s",
                face_record.user_id if face_record else None,
                face_sim,
                settings.face_threshold,
            )
            return VerifyResponse(
                verified=False,
                face_similarity=face_sim,
                reason="No matching profile found.",
            )

        if voice_record is not None and voice_sim >= settings.voice_threshold:
            return _ok(voice_record, None, voice_sim, None)
        logger.info(
            "[IDENTITY] Verify REJECTED (voice-only) user=%s voice_sim=%s "
            "< voice_threshold=%s",
            voice_record.user_id if voice_record else None,
            voice_sim,
            settings.voice_threshold,
        )
        return VerifyResponse(
            verified=False,
            voice_similarity=voice_sim,
            reason="No matching profile found.",
        )

    # ── Register (admin / bootstrap) ─────────────────────────────────────────

    async def register(self, request: RegisterRequest) -> RegisterResponse:
        """Enrol a loyalty profile: embed → FAISS add → SQLite insert.

        At least one biometric (image or audio) must be supplied.  The FAISS
        offsets are minted before the SQLite row is written (they are columns
        of the insert).  ``exists`` is checked before any FAISS mutation so a
        duplicate ``user_id`` cannot leave orphaned vectors behind.
        """
        if not self._inference_ready:
            return RegisterResponse(
                user_id=request.user_id,
                registered=False,
                reason="Inference engines unavailable (models not loaded).",
            )
        if not request.image_base64 and not request.audio_base64:
            return RegisterResponse(
                user_id=request.user_id,
                registered=False,
                reason="At least one biometric (image or audio) is required.",
            )

        async with get_db(self._db_path) as db:
            repo = SqliteProfileRepository(db)
            if await repo.exists(request.user_id):
                return RegisterResponse(
                    user_id=request.user_id,
                    registered=False,
                    reason=f"User '{request.user_id}' is already registered.",
                )

            # Decode + embed both modalities *before* touching FAISS so a
            # malformed payload or an undetected face causes no side effects.
            face_vector = None
            voice_vector = None
            try:
                if request.image_base64:
                    face_vector = self._engines.face.embed(
                        decode_image_bgr(request.image_base64)
                    )
                    if face_vector is None:
                        return RegisterResponse(
                            user_id=request.user_id,
                            registered=False,
                            reason="No face detected in the provided image.",
                        )
                if request.audio_base64:
                    waveform, sample_rate = decode_wav(request.audio_base64)
                    voice_vector = self._engines.voice.embed(waveform, sample_rate)
            except MediaDecodeError as exc:
                return RegisterResponse(
                    user_id=request.user_id,
                    registered=False,
                    reason=f"Could not decode biometric media: {exc}",
                )

            # Offsets first, then the profile row referencing them.
            face_faiss_id = (
                self._face_index.add(face_vector) if face_vector is not None else None
            )
            voice_faiss_id = (
                self._voice_index.add(voice_vector) if voice_vector is not None else None
            )
            record = ProfileRecord(
                user_id=request.user_id,
                name=request.name,
                favorites=request.favorites,
                restrictions=request.restrictions,
                face_faiss_id=face_faiss_id,
                voice_faiss_id=voice_faiss_id,
            )
            await repo.insert(record)
            await db.commit()

        logger.info(
            "[IDENTITY] Registered user_id=%s face_faiss_id=%s voice_faiss_id=%s",
            request.user_id,
            face_faiss_id,
            voice_faiss_id,
        )
        return RegisterResponse(
            user_id=request.user_id,
            registered=True,
            face_faiss_id=face_faiss_id,
            voice_faiss_id=voice_faiss_id,
        )

    # ── Bootstrap (startup auto-enrolment) ───────────────────────────────

    async def bootstrap(self) -> None:
        """Auto-register the profiles from ``identity_config.yaml`` on startup.

        Idempotent: profiles whose ``user_id`` already exists are skipped.  Each
        profile's video is sampled for a representative frame containing a face
        and its optional audio is attached; enrolment is delegated to
        ``register()`` so no pipeline logic is duplicated.  A failing profile is
        logged and skipped — it never aborts startup.
        """
        profiles = self._settings.profiles
        if not profiles:
            logger.info("[BOOTSTRAP] No profiles configured — nothing to register")
            return

        total = len(profiles)
        registered = skipped = failed = 0
        for entry in profiles:
            user_id = entry.get("user_id")
            if not user_id:
                logger.error("[BOOTSTRAP] Profile missing user_id: %r — skipping", entry)
                failed += 1
                continue
            if await self._profile_exists(user_id):
                logger.info("[BOOTSTRAP] user_id=%s already registered — skipping", user_id)
                skipped += 1
                continue
            try:
                if await self._bootstrap_one(entry):
                    registered += 1
                else:
                    failed += 1
            except Exception:  # noqa: BLE001 — one bad profile must not abort startup
                logger.exception("[BOOTSTRAP] Unexpected error registering user_id=%s", user_id)
                failed += 1

        logger.info(
            "[BOOTSTRAP] Summary — attempted=%d registered=%d skipped=%d failed=%d",
            total,
            registered,
            skipped,
            failed,
        )

    async def _profile_exists(self, user_id: str) -> bool:
        async with get_db(self._db_path) as db:
            return await SqliteProfileRepository(db).exists(user_id)

    async def _bootstrap_one(self, entry: dict) -> bool:
        """Enrol a single config profile via ``register()``.  Returns success."""
        user_id = entry["user_id"]
        name = entry.get("name", user_id)
        favorites = entry.get("favorites", [])
        restrictions = entry.get("restrictions", [])
        video_path = entry.get("video_path")
        audio_path = entry.get("audio_path")

        audio_b64 = None
        if audio_path:
            audio_b64 = self._read_file_b64(audio_path)
            if audio_b64 is None:
                logger.warning(
                    "[BOOTSTRAP] user_id=%s audio_path not found: %s", user_id, audio_path
                )

        frames_b64 = self._sample_video_frames_b64(video_path) if video_path else []
        if video_path and not frames_b64:
            logger.error(
                "[BOOTSTRAP] user_id=%s could not read any frame from video_path: %s",
                user_id,
                video_path,
            )

        if not frames_b64 and not audio_b64:
            logger.error(
                "[BOOTSTRAP] user_id=%s has no usable media (video/audio) — skipping",
                user_id,
            )
            return False

        # Video present: try sampled frames until register() detects a face.
        if frames_b64:
            for image_b64 in frames_b64:
                resp = await self.register(
                    RegisterRequest(
                        user_id=user_id,
                        name=name,
                        favorites=favorites,
                        restrictions=restrictions,
                        image_base64=image_b64,
                        audio_base64=audio_b64,
                    )
                )
                if resp.registered:
                    logger.info(
                        "[BOOTSTRAP] Registered user_id=%s face_faiss_id=%s voice_faiss_id=%s",
                        user_id,
                        resp.face_faiss_id,
                        resp.voice_faiss_id,
                    )
                    return True
            logger.error(
                "[BOOTSTRAP] user_id=%s: no sampled frame yielded a detectable face",
                user_id,
            )
            return False

        # Audio-only profile (no video_path).
        resp = await self.register(
            RegisterRequest(
                user_id=user_id,
                name=name,
                favorites=favorites,
                restrictions=restrictions,
                audio_base64=audio_b64,
            )
        )
        if resp.registered:
            logger.info(
                "[BOOTSTRAP] Registered (audio-only) user_id=%s voice_faiss_id=%s",
                user_id,
                resp.voice_faiss_id,
            )
            return True
        logger.error(
            "[BOOTSTRAP] user_id=%s audio-only registration failed: %s", user_id, resp.reason
        )
        return False

    @staticmethod
    def _read_file_b64(path: str) -> str | None:
        """Base64-encode a file's bytes, or return ``None`` if it is missing."""
        p = Path(path)
        if not p.is_file():
            return None
        return base64.b64encode(p.read_bytes()).decode("ascii")

    def _sample_video_frames_b64(self, video_path: str, max_frames: int = 20) -> list[str]:
        """Sample every Nth frame as base64 JPEG (empty list if unreadable)."""
        import cv2  # local import: heavy, container-only dependency

        if not Path(video_path).is_file():
            return []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return []
        step = max(1, self._settings.video_frame_sample_rate)
        frames: list[str] = []
        index = 0
        try:
            while len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if index % step == 0:
                    ok_enc, buf = cv2.imencode(".jpg", frame)
                    if ok_enc:
                        frames.append(base64.b64encode(buf.tobytes()).decode("ascii"))
                index += 1
        finally:
            cap.release()
        return frames
