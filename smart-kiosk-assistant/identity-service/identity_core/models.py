"""Pydantic v2 DTOs for the identity-service REST contract.

Kept in sync with ``kiosk_core/identity/models.py`` on the client side.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LoyaltyProfile(BaseModel):
    user_id: str
    name: str
    favorites: list[str] = Field(default_factory=list)
    restrictions: list[str] = Field(default_factory=list)


class ChallengeResponse(BaseModel):
    challenge_id: str
    prompt_text: str


class VerifyRequest(BaseModel):
    challenge_id: str | None = None
    image_base64: str = Field(description="Base64-encoded camera frame (required).")
    audio_base64: str = Field(description="Base64-encoded WAV audio buffer (required).")


class VerifyResponse(BaseModel):
    verified: bool
    user_id: str | None = None
    profile: LoyaltyProfile | None = None
    face_similarity: float | None = None
    voice_similarity: float | None = None
    fused_score: float | None = None
    reason: str | None = None


class StatsResponse(BaseModel):
    """Operational counts for the identity subsystem (Phase 3)."""

    profiles: int
    face_index_size: int
    voice_index_size: int
    inference_ready: bool


class RegisterRequest(BaseModel):
    """Admin/manual enrolment request (used by bootstrap and ops tooling)."""

    user_id: str
    name: str
    favorites: list[str] = Field(default_factory=list)
    restrictions: list[str] = Field(default_factory=list)
    image_base64: str | None = None
    audio_base64: str | None = None


class RegisterResponse(BaseModel):
    user_id: str
    registered: bool
    face_faiss_id: int | None = None
    voice_faiss_id: int | None = None
    reason: str | None = None
