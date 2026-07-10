"""Pydantic v2 DTOs shared between kiosk-core and identity-service.

These mirror the request/response contract exposed by the identity-service
REST API.  kiosk-core uses them to (de)serialise calls made through
``IdentityClient``; the identity-service reuses the same shapes server-side.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LoyaltyProfile(BaseModel):
    """A registered customer's loyalty profile (metadata only — no embeddings)."""

    user_id: str
    name: str
    favorites: list[str] = Field(default_factory=list)
    restrictions: list[str] = Field(default_factory=list)


class ChallengeResponse(BaseModel):
    """A randomly selected voice challenge prompt for anti-replay liveness."""

    challenge_id: str
    prompt_text: str


class VerifyRequest(BaseModel):
    """Runtime verification request.

    Both modalities are **required** — authentication needs face *and* voice.
    ``image_base64`` is a single captured camera frame (JPEG/PNG, base64).
    ``audio_base64`` is a mono 16 kHz PCM WAV buffer (base64).
    """

    challenge_id: str | None = Field(
        default=None,
        description="Challenge id returned by GET /challenge (anti-replay).",
    )
    image_base64: str = Field(description="Base64-encoded camera frame (required).")
    audio_base64: str = Field(description="Base64-encoded WAV audio buffer (required).")


class VerifyResponse(BaseModel):
    """Result of a verification attempt."""

    verified: bool
    user_id: str | None = None
    profile: LoyaltyProfile | None = None
    face_similarity: float | None = None
    voice_similarity: float | None = None
    fused_score: float | None = None
    reason: str | None = Field(
        default=None,
        description="Human-readable explanation when verified is false.",
    )
