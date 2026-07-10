"""REST API facade for the identity feature (kiosk-core side).

Router prefix: /api/v1/identity
Tags: identity

These endpoints are thin proxies: kiosk-core forwards challenge/verify calls to
the standalone identity-service through ``IdentityClient``.  The router is only
mounted when ``IDENTITY_ENABLED`` is true (see main.py).
"""

from __future__ import annotations

import logging
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException

from kiosk_core.identity.client import IdentityClient
from kiosk_core.identity.models import (
    ChallengeResponse,
    VerifyRequest,
    VerifyResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/identity", tags=["identity"])

# Module-level singleton injected at startup via `init_identity_client()`.
_identity_client: IdentityClient | None = None


def init_identity_client(client: IdentityClient) -> None:
    """Called from main.py lifespan to inject the client singleton."""
    global _identity_client
    _identity_client = client
    logger.info("[IDENTITY-API] IdentityClient registered (base_url=%s)", client.base_url)


def get_identity_client() -> IdentityClient:
    if _identity_client is None:
        raise RuntimeError("IdentityClient not initialised. Call init_identity_client() first.")
    return _identity_client


ClientDep = Annotated[IdentityClient, Depends(get_identity_client)]


@router.get("/challenge", response_model=ChallengeResponse, summary="Get voice challenge")
async def get_challenge(client: ClientDep) -> ChallengeResponse:
    """Return a randomly selected voice challenge prompt for the user to read."""
    try:
        return await client.get_challenge()
    except httpx.HTTPError as exc:
        logger.error("[IDENTITY-API] challenge upstream error: %s", exc)
        raise HTTPException(status_code=502, detail=f"identity-service unavailable: {exc}") from exc


@router.post("/verify", response_model=VerifyResponse, summary="Verify identity (face + voice)")
async def verify(request: VerifyRequest, client: ClientDep) -> VerifyResponse:
    """Run multimodal verification.  Both image and audio are required."""
    if not request.image_base64 or not request.audio_base64:
        raise HTTPException(
            status_code=422,
            detail="Both image_base64 and audio_base64 are required for verification.",
        )
    try:
        return await client.verify(request)
    except httpx.HTTPError as exc:
        logger.error("[IDENTITY-API] verify upstream error: %s", exc)
        raise HTTPException(status_code=502, detail=f"identity-service unavailable: {exc}") from exc
