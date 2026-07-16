"""Typed HTTP client for the standalone identity-service.

Mirrors the existing client pattern (``AnalyzerClient``, ``RagClient``,
``AgentClient``): one class per upstream service, no business logic, raises on
transport/HTTP errors so the caller decides how to degrade.
"""

from __future__ import annotations

import logging

import httpx

from kiosk_core import config
from kiosk_core.identity.models import (
    ChallengeResponse,
    RegisterRequest,
    RegisterResponse,
    VerifyRequest,
    VerifyResponse,
)

logger = logging.getLogger(__name__)


class IdentityClient:
    """HTTP client wrapping the identity-service REST API."""

    def __init__(self, base_url: str | None = None, timeout_seconds: float | None = None):
        self.base_url = (base_url or config.IDENTITY_SERVICE_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds or config.DEFAULT_HTTP_TIMEOUT_SECONDS

    async def get_challenge(self) -> ChallengeResponse:
        """Fetch a random voice challenge prompt (anti-replay liveness)."""
        url = f"{self.base_url}/api/v1/identity/challenge"
        async with httpx.AsyncClient(timeout=self.timeout_seconds, trust_env=False) as client:
            response = await client.get(url)
            response.raise_for_status()
            return ChallengeResponse.model_validate(response.json())

    async def verify(self, request: VerifyRequest) -> VerifyResponse:
        """Run multimodal (face + voice) verification against enrolled profiles."""
        url = f"{self.base_url}/api/v1/identity/verify"
        async with httpx.AsyncClient(timeout=self.timeout_seconds, trust_env=False) as client:
            response = await client.post(url, json=request.model_dump())
            response.raise_for_status()
            return VerifyResponse.model_validate(response.json())

    async def register(self, request: RegisterRequest) -> RegisterResponse:
        """Self-service enrolment: forwards face+voice capture to identity-service."""
        url = f"{self.base_url}/api/v1/identity/register"
        async with httpx.AsyncClient(timeout=self.timeout_seconds, trust_env=False) as client:
            response = await client.post(url, json=request.model_dump())
            response.raise_for_status()
            return RegisterResponse.model_validate(response.json())

    async def health(self) -> bool:
        """Liveness probe — returns True when identity-service responds 200."""
        url = f"{self.base_url}/health"
        try:
            async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
                response = await client.get(url)
                return response.status_code == 200
        except httpx.HTTPError as exc:
            logger.warning("[IDENTITY-CLIENT] health check failed: %s", exc)
            return False
