"""HTTP client for the standalone queue-service (kiosk queue-depth sensor).

Mirrors ``AnalyzerClient``'s pattern: a thin, typed wrapper around one
upstream service's HTTP surface. Used by the QSR MCP read tools
(``kiosk_core/qsr_mcp/mcp_service.py``) to answer "how deep is the queue?"
without kiosk-core re-implementing any counting logic itself — queue-service
remains the single source of truth (see ``queue-service/src/queue_state.py``).
"""
from __future__ import annotations

import logging

import httpx

from kiosk_core import config

logger = logging.getLogger(__name__)


class QueueClient:
    """Reads the live queue snapshot from queue-service's REST API."""

    def __init__(self, base_url: str | None = None, timeout_seconds: float | None = None):
        self.base_url = (base_url or config.QUEUE_SERVICE_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds or config.DEFAULT_HTTP_TIMEOUT_SECONDS
        self._client = httpx.Client(timeout=self.timeout_seconds, trust_env=False)

    def close(self) -> None:
        self._client.close()

    def get_queue_snapshot(self) -> dict | None:
        """Return the latest ``{count, nearby, status, timestamp, version}``
        snapshot from queue-service, or ``None`` if it is unreachable.

        Best-effort by design: a QSR read tool answering "how deep is the
        queue?" should degrade to "unknown" rather than raise and take down
        the whole MCP read path when queue-service is briefly down.
        """
        try:
            response = self._client.get(f"{self.base_url}/api/v1/queue/count")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            logger.warning("[QueueClient] Failed to reach queue-service at %s: %s", self.base_url, exc)
            return None

    def health(self) -> bool:
        try:
            response = self._client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False
