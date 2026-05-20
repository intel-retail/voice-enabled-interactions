import json
from collections.abc import Generator

import httpx

from kiosk_core import config


class RagClient:
    def __init__(self, rag_url: str, timeout_seconds: float | None = None):
        self.rag_url = rag_url
        self.timeout_seconds = timeout_seconds or config.DEFAULT_HTTP_TIMEOUT_SECONDS

    def stream_answer(
        self,
        transcription: str,
        history: list[dict[str, str]] | None = None,
    ) -> Generator[str, None, None]:
        payload: dict[str, object] = {"transcription": transcription}
        if history:
            # Keep only role/content fields and drop empties; rag-service
            # validates role ∈ {user, assistant}.
            cleaned = [
                {"role": str(t.get("role", "")), "content": str(t.get("content", ""))}
                for t in history
                if t.get("content")
            ]
            if cleaned:
                payload["history"] = cleaned
        with httpx.Client(timeout=self.timeout_seconds, trust_env=False) as client:
            with client.stream(
                "POST",
                self.rag_url,
                headers={"Accept": "text/event-stream"},
                json=payload,
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break

                    event = json.loads(payload)
                    if "error" in event:
                        raise RuntimeError(str(event["error"]))

                    token = str(event.get("token", ""))
                    if token:
                        yield token