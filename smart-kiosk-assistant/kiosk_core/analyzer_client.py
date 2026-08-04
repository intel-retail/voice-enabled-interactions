from pathlib import Path

import httpx

from kiosk_core import config


class AnalyzerClient:
    def __init__(self, analyzer_url: str, timeout_seconds: float | None = None):
        self.analyzer_url = analyzer_url
        self.timeout_seconds = timeout_seconds or config.DEFAULT_HTTP_TIMEOUT_SECONDS

    def transcribe_file(
        self,
        file_path: str,
        language: str | None = None,
        temperature: float = 0.0,
        diarization: bool = False,
        session_id: str | None = None,
        speaker_scope_id: str | None = None,
    ) -> dict:
        """POST an audio file to the transcription endpoint.

        Returns the full JSON response dict. When ``diarization=True`` the
        request asks for ``response_format=verbose_json`` so the caller
        receives a ``segments`` list with per-segment ``speaker`` labels.

        When ``session_id`` is provided it is forwarded to the analyzer so
        state that must persist across chunks (e.g. per-session enrolled
        speaker embeddings) is correctly scoped. The analyzer's assigned
        session id — which may differ on the very first call — is also
        surfaced via the ``X-Session-ID`` response header and included in
        the returned dict under the key ``_analyzer_session_id``.

        ``speaker_scope_id`` scopes the analyzer's enrolled primary-speaker
        voice. It must stay constant for the whole conversation — unlike
        ``session_id``, which is regenerated per utterance — otherwise the
        analyzer re-enrols the reference voice from the very audio it is
        judging and can never reject a secondary speaker.
        """
        path = Path(file_path)
        data: dict = {"temperature": str(temperature)}
        if language:
            data["language"] = language
        if diarization:
            data["diarization"] = "true"
            data["response_format"] = "verbose_json"
        if session_id:
            data["session_id"] = session_id
        if speaker_scope_id:
            data["speaker_scope_id"] = speaker_scope_id

        with path.open("rb") as audio_file:
            with httpx.Client(timeout=self.timeout_seconds, trust_env=False) as client:
                response = client.post(
                    self.analyzer_url,
                    files={"file": (path.name, audio_file, "audio/wav")},
                    data=data,
                )
        response.raise_for_status()
        payload = response.json()
        assigned_session = response.headers.get("X-Session-ID")
        if assigned_session and isinstance(payload, dict):
            payload["_analyzer_session_id"] = assigned_session
        return payload
