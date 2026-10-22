import logging
import threading
from pathlib import Path

import httpx

from kiosk_core import config

logger = logging.getLogger(__name__)

# Substrings audio-analyzer uses when an English-only Whisper checkpoint is
# handed a language token. The ASR model lives in audio-analyzer's config, not
# ours, so kiosk-core cannot know ahead of time whether the deployed
# checkpoint is multilingual -- it can only find out by being told.
_NOT_MULTILINGUAL_MARKERS = (
    "not multilingual",
    "cannot specify 'language'",
    'cannot specify "language"',
)

# Latched once the deployed analyzer has rejected a language token, so only
# the FIRST request of the process pays the failed round-trip rather than
# every chunk of every session. Deliberately module-level: this is a property
# of the upstream model, shared by every session, not of one AnalyzerClient.
_language_unsupported = threading.Event()


def _rejects_language(response: httpx.Response) -> bool:
    """True when this error is audio-analyzer refusing a language token.

    Any other 5xx (model crash, OOM, timeout) must NOT be swallowed into a
    silent retry, so the match is on the specific upstream message.
    """
    if response.status_code < 400:
        return False
    try:
        body = response.text.lower()
    except Exception:  # noqa: BLE001 - a body we cannot read is not this error
        return False
    return any(marker in body for marker in _NOT_MULTILINGUAL_MARKERS)


class AnalyzerClient:
    def __init__(self, analyzer_url: str, timeout_seconds: float | None = None):
        self.analyzer_url = analyzer_url
        self.timeout_seconds = timeout_seconds or config.DEFAULT_HTTP_TIMEOUT_SECONDS
        # One session/turn commonly issues 2-3 chunk flushes to the analyzer
        # (adaptive-pause flush, chunk-cap flush, final tail flush). Each
        # AnalyzerClient instance is scoped to a single audio session (see
        # BaseAudioSession.__init__), so a persistent client here reuses one
        # keep-alive TCP connection across all of a turn's ASR calls instead
        # of paying a fresh connection setup on every single flush. Call
        # close() once the owning session finishes (BaseAudioSession does
        # this in _finalize_run).
        self._client = httpx.Client(timeout=self.timeout_seconds, trust_env=False)

    def close(self) -> None:
        self._client.close()

    def transcribe_file(
        self,
        file_path: str,
        language: str | None = None,
        temperature: float = 0.0,
        diarization: bool = False,
        session_id: str | None = None,
        speaker_scope_id: str | None = None,
        prompt: str | None = None,
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
        # Always send the `language` field explicitly. The analyzer's
        # /v1/audio/transcriptions endpoint declares
        # `language: str | None = Form("en")` — if the field is OMITTED
        # from the multipart body, FastAPI applies that "en" default
        # regardless of our intent. A genuinely EMPTY string ("") is also
        # treated as "not provided" by Starlette's multipart form parser
        # (confirmed empirically), so it silently falls back to "en" too.
        # A single space survives multipart parsing as a real value, and
        # the analyzer's own `_normalize_optional_text()` strips it back
        # down to None — this is required to genuinely leave the language
        # unset, e.g. for English-only ASR checkpoints
        # (distil-whisper/distil-small.en) that reject any language token.
        #
        # _language_unsupported latches after the deployed analyzer has told
        # us its checkpoint is English-only, so every subsequent request
        # omits the token instead of re-failing. Without this, the shipped
        # defaults (KIOSK_CORE_ASR_LANGUAGE=en against the English-only
        # distil-small.en) returned HTTP 500 on every single turn and the
        # whole stack transcribed nothing.
        if _language_unsupported.is_set():
            language = None
        data["language"] = language if language else " "
        # Always send the diarization flag explicitly, in both directions. The
        # analyzer's endpoint declares `diarization: bool | None = Form(None)`
        # and falls back to its own `models.asr.diarization` config when the
        # field is ABSENT — so omitting it on preview chunks (as this client
        # used to) silently ran full diarization plus speaker enrollment on
        # every intermediate chunk, on the latency-critical path.
        data["diarization"] = "true" if diarization else "false"
        if diarization:
            data["response_format"] = "verbose_json"
        if session_id:
            data["session_id"] = session_id
        if speaker_scope_id:
            data["speaker_scope_id"] = speaker_scope_id
        if prompt:
            data["prompt"] = prompt

        with path.open("rb") as audio_file:
            response = self._client.post(
                self.analyzer_url,
                files={"file": (path.name, audio_file, "audio/wav")},
                data=data,
            )
        # The deployed checkpoint is English-only and rejected our language
        # token. Latch that fact and retry this chunk once without it, rather
        # than losing the turn's audio to a 500 the operator cannot see.
        if data["language"] != " " and _rejects_language(response):
            if not _language_unsupported.is_set():
                _language_unsupported.set()
                logger.warning(
                    "audio-analyzer rejected language=%r (English-only ASR "
                    "checkpoint); retrying without a language token and "
                    "omitting it for the rest of this process. Set "
                    "KIOSK_CORE_ASR_LANGUAGE='' to silence this, or point "
                    "ASR_MODEL at a multilingual checkpoint.",
                    data["language"],
                )
            data["language"] = " "
            with path.open("rb") as audio_file:
                response = self._client.post(
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
