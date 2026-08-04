import asyncio
import logging
from contextlib import asynccontextmanager, AsyncExitStack
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from kiosk_core import config as cfg
from kiosk_core.api.endpoints import router as api_router
from kiosk_core.models import (
    BrowserWakeWordChunkResponse,
    BrowserWakeWordSessionResponse,
    BrowserWakeWordStartRequest,
    FileSessionStartRequest,
    SessionStartRequest,
    SessionStopResponse,
    WakeWordSessionStartRequest,
)
from kiosk_core.pipeline_latency import pipeline_store
from kiosk_core.service import SessionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

logger = logging.getLogger(__name__)

# Build the MCP http_app once at module level so its lifespan can be wired
# into the FastAPI app's own lifespan below.
_mcp_http_app = None
if cfg.ORDERING_ENABLED:
    from kiosk_core.ordering.mcp_server import mcp
    _mcp_http_app = mcp.http_app()


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncExitStack() as stack:
        # ── Wire MCP http_app lifespan (required for streamable HTTP transport) ──
        if _mcp_http_app is not None:
            await stack.enter_async_context(_mcp_http_app.lifespan(_mcp_http_app))

        # ── Ordering feature startup ─────────────────────────────────────────
        if cfg.ORDERING_ENABLED:
            from kiosk_core.ordering.db import init_db
            from kiosk_core.ordering.service import OrderingService
            from kiosk_core.ordering.api import init_ordering_service
            from kiosk_core.ordering.mcp_server import init_mcp_server

            await init_db(db_path=cfg.KIOSK_DB_PATH)

            ordering_service = OrderingService(upsell_rules_path=cfg.UPSELL_RULES_YAML_PATH)
            seeded = await asyncio.get_event_loop().run_in_executor(
                None, ordering_service.run_seed, cfg.PRODUCTS_YAML_PATH
            )
            logger.info("[STARTUP] Ordering DB ready — %d product(s) seeded", seeded)

            init_ordering_service(ordering_service)
            init_mcp_server(ordering_service)
            logger.info("[STARTUP] MCP server mounted at /mcp (streamable HTTP) ✓")
            logger.info("[STARTUP] Ordering feature enabled ✓")
        else:
            logger.info("[STARTUP] Ordering feature disabled (KIOSK_CORE_ORDERING_ENABLED=false)")

        # ── Identity feature startup ─────────────────────────────────────────
        if cfg.IDENTITY_ENABLED:
            from kiosk_core.identity.client import IdentityClient
            from kiosk_core.identity.api import init_identity_client

            identity_client = IdentityClient(base_url=cfg.IDENTITY_SERVICE_URL)
            init_identity_client(identity_client)
            healthy = await identity_client.health()
            logger.info(
                "[STARTUP] Identity feature enabled ✓ (identity-service=%s, reachable=%s)",
                cfg.IDENTITY_SERVICE_URL,
                healthy,
            )
        else:
            logger.info("[STARTUP] Identity feature disabled (KIOSK_CORE_IDENTITY_ENABLED=false)")

        yield  # application runs


app = FastAPI(title="kiosk-core", lifespan=lifespan)
service = SessionService()
app.include_router(api_router)

# ── Ordering router + MCP mount ──────────────────────────────────────────────
if cfg.ORDERING_ENABLED:
    from kiosk_core.ordering.api import router as ordering_router
    app.include_router(ordering_router)
    if _mcp_http_app is not None:
        app.mount("/mcp", _mcp_http_app)

# ── Identity router ──────────────────────────────────────────────────────────
if cfg.IDENTITY_ENABLED:
    from kiosk_core.identity.api import router as identity_router
    app.include_router(identity_router)


async def _clear_stale_cart_for_new_session(history: list[dict[str, str]] | None = None) -> None:
    """Clear any abandoned draft cart before a brand-new conversation starts.

    A "new session" (one microphone press) is not the same as a "new
    conversation" — the kiosk-ui keeps reusing the same ``conversation_id``
    (and forwards prior turns via ``history``) across all voice turns of a
    single customer visit so the agent retains cart/order state between
    presses (see ``kiosk_core/models.py``). Only clear the draft cart when
    ``history`` is empty, i.e. this really is the first turn of a fresh
    conversation; otherwise this would wipe an in-progress cart on every turn.

    Best-effort: if ordering is disabled or the DB call fails for any reason,
    log and continue — this must never block a new session from starting.
    """
    if not cfg.ORDERING_ENABLED or history:
        return
    try:
        from kiosk_core.ordering.api import get_ordering_service

        ordering_service = get_ordering_service()
        await ordering_service.clear_draft_carts(cfg.DEFAULT_ORDERING_USER_ID)
    except Exception:
        logger.exception("[SESSION_START] Failed to clear stale draft cart — continuing anyway")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/sessions/{session_id}/response-audio/{index}")
def get_response_audio(session_id: str, index: int):
    """Serve a synthesized response-audio WAV segment for browser playback."""
    try:
        path = service.get_response_audio_path(session_id, index)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type="audio/wav", filename=f"response_{index:03d}.wav")
@app.get("/api/v1/identity/enabled")
def identity_enabled() -> dict[str, bool]:
    """Runtime capability flag — always reachable (unlike the gated identity
    router) so kiosk-ui can decide gate-vs-bypass without a rebuild."""
    return {"enabled": cfg.IDENTITY_ENABLED}


@app.get("/api/v1/pipeline/latest")
def pipeline_latest() -> dict:
    """Return the most recent completed voice turn trace with per-stage latencies."""
    trace = pipeline_store.latest()
    if trace is None:
        return {"trace": None, "message": "No completed turns yet"}
    return {"trace": trace}


@app.get("/api/v1/pipeline/recent")
def pipeline_recent(n: int = 5) -> dict:
    """Return the last n completed turn traces (default 5, max 20)."""
    count = min(max(1, n), 20)
    return {"traces": pipeline_store.recent(count)}


@app.get("/api/v1/devices")
def list_devices() -> dict[str, list[dict[str, str | int]]]:
    return {"devices": service.list_input_devices()}


@app.get("/api/v1/capture-mode")
def capture_mode() -> dict[str, object]:
    """Report whether kiosk-core should capture audio directly from a host
    microphone or defer to browser-mic streaming.

    The capture source is controlled by the HOST_MIC env var: when HOST_MIC is
    truthy the backend records from the host machine's microphone
    (recommended='host'); otherwise it streams audio from the browser
    (recommended='browser'). This lets the same build work both locally
    (HOST_MIC=true) and against a remote/headless kiosk-core (HOST_MIC unset)."""
    try:
        devices = service.list_input_devices()
    except Exception:  # noqa: BLE001
        devices = []
    host_available = len(devices) > 0
    return {
        "host_mic_available": host_available,
        "recommended": "host" if cfg.HOST_MIC else "browser",
        "host_devices": devices,
    }


@app.get("/api/v1/sessions")
def list_sessions() -> dict[str, list[dict[str, object]]]:
    return {"sessions": service.list_sessions()}


@app.get("/api/v1/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, object]:
    try:
        return service.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/sessions/start-stream")
async def start_stream_session(request: SessionStartRequest) -> dict[str, object]:
    """Open a browser streaming session.  The caller then pushes audio chunks
    via POST /api/v1/sessions/{session_id}/audio and signals end-of-stream
    via POST /api/v1/sessions/{session_id}/audio/end."""
    await _clear_stale_cart_for_new_session(request.history)
    try:
        return service.start_stream_session(request)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/v1/sessions/{session_id}/audio")
async def push_audio_chunk(session_id: str, request: Request) -> dict[str, str]:
    """Push a raw 16-bit mono PCM WAV chunk into an active browser stream session."""
    wav_bytes = await request.body()
    if not wav_bytes:
        raise HTTPException(status_code=400, detail="Empty audio body")
    try:
        service.push_audio_chunk(session_id, wav_bytes)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "accepted"}


@app.post("/api/v1/sessions/{session_id}/audio/end")
def end_audio_stream(session_id: str) -> dict[str, str]:
    """Signal end-of-stream so the session can finalise and run RAG+TTS."""
    try:
        service.signal_stream_end(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "eos_accepted"}


@app.post("/api/v1/sessions/start", response_model=None)
async def start_session(request: SessionStartRequest) -> dict[str, object]:
    await _clear_stale_cart_for_new_session(request.history)
    try:
        return service.start_session(request)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/v1/sessions/start-after-wakeword", response_model=None)
async def start_session_after_wakeword(request: WakeWordSessionStartRequest) -> dict[str, object]:
    await _clear_stale_cart_for_new_session(request.history)
    try:
        return await run_in_threadpool(service.start_session_after_wakeword, request)
    except TimeoutError as exc:
        raise HTTPException(status_code=408, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Wake-word start failed: {exc}") from exc


@app.post("/api/v1/wakeword/start", response_model=BrowserWakeWordSessionResponse)
async def start_browser_wakeword_session(request: BrowserWakeWordStartRequest) -> BrowserWakeWordSessionResponse:
    try:
        return await run_in_threadpool(service.start_browser_wakeword_session, request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/wakeword/{wakeword_session_id}/audio", response_model=BrowserWakeWordChunkResponse)
async def push_browser_wakeword_audio(wakeword_session_id: str, request: Request) -> BrowserWakeWordChunkResponse:
    wav_bytes = await request.body()
    if not wav_bytes:
        raise HTTPException(status_code=400, detail="Empty audio body")
    try:
        return await run_in_threadpool(service.push_browser_wakeword_audio, wakeword_session_id, wav_bytes)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/v1/wakeword/{wakeword_session_id}/stop", response_model=BrowserWakeWordSessionResponse)
async def stop_browser_wakeword_session(wakeword_session_id: str) -> BrowserWakeWordSessionResponse:
    try:
        return await run_in_threadpool(service.stop_browser_wakeword_session, wakeword_session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/sessions/start-file")
async def start_file_session(
    file: UploadFile = File(...),
    device: int | str | None = Form(None),
    sample_rate: int = Form(cfg.DEFAULT_SAMPLE_RATE),
    chunk_seconds: float = Form(cfg.DEFAULT_CHUNK_SECONDS),
    silence_timeout_seconds: float = Form(cfg.DEFAULT_SILENCE_TIMEOUT_SECONDS),
    max_session_seconds: float = Form(cfg.DEFAULT_MAX_SESSION_SECONDS),
    silence_threshold: int = Form(cfg.DEFAULT_SILENCE_THRESHOLD),
    language: str | None = Form(cfg.DEFAULT_ASR_LANGUAGE),
    temperature: float = Form(0.0),
    analyzer_url: str = Form(cfg.DEFAULT_ANALYZER_URL),
    rag_url: str = Form(cfg.DEFAULT_RAG_URL),
    tts_url: str = Form(cfg.DEFAULT_TTS_URL),
    tts_model: str = Form(cfg.DEFAULT_TTS_MODEL),
    tts_voice: str | None = Form(cfg.DEFAULT_TTS_VOICE),
    tts_language: str | None = Form(cfg.DEFAULT_TTS_LANGUAGE),
    tts_instructions: str | None = Form(cfg.DEFAULT_TTS_INSTRUCTIONS),
    realtime_factor: float = Form(1.0),
) -> dict[str, object]:
    request = FileSessionStartRequest(
        device=device,
        sample_rate=sample_rate,
        chunk_seconds=chunk_seconds,
        silence_timeout_seconds=silence_timeout_seconds,
        max_session_seconds=max_session_seconds,
        silence_threshold=silence_threshold,
        language=language,
        temperature=temperature,
        analyzer_url=analyzer_url,
        rag_url=rag_url,
        tts_url=tts_url,
        tts_model=tts_model,
        tts_voice=tts_voice,
        tts_language=tts_language,
        tts_instructions=tts_instructions,
        realtime_factor=realtime_factor,
    )
    await _clear_stale_cart_for_new_session(request.history)
    try:
        return await run_in_threadpool(service.start_file_session, request, file)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/v1/sessions/{session_id}/stop", response_model=SessionStopResponse)
def stop_session(session_id: str) -> SessionStopResponse:
    try:
        return service.stop_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/v1/sessions/{session_id}/audio/{filename}")
def get_session_audio(session_id: str, filename: str) -> FileResponse:
    """Serve a generated TTS WAV audio file for a session."""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid audio filename")

    session_dir = (Path(__file__).resolve().parent / "generated_audio" / session_id).resolve()
    audio_path = (session_dir / filename).resolve()

    try:
        audio_path.relative_to(session_dir)
    except ValueError as exc:
        raise HTTPException(status_code=404) from exc

    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(audio_path, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8012, reload=False)
