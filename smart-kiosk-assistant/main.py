import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from kiosk_core import config as cfg
from kiosk_core.models import FileSessionStartRequest, SessionStartRequest, SessionStopResponse
from kiosk_core.service import SessionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Ordering feature startup ─────────────────────────────────────────────
    if cfg.ORDERING_ENABLED:
        from kiosk_core.ordering.db import init_db
        from kiosk_core.ordering.service import OrderingService
        from kiosk_core.ordering.api import init_ordering_service

        await init_db(db_path=cfg.KIOSK_DB_PATH)

        ordering_service = OrderingService(upsell_rules_path=cfg.UPSELL_RULES_YAML_PATH)
        # Seed products (runs in threadpool to avoid blocking event loop)
        seeded = await asyncio.get_event_loop().run_in_executor(
            None, ordering_service.run_seed, cfg.PRODUCTS_YAML_PATH
        )
        logger.info("[STARTUP] Ordering DB ready — %d product(s) seeded", seeded)

        init_ordering_service(ordering_service)

        # Mount MCP server for agent tool discovery
        from kiosk_core.ordering.mcp_server import mcp, init_mcp_server
        init_mcp_server(ordering_service)
        app.mount("/mcp", mcp.sse_app())
        logger.info("[STARTUP] MCP server mounted at /mcp (SSE) ✓")
        logger.info("[STARTUP] Ordering feature enabled ✓")
    else:
        logger.info("[STARTUP] Ordering feature disabled (KIOSK_CORE_ORDERING_ENABLED=false)")

    yield  # application runs here


app = FastAPI(title="kiosk-core", lifespan=lifespan)
service = SessionService()

# ── Ordering router ──────────────────────────────────────────────────────────
if cfg.ORDERING_ENABLED:
    from kiosk_core.ordering.api import router as ordering_router
    app.include_router(ordering_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/devices")
def list_devices() -> dict[str, list[dict[str, str | int]]]:
    return {"devices": service.list_input_devices()}


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
def start_stream_session(request: SessionStartRequest) -> dict[str, object]:
    """Open a browser streaming session.  The caller then pushes audio chunks
    via POST /api/v1/sessions/{session_id}/audio and signals end-of-stream
    via POST /api/v1/sessions/{session_id}/audio/end."""
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
def start_session(request: SessionStartRequest) -> dict[str, object]:
    try:
        return service.start_session(request)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/v1/sessions/start-file")
def start_file_session(
    file: UploadFile = File(...),
    device: int | str | None = Form(None),
    sample_rate: int = Form(16000),
    chunk_seconds: float = Form(4.0),
    silence_timeout_seconds: float = Form(1.5),
    max_session_seconds: float = Form(20.0),
    silence_threshold: int = Form(900),
    language: str | None = Form(None),
    temperature: float = Form(0.0),
    analyzer_url: str = Form("http://127.0.0.1:8010/v1/audio/transcriptions"),
    rag_url: str = Form("http://127.0.0.1:8020/api/v1/query"),
    tts_url: str = Form("http://127.0.0.1:8011/v1/audio/speech"),
    tts_model: str = Form("qwen-tts"),
    tts_voice: str | None = Form(None),
    tts_language: str | None = Form("English"),
    tts_instructions: str | None = Form(None),
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
    try:
        return service.start_file_session(request, file)
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
