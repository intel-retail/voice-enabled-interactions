"""identity-service FastAPI application.

Exposes:
  GET  /health                       -> liveness probe
  GET  /api/v1/identity/challenge     -> random voice challenge prompt
  POST /api/v1/identity/verify        -> multimodal (face+voice) verification
  POST /api/v1/identity/register      -> admin/manual enrolment

The heavy biometric pipeline (OpenVINO + FAISS + SQLite) is wired up across
Phases 3–6; this Phase-2 skeleton boots cleanly and serves challenge prompts.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from identity_core.config import load_settings
from identity_core.models import (
    ChallengeResponse,
    RegisterRequest,
    RegisterResponse,
    StatsResponse,
    VerifyRequest,
    VerifyResponse,
)
from identity_core.service import IdentityService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

settings = load_settings()
service = IdentityService(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[STARTUP] identity-service booting (device=%s)", settings.device)
    logger.info(
        "[STARTUP] db_path=%s faiss_dir=%s", settings.db_path, settings.faiss_dir
    )
    # ── Storage bootstrap (Phase 3) ──────────────────────────────────────────
    await service.init_storage()
    # ── Bootstrap automatic test registration (Phase 5) ──────────────────────
    if settings.bootstrap_on_start:
        if settings.profiles:
            logger.info(
                "[STARTUP] BOOTSTRAP_ON_START=true — %d profile(s) configured "
                "(registration wired in Phase 5)",
                len(settings.profiles),
            )
        else:
            logger.info("[STARTUP] BOOTSTRAP_ON_START=true but no profiles configured")
    else:
        logger.info("[STARTUP] BOOTSTRAP_ON_START=false — skipping auto registration")
    yield
    logger.info("[SHUTDOWN] identity-service stopped")


app = FastAPI(title="identity-service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/identity/challenge", response_model=ChallengeResponse)
def get_challenge() -> ChallengeResponse:
    return service.get_challenge()


@app.get("/api/v1/identity/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    return await service.get_stats()


@app.post("/api/v1/identity/verify", response_model=VerifyResponse)
async def verify(request: VerifyRequest) -> VerifyResponse:
    return await service.verify(request)


@app.post("/api/v1/identity/register", response_model=RegisterResponse)
async def register(request: RegisterRequest) -> RegisterResponse:
    return await service.register(request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=False)
