from contextlib import asynccontextmanager, suppress
import asyncio
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.custom_endpoints import router as custom_router
from api.openai_endpoints import router as openai_router
from pipeline import close_shared_pipeline
from utils.config_loader import config
from utils.ensure_model import ensure_model
from utils.logger_config import setup_logger
from utils.preload_models import preload_models

# Feature flag: mount the ordering-agent router only when this is set to
# "true". Defaults to true to preserve backwards-compatibility for the Smart
# Kiosk deployment. Non-kiosk deployments (Healthcare, Education) set this
# to "false" via their environment/compose file so the agent endpoints and
# MCP bootstrap are never invoked.
ORDERING_AGENT_ENABLED = os.getenv("ORDERING_AGENT_ENABLED", "true").lower() == "true"

setup_logger()
logger = logging.getLogger(__name__)


async def _warmup_agent_tools() -> None:
    """Background task: poll kiosk-core until MCP tools are discovered."""
    try:
        from agentic.plugin_loader import load_agent_factory
        await load_agent_factory()().warmup()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("[STARTUP] Agent MCP warmup failed (non-fatal): %s", exc)


def _warmup_rag_models() -> None:
    """Run one throwaway query so embeddings, reranker, and the LLM/OVMS backend
    compile their kernels before the first real turn (otherwise it is ~cold)."""
    from pipeline import get_shared_pipeline

    get_shared_pipeline().answer_question("warmup", max_tokens=8)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_model()
    preload_models()
    logger.info("rag-service initialized (ordering_agent=%s)", ORDERING_AGENT_ENABLED)

    # Warm the inference path (embeddings + reranker + LLM/OVMS) with one
    # throwaway query so the first real turn does not pay kernel-compile cost.
    # Bounded so a slow/unready backend can never stall startup indefinitely.
    try:
        await asyncio.wait_for(asyncio.to_thread(_warmup_rag_models), timeout=60)
        logger.info("[STARTUP] RAG model warmup complete ✓")
    except Exception as exc:  # noqa: BLE001  (includes TimeoutError → lazy fallback)
        logger.warning("[STARTUP] RAG model warmup skipped/failed (non-fatal): %s", exc)

    # Bootstrap the ordering agent only when the feature flag is enabled.
    warmup_task = None
    if ORDERING_AGENT_ENABLED:
        try:
            from agentic.plugin_loader import load_agent_factory
            agent = load_agent_factory()()
            await agent.bootstrap()
            logger.info("[STARTUP] OrderingAgent bootstrapped ✓")
        except BaseException as exc:
            # KeyboardInterrupt / SystemExit must propagate; everything else
            # (including asyncio.CancelledError from MCP's anyio TaskGroup when
            # kiosk-core is not yet reachable) is non-fatal — the agent will
            # retry MCP discovery on the first chat() call.
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            logger.warning("[STARTUP] OrderingAgent bootstrap failed (non-fatal): %s", exc)

        # Keep polling kiosk-core for MCP tools in the background.
        warmup_task = asyncio.create_task(_warmup_agent_tools())

    try:
        yield
    finally:
        if warmup_task is not None:
            warmup_task.cancel()
            with suppress(asyncio.CancelledError):
                await warmup_task
        if ORDERING_AGENT_ENABLED:
            from plugins.kiosk.directive_mode import close_http_client
            await close_http_client()
        close_shared_pipeline()


app = FastAPI(title="rag-service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(getattr(config.api, "cors_allow_origins", ["http://127.0.0.1", "http://localhost"])),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(openai_router)
app.include_router(custom_router)

if ORDERING_AGENT_ENABLED:
    from api.agent_endpoints import router as agent_router
    app.include_router(agent_router)
    logger.info("[STARTUP] Agent endpoints mounted at /api/v1/agent")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=str(getattr(config.server, "host", "0.0.0.0")),
        port=int(getattr(config.server, "port", 8020)),
        reload=False,
    )