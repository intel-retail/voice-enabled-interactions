from contextlib import asynccontextmanager, suppress
import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.custom_endpoints import router as custom_router
from api.openai_endpoints import router as openai_router
from api.agent_endpoints import router as agent_router
from pipeline import close_shared_pipeline
from utils.config_loader import config
from utils.ensure_model import ensure_model
from utils.logger_config import setup_logger
from utils.preload_models import preload_models


setup_logger()
logger = logging.getLogger(__name__)


async def _warmup_agent_tools() -> None:
    """Background task: poll kiosk-core until MCP tools are discovered."""
    try:
        from agentic.ordering_agent import get_ordering_agent
        await get_ordering_agent().warmup()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("[STARTUP] Agent MCP warmup failed (non-fatal): %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_model()
    preload_models()
    logger.info("smart-kiosk-assistant rag-service initialized")

    # Bootstrap the ordering agent (discovers MCP tools from kiosk-core)
    try:
        from agentic.ordering_agent import get_ordering_agent
        agent = get_ordering_agent()
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

    # Keep polling kiosk-core for MCP tools in the background so the first
    # customer turn never pays for tool discovery + agent rebuild (~1.3 s).
    warmup_task = asyncio.create_task(_warmup_agent_tools())

    try:
        yield
    finally:
        warmup_task.cancel()
        with suppress(asyncio.CancelledError):
            await warmup_task
        close_shared_pipeline()


app = FastAPI(title="smart-kiosk-assistant-rag-service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(getattr(config.api, "cors_allow_origins", ["http://127.0.0.1", "http://localhost"])),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(openai_router)
app.include_router(custom_router)
app.include_router(agent_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=str(getattr(config.server, "host", "0.0.0.0")),
        port=int(getattr(config.server, "port", 8020)),
        reload=False,
    )