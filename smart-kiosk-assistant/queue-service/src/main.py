"""queue-service entry point.

Starts the standalone queue-service: initializes logging, loads the
configuration, ensures the detection model is available, then (when
api.enabled is set) starts the FastAPI/uvicorn HTTP server in a daemon
thread before building and running the DLStreamer pipeline.

Graceful SIGINT/SIGTERM handling during the running pipeline is owned by
``QueuePipeline`` (it registers GLib unix signal handlers and tears the
pipeline down before the main loop exits). Here we only guard the bootstrap
window (e.g. a slow model download) so a Ctrl-C exits cleanly.
"""
import logging
import sys
import threading

from config_loader import config
from logger_config import setup_logger
from model_manager import ensure_model
from pipeline import run_pipeline

logger = logging.getLogger(__name__)


def _start_api_server() -> None:
    """Start the FastAPI HTTP server in a background daemon thread.

    The daemon flag ensures the thread is killed automatically when the
    main GLib loop exits. uvicorn is imported here so the import only
    happens when the API is actually needed.
    """
    api_cfg = getattr(config, "api", None)
    if api_cfg is None:
        return
    if not getattr(api_cfg, "enabled", False):
        return

    host = str(getattr(api_cfg, "host", "0.0.0.0"))
    port = int(getattr(api_cfg, "port", 8090))

    import uvicorn
    from api import app

    t = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": app, "host": host, "port": port, "log_level": "warning"},
        daemon=True,
    )
    t.start()
    logger.info("HTTP API server started on %s:%d", host, port)


def main() -> None:
    """Start the queue-service.

    Initializes logging, ensures the detector IR is present, starts the
    optional HTTP API server, then runs the DLStreamer pipeline until
    SIGINT/SIGTERM.
    """
    setup_logger()

    source = getattr(config, "source", None)
    model = getattr(config, "model", None)
    logger.info(
        "Starting queue-service (source=%s, model=%s)",
        getattr(source, "rtsp_url", "<unset>"),
        getattr(model, "name", "<unset>"),
    )

    try:
        ensure_model()
        _start_api_server()
        run_pipeline()
    except KeyboardInterrupt:
        logger.info("Interrupted before pipeline start; shutting down")
    except Exception:  # noqa: BLE001 - log and exit non-zero on fatal startup error
        logger.exception("queue-service terminated with an error")
        sys.exit(1)
    finally:
        logger.info("queue-service stopped")


if __name__ == "__main__":
    main()
