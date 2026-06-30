"""queue-service entry point.

Starts the standalone queue-service: initializes logging, loads the
configuration, ensures the detection model is available, then builds and
runs the DLStreamer pipeline that logs the live queue count.

Graceful SIGINT/SIGTERM handling during the running pipeline is owned by
``QueuePipeline`` (it registers GLib unix signal handlers and tears the
pipeline down before the main loop exits). Here we only guard the bootstrap
window (e.g. a slow model download) so a Ctrl-C exits cleanly.
"""
import logging
import sys

from config_loader import config
from logger_config import setup_logger
from model_manager import ensure_model
from pipeline import run_pipeline

logger = logging.getLogger(__name__)


def main() -> None:
    """Start the queue-service.

    Initializes logging, ensures the detector IR is present, then runs the
    DLStreamer pipeline until SIGINT/SIGTERM.
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
