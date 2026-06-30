"""Logging configuration for queue-service.

Provides ``setup_logger()`` using the same format and conventions as the
other microservices (see
``smart-kiosk-assistant/rag-service/utils/logger_config.py``). The log level
is taken from the ``QUEUE_SERVICE_LOG_LEVEL`` environment variable, falling
back to ``config.logging.level``, then ``INFO``.
"""
import logging
import os

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def _config_level() -> str | None:
    try:
        from config_loader import config

        return getattr(getattr(config, "logging", None), "level", None)
    except Exception:  # noqa: BLE001 - config is optional for logging setup
        return None


def setup_logger() -> logging.Logger:
    level_name = (os.getenv("QUEUE_SERVICE_LOG_LEVEL") or _config_level() or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
    else:
        logging.basicConfig(level=level, format=LOG_FORMAT)
    return root
