"""Ordering domain — product catalogue, cart management, and upsell engine."""

from kiosk_core.ordering.service import OrderingService
from kiosk_core.ordering.db import init_db

__all__ = ["OrderingService", "init_db"]
