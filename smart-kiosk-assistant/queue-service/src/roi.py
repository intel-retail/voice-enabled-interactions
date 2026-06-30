"""Region-of-interest (ROI) geometry for queue-service.

Provides ``ROIManager``: loads the queue polygon from ``queue-config.yaml``
(normalized 0..1 coordinates) and decides whether a person is inside the
monitored queue area via a dependency-free ray-casting point-in-polygon
test. The class is intentionally independent of DLStreamer/GStreamer so it
can be unit tested in isolation.
"""
import logging
from typing import Sequence

logger = logging.getLogger(__name__)

Point = tuple[float, float]
BBox = tuple[float, float, float, float]
_VALID_POINT_MODES = {"foot", "centroid"}


class ROIManager:
    """Holds the queue ROI polygon and tests points/boxes against it.

    Args:
        polygon: Sequence of ``(x, y)`` vertices. When ``None`` the polygon
            is loaded from ``config.roi.polygon``.
        point_mode: ``"foot"`` or ``"centroid"``. When ``None`` it is read
            from ``config.roi.point_mode`` (default ``"foot"``).
        normalized: Whether the polygon (and queried points) use normalized
            ``0..1`` coordinates. Documentation only -- the test is
            unit-agnostic.
    """

    def __init__(
        self,
        polygon: Sequence[Sequence[float]] | None = None,
        point_mode: str | None = None,
        normalized: bool = True,
    ) -> None:
        if polygon is None:
            cfg_polygon, cfg_point_mode = self._load_roi_config()
            polygon = cfg_polygon
            if point_mode is None:
                point_mode = cfg_point_mode

        if point_mode is None:
            point_mode = "foot"
        if point_mode not in _VALID_POINT_MODES:
            logger.warning("Unknown point_mode '%s'; falling back to 'foot'", point_mode)
            point_mode = "foot"

        self.point_mode = point_mode
        self.normalized = normalized
        self._polygon: list[Point] = [(float(x), float(y)) for x, y in polygon]

        if len(self._polygon) < 3:
            logger.warning(
                "ROI polygon has %d point(s); ROI filtering disabled (all points "
                "treated as inside)", len(self._polygon)
            )

    # ── configuration ────────────────────────────────────────────────────────

    @staticmethod
    def _load_roi_config() -> tuple[list, str]:
        from config_loader import config

        roi_cfg = getattr(config, "roi", None)
        polygon = list(getattr(roi_cfg, "polygon", None) or [])
        point_mode = getattr(roi_cfg, "point_mode", "foot")
        return polygon, point_mode

    @property
    def polygon(self) -> list:
        """Normalized polygon vertices (``(x, y)`` in 0..1)."""
        return self._polygon

    # ── geometry ─────────────────────────────────────────────────────────────

    def point_from_bbox(self, bbox: BBox) -> Point:
        """Return the representative point of a bounding box.

        ``bbox`` is ``(x_min, y_min, x_max, y_max)`` in the same coordinate
        space as the polygon. Uses the foot point (bottom-center) or the
        centroid depending on ``point_mode``.
        """
        x_min, y_min, x_max, y_max = bbox
        center_x = (x_min + x_max) / 2.0
        if self.point_mode == "centroid":
            return (center_x, (y_min + y_max) / 2.0)
        return (center_x, y_max)

    def is_inside_roi(self, point: Point) -> bool:
        """Return whether ``point`` lies inside the configured polygon.

        Uses the even-odd ray-casting algorithm. When no valid polygon is
        configured (fewer than 3 vertices), every point is considered inside.
        """
        polygon = self._polygon
        count = len(polygon)
        if count < 3:
            return True

        x, y = point
        inside = False
        j = count - 1
        for i in range(count):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def is_bbox_inside(self, bbox: BBox) -> bool:
        """Convenience: test a bounding box via its foot/centroid point."""
        return self.is_inside_roi(self.point_from_bbox(bbox))