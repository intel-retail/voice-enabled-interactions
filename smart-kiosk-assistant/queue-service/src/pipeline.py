"""DLStreamer pipeline construction and execution.

Builds and runs the GStreamer/DLStreamer pipeline:

    RTSP capture -> decode -> gvadetect (person-detection-retail-0013)
                 -> gvatrack -> gvapython (queue_counter) -> fakesink

The element order is taken from ``conf/pipeline.yaml`` and each element is
configured from ``conf/queue-config.yaml``. Queue-counting logic is NOT
implemented here -- ``pipeline.py`` only registers ``queue_counter.py`` as
the ``gvapython`` callback. This module owns pipeline construction, element
configuration, bus (EOS/ERROR) handling, RTSP reconnection, and the GLib
main loop lifecycle.
"""
from __future__ import annotations

import logging
import os
import signal
from pathlib import Path
from urllib.parse import urlparse

import gi
import yaml

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402  (must follow require_version)


logger = logging.getLogger(__name__)

# queue-service/ (parent of src/)
BASE_DIR = Path(__file__).resolve().parent.parent
CONF_DIR = BASE_DIR / "conf"

# GST_RTSP_LOWER_TRANS_TCP from the GstRTSPLowerTrans flags.
_RTSP_LOWER_TRANS_TCP = 0x04

# Default gvapython binding when not overridden in configuration.
_DEFAULT_COUNTER_CLASS = "QueueCounter"
_DEFAULT_COUNTER_FUNCTION = "process_frame"

# gvapython binding for the optional person metadata filter (YOLO models only).
_FILTER_CLASS = "PersonFilter"
_FILTER_FUNCTION = "process_frame"
# Synthetic element token: built as a gvapython element but configured with the
# PersonFilter module. Lets us reuse the existing element-graph machinery.
_PERSON_FILTER_TYPE = "person_filter"

# RTSP source element types whose transport we configure.
_RTSP_SOURCE_TYPES = {"urisourcebin", "uridecodebin", "rtspsrc"}
_SOURCE_ELEMENT_NAME = "queue_source"


class QueuePipeline:
    """Builds and runs the DLStreamer queue-service pipeline.

    Configuration is read directly from the YAML files under ``conf/``
    because the shared ``config_loader`` is not implemented yet.
    """

    def __init__(self, conf_dir: Path | str | None = None) -> None:
        self._conf_dir = Path(conf_dir) if conf_dir else CONF_DIR
        self._config = self._load_yaml(self._conf_dir / "queue-config.yaml")
        self._pipeline_def = self._load_yaml(self._conf_dir / "pipeline.yaml")

        source_cfg = self._config.get("source", {})
        self._reconnect_delay = float(source_cfg.get("reconnect_delay_seconds", 5.0))
        max_retries = source_cfg.get("max_reconnect_attempts")
        self._max_retries = int(max_retries) if max_retries is not None else None

        debug_cfg = self._config.get("debug", {}) or {}
        self._debug = bool(debug_cfg.get("visualization", False))
        self._debug_sink = str(debug_cfg.get("sink", "autovideosink"))
        self._fps_logging = bool(debug_cfg.get("fps_logging", False))

        api_cfg = self._config.get("api", {}) or {}
        self._api_enabled = bool(api_cfg.get("enabled", False))
        jpeg_quality = int(api_cfg.get("jpeg_quality", 80))
        if self._api_enabled:
            import frame_buffer
            frame_buffer.configure(jpeg_quality)

        # GStreamer's rtspsrc connects through GIO, which honours the proxy
        # environment variables. The internal RTSP host must bypass any HTTP
        # proxy or the connection is wrongly routed and fails immediately.
        self._bypass_proxy_for_rtsp(source_cfg.get("rtsp_url", ""))

        if not Gst.is_initialized():
            Gst.init(None)

        self.pipeline: Gst.Pipeline | None = None
        self.loop = GLib.MainLoop()
        self._bus = None
        self._retries = 0
        self._reconnecting = False
        self._stopping = False

    # ── configuration helpers ────────────────────────────────────────────────

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    @staticmethod
    def _bypass_proxy_for_rtsp(rtsp_url: str) -> None:
        """Add the RTSP source host to ``no_proxy`` so GIO connects directly.

        Must run before ``Gst.init`` so the GIO proxy resolver picks up the
        updated value. Leaves the proxy in place for other hosts (e.g. the
        model download) by only appending the RTSP host.
        """
        host = urlparse(rtsp_url).hostname
        if not host:
            return
        for var in ("no_proxy", "NO_PROXY"):
            entries = [e.strip() for e in os.environ.get(var, "").split(",") if e.strip()]
            if host not in entries:
                entries.append(host)
                os.environ[var] = ",".join(entries)

    def _resolve(self, path: str) -> str:
        """Resolve a config path relative to the service base directory."""
        if os.path.isabs(path):
            return path
        return str((BASE_DIR / path).resolve())

    @staticmethod
    def _format_value(value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def _counter_module_path(self) -> str:
        return str((BASE_DIR / "src" / "queue_counter.py").resolve())

    def _filter_module_path(self) -> str:
        return str((BASE_DIR / "src" / "person_filter.py").resolve())

    def _is_yolo_model(self) -> bool:
        """Decide (from config) whether the detector is a YOLO model.

        YOLO detectors emit every COCO class, so non-person detections must be
        filtered out. The person-only Intel model returns False here and is
        therefore left completely unchanged.
        """
        name = str(self._config.get("model", {}).get("name", "")).lower()
        return "yolo" in name

    # ── pipeline construction ────────────────────────────────────────────────

    def _element_properties(self, etype: str) -> dict[str, object]:
        """Return the property map for a pipeline element type."""
        cfg = self._config
        if etype in {"urisourcebin", "uridecodebin"}:
            return {"name": _SOURCE_ELEMENT_NAME, "uri": cfg["source"]["rtsp_url"]}
        if etype == "rtspsrc":
            return {"name": _SOURCE_ELEMENT_NAME, "location": cfg["source"]["rtsp_url"]}
        if etype == "gvadetect":
            model = cfg["model"]
            props: dict[str, object] = {
                "model": self._resolve(model["ir_path"]),
                "device": model.get("device", "CPU"),
                "threshold": model.get("threshold", 0.5),
                "inference-interval": model.get("inference_interval", 1),
            }
            # model-proc is optional: person-detection-retail-0013 runs
            # without one. Only add it when a valid file is configured.
            proc_path = model.get("proc_path")
            if proc_path:
                resolved_proc = self._resolve(proc_path)
                if os.path.isfile(resolved_proc):
                    props["model-proc"] = resolved_proc
                else:
                    logger.warning(
                        "model-proc '%s' not found; omitting model-proc property",
                        resolved_proc,
                    )
            return props
        if etype == "gvatrack":
            tracker = cfg.get("tracker", {})
            return {"tracking-type": tracker.get("tracking_type", "short-term-imageless")}
        if etype == _PERSON_FILTER_TYPE:
            # PersonFilter removes non-person regions before gvatrack so that
            # tracking, watermark and counting only see persons.
            return {
                "module": self._filter_module_path(),
                "class": _FILTER_CLASS,
                "function": _FILTER_FUNCTION,
            }
        if etype == "gvapython":
            counter = cfg.get("counter", {})
            return {
                "module": self._counter_module_path(),
                "class": counter.get("class", _DEFAULT_COUNTER_CLASS),
                "function": counter.get("function", _DEFAULT_COUNTER_FUNCTION),
            }
        if etype == "fakesink":
            return {"sync": False}
        if etype == "autovideosink":
            return {"sync": False}
        if etype == "appsink":
            # Assign a stable name so get_by_name() works reliably across
            # reconnects: GStreamer increments the auto-name counter globally
            # (appsink0, appsink1, …) every time a new pipeline is built, so
            # the auto-name differs on every reconnect attempt.
            return {"name": "mjpeg_appsink", "emit-signals": True, "max-buffers": 1, "drop": True, "sync": False}
        if etype == "capsfilter":
            return {"caps": "video/x-raw,format=BGRx"}
        # decodebin, videoconvert and any other elements need no properties.
        return {}

    def _build_launch_string(self) -> str:
        elements = self._pipeline_def.get("elements", [])
        if not elements:
            raise ValueError("pipeline.yaml defines no elements")

        types = [element["type"] for element in elements]
        if self._debug:
            types = self._debug_element_chain(types)
        elif self._api_enabled:
            # API enabled but no debug display: use appsink for MJPEG streaming.
            types = self._api_element_chain(types)
        if self._is_yolo_model():
            # Insert the PersonFilter (a gvapython element) right before
            # gvatrack so multi-class YOLO detections are reduced to persons at
            # the metadata level. Intel person-only models skip this entirely.
            idx = next((i for i, t in enumerate(types) if t == "gvatrack"), len(types))
            types.insert(idx, _PERSON_FILTER_TYPE)
        if self._fps_logging:
            # Insert gvafpscounter ahead of the sink so it measures end-to-end
            # throughput (detect + track + count) for A/B model comparison.
            sinks = {"fakesink", "autovideosink", "ximagesink", "xvimagesink"}
            idx = next((i for i, t in enumerate(types) if t in sinks), len(types))
            types.insert(idx, "gvafpscounter")
        segments: list[str] = []
        for etype in types:
            props = self._element_properties(etype)
            segment = self._element_name(etype)
            for key, value in props.items():
                segment += f" {key}={self._format_value(value)}"
            segments.append(segment)

        launch = " ! ".join(segments)
        logger.info("Pipeline: %s", launch)
        return launch

    @staticmethod
    def _element_name(etype: str) -> str:
        """Map a synthetic element token to its real GStreamer element name."""
        if etype == _PERSON_FILTER_TYPE:
            return "gvapython"
        return etype

    def _debug_element_chain(self, types: list[str]) -> list[str]:
        """Insert gvawatermark and a display sink for debug visualization.

        gvawatermark draws boxes/IDs/confidence from the existing gvadetect +
        gvatrack metadata; the queue_counter overlay then adds ROI/count/
        status/FPS before the frame is rendered. fakesink is replaced by a
        videoconvert + display sink. No second inference is performed.
        """
        chain: list[str] = []
        for etype in types:
            if etype == "gvapython":
                # gvawatermark draws boxes/IDs/confidence; videoconvert+BGRx caps
                # force system-memory colour frames so the OpenCV overlay draws
                # in real colour even when inference runs on GPU (VA) memory.
                chain.extend(["gvawatermark", "videoconvert", "capsfilter"])
            if etype == "fakesink":
                chain.extend(["videoconvert", self._debug_sink])
                continue
            chain.append(etype)
        return chain

    def _api_element_chain(self, types: list[str]) -> list[str]:
        """Replace fakesink with appsink branch for MJPEG streaming.

        Inserts gvawatermark + videoconvert + BGRx capsfilter before gvapython
        (so _draw_overlay has colour frames) then routes to an appsink whose
        new-sample signal feeds frame_buffer. Used when api.enabled=true and
        visualization=false.
        """
        chain: list[str] = []
        for etype in types:
            if etype == "gvapython":
                chain.extend(["gvawatermark", "videoconvert", "capsfilter"])
            if etype == "fakesink":
                chain.extend(["videoconvert", "appsink"])
                continue
            chain.append(etype)
        return chain

    def build(self) -> None:
        """Construct the GStreamer pipeline and attach the bus watch."""
        launch_string = self._build_launch_string()
        self.pipeline = Gst.parse_launch(launch_string)

        source = self.pipeline.get_by_name(_SOURCE_ELEMENT_NAME)
        if source is not None:
            self._setup_rtsp_source(source)

        # Wire appsink new-sample signal → frame_buffer when API streaming is on.
        if self._api_enabled and not self._debug:
            appsink = self.pipeline.get_by_name("mjpeg_appsink")
            if appsink is not None:
                appsink.connect("new-sample", self._on_new_sample)
                logger.info("Appsink wired for MJPEG streaming")
            else:
                logger.warning("API enabled but appsink element not found in pipeline")

        self._bus = self.pipeline.get_bus()
        self._bus.add_signal_watch()
        self._bus.connect("message", self._on_bus_message)

    def _on_new_sample(self, appsink) -> int:
        """Appsink callback: pull buffer, convert to numpy, push to frame_buffer."""
        try:
            import numpy as np
            import frame_buffer

            sample = appsink.emit("pull-sample")
            if sample is None:
                return Gst.FlowReturn.OK

            buf = sample.get_buffer()
            caps = sample.get_caps()
            structure = caps.get_structure(0)
            width = structure.get_int("width")[1]
            height = structure.get_int("height")[1]

            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.OK
            try:
                # BGRx (4-channel) from capsfilter; drop alpha channel for cv2
                arr = np.frombuffer(map_info.data, dtype=np.uint8)
                arr = arr.reshape((height, width, 4))
                bgr = arr[:, :, :3].copy()
                frame_buffer.put(bgr)
            finally:
                buf.unmap(map_info)
        except Exception:  # noqa: BLE001
            logger.debug("appsink frame push failed", exc_info=True)
        return Gst.FlowReturn.OK

    def _setup_rtsp_source(self, source: Gst.Element) -> None:
        """Apply RTSP transport settings to the source element.

        ``urisourcebin``/``uridecodebin`` wrap an internal ``rtspsrc`` that is
        only available once the ``source-setup`` signal fires; a bare
        ``rtspsrc`` is configured directly.
        """
        factory = source.get_factory()
        factory_name = factory.get_name() if factory is not None else ""
        if factory_name == "rtspsrc":
            self._configure_rtsp_transport(source)
            return
        try:
            source.connect("source-setup", self._on_source_setup)
        except TypeError:
            logger.debug("Source %s has no 'source-setup' signal", factory_name)

    def _on_source_setup(self, _source_bin: Gst.Element, inner_source: Gst.Element) -> None:
        self._configure_rtsp_transport(inner_source)

    def _configure_rtsp_transport(self, rtsp_source: Gst.Element) -> None:
        source_cfg = self._config.get("source", {})
        transport = str(source_cfg.get("rtsp_transport", "tcp")).lower()
        if transport == "tcp":
            try:
                rtsp_source.set_property("protocols", _RTSP_LOWER_TRANS_TCP)
                logger.info("RTSP transport forced to TCP")
            except Exception:  # noqa: BLE001 - property may be absent
                logger.debug("Could not set protocols=tcp on RTSP source")
        latency = source_cfg.get("latency_ms")
        if latency is not None:
            try:
                rtsp_source.set_property("latency", int(latency))
            except Exception:  # noqa: BLE001
                logger.debug("Could not set latency on RTSP source")

    # ── lifecycle ────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Build (if needed), start the pipeline and run the main loop."""
        if self.pipeline is None:
            self.build()
        self._install_signal_handlers()
        self._start()
        try:
            self.loop.run()
        finally:
            self.stop()

    def _start(self) -> None:
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.error("Failed to set pipeline to PLAYING")
            self._schedule_reconnect()
        else:
            logger.info("queue-service pipeline started")

    def stop(self) -> None:
        """Tear down the pipeline and quit the main loop once."""
        if self._stopping:
            return
        self._stopping = True
        logger.info("Stopping queue-service pipeline")
        self._teardown_pipeline()
        if self.loop.is_running():
            self.loop.quit()

    def _teardown_pipeline(self) -> None:
        if self._bus is not None:
            self._bus.remove_signal_watch()
            self._bus = None
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None

    def _install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, sig, self._on_signal)

    def _on_signal(self) -> bool:
        logger.info("Received shutdown signal")
        self.stop()
        return GLib.SOURCE_REMOVE

    # ── bus / reconnection ───────────────────────────────────────────────────

    def _on_bus_message(self, _bus: Gst.Bus, message: Gst.Message) -> bool:
        mtype = message.type
        if mtype == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error("Pipeline ERROR: %s (%s)", err.message, debug or "")
            self._schedule_reconnect()
        elif mtype == Gst.MessageType.EOS:
            logger.warning("Pipeline reached EOS")
            self._schedule_reconnect()
        elif mtype == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            logger.warning("Pipeline WARNING: %s (%s)", warn.message, debug or "")
        elif mtype == Gst.MessageType.STATE_CHANGED:
            if self.pipeline is not None and message.src == self.pipeline:
                _old, new, _pending = message.parse_state_changed()
                if new == Gst.State.PLAYING:
                    self._retries = 0
        return True

    def _schedule_reconnect(self) -> None:
        if self._stopping or self._reconnecting:
            return
        self._retries += 1
        if self._max_retries is not None and self._retries > self._max_retries:
            logger.error(
                "Max reconnect attempts (%d) exceeded; stopping", self._max_retries
            )
            self.stop()
            return
        self._reconnecting = True
        delay = max(1, int(self._reconnect_delay))
        logger.warning(
            "Scheduling RTSP reconnect #%d in %ds", self._retries, delay
        )
        GLib.timeout_add_seconds(delay, self._do_reconnect)

    def _do_reconnect(self) -> bool:
        self._reconnecting = False
        if self._stopping:
            return GLib.SOURCE_REMOVE
        logger.info("Reconnecting to RTSP source (attempt %d)", self._retries)
        self._teardown_pipeline()
        try:
            self.build()
            self._start()
        except Exception:  # noqa: BLE001 - keep retrying on build failure
            logger.exception("Reconnect attempt failed")
            self._schedule_reconnect()
        return GLib.SOURCE_REMOVE


def run_pipeline(conf_dir: Path | str | None = None) -> None:
    """Convenience entry point used by ``main.py``."""
    QueuePipeline(conf_dir).run()


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("QUEUE_SERVICE_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_pipeline()
