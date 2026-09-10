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

from config_loader import load_config_dict

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import GLib, Gst, GstVideo  # noqa: E402  (must follow require_version)


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

# gvapython binding for the YOLO26 tensor decoder.
#
# YOLO26 cannot go through `gvadetect`: the newest YOLO post-processing
# converter compiled into DLStreamer 2026.1.0 and 2026.2.0 is `yolo_v8`.
# The `yolo_v10`/`yolo_v11` names are *accepted without error* yet decode to
# zero detections, whereas an unknown name is rejected loudly -- a silent
# failure that looks exactly like "the camera sees nobody".
#
# The `decoder` post-processing mode therefore replaces gvadetect with
# `gvainference` (raw tensor out, no post-processing) plus a gvapython element
# running src/yolo_decoder.py, which attaches the person regions itself. Every
# downstream element is unchanged because it still receives ordinary
# GstVideoRegionOfInterestMeta.
_DECODER_CLASS = "YoloDecoder"
_DECODER_FUNCTION = "process_frame"
_DECODER_TYPE = "yolo_decoder"

# model.postproc values. "gvadetect" keeps the stock DLStreamer path (Intel
# person-only models, and YOLO versions whose converter actually works);
# "decoder" selects the gvainference + YoloDecoder path described above.
_POSTPROC_DECODER = "decoder"
_POSTPROC_GVADETECT = "gvadetect"

# Synthetic element tokens for the VA (GPU) zero-copy memory path.
# decodebin already autoplugs vah264dec, which produces VASurface memory. The
# stock chain then used a plain `videoconvert`, which downloads every frame to
# system RAM and colour-converts it on the CPU before gvadetect -- and DLStreamer
# logs an explicit warning about it when device=NPU/GPU:
#   "System memory is being used for inference on device 'NPU'. For optimal
#    performance, use VA memory in the pipeline: vapostproc ! VAMemory ! gvadetect"
# Keeping the surface on the GPU all the way through gvadetect/gvatrack/
# gvawatermark, and only downloading once at the end (for the OpenCV overlay in
# queue_counter.py, which needs system-memory BGRx), measured on this box over a
# 22 s identical run: 17.37 s CPU -> 6.09 s CPU (~65% less, ~0.5 core freed).
#
# NOTE: the download element must ALSO be vapostproc. A plain `videoconvert`
# cannot accept VAMemory on its sink pad and the pipeline dies during caps
# negotiation with "streaming stopped, reason not-linked".
_VA_UPLOAD_TYPE = "va_upload"      # vapostproc, system/VA in -> VA out
_VA_CAPS_TYPE = "va_caps"          # capsfilter pinning video/x-raw(memory:VAMemory)
_VA_DOWNLOAD_TYPE = "va_download"  # vapostproc, VA in -> system BGRx out
_VA_MEMORY_CAPS = "video/x-raw(memory:VAMemory)"

# Synthetic element token for a thread-decoupling leaky queue.
#
# Without these the whole graph -- VA decode, NPU inference, gvatrack,
# gvawatermark, the VA download, the OpenCV overlay and the appsink push --
# runs synchronously on the single thread rtspsrc uses to drain the RTP
# socket. Any hiccup in inference or the Python overlay therefore stops the
# service reading from the socket, which is what made the jitter buffer
# overrun and (with drop-on-latency enabled) discard NAL units.
#
# A `queue` inserts a thread boundary: upstream keeps receiving and decoding
# while downstream is busy. `leaky=downstream` drops the OLDEST buffer once
# the queue is full, which is the whole point of this change -- it relocates
# the drop point from the compressed domain (dropping an RTP packet corrupts
# the frame *and every P-frame after it, until the next IDR*) to the decoded
# domain (dropping a decoded frame just skips it, and is invisible).
#
# The queues are deliberately shallow: a deep queue would trade the corruption
# for latency, and this is a live feed. 4 buffers is ~130 ms at 30 fps.
_QUEUE_TYPE = "queue"
_QUEUE_MAX_BUFFERS = 4



# Element tokens after which a decoupling queue is inserted. Chosen at the
# three points where the cost profile changes sharply:
#   decodebin  -> isolates RTP reception + decode from inference
#   gvatrack   -> isolates inference from the Python/OpenCV overlay
#   (sink)     -> isolates the overlay from the sink, handled separately
_QUEUE_AFTER = ("decodebin", "gvatrack")
_SINK_TYPES = {"appsink", "fakesink", "autovideosink", "ximagesink", "xvimagesink"}

# Render node that must be present for the VA elements to work at all.
_DRI_RENDER_DIR = Path("/dev/dri")

# RTSP source element types whose transport we configure.
_RTSP_SOURCE_TYPES = {"urisourcebin", "uridecodebin", "rtspsrc"}
_SOURCE_ELEMENT_NAME = "queue_source"


class QueuePipeline:
    """Builds and runs the DLStreamer queue-service pipeline.

    ``queue-config.yaml`` is loaded through the shared ``config_loader``
    module so ``QUEUE_SERVICE__*`` environment overrides (e.g.
    ``QUEUE_SERVICE__MODEL__DEVICE``, wired from ``QUEUE_DEVICE`` in
    ``docker-compose.yml``/``.env``) reach the actual pipeline element
    properties. ``pipeline.yaml`` (the element graph shape) has no
    overridable settings and is loaded as plain YAML.
    """

    def __init__(self, conf_dir: Path | str | None = None) -> None:
        self._conf_dir = Path(conf_dir) if conf_dir else CONF_DIR
        self._config = load_config_dict(str(self._conf_dir / "queue-config.yaml"))
        self._pipeline_def = self._load_yaml(self._conf_dir / "pipeline.yaml")

        source_cfg = self._config.get("source", {})
        self._reconnect_delay = float(source_cfg.get("reconnect_delay_seconds", 5.0))
        max_retries = source_cfg.get("max_reconnect_attempts")
        self._max_retries = int(max_retries) if max_retries is not None else None

        debug_cfg = self._config.get("debug", {}) or {}
        self._debug = bool(debug_cfg.get("visualization", False))
        self._debug_sink = str(debug_cfg.get("sink", "autovideosink"))
        self._fps_logging = bool(debug_cfg.get("fps_logging", False))

        self._va_memory = self._resolve_va_memory()

        api_cfg = self._config.get("api", {}) or {}
        self._api_enabled = bool(api_cfg.get("enabled", False))
        jpeg_quality = int(api_cfg.get("jpeg_quality", 80))
        stream_max_height = int(api_cfg.get("stream_max_height", 0))
        stream_max_fps = float(api_cfg.get("stream_max_fps", 0))
        if self._api_enabled:
            import frame_buffer
            frame_buffer.configure(jpeg_quality, stream_max_height, stream_max_fps)

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

    def _resolve_va_memory(self) -> bool:
        """Decide whether to keep frames in GPU (VA) memory through inference.

        ``model.va_memory`` accepts ``auto`` (default), ``true`` or ``false``.
        ``auto`` enables the VA path only when it can actually work: a DRI
        render node must be present, and the inference device must not be CPU
        (a CPU device gains nothing and would pay for the extra download).

        Returns:
            True when the pipeline should be built with the VA zero-copy path.
        """
        model_cfg = self._config.get("model", {}) or {}
        setting = str(model_cfg.get("va_memory", "auto")).strip().lower()
        device = str(model_cfg.get("device", "CPU")).strip().upper()

        if setting in {"false", "no", "off", "0"}:
            logger.info("VA memory path disabled by configuration")
            return False

        has_render_node = any(_DRI_RENDER_DIR.glob("renderD*")) if _DRI_RENDER_DIR.is_dir() else False

        if setting in {"true", "yes", "on", "1"}:
            if not has_render_node:
                # Forcing it would produce an unrunnable pipeline, so refuse
                # rather than crash-loop the service on a CPU-only host.
                logger.warning(
                    "model.va_memory=true but no /dev/dri render node is present; "
                    "falling back to the system-memory path"
                )
                return False
            return True

        if not has_render_node:
            logger.info("VA memory path unavailable (no /dev/dri render node); using system memory")
            return False
        if device == "CPU":
            logger.info("VA memory path skipped for device=CPU; using system memory")
            return False

        logger.info("VA memory path enabled for device=%s (zero-copy into inference)", device)
        return True

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

    def _postproc_mode(self) -> str:
        """Return the configured detection post-processing mode.

        ``decoder`` swaps gvadetect for gvainference + YoloDecoder; anything
        else keeps the stock gvadetect path.
        """
        raw = self._config.get("model", {}).get("postproc")
        return str(raw).strip().lower() if raw else _POSTPROC_GVADETECT

    def _uses_decoder(self) -> bool:
        return self._postproc_mode() == _POSTPROC_DECODER

    def _decoder_module_path(self) -> str:
        return str((BASE_DIR / "src" / "yolo_decoder.py").resolve())

    def _detector_model_path(self) -> str:
        """Resolve the detector IR, preferring a static-shaped variant.

        The YOLO26 export has a dynamic batch dimension, which the NPU plugin
        refuses to compile. ``model_manager.ensure_static_ir`` writes a
        reshaped ``*_static.xml`` next to the original; this returns that file
        when it exists so the pipeline and the provisioning step never
        disagree about which IR is in use.
        """
        configured = self._resolve(self._config["model"]["ir_path"])
        try:
            from model_manager import ensure_static_ir

            return ensure_static_ir(configured, None)
        except Exception:  # noqa: BLE001 - fall back to the configured IR
            logger.debug("Static IR resolution unavailable; using %s", configured)
            return configured

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
                "model": self._detector_model_path(),
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
        if etype == "gvainference":
            # Raw tensor inference: no model-proc and no threshold, because
            # YoloDecoder owns post-processing (and its own thresholding).
            model = cfg["model"]
            return {
                "model": self._detector_model_path(),
                "device": model.get("device", "CPU"),
                "inference-interval": model.get("inference_interval", 1),
            }
        if etype == _DECODER_TYPE:
            return {
                "module": self._decoder_module_path(),
                "class": _DECODER_CLASS,
                "function": _DECODER_FUNCTION,
            }
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
        if etype == "gvawatermark":
            class_label = str(cfg.get("model", {}).get("class_label", "")).strip()
            if class_label:
                return {"displ-cfg": f"hide-roi={class_label}"}
            return {}
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
            # Format only -- never a resolution. See _convert_to_bgrx().
            return {"caps": "video/x-raw,format=BGRx"}
        if etype == _VA_CAPS_TYPE:
            return {"caps": _VA_MEMORY_CAPS}
        if etype == _QUEUE_TYPE:
            # leaky=downstream (2) drops the oldest queued buffer when full.
            # max-size-bytes/-time are zeroed so max-size-buffers is the only
            # bound -- otherwise the byte limit (2 MB by default) is hit first
            # on 1080p BGRx and the queue never reaches its buffer depth.
            return {
                "max-size-buffers": _QUEUE_MAX_BUFFERS,
                "max-size-bytes": 0,
                "max-size-time": 0,
                "leaky": "downstream",
            }
        # decodebin, videoconvert, vapostproc and any other elements need no
        # properties.
        return {}

    def _build_launch_string(self) -> str:
        elements = self._pipeline_def.get("elements", [])
        if not elements:
            raise ValueError("pipeline.yaml defines no elements")

        types = [element["type"] for element in elements]
        if self._va_memory:
            types = self._va_ingress_chain(types)
        if self._debug:
            types = self._debug_element_chain(types)
        elif self._api_enabled:
            # API enabled but no debug display: use appsink for MJPEG streaming.
            types = self._api_element_chain(types)
        if self._uses_decoder():
            # Replace gvadetect with gvainference + YoloDecoder. The decoder
            # emits person regions only, so PersonFilter would be a no-op and
            # is deliberately not inserted on this path.
            types = self._decoder_element_chain(types)
        elif self._is_yolo_model():
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
        # Applied last so the insertion points are computed against the final
        # element list, after the VA / debug / API / filter transforms above.
        types = self._insert_queues(types)
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
        if etype == _DECODER_TYPE:
            return "gvapython"
        if etype in {_VA_UPLOAD_TYPE, _VA_DOWNLOAD_TYPE}:
            return "vapostproc"
        if etype == _VA_CAPS_TYPE:
            return "capsfilter"
        return etype

    def _decoder_element_chain(self, types: list[str]) -> list[str]:
        """Swap ``gvadetect`` for ``gvainference`` + the YOLO tensor decoder.

        Returns a new list; the input is not mutated. If the graph has no
        ``gvadetect`` the chain is returned unchanged, so a hand-edited
        pipeline.yaml cannot silently lose its detector.
        """
        out = list(types)
        try:
            idx = out.index("gvadetect")
        except ValueError:
            logger.warning(
                "postproc=decoder requested but the pipeline has no gvadetect; "
                "leaving the element graph unchanged"
            )
            return out
        out[idx] = "gvainference"
        out.insert(idx + 1, _DECODER_TYPE)
        return out

    def _va_ingress_chain(self, types: list[str]) -> list[str]:
        """Feed gvadetect from GPU (VA) memory instead of system memory.

        Replaces the ``videoconvert`` sitting between the decoder and
        ``gvadetect`` with ``vapostproc`` + a VAMemory capsfilter, so the
        surface produced by the hardware decoder is never round-tripped
        through system RAM just to be handed to the inference element.

        Args:
            types: Element tokens in pipeline order.

        Returns:
            The element tokens with the VA upload substituted in.
        """
        idx = next((i for i, t in enumerate(types) if t == "gvadetect"), None)
        if idx is None:
            return types

        chain = list(types)
        va_upload = [_VA_UPLOAD_TYPE, _VA_CAPS_TYPE]
        if idx > 0 and chain[idx - 1] == "videoconvert":
            chain[idx - 1:idx] = va_upload
        else:
            chain[idx:idx] = va_upload
        return chain

    def _convert_to_bgrx(self) -> list[str]:
        """Elements that land frames in system-memory BGRx for the overlay.

        ``queue_counter.py`` draws with OpenCV and therefore needs CPU-mappable
        BGRx. On the VA path the download must be done by ``vapostproc``;
        ``videoconvert`` cannot accept VAMemory on its sink pad.

        DO NOT ADD A RESOLUTION TO THIS CAPSFILTER. Making this element scale
        as well as convert looks like free performance -- the resize runs on
        the GPU, and the overlay, the frame copy and the JPEG encode all get a
        smaller frame -- but it silently breaks queue counting:

        * Detection metadata (``GstVideoRegionOfInterestMeta``) is attached by
          ``gvadetect`` upstream. GStreamer only carries a meta across a
          transform when that meta can be transformed too, so a scaling
          ``vapostproc`` **drops every ROI meta**. ``frame.regions()`` in
          queue_counter then returns nothing, no tracks are ever confirmed and
          the queue count sits at 0 forever.
        * The failure is deceptive: ``gvawatermark`` sits *before* this element
          and has already burned its boxes into the pixels, so the MJPEG stream
          still shows bounding boxes while the counter sees no detections at
          all. Verified live -- 516 consecutive frames logged "detections=0"
          while the stream showed boxes.
        * Separately, a width that is not a multiple of 64 makes the VA pool
          pad each row, and ``gstgva.VideoFrame.data()`` reshapes the mapping
          as height x width x 4 without consulting the stride, which shears the
          picture into horizontal smears (1440 wide destroyed the image; 1280
          was clean).

        The browser stream is downscaled instead by ``frame_buffer``, on the
        CPU, lazily, only when a client is connected -- see
        ``api.stream_max_height``.
        """
        if self._va_memory:
            return [_VA_DOWNLOAD_TYPE, "capsfilter"]
        return ["videoconvert", "capsfilter"]

    def _insert_queues(self, types: list[str]) -> list[str]:
        """Insert leaky queues to decouple the pipeline's processing stages.

        Applied last, after every other chain transform, so the insertion
        points are computed against the final element list.

        A queue is placed after each element in ``_QUEUE_AFTER`` and
        immediately before the terminal sink. See ``_QUEUE_TYPE`` for why this
        matters: without a thread boundary, a slow overlay stalls RTP
        reception and the video corrupts rather than merely dropping frames.
        """
        chain: list[str] = []
        for etype in types:
            if etype in _SINK_TYPES and chain and chain[-1] != _QUEUE_TYPE:
                chain.append(_QUEUE_TYPE)
            chain.append(etype)
            if etype in _QUEUE_AFTER:
                chain.append(_QUEUE_TYPE)
        return chain

    def _debug_element_chain(self, types: list[str]) -> list[str]:
        """Insert gvawatermark and a display sink for debug visualization.

        gvawatermark stays in the chain for metadata-to-frame integration, but
        person ROI drawing is disabled when model.class_label is configured,
        allowing QueueCounter to be the only bounding-box renderer. fakesink
        is replaced by a videoconvert + display sink. No second inference is
        performed.
        """
        chain: list[str] = []
        for etype in types:
            if etype == "gvapython":
                # gvawatermark draws boxes/IDs/confidence; the convert + BGRx
                # caps force system-memory colour frames so the OpenCV overlay
                # draws in real colour even when inference ran on VA memory.
                chain.append("gvawatermark")
                chain.extend(self._convert_to_bgrx())
            if etype == "fakesink":
                chain.extend(["videoconvert", self._debug_sink])
                continue
            chain.append(etype)
        return chain

    def _api_element_chain(self, types: list[str]) -> list[str]:
        """Replace fakesink with appsink branch for MJPEG streaming.

        Inserts gvawatermark + videoconvert + BGRx capsfilter before gvapython
        (so _draw_overlay has colour frames) then routes to an appsink whose
        new-sample signal feeds frame_buffer. gvawatermark's person ROI drawing
        is disabled when model.class_label is configured, allowing QueueCounter
        to own the final box colors. Used when api.enabled=true and
        visualization=false.
        """
        chain: list[str] = []
        for etype in types:
            if etype == "gvapython":
                chain.append("gvawatermark")
                chain.extend(self._convert_to_bgrx())
            if etype == "fakesink":
                # No videoconvert here: the capsfilter added by
                # _convert_to_bgrx() already pins system-memory BGRx directly
                # upstream of gvapython, and appsink accepts any caps, so a
                # videoconvert would only add an element to the hot path.
                chain.append("appsink")
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
        """Appsink callback: pull buffer, convert to numpy, push to frame_buffer.

        Acts as a fallback producer: ``queue_counter`` already pushes the same
        frame with the overlay drawn on it, so this path is skipped (before the
        expensive frame copy) while that producer is healthy.
        """
        try:
            import numpy as np
            import frame_buffer

            if not frame_buffer.accepts(frame_buffer.SOURCE_APPSINK):
                return Gst.FlowReturn.OK

            sample = appsink.emit("pull-sample")
            if sample is None:
                return Gst.FlowReturn.OK

            buf = sample.get_buffer()
            caps = sample.get_caps()
            structure = caps.get_structure(0)
            width = structure.get_int("width")[1]
            height = structure.get_int("height")[1]

            # Rows may be padded to a hardware alignment boundary (common for
            # VA-downloaded surfaces), so the buffer's real per-row byte count
            # (stride) can exceed width * 4. Reshaping straight to
            # (height, width, 4) then silently assumes zero padding, which
            # shifts every following row's start offset and shows up as
            # diagonal shearing in the MJPEG output.
            stride = self._frame_stride(buf, caps, width)

            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.OK
            try:
                # BGRx (4-channel) from capsfilter; drop alpha channel for cv2
                arr = np.frombuffer(map_info.data, dtype=np.uint8)
                expected = stride * height
                if arr.size < expected:
                    # Stride guess is too large for the mapping -- fall back to
                    # the packed layout rather than raising in a GStreamer
                    # callback (which would silently stop the fallback
                    # producer for the rest of the run).
                    stride = width * 4
                    expected = stride * height
                    if arr.size < expected:
                        return Gst.FlowReturn.OK
                arr = arr[:expected].reshape((height, stride))[:, : width * 4]
                arr = arr.reshape((height, width, 4))
                bgr = arr[:, :, :3].copy()
                frame_buffer.put(bgr, frame_buffer.SOURCE_APPSINK)
            finally:
                buf.unmap(map_info)
        except Exception:  # noqa: BLE001
            logger.debug("appsink frame push failed", exc_info=True)
        return Gst.FlowReturn.OK

    @staticmethod
    def _frame_stride(buf, caps, width: int) -> int:
        """Return the real first-plane row stride, in bytes, for ``buf``.

        Sources are tried most- to least- authoritative:

        1. ``GstVideoMeta`` on the buffer -- the only source that describes
           *this* buffer's actual layout, including any pool padding.
        2. ``GstVideo.VideoInfo`` parsed from the caps -- the format's default
           layout. Note this reports the *unpadded* stride, so it only helps
           when the buffer carries no meta.
        3. ``width * 4`` -- packed BGRx.

        The previous implementation called ``VideoInfo.from_caps()``, which
        raises ``NotImplementedError`` in this GStreamer build ("use
        VideoInfo.new_from_caps instead"). Because the whole callback is
        wrapped in ``except Exception``, that turned into a silent early
        return: the fallback frame producer never pushed a single frame, and
        the failure was invisible at INFO level.
        """
        meta = GstVideo.buffer_get_video_meta(buf)
        if meta is not None and meta.n_planes > 0 and meta.stride[0] > 0:
            return meta.stride[0]
        try:
            info = GstVideo.VideoInfo.new_from_caps(caps)
        except (AttributeError, NotImplementedError):
            info = None
        if info is not None and info.stride[0] > 0:
            return info.stride[0]
        return width * 4

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
                logger.info("RTSP jitter buffer set to %s ms", int(latency))
            except Exception:  # noqa: BLE001
                logger.debug("Could not set latency on RTSP source")
        if bool(source_cfg.get("drop_on_latency", False)):
            try:
                rtsp_source.set_property("drop-on-latency", True)
                logger.info("RTSP drop-on-latency enabled")
            except Exception:  # noqa: BLE001
                logger.debug("Could not set drop-on-latency on RTSP source")

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
