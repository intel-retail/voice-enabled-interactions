"""Model provisioning for queue-service.

Ensures the configured detector's OpenVINO IR exists under ``models/`` at the
path consumed by ``pipeline.py`` (``config.model.ir_path``). The default
detector is Intel Open Model Zoo ``person-detection-retail-0013``.

Provisioning is provider-based so the detector can be changed later (e.g. to
YOLO11) purely through configuration: ``pipeline.py`` only ever reads
``ir_path``/``proc_path``, while this module resolves *how* the IR is
obtained. The ``omz`` provider downloads the pre-converted IR directly from the
Intel Open Model Zoo storage server (the same artifact ``omz_downloader``
fetches), because the ``intel/dlstreamer:2026.3.0-ubuntu24-rc1`` image ships neither
the OMZ CLI tools nor ``openvino-dev``.
"""
import json
import logging
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import yaml

from config_loader import config

logger = logging.getLogger(__name__)

# queue-service/ (parent of src/)
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DEFAULT_PRECISION = "FP16"
DEFAULT_PROVIDER = "omz"

# Open Model Zoo storage server. omz_downloader resolves these same URLs from
# each model's model.yml; for an Intel pre-trained model the IR is served
# pre-converted, so a plain HTTPS GET is sufficient (no conversion tooling).
OMZ_STORAGE_BASE = "https://storage.openvinotoolkit.org/repositories/open_model_zoo"
OMZ_RELEASE = "2023.0"
OMZ_MODELS_BIN_INDEX = "1"
_DOWNLOAD_TIMEOUT_SECONDS = 60

# DL Streamer >= 2026.3 refuses to initialise *any* inference element whose
# model it classifies as YOLO -- including `gvainference`, which performs no
# post-processing at all -- unless a `metadata.yaml` sits next to the IR:
#
#   YOLO model (model_type "YOLO") requires <models>/metadata.yaml
#   to determine the model version and task.
#
# The classification is name-based, so our `yolo26n*.xml` trips it even on the
# `postproc: decoder` path where src/yolo_decoder.py owns decoding entirely and
# DL Streamer never looks at the tensors. 2026.2 had no such gate, so the file
# is absent from every IR we provision. Writing it keeps that raw-tensor path
# working; the values only have to satisfy the gate, because nothing in
# DL Streamer consumes them once `gvainference` is the element in use.
_YOLO_METADATA_FILENAME = "metadata.yaml"
_YOLO_METADATA_TASK = "detect"
# Ultralytics' own exporter stamps the package version here. DL Streamer only
# parses it to pick a YOLO generation, and the decoder path ignores that
# choice, so a plain Ultralytics-era version is enough.
_YOLO_METADATA_VERSION = "8.3.0"
_YOLO_DEFAULT_IMGSZ = [640, 640]
_YOLO_DEFAULT_NAMES = {0: "person"}


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return str((BASE_DIR / path).resolve())


# ── OMZ provider (direct IR download) ────────────────────────────────────────

def _omz_file_url(model_cfg, name: str, precision: str, filename: str) -> str:
    """Build the OMZ storage URL for one IR file.

    The storage layout/release can be overridden per model via the optional
    ``omz_storage_base``/``omz_release``/``omz_index`` config keys.
    """
    base = getattr(model_cfg, "omz_storage_base", None) or OMZ_STORAGE_BASE
    release = getattr(model_cfg, "omz_release", None) or OMZ_RELEASE
    index = str(getattr(model_cfg, "omz_index", None) or OMZ_MODELS_BIN_INDEX)
    return f"{base}/{release}/models_bin/{index}/{name}/{precision}/{filename}"


def _download_file(url: str, dst: Path) -> None:
    """Download ``url`` to ``dst``.

    The OMZ storage server answers unknown paths with an HTTP 200 HTML
    placeholder, so a successful status code is not enough: reject any
    ``text/html`` response and empty downloads.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    request = urllib.request.Request(
        url, headers={"User-Agent": "queue-service/model_manager"}
    )
    try:
        with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
            if response.headers.get_content_type() == "text/html":
                raise RuntimeError(
                    f"Model not found at {url} (server returned an HTML page)"
                )
            with open(dst, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    if dst.stat().st_size == 0:
        raise RuntimeError(f"Downloaded an empty file from {url}")


def _install_ir(src_xml: Path, ir_path: str) -> None:
    """Copy the IR (.xml + .bin) to the path expected by pipeline.py."""
    dst_xml = Path(ir_path)
    dst_xml.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_xml, dst_xml)
    src_bin = src_xml.with_suffix(".bin")
    if src_bin.is_file():
        shutil.copy2(src_bin, dst_xml.with_suffix(".bin"))
    logger.info("Installed IR -> %s", dst_xml)


def _provision_omz(model_cfg, ir_path: str) -> None:
    """Provision an Open Model Zoo IR by downloading it from OMZ storage.

    Fetches ``<name>.xml`` and ``<name>.bin`` for the configured precision into
    an OMZ-style staging layout (``models/<name>/<precision>/``), then installs
    them at ``ir_path`` for ``pipeline.py``.
    """
    name = getattr(model_cfg, "omz_name", None) or getattr(model_cfg, "name")
    precision = getattr(model_cfg, "precision", DEFAULT_PRECISION)

    stage_dir = MODELS_DIR / name / precision
    src_xml = stage_dir / f"{name}.xml"
    src_bin = stage_dir / f"{name}.bin"

    _download_file(_omz_file_url(model_cfg, name, precision, f"{name}.xml"), src_xml)
    _download_file(_omz_file_url(model_cfg, name, precision, f"{name}.bin"), src_bin)

    _install_ir(src_xml, ir_path)


def _provision_local(model_cfg, ir_path: str) -> None:
    """No-op provider for locally supplied IR (e.g. yolov8n.xml/.bin).

    The IR is expected to already exist at ``ir_path``; ``ensure_model`` only
    reaches here when it is missing, so fail clearly instead of downloading.
    """
    raise RuntimeError(
        f"provider 'local' requires the IR to be present at {ir_path}; "
        f"none found. Place the .xml/.bin there or use a download provider."
    )


_PROVIDERS = {
    "omz": _provision_omz,
    "local": _provision_local,
}

# ── public API ───────────────────────────────────────────────────────────────

def _provider_for(model_cfg) -> str:
    explicit = getattr(model_cfg, "source", None) or getattr(model_cfg, "provider", None)
    return str(explicit).lower() if explicit else DEFAULT_PROVIDER


def _is_yolo_model(ir_path: str) -> bool:
    """Return True if DL Streamer would classify this IR as a YOLO model.

    Mirrors DL Streamer's own name-based detection, which is what triggers the
    ``metadata.yaml`` requirement in 2026.3+.

    Args:
        ir_path: Path to the OpenVINO IR ``.xml``.

    Returns:
        True when the IR filename marks it as a YOLO model.
    """
    return "yolo" in Path(ir_path).stem.lower()


def _yolo_labels(ir_path: str) -> dict:
    """Return ``{class_index: label}`` for a YOLO IR.

    Reuses the labels already declared in the model-proc JSON that ships beside
    the IR, so metadata.yaml never contradicts the model. Falls back to a
    person-only map, which is all queue-service counts.

    Args:
        ir_path: Path to the OpenVINO IR ``.xml``.

    Returns:
        Mapping of class index to label name.
    """
    proc_json = Path(os.path.splitext(ir_path)[0] + ".json")
    if not proc_json.is_file():
        # Static IRs are written as <name>_static.xml with no model-proc of
        # their own; fall back to the one belonging to the source IR.
        stem = proc_json.stem
        if stem.endswith("_static"):
            proc_json = proc_json.with_name(stem[: -len("_static")] + ".json")
    try:
        with open(proc_json, encoding="utf-8") as fh:
            postproc = json.load(fh).get("output_postproc") or []
        for entry in postproc:
            labels = entry.get("labels")
            if labels:
                return {i: name for i, name in enumerate(labels)}
    except (OSError, ValueError) as exc:
        logger.debug("Could not read labels from %s: %s", proc_json, exc)
    return dict(_YOLO_DEFAULT_NAMES)


def _yolo_imgsz(ir_path: str) -> list:
    """Return the model's ``[height, width]`` input size.

    Args:
        ir_path: Path to the OpenVINO IR ``.xml``.

    Returns:
        Two-element ``[height, width]`` list, defaulting to 640x640 when the
        IR cannot be read or its spatial dims are dynamic.
    """
    try:
        from openvino import Core  # imported lazily; see ensure_static_ir

        shape = Core().read_model(ir_path).input().get_partial_shape()
        if shape.rank.is_static and len(shape) == 4:
            height, width = shape[2], shape[3]
            if height.is_static and width.is_static:
                return [height.get_length(), width.get_length()]
    except Exception as exc:  # pragma: no cover - diagnostic only
        logger.debug("Could not read input shape from %s: %s", ir_path, exc)
    return list(_YOLO_DEFAULT_IMGSZ)


def ensure_yolo_metadata(ir_path: str) -> str:
    """Write the ``metadata.yaml`` DL Streamer 2026.3+ demands for YOLO IRs.

    No-op for non-YOLO models and when the file already exists, so a
    hand-authored metadata.yaml is never overwritten.

    Args:
        ir_path: Path to the OpenVINO IR ``.xml`` that will be inferenced.

    Returns:
        ``ir_path`` unchanged, so this composes with the other ensure_* steps.

    Raises:
        RuntimeError: The metadata is missing and could not be written. The
            pipeline cannot start without it, so this aborts startup rather
            than leaving a service whose /health passes while it detects
            nothing.
    """
    if not _is_yolo_model(ir_path):
        return ir_path

    metadata_path = Path(ir_path).parent / _YOLO_METADATA_FILENAME
    if metadata_path.is_file():
        logger.debug("YOLO metadata already present: %s", metadata_path)
        return ir_path

    metadata = {
        "description": f"Generated by queue-service for {Path(ir_path).name}",
        "author": "queue-service",
        "version": _YOLO_METADATA_VERSION,
        "task": _YOLO_METADATA_TASK,
        "batch": 1,
        "imgsz": _yolo_imgsz(ir_path),
        "names": _yolo_labels(ir_path),
    }
    # Stage in the same directory so the publish step is a same-filesystem
    # rename-equivalent: a torn write can never be mistaken for a complete
    # file by the next startup. os.link (not os.replace) is the publish
    # primitive because it fails rather than clobbering a metadata.yaml that
    # appeared in the meantime -- the same invariant as the is_file() check
    # above, held across the write.
    tmp_name = None
    try:
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=metadata_path.parent, prefix=".metadata-", suffix=".yaml"
        )
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            yaml.safe_dump(metadata, fh, sort_keys=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.link(tmp_name, metadata_path)
    except FileExistsError:
        logger.debug("YOLO metadata created concurrently: %s", metadata_path)
        return ir_path
    except OSError as exc:
        raise RuntimeError(
            f"Could not write the YOLO metadata DL Streamer 2026.3+ requires "
            f"at {metadata_path}: {exc}"
        ) from exc
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:  # pragma: no cover - best-effort staging cleanup
                pass
    logger.info("Wrote DL Streamer YOLO metadata: %s", metadata_path)
    return ir_path


def ensure_model(model_cfg=None) -> str:
    """Ensure the configured detector IR exists; return its path.

    Skips provisioning when the IR is already present. Otherwise dispatches
    to the configured provider (default ``omz``) to download/prepare it.
    Finally, ensures the IR has a static batch dimension when required by the
    target device (see :func:`ensure_static_ir`).
    """
    model_cfg = model_cfg if model_cfg is not None else getattr(config, "model", None)
    if model_cfg is None:
        raise ValueError("No 'model' section found in configuration")

    ir_path = _resolve(getattr(model_cfg, "ir_path"))
    if os.path.isfile(ir_path):
        logger.info("Model IR already present: %s", ir_path)
        return ensure_yolo_metadata(ensure_static_ir(ir_path, model_cfg))

    provider = _provider_for(model_cfg)
    handler = _PROVIDERS.get(provider)
    if handler is None:
        raise NotImplementedError(
            f"Model provider '{provider}' is not supported yet; "
            f"add a provisioning function to _PROVIDERS"
        )

    logger.info("Model IR missing; provisioning '%s' via '%s' provider",
                getattr(model_cfg, "name", "<unknown>"), provider)
    handler(model_cfg, ir_path)

    if not os.path.isfile(ir_path):
        raise RuntimeError(f"Model provisioning did not produce IR at {ir_path}")
    logger.info("Model ready: %s", ir_path)
    return ensure_yolo_metadata(ensure_static_ir(ir_path, model_cfg))


def ensure_static_ir(ir_path: str, model_cfg=None) -> str:
    """Return an IR path whose batch dimension is static.

    The published YOLO26 export carries a dynamic batch (``[?,3,640,640]``).
    OpenVINO's CPU plugin tolerates that, but the NPU plugin refuses to
    compile it::

        Upper bounds are not specified for node '.../Convolution'
        (type 'Convolution'): input '0' bounds are
        '[9223372036854775807, 3, 640, 640]'

    Reshaping to ``[1,3,640,640]`` fixes it -- measured on this platform the
    reshaped model compiles and runs at 11.6 ms/inference on NPU and 4.3 ms on
    GPU, comfortably inside a 30 fps budget.

    The reshaped IR is written alongside the original as ``<name>_static.xml``
    and reused on subsequent starts. The original is never modified, so a
    rollback is just a config change.

    Args:
        ir_path: Path to the IR as configured.
        model_cfg: Model configuration section (used only for logging).

    Returns:
        Path to an IR safe to compile on the configured device. Falls back to
        ``ir_path`` unchanged if the model is already static, or if reshaping
        is not possible for any reason.
    """
    try:
        import openvino as ov
    except ImportError:
        logger.debug("OpenVINO not importable; skipping static-IR check")
        return ir_path

    static_path = os.path.splitext(ir_path)[0] + "_static.xml"
    if os.path.isfile(static_path):
        # A static IR left over from an earlier export of a different head
        # (e.g. raw 84x8400 vs end-to-end 300x6) silently yields zero
        # detections, so only reuse it if it is at least as new as the source
        # graph (.xml) and weights (.bin).
        source_mtime = os.path.getmtime(ir_path)
        source_bin = os.path.splitext(ir_path)[0] + ".bin"
        if os.path.isfile(source_bin):
            source_mtime = max(source_mtime, os.path.getmtime(source_bin))
        if os.path.getmtime(static_path) >= source_mtime:
            logger.info("Using static IR: %s", static_path)
            return static_path
        logger.warning("Static IR %s is older than %s; regenerating", static_path, ir_path)
        for ext in (".xml", ".bin"):
            try:
                os.remove(os.path.splitext(static_path)[0] + ext)
            except OSError:
                pass

    try:
        core = ov.Core()
        model = core.read_model(ir_path)
        shape = model.input(0).partial_shape
        if shape.is_static:
            return ir_path

        target = [1] + [d.get_length() for d in list(shape)[1:]]
        logger.info(
            "IR '%s' has a dynamic input shape %s; reshaping to %s so it can "
            "compile on NPU/GPU",
            getattr(model_cfg, "name", ir_path), shape, target,
        )
        model.reshape({model.input(0): ov.PartialShape(target)})
        ov.save_model(model, static_path, compress_to_fp16=True)
        logger.info("Wrote static IR: %s", static_path)
        return static_path
    except Exception:  # noqa: BLE001 - never block startup on this
        # A dynamic IR still runs on CPU, so degrade rather than crash.
        logger.exception(
            "Could not produce a static IR from %s; using it as-is. "
            "Inference on NPU/GPU may fail to compile.", ir_path,
        )
        return ir_path


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("QUEUE_SERVICE_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ensure_model()
