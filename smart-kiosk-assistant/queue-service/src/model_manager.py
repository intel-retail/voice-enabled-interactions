"""Model provisioning for queue-service.

Ensures the configured detector's OpenVINO IR exists under ``models/`` at the
path consumed by ``pipeline.py`` (``config.model.ir_path``). The default
detector is Intel Open Model Zoo ``person-detection-retail-0013``.

Provisioning is provider-based so the detector can be changed later (e.g. to
YOLO11) purely through configuration: ``pipeline.py`` only ever reads
``ir_path``/``proc_path``, while this module resolves *how* the IR is
obtained. The ``omz`` provider downloads the pre-converted IR directly from the
Intel Open Model Zoo storage server (the same artifact ``omz_downloader``
fetches), because the ``intel/dlstreamer:2026.1.0-ubuntu24`` image ships neither
the OMZ CLI tools nor ``openvino-dev``.
"""
import logging
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

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


def ensure_model(model_cfg=None) -> str:
    """Ensure the configured detector IR exists; return its path.

    Skips provisioning when the IR is already present. Otherwise dispatches
    to the configured provider (default ``omz``) to download/prepare it.
    """
    model_cfg = model_cfg if model_cfg is not None else getattr(config, "model", None)
    if model_cfg is None:
        raise ValueError("No 'model' section found in configuration")

    ir_path = _resolve(getattr(model_cfg, "ir_path"))
    if os.path.isfile(ir_path):
        logger.info("Model IR already present: %s", ir_path)
        return ir_path

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
    return ir_path


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("QUEUE_SERVICE_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ensure_model()
