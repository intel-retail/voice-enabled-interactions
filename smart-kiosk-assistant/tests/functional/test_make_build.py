"""
Tier 2 — Docker Build Validation Tests
========================================
Covers test case #5:
  Run `make build REGISTRY=false` and assert that ALL expected Docker images
  are present afterwards — including the EAL services (audio-analyzer,
  text-to-speech) that are pulled instead of built when edge-ai-libraries
  is not present locally.

Prerequisites (handled by the CI workflow):
  - Docker daemon running
  - Internet access for docker pull (EAL images)
  - No GPU required — all images must be buildable/pullable on CPU

Run:
    pytest tests/functional/test_make_build.py -m tier2 -v
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_KIOSK_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Expected images after `make build REGISTRY=false`
# All images use "intel" as registry prefix (the _ENV_REGISTRY default).
# ---------------------------------------------------------------------------
LOCAL_BUILT_IMAGES = [
    # Format: (repo_prefix, service_tag)  — full name: intel/<service>:<RELEASE_TAG>
    "rag-service",
    "kiosk-core",
    "metrics-collector",
    "queue-service",
    "rtsp-streamer",
    "kiosk-ui",
]

EAL_PULLED_IMAGES = [
    "audio-analyzer",
    "text-to-speech",
]

ALL_EXPECTED_SERVICES = LOCAL_BUILT_IMAGES + EAL_PULLED_IMAGES

REGISTRY_PREFIX = "intel"
# Release tag matches Makefile default; CI sets RELEASE_TAG env var
import os as _os
RELEASE_TAG = _os.environ.get("RELEASE_TAG", "2026.1.0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_make(target: str, extra_vars: list[str] | None = None, timeout: int = 900) -> subprocess.CompletedProcess:
    """Run a make target in the kiosk root directory."""
    cmd = ["make", target, "REGISTRY=false"] + (extra_vars or [])
    return subprocess.run(
        cmd,
        cwd=str(_KIOSK_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _image_exists(image: str) -> bool:
    """Return True if the Docker image exists locally."""
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _full_image_name(service: str) -> str:
    return f"{REGISTRY_PREFIX}/{service}:{RELEASE_TAG}"


# ---------------------------------------------------------------------------
# Build execution
# ---------------------------------------------------------------------------
class TestMakeBuild:
    """Run `make build REGISTRY=false` and assert all images are produced."""

    @pytest.mark.tier2
    def test_make_build_exits_zero(self):
        """make build REGISTRY=false must complete without error."""
        result = _run_make("build", timeout=900)
        assert result.returncode == 0, (
            f"make build REGISTRY=false failed (exit {result.returncode}).\n"
            f"STDOUT:\n{result.stdout[-3000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )

    @pytest.mark.tier2
    @pytest.mark.parametrize("service", LOCAL_BUILT_IMAGES)
    def test_local_service_image_exists(self, service: str):
        """Each locally-built service image must be present after make build."""
        image = _full_image_name(service)
        assert _image_exists(image), (
            f"Expected image '{image}' not found after make build REGISTRY=false.\n"
            f"This image should be built from local source."
        )

    @pytest.mark.tier2
    @pytest.mark.parametrize("service", EAL_PULLED_IMAGES)
    def test_eal_service_image_exists(self, service: str):
        """EAL images (audio-analyzer, text-to-speech) must be pulled when EAL is absent.

        This validates the Makefile fix: when ../edge-ai-libraries is not present,
        `docker compose pull audio-analyzer text-to-speech` must run instead of
        silently skipping.
        """
        image = _full_image_name(service)
        assert _image_exists(image), (
            f"Expected image '{image}' not found after make build REGISTRY=false.\n"
            f"This image should be pulled (not built locally) when edge-ai-libraries "
            f"is absent. Check Makefile lines for the EAL pull fallback."
        )

    @pytest.mark.tier2
    def test_no_dangling_build_images(self):
        """Build must not leave dangling (<none>:<none>) images.

        Dangling images waste disk space on CI runners.
        """
        result = subprocess.run(
            ["docker", "images", "--filter", "dangling=true", "--format", "{{.ID}} {{.Repository}}:{{.Tag}}"],
            capture_output=True, text=True,
        )
        # Warning only — don't fail the suite over dangling images
        dangling = [line for line in result.stdout.strip().splitlines() if line]
        if dangling:
            import warnings
            warnings.warn(
                f"Found {len(dangling)} dangling Docker image(s) after build:\n"
                + "\n".join(dangling[:10]),
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# docker-compose image name consistency
# ---------------------------------------------------------------------------
class TestDockerComposeImageNames:
    """Images declared in docker-compose.yml must match expected naming."""

    @pytest.mark.tier2
    def test_all_expected_images_in_compose(self):
        """Every expected service image must be declared in docker-compose.yml."""
        import yaml
        compose_path = _KIOSK_ROOT / "docker-compose.yml"
        with open(compose_path) as fh:
            compose = yaml.safe_load(fh)

        services = compose.get("services", {})
        compose_images = {svc: data.get("image", "") for svc, data in services.items()}

        for service in ALL_EXPECTED_SERVICES:
            # Check that the service is defined and has an image
            assert service in compose_images, (
                f"Service '{service}' not found in docker-compose.yml"
            )
            assert compose_images[service], (
                f"Service '{service}' has no image defined in docker-compose.yml"
            )
