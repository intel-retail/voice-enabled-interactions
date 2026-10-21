"""
Tier 2 — Docker Build Validation Tests
========================================
Covers test case #5:
  Run `make build REGISTRY=false` and assert that ALL expected Docker images
  are present afterwards — including the EAL services (audio-analyzer,
  text-to-speech) that are pulled instead of built when edge-ai-libraries
  is not present locally.

metrics-collector is pulled from a *different* image name
(intel/hl-ai-metrics-collector) with its own independent tag
(METRICS_COLLECTOR_TAG), because its source now lives in the edge-ai-suites
repo shared with the multi-modal patient monitoring suite. It is not asserted
here alongside the RELEASE_TAG-based images (`make build` only best-effort
pulls it — see Makefile) — its health is instead validated functionally in
test_full_stack.py.

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
    "queue-service",
    "rtsp-streamer",
    "kiosk-ui",
]

EAL_PULLED_IMAGES = [
    "audio-analyzer",
    "text-to-speech",
]

# Declared in docker-compose.yml but pulled from a separate, independently
# versioned image (intel/hl-ai-metrics-collector:METRICS_COLLECTOR_TAG) whose
# source lives in the edge-ai-suites repo. Not part of the RELEASE_TAG image
# matrix asserted by test_local_service_image_exists /
# test_eal_service_image_exists — only checked for a compose declaration
# below; its actual health is validated in test_full_stack.py.
EXTERNALLY_SOURCED_IMAGES = [
    "metrics-collector",
]

ALL_EXPECTED_SERVICES = LOCAL_BUILT_IMAGES + EAL_PULLED_IMAGES + EXTERNALLY_SOURCED_IMAGES

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


# ---------------------------------------------------------------------------
# REGISTRY / REGISTRY_REPO substitution regression tests (ITEP-96940)
# ---------------------------------------------------------------------------
class TestRegistryRepoSubstitution:
    """`docker-compose.yml` image names must resolve from REGISTRY_REPO, not
    REGISTRY, and the Makefile must forward a REGISTRY_REPO override instead
    of silently dropping it.

    Regression coverage for ITEP-96940: `REGISTRY=true` (the Makefile
    build-mode flag) must never leak into `image:` as a literal
    "true/<service>" registry prefix, and a custom `REGISTRY_REPO` must
    survive both raw `docker compose` and `make` invocations.
    """

    # ovms-llm uses a fixed, externally-published image
    # (openvino/model_server) that is intentionally NOT prefixed by
    # REGISTRY_REPO — it is not one of this repo's own built/pulled images.
    # ovms-cache-init is a one-shot init container on the public `busybox`
    # image for the same reason, so it is exempt on the same grounds.
    _REGISTRY_REPO_EXEMPT_SERVICES = {"ovms-llm", "ovms-cache-init"}

    @staticmethod
    def _compose_config_images(env_overrides: dict[str, str]) -> dict[str, str]:
        """Run `docker compose config` with the given env and return {service: image}."""
        import yaml

        env = {**_os.environ, **env_overrides}
        result = subprocess.run(
            ["docker", "compose", "-f", "docker-compose.yml", "config", "--format", "json"],
            cwd=str(_KIOSK_ROOT),
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"docker compose config failed (exit {result.returncode}).\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )
        import json
        config = json.loads(result.stdout)
        services = config.get("services", {})
        return {svc: data.get("image", "") for svc, data in services.items()}

    @pytest.mark.tier2
    def test_registry_true_does_not_leak_into_image_name(self):
        """REGISTRY=true (Makefile build-mode flag) must not resolve into `image:`.

        This is the exact regression reported in ITEP-96940: a shipped
        `.env`/`.env.example` with REGISTRY=true previously caused every
        `image:` line to resolve to the literal, non-existent "true/<service>"
        registry when raw `docker compose pull/up` was run (bypassing `make`).
        """
        images = self._compose_config_images({"REGISTRY": "true", "REGISTRY_REPO": ""})
        assert images, "docker compose config returned no services"
        for service, image in images.items():
            if service in self._REGISTRY_REPO_EXEMPT_SERVICES:
                continue
            assert not image.startswith("true/"), (
                f"Service '{service}' resolved to '{image}' — REGISTRY=true leaked "
                f"into the image name instead of being ignored by docker-compose.yml"
            )
            assert image.startswith(f"{REGISTRY_PREFIX}/"), (
                f"Service '{service}' resolved to '{image}', expected the default "
                f"'{REGISTRY_PREFIX}/' prefix from REGISTRY_REPO's fallback"
            )

    @pytest.mark.tier2
    def test_registry_repo_override_is_honored_by_raw_compose(self):
        """A custom REGISTRY_REPO must be honored by raw `docker compose config`."""
        custom_repo = "myregistry.example.com/team"
        images = self._compose_config_images({"REGISTRY_REPO": custom_repo})
        assert images, "docker compose config returned no services"
        for service, image in images.items():
            if service in self._REGISTRY_REPO_EXEMPT_SERVICES:
                continue
            assert image.startswith(f"{custom_repo}/"), (
                f"Service '{service}' resolved to '{image}', expected the "
                f"'{custom_repo}/' prefix from the REGISTRY_REPO override"
            )

    @pytest.mark.tier2
    def test_make_forwards_registry_repo_override(self):
        """`make` must forward a custom REGISTRY_REPO override to docker compose.

        Regression coverage: the Makefile previously forwarded `REGISTRY=...`
        to docker compose for every build/up/pull invocation. Since
        docker-compose.yml stopped reading REGISTRY, that forwarding became a
        no-op and silently dropped any custom-registry override
        (e.g. `make build REGISTRY=myregistry.example.com/team`).
        """
        custom_repo = "myregistry.example.com/team"
        result = subprocess.run(
            ["make", "-n", "up", f"REGISTRY={custom_repo}"],
            cwd=str(_KIOSK_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"make -n up REGISTRY={custom_repo} failed (exit {result.returncode}).\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )
        assert f"REGISTRY_REPO={custom_repo}" in result.stdout, (
            f"Expected 'make up REGISTRY={custom_repo}' to forward "
            f"REGISTRY_REPO={custom_repo} to docker compose, but it was not found "
            f"in the dry-run output:\n{result.stdout}"
        )
