"""
Tier 2 — Docker Container Tests
=================================
Covers:
  NEX-T24245  Verify container build
  NEX-T24246  Verify container startup
  NEX-T24247  Verify container status
  NEX-T24248  Verify application logs
  NEX-T24261  Verify container restart recovery

These tests build and run the kiosk-core Docker image directly (without the
full docker-compose stack that requires ML model downloads).  A live Docker
daemon is required; tests are automatically skipped when Docker is unavailable.

IMPORTANT: Only the kiosk-core image is built/run in CI.  The full
           docker-compose stack (audio-analyzer, TTS, RAG) requires multi-GB
           ML model downloads and is deferred to Tier 3 / self-hosted runners.

Run:
    pytest tests/functional/test_docker.py -m tier2 -v -s
"""
import shutil
import subprocess
import time
from pathlib import Path

import pytest

# Resolve to smart-kiosk-assistant/ (two levels up from tests/functional/)
_KIOSK_ROOT = Path(__file__).resolve().parents[2]
_IMAGE_TAG = "kiosk-core-ci-test:latest"
_CONTAINER_NAME = "kiosk-core-ci-test"
_HOST_PORT = 18012  # use a non-standard port to avoid conflicts with local dev
_HEALTH_URL = f"http://127.0.0.1:{_HOST_PORT}/health"
_BUILD_TIMEOUT_SEC = 600  # 10 min for pip install inside Docker
_STARTUP_TIMEOUT_SEC = 30  # kiosk-core starts in <5s without ML models


# ---------------------------------------------------------------------------
# Docker availability helper
# ---------------------------------------------------------------------------
def _docker_available() -> tuple[bool, str]:
    if shutil.which("docker") is None:
        return False, "docker binary not found on PATH"
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return False, "Docker daemon timed out"
    except FileNotFoundError:
        return False, "docker binary not found"
    if result.returncode != 0:
        return False, f"Docker daemon not reachable: {result.stderr.strip()[:200]}"
    return True, ""


def _skip_if_no_docker():
    ok, reason = _docker_available()
    if not ok:
        pytest.skip(f"Docker not available — {reason}")


# ---------------------------------------------------------------------------
# Session-scoped fixture: build image once, clean up after all Tier 2 tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def docker_image():
    """Build the kiosk-core Docker image once for all Tier 2 tests."""
    _skip_if_no_docker()

    result = subprocess.run(
        ["docker", "build", "-t", _IMAGE_TAG, "-f", "Dockerfile", "."],
        cwd=str(_KIOSK_ROOT),
        capture_output=True,
        text=True,
        timeout=_BUILD_TIMEOUT_SEC,
    )
    if result.returncode != 0:
        pytest.fail(
            f"docker build failed (exit {result.returncode}).\n"
            f"STDOUT:\n{result.stdout[-3000:]}\nSTDERR:\n{result.stderr[-3000:]}"
        )
    yield _IMAGE_TAG

    # Cleanup: remove the test image after module tests complete
    subprocess.run(
        ["docker", "rmi", "-f", _IMAGE_TAG],
        capture_output=True,
        timeout=30,
    )


@pytest.fixture(scope="function")
def running_container(docker_image):
    """Start a kiosk-core container and stop it after each test."""
    # Remove any leftover container with the same name
    subprocess.run(
        ["docker", "rm", "-f", _CONTAINER_NAME],
        capture_output=True,
        timeout=10,
    )

    result = subprocess.run(
        [
            "docker", "run", "-d",
            "--name", _CONTAINER_NAME,
            "-p", f"{_HOST_PORT}:8012",
            # Provide dummy downstream URLs so kiosk-core starts without real services
            "-e", "KIOSK_CORE_ANALYZER_URL=http://127.0.0.1:9999/v1/audio/transcriptions",
            "-e", "KIOSK_CORE_RAG_URL=http://127.0.0.1:9999/api/v1/query",
            "-e", "KIOSK_CORE_TTS_URL=http://127.0.0.1:9999/v1/audio/speech",
            docker_image,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        pytest.fail(f"docker run failed: {result.stderr.strip()}")

    container_id = result.stdout.strip()

    # Wait for the service to become healthy
    _wait_for_health(_HOST_PORT, timeout=_STARTUP_TIMEOUT_SEC)

    yield container_id

    # Teardown: always stop and remove the container
    subprocess.run(["docker", "rm", "-f", _CONTAINER_NAME], capture_output=True, timeout=15)


def _wait_for_health(port: int, timeout: int = 30) -> bool:
    """Poll GET /health until 200 or timeout. Returns True on success."""
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=3
            ) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


# ---------------------------------------------------------------------------
# NEX-T24245 — Verify container build
# ---------------------------------------------------------------------------
class TestContainerBuild:
    """NEX-T24245 — All images should build successfully without failures."""

    @pytest.mark.tier2
    def test_docker_daemon_reachable(self):
        """Prerequisite: Docker daemon must be running before any build test."""
        _skip_if_no_docker()

    @pytest.mark.tier2
    def test_kiosk_core_image_builds_successfully(self, docker_image):
        """
        Build kiosk-core image from Dockerfile.
        Validates: Dockerfile syntax, COPY sources exist, pip install succeeds.
        """
        result = subprocess.run(
            ["docker", "image", "inspect", docker_image],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, (
            f"Built image '{docker_image}' not found in local Docker store"
        )

    @pytest.mark.tier2
    def test_built_image_has_correct_label(self, docker_image):
        """Inspect image to confirm it was tagged correctly."""
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", docker_image],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert _IMAGE_TAG in result.stdout, (
            f"Expected image tag '{_IMAGE_TAG}' not in: {result.stdout.strip()}"
        )


# ---------------------------------------------------------------------------
# NEX-T24246 + NEX-T24247 — Verify container startup and status
# ---------------------------------------------------------------------------
class TestContainerStartup:
    """
    NEX-T24246 — All containers should start and remain in running state.
    NEX-T24247 — Smart Kiosk containers should show "Up" status.
    """

    @pytest.mark.tier2
    def test_container_starts_successfully(self, running_container):
        """docker run must exit 0 and return a container ID."""
        assert running_container, "Container ID must be non-empty"

    @pytest.mark.tier2
    def test_container_is_running(self, running_container):
        """docker inspect State.Running must be true."""
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", _CONTAINER_NAME],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "true", (
            f"Container {_CONTAINER_NAME} is not running: {result.stdout.strip()}"
        )

    @pytest.mark.tier2
    def test_container_shows_up_in_docker_ps(self, running_container):
        """docker ps should list the kiosk-core container."""
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={_CONTAINER_NAME}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert _CONTAINER_NAME in result.stdout, (
            f"Container '{_CONTAINER_NAME}' not found in docker ps output: {result.stdout}"
        )

    @pytest.mark.tier2
    def test_health_endpoint_responds_200_from_container(self, running_container):
        """
        NEX-T24260 (container variant) — GET /health on the running container
        must return HTTP 200.
        """
        import urllib.request

        with urllib.request.urlopen(_HEALTH_URL, timeout=10) as resp:
            assert resp.status == 200, f"Expected 200 from {_HEALTH_URL}, got {resp.status}"


# ---------------------------------------------------------------------------
# NEX-T24248 — Verify application logs
# ---------------------------------------------------------------------------
class TestApplicationLogs:
    """NEX-T24248 — Services should initialize without fatal errors."""

    @pytest.mark.tier2
    def test_logs_contain_uvicorn_startup(self, running_container):
        """docker logs must confirm uvicorn started on the expected port."""
        result = subprocess.run(
            ["docker", "logs", _CONTAINER_NAME],
            capture_output=True,
            text=True,
            timeout=15,
        )
        combined = result.stdout + result.stderr
        assert "Uvicorn running" in combined or "Application startup complete" in combined, (
            f"Uvicorn startup message not found in logs.\nLogs:\n{combined[-2000:]}"
        )

    @pytest.mark.tier2
    def test_logs_contain_no_fatal_errors(self, running_container):
        """
        docker logs must not contain FATAL or Traceback at startup.
        (Warnings about missing downstream services are expected and acceptable.)
        """
        result = subprocess.run(
            ["docker", "logs", _CONTAINER_NAME],
            capture_output=True,
            text=True,
            timeout=15,
        )
        combined = (result.stdout + result.stderr).lower()
        fatal_indicators = ["fatal error", "traceback (most recent call last)"]
        found = [ind for ind in fatal_indicators if ind in combined]
        assert not found, (
            f"Fatal startup indicators found in logs: {found}\n"
            f"Relevant log tail:\n{(result.stdout + result.stderr)[-2000:]}"
        )


# ---------------------------------------------------------------------------
# NEX-T24261 — Verify container restart recovery
# ---------------------------------------------------------------------------
class TestContainerRestartRecovery:
    """NEX-T24261 — Service should recover automatically after docker restart."""

    @pytest.mark.tier2
    def test_container_recovers_after_restart(self, running_container):
        """
        docker restart → poll GET /health → must return 200 within
        STARTUP_TIMEOUT_SEC seconds.
        """
        result = subprocess.run(
            ["docker", "restart", _CONTAINER_NAME],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"docker restart failed: {result.stderr.strip()}"
        )

        recovered = _wait_for_health(_HOST_PORT, timeout=_STARTUP_TIMEOUT_SEC)
        assert recovered, (
            f"kiosk-core did not recover within {_STARTUP_TIMEOUT_SEC}s after restart"
        )

    @pytest.mark.tier2
    def test_container_still_running_after_restart(self, running_container):
        """After restart, container State.Running must still be true."""
        subprocess.run(["docker", "restart", _CONTAINER_NAME], capture_output=True, timeout=30)
        _wait_for_health(_HOST_PORT, timeout=_STARTUP_TIMEOUT_SEC)

        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", _CONTAINER_NAME],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.stdout.strip() == "true", (
            "Container not running after restart"
        )
