"""
Tier 1 — Repository & Project Structure Tests
==============================================
Covers:
  NEX-T24243  Verify repository clone
  NEX-T24244  Verify project structure

These tests assert that all expected files, directories, and configuration
artefacts are present in the checked-out repository.  They require no
running services, no Docker daemon, and no ML models.

Run:
    pytest tests/functional/test_structure.py -m tier1 -v
"""
from pathlib import Path

import pytest

# Resolve to smart-kiosk-assistant/ (two levels up from tests/functional/)
_KIOSK_ROOT = Path(__file__).resolve().parents[2]


class TestRepositoryClone:
    """NEX-T24243 — Verify repository clone."""

    @pytest.mark.tier1
    def test_readme_present(self):
        """README.md must exist at the project root."""
        assert (_KIOSK_ROOT / "README.md").is_file(), "README.md not found"

    @pytest.mark.tier1
    def test_dockerfile_present(self):
        """Dockerfile for kiosk-core must be present."""
        assert (_KIOSK_ROOT / "Dockerfile").is_file(), "Dockerfile not found"

    @pytest.mark.tier1
    def test_docker_compose_present(self):
        """docker-compose.yml must be present."""
        assert (_KIOSK_ROOT / "docker-compose.yml").is_file(), "docker-compose.yml not found"

    @pytest.mark.tier1
    def test_requirements_present(self):
        """requirements.txt must be present and non-empty."""
        req = _KIOSK_ROOT / "requirements.txt"
        assert req.is_file(), "requirements.txt not found"
        assert req.stat().st_size > 0, "requirements.txt is empty"

    @pytest.mark.tier1
    def test_main_entrypoint_present(self):
        """main.py (kiosk-core FastAPI entrypoint) must exist."""
        assert (_KIOSK_ROOT / "main.py").is_file(), "main.py not found"


class TestProjectStructure:
    """NEX-T24244 — Verify project structure."""

    @pytest.mark.tier1
    def test_kiosk_core_package_present(self):
        """kiosk_core/ Python package must exist."""
        assert (_KIOSK_ROOT / "kiosk_core").is_dir(), "kiosk_core/ directory not found"

    @pytest.mark.tier1
    def test_docs_directory_present(self):
        """docs/ directory must exist."""
        assert (_KIOSK_ROOT / "docs").is_dir(), "docs/ directory not found"

    @pytest.mark.tier1
    def test_rag_service_present(self):
        """rag-service/ directory must exist."""
        assert (_KIOSK_ROOT / "rag-service").is_dir(), "rag-service/ directory not found"

    @pytest.mark.tier1
    def test_configs_directory_present(self):
        """configs/ directory with per-service YAML overrides must exist."""
        assert (_KIOSK_ROOT / "configs").is_dir(), "configs/ directory not found"

    @pytest.mark.tier1
    def test_metrics_collector_present(self):
        """metrics-collector/ directory must exist (lives under kiosk_core/)."""
        assert (_KIOSK_ROOT / "kiosk_core" / "metrics-collector").is_dir(), (
            "kiosk_core/metrics-collector/ directory not found"
        )

    @pytest.mark.tier1
    def test_kiosk_core_service_module_present(self):
        """kiosk_core/service.py — core session orchestration — must exist."""
        assert (_KIOSK_ROOT / "kiosk_core" / "service.py").is_file(), (
            "kiosk_core/service.py not found"
        )

    @pytest.mark.tier1
    def test_kiosk_core_models_module_present(self):
        """kiosk_core/models.py — Pydantic request/response models — must exist."""
        assert (_KIOSK_ROOT / "kiosk_core" / "models.py").is_file(), (
            "kiosk_core/models.py not found"
        )

    @pytest.mark.tier1
    def test_kiosk_core_config_module_present(self):
        """kiosk_core/config.py — environment-driven configuration — must exist."""
        assert (_KIOSK_ROOT / "kiosk_core" / "config.py").is_file(), (
            "kiosk_core/config.py not found"
        )

    @pytest.mark.tier1
    def test_docker_compose_references_kiosk_core_service(self):
        """docker-compose.yml must declare a kiosk-core service."""
        import yaml  # pyyaml is in requirements.txt

        with open(_KIOSK_ROOT / "docker-compose.yml") as fh:
            compose = yaml.safe_load(fh)

        services = compose.get("services", {})
        assert "kiosk-core" in services, (
            f"kiosk-core service not found in docker-compose.yml; found: {list(services)}"
        )

    @pytest.mark.tier1
    def test_docker_compose_kiosk_core_exposes_port_8012(self):
        """kiosk-core service must expose port 8012 (the documented API port)."""
        import yaml

        with open(_KIOSK_ROOT / "docker-compose.yml") as fh:
            compose = yaml.safe_load(fh)

        ports = compose["services"]["kiosk-core"].get("ports", [])
        port_strings = [str(p) for p in ports]
        assert any("8012" in p for p in port_strings), (
            f"Port 8012 not exposed by kiosk-core in docker-compose.yml; ports: {port_strings}"
        )
