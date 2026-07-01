"""
Tier 1 — YAML Configuration Schema Tests
=========================================
Covers test case #4:
  Validate configs/ordering/products.yaml, configs/ordering/upsell_rules.yaml,
  queue-service/conf/queue-config.yaml, and rag-service/config.yaml against
  expected schemas using Pydantic v2.

These tests are CI-safe: no Docker, no ML models, no audio hardware.

Run:
    pytest tests/functional/test_config_schemas.py -m tier1 -v
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
import yaml
from pydantic import BaseModel, field_validator, model_validator, ValidationError

_KIOSK_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS = _KIOSK_ROOT / "configs"
_PRODUCTS_YAML = _CONFIGS / "ordering" / "products.yaml"
_UPSELL_YAML = _CONFIGS / "ordering" / "upsell_rules.yaml"
_QUEUE_CONFIG = _KIOSK_ROOT / "queue-service" / "conf" / "queue-config.yaml"
_RAG_CONFIG = _KIOSK_ROOT / "rag-service" / "config.yaml"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

VALID_CATEGORIES = {
    "burgers", "wraps", "pizza", "beverages", "sides", "desserts", "breakfast"
}


class ProductSchema(BaseModel):
    product_id: str
    name: str
    category: str
    price: float

    @field_validator("product_id")
    @classmethod
    def product_id_not_empty(cls, v: str) -> str:
        assert v.strip(), "product_id must not be empty"
        return v

    @field_validator("price")
    @classmethod
    def price_positive(cls, v: float) -> float:
        assert v > 0, f"price must be positive, got {v}"
        return v

    @field_validator("category")
    @classmethod
    def category_valid(cls, v: str) -> str:
        assert v in VALID_CATEGORIES, (
            f"category '{v}' not in {VALID_CATEGORIES}"
        )
        return v


class ProductsFileSchema(BaseModel):
    products: list[ProductSchema]

    @field_validator("products")
    @classmethod
    def at_least_one_product(cls, v: list) -> list:
        assert len(v) > 0, "products list must not be empty"
        return v


class UpsellRuleSchema(BaseModel):
    trigger_product_ids: list[str] | None = None
    trigger_categories: list[str] | None = None
    suggest_product_ids: list[str]
    reason: str

    @model_validator(mode="after")
    def has_trigger(self) -> "UpsellRuleSchema":
        has_prod = bool(self.trigger_product_ids)
        has_cat = bool(self.trigger_categories)
        assert has_prod or has_cat, (
            "Each upsell rule must have at least one of trigger_product_ids or "
            "trigger_categories"
        )
        return self

    @field_validator("suggest_product_ids")
    @classmethod
    def suggest_not_empty(cls, v: list) -> list:
        assert len(v) > 0, "suggest_product_ids must not be empty"
        return v

    @field_validator("reason")
    @classmethod
    def reason_not_empty(cls, v: str) -> str:
        assert v.strip(), "reason must not be empty"
        return v


class UpsellFileSchema(BaseModel):
    rules: list[UpsellRuleSchema]

    @field_validator("rules")
    @classmethod
    def at_least_one_rule(cls, v: list) -> list:
        assert len(v) > 0, "rules list must not be empty"
        return v


# Simplified schemas — only validate key fields, not exhaustive
class QueueSourceSchema(BaseModel):
    rtsp_url: str
    rtsp_transport: Literal["tcp", "udp"] = "tcp"

    @field_validator("rtsp_url")
    @classmethod
    def rtsp_url_format(cls, v: str) -> str:
        assert v.startswith("rtsp://"), f"rtsp_url must start with rtsp://, got '{v}'"
        return v


class QueueModelSchema(BaseModel):
    name: str
    provider: Literal["local", "omz"]
    ir_path: str
    device: str
    threshold: float

    @field_validator("threshold")
    @classmethod
    def threshold_range(cls, v: float) -> float:
        assert 0.0 < v <= 1.0, f"threshold must be in (0, 1], got {v}"
        return v

    @field_validator("device")
    @classmethod
    def device_upper(cls, v: str) -> str:
        assert v in ("CPU", "GPU", "NPU"), f"device must be CPU|GPU|NPU, got '{v}'"
        return v


class QueueApiSchema(BaseModel):
    enabled: bool
    port: int
    host: str

    @field_validator("port")
    @classmethod
    def valid_port(cls, v: int) -> int:
        assert 1 <= v <= 65535, f"port {v} out of range"
        return v


class QueueConfigSchema(BaseModel):
    source: QueueSourceSchema
    model: QueueModelSchema
    api: QueueApiSchema


class RagLlmSchema(BaseModel):
    backend: Literal["ovms", "openvino"]
    hf_id: str
    device: str
    weight_format: Literal["int8", "fp16", "fp32"]


class RagEmbeddingSchema(BaseModel):
    hf_id: str
    device: str
    backend: Literal["openvino"]
    weight_format: Literal["int8", "fp16", "fp32"]


class RagModelsSchema(BaseModel):
    llm: RagLlmSchema
    embedding: RagEmbeddingSchema


class RagRetrievalSchema(BaseModel):
    top_k: int
    fetch_k: int

    @field_validator("top_k")
    @classmethod
    def top_k_positive(cls, v: int) -> int:
        assert v > 0, f"top_k must be positive, got {v}"
        return v

    @field_validator("fetch_k")
    @classmethod
    def fetch_k_gte_top_k(cls, v: int) -> int:
        assert v > 0, f"fetch_k must be positive, got {v}"
        return v


class RagConfigSchema(BaseModel):
    models: RagModelsSchema
    retrieval: RagRetrievalSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# products.yaml
# ---------------------------------------------------------------------------
class TestProductsYaml:
    """Validate configs/ordering/products.yaml."""

    @pytest.mark.tier1
    def test_products_yaml_exists(self):
        assert _PRODUCTS_YAML.is_file(), f"products.yaml not found at {_PRODUCTS_YAML}"

    @pytest.mark.tier1
    def test_products_yaml_valid_schema(self):
        """All products must pass schema validation."""
        data = _load_yaml(_PRODUCTS_YAML)
        try:
            schema = ProductsFileSchema.model_validate(data)
        except ValidationError as exc:
            pytest.fail(f"products.yaml schema validation failed:\n{exc}")

    @pytest.mark.tier1
    def test_products_have_unique_ids(self):
        """Each product_id must be unique."""
        data = _load_yaml(_PRODUCTS_YAML)
        ids = [p["product_id"] for p in data["products"]]
        duplicates = [pid for pid in ids if ids.count(pid) > 1]
        assert not duplicates, f"Duplicate product_ids: {set(duplicates)}"

    @pytest.mark.tier1
    def test_products_cover_required_categories(self):
        """Catalogue must include at least burgers, beverages, and sides."""
        data = _load_yaml(_PRODUCTS_YAML)
        categories = {p["category"] for p in data["products"]}
        required = {"burgers", "beverages", "sides"}
        missing = required - categories
        assert not missing, f"Missing required categories: {missing}"

    @pytest.mark.tier1
    def test_products_minimum_count(self):
        """Catalogue must have at least 20 products (regression guard)."""
        data = _load_yaml(_PRODUCTS_YAML)
        count = len(data["products"])
        assert count >= 20, f"Expected ≥20 products, got {count}"


# ---------------------------------------------------------------------------
# upsell_rules.yaml
# ---------------------------------------------------------------------------
class TestUpsellRulesYaml:
    """Validate configs/ordering/upsell_rules.yaml."""

    @pytest.mark.tier1
    def test_upsell_yaml_exists(self):
        assert _UPSELL_YAML.is_file(), f"upsell_rules.yaml not found at {_UPSELL_YAML}"

    @pytest.mark.tier1
    def test_upsell_yaml_valid_schema(self):
        """All rules must pass schema validation."""
        data = _load_yaml(_UPSELL_YAML)
        try:
            UpsellFileSchema.model_validate(data)
        except ValidationError as exc:
            pytest.fail(f"upsell_rules.yaml schema validation failed:\n{exc}")

    @pytest.mark.tier1
    def test_upsell_suggest_ids_exist_in_products(self):
        """Every suggest_product_id must exist in products.yaml."""
        products_data = _load_yaml(_PRODUCTS_YAML)
        known_ids = {p["product_id"] for p in products_data["products"]}

        upsell_data = _load_yaml(_UPSELL_YAML)
        for i, rule in enumerate(upsell_data["rules"]):
            for pid in rule.get("suggest_product_ids", []):
                assert pid in known_ids, (
                    f"Rule #{i}: suggest_product_id '{pid}' not found in products.yaml"
                )

    @pytest.mark.tier1
    def test_upsell_trigger_ids_exist_in_products(self):
        """Every trigger_product_id (when set) must exist in products.yaml."""
        products_data = _load_yaml(_PRODUCTS_YAML)
        known_ids = {p["product_id"] for p in products_data["products"]}

        upsell_data = _load_yaml(_UPSELL_YAML)
        for i, rule in enumerate(upsell_data["rules"]):
            for pid in rule.get("trigger_product_ids", []) or []:
                assert pid in known_ids, (
                    f"Rule #{i}: trigger_product_id '{pid}' not found in products.yaml"
                )

    @pytest.mark.tier1
    def test_upsell_trigger_categories_valid(self):
        """Every trigger_category must be a known catalogue category."""
        upsell_data = _load_yaml(_UPSELL_YAML)
        for i, rule in enumerate(upsell_data["rules"]):
            for cat in rule.get("trigger_categories", []) or []:
                assert cat in VALID_CATEGORIES, (
                    f"Rule #{i}: trigger_category '{cat}' not in {VALID_CATEGORIES}"
                )


# ---------------------------------------------------------------------------
# queue-config.yaml
# ---------------------------------------------------------------------------
class TestQueueConfigYaml:
    """Validate queue-service/conf/queue-config.yaml."""

    @pytest.mark.tier1
    def test_queue_config_exists(self):
        assert _QUEUE_CONFIG.is_file(), f"queue-config.yaml not found at {_QUEUE_CONFIG}"

    @pytest.mark.tier1
    def test_queue_config_valid_schema(self):
        """Key sections of queue-config.yaml must pass schema validation."""
        data = _load_yaml(_QUEUE_CONFIG)
        try:
            QueueConfigSchema.model_validate(data)
        except ValidationError as exc:
            pytest.fail(f"queue-config.yaml schema validation failed:\n{exc}")

    @pytest.mark.tier1
    def test_queue_rtsp_url_references_correct_stream(self):
        """RTSP URL must reference the correct istockphoto stream name."""
        data = _load_yaml(_QUEUE_CONFIG)
        rtsp_url = data["source"]["rtsp_url"]
        assert "istockphoto-2248308153-640_adpp_is" in rtsp_url, (
            f"queue-config.yaml rtsp_url '{rtsp_url}' does not reference "
            f"the expected stream 'istockphoto-2248308153-640_adpp_is'. "
            f"This was a known bug — the fix must not be reverted."
        )

    @pytest.mark.tier1
    def test_queue_api_enabled_by_default(self):
        """API endpoint must be enabled (required for MJPEG stream)."""
        data = _load_yaml(_QUEUE_CONFIG)
        assert data["api"]["enabled"] is True, (
            "queue-config.yaml api.enabled must be true — the MJPEG stream depends on it"
        )


# ---------------------------------------------------------------------------
# rag-service/config.yaml
# ---------------------------------------------------------------------------
class TestRagConfigYaml:
    """Validate rag-service/config.yaml."""

    @pytest.mark.tier1
    def test_rag_config_exists(self):
        assert _RAG_CONFIG.is_file(), f"rag-service/config.yaml not found at {_RAG_CONFIG}"

    @pytest.mark.tier1
    def test_rag_config_valid_schema(self):
        """Key sections of rag-service/config.yaml must pass schema validation."""
        data = _load_yaml(_RAG_CONFIG)
        try:
            RagConfigSchema.model_validate(data)
        except ValidationError as exc:
            pytest.fail(f"rag-service/config.yaml schema validation failed:\n{exc}")

    @pytest.mark.tier1
    def test_rag_llm_uses_qwen_model(self):
        """LLM hf_id must reference a Qwen model (the project standard)."""
        data = _load_yaml(_RAG_CONFIG)
        hf_id = data["models"]["llm"]["hf_id"]
        assert "Qwen" in hf_id or "qwen" in hf_id.lower(), (
            f"Expected a Qwen model in models.llm.hf_id, got '{hf_id}'"
        )

    @pytest.mark.tier1
    def test_rag_fetch_k_gte_top_k(self):
        """fetch_k must be >= top_k for reranking to work correctly."""
        data = _load_yaml(_RAG_CONFIG)
        top_k = data["retrieval"]["top_k"]
        fetch_k = data["retrieval"]["fetch_k"]
        assert fetch_k >= top_k, (
            f"retrieval.fetch_k ({fetch_k}) must be >= top_k ({top_k}) — "
            f"reranker needs fetch_k candidates to select top_k from"
        )

    @pytest.mark.tier1
    def test_rag_devices_are_uppercase(self):
        """OpenVINO device strings must be uppercase (CPU|GPU|NPU)."""
        data = _load_yaml(_RAG_CONFIG)
        devices = {
            "llm": data["models"]["llm"]["device"],
            "embedding": data["models"]["embedding"]["device"],
        }
        for name, device in devices.items():
            assert device == device.upper(), (
                f"models.{name}.device '{device}' must be uppercase — "
                f"OpenVINO rejects lowercase device strings"
            )
