"""
Tier 1 — Ordering API Functional Tests
======================================
Covers product catalogue, order lifecycle, upsell suggestions, and
idempotent catalogue seeding.

The ordering API is initialised in FastAPI lifespan, so these tests use
TestClient as a context manager with a per-test SQLite database.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def _clear_kiosk_modules() -> None:
    """Remove cached app modules so env-driven config is read per test."""
    for mod in list(sys.modules.keys()):
        if mod.startswith("kiosk_core") or mod == "main":
            del sys.modules[mod]


@pytest.fixture(scope="function")
def ordering_client_factory(
    tmp_path: Path,
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., AbstractContextManager[TestClient]]:
    """Return a TestClient factory backed by an isolated ordering database."""

    @contextmanager
    def _factory(db_name: str = "ordering.sqlite3") -> Iterator[TestClient]:
        db_path = tmp_path / db_name
        products_yaml = repo_root / "configs" / "ordering" / "products.yaml"
        upsell_yaml = repo_root / "configs" / "ordering" / "upsell_rules.yaml"

        monkeypatch.setenv("KIOSK_CORE_ORDERING_ENABLED", "true")
        monkeypatch.setenv("KIOSK_CORE_DB_PATH", str(db_path))
        monkeypatch.setenv("KIOSK_CORE_PRODUCTS_YAML", str(products_yaml))
        monkeypatch.setenv("KIOSK_CORE_UPSELL_RULES_YAML", str(upsell_yaml))

        fake_device = {
            "name": "Mock Microphone",
            "max_input_channels": 2,
            "default_samplerate": 16000,
        }
        sd_mock = MagicMock()
        sd_mock.query_devices.return_value = [fake_device]

        with patch.dict("sys.modules", {"sounddevice": sd_mock}):
            from fastapi.testclient import TestClient  # noqa: PLC0415

            _clear_kiosk_modules()
            import main  # noqa: PLC0415 — intentional deferred import

            try:
                with TestClient(main.app) as client:
                    yield client
            finally:
                _clear_kiosk_modules()

    return _factory


@pytest.fixture(scope="function")
def ordering_app(
    ordering_client_factory: Callable[..., AbstractContextManager[TestClient]],
) -> Iterator[TestClient]:
    """FastAPI TestClient with ordering startup completed."""
    with ordering_client_factory() as client:
        yield client


def _item_by_product(order: dict[str, Any], product_id: str) -> dict[str, Any]:
    """Return the order item matching a product id."""
    return next(item for item in order["items"] if item["product_id"] == product_id)


@pytest.mark.tier1
class TestOrderingProducts:
    """Product catalogue endpoint coverage."""

    @pytest.mark.tier1
    def test_products_returns_non_empty_list_with_expected_schema(self, ordering_app: TestClient):
        response = ordering_app.get("/api/v1/products")
        assert response.status_code == 200

        products = response.json()
        assert isinstance(products, list)
        assert products, "Seeded product catalogue must not be empty"

        first = products[0]
        assert {"product_id", "name", "category", "price"} <= set(first)
        assert isinstance(first["product_id"], str)
        assert isinstance(first["name"], str)
        assert isinstance(first["category"], str)
        assert isinstance(first["price"], (int, float))

    @pytest.mark.tier1
    def test_products_category_filter_returns_only_matching_category(self, ordering_app: TestClient):
        response = ordering_app.get("/api/v1/products", params={"category": "burgers"})
        assert response.status_code == 200

        products = response.json()
        assert products, "Expected seeded burger products"
        assert all(product["category"] == "burgers" for product in products)
        assert any(product["product_id"] == "BURGER-VEG-001" for product in products)

    @pytest.mark.tier1
    def test_get_product_returns_known_product_and_unknown_product_404(
        self,
        ordering_app: TestClient,
    ):
        response = ordering_app.get("/api/v1/products/BURGER-VEG-001")
        assert response.status_code == 200
        assert response.json() == {
            "product_id": "BURGER-VEG-001",
            "name": "Crispy Veg Patty Burger",
            "category": "burgers",
            "price": 129.0,
        }

        missing = ordering_app.get("/api/v1/products/DOES-NOT-EXIST")
        assert missing.status_code == 404


@pytest.mark.tier1
class TestOrderingLifecycle:
    """Order creation, mutation, lookup, and confirmation coverage."""

    @pytest.mark.tier1
    def test_full_order_lifecycle_computes_totals_and_confirms(self, ordering_app: TestClient):
        user_id = "tier1-user"

        created = ordering_app.post(
            "/api/v1/orders",
            json={
                "user_id": user_id,
                "items": [{"product_id": "BURGER-VEG-001", "quantity": 2}],
            },
        )
        assert created.status_code == 201, created.text
        order = created.json()
        assert order["order_id"] == 1
        assert order["user_id"] == user_id
        assert order["status"] == "draft"
        assert order["total"] == 258.0
        assert _item_by_product(order, "BURGER-VEG-001")["subtotal"] == 258.0

        fetched = ordering_app.get(f"/api/v1/orders/{order['order_id']}")
        assert fetched.status_code == 200
        assert fetched.json()["total"] == 258.0

        updated = ordering_app.patch(
            f"/api/v1/orders/{order['order_id']}/items",
            json={
                "items": [
                    {"product_id": "BURGER-VEG-001", "quantity": 1},
                    {"product_id": "SIDE-001", "quantity": 2},
                ],
            },
        )
        assert updated.status_code == 200, updated.text
        updated_order = updated.json()
        assert updated_order["total"] == 565.0
        assert _item_by_product(updated_order, "BURGER-VEG-001")["quantity"] == 3
        assert _item_by_product(updated_order, "SIDE-001")["subtotal"] == 178.0

        current = ordering_app.get(f"/api/v1/users/{user_id}/orders/current")
        assert current.status_code == 200
        assert current.json()["order_id"] == order["order_id"]

        confirmed = ordering_app.post(f"/api/v1/orders/{order['order_id']}/confirm")
        assert confirmed.status_code == 200, confirmed.text
        assert confirmed.json()["status"] == "confirmed"
        assert confirmed.json()["total"] == 565.0

    @pytest.mark.tier1
    def test_current_draft_returns_404_for_user_without_orders(self, ordering_app: TestClient):
        response = ordering_app.get("/api/v1/users/no-orders-user/orders/current")
        assert response.status_code == 404


@pytest.mark.tier1
class TestOrderingUpsell:
    """Upsell suggestion endpoint coverage."""

    @pytest.mark.tier1
    def test_upsell_returns_configured_suggestions_for_burger_cart(
        self,
        ordering_app: TestClient,
    ):
        response = ordering_app.post(
            "/api/v1/upsell",
            json={"product_ids": ["BURGER-VEG-001"]},
        )
        assert response.status_code == 200

        suggestions = response.json()
        assert suggestions, "Burger cart should trigger configured upsell rules"

        suggested_ids = {suggestion["product"]["product_id"] for suggestion in suggestions}
        assert {"DRINK-001", "SIDE-001", "DESSERT-001"} <= suggested_ids
        assert all("reason" in suggestion for suggestion in suggestions)

    @pytest.mark.tier1
    def test_upsell_returns_empty_list_for_empty_or_irrelevant_cart(
        self,
        ordering_app: TestClient,
    ):
        empty = ordering_app.post("/api/v1/upsell", json={"product_ids": []})
        assert empty.status_code == 200
        assert empty.json() == []

        irrelevant = ordering_app.post(
            "/api/v1/upsell",
            json={"product_ids": ["UNKNOWN-PRODUCT"]},
        )
        assert irrelevant.status_code == 200
        assert irrelevant.json() == []


@pytest.mark.tier1
class TestOrderingSeed:
    """Catalogue seed idempotency coverage."""

    @pytest.mark.tier1
    def test_app_startup_seed_is_idempotent_for_same_database(
        self,
        ordering_client_factory: Callable[..., AbstractContextManager[TestClient]],
    ):
        with ordering_client_factory("idempotent.sqlite3") as client:
            first = client.get("/api/v1/products")
            assert first.status_code == 200
            first_count = len(first.json())

        with ordering_client_factory("idempotent.sqlite3") as client:
            second = client.get("/api/v1/products")
            assert second.status_code == 200
            second_count = len(second.json())

        assert first_count > 0
        assert second_count == first_count
