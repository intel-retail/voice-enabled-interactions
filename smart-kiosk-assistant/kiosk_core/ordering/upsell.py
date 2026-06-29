"""Rule-based upsell suggestion engine.

Upsell rules are loaded once from ``configs/ordering/upsell_rules.yaml`` at
startup (via :func:`load_upsell_rules`).  The ``UpsellEngine`` is stateless
and thread-safe after initialisation.

Rule YAML format::

    rules:
      - trigger_product_ids: [BURGER-001, BURGER-002]
        suggest_product_ids: [DRINK-001, FRIES-001]
        reason: "Goes great with a burger!"
      - trigger_categories: [pizza]
        suggest_product_ids: [DRINK-001]
        reason: "Customers often pair a drink with pizza."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import yaml

from kiosk_core.ordering.models import Product, UpsellSuggestion

logger = logging.getLogger(__name__)


@dataclass
class _Rule:
    trigger_product_ids: set[str] = field(default_factory=set)
    trigger_categories: set[str] = field(default_factory=set)
    suggest_product_ids: list[str] = field(default_factory=list)
    reason: str = ""


def _parse_rules(raw: list[dict[str, Any]]) -> list[_Rule]:
    rules: list[_Rule] = []
    for entry in raw:
        rules.append(
            _Rule(
                trigger_product_ids=set(entry.get("trigger_product_ids", [])),
                trigger_categories={c.lower() for c in entry.get("trigger_categories", [])},
                suggest_product_ids=entry.get("suggest_product_ids", []),
                reason=entry.get("reason", "You might also like this."),
            )
        )
    return rules


class UpsellEngine:
    """Suggest complementary products given a list of cart product-ids.

    Args:
        rules_path: Path to ``upsell_rules.yaml``.
    """

    def __init__(self, rules_path: str):
        self._rules: list[_Rule] = []
        self._load(rules_path)

    def _load(self, path: str) -> None:
        try:
            with open(path) as fh:
                data = yaml.safe_load(fh) or {}
            self._rules = _parse_rules(data.get("rules", []))
            logger.info("[UPSELL] Loaded %d upsell rules from %s", len(self._rules), path)
        except FileNotFoundError:
            logger.warning("[UPSELL] Rules file not found at %s — no upsell suggestions will be generated", path)
        except Exception as exc:
            logger.error("[UPSELL] Failed to load rules from %s: %s", path, exc)

    def get_suggestions(
        self,
        cart_product_ids: list[str],
        cart_products: list[Product],
        all_products: dict[str, Product],
    ) -> list[UpsellSuggestion]:
        """Return a deduplicated, ordered list of upsell suggestions.

        Args:
            cart_product_ids: product_id values currently in the cart.
            cart_products:    full Product objects for items in the cart.
            all_products:     lookup dict product_id → Product for the whole catalogue.
        """
        cart_ids = set(cart_product_ids)
        cart_categories = {p.category.lower() for p in cart_products}

        seen_product_ids: set[str] = set()
        suggestions: list[UpsellSuggestion] = []

        for rule in self._rules:
            # Check if any cart item triggers this rule
            triggered = bool(rule.trigger_product_ids & cart_ids) or bool(
                rule.trigger_categories & cart_categories
            )
            if not triggered:
                continue

            for pid in rule.suggest_product_ids:
                if pid in cart_ids or pid in seen_product_ids:
                    continue
                product = all_products.get(pid)
                if not product:
                    logger.debug("[UPSELL] Suggested product_id=%s not found in catalogue", pid)
                    continue
                seen_product_ids.add(pid)
                suggestions.append(UpsellSuggestion(product=product, reason=rule.reason))
                logger.debug("[UPSELL] Suggesting %s (%s): %s", product.name, pid, rule.reason)

        logger.info(
            "[UPSELL] cart=%s → %d suggestion(s)", cart_product_ids, len(suggestions)
        )
        return suggestions
