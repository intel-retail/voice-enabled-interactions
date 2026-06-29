"""MCP server for kiosk-core ordering tools.

Exposes ordering operations as MCP tools via HTTP using ``fastmcp``.
The agent connects to this server at ``/mcp`` on the kiosk-core port.

The server is mounted into the FastAPI app — not run as a separate process.

Tools exposed:
  - list_products   — list available menu items (optional category filter)
  - place_order     — create a new draft order
  - update_order    — add/increment items on an existing draft order
  - get_order       — get order summary (items, quantities, total)
  - confirm_order   — confirm a draft order → returns Order ID
  - get_upsell_suggestions — get upsell recommendations for a cart
"""

# NOTE: do NOT add `from __future__ import annotations` here.
# fastmcp resolves tool type-hints eagerly via get_type_hints(); deferring
# annotation evaluation causes `Any` and other typing names to be unresolvable
# in the evaluation namespace on Python 3.11.

import logging
from typing import Any

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("kiosk-ordering")

_ordering_service = None


def init_mcp_server(ordering_service) -> None:
    """Inject the OrderingService singleton into the MCP server."""
    global _ordering_service
    _ordering_service = ordering_service
    logger.info("[MCP-SERVER] OrderingService injected ✓")


def _svc():
    if _ordering_service is None:
        raise RuntimeError("OrderingService not yet initialised in MCP server")
    return _ordering_service


async def _attach_upsell(order_result: dict[str, Any]) -> dict[str, Any]:
    """Attach rule-based upsell suggestions to an order result in-place.

    Computes suggestions deterministically from the products currently in the
    order so the agent always receives them with the order response — rather
    than depending on the LLM to make a separate get_upsell_suggestions call,
    which it does inconsistently. Returns the same dict for convenience.
    """
    from kiosk_core.ordering.models import UpsellRequest

    product_ids = [
        it.get("product_id")
        for it in order_result.get("items", [])
        if it.get("product_id")
    ]
    if not product_ids:
        order_result["upsell_suggestions"] = []
        return order_result
    try:
        suggestions = await _svc().get_upsell_suggestions(
            UpsellRequest(product_ids=product_ids)
        )
        # Pre-format a ready-to-speak display string per suggestion so the LLM
        # echoes the exact name and price verbatim instead of hallucinating
        # prices (e.g. inventing "Pepsi (₹40)" when the real price is ₹59).
        # Cap at 2 — a kiosk upsell should offer one or two complements, not the
        # whole catalogue; this also keeps the round-2 prompt and spoken reply short.
        formatted: list[dict[str, Any]] = []
        for s in suggestions[:2]:
            item = s.model_dump()
            prod = item.get("product", {})
            name = prod.get("name", "")
            price = prod.get("price")
            price_int = int(price) if price is not None and float(price).is_integer() else price
            item["display"] = f"{name} (₹{price_int})" if name else ""
            formatted.append(item)
        order_result["upsell_suggestions"] = formatted
        logger.info(
            "[MCP-SERVER] attached %d upsell suggestion(s) to order: %s",
            len(formatted),
            [f["display"] for f in formatted],
        )
    except Exception as exc:  # upsell must never break order placement
        logger.warning("[MCP-SERVER] upsell attach failed: %s", exc)
        order_result["upsell_suggestions"] = []
    return order_result


async def _resolve_items(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    """Resolve order item references to canonical product_ids.

    Each incoming item may carry the product reference under ``product_id``,
    ``name``, or ``product`` — and that reference may be a real id, a spoken
    name, or a value the LLM fabricated. Every reference is resolved against
    the catalogue via OrderingService.resolve_product so a wrong-format id no
    longer fails the order.

    Returns:
        (resolved_items, None) on success, where resolved_items is a list of
        {product_id, quantity} using real catalogue ids; or (None, error) where
        error is a structured dict with the unresolved reference and a grounded
        ``available_products`` list the agent can offer instead of guessing.
    """
    resolved: list[dict[str, Any]] = []
    for it in items:
        ref = it.get("product_id") or it.get("name") or it.get("product") or ""
        qty = it.get("quantity", 1)
        product = await _svc().resolve_product(ref)
        if product is None:
            suggestions = await _svc().suggest_products(ref)
            logger.warning("[MCP-SERVER] could not resolve product reference '%s'", ref)
            return None, {
                "error": f"No product matches '{ref}'. Do not invent one — offer the customer an item from available_products.",
                "available_products": [
                    {"product_id": p.product_id, "name": p.name, "price": p.price}
                    for p in suggestions
                ],
            }
        resolved.append({"product_id": product.product_id, "quantity": qty})
    return resolved, None


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_products(category: str | None = None) -> list[dict[str, Any]]:
    """List available menu products, optionally filtered by category.

    Args:
        category: Filter by category: burgers, pizza, wraps, sides, beverages, desserts.
                  Omit to list all products.

    Returns:
        List of products with product_id, name, category, and price.
    """
    products = await _svc().list_products(category=category)
    logger.info("[MCP-SERVER] list_products category=%s → %d item(s)", category, len(products))
    return [p.model_dump() for p in products]


@mcp.tool()
async def place_order(user_id: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a new draft order.

    Args:
        user_id: Customer identifier (use "anonymous" if unknown).
        items: List of {product_id, quantity} dicts. product_id may be either a
            catalogue id from list_products OR a plain product name (e.g.
            "Classic Chicken Burger") — the server resolves it to the real item.

    Returns:
        The created order with order_id, items, total, and status="draft", or an
        error dict with ``available_products`` if a reference cannot be matched.
    """
    from kiosk_core.ordering.models import CreateOrderRequest, OrderItemIn

    resolved, err = await _resolve_items(items)
    if err is not None:
        return err
    try:
        item_list = [OrderItemIn(**i) for i in resolved]
        req = CreateOrderRequest(user_id=user_id, items=item_list)
        order = await _svc().place_order(req)
        logger.info("[MCP-SERVER] place_order user=%s order_id=%d total=%.2f", user_id, order.order_id, order.total)
        return await _attach_upsell(order.model_dump(mode="json"))
    except ValueError as exc:
        logger.warning("[MCP-SERVER] place_order user=%s rejected: %s", user_id, exc)
        return {"error": str(exc)}


@mcp.tool()
async def update_order(order_id: int, items: list[dict[str, Any]]) -> dict[str, Any]:
    """Add or increment items on an existing draft order.

    Args:
        order_id: The order to update.
        items: List of {product_id, quantity} dicts to add. product_id may be a
            catalogue id OR a plain product name — the server resolves it.

    Returns:
        Updated order with new items and recalculated total, or an error dict
        with ``available_products`` if a reference cannot be matched.
    """
    from kiosk_core.ordering.models import OrderItemIn

    resolved, err = await _resolve_items(items)
    if err is not None:
        return err
    try:
        item_list = [OrderItemIn(**i) for i in resolved]
        order = await _svc().update_order_items(order_id, item_list)
        logger.info("[MCP-SERVER] update_order order_id=%d new_total=%.2f", order_id, order.total)
        return await _attach_upsell(order.model_dump(mode="json"))
    except ValueError as exc:
        logger.warning("[MCP-SERVER] update_order order_id=%d rejected: %s", order_id, exc)
        return {"error": str(exc)}


@mcp.tool()
async def get_order(order_id: int) -> dict[str, Any] | None:
    """Get the current order summary.

    Args:
        order_id: The order to retrieve.

    Returns:
        Order with items, quantities, total, and status, or null if not found.
    """
    order = await _svc().get_order(order_id)
    if order is None:
        logger.warning("[MCP-SERVER] get_order order_id=%d not found", order_id)
        return None
    logger.info("[MCP-SERVER] get_order order_id=%d status=%s total=%.2f", order_id, order.status, order.total)
    return order.model_dump(mode="json")


@mcp.tool()
async def confirm_order(order_id: int) -> dict[str, Any]:
    """Confirm a draft order and finalise it.

    Args:
        order_id: The draft order to confirm.

    Returns:
        Confirmed order with status="confirmed" and the order_id (use as Order ID),
        or an error dict if the order cannot be confirmed.
    """
    try:
        order = await _svc().confirm_order(order_id)
        logger.info("[MCP-SERVER] confirm_order order_id=%d user=%s total=%.2f ✓", order_id, order.user_id, order.total)
        return order.model_dump(mode="json")
    except ValueError as exc:
        logger.warning("[MCP-SERVER] confirm_order order_id=%d rejected: %s", order_id, exc)
        return {"error": str(exc)}


@mcp.tool()
async def get_upsell_suggestions(product_ids: list[str]) -> list[dict[str, Any]]:
    """Get upsell / pairing recommendations for items in the cart.

    Args:
        product_ids: List of product_ids currently in the customer's cart.

    Returns:
        List of upsell suggestions, each with a product and a reason string.
    """
    from kiosk_core.ordering.models import UpsellRequest

    req = UpsellRequest(product_ids=product_ids)
    suggestions = await _svc().get_upsell_suggestions(req)
    logger.info("[MCP-SERVER] get_upsell_suggestions %d suggestion(s) for %s", len(suggestions), product_ids)
    return [s.model_dump() for s in suggestions]
