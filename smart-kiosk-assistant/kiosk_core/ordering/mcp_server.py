"""MCP server for kiosk-core ordering tools.

Exposes ordering operations as MCP tools via HTTP using ``fastmcp``.
The agent connects to this server at ``/mcp`` on the kiosk-core port.

The server is mounted into the FastAPI app — not run as a separate process.

Tools exposed:
  - list_products   — list available menu items (optional category filter)
  - get_popular_products — bestseller/most-ordered items (optional category)
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
import re
from typing import Any

from fastmcp import FastMCP

from kiosk_core.config import DEFAULT_LIST_PRODUCTS_SUMMARY, UPSELL_MAX_SUGGESTIONS

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


# Values the model supplies when it means "no filter". The tool docstring says
# to omit the argument for the full menu, but the 4B model tends to fill every
# declared parameter instead, and an unmatched category returns zero rows.
_ALL_CATEGORY_ALIASES = frozenset({
    "all", "any", "all categories", "everything", "full", "full menu", "menu",
    "menu items", "items", "food", "all items", "all products", "none",
    "null", "n/a", "-",
    # Dietary words are not categories — they arrive separately via the
    # `dietary` param (injected server-side, see _DIETARY_INJECTED_TOOLS).
    # The model sometimes puts the dietary word here too when the customer
    # asked for e.g. "veg dishes"/"non veg options" with no named category;
    # treating it as "no category filter" lets the dietary param alone do
    # the filtering instead of failing to match any real category.
    "veg", "vegetarian", "vegan", "non veg", "non-veg", "nonveg",
    "non vegetarian", "non-vegetarian", "nonvegetarian", "non_vegetarian",
})

# Singular forms the model uses for categories stored as plurals.
_CATEGORY_SYNONYMS = {
    "burger": "burgers",
    "pizzas": "pizza",
    "wrap": "wraps",
    "side": "sides",
    "beverage": "beverages",
    "drink": "beverages",
    "drinks": "beverages",
    "dessert": "desserts",
    "sweet": "desserts",
    "sweets": "desserts",
}


def _normalise_category(category: str | None) -> str | None:
    """Map a model-supplied category onto a catalogue category.

    Args:
        category: Raw category argument as sent by the agent.

    Returns:
        A catalogue category name, or ``None`` to apply no filter.
    """
    if category is None:
        return None
    cleaned = category.strip().lower()
    if not cleaned or cleaned in _ALL_CATEGORY_ALIASES:
        return None
    return _CATEGORY_SYNONYMS.get(cleaned, cleaned)


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
        # Cap at UPSELL_MAX_SUGGESTIONS — a kiosk upsell should offer a small
        # number of complements, not the whole catalogue.  Each suggestion the
        # agent speaks costs ~8 output tokens of LLM decode, so this cap is the
        # main lever on spoken-reply latency.
        formatted: list[dict[str, Any]] = []
        for s in suggestions[:UPSELL_MAX_SUGGESTIONS]:
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
    dietary: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve order item references to canonical product_ids.

    Each incoming item may carry the product reference under ``product_id``,
    ``name``, or ``product`` — and that reference may be a real id, a spoken
    name, or a value the LLM fabricated. Every reference is resolved against
    the catalogue via OrderingService.resolve_product so a wrong-format id no
    longer fails the order.

    Resolution is deliberately **per item**. An earlier version aborted the
    whole call on the first unresolvable reference, which silently discarded
    the items the customer really did ask for: a single fabricated id in a
    multi-item request ("4 burgers, 3 fries, 1 cold coffee, 4 petty_fries")
    dropped all four lines and left the cart empty, and the agent then had
    nothing to confirm. Valid items must always reach the cart; only the
    fabricated reference is refused.

    Args:
        dietary: ``"vegetarian"``/``"vegan"`` when the agent knows this
            customer stated that preference this session. Only narrows the
            "did you mean" suggestions offered for an *unresolved* reference
            (e.g. plain "burger" matching several products) — it never blocks
            an explicit, uniquely-resolved order for a specific item, so a
            customer who is more specific than their earlier statement is
            never second-guessed.

    Returns:
        ``(resolved, rejected, ambiguous, resolved_display)``. ``resolved``
        holds {product_id, quantity} entries using real catalogue ids, exactly
        what ``OrderItemIn`` expects. ``rejected`` holds {reference,
        suggestions} for every reference not on the menu. ``ambiguous`` holds
        {reference, category, choices} for references that name a whole
        category rather than a product. ``resolved_display`` parallels
        ``resolved`` with the resolved product's real name attached, for
        templating a customer-facing sentence without a second catalogue
        lookup — kept as a separate list rather than an extra key on
        ``resolved`` because ``resolved`` entries are unpacked directly into
        ``OrderItemIn(**i)``, which rejects unknown fields.
    """
    resolved: list[dict[str, Any]] = []
    resolved_display: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    for it in items:
        ref = it.get("product_id") or it.get("name") or it.get("product") or ""
        qty = it.get("quantity", 1)
        product = await _svc().resolve_product(ref)
        if product is None:
            category = await _resolve_category(ref)
            if category is not None:
                # A category is not a fabricated item — the customer named a
                # real section of the menu without picking from it. Refusing it
                # as "not on the menu" is both false and confusing ("we don't
                # have pizza — we do have Margherita Pizza"). Ask which one.
                choices = await _svc().list_products(category=category)
                logger.info(
                    "[MCP-SERVER] reference '%s' is category '%s' — asking customer to choose",
                    ref, category,
                )
                ambiguous.append({"reference": ref, "category": category, "choices": choices})
                continue
            suggestions = await _svc().suggest_products(ref, dietary=dietary)
            logger.warning("[MCP-SERVER] could not resolve product reference '%s'", ref)
            rejected.append({"reference": ref, "suggestions": suggestions})
        else:
            resolved.append({"product_id": product.product_id, "quantity": qty})
            resolved_display.append({"name": product.name, "quantity": qty})
    return resolved, rejected, ambiguous, resolved_display


async def _resolve_category(ref: str) -> str | None:
    """Return the menu category ``ref`` names, if it names one rather than a product.

    Args:
        ref: A free-form product reference from the model, e.g. ``"a pizza"``.

    Returns:
        The canonical category name, or ``None`` when the reference does not
        denote a whole category.
    """
    if not ref:
        return None
    normalised = re.sub(r"[^a-z0-9 ]", " ", ref.lower())
    tokens = {t for t in normalised.split() if t not in {"a", "an", "the", "some", "one"}}
    if not tokens:
        return None
    products = await _svc().list_products(category=None)
    for category in {p.category for p in products}:
        cat_tokens = set(category.lower().split())
        # Match "pizza" against category "pizza", and tolerate a plural.
        if tokens == cat_tokens or {t.rstrip("s") for t in tokens} == {
            c.rstrip("s") for c in cat_tokens
        }:
            return category
    return None


def _ambiguous_payload(ambiguous: list[dict[str, Any]]) -> dict[str, Any]:
    """Ask the customer to choose within a category they named.

    Mirrors ``_rejection_payload``'s design: the real choices are spelled out
    inside the sentence, not left only in a sibling array, because the model
    reliably skips a bare array and falls back to a useless "please try again".
    """
    entry = ambiguous[0]
    choices = entry["choices"][:4]
    offer = ", ".join(f"{p.name} ({p.price:.0f})" for p in choices)
    message = (
        f"'{entry['reference']}' is a menu category, not a single item, so nothing "
        f"was added yet. Do NOT say it is unavailable — we do sell it. Ask the "
        f"customer which one they want and offer exactly these: {offer}."
    )
    return {
        "needs_choice": True,
        "category": entry["category"],
        "choice_message": message,
        "available_products": [
            {"product_id": p.product_id, "name": p.name, "price": p.price} for p in choices
        ],
    }


# A voice kiosk has no keypad: every quantity arrives through ASR, which
# mishears spoken numbers routinely. Observed live — the customer said "one or
# two Classic Chicken Burger", ASR produced "one and 2,000", the model called
# place_order(quantity=2001), the write **succeeded**, and the kiosk announced a
# total of ₹338,169. Every truthfulness guard stayed silent, and correctly so:
# the claim "I've added Classic Chicken Burger" was true. What was missing is
# not honesty but plausibility. This check runs before any write, so the
# implausible line is left untouched and no "added" claim is possible for it.
MAX_ITEM_QUANTITY = 20


def _split_implausible_quantities(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split ``items`` into plausible-quantity and implausible-quantity lines.

    Deliberately **per item**, mirroring ``_resolve_items`` below. An earlier
    version aborted the ENTIRE call the moment any single line exceeded
    ``MAX_ITEM_QUANTITY`` — observed live: "one mango lassi and 100 pepsi"
    refused the whole request, silently dropping the perfectly clear "one
    mango lassi" along with the implausible "100 pepsi". A misheard number on
    one line is not a reason to also discard every other line the customer
    was clear about.

    Args:
        items: Raw item dicts as supplied by the model.

    Returns:
        ``(plausible, implausible)``. ``plausible`` holds every item whose
        quantity is within bounds (or unparsable, treated as trusted rather
        than silently dropped), unchanged and ready for the normal resolve
        path. ``implausible`` holds ``{reference, quantity}`` for every item
        whose quantity was too large to be genuine.
    """
    plausible: list[dict[str, Any]] = []
    implausible: list[dict[str, Any]] = []
    for it in items:
        ref = it.get("product_id") or it.get("name") or it.get("product") or "that item"
        try:
            qty = int(it.get("quantity", 1))
        except (TypeError, ValueError):
            plausible.append(it)
            continue
        if qty > MAX_ITEM_QUANTITY:
            logger.warning(
                "[MCP-SERVER] implausible quantity %d for '%s' — skipping this line only",
                qty, ref,
            )
            implausible.append({"reference": ref, "quantity": qty})
        else:
            plausible.append(it)
    return plausible, implausible


def _quantity_rejection_payload(implausible: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe skipped implausible-quantity lines for the agent.

    Mirrors ``_rejection_payload`` for off-menu items: the message spells out
    exactly which reference(s) were skipped and why, inside the sentence
    itself, so the model cannot silently drop it and either claim success or
    tell the customer to "try again" (retrying the same misheard number fails
    identically).
    """
    refs = ", ".join(f"'{r['reference']}'" for r in implausible)
    plural = "were" if len(implausible) > 1 else "was"
    message = (
        f"The quantity given for {refs} {plural} implausibly large (over "
        f"{MAX_ITEM_QUANTITY} per item) and almost certainly misheard, so "
        f"{plural} not added. Do not guess a number for {refs} and do not "
        f"say it was added — ask the customer how many they would like."
    )
    return {
        "quantity_rejected_items": [r["reference"] for r in implausible],
        "quantity_rejected_message": message,
        "max_quantity": MAX_ITEM_QUANTITY,
    }


def _nothing_resolved_quantity_error(implausible: list[dict[str, Any]]) -> dict[str, Any]:
    """Error payload for a call where every item had an implausible quantity."""
    payload = _quantity_rejection_payload(implausible)
    return {"error": payload["quantity_rejected_message"], **payload}


def _rejection_payload(rejected: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe unavailable references for the agent, with grounded alternatives.

    The alternatives are spelled out inside the sentence itself rather than
    left only in a sibling array: a bare array is too easy for the model to
    skip, after which it says "there was an issue, please try again" and
    strands the customer in a loop, since retrying the same missing item
    always fails.
    """
    refs = ", ".join(f"'{r['reference']}'" for r in rejected)
    # De-duplicate suggestions across all rejected references, preserving order.
    unique: dict[str, Any] = {}
    for r in rejected:
        for p in r["suggestions"]:
            unique.setdefault(p.product_id, p)
    offer = ", ".join(f"{p.name} ({p.price:.0f})" for p in list(unique.values())[:3])
    plural = "are" if len(rejected) > 1 else "is"
    message = (
        f"{refs} {plural} not on the menu. Do not invent them and do not ask the "
        f"customer to try again. Tell them those {plural} unavailable and offer "
        f"these real alternatives instead: {offer}."
        if offer
        else (
            f"{refs} {plural} not on the menu and there are no similar items to "
            f"suggest. Do not invent a product — tell the customer they are "
            f"unavailable and ask what else they'd like."
        )
    )
    return {
        "unavailable_items": [r["reference"] for r in rejected],
        "unavailable_message": message,
        "available_products": [
            {"product_id": p.product_id, "name": p.name, "price": p.price}
            for p in unique.values()
        ],
    }


def _nothing_resolved_error(rejected: list[dict[str, Any]]) -> dict[str, Any]:
    """Error payload for a call where no reference at all could be resolved."""
    if not rejected:
        return {"error": "No items were supplied."}
    payload = _rejection_payload(rejected)
    return {"error": payload["unavailable_message"], **payload}


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_categories() -> list[dict[str, Any]]:
    """List the menu categories available, with how many items each holds.

    Use this when the customer asks what the restaurant serves in general
    ("what do you have?", "show me the menu") without naming a category.
    Offer the category names back and let them choose, then call
    ``list_products`` with the category they pick.

    Returns:
        One ``{category, item_count}`` entry per category, alphabetically.
    """
    products = await _svc().list_products(category=None)
    counts: dict[str, int] = {}
    for product in products:
        counts[product.category] = counts.get(product.category, 0) + 1
    summary = [{"category": name, "item_count": counts[name]} for name in sorted(counts)]
    logger.info("[MCP-SERVER] list_categories → %d categories", len(summary))
    return summary


@mcp.tool()
async def list_products(
    category: str | None = None, dietary: str | None = None
) -> list[dict[str, Any]] | dict[str, Any]:
    """List menu products in a category, or the category list when none is given.

    Args:
        category: One of: burgers, pizza, wraps, sides, beverages, desserts.
                  Call with a category to get that category's products with
                  prices. Call with NO category only to discover which
                  categories exist — that returns category names and item
                  counts, NOT products.
        dietary: Leave unset — the caller fills this in automatically from
                 whatever the customer stated earlier in the conversation
                 (see the ordering agent's dietary tracking).

    Returns:
        With a category: products with product_id, name, category, price,
        and is_veg. Without one: ``{category, item_count}`` entries to offer
        the customer. With a category/item that matches nothing we carry at
        all (e.g. "dosa", "sushi"): ``{category_not_found: True, message,
        categories}`` — see the comment below for why this is a distinct case.
    """
    requested = category
    category = _normalise_category(category)

    products = await _svc().list_products(category=category, dietary=dietary)

    if not products and category is not None:
        # This string never matched a real category (aliases for "everything"
        # like "all"/"menu"/"food" are already caught above). Two genuinely
        # different situations produce this:
        #   1. The model passed a descriptive filter rather than a category
        #      name ("chicken", "veg", "spicy") for something that DOES exist
        #      among our products — e.g. "chicken" should surface the chicken
        #      burgers.
        #   2. The customer asked about something we do not carry at all
        #      ("dosa", "sushi", "biryani") — nothing in the catalogue is
        #      related to it.
        # These must not be handled the same way. Silently falling back to
        # the FULL catalogue (the previous behaviour) fixed nothing for case 2:
        # the model would cherry-pick unrelated items from the dump and
        # present them as if they answered the question (observed: "Do you
        # have dosa?" was answered with a list of sides). Instead, search the
        # catalogue for products whose name actually contains the requested
        # term. Case 1 then still finds its real matches; case 2 finds
        # nothing and gets an honest "we don't carry that" signal instead of
        # unrelated products.
        all_products = await _svc().list_products(category=None, dietary=dietary)
        tokens = [t for t in re.sub(r"[^a-z0-9 ]", " ", (requested or "").lower()).split() if t]
        name_matches = [
            p for p in all_products
            if any(t in p.name.lower() for t in tokens)
        ] if tokens else []

        if name_matches:
            logger.info(
                "[MCP-SERVER] list_products category=%r matched no category but "
                "%d product name(s) contain it", requested, len(name_matches),
            )
            products = name_matches
            category = "_filtered"  # sentinel: already specific, skip the summary branch below
        else:
            category_names = sorted({p.category for p in all_products})
            logger.warning(
                "[MCP-SERVER] list_products category=%r matched nothing on the menu — "
                "reporting as not carried (categories: %s)", requested, category_names,
            )
            return {
                "category_not_found": True,
                "requested": requested,
                "message": (
                    f"'{requested}' is not on the menu and nothing we carry is related "
                    f"to it. Do not invent a product or offer unrelated items as if "
                    f"they answered the question. Tell the customer we don't have "
                    f"that, then name the categories we do serve: "
                    f"{', '.join(category_names)}. Ask which they'd like to see."
                ),
                "categories": category_names,
            }

    # An unfiltered call means "what do you serve?", never "read me the whole
    # catalogue". Returning 26 products made the model recite all of them:
    # ~19 s of generation and ~40 s of synthesised speech. Returning the
    # categories makes the short answer the only answer available.
    #
    # A dietary filter changes this: "show me veg dishes"/"non veg dishes"
    # with no named category is a genuine request for the (much smaller,
    # dietary-narrowed) item list, not a "what categories do you have"
    # question — collapsing it to a category summary silently drops the
    # customer's actual ask. Only apply the summary shortcut when nothing
    # has narrowed the result set.
    if category is None and dietary is None and DEFAULT_LIST_PRODUCTS_SUMMARY:
        counts: dict[str, int] = {}
        for product in products:
            counts[product.category] = counts.get(product.category, 0) + 1
        summary = [
            {"category": name, "item_count": counts[name]} for name in sorted(counts)
        ]
        logger.info(
            "[MCP-SERVER] list_products category=None (requested=%r) → "
            "%d categories (summary of %d item(s))",
            requested, len(summary), len(products),
        )
        return summary

    logger.info(
        "[MCP-SERVER] list_products category=%s (requested=%r) → %d item(s)",
        category, requested, len(products),
    )
    return [p.model_dump() for p in products]


@mcp.tool()
async def get_popular_products(
    category: str | None = None, dietary: str | None = None
) -> list[dict[str, Any]] | dict[str, Any]:
    """Return the restaurant's popular / bestseller / most-ordered dishes.

    Use this for open-ended requests with no specific item or cart in mind —
    "what do you recommend", "suggest something to drink", "what's your most
    ordered dish", "show me your favourites". This is a catalogue flag set by
    the restaurant, not a generated opinion, so it is safe to state as fact.

    Do NOT use ``get_upsell_suggestions`` for this — that tool only pairs
    items already in the customer's cart and requires real cart product_ids;
    calling it with no cart (or an invented product_id) produces an empty or
    fabricated result.

    Args:
        category: Optional category to narrow within (e.g. "beverages" for
                  "suggest something to drink"). Omit for restaurant-wide
                  favourites.
        dietary: Leave unset — the caller fills this in automatically from
                 whatever the customer stated earlier in the conversation
                 (see the ordering agent's dietary tracking).

    Returns:
        Products with product_id, name, category, price, is_veg,
        is_bestseller=true. An empty list means nothing in that category is
        flagged popular — say so, do not invent an answer.
    """
    category = _normalise_category(category)
    products = await _svc().list_bestsellers(category=category, dietary=dietary)
    logger.info(
        "[MCP-SERVER] get_popular_products category=%s dietary=%s → %d item(s)",
        category, dietary, len(products),
    )
    return [p.model_dump() for p in products]


@mcp.tool()
async def place_order(
    user_id: str,
    items: list[dict[str, Any]],
    dietary: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Add items to the customer's cart (creates it if none is open).

    Safe to call for follow-up items: if the customer already has an open
    order, the items are added to it rather than starting a second one.

    Args:
        user_id: Customer identifier (use "anonymous" if unknown).
        items: List of {product_id, quantity}. product_id may be a catalogue id
            OR a plain product name (e.g. "Classic Chicken Burger") — the server
            resolves it.
        dietary: Leave unset — the caller fills this in automatically from
            anything the customer has already said about diet this session.
        dry_run: Internal/agent-only flag (never set by the model — stripped
            from its schema). When True, every resolution step runs normally
            but nothing is persisted; used for speculative draft previews.

    Returns:
        The order (order_id, items, total, status="draft"), or an error
        dict with ``available_products`` if a reference cannot be matched.
    """
    from kiosk_core.ordering.models import CreateOrderRequest, OrderItemIn

    plausible_items, implausible_qty = _split_implausible_quantities(items)
    if not plausible_items:
        return _nothing_resolved_quantity_error(implausible_qty)

    resolved, rejected, ambiguous, resolved_display = await _resolve_items(plausible_items, dietary=dietary)
    if not resolved:
        if ambiguous:
            return _ambiguous_payload(ambiguous)
        if rejected:
            return _nothing_resolved_error(rejected)
        return _nothing_resolved_quantity_error(implausible_qty)
    try:
        item_list = [OrderItemIn(**i) for i in resolved]
        req = CreateOrderRequest(user_id=user_id, items=item_list)
        order = await _svc().place_order(req, dry_run=dry_run)
        logger.info(
            "[MCP-SERVER] %splace_order user=%s order_id=%d total=%.2f",
            "[DRY-RUN] " if dry_run else "", user_id, order.order_id, order.total,
        )
        result = await _attach_upsell(order.model_dump(mode="json"))
        # Names of exactly what this call added, distinct from ``result["items"]``
        # (the whole cart) — lets the caller announce only the new items.
        result["just_added"] = resolved_display
        if rejected:
            # The valid items are in the cart; only the fabricated ones failed.
            logger.warning(
                "[MCP-SERVER] place_order user=%s added %d item(s), refused %s",
                user_id, len(resolved), [r["reference"] for r in rejected],
            )
            result.update(_rejection_payload(rejected))
        if implausible_qty:
            # Same shape as the off-menu partial rejection above: some items
            # really were added in this same call, an implausible-quantity
            # line was skipped, and the caller must disclose it rather than
            # imply everything requested was added.
            logger.warning(
                "[MCP-SERVER] place_order user=%s added %d item(s), skipped implausible qty %s",
                user_id, len(resolved), [r["reference"] for r in implausible_qty],
            )
            result.update(_quantity_rejection_payload(implausible_qty))
        if ambiguous:
            result.update(_ambiguous_payload(ambiguous))
        return result
    except ValueError as exc:
        logger.warning("[MCP-SERVER] place_order user=%s rejected: %s", user_id, exc)
        return {"error": str(exc)}


@mcp.tool()
async def update_order(
    order_id: int, items: list[dict[str, Any]], dietary: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Add or increment items on an existing draft order.

    Args:
        order_id: The order to update.
        items: List of {product_id, quantity} to add. product_id may be a
            catalogue id OR a plain product name — the server resolves it.
        dietary: Leave unset — the caller fills this in automatically from
            anything the customer has already said about diet this session.
        dry_run: Internal/agent-only flag — see ``place_order``.

    Returns:
        Updated order with recalculated total, or an error dict with
        ``available_products`` if a reference cannot be matched.
    """
    from kiosk_core.ordering.models import OrderItemIn

    plausible_items, implausible_qty = _split_implausible_quantities(items)
    if not plausible_items:
        return _nothing_resolved_quantity_error(implausible_qty)

    resolved, rejected, ambiguous, resolved_display = await _resolve_items(plausible_items, dietary=dietary)
    if not resolved:
        if ambiguous:
            return _ambiguous_payload(ambiguous)
        if rejected:
            return _nothing_resolved_error(rejected)
        return _nothing_resolved_quantity_error(implausible_qty)
    try:
        item_list = [OrderItemIn(**i) for i in resolved]
        order = await _svc().update_order_items(order_id, item_list, dry_run=dry_run)
        logger.info(
            "[MCP-SERVER] %supdate_order order_id=%d new_total=%.2f",
            "[DRY-RUN] " if dry_run else "", order_id, order.total,
        )
        result = await _attach_upsell(order.model_dump(mode="json"))
        # Names of exactly what this call added, distinct from ``result["items"]``
        # (the whole cart) — lets the caller announce only the new items.
        result["just_added"] = resolved_display
        if rejected:
            logger.warning(
                "[MCP-SERVER] update_order order_id=%d added %d item(s), refused %s",
                order_id, len(resolved), [r["reference"] for r in rejected],
            )
            result.update(_rejection_payload(rejected))
        if implausible_qty:
            logger.warning(
                "[MCP-SERVER] update_order order_id=%d added %d item(s), skipped implausible qty %s",
                order_id, len(resolved), [r["reference"] for r in implausible_qty],
            )
            result.update(_quantity_rejection_payload(implausible_qty))
        if ambiguous:
            result.update(_ambiguous_payload(ambiguous))
        return result
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
async def get_current_order(user_id: str = "anonymous") -> dict[str, Any] | None:
    """Get the customer's current open (draft) order without needing its id.

    Use this for ANY "what's in my cart / what's my total / what did I order"
    style question, and for "show my order". Never call ``get_order`` with a
    guessed or remembered order_id for these questions — if the id is wrong
    (including any placeholder like ``12345``) the lookup silently returns
    nothing and forces you to falsely tell the customer their cart is empty.
    This tool resolves the order the same way ``remove_from_order`` and
    ``confirm_active_order`` already do, so it always finds the real cart.

    Args:
        user_id: The customer whose cart to look up. Defaults to "anonymous".

    Returns:
        The current draft order with items/total/status, or null if the
        customer genuinely has no open order yet.
    """
    order = await _svc().get_current_order(user_id)
    if order is None:
        logger.info("[MCP-SERVER] get_current_order user=%s has no draft order", user_id)
        return None
    logger.info(
        "[MCP-SERVER] get_current_order user=%s order_id=%d status=%s total=%.2f",
        user_id, order.order_id, order.status, order.total,
    )
    return order.model_dump(mode="json")


@mcp.tool()
async def confirm_active_order(user_id: str = "anonymous", dry_run: bool = False) -> dict[str, Any]:
    """Confirm the customer's current draft order without needing its id.

    Use this when the customer says "yes", "confirm", or "that's all" and you
    do not have an order_id to hand. It resolves the customer's open draft
    order and finalises it in one step.

    Args:
        user_id: The customer placing the order. Defaults to "anonymous".
        dry_run: Internal/agent-only flag — see ``place_order``.

    Returns:
        Confirmed order with status="confirmed" and the order_id, or an error dict.
    """
    order = await _svc().get_current_order(user_id)
    if order is None:
        logger.warning("[MCP-SERVER] confirm_active_order user=%s has no draft order", user_id)
        return {"error": "No open order to confirm."}
    # An order row can exist with every line removed. Confirming it would tell
    # the customer "your order is confirmed" over an empty cart, which is
    # exactly what happened when a batch add was refused wholesale.
    if not order.items:
        logger.warning(
            "[MCP-SERVER] confirm_active_order user=%s order_id=%d is empty — refusing",
            user_id, order.order_id,
        )
        return {
            "error": (
                "The cart is empty, so there is nothing to confirm. Do not tell the "
                "customer their order is confirmed. Ask them what they would like to order."
            )
        }
    try:
        confirmed = await _svc().confirm_order(order.order_id, dry_run=dry_run)
        logger.info(
            "[MCP-SERVER] %sconfirm_active_order user=%s order_id=%d total=%.2f ✓",
            "[DRY-RUN] " if dry_run else "", user_id, confirmed.order_id, confirmed.total,
        )
        return confirmed.model_dump(mode="json")
    except ValueError as exc:
        logger.warning("[MCP-SERVER] confirm_active_order user=%s rejected: %s", user_id, exc)
        return {"error": str(exc)}


@mcp.tool()
async def remove_from_order(
    user_id: str = "anonymous",
    items: list[dict[str, Any]] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Remove one or more items from the customer's current cart.

    Use this whenever the customer asks to take something off their order —
    "remove the aloo tikki burger", "drop the fries and the coke", "I don't
    want the pepsi anymore", "cancel the two burgers". Several items can be
    removed in a single call: pass one entry per item.

    Args:
        user_id: The customer whose cart to modify. Defaults to "anonymous".
        items: List of {product_id, quantity}. product_id may be a catalogue id
            OR the plain spoken product name (e.g. "Aloo Tikki Burger") — the
            server resolves it, preferring items actually in the cart. Omit
            quantity (or pass 0) to remove the item entirely; pass a number
            only when the customer wants to remove some but not all units.

    Returns:
        The updated order with its recalculated total. ``removed`` lists what
        was taken off and ``not_in_cart`` lists any requested item that was not
        there. When the cart ends up empty, ``cart_empty`` is true.
    """
    from kiosk_core.ordering.models import RemoveOrderItem

    if not items:
        return {"error": "No items specified to remove. Ask the customer which item to take off."}

    order = await _svc().get_current_order(user_id)
    if order is None:
        logger.warning("[MCP-SERVER] remove_from_order user=%s has no draft order", user_id)
        return {
            "error": (
                "There is no open order, so nothing can be removed. Tell the "
                "customer their cart is already empty."
            )
        }

    # Resolve each reference against the cart first: the customer is naming
    # something they already ordered, so a cart line is a far better match than
    # a catalogue-wide fuzzy hit (which could resolve "burger" to an item they
    # never ordered and then report it as "not in your cart").
    cart_ids = {item.product_id for item in order.items}
    resolved: list[RemoveOrderItem] = []
    unresolved: list[str] = []
    for it in items:
        ref = it.get("product_id") or it.get("name") or it.get("product") or ""
        raw_qty = it.get("quantity")
        try:
            qty = int(raw_qty) if raw_qty is not None else None
        except (TypeError, ValueError):
            qty = None
        # 0 / negative are the model's other ways of saying "all of it".
        if qty is not None and qty < 1:
            qty = None
        product = await _svc().resolve_product(ref)
        if product is not None and product.product_id in cart_ids:
            resolved.append(RemoveOrderItem(product_id=product.product_id, quantity=qty))
            continue
        # Not in the cart under that resolution — try to match a cart line by
        # name directly before declaring it absent.
        match = await _svc().resolve_cart_item(ref, list(cart_ids))
        if match is not None:
            resolved.append(RemoveOrderItem(product_id=match, quantity=qty))
        else:
            unresolved.append(ref)

    if not resolved:
        in_cart = ", ".join(item.product_name for item in order.items) or "nothing"
        return {
            "error": (
                f"None of those items are in the cart. The cart currently contains: "
                f"{in_cart}. Tell the customer what is actually in their order and "
                f"ask which of those to remove. Do not claim you removed anything."
            ),
            "cart_items": [
                {"product_id": item.product_id, "name": item.product_name, "quantity": item.quantity}
                for item in order.items
            ],
        }

    try:
        updated, not_found = await _svc().remove_order_items(order.order_id, resolved, dry_run=dry_run)
    except ValueError as exc:
        logger.warning("[MCP-SERVER] remove_from_order user=%s rejected: %s", user_id, exc)
        return {"error": str(exc)}

    removed_names: list[str] = []
    remaining_ids = {item.product_id for item in updated.items}
    for item in resolved:
        if item.product_id in not_found:
            continue
        name = next(
            (o.product_name for o in order.items if o.product_id == item.product_id),
            item.product_id,
        )
        removed_names.append(name)

    result = updated.model_dump(mode="json")
    result["removed"] = removed_names
    result["not_in_cart"] = unresolved + [
        next((o.product_name for o in order.items if o.product_id == pid), pid) for pid in not_found
    ]
    result["cart_empty"] = len(remaining_ids) == 0
    logger.info(
        "[MCP-SERVER] %sremove_from_order user=%s order_id=%d removed=%s not_in_cart=%s new_total=%.2f",
        "[DRY-RUN] " if dry_run else "", user_id, updated.order_id, removed_names, result["not_in_cart"], updated.total,
    )
    return result


@mcp.tool()
async def cancel_order(user_id: str = "anonymous", dry_run: bool = False) -> dict[str, Any]:
    """Cancel the customer's entire open draft order in one step.

    Use this ONLY for "cancel my (whole/entire/complete) order", "start over",
    "clear my cart", "forget the whole order" — i.e. the customer wants to
    discard everything, not just one item. For removing specific named items,
    use ``remove_from_order`` instead.

    This deletes the draft directly from the database — it does NOT require
    (or use) any list of item names, so it cannot miss or hallucinate an item.

    Args:
        user_id: The customer whose draft order to cancel. Defaults to "anonymous".
        dry_run: Internal/agent-only flag — see ``place_order``.

    Returns:
        ``{"cancelled": True, "order_id": ..., "items_removed": [...]}`` on
        success, or ``{"error": ...}`` if there was no open order to cancel.
    """
    order = await _svc().cancel_current_order(user_id, dry_run=dry_run)
    if order is None:
        logger.info("[MCP-SERVER] cancel_order user=%s has no draft order", user_id)
        return {
            "error": (
                "There is no open order to cancel. Tell the customer their "
                "cart is already empty."
            )
        }

    items_removed = [item.product_name for item in order.items]
    logger.info(
        "[MCP-SERVER] %scancel_order user=%s order_id=%d cancelled items=%s",
        "[DRY-RUN] " if dry_run else "", user_id, order.order_id, items_removed,
    )
    return {
        "cancelled": True,
        "order_id": order.order_id,
        "items_removed": items_removed,
    }


@mcp.tool()
async def confirm_order(order_id: int, dry_run: bool = False) -> dict[str, Any]:
    """Confirm a draft order and finalise it.

    Args:
        order_id: The draft order to confirm.
        dry_run: Internal/agent-only flag — see ``place_order``.

    Returns:
        Confirmed order with status="confirmed" and the order_id, or an error dict.
    """
    try:
        order = await _svc().confirm_order(order_id, dry_run=dry_run)
        logger.info(
            "[MCP-SERVER] %sconfirm_order order_id=%d user=%s total=%.2f ✓",
            "[DRY-RUN] " if dry_run else "", order_id, order.user_id, order.total,
        )
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
