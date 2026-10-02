"""OrderingService — orchestrates the product catalogue, cart, and upsell.

The service is the single entry point for all ordering business logic.
Repositories are created per-call (each call opens/closes the DB connection).
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import re

from kiosk_core.ordering.db import get_db
from kiosk_core.ordering.models import (
    CreateOrderRequest,
    Order,
    OrderItemIn,
    Product,
    ProductResolution,
    RemoveOrderItem,
    UpsellRequest,
    UpsellSuggestion,
)
from kiosk_core.ordering.repository import (
    SqliteOrderRepository,
    SqliteProductRepository,
)
from kiosk_core.ordering.upsell import UpsellEngine

logger = logging.getLogger(__name__)

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _normalize(text: str) -> str:
    """Lowercase and collapse all non-alphanumeric runs to single spaces.

    Used to compare a free-form product reference (which may be a fabricated
    id like ``spicy_crunch_burger`` or a spoken name like "spicy crunch
    burger") against catalogue product ids and names.
    """
    return _NON_ALNUM.sub(" ", (text or "").lower()).strip()


def _is_veg_filter(dietary: str | None) -> bool | None:
    """Map a captured dietary preference to an ``is_veg`` repository filter.

    ``"vegetarian"``/``"vegan"`` → ``True`` (veg-only). ``"non_vegetarian"``
    (an explicit "suggest me non veg dishes" request) → ``False`` (non-veg
    only — excludes plain-veg items like fries or a brownie, not just
    "chicken burger"). Anything else, including ``"none"`` (a capability
    statement like "I'm not vegetarian" meaning no restriction) or ``None``
    (nothing stated), → ``None`` (unfiltered).
    """
    if dietary in ("vegetarian", "vegan"):
        return True
    if dietary == "non_vegetarian":
        return False
    return None


def _resolve_against(ref: str, products: list[Product]) -> ProductResolution:
    """Pure product-reference resolver: no I/O, deterministic, unit-testable.

    Implements the exact resolution ladder documented on
    ``OrderingService.resolve_product`` (id/name equality, unique substring,
    unique token-subset, token-overlap ambiguity check, difflib fallback),
    but returns an explicit :class:`ProductResolution` instead of collapsing
    "ambiguous" and "no match at all" into the same ``None``.

    Args:
        ref: Free-form reference, already known non-empty by the caller.
        products: The full candidate set to match against (already fetched;
            this function does no database access).

    Returns:
        A ``ProductResolution`` with status ``MATCH``, ``AMBIGUOUS``, or
        ``NOT_FOUND``.
    """
    nref = _normalize(ref)
    if not nref:
        return ProductResolution(status="NOT_FOUND")

    for p in products:
        if _normalize(p.product_id) == nref or _normalize(p.name) == nref:
            return ProductResolution(status="MATCH", product=p, confidence=1.0)

    contains = [p for p in products if nref in _normalize(p.name)]
    if len(contains) == 1:
        return ProductResolution(status="MATCH", product=contains[0], confidence=0.9)

    # Reverse direction: the catalogue name is a whole-word substring INSIDE
    # a longer/noisier reference, rather than the other way around. The
    # check above only ever asks "is the reference short enough to fit
    # inside a product name?" — it can never match when the caller passes a
    # full noisy utterance as the reference (e.g. an agent forwarding
    # "yes please add the paneer tikka burger to my order" verbatim instead
    # of a clean item name). Observed live: a valid "Paneer Tikka Burger"
    # was rejected as off-menu after a compound utterance because neither
    # direction of substring check nor the token-subset check below (which
    # requires every query token to be a name token, and fails the moment
    # the reference carries extra filler words) ever fired. ``\b`` word
    # boundaries keep this from matching a name as a fragment of an
    # unrelated longer word (e.g. "tea" must not match inside "steak").
    reverse_contains = [
        p
        for p in products
        if _normalize(p.name) and re.search(rf"\b{re.escape(_normalize(p.name))}\b", nref)
    ]
    if len(reverse_contains) == 1:
        return ProductResolution(status="MATCH", product=reverse_contains[0], confidence=0.85)

    qtokens = set(nref.split())
    token_hits = [
        p for p in products if qtokens and qtokens.issubset(set(_normalize(p.name).split()))
    ]
    if len(token_hits) == 1:
        return ProductResolution(status="MATCH", product=token_hits[0], confidence=0.9)

    # Before falling back to raw character-similarity, check for a
    # token-overlap tie among plausible candidates. difflib's ratio does
    # not know that a word like "chicken" vs "paneer" is dietary-defining,
    # not stylistic like "spicy" vs "classic" — so it can rank a
    # completely different item above the one the customer actually named
    # just because more characters happen to line up (observed live:
    # "chicken tikka burger" — not on the menu — resolved to "Paneer
    # Tikka Burger" over "Classic Chicken Burger" purely on character
    # ratio, silently swapping a non-veg request for a veg item). A tie in
    # shared *words* between two or more distinct real products means the
    # reference is genuinely ambiguous and must not be silently guessed —
    # same principle as the uniqueness requirement above.
    #
    # The overlap must be at least 2 words before it counts as a
    # meaningful signal: a single shared generic word ("classic" in both
    # "Classic Chicken Burger" and "Classic French Fries") is common
    # across unrelated categories and must not itself block a genuine
    # single-typo match from reaching the difflib fallback below.
    if qtokens:
        overlap_scores = [
            (len(qtokens & set(_normalize(p.name).split())), p) for p in products
        ]
        max_overlap = max((score for score, _ in overlap_scores), default=0)
        if max_overlap >= 2:
            top = [p for score, p in overlap_scores if score == max_overlap]
            if len(top) > 1:
                return ProductResolution(status="AMBIGUOUS", candidates=top)

    name_map = {_normalize(p.name): p for p in products}
    close = difflib.get_close_matches(nref, list(name_map), n=1, cutoff=0.6)
    if close:
        matched_name = close[0]
        ratio = difflib.SequenceMatcher(None, nref, matched_name).ratio()
        return ProductResolution(status="MATCH", product=name_map[matched_name], confidence=round(ratio, 3))

    return ProductResolution(status="NOT_FOUND")


class OrderingService:
    """Business logic for product catalogue, cart management, and upsell.

    Args:
        upsell_rules_path: Path to ``upsell_rules.yaml``.
    """

    def __init__(self, upsell_rules_path: str):
        self._upsell_engine = UpsellEngine(upsell_rules_path)

    # ------------------------------------------------------------------
    # Products
    # ------------------------------------------------------------------

    async def list_products(
        self, category: str | None = None, dietary: str | None = None
    ) -> list[Product]:
        """List catalogue products, optionally restricted by dietary preference.

        Args:
            dietary: ``"vegetarian"``/``"vegan"`` restricts the result to
                veg items only. ``"non_vegetarian"`` restricts to non-veg
                items only (an explicit "suggest me non veg dishes" request).
                Any other value (including None/"none") returns the
                unfiltered list. See :func:`_is_veg_filter`.
        """
        is_veg = _is_veg_filter(dietary)
        async with get_db() as db:
            repo = SqliteProductRepository(db)
            products = await repo.list_all(category=category, is_veg=is_veg)
        logger.debug(
            "[SERVICE] list_products category=%s dietary=%s → %d result(s)",
            category, dietary, len(products),
        )
        return products

    async def list_bestsellers(
        self, category: str | None = None, dietary: str | None = None
    ) -> list[Product]:
        """Return bestseller products, optionally restricted by dietary preference.

        Args:
            dietary: Same semantics as :meth:`list_products` — restricts to
                veg-only or non-veg-only bestsellers when the customer has
                stated a preference.
        """
        is_veg = _is_veg_filter(dietary)
        async with get_db() as db:
            repo = SqliteProductRepository(db)
            products = await repo.list_bestsellers(category=category, is_veg=is_veg)
        logger.debug(
            "[SERVICE] list_bestsellers category=%s dietary=%s → %d result(s)",
            category, dietary, len(products),
        )
        return products

    async def get_product(self, product_id: str) -> Product | None:
        async with get_db() as db:
            repo = SqliteProductRepository(db)
            return await repo.get(product_id)

    async def resolve_product(self, ref: str) -> Product | None:
        """Resolve a free-form product reference to a catalogue Product.

        The LLM does not reliably reproduce exact product ids (it may invent
        ``spicy_crunch_burger`` instead of ``BURGER-NV-002``). This resolves
        a reference — an exact id, a normalised id, or a product name (full,
        partial, or slightly misheard) — to the real Product, so callers never
        depend on the model emitting a perfect identifier.

        This is a thin compatibility wrapper over
        :meth:`resolve_product_detailed` for existing callers that only need
        "did it resolve or not" — see that method for the full resolution
        ladder and an explicit ``MATCH``/``AMBIGUOUS``/``NOT_FOUND`` result.

        Args:
            ref: An id or name reference, e.g. "BURGER-NV-001" or "classic
                chicken burger".

        Returns:
            The matching Product, or None if no confident match exists
            (including when the reference is genuinely ambiguous — an
            ambiguous match is treated as no match, never a guess).
        """
        result = await self.resolve_product_detailed(ref)
        return result.product if result.status == "MATCH" else None

    async def resolve_product_detailed(self, ref: str) -> ProductResolution:
        """Resolve a free-form product reference with an explicit outcome.

        Resolution order (first match wins):
          1. exact product_id
          2. normalised product_id or name equality
          3. unique normalised-substring match on name
          4. unique product whose name contains every query token
          5. reject as ``AMBIGUOUS`` if two or more distinct products tie on
             the most shared query words — see below
          6. closest name by difflib ratio (cutoff 0.6)

        Step 4 is checked before the difflib ratio (6) on purpose: a
        distinguishing word the customer actually said (e.g. "spicy" in
        "spicy chicken burger") is a precise, deliberate signal that a
        character-similarity ratio can lose to a shorter, more generic name
        ("Classic Chicken Burger" scores marginally higher than "Spicy
        Chicken Crunch Burger" against that query by pure ratio, 0.857 vs
        0.851, silently ordering the wrong burger with no error). Requiring
        every query token to be present AND the match to be unique keeps this
        step precise — it only overrides the ratio when it has a strictly
        stronger, unambiguous signal.

        Step 5 guards the same failure mode one level further out: an
        off-menu reference with no full-subset match (step 4) can still share
        two or more words with two or more *different* real items, and
        character ratio alone cannot tell that "chicken" vs "paneer" is
        dietary-defining while "spicy" vs "classic" is not. Observed live:
        "chicken tikka burger" — not on the menu — resolved to "Paneer Tikka
        Burger" over "Classic Chicken Burger" by pure ratio (0.718 vs 0.667),
        silently swapping a non-veg request for a veg item. When the top
        word-overlap score (requiring at least two shared words, so a single
        generic word like "classic" shared with an unrelated item such as
        "Classic French Fries" cannot trigger this on its own) is shared by
        more than one distinct product, the reference is genuinely ambiguous
        and is rejected rather than guessed — the same "ambiguous match is
        treated as no match" principle as step 4's uniqueness requirement.

        Args:
            ref: An id or name reference, e.g. "BURGER-NV-001" or "classic
                chicken burger".

        Returns:
            A ``ProductResolution`` — ``MATCH`` with the product and a
            confidence, ``AMBIGUOUS`` with the tied candidates, or
            ``NOT_FOUND``.
        """
        if not ref:
            return ProductResolution(status="NOT_FOUND")
        async with get_db() as db:
            repo = SqliteProductRepository(db)
            exact = await repo.get(ref)
            if exact is not None:
                return ProductResolution(status="MATCH", product=exact, confidence=1.0)
            products = await repo.list_all()

        return _resolve_against(ref, products)

    async def resolve_cart_item(self, ref: str, cart_product_ids: list[str]) -> str | None:
        """Resolve a spoken reference against the products already in a cart.

        ``resolve_product`` searches the whole catalogue, which is wrong for
        removals: a customer saying "remove the burger" means the burger they
        ordered, and a catalogue-wide match could land on a different burger
        that was never in the cart. This restricts the same matching strategy
        to the cart's own products, so a removal only ever targets something
        the customer actually has.

        Args:
            ref: Free-form product reference from the customer's utterance.
            cart_product_ids: Product ids currently in the cart.

        Returns:
            The matching cart ``product_id``, or None if no confident match.
        """
        if not ref or not cart_product_ids:
            return None

        async with get_db() as db:
            repo = SqliteProductRepository(db)
            products = [p for p in await repo.list_all() if p.product_id in set(cart_product_ids)]
        if not products:
            return None

        nref = _normalize(ref)
        if not nref:
            return None

        for p in products:
            if _normalize(p.product_id) == nref or _normalize(p.name) == nref:
                return p.product_id

        contains = [p for p in products if nref in _normalize(p.name)]
        if len(contains) == 1:
            return contains[0].product_id

        name_map = {_normalize(p.name): p for p in products}
        close = difflib.get_close_matches(nref, list(name_map), n=1, cutoff=0.6)
        if close:
            return name_map[close[0]].product_id

        qtokens = set(nref.split())
        # Either direction is a hit: "burger" ⊂ "Aloo Tikki Burger" and
        # "aloo tikki burger deluxe" ⊃ "Aloo Tikki Burger".
        token_hits = [
            p for p in products
            if qtokens and (
                qtokens.issubset(set(_normalize(p.name).split()))
                or set(_normalize(p.name).split()).issubset(qtokens)
            )
        ]
        if len(token_hits) == 1:
            return token_hits[0].product_id

        return None

    async def suggest_products(
        self, ref: str, n: int = 5, dietary: str | None = None, min_score: float = 0.5
    ) -> list[Product]:
        """Return the ``n`` catalogue products whose names are closest to ``ref``.

        Used to build a grounded "did you mean" list when resolve_product fails,
        so the agent can offer real options instead of inventing items.

        Args:
            dietary: ``"vegetarian"``/``"vegan"`` restricts the ranked pool to
                veg items first; ``"non_vegetarian"`` restricts it to non-veg
                items first — so an ambiguous reference from a customer who
                has stated a preference (e.g. "burger") offers only matching
                alternatives instead of a mix. If restricting would leave
                nothing to suggest, falls back to the full catalogue rather
                than returning an empty list.
            min_score: Minimum ``difflib`` ratio a candidate must clear to be
                offered as a "did you mean" alternative. A genuinely off-menu
                request (e.g. "dosa", "biryani" at a burger/pizza kiosk) has no
                real relationship to anything we sell — its best ratio match is
                typically well below 0.5 (observed: "dosa" → 0.40, "biryani" →
                0.33). Offering it anyway invited the agent to present an
                unrelated item as if it answered the request. Below this floor
                there is nothing genuinely similar, so return no suggestions
                and let the caller fall back to an honest "we don't have
                anything like that" instead of a fabricated-looking match.
        """
        async with get_db() as db:
            repo = SqliteProductRepository(db)
            products = await repo.list_all()
        pool = products
        preferred_is_veg = _is_veg_filter(dietary)
        if preferred_is_veg is not None:
            filtered = [p for p in products if p.is_veg == preferred_is_veg]
            if filtered:
                pool = filtered
        nref = _normalize(ref)
        scored = [
            (p, difflib.SequenceMatcher(None, nref, _normalize(p.name)).ratio())
            for p in pool
        ]
        scored = [(p, s) for p, s in scored if s >= min_score]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [p for p, _ in scored[:n]]

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def place_order(self, request: CreateOrderRequest, dry_run: bool = False) -> Order:
        """Add items to the customer's cart, creating it only if none is open.

        A customer has at most one live cart per visit. The agent cannot be
        relied on to remember an ``order_id`` across turns — unrelated Q&A
        turns push it out of the replayed history window — so it may call this
        instead of :meth:`update_order_items` for a follow-up item. Reusing the
        open draft here makes the two paths converge and prevents a second cart
        from shadowing (and silently dropping) the first.

        Args:
            request: Target user and the items to add.
            dry_run: If ``True``, run every resolution/write step against a
                real transaction (so the returned preview reflects genuine
                would-be totals/ids) but never call ``db.commit()`` — the
                connection close at the end of the ``async with`` block rolls
                the uncommitted transaction back, so nothing is persisted.
                Used by speculative drafting to preview a turn's outcome
                before the customer's utterance is finalised.

        Returns:
            The resulting draft order (or its would-be preview if ``dry_run``).

        Raises:
            ValueError: If any requested ``product_id`` does not exist.
        """
        async with get_db() as db:
            prod_repo = SqliteProductRepository(db)
            order_repo = SqliteOrderRepository(db)

            order_id = await order_repo.consolidate_drafts(request.user_id)
            reused = order_id is not None
            if order_id is None:
                order_id = await order_repo.create(request.user_id)

            for item_in in request.items:
                product = await prod_repo.get(item_in.product_id)
                if product is None:
                    raise ValueError(f"Product not found: {item_in.product_id}")
                await order_repo.add_item(order_id, item_in, product.price)

            total = await order_repo.update_total(order_id)

            if dry_run:
                # Read the would-be order back from THIS connection/transaction
                # — the writes above are visible here (same transaction) but
                # not to any other connection, and never persisted since we
                # never commit. Must fetch here, not via self.get_order()
                # afterward, which would open a fresh connection and see the
                # pre-rollback (i.e. unmodified) state.
                preview = await order_repo.get(order_id)
                logger.info(
                    "[SERVICE] [DRY-RUN] Would %s order_id=%d user=%s total=%.2f (not persisted)",
                    "add to existing" if reused else "place",
                    order_id, request.user_id, total,
                )
                return preview  # type: ignore[return-value]

            await db.commit()

        logger.info(
            "[SERVICE] %s order_id=%d user=%s total=%.2f",
            "Added to existing" if reused else "Placed",
            order_id, request.user_id, total,
        )
        order = await self.get_order(order_id)
        return order  # type: ignore[return-value]

    async def get_order(self, order_id: int) -> Order | None:
        async with get_db() as db:
            repo = SqliteOrderRepository(db)
            return await repo.get(order_id)

    async def get_current_order(self, user_id: str) -> Order | None:
        """Return the latest draft order for a user, if one exists."""
        async with get_db() as db:
            repo = SqliteOrderRepository(db)
            order = await repo.get_current_draft(user_id)
        logger.info("[SERVICE] Retrieved current draft order for user=%s", user_id)
        return order

    async def update_order_items(
        self, order_id: int, items: list[OrderItemIn], dry_run: bool = False
    ) -> Order:
        """Add or increment items on an existing draft order.

        Args:
            order_id: Target draft order.
            items: Items to add/increment.
            dry_run: See :meth:`place_order` — preview only, never persisted.
        """
        async with get_db() as db:
            prod_repo = SqliteProductRepository(db)
            order_repo = SqliteOrderRepository(db)

            # Verify order exists and is still a draft
            order = await order_repo.get(order_id)
            if order is None:
                raise ValueError(f"Order not found: {order_id}")
            if order.status != "draft":
                raise ValueError(f"Order {order_id} is already {order.status} and cannot be modified")

            for item_in in items:
                product = await prod_repo.get(item_in.product_id)
                if product is None:
                    raise ValueError(f"Product not found: {item_in.product_id}")
                await order_repo.add_item(order_id, item_in, product.price)

            total = await order_repo.update_total(order_id)

            if dry_run:
                preview = await order_repo.get(order_id)
                logger.info(
                    "[SERVICE] [DRY-RUN] Would update order_id=%d new_total=%.2f (not persisted)",
                    order_id, total,
                )
                return preview  # type: ignore[return-value]

            await db.commit()

        logger.info("[SERVICE] Updated order_id=%d new_total=%.2f", order_id, total)
        updated = await self.get_order(order_id)
        return updated  # type: ignore[return-value]

    async def remove_order_items(
        self, order_id: int, items: list[RemoveOrderItem], dry_run: bool = False
    ) -> tuple[Order, list[str]]:
        """Remove items from a draft order.

        Args:
            order_id: Target draft order.
            items: Products to remove. A ``quantity`` of ``None`` means
                "remove the whole line" rather than decrement.
            dry_run: See :meth:`place_order` — preview only, never persisted.

        Returns:
            ``(updated_order, not_found)`` where ``not_found`` lists the
            product ids that were not in the cart. A partially-satisfiable
            request still succeeds for the items that were present — the
            caller reports the remainder to the customer.

        Raises:
            ValueError: If the order does not exist or is no longer a draft.
        """
        not_found: list[str] = []
        async with get_db() as db:
            order_repo = SqliteOrderRepository(db)

            order = await order_repo.get(order_id)
            if order is None:
                raise ValueError(f"Order not found: {order_id}")
            if order.status != "draft":
                raise ValueError(
                    f"Order {order_id} is already {order.status} and cannot be modified"
                )

            for item_in in items:
                removed = await order_repo.remove_item(
                    order_id, item_in.product_id, item_in.quantity
                )
                if removed == 0:
                    not_found.append(item_in.product_id)

            total = await order_repo.update_total(order_id)

            if dry_run:
                preview = await order_repo.get(order_id)
                logger.info(
                    "[SERVICE] [DRY-RUN] Would remove %d item(s) from order_id=%d "
                    "new_total=%.2f not_found=%s (not persisted)",
                    len(items) - len(not_found), order_id, total, not_found,
                )
                return preview, not_found  # type: ignore[return-value]

            await db.commit()

        logger.info(
            "[SERVICE] Removed %d item(s) from order_id=%d new_total=%.2f not_found=%s",
            len(items) - len(not_found), order_id, total, not_found,
        )
        updated = await self.get_order(order_id)
        return updated, not_found  # type: ignore[return-value]

    async def clear_draft_carts(self, user_id: str) -> int:
        """Delete any stale (never-confirmed) draft orders for a user.

        Intended to be called when a brand-new conversation/session starts so
        each session begins with an empty cart instead of resurfacing an
        abandoned draft from a previous visit.
        """
        async with get_db() as db:
            repo = SqliteOrderRepository(db)
            deleted = await repo.delete_draft_orders(user_id)
            await db.commit()

        if deleted:
            logger.info("[SERVICE] Cleared %d stale draft cart(s) for user=%s", deleted, user_id)
        return deleted

    async def cancel_current_order(self, user_id: str, dry_run: bool = False) -> Order | None:
        """Cancel (delete) the customer's entire open draft order, if any.

        Distinct from ``remove_order_items``: that call removes one or more
        named items, requiring the caller (the LLM) to enumerate every item
        in the cart from its own memory of the conversation — "cancel my
        whole order" was previously handled that way, which is fragile (a
        forgotten or hallucinated item name produces a wrong result). This
        deletes the draft in one deterministic step directly from the
        database, with no dependency on the caller recalling cart contents.

        Args:
            user_id: The customer whose draft order to cancel.
            dry_run: If ``True``, skip the delete entirely and just return the
                current snapshot — there is nothing else to preview here since
                the return value is already "the order as it was before
                cancellation", not a post-write state.

        Returns:
            A snapshot of the order as it was immediately before cancellation
            (so the caller can report what was cancelled), or ``None`` if the
            customer had no open draft order.
        """
        order = await self.get_current_order(user_id)
        if order is None:
            return None

        if dry_run:
            logger.info(
                "[SERVICE] [DRY-RUN] Would cancel order_id=%d for user=%s (not persisted)",
                order.order_id, user_id,
            )
            return order

        async with get_db() as db:
            repo = SqliteOrderRepository(db)
            deleted = await repo.delete_draft_orders(user_id)
            await db.commit()

        logger.info(
            "[SERVICE] Cancelled order_id=%d for user=%s (%d draft(s) deleted)",
            order.order_id, user_id, deleted,
        )
        return order

    async def confirm_order(self, order_id: int, dry_run: bool = False) -> Order:
        """Confirm a draft order → status becomes 'confirmed'.

        Args:
            order_id: The draft order to confirm.
            dry_run: See :meth:`place_order` — preview only, never persisted.
        """
        async with get_db() as db:
            repo = SqliteOrderRepository(db)
            order = await repo.get(order_id)
            if order is None:
                raise ValueError(f"Order not found: {order_id}")
            if order.status != "draft":
                raise ValueError(f"Order {order_id} is already {order.status}")

            if dry_run:
                # Build the would-be confirmed preview without writing —
                # ``order`` is a draft snapshot, so mark it confirmed only in
                # the returned copy.
                preview = order.model_copy(update={"status": "confirmed"})
                logger.info(
                    "[SERVICE] [DRY-RUN] Would confirm order_id=%d for user=%s (not persisted)",
                    order_id, order.user_id,
                )
                return preview

            await repo.confirm(order_id)
            await db.commit()

        logger.info("[SERVICE] Confirmed order_id=%d for user=%s", order_id, order.user_id)
        confirmed = await self.get_order(order_id)
        return confirmed  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Upsell
    # ------------------------------------------------------------------

    async def get_upsell_suggestions(self, request: UpsellRequest) -> list[UpsellSuggestion]:
        """Return rule-based upsell suggestions for the given cart."""
        async with get_db() as db:
            prod_repo = SqliteProductRepository(db)
            all_products_list = await prod_repo.list_all()
            all_products = {p.product_id: p for p in all_products_list}

        cart_products = [all_products[pid] for pid in request.product_ids if pid in all_products]
        return self._upsell_engine.get_suggestions(
            cart_product_ids=request.product_ids,
            cart_products=cart_products,
            all_products=all_products,
        )

    # ------------------------------------------------------------------
    # Sync bridge for startup (seed)
    # ------------------------------------------------------------------

    def run_seed(self, products_yaml_path: str) -> int:
        """Synchronous wrapper to seed products from YAML on startup."""
        from kiosk_core.ordering.seed import seed_products
        return asyncio.run(seed_products(products_yaml_path))
