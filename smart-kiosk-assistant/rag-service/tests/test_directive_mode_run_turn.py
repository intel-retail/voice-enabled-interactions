"""Unit tests for ``plugins.kiosk.directive_mode.run_turn``.

``run_turn`` drives one streaming directive-mode ordering turn: prose is
released to TTS sentence-by-sentence as it decodes, a single ``<act>`` or
``<check_price>`` directive at the tail is dispatched to the ordering tools,
and every price the customer hears is composed *after the fact* from the tool
payload (the model is forbidden from speaking money). With
``AGENT_DIRECTIVE_MODE=true`` shipped in ``.env.example`` this path handles
every order by default, so it needs coverage (Josh review item 25).

These tests mock the two external touch-points — ``stream_completion`` (the raw
LLM stream) and ``_execute``/``_execute_price`` (the MCP ordering calls) — so
they run with no network, model, or Docker. The fake stream reproduces the real
one's contract exactly: it feeds the model text to ``on_delta`` character by
character and stops early when the callback raises ``_StopGeneration`` (the
signal that the rest of the generation is unusable), then returns the text it
emitted.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from plugins.kiosk import directive_mode


# ── Helpers ───────────────────────────────────────────────────────────────────
def _make_fake_stream(full_text: str):
    """Build a ``stream_completion`` replacement that replays ``full_text``.

    Mirrors the real streamer: deltas are pushed one character at a time, and
    an ``on_delta`` that raises ``_StopGeneration`` ends the stream early (as
    the production ``async with client.stream(...)`` block does), returning
    only the characters emitted so far.
    """

    async def _fake_stream(system_prompt, message, history, on_delta):
        out: list[str] = []
        try:
            for ch in full_text:
                out.append(ch)
                on_delta(ch)
        except directive_mode._StopGeneration:
            pass
        return "".join(out)

    return _fake_stream


def _run_turn(message: str, spoken: list[str]):
    """Invoke ``run_turn`` with throwaway prompt context, collecting prose."""
    return asyncio.run(
        directive_mode.run_turn(
            message=message,
            user_id="user-1",
            history=None,
            menu_block="Classic Chicken Burger — ₹189",
            base_instruction="You are a kiosk.",
            on_safe_sentence=spoken.append,
        )
    )


def _has_money(sentences: list[str]) -> bool:
    """True if any streamed sentence contains a currency symbol or a digit."""
    return any("₹" in s or any(c.isdigit() for c in s) for s in sentences)


# ── <act> success ─────────────────────────────────────────────────────────────
def test_act_directive_speaks_prose_and_appends_payload_total(monkeypatch):
    monkeypatch.setattr(
        directive_mode,
        "stream_completion",
        _make_fake_stream(
            "Got it, one Classic Chicken Burger added. "
            "<act>add|Classic Chicken Burger|1</act>"
        ),
    )
    execute = AsyncMock(
        return_value=({"total": 189.0, "upsell_suggestions": []}, ["add_to_order"], ["add_to_order"])
    )
    monkeypatch.setattr(directive_mode, "_execute", execute)

    spoken: list[str] = []
    result = _run_turn("add a classic chicken burger", spoken)

    assert result is not None
    assert result["directive"] is True
    assert result["tool_calls"] == ["add_to_order"]
    # The confirmation prose is streamed; the price-bearing tail is only in the
    # final reply, never streamed to TTS.
    assert "Got it, one Classic Chicken Burger added." in spoken
    assert not _has_money(spoken), spoken
    assert "Your total is now ₹189." in result["reply"]
    assert "Would you like anything else?" in result["reply"]
    execute.assert_awaited_once()


def test_act_directive_appends_upsell_offer_from_payload(monkeypatch):
    monkeypatch.setattr(
        directive_mode,
        "stream_completion",
        _make_fake_stream("Added. <act>add|Classic Chicken Burger|1</act>"),
    )
    monkeypatch.setattr(
        directive_mode,
        "_execute",
        AsyncMock(
            return_value=(
                {"total": 189.0, "upsell_suggestions": [{"display": "a cola for ₹49"}]},
                ["add_to_order"],
                ["add_to_order"],
            )
        ),
    )

    spoken: list[str] = []
    result = _run_turn("add a burger", spoken)

    assert result is not None
    assert "Would you also like a cola for ₹49?" in result["reply"]
    # The upsell text carries a price but is composed server-side, so it must
    # not have been streamed as prose.
    assert not _has_money(spoken), spoken


# ── <check_price> ─────────────────────────────────────────────────────────────
def test_check_price_directive_composes_price_from_payload(monkeypatch):
    monkeypatch.setattr(
        directive_mode,
        "stream_completion",
        _make_fake_stream("Let me check. <check_price>Classic Chicken Burger</check_price>"),
    )
    monkeypatch.setattr(
        directive_mode,
        "_execute_price",
        AsyncMock(return_value=({"name": "Classic Chicken Burger", "price": 189.0}, ["get_product"], [])),
    )

    spoken: list[str] = []
    result = _run_turn("how much is the classic chicken burger", spoken)

    assert result is not None
    assert result["directive"] is True
    assert "The Classic Chicken Burger costs ₹189." in result["reply"]
    assert not _has_money(spoken), spoken


# ── Fallback paths (return None) ──────────────────────────────────────────────
def test_no_directive_falls_back_to_tool_path(monkeypatch):
    monkeypatch.setattr(
        directive_mode,
        "stream_completion",
        _make_fake_stream("Sure, what would you like to add?"),
    )
    # _execute must never be reached on the fallback path.
    execute = AsyncMock()
    monkeypatch.setattr(directive_mode, "_execute", execute)

    spoken: list[str] = []
    result = _run_turn("hello", spoken)

    assert result is None
    execute.assert_not_awaited()


def test_unparseable_directive_falls_back(monkeypatch):
    monkeypatch.setattr(
        directive_mode,
        "stream_completion",
        _make_fake_stream("Okay. <act>garbage-no-pipes</act>"),
    )
    monkeypatch.setattr(directive_mode, "_execute", AsyncMock())

    result = _run_turn("do something odd", [])

    assert result is None


def test_model_speaking_money_suppresses_prose_and_falls_back(monkeypatch):
    # If in-context drift makes the model narrate a total itself, directive
    # mode must refuse to speak it and bail to the safe tool-calling path.
    monkeypatch.setattr(
        directive_mode,
        "stream_completion",
        _make_fake_stream("Got it. Your total is now ₹500. <act>add|Classic Chicken Burger|1</act>"),
    )
    monkeypatch.setattr(directive_mode, "_execute", AsyncMock())

    spoken: list[str] = []
    result = _run_turn("add a burger", spoken)

    assert result is None
    assert "Got it." in spoken
    assert not _has_money(spoken), spoken


# ── Committed-then-failed (cannot fall back) ──────────────────────────────────
def test_committed_then_failed_returns_partial_failure_reply(monkeypatch):
    monkeypatch.setattr(
        directive_mode,
        "stream_completion",
        _make_fake_stream("Done. <act>add|Classic Chicken Burger|1</act>"),
    )
    # executed=None signals the action was rejected, but committed is non-empty
    # (an earlier write already landed), so a fallback would double-apply.
    monkeypatch.setattr(
        directive_mode,
        "_execute",
        AsyncMock(return_value=(None, ["add_to_order", "remove_from_order"], ["add_to_order"])),
    )

    result = _run_turn("add then remove", [])

    assert result is not None
    assert result["reply"] == directive_mode._PARTIAL_FAILURE_REPLY
    assert result["directive"] is True
    assert result["tool_calls"] == ["add_to_order", "remove_from_order"]


def test_rejected_action_without_commit_falls_back(monkeypatch):
    monkeypatch.setattr(
        directive_mode,
        "stream_completion",
        _make_fake_stream("Sure. <act>add|Classic Chicken Burger|1</act>"),
    )
    # executed=None and nothing committed → safe to re-run via the tool path.
    monkeypatch.setattr(
        directive_mode,
        "_execute",
        AsyncMock(return_value=(None, ["add_to_order"], [])),
    )

    result = _run_turn("add a burger", [])

    assert result is None


# ── Payload with no total (e.g. a removal that emptied the cart) ──────────────
def test_act_directive_with_no_total_has_empty_tail(monkeypatch):
    monkeypatch.setattr(
        directive_mode,
        "stream_completion",
        _make_fake_stream("Removed. <act>remove|Classic Chicken Burger|1</act>"),
    )
    monkeypatch.setattr(
        directive_mode,
        "_execute",
        AsyncMock(return_value=({"total": None}, ["remove_from_order"], ["remove_from_order"])),
    )

    spoken: list[str] = []
    result = _run_turn("remove the burger", spoken)

    assert result is not None
    # No total in the payload → no price tail appended; the reply is just the
    # scrubbed prose.
    assert "Your total" not in result["reply"]
    assert "Removed." in result["reply"]


# ── Confirmation claim is held until the confirm tool commits ─────────────────
def test_confirmation_claim_held_until_confirm_commits(monkeypatch):
    # Directive mode streams prose before the <act> directive runs, so a
    # "Your order is confirmed." sentence must NOT reach TTS until
    # confirm_active_order has actually committed.
    monkeypatch.setattr(
        directive_mode,
        "stream_completion",
        _make_fake_stream("Got it. Your order is confirmed. <act>confirm</act>"),
    )
    monkeypatch.setattr(
        directive_mode,
        "_execute",
        AsyncMock(return_value=({"total": 189.0}, ["confirm_active_order"], ["confirm_active_order"])),
    )

    spoken: list[str] = []
    result = _run_turn("confirm my order", spoken)

    assert result is not None
    # "Got it." streams immediately; the claim is held, then released only
    # after the commit — so by the end of the turn both have reached TTS.
    assert "Got it." in spoken
    assert "Your order is confirmed." in spoken
    assert spoken.index("Got it.") < spoken.index("Your order is confirmed.")
    assert "Your order is confirmed." in result["reply"]


def test_confirmation_claim_dropped_when_confirm_does_not_commit(monkeypatch):
    # The model hallucinates a confirmation on a turn that did NOT confirm
    # (here the directive is an add, not a confirm). The claim must never be
    # synthesised, and must not appear in the streamed prose or the reply.
    monkeypatch.setattr(
        directive_mode,
        "stream_completion",
        _make_fake_stream(
            "Sure. Your order is confirmed. <act>add|Classic Chicken Burger|1</act>"
        ),
    )
    monkeypatch.setattr(
        directive_mode,
        "_execute",
        AsyncMock(return_value=({"total": 189.0}, ["add_to_order"], ["add_to_order"])),
    )

    spoken: list[str] = []
    result = _run_turn("add a burger", spoken)

    assert result is not None
    assert "Sure." in spoken
    assert "Your order is confirmed." not in spoken
    assert "confirmed" not in result["reply"].lower()


def test_confirmation_claim_not_spoken_when_confirm_fails(monkeypatch):
    # confirm_active_order fails (empty/stale cart): _execute returns
    # executed=None with nothing committed, so the turn falls back to the
    # tool path — and the held claim is dropped, never spoken.
    monkeypatch.setattr(
        directive_mode,
        "stream_completion",
        _make_fake_stream("Got it. Your order is confirmed. <act>confirm</act>"),
    )
    monkeypatch.setattr(
        directive_mode,
        "_execute",
        AsyncMock(return_value=(None, ["confirm_active_order"], [])),
    )

    spoken: list[str] = []
    result = _run_turn("confirm my order", spoken)

    assert result is None
    assert "Got it." in spoken
    assert "Your order is confirmed." not in spoken

