"""Tests for OrderingAgent orchestration with mocked ADK/LLM events."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from plugins.kiosk import ordering_agent
from agentic import action_result
from plugins.kiosk import reply_templates
from agentic.mcp_client import MCPTool
from plugins.kiosk.ordering_agent import OrderingAgent


def _run(coro: Any) -> Any:
    """Run an async test helper without depending on pytest-asyncio."""
    return asyncio.run(coro)


def test_make_mcp_callable_invokes_mcp_tool(monkeypatch) -> None:
    """The generated ADK callable delegates to mcp_client.call_tool."""
    fake_call_tool = AsyncMock(return_value={"status": "success", "result": "updated"})
    monkeypatch.setattr(ordering_agent, "call_tool", fake_call_tool)
    mcp_tool = MCPTool(name="update_order", server="core", description="Add items")

    fn = OrderingAgent._make_mcp_callable("update_order", mcp_tool)
    result = _run(fn(order_id="ORD-1", items=[{"product_id": "coke", "quantity": 1}]))

    assert result == {"status": "success", "result": "updated"}
    # dry_run is forced onto every mutating tool server-side and stripped from
    # the model-visible schema, so the model can never set it. Outside a
    # speculative draft it is explicitly False, which is what makes a real
    # turn persist even if a stale value leaked in from elsewhere.
    fake_call_tool.assert_awaited_once_with(
        "update_order",
        {
            "order_id": "ORD-1",
            "items": [{"product_id": "coke", "quantity": 1}],
            "dry_run": False,
        },
    )
    assert fn.__name__ == "update_order"
    assert fn.__doc__ == "Add items"


def test_make_mcp_callable_infers_list_type_for_optional_anyof_items(monkeypatch) -> None:
    """Regression test for a real bug: an optional ``items`` parameter (as on
    ``remove_from_order``) is emitted by FastMCP as ``anyOf: [array, null]``
    with no top-level ``"type"``. The naive lookup fell through to ``str``,
    ADK told the model ``items`` was a string, and the model then sent a
    JSON-encoded or comma-joined string that kiosk-core's Pydantic validation
    rejected outright — silently breaking every multi-item removal.
    """
    fake_call_tool = AsyncMock(return_value={"status": "success", "result": "removed"})
    monkeypatch.setattr(ordering_agent, "call_tool", fake_call_tool)
    mcp_tool = MCPTool(
        name="remove_from_order",
        server="core",
        description="Remove items",
        input_schema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "default": "anonymous"},
                "items": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "object"}},
                        {"type": "null"},
                    ],
                    "default": None,
                },
            },
            "required": [],
        },
    )

    fn = OrderingAgent._make_mcp_callable("remove_from_order", mcp_tool)

    assert fn.__annotations__["items"] is list
    # user_id is stripped from the model-visible schema and injected
    # server-side by _mcp_fn from _user_id_ctx (defaults to "anonymous").
    assert "user_id" not in fn.__annotations__

    _run(fn(items=[{"product_id": "coke", "quantity": 1}]))
    # dry_run=False: see test_make_mcp_callable_invokes_mcp_tool.
    fake_call_tool.assert_awaited_once_with(
        "remove_from_order",
        {
            "user_id": "anonymous",
            "items": [{"product_id": "coke", "quantity": 1}],
            "dry_run": False,
        },
    )


@pytest.mark.parametrize(
    ("message", "expected_tools"),
    [
        ("Do you have vegan burgers?", ["knowledge_lookup"]),
        ("Place an order for a paneer burger", ["list_products", "place_order", "get_upsell_suggestions"]),
        ("Add a coke to my order", ["update_order", "get_upsell_suggestions"]),
        ("Confirm my order", ["confirm_order"]),
    ],
)
def test_chat_records_scripted_adk_tool_calls(
    message: str, expected_tools: list[str], monkeypatch
) -> None:
    """Scripted ADK runner events model OVMS tool-call choices without OVMS."""
    pytest.importorskip("google.adk")

    # chat() calls _refresh_mcp_tools() whenever no MCP tools are registered,
    # which otherwise attempts a real network discovery against kiosk-core.
    # Pretend a tool is already registered so it stays a no-op, matching this
    # test's "model OVMS tool-call choices without OVMS" intent.
    monkeypatch.setattr(ordering_agent, "get_all_tools", lambda: {"dummy": object()})

    agent = OrderingAgent()
    agent._bootstrapped = True
    agent._agent = SimpleNamespace(name="kiosk_ordering_agent")
    agent._session_service = _FakeSessionService()
    agent._runner = _FakeRunner()

    result = _run(agent.chat(message=message, session_id="session-1", user_id="user-1"))

    assert result["tool_calls"] == expected_tools
    assert "reply" in result


class _FakeSessionService:
    """Session service that makes _ensure_session a no-op."""

    async def get_session(self, app_name: str, user_id: str, session_id: str) -> object:
        return object()


class _FakeRunner:
    """ADK Runner replacement that yields scripted tool-call events."""

    async def run_async(
        self,
        user_id: str,
        session_id: str,
        new_message: object,
    ) -> Any:
        text = new_message.parts[0].text.lower()
        if "vegan" in text:
            tools = ["knowledge_lookup"]
            reply = "Yes, we have vegan options."
        elif "place" in text:
            tools = ["list_products", "place_order", "get_upsell_suggestions"]
            reply = "I created a draft order."
        elif "add a coke" in text:
            tools = ["update_order", "get_upsell_suggestions"]
            reply = "I added a coke."
        elif "confirm" in text:
            tools = ["confirm_order"]
            reply = "Your order is confirmed!"
        else:
            tools = []
            reply = "How can I help?"

        # Real ADK events surface tool invocations as `function_call` parts on
        # `event.content` — it never populates a top-level `event.tool_call`
        # (see `_run_turn`'s parsing). Mirror that shape here so this fixture
        # actually exercises the real event-parsing path instead of a stale one.
        for tool_name in tools:
            part = SimpleNamespace(function_call=SimpleNamespace(name=tool_name), text=None)
            yield SimpleNamespace(partial=False, content=SimpleNamespace(parts=[part]))
        reply_part = SimpleNamespace(function_call=None, text=reply)
        yield SimpleNamespace(partial=False, content=SimpleNamespace(parts=[reply_part]))


class TestStripKnowledgeMarkers:
    """Pre-grounding delimiters must never reach TTS."""

    def test_parroted_block_is_unwrapped_keeping_content(self):
        reply = ("[knowledge] The restaurant's name is QuickBite Express. "
                 "The operating hours are Mon-Thu 8 AM-11 PM. [/knowledge]")
        out, forged = ordering_agent._strip_knowledge_markers(reply, pregrounded=True)
        assert "[knowledge]" not in out
        assert "[/knowledge]" not in out
        assert "QuickBite Express" in out
        assert forged is False

    def test_case_insensitive(self):
        out, _ = ordering_agent._strip_knowledge_markers(
            "[KNOWLEDGE] Hi [/Knowledge]", pregrounded=True)
        assert "[" not in out

    def test_clean_reply_untouched(self):
        reply = "We open at 8 AM."
        assert ordering_agent._strip_knowledge_markers(
            reply, pregrounded=True) == (reply, False)

    def test_clean_reply_untouched_without_pregrounding(self):
        # No markers means nothing was claimed, so an ordinary reply on an
        # ungrounded turn is not itself a forgery.
        reply = "Would you like anything else?"
        assert ordering_agent._strip_knowledge_markers(
            reply, pregrounded=False) == (reply, False)

    def test_marker_only_reply_falls_back_to_original(self):
        # Nothing speakable would remain; never return empty.
        out, _ = ordering_agent._strip_knowledge_markers(
            "[knowledge][/knowledge]", pregrounded=True)
        assert out.strip() != ""

    def test_forged_block_without_pregrounding_is_reported(self):
        # Live regression: on a turn with no retrieval and no tool call the
        # model wrapped its own parametric memory in [knowledge] markers, and
        # the unconditional unwrap laundered it into a grounded-looking answer.
        # A marker is not grounding.
        reply = ("[knowledge] The President of India is Droupadi Murmu. "
                 "She took office on 17 July 2022. [/knowledge]")
        out, forged = ordering_agent._strip_knowledge_markers(reply, pregrounded=False)
        assert forged is True
        assert "[knowledge]" not in out

    def test_gate_withholds_sentence_containing_marker(self):
        gate = ordering_agent._SentenceGate(message="hi", emit=None)
        assert gate._is_safe("[knowledge] We open at 8 AM.", ["list_products"]) is False

    def test_truncated_closing_tag_is_trimmed(self):
        # Generation hit its token cap mid-delimiter, leaving a dangling
        # "[/knowledge" (no closing bracket) at the end of the reply.
        reply = ("QuickBite Express is open Mon-Thu 8 AM-11 PM. [/knowledge")
        out, _ = ordering_agent._strip_knowledge_markers(reply, pregrounded=True)
        assert "[/knowledge" not in out
        assert "[" not in out
        assert out.startswith("QuickBite Express")

    def test_truncated_opening_tag_is_trimmed(self):
        reply = "We are open 8 AM-11 PM. [knowledge"
        out, _ = ordering_agent._strip_knowledge_markers(reply, pregrounded=True)
        assert "[knowledge" not in out

    def test_truncated_tag_mid_reply_not_stripped(self):
        # Only trim a truncated tag at the tail; a literal "[" earlier in
        # the reply is not this failure mode and must be left alone.
        reply = "Our hours [see menu for details] are 8 AM-11 PM."
        assert ordering_agent._strip_knowledge_markers(
            reply, pregrounded=True) == (reply, False)


class TestIsOutOfScope:
    """The kiosk may not answer questions outside its remit from memory.

    Every other recovery guard is intent-keyed, so a question matching no
    intent regex reached the customer with no grounding check at all. These
    cases are the live regressions that exposed the gap, plus the in-domain
    turns that must keep working exactly as before.
    """

    @pytest.mark.parametrize("message", [
        "Can you tell me who is the president of India?",
        "Can you tell me the size of football player? the ground.",
        "What is the capital of France?",
        "Who won the world cup?",
    ])
    def test_world_knowledge_questions_are_out_of_scope(self, message):
        assert ordering_agent._is_out_of_scope(message) is True

    @pytest.mark.parametrize("message", [
        "What are the burger options?",
        "Can you suggest something to drink?",
        "What did I order?",
        "How much is the classic chicken burger?",
        "What are your opening hours?",
        "Can you remove the pizza from my order?",
        "Do you have parking?",
        "What is on the menu?",
        "Tell me something about the restaurant.",
    ])
    def test_kiosk_questions_are_in_scope(self, message):
        assert ordering_agent._is_out_of_scope(message) is False

    @pytest.mark.parametrize("message", [
        "hi",
        "Hello there",
        "how are you?",
        "thanks",
        "yes",
        "ok",
        "bye",
    ])
    def test_smalltalk_is_not_refused(self, message):
        assert ordering_agent._is_out_of_scope(message) is False

    @pytest.mark.parametrize("message", [
        "my name is Ravi",
        "I am really hungry today",
        "",
    ])
    def test_statements_are_never_out_of_scope(self, message):
        # A statement wants a conversational reply, not a scope refusal.
        assert ordering_agent._is_out_of_scope(message) is False

    @pytest.mark.parametrize("message", [
        "what else?",
        "which one?",
        "and then?",
        "how many?",
    ])
    def test_short_followups_are_not_refused(self, message):
        # These inherit their subject from the previous turn. Refusing them
        # would break ordinary conversation, and they are far shorter than any
        # world-knowledge question, which always names its subject.
        assert ordering_agent._is_out_of_scope(message) is False

    def test_gate_withholds_everything_until_a_tool_runs(self):
        # The grounding guard only fires when `not tool_calls`, so gate
        # condition (a) is its complete mirror: no sentence from a tool-less
        # turn may be spoken before the guard can replace the reply.
        gate = ordering_agent._SentenceGate(
            message="Can you tell me who is the president of India?", emit=None)
        assert gate._is_safe("The President of India is Droupadi Murmu.", []) is False


class TestStripCitationMarkers:
    """Hallucinated [N] citation markers must never reach TTS."""

    def test_leading_marker_is_stripped(self):
        reply = "[1] QuickBite Express has a shared lot adjacent to the outlet."
        out = ordering_agent._strip_citation_markers(reply)
        assert "[1]" not in out
        assert out.startswith("QuickBite Express")

    def test_multiple_markers_stripped(self):
        reply = "[1] We open at 8 AM. [2] We close at 11 PM."
        out = ordering_agent._strip_citation_markers(reply)
        assert "[1]" not in out and "[2]" not in out

    def test_clean_reply_untouched(self):
        reply = "We open at 8 AM."
        assert ordering_agent._strip_citation_markers(reply) == reply

    def test_marker_only_reply_falls_back_to_original(self):
        assert ordering_agent._strip_citation_markers("[1]").strip() != ""

    def test_does_not_strip_order_id_like_numbers(self):
        # Order ids are plain numbers with no brackets - must be untouched.
        reply = "Your order id is 12345."
        assert ordering_agent._strip_citation_markers(reply) == reply

    def test_gate_withholds_sentence_containing_marker(self):
        gate = ordering_agent._SentenceGate(message="hi", emit=None)
        assert gate._is_safe("[1] We open at 8 AM.", ["knowledge_lookup"]) is False


class TestDedupeRepeatedSentences:
    """A fact echoed twice (raw block + model paraphrase) must speak once."""

    def test_exact_duplicate_sentence_collapsed(self):
        reply = (
            "The restaurant has a seating capacity of 50 people.   "
            "The restaurant has a seating capacity of 50 people."
        )
        out = ordering_agent._dedupe_repeated_sentences(reply)
        assert out == "The restaurant has a seating capacity of 50 people."

    def test_case_and_whitespace_insensitive(self):
        reply = "We open at 8 AM. we OPEN  at 8 am."
        out = ordering_agent._dedupe_repeated_sentences(reply)
        assert out == "We open at 8 AM."

    def test_non_duplicate_reply_untouched(self):
        reply = "We open at 8 AM. We close at 11 PM."
        assert ordering_agent._dedupe_repeated_sentences(reply) == reply

    def test_single_sentence_untouched(self):
        reply = "We open at 8 AM."
        assert ordering_agent._dedupe_repeated_sentences(reply) == reply

    def test_empty_reply_untouched(self):
        assert ordering_agent._dedupe_repeated_sentences("") == ""

    def test_three_way_duplicate_keeps_one(self):
        reply = "QuickBite Express. QuickBite Express. QuickBite Express."
        out = ordering_agent._dedupe_repeated_sentences(reply)
        assert out == "QuickBite Express."


class TestNeedsToolRetryHistoryQuestion:
    """A question about a PAST order action must not force a retry/fallback.

    Regression for a live bug: "What did you remove from my cart earlier?"
    contains the bare _ORDER_ACTION_RE keyword "remove", which used to force
    a tool-call retry and then _ORDER_CLAIM_FALLBACK even though the model's
    original, tool-less answer ("I removed Aloo Tikki Burger from your
    cart.") was already correct — there is no tool to answer "what did you do
    a moment ago", only the turn's own conversation memory.
    """

    def test_history_question_about_removal_does_not_retry(self):
        reply = "I removed Aloo Tikki Burger from your cart."
        message = "What did you remove from my cart earlier?"
        should_retry, _ = ordering_agent._needs_tool_retry(reply, message)
        assert should_retry is False

    def test_history_question_with_leading_clause_does_not_retry(self):
        reply = "I removed Aloo Tikki Burger from your cart."
        message = "In this conversation, what did you remove from my cart?"
        should_retry, _ = ordering_agent._needs_tool_retry(reply, message)
        assert should_retry is False

    def test_history_question_about_adding_does_not_retry(self):
        reply = "You ordered a Classic Chicken Burger earlier."
        message = "What did I add to my cart before?"
        should_retry, _ = ordering_agent._needs_tool_retry(reply, message)
        assert should_retry is False

    def test_fresh_removal_request_still_retries(self):
        # A genuine new action request must still force the tool-call retry.
        reply = "Sure, I'll take care of that."
        message = "Please remove the burger from my order."
        should_retry, nudge = ordering_agent._needs_tool_retry(reply, message)
        assert should_retry is True
        assert nudge == ordering_agent._ORDER_NUDGE

    def test_fresh_confirm_request_still_retries(self):
        reply = "Sure."
        message = "That's all, please confirm my order."
        should_retry, nudge = ordering_agent._needs_tool_retry(reply, message)
        assert should_retry is True
        assert nudge == ordering_agent._ORDER_NUDGE


class TestNeedsToolRetryStartOver:
    """"Start a new order"/"start over" must force a cancel_order retry.

    Regression for a live gap: this phrasing was only ever prompt guidance
    (Rule 6b named "start over" as a cancel_order trigger) with no
    deterministic backstop, the same gap _ORDER_STATUS_RE closed for status
    questions. If the model answered without calling any tool, there was
    nothing to force a retry.
    """

    def test_start_new_order_retries(self):
        reply = "Sure thing!"
        message = "I would like to start a new order"
        should_retry, nudge = ordering_agent._needs_tool_retry(reply, message)
        assert should_retry is True
        assert nudge == ordering_agent._ORDER_NUDGE

    def test_start_over_retries(self):
        reply = "Okay, let's do that."
        message = "Let's start over"
        should_retry, nudge = ordering_agent._needs_tool_retry(reply, message)
        assert should_retry is True
        assert nudge == ordering_agent._ORDER_NUDGE

    def test_start_fresh_retries(self):
        reply = "No problem."
        message = "Can we start fresh please"
        should_retry, nudge = ordering_agent._needs_tool_retry(reply, message)
        assert should_retry is True
        assert nudge == ordering_agent._ORDER_NUDGE

    def test_begin_new_order_retries(self):
        reply = "Sure."
        message = "I want to begin a new order"
        should_retry, nudge = ordering_agent._needs_tool_retry(reply, message)
        assert should_retry is True
        assert nudge == ordering_agent._ORDER_NUDGE

    def test_status_question_does_not_match_start_over(self):
        # Negative control: must not double-fire against _ORDER_STATUS_RE's
        # own phrasing — both regexes route to the same nudge anyway, but
        # they must each be scoped to their own phrasing, not overlap.
        assert not ordering_agent._START_OVER_RE.search("What is the status of my order?")

    def test_plain_cancel_does_not_match_start_over(self):
        # Negative control: bare "cancel" is _ORDER_ACTION_RE's job, not this
        # regex's — confirms no accidental keyword overlap between the two.
        assert not ordering_agent._START_OVER_RE.search("Cancel my order")


class TestMutatingToolsNeverTakeTemplatingShortcut:
    """Cart-mutating tools must never end the ADK turn via `skip_summarization`.

    An earlier version of this fix used a regex to detect "compound"
    utterances ("remove X and add Y") and only excluded those from the
    templating shortcut. That is the wrong layer to decide this at — a
    regex can only recognise wording someone already anticipated, and a
    customer's real phrasing is unbounded. The correct fix is architectural:
    the shortcut is never available for a mutating tool at all, so the model
    always gets the tool result back and decides — using the request and the
    tool results, not surface wording — whether another action is still
    owed. Read-only catalogue tools keep the shortcut since a pure browse
    has no cart side effect to lose.
    """

    def test_mutating_tools_are_excluded_from_the_shortcut(self):
        assert action_result.MUTATING_TOOLS == frozenset({
            "place_order", "update_order", "remove_from_order",
            "cancel_order", "confirm_order", "confirm_active_order",
        })
        # A mutating tool must never be in the set the shortcut is allowed
        # to fire for.
        eligible_for_shortcut = (
            reply_templates.SPEAKABLE_TOOLS - action_result.MUTATING_TOOLS
        )
        for tool in action_result.MUTATING_TOOLS:
            assert tool not in eligible_for_shortcut

    def test_catalogue_tools_remain_eligible_for_the_shortcut(self):
        for tool in ("list_products", "list_categories"):
            assert tool in reply_templates.SPEAKABLE_TOOLS
            assert tool not in action_result.MUTATING_TOOLS

    def test_remove_from_order_never_sets_skip_summarization(self, monkeypatch):
        """End-to-end: even a single, cleanly-templatable removal must leave
        `skip_summarization` unset so ADK always asks the model whether more
        of the customer's request remains.
        """
        fake_call_tool = AsyncMock(return_value={
            "removed": [{"product_id": "classic_chicken_burger", "name": "Classic Chicken Burger"}],
            "not_in_cart": [],
            "total": 438.0,
        })
        monkeypatch.setattr(ordering_agent, "call_tool", fake_call_tool)

        counter_token = ordering_agent._tool_call_count_ctx.set(
            ordering_agent._ToolCallCounter()
        )
        utterance_token = ordering_agent._utterance_ctx.set(
            "remove classic chicken burger and add one margherita pizza instead"
        )
        try:
            mcp_tool = MCPTool(name="remove_from_order", server="core", description="Remove items")
            fn = OrderingAgent._make_mcp_callable("remove_from_order", mcp_tool)

            tool_context = SimpleNamespace(actions=SimpleNamespace(skip_summarization=False))
            result = _run(fn(
                tool_context=tool_context,
                user_id="anonymous",
                items=[{"product_id": "classic_chicken_burger", "quantity": 1}],
            ))
        finally:
            ordering_agent._tool_call_count_ctx.reset(counter_token)
            ordering_agent._utterance_ctx.reset(utterance_token)

        # The template WOULD have produced a clean spoken reply here (that is
        # exactly what made the old regex-gated version wrongly skip), but
        # the shortcut must never fire for a mutating tool regardless.
        assert tool_context.actions.skip_summarization is False
        assert isinstance(result, dict)


class TestKnowledgeQueryRe:
    """A vague "about the restaurant" question must still force knowledge_lookup.

    Regression for a live bug: "Can you tell me something about the
    restaurant?" matched no specific-fact keyword (hours/address/parking/...),
    so no tool call was forced and the model free-hallucinated a generic
    filler answer ("open 10 AM-10 PM", "located at 123 Main Street") that
    matches no real knowledge-base fact.
    """

    @pytest.mark.parametrize(
        "message",
        [
            "Can you tell me something about the restaurant",
            "Tell me about your restaurant",
            "Tell me something about this place",
            "What is the restaurant name?",
            "What are your opening hours?",
        ],
    )
    def test_outlet_questions_match(self, message):
        assert ordering_agent._KNOWLEDGE_QUERY_RE.search(message)

    def test_unrelated_message_does_not_match(self):
        assert not ordering_agent._KNOWLEDGE_QUERY_RE.search(
            "I would like to order one burger"
        )


class TestCatalogueQueryRe:
    """"Do you have <off-menu item>?" must still force list_products.

    Regression for a live bug: "Do you have any sandwiches on the menu?"
    matched the catalogue guard only because it contains "menu", but "Do you
    have sandwiches?" alone matched nothing (no _CATALOGUE_QUERY_RE keyword
    covers "sandwich", since the list is built from real menu categories), so
    the model was free to hallucinate "yes we have sandwiches" — reintroducing
    the exact off-menu-item bug the knowledge-base fix addressed.
    """

    @pytest.mark.parametrize(
        "message",
        [
            "Do you have sandwiches?",
            "Do you have any sandwiches on the menu?",
            "Do you have dosa?",
            "Do you serve pasta?",
            "Have you got any biryani?",
        ],
    )
    def test_off_menu_have_questions_match(self, message):
        assert ordering_agent._CATALOGUE_QUERY_RE.search(message)

    def test_unrelated_message_does_not_match(self):
        assert not ordering_agent._CATALOGUE_QUERY_RE.search(
            "What are your opening hours?"
        )


class TestStripContextBreadcrumb:
    """The cosmetic "[Context: ...]" ingestion tag must be stripped in-place,
    not treated as a leak (regression: it used to share a regex with real
    sensitive fields, so any truthful reply that happened to start with it
    was discarded and replaced with a refusal)."""

    def test_breadcrumb_only_is_stripped_not_replaced(self):
        reply = (
            "[Context: QuickBite Express — Restaurant Knowledge Base] "
            "QuickBite Express is an Indian + Continental Fusion QSR in "
            "Chennai, South India."
        )
        out = ordering_agent._strip_context_breadcrumb(reply)
        assert "[Context:" not in out
        assert "QuickBite Express is an Indian + Continental Fusion QSR" in out

    def test_clean_reply_untouched(self):
        reply = "We have shared parking with 2-wheeler and 4-wheeler bays."
        assert ordering_agent._strip_context_breadcrumb(reply) == reply

    def test_breadcrumb_then_admin_leak_still_falls_back(self):
        # Breadcrumb stripping runs first in the pipeline, but a real
        # sensitive field alongside it must still trigger the hard fallback.
        reply = (
            "[Context: QuickBite Express — Restaurant Knowledge Base] "
            "FSSAI License: 10015033005321"
        )
        stripped = ordering_agent._strip_context_breadcrumb(reply)
        assert ordering_agent._strip_admin_leak(stripped) == ordering_agent._ADMIN_LEAK_FALLBACK


class TestStripLeakedContextTags:
    """A leaked ``[customer_name=X]``/``[user_id=X]``/``[dietary=X]`` scaffold
    tag — the internal signal ``chat()`` prefixes onto the customer's turn —
    must never reach TTS, whether echoed in its exact injected bracket form
    or paraphrased by the model into an invented XML-style tag.
    """

    def test_bracket_tag_prefix_is_stripped_keeping_the_real_reply(self):
        reply = "[customer_name=tester] Please tell me what items are in your order."
        out = ordering_agent._strip_leaked_context_tags(reply)
        assert out == "Please tell me what items are in your order."

    def test_bracket_tag_with_trailing_colon_is_stripped(self):
        reply = "[customer_name=Paris]: Goodbye! Have a great day!"
        out = ordering_agent._strip_leaked_context_tags(reply)
        assert out == "Goodbye! Have a great day!"

    def test_xml_style_tag_with_real_content_around_it_is_stripped(self):
        reply = "Got it <customer_name>Arjun</customer_name>, what would you like?"
        out = ordering_agent._strip_leaked_context_tags(reply)
        assert "<customer_name>" not in out
        assert "</customer_name>" not in out
        assert "Got it" in out and "what would you like?" in out

    def test_bare_xml_tag_reply_gets_a_minimal_natural_substitute(self):
        # Regression: observed live, "What's my name?" got back the ENTIRE
        # reply as a bare `<customer_name>Arjun</customer_name>` with nothing
        # else spoken — stripping alone would leave either "" or just
        # "Arjun", neither of which is an acceptable spoken reply.
        reply = "<customer_name>Arjun</customer_name>"
        out = ordering_agent._strip_leaked_context_tags(reply)
        assert out == "Got it, Arjun."

    def test_bare_bracket_tag_reply_gets_a_minimal_natural_substitute(self):
        reply = "[customer_name=Arjun]"
        out = ordering_agent._strip_leaked_context_tags(reply)
        assert out == "Got it, Arjun."

    def test_bare_tag_with_no_recoverable_name_uses_generic_fallback(self):
        reply = "<user_id>tester</user_id>"
        out = ordering_agent._strip_leaked_context_tags(reply)
        assert out == ordering_agent._CONTEXT_TAG_ONLY_FALLBACK

    def test_user_id_bracket_tag_is_stripped(self):
        reply = "[user_id=tester] Sure, here's your order total: ₹596."
        out = ordering_agent._strip_leaked_context_tags(reply)
        assert out == "Sure, here's your order total: ₹596."

    def test_dietary_bracket_tag_is_stripped(self):
        reply = "[dietary=vegetarian] We have several veg options available."
        out = ordering_agent._strip_leaked_context_tags(reply)
        assert out == "We have several veg options available."

    def test_clean_reply_with_no_tag_is_untouched(self):
        reply = "We're called QuickBite Express. Would you like to know anything else?"
        assert ordering_agent._strip_leaked_context_tags(reply) == reply

    def test_empty_reply_is_returned_unchanged(self):
        assert ordering_agent._strip_leaked_context_tags("") == ""


class TestStripAdminLeak:
    """Internal license/tax/context fields must never reach a customer."""

    def test_context_tag_leak_is_replaced(self):
        reply = (
            "[Context: QuickBite Express — Restaurant Knowledge Base] "
            "FSSAI License: 10015033005321 GST Registration: 33AAACQ5678G1ZM "
            "Hours: Mon-Thu 8 AM-11 PM"
        )
        out = ordering_agent._strip_admin_leak(reply)
        assert out == ordering_agent._ADMIN_LEAK_FALLBACK

    def test_outlet_code_leak_is_replaced(self):
        reply = "Our Outlet Code is QBE-CHN-001, Parent Company QuickBite India Pvt. Ltd."
        assert ordering_agent._strip_admin_leak(reply) == ordering_agent._ADMIN_LEAK_FALLBACK

    def test_clean_reply_untouched(self):
        reply = "We have shared parking with 2-wheeler and 4-wheeler bays."
        assert ordering_agent._strip_admin_leak(reply) == reply

    def test_breadcrumb_alone_no_longer_triggers_fallback(self):
        # Regression: "[Context: ...]" alone used to match _ADMIN_LEAK_RE and
        # blackhole a fully truthful reply. It is cosmetic, not sensitive, and
        # is handled separately by _strip_context_breadcrumb.
        reply = (
            "[Context: QuickBite Express — Restaurant Knowledge Base] "
            "QuickBite Express is located in Chennai, South India."
        )
        assert ordering_agent._strip_admin_leak(reply) == reply

    def test_gate_withholds_sentence_with_admin_leak(self):
        gate = ordering_agent._SentenceGate(message="hi", emit=None)
        assert gate._is_safe(
            "[Context: QuickBite Express] FSSAI License: 123", ["knowledge_lookup"]
        ) is False


class TestStripLeakedDirectives:
    """System-prompt instructions echoed verbatim must never reach TTS."""

    def test_real_mcp_rejection_message_pronoun_form_is_stripped(self):
        # Regression: observed live. mcp_server._rejection_payload's own
        # unavailable-item template reads "...Tell them those are unavailable
        # and offer these real alternatives instead: ...", a pronoun
        # back-reference to "the customer" rather than the literal word
        # "customer". The model echoed this verbatim over TTS because the
        # original _LEAKED_DIRECTIVE_RE only matched openers followed by the
        # literal word "customer"/"the customer", never "them".
        reply = (
            "\"burji\", \"kathi_roll\" are not on the menu. Do not invent "
            "them and do not ask the customer to try again. Tell them those "
            "are unavailable and offer these real alternatives instead: "
            "Chicken Tikka Kathi Roll (149), Paneer Bhurji Kathi Roll (129)."
        )
        cleaned = ordering_agent._strip_leaked_directives(reply)
        assert "Tell them" not in cleaned
        assert "Chicken Tikka Kathi Roll (149)" not in cleaned

    def test_classic_tell_the_customer_form_still_stripped(self):
        reply = "Tell the customer their cart is already empty."
        assert ordering_agent._strip_leaked_directives(reply) == reply  # no trailing text, falls back unchanged
        # With trailing customer-facing text present, only the directive is removed.
        reply2 = "Sure! Tell the customer their cart is already empty. Anything else?"
        cleaned2 = ordering_agent._strip_leaked_directives(reply2)
        assert "Tell the customer" not in cleaned2
        assert "Anything else?" in cleaned2

    def test_ordinary_reply_addressing_customer_as_you_is_untouched(self):
        reply = "I've added your item. Would you like anything else?"
        assert ordering_agent._strip_leaked_directives(reply) == reply

    def test_ordinary_reply_starting_with_offer_or_ask_word_is_untouched(self):
        # "Offer"/"Ask" as ordinary words (not directive verbs targeting
        # "customer"/"them") must not be stripped.
        reply = "Offer valid until midnight, would you like to add fries?"
        assert ordering_agent._strip_leaked_directives(reply) == reply
        reply2 = "Ask away, I'm happy to help with the menu."
        assert ordering_agent._strip_leaked_directives(reply2) == reply2


class TestIsSingleAddUtterance:
    """_is_single_add_utterance must be conservative — only True for provably single adds."""


    @pytest.mark.parametrize("utt", [
        "I want a classic chicken burger.",
        "Add a burger please.",
        "Order me a coffee.",
        "I'd like the veg burger.",
        "Get me one pizza.",
        "I'll have a coke.",
        "Can I get the fries?",
    ])
    def test_single_add_returns_true(self, utt):
        from plugins.kiosk.ordering_agent import _is_single_add_utterance
        assert _is_single_add_utterance(utt) is True, f"Expected True for: {utt!r}"

    @pytest.mark.parametrize("utt", [
        "Remove the burger and add a pizza.",
        "Take off the fries and get me a coke.",
        "Swap the burger for a pizza.",
        "Add a burger and a coke.",
        "I want a burger and also fries.",
        "Add a pizza plus a coke.",
        "I don't want the fries, give me a burger.",
        "Cancel the burger and order a pizza instead.",
        "Remove chicken and add paneer.",
        "No more fries, I want a burger.",
    ])
    def test_compound_returns_false(self, utt):
        from plugins.kiosk.ordering_agent import _is_single_add_utterance
        assert _is_single_add_utterance(utt) is False, f"Expected False for: {utt!r}"


class TestIsOutOfScopeSittingSynonyms:
    """"Sitting capacity"/"sitting option" were missing from the in-domain
    keyword set, so a bare in-domain question about seating was wrongly
    refused as world knowledge. Only "seat"/"seating" matched before this fix.
    """

    @pytest.mark.parametrize("message", [
        "What is the sitting capacity?",
        "Tell me about the sitting option.",
        "How many people can sit here?",
        "Do you have outdoor seats?",
    ])
    def test_seating_related_questions_are_in_scope(self, message):
        assert ordering_agent._is_out_of_scope(message) is False


class TestExtractDietaryPref:
    """Deterministic veg/non-veg preference extraction.

    Regression coverage for the bug where naming BOTH veg and non-veg in the
    same ad-hoc request (e.g. "veg option and non-veg options") collapsed to
    ``non_vegetarian`` only, because ``_VEG_ADHOC_PATTERN`` also matches the
    substring "veg" inside "non-veg" and the non-veg branch was checked (and
    returned) first.
    """

    @pytest.mark.parametrize("message", [
        "I'm vegetarian",
        "I am vegetarian",
    ])
    def test_stated_vegetarian_capability(self, message):
        assert ordering_agent._extract_dietary_pref(message) == "vegetarian"

    def test_stated_vegan_capability(self):
        assert ordering_agent._extract_dietary_pref("I'm vegan") == "vegan"

    @pytest.mark.parametrize("message", [
        "I eat meat",
        "I'm not vegetarian",
    ])
    def test_non_veg_capability_clears_restriction(self, message):
        assert ordering_agent._extract_dietary_pref(message) == "none"

    @pytest.mark.parametrize("message", [
        "suggest me veg dishes",
        "show me vegetarian options",
        "any veg suggestions?",
    ])
    def test_adhoc_veg_only_filter(self, message):
        assert ordering_agent._extract_dietary_pref(message) == "vegetarian"

    @pytest.mark.parametrize("message", [
        "suggest me non veg dishes",
        "show me non-veg options",
        "any nonveg suggestions?",
    ])
    def test_adhoc_non_veg_only_filter(self, message):
        assert ordering_agent._extract_dietary_pref(message) == "non_vegetarian"

    @pytest.mark.parametrize("message", [
        "veg option and non-veg options",
        "show me veg and non veg dishes",
        "suggest veg items and non-veg items too",
    ])
    def test_adhoc_both_veg_and_non_veg_named_shows_full_catalogue(self, message):
        # Naming both is not a valid single positive filter - it must not
        # silently collapse to non_vegetarian and hide every veg item.
        assert ordering_agent._extract_dietary_pref(message) == "none"

    def test_plain_order_by_name_is_not_a_lasting_preference(self):
        # "I'll have the veg burger" resolves via direct-order name matching
        # and must not be misread as a dietary-filter statement.
        assert ordering_agent._extract_dietary_pref("I'll have the veg burger.") is None

    def test_no_dietary_mention_returns_none(self):
        assert ordering_agent._extract_dietary_pref("What are the burger options?") is None

    def test_empty_message_returns_none(self):
        assert ordering_agent._extract_dietary_pref("") is None
