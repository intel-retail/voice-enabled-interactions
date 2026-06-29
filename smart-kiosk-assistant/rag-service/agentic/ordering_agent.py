"""OrderingAgent — Google ADK LlmAgent for the kiosk ordering flow.

The agent:
  1. Answers menu/FAQ questions via the ``knowledge_lookup`` tool (RAG pipeline).
  2. Places, updates, gets, and confirms orders via MCP tools on kiosk-core.
  3. Proactively surfaces upsell suggestions after items are added.
  4. Confirms orders with a friendly message and the Order ID.

Usage::

    agent = OrderingAgent()
    await agent.bootstrap()
    reply = await agent.chat(
        session_id="user-session-1",
        user_id="user123",
        message="I'd like a Paneer Tikka Burger please",
    )
"""

from __future__ import annotations

import logging
from typing import Any

from agentic import config as agent_cfg
from agentic.adk_runtime import create_adk_model, create_runner, create_session_service
from agentic.mcp_client import MCPTool, bootstrap_mcp_tools, call_tool, get_all_tools
from agentic.tools.knowledge_lookup_tool import knowledge_lookup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent instruction prompt
# ---------------------------------------------------------------------------

_AGENT_INSTRUCTION = """
You are the AI ordering assistant for QuickBite Express, a QSR kiosk.
Your job is to help customers discover the menu and place their orders conversationally.

## Tools available to you
- **knowledge_lookup(question)** — answer any question about menu items, prices,
  ingredients, dietary tags, opening hours, or outlet policies.  Always use this
  before saying "I don't know".
- **list_products(category?)** — list all available products, optionally filtered
  by category (burgers, pizza, wraps, sides, beverages, desserts).
- **place_order(user_id, items)** — create a new draft order.  items is a list of
  {product_id, quantity} pairs.  Returns the created order with an order_id.
- **update_order(order_id, items)** — add or increment items on an existing draft order.
- **get_order(order_id)** — show the current order summary (items, quantities, total).
- **confirm_order(order_id)** — finalise the order.  Returns the confirmed Order ID.
- **get_upsell_suggestions(product_ids)** — get complementary product suggestions for
  the items currently in the cart.

## Guidelines
1. When a customer asks about a menu item or outlet info, use **knowledge_lookup**.
2. When a customer wants to order, call **list_products** first if you need the
   exact product_id, then call **place_order**.
3. After placing or updating an order, always call **get_upsell_suggestions** with
   the cart's product_ids and mention the top suggestion naturally.
4. When the customer confirms ("yes", "place it", "looks good"), call **confirm_order**
   and end with: "Your order is confirmed! 🎉 Your Order ID is ORD-XXXXX."
5. Keep responses concise and friendly.  This is a voice kiosk — avoid bullet lists.
6. If the customer asks for their order summary, call **get_order**.
7. Always use the user_id passed to you (default: "anonymous") when placing orders.
""".strip()


# ---------------------------------------------------------------------------
# OrderingAgent
# ---------------------------------------------------------------------------


class OrderingAgent:
    """Wraps the ADK LlmAgent for the ordering flow.

    Call ``await bootstrap()`` once before using ``chat()``.
    """

    def __init__(self) -> None:
        self._agent = None
        self._runner = None
        self._session_service = None
        self._bootstrapped = False

    async def bootstrap(self) -> None:
        """Initialise the ADK model, MCP tools, and runner.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._bootstrapped:
            return

        logger.info("[AGENT] Bootstrapping OrderingAgent …")

        # 1. Discover MCP tools from kiosk-core
        mcp_tools = await bootstrap_mcp_tools(agent_cfg.MCP_CONFIG_PATH)
        logger.info("[AGENT] MCP tools: %s", list(mcp_tools))

        # 2. Build ADK FunctionTools
        from google.adk.agents import LlmAgent
        from google.adk.tools import FunctionTool

        # knowledge_lookup is a native Python async function
        adk_tools = [FunctionTool(knowledge_lookup)]

        # Wrap each MCP tool as an async Python callable → FunctionTool
        for tool_name, mcp_tool in mcp_tools.items():
            adk_tools.append(FunctionTool(self._make_mcp_callable(tool_name, mcp_tool)))

        # 3. Create ADK agent
        model = create_adk_model()
        self._agent = LlmAgent(
            name="kiosk_ordering_agent",
            model=model,
            description="Kiosk ordering assistant — handles menu Q&A and order management",
            instruction=_AGENT_INSTRUCTION,
            tools=adk_tools,
        )

        # 4. Runner + session service
        self._session_service = create_session_service()
        self._runner = create_runner(self._agent, self._session_service)

        self._bootstrapped = True
        logger.info("[AGENT] OrderingAgent ready ✓")

    @staticmethod
    def _make_mcp_callable(tool_name: str, mcp_tool: MCPTool):
        """Dynamically create an async function wrapping an MCP tool call."""

        async def _mcp_fn(**kwargs: Any) -> Any:
            logger.info("[AGENT→MCP] tool=%s args=%s", tool_name, kwargs)
            result = await call_tool(tool_name, kwargs)
            logger.debug("[AGENT→MCP] tool=%s result=%s", tool_name, result)
            return result

        _mcp_fn.__name__ = tool_name
        _mcp_fn.__doc__ = mcp_tool.description
        # Attach JSON schema as __annotations__ hint for ADK
        _mcp_fn.__schema__ = mcp_tool.to_function_schema()  # type: ignore[attr-defined]
        return _mcp_fn

    async def chat(
        self,
        message: str,
        session_id: str,
        user_id: str = "anonymous",
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Run one conversational turn and return the agent's response.

        Args:
            message:    The user's transcribed utterance.
            session_id: Opaque session identifier (maps to ADK session).
            user_id:    The customer's user identifier.
            history:    Previous turns [{role, content}, …] — used to seed
                        the ADK session when it does not yet exist (e.g.
                        after a rag-service restart).

        Returns:
            dict with keys:
              - ``reply``:     str — the agent's text response.
              - ``tool_calls``: list[str] — tools invoked this turn.
        """
        if not self._bootstrapped:
            await self.bootstrap()

        logger.info("[AGENT] chat session=%s user=%s message=%r", session_id, user_id, message[:120])

        from google.genai import types as genai_types

        # Seed the ADK session with prior history if the session does not
        # yet exist (rag-service restart scenario).
        await self._ensure_session(user_id, session_id, history)

        # Prefix the user_id into the first turn so the LLM (and ordering
        # tools) know which customer is speaking without needing a dedicated
        # user-lookup tool.
        full_message = message
        if user_id != "anonymous":
            full_message = f"[user_id={user_id}] {message}"

        content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=full_message)],
        )

        reply_parts: list[str] = []
        tool_calls: list[str] = []

        try:
            # Use run_async — run() is documented as "local testing only"
            # and blocks the event loop thread via queue.Queue().get().
            async for event in self._runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content,
            ):
                if hasattr(event, "tool_call") and event.tool_call:
                    tool_calls.append(event.tool_call.name)
                    logger.info("[AGENT] Tool invoked: %s", event.tool_call.name)
                if hasattr(event, "content") and event.content:
                    for part in getattr(event.content, "parts", []):
                        if hasattr(part, "text") and part.text:
                            reply_parts.append(part.text)
        except Exception as exc:
            logger.error("[AGENT] Error during run: %s", exc, exc_info=True)
            return {"reply": "Sorry, I encountered an error. Please try again.", "tool_calls": []}

        reply = "".join(reply_parts).strip()
        logger.info("[AGENT] Reply length=%d tool_calls=%s", len(reply), tool_calls)
        return {"reply": reply, "tool_calls": tool_calls}

    async def _ensure_session(
        self,
        user_id: str,
        session_id: str,
        history: list[dict[str, str]] | None,
    ) -> None:
        """Create the ADK session and optionally seed it with prior history.

        If the session already exists (normal multi-turn case) this is a
        no-op.  If it does not exist (first turn, or after a rag-service
        restart) we create it and replay any history provided by the caller
        so the agent retains conversation context.
        """
        from google.genai import types as genai_types

        try:
            existing = await self._session_service.get_session(
                app_name=self._agent.name,
                user_id=user_id,
                session_id=session_id,
            )
            if existing is not None:
                return
        except Exception:
            pass  # session service may raise if session not found

        # Build initial events from history so the agent has context
        initial_events: list[genai_types.Content] = []
        for turn in (history or []):
            role = turn.get("role", "")
            text = turn.get("content", "")
            if role in ("user", "assistant") and text:
                adk_role = "model" if role == "assistant" else "user"
                initial_events.append(
                    genai_types.Content(
                        role=adk_role,
                        parts=[genai_types.Part(text=text)],
                    )
                )

        await self._session_service.create_session(
            app_name=self._agent.name,
            user_id=user_id,
            session_id=session_id,
            state={"history_seeded": bool(initial_events)},
        )
        logger.debug(
            "[AGENT] Created session user=%s session=%s history_turns=%d",
            user_id, session_id, len(initial_events),
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_agent_instance: OrderingAgent | None = None


def get_ordering_agent() -> OrderingAgent:
    """Return the module-level OrderingAgent singleton (created lazily)."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = OrderingAgent()
    return _agent_instance
