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
- **knowledge_lookup(question)** — answers questions about ingredients, allergens,
  dietary tags, opening hours, or outlet policies.  Use ONLY for information questions.
- **list_products(category?)** — lists available products with their product_id and
  price.  Always pass category when the customer mentions a food type.
  Valid categories: burgers, pizza, wraps, sides, beverages, desserts.
- **place_order(user_id, items)** — creates a new draft order.  items is a list of
  {product_id, quantity} pairs.  Returns order_id.
- **update_order(order_id, items)** — adds or updates items on an existing draft order.
- **get_order(order_id)** — shows the current order summary (items, quantities, total).
- **confirm_order(order_id)** — finalises the order.  Returns the confirmed Order ID.
- **get_upsell_suggestions(product_ids)** — gets complementary product suggestions
  for the items currently in the cart.

## Decision rules — follow strictly in order

### Rule 1 — Customer wants to ORDER something (e.g. "I want X", "give me X", "a X please", "order for X")
1. Identify the category: burger→"burgers", pizza→"pizza", wrap→"wraps",
   drink/beverage→"beverages", side→"sides", dessert→"desserts".
2. Call **list_products(category=<category>)** — NOT list_products() without category.
3. From the results, find the closest matching product by name.
4. Call **place_order** or **update_order** with that product_id and quantity=1.
5. Call **get_upsell_suggestions** with the cart product_ids.
6. Reply: state the product name and price, mention the upsell, ask to confirm.
   Example: "Great! I've added a Classic Chicken Burger (₹169) to your order.
   Would you also like fries to go with it? Say 'confirm' to place your order."

**NEVER call knowledge_lookup for ordering requests — go straight to list_products.**
**ALWAYS pass category to list_products when the food type is known.**

### Rule 2 — Customer asks an information question (ingredients, "is X vegan?", allergens, hours)
1. Call **knowledge_lookup** to answer.

### Rule 3 — Customer asks to browse menu / "what do you have?" / "show me burgers"
1. Call **list_products(category=<category if mentioned>)**.
2. Reply with the product names and prices in a short conversational list.
   Example: "We have 7 burgers: Classic Chicken Burger ₹169, Spicy Crunch ₹179, ..."

### Rule 4 — Order management
- "show my order" / "what did I order?" → call **get_order**
- "confirm" / "place it" / "that's all" / "yes" → call **confirm_order**, reply:
  "Your order is confirmed! Your Order ID is ORD-XXXXX. Enjoy your meal! 🎉"

## Response style
- Voice kiosk — keep replies short (2-3 sentences max), friendly, conversational.
- Avoid bullet lists in responses — speak in natural sentences.
- Always use the user_id passed to you (default: "anonymous").
- When a product name is unclear (ASR mis-transcription), match the closest product
  from list_products results and confirm: "Did you mean a Crispy Veg Patty Burger?"
- Always state the product name and price when adding to order.

/no_think
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
        MCP tool discovery may silently return 0 tools if kiosk-core is not
        yet ready; ``chat()`` will trigger re-discovery automatically.
        """
        if self._bootstrapped:
            return

        logger.info("[AGENT] Bootstrapping OrderingAgent …")

        # 1. Discover MCP tools from kiosk-core (best-effort at startup;
        #    will be retried on first chat() call if kiosk-core isn't ready yet)
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

    async def _refresh_mcp_tools(self) -> None:
        """Re-discover MCP tools and rebuild the agent if tools are missing.

        Called automatically from ``chat()`` when no MCP tools are registered —
        this recovers from the startup race where rag-service starts before
        kiosk-core, as well as from kiosk-core restarts mid-session.
        """
        if get_all_tools():
            return  # already have tools, nothing to do

        logger.info("[AGENT] No MCP tools registered — retrying discovery from kiosk-core …")
        mcp_tools = await bootstrap_mcp_tools(agent_cfg.MCP_CONFIG_PATH)
        if not mcp_tools:
            logger.warning("[AGENT] MCP re-discovery returned 0 tools — kiosk-core may still be starting")
            return

        logger.info("[AGENT] MCP re-discovery succeeded: %s", list(mcp_tools))

        # Rebuild the agent with the newly discovered tools
        from google.adk.agents import LlmAgent
        from google.adk.tools import FunctionTool

        adk_tools = [FunctionTool(knowledge_lookup)]
        for tool_name, mcp_tool in mcp_tools.items():
            adk_tools.append(FunctionTool(self._make_mcp_callable(tool_name, mcp_tool)))

        model = create_adk_model()
        self._agent = LlmAgent(
            name="kiosk_ordering_agent",
            model=model,
            description="Kiosk ordering assistant — handles menu Q&A and order management",
            instruction=_AGENT_INSTRUCTION,
            tools=adk_tools,
        )
        self._session_service = create_session_service()
        self._runner = create_runner(self._agent, self._session_service)
        logger.info("[AGENT] Agent rebuilt with %d MCP tool(s) ✓", len(mcp_tools))

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

        # If MCP tools weren't available at startup (race with kiosk-core),
        # attempt re-discovery before this turn so ordering tools work.
        await self._refresh_mcp_tools()

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

        # Qwen3 models output chain-of-thought thinking before the final response.
        # Strip everything before the last double-newline that separates thinking
        # from the actual reply, so TTS and the UI get clean text only.
        if "\n\n" in reply:
            # Find last paragraph that looks like the final answer
            # (thinking blocks tend to be long reasoning chains ending with a blank line)
            parts = [p.strip() for p in reply.split("\n\n") if p.strip()]
            # Keep the last block if it looks like a user-facing answer (shorter, no "I need to")
            if len(parts) > 1:
                last_part = parts[-1]
                if not last_part.lower().startswith(("okay,", "alright,", "the user", "i need to", "i should")):
                    reply = last_part
                else:
                    # All looks like thinking — return as-is but log a warning
                    logger.warning("[AGENT] Reply may contain unstripped thinking (%d chars)", len(reply))

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
