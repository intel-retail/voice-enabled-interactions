"""Agentic capability library for the RAG service.

This package provides a Google ADK-based ordering agent that integrates with the
kiosk-core MCP server for tool calls (place_order, update_order, confirm_order,
get_upsell_suggestions, list_products) and with the existing RAG pipeline for
knowledge lookups.

Public interface:
    - ``get_ordering_agent()``  → ``OrderingAgent`` singleton
    - ``agent_chat()``          → run one conversational turn
"""

__all__: list[str] = []
