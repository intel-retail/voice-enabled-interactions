"""QSR MCP surface — exposes kiosk-core to the Central QSR Agent.

See ``kiosk_core/qsr_mcp/mcp_service.py`` for the ``fastmcp``-based
``ServiceServer``-equivalent wiring (read/act tools, event types, durable
log/delivery/policy gate) and ``kiosk_core/qsr_mcp/mcp_server.py`` for the
run/thread wiring, following the same module split used by
``order-accuracy/dine-in``'s ``mcp_service.py`` / ``mcp_server.py``.
"""
