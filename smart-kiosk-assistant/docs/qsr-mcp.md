# Smart Kiosk — QSR MCP Surface

Exposes kiosk-core to the Central QSR Agent (retail-use-cases#100/#103,
`qsr-design`) directly via `fastmcp` — the same MCP framework already used
by `kiosk_core/ordering/mcp_server.py`'s internal `/mcp` mount. This is a
**separate** MCP surface from that ordering mount, which is only for the
internal rag-service ordering agent.

Previously this surface was built on the external `mcp-service-sdk`
package (as Order Accuracy's dine-in/take-away MCP services still are).
It has since been migrated to a direct `fastmcp` implementation that owns
its own durable event log, delivery fan-out, and action policy gate
in-tree (see Module layout), removing the `mcp-service-sdk` dependency
entirely. Tool names, descriptions, input schemas, event types, and the
durable SQLite/JSONL log's on-disk format are unchanged by this migration.

## Module layout

| File | Responsibility |
|---|---|
| `kiosk_core/qsr_mcp/mcp_service.py` | `fastmcp.FastMCP` app, event types, read/act tool implementations |
| `kiosk_core/qsr_mcp/event_log.py` | Durable event log (SQLite/JSONL), same on-disk format as the former `mcp_service_sdk` log |
| `kiosk_core/qsr_mcp/delivery.py` | Multi-sink event delivery with retry + ordered background dispatch (off by default) |
| `kiosk_core/qsr_mcp/autonomy.py` | Outbound **event autonomy** push to the Central QSR Agent's `/autonomy/events` inbox (off by default) |
| `kiosk_core/qsr_mcp/policy.py` | Action policy gate (`GateLevel`, rate limiting) for `set_board_mode` |
| `kiosk_core/qsr_mcp/mcp_server.py` | Runs the MCP app + queue-depth sampling poller on background threads |
| `kiosk_core/qsr_mcp/board_state.py` | Durable (JSON-file) kiosk menu-board mode: `full` / `simplified` |
| `kiosk_core/queue_client.py` | HTTP client to the standalone `queue-service` |

Started from `main.py`'s lifespan when `KIOSK_CORE_QSR_MCP_ENABLED=true`
(default). Listens on `KIOSK_CORE_QSR_MCP_PORT` (default `8014`,
streamable-http transport) — distinct from the ordering feature's `/mcp`.

## Contract (`describe`)

**Event types**
- `queue_depth_sample` — `{count, nearby, status}`, emitted every
  `KIOSK_CORE_QSR_MCP_QUEUE_POLL_SECONDS` (default 30s) by polling
  queue-service.
- `board_mode_changed` — `{mode, reason}`, emitted whenever `set_board_mode`
  runs.

**Read tools**
- `get_queue_depth()` — current count/status from queue-service.
- `get_queue_history(period)` — sample count/avg/max for
  `today|yesterday|all|YYYY-MM-DD`.
- `get_order_stats(period, top_n)` — confirmed order count, revenue, top
  items, from the existing ordering SQLite DB.
- `get_board_state()` — current board mode and when/why it last changed.

**Act tool** (gate: `AUTOMATIC`, rate-limited to 20 calls/60s)
- `set_board_mode(mode, reason=None)` — `mode` is `"full"` or `"simplified"`.
  Implements qsr-design's autonomous action #6: "Kiosk queue deep/short ->
  Kiosk board simplify". The actual MCP wire response wraps the result:
  `{"executed": true, "level": "automatic", "result": {...}}` on success, or
  `{"executed": false, "level": "automatic", "reason": "rate limit exceeded"}`
  when the 20-calls/60s limit is hit.

Also exposed (unchanged by the migration): `describe()` (self-description)
and `subscribe(event_type, condition, callback_url)` (records a
subscription; Kiosk — unlike Order Accuracy's sensor-only services — has no
restriction against exposing `subscribe`).

## Event autonomy (push to the Central QSR Agent)

The tools above are **pull**: the agent calls the kiosk. Event autonomy adds
the **push** direction, so the kiosk reports observations without being
polled. It mirrors the pattern in order-accuracy#214.

The agent's `autonomy` process has no source polling loop — per its
`SERVICE_OWNER_GUIDE`, **the event producer owns change detection and
delivery**. The kiosk's queue poller samples on a fixed interval, so
`AutonomyEventSink` suppresses unchanged samples and only forwards a
`queue_count_changed` when the count moves by at least
`KIOSK_CORE_QSR_AUTONOMY_QUEUE_MIN_DELTA`.

**Flow** — `emit()` writes to the durable log first (authoritative), then
hands the event to a background FIFO worker per destination. Delivery is
best-effort and never blocks the queue poller or a `set_board_mode` call.
A single ordered worker (rather than a thread per event) ensures a retrying
older sample cannot arrive *after* a newer one and leave the agent acting on
a stale queue depth.

**Envelope** posted to `POST {url}` (the agent replies `202 Accepted`):

```json
{
  "event_id": "smart_kiosk-queue_count_changed-<ref_id>",
  "event_type": "queue_count_changed",
  "store_id": "qsr-001",
  "occurred_at": "2025-01-01T12:00:00Z",
  "data": {"queue_count": 11, "source": "smart_kiosk", "source_event_type": "queue_depth_sample", "...": "original payload"}
}
```

`event_id` is stable across retries so the agent can de-duplicate. Internal
event types map to the agent's vocabulary: `queue_depth_sample` →
`queue_count_changed`; `board_mode_changed` passes through unchanged (the
agent's event policy is a wildcard, so unregistered names are safe and
simply yield a no-action explanation).

A `4xx` other than `429` is treated as a permanent rejection and stops
retries — resending a payload the agent refuses to parse cannot succeed.

| Variable | Default | Effect |
|---|---|---|
| `KIOSK_CORE_QSR_AUTONOMY_EVENTS_URL` | *(empty)* | Agent inbox URL, e.g. `http://<agent-host>:8600/autonomy/events`. **Empty disables the whole feature** — the default stack makes no outbound calls. |
| `KIOSK_CORE_QSR_AUTONOMY_EVENTS_TOKEN` | *(empty)* | Optional bearer token |
| `KIOSK_CORE_QSR_AUTONOMY_STORE_ID` | `qsr-001` | Must equal the agent's `QSR_RESTAURANT_ID` or events are rejected. Deliberately separate from `KIOSK_CORE_QSR_MCP_STORE_ID`, which only labels our own durable log. |
| `KIOSK_CORE_QSR_AUTONOMY_EVENT_TYPES` | `queue_depth_sample,board_mode_changed` | Which internal events to forward |
| `KIOSK_CORE_QSR_AUTONOMY_QUEUE_MIN_DELTA` | `1` | Minimum queue-count change before forwarding |
| `KIOSK_CORE_QSR_AUTONOMY_TIMEOUT_SECONDS` | `5.0` | Per-request timeout |
| `KIOSK_CORE_QSR_EXTRA_NO_PROXY` | *(empty)* | Appended to kiosk-core's `NO_PROXY`. **Required behind a corporate proxy**, since the agent runs on another machine. Python's `urllib` matches `no_proxy` by suffix, *not* CIDR — list the host name or exact IP. |

## Configuration

See `kiosk_core/config.py` (`QSR_MCP_*` and `QSR_AUTONOMY_*` constants) /
`.env.example` for all `KIOSK_CORE_QSR_*` environment variables.

## Notes / follow-ups

- Tool names/schemas were derived from `qsr-design` and the Order Accuracy
  reference implementation, since retail-use-cases#100/#103 were not
  reachable from this environment (org SAML SSO). Revisit against the ticket
  text once accessible.
- `set_board_mode` only flips a backend flag today; wiring kiosk-ui to
  visibly react to `board_mode` is a follow-up.
- An existing, populated `qsr_mcp_events.db` (or JSONL log file) from before
  the `fastmcp` migration keeps working unmodified: the SQLite table/columns
  and JSONL record shape in `event_log.py` are byte-for-byte identical to the
  former `mcp_service_sdk`-backed log.
