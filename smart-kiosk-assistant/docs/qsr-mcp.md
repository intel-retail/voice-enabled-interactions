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
| `kiosk_core/qsr_mcp/menu.py` | Menu actuation — add/update catalogue items (sync `sqlite3`) |
| `kiosk_core/qsr_mcp/menu_requests.py` | Durable queue of menu changes awaiting operator approval |
| `kiosk_core/qsr_mcp/api.py` | Operator-only REST endpoints to approve/reject those requests |
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
- `menu_item_added` — `{product_id, name, category, price, reason}`.
- `menu_item_updated` — `{product_id, changed, reason}`, where `changed` is a
  `{field: {from, to}}` diff.
- `menu_change_requested` — `{request_id, action, payload, reason}`, emitted
  when a gated menu change is parked for operator approval.

**Read tools**
- `get_queue_depth()` — current count/status from queue-service.
- `get_queue_history(period)` — sample count/avg/max for
  `today|yesterday|all|YYYY-MM-DD`.
- `get_order_stats(period, top_n)` — confirmed order count, revenue, top
  items, from the existing ordering SQLite DB.
- `get_board_state()` — current board mode and when/why it last changed.
- `get_menu_requests(status)` — menu changes awaiting (or past) operator
  approval; `pending` (default), `approved`, `rejected` or `all`.

**Act tools**

`set_board_mode` (gate: `AUTOMATIC`, rate-limited to 20 calls/60s)
- `set_board_mode(mode, reason=None)` — `mode` is `"full"` or `"simplified"`.
  Implements qsr-design's autonomous action #6: "Kiosk queue deep/short ->
  Kiosk board simplify". The actual MCP wire response wraps the result:
  `{"executed": true, "level": "automatic", "result": {...}}` on success, or
  `{"executed": false, "level": "automatic", "reason": "rate limit exceeded"}`
  when the 20-calls/60s limit is hit.

### Menu actuation

- `add_menu_item(product_id, name, category, price, is_veg=True, is_bestseller=False, reason=None)`
  — gate `NEEDS_APPROVAL` by default. Deliberately **not** an upsert: silently
  overwriting an existing product's price under an "add" call is how a
  catalogue gets corrupted, so a duplicate `product_id` is rejected.
- `update_menu_item(product_id, price=None, is_available=None, name=None, category=None, is_bestseller=None, is_veg=None, reason=None)`
  — gate `AUTOMATIC`. Sparse: omitted fields are left unchanged. Set
  `is_available=false` to take an item off the menu ("86" it).
  `product_id` is immutable — `order_items` rows reference it, so renaming one
  would orphan historical orders.

Both emit events (`menu_item_added` / `menu_item_updated`) and return the
usual `{"executed", "level", "result"}` envelope. A caller-fixable rejection
(duplicate id, bad price, unknown product) comes back as
`{"executed": false, "level", "error"}` rather than a transport error, so the
agent can correct itself instead of retrying an invalid call.

**Unavailable items are enforced, not just hidden.** Setting
`is_available=false` removes the item from `list_products` /
`list_bestsellers`, makes `resolve_product` report `NOT_FOUND` (so the agent
offers an alternative rather than promising an item it cannot sell), *and*
causes `place_order` / `update_order_items` to raise. The order-time check is
the chokepoint that matters: an id can still arrive from a stale cart, a
replayed turn or a direct API call.

**Approval queue.** `PolicyGate` default-denies `NEEDS_APPROVAL` because no
human-approval callback is wired. For `add_menu_item` that alone would make
the tool a dead end, so the request is instead recorded durably and returned
as `{"executed": false, "level": "needs_approval", "request_id": ..., "status": "pending_approval"}`.
An operator actions it over REST — never over MCP, since an agent that could
approve its own request would make the gate meaningless:

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/qsr/menu/requests?status=pending` | List requests |
| POST | `/api/v1/qsr/menu/requests/{id}/approve` | Approve and execute |
| POST | `/api/v1/qsr/menu/requests/{id}/reject` | Reject without executing |

Approval is single-use: the request is marked resolved *before* the write, so
two concurrent operators cannot both execute it. If the write then fails the
request is released as rejected with the error, rather than silently
consuming the approval.

**Persistence vs. `products.yaml`.** The YAML is the *default* catalogue and
is re-applied on every startup. Any row written by these tools is marked
`source='agent'`, and `SqliteProductRepository.seed_upsert` skips those — so
an agent's price change or an 86'd item is not silently reverted by the next
restart, while editing the YAML remains the normal operator workflow for
every item the agent has not taken over.

| Variable | Default | Effect |
|---|---|---|
| `KIOSK_CORE_QSR_MENU_ADD_GATE` | `needs_approval` | Gate for `add_menu_item`. Set `automatic` to let the agent add items outright. |
| `KIOSK_CORE_QSR_MENU_UPDATE_GATE` | `automatic` | Gate for `update_menu_item` |
| `KIOSK_CORE_QSR_MENU_MAX_CALLS` | `20` | Rate limit (calls per window) |
| `KIOSK_CORE_QSR_MENU_PER_SECONDS` | `60.0` | Rate-limit window |
| `KIOSK_CORE_QSR_MENU_REQUESTS_PATH` | `./qsr_menu_requests.json` | Approval queue file |

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
