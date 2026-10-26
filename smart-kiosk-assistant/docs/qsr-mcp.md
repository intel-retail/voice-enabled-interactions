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
| `kiosk_core/qsr_mcp/atomic_file.py` | Crash-safe JSON writes (temp file + `fsync` + `os.replace`) shared by `board_state` / `menu_requests` |
| `kiosk_core/queue_client.py` | HTTP client to the standalone `queue-service` |

**Initialisation is explicit.** Importing `mcp_service` is side-effect free:
`mcp_service.init()` is what opens the durable log, builds the delivery sinks
and the queue-service client, points `board_state` / `menu_requests` at their
configured paths, and registers the policy gates. `main.py` calls it during
startup (and `run_mcp_server` / `start_queue_poller` call it too). It is
idempotent, and takes `force=True` to rebuild — which closes the previous log
and stops the previous dispatcher threads rather than leaking them. Every
runtime entry point also self-initialises, so a direct tool call still works
standalone.

`init()` re-reads per-call configuration, but a handful of values are still
bound at import time and a rebuild will **not** pick them up:
`MCP_SERVICE_ENABLED`, `STORE_ID`, and `MCP_HOST` / `MCP_PORT` /
`MCP_TRANSPORT`. Changing any of those needs a process restart.

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
  `today|yesterday|all|YYYY-MM-DD`. On the SQLite backend this is a single
  `COUNT`/`AVG`/`MAX` over `json_extract(envelope, '$.payload.count')`, so
  `all` stays O(1) in memory however long the log grows; the JSONL backend
  falls back to reducing rows in Python.
- `get_order_stats(period, top_n)` — confirmed order count, revenue, top
  items, from the existing ordering SQLite DB.
- `get_board_state()` — current board mode and when/why it last changed.
- `get_menu_requests(status)` — menu changes awaiting (or past) operator
  approval; `pending` (default), `executing`, `approved`, `rejected` or
  `all`. The response also carries an `incomplete` list of requests stuck in
  `executing` from an interrupted run.

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
| POST | `/api/v1/qsr/menu/items/{id}/release` | Hand an agent-claimed product back to `products.yaml` |

A request moves `pending -> executing -> approved | rejected`:

| Status | Meaning |
|---|---|
| `pending` | Awaiting an operator |
| `executing` | Claimed, write in progress. Seeing this on a freshly started process means an execution was interrupted — the list endpoint reports those separately as `incomplete`, because whether the write landed is unknown and must not be retried blindly |
| `approved` | Executed. Always carries `result` and `executed_at` |
| `rejected` | Refused by an operator, or failed permanently |

Approval is single-use: the claim happens *before* the write, so two
concurrent operators cannot both execute it. `executing` is deliberately a
distinct state from `approved` — collapsing them would make a crash between
the claim and the write indistinguishable from a completed change, leaving
an `approved` request with no result that could never be retried.

The two failure modes are then kept apart. A caller-fixable rejection
(`MenuError` — duplicate id, bad price, unknown product) marks the request
**rejected** with the error, since re-approving it unchanged would fail
identically. Any other failure (a locked database, a disk error) **releases
the request back to `pending`**, so a transient fault really is retryable
rather than consuming the approval.

**Persistence vs. `products.yaml`.** The YAML is the *default* catalogue and
is re-applied on every startup. A row is marked `source='agent'` — which
`SqliteProductRepository.seed_upsert` then skips — only when the agent changes
a field the seeder would otherwise re-apply (`name`, `category`, `price`,
`is_bestseller`, `is_veg`). So an agent's price change is not silently
reverted by the next restart, while editing the YAML remains the normal
operator workflow for every item the agent has not taken over.

An **availability-only** change deliberately does *not* claim the row: the
seeder never writes `is_available`, so an 86'd item stays 86'd across a
restart without permanently detaching it from YAML control. To hand a claimed
row back, `POST /api/v1/qsr/menu/items/{id}/release` — without it, a single
price change would make that item ignore `products.yaml` forever. The release
is operator-only, like approve/reject: letting the agent release its own
claim would allow a restart to undo a change it had just made.

| Variable | Default | Effect |
|---|---|---|
| `KIOSK_CORE_QSR_MENU_ADD_GATE` | `needs_approval` | Gate for `add_menu_item`. Set `automatic` to let the agent add items outright. |
| `KIOSK_CORE_QSR_MENU_UPDATE_GATE` | `automatic` | Gate for `update_menu_item` |
| `KIOSK_CORE_QSR_MENU_MAX_CALLS` | `20` | Rate limit (calls per window) |
| `KIOSK_CORE_QSR_MENU_PER_SECONDS` | `60.0` | Rate-limit window |
| `KIOSK_CORE_QSR_MENU_REQUESTS_PATH` | `./qsr_menu_requests.json` | Approval queue file |

Also exposed (unchanged by the migration): `describe()` (self-description)
and `subscribe(event_type, condition, callback_url)`. **`subscribe` is a
declared no-op.** It validates its arguments and returns
`{"subscribed": false, "implemented": false, "reason": ...}` rather than a
success acknowledgement, because nothing dispatches on those registrations —
acknowledging a subscription that never fires is worse than refusing it. Use
the autonomy push below for server-initiated events.

> **Breaking change.** `subscribed` previously returned the event-type string
> (truthy); it is now the boolean `false`. A caller doing
> `if response["subscribed"]` now correctly sees that nothing was registered,
> but a caller using the value as an identifier will break. Nothing in this
> repository consumes it — **verify against the Central QSR Agent**, which
> lives outside this tree, before deploying.

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

### The two store IDs

There are **two** distinct store identifiers, and mixing them up is the
easiest misconfiguration to make. They are separate because they answer to
different owners:

| Setting | Default | Who defines the value | What it labels |
|---|---|---|---|
| `KIOSK_CORE_QSR_MCP_STORE_ID` | `store-001` | **We do.** Local bookkeeping only. | The `store_id` stamped on every row of our own durable event log (`qsr_mcp_events.db`). Nothing outside this kiosk reads it. |
| `KIOSK_CORE_QSR_AUTONOMY_STORE_ID` | `qsr-001` | **The agent does.** Must equal its `QSR_RESTAURANT_ID`. | The `store_id` field of the envelope POSTed to the agent. A mismatch means the agent rejects every event with a 4xx — which is treated as permanent, so nothing is retried and the events are lost. |

Symptom of getting it wrong: events appear in the local log but the agent
never sees them, and `[QSR-AUTONOMY] event ... rejected: HTTP 4xx` is logged.

If you have no reason to keep them apart, set both to the same value — the
only cost is that the local log is relabelled, which nothing depends on:

```
KIOSK_CORE_QSR_MCP_STORE_ID=qsr-001
KIOSK_CORE_QSR_AUTONOMY_STORE_ID=qsr-001
```

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
