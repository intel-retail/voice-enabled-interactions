"""Unit tests for QSR Agent autonomy event delivery.

Covers the contract the agent's ``HermesEventMenuPolicy`` enforces
(``queue_count_changed`` with a non-negative, non-boolean ``data.queue_count``),
the producer-side change detection its service-owner guide requires, the
retry/permanent-failure split, bearer-token forwarding, and the guarantee that
the pre-existing generic webhook behaviour is unchanged.
"""

import json
import threading
import time

import pytest

from kiosk_core.qsr_mcp.autonomy import AutonomyEventSink, PermanentDeliveryError
from kiosk_core.qsr_mcp.delivery import BackgroundDispatcher, Delivery, WebhookSink
from kiosk_core.qsr_mcp.event_log import new_event

URL = "http://qsr-agent:8600/autonomy/events"


class RecordingPost:
    """Stand-in for ``_post_json`` that records calls and replays statuses."""

    def __init__(self, statuses=(202,)):
        self.statuses = list(statuses)
        self.calls = []

    def __call__(self, url, body, headers, timeout):
        self.calls.append(
            {
                "url": url,
                "body": json.loads(body.decode("utf-8")),
                "headers": headers,
                "timeout": timeout,
            }
        )
        return self.statuses[min(len(self.calls) - 1, len(self.statuses) - 1)]


def queue_event(count, nearby=0, status="LOW"):
    return new_event(
        "queue_depth_sample",
        "smart_kiosk",
        "store-001",
        {"count": count, "nearby": nearby, "status": status},
    )


def board_event(mode="simplified", reason="queue deep"):
    return new_event(
        "board_mode_changed", "smart_kiosk", "store-001", {"mode": mode, "reason": reason}
    )


def make_sink(post, **kwargs):
    kwargs.setdefault("store_id", "qsr-001")
    return AutonomyEventSink(URL, post_json=post, **kwargs)


def make_delivery(post, **kwargs):
    """Wrap a sink the way production does, so eligibility runs via Delivery."""
    return Delivery(sinks=[make_sink(post, **kwargs)], backoff_s=0)


# -- envelope ---------------------------------------------------------------


def test_queue_sample_is_mapped_to_queue_count_changed():
    post = RecordingPost()
    sink = make_sink(post)
    assert sink.push(queue_event(7)) is True

    sent = post.calls[0]["body"]
    assert sent["event_type"] == "queue_count_changed"
    assert sent["store_id"] == "qsr-001"
    assert sent["data"]["queue_count"] == 7
    assert sent["data"]["source"] == "smart_kiosk"
    assert sent["data"]["source_event_type"] == "queue_depth_sample"
    assert sent["occurred_at"].endswith("+00:00")


def test_queue_count_satisfies_agent_policy_validation():
    """Mirrors HermesEventMenuPolicy._validate_event_data for queue events."""
    post = RecordingPost()
    make_sink(post).push(queue_event(0))

    value = post.calls[0]["body"]["data"]["queue_count"]
    assert not isinstance(value, bool)
    assert isinstance(value, int)
    assert value >= 0


def test_board_mode_event_keeps_its_own_name():
    post = RecordingPost()
    assert make_sink(post).push(board_event()) is True

    sent = post.calls[0]["body"]
    assert sent["event_type"] == "board_mode_changed"
    assert sent["data"]["mode"] == "simplified"


def test_event_id_is_stable_across_retries_for_deduplication():
    post = RecordingPost(statuses=(500, 202))
    sink = make_sink(post)
    event = queue_event(4)

    sink.push(event)
    sink.push(event)

    assert post.calls[0]["body"]["event_id"] == post.calls[1]["body"]["event_id"]
    assert event.ref_id in post.calls[0]["body"]["event_id"]


# -- producer-owned change detection ---------------------------------------


def test_unchanged_queue_depth_is_not_forwarded():
    post = RecordingPost()
    delivery = make_delivery(post)

    assert delivery.dispatch(queue_event(5)) is True
    assert delivery.dispatch(queue_event(5)) is True
    assert delivery.dispatch(queue_event(5)) is True

    assert len(post.calls) == 1, "only the first sample should reach the agent"


def test_changed_queue_depth_is_forwarded():
    post = RecordingPost()
    delivery = make_delivery(post)

    delivery.dispatch(queue_event(5))
    delivery.dispatch(queue_event(9))

    assert [c["body"]["data"]["queue_count"] for c in post.calls] == [5, 9]


def test_min_delta_suppresses_jitter():
    post = RecordingPost()
    delivery = make_delivery(post, queue_min_delta=3)

    delivery.dispatch(queue_event(5))
    delivery.dispatch(queue_event(6))  # delta 1 -> suppressed
    delivery.dispatch(queue_event(7))  # delta 2 -> suppressed
    delivery.dispatch(queue_event(9))  # delta 4 -> sent

    assert [c["body"]["data"]["queue_count"] for c in post.calls] == [5, 9]


@pytest.mark.parametrize("bad_count", [None, True, -1, "7", 3.5])
def test_unusable_queue_counts_are_dropped(bad_count):
    post = RecordingPost()
    assert make_delivery(post).dispatch(queue_event(bad_count)) is True
    assert post.calls == []


def test_board_mode_events_are_never_change_filtered():
    post = RecordingPost()
    delivery = make_delivery(post)

    delivery.dispatch(board_event("simplified"))
    delivery.dispatch(board_event("simplified"))

    assert len(post.calls) == 2


def test_retry_is_not_dropped_by_change_detection():
    """Regression: eligibility must be evaluated once per event, not per attempt.

    When ``accepts`` ran inside ``push``, the second attempt saw the same
    queue count as "unchanged", returned success without POSTing, and the
    observation was silently lost on any transient failure.
    """
    post = RecordingPost(statuses=(503, 202))
    delivery = make_delivery(post)

    assert delivery.dispatch(queue_event(8)) is True
    assert len(post.calls) == 2
    assert post.calls[0]["body"] == post.calls[1]["body"]


def test_suppressed_event_does_not_consume_retries():
    post = RecordingPost()
    delivery = make_delivery(post)

    delivery.dispatch(queue_event(5))
    post.calls.clear()
    assert delivery.dispatch(queue_event(5)) is True
    assert post.calls == []


# -- event-type filtering ---------------------------------------------------


def test_event_type_filter_excludes_unlisted_types():
    post = RecordingPost()
    delivery = make_delivery(post, event_types=("queue_depth_sample",))

    assert delivery.dispatch(board_event()) is True
    assert post.calls == []


# -- auth -------------------------------------------------------------------


def test_bearer_token_is_forwarded_when_configured():
    post = RecordingPost()
    make_sink(post, token="s3cret").push(queue_event(2))
    assert post.calls[0]["headers"]["Authorization"] == "Bearer s3cret"


def test_no_authorization_header_without_token():
    post = RecordingPost()
    make_sink(post).push(queue_event(2))
    assert "Authorization" not in post.calls[0]["headers"]


# -- retry semantics --------------------------------------------------------


def test_transient_failure_is_retried_until_accepted():
    post = RecordingPost(statuses=(503, 503, 202))
    delivery = Delivery(sinks=[make_sink(post)], backoff_s=0)

    assert delivery.dispatch(queue_event(6)) is True
    assert len(post.calls) == 3


def test_rate_limit_is_treated_as_transient():
    post = RecordingPost(statuses=(429, 202))
    delivery = Delivery(sinks=[make_sink(post)], backoff_s=0)

    assert delivery.dispatch(queue_event(6)) is True
    assert len(post.calls) == 2


def test_permanent_rejection_stops_retrying():
    post = RecordingPost(statuses=(400,))
    delivery = Delivery(sinks=[make_sink(post)], backoff_s=0)

    assert delivery.dispatch(queue_event(6)) is False
    assert len(post.calls) == 1, "a 4xx body rejection must not be resent"


def test_permanent_delivery_error_is_raised_by_the_sink():
    post = RecordingPost(statuses=(404,))
    with pytest.raises(PermanentDeliveryError):
        make_sink(post).push(queue_event(6))


def test_transport_failure_is_retried_and_reported():
    class Boom:
        calls = 0

        def __call__(self, *args, **kwargs):
            Boom.calls += 1
            raise OSError("connection refused")

    delivery = Delivery(sinks=[make_sink(Boom())], backoff_s=0)
    assert delivery.dispatch(queue_event(6)) is False
    assert Boom.calls == 3


# -- existing behaviour preserved ------------------------------------------


def test_delivery_without_sinks_is_a_no_op():
    delivery = Delivery()
    assert delivery.sinks == ()
    assert delivery.dispatch(queue_event(1)) is True


def test_webhook_url_still_builds_a_webhook_sink():
    delivery = Delivery(webhook_url="http://inbox.example/hook")
    assert len(delivery.sinks) == 1
    assert isinstance(delivery.sinks[0], WebhookSink)


def test_webhook_treats_4xx_as_permanent_and_does_not_retry():
    """The webhook sink must match AutonomyEventSink: a rejected payload is
    not resent, so a misconfigured inbox cannot absorb the retry budget."""
    calls = []

    def post(url, body, headers, timeout):
        calls.append(url)
        return 422

    delivery = Delivery(sinks=[WebhookSink("http://inbox.example/hook", post_json=post)],
                        max_retries=3, backoff_s=0)
    assert delivery.dispatch(queue_event(1)) is False
    assert len(calls) == 1, "a 4xx must not be retried"


def test_webhook_retries_429():
    """429 means "slow down", not "this payload is wrong", so it must use the
    retry budget. AutonomyEventSink already excludes it; the sinks must agree."""
    calls = []

    def post(url, body, headers, timeout):
        calls.append(url)
        return 429

    delivery = Delivery(sinks=[WebhookSink("http://inbox.example/hook", post_json=post)],
                        max_retries=3, backoff_s=0)
    assert delivery.dispatch(queue_event(1)) is False
    assert len(calls) == 3, "429 must be retried, not treated as permanent"


def test_webhook_survives_a_non_urlerror_transport_failure():
    """_post_json can raise http.client.HTTPException or a bare OSError.
    Letting one escape would abort dispatch and skip every later sink."""
    import http.client

    def post(url, body, headers, timeout):
        raise http.client.IncompleteRead(b"")

    sink = WebhookSink("http://inbox.example/hook", post_json=post)
    delivered = []

    class Recorder:
        name = "recorder"

        def push(self, event):
            delivered.append(event.ref_id)
            return True

    delivery = Delivery(sinks=[sink, Recorder()], max_retries=1, backoff_s=0)
    assert delivery.dispatch(queue_event(1)) is False
    assert len(delivered) == 1, "a later sink must still be attempted"


def test_webhook_retries_5xx():
    """A server-side failure is transient and should use the retry budget."""
    calls = []

    def post(url, body, headers, timeout):
        calls.append(url)
        return 503

    delivery = Delivery(sinks=[WebhookSink("http://inbox.example/hook", post_json=post)],
                        max_retries=3, backoff_s=0)
    assert delivery.dispatch(queue_event(1)) is False
    assert len(calls) == 3


def test_webhook_accepts_2xx():
    def post(url, body, headers, timeout):
        return 204

    delivery = Delivery(sinks=[WebhookSink("http://inbox.example/hook", post_json=post)],
                        max_retries=3, backoff_s=0)
    assert delivery.dispatch(queue_event(1)) is True


def test_webhook_and_autonomy_sinks_are_independent():
    """One failing sink must not stop the other from being attempted."""
    post = RecordingPost()

    class FailingSink:
        name = "failing"
        calls = 0

        def push(self, event):
            FailingSink.calls += 1
            return False

    delivery = Delivery(sinks=[FailingSink(), make_sink(post)], backoff_s=0)

    assert delivery.dispatch(queue_event(3)) is False
    assert FailingSink.calls == 3
    assert len(post.calls) == 1


def test_dispatcher_preserves_emission_order_despite_retries():
    """A retrying older event must not overtake a newer one.

    With a thread per event, a slow first delivery would land *after* the
    second, leaving the agent acting on a stale queue depth.
    """
    post = RecordingPost([500, 200] + [200] * 10)
    delivery = Delivery(sinks=[make_sink(post)], backoff_s=0)
    dispatcher = BackgroundDispatcher(delivery, "test-dispatch")

    for count in (3, 11, 25):
        dispatcher.submit(queue_event(count))
    dispatcher.join()

    sent = [c["body"]["data"]["queue_count"] for c in post.calls]
    assert sent == [3, 3, 11, 25]


def test_dispatcher_drops_events_when_backlog_is_full():
    """An unreachable remote must not grow memory without bound."""
    release = threading.Event()

    class BlockingSink:
        calls = 0

        def push(self, event):
            BlockingSink.calls += 1
            release.wait(5)
            return True

    dispatcher = BackgroundDispatcher(
        Delivery(sinks=[BlockingSink()], backoff_s=0), "test-full", max_queue=2
    )
    assert dispatcher.submit(queue_event(1)) is True
    while BlockingSink.calls == 0:
        time.sleep(0.01)
    assert dispatcher.submit(queue_event(2)) is True
    assert dispatcher.submit(queue_event(3)) is True
    assert dispatcher.submit(queue_event(4)) is False

    release.set()
    dispatcher.join()
