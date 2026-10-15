"""Tests for the pooled OVMS HTTP client in plugins/kiosk/directive_mode.py.

stream_completion() used to open a brand-new httpx.AsyncClient (and therefore
a new TCP connection) on every turn. Measured live, this pushed the directive
mode's own documented ~137ms first-content latency out to ~525ms. These tests
guard the fix: one client, created lazily, reused across calls, and cleanly
closable at shutdown.
"""

from __future__ import annotations

import asyncio

import pytest

from plugins.kiosk import directive_mode as dm


@pytest.fixture(autouse=True)
def _reset_pooled_client():
    """Ensure no client leaks between tests."""
    yield
    if dm._HTTP_CLIENT is not None:
        asyncio.run(dm.close_http_client())


class TestPooledClient:
    async def _get(self):
        return await dm._get_http_client()

    def test_first_call_creates_a_client(self):
        client = asyncio.run(self._get())
        assert dm._HTTP_CLIENT is client

    def test_repeated_calls_reuse_the_same_client(self):
        async def _twice():
            first = await dm._get_http_client()
            second = await dm._get_http_client()
            return first, second

        first, second = asyncio.run(_twice())
        assert first is second

    def test_concurrent_first_calls_only_create_one_client(self):
        """Two turns racing to initialize on a cold process must not each
        open their own connection — the lock must serialize creation."""
        async def _race():
            results = await asyncio.gather(
                dm._get_http_client(),
                dm._get_http_client(),
                dm._get_http_client(),
            )
            return results

        clients = asyncio.run(_race())
        assert len({id(c) for c in clients}) == 1

    def test_close_releases_the_client_and_allows_recreation(self):
        async def _close_and_reopen():
            first = await dm._get_http_client()
            await dm.close_http_client()
            assert dm._HTTP_CLIENT is None
            second = await dm._get_http_client()
            return first, second

        first, second = asyncio.run(_close_and_reopen())
        # A fresh client is created post-close; it must not be the closed one.
        assert first is not second

    def test_client_is_a_live_async_client(self):
        import httpx

        client = asyncio.run(self._get())
        assert isinstance(client, httpx.AsyncClient)
        assert not client.is_closed
