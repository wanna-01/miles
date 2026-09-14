from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from miles.rollout.session.core import ProxyRequest, SessionCore


class _BlockingBackend:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def do_proxy(self, *_args, **_kwargs):
        self.started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class _Registry:
    def __init__(self) -> None:
        self.session = SimpleNamespace(lock=asyncio.Lock(), closing=False)
        self.removed = []

    def get_session(self, session_id):
        assert session_id == "session"
        return self.session

    def remove_session(self, session_id):
        self.removed.append(session_id)


@pytest.mark.asyncio
async def test_delete_session_cancels_registered_upstream_proxy():
    backend = _BlockingBackend()
    registry = _Registry()
    core = SessionCore(backend, registry, SimpleNamespace())
    task = core._start_session_proxy(
        "session",
        ProxyRequest(method="POST"),
        "v1/chat/completions",
        body=b"{}",
        headers={},
    )
    waiter = asyncio.create_task(core._finish_session_proxy("session", task))
    await backend.started.wait()

    response = await core.delete_session("session")

    assert response.status_code == 204
    assert backend.cancelled.is_set()
    assert registry.removed == ["session"]
    assert "session" not in core._inflight_proxy_tasks
    with pytest.raises(asyncio.CancelledError):
        await waiter
