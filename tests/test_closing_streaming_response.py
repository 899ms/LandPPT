"""Validate response completion and errors through the ASGI entry point."""

import asyncio

import pytest
from starlette.background import BackgroundTask

from landppt.services.runtime.ai_execution import (
    ai_conversation_context,
    get_current_ai_conversation_id,
)
from landppt.web.responses import ClosingStreamingResponse


@pytest.mark.asyncio
@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
@pytest.mark.parametrize("provider_fails", [False, True])
async def test_response_preserves_context_and_stream_semantics(
    spec_version, provider_fails
):
    events = []
    sent = []
    tasks = []

    async def body():
        tasks.append(asyncio.current_task())
        with ai_conversation_context("stream-session"):
            try:
                yield "first"
                if provider_fails:
                    raise RuntimeError("provider failed")
                yield b"second"
            finally:
                await asyncio.sleep(0)
                tasks.append(asyncio.current_task())
                events.append(("closed", get_current_ai_conversation_id()))
        events.append(("restored", get_current_ai_conversation_id()))

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        sent.append(message)

    async def background():
        events.append(("background", get_current_ai_conversation_id()))

    response = ClosingStreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
        background=BackgroundTask(background),
    )

    async def request():
        with ai_conversation_context("request-session"):
            try:
                await response(
                    {"type": "http", "asgi": {"spec_version": spec_version}},
                    receive,
                    send,
                )
            finally:
                assert get_current_ai_conversation_id() == "request-session"

    if provider_fails:
        with pytest.raises(RuntimeError, match="provider failed"):
            await asyncio.wait_for(request(), timeout=5)
        assert events == [("closed", "stream-session")]
        assert sent[-1]["more_body"] is True
    else:
        await asyncio.wait_for(request(), timeout=5)
        assert events == [
            ("closed", "stream-session"),
            ("restored", "request-session"),
            ("background", "request-session"),
        ]
        assert [message["body"] for message in sent[1:]] == [b"first", b"second", b""]
        assert sent[-1]["more_body"] is False

    assert len(tasks) == 2 and tasks[0] is tasks[1]
    assert sent[0]["status"] == 200
    assert (b"x-accel-buffering", b"no") in sent[0]["headers"]
    assert get_current_ai_conversation_id() is None
