import sys
import types
from pathlib import Path

import pytest

from landppt.ai.base import AIMessage, MessageRole
from landppt.ai.providers import OpenAIProvider


class _FakeChatCompletions:
    def __init__(self):
        self.create_calls = []

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if kwargs.get("stream"):
            async def stream_chunks():
                yield types.SimpleNamespace(
                    choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content="hello"))]
                )

            return stream_chunks()

        return types.SimpleNamespace(
            model=kwargs["model"],
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="hello", tool_calls=[]),
                    finish_reason="stop",
                )
            ],
            usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


class _FakeAsyncOpenAI:
    instances = []

    def __init__(self, **kwargs):
        self.chat = types.SimpleNamespace(completions=_FakeChatCompletions())
        self.responses = types.SimpleNamespace()
        self.__class__.instances.append(self)


def _install_fake_openai(monkeypatch):
    _FakeAsyncOpenAI.instances = []
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=_FakeAsyncOpenAI))


@pytest.mark.asyncio
async def test_opencode_go_reuses_session_for_same_conversation(monkeypatch):
    _install_fake_openai(monkeypatch)
    provider = OpenAIProvider(
        {
            "api_key": "test-key",
            "base_url": "https://opencode.ai/zen/go/v1/",
            "model": "deepseek-v4.1-flash",
        }
    )

    messages = [AIMessage(role=MessageRole.USER, content="hello")]
    await provider.chat_completion(messages, conversation_id="session-a")
    await provider.chat_completion(messages, conversation_id="session-a")

    calls = _FakeAsyncOpenAI.instances[0].chat.completions.create_calls
    assert calls[0]["extra_headers"] == {"x-opencode-session": "session-a"}
    assert calls[1]["extra_headers"] == calls[0]["extra_headers"]


@pytest.mark.asyncio
async def test_opencode_go_uses_different_sessions_for_different_conversations(monkeypatch):
    _install_fake_openai(monkeypatch)
    provider = OpenAIProvider(
        {
            "api_key": "test-key",
            "base_url": "https://opencode.ai/zen/go/v1",
            "model": "deepseek-v4.1-flash",
        }
    )

    messages = [AIMessage(role=MessageRole.USER, content="hello")]
    await provider.chat_completion(messages, conversation_id="session-a")
    await provider.chat_completion(messages, conversation_id="session-b")

    calls = _FakeAsyncOpenAI.instances[0].chat.completions.create_calls
    assert calls[0]["extra_headers"]["x-opencode-session"] != calls[1]["extra_headers"]["x-opencode-session"]


@pytest.mark.asyncio
async def test_non_opencode_openai_compatible_provider_has_no_session_header(monkeypatch):
    _install_fake_openai(monkeypatch)
    provider = OpenAIProvider(
        {
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4.1",
        }
    )

    await provider.chat_completion(
        [AIMessage(role=MessageRole.USER, content="hello")],
        conversation_id="session-a",
    )

    call = _FakeAsyncOpenAI.instances[0].chat.completions.create_calls[0]
    assert "extra_headers" not in call


@pytest.mark.asyncio
async def test_opencode_go_streaming_reuses_session_header(monkeypatch):
    _install_fake_openai(monkeypatch)
    provider = OpenAIProvider(
        {
            "api_key": "test-key",
            "base_url": "https://opencode.ai/zen/go/v1",
            "model": "deepseek-v4.1-flash",
        }
    )

    chunks = []
    async for chunk in provider.stream_chat_completion(
        [AIMessage(role=MessageRole.USER, content="hello")],
        conversation_id="session-a",
    ):
        chunks.append(chunk)

    assert "".join(chunks) == "hello"
    call = _FakeAsyncOpenAI.instances[0].chat.completions.create_calls[0]
    assert call["extra_headers"] == {"x-opencode-session": "session-a"}
    assert call["stream"] is True


def test_opencode_session_header_only_matches_go_base_url():
    from landppt.ai.providers import build_opencode_session_headers

    assert build_opencode_session_headers(
        "https://opencode.ai/zen/go/v1/", "session-a"
    ) == {"x-opencode-session": "session-a"}
    assert build_opencode_session_headers(
        "https://opencode.ai/zen/go/v1/responses", "session-a"
    ) == {"x-opencode-session": "session-a"}
    assert build_opencode_session_headers(
        "https://api.openai.com/v1", "session-a"
    ) == {}


def test_provider_test_session_is_temporary_and_not_fixed():
    from landppt.ai.providers import build_opencode_test_session_headers

    first = build_opencode_test_session_headers("https://opencode.ai/zen/go/v1")
    second = build_opencode_test_session_headers("https://opencode.ai/zen/go/v1")

    assert first.get("x-opencode-session")
    assert second.get("x-opencode-session")
    assert first != second
    assert build_opencode_test_session_headers("https://api.openai.com/v1") == {}


def test_frontend_chat_histories_rotate_and_reuse_session_ids():
    root = Path(__file__).parents[1]
    core = (
        root
        / "src/landppt/web/static/js/pages/project/slides_editor/projectSlidesEditor.core.js"
    ).read_text(encoding="utf-8")
    sidebar = (
        root
        / "src/landppt/web/static/js/pages/project/slides_editor/projectSlidesEditor.aiChat.js"
    ).read_text(encoding="utf-8")
    native = (
        root
        / "src/landppt/web/static/js/pages/project/slides_editor/projectSlidesEditor.nativeChat.js"
    ).read_text(encoding="utf-8")

    assert "let aiChatSessionIds = {};" in core
    assert "let nativeChatSessionIds = {};" in core
    assert "conversation_id: getAIChatSessionId()" in sidebar
    assert "aiChatSessionIds[currentSlideIndex] = newLandPPTConversationId();" in sidebar
    assert "conversation_id: getNativeChatSessionId()" in native
    assert "nativeChatSessionIds[currentSlideIndex] = newLandPPTConversationId();" in native
