import asyncio
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


class _FakeResponses:
    def __init__(self):
        self.create_calls = []

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return types.SimpleNamespace(
            model=kwargs["model"],
            output_text="hello",
            usage=types.SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
            status="completed",
            incomplete_details=None,
        )


class _FakeAsyncOpenAI:
    instances = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.chat = types.SimpleNamespace(completions=_FakeChatCompletions())
        self.responses = _FakeResponses()
        self.__class__.instances.append(self)


def _install_fake_openai(monkeypatch):
    _FakeAsyncOpenAI.instances = []
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=_FakeAsyncOpenAI))


def _clear_collected_langchain_fakes(monkeypatch):
    """Some legacy tests install a collection-time langchain_core stub."""
    for name in list(sys.modules):
        if name == "langchain_core" or name.startswith("langchain_core."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    for name in list(sys.modules):
        if name == "landppt.ai.langchain_adapter" or name.startswith("summeryanyfile"):
            monkeypatch.delitem(sys.modules, name, raising=False)


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


@pytest.mark.asyncio
async def test_opencode_go_responses_api_reuses_session_header(monkeypatch):
    _install_fake_openai(monkeypatch)
    provider = OpenAIProvider(
        {
            "api_key": "test-key",
            "base_url": "https://opencode.ai/zen/go/v1",
            "model": "deepseek-v4.1-flash",
            "use_responses_api": True,
        }
    )

    await provider.chat_completion(
        [AIMessage(role=MessageRole.USER, content="hello")],
        conversation_id="session-a",
    )
    call = _FakeAsyncOpenAI.instances[0].responses.create_calls[0]
    assert call["extra_headers"] == {"x-opencode-session": "session-a"}


@pytest.mark.asyncio
async def test_opencode_provider_does_not_invent_session_when_context_is_missing(monkeypatch):
    _install_fake_openai(monkeypatch)
    provider = OpenAIProvider(
        {
            "api_key": "test-key",
            "base_url": "https://opencode.ai/zen/go/v1",
            "model": "deepseek-v4.1-flash",
        }
    )

    await provider.chat_completion([AIMessage(role=MessageRole.USER, content="hello")])
    call = _FakeAsyncOpenAI.instances[0].chat.completions.create_calls[0]
    assert "extra_headers" not in call


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


def test_opencode_go_client_uses_landppt_user_agent_only_for_opencode(monkeypatch):
    _install_fake_openai(monkeypatch)
    OpenAIProvider(
        {
            "api_key": "test-key",
            "base_url": "https://opencode.ai/zen/go/v1",
            "model": "deepseek-v4.1-flash",
        }
    )
    OpenAIProvider(
        {
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4.1",
        }
    )

    opencode_headers = _FakeAsyncOpenAI.instances[0].init_kwargs.get("default_headers")
    openai_headers = _FakeAsyncOpenAI.instances[1].init_kwargs.get("default_headers")
    assert opencode_headers and opencode_headers["User-Agent"].startswith("LandPPT/")
    assert openai_headers is None


@pytest.mark.asyncio
async def test_langchain_adapter_propagates_context_session_without_cache_coupling(monkeypatch):
    _clear_collected_langchain_fakes(monkeypatch)
    from landppt.ai.base import AIResponse
    from landppt.ai.langchain_adapter import LandPPTChatModel
    from landppt.services.runtime.ai_execution import ai_conversation_context

    captured = []

    class _FakeProvider:
        async def chat_completion(self, **kwargs):
            captured.append(kwargs.get("conversation_id"))
            return AIResponse(content="ok", model="deepseek-v4.1-flash", usage={})

    monkeypatch.setattr(
        "landppt.ai.langchain_adapter.AIProviderFactory.create_provider",
        lambda *args, **kwargs: _FakeProvider(),
    )
    model = LandPPTChatModel(
        provider="openai", model="deepseek-v4.1-flash", base_url="https://opencode.ai/zen/go/v1"
    )

    async def invoke(session):
        with ai_conversation_context(session):
            return await model.ainvoke("hello")

    await invoke("session-a")
    await invoke("session-a")
    assert captured == ["session-a", "session-a"]


@pytest.mark.asyncio
async def test_langchain_adapter_keeps_concurrent_conversations_isolated(monkeypatch):
    _clear_collected_langchain_fakes(monkeypatch)
    from landppt.ai.base import AIResponse
    from landppt.ai.langchain_adapter import LandPPTChatModel
    from landppt.services.runtime.ai_execution import ai_conversation_context

    captured = []

    class _FakeProvider:
        async def chat_completion(self, **kwargs):
            session = kwargs.get("conversation_id")
            await asyncio.sleep(0)
            captured.append(session)
            return AIResponse(content="ok", model="test", usage={})

    monkeypatch.setattr(
        "landppt.ai.langchain_adapter.AIProviderFactory.create_provider",
        lambda *args, **kwargs: _FakeProvider(),
    )
    model = LandPPTChatModel(provider="openai", model="test")

    async def invoke(session):
        with ai_conversation_context(session):
            await model.ainvoke("hello")

    await asyncio.gather(invoke("conversation-a"), invoke("conversation-b"))
    assert sorted(captured) == ["conversation-a", "conversation-b"]


@pytest.mark.asyncio
async def test_summeryanyfile_chain_manager_propagates_session_for_invoke_and_stream(monkeypatch):
    _clear_collected_langchain_fakes(monkeypatch)
    from landppt.services.runtime.ai_execution import ai_conversation_context
    from summeryanyfile.generators.chains import ChainManager

    captured = []

    class _FakeChain:
        async def ainvoke(self, inputs, config):
            from landppt.services.runtime.ai_execution import get_current_ai_conversation_id
            captured.append(("invoke", get_current_ai_conversation_id()))
            return "{}"

        async def astream(self, inputs, config):
            from landppt.services.runtime.ai_execution import get_current_ai_conversation_id
            captured.append(("stream", get_current_ai_conversation_id()))
            yield "{}"

    manager = object.__new__(ChainManager)
    manager._chains = {"test": _FakeChain()}
    with ai_conversation_context("summery-session"):
        await manager.invoke_chain("test", {}, {})
        async for _ in manager.stream_chain("test", {}, {}):
            pass

    assert captured == [("invoke", "summery-session"), ("stream", "summery-session")]


@pytest.mark.asyncio
async def test_summeryanyfile_graph_does_not_turn_provider_failure_into_one_page_outline(monkeypatch):
    _clear_collected_langchain_fakes(monkeypatch)
    from summeryanyfile.graph.nodes import GraphNodes

    async def fail(*args, **kwargs):
        raise RuntimeError("MissingSessionID: x-opencode-session is required")

    nodes = GraphNodes(types.SimpleNamespace())
    nodes.chain_executor = types.SimpleNamespace(execute_with_retry=fail)
    state = {
        "document_chunks": ["content"],
        "project_topic": "topic",
        "project_scenario": "general",
        "project_requirements": "",
        "target_audience": "普通大众",
        "custom_audience": "",
        "ppt_style": "general",
        "custom_style_prompt": "",
        "document_structure": {},
    }

    with pytest.raises(RuntimeError, match="MissingSessionID"):
        await nodes.generate_initial_outline(state, {})


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
