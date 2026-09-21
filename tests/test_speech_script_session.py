import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from landppt.ai.providers import OpenAIProvider
from landppt.services.runtime.ai_execution import ai_conversation_context
from landppt.services.speech_script_service import (
    SpeechScriptCustomization,
    SpeechScriptService,
)


ROOT = Path(__file__).resolve().parents[1]


class _FakeChatCompletions:
    def __init__(self, failures=0):
        self.calls = []
        self.failures = failures

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("provider request failed")
        return types.SimpleNamespace(
            model=kwargs["model"],
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="script", tool_calls=[]),
                    finish_reason="stop",
                )
            ],
            usage=types.SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )


class _FakeAsyncOpenAI:
    instances = []
    failures = 0

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.chat = types.SimpleNamespace(
            completions=_FakeChatCompletions(self.__class__.failures)
        )
        self.responses = types.SimpleNamespace(create=None)
        self.__class__.instances.append(self)


def _make_service(monkeypatch, *, failures=0):
    _FakeAsyncOpenAI.instances = []
    _FakeAsyncOpenAI.failures = failures
    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(AsyncOpenAI=_FakeAsyncOpenAI),
    )
    provider = OpenAIProvider(
        {
            "api_key": "test-key",
            "base_url": "https://opencode.ai/zen/go/v1",
            "model": "deepseek-v4.1-flash",
        }
    )
    service = object.__new__(SpeechScriptService)
    service.user_id = None
    service.ai_provider = provider
    service.provider_settings = {
        "provider": "openai",
        "model": "deepseek-v4.1-flash",
    }

    async def no_existing_contexts(*_args):
        return {}

    service._load_existing_script_contexts = no_existing_contexts
    return service, provider


def _project(slide_count=1):
    return SimpleNamespace(
        project_id="speech-project",
        topic="Session propagation",
        scenario="general",
        slides_data=[
            {"title": f"Slide {index + 1}", "html_content": "content"}
            for index in range(slide_count)
        ],
    )


def _customization():
    return SpeechScriptCustomization()


def _session_headers(provider):
    calls = provider.client.chat.completions.calls
    return [call.get("extra_headers", {}) for call in calls]


@pytest.mark.asyncio
async def test_multi_slide_speech_generation_uses_one_opencode_session(monkeypatch):
    service, provider = _make_service(monkeypatch)

    result = await service.generate_multi_slide_scripts_with_retry(
        _project(slide_count=3),
        [0, 1, 2],
        _customization(),
        max_retries=1,
    )

    assert result.success is True
    headers = _session_headers(provider)
    assert len(headers) == 3
    assert all(headers)
    assert len({item["x-opencode-session"] for item in headers}) == 1


@pytest.mark.asyncio
async def test_speech_generation_retry_keeps_the_same_opencode_session(monkeypatch):
    service, provider = _make_service(monkeypatch, failures=1)

    result = await service.generate_multi_slide_scripts_with_retry(
        _project(),
        [0],
        _customization(),
        max_retries=2,
    )

    assert result.success is True
    headers = _session_headers(provider)
    assert len(headers) == 2
    assert headers[0]["x-opencode-session"] == headers[1]["x-opencode-session"]


@pytest.mark.asyncio
async def test_independent_speech_generations_and_concurrent_runs_do_not_share_sessions(monkeypatch):
    service_a, provider_a = _make_service(monkeypatch)
    service_b, provider_b = _make_service(monkeypatch)

    await service_a.generate_multi_slide_scripts_with_retry(
        _project(), [0], _customization(), max_retries=1
    )
    await service_a.generate_multi_slide_scripts_with_retry(
        _project(), [0], _customization(), max_retries=1
    )
    first = _session_headers(provider_a)[0]["x-opencode-session"]
    second = _session_headers(provider_a)[1]["x-opencode-session"]
    assert first != second

    async def run(service):
        return await service.generate_multi_slide_scripts_with_retry(
            _project(), [0], _customization(), max_retries=1
        )

    await asyncio.gather(run(service_a), run(service_b))
    concurrent = [
        _session_headers(provider_a)[-1]["x-opencode-session"],
        _session_headers(provider_b)[-1]["x-opencode-session"],
    ]
    assert len(set(concurrent)) == 2


@pytest.mark.asyncio
async def test_private_slide_helper_only_inherits_existing_session(monkeypatch):
    service, provider = _make_service(monkeypatch)

    with ai_conversation_context("existing-speech-session"):
        content = await service._generate_script_for_slide(
            _project().slides_data[0],
            0,
            1,
            _project(),
            "",
            _customization(),
        )

    assert content == "script"
    assert _session_headers(provider)[0] == {
        "x-opencode-session": "existing-speech-session"
    }


def test_speech_script_background_entries_are_explicitly_scoped():
    route_text = (ROOT / "src/landppt/web/route_modules/speech_script_routes.py").read_text(
        encoding="utf-8"
    )
    assert '@scoped_ai_conversation("speech-script")' in route_text
    assert '@scoped_ai_conversation("speech-script-humanize")' in route_text
