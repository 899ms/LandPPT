import asyncio
from contextlib import aclosing
from pathlib import Path
from types import SimpleNamespace

import pytest

from landppt.services.enhanced_ppt_service import EnhancedPPTService
from landppt.services.outline.outline_workflow_service import OutlineWorkflowService
from landppt.services.outline.project_outline_repair_service import ProjectOutlineRepairService
from landppt.services.outline.project_outline_workflow_service import ProjectOutlineWorkflowService
from landppt.services.runtime.ai_execution import ai_conversation_context
from landppt.services.runtime.ai_execution import get_current_ai_conversation_id
from landppt.services.slide.slide_authoring_service import SlideAuthoringService
from landppt.services.slide.slide_streaming_service import SlideStreamingService
from summeryanyfile.graph.workflow import WorkflowManager


def _file_outline_request(path):
    return SimpleNamespace(
        file_path=str(path),
        filename=path.name,
        topic="Quarterly Review",
        scenario="general",
        requirements="Keep it concise",
        target_audience="Leadership",
        custom_audience="",
        description="",
        language="zh",
        page_count_mode="ai_decide",
        min_pages=5,
        max_pages=15,
        fixed_pages=None,
        ppt_style="general",
        custom_style_prompt="",
        file_processing_mode="markitdown",
        content_analysis_depth="standard",
    )


@pytest.mark.asyncio
async def test_file_outline_stream_cleanup_restores_context_and_isolates_sessions(tmp_path):
    source_file = tmp_path / "source.md"
    source_file.write_text("# Quarterly Review\nContext", encoding="utf-8")
    created_sessions = []
    closed_generators = []

    class FakeOutline:
        def to_dict(self):
            return {
                "title": "Quarterly Review",
                "slides": [
                    {
                        "page_number": 1,
                        "title": "Quarterly Review",
                        "content_points": ["Context"],
                        "slide_type": "title",
                    }
                ],
            }

    class FakeGenerator:
        async def stream_generate_from_file(self, *_args, **_kwargs):
            try:
                yield {"status": {"step": "generating"}}
                yield {"outline_obj": FakeOutline(), "llm_call_count": 1}
                yield {"status": {"step": "finished"}}
            finally:
                closed_generators.append(self)

    class DummyService:
        def _standardize_summeryfile_outline(self, outline):
            return outline

        async def _validate_and_repair_outline_json(self, outline, _requirements):
            return outline

        def _extract_summeryanyfile_llm_call_count(self, _generator):
            return 1

    workflow = OutlineWorkflowService(DummyService())

    async def create_generator(_request):
        created_sessions.append(get_current_ai_conversation_id())
        return FakeGenerator(), tmp_path

    workflow._create_outline_generator = create_generator
    service = object.__new__(EnhancedPPTService)
    service.outline_workflow = workflow
    request = _file_outline_request(source_file)

    async def consume(stop_on_outline):
        events = []
        stream = service.generate_outline_from_file_streaming(request)
        async with aclosing(stream):
            async for event in stream:
                events.append(event)
                if stop_on_outline and event.get("outline"):
                    break
        return events

    assert get_current_ai_conversation_id() is None
    first_events = await consume(stop_on_outline=True)
    assert any(event.get("outline") for event in first_events)
    assert get_current_ai_conversation_id() is None
    assert len(closed_generators) == 1

    second_events = await consume(stop_on_outline=False)
    assert any(event.get("outline") for event in second_events)
    assert get_current_ai_conversation_id() is None
    assert len(closed_generators) == 2
    assert created_sessions[0] != created_sessions[1]


@pytest.mark.asyncio
async def test_langgraph_execution_propagates_configurable_conversation_id():
    observed = []

    class FakeChain:
        def __init__(self, name, response):
            self.name = name
            self.response = response

        async def ainvoke(self, _inputs, config):
            observed.append(
                {
                    "chain": self.name,
                    "config": config,
                    "conversation_id": get_current_ai_conversation_id(),
                }
            )
            return self.response

    from summeryanyfile.generators.chains import ChainManager

    chain_manager = object.__new__(ChainManager)
    chain_manager._chains = {
        "structure_analysis": FakeChain(
            "structure_analysis",
            '{"title":"Review","type":"report","sections":[],"key_concepts":[],"language":"English","complexity":"medium"}',
        ),
        "initial_outline": FakeChain(
            "initial_outline",
            '{"title":"Review","total_pages":1,"slides":[{"page_number":1,"title":"Review","content_points":["Context"],"slide_type":"title","description":"Opening"}]}',
        ),
        "refine_outline": FakeChain("refine_outline", "{}"),
        "error_recovery": FakeChain("error_recovery", "{}"),
    }

    config = SimpleNamespace(
        conversation_id="explicit-session",
        recursion_limit=10,
        max_slides=5,
        target_language="en",
    )
    workflow = WorkflowManager(chain_manager, config)
    initial_state = {
        "document_chunks": ["Context"],
        "current_index": 0,
        "ppt_title": "",
        "slides": [],
        "total_pages": 0,
        "page_count_mode": "ai_decide",
        "document_structure": {},
        "accumulated_context": "",
        "project_topic": "Review",
        "project_scenario": "general",
        "project_requirements": "",
        "target_audience": "Leadership",
        "custom_audience": "",
        "ppt_style": "general",
        "custom_style_prompt": "",
        "min_pages": 1,
        "max_pages": 5,
        "fixed_pages": None,
    }

    assert get_current_ai_conversation_id() is None
    result = await workflow.execute_workflow(initial_state)

    assert result["ppt_title"] == "Review"
    assert [item["chain"] for item in observed] == [
        "structure_analysis",
        "initial_outline",
    ]
    assert all(item["conversation_id"] == "explicit-session" for item in observed)
    assert all(
        item["config"].get("configurable", {}).get("conversation_id")
        == "explicit-session"
        for item in observed
    )


@pytest.mark.asyncio
async def test_chain_stream_reads_configurable_conversation_id():
    observed = []

    class FakeStreamChain:
        async def astream(self, _inputs, _config):
            observed.append(get_current_ai_conversation_id())
            yield "chunk"

    from summeryanyfile.generators.chains import ChainManager

    chain_manager = object.__new__(ChainManager)
    chain_manager._chains = {"test": FakeStreamChain()}

    chunks = [
        chunk
        async for chunk in chain_manager.stream_chain(
            "test",
            {},
            {"configurable": {"conversation_id": "stream-session"}},
        )
    ]

    assert chunks == ["chunk"]
    assert observed == ["stream-session"]


@pytest.mark.asyncio
async def test_outline_stream_early_close_restores_context_and_isolates_sessions():
    observed_sessions = []
    outline = {
        "title": "Cached deck",
        "slides": [
            {
                "page_number": 1,
                "title": "Cached deck",
                "content_points": ["Context"],
                "slide_type": "title",
            }
        ],
    }

    class FakeProjectManager:
        async def get_project(self, _project_id):
            observed_sessions.append(get_current_ai_conversation_id())
            return SimpleNamespace(outline=outline, confirmed_requirements={})

    workflow = ProjectOutlineWorkflowService(
        SimpleNamespace(project_manager=FakeProjectManager())
    )

    async def skip_stage_update(*_args, **_kwargs):
        return None

    workflow._outline_generation._streaming_service._update_outline_generation_stage = (
        skip_stage_update
    )

    service = object.__new__(EnhancedPPTService)
    service.project_outline_workflow = workflow

    async def consume_first_event():
        stream = service.generate_outline_streaming("project-1")
        event = await anext(stream)
        await stream.aclose()
        return event

    assert get_current_ai_conversation_id() is None
    first_event = await consume_first_event()
    assert '"step": "cached"' in first_event
    assert get_current_ai_conversation_id() is None

    second_event = await consume_first_event()
    assert '"step": "cached"' in second_event
    assert get_current_ai_conversation_id() is None
    assert observed_sessions[0] != observed_sessions[1]


@pytest.mark.asyncio
async def test_slide_stream_early_close_restores_context_and_isolates_sessions():
    observed_sessions = []
    closed_sessions = []

    async def inner_stream(_project_id):
        session_id = get_current_ai_conversation_id()
        observed_sessions.append(session_id)
        try:
            yield "first slide event"
            yield "second slide event"
        finally:
            closed_sessions.append(session_id)

    service = object.__new__(EnhancedPPTService)
    authoring = object.__new__(SlideAuthoringService)
    streaming = object.__new__(SlideStreamingService)
    authoring._service = service
    authoring._streaming_service = streaming
    streaming._service = authoring
    streaming._generate_slides_streaming = inner_stream
    service.slide_authoring = authoring

    async def consume_first_event():
        stream = service.generate_slides_streaming("project-1")
        event = await anext(stream)
        await stream.aclose()
        return event

    assert get_current_ai_conversation_id() is None
    assert await consume_first_event() == "first slide event"
    assert get_current_ai_conversation_id() is None

    assert await consume_first_event() == "first slide event"
    assert get_current_ai_conversation_id() is None
    assert observed_sessions[0] != observed_sessions[1]
    assert closed_sessions == observed_sessions


@pytest.mark.asyncio
async def test_outline_repair_provider_failure_fails_without_fallback(tmp_path):
    provider_calls = []
    fallback_reads = []

    class StubProviderService:
        async def _text_completion_for_role(self, role, prompt=None, temperature=None):
            provider_calls.append(role)
            raise RuntimeError("MissingSessionID: x-opencode-session is required")

    repair_service = ProjectOutlineRepairService(StubProviderService())

    class DummyService:
        def __init__(self):
            self._validate_and_repair_outline_json = (
                repair_service._validate_and_repair_outline_json
            )

        def _standardize_summeryfile_outline(self, outline):
            return outline

        def _extract_summeryanyfile_llm_call_count(self, _generator):
            return 0

        def _read_file_with_fallback_encoding(self, path):
            fallback_reads.append(path)
            return Path(path).read_text(encoding="utf-8")

    class FakeOutline:
        def to_dict(self):
            return {"title": "Quarterly Review", "slides": []}

    class FakeGenerator:
        async def generate_from_file(self, *_args, **_kwargs):
            return FakeOutline()

    source_file = tmp_path / "source.md"
    source_file.write_text("# Quarterly Review\n- revenue up\n", encoding="utf-8")
    workflow = OutlineWorkflowService(DummyService())

    async def create_generator(_request):
        return FakeGenerator(), tmp_path

    workflow._create_outline_generator = create_generator

    result = await workflow.generate_outline_from_file(
        _file_outline_request(source_file)
    )

    assert result.success is False
    assert result.outline is None
    assert fallback_reads == []
    assert len(provider_calls) == 1
    assert "MissingSessionID" in (result.error or "")


@pytest.mark.asyncio
async def test_chain_stream_early_exit_restores_context_and_closes_provider_stream():
    """Early consumer exit must release the conversation scope and close the
    provider stream while that scope is still active."""
    from summeryanyfile.generators.chains import ChainManager

    observed_sessions = []
    closed_sessions = []

    class FakeStreamChain:
        async def astream(self, _inputs, _config):
            observed_sessions.append(get_current_ai_conversation_id())
            try:
                yield "first"
                yield "second"
            finally:
                closed_sessions.append(get_current_ai_conversation_id())

    chain_manager = object.__new__(ChainManager)
    chain_manager._chains = {"test": FakeStreamChain()}
    config = {"configurable": {"conversation_id": "stream-session"}}

    async def consume_one_chunk_and_close():
        stream = chain_manager.stream_chain("test", {}, config)
        async with aclosing(stream):
            async for chunk in stream:
                assert chunk == "first"
                break
        # Snapshot before returning control to the event loop: the provider
        # stream must already be closed here, not left to a later finalizer tick.
        return list(closed_sessions)

    # (a) no ambient conversation: the scope must restore to the pre-entry None
    assert get_current_ai_conversation_id() is None
    assert await consume_one_chunk_and_close() == ["stream-session"]
    assert get_current_ai_conversation_id() is None

    # (b) production shape: an outer workflow already owns a conversation
    with ai_conversation_context("outer"):
        assert get_current_ai_conversation_id() == "outer"
        assert await consume_one_chunk_and_close() == [
            "stream-session",
            "stream-session",
        ]
        assert get_current_ai_conversation_id() == "outer"

    assert get_current_ai_conversation_id() is None
    assert observed_sessions == ["stream-session", "stream-session"]


@pytest.mark.asyncio
async def test_chain_executor_stream_retry_releases_scope_after_consumer_failure():
    """A consumer-side failure mid-stream must release the conversation scope
    before the retry attempt starts, not leave it set for the caller."""
    from summeryanyfile.generators.chains import ChainExecutor, ChainManager

    observed_sessions = []
    attempts = []

    class FakeStreamChain:
        async def astream(self, _inputs, _config):
            observed_sessions.append(get_current_ai_conversation_id())
            yield "first"
            yield "second"

    chain_manager = object.__new__(ChainManager)
    chain_manager._chains = {"test": FakeStreamChain()}
    executor = ChainExecutor(chain_manager, max_retries=2)

    def chunk_callback(chunk):
        attempts.append(chunk)
        if len(attempts) == 1:
            raise RuntimeError("stream consumer failed")
        return None

    config = {"configurable": {"conversation_id": "retry-session"}}

    assert get_current_ai_conversation_id() is None
    result = await executor.execute_with_retry_streaming(
        "test", {}, config, chunk_callback
    )

    assert result == "firstsecond"
    assert attempts == ["first", "first", "second"]
    assert observed_sessions == ["retry-session", "retry-session"]
    assert get_current_ai_conversation_id() is None
