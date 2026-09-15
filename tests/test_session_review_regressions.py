from contextlib import aclosing
from types import SimpleNamespace

import pytest

from landppt.services.enhanced_ppt_service import EnhancedPPTService
from landppt.services.outline.outline_workflow_service import OutlineWorkflowService
from landppt.services.runtime.ai_execution import get_current_ai_conversation_id
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
