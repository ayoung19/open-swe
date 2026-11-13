from pathlib import Path

import pytest
from deepagents.backends import FilesystemBackend
from deepagents.middleware.summarization import SummarizationState
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from rt.offload import AutomaticOffload, RequestedOffload


class _Offloading(AgentMiddleware):
    """Offload before the first model call and end the run, as a requested offload does."""

    # As MDA's own summarization keeps it in every graph's state.
    state_schema = SummarizationState

    def __init__(self, offload: RequestedOffload) -> None:
        self._offload = offload

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: AgentState, runtime: Runtime) -> dict[str, object]:
        update, _outcome = await self._offload.offload(state, runtime)
        return {**update, "jump_to": "end"}


async def test_a_requested_offload_keeps_the_history_and_streams_only_its_status(
    tmp_path: Path,
) -> None:
    model = FakeListChatModel(responses=["A private summary of the old conversation."])
    offload = RequestedOffload(model, FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True))
    offload._lc_helper.keep = ("messages", 2)
    graph = create_agent(
        model=model, middleware=[_Offloading(offload)], checkpointer=InMemorySaver()
    )
    messages = [
        HumanMessage(content="Remember the deployment decision.", id="human-1"),
        AIMessage(content="Use the staging environment.", id="ai-1"),
        HumanMessage(content="What next?", id="human-2"),
        AIMessage(content="Run the smoke tests.", id="ai-2"),
    ]
    config: RunnableConfig = {"configurable": {"thread_id": "offloading-test"}}

    chunks = [
        chunk
        async for chunk in graph.astream(
            {"messages": messages}, config, stream_mode=["messages", "custom"]
        )
    ]

    assert not [data for mode, data in chunks if mode == "messages"]
    statuses = [data for mode, data in chunks if mode == "custom" and isinstance(data, dict)]
    assert [event["status"] for event in statuses] == ["started", "completed"]
    assert statuses[-1]["trigger"] == "manual"
    state = await graph.aget_state(config)
    assert state.values["messages"] == messages
    event = state.values["_summarization_event"]
    assert event["cutoff_index"] == 2
    assert "private summary" in event["summary_message"].content
    assert "deployment decision" in (tmp_path / event["file_path"].lstrip("/")).read_text()

    # Nothing new since the summary: nothing is offloaded again.
    chunks = [chunk async for chunk in graph.astream({}, config, stream_mode="custom")]
    assert chunks[-1]["status"] == "skipped"
    assert (await graph.aget_state(config)).values["_summarization_event"] == event


class TestAutomaticOffload:
    @pytest.mark.parametrize(
        ("addition", "output_limit", "has_usage", "summarized"),
        [
            (0, 64_000, True, False),
            (200_000, 64_000, True, True),
            (360_000, 0, True, True),
            (0, 64_000, False, True),
        ],
    )
    async def test_opaque_reasoning_does_not_trigger_early_but_growth_does(
        self,
        tmp_path: Path,
        addition: int,
        output_limit: int,
        has_usage: bool,
        summarized: bool,
    ) -> None:
        model = ChatOpenAI(
            model="gpt-6.1-sol",
            api_key="test-key",
            profile={"max_input_tokens": 272_000},
            max_tokens=output_limit,
        )
        offload = AutomaticOffload(
            model, FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        )
        offload._lc_helper._summary_model = FakeListChatModel(responses=["Earlier work."])
        serving = FakeListChatModel(responses=["Continue."], profile=model.profile).model_copy(
            update={"max_tokens": output_limit}
        )
        graph = create_agent(
            model=serving,
            middleware=[offload],
            checkpointer=InMemorySaver(),
            system_prompt="Instructions " + "s" * 4000,
        )
        reasoning = AIMessage(
            content=[{"type": "reasoning", "encrypted_content": "opaque" * 100_000, "summary": []}],
            usage_metadata={
                "input_tokens": 150_000,
                "output_tokens": 2000,
                "total_tokens": 152_000,
                "output_token_details": {"reasoning": 2000},
            }
            if has_usage
            else None,
            response_metadata={"model_provider": "openai"},
        )
        messages: list[AnyMessage] = [
            HumanMessage(content="x" * 600_000),
            reasoning,
            HumanMessage(content="Continue " + "y" * addition),
        ]
        assert count_tokens_approximately(messages) > 272_000
        config: RunnableConfig = {"configurable": {"thread_id": "automatic-test"}}
        await graph.ainvoke({"messages": messages}, config)
        state = await graph.aget_state(config)
        assert ("_summarization_event" in state.values) is summarized
        assert isinstance(reasoning.content, list)
        block = reasoning.content[0]
        assert isinstance(block, dict)
        assert block["encrypted_content"] == "opaque" * 100_000

    async def test_current_system_and_tool_schemas_count_toward_the_budget(
        self,
        tmp_path: Path,
    ) -> None:
        model = ChatOpenAI(
            model="gpt-6.1-sol",
            api_key="test-key",
            profile={"max_input_tokens": 272_000},
            max_tokens=64_000,
        )
        offload = AutomaticOffload(model, FilesystemBackend(root_dir=str(tmp_path)))
        messages: list[AnyMessage] = [
            HumanMessage(content="x" * 600_000),
            AIMessage(
                content=[{"type": "reasoning", "encrypted_content": "opaque" * 100_000}],
                usage_metadata={
                    "input_tokens": 150_000,
                    "output_tokens": 2000,
                    "total_tokens": 152_000,
                    "output_token_details": {"reasoning": 2000},
                },
            ),
        ]
        small = ModelRequest(model=model, messages=messages, tools=[])
        large = small.override(
            system_message=SystemMessage(content="s" * 100_000),
            tools=[
                {"name": "lookup", "description": "t" * 100_000, "parameters": {"type": "object"}}
            ],
        )
        assert not offload._over_budget(small)
        assert offload._over_budget(large)
        reasoning = messages[-1]
        assert isinstance(reasoning, AIMessage)
        substantial_reasoning = reasoning.model_copy(
            update={
                "usage_metadata": {
                    "input_tokens": 150_000,
                    "output_tokens": 50_000,
                    "total_tokens": 200_000,
                    "output_token_details": {"reasoning": 50_000},
                }
            }
        )
        assert offload._over_budget(small.override(messages=[messages[0], substantial_reasoning]))
