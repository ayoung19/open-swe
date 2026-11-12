from unittest.mock import AsyncMock, Mock

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from managed_deepagents._memory_instructions import create_memory_instructions_middleware
from pydantic import SecretStr


class TestMemoryCacheBudget:
    @pytest.mark.parametrize("asynchronous", [False, True])
    @pytest.mark.parametrize(
        ("system_breaks", "conversation_breaks", "tool_breaks", "automatic", "anthropic"),
        [
            (3, 0, 0, True, True),
            (2, 1, 1, False, True),
            (2, 0, 0, True, True),
            (0, 0, 0, False, True),
            (0, 0, 0, False, False),
        ],
    )
    async def test_memory_preserves_existing_cache_budget(
        self,
        asynchronous: bool,
        system_breaks: int,
        conversation_breaks: int,
        tool_breaks: int,
        automatic: bool,
        anthropic: bool,
    ) -> None:
        cache = {"type": "ephemeral", "ttl": "1h"}
        request = ModelRequest(
            model=(
                ChatAnthropic(model_name="claude-sonnet-4-5", api_key=SecretStr("test"))
                if anthropic
                else ChatOpenAI(model="gpt-4.1", api_key=SecretStr("test"))
            ),
            system_message=SystemMessage(
                content=[
                    {"type": "text", "text": "Robin instructions", "cache_control": cache}
                    for _ in range(system_breaks)
                ]
            ),
            messages=[
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Hello", "cache_control": cache}
                        for _ in range(conversation_breaks)
                    ]
                )
            ],
            tools=[
                {"name": f"tool_{index}", "cache_control": cache} for index in range(tool_breaks)
            ],
            model_settings={"cache_control": cache} if automatic else {},
        )
        middleware = create_memory_instructions_middleware(
            load_agent_memory=lambda: "Remembered context"
        )
        response = ModelResponse(result=[AIMessage(content="ok")])
        if asynchronous:
            handler = AsyncMock(return_value=response)
            await middleware.awrap_model_call(request, handler)
        else:
            handler = Mock(return_value=response)
            middleware.wrap_model_call(request, handler)

        updated: ModelRequest = handler.call_args.args[0]
        assert updated.system_message is not None
        assert request.system_message is not None
        blocks = updated.system_message.content_blocks
        assert blocks[:-1] == request.system_message.content_blocks
        assert "Remembered context" in str(blocks[-1])
        budget = system_breaks + conversation_breaks + tool_breaks + int(automatic)
        assert ("cache_control" in blocks[-1]) == (anthropic and budget < 4)
        assert updated.messages == request.messages
        assert updated.tools == request.tools
        assert updated.model_settings == request.model_settings
