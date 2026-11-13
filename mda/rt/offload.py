"""Automatic context sizing and explicitly requested conversation offloading.

MDA's own summarization offloads a long conversation on its own. A person may ask for
it sooner, from the dashboard; the run started for that summarizes the conversation as
MDA's summarization would and ends. The event it writes is the one MDA's summarization
applies to every later model call.
"""

import logging
from collections.abc import Awaitable, Callable
from copy import copy
from math import ceil
from typing import Final

from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.summarization import (
    SUMMARIZATION_EVENT_KEY,
    SummarizationMiddleware,
    compute_summarization_defaults,
)
from langchain.agents.middleware.types import (
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from pydantic import JsonValue

# Held state the transcript reads the offload's outcome from, with the next model call.
OFFLOADING_KEY: Final = "conversation_offloading"


logger = logging.getLogger(__name__)


class AutomaticOffload(SummarizationMiddleware):
    """Automatic compaction with token sizing for opaque OpenAI reasoning."""

    def __init__(self, model: BaseChatModel, backend: BackendProtocol) -> None:
        super().__init__(
            model=model,
            backend=backend,
            trim_tokens_to_summarize=None,
            **compute_summarization_defaults(model),
        )

    @property
    def name(self) -> str:
        return "SummarizationMiddleware"

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | ExtendedModelResponse:
        backend = getattr(request.runtime, "backend", None)
        bound = copy(self)
        if isinstance(backend, BackendProtocol):
            bound._backend = backend  # noqa: SLF001
        return await super(AutomaticOffload, bound).awrap_model_call(request, handler)

    def _count_tokens(
        self,
        messages: list[AnyMessage],
        system_message: SystemMessage | None,
        tools: list[BaseTool | dict[str, object]] | None,
    ) -> int:
        if not isinstance(self.model, ChatOpenAI):
            return super()._count_tokens(messages, system_message, tools)
        counted: list[AnyMessage] = []
        reasoning_tokens = 0
        latest_index: int | None = None
        latest_tokens: int | None = None
        latest_reasoning = 0
        corrected = False
        for message in messages:
            usage = message.usage_metadata if isinstance(message, AIMessage) else None
            reasoning = (
                (usage.get("output_token_details") or {}).get("reasoning") if usage else None
            )
            if (
                isinstance(message, AIMessage)
                and isinstance(message.content, list)
                and isinstance(reasoning, int)
                and reasoning >= 0
                and any(
                    isinstance(block, dict)
                    and block.get("type") == "reasoning"
                    and block.get("encrypted_content")
                    for block in message.content
                )
            ):
                # Ciphertext length is unrelated to context size; keep its reported tokens.
                content = [
                    {key: value for key, value in block.items() if key != "encrypted_content"}
                    if isinstance(block, dict) and block.get("type") == "reasoning"
                    else block
                    for block in message.content
                ]
                counted.append(message.model_copy(update={"content": content}))
                reasoning_tokens += reasoning
                corrected = True
            else:
                counted.append(message)
            if isinstance(message, AIMessage) and usage:
                latest_index = len(counted)
                latest_tokens = usage["total_tokens"]
                latest_reasoning = reasoning_tokens
        if not corrected:
            return super()._count_tokens(messages, system_message, tools)
        full = [system_message, *counted] if system_message is not None else counted
        total = count_tokens_approximately(full, tools=tools) + reasoning_tokens
        providers = {
            message.response_metadata.get("model_provider")
            for message in messages
            if isinstance(message, AIMessage)
        }
        if latest_index is not None and latest_tokens is not None and providers == {"openai"}:
            prefix = ([system_message] if system_message is not None else []) + counted[
                :latest_index
            ]
            baseline = count_tokens_approximately(prefix, tools=tools) + latest_reasoning
            if baseline > 0:
                total = ceil(total * min(1.25, max(1.0, latest_tokens / baseline)))
        return total

    @staticmethod
    def _latest_usage(messages: list[AnyMessage]) -> UsageMetadata | None:
        return next(
            (
                message.usage_metadata
                for message in reversed(messages)
                if isinstance(message, AIMessage)
            ),
            None,
        )

    def _should_summarize(self, messages: list[AnyMessage], total_tokens: int) -> bool:
        decision = super()._should_summarize(messages, total_tokens)
        logger.debug(
            "Automatic summarization threshold decision",
            extra={
                "max_input_tokens": self._get_profile_limits(),
                "estimated_tokens": total_tokens,
                "trigger": self._lc_helper.trigger,
                "latest_usage": self._latest_usage(messages),
                "should_summarize": decision,
            },
        )
        return decision

    def _over_budget(self, request: ModelRequest, total_tokens: int | None = None) -> bool:
        count = (
            total_tokens
            if total_tokens is not None
            else self._count_tokens(request.messages, request.system_message, request.tools)
        )
        decision = super()._over_budget(request, count)
        logger.debug(
            "Automatic summarization input budget decision",
            extra={
                "max_input_tokens": request.model.profile.get("max_input_tokens")
                if request.model.profile
                else None,
                "estimated_tokens": count,
                "input_budget": self._input_budget(request),
                "latest_usage": self._latest_usage(request.messages),
                "over_budget": decision,
            },
        )
        return decision


class RequestedOffload(SummarizationMiddleware):
    """MDA's summarization, made to summarize the conversation now."""

    def __init__(self, model: BaseChatModel, backend: BackendProtocol) -> None:
        # Its summary is the run's work, not something to stream to the person.
        summary_model = model.model_copy(
            update={"tags": [*(model.tags or []), "nostream", "langsmith:hidden"]}
        )
        super().__init__(
            model=summary_model, backend=backend, **compute_summarization_defaults(model)
        )

    def _status(self, status: str, **details: JsonValue) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {"status": status, "trigger": "manual", **details}
        get_stream_writer()({"type": OFFLOADING_KEY, **payload})
        return payload

    def _should_summarize(self, messages: list[AnyMessage], total_tokens: int) -> bool:  # noqa: ARG002
        return True

    def _determine_cutoff_index(self, messages: list[AnyMessage]) -> int:
        cutoff = super()._determine_cutoff_index(messages)
        # Nothing but earlier summaries before the cutoff: there is nothing new to offload.
        return cutoff if self._filter_summary_messages(messages[:cutoff]) else 0

    async def _acreate_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        self._status("started")
        try:
            return await super()._acreate_summary(messages_to_summarize)
        except Exception:
            self._status("failed")
            raise

    async def offload(
        self, state: AgentState, runtime: Runtime
    ) -> tuple[dict[str, object], dict[str, JsonValue]]:
        """The conversation summarized: the graph state it writes, and the offload's outcome."""

        async def finish(_request: ModelRequest) -> ModelResponse:
            return ModelResponse(result=[])

        response = await self.awrap_model_call(
            ModelRequest(
                model=self.model,
                messages=state.get("messages", []),
                tools=[],
                state=state,
                runtime=runtime,
            ),
            finish,
        )
        update = (
            response.command.update
            if isinstance(response, ExtendedModelResponse) and response.command
            else None
        )
        event = update.get(SUMMARIZATION_EVENT_KEY) if isinstance(update, dict) else None
        if not isinstance(update, dict) or not isinstance(event, dict):
            return {}, self._status("skipped")
        return update, self._status(
            "completed", cutoff_index=event["cutoff_index"], file_path=event["file_path"]
        )
