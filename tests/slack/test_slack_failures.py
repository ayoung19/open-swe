import json
import logging
import uuid
from unittest.mock import AsyncMock

import httpx2
import openai
import pytest
from runtime_api.model_errors import classify_exception, code_for_error_type

from openswe import completion
from openswe.slack import failures
from openswe.slack.request import SlackRequest
from openswe.slack.retry import SlackRetry


@pytest.fixture(autouse=True)
def clear_status(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    status = AsyncMock(return_value=True)
    monkeypatch.setattr(failures, "set_slack_thread_status", status)
    return status


def _target() -> failures.SlackRequestTarget:
    return failures.SlackRequestTarget(channel_id="C1", thread_ts="1.0", event_id="Ev1")


async def test_unexpected_failure_replies_with_a_uuid7_error_id_that_is_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, clear_status: AsyncMock
) -> None:
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "retry-test-secret-at-least-32-bytes")
    post_reply = AsyncMock(return_value=True)
    monkeypatch.setattr(failures, "post_slack_thread_reply", post_reply)
    target = _target().model_copy(update={"user_id": "U1", "message_ts": "1.5"})

    with caplog.at_level(logging.ERROR, logger=failures.logger.name):
        error_id = await failures.report_slack_failure(target, RuntimeError("boom"))

    assert uuid.UUID(error_id).version == 7
    post_reply.assert_awaited_once()
    clear_status.assert_awaited_once_with("C1", "1.0", "")
    await_args = post_reply.await_args
    assert await_args is not None
    assert await_args.args[:2] == ("C1", "1.0")
    assert "unexpected error" in await_args.args[2]
    assert f"Error ID: `{error_id}`" in await_args.args[2]
    button = await_args.kwargs["blocks"][1]["elements"][0]
    retry = SlackRetry.from_value(json.loads(button["value"]))
    assert retry is not None
    assert retry.model_copy(update={"expires_at": 0}) == SlackRetry(
        channel_id="C1", thread_ts="1.0", user_id="U1", message_ts="1.5", nonce=error_id
    )

    [record] = [r for r in caplog.records if r.getMessage() == "Slack request failed"]
    assert record.error_id == error_id  # type: ignore[attr-defined]
    assert record.slack_channel_id == "C1"  # type: ignore[attr-defined]
    assert record.slack_thread_ts == "1.0"  # type: ignore[attr-defined]
    assert record.slack_event_id == "Ev1"  # type: ignore[attr-defined]
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], RuntimeError)


async def test_status_failure_still_delivers_error_reply(
    monkeypatch: pytest.MonkeyPatch, clear_status: AsyncMock
) -> None:
    clear_status.side_effect = RuntimeError("status unavailable")
    post_reply = AsyncMock(return_value=True)
    monkeypatch.setattr(failures, "post_slack_thread_reply", post_reply)

    error_id = await failures.report_slack_failure(_target(), RuntimeError("boom"))

    assert uuid.UUID(error_id).version == 7
    post_reply.assert_awaited_once()


async def test_run_slack_task_reports_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    post_reply = AsyncMock(return_value=True)
    monkeypatch.setattr(failures, "post_slack_thread_reply", post_reply)

    async def task() -> None:
        raise RuntimeError("boom")

    await failures.run_slack_task(_target(), task())

    post_reply.assert_awaited_once()


@pytest.mark.parametrize("has_thread", [False, True])
@pytest.mark.parametrize("origin", ["button", "reaction", "message", "retry"])
async def test_failure_retry_only_replays_explicitly_replayable_messages(
    monkeypatch: pytest.MonkeyPatch, origin: str, has_thread: bool
) -> None:
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "retry-test-secret-at-least-32-bytes")
    post_reply = AsyncMock(return_value=True)
    monkeypatch.setattr(failures, "post_slack_thread_reply", post_reply)
    request = SlackRequest(
        channel_id="C1",
        thread_ts="1.0",
        event_ts="2.0",
        original_message_ts="1.5" if origin != "button" else "",
        retry_message_ts="1.5" if origin in {"message", "retry"} else "",
        user_id="U1",
        thread_id="thread-1" if has_thread else None,
    )

    await failures.report_slack_failure(request.target, RuntimeError("boom"))

    assert post_reply.await_args is not None
    blocks = post_reply.await_args.kwargs["blocks"]
    if origin in {"button", "reaction"} and not has_thread:
        assert blocks is None
        return
    retry = SlackRetry.from_value(json.loads(blocks[1]["elements"][0]["value"]))
    assert retry is not None
    assert retry.message_ts == ("1.5" if origin in {"message", "retry"} else "")


@pytest.mark.parametrize(
    "status,detail",
    [
        (401, "Invalid API key sk-sensitive123"),
        (402, "Gateway policy blocked the call"),
        (403, "Permission denied"),
        (424, "Secret 'ANTHROPIC_API_KEY' not found"),
    ],
)
async def test_model_access_completion_guides_model_change_without_retry(
    monkeypatch: pytest.MonkeyPatch, status: int, detail: str
) -> None:
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "retry-test-secret-at-least-32-bytes")
    reply = AsyncMock(return_value=True)
    monkeypatch.setattr(completion, "post_slack_thread_reply", reply)
    monkeypatch.setattr(completion, "get_langsmith_error_url", AsyncMock(return_value=None))
    error = openai.APIStatusError(
        "provider error",
        response=httpx2.Response(status, request=httpx2.Request("POST", "https://gateway.invalid")),
        body={"error": {"message": detail}},
    )
    metadata = {
        "source": "slack",
        "source_context": {
            "slack_thread": {"channel_id": "C1", "thread_ts": "1.0", "triggering_user_id": "U1"}
        },
    }

    await completion._post_failure_reply("t1", metadata, "error", classify_exception(error))

    assert reply.await_args is not None
    assert "Pick a different model" in reply.await_args.args[2]
    assert "sk-sensitive123" not in reply.await_args.args[2]
    assert reply.await_args.kwargs["blocks"] is None

    for error_type in (
        "AuthenticationError",
        "PermissionDeniedError",
        "OpenAIAuthenticationError",
        "OpenAIPermissionDeniedError",
    ):
        await completion._post_failure_reply(
            "t1", metadata, "error", code_for_error_type(error_type)
        )
        assert reply.await_args is not None
        assert "Pick a different model" in reply.await_args.args[2]
        assert reply.await_args.kwargs["blocks"] is None
