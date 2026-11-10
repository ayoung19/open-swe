from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openswe.utils import langsmith
from openswe.utils.langsmith import parse_langsmith_locator


@pytest.mark.parametrize(
    "locator",
    [
        "run-1",
        "https://evil.example/o/org/projects/p/project/t/thread-1",
        "https://user@smith.langchain.com/o/org/projects/p/project/t/thread-1",
        "ftp://smith.langchain.com/o/org/projects/p/project/t/thread-1",
        "https://smith.langchain.com/o/org/projects/p/project/t/thread-1/extra",
        "https://smith.langchain.com/o/org/projects/p/project/x/thread-1",
    ],
)
def test_parse_langsmith_locator_rejects_invalid_trace_urls(locator: str) -> None:
    assert parse_langsmith_locator(locator) is None


async def test_get_open_swe_thread_id_from_langsmith_resolves_run_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SimpleNamespace(
        read_run=AsyncMock(return_value=SimpleNamespace(metadata={"thread_id": "thread-1"}))
    )
    monkeypatch.setattr(langsmith, "_build_langsmith_client", lambda: client)

    result = await langsmith.get_open_swe_thread_id_from_langsmith(
        "11111111-1111-4111-8111-111111111111"
    )

    assert result == "thread-1"
    client.read_run.assert_awaited_once_with("11111111-1111-4111-8111-111111111111")


async def test_finding_feedback_creates_then_updates_a_comment_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import UTC, datetime, timedelta
    from uuid import NAMESPACE_URL, uuid4, uuid5

    from langsmith.utils import LangSmithNotFoundError

    trace_id = str(uuid4())
    start = datetime.now(UTC) - timedelta(days=30)
    project_id = uuid4()
    client = SimpleNamespace(
        read_run=AsyncMock(return_value=SimpleNamespace(session_id=project_id, start_time=start)),
        read_feedback=AsyncMock(side_effect=[LangSmithNotFoundError("missing"), object()]),
        create_feedback=AsyncMock(),
    )
    update = AsyncMock()
    monkeypatch.setattr(langsmith, "_build_langsmith_client", lambda: client)
    monkeypatch.setattr(langsmith, "_update_feedback", update)
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    await langsmith.LangSmithLocator.finding_feedback(trace_id, 901, 0)
    await langsmith.LangSmithLocator.finding_feedback(trace_id, 901, 1)
    feedback_id = uuid5(NAMESPACE_URL, f"{trace_id}:901:finding_thumbs_up")
    assert client.create_feedback.await_args.kwargs == {
        "run_id": trace_id,
        "session_id": project_id,
        "start_time": start,
        "key": "finding_thumbs_up",
        "score": 0,
        "comment": "github_comment:901",
        "feedback_id": feedback_id,
    }
    assert update.await_args.args[2] == feedback_id
    assert update.await_args.kwargs == {"score": 1, "comment": "github_comment:901"}


async def test_review_trace_lookup_uses_the_invocation_even_after_fourteen_days(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections.abc import AsyncIterator
    from datetime import UTC, datetime, timedelta
    from unittest.mock import Mock
    from uuid import uuid4

    trace_id = str(uuid4())
    invocation_id = str(uuid4())
    start = datetime.now(UTC) - timedelta(days=30)

    async def results(**kwargs: object) -> AsyncIterator[SimpleNamespace]:
        assert str(invocation_id) in str(kwargs["filter"])
        assert '"thread_id"' in str(kwargs["filter"])
        minimum = kwargs["min_start_time"]
        assert isinstance(minimum, datetime) and minimum < start
        yield SimpleNamespace(trace_id=trace_id)

    client = SimpleNamespace(runs=SimpleNamespace(query=Mock(side_effect=results)))
    monkeypatch.setattr(langsmith, "_build_langsmith_client", lambda: client)
    monkeypatch.setattr(langsmith, "_resolve_project_id_by_name", AsyncMock(return_value="project"))
    assert await langsmith.LangSmithLocator.review_trace_id("reviewer", invocation_id) == trace_id


async def test_finding_feedback_skips_a_missing_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    from uuid import uuid4

    from langsmith.utils import LangSmithNotFoundError

    client = SimpleNamespace(
        read_run=AsyncMock(side_effect=LangSmithNotFoundError("expired")),
        read_feedback=AsyncMock(),
        create_feedback=AsyncMock(),
    )
    monkeypatch.setattr(langsmith, "_build_langsmith_client", lambda: client)
    await langsmith.LangSmithLocator.finding_feedback(str(uuid4()), 901, 1)
    client.create_feedback.assert_not_awaited()
