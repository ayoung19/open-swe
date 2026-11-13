from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from langchain_core.tools import StructuredTool
from langchain_core.tracers.langchain import LangChainTracer
from langgraph.runtime import Runtime
from langsmith import Client

from openswe.agent_api.runs import in_graph_context
from openswe.review.findings import Finding, FindingStatus
from openswe.utils import reviewer_outcomes
from openswe.utils.reviewer_outcomes import (
    FALSE_POSITIVE,
    TRUE_POSITIVE,
    upsert_finding_outcome,
)


class _FakeDataset:
    def __init__(self, dataset_id: str, name: str) -> None:
        self.id = dataset_id
        self.name = name


class _FakeClient:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.updated: list[dict[str, Any]] = []
        self.conflict_once = False

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def read_dataset(self, dataset_name: str) -> _FakeDataset:
        return _FakeDataset("ds_123", dataset_name)

    async def create_example(self, **kwargs: Any) -> None:
        if self.conflict_once:
            self.conflict_once = False
            raise RuntimeError("already exists")
        self.created.append(kwargs)

    def update_example(self, **kwargs: Any) -> None:
        self.updated.append(kwargs)


def _patch_client(monkeypatch, fake: _FakeClient | None) -> None:  # noqa: ANN001
    if fake is None:
        monkeypatch.setattr(reviewer_outcomes, "_outcomes_credentials", lambda: None)
        return
    monkeypatch.setattr(reviewer_outcomes, "_outcomes_credentials", lambda: ("k", "https://api"))
    monkeypatch.setattr(reviewer_outcomes, "async_langsmith_client", lambda *a: fake)
    monkeypatch.setattr(reviewer_outcomes, "sync_langsmith_client", lambda *a: cast(Any, fake))


def _finding() -> Finding:
    return cast(
        Finding,
        {
            "id": "f_abc",
            "file": "app/x.rb",
            "start_line": 10,
            "end_line": 12,
            "side": "RIGHT",
            "diff_hunk": "@@ -1 +1 @@\n-foo\n+bar",
            "title": "NoMethodError on nil",
            "description": "body",
            "severity": "high",
            "confidence": "high",
            "category": "correctness",
            "first_seen_sha": "aaa",
            "github_review_run_id": "run_1",
            "resolution_note": "fixed in def",
        },
    )


async def test_upsert_finding_outcome_builds_payload(monkeypatch) -> None:  # noqa: ANN001
    fake = _FakeClient()
    _patch_client(monkeypatch, fake)

    ok = await upsert_finding_outcome(
        _finding(),
        label=TRUE_POSITIVE,
        label_source="resolved_by_commit",
        repo="o/r",
        pr_number=7,
        pr_url="https://github.com/o/r/pull/7",
        base_sha="aaa",
        head_sha="bbb",
        thread_id="t1",
    )
    assert ok
    assert len(fake.created) == 1
    call = fake.created[0]
    assert call["dataset_id"] == "ds_123"
    assert call["inputs"]["repo"] == "o/r"
    assert call["inputs"]["file"] == "app/x.rb"
    assert call["inputs"]["diff_hunk"].startswith("@@")
    assert call["outputs"]["label"] == TRUE_POSITIVE
    assert call["outputs"]["label_source"] == "resolved_by_commit"
    assert call["outputs"]["finding"]["severity"] == "high"
    assert call["metadata"]["granularity"] == "finding"
    assert call["metadata"]["repo"] == "o/r"
    assert call["metadata"]["run_id"] == "run_1"


async def test_upsert_finding_outcome_updates_on_conflict(monkeypatch) -> None:  # noqa: ANN001
    fake = _FakeClient()
    fake.conflict_once = True
    _patch_client(monkeypatch, fake)

    ok = await upsert_finding_outcome(
        _finding(), label=FALSE_POSITIVE, label_source="dismissed", repo="o/r"
    )
    assert ok
    assert not fake.created
    assert len(fake.updated) == 1


class TestFindingDismissalFeedback:
    @pytest.mark.parametrize(
        ("status", "age", "human", "expected"),
        [
            ("dismissed", 0, True, 1.0),
            ("dismissed", 7 * 86400, True, 1.0),
            ("dismissed", 7 * 86400 + 1, True, 0.0),
            ("dismissed", -1, True, 0.0),
            ("dismissed", 60, False, 0.0),
            ("resolved", 60, True, 0.0),
            ("open", 60, True, 0.0),
        ],
    )
    async def test_feedback_scores_originating_runs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        status: FindingStatus,
        age: int,
        human: bool,
        expected: float,
    ) -> None:
        feedback = AsyncMock(return_value=True)
        monkeypatch.setattr(reviewer_outcomes, "create_langsmith_feedback", feedback)
        start = datetime(2026, 1, 1, tzinfo=UTC)
        finding = _finding()
        finding.update(
            status=status,
            originating_tool_runs={"tool-run": start.isoformat()},
            last_human_reply_at=(start + timedelta(seconds=age)).isoformat(),
            last_human_reply_author="reviewer" if human else None,
            last_human_reply_body="This finding is noise" if human else None,
        )
        await reviewer_outcomes.FindingDismissalFeedback.record(finding)
        feedback.assert_awaited_once_with("tool-run", "finding_dismissed", score=expected)

    @staticmethod
    async def capture(runtime: Runtime) -> object:
        tool = StructuredTool.from_function(
            coroutine=reviewer_outcomes.FindingDismissalFeedback.start,
            name="add_finding",
            description="Record finding origins",
        )
        return await tool.ainvoke({})

    async def test_initializes_each_matching_span(self, monkeypatch: pytest.MonkeyPatch) -> None:
        feedback = AsyncMock(return_value=True)
        monkeypatch.setattr(reviewer_outcomes, "create_langsmith_feedback", feedback)
        client = MagicMock(spec=Client)
        tracer = LangChainTracer(client=client, project_name="test")
        remote_id = str(uuid4())
        remote_start = datetime(2026, 1, 1, tzinfo=UTC)
        result = await in_graph_context(
            {
                "run_name": "add_finding",
                "callbacks": [tracer],
                "configurable": {
                    "tool_spans": [{"run_id": remote_id, "started_at": remote_start.isoformat()}]
                },
            },
            self.capture,
        )
        assert isinstance(result, dict)
        assert result[remote_id] == remote_start.isoformat()
        assert len(result) == 4
        assert {call.args[0] for call in feedback.await_args_list} == set(result)
        assert all(call.kwargs["score"] == 0.0 for call in feedback.await_args_list)
