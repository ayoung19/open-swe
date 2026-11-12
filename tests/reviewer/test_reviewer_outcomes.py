from typing import Any, cast

from openswe.review.findings import Finding
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


async def test_trace_feedback_tracks_published_findings_after_delayed_outcomes() -> None:
    from datetime import UTC, datetime
    from unittest.mock import AsyncMock, patch
    from uuid import uuid4

    from openswe.jobs.queue import Job
    from openswe.review import findings as finding_store
    from openswe.review.findings import FindingState

    published = [
        {**_finding(), "id": "first", "github_review_id": 10, "status": "open"},
        {**_finding(), "id": "second", "github_review_id": 10, "status": "open"},
        {
            **_finding(),
            "id": "later-review",
            "github_review_id": 11,
            "github_review_run_id": "run_2",
            "status": "dismissed",
        },
        {**_finding(), "id": "unpublished", "github_review_id": None, "status": "resolved"},
    ]
    job = Job(
        id=uuid4(),
        kind=FindingState.FEEDBACK_JOB,
        payload={"thread_id": "reviewer-thread"},
        due_at=datetime.now(UTC),
        cron=None,
        timezone="UTC",
        key=None,
        metadata={},
        attempts=0,
        created_at=datetime.now(UTC),
    )
    with (
        patch.object(finding_store, "list_findings", AsyncMock(return_value=published)),
        patch.object(
            finding_store, "create_langsmith_feedback", AsyncMock(return_value=True)
        ) as saved,
    ):
        await FindingState.refresh_feedback(job)
        assert [(call.args, call.kwargs["score"]) for call in saved.await_args_list] == [
            (("run_1", "findings_acted_on"), 0.0),
            (("run_2", "findings_acted_on"), 0.0),
        ]
        published[0]["status"] = "resolved"
        saved.reset_mock()
        await FindingState.refresh_feedback(job)
        assert [(call.args, call.kwargs["score"]) for call in saved.await_args_list] == [
            (("run_1", "findings_acted_on"), 0.5),
            (("run_2", "findings_acted_on"), 0.0),
        ]
        published[1]["status"] = "resolved"
        await FindingState.refresh_feedback(job)
        assert saved.await_args_list[-2].kwargs["score"] == 1.0
        published.clear()
        saved.reset_mock()
        await FindingState.refresh_feedback(job)
        saved.assert_not_awaited()
