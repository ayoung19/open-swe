from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from openswe.review.assessment_feedback import FeedbackSubmission, PublishedAssessment
from openswe.review.routes import get_assessment_feedback, submit_assessment_feedback
from openswe.tools.submit_review_assessment_feedback import submit_review_assessment_feedback
from tests.conftest import FakeUserRecords


async def test_feedback_is_per_person_and_published_assessment(
    registry_db: None, user_records: FakeUserRecords
) -> None:
    assessment = PublishedAssessment(
        review_id=123,
        owner="o",
        repo="r",
        pr_number=1,
        head_sha="a" * 40,
        risk_score=1,
        decision="would_approve",
        explanation="Docs only.",
        run_id="c" * 36,
    )
    await assessment.save()
    await assessment.model_copy(update={"review_id": 124, "head_sha": "b" * 40}).save()
    with (
        patch(
            "openswe.review.assessment_feedback.require_repo_access_for_user",
            AsyncMock(return_value="t"),
        ),
        patch(
            "openswe.review.assessment_feedback.create_langsmith_feedback", new_callable=AsyncMock
        ) as trace_feedback,
    ):
        first = await submit_assessment_feedback(
            "o",
            "r",
            1,
            123,
            FeedbackSubmission(rating="helpful", comment="  Clear rationale  "),
            {"sub": "alice"},
        )
        await submit_assessment_feedback(
            "o",
            "r",
            1,
            123,
            FeedbackSubmission(rating="unhelpful", comment="Missed a migration"),
            {"sub": "bob"},
        )
        saved = await get_assessment_feedback("o", "r", 1, 123, {"sub": "alice"})
        assert saved == first
        assert saved.comment == "Clear rationale"
        assert await get_assessment_feedback("o", "r", 1, 124, {"sub": "alice"}) is None
        edited = await submit_assessment_feedback(
            "o",
            "r",
            1,
            123,
            FeedbackSubmission(rating="unhelpful", comment="Changed my mind"),
            {"sub": "alice"},
        )
        assert edited.comment == "Changed my mind"
        bob = await get_assessment_feedback("o", "r", 1, 123, {"sub": "bob"})
        assert bob and bob.comment == "Missed a migration"
        assert trace_feedback.await_count == 3
        assert trace_feedback.await_args_list[0].args == (
            "c" * 36,
            "review_assessment:123:alice",
        )
        assert trace_feedback.await_args_list[0].kwargs["score"] == 1.0
        assert trace_feedback.await_args_list[0].kwargs["comment"] == "Clear rationale"
        assert trace_feedback.await_args_list[1].kwargs["score"] == 0.0
        assert trace_feedback.await_args_list[2].args == trace_feedback.await_args_list[0].args
        assert trace_feedback.await_args_list[2].kwargs["score"] == 0.0
    assert await PublishedAssessment.get(123) == assessment


async def test_trace_failure_does_not_lose_saved_feedback(
    registry_db: None, user_records: FakeUserRecords
) -> None:
    await PublishedAssessment(
        review_id=123,
        owner="o",
        repo="r",
        pr_number=1,
        head_sha="a" * 40,
        risk_score=1,
        decision="would_approve",
        explanation="Docs only.",
        run_id="c" * 36,
    ).save()
    with (
        patch(
            "openswe.review.assessment_feedback.require_repo_access_for_user",
            AsyncMock(return_value="t"),
        ),
        patch(
            "openswe.review.assessment_feedback.create_langsmith_feedback",
            AsyncMock(side_effect=RuntimeError("trace unavailable")),
        ),
    ):
        saved = await submit_assessment_feedback(
            "o", "r", 1, 123, FeedbackSubmission(rating="unhelpful"), {"sub": "alice"}
        )
    with patch(
        "openswe.review.assessment_feedback.require_repo_access_for_user",
        AsyncMock(return_value="t"),
    ):
        assert await get_assessment_feedback("o", "r", 1, 123, {"sub": "alice"}) == saved


async def test_feedback_cannot_cross_repository_scope(
    registry_db: None, user_records: FakeUserRecords
) -> None:
    await PublishedAssessment(
        review_id=123,
        owner="private",
        repo="r",
        pr_number=1,
        head_sha="a" * 40,
        risk_score=1,
        decision="would_approve",
        explanation="Docs only.",
    ).save()
    with (
        patch(
            "openswe.review.assessment_feedback.require_repo_access_for_user",
            AsyncMock(return_value="t"),
        ),
        pytest.raises(HTTPException) as error,
    ):
        await submit_assessment_feedback(
            "public", "r", 1, 123, FeedbackSubmission(rating="helpful"), {"sub": "alice"}
        )
    assert error.value.status_code == 404
    with (
        patch(
            "openswe.review.assessment_feedback.require_repo_access_for_user",
            AsyncMock(side_effect=HTTPException(403)),
        ),
        pytest.raises(HTTPException) as error,
    ):
        await get_assessment_feedback("private", "r", 1, 123, {"sub": "alice"})
    assert error.value.status_code == 403


async def test_feedback_tool_uses_private_owner_and_repo_permission(
    registry_db: None, user_records: FakeUserRecords
) -> None:
    with (
        patch(
            "openswe.tools.submit_review_assessment_feedback.private_credential_login",
            AsyncMock(return_value=None),
        ),
        pytest.raises(ValueError, match="authenticated owner"),
    ):
        await submit_review_assessment_feedback(
            "o", "r", 1, 123, FeedbackSubmission(rating="helpful")
        )

    await PublishedAssessment(
        review_id=123,
        owner="o",
        repo="r",
        pr_number=1,
        head_sha="a" * 40,
        risk_score=1,
        decision="would_approve",
        explanation="Docs only.",
    ).save()
    with (
        patch(
            "openswe.tools.submit_review_assessment_feedback.private_credential_login",
            AsyncMock(return_value="Alice"),
        ),
        patch(
            "openswe.review.assessment_feedback.require_repo_access_for_user",
            AsyncMock(return_value="t"),
        ),
    ):
        saved = await submit_review_assessment_feedback(
            "o", "r", 1, 123, FeedbackSubmission(rating="helpful", comment="Clear")
        )
        assert saved.login == "alice"
        assert await get_assessment_feedback("o", "r", 1, 123, {"sub": "ALICE"}) == saved
    with (
        patch(
            "openswe.tools.submit_review_assessment_feedback.private_credential_login",
            AsyncMock(return_value="alice"),
        ),
        patch(
            "openswe.review.assessment_feedback.require_repo_access_for_user",
            AsyncMock(side_effect=HTTPException(403)),
        ),
        pytest.raises(HTTPException) as error,
    ):
        await submit_review_assessment_feedback(
            "o", "r", 1, 123, FeedbackSubmission(rating="helpful")
        )
    assert error.value.status_code == 403


@pytest.mark.parametrize(
    "reactions,expected",
    [
        ([], None),
        ([{"user": {"login": "open-swe[bot]"}, "content": "+1"}], None),
        ([{"user": {"login": "alice"}, "content": "heart"}], 0),
        (
            [
                {"user": {"login": "alice"}, "content": "-1"},
                {"user": {"login": "bob"}, "content": "+1"},
            ],
            1,
        ),
        (
            [
                {"user": {"login": "open-swe[bot]"}, "content": "+1"},
                {"user": {"login": "alice"}, "content": "-1"},
            ],
            0,
        ),
    ],
)
async def test_finding_reactions_score_the_originating_trace(
    reactions: list[dict[str, object]],
    expected: int | None,
) -> None:
    from uuid import uuid4

    import httpx2

    from openswe.github.http import GitHubClient
    from openswe.review.reaction_feedback import FindingPublication

    publication = FindingPublication(
        pull_request_id=uuid4(),
        comment_id=901,
        thread_id="reviewer",
        invocation_id="invocation",
        trace_id=str(uuid4()),
    )

    async def respond(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/repos/o/r/pulls/comments/901/reactions"
        return httpx2.Response(200, json=reactions)

    with patch(
        "openswe.review.reaction_feedback.LangSmithLocator.finding_feedback", AsyncMock()
    ) as feedback:
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http:
            await publication.collect_reactions(GitHubClient(http).repo("o", "r"))
    if expected is None:
        feedback.assert_not_awaited()
    else:
        assert feedback.await_args.args == (publication.trace_id, 901, expected)


async def test_closed_pr_records_finding_feedback(registry_db: None) -> None:
    from uuid import uuid4

    import httpx2

    from openswe.github import webhook
    from openswe.github.http import GitHubClient
    from openswe.jobs.queue import Job
    from openswe.review.reaction_feedback import FindingPublication

    trace_id = str(uuid4())
    await FindingPublication.record("o", "r", 7, "reviewer", "invocation", [901])

    async def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=[{"user": {"login": "alice"}, "content": "+1"}])

    with (
        patch("openswe.review.reaction_feedback.queue.schedule", AsyncMock()) as scheduled,
        patch("openswe.webhooks.common.get_thread_metadata_safe", AsyncMock(return_value=None)),
    ):
        await webhook.process_github_pr_close(
            {
                "action": "closed",
                "repository": {"owner": {"login": "o"}, "name": "r"},
                "pull_request": {"number": 7},
            }
        )
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http:
        with (
            patch(
                "openswe.review.reaction_feedback.GitHubClient.as_app",
                return_value=GitHubClient(http),
            ),
            patch(
                "openswe.review.reaction_feedback.LangSmithLocator.review_trace_id",
                AsyncMock(return_value=trace_id),
            ),
            patch(
                "openswe.review.reaction_feedback.LangSmithLocator.finding_feedback", AsyncMock()
            ) as feedback,
        ):
            from datetime import UTC, datetime

            job = Job(
                id=uuid4(),
                kind="review_finding_reactions",
                payload=scheduled.await_args.args[1],
                due_at=datetime.now(UTC),
                cron=None,
                timezone="UTC",
                key=None,
                metadata={},
                attempts=0,
                created_at=datetime.now(UTC),
            )
            await FindingPublication.run(job)
    assert feedback.await_args.args == (trace_id, 901, 1)


async def test_deleted_finding_comment_is_skipped() -> None:
    from uuid import uuid4

    import httpx2

    from openswe.github.http import GitHubClient
    from openswe.review.reaction_feedback import FindingPublication

    publication = FindingPublication(
        pull_request_id=uuid4(),
        comment_id=901,
        thread_id="reviewer",
        invocation_id="invocation",
        trace_id=str(uuid4()),
    )

    async def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(404, json={"message": "Not Found"})

    with patch(
        "openswe.review.reaction_feedback.LangSmithLocator.finding_feedback", AsyncMock()
    ) as feedback:
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http:
            await publication.collect_reactions(GitHubClient(http).repo("o", "r"))
    feedback.assert_not_awaited()
