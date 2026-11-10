"""GitHub finding reactions, correlated with the review invocation that posted them."""

import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid7

import httpx2
from sqlalchemy import BigInteger, ForeignKey, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Mapped, mapped_column

from openswe.database import postgres
from openswe.database.orm import NOW, Base
from openswe.github.http import GitHubClient, RepoClient
from openswe.github.org_membership import INTERNAL_BOT_LOGINS
from openswe.github.pull_requests import PullRequest
from openswe.jobs import queue
from openswe.jobs.queue import Job
from openswe.utils.langsmith import LangSmithLocator

logger = logging.getLogger(__name__)
REACTION_JOB = "review_finding_reactions"
REACTION_SWEEP = "review_finding_reactions_daily"


class FindingPublication(Base):
    __tablename__ = "review_finding_publication"

    pull_request_id: Mapped[UUID] = mapped_column(ForeignKey("pull_request.id", ondelete="CASCADE"))
    comment_id: Mapped[int] = mapped_column(BigInteger, unique=True)
    thread_id: Mapped[str]
    invocation_id: Mapped[str]
    id: Mapped[UUID] = mapped_column(primary_key=True, default_factory=uuid7)
    trace_id: Mapped[str | None] = mapped_column(default=None)
    ten_day_checked: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(server_default=NOW, init=False)
    updated_at: Mapped[datetime] = mapped_column(server_default=NOW, init=False)
    created_by: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"), default=None)
    updated_by: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"), default=None)

    @classmethod
    async def record(
        cls,
        owner: str,
        repo: str,
        number: int,
        thread_id: str,
        invocation_id: str,
        comment_ids: list[int],
    ) -> None:
        if not comment_ids:
            return
        pull = await PullRequest(owner=owner, repo=repo, number=number).ensure()
        async with postgres.session() as session:
            await session.execute(
                insert(cls)
                .values(
                    [
                        {
                            "id": uuid7(),
                            "pull_request_id": pull.id,
                            "comment_id": comment_id,
                            "thread_id": thread_id,
                            "invocation_id": invocation_id,
                        }
                        for comment_id in comment_ids
                    ]
                )
                .on_conflict_do_nothing(index_elements=[cls.comment_id])
            )

    @staticmethod
    async def schedule_close(owner: str, repo: str, number: int) -> None:
        await queue.schedule(
            REACTION_JOB,
            {"owner": owner, "repo": repo, "number": number},
            key=f"{REACTION_JOB}:{owner.lower()}/{repo.lower()}#{number}",
        )

    @classmethod
    async def collect(cls, owner: str, repo: str, number: int, *, aged: bool = False) -> None:
        pull = await PullRequest.get(owner, repo, number)
        if pull is None:
            return
        query = select(cls).where(cls.pull_request_id == pull.id)
        if aged:
            query = query.where(
                cls.created_at <= datetime.now(UTC) - timedelta(days=10),
                cls.ten_day_checked.is_(False),
            )
        async with postgres.session() as session:
            publications = list((await session.scalars(query)).all())
        if not publications:
            return
        async with GitHubClient.as_app(owner, repo) as github:
            repository = github.repo(owner, repo)
            if aged:
                current = await repository.pull_request(number).pull()
                if current.get("state") != "open":
                    return
            for publication in publications:
                await publication.collect_reactions(repository)
                if aged:
                    async with postgres.session() as session:
                        await session.execute(
                            update(cls)
                            .where(cls.id == publication.id)
                            .values(
                                ten_day_checked=True,
                                updated_at=datetime.now(UTC),
                                updated_by=None,
                            )
                        )

    async def collect_reactions(self, repo: RepoClient) -> None:
        try:
            reactions = await repo.review_comment_reactions(self.comment_id)
        except httpx2.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            logger.info("Finding comment is unavailable", extra={"comment_id": self.comment_id})
            return
        qualifying = []
        for reaction in reactions:
            user = reaction.get("user")
            login = user.get("login") if isinstance(user, dict) else None
            if isinstance(login, str) and login.lower() not in INTERNAL_BOT_LOGINS:
                qualifying.append(reaction)
        if not qualifying:
            return
        if self.trace_id is None:
            self.trace_id = await LangSmithLocator.review_trace_id(
                self.thread_id, self.invocation_id
            )
            if self.trace_id is None:
                logger.info(
                    "Review trace is unavailable", extra={"invocation_id": self.invocation_id}
                )
                return
            async with postgres.session() as session:
                await session.execute(
                    update(type(self))
                    .where(type(self).id == self.id)
                    .values(
                        trace_id=self.trace_id,
                        updated_at=datetime.now(UTC),
                        updated_by=None,
                    )
                )
        await LangSmithLocator.finding_feedback(
            self.trace_id,
            self.comment_id,
            int(any(reaction.get("content") == "+1" for reaction in qualifying)),
        )

    @classmethod
    async def daily(cls, _job: Job) -> None:
        async with postgres.session() as session:
            pulls = list(
                (
                    await session.scalars(
                        select(PullRequest)
                        .join(cls, cls.pull_request_id == PullRequest.id)
                        .where(
                            cls.created_at <= datetime.now(UTC) - timedelta(days=10),
                            PullRequest.state == "open",
                            cls.ten_day_checked.is_(False),
                        )
                        .distinct()
                    )
                ).all()
            )
        for pull in pulls:
            await queue.schedule(
                REACTION_JOB,
                {"owner": pull.owner, "repo": pull.repo, "number": pull.number, "aged": True},
                key=f"{REACTION_JOB}:aged:{pull.id}",
            )

    @classmethod
    async def run(cls, job: Job) -> None:
        owner, repo, number = job.payload["owner"], job.payload["repo"], job.payload["number"]
        if not isinstance(owner, str) or not isinstance(repo, str) or not isinstance(number, int):
            raise ValueError("Invalid finding reactions job")
        await cls.collect(owner, repo, number, aged=job.payload.get("aged") is True)
