"""Record finding reaction publications"""

from alembic import op

revision = "425a9a554e5b"
down_revision = "3047efb4bf0d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE review_finding_publication (
            id uuid PRIMARY KEY,
            pull_request_id uuid NOT NULL REFERENCES pull_request(id) ON DELETE CASCADE,
            comment_id bigint NOT NULL UNIQUE,
            thread_id text NOT NULL,
            invocation_id text NOT NULL,
            trace_id text,
            ten_day_checked boolean NOT NULL DEFAULT false,
            created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            created_by uuid REFERENCES users(id) ON DELETE SET NULL,
            updated_by uuid REFERENCES users(id) ON DELETE SET NULL
        )
    """)
    op.execute(
        "CREATE INDEX review_finding_publication_pr_idx ON review_finding_publication (pull_request_id)"
    )


def downgrade() -> None:
    raise NotImplementedError
