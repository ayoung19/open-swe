"""Track finding tool runs"""

from alembic import op

revision = "e756c44ab4ed"
down_revision = "3047efb4bf0d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE pull_request_finding ADD COLUMN originating_tool_runs jsonb NOT NULL DEFAULT '{}'::jsonb"
    )


def downgrade() -> None:
    raise NotImplementedError
