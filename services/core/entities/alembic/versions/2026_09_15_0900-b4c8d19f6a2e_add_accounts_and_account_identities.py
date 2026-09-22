# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add accounts and account identities

Revision ID: b4c8d19f6a2e
Revises: 54a9f4ccf8b1
Create Date: 2026-09-15 09:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b4c8d19f6a2e"
down_revision: Union[str, Sequence[str], None] = "54a9f4ccf8b1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "accounts",
        sa.Column("id", sa.String(length=255), nullable=False, comment="Stable account identifier"),
        sa.Column("type", sa.String(length=32), nullable=False, comment="Account type: user or service"),
        sa.Column(
            "link_key",
            sa.String(length=512),
            nullable=True,
            comment="Optional explicit account-linking key; not used for automatic email linking",
        ),
        sa.Column("display_name", sa.String(length=512), nullable=True),
        sa.Column("primary_email", sa.String(length=320), nullable=True),
        sa.Column("status", sa.String(length=32), server_default="active", nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_accounts_active_link_key",
        "accounts",
        ["link_key"],
        unique=True,
        sqlite_where=sa.text("link_key IS NOT NULL AND status = 'active'"),
        postgresql_where=sa.text("link_key IS NOT NULL AND status = 'active'"),
    )
    op.create_index("idx_accounts_status", "accounts", ["status"], unique=False)
    op.create_index("idx_accounts_type", "accounts", ["type"], unique=False)

    op.create_table(
        "account_identities",
        sa.Column("id", sa.String(length=255), nullable=False, comment="Stable account identity row identifier"),
        sa.Column("account_id", sa.String(length=255), nullable=False),
        sa.Column("issuer", sa.String(length=512), nullable=False),
        sa.Column("subject", sa.String(length=512), nullable=False),
        sa.Column("subject_claim", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="active", nullable=False),
        sa.Column("linked_via", sa.String(length=64), nullable=False),
        sa.Column("linked_by", sa.String(length=255), nullable=True),
        sa.Column("linked_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(), nullable=True),
        sa.Column("link_key_claim_at_link_time", sa.String(length=128), nullable=True),
        sa.Column("link_key_value_at_link_time", sa.String(length=512), nullable=True),
        sa.Column("claims_snapshot", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["account_id"], ["accounts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("issuer", "subject", name="uq_account_identities_issuer_subject"),
    )
    op.create_index("idx_account_identities_account_id", "account_identities", ["account_id"], unique=False)
    op.create_index("idx_account_identities_status", "account_identities", ["status"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("idx_account_identities_status", table_name="account_identities")
    op.drop_index("idx_account_identities_account_id", table_name="account_identities")
    op.drop_table("account_identities")

    op.drop_index("idx_accounts_type", table_name="accounts")
    op.drop_index("idx_accounts_status", table_name="accounts")
    op.drop_index("uq_accounts_active_link_key", table_name="accounts")
    op.drop_table("accounts")
