# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add account profiles and preferences

Revision ID: d3a9f1e27b54
Revises: e8f2b7c41a90
Create Date: 2026-11-10 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d3a9f1e27b54"
down_revision: Union[str, Sequence[str], None] = "e8f2b7c41a90"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "account_profiles",
        sa.Column("account_id", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=256), nullable=True, comment="User override"),
        sa.Column("idp_display_name", sa.String(length=512), nullable=True),
        sa.Column("idp_avatar_url", sa.String(length=2048), nullable=True),
        sa.Column(
            "primary_identity_id",
            sa.String(length=255),
            nullable=True,
            comment="Identity whose claims feed idp_* fields and accounts.primary_email",
        ),
        sa.Column("timezone", sa.String(length=64), nullable=True),
        sa.Column("locale", sa.String(length=35), nullable=True),
        sa.Column("last_login_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("db_version", sa.Integer(), server_default="1", nullable=False),
        sa.ForeignKeyConstraint(["account_id"], ["accounts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["primary_identity_id"], ["account_identities.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("account_id"),
    )
    op.create_table(
        "account_preferences",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("account_id", sa.String(length=255), nullable=False),
        sa.Column("namespace", sa.String(length=128), nullable=False),
        sa.Column("workspace", sa.String(length=255), nullable=True, comment="Null for global preferences"),
        sa.Column("value", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("db_version", sa.Integer(), server_default="1", nullable=False),
        sa.ForeignKeyConstraint(["account_id"], ["accounts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace"], ["workspaces.name"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_account_preferences_global",
        "account_preferences",
        ["account_id", "namespace"],
        unique=True,
        sqlite_where=sa.text("workspace IS NULL"),
        postgresql_where=sa.text("workspace IS NULL"),
    )
    op.create_index(
        "uq_account_preferences_workspace",
        "account_preferences",
        ["account_id", "namespace", "workspace"],
        unique=True,
        sqlite_where=sa.text("workspace IS NOT NULL"),
        postgresql_where=sa.text("workspace IS NOT NULL"),
    )
    op.create_index("idx_account_preferences_workspace", "account_preferences", ["workspace"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("idx_account_preferences_workspace", table_name="account_preferences")
    op.drop_index("uq_account_preferences_workspace", table_name="account_preferences")
    op.drop_index("uq_account_preferences_global", table_name="account_preferences")
    op.drop_table("account_preferences")
    op.drop_table("account_profiles")
