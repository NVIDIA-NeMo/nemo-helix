# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add access_key_workspaces

Revision ID: d3a9c6e1f2b4
Revises: e8f2b7c41a90
Create Date: 2026-11-10 17:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d3a9c6e1f2b4"
down_revision: Union[str, Sequence[str], None] = "e8f2b7c41a90"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "access_key_workspaces",
        sa.Column("credential_id", sa.String(length=255), nullable=False),
        sa.Column("workspace_id", sa.String(length=255), nullable=False),
        sa.Column("issued_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["credential_id"], ["account_credentials.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("credential_id", "workspace_id"),
    )
    op.create_index(
        "idx_access_key_workspaces_workspace_issued",
        "access_key_workspaces",
        ["workspace_id", sa.text("issued_at DESC"), "credential_id"],
    )
    op.create_index(
        "idx_account_credentials_type_issued",
        "account_credentials",
        ["credential_type", sa.text("issued_at DESC"), "id"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("idx_account_credentials_type_issued", table_name="account_credentials")
    op.drop_index("idx_access_key_workspaces_workspace_issued", table_name="access_key_workspaces")
    op.drop_table("access_key_workspaces")
