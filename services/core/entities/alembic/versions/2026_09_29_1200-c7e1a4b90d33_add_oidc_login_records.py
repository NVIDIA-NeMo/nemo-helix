# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add account credentials and OIDC login transactions

Revision ID: c7e1a4b90d33
Revises: b4c8d19f6a2e
Create Date: 2026-09-29 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c7e1a4b90d33"
down_revision: Union[str, Sequence[str], None] = "b4c8d19f6a2e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "auth_login_transactions",
        sa.Column("record_hash", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("record_hash"),
        sa.Index("idx_auth_login_transactions_expires", "expires_at"),
    )
    op.create_table(
        "account_credentials",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("owner_account_id", sa.String(length=255), nullable=False),
        sa.Column("subject_account_id", sa.String(length=255), nullable=False),
        sa.Column("account_identity_id", sa.String(length=255), nullable=True),
        sa.Column("credential_type", sa.String(length=32), nullable=False),
        sa.Column("lookup_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="ACTIVE", nullable=False),
        sa.Column("issued_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("last_used_at", sa.DateTime(), nullable=True),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.Column("revoked_by", sa.String(length=255), nullable=True),
        sa.Column("public_metadata", sa.JSON(), nullable=False),
        sa.Column("encrypted_payload", sa.Text(), nullable=True),
        sa.Column("db_version", sa.Integer(), server_default="1", nullable=False),
        sa.ForeignKeyConstraint(["account_identity_id"], ["account_identities.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["owner_account_id"], ["accounts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["subject_account_id"], ["accounts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("credential_type", "lookup_hash", name="uq_account_credentials_type_lookup_hash"),
    )
    op.create_index("idx_account_credentials_owner", "account_credentials", ["owner_account_id"])
    op.create_index("idx_account_credentials_subject", "account_credentials", ["subject_account_id"])
    op.create_index("idx_account_credentials_identity", "account_credentials", ["account_identity_id"])
    op.create_index("idx_account_credentials_type_status", "account_credentials", ["credential_type", "status"])
    op.create_index("idx_account_credentials_expires", "account_credentials", ["expires_at"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("account_credentials")
    op.drop_table("auth_login_transactions")
