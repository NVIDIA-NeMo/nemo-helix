# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add workspace deletion_error

Revision ID: e8f2b7c41a90
Revises: c7e1a4b90d33
Create Date: 2026-10-19 10:46:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e8f2b7c41a90"
down_revision: Union[str, Sequence[str], None] = "c7e1a4b90d33"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("workspaces", sa.Column("deletion_error", sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("workspaces", "deletion_error")
