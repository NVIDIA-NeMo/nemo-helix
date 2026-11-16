# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add workspace limits

Revision ID: f3a9c2d1b7e4
Revises: e8f2b7c41a90
Create Date: 2026-11-10 10:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f3a9c2d1b7e4"
down_revision: Union[str, Sequence[str], None] = "e8f2b7c41a90"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("workspaces", sa.Column("limits", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("workspaces", "limits")
