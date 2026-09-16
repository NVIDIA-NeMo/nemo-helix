# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQLAlchemy database models for entities service.

These are the database table definitions, separate from Pydantic API schemas.
"""

from datetime import datetime, timezone
from typing import Optional

from nhx.core.entities.entities import Entity, Workspace, WorkspaceDeletionStage
from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.engine.default import DefaultExecutionContext
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .base import Base


def _utcnow() -> datetime:
    """Return current UTC time as a naive datetime with microsecond precision."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def same_as(column: str):
    def default(ctx: DefaultExecutionContext):
        return ctx.get_current_parameters().get(column)

    return default


class DBWorkspace(Base):
    """SQLAlchemy model for Workspace database table."""

    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(255), primary_key=True, comment="System-generated UUID")
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, comment="User-provided workspace name")
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="Optional description")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, server_default=func.now(), nullable=False)
    created_by: Mapped[str] = mapped_column(String(255), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, server_default=func.now(), onupdate=_utcnow, nullable=False
    )
    updated_by: Mapped[str] = mapped_column(String(255), nullable=True, default=same_as("created_by"))

    deletion_stage: Mapped[str | None] = mapped_column(String(100), nullable=True)

    __table_args__ = (Index("idx_workspaces_created", "created_at"),)

    def to_pydantic(self) -> Workspace:
        """Convert SQLAlchemy model to Pydantic model."""
        workspace = Workspace(
            id=self.id,
            name=self.name,
            description=self.description,
            created_at=self.created_at,
            created_by=self.created_by,
            updated_at=self.updated_at,
            updated_by=self.updated_by,
        )
        if self.deletion_stage is not None:
            workspace._deletion_stage = WorkspaceDeletionStage(self.deletion_stage)
        return workspace


class DBEntity(Base):
    """SQLAlchemy model for Entity database table.

    Generic entity storage using JSON for schema-agnostic data.
    Supports both PostgreSQL and SQLite backends.

    Parent-scoped uniqueness:
    - Root entities (parent IS NULL): unique within (workspace, entity_type, name)
    - Child entities (parent IS NOT NULL): unique within (workspace, entity_type, parent, name)
    """

    __tablename__ = "entities"

    id: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
        comment="Compound identifier (entity_type-base58uuid)",
    )

    workspace: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("workspaces.name", ondelete="CASCADE"),
        nullable=False,
        comment="Workspace name",
    )

    entity_type: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Entity type (e.g., 'customization_config')",
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, comment="Entity name")

    parent: Mapped[Optional[str]] = mapped_column(
        String(255),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=True,
        comment="Parent entity ID for nested entities",
    )

    project: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="The name of the project associated with this entity",
    )

    data: Mapped[dict] = mapped_column(JSON, nullable=False, comment="Entity-specific data (JSON)")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, server_default=func.now(), nullable=False)
    created_by: Mapped[str] = mapped_column(String(255), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, server_default=func.now(), onupdate=_utcnow, nullable=False
    )
    updated_by: Mapped[str] = mapped_column(String(255), nullable=True, default=same_as("created_by"))

    db_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default="1",
        comment="Database version of the entity for optimistic locking.",
    )

    __mapper_args__ = {"version_id_col": db_version}

    __table_args__ = (
        # Root entity uniqueness: (workspace, entity_type, name) where parent IS NULL
        Index(
            "uq_entities_root",
            "workspace",
            "entity_type",
            "name",
            unique=True,
            sqlite_where=parent.is_(None),
            postgresql_where=parent.is_(None),
        ),
        # Child entity uniqueness: (workspace, entity_type, parent, name) where parent IS NOT NULL
        Index(
            "uq_entities_child",
            "workspace",
            "entity_type",
            "parent",
            "name",
            unique=True,
            sqlite_where=parent.isnot(None),
            postgresql_where=parent.isnot(None),
        ),
        Index("idx_entities_workspace", "workspace"),
        Index("idx_entities_type", "workspace", "entity_type"),
        Index("idx_entities_lookup", "workspace", "entity_type", "name"),
        Index("idx_entities_parent", "parent"),
        Index("idx_entities_project", "project"),
        Index("idx_entities_updated", "updated_at"),
        Index("idx_entities_created", "created_at"),
    )

    def to_pydantic(self) -> Entity:
        """Convert SQLAlchemy model to Pydantic model."""
        return Entity(
            id=self.id,
            workspace=self.workspace,
            entity_type=self.entity_type,
            name=self.name,
            parent=self.parent,
            project=self.project,
            data=self.data,
            created_at=self.created_at,
            created_by=self.created_by,
            updated_at=self.updated_at,
            updated_by=self.updated_by,
            db_version=self.db_version,
        )
