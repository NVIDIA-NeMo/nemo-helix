# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository interface for Intake sessions."""

from abc import ABC, abstractmethod

from nhx.intake.spans.domain import IntakeSession


class TraceIndexCollisionError(Exception):
    """Deleting roots would tombstone a trace-index key with conflicting owners."""

    def __init__(self, workspace: str, session_id: str) -> None:
        super().__init__(
            f"Session {workspace}/{session_id} cannot be deleted because its root spans have conflicting "
            "trace-index ownership"
        )


class SessionRepository(ABC):
    """Domain-facing interface for session persistence."""

    @abstractmethod
    async def rename_evaluation_references(self, *, workspace: str, original_name: str, deleted_name: str) -> None:
        """Relink stored span versions and trace-index entries, including tombstones."""
        pass

    @abstractmethod
    async def list_evaluation_session_ids(
        self, *, workspace: str, evaluation_name: str, after: str | None, limit: int
    ) -> list[str]:
        """Page distinct current session IDs, including evaluation references on child spans."""
        pass

    @abstractmethod
    async def get_session(self, *, workspace: str, session_id: str) -> IntakeSession | None:
        """Return one session, or ``None`` when it has no current spans."""
        pass

    @abstractmethod
    async def has_live_trace_index(self, *, workspace: str, session_id: str) -> bool:
        """Whether index rows still need cleanup, even when spans are tombstoned."""
        pass

    @abstractmethod
    async def get_evaluation_names(
        self, *, workspace: str, session_id: str, include_deleted: bool = False
    ) -> list[str]:
        """Return every evaluation name referenced by the session's current spans."""
        pass

    @abstractmethod
    async def delete_session(self, *, workspace: str, session_id: str) -> None:
        """Soft-delete current spans, rejecting conflicting trace-index ownership before any write."""
        pass
