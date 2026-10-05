# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository interface for Intake session reads."""

from abc import ABC, abstractmethod

from nhx.intake.spans.domain import IntakeSession


class SessionRepository(ABC):
    """Domain-facing interface for session persistence."""

    @abstractmethod
    async def get_session(self, *, workspace: str, session_id: str) -> IntakeSession | None:
        """Return one session, or ``None`` when it has no current spans."""
        pass
