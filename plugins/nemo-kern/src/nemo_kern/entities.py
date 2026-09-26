# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kern plugin entity definitions — stored in the NeMo Helix entity store.

Only the ``KernSession`` entity lives here; it is managed by
:class:`~nemo_kern.controller.KernSessionController` which upserts sessions
observed from the kern status socket.
"""

from __future__ import annotations

from nemo_helix_plugin.entity import NemoEntity


class KernSession(NemoEntity, entity_type="kern_session"):
    """A live (or recently active) kern agent session.

    Fields are derived from the NDJSON messages emitted on the kern status
    socket (see ``packages/status/protocol.ts``):

    - ``hello`` message supplies ``agent``, ``model``.
    - ``state`` message supplies ``status``, ``ctx_fill``, ``cost``.
    - ``hold`` message supplies ``hold`` (the current hold summary, or None).
    - ``updated_at`` is set to the local epoch timestamp of the last observed
      message for this session.
    """

    agent: str = ""
    """Agent identifier from the ``hello`` message (``agent`` field)."""

    session_path: str = ""
    """The pi session file path (the stable session id used by the controller)."""


    model: str | None = None
    """Model string from the ``hello`` message, e.g. ``"anthropic/claude-opus-4-8"``."""

    status: str = "idle"
    """Working state from the latest ``state`` message: ``"working"`` or ``"idle"``."""

    ctx_fill: float | None = None
    """Context fill fraction 0..1 from the latest ``state`` message (``ctx`` field)."""

    cost: float = 0.0
    """Cumulative session cost from the latest ``state`` message."""

    hold: str | None = None
    """Summary of the current active hold, or ``None`` when no hold is active."""

    last_seen: float = 0.0
    """Unix epoch timestamp (seconds) of the last observed message for this session."""
