# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared wire-level constants and ID conversions for Platform agent sessions."""

from uuid import UUID

import base58

SESSION_ID_HEADER = "X-Nemo-Session-Id"
"""HTTP header carrying the session UUID; legacy and internal requests use entity IDs."""

_SESSION_ID_PREFIX = "agent-session-"


def session_id_to_uuid(session_id: str) -> str:
    """Decode a persisted agent-session entity ID into its UUID string."""
    if not session_id.startswith(_SESSION_ID_PREFIX):
        raise ValueError("Expected an agent-session entity ID")
    suffix = session_id.removeprefix(_SESSION_ID_PREFIX)
    return str(UUID(bytes=base58.b58decode(suffix)))


def session_uuid_to_id(session_uuid: str) -> str:
    """Encode a session UUID as a persisted agent-session entity ID."""
    suffix = base58.b58encode(UUID(session_uuid).bytes).decode("ascii")
    return f"{_SESSION_ID_PREFIX}{suffix}"
