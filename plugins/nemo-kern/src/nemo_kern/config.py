# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the kern plugin.

All paths default to the conventional kern data locations on the host (user max).
Override via environment variables:

    NHX_KERN_DECISIONS_DB=/custom/path/to/decisions.db
    NHX_KERN_FLAGS_DIR=/custom/path/to/kern-sleep/flags
    NHX_KERN_PROPOSALS_FILE=/custom/path/to/core-proposals.md
    NHX_KERN_MEMORY_DIR=/custom/path/to/pi-hermes-memory/entries
    NHX_KERN_STATUS_SOCK=/custom/path/to/status.sock
    NHX_KERN_ACK_FILE=/custom/path/to/studio/acks.json
    NHX_KERN_SESSION_STALE_SECONDS=300
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from nemo_helix_plugin.config import NemoConfig
from pydantic import Field

_AGENT_DIR = Path.home() / ".pi" / "agent"


class KernConfig(NemoConfig):
    """Configuration for the NeMo kern plugin.

    All fields have host-appropriate defaults so the plugin runs out-of-the-box
    without any operator configuration on user max's machine.
    """

    plugin_name: ClassVar[str] = "kern"
    plugin_description: ClassVar[str] = "Configuration for the NeMo kern agent-fleet plugin."

    decisions_db: Path = Field(
        default=_AGENT_DIR / "pi-kern" / "decisions.db",
        description=(
            "Path to the kern gate decisions SQLite database. "
            "Override with NHX_KERN_DECISIONS_DB."
        ),
    )

    flags_dir: Path = Field(
        default=_AGENT_DIR / "kern-sleep" / "flags",
        description=(
            "Directory containing kern sleep flag JSON files. "
            "Override with NHX_KERN_FLAGS_DIR."
        ),
    )

    proposals_file: Path = Field(
        default=_AGENT_DIR / "kern-sleep" / "core-proposals.md",
        description=(
            "Path to the kern sleep core-proposals.md file. "
            "Override with NHX_KERN_PROPOSALS_FILE."
        ),
    )

    memory_dir: Path = Field(
        default=_AGENT_DIR / "pi-hermes-memory" / "entries",
        description=(
            "Directory containing kmem memory entry markdown files. "
            "Override with NHX_KERN_MEMORY_DIR."
        ),
    )

    status_sock: Path = Field(
        default=_AGENT_DIR / "kern" / "status.sock",
        description=(
            "Unix socket path for the kern live-session NDJSON stream. "
            "Override with NHX_KERN_STATUS_SOCK."
        ),
    )

    status_log: Path = Field(
        default=_AGENT_DIR / "kern" / "status.jsonl",
        description=(
            "NDJSON mirror file appended by the pi-kern-status extension "
            "(one line per status message, each stamped with _session_id). "
            "The controller tails this file for live session state. "
            "Override with NHX_KERN_STATUS_LOG."
        ),
    )

    workspace: str = Field(
        default="default",
        description=(
            "Platform workspace for kern session entities. "
            "Override with NHX_KERN_WORKSPACE."
        ),
    )

    ack_file: Path = Field(
        default=_AGENT_DIR / "kern" / "studio" / "acks.json",
        description=(
            "JSON ledger file recording flag ack state. "
            "Override with NHX_KERN_ACK_FILE."
        ),
    )

    session_stale_seconds: float = Field(
        default=300.0,
        description=(
            "Number of seconds after last update before a session is pruned "
            "from the live view. Override with NHX_KERN_SESSION_STALE_SECONDS."
        ),
    )
