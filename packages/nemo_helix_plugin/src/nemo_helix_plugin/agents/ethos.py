# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where an agent's Ethos lives, by convention.

Each agent may have exactly one Ethos fileset, named after the agent, holding the
human-readable ``ETHOS.md``. The location is derived from the agent's
workspace and name rather than stored on the agent.
"""

from __future__ import annotations

from nemo_helix_plugin.refs import FilesetRef

ETHOS_FILENAME = "ETHOS.md"
"""Canonical filename inside the agent's Ethos fileset."""


def ethos_fileset_name(agent_name: str) -> str:
    """Return the conventional fileset name holding an agent's Ethos."""
    return f"{agent_name}-ethos"


def ethos_file_ref(workspace: str, agent_name: str) -> FilesetRef:
    """Return the canonical file ref ``workspace/<name>-ethos#ETHOS.md``."""
    return FilesetRef(f"{workspace}/{ethos_fileset_name(agent_name)}#{ETHOS_FILENAME}")
