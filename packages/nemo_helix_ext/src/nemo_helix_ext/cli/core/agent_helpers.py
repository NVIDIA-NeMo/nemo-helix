# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent mode helpers for the NeMo Helix CLI.

Maps CLI command paths to contextual helper strings shown when --agent-mode is active.
Edit the AGENT_HELPERS dict to evolve what agents see after running commands.
"""

from __future__ import annotations

# Command path -> list of helper strings.
# Command path is derived from ctx.command_path minus the program name,
# e.g. "nemo workspaces list" becomes "workspaces list".
#
# Lookup order: exact match -> parent prefix -> "*" fallback.
AGENT_HELPERS: dict[str, list[str]] = {
    "quickstart": [
        "To learn more about setup, run: nemo docs get-started/setup",
    ],
    "workspaces": [
        "To learn more about workspaces, run: nemo docs get-started/concepts/workspaces",
    ],
    "projects": [
        "To learn more about projects, run: nemo docs get-started/concepts/projects",
    ],
    "entities": [
        "To learn more about entities, run: nemo docs get-started/concepts/entities",
    ],
    "models": [
        "To learn more about models and inference, run: nemo docs run-inference/tutorials/deploy-models",
    ],
    "inference": [
        "To learn more about inference, run: nemo docs run-inference/tutorials/index",
    ],
    "guardrail": [
        "To learn more about guardrails, run: nemo docs guardrails/index",
    ],
    "data-designer": [
        "To learn more about data designer, run: nemo docs data-designer/index",
    ],
    "audit": [
        "To learn more about auditor, run: nemo docs audit/index",
    ],
    "safe-synthesizer": [
        "To learn more about safe synthesizer, run: nemo docs safe-synthesizer/about/index",
    ],
    "files": [
        "To learn more about file management, run: nemo docs get-started/concepts/manage-files",
    ],
    "secrets": [
        "To learn more about secrets, run: nemo docs get-started/concepts/manage-secrets",
    ],
    "auth": [
        "To learn more about authentication, run: nemo docs auth/index",
    ],
    "config": [
        "To learn more about CLI configuration, run: nemo docs cli/configuration",
    ],
    "*": [
        "Run `nemo docs <path>` to read documentation. Run `nemo docs --list` to see available topics.",
    ],
}


def get_agent_helpers(command_path: str) -> list[str]:
    """Look up helpers for a command path with fallback.

    Tries exact match, then progressively shorter prefixes, then "*".
    """
    # Try exact match
    if command_path in AGENT_HELPERS:
        return AGENT_HELPERS[command_path]

    # Try progressively shorter prefixes
    parts = command_path.split()
    while parts:
        parts.pop()
        prefix = " ".join(parts)
        if prefix in AGENT_HELPERS:
            return AGENT_HELPERS[prefix]

    # Global fallback
    return AGENT_HELPERS.get("*", [])
