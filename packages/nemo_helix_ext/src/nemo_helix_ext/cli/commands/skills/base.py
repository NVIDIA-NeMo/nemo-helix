# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base types and protocol for agent skill installers."""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

INSTALLED_SKILL_PREFIX = "nemo-"
SAFE_SKILL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_skill_name(skill_name: str) -> None:
    """Reject skill names that cannot be used as a single safe path component."""
    if not SAFE_SKILL_NAME_PATTERN.fullmatch(skill_name):
        raise ValueError(
            f"Invalid skill name {skill_name!r}: expected a single path component "
            "containing only letters, numbers, dots, underscores, and dashes"
        )


def installed_skill_name(skill_name: str) -> str:
    """Return the skill name exposed to downstream coding agents."""
    validate_skill_name(skill_name)
    if skill_name.startswith(INSTALLED_SKILL_PREFIX):
        return skill_name
    return f"{INSTALLED_SKILL_PREFIX}{skill_name}"


@dataclass
class Skill:
    name: str
    description: str
    version: str
    content: str
    raw: str
    source_dir: Path | None = None
    # Entry-point name under ``nemo.skills`` (e.g. ``"agents"``, ``"platform"``).
    # Useful for programmatic filtering; the human-friendly label is built from
    # ``source_dist`` instead.
    source_plugin: str | None = None
    # Distribution (PyPI / wheel) name that registered this skill's entry point
    # (e.g. ``"nemo-agents-plugin"``, ``"nemo-helix-ext"``,
    # ``"nemo-helix-sdk"``). This is what users see in ``pip list`` / what
    # they ``uv add``'d, and is what the ``Source`` column in
    # ``nemo skills list`` renders.
    source_dist: str | None = None
    preconditions: list[str] = field(default_factory=list)


class Scope(str, Enum):
    PROJECT = "project"
    USER = "user"


class AgentInstaller(Protocol):
    """Protocol that all agent installers must implement."""

    @property
    def name(self) -> str: ...

    @property
    def display_name(self) -> str: ...

    @property
    def supported_scopes(self) -> list[Scope]: ...

    def get_install_path(self, scope: Scope, project_root: Path, skill_name: str) -> Path: ...

    def format_content(self, skill: Skill) -> str: ...

    def install(self, scope: Scope, project_root: Path, skills: dict[str, Skill]) -> list[Path]: ...
