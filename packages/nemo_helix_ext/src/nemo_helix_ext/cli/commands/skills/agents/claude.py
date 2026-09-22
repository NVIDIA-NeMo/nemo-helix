# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude Code agent installer."""

from pathlib import Path

import yaml
from nemo_helix_ext.cli.commands.skills.base import Scope, Skill, installed_skill_name
from nemo_helix_ext.cli.commands.skills.installer import BaseAgentInstaller


class ClaudeInstaller(BaseAgentInstaller):
    name = "claude"
    display_name = "Claude Code"
    supported_scopes = [Scope.PROJECT, Scope.USER]

    def get_install_path(self, scope: Scope, project_root: Path, skill_name: str) -> Path:
        target_name = installed_skill_name(skill_name)
        if scope == Scope.PROJECT:
            return project_root / ".claude" / "skills" / target_name / "SKILL.md"
        return Path.home() / ".claude" / "skills" / target_name / "SKILL.md"

    def format_content(self, skill: Skill) -> str:
        metadata: dict[str, object] = {"name": installed_skill_name(skill.name), "description": skill.description}
        if skill.preconditions:
            metadata["preconditions"] = skill.preconditions
        front_matter = yaml.safe_dump(
            metadata,
            sort_keys=False,
            allow_unicode=True,
        )
        return f"---\n{front_matter}---\n\n{skill.content}"
