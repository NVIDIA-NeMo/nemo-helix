# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable installer for a user-specified skills directory."""

from pathlib import Path

import yaml
from nemo_platform_ext.cli.commands.skills.base import Scope, Skill, installed_skill_name
from nemo_platform_ext.cli.commands.skills.installer import BaseAgentInstaller


class CustomPathInstaller(BaseAgentInstaller):
    """Install standard Agent Skills beneath an explicitly selected directory."""

    name = "custom-path"
    display_name = "Custom path"
    supported_scopes = [Scope.PROJECT, Scope.USER]

    def get_install_path(self, scope: Scope, project_root: Path, skill_name: str) -> Path:
        del scope
        return project_root / installed_skill_name(skill_name) / "SKILL.md"

    def format_content(self, skill: Skill) -> str:
        metadata: dict[str, object] = {"name": installed_skill_name(skill.name), "description": skill.description}
        if skill.preconditions:
            metadata["preconditions"] = skill.preconditions
        front_matter = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True)
        return f"---\n{front_matter}---\n\n{skill.content}"
