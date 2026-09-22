# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Codex agent installer."""

from pathlib import Path

from nemo_platform_ext.cli.commands.skills.base import Scope, Skill, installed_skill_name
from nemo_platform_ext.cli.commands.skills.installer import BaseAgentInstaller, format_agent_skill_content


class CodexInstaller(BaseAgentInstaller):
    name = "codex"
    display_name = "Codex (.agents/skills)"
    supported_scopes = [Scope.PROJECT, Scope.USER]

    def get_install_path(self, scope: Scope, project_root: Path, skill_name: str) -> Path:
        target_name = installed_skill_name(skill_name)
        # Codex discovers skills under `.agents/skills/` (see openai/codex
        # `codex-rs/core-skills/src/loader.rs`). The older `.codex/skills/`
        # layout has been deprecated.
        if scope == Scope.PROJECT:
            return project_root / ".agents" / "skills" / target_name / "SKILL.md"
        return Path.home() / ".agents" / "skills" / target_name / "SKILL.md"

    def format_content(self, skill: Skill) -> str:
        return format_agent_skill_content(skill)
