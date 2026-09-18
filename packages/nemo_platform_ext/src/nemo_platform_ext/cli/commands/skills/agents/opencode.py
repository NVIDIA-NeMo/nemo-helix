# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenCode agent installer."""

from pathlib import Path

from nemo_platform_ext.cli.commands.skills.base import Scope, installed_skill_name
from nemo_platform_ext.cli.commands.skills.installer import BaseAgentInstaller


class OpenCodeInstaller(BaseAgentInstaller):
    name = "opencode"
    display_name = "OpenCode"
    supported_scopes = [Scope.PROJECT, Scope.USER]

    def get_install_path(self, scope: Scope, project_root: Path, skill_name: str) -> Path:
        target_name = installed_skill_name(skill_name)
        if scope == Scope.PROJECT:
            return project_root / ".opencode" / "skills" / target_name / "SKILL.md"
        return Path.home() / ".config" / "opencode" / "skills" / target_name / "SKILL.md"
