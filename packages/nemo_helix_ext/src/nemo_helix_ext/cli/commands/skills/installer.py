# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base agent installer with default install logic."""

import shutil
from pathlib import Path

from nemo_helix_ext.cli.commands.skills.base import Scope, Skill, validate_skill_name


class BaseAgentInstaller:
    """Base class providing default install behavior: loop over skills, format, write."""

    name: str
    display_name: str
    supported_scopes: list[Scope]

    def get_install_path(self, scope: Scope, project_root: Path, skill_name: str) -> Path:
        raise NotImplementedError

    def format_content(self, skill: Skill) -> str:
        """Format skill content for this agent. Default: return raw content."""
        return skill.raw

    def install(self, scope: Scope, project_root: Path, skills: dict[str, Skill]) -> list[Path]:
        """Install all skills. Returns list of paths written."""
        paths: list[Path] = []
        pending: list[tuple[Skill, Path]] = []
        destinations: dict[Path, str] = {}
        for skill_name, skill in skills.items():
            validate_skill_name(skill_name)
            path = self.get_install_path(scope, project_root, skill_name)
            if path in destinations:
                raise ValueError(f"Multiple skills resolve to {path}: {destinations[path]!r} and {skill_name!r}")
            destinations[path] = skill_name
            pending.append((skill, path))
        for skill, path in pending:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.format_content(skill))
            self._copy_companion_files(skill, path)
            paths.append(path)
        return paths

    def _copy_companion_files(self, skill: Skill, installed_path: Path) -> None:
        """Copy non-SKILL.md files from source_dir to the installed skill's directory."""
        if skill.source_dir is None:
            return
        target_dir = installed_path.parent
        for item in skill.source_dir.iterdir():
            if item.name == "SKILL.md":
                continue
            dest = target_dir / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
