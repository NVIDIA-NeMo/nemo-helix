# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OpenCode agent installer."""

from pathlib import Path

from nemo_platform_ext.cli.commands.skills.agents.opencode import OpenCodeInstaller
from nemo_platform_ext.cli.commands.skills.base import Scope, Skill


def _make_skill(name: str = "test-skill", source_dir: Path | None = None) -> Skill:
    return Skill(
        name=name,
        description="desc",
        version="0.1",
        content="# Test",
        raw="---\nname: test\n---\n# Test",
        source_dir=source_dir,
    )


def test_name():
    installer = OpenCodeInstaller()
    assert installer.name == "opencode"


def test_project_install_path(tmp_path: Path):
    installer = OpenCodeInstaller()
    path = installer.get_install_path(Scope.PROJECT, tmp_path, "inference")
    assert path == tmp_path / ".opencode" / "skills" / "nemo-inference" / "SKILL.md"


def test_project_install_path_keeps_existing_nemo_prefix(tmp_path: Path):
    installer = OpenCodeInstaller()
    path = installer.get_install_path(Scope.PROJECT, tmp_path, "nemo-files")
    assert path == tmp_path / ".opencode" / "skills" / "nemo-files" / "SKILL.md"


def test_user_install_path(tmp_path: Path):
    installer = OpenCodeInstaller()
    path = installer.get_install_path(Scope.USER, tmp_path, "inference")
    assert path == Path.home() / ".config" / "opencode" / "skills" / "nemo-inference" / "SKILL.md"


def test_install_copies_companion_files(tmp_path: Path):
    source_dir = tmp_path / "source" / "evaluation"
    source_dir.mkdir(parents=True)
    (source_dir / "SKILL.md").write_text("---\nname: evaluation\n---\n# Evaluation")
    resources = source_dir / "resources"
    resources.mkdir()
    (resources / "guide.md").write_text("# Guide")

    installer = OpenCodeInstaller()
    skills = {"evaluation": _make_skill("evaluation", source_dir=source_dir)}
    installer.install(Scope.PROJECT, tmp_path, skills)

    skill_dir = tmp_path / ".opencode" / "skills" / "nemo-evaluation"
    assert (skill_dir / "SKILL.md").exists()
    assert (skill_dir / "resources" / "guide.md").exists()
