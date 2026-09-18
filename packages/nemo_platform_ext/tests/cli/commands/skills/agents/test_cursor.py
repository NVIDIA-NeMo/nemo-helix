# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Cursor agent installer."""

from pathlib import Path

from nemo_platform_ext.cli.commands.skills.agents.cursor import CursorInstaller
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
    installer = CursorInstaller()
    assert installer.name == "cursor"


def test_supported_scopes():
    installer = CursorInstaller()
    assert installer.supported_scopes == [Scope.PROJECT, Scope.USER]


def test_project_install_path(tmp_path: Path):
    installer = CursorInstaller()
    path = installer.get_install_path(Scope.PROJECT, tmp_path, "inference")
    assert path == tmp_path / ".cursor" / "skills" / "nemo-inference" / "SKILL.md"


def test_project_install_path_keeps_existing_nemo_prefix(tmp_path: Path):
    installer = CursorInstaller()
    path = installer.get_install_path(Scope.PROJECT, tmp_path, "nemo-files")
    assert path == tmp_path / ".cursor" / "skills" / "nemo-files" / "SKILL.md"


def test_user_install_path(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    installer = CursorInstaller()
    path = installer.get_install_path(Scope.USER, tmp_path / "project", "inference")
    assert path == tmp_path / ".cursor" / "skills" / "nemo-inference" / "SKILL.md"


def test_install_creates_files(tmp_path: Path):
    installer = CursorInstaller()
    skills = {"alpha": _make_skill("alpha")}
    paths = installer.install(Scope.PROJECT, tmp_path, skills)
    assert len(paths) == 1
    assert paths[0].exists()


def test_install_copies_companion_files(tmp_path: Path):
    source_dir = tmp_path / "source" / "evaluation"
    source_dir.mkdir(parents=True)
    (source_dir / "SKILL.md").write_text("---\nname: evaluation\n---\n# Evaluation")
    resources = source_dir / "resources"
    resources.mkdir()
    (resources / "llm-judge.md").write_text("# LLM Judge Guide")

    installer = CursorInstaller()
    skills = {"evaluation": _make_skill("evaluation", source_dir=source_dir)}
    installer.install(Scope.PROJECT, tmp_path, skills)

    skill_dir = tmp_path / ".cursor" / "skills" / "nemo-evaluation"
    assert (skill_dir / "SKILL.md").exists()
    assert (skill_dir / "resources" / "llm-judge.md").exists()
