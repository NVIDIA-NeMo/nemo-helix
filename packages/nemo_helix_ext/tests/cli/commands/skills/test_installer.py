# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the base agent installer."""

from pathlib import Path

import pytest
from nemo_helix_ext.cli.commands.skills.base import Scope, Skill, installed_skill_name
from nemo_helix_ext.cli.commands.skills.installer import BaseAgentInstaller


def _make_skill(name: str = "test-skill", content: str = "# Test content", source_dir: Path | None = None) -> Skill:
    return Skill(
        name=name,
        description=f"A {name} skill",
        version="0.1",
        content=content,
        raw=f"---\nname: {name}\n---\n{content}",
        source_dir=source_dir,
    )


class FakeInstaller(BaseAgentInstaller):
    name = "fake"
    display_name = "Fake Agent"
    supported_scopes = [Scope.PROJECT, Scope.USER]

    def get_install_path(self, scope: Scope, project_root: Path, skill_name: str) -> Path:
        del scope
        return project_root / ".fake" / f"{skill_name}.md"

    def format_content(self, skill: Skill) -> str:
        return skill.raw


class NormalizingInstaller(FakeInstaller):
    def get_install_path(self, scope: Scope, project_root: Path, skill_name: str) -> Path:
        del scope
        return project_root / ".fake" / f"{installed_skill_name(skill_name)}.md"


def test_install_creates_files_for_each_skill(tmp_path: Path):
    installer = FakeInstaller()
    skills = {"alpha": _make_skill("alpha"), "beta": _make_skill("beta")}
    paths = installer.install(Scope.PROJECT, tmp_path, skills)
    assert len(paths) == 2
    for path in paths:
        assert path.exists()


def test_install_creates_parent_directories(tmp_path: Path):
    installer = FakeInstaller()
    skills = {"test": _make_skill()}
    paths = installer.install(Scope.PROJECT, tmp_path, skills)
    assert paths[0].parent.is_dir()


def test_install_overwrites_existing_files(tmp_path: Path):
    installer = FakeInstaller()
    skills = {"test": _make_skill(content="new content")}
    installer.install(Scope.PROJECT, tmp_path, skills)
    skills2 = {"test": _make_skill(content="updated content")}
    paths = installer.install(Scope.PROJECT, tmp_path, skills2)
    assert "updated content" in paths[0].read_text()


def test_install_returns_correct_paths(tmp_path: Path):
    installer = FakeInstaller()
    skills = {"alpha": _make_skill("alpha")}
    paths = installer.install(Scope.PROJECT, tmp_path, skills)
    assert paths[0] == tmp_path / ".fake" / "alpha.md"


def test_install_rejects_path_like_skill_names(tmp_path: Path):
    installer = FakeInstaller()
    skills = {"../escape": _make_skill("../escape")}

    with pytest.raises(ValueError, match="Invalid skill name"):
        installer.install(Scope.PROJECT, tmp_path, skills)


def test_install_rejects_duplicate_normalized_destinations_before_writing(tmp_path: Path):
    installer = NormalizingInstaller()
    skills = {
        "foo": _make_skill("foo"),
        "nemo-foo": _make_skill("nemo-foo"),
    }

    with pytest.raises(ValueError, match="Multiple skills resolve to"):
        installer.install(Scope.PROJECT, tmp_path, skills)

    assert not (tmp_path / ".fake" / "nemo-foo.md").exists()


def test_install_copies_companion_files(tmp_path: Path):
    source_dir = tmp_path / "source" / "my-skill"
    source_dir.mkdir(parents=True)
    (source_dir / "SKILL.md").write_text("---\nname: my-skill\n---\n# My Skill")
    resources = source_dir / "resources"
    resources.mkdir()
    (resources / "guide.md").write_text("# Guide content")
    (resources / "reference.md").write_text("# Reference content")

    installer = FakeInstaller()
    skills = {"my-skill": _make_skill("my-skill", source_dir=source_dir)}
    paths = installer.install(Scope.PROJECT, tmp_path, skills)

    target_dir = paths[0].parent
    assert (target_dir / "resources" / "guide.md").exists()
    assert (target_dir / "resources" / "guide.md").read_text() == "# Guide content"
    assert (target_dir / "resources" / "reference.md").exists()


def test_install_without_source_dir_still_works(tmp_path: Path):
    installer = FakeInstaller()
    skills = {"test": _make_skill(content="# Works")}
    paths = installer.install(Scope.PROJECT, tmp_path, skills)
    assert paths[0].exists()
    assert "# Works" in paths[0].read_text()


def test_install_overwrites_companion_files(tmp_path: Path):
    source_dir = tmp_path / "source" / "my-skill"
    source_dir.mkdir(parents=True)
    (source_dir / "SKILL.md").write_text("---\nname: my-skill\n---\n# Skill")
    resources = source_dir / "resources"
    resources.mkdir()
    (resources / "guide.md").write_text("old content")

    installer = FakeInstaller()
    skills = {"my-skill": _make_skill("my-skill", source_dir=source_dir)}
    installer.install(Scope.PROJECT, tmp_path, skills)

    (resources / "guide.md").write_text("new content")
    installer.install(Scope.PROJECT, tmp_path, skills)

    target_dir = (tmp_path / ".fake" / "my-skill.md").parent
    assert (target_dir / "resources" / "guide.md").read_text() == "new content"
