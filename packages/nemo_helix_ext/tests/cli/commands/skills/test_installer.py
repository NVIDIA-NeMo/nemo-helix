# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the base agent installer."""

import json
import logging
from pathlib import Path

import pytest
from nemo_helix_ext.cli.commands.skills.base import Scope, Skill, installed_skill_name
from nemo_helix_ext.cli.commands.skills.installer import (
    INSTALLED_SKILLS_MANIFEST_NAME,
    BaseAgentInstaller,
)


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


class DirectoryLayoutInstaller(BaseAgentInstaller):
    """Installer with the real-world layout: <root>/<skills>/<skill-dir>/SKILL.md.

    Every shipped agent installer writes each skill as a directory under a
    skills root, so manifest/prune behavior is exercised with this shape.
    """

    name = "directory"
    display_name = "Directory Agent"
    supported_scopes = [Scope.PROJECT, Scope.USER]

    def get_install_path(self, scope: Scope, project_root: Path, skill_name: str) -> Path:
        del scope
        return project_root / ".agent" / "skills" / installed_skill_name(skill_name) / "SKILL.md"

    def format_content(self, skill: Skill) -> str:
        return skill.raw


def _skills_root(tmp_path: Path) -> Path:
    return tmp_path / ".agent" / "skills"


def _read_manifest(tmp_path: Path) -> dict:
    return json.loads((_skills_root(tmp_path) / INSTALLED_SKILLS_MANIFEST_NAME).read_text())


def _write_skill_dir(skills_root: Path, name: str, content: str = "user skill") -> None:
    skill_dir = skills_root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\n---\n{content}")


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


class TestInstallManifest:
    """The installed-skills manifest: what it records, what prune removes."""

    def test_full_install_prune_removes_recorded_skill_and_keeps_unrecorded_neighbor(self, tmp_path: Path):
        """A second full install removes recorded skills it no longer ships.

        Directories the manifest does not record (skills the user wrote by
        hand) are never touched.
        """
        installer = DirectoryLayoutInstaller()
        skills_root = _skills_root(tmp_path)

        installer.install(
            Scope.PROJECT,
            tmp_path,
            {"alpha": _make_skill("alpha"), "beta": _make_skill("beta")},
            prune=True,
        )
        assert (skills_root / "nemo-beta" / "SKILL.md").exists()
        assert _read_manifest(tmp_path)["owned"] == ["nemo-alpha", "nemo-beta"]

        # A neighbor the manifest does not record, e.g. a user-written skill.
        _write_skill_dir(skills_root, "nemo-user-skill")

        # A later full install that no longer ships "beta".
        installer.install(Scope.PROJECT, tmp_path, {"alpha": _make_skill("alpha")}, prune=True)

        assert (skills_root / "nemo-alpha" / "SKILL.md").exists()
        assert not (skills_root / "nemo-beta").exists()
        assert (skills_root / "nemo-user-skill" / "SKILL.md").exists()
        assert _read_manifest(tmp_path)["owned"] == ["nemo-alpha"]

    def test_subset_install_records_directory_and_does_not_prune(self, tmp_path: Path):
        """A `--skill` subset records what it wrote and removes nothing."""
        installer = DirectoryLayoutInstaller()
        skills_root = _skills_root(tmp_path)

        installer.install(
            Scope.PROJECT,
            tmp_path,
            {"alpha": _make_skill("alpha"), "beta": _make_skill("beta")},
            prune=True,
        )

        # Default prune=False, matching the `nemo skills install --skill` path.
        installer.install(Scope.PROJECT, tmp_path, {"gamma": _make_skill("gamma")})

        assert (skills_root / "nemo-gamma" / "SKILL.md").exists()
        # The sibling recorded by the earlier full install is untouched.
        assert (skills_root / "nemo-beta" / "SKILL.md").exists()
        assert _read_manifest(tmp_path)["owned"] == ["nemo-alpha", "nemo-beta", "nemo-gamma"]

    def test_prune_overwrites_same_named_skill(self, tmp_path: Path):
        """A skill shipped under a name it already owns is overwritten, not pruned."""
        installer = DirectoryLayoutInstaller()
        skills_root = _skills_root(tmp_path)

        installer.install(Scope.PROJECT, tmp_path, {"alpha": _make_skill("alpha", content="old")}, prune=True)
        installer.install(
            Scope.PROJECT,
            tmp_path,
            {"alpha": _make_skill("alpha", content="updated")},
            prune=True,
        )

        assert "updated" in (skills_root / "nemo-alpha" / "SKILL.md").read_text()
        assert _read_manifest(tmp_path)["owned"] == ["nemo-alpha"]

    def test_install_never_claims_a_preexisting_unrecorded_directory(self, tmp_path: Path):
        installer = DirectoryLayoutInstaller()
        skills_root = _skills_root(tmp_path)
        _write_skill_dir(skills_root, "nemo-alpha")

        installer.install(Scope.PROJECT, tmp_path, {"alpha": _make_skill("alpha"), "beta": _make_skill("beta")})
        assert _read_manifest(tmp_path)["owned"] == ["nemo-beta"]

        installer.install(Scope.PROJECT, tmp_path, {"beta": _make_skill("beta")}, prune=True)
        assert (skills_root / "nemo-alpha" / "SKILL.md").exists()

    def test_prune_drops_a_recorded_directory_that_is_already_gone(self, tmp_path: Path):
        installer = DirectoryLayoutInstaller()
        skills_root = _skills_root(tmp_path)
        skills_root.mkdir(parents=True)
        (skills_root / INSTALLED_SKILLS_MANIFEST_NAME).write_text(json.dumps({"owned": ["nemo-gone"]}))

        installer.install(Scope.PROJECT, tmp_path, {"alpha": _make_skill("alpha")}, prune=True)

        assert _read_manifest(tmp_path)["owned"] == ["nemo-alpha"]

    @pytest.mark.parametrize(
        "manifest_text,expected_warning",
        [
            pytest.param("not json", "unreadable", id="corrupt"),
            pytest.param(json.dumps({"owned": "nemo-a"}), "unexpected shape", id="owned-not-a-list"),
            pytest.param(json.dumps({"owned": ["nemo-a", 1]}), "unexpected shape", id="non-string-entry"),
            pytest.param(json.dumps(["nemo-a"]), "unexpected shape", id="not-a-mapping"),
        ],
    )
    def test_prune_treats_an_invalid_manifest_as_unrecorded(
        self, tmp_path: Path, manifest_text: str, expected_warning: str, caplog: pytest.LogCaptureFixture
    ):
        installer = DirectoryLayoutInstaller()
        skills_root = _skills_root(tmp_path)
        skills_root.mkdir(parents=True)
        (skills_root / INSTALLED_SKILLS_MANIFEST_NAME).write_text(manifest_text)
        _write_skill_dir(skills_root, "nemo-a")

        with caplog.at_level(logging.WARNING, logger="nemo_helix_ext.cli.commands.skills.installer"):
            installer.install(Scope.PROJECT, tmp_path, {"alpha": _make_skill("alpha")}, prune=True)

        assert (skills_root / "nemo-a" / "SKILL.md").exists()
        assert _read_manifest(tmp_path)["owned"] == ["nemo-alpha"]
        assert any(expected_warning in record.message for record in caplog.records)

    def test_prune_keeps_a_recorded_directory_it_cannot_remove(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        """A failed removal stays recorded so a later full install retries it."""
        installer = DirectoryLayoutInstaller()
        skills_root = _skills_root(tmp_path)
        installer.install(
            Scope.PROJECT, tmp_path, {"alpha": _make_skill("alpha"), "beta": _make_skill("beta")}, prune=True
        )

        def _fail(path: Path) -> None:
            raise PermissionError(f"busy: {path}")

        monkeypatch.setattr("nemo_helix_ext.cli.commands.skills.installer.shutil.rmtree", _fail)
        with caplog.at_level(logging.WARNING, logger="nemo_helix_ext.cli.commands.skills.installer"):
            installer.install(Scope.PROJECT, tmp_path, {"alpha": _make_skill("alpha")}, prune=True)

        assert (skills_root / "nemo-beta" / "SKILL.md").exists()
        assert installer.pruned_directories == []
        assert _read_manifest(tmp_path)["owned"] == ["nemo-alpha", "nemo-beta"]
        assert any("Could not remove" in record.message for record in caplog.records)

    def test_prune_never_removes_a_recorded_name_matching_a_written_one_case_insensitively(self, tmp_path: Path):
        installer = DirectoryLayoutInstaller()
        skills_root = _skills_root(tmp_path)
        skills_root.mkdir(parents=True)
        (skills_root / INSTALLED_SKILLS_MANIFEST_NAME).write_text(json.dumps({"owned": ["nemo-ALPHA"]}))

        installer.install(Scope.PROJECT, tmp_path, {"alpha": _make_skill("alpha")}, prune=True)

        assert (skills_root / "nemo-alpha" / "SKILL.md").exists()
        assert installer.pruned_directories == []
        assert _read_manifest(tmp_path)["owned"] == ["nemo-alpha"]

    @pytest.mark.parametrize(
        "recorded",
        [
            pytest.param("team-skill", id="no-installed-prefix"),
            pytest.param("nemo-not-a-dir", id="plain-file"),
            pytest.param("nemo-root-link", id="symlink-to-skills-root"),
            pytest.param("nemo-outside-link", id="symlink-outside-skills-root"),
            pytest.param("../neighbor", id="multi-segment"),
        ],
    )
    def test_prune_leaves_an_unsafe_recorded_name_in_place_and_recorded(self, tmp_path: Path, recorded: str):
        installer = DirectoryLayoutInstaller()
        skills_root = _skills_root(tmp_path)
        skills_root.mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "SKILL.md").write_text("outside content")
        _write_skill_dir(skills_root, "team-skill")
        (skills_root / "nemo-not-a-dir").write_text("user file")
        (skills_root / "nemo-root-link").symlink_to(skills_root, target_is_directory=True)
        (skills_root / "nemo-outside-link").symlink_to(outside, target_is_directory=True)
        (tmp_path / ".agent" / "neighbor").mkdir()
        (skills_root / INSTALLED_SKILLS_MANIFEST_NAME).write_text(json.dumps({"owned": [recorded]}))
        target = skills_root / recorded
        before = target.lstat()

        installer.install(Scope.PROJECT, tmp_path, {"alpha": _make_skill("alpha")}, prune=True)

        assert target.lstat().st_ino == before.st_ino
        assert (outside / "SKILL.md").read_text() == "outside content"
        assert installer.pruned_directories == []
        assert recorded in _read_manifest(tmp_path)["owned"]

    def test_manifest_write_does_not_follow_a_symlink_planted_at_the_manifest_path(self, tmp_path: Path):
        installer = DirectoryLayoutInstaller()
        skills_root = _skills_root(tmp_path)
        victim = tmp_path / "victim.json"
        victim.write_text("precious")
        skills_root.mkdir(parents=True)
        (skills_root / INSTALLED_SKILLS_MANIFEST_NAME).symlink_to(victim)

        installer.install(Scope.PROJECT, tmp_path, {"alpha": _make_skill("alpha")})

        assert victim.read_text() == "precious"
        assert not (skills_root / INSTALLED_SKILLS_MANIFEST_NAME).is_symlink()
        assert _read_manifest(tmp_path)["owned"] == ["nemo-alpha"]
