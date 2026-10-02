# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Codex agent installer."""

from pathlib import Path

import pytest
import yaml
from nemo_helix_ext.cli.commands.skills.agents.codex import CodexInstaller
from nemo_helix_ext.cli.commands.skills.base import Scope, Skill


def _make_skill(
    name: str = "test-skill",
    source_dir: Path | None = None,
    preconditions: list[str] | None = None,
) -> Skill:
    return Skill(
        name=name,
        description=f"A {name} skill",
        version="0.1",
        content=f"# {name}\nSome content.",
        raw=f"---\nname: {name}\ndescription: A {name} skill\nversion: '0.1'\n---\n\n# {name}\nSome content.",
        preconditions=preconditions or [],
        source_dir=source_dir,
    )


def test_name():
    installer = CodexInstaller()
    assert installer.name == "codex"
    assert installer.display_name == "Codex (.agents/skills)"


def test_supported_scopes():
    installer = CodexInstaller()
    assert Scope.PROJECT in installer.supported_scopes
    assert Scope.USER in installer.supported_scopes


def test_project_install_path(tmp_path: Path):
    installer = CodexInstaller()
    path = installer.get_install_path(Scope.PROJECT, tmp_path, "inference")
    assert path == tmp_path / ".agents" / "skills" / "nemo-inference" / "SKILL.md"


def test_project_install_path_keeps_existing_nemo_prefix(tmp_path: Path):
    installer = CodexInstaller()
    path = installer.get_install_path(Scope.PROJECT, tmp_path, "nemo-files")
    assert path == tmp_path / ".agents" / "skills" / "nemo-files" / "SKILL.md"


def test_user_install_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    installer = CodexInstaller()
    path = installer.get_install_path(Scope.USER, tmp_path, "inference")
    assert path == Path.home() / ".agents" / "skills" / "nemo-inference" / "SKILL.md"


def test_format_content_adds_frontmatter():
    installer = CodexInstaller()
    skill = _make_skill("inference")
    content = installer.format_content(skill)
    assert content.startswith("---")
    assert "name: nemo-inference" in content
    assert "description: A inference skill" in content
    assert "# inference" in content


def test_format_content_keeps_existing_nemo_prefix_and_preconditions():
    installer = CodexInstaller()
    skill = _make_skill("nemo-files", preconditions=["nemo_setup_complete", "platform_running"])
    content = installer.format_content(skill)
    front_matter, _, _ = content.partition("\n---\n")
    front_matter = front_matter.removeprefix("---\n")
    parsed = yaml.safe_load(front_matter)
    assert parsed == {
        "name": "nemo-files",
        "description": "A nemo-files skill",
        "preconditions": ["nemo_setup_complete", "platform_running"],
    }


def test_format_content_escapes_yaml_special_chars():
    """Descriptions with `:`, `#`, newlines, or leading `- ` must round-trip via YAML."""
    installer = CodexInstaller()
    tricky = "Use when: foo. Trigger keywords: bar, baz # comment\n- item"
    skill = Skill(
        name="inference",
        description=tricky,
        version="0.1",
        content="# body",
        raw="",
        source_dir=None,
    )
    content = installer.format_content(skill)
    front_matter, _, _ = content.partition("\n---\n")
    front_matter = front_matter.removeprefix("---\n")
    parsed = yaml.safe_load(front_matter)
    assert parsed == {"name": "nemo-inference", "description": tricky}


def test_install_creates_per_skill_directories(tmp_path: Path):
    installer = CodexInstaller()
    skills = {"alpha": _make_skill("alpha"), "beta": _make_skill("beta")}
    paths = installer.install(Scope.PROJECT, tmp_path, skills)
    assert len(paths) == 2
    expected_alpha = tmp_path / ".agents" / "skills" / "nemo-alpha" / "SKILL.md"
    expected_beta = tmp_path / ".agents" / "skills" / "nemo-beta" / "SKILL.md"
    assert expected_alpha in paths
    assert expected_beta in paths
    assert expected_alpha.exists()
    assert expected_beta.exists()
    assert "name: nemo-alpha" in expected_alpha.read_text()


def test_install_does_not_touch_agents_md(tmp_path: Path):
    """Regression: the old installer dumped sections into AGENTS.md; the new
    per-skill-directory installer must leave AGENTS.md untouched."""
    agents_md = tmp_path / "AGENTS.md"
    agents_md.write_text("# My Project\n\nUntouched.\n")
    installer = CodexInstaller()
    skills = {"alpha": _make_skill("alpha")}
    installer.install(Scope.PROJECT, tmp_path, skills)
    assert agents_md.read_text() == "# My Project\n\nUntouched.\n"


def test_install_idempotent(tmp_path: Path):
    installer = CodexInstaller()
    skills = {"alpha": _make_skill("alpha")}
    installer.install(Scope.PROJECT, tmp_path, skills)
    installer.install(Scope.PROJECT, tmp_path, skills)
    skill_md = tmp_path / ".agents" / "skills" / "nemo-alpha" / "SKILL.md"
    # File rewrites cleanly with same content
    assert "name: nemo-alpha" in skill_md.read_text()


def test_install_copies_companion_files(tmp_path: Path):
    source_dir = tmp_path / "source" / "evaluation"
    source_dir.mkdir(parents=True)
    (source_dir / "SKILL.md").write_text("---\nname: evaluation\n---\n# Evaluation")
    resources = source_dir / "resources"
    resources.mkdir()
    (resources / "llm-judge.md").write_text("# LLM Judge Guide")
    (resources / "troubleshooting.md").write_text("# Troubleshooting")

    installer = CodexInstaller()
    skills = {"evaluation": _make_skill("evaluation", source_dir=source_dir)}
    installer.install(Scope.PROJECT, tmp_path, skills)

    skill_dir = tmp_path / ".agents" / "skills" / "nemo-evaluation"
    assert (skill_dir / "SKILL.md").exists()
    assert (skill_dir / "resources" / "llm-judge.md").read_text() == "# LLM Judge Guide"
    assert (skill_dir / "resources" / "troubleshooting.md").read_text() == "# Troubleshooting"


def test_user_install_writes_to_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    installer = CodexInstaller()
    skills = {"alpha": _make_skill("alpha")}
    installer.install(Scope.USER, tmp_path / "project", skills)
    assert (tmp_path / ".agents" / "skills" / "nemo-alpha" / "SKILL.md").exists()
    # No project-local file written under USER scope
    assert not (tmp_path / "project" / ".agents").exists()
