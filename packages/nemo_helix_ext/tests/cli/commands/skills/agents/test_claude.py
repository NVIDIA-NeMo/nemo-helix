# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Claude Code agent installer."""

from pathlib import Path

import yaml
from nemo_helix_ext.cli.commands.skills.agents.claude import ClaudeInstaller
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
    installer = ClaudeInstaller()
    assert installer.name == "claude"
    assert installer.display_name == "Claude Code"


def test_supported_scopes():
    installer = ClaudeInstaller()
    assert Scope.PROJECT in installer.supported_scopes
    assert Scope.USER in installer.supported_scopes


def test_project_install_path(tmp_path: Path):
    installer = ClaudeInstaller()
    path = installer.get_install_path(Scope.PROJECT, tmp_path, "inference")
    assert path == tmp_path / ".claude" / "skills" / "nemo-inference" / "SKILL.md"


def test_project_install_path_keeps_existing_nemo_prefix(tmp_path: Path):
    installer = ClaudeInstaller()
    path = installer.get_install_path(Scope.PROJECT, tmp_path, "nemo-files")
    assert path == tmp_path / ".claude" / "skills" / "nemo-files" / "SKILL.md"


def test_user_install_path(tmp_path: Path):
    installer = ClaudeInstaller()
    path = installer.get_install_path(Scope.USER, tmp_path, "inference")
    assert path == Path.home() / ".claude" / "skills" / "nemo-inference" / "SKILL.md"


def test_format_content_adds_frontmatter():
    installer = ClaudeInstaller()
    skill = _make_skill("inference")
    content = installer.format_content(skill)
    assert content.startswith("---")
    assert "name: nemo-inference" in content
    assert "# inference" in content


def test_format_content_keeps_existing_nemo_prefix_and_preconditions():
    installer = ClaudeInstaller()
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
    installer = ClaudeInstaller()
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


def test_install_creates_files(tmp_path: Path):
    installer = ClaudeInstaller()
    skills = {"alpha": _make_skill("alpha"), "beta": _make_skill("beta")}
    paths = installer.install(Scope.PROJECT, tmp_path, skills)
    assert len(paths) == 2
    for path in paths:
        assert path.exists()
        assert "---" in path.read_text()


def test_install_copies_companion_files(tmp_path: Path):
    source_dir = tmp_path / "source" / "evaluation"
    source_dir.mkdir(parents=True)
    (source_dir / "SKILL.md").write_text("---\nname: evaluation\n---\n# Evaluation")
    resources = source_dir / "resources"
    resources.mkdir()
    (resources / "llm-judge.md").write_text("# LLM Judge Guide")

    installer = ClaudeInstaller()
    skills = {"evaluation": _make_skill("evaluation", source_dir=source_dir)}
    installer.install(Scope.PROJECT, tmp_path, skills)

    skill_dir = tmp_path / ".claude" / "skills" / "nemo-evaluation"
    assert (skill_dir / "SKILL.md").exists()
    assert (skill_dir / "resources" / "llm-judge.md").exists()
    assert (skill_dir / "resources" / "llm-judge.md").read_text() == "# LLM Judge Guide"
