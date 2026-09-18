# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the skills CLI commands."""

from pathlib import Path
from unittest.mock import patch

import pytest
from nemo_platform_ext.cli.app import app
from nemo_platform_ext.cli.commands.skills.base import installed_skill_name
from nemo_platform_ext.cli.commands.skills.registry import (
    DuplicateSkillError,
    SkillProvider,
    _load_skills_cached,
)
from nemo_platform_ext.skills import skills_dir as platform_skills_dir
from typer.testing import CliRunner

from ...utils import assert_exit_code

runner = CliRunner()


def _example_plugin_skills_root() -> Path:
    relative_path = Path("plugins/example-plugin/src/nemo_example_plugin/skills")
    for parent in Path(__file__).resolve().parents:
        if (parent / relative_path).is_dir():
            return parent / relative_path
    raise FileNotFoundError(f"Could not find {relative_path} from {__file__}")


def _platform_skill_names() -> set[str]:
    """Names of skills shipped under `nemo_platform_ext.skills`, derived from disk.

    Hardcoding the set here would make every test break whenever we add or
    remove a built-in skill, so we discover it the same way the registry does.
    """
    return {
        entry.name for entry in platform_skills_dir().iterdir() if entry.is_dir() and not entry.name.startswith("_")
    }


@pytest.fixture(autouse=True)
def _clear_skill_registry_caches():
    """Reset the registry caches around every test in this module.

    ``_load_skills_cached`` is an ``lru_cache``, and the underlying
    ``nemo_platform_plugin.discovery.discover_entry_points`` is
    ``@cache``-decorated. If any earlier test (in this module or elsewhere)
    populated either cache with a value computed under monkeypatched conditions,
    later tests would inherit that stale view and the CLI would render ``[]``
    skills. Clearing both sides on entry and exit isolates each test from its
    neighbours regardless of test order — which matters under pytest-xdist where
    ordering is not fixed.
    """
    from nemo_platform_plugin.discovery import discover_entry_points as _discover_eps

    _load_skills_cached.cache_clear()
    _discover_eps.cache_clear()
    yield
    _load_skills_cached.cache_clear()
    _discover_eps.cache_clear()


@pytest.fixture
def example_plugin_skills(monkeypatch):
    """Layer the example-plugin skills directory on top of the platform's own skills."""
    _load_skills_cached.cache_clear()
    monkeypatch.setattr(
        "nemo_platform_ext.cli.commands.skills.registry._discover_skill_providers",
        lambda: {
            "platform": SkillProvider(name="platform", path=platform_skills_dir(), dist_name="nemo-platform-ext"),
            "example": SkillProvider(
                name="example", path=_example_plugin_skills_root(), dist_name="nemo-example-plugin"
            ),
        },
    )
    yield
    _load_skills_cached.cache_clear()


class TestList:
    def test_list_shows_skills(self):
        result = runner.invoke(app, "skills list")
        assert_exit_code(result, 0)
        assert "inference" in result.stdout

    def test_list_shows_plugin_skills(self, example_plugin_skills):
        result = runner.invoke(app, "skills list")
        assert_exit_code(result, 0)
        assert "example-hello" in result.stdout
        assert "example-debug" in result.stdout

    def test_list_shows_version(self):
        result = runner.invoke(app, "skills list")
        assert_exit_code(result, 0)
        assert "0.1" in result.stdout

    def test_list_duplicate_skill_error_is_clean(self, monkeypatch):
        monkeypatch.setattr(
            "nemo_platform_ext.cli.commands.skills.cli.load_skills",
            lambda: (_ for _ in ()).throw(DuplicateSkillError("duplicate skill")),
        )
        result = runner.invoke(app, "skills list")
        assert_exit_code(result, 1)
        assert "Error: duplicate skill" in result.output

    def test_stream_requires_json_output_before_loading_skills(self):
        with patch("nemo_platform_ext.cli.commands.skills.cli.load_skills") as mock_load_skills:
            result = runner.invoke(app, "skills list --output table --stream")

        assert_exit_code(result, 2)
        assert "--stream requires --output json or --output raw" in result.stderr
        mock_load_skills.assert_not_called()

    def test_list_source_column_shows_distribution_names(self, example_plugin_skills):
        """JSON output is stable and easy to assert against the new ``Source`` semantics.

        The platform's bundled skills must collapse to ``nemo-platform``; plugins
        must render under their distribution name.
        """
        import json

        result = runner.invoke(app, "skills list -f json")
        assert_exit_code(result, 0)
        data = json.loads(result.stdout)
        by_name = {item["name"]: item for item in data}

        # Platform skills always come from the `nemo-platform-ext` (or
        # vendored `nemo-platform-sdk`) distribution; both collapse to
        # `nemo-platform` in the user-facing column.
        assert by_name["inference"]["source"] == "nemo-platform"
        # Raw entry-point name preserved for programmatic consumers.
        assert by_name["inference"]["source_plugin"] == "platform"

        # Plugins render under the distribution name that ships them.
        assert by_name["example-hello"]["source"] == "nemo-example-plugin"
        assert by_name["example-hello"]["source_dist"] == "nemo-example-plugin"
        assert by_name["example-hello"]["source_plugin"] == "example"

    def test_list_source_filter_keeps_only_matching_dist(self, example_plugin_skills):
        """`--source nemo-platform` returns only the platform's own skills."""
        import json

        result = runner.invoke(app, "skills list -f json --source nemo-platform")
        assert_exit_code(result, 0)
        names = {item["name"] for item in json.loads(result.stdout)}
        assert names == _platform_skill_names()

    def test_list_source_filter_repeatable_unions_results(self, example_plugin_skills):
        """`--source` can be repeated; results are the union (OR)."""
        import json

        result = runner.invoke(
            app,
            "skills list -f json --source nemo-platform --source nemo-example-plugin",
        )
        assert_exit_code(result, 0)
        names = {item["name"] for item in json.loads(result.stdout)}
        assert names == _platform_skill_names() | {"example-hello", "example-debug"}

    def test_list_source_filter_is_case_insensitive(self, example_plugin_skills):
        import json

        result = runner.invoke(app, "skills list -f json --source Nemo-Platform")
        assert_exit_code(result, 0)
        names = {item["name"] for item in json.loads(result.stdout)}
        assert names == _platform_skill_names()

    def test_list_source_filter_unknown_source_errors(self, example_plugin_skills):
        """Unknown source must be a hard error listing what's actually available."""
        result = runner.invoke(app, "skills list --source nemo-does-not-exist")
        assert_exit_code(result, 1)
        assert "Unknown source(s): nemo-does-not-exist" in result.output
        # The error must show the user what they could have typed.
        assert "nemo-platform" in result.output
        assert "nemo-example-plugin" in result.output


class TestShow:
    def test_show_with_name_prints_content(self):
        result = runner.invoke(app, "skills show inference")
        assert_exit_code(result, 0)
        assert "name: inference" in result.stdout

    def test_show_with_agent_formats_content(self):
        result = runner.invoke(app, "skills show --agent claude inference")
        assert_exit_code(result, 0)
        assert "name: nemo-inference" in result.stdout

    def test_show_unknown_skill_errors(self):
        result = runner.invoke(app, "skills show nonexistent")
        assert_exit_code(result, 1)
        assert "Unknown skill" in result.output

    def test_show_unknown_agent_errors(self):
        result = runner.invoke(app, "skills show --agent unknown inference")
        assert_exit_code(result, 1)
        assert "Unsupported agent" in result.output

    def test_show_plugin_skill_prints_content(self, example_plugin_skills):
        result = runner.invoke(app, "skills show example-hello")
        assert_exit_code(result, 0)
        assert "Example Hello" in result.stdout

    def test_show_each_skill(self):
        for skill in _platform_skill_names():
            result = runner.invoke(app, f"skills show {skill}")
            assert_exit_code(result, 0)
            assert len(result.stdout.strip()) > 0

    def test_show_duplicate_skill_error_is_clean(self, monkeypatch):
        monkeypatch.setattr(
            "nemo_platform_ext.cli.commands.skills.cli.load_skills",
            lambda: (_ for _ in ()).throw(DuplicateSkillError("duplicate skill")),
        )
        result = runner.invoke(app, "skills show setup")
        assert_exit_code(result, 1)
        assert "Error: duplicate skill" in result.output


class TestInstall:
    def test_install_requires_agent(self):
        result = runner.invoke(app, "skills install")
        assert result.exit_code != 0

    def test_install_claude_project_creates_multiple_files(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, "skills install --agent claude")
        assert_exit_code(result, 0)
        # Every built-in skill should install — exercise the multi-skill path
        # without hardcoding which platform skills exist today.
        for skill_name in _platform_skill_names():
            installed_name = installed_skill_name(skill_name)
            assert (tmp_path / ".claude" / "skills" / installed_name / "SKILL.md").exists()
        assert "Installed" in result.stdout

    def test_install_selective_skills(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, "skills install --agent claude --skill inference")
        assert_exit_code(result, 0)
        assert (tmp_path / ".claude" / "skills" / "nemo-inference" / "SKILL.md").exists()
        assert not (tmp_path / ".claude" / "skills" / "nemo-setup" / "SKILL.md").exists()

    def test_install_prefixed_skill_does_not_double_prefix(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, "skills install --agent claude --skill nemo-files")
        assert_exit_code(result, 0)
        skill_file = tmp_path / ".claude" / "skills" / "nemo-files" / "SKILL.md"
        assert skill_file.exists()
        assert not (tmp_path / ".claude" / "skills" / "nemo-nemo-files").exists()
        assert "name: nemo-files" in skill_file.read_text()

    @pytest.mark.parametrize(
        ("agent", "relative_path"),
        [
            ("claude", ".claude/skills/nemo-inference/SKILL.md"),
            ("codex", ".agents/skills/nemo-inference/SKILL.md"),
            ("cursor", ".cursor/skills/nemo-inference/SKILL.md"),
            ("opencode", ".opencode/skills/nemo-inference/SKILL.md"),
        ],
    )
    def test_install_uses_cwd_when_parent_has_git_directory(
        self,
        tmp_path: Path,
        monkeypatch,
        agent: str,
        relative_path: str,
    ):
        (tmp_path / ".git").mkdir()
        project_dir = tmp_path / "project" / "subdir"
        project_dir.mkdir(parents=True)
        monkeypatch.chdir(project_dir)

        result = runner.invoke(app, ["skills", "install", "--agent", agent, "--skill", "inference"])

        assert_exit_code(result, 0)
        expected_path = project_dir / relative_path
        assert expected_path.exists()
        assert str(expected_path) in result.stdout
        assert not (tmp_path / relative_path).exists()

    @pytest.mark.parametrize("option", ["--project-dir", "--project-root"])
    def test_install_accepts_explicit_project_dir(self, tmp_path: Path, monkeypatch, option: str):
        cwd = tmp_path / "cwd"
        project_dir = tmp_path / "project"
        cwd.mkdir()
        project_dir.mkdir()
        monkeypatch.chdir(cwd)

        result = runner.invoke(
            app,
            ["skills", "install", "--agent", "claude", "--skill", "inference", option, str(project_dir)],
        )

        assert_exit_code(result, 0)
        expected_path = project_dir / ".claude" / "skills" / "nemo-inference" / "SKILL.md"
        assert expected_path.exists()
        assert str(expected_path) in result.stdout
        assert not (cwd / ".claude").exists()

    def test_install_invalid_skill_name_errors(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, "skills install --agent claude --skill nonexistent")
        assert_exit_code(result, 1)
        assert "Unknown skill" in result.output

    def test_install_duplicate_skill_error_is_clean(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "nemo_platform_ext.cli.commands.skills.cli.load_skills",
            lambda: (_ for _ in ()).throw(DuplicateSkillError("duplicate skill")),
        )
        result = runner.invoke(app, "skills install --agent claude --skill inference")
        assert_exit_code(result, 1)
        assert "Error: duplicate skill" in result.output

    def test_install_plugin_skill_copies_companion_files(self, tmp_path: Path, monkeypatch, example_plugin_skills):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, "skills install --agent claude --skill example-debug")
        assert_exit_code(result, 0)
        skill_dir = tmp_path / ".claude" / "skills" / "nemo-example-debug"
        assert (skill_dir / "SKILL.md").exists()
        assert (skill_dir / "resources" / "notes.md").exists()

    def test_install_codex_writes_per_skill_directories(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, "skills install --agent codex")
        assert_exit_code(result, 0)
        inference = tmp_path / ".agents" / "skills" / "nemo-inference" / "SKILL.md"
        assert inference.exists()
        assert "name: nemo-inference" in inference.read_text()
        # The old marker-comment layout is gone: AGENTS.md must not be touched.
        assert not (tmp_path / "AGENTS.md").exists()

    def test_install_unknown_agent_errors(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, "skills install --agent unknown")
        assert_exit_code(result, 1)
        assert "Unsupported agent" in result.output

    def test_install_cursor_user_scope(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        result = runner.invoke(app, "skills install --agent cursor --user --skill inference")
        assert_exit_code(result, 0)
        assert (tmp_path / ".cursor" / "skills" / "nemo-inference" / "SKILL.md").exists()

    def test_install_requires_agent_or_path(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, "skills install --skill inference")
        assert_exit_code(result, 1)
        assert "provide --agent or --path" in result.output

    def test_install_to_custom_path(self, tmp_path: Path):
        skills_path = tmp_path / "custom" / "skills"
        result = runner.invoke(
            app,
            ["skills", "install", "--path", str(skills_path), "--skill", "inference"],
        )
        assert_exit_code(result, 0)
        skill_file = skills_path / "nemo-inference" / "SKILL.md"
        assert skill_file.exists()
        assert "name: nemo-inference" in skill_file.read_text()


class TestFindProjectRoot:
    def test_uses_cwd_even_when_parent_has_git_directory(self, tmp_path: Path, monkeypatch):
        (tmp_path / ".git").mkdir()
        subdir = tmp_path / "a" / "b"
        subdir.mkdir(parents=True)
        monkeypatch.chdir(subdir)
        from nemo_platform_ext.cli.commands.skills.cli import _find_project_root

        assert _find_project_root() == subdir

    def test_uses_cwd_without_git(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from nemo_platform_ext.cli.commands.skills.cli import _find_project_root

        assert _find_project_root() == tmp_path

    def test_uses_explicit_project_dir(self, tmp_path: Path):
        from nemo_platform_ext.cli.commands.skills.cli import _find_project_root

        assert _find_project_root(tmp_path) == tmp_path


class TestHelpText:
    def test_skills_help(self):
        result = runner.invoke(app, "skills --help")
        assert_exit_code(result, 0)

    def test_list_help(self):
        result = runner.invoke(app, "skills list --help")
        assert_exit_code(result, 0)

    def test_show_help(self):
        result = runner.invoke(app, "skills show --help")
        assert_exit_code(result, 0)

    def test_install_help(self):
        result = runner.invoke(app, "skills install --help")
        assert_exit_code(result, 0)
