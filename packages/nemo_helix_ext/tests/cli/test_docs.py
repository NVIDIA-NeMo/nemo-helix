# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the nemo docs command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from nemo_helix_ext.cli.app import app
from nemo_helix_ext.quickstart.config import QuickstartConfig
from typer.testing import CliRunner

runner = CliRunner()

qs_no_auth = QuickstartConfig(auth_enabled=False)


def _invoke(*args: str, env: dict[str, str] | None = None):
    """Invoke the CLI with auth disabled."""
    with patch("nemo_helix_ext.quickstart.QuickstartConfig.load", return_value=qs_no_auth):
        return runner.invoke(app, list(args), env=env)


def _listed_topics(output: str) -> list[str]:
    return [line.strip() for line in output.splitlines() if line.startswith("  ")]


class TestDocsCommand:
    def test_docs_help(self):
        result = _invoke("docs", "--help")
        assert result.exit_code == 0
        assert "Read NeMo Helix documentation" in result.stdout

    def test_docs_list(self):
        result = _invoke("docs", "--list")
        assert result.exit_code == 0
        assert "get-started/setup" in result.stdout

    def test_docs_list_independent_of_cwd(self, tmp_path: Path, monkeypatch):
        unrelated_docs = tmp_path / "docs"
        unrelated_docs.mkdir()
        (unrelated_docs / "jenkins-pipeline-guide.mdx").write_text("# Wrong docs root\n")

        monkeypatch.delenv("NHX_DOCS_ROOT", raising=False)
        monkeypatch.chdir(tmp_path)

        result = _invoke("docs", "--list")

        assert result.exit_code == 0
        assert "get-started/setup" in result.stdout
        assert "jenkins-pipeline-guide" not in result.stdout

    def test_docs_root_prefers_repo_docs_with_mdx(self, tmp_path: Path, monkeypatch):
        from nemo_helix_ext.cli.commands import docs

        monkeypatch.delenv("NHX_DOCS_ROOT", raising=False)
        repo_docs = tmp_path / "docs"
        commands_dir = tmp_path / "src" / "nemo_helix_ext" / "cli" / "commands"
        repo_docs.mkdir(parents=True)
        commands_dir.mkdir(parents=True)
        (repo_docs / "index.mdx").write_text("# Repo docs\n")

        result = docs._find_docs_root(commands_dir / "docs.py")

        assert result == repo_docs.resolve()

    def test_docs_root_uses_package_snapshot_without_repo_docs(self, tmp_path: Path, monkeypatch):
        from nemo_helix_ext.cli.commands import docs

        monkeypatch.delenv("NHX_DOCS_ROOT", raising=False)
        package_docs = tmp_path / "site-packages" / "nemo_helix" / "cli" / "docs"
        commands_dir = tmp_path / "site-packages" / "nemo_helix" / "cli" / "commands"
        package_docs.mkdir(parents=True)
        commands_dir.mkdir(parents=True)
        (package_docs / "index.mdx").write_text("# Packaged docs\n")

        result = docs._find_docs_root(commands_dir / "docs.py")

        assert result == package_docs.resolve()

    def test_list_docs_skips_underscore_and_hidden_paths(self, tmp_path: Path):
        from nemo_helix_ext.cli.commands import docs

        (tmp_path / "index.mdx").write_text("# Home\n")
        (tmp_path / "guide").mkdir()
        (tmp_path / "guide" / "topic.mdx").write_text("# Topic\n")
        (tmp_path / "_snippets").mkdir()
        (tmp_path / "_snippets" / "fragment.mdx").write_text("snippet")
        (tmp_path / ".hidden").mkdir()
        (tmp_path / ".hidden" / "secret.mdx").write_text("secret")

        assert docs._list_docs(tmp_path) == ["guide/topic", "index"]

    def test_docs_no_args_shows_list(self):
        result = _invoke("docs")
        assert result.exit_code == 0
        assert "Available documentation topics" in result.stdout

    def test_docs_read_setup(self):
        result = _invoke("docs", "get-started/setup")
        assert result.exit_code == 0
        assert "Setup" in result.stdout
        assert "nemo-setup" in result.stdout

    def test_docs_read_with_mdx_extension(self):
        result = _invoke("docs", "get-started/setup.mdx")
        assert result.exit_code == 0
        assert "Setup" in result.stdout
        assert "nemo-setup" in result.stdout

    def test_docs_nonexistent(self):
        result = _invoke("docs", "nonexistent/path")
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_docs_path_traversal_blocked(self, tmp_path: Path):
        result = _invoke("docs", "../../etc/passwd", env={"NHX_DOCS_ROOT": str(tmp_path)})
        assert result.exit_code == 1
        assert "invalid path" in result.output.lower()

    def test_docs_env_override(self, tmp_path: Path):
        doc_file = tmp_path / "test-topic.mdx"
        doc_file.write_text("# Test Topic\nHello from test.")

        result = _invoke("docs", "test-topic", env={"NHX_DOCS_ROOT": str(tmp_path)})
        assert result.exit_code == 0
        assert "Hello from test" in result.stdout
