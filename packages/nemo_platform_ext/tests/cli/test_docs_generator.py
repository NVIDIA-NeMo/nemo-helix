# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
from pathlib import Path

import typer
from nemo_platform_ext.cli.core.lazy_load import ManifestBackedNmpGroup, attach_lazy_entries
from nemo_platform_ext.cli.manifest import TopLevelEntry


def _load_docs_generator():
    for parent in Path(__file__).resolve().parents:
        docs_generator_path = parent / "packages" / "nemo_platform_ext" / "scripts" / "docs_generator.py"
        if docs_generator_path.exists():
            spec = importlib.util.spec_from_file_location("nemo_platform_ext_docs_generator", docs_generator_path)
            if spec is None or spec.loader is None:
                break
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise RuntimeError("Could not locate docs_generator.py")


_docs_generator = _load_docs_generator()
generate_docs = _docs_generator.generate_docs
generate_index_snippet = _docs_generator.generate_index_snippet
enable_plugin_cli_docs = _docs_generator._enable_plugin_cli_docs
documented_plugin_clis = _docs_generator._DOCUMENTED_PLUGIN_CLIS
plugin_docs_discovery_env = _docs_generator._PLUGIN_DOCS_DISCOVERY_ENV


def test_cli_docs_use_supported_plugins_regardless_of_environment(monkeypatch):
    assert documented_plugin_clis == (
        "agents",
        "anonymizer",
        "auditor",
        "customization",
        "data-designer",
        "evaluator",
        "insights",
        "safe-synthesizer",
    )

    for name in plugin_docs_discovery_env:
        monkeypatch.setenv(name, "agent-hardener")

    enable_plugin_cli_docs()

    assert all(os.environ[name] == value for name, value in plugin_docs_discovery_env.items())
    assert os.environ["NEMO_PLUGIN_ALLOWLIST"] == "*"
    assert os.environ["NEMO_PLUGIN_CLI_ALLOWLIST"] == ",".join(documented_plugin_clis)
    assert "agent-hardener" not in documented_plugin_clis


def test_cli_docs_main_includes_supported_plugin_commands(tmp_path):
    env = os.environ.copy()
    env.update({name: "agent-hardener" for name in plugin_docs_discovery_env})
    repo_root = Path(_docs_generator.__file__).resolve().parents[3]

    result = subprocess.run(
        ["uv", "run", "--project", str(repo_root), str(_docs_generator.__file__), "summary"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
        cwd=tmp_path,
    )

    functional_plugins_row = next(
        line for line in result.stdout.splitlines() if line.startswith("| Functional plugins |")
    )
    for plugin_name in documented_plugin_clis:
        assert f"`{plugin_name}`" in functional_plugins_row
    assert "`agent-hardener`" not in functional_plugins_row


def test_cli_docs_main_documents_supported_plugin_subcommands(tmp_path):
    env = os.environ.copy()
    env.update({name: "agent-hardener" for name in plugin_docs_discovery_env})
    repo_root = Path(_docs_generator.__file__).resolve().parents[3]

    result = subprocess.run(
        ["uv", "run", "--project", str(repo_root), str(_docs_generator.__file__), "reference"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
        cwd=tmp_path,
    )

    for plugin_name in documented_plugin_clis:
        assert f"### nemo {plugin_name}\n" in result.stdout
        assert re.search(rf"^#### nemo {re.escape(plugin_name)} \S", result.stdout, re.MULTILINE), (
            f"expected at least one documented subcommand for `nemo {plugin_name}`"
        )
    assert "agent-hardener" not in result.stdout


def test_index_snippet_skips_hidden_lazy_commands_without_loading():
    docs_app = typer.Typer(cls=ManifestBackedNmpGroup)

    @docs_app.callback()
    def main() -> None:
        """Test CLI."""

    @docs_app.command(rich_help_panel="Setup")
    def visible() -> None:
        """Visible command."""

    attach_lazy_entries(
        main,
        (
            TopLevelEntry(
                import_path="unused_docs_generator_test.module:visible",
                help="Manifest visible command help.",
                name="visible",
                panel="Setup",
                kind="command",
            ),
            TopLevelEntry(
                import_path="missing_docs_generator_test.module:app",
                help="Hidden command.",
                name="hidden-command",
                panel="Setup",
                kind="group",
                hidden=True,
            ),
        ),
    )

    snippet = generate_index_snippet(docs_app, name="nemo")
    reference = generate_docs(docs_app, name="nemo")

    assert "`visible`" in snippet
    assert "Manifest visible command help." in reference
    assert "Visible command." not in reference
    assert "hidden-command" not in snippet
    assert "Hidden command." not in snippet
    assert "* `--help, -h`: Show this message and exit." in reference
