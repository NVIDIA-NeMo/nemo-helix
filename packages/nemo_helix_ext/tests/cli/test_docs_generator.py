# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
from pathlib import Path

import pytest
import typer
from nemo_helix_ext.cli.core.lazy_load import ManifestBackedNhxGroup, attach_lazy_entries
from nemo_helix_ext.cli.manifest import TopLevelEntry


def _load_docs_generator():
    for parent in Path(__file__).resolve().parents:
        docs_generator_path = parent / "packages" / "nemo_helix_ext" / "scripts" / "docs_generator.py"
        if docs_generator_path.exists():
            spec = importlib.util.spec_from_file_location("nemo_helix_ext_docs_generator", docs_generator_path)
            if spec is None or spec.loader is None:
                break
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise RuntimeError("Could not locate docs_generator.py")


_docs_generator = _load_docs_generator()
generate_docs = _docs_generator.generate_docs
generate_index_snippet = _docs_generator.generate_index_snippet
write_docs_files = _docs_generator.write_docs_files
with_trailing_newline = _docs_generator._with_trailing_newline
enable_plugin_cli_docs = _docs_generator._enable_plugin_cli_docs
documented_plugin_clis = _docs_generator._DOCUMENTED_PLUGIN_CLIS
plugin_docs_discovery_env = _docs_generator._PLUGIN_DOCS_DISCOVERY_ENV


def test_cli_docs_use_supported_plugins_regardless_of_environment(monkeypatch):
    assert documented_plugin_clis == (
        "agents",
        "anonymizer",
        "garak",
        "customization",
        "data-designer",
        "evals",
        "experiments",
        "guardrail",
        "compass",
        "intake",
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
    docs_app = typer.Typer(cls=ManifestBackedNhxGroup)

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


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "https://github.com/<org>/<repo>/tree/<ref>/<dir>",
            "https://github.com/&lt;org>/&lt;repo>/tree/&lt;ref>/&lt;dir>",
        ),
        ("Use <name> with {key}", r"Use &lt;name> with \{key}"),
        ("Use `<name>` and `{key}`", "Use `<name>` and `{key}`"),
        ("Use ``<name> {key}``", "Use ``<name> {key}``"),
        ("Use ``<name> with a ` tick``", "Use ``<name> with a ` tick``"),
        ("Unmatched ` <name>", "Unmatched ` &lt;name>"),
        ("```text\n<name> {key}\n```", "```text\n<name> {key}\n```"),
    ],
)
def test_escape_mdx_keeps_placeholders_out_of_jsx(source: str, expected: str) -> None:
    assert _docs_generator._escape_mdx(source) == expected


def test_generated_option_help_escapes_url_placeholders() -> None:
    docs_app = typer.Typer()

    @docs_app.command()
    def fetch(
        target: str = typer.Option(help="GitHub tree URL: https://github.com/<org>/<repo>/tree/<ref>/<dir>"),
    ) -> None:
        """Fetch tasks."""

    reference = generate_docs(docs_app, name="nemo")

    assert "https://github.com/&lt;org>/&lt;repo>/tree/&lt;ref>/&lt;dir>" in reference
    assert "<org>" not in reference
    assert r"\<org>" not in reference


def test_write_docs_files_matches_individual_generators(tmp_path):
    docs_app = typer.Typer()

    @docs_app.callback()
    def main() -> None:
        """Test CLI."""

    @docs_app.command(rich_help_panel="Setup")
    def visible() -> None:
        """Visible command."""

    reference_path = tmp_path / "docs/cli/reference.mdx"
    summary_path = tmp_path / "docs/fern/snippets/_snippets/cli-summary.mdx"

    write_docs_files(docs_app, reference_path, summary_path, name="nemo")

    assert reference_path.read_text(encoding="utf-8") == generate_docs(docs_app, name="nemo")
    assert summary_path.read_text(encoding="utf-8") == with_trailing_newline(
        generate_index_snippet(docs_app, name="nemo")
    )


def test_committed_cli_reference_has_no_bare_placeholder_urls() -> None:
    """A URL template in help text must be a code span, not an autolinked dead URL.

    Fern autolinks a bare ``https://`` URL in prose, so an escaped placeholder such as
    ``https://github.com/&lt;org>/...`` becomes a live link whose href GitHub cannot resolve.
    """
    reference = Path(_docs_generator._REFERENCE_DOCS_PATH).read_text(encoding="utf-8")
    prose = re.sub(r"(`+)(?:(?!\1).)+?\1", "", reference)
    offenders = [line for line in prose.splitlines() if re.search(r"https?://\S*&lt;", line)]
    assert offenders == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("pass --auto to use env vars", "pass `--auto` to use env vars"),
        ("Defaults to --email.", "Defaults to `--email`."),
        ("Use `--auto` already coded", "Use `--auto` already coded"),
        ("a well-known -- separator", "a well-known -- separator"),
        ("re-run with -x", "re-run with -x"),
        ("  nemo --base-url http://localhost:8080 setup", "  `nemo --base-url http://localhost:8080 setup`"),
        ("```shell\nnemo x --flag\n```", "```shell\nnemo x --flag\n```"),
    ],
)
def test_escape_mdx_code_formats_prose_flags(source: str, expected: str) -> None:
    # Fern's typographer renders a bare "--flag" as an em dash, which breaks copy-paste.
    assert _docs_generator._escape_mdx(source) == expected


def test_examples_separated_by_blank_lines_stay_in_one_code_block() -> None:
    docs_app = typer.Typer()

    @docs_app.command()
    def upload() -> None:
        """Upload files.

        Examples:
            # First
            nemo files upload ./a my-fileset

            # Second
            nemo files upload ./b my-fileset --remote-path uploads/

        Trailing prose.
        """

    reference = generate_docs(docs_app, name="nemo")

    block = reference.split("```shell\n", 1)[1].split("```", 1)[0]
    assert "nemo files upload ./a my-fileset" in block
    assert "nemo files upload ./b my-fileset --remote-path uploads/" in block
    assert "\\# Second" not in reference
    assert "Trailing prose." in reference.split("```shell", 1)[1].split("```", 2)[1]
