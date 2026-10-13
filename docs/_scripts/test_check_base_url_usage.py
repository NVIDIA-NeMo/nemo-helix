# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from docs._scripts import check_base_url_usage
from docs._scripts.check_base_url_usage import check_paths


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _lines(tmp_path: Path, text: str, name: str = "page.mdx") -> list[int]:
    _write(tmp_path, name, text)
    return [violation.line for violation in check_paths([tmp_path])]


@pytest.mark.parametrize(
    "code",
    [
        'base_url=os.environ.get("NHX_BASE_URL", "http://localhost:8080"),',
        'base_url=os.environ.get("NHX_BASE_URL") or "http://localhost:8080",',
        "BASE_URL = os.getenv('NHX_BASE_URL', 'http://localhost:8080')",
        "url = f\"{os.environ['NHX_BASE_URL']}/apis\"",
        'if os.environ["NHX_BASE_URL"] == "x": pass',
    ],
)
def test_flags_python_reads(tmp_path: Path, code: str) -> None:
    assert _lines(tmp_path, f"# Page\n\n```python\nimport os\n{code}\n```\n") == [5]


@pytest.mark.parametrize(
    "code",
    [
        "client = NemoClient.from_config()",
        'os.environ["NHX_BASE_URL"] = "<YOUR_NHX_BASE_URL>"',
    ],
)
def test_allows_python_without_reads(tmp_path: Path, code: str) -> None:
    assert _lines(tmp_path, f"```python\nimport os\n{code}\n```\n") == []


@pytest.mark.parametrize(
    "code",
    [
        "export NHX_BASE_URL=http://localhost:8080",
        "NHX_BASE_URL=https://nhx.example.com nemo models list",
        'export NHX_BASE_URL="https://nhx.example.com"',
    ],
)
def test_flags_shell_literal_assignments(tmp_path: Path, code: str) -> None:
    assert _lines(tmp_path, f"```bash\n{code}\n```\n") == [2]


@pytest.mark.parametrize(
    "code",
    [
        ': "${NHX_BASE_URL:=http://localhost:8080}"',
        'curl "${NHX_BASE_URL:-http://localhost:8080}/health/ready"',
        'export NHX_STUDIO_URL="$NHX_BASE_URL/studio"',
        'export NHX_BASE_URL="${NHX_BASE_URL:-http://localhost:8080}"',
    ],
)
def test_allows_shell_defaults_and_references(tmp_path: Path, code: str) -> None:
    assert _lines(tmp_path, f"```bash\n{code}\n```\n") == []


def test_ignores_other_languages_and_prose(tmp_path: Path) -> None:
    text = (
        "Run `export NHX_BASE_URL=https://nhx.example.com` first.\n\n"
        "```yaml\nenv:\n  NHX_BASE_URL: http://localhost:8080\n```\n\n"
        "```text\nexport NHX_BASE_URL=http://localhost:8080\n```\n"
    )
    assert _lines(tmp_path, text) == []


def test_inline_marker_exempts_only_its_line(tmp_path: Path) -> None:
    text = (
        "```bash\n"
        "export NHX_BASE_URL=http://localhost:8080  # nhx-base-url-allow: starts the platform locally\n"
        "export NHX_BASE_URL=http://localhost:9090\n"
        "```\n"
    )
    assert _lines(tmp_path, text) == [3]


@pytest.mark.parametrize(
    ("name", "marker"),
    [
        ("page.mdx", "{/* @nemo-docs: allow-nhx-base-url */}"),
        ("page.md", "<!-- @nemo-docs: allow-nhx-base-url -->"),
        ("page.mdx", "  {/* @nemo-docs: allow-nhx-base-url */}"),
    ],
)
def test_block_marker_exempts_next_block_only(tmp_path: Path, name: str, marker: str) -> None:
    text = f"{marker}\n```bash\nexport NHX_BASE_URL=https://nhx.example.com\n```\n\n```bash\nexport NHX_BASE_URL=https://nhx.example.com\n```\n"
    assert _lines(tmp_path, text, name=name) == [7]


def test_block_marker_does_not_carry_past_prose(tmp_path: Path) -> None:
    text = "{/* @nemo-docs: allow-nhx-base-url */}\nSome prose.\n\n```bash\nexport NHX_BASE_URL=https://nhx.example.com\n```\n"
    assert _lines(tmp_path, text) == [5]


def test_checks_notebook_cells(tmp_path: Path) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": ["Optional:\n", "\n", "```python\n", 'os.environ["NHX_BASE_URL"] = "<URL>"\n', "```\n"],
            },
            {"cell_type": "code", "source": ["client = NemoClient.from_config()\n"]},
            {
                "cell_type": "code",
                "source": ['NHX_BASE_URL = os.environ.get("NHX_BASE_URL", "http://localhost:8080")\n'],
            },
        ]
    }
    _write(tmp_path, "tutorial.ipynb", json.dumps(notebook))
    violations = check_paths([tmp_path])
    assert [(v.path.name, v.line) for v in violations] == [("tutorial.ipynb", 3)]


def test_accepts_explicit_notebook_path(tmp_path: Path) -> None:
    notebook = {"cells": [{"cell_type": "code", "source": ['os.getenv("NHX_BASE_URL")\n']}]}
    path = _write(tmp_path, "tutorial.ipynb", json.dumps(notebook))
    assert [(v.path.name, v.line) for v in check_paths([path])] == [("tutorial.ipynb", 1)]


def test_skips_generated_docs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    generated = _write(
        tmp_path, "reference.mdx", "```shell\nNHX_BASE_URL=https://nhx.example.com nemo setup --auto\n```\n"
    )
    monkeypatch.setattr(check_base_url_usage, "GENERATED_DOCS", {generated.resolve()})
    assert check_paths([tmp_path]) == []


def test_main_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    clean = tmp_path / "clean"
    clean.mkdir()
    _write(clean, "page.mdx", "```python\nclient = NemoClient.from_config()\n```\n")
    monkeypatch.setattr("sys.argv", ["check_base_url_usage", str(clean)])
    assert check_base_url_usage.main() == 0

    dirty = tmp_path / "dirty"
    dirty.mkdir()
    page = _write(dirty, "page.mdx", "```bash\nexport NHX_BASE_URL=http://localhost:8080\n```\n")
    monkeypatch.setattr("sys.argv", ["check_base_url_usage", str(dirty)])
    assert check_base_url_usage.main() == 1
    assert f"{page}:2:" in capsys.readouterr().out
