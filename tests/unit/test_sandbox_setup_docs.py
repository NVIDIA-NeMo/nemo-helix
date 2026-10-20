# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard OpenShell setup instructions against pruning the platform workspace."""

import json
import re
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "plugins/nemo-deployments"
SKILL = PLUGIN / "src/nemo_deployments_plugin/skills/deploy-sandbox/SKILL.md"
NOTEBOOK = PLUGIN / "examples/openshell/DEMO.ipynb"


def _notebook_sources() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text())
    return ["".join(cell["source"]) for cell in notebook["cells"]]


@pytest.mark.parametrize("path", [SKILL, NOTEBOOK], ids=["skill", "notebook"])
def test_sandbox_syncs_preserve_bootstrapped_workspace(path: Path) -> None:
    text = path.read_text() if path == SKILL else "\n".join(_notebook_sources())
    # Join wrapped inline commands as well as shell line continuations. Scan prose
    # and recovery tables too, not just the main setup code block.
    text = re.sub(r"`(uv sync[^`]+)`", lambda match: match[0].replace("\n", " "), text)
    text = text.replace("\\\n", " ")
    commands = re.findall(r"uv sync[^`\n]*", text)
    assert commands, f"No workspace setup instructions found in {path}"
    for command in commands:
        tokens = shlex.split(command)
        assert tokens == ["uv", "sync", "--frozen", "--all-packages", "--extra", "openshell"], command


def test_notebook_setup_shell_syntax() -> None:
    setup_cells = [source for source in _notebook_sources() if source.startswith("%%bash") and "uv sync" in source]
    assert len(setup_cells) == 1
    # Parse only: never execute installation from a documentation regression test.
    subprocess.run(["bash", "-n"], input=setup_cells[0].removeprefix("%%bash"), text=True, check=True)


def test_openshell_test_targets_preserve_workspace() -> None:
    makefile = (ROOT / "Makefile").read_text()
    for target in ("test-deployments-openshell", "test-jobs-openshell"):
        recipe = makefile.split(f"{target}: ", 1)[1].split("\n\n", 1)[0]
        command = next(line.strip() for line in recipe.splitlines() if "$(UV) run" in line)
        assert shlex.split(command.removesuffix("\\").strip()) == [
            "$(UV)",
            "run",
            "--frozen",
            "--all-packages",
            "--extra",
            "openshell",
        ]


def test_sandbox_setup_extends_bootstrap_without_changing_default() -> None:
    makefile = (ROOT / "Makefile").read_text()
    bootstrap = makefile.split("bootstrap-python: ", 1)[1].split("\n\n", 1)[0]
    assert "$(UV) sync --python $(PYTHON_VERSION) --frozen --all-packages" in bootstrap
    assert "--extra openshell" not in bootstrap
    skill = SKILL.read_text()
    notebook = "\n".join(_notebook_sources())
    for text in (skill, notebook):
        assert "make bootstrap-python" in text
        assert "platform-independent" in text
        assert "glibc >= 2.39" in text  # Keep the deployment's verified host restriction.
        assert "unverified on macOS" in text
        assert '.venv/bin/python -c "import python_on_whales, openshell, nhx.core.models"' in text
        assert "DEPS_OK" in text
        assert "uv run" in text  # Warn about implicit resyncs dropping the extra.
    assert "Linux-only" in skill
    assert "verified only on Linux" in notebook
