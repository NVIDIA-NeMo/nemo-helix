# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard: the ``nemo`` CLI depends only on the typed clients in ``nemo_platform_plugin``.

Two complementary checks:

1. A static scan asserts no module under ``nemo_platform_ext.cli`` imports the
   generated ``nemo_platform`` SDK. Modules still being migrated are listed in
   ``MIGRATING`` and must be removed from it as they land.
2. A runtime check runs the CLI in a subprocess with ``nemo_platform`` made
   un-importable, proving the CLI's import graph and the migrated command
   groups work without the generated SDK installed.
3. The same subprocess technique covers the plugin seam: the
   ``nemo_platform_plugin`` discovery/CLI/scheduler modules import cleanly and
   ``nemo --help`` still lists the plugin groups registered via ``nemo.cli``.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import nemo_platform_ext.cli as cli_package
import pytest

CLI_ROOT = Path(cli_package.__file__).parent

# Modules (relative to the cli package) that still import the generated SDK.
# Shrinks to empty as command groups migrate to typed clients.
MIGRATING: frozenset[str] = frozenset()
MIGRATING_DIRS: tuple[str, ...] = ("commands/api/",)

STAINLESS_PACKAGE = "nemo_platform"


def _imports_stainless(tree: ast.AST) -> list[str]:
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == STAINLESS_PACKAGE or alias.name.startswith(f"{STAINLESS_PACKAGE}."):
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == STAINLESS_PACKAGE or node.module.startswith(f"{STAINLESS_PACKAGE}."):
                hits.append(node.module)
    return hits


def _cli_modules() -> list[Path]:
    return sorted(p for p in CLI_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def test_cli_modules_do_not_import_generated_sdk() -> None:
    offenders: dict[str, list[str]] = {}
    for path in _cli_modules():
        relative = path.relative_to(CLI_ROOT).as_posix()
        if relative in MIGRATING or relative.startswith(MIGRATING_DIRS):
            continue
        hits = _imports_stainless(ast.parse(path.read_text(encoding="utf-8")))
        if hits:
            offenders[relative] = hits
    assert not offenders, f"CLI modules importing the generated SDK: {json.dumps(offenders, indent=2)}"


def test_migrating_allow_list_is_current() -> None:
    """Entries in MIGRATING must still exist and still import the SDK; otherwise remove them."""
    stale = []
    for relative in MIGRATING:
        path = CLI_ROOT / relative
        if not path.exists() or not _imports_stainless(ast.parse(path.read_text(encoding="utf-8"))):
            stale.append(relative)
    assert not stale, f"MIGRATING entries no longer needed: {stale}"


_RUNTIME_PROBE = r"""
import importlib.abc
import json
import sys


class _BlockGeneratedSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "nemo_platform" or name.startswith("nemo_platform."):
            raise ImportError(f"generated SDK is not available: {name}")
        return None


sys.meta_path.insert(0, _BlockGeneratedSDK())

from typer.testing import CliRunner  # noqa: E402

from nemo_platform_ext.cli.app import app  # noqa: E402

runner = CliRunner()
results = {}
for args in json.loads(sys.argv[1]):
    result = runner.invoke(app, args, env={"NMP_BASE_URL": "http://127.0.0.1:1"})
    results[" ".join(args)] = {
        "exit_code": result.exit_code,
        "output": (result.stdout + result.stderr)[-2000:],
        "exception": repr(result.exception) if result.exception else None,
    }
print(json.dumps(results))
"""

# Commands that must work without the generated SDK. Migrated groups add
# their ``--help`` and a ``-f code`` invocation (which builds the request
# without a server) here.
#
# ``setup`` and ``services`` are Stainless-free themselves but are not probed
# here: both import the local-services runner (``nmp.platform_runner.config``,
# ``nemo_platform_ext.local.process``), which pulls in server-side packages
# (``nmp.common.service``) that still depend on the generated SDK.
RUNTIME_COMMANDS: list[list[str]] = [
    ["--help"],
    ["--version"],
    ["jobs", "--help"],
    ["jobs", "list", "-f", "code"],
    ["secrets", "--help"],
    ["secrets", "list", "-f", "code"],
    ["secrets", "create", "abc", "--value", "v", "-f", "code"],
    ["secrets", "admin", "rotate-encryption-keys", "-f", "code"],
    ["workspaces", "list", "-f", "code"],
    ["files", "filesets", "list", "-f", "code"],
    ["models", "list", "-f", "code"],
    ["adapters", "list", "-f", "code"],
    ["inference", "providers", "list", "-f", "code"],
    ["inference", "--help"],
    ["projects", "list", "-f", "code"],
    ["iam", "role-bindings", "list", "-f", "code"],
    ["guardrail", "configs", "list", "-f", "code"],
    ["intake", "traces", "list", "-f", "code"],
    ["experiments", "list", "-f", "code"],
    ["wait", "--help"],
    ["chat", "--help"],
]


def _run_probe(commands: list[list[str]], tmp_path: Path) -> dict[str, dict[str, object]]:
    env = {**os.environ, "NMP_CONFIG_FILE": str(tmp_path / "config.yaml")}
    for key in list(env):
        if key.startswith("NMP_") and key != "NMP_CONFIG_FILE":
            env.pop(key)
    completed = subprocess.run(
        [sys.executable, "-c", _RUNTIME_PROBE, json.dumps(commands)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert completed.returncode == 0, f"probe crashed:\nstdout={completed.stdout}\nstderr={completed.stderr}"
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_cli_runs_without_generated_sdk_installed(tmp_path: Path) -> None:
    results = _run_probe(RUNTIME_COMMANDS, tmp_path)
    failures = {cmd: info for cmd, info in results.items() if info["exit_code"] != 0}
    assert not failures, f"commands failing without the generated SDK: {json.dumps(failures, indent=2)}"


# Plugin CLI groups discovered through the ``nemo.cli`` entry-point group. The
# discovery path (``nemo_platform_plugin.discovery`` and the modules it pulls in)
# must not need the generated SDK, or every plugin group silently disappears
# from ``nemo --help``.
PLUGIN_GROUPS: tuple[str, ...] = ("agents", "guardrail", "intake", "experiments")


def test_plugin_groups_are_discovered_without_generated_sdk(tmp_path: Path) -> None:
    results = _run_probe([["--help"], ["agents", "--help"]], tmp_path)

    root_help = results["--help"]
    assert root_help["exit_code"] == 0, root_help
    missing = [group for group in PLUGIN_GROUPS if f"Plugin commands for {group}." not in str(root_help["output"])]
    assert not missing, f"plugin groups missing from `nemo --help` without the generated SDK: {missing}"

    agents_help = results["agents --help"]
    assert agents_help["exit_code"] == 0, agents_help
    assert "unavailable due to import error" not in str(agents_help["output"]), agents_help["output"]


_BLOCK_GENERATED_SDK = (
    "import importlib.abc, sys\n"
    "class B(importlib.abc.MetaPathFinder):\n"
    "    def find_spec(self, name, path, target=None):\n"
    "        if name == 'nemo_platform' or name.startswith('nemo_platform.'):\n"
    "            raise ImportError(name)\n"
    "sys.meta_path.insert(0, B())\n"
)


@pytest.mark.parametrize("module", ["nemo_platform_ext.cli.app", "nemo_platform_ext.cli.core.context"])
def test_core_modules_import_without_generated_sdk(module: str) -> None:
    probe = _BLOCK_GENERATED_SDK + f"import {module}\n"
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr


# Plugin-contract modules the CLI imports to discover and drive plugins. Every
# module they pull in transitively must be importable without the generated SDK.
PLUGIN_CONTRACT_MODULES: tuple[str, ...] = (
    "nemo_platform_plugin.discovery",
    "nemo_platform_plugin.cli",
    "nemo_platform_plugin.commands",
    "nemo_platform_plugin.scheduler",
)


def test_plugin_contract_import_graph_is_free_of_generated_sdk() -> None:
    probe = (
        _BLOCK_GENERATED_SDK
        + "import importlib\n"
        + f"for module in {list(PLUGIN_CONTRACT_MODULES)!r}:\n"
        + "    importlib.import_module(module)\n"
        + "loaded = sorted(name for name in sys.modules if name == 'nemo_platform' or name.startswith('nemo_platform.'))\n"
        + "assert not loaded, loaded\n"
    )
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
