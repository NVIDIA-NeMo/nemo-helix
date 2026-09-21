# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import-boundary tests for Intake ClickHouse persistence."""

import ast
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "nmp" / "intake"
_RAW_CLIENT_MODULE = "nmp.intake.spans.clickhouse_client"
# The plugin CLI talks to Intake over HTTP and never touches ClickHouse; its Typer
# ``@app.command`` registrations would otherwise trip the ``.command(...)`` heuristic.
_CLI_SOURCE_ROOT = _SOURCE_ROOT / "cli_commands"

# Only service composition roots, the local lifecycle provisioner, and the
# executor may depend on the raw runtime client.
_EXPECTED_RAW_CLIENT_IMPORTS = {
    "api/v2/experiments/dependencies.py",
    "local_clickhouse.py",
    "repository/clickhouse/executor.py",
    "service.py",
    "spans/api/dependencies.py",
}
_EXPECTED_LOW_LEVEL_CALL_MODULES = {
    "local_clickhouse.py",
    "repository/clickhouse/executor.py",
    "spans/clickhouse_client.py",
    "spans/clickhouse_migrations.py",
}


def test_raw_clickhouse_client_imports_are_confined_to_approved_modules() -> None:
    imports = {
        path.relative_to(_SOURCE_ROOT).as_posix() for path in _SOURCE_ROOT.rglob("*.py") if _imports_raw_client(path)
    }

    unexpected = imports - _EXPECTED_RAW_CLIENT_IMPORTS
    missing = _EXPECTED_RAW_CLIENT_IMPORTS - imports
    assert imports == _EXPECTED_RAW_CLIENT_IMPORTS, (
        f"Raw client boundary changed; use ClickHouseExecutor for unexpected imports: "
        f"{sorted(unexpected)}; remove stale expected imports: {sorted(missing)}"
    )


def test_low_level_clickhouse_calls_are_confined_to_approved_modules() -> None:
    callers = {
        path.relative_to(_SOURCE_ROOT).as_posix()
        for path in _SOURCE_ROOT.rglob("*.py")
        if not path.is_relative_to(_CLI_SOURCE_ROOT) and _calls_low_level_clickhouse(path)
    }

    unexpected = callers - _EXPECTED_LOW_LEVEL_CALL_MODULES
    missing = _EXPECTED_LOW_LEVEL_CALL_MODULES - callers
    assert callers == _EXPECTED_LOW_LEVEL_CALL_MODULES, (
        f"Low-level ClickHouse boundary changed; route runtime data operations through ClickHouseExecutor: "
        f"{sorted(unexpected)}; remove stale expected callers: {sorted(missing)}"
    )


def _imports_raw_client(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == _RAW_CLIENT_MODULE:
            return True
        if isinstance(node, ast.Import) and any(alias.name == _RAW_CLIENT_MODULE for alias in node.names):
            return True
    return False


def _calls_low_level_clickhouse(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr in {"query", "command"}:
            return True
        if node.func.attr != "insert":
            continue
        receiver = node.func.value
        if isinstance(receiver, ast.Attribute) and receiver.attr == "_executor":
            continue
        return True
    return False
