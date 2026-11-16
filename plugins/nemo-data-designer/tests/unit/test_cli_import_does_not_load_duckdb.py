# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Building the ``nemo data-designer`` CLI must not import DuckDB.

``import duckdb`` opens DuckDB's default in-memory connection (the package
builds SQL types at import time). In some sandboxes that connection aborts the
process natively, which took down ``nemo data-designer --help`` even though no
CLI command opens DuckDB locally: the work runs on the platform. Data Designer
imports every seed-reader plugin when ``data_designer.interface`` loads, so
our readers must keep the import inside ``create_duckdb_connection``.

Each check runs in a fresh interpreter so modules loaded by other tests in the
same pytest process cannot hide or fake an import.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_ASSERT_NO_DUCKDB = textwrap.dedent(
    """
    import sys

    loaded = sorted(name for name in ("duckdb", "_duckdb") if name in sys.modules)
    assert not loaded, f"DuckDB was imported: {loaded}"
    """
)


def _run_isolated(script: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    # get_cli() seeds default model settings under DATA_DESIGNER_HOME; keep that off the user's home.
    env["DATA_DESIGNER_HOME"] = str(tmp_path / "data-designer-home")
    return subprocess.run(
        [sys.executable, "-c", script + _ASSERT_NO_DUCKDB],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize(
    "module",
    [
        "data_designer_nemo.fileset_file_seed_reader",
        "data_designer_nemo.person_reader",
        "data_designer_nemo.context.check_models",
        # Loads every registered seed-reader plugin, including ours.
        "data_designer.interface.data_designer",
    ],
)
def test_importing_reader_modules_does_not_load_duckdb(module: str, tmp_path: Path) -> None:
    result = _run_isolated(f"import {module}\n", tmp_path)

    assert result.returncode == 0, result.stderr


def test_building_cli_help_does_not_load_duckdb(tmp_path: Path) -> None:
    script = textwrap.dedent(
        """
        from nemo_data_designer_plugin.cli.main import DataDesignerCLI
        from typer.testing import CliRunner

        result = CliRunner().invoke(DataDesignerCLI().get_cli(), ["--help"])
        assert result.exit_code == 0, result.output
        assert "validate" in result.output, result.output
        """
    )

    result = _run_isolated(script, tmp_path)

    assert result.returncode == 0, result.stderr
