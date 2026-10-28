# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compass dependency errors remain actionable even without Compass installed."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("missing_dependency", ["compass", "trace_ingest", "unexpected_dependency"])
def test_missing_compass_dependencies(missing_dependency: str) -> None:
    # A fresh process exercises adapter imports without cached compass dependencies.
    script = """
import asyncio
import importlib.abc
import sys
from unittest.mock import AsyncMock

missing = sys.argv[1]

# Simulate a direct or transitive import failure without needing Compass installed.
class BlockDependency(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "compass":
            raise ModuleNotFoundError(f"No module named '{missing}'", name=missing)

sys.meta_path.insert(0, BlockDependency())
from nemo_compass_plugin import fabric_adapter
from nemo_compass_plugin.analysis.run import run_compass

async def check():
    client = AsyncMock()
    expected = ModuleNotFoundError if missing == "unexpected_dependency" else RuntimeError
    try:
        await run_compass(agent="agent", ethos=None, workspace="default", base_url=None, client=client)
    except expected as error:
        if missing == "unexpected_dependency":
            assert error.name == missing
        else:
            assert error.__cause__.name == missing
            message = str(error)
            for text in ("nemo-compass==0.2.0", "Python 3.12 or 3.13", "nemo-helix[all]"):
                assert text in message
    else:
        raise AssertionError("Missing dependency did not fail compass execution")
    client.close.assert_awaited_once()

asyncio.run(check())
"""
    result = subprocess.run(
        [sys.executable, "-c", script, missing_dependency], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
