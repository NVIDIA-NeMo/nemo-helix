# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boot the plugin the way the platform does: a clean process with no test env."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

# These boot the plugin in a child process, so guard the module instead: a default sync leaves
# it uninstalled and the repo-wide run still sweeps here.
pytest.importorskip("scaled_evals", reason="scaled-evals plugin not installed")


# Runs in a subprocess so conftest's env presets can't mask a real boot failure.
BOOT_PROBE = """
from importlib.metadata import entry_points

service_entrypoint = next(iter(entry_points(group="nemo.services", name="scaled-evals")))
print("DISCOVERED:", service_entrypoint.name == "scaled-evals")
service_entrypoint.load()()
"""


def test_plugin_loads_without_test_env() -> None:
    result = subprocess.run(
        [sys.executable, "-c", BOOT_PROBE],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=300,
    )
    assert result.returncode == 0, result.stderr

    # The entry-point loader swallows import errors, so a crash here would
    # silently drop the plugin.
    assert "DISCOVERED: True" in result.stdout, result.stdout
