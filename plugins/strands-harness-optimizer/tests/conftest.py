# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures, and the import path for an uninstalled checkout.

The plugin is opt-in (see README, "Install"), so a repo-wide pytest run may collect these tests
without the package installed.  An installed copy wins: the path is only added when the package
cannot be found.  Strands itself is guarded per module with ``pytest.importorskip``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

if importlib.util.find_spec("strands_harness_optimizer_plugin") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402
from nemo_helix_plugin.job_context import JobContext, StoragePaths  # noqa: E402
from nemo_helix_plugin.job_results import LocalJobResults  # noqa: E402


@pytest.fixture
def source_agent() -> dict[str, Any]:
    return {
        "config_format": "nemo-agents-spec-v1",
        "name": "calculator-agent",
        "default_harness": "deepagents",
        "harnesses": {"deepagents": {"kind": "deepagents"}},
        "models": {
            "default": {
                "provider": "nvidia",
                "model": "calculator-model",
                "api_key_env": "NVIDIA_API_KEY",
                "temperature": 0.0,
            }
        },
        "instructions": {"system": {"content": "old prompt"}},
    }


@pytest.fixture
def ctx(tmp_path: Path) -> JobContext:
    ephemeral = tmp_path / "ephemeral"
    ephemeral.mkdir()
    return JobContext(
        workspace="default",
        storage=StoragePaths(ephemeral=ephemeral),
        results=LocalJobResults(root=tmp_path / "job-results"),
    )
