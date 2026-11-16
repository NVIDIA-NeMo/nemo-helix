# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import LocalJobResults


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
