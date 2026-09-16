# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`nemo_helix_plugin.run_dependencies` public API boundaries."""

from __future__ import annotations

import subprocess
import sys

from nemo_helix_plugin import run_dependencies


def test_run_dependencies_exports_public_run_dependency_helpers() -> None:
    assert run_dependencies.__all__ == ["LocalRunError", "resolve_run_kwargs"]


def test_dispatcher_import_does_not_import_scheduler() -> None:
    script = (
        "import sys; import nemo_helix_plugin.tasks.dispatcher; print('nemo_helix_plugin.scheduler' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"
