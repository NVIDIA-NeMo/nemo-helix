# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test env for scaled-evals Settings import (must run before package imports)."""

from __future__ import annotations

import os

import pytest

# Each test module guards its own imports, rather than this conftest ignoring the directory:
# a `collect_ignore_glob` here behaved as intended in a standalone reproduction but did not
# take effect in the repo-wide CI run, and per-module skips do not depend on that mechanism.
#
# conftest runs before any of those per-module guards, so it must import nothing the plugin
# brings in.
os.environ.setdefault("DATABASE_SSL_MODE", "disable")
os.environ.setdefault("CONTROL_PLANE_AUTH_ENABLED", "false")


@pytest.fixture(autouse=True)
def _no_immediate_job_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep API tests from reaching Platform Jobs after creating an evaluation."""
    try:
        import scaled_evals.api.routers.evaluations as evaluation_routes  # noqa: PLC0415
    except ImportError:
        return
    monkeypatch.setattr(evaluation_routes, "_submit_now", lambda _evaluation_id: None)
