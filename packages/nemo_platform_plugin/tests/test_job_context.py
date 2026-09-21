# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`nemo_platform_plugin.job_context`.

Pin the :class:`JobContext` contract:

- Concrete dataclass — construction is always explicit.
- ``job_id`` is ``str | None``; ``None`` means "purely local run".
- ``results`` is typed :class:`JobResults` (sync) — there is no async
  twin; ``NemoJob.run`` runs in the task container where ``save`` is
  invoked synchronously.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import LocalJobResults
from nemo_platform_plugin.job_usage import LocalJobUsageReporter


def _make_storage(tmp_path: Path) -> StoragePaths:
    return StoragePaths(
        ephemeral=tmp_path / "ephemeral",
        persistent=tmp_path / "persistent",
    )


class TestJobContext:
    def test_job_id_defaults_to_none_for_local_runs(self, tmp_path: Path) -> None:
        ctx = JobContext(
            workspace="ws",
            storage=_make_storage(tmp_path),
            results=LocalJobResults(root=tmp_path / "r"),
        )
        assert ctx.job_id is None

    def test_job_id_can_be_set_for_platform_runs(self, tmp_path: Path) -> None:
        ctx = JobContext(
            workspace="ws",
            storage=_make_storage(tmp_path),
            results=LocalJobResults(root=tmp_path / "r"),
            job_id="550e8400-e29b-41d4-a716-446655440000",
        )
        assert ctx.job_id == "550e8400-e29b-41d4-a716-446655440000"

    def test_storage_and_results_round_trip(self, tmp_path: Path) -> None:
        storage = _make_storage(tmp_path)
        results = LocalJobResults(root=tmp_path / "r")
        ctx = JobContext(workspace="ws", storage=storage, results=results)
        assert ctx.storage is storage
        assert ctx.results is results
        assert ctx.workspace == "ws"

    def test_usage_defaults_to_local_reporter(self, tmp_path: Path) -> None:
        ctx = JobContext(
            workspace="ws",
            storage=_make_storage(tmp_path),
            results=LocalJobResults(root=tmp_path / "r"),
        )

        assert isinstance(ctx.usage, LocalJobUsageReporter)

    def test_results_is_required(self, tmp_path: Path) -> None:
        # Omitting ``results`` raises at construction; the dataclass
        # enforces it (no silent ``None`` handed to the job).
        with pytest.raises(TypeError, match="results"):
            JobContext(workspace="ws", storage=_make_storage(tmp_path))  # ty: ignore[missing-argument]
