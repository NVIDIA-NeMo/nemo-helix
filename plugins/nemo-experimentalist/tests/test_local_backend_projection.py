# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LocalExperimentalistBackend's best-effort projection onto native platform Experiments.

Projection failures must never fail the run or the local-file persistence (spec §3 / F).
"""

from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest
from doubles import fake_client, make_candidate
from nemo_experimentalist_plugin.entities import ExperimentRun
from nemo_experimentalist_plugin.experimentalist.experiment_mirror import ExperimentMirror
from nemo_experimentalist_plugin.experimentalist.experimentalist_backend import LocalExperimentalistBackend
from nemo_experimentalist_plugin.experimentalist.result import ExperimentalistResult

pytestmark = pytest.mark.asyncio


def _run() -> ExperimentRun:
    return ExperimentRun(workspace="default", agent="a", config_snapshot={}, status="running", progress_completed=0)


async def test_create_run_projects_when_client_present(tmp_path: Path) -> None:
    be = LocalExperimentalistBackend(client=fake_client(), path=tmp_path)
    be._mirrors["default"] = AsyncMock()  # injected so no real SDK call is made
    await be.create_run(workspace="default", run=_run())
    be._mirrors["default"].ensure_group.assert_awaited_once()
    assert (tmp_path / "eval-and-optimize" / "run.json").exists()


async def test_create_candidate_projects_when_client_present(tmp_path: Path) -> None:
    be = LocalExperimentalistBackend(client=fake_client(), path=tmp_path)
    be._mirrors["default"] = AsyncMock()
    await be.create_candidate(
        workspace="default",
        candidate=make_candidate(run_id="run-1", label="agent-0", generation=0, description="baseline"),
    )
    be._mirrors["default"].project_candidate.assert_awaited_once()


async def test_persist_result_projects_finalize(tmp_path: Path) -> None:
    be = LocalExperimentalistBackend(client=fake_client(), path=tmp_path)
    be._mirrors["default"] = AsyncMock()
    await be.persist_result(
        workspace="default",
        result=ExperimentalistResult(summary="done", run_id="run-1", progress_completed=1, winner=None),
    )
    be._mirrors["default"].finalize.assert_awaited_once()


async def test_projection_failure_is_swallowed(tmp_path: Path) -> None:
    # A projection failure must never fail the run or the local-file write.
    class _BoomMirror:
        async def ensure_group(self, _run: ExperimentRun) -> None:
            raise RuntimeError("platform down")

    be = LocalExperimentalistBackend(client=fake_client(), path=tmp_path)
    be._mirrors["default"] = cast(ExperimentMirror, _BoomMirror())
    out = await be.create_run(workspace="default", run=_run())  # must NOT raise
    assert out.id
    assert (tmp_path / "eval-and-optimize" / "run.json").exists()
