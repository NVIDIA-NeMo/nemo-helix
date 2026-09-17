# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-Harbor parity for an errored attempt that still has a verifier reward."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from harbor.models.trial.result import TrialResult
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import (
    HarborRewardMetric,
    HarborRuntimeConfig,
    run_harbor_eval,
)
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrialStatus

from packages.nemo_evaluator_sdk.tests.harbor_fixtures import ErrorAwareQualityMetric

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
    pytest.mark.timeout(300),
    pytest.mark.xdist_group("harbor-evaluator-parity"),
]

_FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "harbor_sdk_error_scoring"
_TASK_NAME = "harbor/timeout-with-reward"


def _require_docker() -> None:
    """Skip locally, but fail CI, when genuine Harbor execution is unavailable."""
    try:
        completed = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        reason = f"Docker is unavailable for the Harbor error-scoring parity test: {exc}"
    else:
        if completed.returncode == 0:
            return
        reason = "Docker is unavailable for the Harbor error-scoring parity test"
        if completed.stderr.strip():
            reason = f"{reason}: {completed.stderr.strip()}"

    if os.environ.get("CI"):
        pytest.fail(reason)
    pytest.skip(reason)


async def test_sdk_preserves_a_real_harbor_error_and_finite_reward(tmp_path: Path) -> None:
    _require_docker()
    jobs_dir = tmp_path / "jobs"
    config = HarborRuntimeConfig(
        jobs_dir=jobs_dir,
        job_name="sdk-error-parity",
        agent_name="oracle",
        n_concurrent_trials=1,
    )

    result = await run_harbor_eval(
        config,
        _FIXTURE_ROOT / "dataset",
        metrics=[HarborRewardMetric(), ErrorAwareQualityMetric()],
    )

    result_paths = list((jobs_dir / "sdk-error-parity").glob("*/result.json"))
    assert len(result_paths) == 1
    harbor_result = TrialResult.model_validate_json(result_paths[0].read_bytes())
    assert harbor_result.task_name == _TASK_NAME
    assert harbor_result.exception_info is not None
    assert harbor_result.exception_info.exception_type == "AgentTimeoutError"
    assert harbor_result.verifier_result is not None
    assert harbor_result.verifier_result.rewards is not None
    assert harbor_result.verifier_result.rewards["reward"] == 0.8

    [trial] = result.trials
    assert trial.status is AgentEvalTrialStatus.PARTIAL
    assert trial.error is not None
    assert trial.error.type == "AgentTimeoutError"

    harbor_rewards = {
        (harbor_result.task_name, harbor_result.trial_name): harbor_result.verifier_result.rewards["reward"]
    }
    sdk_rewards = {
        (score.task_id, score.trial_id): output.value
        for score in result.scores
        if score.metric_type == "harbor_reward"
        for output in score.outputs
        if output.name == "reward"
    }
    assert sdk_rewards == harbor_rewards
    assert result.summary.score("harbor_reward.reward").mean == 0.8
    [quality_score] = [score for score in result.scores if score.metric_type == "error_aware_quality"]
    assert quality_score.trial_id == trial.id
    assert [(output.name, output.value) for output in quality_score.outputs] == [("quality", 1.0)]
    assert quality_score.diagnostics == []
    assert result.summary.metric_coverage["error_aware_quality"]["quality"].model_dump() == {
        "total": 1,
        "scored": 1,
        "failed": 0,
        "missing": 0,
    }
    assert result.summary.error_trial_ids == {"AgentTimeoutError": [trial.id]}
