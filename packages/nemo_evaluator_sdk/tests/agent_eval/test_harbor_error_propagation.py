# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Harbor, real error: the exception-propagation path end to end.

The unit half of this lives in ``test_harbor_runtime.py`` and replays a captured ``result.json``, so
it runs on every PR. This one actually runs Harbor in Docker, which is the only way to catch Harbor
changing *what* it stamps rather than the SDK mis-reading what it stamped.

Marked ``integration`` rather than ``e2e``/``slow`` on purpose: that combination (used by
``test_harbor_runtime_e2e.py``) is selected by no make target and no CI job. ``integration`` at least
runs wherever the plugin's ``test_harbor_plugin_run.py`` does. This older error-rollup check remains
optional; the Experimentalist integration suite owns the required Harbor 0.20 error-plus-reward
parity contract.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import (
    HarborRuntimeConfig,
    run_harbor_eval,
)
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrialStatus

pytestmark = [pytest.mark.integration]

#: A Harbor dataset owned by the tests, holding exactly one permanently-failing task. Deliberately
#: not in ``examples/``: a dataset users are pointed at should not ship a task that always fails.
_ERROR_DATASET_DIR = Path(__file__).parent / "fixtures" / "harbor_error_dataset"
_ERROR_TASK_NAME = "harbor/injected-runtime-error"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        # Bound the probe: a wedged daemon can otherwise hang until the test-level timeout.
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except subprocess.TimeoutExpired:
        return False


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_a_real_harbor_timeout_lands_in_the_summary_error_rollup(tmp_path: Path) -> None:
    """The fixture's oracle sleeps past a 1s agent timeout, so Harbor writes ``exception_info``.

    ``AgentTimeoutError`` is deterministic here only because no timeout multiplier is set: a
    fractional ``agent_timeout_multiplier`` races Harbor's outer ``wait_for`` against the oracle's
    inner Docker timeout, and the inner one raises a plain ``RuntimeError`` instead.
    """
    pytest.importorskip("harbor")
    if not _docker_available():
        pytest.skip("Docker daemon is required to run a Harbor job")

    config = HarborRuntimeConfig(jobs_dir=tmp_path / "jobs", agent_name="oracle")
    result = await run_harbor_eval(config, _ERROR_DATASET_DIR)

    [trial] = result.trials
    assert trial.task_id == _ERROR_TASK_NAME
    # Errored but still scoreable: FAILED would short-circuit the metric entirely.
    assert trial.status is AgentEvalTrialStatus.PARTIAL
    assert trial.error is not None
    assert trial.error.type == "AgentTimeoutError"
    assert trial.error.message is not None and "timed out" in trial.error.message

    # The point of AALGO-428: no re-walking result.trials, no reconstruction helper -- the summary
    # already carries Harbor's exception_stats shape, keyed by trial id.
    assert result.summary.error_trial_ids == {"AgentTimeoutError": [trial.id]}
    assert result.summary.error_count == 1
