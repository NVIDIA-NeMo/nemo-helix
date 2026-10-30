# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A trial whose sandbox stopped under it reports that, not the error Harbor hit next."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nhx_evals_sdk.agent_eval.runtimes.harbor.trial_adapter import (
    SANDBOX_TERMINATED_ERROR_TYPE,
    _trial_from_harbor_result,
)
from nhx_evals_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus

# What Harbor records when the agent's sandbox was evicted before the verifier ran.
_FOLLOW_ON_ERROR = {
    "exception_type": "DownloadVerifierDirError",
    "exception_message": "Failed to download verifier directory from environment",
    "exception_traceback": "Traceback (most recent call last): ...",
    "occurred_at": "2026-10-30T13:05:10",
}
_EVICTED = {
    "backend": "opensandbox",
    "sandbox_id": "45c30f94",
    "role": "agent",
    "cause": "evicted",
    "detail": "Kubernetes evicted the sandbox pod: Pod ephemeral local storage usage exceeds the total limit of containers 200Mi.",
    "evidence": {
        "state": "Pending",
        "reason": "CREATING",
        "message": "Sandbox is being created",
        "events": ["Evicted: Pod ephemeral local storage usage exceeds the total limit of containers 200Mi."],
    },
    "observed_at": "2026-10-30T20:05:12+00:00",
}


def _trial(tmp_path: Path, *, exception_info: Any = None, records: dict[str, Any] | None = None) -> AgentEvalTrial:
    for name, record in (records or {}).items():
        content = record if isinstance(record, str) else json.dumps(record)
        (tmp_path / f"nemo-sandbox-termination-{name}.json").write_text(content)
    data = {
        "task_name": "task-1",
        "trial_name": "trial-1",
        "verifier_result": None,
        "exception_info": exception_info,
    }
    return _trial_from_harbor_result(tmp_path, data, reward_key="reward")


def test_a_stopped_sandbox_replaces_the_follow_on_error(tmp_path: Path) -> None:
    trial = _trial(tmp_path, exception_info=_FOLLOW_ON_ERROR, records={"45c30f94": _EVICTED})

    assert trial.error is not None
    assert trial.error.type == SANDBOX_TERMINATED_ERROR_TYPE
    assert trial.error.message == (
        "agent sandbox 45c30f94 evicted: Kubernetes evicted the sandbox pod: Pod ephemeral local storage usage "
        "exceeds the total limit of containers 200Mi.. Harbor then reported DownloadVerifierDirError: Failed to "
        "download verifier directory from environment"
    )
    # Harbor's traceback and timestamp are the evidence for what failed next; both are kept.
    assert trial.error.traceback == _FOLLOW_ON_ERROR["exception_traceback"]
    assert trial.error.occurred_at is not None and trial.error.occurred_at.isoformat() == "2026-10-30T13:05:10"
    assert trial.status is AgentEvalTrialStatus.PARTIAL


def test_every_stopped_sandbox_is_named(tmp_path: Path) -> None:
    verifier = {**_EVICTED, "sandbox_id": "b", "role": "verifier", "cause": "deleted", "detail": "it was deleted"}

    trial = _trial(
        tmp_path, exception_info=_FOLLOW_ON_ERROR, records={"a": {**_EVICTED, "sandbox_id": "a"}, "b": verifier}
    )

    assert trial.error is not None and trial.error.message is not None
    assert trial.error.message.startswith("agent sandbox a evicted: ")
    assert "; verifier sandbox b deleted: it was deleted. Harbor then reported" in trial.error.message


def test_a_trial_without_an_error_keeps_none(tmp_path: Path) -> None:
    """A sandbox that stopped after the trial got what it needed changed nothing the trial reports."""
    trial = _trial(tmp_path, records={"45c30f94": _EVICTED})

    assert trial.error is None


def test_an_error_without_a_stopped_sandbox_is_harbors(tmp_path: Path) -> None:
    trial = _trial(tmp_path, exception_info=_FOLLOW_ON_ERROR)

    assert trial.error is not None and trial.error.type == "DownloadVerifierDirError"


def test_unreadable_records_are_skipped_rather_than_raising(tmp_path: Path) -> None:
    trial = _trial(
        tmp_path,
        exception_info=_FOLLOW_ON_ERROR,
        records={"torn": '{"sandbox_id": "x", "det', "list": [1, 2], "no-detail": {"sandbox_id": "y"}},
    )

    assert trial.error is not None and trial.error.type == "DownloadVerifierDirError"
