# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import importlib
import json
import logging
import os
import sys
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

pytest.importorskip("harbor", reason="Harbor tests require the optional harbor extra")

from harbor_fixtures import ErrorAwareQualityMetric, write_harbor_trial_result
from nemo_evaluator_sdk.agent_eval.evaluator import AgentEvaluator
from nemo_evaluator_sdk.agent_eval.metrics import AgentPhaseSuccessMetric
from nemo_evaluator_sdk.agent_eval.results import AgentEvalSummary
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import (
    HarborAgentTaskRunner,
    HarborRewardMetric,
    HarborRuntimeConfig,
    HarborTasksetLoader,
    _build_native_job,
    build_trials_from_job_dir,
    discover_harbor_tasks,
    scoped_harbor_agent_import,
)
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_trial_adapter import (
    _final_agent_message,
    _rewards_mapping,
    _trial_from_harbor_result,
)
from nemo_evaluator_sdk.agent_eval.tasks import (
    AgentEvalRunConfig,
    AgentEvalTask,
    SemanticReducer,
    SemanticView,
    ViewSignal,
)
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, TrialError
from nemo_evaluator_sdk.metrics.protocol import CandidateOutput, DatasetRow, MetricInput
from nemo_evaluator_sdk.metrics.utils import metric_type_name
from nemo_evaluator_sdk.values.evidence import ATIFTraceHandle, OTLPTraceHandle, read_atif
from pydantic import BaseModel, ValidationError

_HELLO_WORLD_DATASET = Path(__file__).resolve().parents[2] / "examples" / "harbor" / "hello_world_dataset"
_FIXTURES = Path(__file__).parent / "fixtures"
# A verbatim Harbor result.json from a real agent-timeout run, host paths scrubbed.
_HARBOR_ERROR_RESULT = _FIXTURES / "harbor_error_result.json"
_EXPECTED_MAX_TRACEBACK_CHARS = 8192
_MISSING = object()


class _BadFloat(float):
    def __float__(self) -> float:
        raise RuntimeError("must not be called")


class _BadStr(str):
    def __len__(self) -> int:
        raise RuntimeError("must not be called")


def _write_trial(
    job_dir: Path,
    trial_name: str,
    task_name: str,
    *,
    reward: float | None,
    exception: str | Mapping[str, object] | None = None,
) -> None:
    """Write one complete Harbor-valid trial result."""
    write_harbor_trial_result(
        job_dir / trial_name,
        task_name=task_name,
        rewards=None if reward is None else {"reward": reward},
        exception=exception,
    )


def _write_rewards_trial(job_dir: Path, trial_name: str, task_name: str, rewards: Mapping[str, float | int]) -> None:
    write_harbor_trial_result(job_dir / trial_name, task_name=task_name, rewards=rewards)


def _adapt_raw_trial(
    tmp_path: Path,
    *,
    rewards: Mapping[object, object] | None,
    exception_info: object | None = None,
    reward_key: str = "reward",
) -> AgentEvalTrial:
    """Exercise defensive normalization below the Harbor-valid file boundary."""
    trial_dir = tmp_path / "raw__trial"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "verifier").mkdir()
    return _trial_from_harbor_result(
        trial_dir,
        {
            "task_name": "t",
            "trial_name": trial_dir.name,
            "verifier_result": None if rewards is None else {"rewards": rewards},
            "exception_info": exception_info,
        },
        reward_key=reward_key,
    )


@pytest.mark.parametrize(
    ("value", "expected", "reason"),
    [
        (1, 1.0, None),
        ("1.25", 1.25, None),
        (True, None, "boolean"),
        ("bad", None, "non_numeric"),
        (float("nan"), None, "non_finite"),
        (float("inf"), None, "non_finite"),
        (float("-inf"), None, "non_finite"),
    ],
)
def test_harbor_reward_parser_is_total_and_finite(value: object, expected: float | None, reason: str | None) -> None:
    parsed = _rewards_mapping({"verifier_result": {"rewards": {"score": value, "sibling": 2}}})

    assert parsed.values == ({"score": expected, "sibling": 2.0} if expected is not None else {"sibling": 2.0})
    assert parsed.rejected_by_key == ({} if reason is None else {"score": reason})
    assert parsed.rejected_entries == ()


def test_harbor_reward_parser_preserves_punctuation_and_redacts_bad_keys() -> None:
    parsed = _rewards_mapping(
        {
            "verifier_result": {
                "rewards": {
                    "criteria.legal": 1,
                    "": 2,
                    "bad\nkey": 3,
                    "x" * 256: 4,
                    "score.pass@2": 5,
                }
            }
        }
    )

    assert parsed.values == {"criteria.legal": 1.0}
    assert parsed.rejected_by_key == {}
    assert parsed.rejected_entries == ("invalid_key", "invalid_key", "invalid_key", "reserved_key")


def test_harbor_reward_parser_rejects_hostile_subclasses_without_losing_siblings() -> None:
    parsed = _rewards_mapping(
        {
            "verifier_result": {
                "rewards": {
                    "reward": 1.0,
                    "bad_value": _BadFloat(2.0),
                    _BadStr("bad_key"): 3.0,
                }
            }
        }
    )

    assert parsed.values == {"reward": 1.0}
    assert parsed.rejected_by_key == {"bad_value": "non_numeric"}
    assert parsed.rejected_entries == ("invalid_key",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "expected_value", "reason"),
    [
        pytest.param(_MISSING, 0.0, "absent", id="missing"),
        pytest.param(True, 0.0, "boolean", id="boolean"),
        pytest.param("1.25", 1.25, None, id="finite-numeric-string"),
        pytest.param("bad", 0.0, "non_numeric", id="non-numeric-string"),
        pytest.param(float("nan"), 0.0, "non_finite", id="nan"),
        pytest.param(float("inf"), 0.0, "non_finite", id="positive-infinity"),
        pytest.param(float("-inf"), 0.0, "non_finite", id="negative-infinity"),
    ],
)
async def test_primary_reward_matrix_survives_adaptation_and_metric_diagnostics(
    tmp_path: Path,
    raw: object,
    expected_value: float,
    reason: str | None,
) -> None:
    rewards: dict[object, object] = {"sibling": 0.5}
    if raw is not _MISSING:
        rewards["score"] = raw
    trial = _adapt_raw_trial(tmp_path, rewards=rewards, reward_key="score")
    result = await HarborRewardMetric(output_name="score", reward_keys=("score",)).compute_scores(
        MetricInput(row=DatasetRow(data={}), candidate=CandidateOutput(metadata=trial.metadata))
    )

    assert [(output.name, output.value) for output in result.outputs] == [("score", expected_value)]
    if reason is None:
        assert trial.metadata["reward_details"]["score"] == expected_value
        assert "score" not in trial.metadata["reward_rejections"]
        assert result.diagnostics == []
        assert trial.status is AgentEvalTrialStatus.COMPLETED
    else:
        expected_rejections = {} if raw is _MISSING else {"score": reason}
        assert trial.metadata["reward_rejections"] == expected_rejections
        assert [diagnostic.details for diagnostic in result.diagnostics] == [{"output": "score", "reason": reason}]
        assert trial.status is AgentEvalTrialStatus.PARTIAL


@pytest.mark.asyncio
async def test_harbor_valid_boolean_primary_remains_unusable_after_file_validation(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    _write_rewards_trial(job_dir, "t__a", "t", {"score": 1.0, "sibling": 0.5})
    result_path = job_dir / "t__a" / "result.json"
    payload = json.loads(result_path.read_text())
    payload["verifier_result"]["rewards"]["score"] = True
    result_path.write_text(json.dumps(payload))
    task = AgentEvalTask(id="t", intent="t", inputs={"instruction": "t"}, metrics=[HarborRewardMetric()])

    trial = build_trials_from_job_dir(job_dir, [task], reward_key="score")[0]
    result = await HarborRewardMetric(output_name="score", reward_keys=("score",)).compute_scores(
        MetricInput(row=DatasetRow(data={}), candidate=CandidateOutput(metadata=trial.metadata))
    )

    assert trial.status is AgentEvalTrialStatus.PARTIAL
    assert trial.metadata["reward"] is None
    assert trial.metadata["reward_rejections"] == {"score": "boolean"}
    assert [(output.name, output.value) for output in result.outputs] == [("score", 0.0)]
    assert [diagnostic.details for diagnostic in result.diagnostics] == [{"output": "score", "reason": "boolean"}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "expected_secondary", "reason"),
    [
        pytest.param(_MISSING, None, "absent", id="missing"),
        pytest.param(True, None, "boolean", id="boolean"),
        pytest.param("1.25", 1.25, None, id="finite-numeric-string"),
        pytest.param("bad", None, "non_numeric", id="non-numeric-string"),
        pytest.param(float("nan"), None, "non_finite", id="nan"),
        pytest.param(float("inf"), None, "non_finite", id="positive-infinity"),
        pytest.param(float("-inf"), None, "non_finite", id="negative-infinity"),
    ],
)
async def test_secondary_reward_matrix_survives_adaptation_and_metric_diagnostics(
    tmp_path: Path,
    raw: object,
    expected_secondary: float | None,
    reason: str | None,
) -> None:
    rewards: dict[object, object] = {"score": 1.0}
    if raw is not _MISSING:
        rewards["format_ok"] = raw
    trial = _adapt_raw_trial(tmp_path, rewards=rewards, reward_key="score")
    result = await HarborRewardMetric(output_name="score", reward_keys=("score", "format_ok")).compute_scores(
        MetricInput(row=DatasetRow(data={}), candidate=CandidateOutput(metadata=trial.metadata))
    )

    expected_outputs = [("score", 1.0)]
    if expected_secondary is not None:
        expected_outputs.append(("format_ok", expected_secondary))
    assert [(output.name, output.value) for output in result.outputs] == expected_outputs
    assert trial.status is AgentEvalTrialStatus.COMPLETED
    if reason is None:
        assert trial.metadata["reward_details"]["format_ok"] == expected_secondary
        assert "format_ok" not in trial.metadata["reward_rejections"]
        assert result.diagnostics == []
    else:
        expected_rejections = {} if raw is _MISSING else {"format_ok": reason}
        assert trial.metadata["reward_rejections"] == expected_rejections
        assert [diagnostic.details for diagnostic in result.diagnostics] == [{"output": "format_ok", "reason": reason}]


@pytest.mark.parametrize("reward_key", ["", "bad\tkey", "x" * 256, "score.pass@2"])
def test_harbor_runner_rejects_invalid_primary_before_running(tmp_path: Path, reward_key: str) -> None:
    with pytest.raises(ValueError, match="reward_key"):
        HarborAgentTaskRunner(job_dir=tmp_path, reward_key=reward_key)
    with pytest.raises(ValueError, match="reward_key"):
        build_trials_from_job_dir(tmp_path, [], reward_key=reward_key)


@pytest.mark.asyncio
async def test_harbor_runner_scores_through_agent_evaluator_and_adapts_legacy_payload(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    # Top-level aggregate result.json must be ignored (only */result.json are trials).
    (job_dir / "result.json").write_text(json.dumps({"stats": {}}))
    _write_trial(job_dir, "pass-task__aaa", "pass-task", reward=1.0)
    _write_trial(job_dir, "fail-task__bbb", "fail-task", reward=0.0, exception="NonZeroAgentExitCodeError")
    # A trial whose verifier emitted no reward at all (verifier_result=None).
    _write_trial(job_dir, "noreward-task__ccc", "noreward-task", reward=None)

    tasks = [
        AgentEvalTask(id="pass-task", intent="x", inputs={"instruction": "p"}, metrics=[HarborRewardMetric()]),
        AgentEvalTask(id="fail-task", intent="y", inputs={"instruction": "q"}, metrics=[HarborRewardMetric()]),
        AgentEvalTask(id="noreward-task", intent="z", inputs={"instruction": "r"}, metrics=[HarborRewardMetric()]),
    ]

    # Direct adaptation: reward stays metadata while usage is typed; exceptions remain scoreable.
    trials = {t.task_id: t for t in build_trials_from_job_dir(job_dir, tasks)}
    assert trials["pass-task"].status == AgentEvalTrialStatus.COMPLETED
    assert trials["pass-task"].metadata["reward"] == 1.0
    assert trials["pass-task"].measurements.prompt_tokens == 100
    assert trials["pass-task"].measurements.cost_usd == 0.25
    assert "prompt_tokens" not in trials["pass-task"].metadata
    assert "cost_usd" not in trials["pass-task"].metadata
    assert trials["pass-task"].evidence is not None
    assert trials["fail-task"].status == AgentEvalTrialStatus.PARTIAL
    assert trials["fail-task"].error is not None
    assert trials["fail-task"].error.type == "NonZeroAgentExitCodeError"
    # The typed field replaced the metadata stamp outright: one source of truth, no mirror to drift.
    assert "exception_type" not in trials["fail-task"].metadata
    # Missing reward: no explicit reward -> PARTIAL, metadata reward is None, scores as 0.0.
    assert trials["noreward-task"].status == AgentEvalTrialStatus.PARTIAL
    assert trials["noreward-task"].metadata["reward"] is None

    # run_job is awaited exactly once, then the job dir is adapted and scored end-to-end.
    calls = []
    runner = HarborAgentTaskRunner(job_dir=job_dir, run_job=lambda: _record(calls))
    result = await AgentEvaluator().run(tasks=tasks, target=runner, config=AgentEvalRunConfig())
    assert calls == ["ran"]

    rewards_by_task = {score.task_id: score.outputs[0].value for score in result.scores if score.outputs}
    assert rewards_by_task == {"pass-task": 1.0, "fail-task": 0.0, "noreward-task": 0.0}

    # ...and the summary carries Harbor's own trial-keyed shape, with no reconstruction needed.
    assert result.summary.error_trial_ids == {"NonZeroAgentExitCodeError": ["fail-task__bbb"]}
    assert result.summary.error_count == 1


async def _record(calls: list[str]) -> None:
    calls.append("ran")


def _reward_stats_from_summary(
    summary: AgentEvalSummary,
    *,
    metric_type: str = "harbor_reward",
    output_name: str = "reward",
) -> dict[str, dict[float, list[str]]]:
    """Rebuild Harbor's *own* ``reward_stats`` — ``{reward_key: {value: [trial_name, ...]}}``.

    Harbor builds it as ``reward_stats.setdefault(value, []).append(trial_result.trial_name)``: keyed
    by the raw numeric value, listing trial names. ``_trial_from_harbor_result`` stamps Harbor's
    ``trial_name`` straight onto ``AgentEvalTrial.id``, which is the ``trial_id`` each record now
    carries, so this reproduces Harbor's shape rather than approximating it.

    A ``None`` value is a trial that died before the verifier ran, and is skipped here — as is any
    non-numeric value, since Harbor keys ``reward_stats`` by the reward value itself.

    One caveat this does *not* reproduce: Harbor files a trial in ``reward_stats`` only when
    ``verifier_result.rewards`` exists, so a trial that crashed before the verifier ran appears in
    ``exception_stats`` alone. The SDK synthesises ``0.0`` for it (see ``HarborRewardMetric``), so it
    also lands in ``task_metric_values`` — ``gamma__a`` below is exactly that case. Reconciling the
    two is AALGO-441, not this helper.
    """
    key = f"{metric_type}.{output_name}"
    stats: dict[str, dict[float, list[str]]] = {}
    for values_by_key in summary.task_metric_values.values():
        for record in values_by_key.get(key, []):
            if not isinstance(record.value, bool | int | float):
                continue
            stats.setdefault(output_name, {}).setdefault(float(record.value), []).append(record.trial_id)
    return stats


@pytest.mark.asyncio
async def test_harbor_reward_stats_is_derivable_from_summary_task_metric_values(tmp_path: Path) -> None:
    """The summary alone reproduces Harbor's ``reward_stats``, which is what AALGO-310 exists to enable.

    Harbor groups rewards by ``trial_name`` and keys them by the raw ``float | int``. Both were out of
    reach while values were bare numbers indexed by position; now that each record names its trial,
    AALGO-441 can rebuild the real shape without re-walking ``result.scores``.

    ``exception_stats`` is now covered too, by
    :func:`test_harbor_exception_stats_is_read_straight_off_the_summary` below (AALGO-428).
    """
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    # Two trials each for alpha (flaky) and beta (solved); one for gamma, whose verifier emitted
    # no reward at all -> PARTIAL trial that still scores 0.0.
    _write_trial(job_dir, "alpha__a", "alpha", reward=1.0)
    _write_trial(job_dir, "alpha__b", "alpha", reward=0.0)
    _write_trial(job_dir, "beta__a", "beta", reward=1.0)
    _write_trial(job_dir, "beta__b", "beta", reward=1.0)
    _write_trial(job_dir, "gamma__a", "gamma", reward=None)

    tasks = [
        AgentEvalTask(id=task_id, intent="x", inputs={"instruction": "p"}, metrics=[HarborRewardMetric()])
        for task_id in ("alpha", "beta", "gamma")
    ]
    runner = HarborAgentTaskRunner(job_dir=job_dir, run_job=lambda: _record([]))
    result = await AgentEvaluator().run(tasks=tasks, target=runner, config=AgentEvalRunConfig())

    recorded = {
        task_id: {key: [(a.trial_id, a.value) for a in records] for key, records in by_key.items()}
        for task_id, by_key in result.summary.task_metric_values.items()
    }
    assert recorded == {
        "alpha": {"harbor_reward.reward": [("alpha__a", 1.0), ("alpha__b", 0.0)]},
        "beta": {"harbor_reward.reward": [("beta__a", 1.0), ("beta__b", 1.0)]},
        "gamma": {"harbor_reward.reward": [("gamma__a", 0.0)]},
    }

    # Harbor's own shape: raw float keys, trial names in the lists.
    assert _reward_stats_from_summary(result.summary) == {
        "reward": {1.0: ["alpha__a", "beta__a", "beta__b"], 0.0: ["alpha__b", "gamma__a"]}
    }


@pytest.mark.asyncio
async def test_harbor_exception_stats_is_read_straight_off_the_summary(tmp_path: Path) -> None:
    """AALGO-428: ``summary.error_trial_ids`` *is* Harbor's ``exception_stats``, not an approximation.

    The proof is the absence of a helper. ``reward_stats`` needs ``_reward_stats_from_summary`` above
    to re-key task-major records into Harbor's shape; this needs nothing — the field is already
    ``{exception type: [trial_name, ...]}``, because Harbor's ``trial_name`` is ``AgentEvalTrial.id``.

    Two things this pins that a later refactor could plausibly "tidy" away:

    - ``beta__a`` errored *and* scored 1.0. It appears in the rollup **and** counts as a pass in
      ``task_metric_values`` — Harbor double-files it the same way, because its reward and exception
      branches are independent.
    - ``gamma__a`` is ``PARTIAL``, not ``FAILED`` (an errored Harbor trial stays scoreable). Filtering
      this rollup by status would drop exactly the trials it exists to name.
    """
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    _write_trial(job_dir, "alpha__a", "alpha", reward=1.0)  # clean
    _write_trial(job_dir, "alpha__b", "alpha", reward=0.0, exception={"exception_type": "RuntimeError"})
    _write_trial(job_dir, "beta__a", "beta", reward=1.0, exception={"exception_type": "RuntimeError"})
    _write_trial(job_dir, "gamma__a", "gamma", reward=None, exception={"exception_type": "TimeoutError"})

    tasks = [
        AgentEvalTask(id=task_id, intent="x", inputs={"instruction": "p"}, metrics=[HarborRewardMetric()])
        for task_id in ("alpha", "beta", "gamma")
    ]
    runner = HarborAgentTaskRunner(job_dir=job_dir, run_job=lambda: _record([]))
    result = await AgentEvaluator().run(tasks=tasks, target=runner, config=AgentEvalRunConfig())

    assert result.summary.error_trial_ids == {
        "RuntimeError": ["alpha__b", "beta__a"],
        "TimeoutError": ["gamma__a"],
    }
    assert result.summary.error_count == 3

    # The errored-but-rewarded trial keeps its reward and its pass, exactly as Harbor reports it.
    assert _reward_stats_from_summary(result.summary)["reward"][1.0] == ["alpha__a", "beta__a"]
    by_trial = {t.id: t for t in result.trials}
    assert by_trial["beta__a"].status is AgentEvalTrialStatus.PARTIAL
    assert by_trial["gamma__a"].status is AgentEvalTrialStatus.PARTIAL


@pytest.mark.asyncio
async def test_errored_harbor_rewards_and_metric_owned_exclusions_are_independent(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    _write_rewards_trial(job_dir, "t__a_success", "t", {"reward": 1.0, "format_ok": 1.0})
    _write_trial(job_dir, "t__b_runtime_reward", "t", reward=0.8, exception="RuntimeError")
    _write_trial(job_dir, "t__c_runtime_missing", "t", reward=None, exception="RuntimeError")
    _write_trial(job_dir, "t__d_timeout", "t", reward=0.6, exception="AgentTimeoutError")
    task = AgentEvalTask(
        id="t",
        intent="test",
        inputs={"instruction": "test"},
        metrics=[HarborRewardMetric(), ErrorAwareQualityMetric()],
        views={
            "quality": SemanticView(
                reducer=SemanticReducer.MEAN,
                signals=[ViewSignal(metric="error_aware_quality", output="quality")],
            )
        },
    )

    result = await AgentEvaluator().run(tasks=[task], target=HarborAgentTaskRunner(job_dir=job_dir))

    assert [(trial.id, trial.status) for trial in result.trials] == [
        ("t__a_success", AgentEvalTrialStatus.COMPLETED),
        ("t__b_runtime_reward", AgentEvalTrialStatus.PARTIAL),
        ("t__c_runtime_missing", AgentEvalTrialStatus.PARTIAL),
        ("t__d_timeout", AgentEvalTrialStatus.PARTIAL),
    ]
    harbor_scores = [score for score in result.scores if score.metric_type == "harbor_reward"]
    assert [(score.trial_id, score.outputs[0].value) for score in harbor_scores] == [
        ("t__a_success", 1.0),
        ("t__b_runtime_reward", 0.8),
        ("t__c_runtime_missing", 0.0),
        ("t__d_timeout", 0.6),
    ]
    missing_reward_score = next(score for score in harbor_scores if score.trial_id == "t__c_runtime_missing")
    assert {"output": "reward", "reason": "absent"} in [
        diagnostic.details for diagnostic in missing_reward_score.diagnostics
    ]
    assert result.summary.score("harbor_reward.reward").mean == pytest.approx(0.6)
    assert result.summary.metric_coverage["harbor_reward"]["format_ok"].model_dump() == {
        "total": 4,
        "scored": 1,
        "failed": 0,
        "missing": 3,
    }

    quality_scores = [score for score in result.scores if score.metric_type == "error_aware_quality"]
    assert [[output.value for output in score.outputs] for score in quality_scores] == [[1.0], [], [], [1.0]]
    assert [
        diagnostic.details
        for score in quality_scores
        for diagnostic in score.diagnostics
        if diagnostic.details is not None
    ] == [
        {"output": "quality", "reason": "excluded_error_type"},
        {"output": "quality", "reason": "excluded_error_type"},
    ]
    assert result.summary.metric_coverage["error_aware_quality"]["quality"].model_dump() == {
        "total": 4,
        "scored": 2,
        "failed": 0,
        "missing": 2,
    }
    quality_view = result.summary.score("view.quality")
    assert (quality_view.count, quality_view.nan_count, quality_view.mean) == (2, 2, 1.0)
    assert result.summary.error_trial_ids == {
        "RuntimeError": ["t__b_runtime_reward", "t__c_runtime_missing"],
        "AgentTimeoutError": ["t__d_timeout"],
    }
    assert result.summary.trial_count == 4


def test_reward_with_no_matching_reward_key_is_partial_and_warns(tmp_path: Path, caplog) -> None:
    # Verifier emitted a reward, but under a key we didn't ask for: no guessing —
    # the trial is treated as having no reward (None -> PARTIAL, scores 0.0) and warns.
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    _write_trial(job_dir, "t__aaa", "t", reward=1.0)  # emitted under "reward"
    tasks = [AgentEvalTask(id="t", intent="x", inputs={"instruction": "p"}, metrics=[HarborRewardMetric()])]

    with caplog.at_level(logging.WARNING):
        trials = build_trials_from_job_dir(job_dir, tasks, reward_key="missing")

    assert trials[0].metadata["reward"] is None
    assert trials[0].status == AgentEvalTrialStatus.PARTIAL
    assert "none matches reward_key" in caplog.text


def test_rejected_primary_is_distinguished_and_trial_metadata_is_sanitized(tmp_path: Path, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        trial = _adapt_raw_trial(
            tmp_path,
            rewards={
                "score": True,
                "format_ok": "1",
                "shape_ok": "bad",
                "": 1,
                "derived.pass@2": 1,
            },
            reward_key="score",
        )

    assert trial.metadata["reward"] is None
    assert trial.metadata["reward_details"] == {"format_ok": 1.0}
    assert trial.metadata["reward_rejections"] == {"score": "boolean", "shape_ok": "non_numeric"}
    assert trial.metadata["reward_entry_rejections"] == ["invalid_key", "reserved_key"]
    assert "emitted but rejected as boolean" in caplog.text
    assert "derived.pass@2" not in repr(trial.metadata)


@pytest.mark.asyncio
async def test_harbor_runner_finalizes_sparse_outputs_per_task(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    _write_rewards_trial(job_dir, "a__1", "A", {"score": 1, "format_ok": 1, "shape.ok": 0.5})
    _write_rewards_trial(job_dir, "a__2", "A", {"score": 0})
    _write_rewards_trial(job_dir, "b__1", "B", {"score": 1})

    view = SemanticView(
        reducer=SemanticReducer.MEAN,
        signals=[ViewSignal(metric="agent_phase_success", output="agent_phase_success")],
    )
    tasks = [
        AgentEvalTask(
            id="A",
            intent="A",
            inputs={"instruction": "a"},
            metrics=[HarborRewardMetric(), AgentPhaseSuccessMetric()],
            views={"agent_ok": view},
        ),
        AgentEvalTask(id="B", intent="B", inputs={"instruction": "b"}, metrics=[HarborRewardMetric()]),
    ]

    result = await AgentEvaluator().run(
        tasks=tasks,
        target=HarborAgentTaskRunner(job_dir=job_dir, reward_key="score"),
    )

    by_task = {task.id: task for task in result.tasks}
    a_metric = next(metric for metric in by_task["A"].metrics if metric_type_name(metric) == "harbor_reward")
    b_metric = next(metric for metric in by_task["B"].metrics if metric_type_name(metric) == "harbor_reward")
    assert [spec.name for spec in a_metric.output_spec()] == ["score", "format_ok", "shape.ok"]
    assert [spec.required for spec in a_metric.output_spec()] == [True, False, False]
    assert [spec.name for spec in b_metric.output_spec()] == ["score"]
    assert metric_type_name(by_task["A"].metrics[1]) == "agent_phase_success"
    assert by_task["A"].views == {"agent_ok": view}

    format_score = result.summary.score("harbor_reward.format_ok")
    shape_score = result.summary.score("harbor_reward.shape.ok")
    assert (format_score.count, format_score.nan_count, format_score.mean) == (1, 1, 1.0)
    assert (shape_score.count, shape_score.nan_count, shape_score.mean) == (1, 1, 0.5)
    assert result.summary.metric_coverage["harbor_reward"]["format_ok"].missing == 1
    assert result.summary.metric_coverage["harbor_reward"]["shape.ok"].missing == 1

    a_scores = [score for score in result.scores if score.task_id == "A" and score.metric_type == "harbor_reward"]
    assert [[output.name for output in score.outputs] for score in a_scores] == [
        ["score", "format_ok", "shape.ok"],
        ["score"],
    ]


@pytest.mark.asyncio
async def test_harbor_runner_retains_predeclared_secondary_when_every_trial_omits_it(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    _write_rewards_trial(job_dir, "a__1", "A", {"score": 1})
    _write_rewards_trial(job_dir, "a__2", "A", {"score": 0})

    task = AgentEvalTask(
        id="A",
        intent="A",
        inputs={"instruction": "a"},
        metrics=[HarborRewardMetric(output_name="score", reward_keys=("score", "format_ok"))],
        views={
            "format_quality": SemanticView(
                reducer=SemanticReducer.MEAN,
                signals=[ViewSignal(metric="harbor_reward", output="format_ok")],
            )
        },
    )

    result = await AgentEvaluator().run(
        tasks=[task],
        target=HarborAgentTaskRunner(job_dir=job_dir, reward_key="score"),
    )

    output_specs = result.tasks[0].metrics[0].output_spec()
    assert [(spec.name, spec.required) for spec in output_specs] == [("score", True), ("format_ok", False)]

    format_score = result.summary.score("harbor_reward.format_ok")
    assert (format_score.count, format_score.nan_count, format_score.mean) == (0, 2, None)
    assert result.summary.metric_coverage["harbor_reward"]["format_ok"].model_dump() == {
        "total": 2,
        "scored": 0,
        "failed": 0,
        "missing": 2,
    }

    format_view = result.summary.score("view.format_quality")
    assert (format_view.count, format_view.nan_count, format_view.mean) == (0, 2, None)
    for k in (1, 2):
        pass_score = result.summary.score(f"harbor_reward.format_ok.pass@{k}")
        assert (pass_score.count, pass_score.nan_count, pass_score.mean) == (0, 1, None)


def test_harbor_scoring_task_output_order_is_attempt_order_independent(tmp_path: Path) -> None:
    runner = HarborAgentTaskRunner(job_dir=tmp_path, reward_key="score")
    task = AgentEvalTask(id="A", intent="A", inputs={"instruction": "a"}, metrics=[HarborRewardMetric()])
    trials = [
        AgentEvalTrial(
            id="a2",
            task_id="A",
            status=AgentEvalTrialStatus.PARTIAL,
            output=None,
            metadata={"reward_details": {"z": 1.0}, "reward_rejections": {"a": "boolean"}},
        ),
        AgentEvalTrial(
            id="a1",
            task_id="A",
            status=AgentEvalTrialStatus.PARTIAL,
            output=None,
            metadata={"reward_details": {"middle": 1.0}},
        ),
    ]

    forward = runner.scoring_metrics(task, trials)[0].output_spec()
    reverse = runner.scoring_metrics(task, list(reversed(trials)))[0].output_spec()

    assert [spec.name for spec in forward] == ["score", "a", "middle", "z"]
    assert [spec.name for spec in reverse] == ["score", "a", "middle", "z"]


def test_task_discovery_and_taskset_loader_over_bundled_dataset() -> None:
    # Discovery reads the bundled hello-world dataset directory the same way Harbor
    # does: id comes from [task] name, and each task is scored by a reward metric.
    tasks = discover_harbor_tasks(_HELLO_WORLD_DATASET)
    assert {task.id for task in tasks} == {"harbor/hello-world"}
    by_id = {task.id: task for task in tasks}
    task = by_id["harbor/hello-world"]
    # `intent` is the human-facing task name (metadata), NOT the instruction; the instruction the
    # agent acts on comes from instruction.md and lives in inputs["instruction"].
    assert task.intent == "harbor/hello-world"
    assert task.inputs["instruction"] == 'Create a file called hello.txt with "Hello, world!" as the content.'
    assert [metric_type_name(metric) for metric in task.metrics] == ["harbor_reward"]
    # The dataset dir and task dir are stamped on the task so a native runner can
    # recover them without a separate dataset_path argument.
    assert task.metadata["harbor_dataset_path"] == str(_HELLO_WORLD_DATASET)
    assert task.metadata["harbor_task_dir"] == str(_HELLO_WORLD_DATASET / "hello-world")

    # The loader wraps discovery as an AgentEvalTaskset and honors `limit`.
    loader = HarborTasksetLoader(_HELLO_WORLD_DATASET)
    assert loader.name == "harbor"
    taskset = loader.load()
    assert {t.id for t in taskset.tasks} == {"harbor/hello-world"}
    assert taskset.metadata["harbor_dataset_path"] == str(_HELLO_WORLD_DATASET)
    # A limit at/above the task count is a no-op (an empty taskset is invalid).
    assert {t.id for t in loader.load(limit=5).tasks} == {"harbor/hello-world"}


def test_harbor_folder_names_prefer_task_directories() -> None:
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _harbor_folder_names

    tasks = discover_harbor_tasks(_HELLO_WORLD_DATASET)
    assert set(_harbor_folder_names(tasks) or []) == {"hello-world"}
    assert _harbor_folder_names([AgentEvalTask(id="x", intent="x", inputs={}, metrics=[])]) is None


def test_discovery_fails_loudly_on_malformed_task(tmp_path: Path) -> None:
    # A malformed task.toml raises a clear, path-named error rather than crashing
    # cryptically or silently dropping the task (which would shrink eval coverage).
    task_dir = tmp_path / "bad-task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text('[task]\nname = "oops')  # unterminated string
    with pytest.raises(ValueError, match=r"malformed Harbor task config at .*bad-task"):
        discover_harbor_tasks(tmp_path)


def test_runtime_config_defaults_and_runner_requires_a_source() -> None:
    # Config holds only plain fields (importing the module never needs harbor).
    config = HarborRuntimeConfig(jobs_dir=Path("/tmp/jobs"))
    assert config.agent_name == "oracle"
    assert config.reward_key == "reward"

    # A fully under-specified construction is rejected up front.
    with pytest.raises(ValueError):
        HarborAgentTaskRunner()

    # Native mode no longer needs dataset_path at construction; it is recovered from
    # the tasks at run time. Tasks without that metadata (and no override) fail loudly
    # when run (before Harbor is imported, so this needs no harbor install).
    runner = HarborAgentTaskRunner(config=config)
    with pytest.raises(ValueError):
        asyncio.run(runner.run_tasks([AgentEvalTask(id="t", intent="x", inputs={})]))


def _cached_task(dataset_path: Path, task_dir: Path, task_id: str = "t") -> AgentEvalTask:
    """A task whose dataset and on-disk directory the cache stamp can resolve."""
    return AgentEvalTask(
        id=task_id,
        intent="x",
        inputs={"instruction": "x"},
        metrics=[HarborRewardMetric()],
        metadata={"harbor_dataset_path": str(dataset_path), "harbor_task_dir": str(task_dir)},
    )


def _seed_cached_job(tmp_path: Path, *, task_id: str = "t") -> tuple[HarborRuntimeConfig, Path, AgentEvalTask]:
    """A complete job dir plus the config/task that produced it, stamped as a real run would."""
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _cache_stamp, _write_cache_stamp

    dataset_path = tmp_path / "dataset"
    task_dir = dataset_path / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(f'[task]\nname = "{task_id}"\n')

    # jobs_dir deliberately nested under the dataset dir: the digest must exclude it,
    # or it would hash its own growing results tree and never stabilize.
    jobs_dir = dataset_path / "jobs"
    job_dir = jobs_dir / "cached-job"
    job_dir.mkdir(parents=True)
    _write_trial(job_dir, f"{task_id}__aaa", task_id, reward=1.0)

    config = HarborRuntimeConfig(jobs_dir=jobs_dir, job_name="cached-job")
    task = _cached_task(dataset_path, task_dir, task_id)
    _write_cache_stamp(job_dir, _cache_stamp(config, dataset_path, [task]))
    return config, job_dir, task


@pytest.mark.asyncio
async def test_metric_selection_changes_cached_scoring_but_not_execution_cache_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _cache_stamp

    config, _job_dir, task = _seed_cached_job(tmp_path)
    dataset_path = Path(str(task.metadata["harbor_dataset_path"]))
    with_quality = task.model_copy(update={"metrics": [HarborRewardMetric(), ErrorAwareQualityMetric()]})
    run_calls: list[bool] = []
    _spy_on_run_job(monkeypatch, run_calls)

    without_result = await AgentEvaluator().run(tasks=[task], target=HarborAgentTaskRunner(config=config))
    with_result = await AgentEvaluator().run(tasks=[with_quality], target=HarborAgentTaskRunner(config=config))

    assert _cache_stamp(config, dataset_path, [task]) == _cache_stamp(config, dataset_path, [with_quality])
    assert run_calls == []
    assert [trial.id for trial in without_result.trials] == [trial.id for trial in with_result.trials]
    without_descriptors = without_result.tasks[0].model_dump(mode="json")["metrics"]
    with_descriptors = with_result.tasks[0].model_dump(mode="json")["metrics"]
    assert [descriptor["type"] for descriptor in without_descriptors] == ["harbor_reward"]
    assert [descriptor["type"] for descriptor in with_descriptors] == ["harbor_reward", "error_aware_quality"]
    assert with_descriptors[0] == without_descriptors[0]
    assert with_descriptors[1]["outputs"] == [
        {"name": "quality", "description": None, "value_schema": "ContinuousScore", "required": False}
    ]
    assert [score.metric_type for score in without_result.scores] == ["harbor_reward"]
    assert [score.metric_type for score in with_result.scores] == ["harbor_reward", "error_aware_quality"]
    assert [(output.name, output.value) for output in with_result.scores[1].outputs] == [("quality", 1.0)]


@pytest.mark.asyncio
async def test_native_runner_uses_job_dir_as_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A native run whose job_dir covers every requested task AND carries a matching
    # cache stamp is re-adapted, not re-run. Adaptation still imports Harbor lazily
    # to validate the persisted result against Harbor's own schema.
    config, _job_dir, task = _seed_cached_job(tmp_path)
    run_calls: list[bool] = []
    _spy_on_run_job(monkeypatch, run_calls)
    trials = await HarborAgentTaskRunner(config=config).run_tasks([task])

    assert [trial.task_id for trial in trials] == ["t"]
    assert trials[0].metadata["reward"] == 1.0
    assert run_calls == []


@pytest.mark.asyncio
async def test_cached_trials_finalize_sparse_reward_outputs(tmp_path: Path) -> None:
    config, job_dir, task = _seed_cached_job(tmp_path)
    _write_rewards_trial(job_dir, "t__aaa", "t", {"reward": 1.0, "format_ok": 1.0})

    result = await AgentEvaluator().run(tasks=[task], target=HarborAgentTaskRunner(config=config))

    metric = result.tasks[0].metrics[0]
    assert [spec.name for spec in metric.output_spec()] == ["reward", "format_ok"]
    assert [(output.name, output.value) for output in result.scores[0].outputs] == [
        ("reward", 1.0),
        ("format_ok", 1.0),
    ]


def _stamp_for(config: HarborRuntimeConfig, task: AgentEvalTask, job_dir: Path) -> None:
    """Stamp ``job_dir`` as though ``config`` had just produced it."""
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _cache_stamp, _write_cache_stamp

    dataset_path = Path(str(task.metadata["harbor_dataset_path"]))
    _write_cache_stamp(job_dir, _cache_stamp(config, dataset_path, [task]))


def _spy_on_run_job(monkeypatch: pytest.MonkeyPatch, calls: list[bool]) -> None:
    """Replace the native job build so run_tasks is observable without Harbor.

    Records whether the run was attempted and what force_rerun it was built with.
    """
    from nemo_evaluator_sdk.agent_eval.runtimes import harbor_runtime

    def fake(config, _dataset_path, _task_names, *, job_name=None, force_rerun=None):
        async def run_job() -> None:
            calls.append(bool(force_rerun))

        return config.jobs_dir / (job_name or "job"), run_job

    monkeypatch.setattr(harbor_runtime, "_build_native_job", fake)


@pytest.mark.asyncio
async def test_unstamped_job_dir_is_not_trusted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A complete job dir with no stamp predates this check, or was written by plain
    # Harbor. Re-running is the safe reading.
    config, job_dir, task = _seed_cached_job(tmp_path)
    (job_dir / ".nemo-eval-harbor-cache.json").unlink()
    calls: list[bool] = []
    _spy_on_run_job(monkeypatch, calls)

    await HarborAgentTaskRunner(config=config).run_tasks([task])

    assert calls == [True], "an unstamped dir must be re-run, and discarded rather than resumed"


@pytest.mark.asyncio
async def test_changed_inputs_discard_the_job_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A stamp mismatch means the surviving results were produced by different
    # inputs, so they must be deleted rather than resumed onto.
    config, _job_dir, task = _seed_cached_job(tmp_path)
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "wrapper.py").write_text("x = 1\n")
    config = config.model_copy(update={"agent_import_path": "wrapper:Agent", "agent_dir": agent_dir})
    calls: list[bool] = []
    _spy_on_run_job(monkeypatch, calls)

    await HarborAgentTaskRunner(config=config).run_tasks([task])

    assert calls == [True], "changed inputs must discard, not resume"


@pytest.mark.asyncio
async def test_under_covered_job_resumes_with_agent_dir_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression for AALGO-430. This used to discard unconditionally: the scoped
    # import path carried a fresh uuid per run, so Harbor's JobConfig never matched
    # and it raised FileExistsError instead of resuming. Now the path is
    # content-addressed, so an unchanged agent resumes and keeps completed Docker
    # work — the same as the agent_dir-unset case below.
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "wrapper.py").write_text("x = 1\n")
    config, job_dir, task = _seed_cached_job(tmp_path)
    config = config.model_copy(update={"agent_import_path": "wrapper:Agent", "agent_dir": agent_dir, "n_attempts": 2})
    _stamp_for(config, task, job_dir)  # stamp matches; only coverage is short
    calls: list[bool] = []
    _spy_on_run_job(monkeypatch, calls)

    await HarborAgentTaskRunner(config=config).run_tasks([task])

    assert calls == [False], "an unchanged agent must resume, not discard completed trials"


@pytest.mark.asyncio
async def test_under_covered_job_resumes_when_harbor_can(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Inputs unchanged, some attempts missing, agent_dir unset — Harbor's
    # AgentConfig is deterministic, so it resumes per trial. Discarding would throw
    # away completed Docker work for nothing. The agent_dir-set case above now
    # behaves identically (AALGO-430).
    config, job_dir, task = _seed_cached_job(tmp_path)
    config = config.model_copy(update={"n_attempts": 2})
    _stamp_for(config, task, job_dir)  # stamp matches the new config
    calls: list[bool] = []
    _spy_on_run_job(monkeypatch, calls)

    await HarborAgentTaskRunner(config=config).run_tasks([task])

    assert calls == [False], "a resumable miss must not delete completed trials"


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["agent", "agent_kwargs", "task", "option"])
async def test_changed_inputs_invalidate_the_cache(tmp_path: Path, mutation: str) -> None:
    # Each of these changes what a run would produce, so the stamped dir must not be
    # served. Reaching run_job (and failing there) is the observable signal.
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _cache_is_stale, _cache_stamp

    config, job_dir, task = _seed_cached_job(tmp_path)
    dataset_path = Path(str(task.metadata["harbor_dataset_path"]))

    if mutation == "agent":
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "wrapper.py").write_text("x = 1\n")
        config = config.model_copy(
            update={"agent_import_path": "wrapper:Agent", "agent_dir": agent_dir},
        )
    elif mutation == "agent_kwargs":
        config = config.model_copy(update={"agent_kwargs": {"fabric_telemetry": "relay"}})
    elif mutation == "task":
        (dataset_path / "t" / "task.toml").write_text('[task]\nname = "t"\nchanged = true\n')
    else:
        config = config.model_copy(update={"n_attempts": 2})

    assert _cache_is_stale(job_dir, _cache_stamp(config, dataset_path, [task])) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_shape", ["builtin", "installed_import_path", "agent_dir"])
async def test_agent_kwargs_reach_harbor_agent_config_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_shape: str
) -> None:
    """``agent_kwargs`` must land on Harbor's ``AgentConfig.kwargs`` (its ``--ak``) for every agent shape.

    Harbor merges ``AgentConfig.kwargs`` into the agent constructor for built-in and import-path agents
    alike, so a shape that dropped them would silently run the agent with its defaults.
    """
    import harbor.job

    agent_kwargs = {"fabric_adapter_id": "nvidia.fabric.codex", "fabric_harness_settings": {"turns": [1, 2]}}
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    agent_options: dict[str, object] = {"agent_kwargs": agent_kwargs}
    if agent_shape == "installed_import_path":
        agent_options["agent_import_path"] = "mypkg.agent:WrappedAgent"
    elif agent_shape == "agent_dir":
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "harbor_wrapper.py").write_text("x = 1\n", encoding="utf-8")
        agent_options.update(agent_import_path="harbor_wrapper:WrappedAgent", agent_dir=agent_dir)
    config = HarborRuntimeConfig(jobs_dir=jobs_dir, job_name="kwargs-job", **agent_options)

    created: list[Any] = []

    class FakeJob:
        @classmethod
        async def create(cls, job_config: Any) -> "FakeJob":
            created.append(job_config)
            return cls()

        async def run(self) -> None:
            return None

    monkeypatch.setattr(harbor.job, "Job", FakeJob)
    _, run_job = _build_native_job(config, tmp_path / "dataset", None, job_name="kwargs-job")
    await run_job()

    assert len(created) == 1
    (agent_config,) = created[0].agents
    assert agent_config.kwargs == agent_kwargs
    if agent_shape == "builtin":
        assert agent_config.name == "oracle"
    else:
        assert agent_config.import_path is not None and agent_config.import_path.endswith(":WrappedAgent")


@pytest.mark.asyncio
async def test_cosmetic_options_do_not_evict_the_cache(tmp_path: Path) -> None:
    # Presentation and placement knobs change nothing about the results; evicting on
    # them would cost a full Docker re-run for nothing.
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _cache_is_stale, _cache_stamp

    config, job_dir, task = _seed_cached_job(tmp_path)
    dataset_path = Path(str(task.metadata["harbor_dataset_path"]))
    relaxed = config.model_copy(update={"quiet": False, "n_concurrent_trials": 1, "reward_key": "other"})

    assert _cache_is_stale(job_dir, _cache_stamp(relaxed, dataset_path, [task])) is False


@pytest.mark.asyncio
async def test_task_subset_of_a_cached_run_still_hits(tmp_path: Path) -> None:
    # Stamping per task (not one job-wide hash) means evaluating a subset of a
    # previously cached job is still a hit.
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import (
        _cache_is_stale,
        _cache_stamp,
        _write_cache_stamp,
    )

    config, job_dir, task_a = _seed_cached_job(tmp_path, task_id="t")
    dataset_path = Path(str(task_a.metadata["harbor_dataset_path"]))
    task_b_dir = dataset_path / "u"
    task_b_dir.mkdir()
    (task_b_dir / "task.toml").write_text('[task]\nname = "u"\n')
    task_b = _cached_task(dataset_path, task_b_dir, "u")

    _write_cache_stamp(job_dir, _cache_stamp(config, dataset_path, [task_a, task_b]))

    assert _cache_is_stale(job_dir, _cache_stamp(config, dataset_path, [task_a])) is False


def test_unpinned_job_name_writes_no_stamp_and_reads_no_files(tmp_path: Path) -> None:
    # The default timestamped job name can never hit the cache, so the fingerprint
    # must not be computed at all — this is the path plugins/nemo-evaluator takes.
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _resolve_job_dir

    config = HarborRuntimeConfig(jobs_dir=tmp_path / "jobs")
    first = _resolve_job_dir(config)[1]

    assert config.job_name is None
    assert first.parent == tmp_path / "jobs"
    assert not list((tmp_path / "jobs").glob("**/.nemo-eval-harbor-cache.json"))


@pytest.mark.parametrize("job_name", ["..", "../outside", "/tmp/harbor-escape", "."])
def test_resolve_job_dir_rejects_paths_outside_jobs_dir(tmp_path: Path, job_name: str) -> None:
    # force_rerun shutil.rmtree's this path. A job_name that escapes jobs_dir would
    # delete arbitrary directories; Harbor's native evaluator already refuses that.
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _resolve_job_dir

    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    config = HarborRuntimeConfig(jobs_dir=jobs_dir, job_name=job_name)

    with pytest.raises(ValueError, match="strict descendant"):
        _resolve_job_dir(config)


def test_resolve_job_dir_accepts_a_nested_descendant(tmp_path: Path) -> None:
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _resolve_job_dir

    jobs_dir = tmp_path / "jobs"
    config = HarborRuntimeConfig(jobs_dir=jobs_dir, job_name="debug-rerun")

    name, job_dir = _resolve_job_dir(config)

    assert name == "debug-rerun"
    assert job_dir == (jobs_dir / "debug-rerun").resolve()
    assert job_dir.is_relative_to(jobs_dir.resolve())
    assert job_dir != jobs_dir.resolve()


def test_build_native_job_rejects_escaping_job_name_before_rmtree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # _build_native_job concatenates jobs_dir/job_name independently of _resolve_job_dir.
    # The containment check must live here too, or force_rerun still rmtree's the escape.
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("do not delete", encoding="utf-8")

    rmtree_calls: list[Path] = []
    monkeypatch.setattr(
        "nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime.shutil.rmtree",
        lambda path, **kwargs: rmtree_calls.append(Path(path)),
    )

    config = HarborRuntimeConfig(jobs_dir=jobs_dir, job_name="../outside", force_rerun=True)
    with pytest.raises(ValueError, match="strict descendant"):
        _build_native_job(config, tmp_path / "dataset", None, job_name="../outside", force_rerun=True)

    assert rmtree_calls == []
    assert marker.read_text(encoding="utf-8") == "do not delete"


def test_cache_stamp_survives_harbors_stray_directory_sweep(tmp_path: Path) -> None:
    # Harbor rmtree's any *directory* in a job dir lacking result.json. The stamp must
    # therefore be a file, or it would be silently deleted on the next Harbor run.
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import CACHE_STAMP_FILENAME

    _config, job_dir, _task = _seed_cached_job(tmp_path)
    stamp = job_dir / CACHE_STAMP_FILENAME

    assert stamp.is_file()
    assert not stamp.is_dir()


def test_cache_stamp_handles_a_missing_agent_dir(tmp_path: Path) -> None:
    # agent_dir is None for every built-in-agent caller (including nemo-evaluator).
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _cache_stamp

    config, _job_dir, task = _seed_cached_job(tmp_path)
    stamp = _cache_stamp(config, Path(str(task.metadata["harbor_dataset_path"])), [task])

    assert config.agent_dir is None
    assert stamp["agent"] == "<none>"


def test_unresolvable_task_dir_is_always_stale(tmp_path: Path) -> None:
    # A task we cannot locate on disk must never be silently omitted from the
    # fingerprint — that would be a stale-cache hole.
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _cache_is_stale, _cache_stamp

    config, job_dir, _task = _seed_cached_job(tmp_path)
    orphan = AgentEvalTask(id="ghost", intent="x", inputs={"instruction": "x"}, metrics=[HarborRewardMetric()])
    stamp = _cache_stamp(config, tmp_path / "nonexistent-dataset", [orphan])

    assert stamp["tasks"]["ghost"] == "<unresolved>"
    assert _cache_is_stale(job_dir, stamp) is True


def test_multiple_attempts_map_to_one_trial_each(tmp_path: Path) -> None:
    # n_attempts > 1: Harbor writes one result.json per attempt, and each becomes a
    # distinct trial for the same task id (so the summary can aggregate over attempts).
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    _write_trial(job_dir, "t__a1B2", "t", reward=1.0)
    _write_trial(job_dir, "t__Z9y8", "t", reward=0.0)
    tasks = [AgentEvalTask(id="t", intent="x", inputs={"instruction": "p"}, metrics=[HarborRewardMetric()])]

    trials = build_trials_from_job_dir(job_dir, tasks)
    assert [trial.task_id for trial in trials] == ["t", "t"]
    assert [trial.id for trial in trials] == ["t__Z9y8", "t__a1B2"]
    assert all("harbor_attempt" not in trial.metadata for trial in trials)
    assert sorted(trial.metadata["reward"] for trial in trials) == [0.0, 1.0]


def test_cache_counts_harbor_valid_physical_attempts(tmp_path: Path) -> None:
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _all_tasks_cached

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    tasks = [AgentEvalTask(id="t", intent="x", inputs={"instruction": "p"}, metrics=[HarborRewardMetric()])]

    # One completed attempt: enough for n_attempts=1, not for n_attempts=2.
    _write_trial(job_dir, "t__aaa", "t", reward=1.0)
    assert _all_tasks_cached(job_dir, tasks, n_attempts=1) is True
    assert _all_tasks_cached(job_dir, tasks, n_attempts=2) is False

    # Harbor considers a valid errored result complete, so it is the second attempt.
    _write_trial(job_dir, "t__bbb", "t", reward=0.0, exception="NonZeroAgentExitCodeError")
    assert _all_tasks_cached(job_dir, tasks, n_attempts=2) is True

    # Extra valid attempts do not invalidate already-satisfied coverage.
    _write_trial(job_dir, "t__ccc", "t", reward=1.0)
    assert _all_tasks_cached(job_dir, tasks, n_attempts=2) is True


def test_scoped_agent_import_makes_wrapper_importable_then_cleans_up(tmp_path: Path) -> None:
    # import_path without agent_dir is allowed (Harbor imports an installed module directly);
    # only a dangling agent_dir (no import_path) is rejected.
    HarborRuntimeConfig(jobs_dir=tmp_path, agent_import_path="mypkg.agent:WrappedAgent")
    with pytest.raises(ValidationError):
        HarborRuntimeConfig(jobs_dir=tmp_path, agent_dir=tmp_path)

    # Inside the scope the user's harbor_wrapper.py resolves under a synthetic package,
    # and the yielded path preserves the :attribute suffix Harbor imports.
    (tmp_path / "harbor_wrapper.py").write_text("class WrappedAgent:\n    value = 42\n")
    with scoped_harbor_agent_import(tmp_path, "harbor_wrapper:WrappedAgent") as scoped_import:
        module_name, _, attribute = scoped_import.partition(":")
        assert attribute == "WrappedAgent"
        module = importlib.import_module(module_name)
        assert module.WrappedAgent.value == 42
        package = module_name.rsplit(".", 1)[0]
        assert package in sys.modules

    # On exit the injected module and its synthetic package are gone from sys.modules.
    assert module_name not in sys.modules
    assert package not in sys.modules


def test_digest_ignores_an_exclusion_that_contains_the_whole_tree(tmp_path: Path) -> None:
    # jobs_dir sitting *above* the agent/task dir is a legitimate layout. Applying the
    # exclusion there would match every entry and yield an empty digest, silently
    # disabling invalidation for the entire directory.
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _digest_directory

    work = tmp_path / "work"
    (work / "agent").mkdir(parents=True)
    (work / "agent" / "a.py").write_text("v1")

    before = _digest_directory(work / "agent", exclude=frozenset({work}))
    assert before != hashlib.sha256().hexdigest(), "an ancestor exclusion must not empty the digest"

    (work / "agent" / "a.py").write_text("v2-DIFFERENT")
    assert _digest_directory(work / "agent", exclude=frozenset({work})) != before


def test_digest_still_ignores_a_jobs_dir_nested_inside_the_tree(tmp_path: Path) -> None:
    # The case the exclusion actually exists for: results written under the hashed
    # tree must not make the fingerprint move on every run.
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _digest_directory

    agent = tmp_path / "agent"
    (agent / "jobs").mkdir(parents=True)
    (agent / "a.py").write_text("src")
    (agent / "jobs" / "result.json").write_text("{}")

    before = _digest_directory(agent, exclude=frozenset({agent / "jobs"}))
    (agent / "jobs" / "result.json").write_text('{"more": "output"}')
    assert _digest_directory(agent, exclude=frozenset({agent / "jobs"})) == before


def test_task_dir_outside_the_active_dataset_is_rediscovered(tmp_path: Path) -> None:
    # `dataset_path` can be overridden on the runner, so a task stamped during
    # discovery under dataset A must not be fingerprinted when Harbor runs dataset B.
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _task_dirs_for

    dataset_a, dataset_b = tmp_path / "dsA", tmp_path / "dsB"
    for dataset in (dataset_a, dataset_b):
        (dataset / "t").mkdir(parents=True)
        (dataset / "t" / "task.toml").write_text('[task]\nname = "t"\n')

    task = AgentEvalTask(
        id="t",
        intent="x",
        inputs={"instruction": "x"},
        metadata={"harbor_dataset_path": str(dataset_a), "harbor_task_dir": str(dataset_a / "t")},
    )

    resolved = _task_dirs_for(dataset_b, [task])["t"]
    assert resolved is not None
    assert resolved.resolve().is_relative_to(dataset_b.resolve()), "must fingerprint the dataset Harbor runs"


def test_vanished_task_dir_is_rediscovered(tmp_path: Path) -> None:
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _task_dirs_for

    dataset = tmp_path / "ds"
    (dataset / "t").mkdir(parents=True)
    (dataset / "t" / "task.toml").write_text('[task]\nname = "t"\n')
    task = AgentEvalTask(
        id="t",
        intent="x",
        inputs={"instruction": "x"},
        metadata={"harbor_task_dir": str(tmp_path / "gone" / "t")},
    )

    assert _task_dirs_for(dataset, [task])["t"] == dataset / "t"


def test_symlink_loop_degrades_the_stamp_instead_of_killing_the_run(tmp_path: Path) -> None:
    """A loop under a task dir must not take down a run over a best-effort fingerprint.

    Deliberately a *loop*, not a dangling or vanished link. On CPython 3.12 — the
    floor this package targets — `Path.resolve()` translates `ELOOP` into
    ``RuntimeError``, which is **not** an ``OSError``, so catching only ``OSError``
    leaves this crashing. It also raises deterministically rather than as a race, so
    the failure is reproducible rather than occasional.

    What this pins is :func:`_safe_resolve`'s exception set. The loop is reached
    through :func:`_digest_directory`'s walk, which already routes every path through
    ``_safe_resolve``. It does **not** exercise ``_cache_stamp``'s own resolve calls:
    ``_task_dirs_for`` filters candidates with ``is_dir()``, which returns ``False``
    for a loop, so a looping path never reaches them. Those calls use
    ``_safe_resolve`` for consistency, not because a live crash was demonstrated.
    """
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import HarborRuntimeConfig, _cache_stamp, _safe_resolve

    dataset = tmp_path / "ds"
    task_dir = dataset / "t"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text('[task]\nname = "t"\n')
    loop = task_dir / "loop"
    loop.symlink_to(loop)

    # Pin the premise: if this stops raising, the guard below is no longer load-bearing.
    # From 3.13 `resolve()` returns the path for a loop instead of raising.
    if sys.version_info < (3, 13):
        with pytest.raises(RuntimeError):
            loop.resolve()
    assert _safe_resolve(loop) == loop.absolute(), "_safe_resolve must swallow the loop, not just OSError"

    task = AgentEvalTask(
        id="t",
        intent="x",
        inputs={"instruction": "x"},
        metadata={"harbor_task_dir": str(task_dir)},
    )
    config = HarborRuntimeConfig(jobs_dir=tmp_path / "jobs", job_name="pinned")

    stamp = _cache_stamp(config, dataset, [task])

    assert set(stamp["tasks"]) == {"t"}, "the task must still be fingerprinted, not dropped"


@pytest.mark.asyncio
async def test_inputs_changing_mid_run_leaves_the_job_unstamped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # If the candidate is edited while Harbor is running, the results came from the
    # OLD sources. Stamping the new fingerprint onto them would let a later run serve
    # them as if they matched — so the job dir is deliberately left unstamped.
    from nemo_evaluator_sdk.agent_eval.runtimes import harbor_runtime

    config, job_dir, task = _seed_cached_job(tmp_path)
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "wrapper.py").write_text("v1\n")
    config = config.model_copy(update={"agent_import_path": "wrapper:Agent", "agent_dir": agent_dir})
    (job_dir / harbor_runtime.CACHE_STAMP_FILENAME).unlink()

    def fake(cfg, _dataset_path, _task_names, *, job_name=None, force_rerun=None):
        async def run_job() -> None:
            (agent_dir / "wrapper.py").write_text("v2-EDITED-MID-RUN\n")

        return cfg.jobs_dir / (job_name or "job"), run_job

    monkeypatch.setattr(harbor_runtime, "_build_native_job", fake)

    await HarborAgentTaskRunner(config=config).run_tasks([task])

    assert not (job_dir / harbor_runtime.CACHE_STAMP_FILENAME).exists(), (
        "results produced from pre-edit sources must not be stamped with the post-edit fingerprint"
    )


def test_digest_is_injective_over_separator_bearing_contents(tmp_path: Path) -> None:
    """Distinct trees must never share a digest, even when contents embed the framing.

    A collision here fails CLOSED - the digest matches, so stale results are served.
    The historical bug was concatenating ``name \\0 content \\0`` with no length
    framing: because file *contents* may contain NUL, ``{a: b"", b: b"Z"}`` and
    ``{a: b"\\0b\\0Z"}`` produced an identical byte stream.

    Rather than pin one hand-built pair to one encoding, this asserts the property:
    every tree below is structurally different, so every digest must differ. The
    contents are chosen to embed the separators an unframed encoding would rely on.
    """
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _digest_directory

    trees: dict[str, dict[str, bytes]] = {
        "two_files_empty_then_z": {"a": b"", "b": b"Z"},
        "one_file_absorbing_nul": {"a": b"\0b\0Z"},
        "one_file_absorbing_nul_and_mode": {"a": b"\0b\0-\0Z"},
        "three_files": {"a": b"", "b": b"", "c": b"Z"},
        "two_files_swapped": {"a": b"Z", "b": b""},
        "one_file_named_b": {"b": b"Z"},
    }

    digests: dict[str, str] = {}
    for name, files in trees.items():
        root = tmp_path / name
        root.mkdir()
        for filename, content in files.items():
            (root / filename).write_bytes(content)
        digests[name] = _digest_directory(root)

    collisions = {
        (left, right) for left in digests for right in digests if left < right and digests[left] == digests[right]
    }
    assert not collisions, f"distinct trees produced identical digests: {sorted(collisions)}"


def test_digest_tracks_the_execute_bit(tmp_path: Path) -> None:
    # Harbor discovers and runs tests/test.sh; flipping +x changes what happens
    # without changing a single byte of content.
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _digest_directory

    task = tmp_path / "task"
    task.mkdir()
    script = task / "test.sh"
    script.write_text("#!/bin/sh\necho hi\n")

    script.chmod(0o644)
    non_executable = _digest_directory(task)
    script.chmod(0o755)
    assert _digest_directory(task) != non_executable, "+x must invalidate"


def test_digest_ignores_read_write_permission_noise(tmp_path: Path) -> None:
    # Only the execute bit is tracked, mirroring git: umask differences between two
    # checkouts of the same sources must not evict a usable cache.
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _digest_directory

    task = tmp_path / "task"
    task.mkdir()
    source = task / "a.py"
    source.write_text("x = 1\n")

    source.chmod(0o644)
    before = _digest_directory(task)
    source.chmod(0o600)
    assert _digest_directory(task) == before


def test_digest_covers_vendored_dependencies_but_not_the_environment(tmp_path: Path) -> None:
    # node_modules ships with the agent and changes what it does, so it counts.
    # .venv is environment the Harbor wrapper never uploads, so it does not.
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _digest_directory

    agent = tmp_path / "agent"
    (agent / "node_modules" / "lib").mkdir(parents=True)
    (agent / ".venv").mkdir()
    (agent / "main.js").write_text("x")
    (agent / "node_modules" / "lib" / "index.js").write_text("v1")
    (agent / ".venv" / "marker").write_text("1")

    before = _digest_directory(agent)
    (agent / "node_modules" / "lib" / "index.js").write_text("v2-DIFFERENT")
    assert _digest_directory(agent) != before, "a vendored dependency change must invalidate"

    after_dep = _digest_directory(agent)
    (agent / ".venv" / "marker").write_text("2")
    assert _digest_directory(agent) == after_dep, ".venv churn must not evict the cache"


def test_digest_survives_a_dangling_symlink(tmp_path: Path) -> None:
    # A link whose *target* is missing: is_dir()/is_file() are both False, so it is
    # recorded as a marker. (readlink still succeeds here - see the test below for
    # the case where readlink itself fails.)
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _digest_directory

    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "real.py").write_text("x")
    (agent / "dangling").symlink_to(tmp_path / "does-not-exist")

    assert _digest_directory(agent)  # no raise


def test_digest_distinguishes_a_file_from_a_directory_of_the_same_name(tmp_path: Path) -> None:
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _digest_directory

    as_file = tmp_path / "as_file"
    as_file.mkdir()
    (as_file / "thing").write_text("")

    as_dir = tmp_path / "as_dir"
    as_dir.mkdir()
    (as_dir / "thing").mkdir()

    assert _digest_directory(as_file) != _digest_directory(as_dir)


def test_digest_survives_readlink_failing_mid_walk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The link itself disappearing between is_symlink() and readlink is a real race
    # against any process cleaning up the tree. It must degrade to a marker rather
    # than raise out of run_tasks and kill an otherwise-good evaluation.
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _digest_directory

    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "real.py").write_text("x")
    (agent / "link").symlink_to(agent / "real.py")

    def exploding_readlink(*_args: object, **_kwargs: object) -> str:
        raise OSError(2, "No such file or directory")

    monkeypatch.setattr(os, "readlink", exploding_readlink)

    assert _digest_directory(agent)  # must not raise


def _scoped_path(agent_dir: Path) -> str:
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import scoped_harbor_agent_import

    with scoped_harbor_agent_import(agent_dir, "wrapper:Agent") as scoped:
        return scoped


def test_scoped_import_path_is_stable_for_unchanged_contents(tmp_path: Path) -> None:
    # The import path lands in Harbor's JobConfig, which Harbor compares field-by-field
    # when deciding whether a job dir may be resumed. A per-run random suffix made that
    # comparison fail every time, so Harbor could never resume (AALGO-430).
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "wrapper.py").write_text("x = 1\n")

    assert _scoped_path(agent_dir) == _scoped_path(agent_dir)


def test_scoped_import_path_changes_when_the_agent_changes(tmp_path: Path) -> None:
    # The flip side: an edited agent must NOT resume a job dir built from the old one.
    # Harbor's own config check now catches that without help from the cache stamp.
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "wrapper.py").write_text("x = 1\n")
    before = _scoped_path(agent_dir)

    (agent_dir / "wrapper.py").write_text("x = 2\n")

    assert _scoped_path(agent_dir) != before


def test_distinct_agents_do_not_share_a_scoped_package(tmp_path: Path) -> None:
    # Content-addressing must not collapse different agents onto one sys.modules entry.
    first = tmp_path / "a"
    second = tmp_path / "b"
    for path, body in ((first, "x = 1\n"), (second, "x = 2\n")):
        path.mkdir()
        (path / "wrapper.py").write_text(body)

    assert _scoped_path(first) != _scoped_path(second)


def test_overlapping_scopes_on_one_agent_survive_the_inner_exit(tmp_path: Path) -> None:
    # Identical contents now share a package name, so teardown is refcounted: the inner
    # scope exiting must not strip sys.modules out from under the outer one.
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import scoped_harbor_agent_import

    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "wrapper.py").write_text("VALUE = 7\n")

    with scoped_harbor_agent_import(agent_dir, "wrapper:Agent") as outer:
        package = outer.split(":")[0].rsplit(".", 1)[0]
        with scoped_harbor_agent_import(agent_dir, "wrapper:Agent"):
            pass
        # Inner scope closed; the outer one is still open and must still resolve.
        assert package in sys.modules
        assert importlib.import_module(f"{package}.wrapper").VALUE == 7

    assert package not in sys.modules, "the last scope to exit must clean up"


def test_same_named_identical_agents_do_not_repoint_an_open_scope(tmp_path: Path) -> None:
    # Content-addressing means two directories with the same basename and identical
    # contents share a package name. The second install must not swap `__path__` out
    # from under the first, still-open scope.
    #
    # Deliberately NOT fixed by hashing the resolved path into the package name: that
    # would make the name location-dependent, so the same agent evaluated from a
    # different path would produce a different JobConfig and Harbor would refuse to
    # resume — reintroducing AALGO-430. The trees are byte-identical here, so keeping
    # the first path is correct.
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _AGENT_PACKAGE_REFCOUNTS

    first = tmp_path / "one" / "agent"
    second = tmp_path / "two" / "agent"
    for agent_dir in (first, second):
        agent_dir.mkdir(parents=True)
        (agent_dir / "wrapper.py").write_text("VALUE = 1\n")

    with scoped_harbor_agent_import(first, "wrapper:Agent") as outer:
        package = outer.rsplit(".", 1)[0]
        assert sys.modules[package].__path__ == [str(first)]
        with scoped_harbor_agent_import(second, "wrapper:Agent") as inner:
            assert inner == outer, "identical contents and basename must share one package"
            assert sys.modules[package].__path__ == [str(first)], (
                "the second install must not repoint a scope that is still open"
            )
        assert sys.modules[package].__path__ == [str(first)], "the inner exit must not tear down the outer scope"

    assert package not in sys.modules
    assert _AGENT_PACKAGE_REFCOUNTS == {}


def test_scoped_import_teardown_is_complete_after_overlap(tmp_path: Path) -> None:
    # Refcounting must not leak: no stray refcount entries or sys.modules residue.
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import (
        _AGENT_IMPORT_ROOT,
        _AGENT_PACKAGE_REFCOUNTS,
        scoped_harbor_agent_import,
    )

    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "wrapper.py").write_text("x = 1\n")

    with scoped_harbor_agent_import(agent_dir, "wrapper:Agent"):
        with scoped_harbor_agent_import(agent_dir, "wrapper:Agent"):
            pass

    assert _AGENT_PACKAGE_REFCOUNTS == {}
    assert not [name for name in sys.modules if name.startswith(f"{_AGENT_IMPORT_ROOT}.")]


def test_scoped_import_path_ignores_a_jobs_dir_nested_under_the_agent(tmp_path: Path) -> None:
    # jobs_dir is caller-chosen and may sit *under* agent_dir. Without the same
    # exclusion the cache stamp applies, Harbor's own results would feed the package
    # name, so the import path would move every run and the resume this whole change
    # exists to enable could never happen.
    agent_dir = tmp_path / "agent"
    jobs_dir = agent_dir / "results"
    jobs_dir.mkdir(parents=True)
    (agent_dir / "wrapper.py").write_text("x = 1\n")
    excluded = frozenset({jobs_dir.resolve()})

    with scoped_harbor_agent_import(agent_dir, "wrapper:Agent", exclude=excluded) as before:
        pass
    (jobs_dir / "trial-a").mkdir()
    (jobs_dir / "trial-a" / "result.json").write_text("{}")
    with scoped_harbor_agent_import(agent_dir, "wrapper:Agent", exclude=excluded) as after:
        pass

    assert before == after, "accumulating results must not move the agent's import path"


def test_failed_scoped_import_install_does_not_wedge_the_refcount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The refcount is taken only once the sys.modules injection has succeeded. Taking
    # it first would strand the count at 1 when the injection raises — no scope ever
    # opened, so nothing decrements it, and the package could never be torn down again.
    from nemo_evaluator_sdk.agent_eval.runtimes import harbor_runtime
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _AGENT_IMPORT_ROOT, _AGENT_PACKAGE_REFCOUNTS

    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "wrapper.py").write_text("x = 1\n")

    def explode(_name: str) -> ModuleType:
        raise RuntimeError("synthetic package could not be built")

    with monkeypatch.context() as patched:
        patched.setattr(harbor_runtime, "ModuleType", explode)
        with pytest.raises(RuntimeError, match="synthetic package"):
            with scoped_harbor_agent_import(agent_dir, "wrapper:Agent"):
                pass

    assert _AGENT_PACKAGE_REFCOUNTS == {}, "a failed install must not leave a refcount behind"
    # And a later scope must still install and then fully tear down.
    with scoped_harbor_agent_import(agent_dir, "wrapper:Agent"):
        pass
    assert _AGENT_PACKAGE_REFCOUNTS == {}
    assert not [name for name in sys.modules if name.startswith(f"{_AGENT_IMPORT_ROOT}.")]


class _DriftConfig(BaseModel):
    """Stands in for Harbor's JobConfig: a field it ignores, one it compares, one defaulted."""

    job_name: str = "job"
    n_concurrent_trials: int = 4
    quiet: bool = True


def _stub_harbor(
    monkeypatch: pytest.MonkeyPatch,
    job_create: Callable[[object], Awaitable[object]],
    verifier_calls: list[dict[str, object]] | None = None,
) -> None:
    """Install a minimal fake ``harbor`` package so ``run_job`` can execute.

    Only the names ``_build_native_job``'s ``run_job`` imports are provided.
    ``job_create`` becomes ``Job.create``; every config class is a permissive stub,
    since what is under test is the control flow around Harbor, not the payload.
    ``verifier_calls`` records ``VerifierConfig`` kwargs, which no caller can read back
    off the permissive ``JobConfig`` stub.
    """

    def _module(name: str, **attrs: object) -> None:
        module = ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)

    def _anything(*_args: object, **_kwargs: object) -> object:
        return object()

    def _verifier_config(**kwargs: object) -> object:
        if verifier_calls is not None:
            verifier_calls.append(kwargs)
        return object()

    class _Job:
        create = staticmethod(job_create)

    _module("harbor")
    # JobConfig is a real model, not `_anything`: the resume-refusal path reads and
    # re-validates the persisted one to report what differed. Pydantic ignores the
    # kwargs _build_native_job passes that this stand-in doesn't declare.
    _module("harbor.job", DatasetConfig=_anything, Job=_Job, JobConfig=_DriftConfig)
    _module("harbor.models")
    _module("harbor.models.job")
    _module("harbor.models.job.config", RetryConfig=_anything)
    _module("harbor.models.trial")
    _module(
        "harbor.models.trial.config",
        AgentConfig=_anything,
        ArtifactConfig=_anything,
        VerifierConfig=_verifier_config,
    )


class _FakeJob:
    async def run(self) -> None:
        return None


@pytest.mark.asyncio
async def test_harbor_refusing_to_resume_discards_and_reruns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Harbor compares its whole persisted JobConfig (and lock.json) before resuming and
    # raises FileExistsError on any mismatch. The SDK cache stamp is looser on purpose:
    # `quiet`, `n_concurrent_trials` and the `task_names` filter change the JobConfig
    # without changing the results, so a job dir that passes the stamp — and is
    # therefore handed over with force_rerun=False — can still be rejected by Harbor.
    # That must degrade to a clean re-run rather than crash the evaluation.
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "pinned"
    (job_dir / "old-trial").mkdir(parents=True)
    # The dir was produced on a 10-core box; this run defaults to 4. Nothing about the
    # results changed, so the SDK stamp would still call it fresh — Harbor won't.
    (job_dir / "config.json").write_text(
        _DriftConfig(job_name="pinned", n_concurrent_trials=10).model_dump_json(exclude_defaults=True),
        encoding="utf-8",
    )
    dir_existed_at_attempt: list[bool] = []

    async def create(_config: object) -> _FakeJob:
        dir_existed_at_attempt.append(job_dir.exists())
        if len(dir_existed_at_attempt) == 1:
            raise FileExistsError(
                f"Job directory {job_dir} already exists and cannot be resumed with a different config."
            )
        return _FakeJob()

    _stub_harbor(monkeypatch, create)
    config = HarborRuntimeConfig(jobs_dir=jobs_dir, job_name="pinned")
    _built, run_job = _build_native_job(config, tmp_path / "dataset", None, job_name="pinned", force_rerun=False)

    with caplog.at_level(logging.WARNING):
        await run_job()

    assert dir_existed_at_attempt == [True, False], "the refused dir must be discarded before the retry"
    assert not (job_dir / "old-trial").exists(), "the stale trial must be gone, not resumed onto"
    assert "refused to resume" in caplog.text, "silently deleting completed trials must be visible"
    assert "n_concurrent_trials: 10 -> 4" in caplog.text, "the warning must name what forced the discard"


@pytest.mark.asyncio
async def test_trace_dir_is_published_to_the_verifier_as_trace_dir_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Drop this and every trace-reading metric reads an empty directory instead of raising.
    verifier_calls: list[dict[str, object]] = []

    async def create(_config: object) -> _FakeJob:
        return _FakeJob()

    _stub_harbor(monkeypatch, create, verifier_calls)
    config = HarborRuntimeConfig(jobs_dir=tmp_path / "jobs", job_name="pinned", trace_dir="/app/traces")
    _built, run_job = _build_native_job(config, tmp_path / "dataset", None, job_name="pinned", force_rerun=False)

    await run_job()

    assert verifier_calls == [{"env": {"TRACE_DIR": "/app/traces"}}]


@pytest.mark.asyncio
async def test_file_exists_error_without_a_job_dir_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The retry is scoped to Harbor's resume refusal. A FileExistsError raised with no
    # job dir to discard is something else entirely and must not be swallowed, nor
    # turned into a second Docker run.
    attempts: list[int] = []

    async def create(_config: object) -> _FakeJob:
        attempts.append(1)
        raise FileExistsError("something unrelated")

    _stub_harbor(monkeypatch, create)
    config = HarborRuntimeConfig(jobs_dir=tmp_path / "jobs", job_name="pinned")
    _built, run_job = _build_native_job(config, tmp_path / "dataset", None, job_name="pinned", force_rerun=False)

    with pytest.raises(FileExistsError, match="something unrelated"):
        await run_job()

    assert attempts == [1], "an unrelated FileExistsError must not be retried"


@pytest.mark.asyncio
async def test_unrelated_file_exists_error_mid_run_leaves_the_job_dir_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The dangerous shape: a job dir that *does* exist, and a FileExistsError raised
    # from inside the run rather than by Harbor's resume check — a trial, a hook, an
    # environment build. Treating that as drift would delete completed work and re-run
    # for an error that has nothing to do with the config.
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "pinned"
    (job_dir / "finished-trial").mkdir(parents=True)
    attempts: list[int] = []

    class _ExplodingJob:
        async def run(self) -> None:
            attempts.append(1)
            raise FileExistsError(17, "File exists", str(tmp_path / "scratch" / "artifact.tar"))

    async def create(_config: object) -> _ExplodingJob:
        return _ExplodingJob()

    _stub_harbor(monkeypatch, create)
    config = HarborRuntimeConfig(jobs_dir=jobs_dir, job_name="pinned")
    _built, run_job = _build_native_job(config, tmp_path / "dataset", None, job_name="pinned", force_rerun=False)

    with pytest.raises(FileExistsError):
        await run_job()

    assert attempts == [1], "an unrelated failure must not be retried"
    assert (job_dir / "finished-trial").exists(), "completed work must survive an error that is not resume drift"


@pytest.mark.parametrize(
    ("message", "errno", "expected"),
    [
        ("Job directory {job_dir} already exists and cannot be resumed with a different config.", None, True),
        ("Job directory {job_dir} already has a lock.json that does not match the resolved job lock.", None, True),
        # Same words, but an OS-level EEXIST: errno is set, so it is not Harbor's refusal.
        ("Job directory {job_dir} already exists and cannot be resumed with a different config.", 17, False),
        # A refusal naming a *different* job dir is not ours to act on.
        ("Job directory /somewhere/else already exists and cannot be resumed with a different config.", None, False),
        ("[Errno 17] File exists: '{job_dir}/trial/artifact.tar'", None, False),
    ],
)
def test_only_harbors_resume_refusal_authorises_deleting_the_job_dir(
    tmp_path: Path, message: str, errno: int | None, expected: bool
) -> None:
    # Deleting a job dir is the one irreversible thing this runtime does, so the
    # predicate that authorises it is pinned directly. Anything unrecognised must
    # answer False and let the error propagate — the safe direction if Harbor rewords.
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _is_harbor_resume_refusal

    job_dir = tmp_path / "jobs" / "pinned"
    rendered = message.format(job_dir=job_dir)
    exc = FileExistsError(rendered) if errno is None else FileExistsError(errno, "File exists", rendered)

    assert _is_harbor_resume_refusal(exc, job_dir) is expected


def test_job_config_drift_names_the_field_that_forced_the_discard(tmp_path: Path) -> None:
    # Harbor says only *that* a config differs, so the discard looks arbitrary in the
    # log. This pins three things at once: the differing field is named, a field Harbor
    # ignores is not, and a field left at its default is not — the last only holds
    # because the persisted JSON (written with exclude_defaults=True) is re-validated
    # rather than compared raw, which would see a missing key as a difference.
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _describe_job_config_drift

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    stored = _DriftConfig(job_name="pinned", n_concurrent_trials=10)
    (job_dir / "config.json").write_text(stored.model_dump_json(exclude_defaults=True), encoding="utf-8")

    drift = _describe_job_config_drift(job_dir, _DriftConfig(job_name="renamed", n_concurrent_trials=4))

    assert drift == "n_concurrent_trials: 10 -> 4"


def test_job_config_drift_is_silent_when_it_cannot_tell(tmp_path: Path) -> None:
    # No config.json is the lock.json-refusal case: there is no JobConfig difference to
    # report. Diagnostics must degrade to silence, never to a raised exception that
    # would mask the FileExistsError being explained.
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _describe_job_config_drift

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    assert _describe_job_config_drift(job_dir, _DriftConfig()) == ""
    (job_dir / "config.json").write_text("{not json", encoding="utf-8")
    assert _describe_job_config_drift(job_dir, _DriftConfig()) == ""


# Harbor JobConfig field -> the HarborRuntimeConfig fields that feed it. Each of these
# must sit inside the cache fingerprint: if one drops out, the stamp could call a job
# dir reusable that Harbor will then reject. `agents` also carries the agent contents,
# which the stamp digests separately (that is why `agent_dir` itself is irrelevant).
_STAMP_COVERED_HARBOR_FIELDS = {
    "n_attempts": {"n_attempts"},
    "artifacts": {"artifacts", "trace_dir"},
    # Carries TRACE_DIR, the container trace path verifiers read.
    "verifier": {"trace_dir"},
    "retry": {"max_retries"},
    "agents": {"agent_name", "agent_import_path", "agent_model_name"},
    "timeout_multiplier": {"timeout_multiplier"},
    "agent_timeout_multiplier": {"agent_timeout_multiplier"},
    "verifier_timeout_multiplier": {"verifier_timeout_multiplier"},
    "agent_setup_timeout_multiplier": {"agent_setup_timeout_multiplier"},
    "environment_build_timeout_multiplier": {"environment_build_timeout_multiplier"},
}
# Left at Harbor's defaults by _build_native_job, so two SDK-built configs can never
# disagree on them. (A dir written by the Harbor CLI could, but it carries no SDK cache
# stamp, so it is stale and gets discarded before Harbor ever sees it.)
_SDK_NEVER_SETS = {"install_only", "environment", "metrics", "tasks", "extra_instruction_paths"}
# Compared by Harbor, deliberately *not* keyed by the SDK stamp. Harbor asks "can I
# resume this directory?"; the stamp asks "did these inputs produce these results?".
# Where the answers diverge, _build_native_job absorbs Harbor's refusal.
_KNOWINGLY_LOOSER = {
    "jobs_dir": "implied by having found the job dir at all",
    "n_concurrent_trials": "scheduling only; keying it would discard a cached run on a box with a different core count",
    "quiet": "display only; changes nothing about the results",
    "datasets": "`path` is covered by the per-task digests; the `task_names` filter is left unkeyed so a subset of a "
    "cached job still hits (see _stamp_coverage)",
}


def test_harbor_still_words_its_resume_refusals_the_way_we_match_them() -> None:
    # The predicate that authorises deleting a job dir keys off Harbor's message text.
    # If Harbor rewords, the predicate stops matching and the refusal propagates as a
    # crash — the safe direction, but a silent loss of the graceful re-run. Catch that
    # at upgrade time here instead of in someone's failed experiment.
    pytest.importorskip("harbor.job", reason="harbor needs python >= 3.12")
    import inspect

    from harbor.job import Job
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _HARBOR_RESUME_REFUSALS

    source = inspect.getsource(Job)
    for phrase in _HARBOR_RESUME_REFUSALS:
        assert phrase in source, (
            f"Harbor no longer raises its resume refusal with {phrase!r}. Re-read Job._maybe_init_existing_job "
            "and Job._write_job_lock, then update _HARBOR_RESUME_REFUSALS. Keep each phrase inside a single "
            "source string literal — one spanning an implicit concatenation will not be found here."
        )


def test_harbor_job_config_equality_still_behaves_as_the_retry_assumes() -> None:
    # The FileExistsError retry exists because Harbor compares its whole JobConfig and
    # ignores only identity/logging fields. Pin that behaviourally, so a Harbor upgrade
    # that changes the rule surfaces here rather than as a mystery re-run in production.
    job_config = pytest.importorskip("harbor.models.job.config", reason="harbor needs python >= 3.12")

    baseline = job_config.JobConfig(job_name="a")
    assert baseline == job_config.JobConfig(job_name="b"), "job_name must stay outside Harbor's comparison"
    assert baseline != job_config.JobConfig(job_name="a", n_concurrent_trials=99), (
        "n_concurrent_trials must stay inside it — that is the case the retry absorbs"
    )


def test_every_harbor_job_config_field_is_classified_against_the_sdk_stamp() -> None:
    # Drift guard. The SDK's fingerprint is deliberately looser than Harbor's
    # comparison, but only in ways we have reasoned about. A Harbor upgrade that adds a
    # compared field would silently widen that gap into unexplained full re-runs, so
    # every field must land in exactly one bucket before it can ship.
    job_config = pytest.importorskip("harbor.models.job.config", reason="harbor needs python >= 3.12")
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import (
        _CACHE_IRRELEVANT_OPTIONS,
        _HARBOR_EQ_IGNORED_FIELDS,
    )

    # Derived, not hardcoded: dropping a field into _CACHE_IRRELEVANT_OPTIONS re-checks
    # here instead of quietly diverging from a copied list.
    fingerprinted = set(HarborRuntimeConfig.model_fields) - set(_CACHE_IRRELEVANT_OPTIONS)
    for harbor_field, sdk_fields in _STAMP_COVERED_HARBOR_FIELDS.items():
        missing = sdk_fields - fingerprinted
        assert not missing, (
            f"Harbor compares {harbor_field!r}, but {sorted(missing)} left the cache fingerprint. "
            "Either restore it, or move the field to _KNOWINGLY_LOOSER with a reason."
        )

    classified = (
        set(_HARBOR_EQ_IGNORED_FIELDS) | _SDK_NEVER_SETS | set(_STAMP_COVERED_HARBOR_FIELDS) | set(_KNOWINGLY_LOOSER)
    )
    actual = set(job_config.JobConfig.model_fields)
    assert not actual - classified, (
        f"Harbor's JobConfig grew {sorted(actual - classified)}. Classify each one: covered by the cache stamp "
        "(_STAMP_COVERED_HARBOR_FIELDS), never set by the SDK (_SDK_NEVER_SETS), or knowingly unkeyed "
        "(_KNOWINGLY_LOOSER, with a reason)."
    )
    assert not classified - actual, (
        f"{sorted(classified - actual)} no longer exist on Harbor's JobConfig; drop them from the classification."
    )


def test_trial_dir_without_result_json_is_skipped_and_its_task_reported_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A trial that died before the verifier leaves a dir but no result.json.

    Harbor still creates the trial dir (and often an exception.txt), so the adapter
    has to tolerate the absence rather than raise, while the task it belonged to
    must not silently vanish from the run — it is reported as having no result.
    """
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    _write_trial(job_dir, "ok-task__aaa", "ok-task", reward=1.0)
    crashed = job_dir / "crashed-task__bbb"
    (crashed / "agent").mkdir(parents=True)
    (crashed / "exception.txt").write_text("Traceback (most recent call last): ...")

    tasks = [
        AgentEvalTask(id="ok-task", intent="x", inputs={"instruction": "p"}, metrics=[HarborRewardMetric()]),
        AgentEvalTask(id="crashed-task", intent="y", inputs={"instruction": "q"}, metrics=[HarborRewardMetric()]),
    ]
    with caplog.at_level(logging.WARNING):
        trials = build_trials_from_job_dir(job_dir, tasks)

    assert [trial.task_id for trial in trials] == ["ok-task"]
    assert "crashed-task" in caplog.text


def test_unreadable_result_json_is_skipped_without_raising(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A truncated result.json must not take the whole job's adaptation down."""
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    _write_trial(job_dir, "ok-task__aaa", "ok-task", reward=1.0)
    broken = job_dir / "broken-task__bbb"
    broken.mkdir()
    (broken / "result.json").write_text("{not json")

    tasks = [
        AgentEvalTask(id="ok-task", intent="x", inputs={"instruction": "p"}, metrics=[HarborRewardMetric()]),
        AgentEvalTask(id="broken-task", intent="y", inputs={"instruction": "q"}, metrics=[HarborRewardMetric()]),
    ]
    with caplog.at_level(logging.WARNING):
        trials = build_trials_from_job_dir(job_dir, tasks)

    assert [trial.task_id for trial in trials] == ["ok-task"]
    assert "broken-task" in caplog.text


@pytest.mark.parametrize(
    ("exception_info", "expected_type"),
    [
        ({"exception_type": "TimeoutError"}, "TimeoutError"),
        ({"type": "TimeoutError"}, "TimeoutError"),
        ({"name": "TimeoutError"}, "TimeoutError"),
        ({"class": "TimeoutError"}, "TimeoutError"),
        # A mapping Harbor filled with something unexpected still counts as failed.
        ({"message": "boom"}, "UnknownException"),
        ({}, "UnknownException"),
        # Older/other writers put a bare value there.
        ("NonZeroAgentExitCodeError", "NonZeroAgentExitCodeError"),
        (17, "17"),
        # Degenerate values must normalise, never raise: _trial_from_harbor_result runs outside the
        # only try/except in build_trials_from_job_dir, so a ValidationError here would abandon every
        # remaining trial in the job dir.
        ("", "UnknownException"),
        ("   ", "UnknownException"),
        ({"exception_type": ""}, "UnknownException"),
        ({"exception_type": "   "}, "UnknownException"),
        ({"exception_type": 123}, "UnknownException"),
        ({"exception_type": None}, "UnknownException"),
    ],
)
def test_exception_info_shapes_all_resolve_to_a_type(
    tmp_path: Path, exception_info: object, expected_type: str
) -> None:
    """Any non-null exception_info must mark the trial failed, whatever its shape.

    Only a bare string was covered before, so a mapping — which is what Harbor
    actually writes — went untested. Resolving to None here would silently promote
    a crashed trial to COMPLETED and let it score.
    """
    trial = _adapt_raw_trial(tmp_path, rewards={"reward": 1.0}, exception_info=exception_info)

    assert trial.error is not None
    assert trial.error.type == expected_type
    assert trial.status is AgentEvalTrialStatus.PARTIAL


def test_non_string_message_and_traceback_are_dropped_rather_than_carried(tmp_path: Path) -> None:
    # Same totality requirement as the parametrization above: a producer that put a number (or a
    # nested object) where a string belongs must not take down the whole job dir.
    trial = _adapt_raw_trial(
        tmp_path,
        rewards={"reward": 1.0},
        exception_info={"exception_type": "RuntimeError", "exception_message": 42, "exception_traceback": {"a": 1}},
    )

    assert trial.error is not None
    assert trial.error.type == "RuntimeError"
    assert trial.error.message is None
    assert trial.error.traceback is None


def test_a_real_harbor_exception_payload_round_trips_every_field(tmp_path: Path) -> None:
    """Shaped after a real Harbor result.json: an agent that died with exit 127.

    ``occurred_at`` is naive local wall time while Harbor stamps trial start/finish in UTC, so it is
    kept exactly as written rather than normalised — inventing an offset would fabricate precision.
    """
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    _write_trial(
        job_dir,
        "debug-agent-runtime-error__KFtcHEw",
        "t",
        reward=None,
        exception={
            "exception_type": "RuntimeError",
            "exception_message": "Agent process failed with exit code 127: python: command not found\n",
            "exception_traceback": 'Traceback (most recent call last):\n  File "/Users/x/harbor/trial.py", line 354\n',
            "occurred_at": "2026-08-13T17:22:32.230852",
        },
    )

    [trial] = build_trials_from_job_dir(
        job_dir, [AgentEvalTask(id="t", intent="x", inputs={"instruction": "p"}, metrics=[HarborRewardMetric()])]
    )

    assert trial.error is not None
    assert trial.error.type == "RuntimeError"
    assert trial.error.message is not None and "exit code 127" in trial.error.message
    assert trial.error.traceback is not None and trial.error.traceback.startswith("Traceback")
    occurred_at = trial.error.occurred_at
    assert occurred_at is not None
    assert occurred_at == datetime(2026, 8, 13, 17, 22, 32, 230852)
    assert occurred_at.tzinfo is None  # naive, as Harbor wrote it


def test_an_oversized_traceback_is_truncated(tmp_path: Path) -> None:
    # Bundles are portable and a traceback is diagnostic text, not something anyone joins on, so it
    # is bounded rather than faithful.
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    _write_trial(
        job_dir,
        "t__aaa",
        "t",
        reward=1.0,
        exception={
            "exception_type": "RuntimeError",
            "exception_traceback": "x" * (_EXPECTED_MAX_TRACEBACK_CHARS * 3),
        },
    )

    [trial] = build_trials_from_job_dir(
        job_dir, [AgentEvalTask(id="t", intent="x", inputs={"instruction": "p"}, metrics=[HarborRewardMetric()])]
    )

    assert trial.error is not None
    assert trial.error.traceback is not None
    assert len(trial.error.traceback) == _EXPECTED_MAX_TRACEBACK_CHARS


def test_absent_exception_info_leaves_the_trial_completed(tmp_path: Path) -> None:
    """The negative case that gives the parametrization above its meaning."""
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    _write_trial(job_dir, "t__aaa", "t", reward=1.0, exception=None)

    trials = build_trials_from_job_dir(
        job_dir, [AgentEvalTask(id="t", intent="x", inputs={"instruction": "p"}, metrics=[HarborRewardMetric()])]
    )

    assert trials[0].error is None
    assert trials[0].status is AgentEvalTrialStatus.COMPLETED


def test_the_typed_error_is_the_only_carrier(tmp_path: Path) -> None:
    """The adapter records the failure once, on ``error`` -- nothing is mirrored into metadata.

    A pre-``TrialError`` bundle put the type in free-form metadata. That is not interpreted on load,
    so the two paths do not converge: only a trial adapted by this code carries an error.
    """
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    _write_trial(job_dir, "t__aaa", "t", reward=0.0, exception={"exception_type": "RuntimeError"})

    [fresh] = build_trials_from_job_dir(
        job_dir, [AgentEvalTask(id="t", intent="x", inputs={"instruction": "p"}, metrics=[HarborRewardMetric()])]
    )
    legacy = AgentEvalTrial.model_validate(
        {"id": "t__aaa", "task_id": "t", "status": "partial", "metadata": {"exception_type": "RuntimeError"}}
    )

    assert fresh.error is not None and fresh.error.type == "RuntimeError"
    assert "exception_type" not in fresh.metadata
    assert legacy.error is None  # free-form metadata is never promoted to a typed error


def test_occurred_at_is_not_advertised_as_rfc_3339() -> None:
    # RFC 3339 date-time requires an offset, and Harbor writes a naive local clock. Advertising
    # `format: date-time` would make a strict client parse a zoneless string into its own zone,
    # silently shifting the instant.
    schema = TrialError.model_json_schema()["properties"]["occurred_at"]
    assert "format" not in schema
    assert schema["type"] == "string"


def test_both_naive_and_aware_timestamps_survive_a_round_trip() -> None:
    naive = TrialError(type="E", occurred_at=datetime(2026, 8, 13, 17, 22, 32, 230852))
    aware = TrialError(type="E", occurred_at=datetime(2026, 8, 14, 0, 22, 25, tzinfo=timezone.utc))

    assert TrialError.model_validate(naive.model_dump(mode="json")).occurred_at == naive.occurred_at
    assert TrialError.model_validate(aware.model_dump(mode="json")).occurred_at == aware.occurred_at
    # The naive one serialises without an offset, which is exactly why the schema cannot claim one.
    assert naive.model_dump(mode="json")["occurred_at"] == "2026-08-13T17:22:32.230852"


def test_a_real_harbor_error_payload_reaches_the_summary_rollup(tmp_path: Path) -> None:
    """The exception path against real Harbor bytes, not a hand-written dict.

    Every other test here builds ``exception_info`` with :func:`_write_trial`, which means the whole
    chain is only ever verified against payloads we wrote ourselves. This one replays a captured
    ``result.json`` from an actual Harbor run whose agent blew a 1s timeout, so a change to Harbor's
    on-disk shape shows up as a failure rather than as silently-empty rollups.

    It also pins the case the synthetic tests can only assert by construction: Harbor ran the verifier
    *after* recording the timeout, so this trial carries an error **and** a real reward. It therefore
    appears in ``error_trial_ids`` and in ``task_metric_values`` at once.
    """
    payload = json.loads(_HARBOR_ERROR_RESULT.read_text(encoding="utf-8"))
    trial_name, task_name = payload["trial_name"], payload["task_name"]

    job_dir = tmp_path / "job"
    (job_dir / trial_name).mkdir(parents=True)
    (job_dir / trial_name / "result.json").write_text(json.dumps(payload), encoding="utf-8")

    task = AgentEvalTask(id=task_name, intent="x", inputs={"instruction": "p"}, metrics=[HarborRewardMetric()])
    [trial] = build_trials_from_job_dir(job_dir, [task])

    assert trial.error is not None
    assert trial.error.type == "AgentTimeoutError"
    assert trial.error.message == "Agent execution timed out after 1.0 seconds"
    assert trial.error.traceback is not None and "AgentTimeoutError" in trial.error.traceback
    # Harbor stamps a naive local clock here while writing trial start/finish in UTC; the SDK keeps it
    # exactly as written rather than inventing an offset.
    occurred_at = trial.error.occurred_at
    assert occurred_at is not None and occurred_at.tzinfo is None
    # Errored, but still scored -- FAILED would exclude it from scoring entirely.
    assert trial.status is AgentEvalTrialStatus.PARTIAL
    assert trial.metadata["reward"] == 0.0
    assert "exception_info" not in trial.metadata

    result_evidence = trial.get_evidence("result")
    assert result_evidence is not None
    assert result_evidence.ref is not None
    raw_result = json.loads(Path(result_evidence.ref).read_text(encoding="utf-8"))
    assert raw_result["exception_info"] == payload["exception_info"]

    summary = AgentEvalSummary.from_scores([], trials=[trial])
    assert summary.error_trial_ids == {"AgentTimeoutError": [trial_name]}
    assert summary.error_count == 1


def _atif_trajectory_payload() -> dict[str, object]:
    """A minimal ATIF trajectory shaped like the one Harbor's ATIF-capable agents write."""
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": "s1",
        "agent": {"name": "codex", "version": "1.0"},
        "steps": [
            {"step_id": 1, "source": "user", "message": "solve it", "timestamp": "2026-01-01T00:00:00+00:00"},
            {"step_id": 2, "source": "agent", "message": "thinking", "timestamp": "2026-01-01T00:00:01+00:00"},
            {"step_id": 3, "source": "agent", "message": "204", "timestamp": "2026-01-01T00:00:02+00:00"},
        ],
    }


def _trial_with_trajectory(tmp_path: Path, payload: object) -> AgentEvalTrial:
    job_dir = tmp_path / "job"
    _write_trial(job_dir, "t__1", "t", reward=1.0)
    trial_dir = job_dir / "t__1"
    if payload is None:
        (trial_dir / "agent" / "trajectory.json").unlink()
    else:
        (trial_dir / "agent" / "trajectory.json").write_text(json.dumps(payload))
    data = json.loads((trial_dir / "result.json").read_text())
    return _trial_from_harbor_result(trial_dir, data, reward_key="reward")


def test_atif_trajectory_is_labelled_atif(tmp_path: Path) -> None:
    trial = _trial_with_trajectory(tmp_path, _atif_trajectory_payload())

    # Harbor names the file trajectory.json, so only its contents identify it as ATIF.
    trace = trial.get_evidence("trace")
    assert trace is not None
    assert trace.format == "atif"
    assert trace.description == "Collected Harbor artifact agent/trajectory.json."
    assert trace.metadata == {}


def test_absent_trajectory_leaves_no_trace_evidence(tmp_path: Path) -> None:
    trial = _trial_with_trajectory(tmp_path, None)

    assert trial.get_evidence("trace") is None


def test_non_atif_trajectory_is_an_artifact_not_standard_trace(tmp_path: Path) -> None:
    trial = _trial_with_trajectory(tmp_path, {"not": "atif"})

    assert trial.get_evidence("trace") is None
    artifact = trial.get_evidence("artifact:agent/trajectory.json")
    assert artifact is not None
    assert artifact.kind == "artifact"


def _adapted_measurements(tmp_path: Path, result_data: dict[str, object]) -> dict[str, int | float]:
    trial_dir = tmp_path / "measurement-trial"
    trial_dir.mkdir(exist_ok=True)
    payload: dict[str, object] = {
        "task_name": "task",
        "trial_name": "task__1",
        "verifier_result": {"rewards": {"reward": 1.0}},
        "exception_info": None,
    }
    payload.update(result_data)
    trial = _trial_from_harbor_result(trial_dir, payload, reward_key="reward")
    assert (
        not {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cache_creation_tokens",
            "cache_read_tokens",
            "runtime_sec",
            "cost_usd",
        }
        & trial.metadata.keys()
    )
    return trial.measurements.model_dump(exclude_none=True)


def test_trial_measurements_use_one_source_and_are_total(tmp_path: Path) -> None:
    assert _adapted_measurements(
        tmp_path,
        {
            "agent_result": {"n_input_tokens": 2, "cost_usd": 0.25},
            "step_results": [{"agent_result": {"n_input_tokens": 100, "n_output_tokens": 3}}],
        },
    ) == {"prompt_tokens": 2, "cost_usd": 0.25}
    assert _adapted_measurements(
        tmp_path,
        {
            "step_results": [
                {"agent_result": {"n_input_tokens": 2, "n_output_tokens": 1, "cost_usd": 0.1}},
                {"agent_result": {"n_input_tokens": 3, "n_cache_tokens": 4, "cost_usd": 0.2}},
            ]
        },
    ) == {
        "prompt_tokens": 5,
        "completion_tokens": 1,
        "total_tokens": 6,
        "cache_read_tokens": 4,
        "cost_usd": pytest.approx(0.3),
    }
    assert _adapted_measurements(
        tmp_path, {"agent_result": {"n_input_tokens": 0, "n_output_tokens": 0, "n_cache_tokens": 0, "cost_usd": 0}}
    ) == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cache_read_tokens": 0, "cost_usd": 0.0}
    assert (
        _adapted_measurements(tmp_path, {"agent_result": {}, "step_results": [{"agent_result": {"n_input_tokens": 9}}]})
        == {}
    )
    assert _adapted_measurements(tmp_path, {"step_results": 7}) == {}
    assert (
        _adapted_measurements(
            tmp_path, {"step_results": [None, "bad", {"agent_result": 7}, {"agent_result": {"n_input_tokens": -2}}]}
        )
        == {}
    )


def test_null_top_level_agent_result_uses_step_measurements(tmp_path: Path) -> None:
    assert _adapted_measurements(
        tmp_path,
        {
            "agent_result": None,
            "step_results": [
                {"agent_result": {"n_input_tokens": 2, "n_output_tokens": 1}},
                {"agent_result": {"n_input_tokens": 3, "n_output_tokens": 2}},
            ],
        },
    ) == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}


@pytest.mark.parametrize("bad", [True, "12", float("nan"), float("inf"), float("-inf")])
def test_trial_measurements_reject_malformed_values(tmp_path: Path, bad: object) -> None:
    assert _adapted_measurements(
        tmp_path,
        {"agent_result": {"n_input_tokens": bad, "n_output_tokens": 2, "cost_usd": bad}},
    ) == {"completion_tokens": 2}


def test_invalid_step_field_poisons_only_that_aggregate(tmp_path: Path) -> None:
    assert _adapted_measurements(
        tmp_path,
        {
            "step_results": [
                {"agent_result": {"n_input_tokens": 5, "n_output_tokens": 1}},
                {"agent_result": {"n_input_tokens": "bad", "n_output_tokens": 2}},
            ]
        },
    ) == {"completion_tokens": 3}


def test_trial_measurements_omit_numeric_and_aggregate_cost_overflow(tmp_path: Path) -> None:
    assert _adapted_measurements(tmp_path, {"agent_result": {"n_input_tokens": 1, "cost_usd": 10**1000}}) == {
        "prompt_tokens": 1
    }
    assert _adapted_measurements(
        tmp_path,
        {
            "step_results": [
                {"agent_result": {"n_input_tokens": 1, "cost_usd": 1e308}},
                {"agent_result": {"n_input_tokens": 2, "cost_usd": 1e308}},
            ]
        },
    ) == {"prompt_tokens": 3}


def test_trial_measurements_use_top_level_runtime_without_combining_steps(tmp_path: Path) -> None:
    assert _adapted_measurements(
        tmp_path,
        {
            "agent_execution": {
                "started_at": "2026-01-01T00:00:00+00:00",
                "finished_at": "2026-01-01T00:00:02.500000+00:00",
            },
            "step_results": [
                {
                    "agent_execution": {
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "finished_at": "2026-01-01T00:00:10+00:00",
                    }
                }
            ],
        },
    ) == {"runtime_sec": 2.5}


def test_trial_measurements_sum_step_runtime_when_top_level_is_absent(tmp_path: Path) -> None:
    assert _adapted_measurements(
        tmp_path,
        {
            "agent_execution": None,
            "step_results": [
                {
                    "agent_execution": {
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "finished_at": "2026-01-01T00:00:01+00:00",
                    }
                },
                {},
                {
                    "agent_execution": {
                        "started_at": "2026-01-01T00:00:03+00:00",
                        "finished_at": "2026-01-01T00:00:05+00:00",
                    }
                },
            ],
        },
    ) == {"runtime_sec": 3.0}


@pytest.mark.parametrize(
    "window",
    [
        {"started_at": "2026-01-01T00:00:00+00:00"},
        {
            "started_at": "2026-01-01T00:00:02+00:00",
            "finished_at": "2026-01-01T00:00:01+00:00",
        },
    ],
)
def test_invalid_step_runtime_poisons_the_runtime_aggregate(tmp_path: Path, window: object) -> None:
    assert (
        _adapted_measurements(
            tmp_path,
            {
                "step_results": [
                    {
                        "agent_execution": {
                            "started_at": "2026-01-01T00:00:00+00:00",
                            "finished_at": "2026-01-01T00:00:01+00:00",
                        }
                    },
                    {"agent_execution": window},
                ]
            },
        )
        == {}
    )


def _write_evidence_trial(job_dir: Path, trial_name: str = "task__A1b2") -> Path:
    _write_trial(job_dir, trial_name, "task", reward=1.0)
    trial_dir = job_dir / trial_name
    (trial_dir / "config.json").write_text("{}", encoding="utf-8")
    (trial_dir / "trial.log").write_text("run", encoding="utf-8")
    (trial_dir / "verifier" / "reward.json").write_text('{"reward": 1}', encoding="utf-8")
    return trial_dir


def _adapt_evidence_trial(job_dir: Path) -> AgentEvalTrial:
    task = AgentEvalTask(id="task", intent="x", inputs={"instruction": "p"}, metrics=[HarborRewardMetric()])
    return build_trials_from_job_dir(job_dir, [task])[0]


def test_harbor_evidence_descriptors_are_typed_absolute_and_described(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    trial_dir = _write_evidence_trial(job_dir)
    (trial_dir / "artifacts" / "output.txt").parent.mkdir(parents=True)
    (trial_dir / "artifacts" / "output.txt").write_text("done", encoding="utf-8")

    trial = _adapt_evidence_trial(job_dir)

    expected = {
        "trial_dir": ("filesystem", "dir", trial_dir.resolve()),
        "config": ("json", "json", (trial_dir / "config.json").resolve()),
        "json:config.json": ("json", "json", (trial_dir / "config.json").resolve()),
        "result": ("json", "json", (trial_dir / "result.json").resolve()),
        "json:result.json": ("json", "json", (trial_dir / "result.json").resolve()),
        "log:trial.log": ("log", "text", (trial_dir / "trial.log").resolve()),
        "log:verifier/reward.json": ("log", "json", (trial_dir / "verifier" / "reward.json").resolve()),
        "artifact:artifacts/output.txt": ("artifact", None, (trial_dir / "artifacts" / "output.txt").resolve()),
        "artifact:output.txt": ("artifact", None, (trial_dir / "artifacts" / "output.txt").resolve()),
    }
    for name, (kind, evidence_format, path) in expected.items():
        descriptor = trial.get_evidence(name)
        assert descriptor is not None
        assert (descriptor.kind, descriptor.format, descriptor.ref) == (kind, evidence_format, str(path))
        assert isinstance(descriptor.description, str)

    logs = trial.get_evidence("logs")
    final_state = trial.get_evidence("final_state")
    verifier_logs = trial.get_evidence("verifier_logs")
    assert logs is not None
    assert final_state is not None
    assert verifier_logs is not None
    assert (logs.kind, logs.format, logs.ref) == ("logs", "dir", str((trial_dir / "agent").resolve()))
    assert (final_state.kind, final_state.format, final_state.ref) == (
        "filesystem",
        "dir",
        str((trial_dir / "artifacts").resolve()),
    )
    assert (verifier_logs.kind, verifier_logs.format, verifier_logs.ref) == (
        "logs",
        "dir",
        str((trial_dir / "verifier").resolve()),
    )
    assert set(trial.evidence.descriptors if trial.evidence is not None else ()) == {
        "logs",
        "final_state",
        "verifier_logs",
        "trial_dir",
        "config",
        "json:config.json",
        "result",
        "json:result.json",
        "log:trial.log",
        "log:verifier/reward.json",
        "artifact:agent/trajectory.json",
        "artifact:artifacts/output.txt",
        "artifact:output.txt",
    }


def test_harbor_manifest_descriptions_preserve_their_original_paths(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    trial_dir = _write_evidence_trial(job_dir)
    artifacts_manifest = trial_dir / "artifacts" / "manifest.json"
    artifacts_manifest.parent.mkdir(parents=True)
    artifacts_manifest.write_text("{}", encoding="utf-8")
    root_manifest = trial_dir / "manifest.json"
    root_manifest.write_text("{}", encoding="utf-8")

    trial = _adapt_evidence_trial(job_dir)

    artifact_descriptor = trial.get_evidence("artifact:artifacts/manifest.json")
    root_descriptor = trial.get_evidence("artifact:manifest.json")
    assert artifact_descriptor is not None
    assert root_descriptor is not None
    assert artifact_descriptor.description == (
        "Harbor artifact manifest. Lists collected artifact files and the environment paths they were copied from."
    )
    assert root_descriptor.description == "Collected Harbor artifact manifest.json."


def test_harbor_evidence_keys_preserve_colliding_files_and_noncolliding_aliases(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    trial_dir = _write_evidence_trial(job_dir)
    paths = [
        "artifacts/traces/x.jsonl",
        "traces/x.jsonl",
        "artifacts/traces/y.jsonl",
        "artifacts/foo.txt",
        "foo.txt",
        "artifacts/only.txt",
    ]
    for relative in paths:
        path = trial_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"resourceSpans": []}\n' if path.suffix == ".jsonl" else relative, encoding="utf-8")

    trial = _adapt_evidence_trial(job_dir)

    expected_refs = {
        "trace:artifacts/traces/x.jsonl": trial_dir / "artifacts/traces/x.jsonl",
        "trace:traces/x.jsonl": trial_dir / "traces/x.jsonl",
        "trace:artifacts/traces/y.jsonl": trial_dir / "artifacts/traces/y.jsonl",
        "trace:traces/y.jsonl": trial_dir / "artifacts/traces/y.jsonl",
        "artifact:artifacts/foo.txt": trial_dir / "artifacts/foo.txt",
        "artifact:foo.txt": trial_dir / "foo.txt",
        "artifact:artifacts/only.txt": trial_dir / "artifacts/only.txt",
        "artifact:only.txt": trial_dir / "artifacts/only.txt",
    }
    for name, path in expected_refs.items():
        descriptor = trial.get_evidence(name)
        assert descriptor is not None
        assert descriptor.ref == str(path.resolve())


def test_otlp_becomes_the_standard_trace_when_both_formats_are_present(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    trial_dir = _write_evidence_trial(job_dir)
    traces = trial_dir / "nested" / "traces"
    traces.mkdir(parents=True)
    otlp = traces / "trace.jsonl"
    atif = traces / "trajectory.atif.json"
    otlp.write_text('{"resourceSpans": []}\n', encoding="utf-8")
    atif.write_text(json.dumps(_atif_trajectory_payload()), encoding="utf-8")

    trial = _adapt_evidence_trial(job_dir)

    standard = trial.get_evidence("trace")
    assert standard is not None
    assert standard.ref == str(otlp.resolve())
    assert standard.format == "otlp"
    assert standard.description == "Agent execution trace JSONL for nested/traces/trace.jsonl."
    assert standard.metadata == {}
    # The ATIF view stays reachable, both by format and through its path key.
    by_format = trial.get_evidence("trace:atif")
    assert by_format is not None
    assert by_format.ref == str(atif.resolve())
    assert trial.get_evidence("trace:nested/traces/trajectory.atif.json") is not None


def test_atif_becomes_the_standard_trace_when_no_otlp_trace_exists(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    trial_dir = _write_evidence_trial(job_dir)
    traces = trial_dir / "traces"
    traces.mkdir()
    atif = traces / "trajectory.atif.json"
    atif.write_text(json.dumps(_atif_trajectory_payload()), encoding="utf-8")

    trial = _adapt_evidence_trial(job_dir)

    standard = trial.get_evidence("trace")
    assert standard is not None
    assert standard.format == "atif"
    assert standard.ref == str(atif.resolve())


@pytest.mark.asyncio
async def test_named_atif_trace_keeps_its_identity_and_validates_lazily(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    trial_dir = _write_evidence_trial(job_dir)
    traces = trial_dir / "traces"
    traces.mkdir()
    atif = traces / "broken.atif.json"
    atif.write_text(json.dumps({"schema_version": "ATIF-v1.7", "steps": []}), encoding="utf-8")

    trial = _adapt_evidence_trial(job_dir)

    extension = trial.get_evidence("trace:traces/broken.atif.json")
    selected = trial.get_evidence("trace")
    assert extension is not None
    assert selected is not None
    assert (extension.kind, extension.format, extension.ref) == ("trace", "atif", str(atif.resolve()))
    assert extension.description == "Agent execution ATIF trajectory for traces/broken.atif.json."
    assert extension.metadata == {}
    assert selected.ref == extension.ref
    assert trial.evidence is not None
    handle = await trial.evidence.trace()
    assert isinstance(handle, ATIFTraceHandle)
    with pytest.raises(ValidationError):
        await handle.trace()


def test_a_trial_with_no_trace_artifact_has_no_standard_trace(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    trial_dir = _write_evidence_trial(job_dir)
    (trial_dir / "agent" / "trajectory.json").unlink()

    trial = _adapt_evidence_trial(job_dir)

    assert trial.get_evidence("trace") is None
    assert trial.get_evidence("trace:atif") is None
    assert trial.get_evidence("trace:otlp") is None


@pytest.mark.parametrize(
    ("source_name", "legacy_name"),
    [
        (
            "nemo_evaluator_sdk.agent_eval.runtimes.harbor_trial_adapter",
            "nemo_platform.beta.evaluator.agent_eval.runtimes.harbor_trial_adapter",
        ),
        (
            "nemo_evaluator_sdk.agent_eval.trials",
            "nemo_platform.beta.evaluator.agent_eval.trials",
        ),
    ],
)
def test_legacy_harbor_trial_contract_import_resolves_to_source_module(source_name: str, legacy_name: str) -> None:
    source = importlib.import_module(source_name)
    legacy = importlib.import_module(legacy_name)
    assert source.__file__ is not None
    assert legacy.__file__ is not None

    assert Path(legacy.__file__).resolve() == Path(source.__file__).resolve()


def test_atif_trajectory_supplies_the_final_answer(tmp_path: Path) -> None:
    trial = _trial_with_trajectory(tmp_path, _atif_trajectory_payload())

    assert trial.output is not None
    assert trial.output.output_text == "204"


def test_non_atif_trajectory_yields_no_final_answer(tmp_path: Path) -> None:
    trial = _trial_with_trajectory(tmp_path, {"not": "atif"})

    assert trial.output is not None
    assert trial.output.output_text is None


def test_final_answer_ignores_trailing_non_agent_steps(tmp_path: Path) -> None:
    payload = _atif_trajectory_payload()
    steps = cast(list[dict[str, object]], payload["steps"])
    assert isinstance(steps, list)
    steps.append({"step_id": 4, "source": "system", "message": "cleanup", "timestamp": "2026-01-01T00:00:03+00:00"})

    trial = _trial_with_trajectory(tmp_path, payload)

    assert trial.output is not None
    assert trial.output.output_text == "204"


def test_empty_final_agent_message_is_not_backfilled_from_earlier_reasoning(tmp_path: Path) -> None:
    # Only the last agent step can be the answer; an earlier one is intermediate reasoning.
    payload = _atif_trajectory_payload()
    steps = cast(list[dict[str, object]], payload["steps"])
    assert isinstance(steps, list)
    steps.append({"step_id": 4, "source": "agent", "message": "", "timestamp": "2026-01-01T00:00:03+00:00"})

    trial = _trial_with_trajectory(tmp_path, payload)

    assert trial.output is not None
    assert trial.output.output_text is None


@pytest.mark.asyncio
async def test_both_trace_formats_are_readable_when_a_trial_emits_both(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    trial_dir = _write_evidence_trial(job_dir)
    traces = trial_dir / "traces"
    traces.mkdir()
    (traces / "trace.jsonl").write_text('{"resourceSpans": []}\n', encoding="utf-8")
    (traces / "trajectory.atif.json").write_text(json.dumps(_atif_trajectory_payload()), encoding="utf-8")

    trial = _adapt_evidence_trial(job_dir)

    assert trial.evidence is not None
    assert isinstance(await trial.evidence.trace(format="atif"), ATIFTraceHandle)
    assert isinstance(await trial.evidence.trace(format="otlp"), OTLPTraceHandle)
    # OTLP is primary, and naming that format reuses its handle rather than reparsing.
    primary = trial.get_evidence("trace")
    assert primary is not None
    assert primary.format == "otlp"
    assert await trial.evidence.trace() is await trial.evidence.trace(format="otlp")


@pytest.mark.asyncio
async def test_atif_stays_readable_by_format_when_it_is_also_the_primary(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    trial_dir = _write_evidence_trial(job_dir)
    traces = trial_dir / "traces"
    traces.mkdir()
    (traces / "trajectory.atif.json").write_text(json.dumps(_atif_trajectory_payload()), encoding="utf-8")

    trial = _adapt_evidence_trial(job_dir)

    assert trial.evidence is not None
    assert isinstance(await trial.evidence.trace(format="atif"), ATIFTraceHandle)


def test_harbors_builtin_atif_dump_is_reachable_by_format(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    trial_dir = _write_evidence_trial(job_dir)
    traces = trial_dir / "traces"
    traces.mkdir()
    (traces / "trace.jsonl").write_text('{"resourceSpans": []}\n', encoding="utf-8")
    dump = trial_dir / "agent" / "trajectory.json"
    dump.write_text(json.dumps(_atif_trajectory_payload()), encoding="utf-8")

    # agent/trajectory.json declares no format in its path, so it reaches trace:atif
    # only through the legacy fallback, and only when it parses as ATIF.
    trial = _adapt_evidence_trial(job_dir)

    atif = trial.get_evidence("trace:atif")
    assert atif is not None
    assert atif.ref == str(dump.resolve())


def _otlp_answer_trace(text: str) -> str:
    span = {
        "traceId": "0" * 32,
        "spanId": "a" * 16,
        "name": "run",
        "startTimeUnixNano": "1",
        "endTimeUnixNano": "9",
        "attributes": [
            {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}},
            {"key": "output.value", "value": {"stringValue": text}},
        ],
    }
    return json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": [span]}]}]}) + "\n"


def test_output_text_comes_from_the_otlp_trace_when_the_trial_has_one(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    trial_dir = _write_evidence_trial(job_dir)
    traces = trial_dir / "traces"
    traces.mkdir()
    (traces / "trace.jsonl").write_text(_otlp_answer_trace("otlp answer"), encoding="utf-8")
    (trial_dir / "agent" / "trajectory.json").write_text(json.dumps(_atif_trajectory_payload()), encoding="utf-8")

    trial = _adapt_evidence_trial(job_dir)

    assert trial.output is not None
    assert trial.output.output_text == "otlp answer"


def test_output_text_falls_back_to_atif_when_the_otlp_trace_carries_no_answer(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    trial_dir = _write_evidence_trial(job_dir)
    traces = trial_dir / "traces"
    traces.mkdir()
    # A span tree with no output attribute yields nothing to read, so ATIF still answers.
    (traces / "trace.jsonl").write_text('{"resourceSpans": []}\n', encoding="utf-8")
    (trial_dir / "agent" / "trajectory.json").write_text(json.dumps(_atif_trajectory_payload()), encoding="utf-8")

    trial = _adapt_evidence_trial(job_dir)

    atif_answer = _final_agent_message(read_atif(trial_dir / "agent" / "trajectory.json"))
    assert atif_answer
    assert trial.output is not None
    assert trial.output.output_text == atif_answer


def test_output_text_is_absent_when_the_trial_has_no_readable_trace(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    trial_dir = _write_evidence_trial(job_dir)
    (trial_dir / "agent" / "trajectory.json").unlink()

    trial = _adapt_evidence_trial(job_dir)

    assert trial.output is not None
    assert trial.output.output_text is None


def test_an_unreadable_otlp_trace_does_not_lose_the_atif_answer(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    trial_dir = _write_evidence_trial(job_dir)
    traces = trial_dir / "traces"
    traces.mkdir()
    (traces / "trace.jsonl").write_text("{not json\n", encoding="utf-8")
    (trial_dir / "agent" / "trajectory.json").write_text(json.dumps(_atif_trajectory_payload()), encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        trial = _adapt_evidence_trial(job_dir)

    assert "Ignoring unreadable OTLP trace" in caplog.text
    assert trial.output is not None
    assert trial.output.output_text == _final_agent_message(read_atif(trial_dir / "agent" / "trajectory.json"))


def test_an_empty_message_envelope_falls_back_to_the_atif_answer(tmp_path: Path) -> None:
    # An OTLP trace whose only output is an envelope with no assistant text is not an answer,
    # so the trial's ATIF trajectory still supplies one.
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    trial_dir = _write_evidence_trial(job_dir)
    traces = trial_dir / "traces"
    traces.mkdir()
    span = {
        "traceId": "0" * 32,
        "spanId": "a" * 16,
        "name": "run",
        "startTimeUnixNano": "1",
        "endTimeUnixNano": "9",
        "attributes": [
            {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}},
            {"key": "gen_ai.output.messages", "value": {"stringValue": "[]"}},
        ],
    }
    (traces / "trace.jsonl").write_text(
        json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": [span]}]}]}) + "\n", encoding="utf-8"
    )
    (trial_dir / "agent" / "trajectory.json").write_text(json.dumps(_atif_trajectory_payload()), encoding="utf-8")

    trial = _adapt_evidence_trial(job_dir)

    assert trial.output is not None
    assert trial.output.output_text == _final_agent_message(read_atif(trial_dir / "agent" / "trajectory.json"))


@pytest.mark.asyncio
async def test_an_unreadable_otlp_trace_is_not_promoted_over_a_valid_atif_one(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    trial_dir = _write_evidence_trial(job_dir)
    traces = trial_dir / "traces"
    traces.mkdir()
    (traces / "trace.jsonl").write_text("{not json at all\n", encoding="utf-8")
    (trial_dir / "agent" / "trajectory.json").write_text(json.dumps(_atif_trajectory_payload()), encoding="utf-8")

    trial = _adapt_evidence_trial(job_dir)

    primary = trial.get_evidence("trace")
    assert primary is not None
    assert primary.format == "atif"
    assert trial.evidence is not None
    assert isinstance(await trial.evidence.trace(), ATIFTraceHandle)
    # The malformed file is demoted, not hidden.
    assert trial.get_evidence("trace:otlp") is not None
