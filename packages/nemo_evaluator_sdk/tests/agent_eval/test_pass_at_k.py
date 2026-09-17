# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-level pass@k aggregation: the unbiased estimator, output gating, threshold, and uniformity
across metric types (so Gym and Harbor trials aggregate identically)."""

from __future__ import annotations

import pytest
from nemo_evaluator_sdk.agent_eval.evaluator import AgentEvaluator
from nemo_evaluator_sdk.agent_eval.results import AgentEvalSummary, _pass_at_k, metric_values
from nemo_evaluator_sdk.agent_eval.scores import (
    TRIAL_STATUS_DETAIL,
    AgentEvalDiagnostic,
    AgentEvalDiagnosticSeverity,
    AgentEvalScoreStatus,
    AgentEvalTaskScore,
    is_trial_failure,
)
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput
from nemo_evaluator_sdk.metrics.protocol import Metric, MetricInput, MetricOutput, MetricResult
from nemo_evaluator_sdk.values.protocol import MetricOutputSpec


class _ScoreMetric:
    """Minimal metric declaring continuous-score outputs (pass@k-eligible)."""

    def __init__(self, metric_type: str, *output_names: str) -> None:
        self._type = metric_type
        self._output_names = output_names or ("reward",)

    @property
    def type(self) -> str:
        return self._type

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score(name, required=name == "reward") for name in self._output_names]

    async def compute_scores(self, input: MetricInput) -> MetricResult:  # pragma: no cover - not exercised
        raise NotImplementedError


class _LabelMetric:
    """Minimal metric declaring a label output (must NOT get pass@k)."""

    @property
    def type(self) -> str:
        return "verdict"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.label("category")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:  # pragma: no cover - not exercised
        raise NotImplementedError


def _task(task_id: str, *metrics: Metric) -> AgentEvalTask:
    return AgentEvalTask(id=task_id, intent="t", inputs={}, metrics=list(metrics))


def _score(task_id: str, trial_id: str, metric_type: str, name: str, value: object) -> AgentEvalTaskScore:
    return _score_outputs(task_id, trial_id, metric_type, {name: value})


def _score_outputs(task_id: str, trial_id: str, metric_type: str, outputs: dict[str, object]) -> AgentEvalTaskScore:
    return AgentEvalTaskScore(
        id=f"{task_id}:{trial_id}:{metric_type}",
        run_id="run",
        task_id=task_id,
        trial_id=trial_id,
        metric_type=metric_type,
        status=AgentEvalScoreStatus.COMPLETED,
        outputs=[MetricOutput(name=name, value=value) for name, value in outputs.items()],
    )


def _failed_score(task_id: str, trial_id: str, metric_type: str, *, details: dict[str, object]) -> AgentEvalTaskScore:
    """A FAILED score. ``details`` decides which kind: a trial that failed, or a metric that raised."""
    return AgentEvalTaskScore(
        id=f"{task_id}:{trial_id}:{metric_type}",
        run_id="run",
        task_id=task_id,
        trial_id=trial_id,
        metric_type=metric_type,
        status=AgentEvalScoreStatus.FAILED,
        outputs=[],
        diagnostics=[AgentEvalDiagnostic(severity=AgentEvalDiagnosticSeverity.ERROR, message="boom", details=details)],
    )


def test_pass_at_k_unbiased_estimator() -> None:
    assert _pass_at_k(2, 0, 1) == 0.0  # no passes
    assert _pass_at_k(2, 2, 1) == 1.0  # all pass
    assert _pass_at_k(2, 1, 1) == pytest.approx(0.5)
    assert _pass_at_k(2, 1, 2) == 1.0  # n-c < k -> guaranteed hit
    assert _pass_at_k(3, 1, 1) == pytest.approx(1 / 3)
    assert _pass_at_k(4, 2, 2) == pytest.approx(5 / 6)  # 1 - C(2,2)/C(4,2)


def test_task_pass_at_k_gated_and_uniform_across_metric_types() -> None:
    # Two tasks x 2 trials, scored by two reward metric types (mimicking Gym + Harbor) plus a label
    # metric. t1 rewards [1.0, 0.0] (1/2 pass); t2 [1.0, 1.0] (2/2). pass@1 = mean(0.5, 1.0) = 0.75.
    gym, harbor, label = _ScoreMetric("gym_reward"), _ScoreMetric("harbor_reward"), _LabelMetric()
    tasks = [_task("t1", gym, harbor, label), _task("t2", gym, harbor, label)]
    scores: list[AgentEvalTaskScore] = []
    for mt in ("gym_reward", "harbor_reward"):
        scores += [
            _score("t1", "a0", mt, "reward", 1.0),
            _score("t1", "a1", mt, "reward", 0.0),
            _score("t2", "a0", mt, "reward", 1.0),
            _score("t2", "a1", mt, "reward", 1.0),
        ]
    scores.append(_score("t1", "a0", "verdict", "category", "good"))  # label — ignored by pass@k

    summary = AgentEvalSummary.from_scores(scores, tasks=tasks)
    by_name = {score.name: score for score in summary.scores.scores}

    # The label is *recorded* -- it is per-trial evidence -- but stays out of pass@k. Those two facts
    # together are the whole point of keeping _TASK_METRIC_VALUE_SCHEMAS wider than the pass@k set.
    assert {key: metric_values(a) for key, a in summary.task_metric_values["t1"].items()} == {
        "gym_reward.reward": [1.0, 0.0],
        "harbor_reward.reward": [1.0, 0.0],
        "verdict.category": ["good"],
    }
    assert {key: metric_values(a) for key, a in summary.task_metric_values["t2"].items()} == {
        "gym_reward.reward": [1.0, 1.0],
        "harbor_reward.reward": [1.0, 1.0],
        "verdict.category": [],  # t2 declares the label metric but no verdict was scored for it
    }
    for metric_type in ("gym_reward", "harbor_reward"):  # uniform across runners
        assert by_name[f"{metric_type}.reward.pass@1"].mean == pytest.approx(0.75)
        assert by_name[f"{metric_type}.reward.pass@2"].mean == pytest.approx(1.0)
    assert not any(name.startswith("verdict") and ".pass@" in name for name in by_name)  # label not eligible


def test_partial_credit_is_not_a_pass() -> None:
    # pass@k answers "did the agent solve the task", so only full credit counts: of values 0.5 and
    # 1.0, exactly one is a pass.
    tasks = [_task("t1", _ScoreMetric("reward"))]
    scores = [_score("t1", "a0", "reward", "reward", 0.5), _score("t1", "a1", "reward", "reward", 1.0)]

    by_name = {s.name: s for s in AgentEvalSummary.from_scores(scores, tasks=tasks).scores.scores}

    assert by_name["reward.reward.pass@1"].mean == pytest.approx(0.5)
    assert by_name["reward.reward.pass@2"].mean == pytest.approx(1.0)  # one of the two trials passed


def test_population_and_sample_stats_are_both_reported() -> None:
    # The two conventions answer different questions, so both are named explicitly rather than
    # leaving the divisor implicit: population divides by n, sample (Bessel) by n-1.
    tasks = [_task("t1", _ScoreMetric("reward")), _task("t2", _ScoreMetric("reward"))]
    values = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]  # n=10, mean=0.6, sum_sq_dev=2.4
    scores = [_score(f"t{1 + index % 2}", f"a{index}", "reward", "reward", value) for index, value in enumerate(values)]

    aggregate = AgentEvalSummary.from_scores(scores, tasks=tasks).score("reward.reward")

    assert aggregate.count == 10
    assert aggregate.mean == pytest.approx(0.6)
    assert aggregate.std_dev == pytest.approx((2.4 / 10) ** 0.5)  # population
    assert aggregate.sample_std_dev == pytest.approx((2.4 / 9) ** 0.5)  # sample (Bessel-corrected)
    assert aggregate.variance == pytest.approx(2.4 / 10)
    assert aggregate.sample_variance == pytest.approx(2.4 / 9)


def test_sample_stats_undefined_for_a_single_value() -> None:
    # Sample statistics need at least two observations; None means "not estimable", not zero.
    tasks = [_task("t1", _ScoreMetric("reward"))]
    scores = [_score("t1", "a0", "reward", "reward", 1.0)]

    aggregate = AgentEvalSummary.from_scores(scores, tasks=tasks).score("reward.reward")

    assert aggregate.std_dev == 0.0  # population is well-defined for one value
    assert aggregate.sample_std_dev is None
    assert aggregate.sample_variance is None


def test_a_failed_trial_counts_as_not_passing_not_absent() -> None:
    # The agent solved the task once and its other rollout died. Dropping the dead one would report
    # pass@1 = 1.0 ("solved it first try") and make pass@2 vanish along with the trial that earned it.
    tasks = [_task("t1", _ScoreMetric("reward"))]
    scores = [
        _score("t1", "a0", "reward", "reward", 1.0),
        _failed_score("t1", "a1", "reward", details={TRIAL_STATUS_DETAIL: "failed"}),
    ]

    summary = AgentEvalSummary.from_scores(scores, tasks=tasks)
    by_name = {s.name: s for s in summary.scores.scores}

    assert metric_values(summary.task_metric_values["t1"]["reward.reward"]) == [1.0, None]
    assert by_name["reward.reward.pass@1"].mean == pytest.approx(0.5)  # 1 of 2 trials, not 1 of 1
    assert by_name["reward.reward.pass@2"].mean == pytest.approx(1.0)
    assert by_name["reward.reward.pass@1"].nan_count == 0  # the task was measured, so nothing is missing


def test_a_metric_that_raised_leaves_the_trial_unmeasured_rather_than_failed() -> None:
    # The distinction the trial_status detail exists for: a judge that timed out tells us nothing about
    # whether the agent passed, so it must not be charged to the agent the way a dead rollout is.
    # t1: [1.0, metric raised] -> n=1, pass@1 = 1.0.  t2: [1.0, 0.0] -> n=2, pass@1 = 0.5, pass@2 = 1.0.
    tasks = [_task("t1", _ScoreMetric("reward")), _task("t2", _ScoreMetric("reward"))]
    scores = [
        _score("t1", "a0", "reward", "reward", 1.0),
        _failed_score("t1", "a1", "reward", details={"exception_type": "TimeoutError"}),
        _score("t2", "a0", "reward", "reward", 1.0),
        _score("t2", "a1", "reward", "reward", 0.0),
    ]

    summary = AgentEvalSummary.from_scores(scores, tasks=tasks)
    by_name = {s.name: s for s in summary.scores.scores}

    assert metric_values(summary.task_metric_values["t1"]["reward.reward"]) == [1.0]
    assert by_name["reward.reward.pass@1"].mean == pytest.approx(0.75)  # mean(1.0, 0.5), not mean(0.5, 0.5)
    # t1 had two applicable attempts but only one measured value, so pass@2 stays visibly
    # unestimable instead of silently dropping the task.
    assert by_name["reward.reward.pass@2"].count == 1
    assert by_name["reward.reward.pass@2"].nan_count == 1


def test_sparse_optional_output_emits_pass_rows_through_applicable_attempts() -> None:
    tasks = [_task("A", _ScoreMetric("harbor_reward", "format_ok", "reward"))]
    scores = [
        _score_outputs("A", "a1", "harbor_reward", {"reward": 1.0, "format_ok": 1.0}),
        _score_outputs("A", "a2", "harbor_reward", {"reward": 0.0}),
    ]

    summary = AgentEvalSummary.from_scores(scores, tasks=tasks)
    format_pass_1 = summary.score("harbor_reward.format_ok.pass@1")
    format_pass_2 = summary.score("harbor_reward.format_ok.pass@2")

    assert (format_pass_1.count, format_pass_1.nan_count, format_pass_1.mean) == (1, 0, 1.0)
    assert (format_pass_2.count, format_pass_2.nan_count, format_pass_2.mean) == (0, 1, None)
    # Exact fully qualified names prove alphabetical output order never selects the primary.
    assert summary.score("harbor_reward.reward.pass@1").mean == pytest.approx(0.5)


def test_all_omitted_optional_output_retains_each_applicable_pass_row() -> None:
    tasks = [_task("A", _ScoreMetric("harbor_reward", "reward", "format_ok"))]
    scores = [
        _score_outputs("A", "a1", "harbor_reward", {"reward": 1.0}),
        _score_outputs("A", "a2", "harbor_reward", {"reward": 0.0}),
    ]

    summary = AgentEvalSummary.from_scores(scores, tasks=tasks)

    for k in (1, 2):
        score = summary.score(f"harbor_reward.format_ok.pass@{k}")
        assert (score.count, score.nan_count, score.mean) == (0, 1, None)


def test_failed_metric_stays_out_of_n_while_failed_trial_is_a_non_pass() -> None:
    tasks = [
        _task("trial-failed", _ScoreMetric("harbor_reward", "reward", "format_ok")),
        _task("metric-failed", _ScoreMetric("harbor_reward", "reward", "format_ok")),
    ]
    scores = [
        _score_outputs("trial-failed", "a1", "harbor_reward", {"reward": 1.0, "format_ok": 1.0}),
        _failed_score("trial-failed", "a2", "harbor_reward", details={TRIAL_STATUS_DETAIL: "failed"}),
        _score_outputs("metric-failed", "b1", "harbor_reward", {"reward": 1.0, "format_ok": 1.0}),
        _failed_score("metric-failed", "b2", "harbor_reward", details={"exception_type": "TimeoutError"}),
    ]

    summary = AgentEvalSummary.from_scores(scores, tasks=tasks)

    assert summary.score("harbor_reward.format_ok.pass@1").mean == pytest.approx(0.75)
    pass_2 = summary.score("harbor_reward.format_ok.pass@2")
    assert (pass_2.count, pass_2.nan_count, pass_2.mean) == (1, 1, 1.0)


def test_pass_at_two_counts_each_declaring_task_even_when_one_has_n_one() -> None:
    tasks = [
        _task("A", _ScoreMetric("harbor_reward", "reward", "format_ok")),
        _task("B", _ScoreMetric("harbor_reward", "reward", "format_ok")),
        _task("C", _ScoreMetric("harbor_reward", "reward")),
    ]
    scores = [
        _score_outputs("A", "a1", "harbor_reward", {"reward": 1.0, "format_ok": 1.0}),
        _score_outputs("A", "a2", "harbor_reward", {"reward": 0.0, "format_ok": 0.0}),
        _score_outputs("B", "b1", "harbor_reward", {"reward": 1.0, "format_ok": 1.0}),
        _score_outputs("B", "b2", "harbor_reward", {"reward": 0.0}),
        _score_outputs("C", "c1", "harbor_reward", {"reward": 1.0}),
        _score_outputs("C", "c2", "harbor_reward", {"reward": 0.0}),
    ]

    summary = AgentEvalSummary.from_scores(scores, tasks=tasks)
    pass_2 = summary.score("harbor_reward.format_ok.pass@2")

    assert (pass_2.count, pass_2.nan_count, pass_2.mean) == (1, 1, 1.0)
    coverage = summary.metric_coverage["harbor_reward"]["format_ok"]
    assert coverage.total == 4  # task C does not declare format_ok and creates no opportunity


def test_a_task_with_no_usable_value_is_reported_as_nan_count_uniformly_across_k() -> None:
    # Every trial on t2 was unmeasurable, so it contributes to no estimate at any k. Silently
    # narrowing the denominator is what made a shrunken pass@k indistinguishable from a clean one.
    tasks = [_task("t1", _ScoreMetric("reward")), _task("t2", _ScoreMetric("reward"))]
    scores = [
        _score("t1", "a0", "reward", "reward", 1.0),
        _score("t1", "a1", "reward", "reward", 1.0),
        _failed_score("t2", "a0", "reward", details={"exception_type": "TimeoutError"}),
        _failed_score("t2", "a1", "reward", details={"exception_type": "TimeoutError"}),
    ]

    by_name = {s.name: s for s in AgentEvalSummary.from_scores(scores, tasks=tasks).scores.scores}

    for k in (1, 2):
        assert by_name[f"reward.reward.pass@{k}"].mean == pytest.approx(1.0)
        assert by_name[f"reward.reward.pass@{k}"].count == 1  # only t1 could be estimated
        assert by_name[f"reward.reward.pass@{k}"].nan_count == 1  # ...and t2 says so, at every k


def test_pass_at_k_skipped_without_task_specs() -> None:
    # No tasks -> no metric specs -> pass@k can't know which outputs are score-like, so none emitted.
    scores = [_score("t1", "a0", "reward", "reward", 1.0)]
    summary = AgentEvalSummary.from_scores(scores, tasks=None)
    assert not any(".pass@" in score.name for score in summary.scores.scores)


def test_a_trial_status_detail_that_is_not_a_failure_does_not_make_one() -> None:
    # `trial_status` is a natural key for a hand-built score to carry for its own reasons, and the
    # model is public. Matching on the value rather than the key's presence keeps a metric failure that
    # happens to mention a trial status out of the pass@k denominator.
    tasks = [_task("t1", _ScoreMetric("reward"))]
    scores = [
        _score("t1", "a0", "reward", "reward", 1.0),
        _failed_score("t1", "a1", "reward", details={TRIAL_STATUS_DETAIL: "completed", "exception_type": "ValueError"}),
    ]

    assert is_trial_failure(scores[1]) is False
    by_name = {s.name: s for s in AgentEvalSummary.from_scores(scores, tasks=tasks).scores.scores}
    assert by_name["reward.reward.pass@1"].mean == pytest.approx(1.0)  # n=1, not charged the bad trial


@pytest.mark.asyncio
async def test_is_trial_failure_matches_what_the_evaluator_actually_stamps() -> None:
    """Pin both producers of a FAILED score against the discriminator that reads them.

    ``is_trial_failure`` keys off a diagnostic detail rather than a typed field, so the classification
    is only as durable as the two sites that stamp it. Exercising the real scoring path — rather than
    hand-built scores — is what makes an edit to either site fail here instead of quietly re-routing
    dead rollouts out of the pass@k denominator (or judge timeouts into it).
    """

    class _RaisingMetric(_ScoreMetric):
        async def compute_scores(self, input: MetricInput) -> MetricResult:
            raise TimeoutError("judge timed out")

    tasks = [
        AgentEvalTask(id="t1", intent="t", inputs={}, metrics=[_ScoreMetric("reward")]),
        AgentEvalTask(id="t2", intent="t", inputs={}, metrics=[_RaisingMetric("reward")]),
    ]
    trials = [
        AgentEvalTrial(id="a0", task_id="t1", status=AgentEvalTrialStatus.FAILED, output=AgentOutput()),
        AgentEvalTrial(id="a1", task_id="t2", status=AgentEvalTrialStatus.COMPLETED, output=AgentOutput()),
    ]

    result = await AgentEvaluator().run(tasks=tasks, trials=trials)
    by_task = {score.task_id: score for score in result.scores}

    assert by_task["t1"].status is AgentEvalScoreStatus.FAILED
    assert by_task["t2"].status is AgentEvalScoreStatus.FAILED
    assert is_trial_failure(by_task["t1"]) is True  # the trial died -> did not pass
    assert is_trial_failure(by_task["t2"]) is False  # the metric died -> unmeasured
