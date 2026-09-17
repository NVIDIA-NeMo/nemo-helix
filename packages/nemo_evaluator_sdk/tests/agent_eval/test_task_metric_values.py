# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
from nemo_evaluator_sdk.agent_eval.results import (
    AgentEvalSummary,
    TrialMetricValue,
    TrialMetricValueType,
    metric_values,
    numeric_metric_values,
)
from nemo_evaluator_sdk.agent_eval.scores import (
    TRIAL_STATUS_DETAIL,
    AgentEvalDiagnostic,
    AgentEvalDiagnosticSeverity,
    AgentEvalScoreStatus,
    AgentEvalTaskScore,
)
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask, SemanticReducer, SemanticView, ViewSignal
from nemo_evaluator_sdk.metrics.protocol import Metric, MetricInput, MetricOutput, MetricResult
from nemo_evaluator_sdk.values.protocol import MetricOutputSpec
from pydantic import RootModel, ValidationError


def _vendored_module(name: str) -> Any:
    return import_module(f"nemo_platform.beta.evaluator.agent_eval.{name}")


class _TokenCount(RootModel[int]):
    """A free-model output: numeric, but a measurement rather than a per-trial score."""


class _Metric:
    def __init__(self, metric_type: str, *outputs: MetricOutputSpec) -> None:
        self._type = metric_type
        self._outputs = outputs

    @property
    def type(self) -> str:
        return self._type

    def output_spec(self) -> list[MetricOutputSpec]:
        return list(self._outputs)

    async def compute_scores(self, input: MetricInput) -> MetricResult:  # pragma: no cover
        raise NotImplementedError


class _SinglePassScores(list[AgentEvalTaskScore]):
    def __init__(self, scores: list[AgentEvalTaskScore]) -> None:
        super().__init__(scores)
        self.iterations = 0

    def __iter__(self) -> Iterator[AgentEvalTaskScore]:
        self.iterations += 1
        assert self.iterations == 1, "scores were rescanned"
        return super().__iter__()


def _task(task_id: str, *metrics: _Metric) -> AgentEvalTask:
    return AgentEvalTask(id=task_id, intent="test", inputs={}, metrics=list(metrics))


def _pairs(
    records: dict[str, dict[str, list[TrialMetricValue]]],
) -> dict[str, dict[str, list[tuple[str, float | int | bool | str | None]]]]:
    """Flatten trial-metric records to ``(trial_id, value)`` so assertions stay readable."""
    return {
        task_id: {key: [(a.trial_id, a.value) for a in values] for key, values in by_key.items()}
        for task_id, by_key in records.items()
    }


def _score(
    task_id: str,
    trial_id: str,
    metric_type: str,
    output_name: str,
    value: object,
    *,
    status: AgentEvalScoreStatus = AgentEvalScoreStatus.COMPLETED,
) -> AgentEvalTaskScore:
    return _score_outputs(
        task_id,
        trial_id,
        metric_type,
        {output_name: value},
        status=status,
    )


def _score_outputs(
    task_id: str,
    trial_id: str,
    metric_type: str,
    outputs: dict[str, object],
    *,
    status: AgentEvalScoreStatus = AgentEvalScoreStatus.COMPLETED,
) -> AgentEvalTaskScore:
    return AgentEvalTaskScore(
        id=f"run:{task_id}:{trial_id}:{metric_type}",
        run_id="run",
        task_id=task_id,
        trial_id=trial_id,
        metric_type=metric_type,
        status=status,
        outputs=[MetricOutput(name=name, value=value) for name, value in outputs.items()],
    )


def _harbor_task(task_id: str, *, format_ok: bool = True, with_view: bool = False) -> AgentEvalTask:
    outputs = [MetricOutputSpec.continuous_score("reward")]
    if format_ok:
        outputs.append(MetricOutputSpec.continuous_score("format_ok", required=False))
    views = (
        {
            "format_quality": SemanticView(
                reducer=SemanticReducer.SINGLE,
                signals=[ViewSignal(metric="harbor_reward", output="format_ok")],
            )
        }
        if with_view
        else {}
    )
    return AgentEvalTask(
        id=task_id,
        intent="test",
        inputs={},
        metrics=[_Metric("harbor_reward", *outputs)],
        views=views,
    )


def _failed_score(
    task_id: str, trial_id: str, *, trial_failed: bool, metric_type: str = "reward"
) -> AgentEvalTaskScore:
    details = {TRIAL_STATUS_DETAIL: "failed"} if trial_failed else {"exception_type": "TimeoutError"}
    return AgentEvalTaskScore(
        id=f"run:{task_id}:{trial_id}:{metric_type}",
        run_id="run",
        task_id=task_id,
        trial_id=trial_id,
        metric_type=metric_type,
        status=AgentEvalScoreStatus.FAILED,
        diagnostics=[
            AgentEvalDiagnostic(
                severity=AgentEvalDiagnosticSeverity.ERROR,
                message="failed",
                details=details,
            )
        ],
    )


def test_summary_exposes_ordered_native_metric_values_per_task() -> None:
    reward = _Metric("reward", MetricOutputSpec.continuous_score("score"))
    retries = _Metric("retries", MetricOutputSpec.discrete_score("count"))
    verdict = _Metric("verdict", MetricOutputSpec.label("label"))
    complete = _Metric("complete", MetricOutputSpec.boolean("passed"))
    tasks = [_task("task-a", reward, retries, verdict, complete), _task("task-b", reward)]
    scores = [
        _score("task-a", "trial-0", "reward", "score", 1.0),
        _score("task-a", "trial-0", "retries", "count", 2),
        _score("task-a", "trial-0", "verdict", "label", "good"),
        _score("task-a", "trial-0", "complete", "passed", True),
        _score(
            "task-a",
            "trial-1",
            "reward",
            "score",
            0.25,
            status=AgentEvalScoreStatus.PARTIAL,
        ),
        _score("task-a", "trial-1", "retries", "count", 3),
        _score("task-a", "trial-1", "complete", "passed", False),
        _score("task-b", "trial-0", "reward", "score", 0.0),
    ]

    summary = AgentEvalSummary.from_scores(scores, tasks=tasks)

    assert _pairs(summary.task_metric_values) == {
        "task-a": {
            "complete.passed": [("trial-0", True), ("trial-1", False)],
            "retries.count": [("trial-0", 2), ("trial-1", 3)],
            "reward.score": [("trial-0", 1.0), ("trial-1", 0.25)],
            "verdict.label": [("trial-0", "good")],
        },
        "task-b": {"reward.score": [("trial-0", 0.0)]},
    }

    # `==` cannot see the type change (2 == 2.0, True == 1.0), so assert it directly -- preserving
    # the metric's own type is the whole point.
    records = summary.task_metric_values["task-a"]
    count = records["retries.count"][0].value
    assert isinstance(count, int) and not isinstance(count, bool)
    assert records["complete.passed"][0].value is True
    assert isinstance(records["reward.score"][0].value, float)
    assert records["verdict.label"][0].value_type is TrialMetricValueType.LABEL


def test_task_metric_values_scans_scores_once() -> None:
    tasks = [_task("task-a", _Metric("reward", MetricOutputSpec.continuous_score("score")))]
    scores = _SinglePassScores(
        [
            _score("task-a", "trial-0", "reward", "score", 1.0),
            _score("task-a", "trial-1", "reward", "score", 0.0),
        ]
    )

    assert _pairs(AgentEvalSummary.from_scores(scores, tasks=tasks).task_metric_values) == {
        "task-a": {"reward.score": [("trial-0", 1.0), ("trial-1", 0.0)]}
    }


def test_sparse_optional_output_is_consistent_across_summary_consumers() -> None:
    task = _harbor_task("A", with_view=True)
    scores = [
        _score_outputs("A", "a1", "harbor_reward", {"reward": 1.0, "format_ok": 1.0}),
        _score_outputs("A", "a2", "harbor_reward", {"reward": 0.0}),
    ]

    summary = AgentEvalSummary.from_scores(scores, tasks=[task])
    raw = summary.score("harbor_reward.format_ok")
    coverage = summary.metric_coverage["harbor_reward"]["format_ok"]
    view = summary.score("view.format_quality")

    assert (raw.count, raw.nan_count, raw.mean) == (1, 1, 1.0)
    assert raw.sample_std_dev is None and raw.sample_variance is None
    assert (coverage.total, coverage.scored, coverage.missing, coverage.failed) == (2, 1, 1, 0)
    assert raw.count is not None
    assert raw.count + raw.nan_count == 2
    assert coverage.scored + coverage.missing + coverage.failed == coverage.total
    assert _pairs(summary.task_metric_values)["A"]["harbor_reward.format_ok"] == [("a1", 1.0)]
    assert (view.count, view.nan_count, view.mean) == (1, 1, 1.0)


def test_all_omitted_optional_output_retains_unmeasured_summary_rows() -> None:
    task = _harbor_task("A", with_view=True)
    scores = [
        _score_outputs("A", "a1", "harbor_reward", {"reward": 1.0}),
        _score_outputs("A", "a2", "harbor_reward", {"reward": 0.0}),
    ]

    summary = AgentEvalSummary.from_scores(scores, tasks=[task])
    raw = summary.score("harbor_reward.format_ok")
    coverage = summary.metric_coverage["harbor_reward"]["format_ok"]
    view = summary.score("view.format_quality")

    assert (raw.count, raw.nan_count, raw.mean) == (0, 2, None)
    assert (coverage.total, coverage.scored, coverage.missing, coverage.failed) == (2, 0, 2, 0)
    assert _pairs(summary.task_metric_values)["A"]["harbor_reward.format_ok"] == []
    assert (view.count, view.nan_count, view.mean) == (0, 2, None)
    assert summary.score("harbor_reward.format_ok.pass@1").nan_count == 1
    assert summary.score("harbor_reward.format_ok.pass@2").nan_count == 1


def test_failed_trial_and_failed_metric_have_distinct_observations() -> None:
    task = _harbor_task("A", with_view=True)
    trial_failure = AgentEvalSummary.from_scores(
        [
            _score_outputs("A", "a1", "harbor_reward", {"reward": 1.0, "format_ok": 1.0}),
            _failed_score("A", "a2", trial_failed=True, metric_type="harbor_reward"),
        ],
        tasks=[task],
    )
    metric_failure = AgentEvalSummary.from_scores(
        [
            _score_outputs("A", "a1", "harbor_reward", {"reward": 1.0, "format_ok": 1.0}),
            _failed_score("A", "a2", trial_failed=False, metric_type="harbor_reward"),
        ],
        tasks=[task],
    )

    for summary in (trial_failure, metric_failure):
        raw = summary.score("harbor_reward.format_ok")
        coverage = summary.metric_coverage["harbor_reward"]["format_ok"]
        assert (raw.count, raw.nan_count, raw.mean) == (1, 1, 1.0)
        assert (coverage.total, coverage.scored, coverage.missing, coverage.failed) == (2, 1, 0, 1)
        assert summary.score("view.format_quality").nan_count == 1
    assert _pairs(trial_failure.task_metric_values)["A"]["harbor_reward.format_ok"] == [
        ("a1", 1.0),
        ("a2", None),
    ]
    assert _pairs(metric_failure.task_metric_values)["A"]["harbor_reward.format_ok"] == [("a1", 1.0)]
    assert trial_failure.score("harbor_reward.format_ok.pass@1").mean == pytest.approx(0.5)
    assert metric_failure.score("harbor_reward.format_ok.pass@1").mean == pytest.approx(1.0)


def test_output_applicability_is_task_local_for_every_summary_consumer() -> None:
    tasks = [_harbor_task("A", with_view=True), _harbor_task("B", format_ok=False)]
    scores = [
        _score_outputs("A", "a1", "harbor_reward", {"reward": 1.0, "format_ok": 1.0}),
        _score_outputs("A", "a2", "harbor_reward", {"reward": 0.0}),
        _score_outputs("B", "b1", "harbor_reward", {"reward": 1.0}),
        _score_outputs("B", "b2", "harbor_reward", {"reward": 0.0}),
    ]

    summary = AgentEvalSummary.from_scores(scores, tasks=tasks)
    raw = summary.score("harbor_reward.format_ok")
    coverage = summary.metric_coverage["harbor_reward"]["format_ok"]

    assert (raw.count, raw.nan_count, raw.mean) == (1, 1, 1.0)
    assert (coverage.total, coverage.scored, coverage.missing, coverage.failed) == (2, 1, 1, 0)
    assert "harbor_reward.format_ok" not in summary.task_metric_values["B"]
    assert summary.score("view.format_quality").nan_count == 1


def test_failed_trials_are_recorded_but_metric_failures_are_unmeasured() -> None:
    tasks = [
        _task("flaky", _Metric("reward", MetricOutputSpec.continuous_score("score"))),
        _task("unmeasured", _Metric("reward", MetricOutputSpec.continuous_score("score"))),
    ]
    scores = [
        _score("flaky", "trial-0", "reward", "score", 1.0),
        _failed_score("flaky", "trial-1", trial_failed=True),
        _failed_score("unmeasured", "trial-0", trial_failed=False),
    ]

    summary = AgentEvalSummary.from_scores(scores, tasks=tasks)

    assert _pairs(summary.task_metric_values) == {
        # The dead trial keeps its identity, paired with its None; the unmeasured one has no entry at
        # all, so it is nameable from neither -- that asymmetry is what pass@k depends on.
        "flaky": {"reward.score": [("trial-0", 1.0), ("trial-1", None)]},
        "unmeasured": {"reward.score": []},
    }


def test_a_task_that_produced_no_trial_is_unmeasured_and_counted_in_pass_at_k_nan() -> None:
    # from_scores can be handed a task list wider than the scores -- a caller re-aggregating a subset.
    # The task still declares the metric, so it holds an empty value list and counts as missing
    # coverage: excluding it would report pass@k over a denominator smaller than the task set asked
    # for. A full run cannot reach this state; AgentEvaluator._score_trials refuses to score when a
    # task produced no trial, which test_evaluator.py::test_run_rejects_tasks_without_trials pins.
    reward = _Metric("reward", MetricOutputSpec.continuous_score("score"))
    tasks = [_task("scored", reward), _task("never-ran", reward)]
    scores = [_score("scored", "trial-0", "reward", "score", 1.0)]

    summary = AgentEvalSummary.from_scores(scores, tasks=tasks)
    by_name = {score.name: score for score in summary.scores.scores}

    assert _pairs(summary.task_metric_values) == {
        "scored": {"reward.score": [("trial-0", 1.0)]},
        "never-ran": {"reward.score": []},
    }
    assert by_name["reward.score.pass@1"].mean == 1.0  # the one measured task passed
    assert by_name["reward.score.pass@1"].count == 1
    assert by_name["reward.score.pass@1"].nan_count == 1  # ...and the unrun one is not hidden


def test_outputs_declared_under_an_unretained_schema_stay_out_even_when_numeric() -> None:
    # Token measurements and other free models are excluded by their declared schema. Emitting a
    # numeric value must not add them back: the value is a measurement, not a per-trial score.
    tasks = [
        _task(
            "task-a",
            _Metric("reward", MetricOutputSpec.continuous_score("score")),
            _Metric("usage", MetricOutputSpec.model("prompt_tokens", _TokenCount)),
        )
    ]
    scores = [
        _score("task-a", "trial-0", "reward", "score", 1.0),
        _score("task-a", "trial-0", "usage", "prompt_tokens", 1234),
    ]

    assert _pairs(AgentEvalSummary.from_scores(scores, tasks=tasks).task_metric_values) == {
        "task-a": {"reward.score": [("trial-0", 1.0)]}
    }


def test_task_specs_do_not_rediscover_undeclared_outputs() -> None:
    # With task specs, declared output pairs are the complete applicability contract. Task-a's
    # unretained model output and task-b's undeclared output must both stay out of task values.
    tasks = [
        _task("task-a", _Metric("usage", MetricOutputSpec.model("prompt_tokens", _TokenCount))),
        _task("task-b", _Metric("reward", MetricOutputSpec.continuous_score("score"))),
    ]
    scores = [
        _score("task-a", "trial-0", "usage", "prompt_tokens", 100),
        _score("task-b", "trial-0", "reward", "score", 1.0),
        _score("task-b", "trial-0", "usage", "prompt_tokens", 250),  # undeclared on task-b
    ]

    records = AgentEvalSummary.from_scores(scores, tasks=tasks).task_metric_values

    # task-a declared it under an unretained schema, so it is not a key there at all -- not even an
    # empty one -- and the numeric value it emitted cannot add it back.
    assert records["task-a"] == {}
    assert records["task-b"].keys() == {"reward.score"}


def test_nan_metric_values_survive_json_as_a_string() -> None:
    # A metric may legitimately score NaN. json.dumps would write a bare NaN token, which is not
    # valid JSON, so summary.json must carry the string form -- and read it back as a float.
    tasks = [_task("task-a", _Metric("reward", MetricOutputSpec.continuous_score("score")))]
    summary = AgentEvalSummary.from_scores([_score("task-a", "trial-0", "reward", "score", float("nan"))], tasks=tasks)

    payload = summary.model_dump(mode="json")
    record = payload["task_metric_values"]["task-a"]["reward.score"][0]
    assert record["value"] == "NaN"
    assert record["value_type"] == "number"  # what tells it apart from a label reading "NaN"

    # Strict JSON: no bare NaN/Infinity tokens anywhere in the serialized bundle.
    def _reject(constant: str) -> float:
        raise AssertionError(f"summary.json contains a bare {constant} token")

    reloaded = json.loads(json.dumps(payload), parse_constant=_reject)
    value = AgentEvalSummary.model_validate(reloaded).task_metric_values["task-a"]["reward.score"][0].value
    assert isinstance(value, float) and math.isnan(value)


def test_a_label_and_a_real_nan_are_distinguishable_on_the_wire() -> None:
    # The reason value_type is required on the wire: strict JSON has no NaN literal, so a real NaN
    # travels as the string "NaN" -- the same three bytes as a judge label that happens to read "NaN".
    real_nan = TrialMetricValue(trial_id="t0", value=float("nan"))
    label = TrialMetricValue(trial_id="t1", value_type=TrialMetricValueType.LABEL, value="NaN")

    dumped = [a.model_dump(mode="json") for a in (real_nan, label)]
    assert [d["value"] for d in dumped] == ["NaN", "NaN"]  # identical payloads...
    assert [d["value_type"] for d in dumped] == ["number", "label"]  # ...told apart by the type

    back = [TrialMetricValue.model_validate(d) for d in json.loads(json.dumps(dumped))]
    assert isinstance(back[0].value, float) and math.isnan(back[0].value)
    assert back[1].value == "NaN" and isinstance(back[1].value, str)


def test_a_payload_without_value_type_still_loads() -> None:
    # Bundles written before value_type existed, and hand-built records. The old encoding gave a
    # string exactly one meaning -- the NaN escape -- so that is how a bare string is read.
    legacy_nan = TrialMetricValue.model_validate({"trial_id": "t0", "value": "NaN"})
    assert legacy_nan.value_type is TrialMetricValueType.NUMBER
    assert isinstance(legacy_nan.value, float) and math.isnan(legacy_nan.value)

    assert TrialMetricValue.model_validate({"trial_id": "t1", "value": 5}).value_type is TrialMetricValueType.NUMBER
    assert TrialMetricValue.model_validate({"trial_id": "t2", "value": None}).value_type is TrialMetricValueType.MISSING
    assert TrialMetricValue.model_validate({"trial_id": "t3", "value": "ok"}).value_type is TrialMetricValueType.LABEL


def test_a_record_cannot_claim_one_kind_and_carry_another() -> None:
    for payload in (
        {"trial_id": "t", "value_type": "number", "value": "good"},
        {"trial_id": "t", "value_type": "label", "value": 1.0},
        {"trial_id": "t", "value_type": "label", "value": None},
        {"trial_id": "t", "value_type": "missing", "value": 1.0},
    ):
        with pytest.raises(ValidationError):
            TrialMetricValue.model_validate(payload)


def test_value_type_is_always_present_in_serialized_output() -> None:
    # Optional for the caller (derived from the value), but never absent from what a reader sees --
    # they must not have to guess whether a string is an escaped float or a label.
    for record in (
        TrialMetricValue(trial_id="t", value=1.0),
        TrialMetricValue(trial_id="t", value="good"),
        TrialMetricValue(trial_id="t", value=None),
        TrialMetricValue(trial_id="t", value=float("nan")),
    ):
        assert "value_type" in record.model_dump(mode="json")


def test_without_tasks_there_is_no_spec_to_filter_on() -> None:
    # No tasks means no declared schemas to consult, so every numeric output observed is retained.
    scores = [_score("task-a", "trial-0", "usage", "prompt_tokens", 1234)]

    summary = AgentEvalSummary.from_scores(scores)

    assert _pairs(summary.task_metric_values) == {"task-a": {"usage.prompt_tokens": [("trial-0", 1234)]}}
    assert isinstance(summary.task_metric_values["task-a"]["usage.prompt_tokens"][0].value, int)


def test_values_align_across_keys_by_trial_id_not_position() -> None:
    # A metric that raised drops its record entirely while a dead trial holds its slot as None, so a
    # metric failure shortens its own key's list without shortening its neighbour's. Index i of two
    # keys is then two different trials -- which is exactly why every record names its trial.
    tasks = [
        _task(
            "task-a",
            _Metric("reward", MetricOutputSpec.continuous_score("score")),
            _Metric("steps", MetricOutputSpec.discrete_score("count")),
        )
    ]
    scores = [
        _score("task-a", "trial-0", "reward", "score", 1.0),
        _score("task-a", "trial-0", "steps", "count", 5),
        _failed_score("task-a", "trial-1", trial_failed=False),  # the reward judge timed out
        _score("task-a", "trial-1", "steps", "count", 9),
        _score("task-a", "trial-2", "reward", "score", 0.0),
        _score("task-a", "trial-2", "steps", "count", 7),
    ]

    records = AgentEvalSummary.from_scores(scores, tasks=tasks).task_metric_values["task-a"]

    # Position lies: index 1 is trial-2 under reward.score but trial-1 under steps.count.
    assert records["reward.score"][1].trial_id == "trial-2"
    assert records["steps.count"][1].trial_id == "trial-1"
    # trial_id tells the truth, so a join across the two keys is now possible and correct.
    steps_by_trial = {a.trial_id: a.value for a in records["steps.count"]}
    assert [(a.trial_id, a.value, steps_by_trial[a.trial_id]) for a in records["reward.score"]] == [
        ("trial-0", 1.0, 5),
        ("trial-2", 0.0, 7),
    ]


def test_undeclared_outputs_are_discovered_only_without_task_specs() -> None:
    tasks = [_task("task-a", _Metric("reward", MetricOutputSpec.continuous_score("score")))]
    scores = [
        _score("task-a", "trial-0", "reward", "score", 1.0),
        _score("task-a", "trial-0", "verdict", "grade", "excellent"),  # no task declares this
    ]

    assert _pairs(AgentEvalSummary.from_scores(scores, tasks=tasks).task_metric_values) == {
        "task-a": {"reward.score": [("trial-0", 1.0)]}
    }
    assert _pairs(AgentEvalSummary.from_scores(scores).task_metric_values) == {
        "task-a": {"reward.score": [("trial-0", 1.0)], "verdict.grade": [("trial-0", "excellent")]}
    }


def test_a_label_under_a_scorelike_key_does_not_break_pass_at_k() -> None:
    # A hand-built score can carry a label under a task-declared score output. It must land in
    # nan_count as an unmeasured task, not raise on `"good" >= 1.0`.
    reward = _Metric("reward", MetricOutputSpec.continuous_score("score"))
    tasks = [_task("scored", reward), _task("labelled", reward)]
    scores = [
        _score("scored", "trial-0", "reward", "score", 1.0),
        _score("labelled", "trial-0", "reward", "score", "good"),  # a label under a scorelike key
    ]

    summary = AgentEvalSummary.from_scores(scores, tasks=tasks)
    by_name = {score.name: score for score in summary.scores.scores}

    assert _pairs(summary.task_metric_values)["labelled"] == {"reward.score": [("trial-0", "good")]}
    assert by_name["reward.score.pass@1"].mean == 1.0  # the one measurable task passed
    assert by_name["reward.score.pass@1"].count == 1
    assert by_name["reward.score.pass@1"].nan_count == 1  # the labelled task is unmeasured, not failed


def test_a_view_over_a_label_output_reduces_to_nothing_rather_than_raising() -> None:
    # Views read MetricOutput through _semantic_value, which still projects to float and drops
    # strings -- so widening the value record cannot leak a label into view arithmetic. This pins
    # that separation; it fails loudly if views are ever rewired onto task_metric_values.
    verdict = _Metric("verdict", MetricOutputSpec.label("grade"))
    task = AgentEvalTask(
        id="task-a",
        intent="test",
        inputs={},
        metrics=[verdict],
        views={
            "quality": SemanticView(
                reducer=SemanticReducer.SINGLE, signals=[ViewSignal(metric="verdict", output="grade")]
            )
        },
    )
    scores = [_score("task-a", "trial-0", "verdict", "grade", "excellent")]

    summary = AgentEvalSummary.from_scores(scores, tasks=[task])
    view = summary.score("view.quality")

    assert view.count == 0 and view.nan_count == 1  # reduced to nothing, no exception
    assert _pairs(summary.task_metric_values) == {"task-a": {"verdict.grade": [("trial-0", "excellent")]}}


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize(
    "reducer",
    [SemanticReducer.MEAN, SemanticReducer.WEIGHTED_MEAN, SemanticReducer.ALL, SemanticReducer.ANY],
)
@pytest.mark.parametrize("reverse_signals", [False, True], ids=["finite-first", "nonfinite-first"])
def test_a_view_rejects_any_nonfinite_signal_before_reduction(
    nonfinite: float,
    reducer: SemanticReducer,
    *,
    reverse_signals: bool,
) -> None:
    metrics: list[Metric] = [
        _Metric("finite", MetricOutputSpec.continuous_score("score")),
        _Metric("nonfinite", MetricOutputSpec.continuous_score("score")),
    ]
    signals = [
        ViewSignal(metric="finite", output="score"),
        ViewSignal(metric="nonfinite", output="score"),
    ]
    task = AgentEvalTask(
        id="task-a",
        intent="test",
        inputs={},
        metrics=metrics,
        views={
            "quality": SemanticView(
                reducer=reducer,
                signals=list(reversed(signals)) if reverse_signals else signals,
            )
        },
    )
    scores = [
        _score("task-a", "trial-0", "finite", "score", 1.0),
        _score("task-a", "trial-0", "nonfinite", "score", nonfinite),
    ]

    view = AgentEvalSummary.from_scores(scores, tasks=[task]).score("view.quality")

    assert (view.count, view.nan_count, view.mean) == (0, 1, None)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
def test_a_single_signal_view_rejects_nonfinite_values(nonfinite: float) -> None:
    metric = _Metric("reward", MetricOutputSpec.continuous_score("score"))
    task = AgentEvalTask(
        id="task-a",
        intent="test",
        inputs={},
        metrics=[metric],
        views={
            "quality": SemanticView(
                reducer=SemanticReducer.SINGLE,
                signals=[ViewSignal(metric="reward", output="score")],
            )
        },
    )

    view = AgentEvalSummary.from_scores(
        [_score("task-a", "trial-0", "reward", "score", nonfinite)], tasks=[task]
    ).score("view.quality")

    assert (view.count, view.nan_count, view.mean) == (0, 1, None)


def test_a_one_signal_view_preserves_duplicate_score_attempts() -> None:
    metric = _Metric("reward", MetricOutputSpec.continuous_score("score"))
    task = AgentEvalTask(
        id="task-a",
        intent="test",
        inputs={},
        metrics=[metric],
        views={
            "quality": SemanticView(
                reducer=SemanticReducer.SINGLE,
                signals=[ViewSignal(metric="reward", output="score")],
            )
        },
    )
    scores = [
        _score("task-a", "duplicate", "reward", "score", 1.0),
        _score("task-a", "duplicate", "reward", "score", 0.0),
    ]

    summary = AgentEvalSummary.from_scores(scores, tasks=[task])
    view = summary.score("view.quality")

    assert (view.count, view.nan_count, view.mean) == (2, 0, 0.5)
    assert summary.score("reward.score").count == 2
    assert summary.metric_coverage["reward"]["score"].total == 2
    assert _pairs(summary.task_metric_values)["task-a"]["reward.score"] == [
        ("duplicate", 1.0),
        ("duplicate", 0.0),
    ]


def test_repeated_view_attempts_pair_signals_by_occurrence_and_retain_absent_signal_opportunities() -> None:
    metrics: list[Metric] = [
        _Metric("a", MetricOutputSpec.continuous_score("score")),
        _Metric("b", MetricOutputSpec.continuous_score("score")),
        _Metric("other", MetricOutputSpec.continuous_score("score")),
    ]
    task = AgentEvalTask(
        id="task-a",
        intent="test",
        inputs={},
        metrics=metrics,
        views={
            "quality": SemanticView(
                reducer=SemanticReducer.MEAN,
                signals=[ViewSignal(metric="a", output="score"), ViewSignal(metric="b", output="score")],
            )
        },
    )
    scores = [
        _score("task-a", "repeated", "a", "score", 1.0),
        _score("task-a", "repeated", "b", "score", 1.0),
        _score("task-a", "repeated", "a", "score", 0.0),
        _score("task-a", "other-only", "other", "score", 1.0),
    ]

    view = AgentEvalSummary.from_scores(scores, tasks=[task]).score("view.quality")

    # repeated occurrence 0 pairs a=1 with b=1; occurrence 1 has no b. The other-only trial creates
    # one more unmeasured opportunity even though neither view signal has a score for it.
    assert (view.count, view.nan_count, view.mean) == (1, 2, 1.0)


def test_dead_trials_are_nameable_from_the_summary_alone() -> None:
    # AALGO-428 needs to say *which* trial died to roll up exception types. Before records carried a
    # trial id the summary could count dead trials but not name one; now it is a lookup key out to
    # trials.jsonl, where the error lives.
    tasks = [_task("task-a", _Metric("reward", MetricOutputSpec.continuous_score("score")))]
    scores = [
        _score("task-a", "trial-0", "reward", "score", 1.0),
        _failed_score("task-a", "trial-1", trial_failed=True),
        _failed_score("task-a", "trial-2", trial_failed=False),  # metric raised: unmeasured, not dead
    ]

    records = AgentEvalSummary.from_scores(scores, tasks=tasks).task_metric_values["task-a"]["reward.score"]

    assert {a.trial_id for a in records if a.value is None} == {"trial-1"}


def test_duplicate_trial_ids_are_two_records_not_one() -> None:
    # Nothing enforces trial-id uniqueness, so the value list must never be re-keyed by trial id:
    # collapsing two records into one would silently drop pass@k's n. A list cannot lose cardinality.
    tasks = [_task("task-a", _Metric("reward", MetricOutputSpec.continuous_score("score")))]
    scores = [
        _score("task-a", "dup", "reward", "score", 1.0),
        _score("task-a", "dup", "reward", "score", 0.0),
    ]

    summary = AgentEvalSummary.from_scores(scores, tasks=tasks)
    by_name = {score.name: score for score in summary.scores.scores}

    assert _pairs(summary.task_metric_values) == {"task-a": {"reward.score": [("dup", 1.0), ("dup", 0.0)]}}
    assert by_name["reward.score.pass@1"].mean == pytest.approx(0.5)  # n=2, not n=1
    assert by_name["reward.score.pass@2"].mean == pytest.approx(1.0)


def test_metric_values_projects_to_a_bare_value_list() -> None:
    # The projection pass@k reads: order, cardinality and None-vs-absent preserved exactly.
    records = [
        TrialMetricValue(trial_id="t0", value=1.0),
        TrialMetricValue(trial_id="t1", value=None),
        TrialMetricValue(trial_id="t2", value=0.0),
    ]

    assert metric_values(records) == [1.0, None, 0.0]

    # A label is preserved faithfully by metric_values and dropped by the arithmetic projection:
    # a categorical verdict is an unmeasured trial, not a failed one.
    with_label = [*records, TrialMetricValue(trial_id="t3", value="excellent")]
    assert metric_values(with_label) == [1.0, None, 0.0, "excellent"]
    assert numeric_metric_values(with_label) == [1.0, None, 0.0]
    assert numeric_metric_values([]) == []
    assert metric_values([]) == []


def test_pass_at_k_aggregates_keep_every_declaring_task_visible_at_each_k() -> None:
    """Golden table for failures, missing measurements, and tasks with no attempts.

    Every branch pass@k distinguishes is present: a task that always passes, one whose trials
    include a dead one (None counts toward ``n``), one whose metric raised on a trial (dropped
    from ``n``, so it falls out of ``k=2``), and two that yielded nothing at all (``nan_count``).
    """
    reward = _Metric("reward", MetricOutputSpec.continuous_score("score"))
    passed = _Metric("complete", MetricOutputSpec.boolean("passed"))
    tasks = [_task(t, reward, passed) for t in ("solved", "flaky", "judged-out", "unmeasured", "never-ran")]
    scores = [
        _score("solved", "s0", "reward", "score", 1.0),
        _score("solved", "s0", "complete", "passed", True),
        _score("solved", "s1", "reward", "score", 1.0),
        _score("solved", "s1", "complete", "passed", True),
        _score("solved", "s2", "reward", "score", 1.0),
        _score("solved", "s2", "complete", "passed", True),
        _score("flaky", "f0", "reward", "score", 1.0),
        _score("flaky", "f0", "complete", "passed", True),
        _failed_score("flaky", "f1", trial_failed=True),
        _failed_score("flaky", "f1", trial_failed=True, metric_type="complete"),
        _score("flaky", "f2", "reward", "score", 0.0),
        _score("flaky", "f2", "complete", "passed", False),
        _score("judged-out", "j0", "reward", "score", 1.0),
        _score("judged-out", "j0", "complete", "passed", True),
        _failed_score("judged-out", "j1", trial_failed=False),
        _failed_score("judged-out", "j1", trial_failed=False, metric_type="complete"),
        _failed_score("unmeasured", "u0", trial_failed=False),
        _failed_score("unmeasured", "u0", trial_failed=False, metric_type="complete"),
    ]

    summary = AgentEvalSummary.from_scores(scores, tasks=tasks)
    actual = {s.name: (s.mean, s.count, s.nan_count) for s in summary.scores.scores if ".pass@" in s.name}

    assert actual == {
        "complete.passed.pass@1": (pytest.approx(0.7777777777777777), 3, 2),
        "complete.passed.pass@2": (pytest.approx(0.8333333333333333), 2, 3),
        "complete.passed.pass@3": (pytest.approx(1.0), 2, 3),
        "reward.score.pass@1": (pytest.approx(0.7777777777777777), 3, 2),
        "reward.score.pass@2": (pytest.approx(0.8333333333333333), 2, 3),
        "reward.score.pass@3": (pytest.approx(1.0), 2, 3),
    }


def test_summary_without_task_metric_values_loads_as_empty() -> None:
    assert AgentEvalSummary.model_validate({}).task_metric_values == {}


def test_vendored_summary_accepts_task_metric_values() -> None:
    VendoredAgentEvalSummary = _vendored_module("results").AgentEvalSummary

    payload = {
        "task_metric_values": {
            "task-a": {"reward.score": [{"trial_id": "t0", "value": 1.0}, {"trial_id": "t1", "value": None}]}
        }
    }

    # The payload deliberately omits value_type, so this doubles as the legacy-derivation regression.
    records = VendoredAgentEvalSummary.model_validate(payload).task_metric_values
    assert [(a.trial_id, a.value) for a in records["task-a"]["reward.score"]] == [("t0", 1.0), ("t1", None)]


def test_vendored_module_exposes_the_public_value_api() -> None:
    # The byte-copy test below proves file parity, not that the names are usable through the shipped
    # package. These are the surface a consumer of nemo-platform actually imports.
    vendored_results = _vendored_module("results")
    VendoredSummary = vendored_results.AgentEvalSummary
    VendoredValue = vendored_results.TrialMetricValue
    VendoredType = vendored_results.TrialMetricValueType
    vendored_numeric = vendored_results.numeric_metric_values

    records = [VendoredValue(trial_id="t0", value=1.0), VendoredValue(trial_id="t1", value="good")]
    assert vendored_numeric(records) == [1.0]  # the label is dropped, as in the source module
    assert records[1].value_type is VendoredType.LABEL

    summary = VendoredSummary(task_metric_values={"task-a": {"reward.score": records}})
    [outcomes] = summary.task_outcomes()
    assert outcomes.task_id == "task-a" and outcomes.outcomes[0].metric_name == "reward.score"


def test_legacy_results_import_resolves_to_the_source_module() -> None:
    # The SDK exposes this legacy path through a runtime alias, not a rewritten copy, so import
    # compatibility should point at the canonical source file.
    import nemo_evaluator_sdk.agent_eval.results as source

    legacy = _vendored_module("results")

    assert Path(legacy.__file__).resolve() == Path(source.__file__).resolve()


def test_gym_example_rejects_a_bundle_written_before_task_metric_values(tmp_path: Path) -> None:
    # The field defaults to empty, so an older bundle would load cleanly and simply show no per-task
    # section -- a reader would take that as "no per-task outcomes" rather than "this script cannot
    # see them". Fail with a version message instead.
    from packages.nemo_evaluator_sdk.examples.gym.inspect_results import BundleFormatError, load_bundle

    (tmp_path / "summary.json").write_text(json.dumps({"task_count": 2}), encoding="utf-8")

    with pytest.raises(BundleFormatError, match="predates summary.task_metric_values"):
        load_bundle(tmp_path)


@pytest.mark.parametrize(
    ("label", "raw"),
    [
        # A root that is not a container: `"task_metric_values" not in 1` raises TypeError.
        ("null root", "null"),
        ("number root", "1"),
        ("bool root", "true"),
        # A root that *is* a container, so the membership test passes and validation is reached --
        # a bare string matches by substring, which is the sharpest edge of the three.
        ("array root", '["task_metric_values"]'),
        ("string root", '"task_metric_values"'),
        # A well-formed object whose field types are wrong, or that carries an extra key
        # (`AgentEvalSummary` is extra="forbid").
        ("wrong field type", '{"task_metric_values": []}'),
        ("unknown field", '{"task_metric_values": {}, "bogus": 1}'),
        ("bad nested record", '{"task_metric_values": {"t": {"m": [{"trial_id": 5, "value": 1.0}]}}}'),
    ],
)
def test_gym_example_reports_a_malformed_bundle_as_a_bundle_format_error(tmp_path: Path, label: str, raw: str) -> None:
    # main() turns BundleFormatError into an exit code and lets everything else become a traceback,
    # so every unreadable shape has to arrive as that one type -- not TypeError from the membership
    # test, and not a raw pydantic ValidationError.
    from packages.nemo_evaluator_sdk.examples.gym.inspect_results import BundleFormatError, load_bundle

    (tmp_path / "summary.json").write_text(raw, encoding="utf-8")

    with pytest.raises(BundleFormatError):
        load_bundle(tmp_path)


def test_summary_task_outcomes_name_their_own_keys() -> None:
    # task_outcomes() lifts the nested dicts into models whose fields are named, for callers that
    # want a typed object to pass around. It is a read-time view: the summary keeps the dict shape.
    summary = AgentEvalSummary(
        task_metric_values={
            "task-b": {"gym_reward.reward": [TrialMetricValue(trial_id="task-b__bbb", value=0.0)]},
            "task-a": {
                "steps.count": [TrialMetricValue(trial_id="task-a__aaa", value=5)],
                "gym_reward.reward": [TrialMetricValue(trial_id="task-a__aaa", value=1.0)],
            },
        }
    )

    outcomes = summary.task_outcomes()

    assert [o.task_id for o in outcomes] == ["task-a", "task-b"]  # sorted by task
    assert [o.metric_name for o in outcomes[0].outcomes] == ["gym_reward.reward", "steps.count"]  # then metric
    assert [(t.trial_id, t.value) for t in outcomes[0].outcomes[0].trials] == [("task-a__aaa", 1.0)]


def _mixed_metric_summary() -> AgentEvalSummary:
    """One task measured by the metric, one that declared it but yielded nothing, one scored by another."""
    return AgentEvalSummary(
        task_metric_values={
            "task-a": {
                "gym_reward.reward": [
                    TrialMetricValue(trial_id="task-a__aaa", value=1.0),
                    TrialMetricValue(trial_id="task-a__bbb", value=0.0),
                ],
                "steps.count": [TrialMetricValue(trial_id="task-a__aaa", value=7)],
            },
            "task-b": {"gym_reward.reward": []},
            "task-c": {"judge.rating": [TrialMetricValue(trial_id="task-c__ccc", value=0.5)]},
        }
    )


def test_task_outcomes_narrows_to_one_metric() -> None:
    # The filter is what makes the typed view usable for a single-metric report; without it a caller
    # has to re-filter the nested lists by hand, which is what the dict shape already made them do.
    outcomes = _mixed_metric_summary().task_outcomes("gym_reward.reward")

    assert [o.task_id for o in outcomes] == ["task-a", "task-b"]
    assert [[oc.metric_name for oc in o.outcomes] for o in outcomes] == [["gym_reward.reward"], ["gym_reward.reward"]]
    assert [(t.trial_id, t.value) for t in outcomes[0].outcomes[0].trials] == [
        ("task-a__aaa", 1.0),
        ("task-a__bbb", 0.0),
    ]


def test_task_outcomes_drops_a_task_the_metric_never_measured_but_keeps_an_empty_one() -> None:
    # The two states are not the same and must not be flattened together. task-b declared the metric
    # and yielded no usable value -- that is missing coverage, so it stays with an empty trials list
    # and reports as unmeasured. task-c was scored by a different metric entirely; reporting it would
    # invent coverage the run never asked for.
    outcomes = _mixed_metric_summary().task_outcomes("gym_reward.reward")

    by_task = {o.task_id: o for o in outcomes}
    assert "task-c" not in by_task
    assert by_task["task-b"].outcomes[0].trials == []

    # Unfiltered, every task is present including the one scored by another metric.
    assert [o.task_id for o in _mixed_metric_summary().task_outcomes()] == ["task-a", "task-b", "task-c"]


def test_gym_example_reads_task_outcomes_from_summary() -> None:
    # The example reads the SDK's typed view rather than re-deriving one: each row names its own
    # task and metric, and each trial the rollout that produced it.
    from packages.nemo_evaluator_sdk.examples.gym.inspect_results import show_per_task

    outcomes = _mixed_metric_summary().task_outcomes("gym_reward.reward")

    assert [(o.task_id, o.outcomes[0].metric_name) for o in outcomes] == [
        ("task-a", "gym_reward.reward"),
        ("task-b", "gym_reward.reward"),
    ]
    # The display path is what the projection to floats exists for; it must not raise on an empty
    # outcome, which is the task that reports as unmeasured.
    show_per_task(outcomes)
