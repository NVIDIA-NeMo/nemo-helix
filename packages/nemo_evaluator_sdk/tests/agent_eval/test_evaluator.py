# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from nemo_evaluator_sdk.agent_eval.evaluator import (
    AgentEvaluator,
    _live_response_usage_measurements,
    _metric_row,
    _new_run_id,
    _task_row,
    _trial_from_sample,
    _trial_sample,
    _warn_invalid_live_usage_counts,
)
from nemo_evaluator_sdk.agent_eval.results import AgentEvalSummary
from nemo_evaluator_sdk.agent_eval.scores import (
    AgentEvalDiagnosticSeverity,
    AgentEvalScoreStatus,
    AgentEvalTaskScore,
)
from nemo_evaluator_sdk.agent_eval.tasks import (
    AgentEvalRunConfig,
    AgentEvalTask,
    SemanticReducer,
    SemanticView,
    ViewSignal,
)
from nemo_evaluator_sdk.agent_eval.trials import (
    AgentEvalTrial,
    AgentEvalTrialStatus,
    AgentOutput,
    RunnerInfo,
    TrialError,
    TrialMeasurements,
)
from nemo_evaluator_sdk.agent_inference import AgentInferenceContext, AgentInvocationResult, AgentInvocationStatus
from nemo_evaluator_sdk.enums import AgentFormat, ModelFormat
from nemo_evaluator_sdk.metrics.protocol import (
    Metric,
    MetricDiagnostic,
    MetricInput,
    MetricOutput,
    MetricOutputSpec,
    MetricResult,
)
from nemo_evaluator_sdk.values import (
    Agent,
    GenericAgent,
    Model,
    NemoAgentToolkitAgent,
    RunConfigOnline,
    RunConfigOnlineModel,
)
from nemo_evaluator_sdk.values.evidence import CandidateEvidence, EvidenceDescriptor
from pydantic import ValidationError


def test_trial_from_sample_falls_back_to_reasoning_content() -> None:
    task = AgentEvalTask(id="task-1", intent="Answer.", inputs={"instruction": "Q?"})
    target = Model(name="reasoning-model", url="https://example/v1/chat/completions")
    response = {"choices": [{"message": {"content": None, "reasoning_content": "the reasoned answer"}}]}

    # Empty content falls back to reasoning_content; explicit content wins when present.
    fallback = _trial_from_sample(task, target, {"output_text": "  ", "response": response})
    assert fallback.output is not None
    assert fallback.output.output_text == "the reasoned answer"

    explicit = _trial_from_sample(task, target, {"output_text": "final answer", "response": response})
    assert explicit.output is not None
    assert explicit.output.output_text == "final answer"


def test_trial_from_sample_fallback_trace_has_trace_kind() -> None:
    task = AgentEvalTask(id="task-1", intent="Answer.", inputs={"instruction": "Q?"})
    target = Model(name="target", url="https://example/v1/chat/completions")

    trial = _trial_from_sample(task, target, {"output_text": "answer"})

    assert trial.evidence is not None
    trace = trial.evidence.require("trace", kind="trace")
    assert trace.format == "json"


def test_trial_from_sample_preserves_typed_trace_and_canonical_metadata() -> None:
    task = AgentEvalTask(id="task-1", intent="Answer.", inputs={"instruction": "Q?"})
    target = Model(name="canonical-target", url="https://example/v1/chat/completions")
    typed_trace = {
        "schema_version": "ATIF-v1.7",
        "steps": [{"source": "user", "message": "Q?"}],
    }

    trial = _trial_from_sample(
        task,
        target,
        {
            "output_text": "answer",
            "trajectory": [{"legacy": True}],
            "evidence": CandidateEvidence(
                descriptors={
                    "trace": EvidenceDescriptor(
                        kind="trace",
                        format="atif",
                        data=typed_trace,
                    )
                }
            ),
            "invocation_metadata": {
                "endpoint": "/generate/stream",
                "model_id": "spoofed-model",
                "target_name": "spoofed-target",
                "generated": False,
            },
        },
    )

    assert trial.evidence is not None
    trace = trial.evidence.require("trace", kind="trace")
    assert trace.format == "atif"
    assert trace.data == typed_trace
    assert trial.metadata["endpoint"] == "/generate/stream"
    assert trial.metadata["model_id"] == "canonical-target"
    assert trial.metadata["target_name"] == "canonical-target"
    assert trial.metadata["generated"] is True


@pytest.mark.parametrize(
    ("target", "usage", "expected"),
    [
        pytest.param(
            Model(name="chat-model", url="https://example/v1/chat/completions"),
            {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 999,
                "prompt_tokens_details": {"cached_tokens": 3},
            },
            TrialMeasurements(prompt_tokens=10, completion_tokens=2, cache_read_tokens=3),
            id="openai-chat",
        ),
        pytest.param(
            Model(name="responses-model", url="https://example/v1/responses"),
            {
                "input_tokens": 8,
                "output_tokens": 4,
                "input_tokens_details": {"cached_tokens": 2},
            },
            TrialMeasurements(prompt_tokens=8, completion_tokens=4, cache_read_tokens=2),
            id="openai-responses",
        ),
        pytest.param(
            GenericAgent(
                name="anthropic-agent",
                url="https://example/agent",
                format=AgentFormat.GENERIC,
                body={"input": "{{ instruction }}"},
                response_path="$.answer",
            ),
            {
                "input_tokens": 5,
                "output_tokens": 2,
                "cache_creation_input_tokens": 1,
                "cache_read_input_tokens": 4,
            },
            TrialMeasurements(
                prompt_tokens=10,
                completion_tokens=2,
                cache_creation_tokens=1,
                cache_read_tokens=4,
            ),
            id="anthropic-agent",
        ),
    ],
)
def test_trial_from_sample_normalizes_live_usage(
    target: Model | Agent,
    usage: dict[str, Any],
    expected: TrialMeasurements,
) -> None:
    task = AgentEvalTask(id="task-1", intent="Answer.", inputs={"instruction": "Q?"})

    trial = _trial_from_sample(
        task,
        target,
        {"output_text": "answer", "response": {"usage": usage}},
    )

    assert trial.measurements == expected
    assert trial.measurements.total_tokens == expected.total_tokens
    assert trial.measurements.runtime_sec is None
    assert trial.measurements.cost_usd is None


def test_trial_from_sample_omits_ambiguous_prompt_cache_usage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    task = AgentEvalTask(id="task-1", intent="Answer.", inputs={"instruction": "Q?"})
    target = Model(name="target", url="https://example/v1/responses")

    trial = _trial_from_sample(
        task,
        target,
        {
            "output_text": "answer",
            "response": {
                "usage": {
                    "input_tokens": 8,
                    "output_tokens": 4,
                    "input_tokens_details": {"cached_tokens": 2},
                    "cache_read_input_tokens": 2,
                }
            },
        },
    )

    assert trial.measurements == TrialMeasurements(completion_tokens=4)
    assert "both inclusive cache details and separate cache fields" in caplog.text


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        pytest.param(
            {
                "input_tokens": 5,
                "output_tokens": 2,
                "input_tokens_details": {"cached_tokens": None},
                "cache_read_input_tokens": 4,
            },
            TrialMeasurements(prompt_tokens=9, completion_tokens=2, cache_read_tokens=4),
            id="null-inclusive-placeholder",
        ),
        pytest.param(
            {
                "input_tokens": 5,
                "output_tokens": 2,
                "input_tokens_details": {"cached_tokens": 4},
                "cache_read_input_tokens": None,
            },
            TrialMeasurements(prompt_tokens=5, completion_tokens=2, cache_read_tokens=4),
            id="null-separate-placeholder",
        ),
    ],
)
def test_live_response_usage_ignores_null_cache_format_placeholders(
    usage: dict[str, Any],
    expected: TrialMeasurements,
) -> None:
    assert _live_response_usage_measurements({"usage": usage}, trial_id="trial-1") == expected


def test_trial_from_sample_warns_and_preserves_independent_valid_usage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    task = AgentEvalTask(id="task-1", intent="Answer.", inputs={"instruction": "Q?"})
    target = Model(name="target", url="https://example/v1/chat/completions")

    trial = _trial_from_sample(
        task,
        target,
        {
            "output_text": "answer",
            "response": {
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": "bad",
                    "prompt_tokens_details": {"cached_tokens": -1},
                }
            },
        },
    )

    assert trial.measurements == TrialMeasurements(prompt_tokens=8)
    assert "completion_tokens='bad'" in caplog.text
    assert "prompt_tokens_details.cached_tokens=-1" in caplog.text


def test_live_response_usage_measurements_empty_when_usage_is_absent() -> None:
    assert _live_response_usage_measurements(None, trial_id="trial-1") == TrialMeasurements()
    assert _live_response_usage_measurements("not-a-mapping", trial_id="trial-1") == TrialMeasurements()
    assert _live_response_usage_measurements({}, trial_id="trial-1") == TrialMeasurements()
    assert _live_response_usage_measurements({"usage": None}, trial_id="trial-1") == TrialMeasurements()


def test_live_response_usage_measurements_omits_non_object_usage(caplog: pytest.LogCaptureFixture) -> None:
    assert _live_response_usage_measurements({"usage": "tokens"}, trial_id="trial-1") == TrialMeasurements()
    assert "usage='tokens'" in caplog.text


def test_live_response_usage_measurements_prefers_chat_aliases() -> None:
    measurements = _live_response_usage_measurements(
        {
            "usage": {
                "prompt_tokens": 10,
                "input_tokens": 1,
                "completion_tokens": 2,
                "output_tokens": 9,
                "total_tokens": 999,
            }
        },
        trial_id="trial-1",
    )

    assert measurements == TrialMeasurements(prompt_tokens=10, completion_tokens=2)


def test_live_response_usage_measurements_records_zero_usage() -> None:
    measurements = _live_response_usage_measurements(
        {"usage": {"prompt_tokens": 0, "completion_tokens": 0}},
        trial_id="trial-1",
    )

    assert measurements.prompt_tokens == 0
    assert measurements.completion_tokens == 0
    assert measurements.total_tokens == 0


def test_invalid_separate_cache_poisons_prompt_only(caplog: pytest.LogCaptureFixture) -> None:
    measurements = _live_response_usage_measurements(
        {"usage": {"input_tokens": 5, "output_tokens": 2, "cache_read_input_tokens": "bad"}},
        trial_id="trial-1",
    )

    assert measurements == TrialMeasurements(completion_tokens=2)
    assert "cache_read_input_tokens='bad'" in caplog.text


def test_separate_cache_without_prompt_does_not_invent_a_prompt_total() -> None:
    measurements = _live_response_usage_measurements(
        {"usage": {"output_tokens": 2, "cache_read_input_tokens": 4}},
        trial_id="trial-1",
    )

    assert measurements == TrialMeasurements(completion_tokens=2, cache_read_tokens=4)


def test_inclusive_cache_falls_back_from_chat_details_to_responses_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    measurements = _live_response_usage_measurements(
        {
            "usage": {
                "input_tokens": 8,
                "output_tokens": 1,
                "prompt_tokens_details": "not-an-object",
                "input_tokens_details": {"cached_tokens": 3},
            }
        },
        trial_id="trial-1",
    )

    assert measurements == TrialMeasurements(prompt_tokens=8, completion_tokens=1, cache_read_tokens=3)
    assert "prompt_tokens_details='not-an-object'" in caplog.text


def test_warn_invalid_live_usage_counts_skips_missing_and_null_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    usage = {"prompt_tokens": 8, "completion_tokens": None}

    _warn_invalid_live_usage_counts("trial-1", usage)

    assert caplog.text == ""
    assert usage == {"prompt_tokens": 8, "completion_tokens": None}


def test_warn_invalid_live_usage_counts_logs_malformed_fields(caplog: pytest.LogCaptureFixture) -> None:
    _warn_invalid_live_usage_counts(
        "trial-1",
        {
            "prompt_tokens": True,
            "prompt_tokens_details": [1],
            "input_tokens_details": {"cached_tokens": -1},
        },
    )

    assert "prompt_tokens=True" in caplog.text
    assert "prompt_tokens_details=[1]" in caplog.text
    assert "input_tokens_details.cached_tokens=-1" in caplog.text


def test_generated_run_ids_are_unique_within_the_same_second() -> None:
    with patch("nemo_evaluator_sdk.agent_eval.evaluator.datetime") as mock_datetime:
        mock_datetime.now.return_value.strftime.return_value = "20260628120000"

        first = _new_run_id()
        second = _new_run_id()

    assert first != second
    assert first.startswith("agent-eval-20260628120000-")


def test_metric_row_exposes_reference_but_task_row_hides_it() -> None:
    # ``reference`` is grader-only held-out ground truth: metrics must see it, the agent (via the
    # generation ``_task_row``) must not.
    task = AgentEvalTask(
        id="task-1",
        intent="Fix the bug.",
        inputs={"instruction": "Fix calculator.py."},
        reference={"test_calculator.py": "def test_add(): assert add(2, 3) == 5"},
    )
    trial = _candidate_trial()

    metric_row = _metric_row(task, trial)
    assert metric_row["reference"] == {"test_calculator.py": "def test_add(): assert add(2, 3) == 5"}

    task_row = _task_row(task)
    assert "reference" not in task_row


def test_metric_row_keeps_measurements_separate_from_trial_metadata() -> None:
    task = AgentEvalTask(id="task-1", intent="Fix it.", inputs={})
    trial = AgentEvalTrial(
        id="trial-1",
        task_id=task.id,
        status=AgentEvalTrialStatus.PARTIAL,
        output=None,
        measurements=TrialMeasurements(prompt_tokens=8, completion_tokens=2, runtime_sec=1.5),
        metadata={"prompt_tokens": 999, "provenance": "kept"},
    )

    assert _trial_sample(trial) == {}
    row = _metric_row(task, trial)
    assert row["trial"]["measurements"] == trial.measurements.model_dump(mode="json")
    assert row["trial"]["metadata"] == {"prompt_tokens": 999, "provenance": "kept"}
    assert trial.metadata == {"prompt_tokens": 999, "provenance": "kept"}


def test_metric_row_measurements_dump_does_not_alias_durable_trial_state() -> None:
    trial = AgentEvalTrial(
        id="trial-1",
        task_id="task-1",
        status=AgentEvalTrialStatus.PARTIAL,
        measurements=TrialMeasurements(prompt_tokens=8, completion_tokens=2),
    )
    original = trial.measurements.model_copy()

    row = _metric_row(AgentEvalTask(id="task-1", intent="Fix it.", inputs={}), trial)
    row["trial"]["measurements"]["prompt_tokens"] = -1

    assert trial.measurements == original


def test_metric_row_metadata_does_not_alias_durable_trial_state() -> None:
    trial = AgentEvalTrial(
        id="trial-1",
        task_id="task-1",
        status=AgentEvalTrialStatus.PARTIAL,
        metadata={"provenance": "kept"},
    )

    row = _metric_row(AgentEvalTask(id="task-1", intent="Fix it.", inputs={}), trial)
    row["trial"]["metadata"]["provenance"] = "changed"

    assert trial.metadata == {"provenance": "kept"}


class _ConstantMetric:
    @property
    def type(self) -> str:
        return "constant_metric"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("score")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        return MetricResult(outputs=[MetricOutput(name="score", value=0.75)])


class _MeasurementCaptureMetric(_ConstantMetric):
    def __init__(self) -> None:
        self.input: MetricInput | None = None

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        self.input = input
        return await super().compute_scores(input)


class _EvidenceMetric:
    def __init__(self) -> None:
        self.inputs: list[MetricInput] = []

    @property
    def type(self) -> str:
        return "evidence_metric"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("score")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        self.inputs.append(input)
        return MetricResult(outputs=[MetricOutput(name="score", value=1.0)])


class _OtherMetric:
    @property
    def type(self) -> str:
        return "other_metric"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("quality")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        return MetricResult(outputs=[MetricOutput(name="quality", value=0.25)])


class _FailingMetric:
    @property
    def type(self) -> str:
        return "failing_metric"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("score")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        raise RuntimeError("missing final_state evidence")


class _DiagnosticMetric:
    @property
    def type(self) -> str:
        return "diagnostic_metric"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("score")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        return MetricResult(
            outputs=[MetricOutput(name="score", value=0.5)],
            diagnostics=[
                MetricDiagnostic(message="criterion C-001 passed", details={"verdict": "pass", "id": "C-001"})
            ],
        )


def _task(metric: Any | None = None, *, task_id: str = "task-1") -> AgentEvalTask:
    return AgentEvalTask(
        id=task_id,
        intent="Answer a professional benchmark prompt.",
        inputs={"instruction": "What is the answer?", "domain": "Finance MBA"},
        metrics=[metric or _ConstantMetric()],
        metadata={"benchmark": "Example", "domain": "Finance MBA"},
    )


def _candidate_trial() -> AgentEvalTrial:
    return AgentEvalTrial(
        id="trial-1",
        task_id="task-1",
        status=AgentEvalTrialStatus.COMPLETED,
        output=AgentOutput(output_text="Candidate answer"),
        metadata={"model_id": "candidate"},
    )


def _task_score(
    run_id: str,
    task_id: str,
    trial_id: str,
    metric_type: str,
    output_name: str,
    output_value: float,
) -> AgentEvalTaskScore:
    return AgentEvalTaskScore(
        id=f"{run_id}:{task_id}:{trial_id}:{metric_type}",
        run_id=run_id,
        task_id=task_id,
        trial_id=trial_id,
        metric_type=metric_type,
        status=AgentEvalScoreStatus.COMPLETED,
        outputs=[MetricOutput(name=output_name, value=output_value)],
    )


class _TaskRunner:
    def __init__(self) -> None:
        self.config: AgentEvalRunConfig | None = None

    def runner_info(self) -> RunnerInfo:
        return RunnerInfo(name="test_runner", kind="runner")

    async def run_tasks(
        self,
        tasks: Sequence[AgentEvalTask],
        config: AgentEvalRunConfig | None = None,
    ) -> list[AgentEvalTrial]:
        self.config = config
        return [
            AgentEvalTrial(
                id=f"{task.id}:runtime",
                task_id=task.id,
                status=AgentEvalTrialStatus.COMPLETED,
                output=AgentOutput(output_text="Runtime answer"),
                metadata={"model_id": "runtime"},
            )
            for task in tasks
        ]


class _FinalizingTaskRunner(_TaskRunner):
    def __init__(self, replacement: Any, events: list[str]) -> None:
        super().__init__()
        self.replacement = replacement
        self.events = events

    async def run_tasks(
        self,
        tasks: Sequence[AgentEvalTask],
        config: AgentEvalRunConfig | None = None,
    ) -> list[AgentEvalTrial]:
        self.events.append("run_tasks")
        return await super().run_tasks(tasks, config)

    def scoring_metrics(
        self,
        task: AgentEvalTask,
        trials: Sequence[AgentEvalTrial],
    ) -> Sequence[Metric]:
        self.events.append(f"finalize:{len(trials)}")
        return [self.replacement]


class _OrderingMetric(_ConstantMetric):
    def __init__(self, events: list[str]) -> None:
        self.events = events

    @property
    def type(self) -> str:
        return "finalized_metric"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("finalized_score")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        self.events.append("score")
        return MetricResult(outputs=[MetricOutput(name="finalized_score", value=0.75)])


class _MutableMetadataBox:
    """Arbitrary mutable metadata leaf with identity-only equality."""

    def __init__(self, value: str) -> None:
        self.value = value


@pytest.mark.asyncio
async def test_target_can_finalize_scoring_tasks_after_trials_and_before_scoring() -> None:
    events: list[str] = []
    metric = _OrderingMetric(events)
    box = _MutableMetadataBox("original")
    task = _task().model_copy(update={"metadata": {"box": box}})

    result = await AgentEvaluator().run(tasks=[task], target=_FinalizingTaskRunner(metric, events))

    assert events == ["run_tasks", "finalize:1", "score"]
    assert result.tasks[0].metrics == [metric]
    assert result.tasks[0].metadata["box"] is box
    assert "finalized_metric.finalized_score" in result.summary.scores.scores_by_name
    assert "constant_metric.score" not in result.summary.scores.scores_by_name


@pytest.mark.asyncio
async def test_non_provider_target_keeps_original_scoring_tasks() -> None:
    task = _task()

    result = await AgentEvaluator().run(tasks=[task], target=_TaskRunner())

    assert result.tasks == [task]


class _DuplicatingTaskRunner(_TaskRunner):
    def scoring_metrics(
        self,
        task: AgentEvalTask,
        trials: Sequence[AgentEvalTrial],
    ) -> Sequence[Metric]:
        return [_ConstantMetric(), _ConstantMetric()]


@pytest.mark.asyncio
async def test_finalized_tasks_are_revalidated_for_duplicate_metric_types() -> None:
    with pytest.raises(ValueError, match="duplicate task metric types"):
        await AgentEvaluator().run(tasks=[_task()], target=_DuplicatingTaskRunner())


@pytest.mark.asyncio
async def test_finalized_tasks_are_revalidated_for_view_references() -> None:
    events: list[str] = []
    task = _task().model_copy(
        update={
            "views": {
                "quality": SemanticView(
                    reducer=SemanticReducer.MEAN,
                    signals=[ViewSignal(metric="constant_metric", output="score")],
                )
            }
        }
    )

    # The replacement metric's type is "finalized_metric", so the view's signal no longer resolves.
    with pytest.raises(ValueError, match="references unknown"):
        await AgentEvaluator().run(tasks=[task], target=_FinalizingTaskRunner(_OrderingMetric(events), events))


class _TrialMovingTaskRunner(_TaskRunner):
    """Breaks the no-mutation convention by reassigning a trial to another task."""

    def scoring_metrics(
        self,
        task: AgentEvalTask,
        trials: Sequence[AgentEvalTrial],
    ) -> Sequence[Metric]:
        if task.id == "a":
            trials[0].task_id = "b"
        return list(task.metrics)


class _TrialSwappingTaskRunner(_TaskRunner):
    """Breaks the no-mutation convention while preserving one trial per task."""

    def __init__(self) -> None:
        super().__init__()
        self.trials_by_original_task: dict[str, AgentEvalTrial] = {}

    async def run_tasks(
        self,
        tasks: Sequence[AgentEvalTask],
        config: AgentEvalRunConfig | None = None,
    ) -> list[AgentEvalTrial]:
        trials = await super().run_tasks(tasks, config)
        self.trials_by_original_task = {trial.task_id: trial for trial in trials}
        return trials

    def scoring_metrics(
        self,
        task: AgentEvalTask,
        trials: Sequence[AgentEvalTrial],
    ) -> Sequence[Metric]:
        if task.id == "a":
            trial_a = self.trials_by_original_task["a"]
            trial_b = self.trials_by_original_task["b"]
            trial_a.task_id, trial_b.task_id = trial_b.task_id, trial_a.task_id
        return list(task.metrics)


@pytest.mark.asyncio
async def test_scoring_rejects_moved_trial_assignment() -> None:
    with pytest.raises(
        ValueError,
        match=r"must not mutate trials \(trial 'a:runtime' task_id 'a' -> 'b'\)",
    ):
        await AgentEvaluator().run(
            tasks=[_task(task_id="a"), _task(task_id="b")],
            target=_TrialMovingTaskRunner(),
        )


@pytest.mark.asyncio
async def test_scoring_rejects_balanced_trial_assignment_swap() -> None:
    with pytest.raises(
        ValueError,
        match=r"must not mutate trials \(trial 'a:runtime' task_id 'a' -> 'b'; trial 'b:runtime' task_id 'b' -> 'a'\)",
    ):
        await AgentEvaluator().run(
            tasks=[_task(task_id="a"), _task(task_id="b")],
            target=_TrialSwappingTaskRunner(),
        )


def test_run_rejects_trials_and_target_together() -> None:
    model = Model(url="https://model.test/v1/chat/completions", name="target", format=ModelFormat.OPEN_AI)

    with pytest.raises(ValueError, match="provide exactly one"):
        AgentEvaluator().run_sync(
            tasks=[_task()],
            trials=[_candidate_trial()],
            target=model,
        )


def test_run_rejects_neither_trials_nor_target() -> None:
    with pytest.raises(ValueError, match="provide exactly one"):
        AgentEvaluator().run_sync(tasks=[_task()])


@pytest.mark.asyncio
async def test_run_writes_nothing_until_persist_is_called(tmp_path: Path) -> None:
    # The point of the change: computing an evaluation and storing one are separate decisions, so a
    # run given a work_dir still leaves it empty until the caller asks for a bundle.
    result = await AgentEvaluator().run(
        tasks=[_task()],
        trials=[_candidate_trial()],
        config=AgentEvalRunConfig(work_dir=tmp_path, parallelism=1),
    )

    assert not (tmp_path / "run.json").exists()
    assert not (tmp_path / "report.html").exists()
    # work_dir comes from the config, so it is known at construction — never patched on afterwards.
    assert result.work_dir == tmp_path

    result.persist()
    assert (tmp_path / "run.json").is_file()


@pytest.mark.asyncio
async def test_scores_imported_trials_with_metric_and_persists_bundle(tmp_path: Path) -> None:
    result = await AgentEvaluator().run(
        tasks=[_task()],
        trials=[_candidate_trial()],
        config=AgentEvalRunConfig(work_dir=tmp_path, parallelism=1),
    )
    # run() no longer writes anything; persisting is the caller's call and defaults to the work_dir.
    location = result.persist()

    assert result.summary.score("constant_metric.score").mean == 0.75
    assert location.output_dir == tmp_path
    assert location.dashboard_path == tmp_path / "report.html"
    assert (tmp_path / "run.json").exists()
    assert (tmp_path / "scores.jsonl").exists()
    assert "run_id" not in json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

    score_payload = json.loads((tmp_path / "scores.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert score_payload["id"] == f"{result.run_id}:task-1:trial-1:constant_metric"
    assert score_payload["run_id"] == result.run_id
    assert score_payload["status"] == "completed"
    assert score_payload["diagnostics"] == []

    run_payload = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert run_payload == {
        "artifacts": {
            "metadata": "metadata.json",
            "scores": "scores.jsonl",
            "summary": "summary.json",
            "tasks": "tasks.jsonl",
            "trials": "trials.jsonl",
        },
        "dashboard_path": str(tmp_path / "report.html"),
        "output_dir": str(tmp_path),
        "run_id": result.run_id,
    }
    assert result.scores[0].metric_type == "constant_metric"
    assert result.scores[0].outputs[0].value == 0.75


@pytest.mark.asyncio
async def test_scores_partial_trials() -> None:
    result = await AgentEvaluator().run(
        tasks=[_task()],
        trials=[
            AgentEvalTrial(
                id="trial-1",
                task_id="task-1",
                status=AgentEvalTrialStatus.PARTIAL,
                output=AgentOutput(output_text="Partial answer"),
            )
        ],
    )

    assert result.summary.score("constant_metric.score").mean == 0.75


@pytest.mark.parametrize(
    ("measurements", "output", "expected_candidate_metadata"),
    [
        pytest.param(
            TrialMeasurements(prompt_tokens=8, completion_tokens=2, runtime_sec=1.5),
            AgentOutput(output_text="answer", response={"answer": 42}, metadata={"shared": "output"}),
            {"provenance": "kept", "shared": "output"},
            id="populated-output",
        ),
        pytest.param(
            TrialMeasurements(),
            None,
            {},
            id="empty-outputless",
        ),
        pytest.param(
            TrialMeasurements(prompt_tokens=0, completion_tokens=0, runtime_sec=0.0, cost_usd=0.0),
            AgentOutput(output_text="zero", metadata={"shared": "output"}),
            {"provenance": "kept", "shared": "output"},
            id="zero-output",
        ),
    ],
)
@pytest.mark.asyncio
async def test_metric_input_uses_only_canonical_trial_measurements(
    measurements: TrialMeasurements,
    output: AgentOutput | None,
    expected_candidate_metadata: dict[str, Any],
) -> None:
    metric = _MeasurementCaptureMetric()
    task = AgentEvalTask(id="task-1", intent="Answer.", inputs={}, metrics=[metric])
    evidence = (
        CandidateEvidence(descriptors={"trace": EvidenceDescriptor(kind="trace", format="json", data={})})
        if output is not None
        else None
    )
    trial = AgentEvalTrial(
        id="trial-1",
        task_id=task.id,
        status=AgentEvalTrialStatus.PARTIAL,
        output=output,
        evidence=evidence,
        measurements=measurements,
        metadata={"provenance": "kept", "shared": "trial"},
    )
    original_trial = trial.model_copy(deep=True)

    result = await AgentEvaluator().run(tasks=[task], trials=[trial])

    assert result.scores[0].status == AgentEvalScoreStatus.COMPLETED
    assert metric.input is not None
    assert metric.input.row.data["trial"]["measurements"] == trial.measurements.model_dump(mode="json")
    assert metric.input.row.data["trial"]["metadata"] == trial.metadata
    assert metric.input.candidate.metadata == expected_candidate_metadata
    assert metric.input.candidate.output_text == (output.output_text if output is not None else None)
    assert metric.input.candidate.response == (output.response if output is not None else None)
    assert metric.input.candidate.evidence == evidence
    assert trial == original_trial


@pytest.mark.asyncio
async def test_measurement_named_metadata_remains_opaque_in_metric_input() -> None:
    metric = _MeasurementCaptureMetric()
    task = AgentEvalTask(id="task-1", intent="Answer.", inputs={}, metrics=[metric])
    trial = AgentEvalTrial(
        id="trial-1",
        task_id=task.id,
        status=AgentEvalTrialStatus.PARTIAL,
        output=AgentOutput(output_text="answer"),
        measurements=TrialMeasurements(prompt_tokens=8),
        metadata={"prompt_tokens": 999},
    )

    await AgentEvaluator().run(tasks=[task], trials=[trial])

    assert metric.input is not None
    assert metric.input.candidate.metadata["prompt_tokens"] == 999
    assert metric.input.row.data["trial"]["metadata"]["prompt_tokens"] == 999
    assert metric.input.row.data["trial"]["measurements"]["prompt_tokens"] == 8


@pytest.mark.asyncio
async def test_run_revalidates_model_copy_corrupted_measurements_before_scoring() -> None:
    metric = _MeasurementCaptureMetric()
    task = AgentEvalTask(id="task-1", intent="Answer.", inputs={}, metrics=[metric])
    trial = AgentEvalTrial(
        id="trial-1",
        task_id=task.id,
        status=AgentEvalTrialStatus.PARTIAL,
        measurements=TrialMeasurements(prompt_tokens=8, completion_tokens=2),
    )
    corrupted_measurements = trial.measurements.model_copy(update={"prompt_tokens": -1})
    corrupted_trial = trial.model_copy(update={"measurements": corrupted_measurements})

    with pytest.raises(ValidationError):
        await AgentEvaluator().run(tasks=[task], trials=[corrupted_trial])

    assert metric.input is None


@pytest.mark.asyncio
async def test_run_uses_revalidated_model_copy_measurements_when_scoring() -> None:
    metric = _MeasurementCaptureMetric()
    task = AgentEvalTask(id="task-1", intent="Answer.", inputs={}, metrics=[metric])
    trial = AgentEvalTrial(
        id="trial-1",
        task_id=task.id,
        status=AgentEvalTrialStatus.PARTIAL,
    ).model_copy(update={"measurements": {"prompt_tokens": 8, "completion_tokens": 2}})

    result = await AgentEvaluator().run(tasks=[task], trials=[trial])

    assert result.scores[0].status == AgentEvalScoreStatus.COMPLETED
    assert metric.input is not None
    assert metric.input.row.data["trial"]["measurements"] == TrialMeasurements(
        prompt_tokens=8, completion_tokens=2
    ).model_dump(mode="json")
    assert result.trials[0].measurements == TrialMeasurements(prompt_tokens=8, completion_tokens=2)


@pytest.mark.asyncio
async def test_target_runtime_produces_trials_before_scoring() -> None:
    runtime = _TaskRunner()
    result = await AgentEvaluator().run(
        tasks=[_task()],
        target=runtime,
    )

    assert result.trials[0].id == "task-1:runtime"
    assert runtime.config is not None
    assert runtime.config.run_id == result.run_id
    assert result.summary.score("constant_metric.score").mean == 0.75


@pytest.mark.asyncio
async def test_live_model_generation_with_mocked_inference() -> None:
    async def fake_model_inference(
        model: Model,
        request: dict[str, Any],
        max_retries: int | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del model, max_retries, kwargs
        assert request["messages"][0]["content"] == "What is the answer?"
        assert "prompt" not in request
        return {
            "choices": [{"message": {"role": "assistant", "content": "Generated model answer"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 3},
            },
        }

    model = Model(url="https://model.test/v1/chat/completions", name="target-model", format=ModelFormat.OPEN_AI)
    result = await AgentEvaluator(inference_fn=fake_model_inference).run(
        tasks=[_task()],
        target=model,
        config=AgentEvalRunConfig(params=RunConfigOnlineModel(parallelism=1)),
    )

    assert result.trials[0].metadata["model_id"] == "target-model"
    assert result.trials[0].output is not None
    assert result.trials[0].output.output_text == "Generated model answer"
    assert result.trials[0].measurements == TrialMeasurements(
        prompt_tokens=10,
        completion_tokens=2,
        cache_read_tokens=3,
    )
    assert result.summary.score("constant_metric.score").mean == 0.75


@pytest.mark.asyncio
async def test_live_model_generation_completions_endpoint_prompts_with_instruction() -> None:
    # A bare /v1/completions endpoint uses the `{"prompt": ...}` request shape; the task instruction
    # is rendered into that wire field (there is no chat `messages` wrapper).
    async def fake_model_inference(
        model: Model,
        request: dict[str, Any],
        max_retries: int | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del model, max_retries, kwargs
        assert request["prompt"] == "Use the task instruction."
        assert "messages" not in request
        return {"choices": [{"text": "Generated model answer"}]}

    task = AgentEvalTask(
        id="task-1",
        intent="Answer the instruction.",
        inputs={"instruction": "Use the task instruction."},
        metrics=[_ConstantMetric()],
    )
    model = Model(url="https://model.test/v1/completions", name="target-model", format=ModelFormat.OPEN_AI)

    await AgentEvaluator(inference_fn=fake_model_inference).run(
        tasks=[task],
        target=model,
        config=AgentEvalRunConfig(params=RunConfigOnlineModel(parallelism=1)),
    )


@pytest.mark.asyncio
async def test_metric_failure_records_failed_score_and_does_not_stop_other_metrics() -> None:
    task = _task(metric=_FailingMetric())
    other_task = AgentEvalTask(
        id="task-2",
        intent="Answer another prompt.",
        inputs={"instruction": "Another question?"},
        metrics=[_OtherMetric()],
    )
    trials = [
        _candidate_trial(),
        AgentEvalTrial(
            id="trial-2",
            task_id="task-2",
            status=AgentEvalTrialStatus.COMPLETED,
            output=AgentOutput(output_text="Other answer"),
        ),
    ]

    result = await AgentEvaluator().run(tasks=[task, other_task], trials=trials)

    failed = next(item for item in result.scores if item.metric_type == "failing_metric")
    completed = next(item for item in result.scores if item.metric_type == "other_metric")
    assert failed.status.value == "failed"
    assert failed.outputs == []
    assert failed.diagnostics[0].message == "missing final_state evidence"
    assert completed.status.value == "completed"
    assert completed.outputs[0].value == 0.25
    assert result.summary.metric_coverage["failing_metric"]["score"].failed == 1
    assert result.summary.metric_coverage["other_metric"]["quality"].scored == 1


@pytest.mark.asyncio
async def test_success_metric_diagnostics_are_persisted() -> None:
    # A successful metric's own diagnostics (e.g. per-criterion judge verdicts) must reach the score:
    # previously only the failure path recorded diagnostics and the success path dropped them.
    result = await AgentEvaluator().run(tasks=[_task(_DiagnosticMetric())], trials=[_candidate_trial()])

    (score,) = result.scores
    assert score.status.value == "completed"
    (diagnostic,) = score.diagnostics
    assert diagnostic.severity == AgentEvalDiagnosticSeverity.INFO
    assert diagnostic.message == "criterion C-001 passed"
    assert diagnostic.source == "diagnostic_metric"
    assert diagnostic.details == {"verdict": "pass", "id": "C-001"}


@pytest.mark.asyncio
async def test_metric_failure_can_fail_fast_for_development() -> None:
    with pytest.raises(RuntimeError, match="missing final_state evidence"):
        await AgentEvaluator().run(
            tasks=[_task(metric=_FailingMetric())],
            trials=[_candidate_trial()],
            config=AgentEvalRunConfig(fail_fast=True),
        )


def test_summary_reports_coverage_and_merges_views_into_scores() -> None:
    task = AgentEvalTask(
        id="task-1",
        intent="Answer a prompt.",
        inputs={"instruction": "Question?"},
        metrics=[_ConstantMetric(), _OtherMetric()],
        views={
            "outcome_correctness": SemanticView(
                reducer=SemanticReducer.MEAN,
                signals=[
                    ViewSignal(metric="constant_metric", output="score"),
                    ViewSignal(metric="other_metric", output="quality"),
                ],
            )
        },
    )
    scores = [
        _task_score("run-1", "task-1", "trial-1", "constant_metric", "score", 1.0),
        _task_score("run-1", "task-1", "trial-1", "other_metric", "quality", 0.0),
    ]

    summary = AgentEvalSummary.from_scores(scores, tasks=[task])

    assert summary.score("constant_metric.score").mean == 1.0
    assert summary.score("other_metric.quality").mean == 0.0
    assert summary.score("view.outcome_correctness").mean == 0.5
    assert summary.metric_coverage["constant_metric"]["score"].total == 1
    assert summary.metric_coverage["constant_metric"]["score"].scored == 1


@pytest.mark.asyncio
async def test_live_agent_generation_preserves_trace_evidence_for_metrics() -> None:
    metric = _EvidenceMetric()

    async def fake_agent_inference(
        agent: Agent,
        request: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        del agent, kwargs
        # A generic HTTP agent receives the task row directly — no chat/completions wrapper — so its
        # `body` template can reference task inputs such as `{{ instruction }}`.
        assert "messages" not in request
        assert request["instruction"] == "What is the answer?"
        return {
            "choices": [{"message": {"role": "assistant", "content": "Generated agent answer"}}],
            "trajectory": [{"tool": "search", "line": 3}],
            "usage": {
                "input_tokens": 5,
                "output_tokens": 2,
                "cache_creation_input_tokens": 1,
                "cache_read_input_tokens": 4,
            },
        }

    agent = GenericAgent(
        url="https://agent.test",
        name="target-agent",
        format=AgentFormat.GENERIC,
        body={"input": "{{ instruction }}"},
        response_path="$.answer",
    )
    result = await AgentEvaluator(inference_fn=fake_agent_inference).run(
        tasks=[_task(metric)],
        target=agent,
        config=AgentEvalRunConfig(params=RunConfigOnline(parallelism=1)),
    )

    assert result.trials[0].evidence is not None
    assert result.trials[0].evidence.require("trace").kind == "trace"
    assert result.trials[0].output is not None
    assert result.trials[0].output.output_text == "Generated agent answer"
    assert result.trials[0].measurements == TrialMeasurements(
        prompt_tokens=10,
        completion_tokens=2,
        cache_creation_tokens=1,
        cache_read_tokens=4,
    )
    assert metric.inputs[0].candidate.evidence == result.trials[0].evidence


@pytest.mark.asyncio
async def test_live_agent_typed_partial_invocation_preserves_non_trace_evidence() -> None:
    metric = _EvidenceMetric()

    async def fake_agent_inference(
        agent: Agent,
        request: dict[str, Any],
        **kwargs: Any,
    ) -> AgentInvocationResult:
        del agent, request, kwargs
        return AgentInvocationResult(
            status=AgentInvocationStatus.PARTIAL,
            response={"choices": [{"message": {"role": "assistant", "content": None}}]},
            evidence=CandidateEvidence(
                descriptors={
                    "stream_events": EvidenceDescriptor(
                        kind="agent_stream_events",
                        format="json",
                        data=[{"channel": "data", "payload": {"value": {"error": "auth required"}}}],
                    )
                }
            ),
            metadata={"stream_error": "auth required"},
        )

    agent = NemoAgentToolkitAgent(url="https://agent.test", name="target-agent", format=AgentFormat.NEMO_AGENT_TOOLKIT)
    result = await AgentEvaluator(inference_fn=fake_agent_inference).run(
        tasks=[_task(metric)],
        target=agent,
        config=AgentEvalRunConfig(params=RunConfigOnline(parallelism=1)),
    )

    trial = result.trials[0]
    assert trial.status is AgentEvalTrialStatus.PARTIAL
    assert trial.evidence is not None
    assert trial.evidence.require("stream_events").kind == "agent_stream_events"
    assert "trace" not in trial.evidence.names()
    assert metric.inputs[0].candidate.evidence == trial.evidence
    assert result.scores[0].status is AgentEvalScoreStatus.COMPLETED


@pytest.mark.asyncio
async def test_live_agent_typed_failed_invocation_retains_output_and_evidence() -> None:
    async def fake_agent_inference(
        agent: Agent,
        request: dict[str, Any],
        **kwargs: Any,
    ) -> AgentInvocationResult:
        del agent, request, kwargs
        return AgentInvocationResult(
            status=AgentInvocationStatus.FAILED,
            response={"choices": [{"message": {"role": "assistant", "content": "answer"}}]},
            output_text="answer",
            evidence=CandidateEvidence(
                descriptors={
                    "raw_stream": EvidenceDescriptor(
                        kind="agent_stream",
                        format="text",
                        data="data: answer\n",
                    ),
                    "translation_error": EvidenceDescriptor(
                        kind="error",
                        format="json",
                        data={"error": "invalid ATIF"},
                    ),
                }
            ),
        )

    agent = NemoAgentToolkitAgent(
        url="https://agent.test",
        name="target-agent",
        format=AgentFormat.NEMO_AGENT_TOOLKIT,
    )
    result = await AgentEvaluator(inference_fn=fake_agent_inference).run(
        tasks=[_task()],
        target=agent,
        config=AgentEvalRunConfig(params=RunConfigOnline(parallelism=1)),
    )

    trial = result.trials[0]
    assert trial.status is AgentEvalTrialStatus.FAILED
    assert trial.output is not None
    assert trial.output.output_text == "answer"
    assert trial.evidence is not None
    assert trial.evidence.require("raw_stream").data == "data: answer\n"
    assert trial.evidence.require("translation_error").kind == "error"


@pytest.mark.asyncio
async def test_generation_boundary_names_agent_eval_context() -> None:
    agent = NemoAgentToolkitAgent(url="https://agent.test", name="target-agent", format=AgentFormat.NEMO_AGENT_TOOLKIT)
    sample = {
        "output_text": "answer",
        "response": {"choices": [{"message": {"role": "assistant", "content": "answer"}}]},
    }

    with patch(
        "nemo_evaluator_sdk.agent_eval.evaluator._generate_sample",
        new_callable=AsyncMock,
        return_value=sample,
    ) as mock_generate:
        await AgentEvaluator(inference_fn=AsyncMock()).run(
            tasks=[_task()],
            target=agent,
            config=AgentEvalRunConfig(
                run_id="run-123",
                params=RunConfigOnline(parallelism=1),
            ),
        )

    assert mock_generate.await_args is not None
    assert mock_generate.await_args.kwargs["agent_eval_context"] == {
        "run_id": "run-123",
        "task_id": "task-1",
        "invocation_id": "run-123:task-1:target-agent",
    }
    assert "template_context" not in mock_generate.await_args.kwargs


@pytest.mark.asyncio
async def test_default_agent_invocation_receives_run_context_and_evidence_dir(tmp_path: Path) -> None:
    invocation = AgentInvocationResult(
        status=AgentInvocationStatus.COMPLETED,
        response={"choices": [{"message": {"role": "assistant", "content": "answer"}}]},
        output_text="answer",
    )
    agent = NemoAgentToolkitAgent(url="https://agent.test", name="target-agent", format=AgentFormat.NEMO_AGENT_TOOLKIT)
    prompt_template = {
        "input_message": "{{ item.instruction }}",
        "conversation_id": ("{{ agent_eval.run_id }}-{{ agent_eval.task_id }}-{{ agent_eval.invocation_id }}"),
    }

    with patch(
        "nemo_evaluator_sdk.agent_inference.invoke_agent",
        new_callable=AsyncMock,
        return_value=invocation,
    ) as mock_invoke:
        await AgentEvaluator().run(
            tasks=[_task()],
            target=agent,
            config=AgentEvalRunConfig(
                run_id="run-123",
                work_dir=tmp_path,
                prompt_template=prompt_template,
                params=RunConfigOnline(parallelism=1),
            ),
        )

    assert mock_invoke.await_args is not None
    assert mock_invoke.await_args.args[1] == {
        "input_message": "What is the answer?",
        "conversation_id": "run-123-task-1-run-123:task-1:target-agent",
    }
    assert mock_invoke.await_args.kwargs["evidence_dir"] == tmp_path / "evidence" / "000000-task-1"


@pytest.mark.asyncio
async def test_default_agent_evidence_dirs_are_confined_and_unique(tmp_path: Path) -> None:
    captured_dirs: list[Path] = []
    invocation = AgentInvocationResult(
        status=AgentInvocationStatus.COMPLETED,
        response={"choices": [{"message": {"role": "assistant", "content": "answer"}}]},
        output_text="answer",
    )

    async def fake_invoke(agent: Agent, request: dict[str, Any], **kwargs: Any) -> AgentInvocationResult:
        del agent, request
        evidence_dir = kwargs["evidence_dir"]
        assert isinstance(evidence_dir, Path)
        captured_dirs.append(evidence_dir)
        return invocation

    tasks = [
        _task(task_id=".."),
        _task(task_id="a/b"),
        _task(task_id="a?b"),
    ]
    agent = NemoAgentToolkitAgent(url="https://agent.test", name="target-agent", format=AgentFormat.NEMO_AGENT_TOOLKIT)

    with patch("nemo_evaluator_sdk.agent_inference.invoke_agent", side_effect=fake_invoke):
        await AgentEvaluator().run(
            tasks=tasks,
            target=agent,
            config=AgentEvalRunConfig(
                run_id="run-123",
                work_dir=tmp_path,
                params=RunConfigOnline(parallelism=1),
            ),
        )

    evidence_root = (tmp_path / "evidence").resolve()
    assert {path.name for path in captured_dirs} == {
        "task-000000",
        "000001-a-b",
        "000002-a-b",
    }
    assert len({path.resolve() for path in captured_dirs}) == len(tasks)
    assert all(path.resolve().parent == evidence_root for path in captured_dirs)


@pytest.mark.asyncio
async def test_agent_inference_factory_receives_per_task_context(tmp_path: Path) -> None:
    invocation = AgentInvocationResult(
        status=AgentInvocationStatus.COMPLETED,
        response={"choices": [{"message": {"role": "assistant", "content": "answer"}}]},
        output_text="answer",
    )
    agent = NemoAgentToolkitAgent(url="https://agent.test", name="target-agent", format=AgentFormat.NEMO_AGENT_TOOLKIT)

    contexts: list[AgentInferenceContext] = []

    async def inference_fn(agent, request, **kwargs):
        del agent, request, kwargs
        return invocation

    def factory(context: AgentInferenceContext):
        contexts.append(context)
        return inference_fn

    await AgentEvaluator(agent_inference_fn_factory=factory).run(
        tasks=[_task()],
        target=agent,
        config=AgentEvalRunConfig(
            run_id="run-123",
            work_dir=tmp_path,
            params=RunConfigOnline(parallelism=1),
        ),
    )

    assert len(contexts) == 1
    assert contexts[0].metadata == {
        "run_id": "run-123",
        "task_id": "task-1",
        "invocation_id": "run-123:task-1:target-agent",
    }
    assert contexts[0].evidence_dir == tmp_path / "evidence" / "000000-task-1"


def test_agent_evaluator_rejects_direct_inference_and_factory() -> None:
    async def inference_fn(agent, request, **kwargs):
        del agent, request, kwargs
        return {}

    invalid_kwargs: Any = {
        "inference_fn": inference_fn,
        "agent_inference_fn_factory": lambda context: inference_fn,
    }
    with pytest.raises(ValueError, match="inference_fn.*agent_inference_fn_factory"):
        # Deliberately bypass the overload contract to verify the runtime guard for
        # dynamically typed callers.
        AgentEvaluator(**invalid_kwargs)


@pytest.mark.asyncio
async def test_live_generation_ignore_request_failure_records_failed_trial() -> None:
    async def failing_model_inference(
        model: Model,
        request: dict[str, Any],
        max_retries: int | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del model, request, max_retries, kwargs
        raise RuntimeError("inference boom")

    model = Model(url="https://model.test/v1/chat/completions", name="target-model", format=ModelFormat.OPEN_AI)

    result = await AgentEvaluator(inference_fn=failing_model_inference).run(
        tasks=[_task()],
        target=model,
        config=AgentEvalRunConfig(params=RunConfigOnlineModel(parallelism=1, ignore_request_failure=True)),
    )

    assert result.trials[0].status is AgentEvalTrialStatus.FAILED
    assert result.trials[0].output is None
    assert result.trials[0].metadata["error"] == "inference boom"
    assert result.scores[0].status is AgentEvalScoreStatus.FAILED
    assert result.summary.metric_coverage["constant_metric"]["score"].failed == 1

    # Without ignore_request_failure the run aborts on the first failed request.
    with pytest.raises(RuntimeError, match="inference boom"):
        await AgentEvaluator(inference_fn=failing_model_inference).run(
            tasks=[_task()],
            target=model,
            config=AgentEvalRunConfig(params=RunConfigOnlineModel(parallelism=1)),
        )


@pytest.mark.asyncio
async def test_run_rejects_tasks_without_trials() -> None:
    other_task = AgentEvalTask(
        id="task-2",
        intent="Answer another prompt.",
        inputs={"instruction": "Another question?"},
        metrics=[_OtherMetric()],
    )

    with pytest.raises(ValueError, match=r"no trials produced for tasks: \['task-2'\]"):
        await AgentEvaluator().run(tasks=[_task(), other_task], trials=[_candidate_trial()])


class _ErroringTaskRunner(_TaskRunner):
    """A runner whose trials carry a typed error, as the Harbor adapter's do."""

    async def run_tasks(
        self,
        tasks: Sequence[AgentEvalTask],
        config: AgentEvalRunConfig | None = None,
    ) -> list[AgentEvalTrial]:
        trials = await super().run_tasks(tasks, config)
        return [trial.model_copy(update={"error": TrialError(type="RuntimeError", message="boom")}) for trial in trials]


@pytest.mark.asyncio
async def test_run_threads_trials_into_the_summary_error_rollup() -> None:
    """Guards the one line wiring ``trials=`` into ``from_scores``.

    Every other rollup test builds the summary directly, so dropping that argument would leave them
    all green while silently emptying the rollup for every real run.
    """
    result = await AgentEvaluator().run(tasks=[_task()], target=_ErroringTaskRunner())

    assert result.summary.error_trial_ids == {"RuntimeError": [trial.id for trial in result.trials]}
    assert result.summary.error_count == len(result.trials)


def test_metric_row_exposes_the_typed_trial_error() -> None:
    """The replacement for reading ``metadata["exception_type"]``.

    Harbor no longer writes that key at all (``test_harbor_runtime.py`` asserts its absence), so this
    is the only path by which a metric can grade on *how* a trial failed. A pre-``TrialError`` bundle
    still carries the old key in its metadata, but nothing reads it.
    """
    task = AgentEvalTask(id="task-1", intent="Fix it.", inputs={"instruction": "Q?"})
    trial = _candidate_trial().model_copy(
        update={"error": TrialError(type="RuntimeError", message="boom", traceback="Traceback...\n")}
    )

    row = _metric_row(task, trial)

    assert row["trial"]["error"] == {
        "type": "RuntimeError",
        "message": "boom",
        "traceback": "Traceback...\n",
        "occurred_at": None,
    }


def test_metric_row_error_is_none_for_a_trial_that_did_not_fail() -> None:
    # Always present, so a metric can read it unconditionally rather than probing for the key.
    row = _metric_row(AgentEvalTask(id="task-1", intent="Fix it.", inputs={}), _candidate_trial())

    assert row["trial"]["error"] is None
    assert row["trial"]["measurements"] == TrialMeasurements().model_dump(mode="json")


class _PreflightCountingMetric(_ConstantMetric):
    """Metric that records how many times its preflight ran."""

    preflight_calls: int = 0

    async def preflight(self) -> None:
        self.preflight_calls += 1


@pytest.mark.asyncio
async def test_run_preflights_metrics_before_scoring_imported_trials() -> None:
    """Agent-eval scores metrics directly, so it must run their preflight itself.

    An LLM judge detects its endpoint's structured-output encoding in preflight. Without this the
    judge scores using the provisional encoding new_hooks guessed, silently losing enforcement on
    an endpoint that does not honour it.
    """
    metric = _PreflightCountingMetric()

    await AgentEvaluator().run(
        tasks=[_task(metric)],
        trials=[_candidate_trial()],
        config=AgentEvalRunConfig(parallelism=1),
    )

    assert metric.preflight_calls == 1


@pytest.mark.asyncio
async def test_run_preflights_each_metric_once_across_tasks() -> None:
    """The same metric object is scored once per trial; preflight must not repeat per scoring."""
    metric = _PreflightCountingMetric()

    await AgentEvaluator().run(
        tasks=[_task(metric, task_id="task-1"), _task(metric, task_id="task-2")],
        trials=[
            _candidate_trial(),
            AgentEvalTrial(
                id="trial-2",
                task_id="task-2",
                status=AgentEvalTrialStatus.COMPLETED,
                output=AgentOutput(output_text="Candidate answer"),
                metadata={"model_id": "candidate"},
            ),
        ],
        config=AgentEvalRunConfig(parallelism=1),
    )

    assert metric.preflight_calls == 1


class _ProbingMetric(_ConstantMetric):
    """Metric whose preflight probes a remote endpoint, as an LLM judge's does."""

    preflight_calls: int = 0

    async def preflight(self) -> None:
        self.preflight_calls += 1
        raise RuntimeError("endpoint probe failed")


def _failed_trial(task_id: str = "task-1", trial_id: str = "trial-1") -> AgentEvalTrial:
    return AgentEvalTrial(
        id=trial_id,
        task_id=task_id,
        status=AgentEvalTrialStatus.FAILED,
        output=AgentOutput(output_text=""),
    )


@pytest.mark.asyncio
async def test_failed_trials_are_scored_without_preflighting_their_metric() -> None:
    """A run whose trials all failed must not probe the judge endpoint.

    Failed trials short-circuit to a failed score without invoking the metric, so preflighting is
    both a wasted remote call and a new way for the run to abort: preflight resolves the judge
    model, and that raises for an unresolved reference. Importing a batch of failed trials must
    still yield failed scores rather than an exception.
    """
    metric = _ProbingMetric()

    result = await AgentEvaluator().run(
        tasks=[_task(metric)],
        trials=[_failed_trial()],
        config=AgentEvalRunConfig(parallelism=1),
    )

    assert metric.preflight_calls == 0
    assert [score.status for score in result.scores] == [AgentEvalScoreStatus.FAILED]


@pytest.mark.asyncio
async def test_metric_is_preflighted_when_any_trial_is_scoreable() -> None:
    """One completed trial is enough to need the endpoint resolved."""
    metric = _PreflightCountingMetric()

    await AgentEvaluator().run(
        tasks=[_task(metric, task_id="task-1"), _task(metric, task_id="task-2")],
        trials=[
            _failed_trial(task_id="task-1", trial_id="trial-1"),
            AgentEvalTrial(
                id="trial-2",
                task_id="task-2",
                status=AgentEvalTrialStatus.COMPLETED,
                output=AgentOutput(output_text="Candidate answer"),
            ),
        ],
        config=AgentEvalRunConfig(parallelism=1),
    )

    assert metric.preflight_calls == 1
