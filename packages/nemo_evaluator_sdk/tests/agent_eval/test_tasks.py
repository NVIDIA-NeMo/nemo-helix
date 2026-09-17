# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_evaluator_sdk.agent_eval.tasks import (
    AgentEvalTask,
    AgentEvalTaskset,
    AgentEvalTasksetLoader,
    SemanticReducer,
    SemanticView,
    ViewSignal,
)
from nemo_evaluator_sdk.metrics.protocol import MetricInput, MetricOutputSpec, MetricResult
from pydantic import ValidationError


def test_taskset_requires_tasks_and_unique_ids() -> None:
    task = AgentEvalTask(id="t", intent="i", inputs={})
    assert AgentEvalTaskset(tasks=[task]).tasks == [task]
    with pytest.raises(ValidationError, match="at least 1"):
        AgentEvalTaskset(tasks=[])
    with pytest.raises(ValidationError, match="duplicate taskset task ids"):
        AgentEvalTaskset(tasks=[task, AgentEvalTask(id="t", intent="i", inputs={})])


def test_loader_protocol_is_satisfied_by_a_named_load_adapter() -> None:
    class _Loader:
        name = "fake"

        def load(self, *, source: object = None, limit: object = None, evidence_dir: object = None) -> AgentEvalTaskset:
            return AgentEvalTaskset(tasks=[AgentEvalTask(id="t", intent="i", inputs={})])

    loader = _Loader()
    assert isinstance(loader, AgentEvalTasksetLoader)
    assert [task.id for task in loader.load().tasks] == ["t"]


class _Metric:
    @property
    def type(self) -> str:
        return "example_metric"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("score")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        raise NotImplementedError


def test_task_serializes_metric_instances_as_descriptors() -> None:
    task = AgentEvalTask(
        id="task-1",
        intent="answer the prompt",
        inputs={"instruction": "Question?"},
        metrics=[_Metric()],
    )

    assert task.model_dump(mode="json")["metrics"] == [
        {
            "type": "example_metric",
            "outputs": [{"name": "score", "description": None, "value_schema": "ContinuousScore"}],
        }
    ]


def test_task_serializes_optional_metric_outputs_only_with_required_false() -> None:
    class _OptionalMetric(_Metric):
        def output_spec(self) -> list[MetricOutputSpec]:
            return [MetricOutputSpec.continuous_score("score", required=False)]

    task = AgentEvalTask(
        id="task-1",
        intent="answer the prompt",
        inputs={"instruction": "Question?"},
        metrics=[_OptionalMetric()],
    )

    assert task.model_dump(mode="json")["metrics"] == [
        {
            "type": "example_metric",
            "outputs": [{"name": "score", "description": None, "value_schema": "ContinuousScore", "required": False}],
        }
    ]


def test_task_required_metric_descriptor_bytes_remain_stable() -> None:
    task = AgentEvalTask(
        id="task-1",
        intent="answer the prompt",
        inputs={"instruction": "Question?"},
        metrics=[_Metric()],
    )

    assert task.model_dump_json() == (
        '{"id":"task-1","intent":"answer the prompt","inputs":{"instruction":"Question?"},'
        '"reference":{},"metrics":[{"type":"example_metric","outputs":[{"name":"score",'
        '"description":null,"value_schema":"ContinuousScore"}]}],"views":{},"metadata":{}}'
    )


def test_task_rejects_duplicate_metric_types() -> None:
    with pytest.raises(ValueError, match="duplicate task metric types"):
        AgentEvalTask(
            id="task-1",
            intent="answer the prompt",
            inputs={"instruction": "Question?"},
            metrics=[_Metric(), _Metric()],
        )


def test_task_validates_view_signals_against_metric_outputs() -> None:
    with pytest.raises(ValueError, match="unknown output"):
        AgentEvalTask(
            id="task-1",
            intent="answer the prompt",
            inputs={"instruction": "Question?"},
            metrics=[_Metric()],
            views={
                "outcome_correctness": SemanticView(
                    reducer=SemanticReducer.SINGLE,
                    signals=[ViewSignal(metric="example_metric", output="missing")],
                )
            },
        )
