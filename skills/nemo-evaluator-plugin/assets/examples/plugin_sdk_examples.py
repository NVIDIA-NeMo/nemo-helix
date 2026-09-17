# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concise examples for the Evaluator plugin SDK surfaces.

These functions are intentionally not called at import time. Copy the one that
matches the feature being used and supply a configured NeMo Platform client.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def capital_france_metric() -> Any:
    """Return an output-only metric suitable for stored agent-evaluation tasks."""
    from nemo_evaluator_sdk import StringCheckMetric

    return StringCheckMetric(
        operation="equals",
        left_template="{{sample.output_text | trim}}",
        right_template="Paris",
    )


def evaluate_standalone() -> Any:
    """Evaluate one deterministic metric in process."""
    from nemo_evaluator_sdk import Evaluator, ExactMatchMetric

    return Evaluator().run_sync(
        metrics=[
            ExactMatchMetric(
                reference="{{item.expected}}",
                candidate="{{item.output}}",
            )
        ],
        dataset=[
            {"expected": "Paris", "output": "Paris"},
            {"expected": "Paris", "output": "London"},
        ],
    )


def submit_and_collect(client: Any, output_dir: Path) -> tuple[Any, Path]:
    """Submit one metric, wait for completion, and retrieve its artifacts."""
    from nemo_evaluator_sdk import ExactMatchMetric

    job = client.evaluator.submit(
        metric=ExactMatchMetric(
            reference="{{item.expected}}",
            candidate="{{item.output}}",
        ),
        dataset=[{"expected": "Paris", "output": "Paris"}],
    )
    job.wait_until_done()
    return job.get_result(), job.download_artifacts(output_dir)


def store_resources(client: Any) -> None:
    """Store one metric, task, and taskset."""
    from nemo_evaluator.api.schemas import (
        EvaluatorTaskDefinition,
        MetricRef,
        TaskInput,
        TaskInputs,
        TaskRef,
        TasksetInput,
    )

    client.evaluator.metrics.create(
        "answer-exact",
        metric=capital_france_metric(),
    )
    client.evaluator.tasks.create(
        "capital-france",
        task=TaskInput(
            spec=EvaluatorTaskDefinition(
                kind="evaluator",
                intent="Name the capital of France.",
                inputs=TaskInputs(instruction="What is the capital of France?"),
                metrics=[MetricRef("answer-exact")],
            ),
        ),
    )
    client.evaluator.tasksets.create(
        "geography",
        taskset=TasksetInput(tasks=[TaskRef("capital-france")]),
    )


def build_agent_eval_spec(metric_bundle: Any) -> Any:
    """Build a durable task evaluation with a runner target."""
    from nemo_evaluator.api.schemas import TaskInputs
    from nemo_evaluator.jobs.agent_spec import (
        AgentEvalInputSpec,
        AgentEvalTaskInput,
        FabricRunnerTarget,
    )

    return AgentEvalInputSpec(
        tasks=[
            AgentEvalTaskInput(
                id="capital-france",
                intent="Name the capital of France.",
                inputs=TaskInputs(instruction="What is the capital of France?"),
                metrics=[metric_bundle],
            )
        ],
        target=FabricRunnerTarget(
            config={
                "metadata": {"name": "geography-smoke"},
                "harness": {"adapter_id": "nvidia.fabric.codex"},
            }
        ),
        max_concurrent_tasks=2,
        labels={"benchmark": "geography-smoke"},
    )


def build_gym_agent_eval_spec() -> Any:
    """Build a sandboxed Gym evaluation for a stored taskset and environment FileSet."""
    from nemo_evaluator.api.schemas import TasksetRef
    from nemo_evaluator.filesets import FilesetRef
    from nemo_evaluator.jobs.agent_spec import AgentEvalInputSpec, GymRunnerTarget

    return AgentEvalInputSpec(
        tasks=TasksetRef("default/my-gym-taskset"),
        target=GymRunnerTarget(
            environment=FilesetRef(root="default/my-gym-environment"),
            agent="simple_agent",
            agent_config="responses_api_agents/simple_agent/configs/simple_agent.yaml",
            resources_server="custom_greeting",
            num_repeats=1,
            concurrency=1,
            hydra_params={
                "policy_base_url": (
                    "http://nemo-platform-api.default.svc.cluster.local:8080/"
                    "apis/inference-gateway/v2/workspaces/default/model/my-model/-/v1"
                ),
                "policy_api_key": "not-used",
                "policy_model_name": "my-model",
            },
        ),
        max_concurrent_tasks=1,
        fail_fast=True,
        labels={"benchmark": "custom-gym-environment"},
    )
