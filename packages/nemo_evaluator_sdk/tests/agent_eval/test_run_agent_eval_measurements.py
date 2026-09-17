# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult, AgentEvalSummary
from nemo_evaluator_sdk.agent_eval.scores import AgentEvalScoreStatus, AgentEvalTaskScore
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, TrialMeasurements
from nemo_evaluator_sdk.metrics.protocol import MetricOutput
from pydantic import ValidationError


def _example(name: str):
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(f"examples.run_agent_eval.{name}")


@pytest.mark.asyncio
async def test_workflow_runtime_records_typed_runtime_only(tmp_path: Path) -> None:
    workflow = _example("workflow_runtime")
    task = AgentEvalTask(id="task-1", intent="Answer.", inputs={"instruction": "hello"})

    [trial] = await workflow.WorkflowAgentRuntime().run_tasks(
        [task], config=AgentEvalRunConfig(work_dir=tmp_path, parallelism=1)
    )

    assert trial.measurements.runtime_sec is not None
    assert trial.measurements.runtime_sec >= 0
    assert "runtime_sec" not in trial.metadata


def test_platform_runtime_uses_outer_runtime_and_canonical_cache_total(tmp_path: Path) -> None:
    platform = _example("platform_runtime")
    layout = platform.AgenticRunLayout(
        run_dir=tmp_path,
        agent_log_dir=tmp_path / "agent",
        workspace_dir=tmp_path / "workspace",
        state_dir=tmp_path / "state",
        instruction_path=tmp_path / "instruction.md",
    )
    for directory in (layout.agent_log_dir, layout.workspace_dir, layout.state_dir):
        directory.mkdir()
    (layout.agent_log_dir / "nat_agent.log").write_text(
        json.dumps(
            {
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "cache_creation_input_tokens": 1,
                    "cache_read_input_tokens": 4,
                },
                "duration_ms": 999_000,
            }
        ),
        encoding="utf-8",
    )

    trial = platform.build_trial_from_artifacts(
        task=AgentEvalTask(id="task-1", intent="Answer.", inputs={}),
        layout=layout,
        runtime_name="workflow",
        agent_model="model",
        exit_code=0,
        agent_ok=True,
        runtime_sec=4.0,
    )

    assert trial.measurements == TrialMeasurements(
        prompt_tokens=10,
        completion_tokens=2,
        cache_creation_tokens=1,
        cache_read_tokens=4,
        runtime_sec=4.0,
    )
    assert (
        not {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cache_creation_tokens",
            "cache_read_tokens",
            "runtime_sec",
            "duration_ms",
        }
        & trial.metadata.keys()
    )


def test_example_usage_handles_inclusive_ambiguous_and_invalid_cache_shapes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    usage = _example("usage")

    inclusive = usage.extract_usage_metrics(
        json.dumps({"usage": {"input_tokens": 5, "output_tokens": 2, "cached_input_tokens": 4}})
    )
    assert inclusive["prompt_tokens"] == 5
    assert inclusive["cache_read_tokens"] == 4
    assert inclusive["total_tokens"] == 7

    ambiguous = usage.extract_usage_metrics(
        json.dumps(
            {
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "cached_input_tokens": 4,
                    "cache_read_input_tokens": 1,
                }
            }
        )
    )
    assert ambiguous["prompt_tokens"] is None
    assert ambiguous["completion_tokens"] == 2
    assert ambiguous["cache_read_tokens"] is None

    invalid = usage.extract_usage_metrics(
        json.dumps({"usage": {"input_tokens": 5, "output_tokens": 2, "cache_read_input_tokens": "bad"}})
    )
    assert invalid["prompt_tokens"] is None
    assert invalid["completion_tokens"] == 2
    assert "omitting ambiguous prompt/cache measurements" in caplog.text
    assert "cache_read_input_tokens='bad'" in caplog.text


@pytest.mark.parametrize(
    ("usage_payload", "expected"),
    [
        pytest.param(
            {
                "input_tokens": 5,
                "output_tokens": 2,
                "cached_input_tokens": None,
                "cache_read_input_tokens": 4,
            },
            {
                "prompt_tokens": 9,
                "completion_tokens": 2,
                "total_tokens": 11,
                "cache_creation_tokens": None,
                "cache_read_tokens": 4,
                "duration_ms": None,
            },
            id="null-inclusive-placeholder",
        ),
        pytest.param(
            {
                "input_tokens": 5,
                "output_tokens": 2,
                "cached_input_tokens": 4,
                "cache_read_input_tokens": None,
            },
            {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
                "cache_creation_tokens": None,
                "cache_read_tokens": 4,
                "duration_ms": None,
            },
            id="null-separate-placeholder",
        ),
    ],
)
def test_example_usage_ignores_null_cache_format_placeholders(
    usage_payload: dict,
    expected: dict,
) -> None:
    usage = _example("usage")

    assert usage.extract_usage_metrics(json.dumps({"usage": usage_payload})) == expected


def test_example_usage_ambiguous_message_invalidates_trial_wide_prompt_and_cache() -> None:
    usage = _example("usage")

    metrics = usage.extract_usage_metrics(
        json.dumps(
            {
                "messages": [
                    {
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 1,
                            "cache_read_input_tokens": 2,
                        }
                    },
                    {
                        "usage": {
                            "input_tokens": 20,
                            "output_tokens": 2,
                            "cache_read_input_tokens": 3,
                            "cached_input_tokens": 3,
                        }
                    },
                ]
            }
        )
    )

    assert metrics == {
        "prompt_tokens": None,
        "completion_tokens": 3,
        "total_tokens": None,
        "cache_creation_tokens": None,
        "cache_read_tokens": None,
        "duration_ms": None,
    }


def test_example_reporting_reads_typed_measurements() -> None:
    runner = _example("run_agent_eval")
    trial = AgentEvalTrial(
        id="trial-1",
        task_id="task-1",
        status=AgentEvalTrialStatus.PARTIAL,
        measurements=TrialMeasurements(prompt_tokens=8, completion_tokens=2, runtime_sec=1.5),
        metadata={"total_tokens": 999, "runtime_sec": 999},
    )
    result = AgentEvalResult(run_id="run-1", tasks=[], trials=[trial], scores=[], summary=AgentEvalSummary())

    output = io.StringIO()
    with redirect_stdout(output):
        runner._print_measurements(result)

    assert output.getvalue().splitlines() == [
        "  total_tokens: 10 across 1/1 trials",
        "  runtime_sec: 1.5 across 1/1 trials",
    ]


def test_gating_reads_typed_measurements_and_reports_unscored_tasks() -> None:
    gating = _example("gating")
    tasks = [
        AgentEvalTask(id="task-1", intent="Answer.", inputs={}),
        AgentEvalTask(id="task-2", intent="Answer.", inputs={}),
    ]
    trials = [
        AgentEvalTrial(
            id=f"{task.id}:trial",
            task_id=task.id,
            status=AgentEvalTrialStatus.PARTIAL,
            measurements=TrialMeasurements(prompt_tokens=5, completion_tokens=1, runtime_sec=2.0),
        )
        for task in tasks
    ]
    score = AgentEvalTaskScore(
        id="score-1",
        run_id="run-1",
        task_id="task-1",
        trial_id=trials[0].id,
        metric_type="verifier",
        status=AgentEvalScoreStatus.COMPLETED,
        outputs=[MetricOutput(name="verifier_reward", value=1.0)],
    )
    result = AgentEvalResult(
        run_id="run-1",
        tasks=tasks,
        trials=trials,
        scores=[score],
        summary=AgentEvalSummary(),
    )

    summary = gating.summarize_run(result)

    assert summary["total_tokens_sum"] == 12
    assert summary["runtime_sec_sum"] == 4.0
    assert summary["passed_tasks"] == 1
    assert summary["unscored_task_ids"] == ["task-2"]


def test_fabric_measurements_remain_outside_runtime_gating() -> None:
    gating = _example("gating")
    task = AgentEvalTask(id="task-1", intent="Answer.", inputs={})
    trial = AgentEvalTrial(
        id="task-1:fabric",
        task_id=task.id,
        status=AgentEvalTrialStatus.PARTIAL,
        measurements=TrialMeasurements(prompt_tokens=8, completion_tokens=2, cost_usd=0.25),
        metadata={"adapter_id": "nvidia.fabric.codex"},
    )
    result = AgentEvalResult(
        run_id="run-1",
        tasks=[task],
        trials=[trial],
        scores=[],
        summary=AgentEvalSummary(),
    )

    summary = gating.summarize_run(result)

    assert summary["total_tokens_sum"] == 10
    assert summary["runtime_sec_sum"] is None
    assert summary["avg_runtime_sec"] is None
    assert summary["runtime_metrics_coverage"] == 0.0
    assert summary["runtime_metrics_available_tasks"] == 0
    assert summary["runtime_metrics_unavailable_tasks"] == ["task-1"]


@pytest.mark.parametrize("filename", ["trial.json", "trials.jsonl"])
@pytest.mark.parametrize(
    ("payload", "expected_measurements"),
    [
        pytest.param(
            {
                "measurements": {
                    "prompt_tokens": 8,
                    "completion_tokens": 2,
                    "runtime_sec": 0,
                    "cost_usd": 0,
                },
                "metadata": {"reward": 0.8},
            },
            TrialMeasurements(prompt_tokens=8, completion_tokens=2, runtime_sec=0, cost_usd=0),
            id="typed",
        ),
        pytest.param(
            {"metadata": {"prompt_tokens": 8, "completion_tokens": 2, "duration_ms": 1500}},
            TrialMeasurements(),
            id="metadata-only",
        ),
        pytest.param(
            {
                "measurements": {"prompt_tokens": 8, "completion_tokens": 2},
                "metadata": {"prompt_tokens": 999, "completion_tokens": 999, "duration_ms": 1500},
            },
            TrialMeasurements(prompt_tokens=8, completion_tokens=2),
            id="conflicting-metadata",
        ),
    ],
)
def test_example_stored_trial_loaders_preserve_typed_measurements_and_opaque_metadata(
    tmp_path: Path,
    filename: str,
    payload: dict[str, object],
    expected_measurements: TrialMeasurements,
) -> None:
    workflow = _example("workflow_runtime")
    row = {"id": "stored-trial", "task_id": "task-1", "status": "partial", **payload}
    serialized = json.dumps(row)
    (tmp_path / filename).write_text(serialized + ("\n" if filename.endswith("jsonl") else ""), encoding="utf-8")

    [trial] = workflow.load_stored_trials(tmp_path)

    assert trial.measurements == expected_measurements
    assert trial.metadata == payload.get("metadata", {})


@pytest.mark.parametrize("filename", ["trial.json", "trials.jsonl"])
@pytest.mark.parametrize("measurements", [None, {"prompt_tokens": -1}])
def test_example_stored_trial_loaders_reject_invalid_typed_measurements(
    tmp_path: Path,
    filename: str,
    measurements: object,
) -> None:
    workflow = _example("workflow_runtime")
    serialized = json.dumps(
        {
            "id": "stored-trial",
            "task_id": "task-1",
            "status": "partial",
            "measurements": measurements,
            "metadata": {"prompt_tokens": 8, "duration_ms": 1500},
        }
    )
    (tmp_path / filename).write_text(serialized + ("\n" if filename.endswith("jsonl") else ""), encoding="utf-8")

    with pytest.raises(ValidationError):
        workflow.load_stored_trials(tmp_path)
