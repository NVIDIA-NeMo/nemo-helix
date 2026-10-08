# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for task-local Evaluator benchmark metrics."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from evaluator_agent_eval.artifacts import AgentArtifacts
from evaluator_agent_eval.factory import build_evaluator_scoring_row, capture_agent_attempt
from evaluator_agent_eval.runner import score_evaluator_rows
from evaluator_agent_eval.schemas import CapturedAgentAttempt
from evaluator_agent_eval.task_config import AgenticUseTaskConfig, load_agentic_use_task_config
from evaluator_agent_eval.task_metric_utils import extract_marker_json_object
from nhx_evals_sdk.metrics.base import Metric


def _load_task_metrics(task_name: str) -> ModuleType:
    agentic_use_dir = Path(__file__).parents[2]
    path = agentic_use_dir / task_name / "tests" / "task_metrics.py"
    spec = importlib.util.spec_from_file_location(f"{task_name}_task_metrics", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not import task metrics from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_surface_discovery_metric_normalizes_required_term_success(evaluator_task_dir: Path, agent_log_dir: Path):
    (agent_log_dir / "final_message.txt").write_text(
        """
Use packages/nhx_evals_sdk with Evaluator, ExactMatchMetric, and run_sync.

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path("packages/nhx_evals_sdk/src").resolve()))

from nhx_evals_sdk import Evaluator, ExactMatchMetric

rows = [
    {"question": "2+2?", "expected": "4", "prediction": "4"},
    {"question": "Capital of France?", "expected": "Paris", "prediction": "Lyon"},
]

result = Evaluator().run_sync(
    metrics=ExactMatchMetric(
        reference="{{item.expected}}",
        candidate="{{item.prediction}}",
    ),
    dataset=rows,
)
```
""".strip(),
        encoding="utf-8",
    )
    artifacts = AgentArtifacts.from_dir(agent_log_dir)
    task_config = load_agentic_use_task_config(evaluator_task_dir)
    attempt = capture_agent_attempt(task_dir=evaluator_task_dir, artifacts=artifacts)
    module = _load_task_metrics("evaluator-standalone-sdk-surface-discovery")

    scores = _score_task_metric(
        metric=module.SurfaceDiscoveryMetric(),
        task_dir=evaluator_task_dir,
        artifacts=artifacts,
        task_config=task_config,
        attempt=attempt,
    )

    assert scores["task_success"] == [1.0]
    assert scores["verification_score"] == [1.0]
    assert scores["output_schema_valid"] == [1.0]


def test_surface_discovery_metric_accepts_async_invocation(evaluator_task_dir: Path, agent_log_dir: Path):
    (agent_log_dir / "final_message.txt").write_text(
        """
Use packages/nhx_evals_sdk with Evaluator, ExactMatchMetric, and run.

```python
from pathlib import Path
import asyncio
import sys

sys.path.insert(0, str(Path("packages/nhx_evals_sdk/src").resolve()))

from nhx_evals_sdk import Evaluator, ExactMatchMetric

rows = [
    {"question": "2+2?", "expected": "4", "prediction": "4"},
    {"question": "Capital of France?", "expected": "Paris", "prediction": "Lyon"},
]

async def main() -> None:
    result = await Evaluator().run(
        metrics=ExactMatchMetric(
            reference="{{item.expected}}",
            candidate="{{item.prediction}}",
        ),
        dataset=rows,
    )
    print(result)

asyncio.run(main())
```
""".strip(),
        encoding="utf-8",
    )
    artifacts = AgentArtifacts.from_dir(agent_log_dir)
    task_config = load_agentic_use_task_config(evaluator_task_dir)
    attempt = capture_agent_attempt(task_dir=evaluator_task_dir, artifacts=artifacts)
    module = _load_task_metrics("evaluator-standalone-sdk-surface-discovery")

    scores = _score_task_metric(
        metric=module.SurfaceDiscoveryMetric(),
        task_dir=evaluator_task_dir,
        artifacts=artifacts,
        task_config=task_config,
        attempt=attempt,
    )

    assert scores["task_success"] == [1.0]
    assert scores["verification_score"] == [1.0]
    assert scores["output_schema_valid"] == [1.0]


def test_surface_discovery_metric_selects_complete_snippet(evaluator_task_dir: Path, agent_log_dir: Path):
    (agent_log_dir / "final_message.txt").write_text(
        """
Use packages/nhx_evals_sdk with Evaluator, ExactMatchMetric, and run_sync.

```python
from nhx_evals_sdk import Evaluator, ExactMatchMetric
```

```python
from nhx_evals_sdk import Evaluator, ExactMatchMetric

rows = [
    {"question": "2+2?", "expected": "4", "prediction": "4"},
    {"question": "Capital of France?", "expected": "Paris", "prediction": "Lyon"},
]

result = Evaluator().run_sync(
    metrics=ExactMatchMetric(
        reference="{{item.expected}}",
        candidate="{{item.prediction}}",
    ),
    dataset=rows,
)
```
""".strip(),
        encoding="utf-8",
    )
    artifacts = AgentArtifacts.from_dir(agent_log_dir)
    task_config = load_agentic_use_task_config(evaluator_task_dir)
    attempt = capture_agent_attempt(task_dir=evaluator_task_dir, artifacts=artifacts)
    module = _load_task_metrics("evaluator-standalone-sdk-surface-discovery")

    scores = _score_task_metric(
        metric=module.SurfaceDiscoveryMetric(),
        task_dir=evaluator_task_dir,
        artifacts=artifacts,
        task_config=task_config,
        attempt=attempt,
    )

    assert scores["task_success"] == [1.0]
    assert scores["verification_score"] == [1.0]
    assert scores["output_schema_valid"] == [1.0]


def test_surface_discovery_metric_rejects_prose_only_answer(evaluator_task_dir: Path, agent_log_dir: Path):
    (agent_log_dir / "final_message.txt").write_text(
        (
            "Use packages/nhx_evals_sdk with Evaluator, run_sync, and ExactMatchMetric. "
            "Create the two-row dataset and call Evaluator().run_sync(...) with the expected templates."
        ),
        encoding="utf-8",
    )
    artifacts = AgentArtifacts.from_dir(agent_log_dir)
    task_config = load_agentic_use_task_config(evaluator_task_dir)
    attempt = capture_agent_attempt(task_dir=evaluator_task_dir, artifacts=artifacts)
    module = _load_task_metrics("evaluator-standalone-sdk-surface-discovery")

    scores = _score_task_metric(
        metric=module.SurfaceDiscoveryMetric(),
        task_dir=evaluator_task_dir,
        artifacts=artifacts,
        task_config=task_config,
        attempt=attempt,
    )

    assert scores["task_success"] == [0.0]
    assert scores["output_schema_valid"] == [0.0]


def test_exact_match_metric_requires_runnable_sdk_code(tmp_path: Path, agent_log_dir: Path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (task_dir / "task.toml").write_text(
        """
version = "1.0"

[evaluator.surface]
constraint = "standalone_sdk"
allowed = ["standalone_sdk"]
forbidden = ["cli", "plugin_sdk", "legacy_service"]

[evaluator.expected]
required_terms = ["packages/nhx_evals_sdk", "Evaluator", "run_sync", "ExactMatchMetric", "2+2?", "Capital of France?", "0.5"]
""".strip(),
        encoding="utf-8",
    )
    (task_dir / "instruction.md").write_text("Run exact match.", encoding="utf-8")
    (workspace_dir / "solution.py").write_text(
        """
from pathlib import Path
import sys

sys.path.insert(0, str(Path("packages/nhx_evals_sdk/src").resolve()))

from nhx_evals_sdk import Evaluator, ExactMatchMetric

rows = [
    {"question": "2+2?", "expected": "4", "prediction": "4"},
    {"question": "Capital of France?", "expected": "Paris", "prediction": "Lyon"},
]
result = Evaluator().run_sync(
    metrics=ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.prediction}}"),
    dataset=rows,
)
result.print_summary()
""".strip(),
        encoding="utf-8",
    )
    (agent_log_dir / "final_message.txt").write_text(
        "Wrote workspace/solution.py using packages/nhx_evals_sdk; the summary has aggregate score 0.5.",
        encoding="utf-8",
    )
    artifacts = AgentArtifacts.from_dir(agent_log_dir, workspace_dir=workspace_dir)
    task_config = load_agentic_use_task_config(task_dir)
    attempt = capture_agent_attempt(task_dir=task_dir, artifacts=artifacts)
    module = _load_task_metrics("evaluator-standalone-sdk-simple-exact-match")

    scores = _score_task_metric(
        metric=module.ExactMatchEvaluationMetric(),
        task_dir=task_dir,
        artifacts=artifacts,
        task_config=task_config,
        attempt=attempt,
    )

    assert scores["task_success"] == [1.0]
    assert scores["verification_score"] == [1.0]
    assert scores["output_schema_valid"] == [1.0]


def test_exact_match_metric_rejects_code_that_runs_without_expected_scores(tmp_path: Path, agent_log_dir: Path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (task_dir / "task.toml").write_text(
        """
version = "1.0"

[evaluator.surface]
constraint = "standalone_sdk"
allowed = ["standalone_sdk"]
forbidden = ["cli", "plugin_sdk", "legacy_service"]

[evaluator.expected]
required_terms = ["packages/nhx_evals_sdk", "Evaluator", "run_sync", "ExactMatchMetric", "2+2?", "Capital of France?", "0.5"]
""".strip(),
        encoding="utf-8",
    )
    (task_dir / "instruction.md").write_text("Run exact match.", encoding="utf-8")
    (workspace_dir / "solution.py").write_text(
        """
from nhx_evals_sdk import Evaluator, ExactMatchMetric

# This mentions run_sync but does not perform the required evaluation.
print("not the expected row scores")
print("0.5")
""".strip(),
        encoding="utf-8",
    )
    (agent_log_dir / "final_message.txt").write_text(
        "Wrote workspace/solution.py using packages/nhx_evals_sdk for 2+2? and Capital of France?.",
        encoding="utf-8",
    )
    artifacts = AgentArtifacts.from_dir(agent_log_dir, workspace_dir=workspace_dir)
    task_config = load_agentic_use_task_config(task_dir)
    attempt = capture_agent_attempt(task_dir=task_dir, artifacts=artifacts)
    module = _load_task_metrics("evaluator-standalone-sdk-simple-exact-match")

    scores = _score_task_metric(
        metric=module.ExactMatchEvaluationMetric(),
        task_dir=task_dir,
        artifacts=artifacts,
        task_config=task_config,
        attempt=attempt,
    )

    assert scores["task_success"] == [0.0]
    assert scores["output_schema_valid"] == [0.0]


def test_exact_match_metric_rejects_missing_solution_artifact(tmp_path: Path, agent_log_dir: Path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (task_dir / "task.toml").write_text(
        """
version = "1.0"

[evaluator.surface]
constraint = "standalone_sdk"
allowed = ["standalone_sdk"]
forbidden = ["cli", "plugin_sdk", "legacy_service"]

[evaluator.expected]
required_terms = ["packages/nhx_evals_sdk", "Evaluator", "run_sync", "ExactMatchMetric", "2+2?", "Capital of France?", "0.5"]
""".strip(),
        encoding="utf-8",
    )
    (task_dir / "instruction.md").write_text("Run exact match.", encoding="utf-8")
    (agent_log_dir / "final_message.txt").write_text(
        "I would use packages/nhx_evals_sdk with Evaluator, run_sync, and ExactMatchMetric.",
        encoding="utf-8",
    )
    artifacts = AgentArtifacts.from_dir(agent_log_dir, workspace_dir=workspace_dir)
    task_config = load_agentic_use_task_config(task_dir)
    attempt = capture_agent_attempt(task_dir=task_dir, artifacts=artifacts)
    module = _load_task_metrics("evaluator-standalone-sdk-simple-exact-match")

    scores = _score_task_metric(
        metric=module.ExactMatchEvaluationMetric(),
        task_dir=task_dir,
        artifacts=artifacts,
        task_config=task_config,
        attempt=attempt,
    )

    assert scores["task_success"] == [0.0]
    assert scores["output_schema_valid"] == [0.0]


@pytest.mark.asyncio
async def test_exact_match_metric_attaches_summary_diff_diagnostics(tmp_path: Path, agent_log_dir: Path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (task_dir / "task.toml").write_text(
        """
version = "1.0"

[evaluator.surface]
constraint = "standalone_sdk"
allowed = ["standalone_sdk"]
forbidden = ["cli", "plugin_sdk", "legacy_service"]

[evaluator.expected]
required_terms = ["packages/nhx_evals_sdk", "Evaluator", "run_sync", "ExactMatchMetric", "2+2?", "Capital of France?", "0.5"]
""".strip(),
        encoding="utf-8",
    )
    (workspace_dir / "solution.py").write_text(
        """
from nhx_evals_sdk import Evaluator, ExactMatchMetric

# Mentions run_sync / ExactMatchMetric but prints the wrong summary.
print("not the expected summary")
""".strip(),
        encoding="utf-8",
    )
    module = _load_task_metrics("evaluator-standalone-sdk-simple-exact-match")
    result = await module.ExactMatchEvaluationMetric().compute_scores(
        {
            "output_text": "packages/nhx_evals_sdk Evaluator run_sync ExactMatchMetric 2+2? Capital of France? 0.5",
            "workspace_dir": str(workspace_dir),
            "task_dir": str(task_dir),
            "final_answer_extracted": True,
        },
        {},
    )

    assert [output.name for output in result.outputs] == [
        "task_success",
        "verification_score",
        "output_schema_valid",
    ]
    assert result.outputs[2].value == 0.0
    assert result.diagnostics
    assert result.diagnostics[0].message == "summary mismatch"
    assert result.diagnostics[0].details is not None
    assert "expected" in result.diagnostics[0].details
    assert result.diagnostics[0].details["actual"].strip() == "not the expected summary"
    assert "--- expected" in result.diagnostics[0].details["diff"]
    assert "+++ actual" in result.diagnostics[0].details["diff"]


def _score_task_metric(
    *,
    metric: Metric,
    task_dir: Path,
    artifacts: AgentArtifacts,
    task_config: AgenticUseTaskConfig,
    attempt: CapturedAgentAttempt,
) -> dict[str, list[float]]:
    row = build_evaluator_scoring_row(
        task_dir=task_dir,
        artifacts=artifacts,
        task_config=task_config,
        attempt=attempt,
    )
    scored = score_evaluator_rows([row], additional_metrics=[metric])
    if scored.row_scores and scored.row_scores[0].error:
        raise AssertionError(scored.row_scores[0].error)
    score_values: dict[str, list[float]] = {
        "task_success": [],
        "verification_score": [],
        "output_schema_valid": [],
    }
    for metric_scores in scored.row_scores[0].metrics.values():
        for score in metric_scores:
            if score.name in score_values:
                score_values[score.name].append(score.value)
    return score_values


def test_agent_target_metric_requires_runnable_agent_target_code(tmp_path: Path, agent_log_dir: Path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        """
version = "1.0"

[evaluator.surface]
constraint = "standalone_sdk"
allowed = ["standalone_sdk"]
forbidden = ["cli", "plugin_sdk", "legacy_service"]

[evaluator.expected]
required_terms = [
  "packages/nhx_evals_sdk",
  "Evaluator",
  "run_sync",
  "Agent",
  "nhx_evals_sdk.values.agents",
  "AgentFormat.GENERIC",
  "ExactMatchMetric",
  "response_path",
  "$.answer",
  "trajectory_path",
  "$.trajectory",
  "candidate_agent_runtime",
  "candidate_agent_model",
]
""".strip(),
        encoding="utf-8",
    )
    (task_dir / "instruction.md").write_text("Configure an agent target.", encoding="utf-8")
    (agent_log_dir / "final_message.txt").write_text(
        """
Use packages/nhx_evals_sdk with Evaluator.run_sync, Agent, AgentFormat.GENERIC, and ExactMatchMetric.
The metadata identifies candidate_agent_runtime and candidate_agent_model.

```python
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
import json

from nhx_evals_sdk import Evaluator, ExactMatchMetric
from nhx_evals_sdk.enums import AgentFormat
from nhx_evals_sdk.values.agents import Agent


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", "0")))
        body = {"answer": "4", "trajectory": {"steps": [{"tool_calls": [{"name": "calculator"}]}]}}
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


server = HTTPServer(("127.0.0.1", 0), Handler)
thread = Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    agent = Agent(
        url=f"http://127.0.0.1:{server.server_port}",
        name="mock-calculator-agent",
        format=AgentFormat.GENERIC,
        body={"prompt": "{{ prompt }}"},
        response_path="$.answer",
        trajectory_path="$.trajectory",
    )
    rows = [
        {
            "prompt": "What is 2+2?",
            "expected": "4",
            "candidate_agent_runtime": "mock-agent",
            "candidate_agent_model": "mock-calculator-v1",
        }
    ]
    result = Evaluator().run_sync(
        metrics=[ExactMatchMetric(reference="{{ item.expected }}")],
        dataset=rows,
        target=agent,
        prompt_template="{{ item.prompt }}",
    )
    payload = {
        "answer": result.row_scores[0].sample["output_text"],
        "exact_match": result.row_scores[0].metrics["exact-match"][0].value,
        "trajectory_tool_calls": result.row_scores[0].sample["trajectory"]["steps"][0]["tool_calls"],
        "candidate_agent_runtime": rows[0]["candidate_agent_runtime"],
        "candidate_agent_model": rows[0]["candidate_agent_model"],
    }
    print(json.dumps(payload, indent=2))
finally:
    server.shutdown()
```
""".strip(),
        encoding="utf-8",
    )
    artifacts = AgentArtifacts.from_dir(agent_log_dir)
    task_config = load_agentic_use_task_config(task_dir)
    attempt = capture_agent_attempt(task_dir=task_dir, artifacts=artifacts)
    module = _load_task_metrics("evaluator-standalone-sdk-agent-target")

    scores = _score_task_metric(
        metric=module.AgentTargetConfigurationMetric(),
        task_dir=task_dir,
        artifacts=artifacts,
        task_config=task_config,
        attempt=attempt,
    )

    assert scores["task_success"] == [1.0]
    assert scores["verification_score"] == [1.0]
    assert scores["output_schema_valid"] == [1.0]


def test_agent_target_metric_reads_workspace_python_artifact(tmp_path: Path, agent_log_dir: Path):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "agent_target_exact_match.py").write_text(
        "from nhx_evals_sdk import Evaluator\n"
        "from nhx_evals_sdk.values.agents import Agent\n"
        "Evaluator().run_sync(target=Agent(response_path='$.answer', trajectory_path='$.trajectory'))\n",
        encoding="utf-8",
    )
    artifacts = AgentArtifacts.from_dir(agent_log_dir, workspace_dir=workspace_dir)
    module = _load_task_metrics("evaluator-standalone-sdk-agent-target")

    code = module._extract_workspace_python_code(artifacts)

    assert code is not None
    assert "agent_target_exact_match.py" not in code
    assert "Evaluator().run_sync" in code


def test_surface_adherence_metric_authoring_runs_candidate_metric(tmp_path: Path, agent_log_dir: Path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        """
version = "1.0"

[evaluator.surface]
constraint = "standalone_sdk"
allowed = ["standalone_sdk"]
forbidden = ["cli", "plugin_sdk", "legacy_service"]

[evaluator.expected]
required_terms = [
  "packages/nhx_evals_sdk",
  "Metric",
  "type",
  "compute_scores",
  "score_names",
  "MetricResult",
  "MetricScore",
  "surface_adherence",
  "surface_violation_count",
  "observed_surfaces",
  "allowed_surfaces",
  "forbidden_surfaces",
]
""".strip(),
        encoding="utf-8",
    )
    (task_dir / "instruction.md").write_text("Author a surface-adherence metric.", encoding="utf-8")
    (agent_log_dir / "final_message.txt").write_text(
        """
Use packages/nhx_evals_sdk and implement a Metric-compatible class with compute_scores and score_names.

```python
from nhx_evals_sdk.values.results import MetricResult, MetricScore


class SurfaceAdherenceMetric:
    @property
    def type(self) -> str:
        return "agent_eval/surface_adherence"

    def score_names(self) -> list[str]:
        return ["surface_adherence", "surface_violation_count"]

    async def compute_scores(self, item: dict, sample: dict) -> MetricResult:
        observed_surfaces = set(item["observed_surfaces"])
        allowed_surfaces = set(item["allowed_surfaces"])
        forbidden_surfaces = set(item["forbidden_surfaces"])
        violations = (observed_surfaces - allowed_surfaces) | (observed_surfaces & forbidden_surfaces)
        adherence = 1.0 if not violations else 0.0
        return MetricResult(
            scores=[
                MetricScore(name="surface_adherence", value=adherence),
                MetricScore(name="surface_violation_count", value=float(len(violations))),
            ]
        )
```
""".strip(),
        encoding="utf-8",
    )
    artifacts = AgentArtifacts.from_dir(agent_log_dir)
    task_config = load_agentic_use_task_config(task_dir)
    attempt = capture_agent_attempt(task_dir=task_dir, artifacts=artifacts)
    module = _load_task_metrics("evaluator-standalone-sdk-surface-adherence-metric")

    scores = _score_task_metric(
        metric=module.SurfaceAdherenceMetricAuthoringMetric(),
        task_dir=task_dir,
        artifacts=artifacts,
        task_config=task_config,
        attempt=attempt,
    )

    assert scores["task_success"] == [1.0]
    assert scores["verification_score"] == [1.0]
    assert scores["output_schema_valid"] == [1.0]


def test_surface_adherence_metric_reads_workspace_python_artifact(tmp_path: Path, agent_log_dir: Path):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "surface_adherence_metric.py").write_text(
        "from nhx_evals_sdk.values.results import MetricResult, MetricScore\n"
        "class SurfaceAdherenceMetric:\n"
        "    async def compute_scores(self, item, sample):\n"
        "        return MetricResult(scores=[MetricScore(name='surface_adherence', value=1.0)])\n",
        encoding="utf-8",
    )
    artifacts = AgentArtifacts.from_dir(agent_log_dir, workspace_dir=workspace_dir)
    module = _load_task_metrics("evaluator-standalone-sdk-surface-adherence-metric")

    code = module._extract_workspace_python_code(artifacts)

    assert code is not None
    assert "compute_scores" in code
    assert "MetricScore" in code


def test_extract_marker_json_object_skips_trailing_malformed_marker():
    stdout = "\n".join(
        [
            "prefix",
            '__RESULT__={"score": 1}',
            '__RESULT__={"truncated"',
        ]
    )

    assert extract_marker_json_object(stdout, marker="__RESULT__=") == {"score": 1}
