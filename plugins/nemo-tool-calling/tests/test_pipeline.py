# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml
from nemo_tool_calling_plugin.analyzer import (
    Analysis,
    AnalysisContext,
    AnalyzerFinding,
    ConfigFix,
    SchemaFix,
    ToolDefinition,
)
from nemo_tool_calling_plugin.overlay_proxy import PROXY_MODULE
from nemo_tool_calling_plugin.pipeline import PipelineInputs, Prompt, VariantId, run_pipeline
from nemo_tool_calling_plugin.scoring import RunRecord, ToolCall

SERVER = "data_designer"
TOOL = "generate_data_designer_job_request"
REJECTED = "TOOL CALL REJECTED: 422 Unprocessable Entity:\n- columns[0] (review): unknown field 'sampler_type '"
ACCEPTED = '{"status": "accepted"}'
GOOD_ARGS = {
    "job_request": {
        "name": "reviews",
        "spec": {
            "num_records": 10,
            "config": {
                "model_configs": [],
                "columns": [{"name": "id", "column_type": "sampler", "sampler_type": "uuid", "params": {}}],
            },
        },
    }
}  # noqa: E501
BAD_ARGS = {
    "job_request": {
        "spec": {
            "num_records": 10,
            "config": {"columns": [{"name": "review", "column_type": "sampler", "sampler_type ": "uuid"}]},
        }
    }
}  # noqa: E501

_RATE = {"type": "number", "minimum": 0, "maximum": 1}
_METRICS = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "runs",
        "runs_with_errors",
        "tool_calls",
        "rejected_calls",
        "tool_error_rate",
        "signals",
        "first_call_accepted_rate",
        "task_success_rate",
        "avg_calls_per_run",
    ],
    "properties": {
        "runs": {"type": "integer"},
        "runs_with_errors": {"type": "integer"},
        "tool_calls": {"type": "integer"},
        "rejected_calls": {"type": "integer"},
        "tool_error_rate": _RATE,
        "signals": {
            "type": "object",
            "additionalProperties": False,
            "required": ["tool_error", "schema_invalid", "error_text", "retry_loop_runs"],
            "properties": {
                key: {"type": "integer"} for key in ("tool_error", "schema_invalid", "error_text", "retry_loop_runs")
            },
        },
        "first_call_accepted_rate": _RATE,
        "task_success_rate": _RATE,
        "avg_calls_per_run": {"type": "number"},
    },
}
_FINDING = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "source", "severity", "tool", "title", "detail", "evidence"],
    "properties": {
        "id": {"type": "string"},
        "source": {"enum": ["schema-lint", "failure-analysis"]},
        "severity": {"enum": ["high", "medium", "low"]},
        "tool": {"type": ["string", "null"]},
        "title": {"type": "string"},
        "detail": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
}
RESULT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "version",
        "agent",
        "analyzer_model",
        "dataset",
        "variants",
        "findings",
        "suggestions",
        "recommendation",
    ],
    "properties": {
        "version": {"const": 1},
        "agent": {"type": "string"},
        "analyzer_model": {"type": "string"},
        "dataset": {
            "type": "object",
            "additionalProperties": False,
            "required": ["prompts", "runs_per_prompt"],
            "properties": {"prompts": {"type": "integer"}, "runs_per_prompt": {"type": "integer"}},
        },
        "variants": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "label", "description", "metrics", "rejection_reasons", "per_prompt", "examples"],
                "properties": {
                    "id": {"enum": ["baseline", "config-fix", "schema-fix"]},
                    "label": {"type": "string"},
                    "description": {"type": "string"},
                    "metrics": _METRICS,
                    "rejection_reasons": {"type": "object", "additionalProperties": {"type": "integer"}},
                    "per_prompt": {"type": "object", "additionalProperties": _METRICS},
                    "examples": {
                        "type": "array",
                        "maxItems": 5,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "prompt_id",
                                "tool",
                                "signals",
                                "arguments_excerpt",
                                "rejection",
                                "schema_errors",
                            ],
                            "properties": {
                                "prompt_id": {"type": "string"},
                                "tool": {"type": "string"},
                                "signals": {
                                    "type": "array",
                                    "items": {"enum": ["tool_error", "schema_invalid", "error_text"]},
                                },
                                "arguments_excerpt": {"type": "string"},
                                "rejection": {"type": "string"},
                                "schema_errors": {"type": "array", "items": {"type": "string"}},
                            },
                        },
                    },
                },
            },
        },
        "findings": {"type": "array", "items": _FINDING},
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "kind",
                    "title",
                    "rationale",
                    "applies_automatically",
                    "artifact",
                    "diff_artifact",
                    "verified_by_variant",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "kind": {"enum": ["agent-config", "tool-schema"]},
                    "title": {"type": "string"},
                    "rationale": {"type": "string"},
                    "applies_automatically": {"type": "boolean"},
                    "artifact": {"type": "string"},
                    "diff_artifact": {"type": "string"},
                    "verified_by_variant": {"enum": ["config-fix", "schema-fix", None]},
                },
            },
        },
        "recommendation": {"type": "string"},
    },
}


class FakeRunner:
    def __init__(self, rejected_by_variant: dict[str, int], runs_per_prompt: int) -> None:
        self._rejected = rejected_by_variant
        self._runs_per_prompt = runs_per_prompt
        self.configs: dict[str, dict[str, Any]] = {}

    def run(self, variant: VariantId, agent_config: dict[str, Any], prompts: Sequence[Prompt]) -> list[RunRecord]:
        self.configs[variant] = agent_config
        rejected = self._rejected[variant]
        runs: list[RunRecord] = []
        for prompt in prompts:
            for attempt in range(1, self._runs_per_prompt + 1):
                calls = [ToolCall(TOOL, BAD_ARGS, REJECTED) for _ in range(rejected)]
                calls.append(ToolCall(TOOL, GOOD_ARGS, ACCEPTED))
                runs.append(RunRecord(prompt.id, attempt, calls))
        return runs


class FakeAnalyzer:
    model = "default/fake-analyzer"

    def __init__(self, analysis: Analysis) -> None:
        self._analysis = analysis
        self.context: AnalysisContext | None = None

    def analyze(self, context: AnalysisContext) -> Analysis:
        self.context = context
        return self._analysis


@pytest.fixture
def agent_config() -> dict[str, Any]:
    return {
        "config_format": "nemo-agents-spec-v1",
        "name": "data-designer-toolcall",
        "instructions": {"system": {"content": "Call the tool."}},
        "mcp": {
            "servers": {SERVER: {"transport": "stdio", "url": "python", "args": ["../mcps/dd_job_request_server.py"]}}
        },
    }


def _inputs(
    agent_config: dict[str, Any], pre_fix_tool: dict[str, Any], tmp_path: Path, **overrides: Any
) -> PipelineInputs:
    tool = ToolDefinition(SERVER, TOOL, pre_fix_tool["description"], pre_fix_tool["inputSchema"])
    fields: dict[str, Any] = {
        "agent": "default/data-designer-toolcall",
        "agent_config": agent_config,
        "prompts": [Prompt("product-reviews", "reviews"), Prompt("orders", "orders")],
        "runs_per_prompt": 2,
        "tools": [tool],
        "overlay_dir": tmp_path / "overlays",
        "models": ["default/nemotron3-5-lightning"],
    }
    return PipelineInputs(**{**fields, **overrides})


def _analysis(post_fix_tool: dict[str, Any]) -> Analysis:
    return Analysis(
        findings=[AnalyzerFinding(title="Keys corrupted", detail="Open column object", severity="high", tool=TOOL)],
        config_fix=ConfigFix(
            title="Name the model", rationale="Invented providers.", system_prompt="Call the tool. Use default/m."
        ),
        schema_fix=SchemaFix(
            title="One schema per column type",
            rationale="Order-constrained grammars.",
            tools=(ToolDefinition(SERVER, TOOL, post_fix_tool["description"], post_fix_tool["inputSchema"]),),
        ),
    )


def test_pipeline_writes_every_contract_file(
    tmp_path: Path, agent_config: dict[str, Any], pre_fix_tool: dict[str, Any], post_fix_tool: dict[str, Any]
) -> None:
    runner = FakeRunner({"baseline": 3, "config-fix": 1, "schema-fix": 0}, runs_per_prompt=2)
    analyzer = FakeAnalyzer(_analysis(post_fix_tool))
    results = tmp_path / "results"

    result = run_pipeline(
        _inputs(agent_config, pre_fix_tool, tmp_path), runner=runner, analyzer=analyzer, results_dir=results
    )

    on_disk = json.loads((results / "tool-calling-result.json").read_text())
    assert on_disk == result
    jsonschema.validate(result, RESULT_SCHEMA)
    assert [variant["id"] for variant in result["variants"]] == ["baseline", "config-fix", "schema-fix"]
    rates = [variant["metrics"]["tool_error_rate"] for variant in result["variants"]]
    assert rates == [0.75, 0.5, 0.0]
    assert result["analyzer_model"] == "default/fake-analyzer"
    assert result["dataset"] == {"prompts": 2, "runs_per_prompt": 2}
    assert "75%" in result["recommendation"] and "0%" in result["recommendation"]
    assert {path.relative_to(results).as_posix() for path in results.rglob("*") if path.is_file()} == {
        "tool-calling-result.json",
        "report.md",
        "candidate-agent.yaml",
        "candidate-agent.diff",
        f"schema-patch/{SERVER}/{TOOL}.json",
        "schema-patch.diff",
    }
    assert yaml.safe_load((results / "candidate-agent.yaml").read_text())["instructions"]["system"]["content"] == (
        "Call the tool. Use default/m."
    )
    assert "+    content: Call the tool. Use default/m." in (results / "candidate-agent.diff").read_text()
    assert (results / "schema-patch.diff").read_text().startswith(f"--- a/{SERVER}/{TOOL}.json")
    patch = json.loads((results / f"schema-patch/{SERVER}/{TOOL}.json").read_text())
    assert set(patch) == {"name", "description", "inputSchema"}
    assert [s["verified_by_variant"] for s in result["suggestions"]] == ["config-fix", "schema-fix"]
    assert result["suggestions"][1]["artifact"] == f"schema-patch/{SERVER}/{TOOL}.json"
    assert "# Tool-calling report: default/data-designer-toolcall" in (results / "report.md").read_text()


def test_pipeline_signals_follow_the_schema_each_variant_advertised(
    tmp_path: Path, agent_config: dict[str, Any], pre_fix_tool: dict[str, Any], post_fix_tool: dict[str, Any]
) -> None:
    runner = FakeRunner({"baseline": 1, "config-fix": 1, "schema-fix": 1}, runs_per_prompt=1)

    result = run_pipeline(
        _inputs(agent_config, pre_fix_tool, tmp_path, runs_per_prompt=1),
        runner=runner,
        analyzer=FakeAnalyzer(_analysis(post_fix_tool)),
        results_dir=tmp_path / "results",
    )

    signals = {variant["id"]: variant["metrics"]["signals"] for variant in result["variants"]}
    assert signals["baseline"]["schema_invalid"] == 0
    assert signals["schema-fix"]["schema_invalid"] == 2
    assert signals["baseline"]["retry_loop_runs"] == 2
    loose = [
        finding for finding in result["findings"] if finding["title"] == "Schema too loose to catch its own errors"
    ]
    assert len(loose) == 1


def test_schema_fix_variant_runs_behind_the_overlay_proxy(
    tmp_path: Path, agent_config: dict[str, Any], pre_fix_tool: dict[str, Any], post_fix_tool: dict[str, Any]
) -> None:
    runner = FakeRunner({"baseline": 1, "config-fix": 1, "schema-fix": 0}, runs_per_prompt=1)

    run_pipeline(
        _inputs(agent_config, pre_fix_tool, tmp_path),
        runner=runner,
        analyzer=FakeAnalyzer(_analysis(post_fix_tool)),
        results_dir=tmp_path / "results",
    )

    server = runner.configs["schema-fix"]["mcp"]["servers"][SERVER]
    assert PROXY_MODULE in server["args"]
    assert server["args"][-2:] == ["python", "../mcps/dd_job_request_server.py"]
    overlay = json.loads(Path(server["args"][server["args"].index("--overlay") + 1]).read_text())
    assert overlay[0]["inputSchema"] == post_fix_tool["inputSchema"]
    assert runner.configs["schema-fix"]["instructions"]["system"]["content"] == "Call the tool. Use default/m."
    assert runner.configs["baseline"] is agent_config


def test_pipeline_without_suggestions_runs_only_the_baseline(
    tmp_path: Path, agent_config: dict[str, Any], pre_fix_tool: dict[str, Any]
) -> None:
    runner = FakeRunner({"baseline": 2}, runs_per_prompt=1)
    analyzer = FakeAnalyzer(Analysis(errors=["Config-fix analysis failed: down"]))

    result = run_pipeline(
        _inputs(agent_config, pre_fix_tool, tmp_path), runner=runner, analyzer=analyzer, results_dir=tmp_path / "r"
    )

    jsonschema.validate(result, RESULT_SCHEMA)
    assert [variant["id"] for variant in result["variants"]] == ["baseline"]
    assert result["suggestions"] == []
    assert any(finding["title"] == "Analyzer call failed" for finding in result["findings"])
    assert sorted(path.name for path in (tmp_path / "r").iterdir()) == ["report.md", "tool-calling-result.json"]
    assert analyzer.context is not None and analyzer.context.models == ["default/nemotron3-5-lightning"]
