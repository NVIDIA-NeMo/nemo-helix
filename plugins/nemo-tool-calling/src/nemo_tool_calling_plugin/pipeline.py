# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Baseline, diagnose, verify, report: the strategy without the platform plumbing."""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from nemo_tool_calling_plugin.analyzer import (
    Analysis,
    AnalysisContext,
    Analyzer,
    ConfigFix,
    SchemaFix,
    ToolDefinition,
    check_schema_fix,
)
from nemo_tool_calling_plugin.overlay_proxy import proxied_server_spec
from nemo_tool_calling_plugin.report import ResultFiles, recommendation, write_results
from nemo_tool_calling_plugin.schema_lint import lint_tool
from nemo_tool_calling_plugin.scoring import (
    DEFAULT_CALL_BUDGET,
    RunRecord,
    accepted_calls,
    annotate_schema_errors,
    loose_schema_findings,
    variant_summary,
)

logger = logging.getLogger(__name__)

VariantId = Literal["baseline", "config-fix", "schema-fix"]
VERIFY_VARIANTS: tuple[VariantId, ...] = ("config-fix", "schema-fix")
RESULT_VERSION = 1


@dataclass(frozen=True)
class Prompt:
    id: str
    prompt: str


class VariantRunner(Protocol):
    def run(self, variant: VariantId, agent_config: dict[str, Any], prompts: Sequence[Prompt]) -> list[RunRecord]: ...


@dataclass
class PipelineInputs:
    agent: str
    agent_config: dict[str, Any]
    prompts: list[Prompt]
    runs_per_prompt: int
    tools: list[ToolDefinition]
    overlay_dir: Path
    models: list[str] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)
    verify: tuple[VariantId, ...] = VERIFY_VARIANTS
    call_budget: int = DEFAULT_CALL_BUDGET


def system_prompt(config: dict[str, Any]) -> str:
    return str(((config.get("instructions") or {}).get("system") or {}).get("content") or "")


def with_system_prompt(config: dict[str, Any], prompt: str) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    updated.setdefault("instructions", {}).setdefault("system", {})["content"] = prompt
    return updated


def with_schema_overlay(
    config: dict[str, Any], fix: SchemaFix, overlay_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Route each patched stdio server through the overlay proxy; report servers that cannot be proxied."""
    updated = copy.deepcopy(config)
    servers: dict[str, Any] = (updated.get("mcp") or {}).get("servers") or {}
    findings: list[dict[str, Any]] = []
    by_server: dict[str, list[ToolDefinition]] = {}
    for tool in fix.tools:
        by_server.setdefault(tool.server, []).append(tool)
    for server, tools in by_server.items():
        spec = servers.get(server)
        transport = str((spec or {}).get("transport", "")).strip().lower()
        if spec is None or transport not in ("stdio", "command", "process"):
            findings.append(
                {
                    "id": f"schema-overlay-unsupported-{server}",
                    "source": "failure-analysis",
                    "severity": "low",
                    "tool": tools[0].name,
                    "title": f"Schema patch for server {server!r} was not verified",
                    "detail": "The schema-overlay proxy wraps stdio MCP servers only.",
                    "evidence": [],
                }
            )
            continue
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_dir / f"{server}.json"
        overlay_path.write_text(json.dumps([tool.to_mcp() for tool in tools], indent=2), encoding="utf-8")
        servers[server] = proxied_server_spec(spec, overlay_path)
    return updated, findings


def _advertised_schemas(tools: Sequence[ToolDefinition], fix: SchemaFix | None = None) -> dict[str, dict[str, Any]]:
    schemas = {tool.name: tool.input_schema for tool in tools}
    if fix is not None:
        schemas.update({tool.name: tool.input_schema for tool in fix.tools})
    return schemas


def _variant(
    variant: VariantId,
    label: str,
    description: str,
    runs: Sequence[RunRecord],
    *,
    schemas: dict[str, dict[str, Any]],
    call_budget: int,
) -> dict[str, Any]:
    annotate_schema_errors(runs, schemas)
    return {"id": variant, "label": label, "description": description, **variant_summary(runs, call_budget)}


def _analysis_findings(analysis: Analysis) -> list[dict[str, Any]]:
    findings = [
        {
            "id": f"analysis-{index + 1}",
            "source": "failure-analysis",
            "severity": finding.severity,
            "tool": finding.tool,
            "title": finding.title,
            "detail": finding.detail,
            "evidence": list(finding.evidence),
        }
        for index, finding in enumerate(analysis.findings)
    ]
    findings.extend(
        {
            "id": f"analysis-error-{index + 1}",
            "source": "failure-analysis",
            "severity": "low",
            "tool": None,
            "title": "Analyzer call failed",
            "detail": error,
            "evidence": [],
        }
        for index, error in enumerate(analysis.errors)
    )
    return findings


def _suggestions(
    config_fix: ConfigFix | None, schema_fix: SchemaFix | None, ran: set[str], files: ResultFiles
) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    if config_fix is not None:
        suggestions.append(
            {
                "id": "config-fix",
                "kind": "agent-config",
                "title": config_fix.title,
                "rationale": config_fix.rationale,
                "applies_automatically": True,
                "artifact": files.candidate_agent,
                "diff_artifact": files.candidate_agent_diff,
                "verified_by_variant": "config-fix" if "config-fix" in ran else None,
            }
        )
    if schema_fix is not None:
        suggestions.append(
            {
                "id": "schema-fix",
                "kind": "tool-schema",
                "title": schema_fix.title,
                "rationale": schema_fix.rationale,
                "applies_automatically": False,
                "artifact": files.schema_patches[0] if len(files.schema_patches) == 1 else "schema-patch/",
                "diff_artifact": files.schema_patch_diff,
                "verified_by_variant": "schema-fix" if "schema-fix" in ran else None,
            }
        )
    return suggestions


def run_pipeline(
    inputs: PipelineInputs, *, runner: VariantRunner, analyzer: Analyzer, results_dir: Path
) -> dict[str, Any]:
    base = inputs.agent_config
    logger.info("Baseline: %d prompt(s) x %d run(s)", len(inputs.prompts), inputs.runs_per_prompt)
    baseline = runner.run("baseline", base, inputs.prompts)
    published = _advertised_schemas(inputs.tools)
    budget = inputs.call_budget
    variants = [
        _variant(
            "baseline", "Current agent", "The agent as registered.", baseline, schemas=published, call_budget=budget
        )
    ]

    lint = [finding.to_finding() for tool in inputs.tools for finding in lint_tool(tool.to_mcp())]
    lint = [*loose_schema_findings(baseline), *lint]
    baseline_summary = variants[0]
    analysis = analyzer.analyze(
        AnalysisContext(
            agent=inputs.agent,
            system_prompt=system_prompt(base),
            tools=inputs.tools,
            metrics=baseline_summary["metrics"],
            rejection_reasons=baseline_summary["rejection_reasons"],
            examples=baseline_summary["examples"],
            lint_findings=lint,
            models=inputs.models,
            providers=inputs.providers,
        )
    )
    findings = [*lint, *_analysis_findings(analysis)]

    schema_fix = analysis.schema_fix
    if schema_fix is not None:
        accepted = {tool.name: accepted_calls(baseline, tool.name) for tool in schema_fix.tools}
        schema_fix, check_findings = check_schema_fix(schema_fix, accepted)
        findings.extend(check_findings)

    candidate = with_system_prompt(base, analysis.config_fix.system_prompt) if analysis.config_fix else base
    ran: set[str] = {"baseline"}
    if analysis.config_fix is not None and "config-fix" in inputs.verify:
        logger.info("Verifying the config fix")
        runs = runner.run("config-fix", candidate, inputs.prompts)
        description = "The agent with the suggested prompt edit."
        variants.append(_variant("config-fix", "Config fix", description, runs, schemas=published, call_budget=budget))
        ran.add("config-fix")
    if schema_fix is not None and "schema-fix" in inputs.verify:
        overlaid, overlay_findings = with_schema_overlay(candidate, schema_fix, inputs.overlay_dir)
        findings.extend(overlay_findings)
        if not overlay_findings:
            logger.info("Verifying the schema patch through the schema-overlay proxy")
            runs = runner.run("schema-fix", overlaid, inputs.prompts)
            description = (
                "The config fix plus the patched tool schema, served through a schema-overlay proxy."
                if analysis.config_fix is not None
                else "The agent with the patched tool schema, served through a schema-overlay proxy."
            )
            patched = _advertised_schemas(inputs.tools, schema_fix)
            variants.append(
                _variant("schema-fix", "Config + schema fix", description, runs, schemas=patched, call_budget=budget)
            )
            ran.add("schema-fix")

    files = write_results(
        results_dir,
        original_config=base,
        candidate_config=candidate if analysis.config_fix is not None else None,
        tools=inputs.tools,
        schema_fix=schema_fix,
    )
    result: dict[str, Any] = {
        "version": RESULT_VERSION,
        "agent": inputs.agent,
        "analyzer_model": analyzer.model,
        "dataset": {"prompts": len(inputs.prompts), "runs_per_prompt": inputs.runs_per_prompt},
        "variants": variants,
        "findings": findings,
        "suggestions": _suggestions(analysis.config_fix, schema_fix, ran, files),
        "recommendation": recommendation(variants, has_config_fix=analysis.config_fix is not None),
    }
    files.write_summary(result)
    return result
