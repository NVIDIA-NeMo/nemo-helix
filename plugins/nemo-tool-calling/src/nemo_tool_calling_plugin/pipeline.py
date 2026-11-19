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
    STEP_LABELS,
    Analysis,
    AnalysisContext,
    AnalysisStep,
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
    traced_baseline: list[RunRecord] | None = None
    trace_window: dict[str, str] | None = None

    def dataset(self) -> dict[str, Any]:
        replay = {"prompts": len(self.prompts), "runs_per_prompt": self.runs_per_prompt}
        if self.traced_baseline is None:
            return {"source": "dataset", **replay}
        return {"source": "traces", "sessions": len(self.traced_baseline), "window": self.trace_window, **replay}


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
    return findings


def _analyzer_error_findings(analysis: Analysis) -> list[dict[str, Any]]:
    return [
        {
            "id": f"analysis-error-{failure.step}",
            "source": "failure-analysis",
            "severity": "high",
            "tool": None,
            "title": f"Analyzer failed, so no {STEP_LABELS[failure.step]} was proposed",
            "detail": f"{failure.cause()[0].upper()}{failure.cause()[1:]}. {failure.remedy()}",
            "evidence": [failure.message],
        }
        for failure in analysis.errors
    ]


NO_REPLAY_PROMPTS = "no user prompt could be recovered from the traced sessions; pass a prompt dataset to verify it"


def _gap_notes(
    analysis: Analysis,
    *,
    schema_fix: SchemaFix | None,
    ran: set[str],
    unverified_reasons: dict[str, str],
) -> list[str]:
    """Why a suggestion is missing or was not verified, one sentence each."""
    failed = {failure.step: failure for failure in analysis.errors}
    notes: list[str] = []
    if len(failed) == len(STEP_LABELS) and len({failure.cause() for failure in failed.values()}) == 1:
        failure = failed["config-fix"]
        notes.append(f"No fix was proposed: {failure.cause()}. {failure.remedy()}")
    else:
        notes.extend(
            f"No {STEP_LABELS[step]} was proposed: {failure.cause()}. {failure.remedy()}"
            for step, failure in failed.items()
        )
    if not failed and analysis.config_fix is None and analysis.schema_fix is None:
        notes.append("The analyzer proposed no fix; see the findings.")
    if analysis.schema_fix is not None and schema_fix is None:
        notes.append("The schema patch was dropped because it is not valid JSON Schema; see the findings.")
    proposed: dict[AnalysisStep, bool] = {
        "config-fix": analysis.config_fix is not None,
        "schema-fix": schema_fix is not None,
    }
    unverified = [step for step, made in proposed.items() if made and step not in ran]
    reasons = {unverified_reasons.get(step, "") for step in unverified}
    if len(unverified) == 2 and len(reasons) == 1:
        notes.append(f"The prompt fix and schema patch were proposed but not verified: {reasons.pop()}.")
    else:
        notes.extend(
            f"The {STEP_LABELS[step]} was proposed but not verified: {unverified_reasons.get(step, 'see the findings')}."
            for step in unverified
        )
    return notes


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
    if inputs.traced_baseline is None:
        logger.info("Baseline: %d prompt(s) x %d run(s)", len(inputs.prompts), inputs.runs_per_prompt)
        baseline = runner.run("baseline", base, inputs.prompts)
        description = "The agent as registered."
    else:
        baseline = inputs.traced_baseline
        description = f"The agent as registered, scored from {len(baseline)} traced session(s) in Intake."
    published = _advertised_schemas(inputs.tools)
    budget = inputs.call_budget
    variants = [_variant("baseline", "Current agent", description, baseline, schemas=published, call_budget=budget)]

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
    findings = [*_analyzer_error_findings(analysis), *lint, *_analysis_findings(analysis)]
    verify = inputs.verify if inputs.prompts else ()
    unverified_reasons: dict[str, str] = {
        step: NO_REPLAY_PROMPTS if not inputs.prompts else "`verify` leaves it out"
        for step in VERIFY_VARIANTS
        if step not in verify
    }
    if not inputs.prompts and inputs.verify:
        findings.append(
            {
                "id": "no-replay-prompts",
                "source": "failure-analysis",
                "severity": "low",
                "tool": None,
                "title": "Suggestions were not verified",
                "detail": "No user prompt could be recovered from the traced sessions, so nothing was replayed. "
                "Pass a prompt dataset to verify the suggestions.",
                "evidence": [],
            }
        )

    schema_fix = analysis.schema_fix
    if schema_fix is not None:
        accepted = {tool.name: accepted_calls(baseline, tool.name) for tool in schema_fix.tools}
        schema_fix, check_findings = check_schema_fix(schema_fix, accepted)
        findings.extend(check_findings)

    candidate = with_system_prompt(base, analysis.config_fix.system_prompt) if analysis.config_fix else base
    ran: set[str] = {"baseline"}
    if analysis.config_fix is not None and "config-fix" in verify:
        logger.info("Verifying the config fix")
        runs = runner.run("config-fix", candidate, inputs.prompts)
        description = "The agent with the suggested prompt edit."
        variants.append(_variant("config-fix", "Config fix", description, runs, schemas=published, call_budget=budget))
        ran.add("config-fix")
    if schema_fix is not None and "schema-fix" in verify:
        overlaid, overlay_findings = with_schema_overlay(candidate, schema_fix, inputs.overlay_dir)
        findings.extend(overlay_findings)
        if overlay_findings:
            unverified_reasons["schema-fix"] = "the schema-overlay proxy can only serve stdio MCP servers"
        else:
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
    advice = recommendation(
        variants,
        has_config_fix=analysis.config_fix is not None,
        notes=_gap_notes(analysis, schema_fix=schema_fix, ran=ran, unverified_reasons=unverified_reasons),
    )
    result: dict[str, Any] = {
        "version": RESULT_VERSION,
        "agent": inputs.agent,
        "analyzer_model": analyzer.model,
        "dataset": inputs.dataset(),
        "variants": variants,
        "findings": findings,
        "suggestions": _suggestions(analysis.config_fix, schema_fix, ran, files),
        "recommended_variant": advice.variant,
        "recommendation": advice.text,
    }
    files.write_summary(result)
    return result
