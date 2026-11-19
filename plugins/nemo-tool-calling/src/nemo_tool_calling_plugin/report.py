# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write the results files described in CONTRACT.md."""

from __future__ import annotations

import difflib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from nemo_tool_calling_plugin.analyzer import SchemaFix, ToolDefinition

RESULT_FILENAME = "tool-calling-result.json"
REPORT_FILENAME = "report.md"
CANDIDATE_AGENT_FILENAME = "candidate-agent.yaml"
CANDIDATE_AGENT_DIFF_FILENAME = "candidate-agent.diff"
SCHEMA_PATCH_DIR = "schema-patch"
SCHEMA_PATCH_DIFF_FILENAME = "schema-patch.diff"


@dataclass
class ResultFiles:
    root: Path
    candidate_agent: str | None = None
    candidate_agent_diff: str | None = None
    schema_patches: list[str] = field(default_factory=list)
    schema_patch_diff: str | None = None

    def write_summary(self, result: dict[str, Any]) -> None:
        (self.root / RESULT_FILENAME).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (self.root / REPORT_FILENAME).write_text(render_markdown(result), encoding="utf-8")


def _yaml(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=120)


def _tool_json(tool: ToolDefinition) -> str:
    return json.dumps(tool.to_mcp(), indent=2, ensure_ascii=False) + "\n"


def _diff(before: str, after: str, *, name: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile=f"a/{name}", tofile=f"b/{name}"
        )
    )


def write_results(
    root: Path,
    *,
    original_config: dict[str, Any],
    candidate_config: dict[str, Any] | None,
    tools: Sequence[ToolDefinition],
    schema_fix: SchemaFix | None,
) -> ResultFiles:
    root.mkdir(parents=True, exist_ok=True)
    files = ResultFiles(root=root)
    if candidate_config is not None:
        candidate = _yaml(candidate_config)
        (root / CANDIDATE_AGENT_FILENAME).write_text(candidate, encoding="utf-8")
        (root / CANDIDATE_AGENT_DIFF_FILENAME).write_text(
            _diff(_yaml(original_config), candidate, name=CANDIDATE_AGENT_FILENAME), encoding="utf-8"
        )
        files.candidate_agent = CANDIDATE_AGENT_FILENAME
        files.candidate_agent_diff = CANDIDATE_AGENT_DIFF_FILENAME
    if schema_fix is not None:
        published = {(tool.server, tool.name): tool for tool in tools}
        diffs: list[str] = []
        for tool in schema_fix.tools:
            relative = f"{SCHEMA_PATCH_DIR}/{tool.server}/{tool.name}.json"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            patched = _tool_json(tool)
            path.write_text(patched, encoding="utf-8")
            files.schema_patches.append(relative)
            original = published.get((tool.server, tool.name))
            diffs.append(
                _diff(_tool_json(original) if original else "", patched, name=f"{tool.server}/{tool.name}.json")
            )
        (root / SCHEMA_PATCH_DIFF_FILENAME).write_text("".join(diffs), encoding="utf-8")
        files.schema_patch_diff = SCHEMA_PATCH_DIFF_FILENAME
    return files


def _percent(rate: float) -> str:
    return f"{round(rate * 100)}%"


@dataclass(frozen=True)
class Recommendation:
    text: str
    variant: Literal["config-fix", "schema-fix"] | None


def recommendation(
    variants: Sequence[dict[str, Any]], *, has_config_fix: bool, notes: Sequence[str] = ()
) -> Recommendation:
    """What to do, from the verified variants, followed by why any suggestion is missing or unverified.

    The sentence and ``variant`` come from the same comparisons, so the text never recommends a variant other than
    the one ``recommended_variant`` names.
    """
    by_id = {variant["id"]: variant["metrics"] for variant in variants}
    baseline = by_id["baseline"]["tool_error_rate"]
    config = by_id.get("config-fix")
    schema = by_id.get("schema-fix")
    parts: list[str] = []
    recommended: Literal["config-fix", "schema-fix"] | None = None
    if config is None and schema is None:
        parts.append(
            "No rejected tool calls in the baseline."
            if baseline == 0
            else f"The baseline rejects {_percent(baseline)} of tool calls."
        )
    if config is not None and config["tool_error_rate"] < baseline:
        recommended = "config-fix"
        parts.append(
            f"Apply the prompt fix now: it cuts the tool error rate from {_percent(baseline)} to "
            f"{_percent(config['tool_error_rate'])}."
        )
    elif config is not None:
        parts.append(
            f"The prompt fix did not lower the tool error rate ({_percent(baseline)} -> "
            f"{_percent(config['tool_error_rate'])}); don't apply it as is."
        )
    if schema is not None:
        best_so_far = min(baseline, config["tool_error_rate"]) if config is not None else baseline
        together = "together with the prompt fix " if has_config_fix else ""
        if schema["tool_error_rate"] < best_so_far:
            recommended = "schema-fix"
            parts.append(
                f"Send the schema patch to the tool's owner: {together}it brings the tool error rate to "
                f"{_percent(schema['tool_error_rate'])}."
            )
        else:
            parts.append(f"The schema patch did not help further ({_percent(schema['tool_error_rate'])}).")
    if baseline == 0 and config is None and schema is None and not notes:
        parts.append("Nothing to fix.")
    parts.extend(notes)
    return Recommendation(text=" ".join(parts), variant=recommended)


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# Tool-calling report: {result['agent']}",
        "",
        result["recommendation"],
        "",
        f"Recommended variant: {_variant_label(result)}.",
        "",
        f"Dataset: {result['dataset']['prompts']} prompts x {result['dataset']['runs_per_prompt']} runs. "
        f"Analyzer: `{result['analyzer_model']}`.",
        "",
        "## Variants",
        "",
        "| Variant | Runs | Errored runs | Tool calls | Rejected | Error rate | Schema-invalid | Retry-loop runs "
        "| First call OK | Task success |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for variant in result["variants"]:
        m = variant["metrics"]
        lines.append(
            f"| {variant['label']} | {m['runs']} | {m['runs_with_errors']} | {m['tool_calls']} | "
            f"{m['rejected_calls']} | {_percent(m['tool_error_rate'])} | {m['signals']['schema_invalid']} | "
            f"{m['signals']['retry_loop_runs']} | {_percent(m['first_call_accepted_rate'])} | "
            f"{_percent(m['task_success_rate'])} |"
        )
    for variant in result["variants"]:
        if not variant["rejection_reasons"]:
            continue
        lines += ["", f"### Rejection reasons: {variant['label']}", ""]
        lines += [f"- {reason}: {count}" for reason, count in variant["rejection_reasons"].items()]
    lines += ["", "## Findings", ""]
    for finding in result["findings"]:
        lines.append(f"- **{finding['severity']}** ({finding['source']}) {finding['title']}: {finding['detail']}")
    lines += ["", "## Suggestions", ""]
    for suggestion in result["suggestions"]:
        verified = suggestion.get("verified_by_variant") or "not verified"
        lines.append(
            f"- **{suggestion['title']}** ({suggestion['kind']}, verified by: {verified}). "
            f"{suggestion['rationale']} Artifact: `{suggestion['artifact']}`, diff: `{suggestion['diff_artifact']}`."
        )
    return "\n".join(lines) + "\n"


def _variant_label(result: dict[str, Any]) -> str:
    recommended = result["recommended_variant"]
    if recommended is None:
        return "none"
    label = next((variant["label"] for variant in result["variants"] if variant["id"] == recommended), recommended)
    return f"**{label}** (`{recommended}`)"
