# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM-assisted gap analysis using the Anthropic API."""

import json
import os
import re
from datetime import datetime, timezone

import anthropic
from nemo_agents_plugin.improvement.analysis.mechanical import cluster_evals, mechanical_analysis
from nemo_agents_plugin.improvement.models import (
    BaselineEntry,
    BatchResult,
    EvalCluster,
    GapAnalysis,
    GapCategory,
    Hypothesis,
    MechanicalAnalysis,
    to_json,
)
from nemo_agents_plugin.improvement.traces.base import TraceParser, TraceSummary

# Schema description for Claude's structured output
HYPOTHESIS_SCHEMA = """
Output a JSON array with ONE hypothesis object per cluster. Each object must have:
{
  "cluster_id": "string - the cluster_id from the clusters above",
  "eval_names": ["list of eval names in this cluster"],
  "root_cause": "string - shared root cause across the cluster",
  "category": "string - one of: missing_skill, inadequate_skill, cli_gap, cli_ergonomics, fundamental",
  "proposed_fix": "string - specific change to make (which file, what content)",
  "affected_files": ["list of file paths to modify"],
  "expected_impact": "string - expected improvement across the cluster",
  "confidence": 0.8  // float 0.0-1.0
}

Categories explained:
- missing_skill: No skill exists for this workflow. Fix: create a new skill file under the agent's skills path.
- inadequate_skill: A skill exists but is incomplete/misleading. Fix: update the skill.
- cli_gap: The agent's CLI/tooling doesn't expose the right command or flag. Fix: update the CLI/tooling.
- cli_ergonomics: Command exists but output is hard for agents to parse. Fix: improve output format.
- fundamental: Not addressable via skills/tooling changes (model capability limit, etc.). Note: still include these but with low confidence.

IMPORTANT: Do NOT suggest changes to eval definitions (instruction.md, task.toml, workflow.yml,
test_outputs.py), Dockerfiles, or anything inside the evals directory. The goal is to improve
the agent (its skills, its tooling) so it performs better on the evals as they are.

Output one hypothesis per cluster. Skip clusters where no actionable fix exists.
""".strip()


def _build_analysis_prompt(
    mechanical: MechanicalAnalysis,
    clusters: list[EvalCluster],
    trace_excerpts: dict[str, str],
    skills_path: str = ".agents/skills",
) -> str:
    """Build the prompt for the LLM to analyze eval results.

    The prompt is agent-agnostic — it does not assume a specific CLI tool or
    repo layout. The only project-specific input is *skills_path*, which tells
    the analyzer where the writable skills directory lives.
    """
    return f"""You are analyzing eval results for an agent under improvement.
Each eval is a containerized task: an instruction, a verifier, and a pass/fail outcome.

## Mechanical Analysis (computed from results)

{to_json(mechanical)}

## Eval Clusters (grouped by shared error patterns or optimization signals)

{_format_clusters(clusters)}

## Session Trace Excerpts (from worst-performing evals)

{_format_trace_excerpts(trace_excerpts)}

## Available Improvement Levers

1. **Agent skills** under `{skills_path}` — markdown files that provide contextual guidance to the agent when triggered by keywords or task patterns. **This is the ONLY writable scope for the nemo-optimize-skills strategy.**
2. (Future strategies may also touch CLI tooling, agent config, or prompts — but the current loop only writes skills.)

## Your Task

For EACH cluster above, analyze the shared root cause and propose ONE targeted fix that lives under `{skills_path}`.

For **failing** clusters (signal: error_pattern, unclustered, missing_skill):
- What error patterns the evals in each cluster share
- Whether a skill would have prevented the shared failure mode
- Whether the agent went down a wrong path that a skill could redirect

For **optimization** clusters (signal: new_eval, slow_outlier, tool_heavy, token_heavy):
- The eval already PASSES — your hypothesis should reduce wallclock time, tool calls, or token usage WITHOUT regressing correctness
- Where the agent wastes time: unnecessary tool calls, retries, exploring wrong paths, re-reading context
- Whether a skill with concrete examples would let the agent skip discovery steps
- Whether existing skill content is too long / verbose, inflating token usage
- Whether the agent uses an inefficient approach that a skill hint could correct
- For `token_heavy`: focus on skill conciseness and giving the agent a direct path to the answer
- For `slow_outlier`: focus on reducing back-and-forth exploration via better skills
- For `tool_heavy`: focus on collapsing multi-step probing into a single direct command

All file paths in `affected_files` must be under `{skills_path}`. The loop will reject any change outside that scope.

{HYPOTHESIS_SCHEMA}
"""


def _format_clusters(clusters: list[EvalCluster]) -> str:
    parts = []
    for c in clusters:
        patterns = "\n".join(f"  - {p}" for p in c.shared_patterns[:5])
        if patterns:
            parts.append(
                f"### {c.cluster_id} ({len(c.eval_names)} evals, signal: {c.signal_type})\n"
                f"Evals: {', '.join(c.eval_names)}\n"
                f"Shared patterns:\n{patterns}"
            )
        else:
            parts.append(
                f"### {c.cluster_id} ({len(c.eval_names)} evals, signal: {c.signal_type})\n"
                f"Evals: {', '.join(c.eval_names)}\n"
                f"{c.description or '(no shared error patterns)'}"
            )
    return "\n\n".join(parts)


def _format_trace_excerpts(excerpts: dict[str, str]) -> str:
    parts = []
    for eval_name, excerpt in excerpts.items():
        parts.append(f"### {eval_name}\n```\n{excerpt}\n```")
    return "\n\n".join(parts)


def _trace_excerpt_for(
    batch: BatchResult,
    eval_name: str,
    parser: TraceParser,
    cache: dict[str, TraceSummary],
) -> str | None:
    """Return the parser's trace excerpt for *eval_name*, or None if absent.

    Caches the per-eval ``TraceSummary`` because parsing a session JSONL is
    not free and the collector visits each eval at most once per batch.
    """
    if eval_name in cache:
        return cache[eval_name].trace_excerpt or None

    result = batch.get_result(eval_name)
    if not result or not result.job_dir or not result.job_dir.exists():
        return None

    summary = parser.summarize(result.job_dir, eval_name)
    cache[eval_name] = summary
    return summary.trace_excerpt or None


def _collect_trace_excerpts(
    batch: BatchResult,
    mechanical: MechanicalAnalysis,
    clusters: list[EvalCluster],
    parser: TraceParser | None,
    max_per_cluster: int = 2,
    max_total: int = 10,
) -> dict[str, str]:
    """Collect trace excerpts, at least one per cluster.

    Returns an empty dict when *parser* is None (no parser registered for
    this batch's agent — the LLM analyzer runs on mechanical-analysis-only
    inputs).
    """
    if parser is None:
        return {}

    excerpts: dict[str, str] = {}
    summary_cache: dict[str, TraceSummary] = {}

    for cluster in clusters:
        collected_for_cluster = 0
        for eval_name in cluster.eval_names:
            if len(excerpts) >= max_total:
                return excerpts
            if collected_for_cluster >= max_per_cluster:
                break
            if eval_name in excerpts:
                continue

            excerpt = _trace_excerpt_for(batch, eval_name, parser, summary_cache)
            if excerpt:
                excerpts[eval_name] = excerpt
                collected_for_cluster += 1

    # Fill remaining slots with slowest evals not yet included
    for name, _ in mechanical.slowest_evals:
        if len(excerpts) >= max_total:
            break
        if name in excerpts:
            continue
        excerpt = _trace_excerpt_for(batch, name, parser, summary_cache)
        if excerpt:
            excerpts[name] = excerpt

    return excerpts


def _extract_json(text: str) -> str:
    """Extract JSON from a response that may contain preamble text and code fences."""
    stripped = text.strip()
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", stripped, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # Find the first [ or { and take everything from there
    for i, ch in enumerate(stripped):
        if ch in ("[", "{"):
            candidate = stripped[i:]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
            break

    return stripped


def _parse_hypotheses(raw_json: str) -> list[Hypothesis]:
    """Parse Claude's JSON output into Hypothesis objects."""
    text = _extract_json(raw_json)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse Claude's response as JSON: {e}\nResponse (first 500 chars): {text[:500]}"
        ) from e
    if isinstance(data, dict) and "hypotheses" in data:
        data = data["hypotheses"]

    hypotheses = []
    for item in data:
        try:
            hypotheses.append(
                Hypothesis(
                    cluster_id=item.get("cluster_id", item.get("eval_name", "")),
                    eval_names=item.get("eval_names", [item["eval_name"]] if "eval_name" in item else []),
                    root_cause=item["root_cause"],
                    category=GapCategory(item["category"]),
                    proposed_fix=item["proposed_fix"],
                    affected_files=item.get("affected_files", []),
                    expected_impact=item.get("expected_impact", ""),
                    confidence=float(item.get("confidence", 0.5)),
                )
            )
        except (KeyError, ValueError):
            continue

    return sorted(hypotheses, key=lambda h: h.confidence, reverse=True)


DEFAULT_MODEL = os.environ.get("SELF_IMPROVE_MODEL", "aws/anthropic/bedrock-claude-sonnet-4-5-v1")


async def llm_analyze(
    batch: BatchResult,
    mechanical: MechanicalAnalysis,
    clusters: list[EvalCluster],
    parser: TraceParser | None,
    max_hypotheses: int = 10,
    skills_path: str = ".agents/skills",
) -> list[Hypothesis]:
    """Call Anthropic API to analyze eval results and generate hypotheses."""
    if not clusters:
        return []
    trace_excerpts = _collect_trace_excerpts(batch, mechanical, clusters, parser)
    prompt = _build_analysis_prompt(mechanical, clusters, trace_excerpts, skills_path=skills_path)

    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    client = anthropic.AsyncAnthropic(base_url=base_url) if base_url else anthropic.AsyncAnthropic()
    try:
        response = await client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=16384,
            system="You are a JSON-only assistant. Output only valid JSON with no preamble, explanation, or markdown fences.",
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APITimeoutError as err:
        raise RuntimeError(f"Anthropic API timed out for model {DEFAULT_MODEL}") from err
    except anthropic.APIConnectionError as err:
        raise RuntimeError(f"Failed to connect to Anthropic API: {err}") from err

    block = response.content[0]
    if not isinstance(block, anthropic.types.TextBlock):
        raise RuntimeError(f"Anthropic API returned non-text block: {type(block).__name__}")
    raw = block.text
    if not raw or not raw.strip():
        raise RuntimeError("Anthropic API returned empty response")

    hypotheses = _parse_hypotheses(raw)
    return hypotheses[:max_hypotheses]


async def generate_gap_analysis(
    batch: BatchResult,
    parser: TraceParser | None,
    baselines: dict[str, BaselineEntry] | None = None,
    max_hypotheses: int = 10,
    skills_path: str = ".agents/skills",
) -> GapAnalysis:
    """Full pipeline: mechanical analysis -> clustering -> LLM analysis -> GapAnalysis."""
    mechanical = mechanical_analysis(batch, parser, baselines)
    clusters = cluster_evals(mechanical, baselines, batch)
    hypotheses = await llm_analyze(batch, mechanical, clusters, parser, max_hypotheses, skills_path=skills_path)

    return GapAnalysis(
        batch_id=batch.batch_id,
        mechanical=mechanical,
        clusters=clusters,
        hypotheses=hypotheses,
        generated_at=datetime.now(timezone.utc),
    )
