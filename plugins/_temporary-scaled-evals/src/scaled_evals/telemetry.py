# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

from scaled_evals.intake.atif_payload import TrialPayload, load_json_if_exists, trial_payloads

_TOKEN_FIELDS = {
    "input_tokens": ("total_prompt_tokens", "prompt_tokens", "input_tokens"),
    "output_tokens": ("total_completion_tokens", "completion_tokens", "output_tokens"),
    "cached_tokens": ("total_cached_tokens", "cached_tokens"),
    "cache_creation_tokens": ("total_cache_creation_tokens", "cache_creation_tokens"),
}
_RAW_ARTIFACT_NAMES = {
    "result.json": "result",
    "trajectory.json": "trajectory",
    "trajectory.json.bak": "native_trajectory",
    "transcript.json": "transcript",
    "transcript.jsonl": "transcript",
    "intake-upload.json": "intake_diagnostic",
}


def _non_negative_number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return value


def _first_metric(metrics: dict[str, Any], names: tuple[str, ...]) -> int | None:
    for name in names:
        value = _non_negative_number(metrics.get(name))
        if value is not None:
            return int(value)
    return None


def _payload_metric(payload: TrialPayload, names: tuple[str, ...]) -> int | None:
    body = payload.payload
    metrics = body.get("final_metrics")
    if not isinstance(metrics, dict):
        return None
    value = _first_metric(metrics, names)
    if value is not None:
        return value
    metrics_extra = metrics.get("extra")
    return _first_metric(metrics_extra, names) if isinstance(metrics_extra, dict) else None


def _raw_usage_source(payload: TrialPayload) -> str:
    extra = payload.payload.get("extra")
    if not isinstance(extra, dict):
        return "unknown"
    trial_result = extra.get("trial_result")
    if isinstance(trial_result, dict):
        agent_result = trial_result.get("agent_result")
        if isinstance(agent_result, dict) and any(
            key in agent_result for key in ("n_input_tokens", "n_output_tokens", "n_cache_tokens")
        ):
            return "framework-result"
    trial_dir = extra.get("trial_dir")
    if isinstance(trial_dir, str):
        trajectory = load_json_if_exists(Path(trial_dir) / "agent" / "trajectory.json")
        final_metrics = trajectory.get("final_metrics")
        if isinstance(final_metrics, dict) and any(
            key in final_metrics for names in _TOKEN_FIELDS.values() for key in names
        ):
            return "atif"
        for step in trajectory.get("steps") or []:
            metrics = step.get("metrics") if isinstance(step, dict) else None
            if isinstance(metrics, dict) and any(key in metrics for names in _TOKEN_FIELDS.values() for key in names):
                return "atif"
    return "unknown"


def _cost(payload: TrialPayload) -> tuple[float | None, str]:
    body = payload.payload
    final_metrics = body.get("final_metrics")
    cost = None
    if isinstance(final_metrics, dict):
        raw_cost = _non_negative_number(final_metrics.get("total_cost_usd"))
        cost = float(raw_cost) if raw_cost is not None else None
    extra = body.get("extra")
    trial_result = extra.get("trial_result") if isinstance(extra, dict) else None
    agent_result = trial_result.get("agent_result") if isinstance(trial_result, dict) else None
    if isinstance(agent_result, dict) and _non_negative_number(agent_result.get("cost_usd")) is not None:
        return cost, "provider"
    if isinstance(final_metrics, dict):
        source = final_metrics.get("cost_source")
        metrics_extra = final_metrics.get("extra")
        if source is None and isinstance(metrics_extra, dict):
            source = metrics_extra.get("cost_source")
        if source in {"provider", "estimated"}:
            return cost, str(source)
    return cost, "unknown"


def _interactions_known(payload: TrialPayload) -> bool:
    extra = payload.payload.get("extra")
    if isinstance(extra, dict) and extra.get("trajectory_status") in {"missing", "unreadable"}:
        return False
    return isinstance(payload.payload.get("steps"), list)


def _artifact_refs(job_dir: Path) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    if not job_dir.is_dir():
        return refs
    for path in sorted(candidate for candidate in job_dir.rglob("*") if candidate.is_file()):
        relation = _RAW_ARTIFACT_NAMES.get(path.name)
        if relation is None and not path.name.startswith("transcript."):
            continue
        refs.append(
            {
                "relation": relation or "transcript",
                "path": path.relative_to(job_dir).as_posix(),
            }
        )
    return refs


def summarize_job_telemetry(job_dir: Path, *, evaluation_run_id: str) -> dict[str, Any]:
    """Derive portable facts from the same ATIF payloads handed to Intake.

    Missing source facts remain ``None``. In particular, synthesized ATIF records
    do not turn absent usage into zero usage.
    """
    payloads = trial_payloads(
        job_dir,
        "telemetry",
        "telemetry",
        "scaled-evals",
        evaluation_run_id=evaluation_run_id,
    )
    sources = [_raw_usage_source(payload) for payload in payloads]
    known_usage = [(payload, source) for payload, source in zip(payloads, sources, strict=True) if source != "unknown"]
    token_totals: dict[str, int | None] = {}
    for target, names in _TOKEN_FIELDS.items():
        values = []
        for payload, _source in known_usage:
            value = _payload_metric(payload, names)
            if value is not None:
                values.append(value)
        token_totals[target] = sum(values) if values else None

    known_sources = sorted(set(source for source in sources if source != "unknown"))
    usage_source = known_sources[0] if len(known_sources) == 1 else ("mixed" if known_sources else "unknown")
    turns = 0
    tool_calls = 0
    interactions_known = False
    costs: list[float] = []
    cost_sources: set[str] = set()
    for payload in payloads:
        steps = payload.payload.get("steps")
        if _interactions_known(payload) and isinstance(steps, list):
            interactions_known = True
            for step in steps:
                if not isinstance(step, dict):
                    continue
                if step.get("source") in {"user", "agent"}:
                    turns += 1
                calls = step.get("tool_calls")
                if isinstance(calls, list):
                    tool_calls += len(calls)
        cost, source = _cost(payload)
        if cost is not None:
            costs.append(cost)
        cost_sources.add(source)

    cost_source = next(iter(cost_sources)) if len(cost_sources) == 1 else "unknown"
    if cost_source not in {"provider", "estimated"}:
        cost_source = "unknown"
    return {
        **token_totals,
        "usage_source": usage_source,
        "turn_count": turns if interactions_known else None,
        "tool_call_count": tool_calls if interactions_known else None,
        "cost_usd": sum(costs) if costs else None,
        "cost_source": cost_source,
        "raw_artifact_refs": _artifact_refs(job_dir),
        "intake_expected_records": len(payloads),
        "intake_run_refs": [payload.external_id for payload in payloads],
    }
