# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token-usage normalization for Agents jobs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nemo_agents_plugin.improvement.traces.claude_code import extract_token_usage
from nemo_helix_plugin.job_usage import JobTokenUsage


def fabric_output_token_usage(output: Any) -> JobTokenUsage | None:
    """Read provider totals from a terminal Fabric output, when present."""
    if not isinstance(output, Mapping):
        return None
    usage = output.get("usage")
    if not isinstance(usage, Mapping):
        return None

    input_tokens = _first_nonnegative_int(usage, "input_tokens", "prompt_tokens")
    output_tokens = _first_nonnegative_int(usage, "output_tokens", "completion_tokens")
    if input_tokens is None and output_tokens is None:
        return None
    return JobTokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def evaluation_batch_token_usage(
    batch_dir: Path,
    *,
    runner: str,
    expected_executions: int,
) -> JobTokenUsage | None:
    """Sum complete raw execution records without using aggregate reports."""
    if expected_executions < 1:
        return None
    if runner == "harbor":
        usages = _harbor_execution_usages(batch_dir)
    elif runner == "nat":
        usages = _nat_execution_usages(batch_dir)
    else:
        return None

    if len(usages) != expected_executions or any(usage is None for usage in usages):
        return None
    complete = [usage for usage in usages if usage is not None]
    return JobTokenUsage(
        input_tokens=sum(usage.input_tokens or 0 for usage in complete),
        output_tokens=sum(usage.output_tokens or 0 for usage in complete),
    )


def _harbor_execution_usages(batch_dir: Path) -> list[JobTokenUsage | None]:
    usages: list[JobTokenUsage | None] = []
    for job_dir in sorted(path for path in batch_dir.iterdir() if path.is_dir()):
        trial_result = _find_harbor_trial_result(job_dir)
        if trial_result is None:
            usages.append(None)
            continue
        usages.append(_harbor_trial_usage(trial_result))
    return usages


def _find_harbor_trial_result(job_dir: Path) -> Path | None:
    for result_path in sorted(job_dir.rglob("result.json")):
        payload = _load_json_object(result_path)
        if payload is not None and "task_name" in payload:
            return result_path
    return None


def _harbor_trial_usage(result_path: Path) -> JobTokenUsage | None:
    payload = _load_json_object(result_path)
    if payload is None:
        return None
    agent_result = payload.get("agent_result")
    if isinstance(agent_result, Mapping):
        input_tokens = _nonnegative_int(agent_result.get("n_input_tokens"))
        output_tokens = _nonnegative_int(agent_result.get("n_output_tokens"))
        cache_tokens = _nonnegative_int(agent_result.get("n_cache_tokens"))
        if input_tokens is not None and output_tokens is not None:
            return JobTokenUsage(
                input_tokens=input_tokens + (cache_tokens or 0),
                output_tokens=output_tokens,
            )

    trace_usage = extract_token_usage(result_path.parent)
    if trace_usage is None:
        return None
    return JobTokenUsage(
        input_tokens=trace_usage.input_tokens + trace_usage.cache_tokens,
        output_tokens=trace_usage.output_tokens,
    )


def _nat_execution_usages(batch_dir: Path) -> list[JobTokenUsage | None]:
    usages: list[JobTokenUsage | None] = []
    for run_dir in sorted(path for path in batch_dir.iterdir() if path.is_dir()):
        result_path = run_dir / "result.json"
        payload = _load_json_object(result_path)
        if payload is None:
            usages.append(None)
            continue
        metrics = payload.get("metrics")
        if not isinstance(metrics, Mapping):
            usages.append(None)
            continue
        input_tokens = _nonnegative_int(metrics.get("prompt_tokens"))
        output_tokens = _nonnegative_int(metrics.get("completion_tokens"))
        usages.append(
            JobTokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
            if input_tokens is not None and output_tokens is not None
            else None
        )
    return usages


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _first_nonnegative_int(values: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in values:
            return _nonnegative_int(values[key])
    return None


def _nonnegative_int(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None
