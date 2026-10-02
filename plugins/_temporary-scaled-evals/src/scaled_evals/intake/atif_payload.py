# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor ATIF → NeMo Helix Intake payload building.

Adapted from an internal ATIF upload helper. Upload via
:mod:`scaled_evals.intake.client`.
"""
# ruff: noqa: E501 — ported mapping module; line wraps deferred.

from __future__ import annotations

import importlib.metadata
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from scaled_evals.intake.harbor_lab_pricing import (
    HARBOR_LAB_PRICING_SOURCE,
    estimate_cost,
)

DEFAULT_SOURCE = "scaled-evals"
PLATFORM_AGENTIC_USE_DATASET = "platform-agentic-use"
GENERATED_STAGE_MARKERS = ("_skills_stage.json", "_gke_image_stage.json")
SUPPORTED_ATIF_SCHEMA_VERSIONS = {
    "ATIF-v1.0",
    "ATIF-v1.1",
    "ATIF-v1.2",
    "ATIF-v1.3",
    "ATIF-v1.4",
    "ATIF-v1.5",
    "ATIF-v1.6",
    "ATIF-v1.7",
}


class IntakeError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrialPayload:
    external_id: str
    payload: dict[str, Any]


def installed_package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_metadata() -> dict[str, Any]:
    return {
        "harbor": {
            "package": "harbor",
            "version": installed_package_version("harbor"),
        },
        "python": {"version": sys.version.split()[0]},
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path)


def reward_from_trial(trial_result: dict[str, Any]) -> Any:
    rewards = (trial_result.get("verifier_result") or {}).get("rewards") or {}
    if "reward" in rewards:
        return rewards["reward"]
    if rewards:
        return next(iter(rewards.values()))
    return None


def total_errors(job_result: dict[str, Any]) -> int:
    stats = job_result.get("stats") or {}
    if isinstance(stats.get("n_errored_trials"), int):
        return stats["n_errored_trials"]
    evals = stats.get("evals") or {}
    total = 0
    for item in evals.values():
        if isinstance(item, dict) and isinstance(item.get("n_errors"), int):
            total += item["n_errors"]
    return total


def first_mapping(items: object) -> dict[str, Any]:
    if isinstance(items, list) and items and isinstance(items[0], dict):
        return items[0]
    return {}


def platform_agentic_use_dataset(path: object) -> str | None:
    if not path:
        return None

    dataset_path = Path(str(path))
    if PLATFORM_AGENTIC_USE_DATASET in dataset_path.parts:
        return PLATFORM_AGENTIC_USE_DATASET
    if (dataset_path / "_platform_agentic_use_stage.json").exists():
        return PLATFORM_AGENTIC_USE_DATASET
    return None


def generated_stage_source_dataset(path: object) -> str | None:
    if not path:
        return None

    dataset_path = Path(str(path))
    for candidate in (dataset_path, *dataset_path.parents):
        for marker_name in GENERATED_STAGE_MARKERS:
            marker = candidate / marker_name
            if not marker.exists():
                continue
            metadata = load_json(marker)
            source = metadata.get("source")
            if source:
                return Path(str(source)).name
    return None


def dataset_name(job_config: dict[str, Any], trial_result: dict[str, Any]) -> str:
    dataset = first_mapping(job_config.get("datasets"))
    task_config = (trial_result.get("config") or {}).get("task") or {}
    path = dataset.get("path") or task_config.get("path")
    staged_dataset = platform_agentic_use_dataset(path)
    if staged_dataset:
        return staged_dataset
    generated_source_dataset = generated_stage_source_dataset(path)
    if generated_source_dataset:
        return generated_source_dataset

    if dataset.get("name"):
        name = str(dataset["name"])
        version = dataset.get("version")
        return f"{name}@{version}" if version else name
    source = trial_result.get("source") or task_config.get("source")
    if source:
        return str(source)
    if path:
        return Path(str(path)).name
    return "harbor"


def resolved_app_name(app_name: str, dataset: str) -> str:
    return dataset if app_name == "auto" else app_name


def agent_name(job_config: dict[str, Any], trial_result: dict[str, Any]) -> str:
    agent = first_mapping(job_config.get("agents"))
    trial_agent = (trial_result.get("config") or {}).get("agent") or {}
    agent_info = trial_result.get("agent_info") or {}
    return str(
        agent_info.get("name")
        or agent.get("name")
        or trial_agent.get("name")
        or agent.get("import_path")
        or trial_agent.get("import_path")
        or "unknown"
    )


def model_name(job_config: dict[str, Any], trial_result: dict[str, Any], trajectory: dict[str, Any]) -> str:
    agent = first_mapping(job_config.get("agents"))
    trial_agent = (trial_result.get("config") or {}).get("agent") or {}
    agent_info = trial_result.get("agent_info") or {}
    return str(
        ((agent_info.get("model_info") or {}).get("name"))
        or agent_info.get("model_name")
        or agent.get("model_name")
        or trial_agent.get("model_name")
        or (trajectory.get("agent") or {}).get("model_name")
        or "unknown"
    )


def count_tool_calls(trajectory: dict[str, Any]) -> int:
    total = 0
    for step in trajectory.get("steps") or []:
        if isinstance(step, dict):
            calls = step.get("tool_calls") or []
            if isinstance(calls, list):
                total += len(calls)
    return total


def final_message(trajectory: dict[str, Any], trial_result: dict[str, Any]) -> str:
    for step in reversed(trajectory.get("steps") or []):
        if not isinstance(step, dict):
            continue
        message = step.get("message")
        if isinstance(message, str) and message.strip() and message.strip() != "(tool use)":
            return message.strip()
    reward = reward_from_trial(trial_result)
    return f"Harbor trial completed with reward={reward!r}"


def stringify_content(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def openai_tool_calls(step: dict[str, Any]) -> list[dict[str, Any]]:
    calls = step.get("tool_calls") or []
    if not isinstance(calls, list):
        return []

    converted: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        arguments = call.get("arguments")
        converted.append(
            {
                "id": str(call.get("tool_call_id") or f"tool-call-{index}"),
                "type": "function",
                "function": {
                    "name": str(call.get("function_name") or call.get("name") or "tool"),
                    "arguments": stringify_content(arguments),
                },
            }
        )
    return converted


def observation_messages(step: dict[str, Any]) -> list[dict[str, Any]]:
    observation = step.get("observation") or {}
    if not isinstance(observation, dict):
        return []

    results = observation.get("results") or []
    if not isinstance(results, list):
        return []

    messages: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        messages.append(
            {
                "role": "tool",
                "tool_call_id": str(result.get("source_call_id") or ""),
                "content": stringify_content(result.get("content")),
            }
        )
    return messages


def message_from_step(step: dict[str, Any]) -> dict[str, Any] | None:
    source = str(step.get("source") or "")
    if source == "user":
        role = "user"
    elif source == "agent":
        role = "assistant"
    else:
        return None

    tool_calls = openai_tool_calls(step)
    content = stringify_content(step.get("message")).strip()
    if content == "(tool use)":
        content = ""
    if not content and not tool_calls:
        return None

    message: dict[str, Any] = {"role": role, "content": content}
    if tool_calls and role == "assistant":
        message["tool_calls"] = tool_calls
    return message


def messages_for_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for step in steps:
        message = message_from_step(step)
        if message is not None:
            messages.append(message)
        messages.extend(observation_messages(step))
    return messages


def final_agent_step_index(steps: list[dict[str, Any]]) -> int | None:
    for index in range(len(steps) - 1, -1, -1):
        step = steps[index]
        if step.get("source") == "agent" and message_from_step(step) is not None:
            return index
    return None


def request_messages_for_trial(steps: list[dict[str, Any]], response_index: int | None) -> list[dict[str, Any]]:
    request_steps = steps if response_index is None else steps[:response_index]
    messages = messages_for_steps(request_steps)
    return messages or [{"role": "user", "content": "Harbor evaluation trial."}]


def response_message_for_step(step: dict[str, Any]) -> dict[str, Any]:
    message = message_from_step(step)
    if message is None:
        message = {"role": "assistant", "content": ""}
    message["role"] = "assistant"
    return message


def response_message_for_trial(
    steps: list[dict[str, Any]],
    response_index: int | None,
    trajectory: dict[str, Any],
    trial_result: dict[str, Any],
) -> dict[str, Any]:
    if response_index is not None:
        return response_message_for_step(steps[response_index])
    return {"role": "assistant", "content": final_message(trajectory, trial_result)}


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def summed_step_metric(steps: list[dict[str, Any]], key: str) -> int | float | None:
    total: int | float = 0
    found = False
    for step in steps:
        metrics = step.get("metrics")
        if not isinstance(metrics, dict):
            continue
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            total += value
            found = True
    return total if found else None


def final_step_timestamp(steps: list[dict[str, Any]]) -> Any:
    for step in reversed(steps):
        if step.get("timestamp"):
            return step["timestamp"]
    return None


def usage_from_trial(
    trial_result: dict[str, Any],
    trajectory: dict[str, Any],
    agent_result: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    steps = [step for step in trajectory.get("steps") or [] if isinstance(step, dict)]
    final_metrics = trajectory.get("final_metrics")
    if not isinstance(final_metrics, dict):
        final_metrics = {}

    usage = {
        "model": model,
        "started_at": trial_result.get("started_at"),
        "ended_at": trial_result.get("finished_at") or final_step_timestamp(steps),
        "input_tokens": first_present(
            agent_result.get("n_input_tokens"),
            final_metrics.get("prompt_tokens"),
            final_metrics.get("input_tokens"),
            summed_step_metric(steps, "prompt_tokens"),
            0,
        ),
        "output_tokens": first_present(
            agent_result.get("n_output_tokens"),
            final_metrics.get("completion_tokens"),
            final_metrics.get("output_tokens"),
            summed_step_metric(steps, "completion_tokens"),
            0,
        ),
        "cached_tokens": first_present(
            agent_result.get("n_cache_tokens"),
            final_metrics.get("cached_tokens"),
            summed_step_metric(steps, "cached_tokens"),
            0,
        ),
    }
    cost_usd = first_present(
        agent_result.get("cost_usd"),
        final_metrics.get("cost_usd"),
        summed_step_metric(steps, "cost_usd"),
    )
    if cost_usd is not None:
        usage["cost_usd"] = cost_usd
    return {key: value for key, value in usage.items() if value is not None}


def atif_schema_version(trajectory: dict[str, Any]) -> str:
    version = trajectory.get("schema_version")
    if isinstance(version, str) and version in SUPPORTED_ATIF_SCHEMA_VERSIONS:
        return version
    return "ATIF-v1.6"


def atif_agent_payload(
    trajectory: dict[str, Any],
    resolved_agent_name: str,
    resolved_model_name: str,
) -> dict[str, Any]:
    raw_agent = trajectory.get("agent")
    raw = dict(raw_agent) if isinstance(raw_agent, dict) else {}
    payload: dict[str, Any] = {
        "name": str(raw.pop("name", None) or resolved_agent_name or "harbor-agent"),
        "version": str(raw.pop("version", None) or raw.pop("agent_version", None) or "unknown"),
        "model_name": str(raw.pop("model_name", None) or resolved_model_name or "unknown"),
    }
    tool_definitions = raw.pop("tool_definitions", None)
    if isinstance(tool_definitions, list):
        payload["tool_definitions"] = [item for item in tool_definitions if isinstance(item, dict)]
    if raw:
        payload["extra"] = raw
    return payload


def integer_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def number_or_none(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def atif_step_metrics(metrics: Any) -> dict[str, Any] | None:
    if not isinstance(metrics, dict):
        return None
    payload: dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
        value = integer_or_none(metrics.get(key))
        if value is not None:
            payload[key] = value
    cost_usd = number_or_none(metrics.get("cost_usd"))
    if cost_usd is not None:
        payload["cost_usd"] = cost_usd
    for key in ("prompt_token_ids", "completion_token_ids", "logprobs"):
        value = metrics.get(key)
        if isinstance(value, list):
            payload[key] = value
    extra = {
        key: value
        for key, value in metrics.items()
        if key not in payload and key not in {"prompt_tokens", "completion_tokens", "cached_tokens", "cost_usd"}
    }
    if extra:
        payload["extra"] = extra
    return payload or None


def atif_final_metrics_payload(
    trajectory: dict[str, Any],
    steps: list[dict[str, Any]],
    usage: dict[str, Any],
) -> dict[str, Any]:
    raw = trajectory.get("final_metrics")
    final_metrics = dict(raw) if isinstance(raw, dict) else {}
    payload: dict[str, Any] = {}
    field_sources = {
        "total_prompt_tokens": (
            final_metrics.get("total_prompt_tokens"),
            final_metrics.get("prompt_tokens"),
            final_metrics.get("input_tokens"),
            usage.get("input_tokens"),
        ),
        "total_completion_tokens": (
            final_metrics.get("total_completion_tokens"),
            final_metrics.get("completion_tokens"),
            final_metrics.get("output_tokens"),
            usage.get("output_tokens"),
        ),
        "total_cached_tokens": (
            final_metrics.get("total_cached_tokens"),
            final_metrics.get("cached_tokens"),
            usage.get("cached_tokens"),
        ),
    }
    for target, candidates in field_sources.items():
        for candidate in candidates:
            value = integer_or_none(candidate)
            if value is not None:
                payload[target] = value
                break
    for candidate in (
        final_metrics.get("total_cost_usd"),
        final_metrics.get("cost_usd"),
        usage.get("cost_usd"),
    ):
        value = number_or_none(candidate)
        if value is not None:
            payload["total_cost_usd"] = value
            break
    payload["total_steps"] = len(steps)
    consumed = {
        "total_prompt_tokens",
        "prompt_tokens",
        "input_tokens",
        "total_completion_tokens",
        "completion_tokens",
        "output_tokens",
        "total_cached_tokens",
        "cached_tokens",
        "total_cost_usd",
        "cost_usd",
        "total_steps",
    }
    extra = {key: value for key, value in final_metrics.items() if key not in consumed}
    if extra:
        payload["extra"] = extra
    return payload


_SESSION_TOTAL_FIELDS = {
    "total_calls": "calls",
    "total_prompt_tokens": "prompt_tokens",
    "total_cached_tokens": "cached_tokens",
    "total_cache_creation_tokens": "cache_creation_tokens",
    "total_completion_tokens": "completion_tokens",
}


class _SwitchyardModelUsage(BaseModel):
    """Validated model counters from one Switchyard routing session."""

    model_config = ConfigDict(extra="ignore", strict=True)

    calls: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    cached_tokens: int = Field(ge=0)
    cache_creation_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


class _SwitchyardSessionStats(BaseModel):
    """Validated, internally consistent Switchyard session snapshot."""

    model_config = ConfigDict(extra="ignore", strict=True)

    session_id: str = Field(min_length=1)
    total_calls: int = Field(ge=0)
    total_prompt_tokens: int = Field(ge=0)
    total_cached_tokens: int = Field(ge=0)
    total_cache_creation_tokens: int = Field(ge=0)
    total_completion_tokens: int = Field(ge=0)
    models: dict[str, _SwitchyardModelUsage]

    @field_validator("models")
    @classmethod
    def _nonempty_named_models(cls, models: dict[str, _SwitchyardModelUsage]) -> dict[str, _SwitchyardModelUsage]:
        if not models:
            raise ValueError("models must not be empty")
        if any(not model.strip() for model in models):
            raise ValueError("model names must not be blank")
        return models

    @model_validator(mode="after")
    def _totals_match_models(self) -> _SwitchyardSessionStats:
        for total_field, model_field in _SESSION_TOTAL_FIELDS.items():
            if getattr(self, total_field) != sum(getattr(usage, model_field) for usage in self.models.values()):
                raise ValueError(f"{total_field} does not match model totals")
        return self


def _validated_switchyard_session(value: Any) -> _SwitchyardSessionStats | None:
    try:
        return _SwitchyardSessionStats.model_validate(value)
    except ValidationError:
        return None


def switchyard_session_stats(routing_stats: dict[str, Any], session_id: str) -> dict[str, Any] | None:
    """Return one internally consistent Switchyard session snapshot."""
    sessions = routing_stats.get("sessions")
    raw = sessions.get(session_id) if isinstance(sessions, dict) else None
    if raw is None and routing_stats.get("session_id") == session_id:
        raw = routing_stats
    session = _validated_switchyard_session(raw)
    if session is None or session.session_id != session_id:
        return None
    return session.model_dump()


def switchyard_model_metrics(routing_stats: dict[str, Any]) -> dict[str, dict[str, int | float]]:
    """Aggregate per-model usage metrics from a final Switchyard stats snapshot.

    Cache hit rate is the fraction of a model's input tokens served from cache.
    Models with no input tokens have a rate of ``0.0``.
    """
    raw_sessions = routing_stats.get("sessions")
    sessions: list[_SwitchyardSessionStats] = []
    if isinstance(raw_sessions, dict):
        if not raw_sessions:
            return {}
        for session_id, raw_session in raw_sessions.items():
            session = _validated_switchyard_session(raw_session)
            if session is None or session.session_id != session_id:
                return {}
            sessions.append(session)
    else:
        session = _validated_switchyard_session(routing_stats)
        if session is None:
            return {}
        sessions.append(session)

    input_tokens: dict[str, int] = {}
    output_tokens: dict[str, int] = {}
    cached_tokens: dict[str, int] = {}

    for session in sessions:
        for model, usage in session.models.items():
            input_tokens[model] = input_tokens.get(model, 0) + usage.prompt_tokens
            output_tokens[model] = output_tokens.get(model, 0) + usage.completion_tokens
            cached_tokens[model] = cached_tokens.get(model, 0) + usage.cached_tokens

    if not input_tokens:
        return {}
    return {
        "input_tokens_by_model": dict(sorted(input_tokens.items())),
        "output_tokens_by_model": dict(sorted(output_tokens.items())),
        "cache_hit_rate_by_model": {
            model: cached_tokens[model] / tokens if tokens else 0.0 for model, tokens in sorted(input_tokens.items())
        },
    }


def native_atif_model_name(trajectory: dict[str, Any]) -> str | None:
    agent = trajectory.get("agent")
    if not isinstance(agent, dict):
        return None
    model = agent.get("model_name")
    if not isinstance(model, str) or not model.strip():
        return None
    return model.strip()


def _is_gateway_model_name(model: str) -> bool:
    normalized = model.strip().casefold().strip("/")
    return normalized == "unknown" or normalized == "switchyard" or normalized.endswith("/switchyard")


def preserve_native_single_model_metrics(
    native_model: str | None,
    final_metrics: dict[str, Any],
    session_stats: dict[str, Any],
) -> bool:
    """Keep complete native totals when they agree with a concrete single routed model."""
    if native_model is None or _is_gateway_model_name(native_model):
        return False
    if len(session_stats["models"]) != 1:
        return False
    for field in (
        "total_prompt_tokens",
        "total_cached_tokens",
        "total_completion_tokens",
    ):
        if integer_or_none(final_metrics.get(field)) != session_stats[field]:
            return False
    total_cost = number_or_none(final_metrics.get("total_cost_usd"))
    if total_cost is None or total_cost < 0:
        return False
    has_billable_tokens = bool(session_stats["total_prompt_tokens"] or session_stats["total_completion_tokens"])
    return not has_billable_tokens or total_cost > 0


def hydrate_atif_from_switchyard(
    payload: dict[str, Any],
    session_stats: dict[str, Any],
    *,
    native_model: str | None,
) -> None:
    """Hydrate only ATIF root attribution from authoritative session totals."""
    raw_models = session_stats["models"]
    routed_models: dict[str, dict[str, Any]] = {}
    total_cost = 0.0
    complete_cost = True
    for model, usage in raw_models.items():
        model_usage = dict(usage)
        priced = estimate_cost(
            model,
            usage["prompt_tokens"],
            usage["cached_tokens"],
            usage["cache_creation_tokens"],
            usage["completion_tokens"],
        )
        if priced is None:
            model_usage["pricing"] = {
                "source": HARBOR_LAB_PRICING_SOURCE,
                "status": "unknown_model",
            }
            if usage["prompt_tokens"] or usage["completion_tokens"]:
                complete_cost = False
        else:
            model_usage["cost_usd"] = priced["total_cost"]
            model_usage["pricing"] = {
                "source": HARBOR_LAB_PRICING_SOURCE,
                "status": "priced",
                "matched_model": priced["matched_model"],
                "input_cost_usd": priced["input_cost"],
                "cache_cost_usd": priced["cache_cost"],
                "cache_creation_cost_usd": priced["cache_creation_cost"],
                "output_cost_usd": priced["output_cost"],
            }
            total_cost += float(priced["total_cost"])
        routed_models[model] = model_usage

    final_metrics = payload.get("final_metrics")
    if not isinstance(final_metrics, dict):
        final_metrics = {}
        payload["final_metrics"] = final_metrics
    final_metrics.update(
        {
            "total_prompt_tokens": session_stats["total_prompt_tokens"],
            "total_cached_tokens": session_stats["total_cached_tokens"],
            "total_completion_tokens": session_stats["total_completion_tokens"],
        }
    )
    if complete_cost:
        final_metrics["total_cost_usd"] = total_cost
    else:
        final_metrics.pop("total_cost_usd", None)

    models = list(raw_models)
    resolved_model = models[0] if len(models) == 1 else "unknown"
    replace_model = native_model is None or _is_gateway_model_name(native_model)
    if replace_model:
        agent = payload.get("agent")
        if isinstance(agent, dict):
            agent["model_name"] = resolved_model

    extra = payload.get("extra")
    if not isinstance(extra, dict):
        extra = {}
        payload["extra"] = extra
    experiment = extra.get("experiment")
    if replace_model and isinstance(experiment, dict):
        experiment["model"] = resolved_model
    extra["switchyard_routing"] = {
        "source": "switchyard-session-stats",
        "pricing_source": HARBOR_LAB_PRICING_SOURCE,
        "session_id": session_stats["session_id"],
        "total_calls": session_stats["total_calls"],
        "total_prompt_tokens": session_stats["total_prompt_tokens"],
        "total_cached_tokens": session_stats["total_cached_tokens"],
        "total_cache_creation_tokens": session_stats["total_cache_creation_tokens"],
        "total_completion_tokens": session_stats["total_completion_tokens"],
        "cost_status": "complete" if complete_cost else "unknown_pricing",
        "models": routed_models,
    }


def _persist_hydrated_trajectory(path: Path, payload: dict[str, Any]) -> None:
    """Preserve the native trajectory once, then write the exact Intake payload."""
    backup_path = path.with_name(f"{path.name}.bak")
    try:
        original = path.read_bytes()
        try:
            with backup_path.open("xb") as backup:
                backup.write(original)
        except FileExistsError:
            pass
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise IntakeError(f"failed to persist hydrated ATIF trajectory {path}: {exc}") from exc


def atif_content(value: Any) -> str | list[Any]:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        valid_parts = [item for item in value if isinstance(item, dict) and isinstance(item.get("type"), str)]
        if len(valid_parts) == len(value):
            return value
    return stringify_content(value)


def atif_tool_calls(step: dict[str, Any]) -> list[dict[str, Any]] | None:
    raw_calls = step.get("tool_calls")
    if not isinstance(raw_calls, list):
        return None
    calls: list[dict[str, Any]] = []
    for index, call in enumerate(raw_calls):
        if not isinstance(call, dict):
            continue
        arguments = call.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {"value": arguments} if arguments is not None else {}
        calls.append(
            {
                "tool_call_id": str(call.get("tool_call_id") or call.get("id") or f"tool-call-{index}"),
                "function_name": str(call.get("function_name") or call.get("name") or "tool"),
                "arguments": arguments,
            }
        )
    return calls or None


def atif_observation(observation: Any) -> dict[str, Any] | None:
    if not isinstance(observation, dict):
        return None
    raw_results = observation.get("results")
    if not isinstance(raw_results, list):
        return None
    results: list[dict[str, Any]] = []
    for result in raw_results:
        if not isinstance(result, dict):
            continue
        payload: dict[str, Any] = {}
        source_call_id = result.get("source_call_id")
        if source_call_id is not None:
            payload["source_call_id"] = str(source_call_id)
        if "content" in result:
            payload["content"] = atif_content(result.get("content"))
        refs = result.get("subagent_trajectory_ref")
        if isinstance(refs, list):
            normalized_refs: list[dict[str, Any]] = []
            for ref in refs:
                if not isinstance(ref, dict):
                    continue
                normalized_ref = {
                    key: str(ref[key])
                    for key in ("trajectory_id", "trajectory_path", "session_id")
                    if ref.get(key) is not None
                }
                extra = {
                    key: value
                    for key, value in ref.items()
                    if key not in {"trajectory_id", "trajectory_path", "session_id"}
                }
                if extra:
                    normalized_ref["extra"] = extra
                if normalized_ref:
                    normalized_refs.append(normalized_ref)
            if normalized_refs:
                payload["subagent_trajectory_ref"] = normalized_refs
        extra = {
            key: value
            for key, value in result.items()
            if key not in {"source_call_id", "content", "subagent_trajectory_ref"}
        }
        if extra:
            payload["extra"] = extra
        if payload:
            results.append(payload)
    return {"results": results} if results else None


def merge_observation(agent_step: dict[str, Any], observation: dict[str, Any]) -> None:
    existing = agent_step.get("observation")
    if not isinstance(existing, dict):
        agent_step["observation"] = observation
        return
    existing_results = existing.setdefault("results", [])
    if isinstance(existing_results, list):
        existing_results.extend(observation.get("results") or [])


def atif_step_payload(step: dict[str, Any], step_id: int) -> dict[str, Any] | None:
    source = step.get("source")
    if source not in {"system", "user", "agent"}:
        return None
    payload: dict[str, Any] = {
        "step_id": step_id,
        "source": source,
        "message": atif_content(step.get("message", "")),
    }
    timestamp = step.get("timestamp")
    if isinstance(timestamp, str) and timestamp:
        payload["timestamp"] = timestamp
    if isinstance(step.get("is_copied_context"), bool):
        payload["is_copied_context"] = step["is_copied_context"]
    llm_call_count = integer_or_none(step.get("llm_call_count"))
    if llm_call_count is not None:
        payload["llm_call_count"] = llm_call_count
    extra = step.get("extra")
    if isinstance(extra, dict) and extra:
        payload["extra"] = extra

    if source == "agent":
        model_name = step.get("model_name")
        if isinstance(model_name, str) and model_name:
            payload["model_name"] = model_name
        if step.get("reasoning_effort") is not None:
            payload["reasoning_effort"] = step["reasoning_effort"]
        reasoning_content = step.get("reasoning_content")
        if isinstance(reasoning_content, str):
            payload["reasoning_content"] = reasoning_content
        tool_calls = atif_tool_calls(step)
        if tool_calls:
            payload["tool_calls"] = tool_calls
        observation = atif_observation(step.get("observation"))
        if observation:
            payload["observation"] = observation
        metrics = atif_step_metrics(step.get("metrics"))
        if metrics:
            payload["metrics"] = metrics
    return payload


def atif_steps_payload(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    last_agent_step: dict[str, Any] | None = None
    for raw_step in trajectory.get("steps") or []:
        if not isinstance(raw_step, dict):
            continue
        if raw_step.get("source") == "environment":
            observation = atif_observation(raw_step.get("observation"))
            if observation and last_agent_step is not None:
                merge_observation(last_agent_step, observation)
            continue
        step = atif_step_payload(raw_step, len(steps) + 1)
        if step is None:
            continue
        steps.append(step)
        last_agent_step = step if step.get("source") == "agent" else None
    return steps


def atif_payload_for_trial(
    *,
    trajectory: dict[str, Any],
    job_id: str,
    job_config: dict[str, Any],
    job_result: dict[str, Any],
    trial_result: dict[str, Any],
    job_dir: Path,
    trial_dir: Path,
    external_id: str,
    session_id: str,
    trial_id: str,
    trial_name: str,
    task_name: str,
    dataset: str,
    app_ref: str,
    source: str,
    resolved_agent_name: str,
    resolved_model_name: str,
    usage: dict[str, Any],
    test_case_id: str | None = None,
    evaluation_id: str | None = None,
) -> dict[str, Any]:
    steps = atif_steps_payload(trajectory)
    task_id = trial_result.get("task_id") or task_name
    verifier_result = trial_result.get("verifier_result")
    resolved_test_case_id = test_case_id or _scalar_task_id(task_id) or task_name
    payload: dict[str, Any] = {
        "schema_version": atif_schema_version(trajectory),
        "session_id": session_id,
        "agent": atif_agent_payload(trajectory, resolved_agent_name, resolved_model_name),
        "final_metrics": atif_final_metrics_payload(trajectory, steps, usage),
        "steps": steps,
        "extra": {
            "source": source,
            "format": "atif",
            "app": app_ref,
            "external_id": external_id,
            "job_id": job_id,
            "job_name": str(job_config.get("job_name") or job_dir.name),
            "job_dir": str(job_dir),
            "trial_dir": str(trial_dir),
            "trial_id": trial_id,
            "trial_name": trial_name,
            "task_id": task_id,
            "task_name": task_name,
            "trial_uri": f"harbor://{job_id}/{trial_name}",
            "started_at": trial_result.get("started_at"),
            "finished_at": trial_result.get("finished_at"),
            "exception_info": trial_result.get("exception_info"),
            "verifier_result": verifier_result,
            "verifier": {
                "reward": reward_from_trial(trial_result),
                "tool_calls": count_tool_calls(trajectory),
            },
            "experiment": {
                "id": job_id,
                "job_name": str(job_config.get("job_name") or job_dir.name),
                "task_id": trial_result.get("task_id"),
                "start_time": job_result.get("started_at"),
                "num_attempts": job_config.get("n_attempts"),
                "num_trials": job_result.get("n_total_trials"),
                "num_errors": total_errors(job_result),
                "model": resolved_model_name,
                "agent_name": resolved_agent_name,
                "dataset_name": dataset,
                "source": source,
                "job_dir": str(job_dir),
            },
            "runtime": runtime_metadata(),
            "trajectory_session_id": trajectory.get("session_id"),
        },
    }
    if evaluation_id:
        # Canonical intake context (post experiment_context deprecation): only evaluation_id +
        # test_case_id survive ingest. Run/trial metadata lives in extra (raw) and on the
        # Experiment entity, not here.
        payload["evaluation_context"] = {
            "evaluation_id": evaluation_id,
            "test_case_id": resolved_test_case_id,
        }
    if trajectory.get("continued_trajectory_ref"):
        payload["continued_trajectory_ref"] = trajectory["continued_trajectory_ref"]
    if trajectory.get("notes"):
        payload["notes"] = trajectory["notes"]
    return payload


def trial_payloads(
    job_dir: Path,
    workspace: str,
    app_name: str,
    source: str,
    *,
    evaluation_run_id: str,
    test_case_id: str | None = None,
    evaluation_id: str | None = None,
    routing_stats: dict[str, Any] | None = None,
) -> list[TrialPayload]:
    job_result = load_json_if_exists(job_dir / "result.json")
    job_config = load_json_if_exists(job_dir / "config.json")
    eval_run_id = evaluation_run_id
    if routing_stats is None:
        routing_stats = load_json_if_exists(job_dir / "switchyard" / "routing_stats_final.json")

    payloads: list[TrialPayload] = []
    session_candidates: list[tuple[TrialPayload, Path | None, tuple[str, ...], str | None]] = []
    trial_records: list[tuple[Path, Path | None, dict[str, Any], dict[str, Any], str]] = []
    for trial_dir in sorted(child for child in job_dir.iterdir() if child.is_dir()):
        result_path = trial_dir / "result.json"
        atif_path = trial_dir / "agent" / "trajectory.json"
        if not result_path.exists():
            continue

        try:
            trial_result = load_json(result_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"scaled-evals Intake: skipping {trial_dir.name}: {exc}", file=sys.stderr)
            continue

        if atif_path.exists():
            try:
                trajectory = load_json(atif_path)
                trajectory_status = "native"
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(
                    f"scaled-evals Intake: synthesizing {trial_dir.name} without its unreadable trajectory: {exc}",
                    file=sys.stderr,
                )
                trajectory = {
                    "schema_version": "ATIF-v1.6",
                    "steps": [],
                    "notes": "Native trajectory was unreadable; scaled-evals synthesized this task record.",
                }
                trajectory_status = "unreadable"
        else:
            trajectory = {
                "schema_version": "ATIF-v1.6",
                "steps": [],
                "notes": "No native trajectory was produced; scaled-evals synthesized this task record.",
            }
            trajectory_status = "missing"
        trial_records.append((trial_dir, atif_path, trial_result, trajectory, trajectory_status))

    if not trial_records:
        task_name = test_case_id or str(job_config.get("job_name") or job_dir.name)
        trial_id = f"{evaluation_run_id}:no-trajectory"
        trial_records.append(
            (
                job_dir,
                None,
                {
                    "id": trial_id,
                    "task_name": task_name,
                    "trial_name": evaluation_run_id,
                },
                {
                    "schema_version": "ATIF-v1.6",
                    "steps": [],
                    "notes": "No readable trial result or native trajectory was produced; "
                    "scaled-evals synthesized this task record.",
                },
                "missing",
            )
        )

    for trial_dir, atif_path, trial_result, trajectory, trajectory_status in trial_records:
        trial_id = str(trial_result.get("id") or trial_dir.name)
        trial_name = str(trial_result.get("trial_name") or trial_dir.name)
        task_name = str(trial_result.get("task_name") or "harbor-trial")
        agent_result = trial_result.get("agent_result") or {}
        dataset = dataset_name(job_config, trial_result)
        intake_app = resolved_app_name(app_name, dataset)
        app_ref = f"{workspace}/{intake_app}"
        resolved_agent_name = agent_name(job_config, trial_result)
        native_model = native_atif_model_name(trajectory)
        resolved_model_name = native_model or model_name(job_config, trial_result, trajectory)
        external_id = f"{source}:{evaluation_run_id}:{trial_id}"
        session_id = trial_name
        usage = usage_from_trial(trial_result, trajectory, agent_result, resolved_model_name)
        payload = atif_payload_for_trial(
            trajectory=trajectory,
            job_id=eval_run_id,
            job_config=job_config,
            job_result=job_result,
            trial_result=trial_result,
            job_dir=job_dir,
            trial_dir=trial_dir,
            external_id=external_id,
            session_id=session_id,
            trial_id=trial_id,
            trial_name=trial_name,
            task_name=task_name,
            dataset=dataset,
            app_ref=app_ref,
            source=source,
            resolved_agent_name=resolved_agent_name,
            resolved_model_name=resolved_model_name,
            usage=usage,
            test_case_id=test_case_id,
            evaluation_id=evaluation_id,
        )
        payload["extra"]["job_config"] = job_config
        payload["extra"]["job_result"] = job_result
        payload["extra"]["trial_result"] = trial_result
        if trajectory_status != "native":
            payload["extra"]["trajectory_status"] = trajectory_status
        trial_payload = TrialPayload(external_id=external_id, payload=payload)
        payloads.append(trial_payload)
        candidates = tuple(
            dict.fromkeys(
                value
                for candidate in (trajectory.get("session_id"), trial_name, trial_id)
                if candidate is not None and (value := str(candidate).strip())
            )
        )
        session_candidates.append((trial_payload, atif_path, candidates, native_model))

    for trial_payload, atif_path, candidates, native_model in session_candidates:
        candidate_ids = (*candidates, eval_run_id) if len(payloads) == 1 else candidates
        for candidate in candidate_ids:
            session_stats = switchyard_session_stats(routing_stats, candidate)
            if session_stats is not None:
                final_metrics = trial_payload.payload.get("final_metrics")
                preserve_native_metrics = isinstance(final_metrics, dict) and preserve_native_single_model_metrics(
                    native_model,
                    final_metrics,
                    session_stats,
                )
                if not preserve_native_metrics:
                    hydrate_atif_from_switchyard(
                        trial_payload.payload,
                        session_stats,
                        native_model=native_model,
                    )
                if atif_path is not None and (
                    not preserve_native_metrics or atif_path.with_name(f"{atif_path.name}.bak").is_file()
                ):
                    _persist_hydrated_trajectory(atif_path, trial_payload.payload)
                break

    return payloads


def _scalar_task_id(value: Any) -> str | None:
    """Return a stable producer id without stringifying Harbor task objects."""
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        text = str(value).strip()
        return text or None
    return None
