# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt one Harbor result directory into an SDK :class:`AgentEvalTrial`.

This module owns the complete Harbor trial-data seam: identity, reward and error
normalization, measurements, Harbor-valid result-file discovery, and
collision-safe evidence discovery. Harbor job execution and cache orchestration
remain in :mod:`harbor_runtime`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

from nemo_evaluator_sdk.agent_eval.reward_keys import (
    HarborRewardValueRejection,
    ParsedHarborRewards,
    RewardKeyRejection,
    reward_key_rejection,
)
from nemo_evaluator_sdk.agent_eval.trials import (
    UNKNOWN_ERROR_TYPE,
    AgentEvalTrial,
    AgentEvalTrialStatus,
    AgentOutput,
    TrialError,
    TrialMeasurements,
    standard_evidence_descriptors,
)
from nemo_evaluator_sdk.values.evidence import (
    EVIDENCE_FORMAT_ATIF,
    EVIDENCE_FORMAT_JSON,
    EVIDENCE_FORMAT_OTLP,
    EVIDENCE_FORMAT_TEXT,
    EVIDENCE_TRACE,
    CandidateEvidence,
    EvidenceDescriptor,
    Trajectory,
    read_atif,
)
from nemo_evaluator_sdk.values.otlp import (
    final_output_text,
    parse_resource_spans,
    resource_spans_from_text,
)
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans
from pydantic import ValidationError

logger = logging.getLogger(__name__)

_ATIF_TRACE_SUFFIX = ".atif.json"
_TRIAL_DIR_DESCRIPTION = (
    "Harbor trial output directory for one task attempt. Contains config.json, result.json, "
    "trial.log, agent logs, verifier logs, and collected artifacts."
)
_TRIAL_CONFIG_DESCRIPTION = (
    "Harbor trial configuration snapshot. Records the task, agent, environment, verifier, "
    "artifact collection, and job id used for this attempt."
)
_TRIAL_RESULT_DESCRIPTION = (
    "Harbor trial result JSON. Contains task and trial identifiers, agent info, verifier rewards, "
    "exception info, phase timings, and token or cost usage."
)
_TRIAL_LOG_DESCRIPTIONS = {
    "agent/oracle.txt": "Oracle-agent log captured when Harbor runs the reference solution.",
    "agent/setup/stdout.txt": "Agent setup stdout captured while Harbor uploads the agent and installs dependencies.",
    "exception.txt": "Exception traceback captured by Harbor for a failed trial.",
    "trial.log": (
        "Harbor trial orchestration log covering environment setup, agent execution, verifier execution, "
        "artifact collection, and cleanup."
    ),
    "verifier/reward.json": (
        "Verifier reward JSON written under /logs/verifier. Harbor parses numeric entries from this file "
        "as trial metrics."
    ),
    "verifier/reward.txt": (
        "Verifier scalar reward file written under /logs/verifier. Harbor parses this file as the reward metric."
    ),
    "verifier/test-stderr.txt": "Verifier stderr captured while Harbor runs the task tests from the /tests directory.",
    "verifier/test-stdout.txt": "Verifier stdout captured while Harbor runs the task tests from the /tests directory.",
}
_MAX_TRACEBACK_CHARS = 8192
_HARBOR_EXTRA_REQUIRED_MESSAGE = (
    "Harbor execution and result adaptation require the optional `harbor` extra on Python >=3.12. "
    "If NeMo Platform is installed as a package, run: uv pip install 'harbor>=0.20,<0.21' "
    "(the `all`, `services`, `plugins`, and `nemo-experimentalist-plugin` extras already include it). "
    "From a NeMo Platform source checkout's repository root, run: "
    "uv sync --frozen --package nemo-evaluator-sdk --extra harbor"
)


def _iter_harbor_trial_results(job_dir: Path) -> Iterator[tuple[Path, Mapping[str, Any]]]:
    """Yield Harbor-valid result mappings in deterministic path order.

    Harbor remains a lazy import. The yielded mapping stays uncoerced because the
    SDK intentionally applies stricter reward rules than Harbor's Pydantic model.
    """
    try:
        from harbor.models.trial.result import TrialResult  # ty: ignore[unresolved-import,unused-ignore-comment]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_HARBOR_EXTRA_REQUIRED_MESSAGE) from exc

    for result_path in sorted(job_dir.glob("*/result.json")):
        try:
            result_data = json.loads(result_path.read_bytes())
            TrialResult.model_validate(result_data)
        except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Skipping invalid Harbor trial result %s: %s", result_path, exc)
            continue
        yield result_path.parent, cast(dict[str, Any], result_data)


def _trial_from_harbor_result(
    trial_dir: Path,
    data: Mapping[str, Any],
    *,
    reward_key: str,
) -> AgentEvalTrial:
    """Normalize one completed Harbor attempt into an SDK trial.

    Args:
        trial_dir: Directory containing the Harbor attempt result and evidence files.
        data: Parsed contents of the attempt's ``result.json`` file.
        reward_key: Verifier reward name to expose as the trial's primary reward.

    Returns:
        The normalized trial, including its identity, reward, error, measurements,
        and evidence descriptors.
    """
    task_id = str(data["task_name"])
    trial_id = str(data.get("trial_name") or trial_dir.name)
    rewards = _rewards_mapping(data)
    reward = _primary_reward(rewards, reward_key)
    error = _trial_error(_reported_exception_info(data))

    metadata: dict[str, Any] = {
        "reward": reward,
        **rewards.to_metadata(),
        "harbor_trial_dir": str(trial_dir),
    }
    measurements = _trial_measurements(data)

    is_complete = error is None and reward is not None
    status = AgentEvalTrialStatus.COMPLETED if is_complete else AgentEvalTrialStatus.PARTIAL

    extension_descriptors, atif_trace, otlp_trace = _harbor_extension_evidence(trial_dir)
    descriptors = standard_evidence_descriptors(
        logs_dir=(trial_dir / "agent").resolve(),
        final_state_dir=(trial_dir / "artifacts").resolve(),
        verifier_logs_dir=(trial_dir / "verifier").resolve(),
    )
    descriptors.update(extension_descriptors)

    # Discovery names an OTLP trace from its artifact path alone; only reading it proves it
    # parses. Both the primary trace and the answer follow from that one read.
    otlp_spans = _read_otlp(otlp_trace)
    primary_trace = otlp_trace if otlp_spans is not None else atif_trace
    if primary_trace is not None:
        descriptors[EVIDENCE_TRACE] = primary_trace

    return AgentEvalTrial(
        id=trial_id,
        task_id=task_id,
        status=status,
        output=AgentOutput(
            output_text=_trial_output_text(otlp_spans, atif_trace),
            metadata={"harbor_trial_dir": str(trial_dir)},
        ),
        evidence=CandidateEvidence(descriptors=descriptors),
        error=error,
        measurements=measurements,
        metadata=metadata,
    )


def _trial_output_text(otlp_spans: list[ResourceSpans] | None, atif_trace: EvidenceDescriptor | None) -> str | None:
    """The agent's answer for the trial, from its OTLP trace when that carries one.

    ATIF remains the fallback because an agent may emit no OTLP trace, and OTLP spans that
    carry no output attribute yield nothing to read. Both being absent is normal: oracle and
    nop produce neither.
    """
    if otlp_spans is not None:
        answer = final_output_text(otlp_spans)
        if answer:
            return answer
    if atif_trace is None or atif_trace.ref is None:
        return None
    return _final_agent_message(read_atif(Path(atif_trace.ref)))


def _read_otlp(descriptor: EvidenceDescriptor | None) -> list[ResourceSpans] | None:
    """Parse a trial's OTLP trace, or ``None`` when it is absent or will not read."""
    if descriptor is None or descriptor.ref is None:
        return None
    path = Path(descriptor.ref)
    try:
        return parse_resource_spans(resource_spans_from_text(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as error:
        logger.warning("Ignoring unreadable OTLP trace %s: %s", path, error)
        return None


def _final_agent_message(trajectory: Trajectory | None) -> str | None:
    """The agent's last message, which is its user-visible answer for the trial.

    Only the *last* agent step can be the answer. An earlier one is intermediate reasoning, so an
    agent that ends on an empty message has produced no answer rather than the previous one.
    """
    if trajectory is None:
        return None
    for step in reversed(trajectory.steps):
        if step.source == "agent":
            return step.message or None
    return None


def _add_extension_descriptor(
    descriptors: dict[str, EvidenceDescriptor],
    *,
    family: str,
    relative_path: Path,
    descriptor: EvidenceDescriptor,
) -> None:
    """Register an extension descriptor under canonical and compatible keys.

    The canonical key includes the complete trial-relative path, for example
    ``artifact:artifacts/output.txt``. This keeps the mapping injective when a
    trial contains both ``output.txt`` and ``artifacts/output.txt``.

    Experimentalist consumers remove the leading ``artifacts/`` directory
    while naming collected artifacts, so they look up the same file through the
    legacy alias ``artifact:output.txt``. The alias is retained during migration
    to avoid breaking those consumers. ``setdefault`` ensures that an alias never
    replaces an exact key belonging to another file.

    Args:
        descriptors: Mutable evidence mapping to update.
        family: Evidence key prefix, such as ``artifact``, ``log``, or ``trace``.
        relative_path: Artifact path relative to the Harbor trial directory.
        descriptor: Descriptor that points to the artifact.

    Returns:
        None. The supplied ``descriptors`` mapping is updated in place.
    """
    descriptors[f"{family}:{relative_path.as_posix()}"] = descriptor
    if relative_path.parts and relative_path.parts[0] == "artifacts" and len(relative_path.parts) > 1:
        legacy_path = Path(*relative_path.parts[1:]).as_posix()
        descriptors.setdefault(f"{family}:{legacy_path}", descriptor)


def _is_trial_log_path(relative_path: str) -> bool:
    """Determine whether a trial-relative path is a known Harbor log.

    Args:
        relative_path: POSIX-style path relative to the Harbor trial directory.

    Returns:
        ``True`` for a known Harbor log or agent command stdout file; otherwise
        ``False``.
    """
    if relative_path in _TRIAL_LOG_DESCRIPTIONS:
        return True
    parts = relative_path.split("/")
    return len(parts) == 3 and parts[0] == "agent" and parts[1].startswith("command-") and parts[2] == "stdout.txt"


def _trace_format_from_path(relative_path: Path) -> Literal["otlp", "atif"] | None:
    """Identify the trace format declared by Harbor's artifact path convention.

    This classifies evidence without reading it: Harbor's path convention is the
    contract, so discovery stays a catalog rather than a full ingest.
    ``EvidenceDescriptor.format`` is a parser hint; ATIF content is validated
    lazily when a consumer materializes the trace.

    Args:
        relative_path: Artifact path relative to the Harbor trial directory.

    Returns:
        ``"atif"`` or ``"otlp"`` for a supported file below a ``traces``
        directory, or ``None`` when the path does not declare a recognized trace
        format.
    """
    if "traces" not in relative_path.parts[:-1]:
        return None
    if relative_path.as_posix().endswith(_ATIF_TRACE_SUFFIX):
        return EVIDENCE_FORMAT_ATIF
    if relative_path.suffix == ".jsonl":
        return EVIDENCE_FORMAT_OTLP
    return None


def _display_artifact_path(relative_path: Path) -> Path:
    """Produce the historical display path for a Harbor artifact.

    Args:
        relative_path: Artifact path relative to the Harbor trial directory.

    Returns:
        The path without a leading ``artifacts/`` component when present, or the
        original relative path otherwise.
    """
    if relative_path.parts and relative_path.parts[0] == "artifacts" and len(relative_path.parts) > 1:
        return Path(*relative_path.parts[1:])
    return relative_path


def _harbor_extension_evidence(
    trial_dir: Path,
) -> tuple[dict[str, EvidenceDescriptor], EvidenceDescriptor | None, EvidenceDescriptor | None]:
    """Describe every Harbor trial file and pick the standard trace.

    Every discovered file remains available through an extension evidence key.
    The first trace of each encoding is additionally registered under ``trace:atif``
    and ``trace:otlp``, so a metric can read either view of a trial that emits both.
    Harbor's built-in ATIF dump ``agent/trajectory.json`` is used as a fallback when
    no path-declared ATIF file is present.

    Args:
        trial_dir: Harbor attempt directory to inspect recursively.

    Returns:
        All discovered extension descriptors, then the ATIF and OTLP traces, either of
        which is ``None`` when the trial emitted no trace of that encoding. Choosing
        between them is the caller's, which can read the OTLP trace before promoting it.
    """
    descriptors: dict[str, EvidenceDescriptor] = {
        "trial_dir": EvidenceDescriptor(
            kind="filesystem",
            format="dir",
            ref=str(trial_dir.resolve()),
            description=_TRIAL_DIR_DESCRIPTION,
        )
    }
    collected_traces: dict[str, EvidenceDescriptor | None] = {
        EVIDENCE_FORMAT_OTLP: None,
        EVIDENCE_FORMAT_ATIF: None,
    }
    # ATIF from Harbor's older dump path ``agent/trajectory.json``. Used only when
    # no canonical ATIF trace is found under ``traces/*.atif.json``.
    legacy_atif: EvidenceDescriptor | None = None

    try:
        files = sorted(path for path in trial_dir.rglob("*") if path.is_file())
    except OSError as exc:
        logger.warning("Could not enumerate Harbor trial evidence under %s: %s", trial_dir, exc)
        files = []

    for path in files:
        relative_path = path.relative_to(trial_dir)
        relative_name = relative_path.as_posix()
        try:
            ref = str(path.resolve())
        except OSError as exc:
            logger.warning("Skipping unreadable Harbor trial evidence %s: %s", path, exc)
            continue

        if relative_name == "config.json":
            descriptor = EvidenceDescriptor(
                kind="json",
                format=EVIDENCE_FORMAT_JSON,
                ref=ref,
                description=_TRIAL_CONFIG_DESCRIPTION,
            )
            descriptors["config"] = descriptor
            _add_extension_descriptor(
                descriptors,
                family="json",
                relative_path=relative_path,
                descriptor=descriptor,
            )
            continue
        if relative_name == "result.json":
            descriptor = EvidenceDescriptor(
                kind="json",
                format=EVIDENCE_FORMAT_JSON,
                ref=ref,
                description=_TRIAL_RESULT_DESCRIPTION,
            )
            descriptors["result"] = descriptor
            _add_extension_descriptor(
                descriptors,
                family="json",
                relative_path=relative_path,
                descriptor=descriptor,
            )
            continue

        discovered_format = _trace_format_from_path(relative_path)
        if discovered_format is not None:
            display_path = _display_artifact_path(relative_path).as_posix()
            description = (
                f"Agent execution ATIF trajectory for {display_path}."
                if discovered_format == EVIDENCE_FORMAT_ATIF
                else f"Agent execution trace JSONL for {display_path}."
            )
            descriptor = EvidenceDescriptor(
                kind=EVIDENCE_TRACE,
                format=discovered_format,
                ref=ref,
                description=description,
            )
            _add_extension_descriptor(
                descriptors,
                family=EVIDENCE_TRACE,
                relative_path=relative_path,
                descriptor=descriptor,
            )
            if collected_traces[discovered_format] is None:
                collected_traces[discovered_format] = descriptor
            continue

        if relative_name == "agent/trajectory.json" and read_atif(path) is not None:
            descriptor = EvidenceDescriptor(
                kind=EVIDENCE_TRACE,
                format=EVIDENCE_FORMAT_ATIF,
                ref=ref,
                description="Collected Harbor artifact agent/trajectory.json.",
            )
            _add_extension_descriptor(
                descriptors,
                family=EVIDENCE_TRACE,
                relative_path=relative_path,
                descriptor=descriptor,
            )
            legacy_atif = descriptor
            continue

        if _is_trial_log_path(relative_name):
            description = _TRIAL_LOG_DESCRIPTIONS.get(relative_name)
            if description is None and relative_name.startswith("agent/command-"):
                description = "Agent command stdout captured while Harbor runs the benchmark agent."
            descriptor = EvidenceDescriptor(
                kind="log",
                format=EVIDENCE_FORMAT_JSON if path.suffix == ".json" else EVIDENCE_FORMAT_TEXT,
                ref=ref,
                description=description or f"Harbor trial log {relative_name}.",
            )
            _add_extension_descriptor(
                descriptors,
                family="log",
                relative_path=relative_path,
                descriptor=descriptor,
            )
            continue

        display_path = _display_artifact_path(relative_path).as_posix()
        description = f"Collected Harbor artifact {display_path}."
        if relative_path.parts[0] == "artifacts" and display_path == "manifest.json":
            description = (
                "Harbor artifact manifest. Lists collected artifact files and the environment paths they were "
                "copied from."
            )
        descriptor = EvidenceDescriptor(
            kind="artifact",
            ref=ref,
            description=description,
        )
        _add_extension_descriptor(
            descriptors,
            family="artifact",
            relative_path=relative_path,
            descriptor=descriptor,
        )

    atif_trace = collected_traces[EVIDENCE_FORMAT_ATIF]
    if atif_trace is None:
        atif_trace = legacy_atif
    otlp_trace = collected_traces[EVIDENCE_FORMAT_OTLP]
    for evidence_format, descriptor in ((EVIDENCE_FORMAT_ATIF, atif_trace), (EVIDENCE_FORMAT_OTLP, otlp_trace)):
        if descriptor is not None:
            descriptors[f"{EVIDENCE_TRACE}:{evidence_format}"] = descriptor

    return descriptors, atif_trace, otlp_trace


def _rewards_mapping(data: Mapping[str, Any]) -> ParsedHarborRewards:
    """Normalize Harbor verifier rewards into numeric SDK values.

    Args:
        data: Parsed Harbor ``result.json`` payload.

    Returns:
        Finite numeric values plus sanitized rejection diagnostics. Unusable
        values and unsafe keys are omitted from the numeric mapping.
    """
    verifier_result = data.get("verifier_result")
    if not isinstance(verifier_result, Mapping):
        return ParsedHarborRewards()
    rewards = verifier_result.get("rewards")
    if not isinstance(rewards, Mapping):
        return ParsedHarborRewards()
    values: dict[str, float] = {}
    rejected_by_key: dict[str, HarborRewardValueRejection] = {}
    rejected_entries: list[RewardKeyRejection] = []
    for key, value in rewards.items():
        key_rejection = reward_key_rejection(key)
        if key_rejection is not None:
            rejected_entries.append(key_rejection)
            continue
        assert type(key) is str
        if isinstance(value, bool):
            rejected_by_key[key] = "boolean"
            continue
        if type(value) not in (int, float, str):
            rejected_by_key[key] = "non_numeric"
            continue
        try:
            parsed = float(value)
        except OverflowError:  # an int too large to represent
            rejected_by_key[key] = "non_finite"
            continue
        except ValueError:  # a string that is not a number
            rejected_by_key[key] = "non_numeric"
            continue
        if not math.isfinite(parsed):
            rejected_by_key[key] = "non_finite"
            continue
        values[key] = parsed
    return ParsedHarborRewards(
        values=values,
        rejected_by_key=rejected_by_key,
        rejected_entries=tuple(rejected_entries),
    )


def _primary_reward(rewards: ParsedHarborRewards, reward_key: str) -> float | None:
    """Select the configured primary reward.

    Args:
        rewards: Normalized verifier rewards keyed by reward name.
        reward_key: Reward name selected by the runtime configuration.

    Returns:
        The selected reward, or ``None`` when ``reward_key`` is absent. A warning
        is emitted when other rewards exist but none matches the configured key.
    """
    if reward_key in rewards.values:
        return rewards.values[reward_key]
    if reason := rewards.rejected_by_key.get(reward_key):
        logger.warning(
            "Harbor trial reward_key=%r was emitted but rejected as %s; treating as unusable",
            reward_key,
            reason,
        )
    elif rewards.values:
        logger.warning(
            "Harbor trial emitted rewards %s but none matches reward_key=%r; treating the trial as having no reward",
            sorted(rewards.values),
            reward_key,
        )
    return None


def _reported_exception_info(data: Mapping[str, Any]) -> Any:
    """The exception Harbor recorded for a trial, from the trial or from whichever step raised.

    Harbor sets the top-level ``exception_info`` for a single-step trial, but a multi-step trial
    leaves it ``None`` and records the exception on the ``step_results`` entry that raised.

    Args:
        data: Parsed Harbor ``result.json`` payload.

    Returns:
        The first exception payload found, trial-level first, or ``None`` when no step raised.
    """
    trial_level = data.get("exception_info")
    if trial_level is not None:
        return trial_level
    steps = data.get("step_results")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        return None
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        step_level = step.get("exception_info")
        if step_level is not None:
            return step_level
    return None


def _trial_error(exception_info: Any) -> TrialError | None:
    """Normalize any Harbor ``exception_info`` shape without raising.

    Args:
        exception_info: Exception payload emitted by Harbor, which may be a
            mapping, scalar value, or ``None``.

    Returns:
        A normalized trial error with a bounded traceback, or ``None`` when Harbor
        reported no exception.
    """
    if exception_info is None:
        return None
    if not isinstance(exception_info, Mapping):
        return TrialError(type=str(exception_info).strip() or UNKNOWN_ERROR_TYPE)

    error_type = ""
    for key in ("exception_type", "type", "name", "class"):
        value = exception_info.get(key)
        if isinstance(value, str) and value.strip():
            error_type = value
            break

    traceback = _first_string(exception_info, ("exception_traceback", "traceback"))
    return TrialError(
        type=error_type or UNKNOWN_ERROR_TYPE,
        message=_first_string(exception_info, ("exception_message", "message")),
        traceback=traceback[:_MAX_TRACEBACK_CHARS] if traceback is not None else None,
        occurred_at=_error_timestamp(exception_info.get("occurred_at")),
    )


def _first_string(payload: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    """Find the first string value among candidate mapping keys.

    Args:
        payload: Mapping to inspect.
        keys: Candidate keys in lookup order.

    Returns:
        The first string value found, including an empty string, or ``None`` when
        no candidate contains a string.
    """
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def _error_timestamp(value: Any) -> datetime | None:
    """Parse an error timestamp without inventing a timezone.

    Args:
        value: Harbor timestamp value, typically a ``datetime`` or ISO-formatted
            string.

    Returns:
        The supplied ``datetime`` or a parsed ISO timestamp. Returns ``None`` for
        unsupported or invalid values and preserves whether the timestamp is
        timezone-aware or naive.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        with contextlib.suppress(ValueError):
            return datetime.fromisoformat(value)
    return None


def _token_measurements(agent_result: Any, *, trial_id: str) -> tuple[dict[str, int | float], set[str]]:
    """Extract token counts and cost from one Harbor agent result.

    Args:
        agent_result: Harbor agent-result payload to inspect.

    Returns:
        Valid SDK measurement values and fields that were explicitly invalid.
    """
    if not isinstance(agent_result, Mapping):
        return {}, set()
    mapping = {
        "prompt_tokens": "n_input_tokens",
        "completion_tokens": "n_output_tokens",
        "cache_read_tokens": "n_cache_tokens",
    }
    out: dict[str, int | float] = {}
    invalid: set[str] = set()
    for sdk_key, harbor_key in mapping.items():
        if harbor_key not in agent_result or agent_result[harbor_key] is None:
            continue
        value = agent_result.get(harbor_key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            out[sdk_key] = value
        else:
            invalid.add(sdk_key)
            logger.warning(
                "Harbor trial %s has invalid %s=%r; omitting %s",
                trial_id,
                harbor_key,
                value,
                sdk_key,
            )
    if "cost_usd" in agent_result and agent_result["cost_usd"] is not None:
        cost = agent_result["cost_usd"]
        if _is_finite_nonnegative_number(cost):
            out["cost_usd"] = float(cost)
        else:
            invalid.add("cost_usd")
            logger.warning("Harbor trial %s has invalid cost_usd=%r; omitting cost_usd", trial_id, cost)
    return out, invalid


def _trial_measurements(data: Mapping[str, Any]) -> TrialMeasurements:
    """Build canonical Harbor measurements from one result source.

    A top-level ``agent_result`` takes precedence. When it is absent, valid
    measurements from step-level agent results are aggregated exactly once.

    Args:
        data: Parsed Harbor ``result.json`` payload.

    Returns:
        Validated measurements; malformed optional source values are omitted.
    """
    trial_id = str(data.get("trial_name") or data.get("task_name") or "unknown")
    top_level = data.get("agent_result")
    if isinstance(top_level, Mapping):
        values, _ = _token_measurements(top_level, trial_id=trial_id)
    else:
        values = _aggregate_step_measurements(data.get("step_results"), trial_id=trial_id)

    runtime = _harbor_runtime_sec(data, trial_id=trial_id)
    if runtime is not None:
        values["runtime_sec"] = runtime
    return TrialMeasurements.model_validate(values)


def _aggregate_step_measurements(step_results: Any, *, trial_id: str) -> dict[str, int | float]:
    if not isinstance(step_results, list):
        return {}

    contributors: dict[str, list[int | float]] = {}
    invalid: set[str] = set()
    for step in step_results:
        if not isinstance(step, Mapping):
            continue
        captured, rejected = _token_measurements(step.get("agent_result"), trial_id=trial_id)
        invalid.update(rejected)
        for key, value in captured.items():
            contributors.setdefault(key, []).append(value)

    totals: dict[str, int | float] = {}
    for key, values in contributors.items():
        if key in invalid:
            continue
        try:
            total = math.fsum(float(value) for value in values) if key == "cost_usd" else sum(int(v) for v in values)
        except OverflowError:
            total = float("inf")
        if _is_finite_nonnegative_number(total):
            totals[key] = float(total) if key == "cost_usd" else int(total)
        else:
            logger.warning("Harbor trial %s has invalid aggregate %s=%r; omitting it", trial_id, key, total)
    return totals


def _harbor_runtime_sec(data: Mapping[str, Any], *, trial_id: str) -> float | None:
    top_level = data.get("agent_execution")
    if top_level is not None:
        duration = _execution_duration(top_level)
        if duration is None:
            logger.warning(
                "Harbor trial %s has an invalid top-level agent_execution window; omitting runtime", trial_id
            )
        return duration

    step_results = data.get("step_results")
    if not isinstance(step_results, list):
        return None
    durations: list[float] = []
    for step in step_results:
        if not isinstance(step, Mapping) or step.get("agent_execution") is None:
            continue
        duration = _execution_duration(step.get("agent_execution"))
        if duration is None:
            logger.warning("Harbor trial %s has an invalid step agent_execution window; omitting runtime", trial_id)
            return None
        durations.append(duration)
    if not durations:
        return None
    try:
        total = math.fsum(durations)
    except OverflowError:
        total = float("inf")
    if not math.isfinite(total):
        logger.warning("Harbor trial %s has a non-finite aggregate runtime; omitting runtime", trial_id)
        return None
    return total


def _execution_duration(window: Any) -> float | None:
    if not isinstance(window, Mapping):
        return None
    started = _as_datetime(window.get("started_at"))
    finished = _as_datetime(window.get("finished_at"))
    if started is None or finished is None:
        return None
    try:
        duration = (finished - started).total_seconds()
    except TypeError:
        return None
    if not math.isfinite(duration) or duration < 0:
        return None
    return duration


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        with contextlib.suppress(ValueError):
            return datetime.fromisoformat(value)
    return None


def _is_finite_nonnegative_number(value: Any) -> bool:
    if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False
