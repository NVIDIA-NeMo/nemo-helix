# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pieces of the NeMo Fabric agent-eval runtime.

:class:`~nemo_evaluator_sdk.agent_eval.runtimes.fabric.runtime.FabricAgentRuntime` runs a Fabric
config either in-process on the host or inside a sandbox
(:mod:`~nemo_evaluator_sdk.agent_eval.runtimes.fabric._sandbox_execution`). Both paths hand their
outcome back through the types here, so one trial-mapping step serves both.

Trajectory capture is built from ``nemo_fabric``'s own typed config objects (a hard dependency), so
Fabric owns the schema: a breaking Fabric change fails construction here rather than silently
producing a block Fabric ignores.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from nemo_evaluator_sdk.agent_eval.runtimes.fabric.otlp_writer import otlp_endpoint_fields
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import SkillProvenance
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, TrialMeasurements
from nemo_evaluator_sdk.values.evidence import CandidateEvidence, EvidenceDescriptor
from pydantic import JsonValue

# The file-exporter output names we choose (Relay accepts these as inputs).
ATIF_FILENAME_TEMPLATE = "trajectory-{session_id}.atif.json"
ATOF_FILENAME = "events.atof.jsonl"
#: ATIF ``agent.version``. Reports the agent *framework* rather than a real version; the resolved
#: nemo-fabric version would be the better answer.
FABRIC_AGENT_VERSION = "fabric"
#: ``kind`` Fabric stamps on the promoted Relay ATIF artifact.
ATIF_ARTIFACT_KIND = "atif"


def to_mapping(config: Any) -> dict[str, Any]:
    """Normalize a typed Fabric config or a plain mapping to a plain dict."""
    # A typed Fabric config exposes ``to_mapping()``; a plain mapping is used as-is. Both are
    # str-keyed at runtime, but the getattr + optional (unresolved) ``FabricConfig`` type defeat static
    # narrowing, so cast the known-good source before building the dict.
    to_mapping_method = getattr(config, "to_mapping", None)
    source = to_mapping_method() if callable(to_mapping_method) else config
    return dict(cast(Mapping[str, Any], source))


def safe_path_name(value: str) -> str:
    """Filesystem-safe rendering of an arbitrary id (alnum/``._-`` kept, else ``-``; trimmed to 120)."""
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in value).strip(".-")[:120]


def task_subdir_name(index: int, task_id: str) -> str:
    """Deterministic per-task evidence subdir name (``000000-<safe-id>``)."""
    safe = safe_path_name(task_id)
    return f"{index:06d}-{safe}" if safe else f"task-{index:06d}"


def extract_output_text(output: object) -> str | None:
    """Pull the user-visible message out of a Fabric output value (already unwrapped from the result).

    Harness outputs vary; adapters commonly nest the final message under ``response`` (the codex-cli
    adapter does). Prefer a string ``response``/``output_text``/``text``/``message``, else stringify.
    """
    if output is None:
        return None
    if isinstance(output, str):
        return output
    if isinstance(output, Mapping):
        for key in ("response", "output_text", "text", "message"):
            value = output.get(key)
            if isinstance(value, str):
                return value
    return json.dumps(output, default=str)


def normalize_output(output: Any) -> JsonValue:
    """Unwrap a Fabric ``RunResult.output`` into the plain JSON value the trial response stores.

    Fabric wraps object outputs in a ``RunOutput`` mapping; copy it into a plain dict. Raw JSON
    outputs pass through unchanged.
    """
    if isinstance(output, Mapping):
        return dict(output)
    return cast(JsonValue, output)


@dataclass(frozen=True)
class ArtifactView:
    """One entry of a Fabric artifact manifest, with ``path`` already on the host filesystem."""

    name: str
    kind: str | None
    path: str
    media_type: str | None


@dataclass(frozen=True)
class ResultView:
    """A Fabric ``RunResult`` read the same way however it arrived.

    On the host the result is the SDK object; in a sandbox it is the JSON the in-sandbox driver
    printed. Trial mapping only ever sees this view, so both modes produce one trial shape.
    """

    payload: Mapping[str, Any]
    status: str
    output: JsonValue
    error: Mapping[str, Any] | None
    harness: str | None
    adapter_id: str | None
    adapter_kind: str | None
    invocation_id: str | None
    artifacts: tuple[ArtifactView, ...]
    telemetry: tuple[dict[str, Any], ...]
    events: tuple[dict[str, Any], ...]

    @classmethod
    def from_result(cls, result: Any) -> ResultView:
        """View a ``nemo_fabric.RunResult`` (or anything with its attributes)."""
        error = result.error
        return cls(
            payload=result.to_mapping(),
            status=str(result.status),
            output=normalize_output(result.output),
            error=None if error is None else {"stage": error.stage, "code": error.code, "message": error.message},
            harness=result.harness,
            adapter_id=result.adapter_id,
            adapter_kind=result.adapter_kind,
            invocation_id=result.invocation_id,
            artifacts=tuple(
                ArtifactView(
                    name=artifact.name, kind=artifact.kind, path=str(artifact.path), media_type=artifact.media_type
                )
                for artifact in result.artifacts.artifacts
            ),
            telemetry=tuple(
                {"provider": ref.provider, "kind": ref.kind, "uri": ref.uri, "trace_id": ref.trace_id}
                for ref in result.telemetry
            ),
            events=tuple({"kind": event.kind, "message": event.message} for event in result.events),
        )

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, artifact_path: Callable[[str], str | None] = lambda path: path
    ) -> ResultView:
        """View the ``RunResult.to_mapping()`` JSON a sandbox driver printed.

        ``artifact_path`` maps each artifact's in-sandbox path to where it now lives on the host;
        returning ``None`` drops an artifact that was not brought across.
        """
        error = payload.get("error")
        artifacts: list[ArtifactView] = []
        manifest = payload.get("artifacts")
        entries = manifest.get("artifacts") if isinstance(manifest, Mapping) else None
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, Mapping) or not entry.get("name") or not entry.get("path"):
                continue
            path = artifact_path(str(entry["path"]))
            if path is None:
                continue
            artifacts.append(
                ArtifactView(
                    name=str(entry["name"]),
                    kind=_optional_str(entry.get("kind")),
                    path=path,
                    media_type=_optional_str(entry.get("media_type")),
                )
            )
        return cls(
            payload=payload,
            status=str(payload.get("status")),
            output=normalize_output(payload.get("output")),
            error={"stage": error.get("stage"), "code": error.get("code"), "message": error.get("message")}
            if isinstance(error, Mapping)
            else None,
            harness=_optional_str(payload.get("harness")),
            adapter_id=_optional_str(payload.get("adapter_id")),
            adapter_kind=_optional_str(payload.get("adapter_kind")),
            invocation_id=_optional_str(payload.get("invocation_id")),
            artifacts=tuple(artifacts),
            telemetry=tuple(
                {key: ref.get(key) for key in ("provider", "kind", "uri", "trace_id")}
                for ref in _mapping_list(payload.get("telemetry"))
            ),
            events=tuple(
                {key: event.get(key) for key in ("kind", "message")} for event in _mapping_list(payload.get("events"))
            ),
        )

    def failure(self) -> Mapping[str, Any]:
        """The error mapping a non-``succeeded`` result is graded from."""
        if self.error is None:
            return {"code": self.status, "message": "Fabric run did not succeed"}
        return self.error


@dataclass
class TaskRun:
    """What one execution mode hands back for one task: a result, or the exception that stopped it.

    Paths are on the host. The sandbox mode downloads its ``/out`` tree into the evidence dir first,
    so ``workspace_dir`` and ``relay_dir`` sit in the same place for both modes.
    """

    workspace_dir: Path
    relay_dir: Path
    skill_provenances: list[SkillProvenance] = field(default_factory=list)
    result: ResultView | None = None
    error: Exception | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _mapping_list(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [cast(Mapping[str, Any], item) for item in value if isinstance(item, Mapping)]


def build_failed_trial(
    task: AgentEvalTask,
    evidence_dir: Path,
    error: Exception | Mapping[str, Any],
    *,
    runtime_name: str,
    trial_id_suffix: str,
    extra_metadata: Mapping[str, Any] | None = None,
    measurements: TrialMeasurements | None = None,
) -> AgentEvalTrial:
    """Persist ``error.json`` and build a FAILED trial with the standard error evidence + metadata.

    ``error`` is either a raised exception or a Fabric error mapping (``stage``/``code``/``message``).
    """
    if isinstance(error, Mapping):
        error_type = str(error.get("code") or error.get("stage") or "FabricError")
        error_message = str(error.get("message") or error)
    else:
        error_type = error.__class__.__name__
        error_message = str(error)
    error_path = evidence_dir / "error.json"
    error_path.write_text(json.dumps({"error_type": error_type, "error": error_message}) + "\n", encoding="utf-8")
    return AgentEvalTrial(
        id=f"{task.id}:{trial_id_suffix}",
        task_id=task.id,
        status=AgentEvalTrialStatus.FAILED,
        output=None,
        evidence=CandidateEvidence(
            descriptors={"error": EvidenceDescriptor(kind="error", format="json", ref=str(error_path))},
            metadata={"runtime": runtime_name},
        ),
        measurements=measurements if measurements is not None else TrialMeasurements(),
        metadata={
            **(dict(extra_metadata) if extra_metadata else {}),
            "runtime": runtime_name,
            "error_type": error_type,
            "error": error_message,
            # A failed trial did not complete its agent phase; stamp it explicitly so
            # AgentPhaseSuccessMetric scores it False rather than by omission.
            "agent_ok": False,
        },
    )


def relay_observability(
    *,
    relay_dir: str,
    agent_name: str,
    agent_version: str,
    extra: Mapping[str, Any] | None = None,
    otlp_endpoint: str | None = None,
) -> Any:
    """Relay's ATIF/ATOF file-exporter observability config, as ``nemo_fabric.RelayObservabilityConfig``.

    Built from Fabric's own typed relay models rather than ``nemo_relay``'s: Fabric is what has to
    accept the block, so a breaking change there fails construction here instead of being silently
    dropped from a config Fabric no longer understands.

    ``nemo_fabric`` is imported here rather than at module scope: it is a native extension, and this
    module is reachable from the evaluator plugin's job imports, so an eager import would charge every
    consumer for trajectory capture they may never use.
    """
    from nemo_fabric import (  # ty: ignore[unresolved-import]
        RelayAtifConfig,
        RelayAtofConfig,
        RelayAtofFileSinkConfig,
        RelayObservabilityConfig,
        RelayOpenTelemetryConfig,
        RelayOpenTelemetryEndpointConfig,
    )

    opentelemetry = None
    if otlp_endpoint is not None:
        fields = otlp_endpoint_fields(endpoint=otlp_endpoint, service_name=agent_name)
        opentelemetry = RelayOpenTelemetryConfig(enabled=True, endpoints=[RelayOpenTelemetryEndpointConfig(**fields)])
    return RelayObservabilityConfig(
        opentelemetry=opentelemetry,
        atif=RelayAtifConfig(
            enabled=True,
            output_directory=relay_dir,
            filename_template=ATIF_FILENAME_TEMPLATE,
            agent_name=agent_name,
            agent_version=agent_version,
            extra=dict(extra) if extra else None,
        ),
        atof=RelayAtofConfig(
            enabled=True,
            sinks=[
                RelayAtofFileSinkConfig(
                    output_directory=relay_dir,
                    filename=ATOF_FILENAME,
                    mode="overwrite",
                )
            ],
        ),
    )


def relay_telemetry_fragment(
    config: Mapping[str, Any],
    *,
    relay_dir: str,
    agent_name: str,
    agent_version: str,
    extra: Mapping[str, Any] | None = None,
    otlp_endpoint: str | None = None,
) -> dict[str, Any]:
    """The Fabric config keys that enable Relay's file exporter, serialized by Fabric itself.

    ``config`` supplies only the identity Fabric requires to validate (metadata + harness/workflow);
    nothing else about it is carried over. Returning Fabric's own serialization of the keys, rather
    than a hand-written envelope, is what keeps the block in whatever shape Fabric currently reads —
    a config Fabric does not recognize is ignored rather than rejected, so a drifted envelope
    produces a run with no trajectory at all and no error.
    """
    from nemo_fabric import FabricConfig  # ty: ignore[unresolved-import]

    probe = FabricConfig.from_mapping(config)
    probe.enable_relay(
        output_dir=relay_dir,
        observability=relay_observability(
            relay_dir=relay_dir,
            agent_name=agent_name,
            agent_version=agent_version,
            extra=extra,
            otlp_endpoint=otlp_endpoint,
        ),
    )
    mapping = probe.to_mapping()
    return {key: mapping[key] for key in ("telemetry", "relay")}
