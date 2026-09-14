# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the host and containerized NeMo Fabric agent-eval runtimes.

:class:`~nemo_evaluator_sdk.agent_eval.runtimes.fabric.runtime.FabricAgentRuntime` (host) and
:class:`~nemo_evaluator_sdk.agent_eval.runtimes.fabric.container_runtime.FabricContainerRuntime`
(sandbox) map a Fabric ``RunResult`` to the *same* trial/evidence contract, so the pieces they share
live here — one definition, so the two runtimes cannot drift apart.

Trajectory capture is built from ``nemo_fabric``'s own typed config objects (a hard dependency), so
Fabric owns the schema: a breaking Fabric change fails construction here rather than silently
producing a block Fabric ignores.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nemo_evaluator_sdk.agent_eval.runtimes.fabric.otlp_writer import otlp_endpoint_fields
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus
from nemo_evaluator_sdk.values.evidence import CandidateEvidence, EvidenceDescriptor

# The file-exporter output names we choose (Relay accepts these as inputs). Shared so both runtimes
# emit the trajectory under identical names.
ATIF_FILENAME_TEMPLATE = "trajectory-{session_id}.atif.json"
ATOF_FILENAME = "events.atof.jsonl"
#: ATIF ``agent.version``. Both runtimes report the agent *framework* here so a consumer can group
#: host and container traces together; ``agent.name`` is what distinguishes them. Not a real version
#: yet — reporting the resolved nemo-fabric version would be the better answer.
FABRIC_AGENT_VERSION = "fabric"


def safe_path_name(value: str) -> str:
    """Filesystem-safe rendering of an arbitrary id (alnum/``._-`` kept, else ``-``; trimmed to 120)."""
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in value).strip(".-")[:120]


def task_subdir_name(index: int, task_id: str) -> str:
    """Deterministic per-task evidence subdir name (``000000-<safe-id>``) shared by both runtimes."""
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


def build_failed_trial(
    task: AgentEvalTask,
    evidence_dir: Path,
    error: Exception | Mapping[str, Any],
    *,
    runtime_name: str,
    trial_id_suffix: str,
    extra_metadata: Mapping[str, Any] | None = None,
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
        metadata={
            **(dict(extra_metadata) if extra_metadata else {}),
            "runtime": runtime_name,
            "error_type": error_type,
            "error": error_message,
            # A failed trial did not complete its agent phase; stamp it explicitly (matching the host
            # Fabric/Codex runtimes) so AgentPhaseSuccessMetric scores it False rather than by omission.
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
            otlp_endpoint=otlp_endpoint,
        ),
    )
    mapping = probe.to_mapping()
    return {key: mapping[key] for key in ("telemetry", "relay")}
