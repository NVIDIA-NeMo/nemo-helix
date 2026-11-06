# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The trace evidence a Fabric run declares, read the same way by every runtime that publishes it.

Both the Fabric runtime and Gym (whose registered agent drives Fabric) surface a Relay trace a
run left on disk. The pieces they share live here rather than in either runtime's package, so
neither imports the other: how a run's artifact manifest is read, and how a trace is filed as
evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nhx_evals_sdk.values.evidence import (
    EVIDENCE_FORMAT_ATIF,
    EVIDENCE_FORMAT_OTLP,
    EVIDENCE_TRACE,
    EvidenceDescriptor,
)

#: ``kind`` Fabric stamps on the promoted Relay ATIF artifact.
ATIF_ARTIFACT_KIND = "atif"
#: ``kind`` Fabric stamps on the Relay OpenTelemetry file it declares as a run artifact.
OTLP_ARTIFACT_KIND = "otlp"


@dataclass(frozen=True)
class ArtifactView:
    """One entry of a Fabric artifact manifest, with ``path`` already on the host filesystem."""

    name: str
    kind: str | None
    path: str
    media_type: str | None


def declared_artifacts(
    result: Mapping[str, Any], *, artifact_path: Callable[[str], str | None] = lambda path: path
) -> tuple[ArtifactView, ...]:
    """The artifacts a ``RunResult.to_mapping()`` payload declares, skipping malformed entries.

    ``artifact_path`` maps each artifact's recorded path to where it now lives on the host;
    returning ``None`` drops an artifact that was not brought across.
    """
    manifest = result.get("artifacts")
    entries = manifest.get("artifacts") if isinstance(manifest, Mapping) else None
    artifacts: list[ArtifactView] = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, Mapping) or not entry.get("name") or not entry.get("path"):
            continue
        path = artifact_path(str(entry["path"]))
        if path is None:
            continue
        kind, media_type = entry.get("kind"), entry.get("media_type")
        artifacts.append(
            ArtifactView(
                name=str(entry["name"]),
                kind=None if kind is None else str(kind),
                path=path,
                media_type=None if media_type is None else str(media_type),
            )
        )
    return tuple(artifacts)


def register_trace_evidence(
    descriptors: dict[str, EvidenceDescriptor], *, atif: Path | None, otlp: Path | None
) -> None:
    """File each trace under its format-qualified key, and the primary one under ``trace``.

    OTLP takes the primary key when it exists, ATIF otherwise -- the rule Harbor and Gym follow, so
    a metric that asks for neither format gets the same kind of answer whatever ran the trial.

    The two are admitted on different terms. ATIF is registered on a path alone, because Fabric
    declares it in the run's artifact manifest and ``ATIFTraceHandle`` validates it on read. OTLP is
    one we wrote ourselves, so an empty file means the exporter never flushed and is not a trace.
    """
    admitted = [(atif, EVIDENCE_FORMAT_ATIF)] if atif is not None else []
    if otlp is not None and otlp.is_file() and otlp.stat().st_size > 0:
        admitted.append((otlp, EVIDENCE_FORMAT_OTLP))
    for path, trace_format in admitted:
        descriptor = EvidenceDescriptor(kind=EVIDENCE_TRACE, format=trace_format, ref=str(path))
        descriptors[EVIDENCE_TRACE] = descriptor
        descriptors[f"{EVIDENCE_TRACE}:{trace_format}"] = descriptor
