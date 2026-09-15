# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn the exports an :mod:`otlp_receiver` stored into the trace evidence a metric reads.

Split from the receiver because these need protobuf and the receiver must not: it is seeded into a
sandbox image that carries Fabric and nothing of this package's own dependencies.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

from google.protobuf.json_format import MessageToJson
from google.protobuf.message import DecodeError
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.otlp_receiver import EXPORT_SUFFIX
from nemo_evaluator_sdk.values.evidence import (
    EVIDENCE_FORMAT_ATIF,
    EVIDENCE_FORMAT_OTLP,
    EVIDENCE_TRACE,
    EvidenceDescriptor,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

logger = logging.getLogger(__name__)

#: Where a trial's trace lands, relative to its evidence directory. ``traces`` because that is
#: where Harbor puts its own, so the layout reads the same across runners.
TRACES_SUBDIR = "traces"
OTLP_FILENAME = "relay.jsonl"


def traces_dir(evidence_dir: Path) -> Path:
    return evidence_dir / TRACES_SUBDIR


def otlp_trace_path(evidence_dir: Path) -> Path:
    return traces_dir(evidence_dir) / OTLP_FILENAME


def otlp_endpoint_fields(*, endpoint: str, service_name: str) -> dict[str, Any]:
    """Relay OpenTelemetry endpoint settings, as plain values.

    Returned unmaterialized because the two runtimes build the same endpoint out of different
    classes -- ``nemo_fabric``'s ``Relay*`` models for a typed config object, ``nemo_relay``'s own
    for a config dict -- and only the values are shared.
    """
    return {
        # `full` rather than `gen_ai`/`openinference`: a projection chooses what to drop, and the
        # captured file is the trial's whole trace rather than one view of it.
        "type": "full",
        "endpoint": endpoint,
        "transport": "http_binary",
        "service_name": service_name,
    }


def export_to_json_line(body: bytes) -> str | None:
    """One OTLP/JSON line for a serialized export, or None when it carries no spans.

    Raises:
        google.protobuf.message.DecodeError: ``body`` is not an export request.
    """
    request = ExportTraceServiceRequest()
    request.ParseFromString(body)
    if not request.resource_spans:
        return None
    return MessageToJson(request, indent=None)


def fold_exports(directory: Path) -> int:
    """Fold the receiver's stored exports into one OTLP/JSON file, returning how many were folded.

    Streamed through a temporary file and moved into place, so a trace larger than memory still
    folds. Additive: a fold consumes its inputs, so repeating it changes nothing, and a fold that
    finds new exports appends to the trace already written rather than replacing it.
    """
    exports = sorted(directory.glob(f"*{EXPORT_SUFFIX}"))
    if not exports:
        return 0
    trace = directory / OTLP_FILENAME
    partial = directory / f"{OTLP_FILENAME}.partial"
    folded = 0
    try:
        with partial.open("w", encoding="utf-8") as handle:
            # Carried forward rather than overwritten: the rewrite is atomic, so an earlier fold's
            # spans have to be copied in or `os.replace` drops them.
            if trace.exists():
                with trace.open("r", encoding="utf-8") as previous:
                    shutil.copyfileobj(previous, handle)
            for export in exports:
                try:
                    line = export_to_json_line(export.read_bytes())
                except (OSError, DecodeError):
                    logger.warning("Discarding an unreadable OTLP export at %s.", export)
                    continue
                if line is None:
                    continue
                handle.write(f"{line}\n")
                folded += 1
    except BaseException:
        # Whatever went wrong, a half-written trace must not be left where a complete one goes.
        partial.unlink(missing_ok=True)
        raise
    if not folded:
        partial.unlink(missing_ok=True)
        _drop(exports)
        return 0
    os.replace(partial, trace)
    _drop(exports)
    return folded


def _drop(exports: list[Path]) -> None:
    """Consume the exports a fold has read, best-effort.

    Every export that got this far was either folded into the trace -- a lossless re-encoding, so
    keeping both would double a trace's footprint -- or read and discarded as unusable. Leaving
    either behind means the next fold re-reads it. Only a fold that *raises* keeps its inputs, so
    there is something to retry from.

    Cleanup never raises: it runs after the trace is committed, so failing here would cost the
    caller a trace that is already safely on disk. It is loud instead, because a surviving input is
    one a later fold would append a second time.
    """
    for export in exports:
        try:
            export.unlink(missing_ok=True)
        except OSError as error:
            logger.warning(
                "Could not consume the OTLP export at %s (%s); folding this directory again would duplicate its spans.",
                export,
                error,
            )


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
