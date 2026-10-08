# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal OTLP JSONL trace writer — standard library only.

Harbor collects `/app/traces` into the trial's `artifacts/traces/` directory, and
the Experimentalist's `TrialResult.trace` points at the first `*.jsonl` it finds
there. The Analyzer's TraceExplorer reads that file, so the shape below is the
contract: one `ExportTraceServiceRequest` JSON object per line, spans under
`resourceSpans[].scopeSpans[].spans[]`, OpenInference semantics in attributes.

Real agents get this for free from an OpenTelemetry SDK exporter (see
`examples/terminal-bench-agent/tracing.py`). This one is hand-rolled so the task
container needs no dependencies beyond the Python base image.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

TRACES_DIR = Path(os.environ.get("TRACE_DIR", "/app/traces"))


def _attrs(values: dict[str, str]) -> list[dict]:
    return [{"key": key, "value": {"stringValue": value}} for key, value in values.items()]


def write_trace(session_id: str, instruction: str, answer: str, handler: str) -> Path:
    """Write a two-span trace (agent chain + its handler) as OTLP JSONL.

    Args:
        session_id: Stable id for this run; becomes the trace's session.
        instruction: The task instruction the agent received.
        answer: The line the agent produced.
        handler: Name of the code path that produced the answer.

    Returns:
        Path of the written JSONL file.
    """
    now = time.time_ns()
    trace_id = f"{now:032x}"[-32:]
    root_span_id, child_span_id = f"{now:016x}"[-16:], f"{now + 1:016x}"[-16:]

    document = {
        "resourceSpans": [
            {
                "resource": {"attributes": _attrs({"service.name": "hello-harbor-agent", "session.id": session_id})},
                "scopeSpans": [
                    {
                        "scope": {"name": "hello-harbor-agent"},
                        "spans": [
                            {
                                "traceId": trace_id,
                                "spanId": root_span_id,
                                "name": "HelloAgent.solve",
                                "kind": "SPAN_KIND_INTERNAL",
                                "startTimeUnixNano": str(now),
                                "endTimeUnixNano": str(now + 1_000_000),
                                "attributes": _attrs(
                                    {
                                        "openinference.span.kind": "CHAIN",
                                        "input.value": instruction,
                                        "output.value": answer,
                                    }
                                ),
                                "status": {"code": "STATUS_CODE_OK"},
                            },
                            {
                                "traceId": trace_id,
                                "spanId": child_span_id,
                                "parentSpanId": root_span_id,
                                "name": handler,
                                "kind": "SPAN_KIND_INTERNAL",
                                "startTimeUnixNano": str(now),
                                "endTimeUnixNano": str(now + 500_000),
                                "attributes": _attrs(
                                    {
                                        "openinference.span.kind": "TOOL",
                                        "tool.name": handler,
                                        "input.value": instruction,
                                        "output.value": answer,
                                    }
                                ),
                                "status": {"code": "STATUS_CODE_OK"},
                            },
                        ],
                    }
                ],
            }
        ]
    }

    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACES_DIR / "agent.jsonl"
    path.write_text(json.dumps(document, separators=(",", ":")) + "\n", encoding="utf-8")
    return path
