# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OTLP trace ingress for agents in leased sandboxes.

    POST /v2/workspaces/{workspace}/otlp/v1/traces

An agent points its OTLP/HTTP exporter at ``.../v2/workspaces/{workspace}/otlp``. The gateway accepts
spans only from an address a call lease holds in that workspace, moves them into the lease's trial, and
sends them to Intake with its own spans.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request, Response, status
from google.protobuf.message import DecodeError
from nhx.core.inference_gateway.api.authz import MODEL_EXEC_PERMISSION, enforce_delegated_workspace_access
from nhx.core.inference_gateway.api.call_recording import global_call_recorder
from nhx.core.inference_gateway.api.lease_cache import global_lease_cache
from nhx.core.inference_gateway.api.trial_spans import adopt_agent_spans
from nhx.core.inference_gateway.api.validation import validate_entity_name
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

PROTOBUF_MEDIA_TYPE = "application/x-protobuf"
# Intake refuses larger OTLP bodies, so one this size could never be forwarded whole.
MAX_EXPORT_BYTES = 5 * 1024 * 1024


@router.post(
    "/v2/workspaces/{workspace}/otlp/v1/traces",
    summary="Ingest Agent Traces",
    response_description="OTLP export response",
    operation_id="ingest_agent_traces",
    status_code=status.HTTP_200_OK,
    response_class=Response,
    responses={
        403: {"description": "No call lease holds the caller's address in this workspace."},
        413: {"description": "The export is larger than 5 MiB."},
        415: {"description": "The body is not OTLP protobuf."},
    },
)
async def ingest_agent_traces(workspace: str, request: Request) -> Response:
    """Accept an OTLP/HTTP protobuf trace export from an agent in a leased sandbox."""
    validate_entity_name(workspace, field_name="workspace")
    await enforce_delegated_workspace_access(workspace, MODEL_EXEC_PERMISSION)
    if request.headers.get("content-type", "").split(";")[0].strip() != PROTOBUF_MEDIA_TYPE:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=f"Send OTLP as {PROTOBUF_MEDIA_TYPE}.")
    body = await request.body()
    if len(body) > MAX_EXPORT_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, detail="OTLP exports are limited to 5 MiB.")

    client_address = request.client.host if request.client else None
    lease = global_lease_cache().lookup(client_address, datetime.now(UTC))
    if lease is None or lease.workspace != workspace:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail=f"No call lease holds {client_address} in workspace '{workspace}'.",
        )
    try:
        export = ExportTraceServiceRequest.FromString(body)
    except DecodeError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"Invalid OTLP protobuf: {exc}") from exc

    recorder = global_call_recorder()
    if recorder is not None:
        for spans in adopt_agent_spans(export, lease):
            recorder.record(workspace, spans)
    return Response(content=ExportTraceServiceResponse().SerializeToString(), media_type=PROTOBUF_MEDIA_TYPE)
