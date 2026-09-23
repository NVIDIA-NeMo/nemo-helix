# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import MLflow traces and assessments into NeMo Intake."""

from __future__ import annotations

import argparse
import base64
import binascii
import importlib
import importlib.util
import os
from typing import Any

from _import_common import (
    ImportBundle,
    add_common_arguments,
    add_provider_raw,
    add_provider_signal_raw,
    load_json,
    metadata_annotation,
    nanoseconds_to_datetime,
    normalize_kind,
    normalize_status,
    parse_json_value,
    partition_record,
    project_signal,
    run_import,
    set_if,
    to_jsonable,
    validate_common_arguments,
)

SOURCE = "mlflow"
INFO_FIELDS = {"trace_id", "trace_metadata", "assessments"}
SPAN_FIELDS = {
    "trace_id",
    "span_id",
    "parent_span_id",
    "name",
    "start_time_unix_nano",
    "end_time_unix_nano",
    "status",
    "attributes",
}
ASSESSMENT_FIELDS = {"assessment_name", "span_id", "feedback", "expectation"}


def map_mlflow_export(payload: Any, *, project: str | None, include_feedback: bool) -> ImportBundle:
    traces = payload.get("traces", []) if isinstance(payload, dict) else payload
    if not isinstance(traces, list):
        raise ValueError("MLflow export must be a trace array or an object with `traces`")
    bundle = ImportBundle(source=SOURCE)

    for trace_index, trace_value in enumerate(traces):
        spans_by_id: dict[str, dict[str, Any]] = {}
        mapped_trace_id = ""
        trace = _object(trace_value, "MLflow trace")
        info = _object(trace.get("info", {}), "MLflow trace info")
        data = _object(trace.get("data", {}), "MLflow trace data")
        trace_id = _mlflow_trace_id(info.get("trace_id"))
        trace_metadata = _object(info.get("trace_metadata") or {}, "MLflow trace metadata")
        session_id = str(
            trace_metadata.get("mlflow.trace.session")
            or trace_metadata.get("mlflow.trace.session_id")
            or trace_metadata.get("session_id")
            or trace_id
        )
        info_raw, info_coverage = partition_record(
            info,
            record_name=f"trace[{trace_index}].info",
            mapped_fields=INFO_FIELDS,
        )
        bundle.coverage.append(info_coverage)
        data_raw, data_coverage = partition_record(
            data,
            record_name=f"trace[{trace_index}].data",
            mapped_fields={"spans"},
        )
        bundle.coverage.append(data_coverage)

        for span_index, span_value in enumerate(data.get("spans", [])):
            native = _object(span_value, "MLflow span")
            serialized_attributes = _object(native.get("attributes", {}), "MLflow span attributes")
            attributes = {key: parse_json_value(value) for key, value in serialized_attributes.items()}
            status = native.get("status") or {}
            status_value = status.get("code") if isinstance(status, dict) else status
            external_span_id = _mlflow_span_id(native.get("span_id"))
            external_trace_id = trace_id or _mlflow_trace_id(native.get("trace_id"))
            if not external_span_id or not external_trace_id:
                raise ValueError("MLflow spans require trace_id and span_id")
            if native.get("start_time_unix_nano") is None:
                raise ValueError("MLflow spans require start_time_unix_nano")
            if mapped_trace_id and mapped_trace_id != external_trace_id:
                raise ValueError("MLflow trace contains spans with different trace IDs")
            mapped_trace_id = external_trace_id
            normalized_attributes = dict(attributes)
            span_type = normalized_attributes.pop("mlflow.spanType", None)
            span_input = parse_json_value(normalized_attributes.pop("mlflow.spanInputs", None))
            span_output = parse_json_value(normalized_attributes.pop("mlflow.spanOutputs", None))
            set_if(normalized_attributes, "gen_ai.request.model", attributes.get("mlflow.llm.model"))
            set_if(normalized_attributes, "gen_ai.system", attributes.get("mlflow.llm.provider"))
            set_if(normalized_attributes, "gen_ai.project", project)
            span = {
                "span_id": external_span_id,
                "trace_id": external_trace_id,
                "session_id": session_id or external_trace_id,
                "parent_span_id": _mlflow_span_id(native.get("parent_span_id")) or None,
                "name": str(native.get("name") or ""),
                "kind": normalize_kind(span_type),
                "status": normalize_status(status_value),
                "started_at": nanoseconds_to_datetime(native["start_time_unix_nano"]),
                "ended_at": (
                    nanoseconds_to_datetime(native["end_time_unix_nano"])
                    if native.get("end_time_unix_nano") is not None
                    else None
                ),
                "input": span_input,
                "output": span_output,
                "attributes": normalized_attributes,
            }
            span_raw, span_coverage = partition_record(
                native,
                record_name=f"trace[{trace_index}].span[{span_index}]",
                mapped_fields=SPAN_FIELDS,
            )
            bundle.coverage.append(span_coverage)
            raw = {**info_raw, **data_raw, **span_raw}
            raw["native_ids"] = {
                "trace_id": native.get("trace_id"),
                "span_id": native.get("span_id"),
                "parent_span_id": native.get("parent_span_id"),
            }
            raw["status"] = to_jsonable(status)
            if trace_metadata:
                raw["trace_metadata"] = trace_metadata
            add_provider_raw(span, SOURCE, raw)
            bundle.spans.append(span)
            spans_by_id[external_span_id] = span

        if include_feedback:
            _map_assessments(
                bundle,
                info.get("assessments") or [],
                spans_by_id=spans_by_id,
                trace_id=mapped_trace_id,
                session_id=session_id or mapped_trace_id,
                record_prefix=f"trace[{trace_index}]",
            )
    if isinstance(payload, dict):
        export_raw, coverage = partition_record(
            payload,
            record_name="export",
            mapped_fields={"traces"},
            ignored_fields={"cursor", "next_cursor", "next_page_token", "count"},
        )
        bundle.coverage.append(coverage)
        if export_raw and bundle.spans:
            add_provider_raw(bundle.spans[0], SOURCE, {"export": export_raw})
    bundle.validate()
    return bundle


def _map_assessments(
    bundle: ImportBundle,
    values: list[Any],
    *,
    spans_by_id: dict[str, dict[str, Any]],
    trace_id: str,
    session_id: str,
    record_prefix: str,
) -> None:
    trace_spans = [span for span in bundle.spans if span["trace_id"] == trace_id]
    root = next((span for span in trace_spans if not span.get("parent_span_id")), None)
    for index, value in enumerate(values):
        assessment = _object(value, "MLflow assessment")
        raw, coverage = partition_record(
            assessment,
            record_name=f"{record_prefix}.assessment[{index}]",
            mapped_fields=ASSESSMENT_FIELDS,
        )
        bundle.coverage.append(coverage)
        name = str(assessment.get("assessment_name") or "assessment")
        target = spans_by_id.get(_mlflow_span_id(assessment.get("span_id"))) or root
        if target is None:
            raise ValueError(f"MLflow assessment {name!r} does not target an exported span")
        source = _object(assessment.get("source") or {}, "MLflow assessment source")
        source_type = str(source.get("source_type") or source.get("type") or "HUMAN").upper()
        automated = source_type in {"LLM_JUDGE", "AI_JUDGE", "CODE"}
        feedback = assessment.get("feedback")
        expectation = assessment.get("expectation")
        signal_value: Any = None
        comment: str | None = assessment.get("rationale")
        if isinstance(feedback, dict):
            signal_value = feedback.get("value")
            comment = feedback.get("rationale") or comment
        elif feedback is not None:
            signal_value = feedback
        evaluations, annotations = project_signal(
            provider=SOURCE,
            span_id=str(target["span_id"]),
            session_id=session_id,
            name=name,
            value=signal_value,
            comment=comment,
            automated=automated,
        )
        bundle.evaluator_results.extend(evaluations)
        bundle.annotations.extend(annotations)
        if expectation is not None:
            bundle.annotations.append(
                metadata_annotation(
                    span_id=str(target["span_id"]),
                    session_id=session_id,
                    metadata={f"{SOURCE}.expectation.{name}": to_jsonable(expectation)},
                )
            )
        signal_raw = {**raw, **assessment}
        add_provider_signal_raw(target, SOURCE, {key: item for key, item in signal_raw.items() if item is not None})


def fetch_mlflow(args: argparse.Namespace) -> dict[str, Any]:
    if importlib.util.find_spec("mlflow") is None:
        raise RuntimeError("MLflow live import requires the `mlflow` package")
    mlflow = importlib.import_module("mlflow")
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    lower_ms = int(args.since.timestamp() * 1000)
    upper_ms = int(args.until.timestamp() * 1000)
    traces = mlflow.search_traces(
        experiment_ids=[args.project],
        filter_string=f"timestamp_ms >= {lower_ms} AND timestamp_ms < {upper_ms}",
        return_type="list",
    )
    return {"traces": [to_jsonable(trace) for trace in traces]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_common_arguments(parser, args)
    payload = load_json(args.input) if args.input else fetch_mlflow(args)
    bundle = map_mlflow_export(payload, project=args.project, include_feedback=args.include_feedback)
    return run_import(bundle, args)


def _object(value: Any, label: str) -> dict[str, Any]:
    converted = to_jsonable(value)
    if not isinstance(converted, dict):
        raise ValueError(f"{label} must be an object")
    return converted


def _mlflow_span_id(value: Any) -> str:
    return _mlflow_id(value, byte_length=8)


def _mlflow_trace_id(value: Any) -> str:
    return _mlflow_id(value, byte_length=16)


def _mlflow_id(value: Any, *, byte_length: int) -> str:
    text = str(value or "")
    try:
        decoded = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        return text
    return decoded.hex() if len(decoded) == byte_length else text


if __name__ == "__main__":
    raise SystemExit(main())
