# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import Arize Phoenix spans and annotations into NeMo Intake."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
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
    validated_service_url,
)

SOURCE = "phoenix"
MAX_PAGES = 1000
SPAN_FIELDS = {
    "traceId",
    "trace_id",
    "spanId",
    "span_id",
    "parentSpanId",
    "parent_span_id",
    "name",
    "kind",
    "startTimeUnixNano",
    "start_time_unix_nano",
    "endTimeUnixNano",
    "end_time_unix_nano",
    "status",
    "attributes",
    "resource",
}
ANNOTATION_FIELDS = {"span_id", "name", "annotator_kind", "result", "metadata"}


def map_phoenix_export(payload: Any, *, project: str | None, include_feedback: bool) -> ImportBundle:
    if not isinstance(payload, dict):
        raise ValueError("Phoenix export must be an object")
    native_spans = payload.get("spans", payload.get("data"))
    if not isinstance(native_spans, list):
        raise ValueError("Phoenix export must contain a `spans` or `data` array")
    bundle = ImportBundle(source=SOURCE)
    spans_by_id: dict[str, dict[str, Any]] = {}

    for index, value in enumerate(native_spans):
        native = _object(value, "Phoenix OTLP span")
        span_id = str(_alias(native, "spanId", "span_id") or "")
        trace_id = str(_alias(native, "traceId", "trace_id") or "")
        if not span_id or not trace_id:
            raise ValueError("Phoenix spans require traceId and spanId")
        span_attributes = _attributes(native.get("attributes"))
        resource = _object(native.get("resource") or {}, "Phoenix resource")
        resource_attributes = _attributes(resource.get("attributes"))
        attributes = {**resource_attributes, **span_attributes}
        set_if(attributes, "gen_ai.project", project)
        session_id = str(
            attributes.get("session.id")
            or attributes.get("session_id")
            or attributes.get("openinference.session.id")
            or trace_id
        )
        status = native.get("status") or {}
        status_code = status.get("code") if isinstance(status, dict) else status
        start_ns = _alias(native, "startTimeUnixNano", "start_time_unix_nano")
        end_ns = _alias(native, "endTimeUnixNano", "end_time_unix_nano")
        if start_ns is None:
            raise ValueError("Phoenix spans require startTimeUnixNano")
        span = {
            "span_id": span_id,
            "trace_id": trace_id,
            "session_id": session_id,
            "parent_span_id": _alias(native, "parentSpanId", "parent_span_id") or None,
            "name": str(native.get("name") or ""),
            "kind": normalize_kind(attributes.get("openinference.span.kind") or native.get("kind")),
            "status": normalize_status(status_code),
            "started_at": nanoseconds_to_datetime(start_ns),
            "ended_at": nanoseconds_to_datetime(end_ns) if end_ns is not None else None,
            "input": parse_json_value(attributes.get("input.value")),
            "output": parse_json_value(attributes.get("output.value")),
            "attributes": attributes,
        }
        raw, coverage = partition_record(
            native,
            record_name=f"span[{index}]",
            mapped_fields=SPAN_FIELDS,
        )
        bundle.coverage.append(coverage)
        if resource:
            raw["resource"] = resource
        raw["native_classification"] = {"kind": native.get("kind"), "status": to_jsonable(status)}
        add_provider_raw(span, SOURCE, raw)
        bundle.spans.append(span)
        spans_by_id[span_id] = span

    if include_feedback:
        for index, value in enumerate(payload.get("annotations", [])):
            annotation = _object(value, "Phoenix annotation")
            raw, coverage = partition_record(
                annotation,
                record_name=f"annotation[{index}]",
                mapped_fields=ANNOTATION_FIELDS,
            )
            bundle.coverage.append(coverage)
            target = spans_by_id.get(str(annotation.get("span_id") or ""))
            if target is None:
                raise ValueError(f"Phoenix annotation {annotation.get('id')!r} does not target an exported span")
            result = _object(annotation.get("result") or {}, "Phoenix annotation result")
            value_to_project = result.get("score")
            if value_to_project is None:
                value_to_project = result.get("label")
            automated = str(annotation.get("annotator_kind") or "HUMAN").upper() in {
                "LLM",
                "CODE",
            }
            evaluations, annotations = project_signal(
                provider=SOURCE,
                span_id=str(target["span_id"]),
                session_id=str(target["session_id"]),
                name=str(annotation.get("name") or "annotation"),
                value=value_to_project,
                comment=result.get("explanation"),
                automated=automated,
            )
            bundle.evaluator_results.extend(evaluations)
            bundle.annotations.extend(annotations)
            if not automated and result.get("score") is not None and result.get("label") is not None:
                bundle.annotations.append(
                    {
                        "span_id": str(target["span_id"]),
                        "session_id": str(target["session_id"]),
                        "kind": "label",
                        "name": str(annotation.get("name") or "annotation")[:256],
                        "value_type": "text",
                        "value": str(result["label"]),
                    }
                )
            if annotation.get("metadata"):
                bundle.annotations.append(
                    metadata_annotation(
                        span_id=str(target["span_id"]),
                        session_id=str(target["session_id"]),
                        metadata={"phoenix.annotation": to_jsonable(annotation["metadata"])},
                    )
                )
            add_provider_signal_raw(target, SOURCE, {**raw, **annotation})
    export_raw, coverage = partition_record(
        payload,
        record_name="export",
        mapped_fields={"spans", "data", "annotations"},
        ignored_fields={"cursor", "next_cursor", "count"},
    )
    bundle.coverage.append(coverage)
    if export_raw and bundle.spans:
        add_provider_raw(bundle.spans[0], SOURCE, {"export": export_raw})
    bundle.validate()
    return bundle


def fetch_phoenix(args: argparse.Namespace) -> dict[str, Any]:
    base_url = validated_service_url(args.phoenix_base_url, label="Phoenix base URL")
    headers: dict[str, str] = {}
    if key := os.environ.get("PHOENIX_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    project = quote(args.project, safe="")
    spans = _fetch_pages(
        f"{base_url}/v1/projects/{project}/spans/otlpv1",
        headers=headers,
        params={"start_time": args.since.isoformat(), "end_time": args.until.isoformat(), "limit": 1000},
    )
    annotations: list[dict[str, Any]] = []
    if args.include_feedback:
        span_ids = [str(_alias(span, "spanId", "span_id")) for span in spans]
        for start in range(0, len(span_ids), 100):
            params: list[tuple[str, str | int]] = [("limit", 10000)]
            params.extend(("span_ids", span_id) for span_id in span_ids[start : start + 100])
            annotations.extend(
                _fetch_pages(
                    f"{base_url}/v1/projects/{project}/span_annotations",
                    headers=headers,
                    params=params,
                )
            )
    return {"spans": spans, "annotations": annotations}


def _fetch_pages(
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, str | int] | list[tuple[str, str | int]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    for _ in range(MAX_PAGES):
        page_params = list(params.items()) if isinstance(params, dict) else list(params)
        if cursor:
            page_params.append(("cursor", cursor))
        response = requests.get(url, headers=headers, params=page_params, timeout=60, allow_redirects=False)
        if response.status_code != 200:
            raise RuntimeError(f"Phoenix GET returned {response.status_code}: {response.text[:2000]}")
        payload = _object(response.json(), "Phoenix response")
        data = payload.get("data", [])
        if not isinstance(data, list):
            raise ValueError("Phoenix response `data` must be an array")
        results.extend(_object(item, "Phoenix response item") for item in data)
        next_cursor = payload.get("next_cursor")
        if not next_cursor:
            return results
        next_cursor = str(next_cursor)
        if next_cursor in seen_cursors:
            raise RuntimeError("Phoenix pagination returned a repeated cursor")
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    raise RuntimeError(f"Phoenix pagination exceeded {MAX_PAGES} pages")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        "--phoenix-base-url",
        default=os.environ.get("PHOENIX_BASE_URL", "http://127.0.0.1:6006"),
    )
    parser.add_argument("--annotations-input", type=Path, help="Optional JSON file containing `annotations`.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_common_arguments(parser, args)
    payload = _object(load_json(args.input) if args.input else fetch_phoenix(args), "Phoenix export")
    if args.annotations_input:
        annotation_payload = load_json(args.annotations_input)
        annotations = (
            annotation_payload.get("annotations", annotation_payload)
            if isinstance(annotation_payload, dict)
            else annotation_payload
        )
        if not isinstance(annotations, list):
            raise ValueError("Phoenix annotations input must be an array or an object containing `annotations`")
        payload["annotations"] = annotations
    bundle = map_phoenix_export(payload, project=args.project, include_feedback=args.include_feedback)
    return run_import(bundle, args)


def _object(value: Any, label: str) -> dict[str, Any]:
    converted = to_jsonable(value)
    if not isinstance(converted, dict):
        raise ValueError(f"{label} must be an object")
    return converted


def _alias(value: dict[str, Any], camel: str, snake: str) -> Any:
    return value[camel] if camel in value else value.get(snake)


def _attributes(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): _any_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return {
            str(item["key"]): _any_value(item.get("value"))
            for item in value
            if isinstance(item, dict) and "key" in item
        }
    raise ValueError("OTLP attributes must be an object or key/value array")


def _any_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    aliases = (
        ("stringValue", "string_value"),
        ("boolValue", "bool_value"),
        ("intValue", "int_value"),
        ("doubleValue", "double_value"),
        ("bytesValue", "bytes_value"),
    )
    for camel, snake in aliases:
        if camel in value:
            return value[camel]
        if snake in value:
            return value[snake]
    array = value.get("arrayValue", value.get("array_value"))
    if isinstance(array, dict):
        return [_any_value(item) for item in array.get("values", [])]
    mapping = value.get("kvlistValue", value.get("kvlist_value"))
    if isinstance(mapping, dict):
        return _attributes(mapping.get("values", []))
    return to_jsonable(value)


if __name__ == "__main__":
    raise SystemExit(main())
