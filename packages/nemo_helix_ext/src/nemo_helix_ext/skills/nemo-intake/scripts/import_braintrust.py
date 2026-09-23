# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import Braintrust project-log spans and inline feedback into NeMo Intake."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
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
    normalize_datetime,
    normalize_kind,
    normalize_status,
    partition_record,
    project_signal,
    run_import,
    set_if,
    to_jsonable,
    validate_common_arguments,
    validated_service_url,
)

SOURCE = "braintrust"
MAX_PAGES = 1000
EVENT_FIELDS = {
    "id",
    "span_id",
    "root_span_id",
    "span_parents",
    "span_attributes",
    "created",
    "input",
    "output",
    "error",
    "metrics",
    "metadata",
    "scores",
    "expected",
    "comments",
    "classifications",
    "name",
}


def map_braintrust_export(payload: Any, *, project: str | None, include_feedback: bool) -> ImportBundle:
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise ValueError("Braintrust export must contain an `events` array")
    bundle = ImportBundle(source=SOURCE)
    seen_ids: set[str] = set()
    for index, value in enumerate(payload["events"]):
        event = _object(value, "Braintrust project-log event")
        event_id = str(event.get("id") or "")
        if not event_id:
            raise ValueError("Braintrust events require `id`")
        if event_id in seen_ids:
            continue
        seen_ids.add(event_id)
        span_id = str(event.get("span_id") or event_id)
        trace_id = str(event.get("root_span_id") or span_id)
        parents = event.get("span_parents") or []
        parent_span_id = str(parents[-1]) if isinstance(parents, list) and parents else None
        span_attributes = _object(event.get("span_attributes") or {}, "Braintrust span attributes")
        metadata = _object(event.get("metadata") or {}, "Braintrust metadata")
        metrics = _object(event.get("metrics") or {}, "Braintrust metrics")
        session_id = str(metadata.get("session_id") or metadata.get("thread_id") or trace_id)
        started_at = metrics.get("start") if metrics.get("start") is not None else event.get("created")
        ended_at = metrics.get("end")
        if ended_at is None and metrics.get("duration") is not None:
            ended_at = _epoch_seconds(started_at) + float(metrics["duration"])
        attributes: dict[str, Any] = {}
        set_if(attributes, "gen_ai.project", project)
        set_if(attributes, "gen_ai.request.model", metadata.get("model") or span_attributes.get("model"))
        set_if(attributes, "gen_ai.system", metadata.get("provider") or span_attributes.get("provider"))
        set_if(attributes, "gen_ai.usage.input_tokens", metrics.get("prompt_tokens"))
        set_if(attributes, "gen_ai.usage.output_tokens", metrics.get("completion_tokens"))
        set_if(attributes, "gen_ai.usage.total_tokens", metrics.get("tokens"))
        set_if(attributes, "gen_ai.usage.cost", metrics.get("cost"))
        if event.get("error"):
            set_if(attributes, "exception.message", event["error"])
        span = {
            "span_id": span_id,
            "trace_id": trace_id,
            "session_id": session_id,
            "parent_span_id": parent_span_id,
            "name": str(span_attributes.get("name") or event.get("name") or ""),
            "kind": normalize_kind(span_attributes.get("type")),
            "status": normalize_status("success" if ended_at is not None else None, error=event.get("error")),
            "started_at": normalize_datetime(started_at),
            "ended_at": normalize_datetime(ended_at) if ended_at is not None else None,
            "input": to_jsonable(event.get("input")),
            "output": to_jsonable(event.get("output")),
            "attributes": attributes,
        }
        raw, coverage = partition_record(
            event,
            record_name=f"event[{index}]",
            mapped_fields=EVENT_FIELDS,
        )
        bundle.coverage.append(coverage)
        if span_attributes:
            raw["span_attributes"] = span_attributes
        if metadata:
            raw["metadata"] = metadata
        if metrics:
            raw["metrics"] = metrics
        raw["native_identity"] = {
            "id": event.get("id"),
            "span_id": event.get("span_id"),
            "root_span_id": event.get("root_span_id"),
            "span_parents": to_jsonable(event.get("span_parents")),
            "created": event.get("created"),
        }
        for signal_field in ("scores", "expected", "comments", "classifications"):
            if event.get(signal_field) is not None:
                raw[signal_field] = to_jsonable(event[signal_field])
        add_provider_raw(span, SOURCE, raw)
        bundle.spans.append(span)

        if include_feedback:
            _map_feedback(bundle, event, span)
    export_raw, coverage = partition_record(
        payload,
        record_name="export",
        mapped_fields={"events"},
        ignored_fields={"cursor", "next_cursor", "count"},
    )
    bundle.coverage.append(coverage)
    if export_raw and bundle.spans:
        add_provider_raw(bundle.spans[0], SOURCE, {"export": export_raw})
    bundle.validate()
    return bundle


def _map_feedback(bundle: ImportBundle, event: dict[str, Any], span: dict[str, Any]) -> None:
    for name, score in _object(event.get("scores") or {}, "Braintrust scores").items():
        evaluations, annotations = project_signal(
            provider=SOURCE,
            span_id=str(span["span_id"]),
            session_id=str(span["session_id"]),
            name=name,
            value=score,
            comment=None,
            automated=True,
        )
        bundle.evaluator_results.extend(evaluations)
        bundle.annotations.extend(annotations)
        add_provider_signal_raw(span, SOURCE, {"type": "score", "name": name, "value": score})
    if event.get("expected") is not None:
        bundle.annotations.append(
            metadata_annotation(
                span_id=str(span["span_id"]),
                session_id=str(span["session_id"]),
                metadata={"braintrust.expected": to_jsonable(event["expected"])},
            )
        )
    for comment in _comments(event.get("comments")):
        _, annotations = project_signal(
            provider=SOURCE,
            span_id=str(span["span_id"]),
            session_id=str(span["session_id"]),
            name="comment",
            value=None,
            comment=comment,
            automated=False,
        )
        bundle.annotations.extend(annotations)
    classifications = event.get("classifications")
    if isinstance(classifications, list):
        for classification in classifications:
            if isinstance(classification, str):
                bundle.annotations.append(
                    {
                        "span_id": str(span["span_id"]),
                        "session_id": str(span["session_id"]),
                        "kind": "label",
                        "value_type": "text",
                        "value": classification,
                    }
                )
            else:
                bundle.annotations.append(
                    metadata_annotation(
                        span_id=str(span["span_id"]),
                        session_id=str(span["session_id"]),
                        metadata={"braintrust.classification": to_jsonable(classification)},
                    )
                )
    elif classifications is not None:
        bundle.annotations.append(
            metadata_annotation(
                span_id=str(span["span_id"]),
                session_id=str(span["session_id"]),
                metadata={"braintrust.classifications": to_jsonable(classifications)},
            )
        )


def fetch_braintrust(args: argparse.Namespace) -> dict[str, Any]:
    base_url = validated_service_url(args.braintrust_base_url, label="Braintrust base URL")
    api_key = os.environ.get("BRAINTRUST_API_KEY")
    if not api_key:
        raise RuntimeError("Braintrust live import requires BRAINTRUST_API_KEY")
    url = f"{base_url}/v1/project_logs/{quote(args.project, safe='')}/fetch"
    events: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_cursors: set[str] = set()
    cursor: str | None = None
    for _ in range(MAX_PAGES):
        params: dict[str, str | int] = {"limit": 1000}
        if cursor:
            params["cursor"] = cursor
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            params=params,
            timeout=60,
            allow_redirects=False,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Braintrust GET returned {response.status_code}: {response.text[:2000]}")
        payload = _object(response.json(), "Braintrust response")
        page_value = payload.get("events", [])
        if not isinstance(page_value, list):
            raise ValueError("Braintrust response `events` must be an array")
        page = [_object(item, "Braintrust event") for item in page_value]
        if not page:
            break
        for event in page:
            event_id = str(event.get("id") or "")
            if not event_id:
                raise ValueError("Braintrust events require `id`")
            if event_id in seen_ids:
                continue
            seen_ids.add(event_id)
            metrics_start = _object(event.get("metrics") or {}, "metrics").get("start")
            created_at = _as_datetime(event.get("created") if event.get("created") is not None else metrics_start)
            if args.since <= created_at < args.until:
                events.append(event)
        next_cursor = payload.get("cursor")
        if not next_cursor:
            break
        next_cursor = str(next_cursor)
        if next_cursor in seen_cursors:
            raise RuntimeError("Braintrust pagination returned a repeated cursor")
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    else:
        raise RuntimeError(f"Braintrust fetch exceeded {MAX_PAGES} pages")
    return {"events": events}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        "--braintrust-base-url",
        default=os.environ.get("BRAINTRUST_API_URL", "https://api.braintrust.dev"),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_common_arguments(parser, args)
    payload = load_json(args.input) if args.input else fetch_braintrust(args)
    bundle = map_braintrust_export(payload, project=args.project, include_feedback=args.include_feedback)
    return run_import(bundle, args)


def _object(value: Any, label: str) -> dict[str, Any]:
    converted = to_jsonable(value)
    if not isinstance(converted, dict):
        raise ValueError(f"{label} must be an object")
    return converted


def _epoch_seconds(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    return _as_datetime(value).timestamp()


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    raise ValueError(f"Unsupported Braintrust timestamp: {value!r}")


def _comments(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    comments: list[str] = []
    for item in value:
        if isinstance(item, str):
            comments.append(item)
        elif isinstance(item, dict):
            text = item.get("text") or item.get("comment")
            if text:
                comments.append(str(text))
    return comments


if __name__ == "__main__":
    raise SystemExit(main())
