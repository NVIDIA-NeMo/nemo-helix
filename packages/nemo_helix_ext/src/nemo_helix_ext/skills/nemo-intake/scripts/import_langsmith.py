# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import LangSmith runs and feedback into NeMo Intake."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

SOURCE = "langsmith"
RUN_FIELDS = {
    "id",
    "trace_id",
    "parent_run_id",
    "name",
    "run_type",
    "start_time",
    "end_time",
    "inputs",
    "outputs",
    "error",
    "status",
    "extra",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "total_cost",
    "prompt_cost",
    "completion_cost",
}
FEEDBACK_FIELDS = {"run_id", "key", "score", "value", "comment", "correction", "feedback_source"}


def map_langsmith_export(payload: Any, *, project: str | None, include_feedback: bool) -> ImportBundle:
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
        raise ValueError("LangSmith export must contain a `runs` array")
    bundle = ImportBundle(source=SOURCE)
    spans_by_id: dict[str, dict[str, Any]] = {}

    for index, value in enumerate(payload["runs"]):
        run = _object(value, "LangSmith run")
        run_id = str(run.get("id") or "")
        trace_id = str(run.get("trace_id") or run_id)
        if not run_id:
            raise ValueError("LangSmith runs require `id`")
        if run.get("start_time") is None:
            raise ValueError("LangSmith runs require start_time")
        extra = _object(run.get("extra") or {}, "LangSmith run extra")
        metadata = _object(extra.get("metadata") or {}, "LangSmith run metadata")
        session_id = str(metadata.get("session_id") or metadata.get("thread_id") or run.get("session_id") or trace_id)
        attributes: dict[str, Any] = {}
        set_if(attributes, "gen_ai.project", project)
        set_if(
            attributes,
            "gen_ai.request.model",
            metadata.get("ls_model_name") or metadata.get("model_name") or _nested(extra, "invocation_params", "model"),
        )
        set_if(attributes, "gen_ai.system", metadata.get("ls_provider") or metadata.get("provider"))
        set_if(attributes, "gen_ai.usage.input_tokens", run.get("prompt_tokens"))
        set_if(attributes, "gen_ai.usage.output_tokens", run.get("completion_tokens"))
        set_if(attributes, "gen_ai.usage.total_tokens", run.get("total_tokens"))
        set_if(attributes, "gen_ai.usage.cost", run.get("total_cost"))
        set_if(attributes, "llm.cost.prompt", run.get("prompt_cost"))
        set_if(attributes, "llm.cost.completion", run.get("completion_cost"))
        if run.get("error"):
            set_if(attributes, "exception.message", run["error"])
        span = {
            "span_id": run_id,
            "trace_id": trace_id,
            "session_id": session_id,
            "parent_span_id": run.get("parent_run_id") or None,
            "name": str(run.get("name") or ""),
            "kind": normalize_kind(run.get("run_type")),
            "status": normalize_status(run.get("status"), error=run.get("error")),
            "started_at": normalize_datetime(run["start_time"]),
            "ended_at": normalize_datetime(run["end_time"]) if run.get("end_time") else None,
            "input": to_jsonable(run.get("inputs")),
            "output": to_jsonable(run.get("outputs")),
            "attributes": attributes,
        }
        raw, coverage = partition_record(
            run,
            record_name=f"run[{index}]",
            mapped_fields=RUN_FIELDS,
        )
        bundle.coverage.append(coverage)
        if extra:
            raw["extra"] = extra
        raw["native_classification"] = {"run_type": run.get("run_type"), "status": run.get("status")}
        add_provider_raw(span, SOURCE, raw)
        bundle.spans.append(span)
        spans_by_id[run_id] = span

    if include_feedback:
        for index, value in enumerate(payload.get("feedback", [])):
            feedback = _object(value, "LangSmith feedback")
            raw, coverage = partition_record(
                feedback,
                record_name=f"feedback[{index}]",
                mapped_fields=FEEDBACK_FIELDS,
            )
            bundle.coverage.append(coverage)
            target = spans_by_id.get(str(feedback.get("run_id") or ""))
            if target is None:
                raise ValueError(f"LangSmith feedback {feedback.get('id')!r} does not target an exported run")
            source = _object(feedback.get("feedback_source") or {}, "LangSmith feedback source")
            source_type = str(source.get("type") or "api").lower()
            automated = source_type in {"model", "evaluator", "automated", "code"}
            signal_value = feedback.get("score")
            if signal_value is None:
                signal_value = feedback.get("value")
            evaluations, annotations = project_signal(
                provider=SOURCE,
                span_id=str(target["span_id"]),
                session_id=str(target["session_id"]),
                name=str(feedback.get("key") or "feedback"),
                value=signal_value,
                comment=feedback.get("comment"),
                automated=automated,
            )
            bundle.evaluator_results.extend(evaluations)
            bundle.annotations.extend(annotations)
            if feedback.get("correction") is not None:
                bundle.annotations.append(
                    metadata_annotation(
                        span_id=str(target["span_id"]),
                        session_id=str(target["session_id"]),
                        metadata={"langsmith.correction": to_jsonable(feedback["correction"])},
                    )
                )
            add_provider_signal_raw(target, SOURCE, {**raw, **feedback})
    export_raw, coverage = partition_record(
        payload,
        record_name="export",
        mapped_fields={"runs", "feedback"},
        ignored_fields={"cursor", "next_cursor", "count"},
    )
    bundle.coverage.append(coverage)
    if export_raw and bundle.spans:
        add_provider_raw(bundle.spans[0], SOURCE, {"export": export_raw})
    bundle.validate()
    return bundle


def fetch_langsmith(args: argparse.Namespace) -> dict[str, Any]:
    if importlib.util.find_spec("langsmith") is None:
        raise RuntimeError("LangSmith live import requires the `langsmith` package")
    langsmith = importlib.import_module("langsmith")
    client = langsmith.Client(
        api_url=validated_service_url(args.langsmith_endpoint, label="LangSmith endpoint"),
        api_key=os.environ.get("LANGSMITH_API_KEY"),
    )
    runs: list[dict[str, Any]] = []
    for run in client.list_runs(project_name=args.project, start_time=args.since):
        converted = _object(run, "LangSmith run")
        started_at = datetime.fromisoformat(str(converted["start_time"]).replace("Z", "+00:00"))
        if started_at.tzinfo is None or started_at.utcoffset() is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        if started_at >= args.until:
            continue
        runs.append(converted)
    feedback: list[dict[str, Any]] = []
    if args.include_feedback:
        run_ids = [run["id"] for run in runs]
        for start in range(0, len(run_ids), 100):
            feedback.extend(
                _object(item, "LangSmith feedback")
                for item in client.list_feedback(run_ids=run_ids[start : start + 100])
            )
    return {"runs": runs, "feedback": feedback}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        "--langsmith-endpoint",
        default=os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
    )
    parser.add_argument("--feedback-input", type=Path, help="Optional JSON file containing `feedback`.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_common_arguments(parser, args)
    payload = _object(load_json(args.input) if args.input else fetch_langsmith(args), "LangSmith export")
    if args.feedback_input:
        feedback_payload = load_json(args.feedback_input)
        feedback = (
            feedback_payload.get("feedback", feedback_payload)
            if isinstance(feedback_payload, dict)
            else feedback_payload
        )
        if not isinstance(feedback, list):
            raise ValueError("LangSmith feedback input must be an array or an object containing `feedback`")
        payload["feedback"] = feedback
    bundle = map_langsmith_export(payload, project=args.project, include_feedback=args.include_feedback)
    return run_import(bundle, args)


def _object(value: Any, label: str) -> dict[str, Any]:
    converted = to_jsonable(value)
    if not isinstance(converted, dict):
        raise ValueError(f"{label} must be an object")
    return converted


def _nested(value: dict[str, Any], first: str, second: str) -> Any:
    nested = value.get(first)
    return nested.get(second) if isinstance(nested, dict) else None


if __name__ == "__main__":
    raise SystemExit(main())
