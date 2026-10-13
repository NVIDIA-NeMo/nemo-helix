# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed demo Intake telemetry for the Insights Analyst execute-job demo.

Expands the scenario spec in ``demo_spans.json`` into Intake spans and
annotations, then posts them::

    POST /apis/intake/v2/workspaces/{workspace}/ingest/spans
    POST /apis/intake/v2/workspaces/{workspace}/annotations

The corpus is sized against the Analyst's own bar: it files an Insight only for
patterns it can evidence with at least three representative traces, and it ranks
issues recurring across many sessions above one-offs. It also starts from
feedback, so each failing scenario carries negative ``feedback`` annotations and
a low ``helpfulness`` label.

``gen_ai.agent.name`` is stamped on every span — that attribute is what the
Analyst's ``agent_name`` span filter matches on, so it must equal the target
agent name. Feedback is attached at session level (no ``span_id``), which is
both the realistic shape for an end-user thumbs-down and the id the Analyst
correlates back to spans with.

Re-running is safe. Intake's span table sorts on ``start_time``, so a repeat
post only updates in place when every span keeps its original timestamp; a
reseed therefore reuses the origin of the corpus already in Intake instead of
re-anchoring to now. Annotations have no natural key, so the script skips any
whose ``(session_id, kind, name, value)`` already exists.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

DEFAULT_SPEC = Path(__file__).with_name("demo_spans.json")
AGENT_NAME_ATTRIBUTE = "gen_ai.agent.name"
HELPFULNESS_LABEL = "helpfulness"


def main() -> None:
    args = _parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    sessions = _expand_sessions(spec)
    source = args.source or spec.get("source", "insights-demo")

    if args.dry_run:
        spans, annotations = _render(sessions, args.target_agent, args.started_at)
        print(json.dumps({"spans": {"source": source, "spans": spans}, "annotations": annotations}, indent=2))
        return

    base_url = args.base_url.rstrip("/")
    workspace = args.workspace
    with httpx.Client(timeout=args.request_timeout) as client:
        origin = args.started_at or _existing_origin(client, base_url, workspace, source, sessions)
        spans, annotations = _render(sessions, args.target_agent, origin)

        payload = {"source": source, "spans": spans}
        response = client.post(f"{base_url}/apis/intake/v2/workspaces/{workspace}/ingest/spans", json=payload)
        if response.status_code != 201:
            raise SystemExit(f"Span ingest failed ({response.status_code}): {response.text}")
        print(f"Ingested {len(spans)} spans across {len(sessions)} sessions as source '{source}'.")

        if args.skip_annotations:
            print("Skipped annotations (--skip-annotations).")
        else:
            created_count = 0
            for annotation in annotations:
                if _annotation_exists(client, base_url, workspace, annotation):
                    continue
                created = client.post(f"{base_url}/apis/intake/v2/workspaces/{workspace}/annotations", json=annotation)
                if created.status_code != 201:
                    raise SystemExit(f"Annotation create failed ({created.status_code}): {created.text}")
                created_count += 1
            print(f"Created {created_count} annotations ({len(annotations) - created_count} already present).")

        _verify(client, base_url, workspace, args.target_agent, len(spans))


def _existing_origin(
    client: httpx.Client,
    base_url: str,
    workspace: str,
    source: str,
    sessions: list[dict[str, Any]],
) -> datetime | None:
    """Recover the origin of a previous seed so a reseed overwrites rather than duplicates.

    Intake sorts spans on ``start_time``, so re-posting a span with a new
    timestamp adds a second row with the same span id instead of replacing it,
    and the Analyst rejects the resulting trace.
    """
    anchor = sessions[0]
    root = next(template for template in anchor["span_templates"] if "parent_suffix" not in template)
    root_span_id = f"{anchor['id_prefix']}-{root['suffix']}"
    response = client.get(
        f"{base_url}/apis/intake/v2/workspaces/{workspace}/spans",
        params={
            "filter[trace_id]": anchor["trace_id"],
            "filter[source]": source,
            "page_size": 100,
            "mode": "summary",
        },
    )
    response.raise_for_status()
    starts = sorted(
        datetime.fromisoformat(span["started_at"]).replace(tzinfo=timezone.utc)
        for span in response.json()["data"]
        if span["span_id"] == root_span_id
    )
    if not starts:
        return None
    if len(starts) > 1:
        print(
            f"Warning: span '{root_span_id}' already exists {len(starts)} times in Intake from earlier seeds; "
            "the Analyst will reject these traces until the duplicates are removed. Reusing the newest origin."
        )
    origin = starts[-1] - timedelta(seconds=anchor["base_offset"] + root["start_offset_seconds"])
    print(f"Reusing the existing corpus origin {origin.isoformat()} so spans update in place.")
    return origin


def _annotation_exists(client: httpx.Client, base_url: str, workspace: str, annotation: dict[str, Any]) -> bool:
    params = {"filter[session_id]": annotation["session_id"], "filter[kind]": annotation["kind"], "page_size": 100}
    if "name" in annotation:
        params["filter[name]"] = annotation["name"]
    response = client.get(f"{base_url}/apis/intake/v2/workspaces/{workspace}/annotations", params=params)
    response.raise_for_status()
    return any(existing.get("value") == annotation["value"] for existing in response.json()["data"])


def _verify(client: httpx.Client, base_url: str, workspace: str, target_agent: str, expected_spans: int) -> None:
    """Read the corpus back; ingest returns an empty 201 body."""
    groups = client.get(
        f"{base_url}/apis/intake/v2/workspaces/{workspace}/spans/groups",
        params={"by": "session_id", "filter[agent_name]": target_agent, "page_size": 1000},
    )
    groups.raise_for_status()
    session_count = len(groups.json()["data"])

    errors = client.get(
        f"{base_url}/apis/intake/v2/workspaces/{workspace}/spans",
        params={"filter[agent_name]": target_agent, "filter[status]": "error", "page_size": 1000, "mode": "summary"},
    )
    errors.raise_for_status()
    error_count = len(errors.json()["data"])

    print(
        f"Intake reports {session_count} session(s) and {error_count} error span(s) "
        f"for agent_name='{target_agent}' in '{workspace}'."
    )
    if session_count == 0:
        raise SystemExit("Ingest returned 201 but nothing is readable back. Check that ClickHouse is running.")
    if error_count == 0:
        raise SystemExit(f"Seeded {expected_spans} spans but no error spans are queryable; the Analyst needs those.")


def _expand_sessions(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the scenario spec into one entry per session."""
    sessions: list[dict[str, Any]] = []
    scenario_offset = 0
    for scenario in spec["scenarios"]:
        key = scenario["key"]
        cases = scenario["cases"]
        stride = scenario.get("session_stride_seconds", 300)
        negative = scenario.get("negative_feedback_sessions", 0)
        positive = scenario.get("positive_feedback_sessions", 0)
        for index in range(scenario["session_count"]):
            case = cases[index % len(cases)]
            feedback = "negative" if index < negative else ("positive" if index < positive else None)
            sessions.append(
                {
                    "session_id": f"demo-{key}-{index + 1:02d}",
                    "trace_id": f"demo-trace-{key}-{index + 1:02d}",
                    "base_offset": scenario_offset + index * stride,
                    "span_templates": scenario["spans"],
                    "values": {**case, "n": f"{index + 1:02d}"},
                    "id_prefix": f"demo-{key}-{index + 1:02d}",
                    "feedback": feedback,
                    "helpfulness_score": scenario.get("helpfulness_score") if feedback else None,
                }
            )
        scenario_offset += scenario["session_count"] * stride
    return sessions


def _render(
    sessions: list[dict[str, Any]],
    target_agent: str,
    started_at: datetime | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Materialize spans and annotations with absolute timestamps."""
    span_extent = max(
        session["base_offset"] + template["start_offset_seconds"] + template.get("duration_seconds", 0)
        for session in sessions
        for template in session["span_templates"]
    )
    # Anchor so the newest span lands just before now: the fixture stores
    # relative offsets, and a large corpus would otherwise run into the future.
    origin = started_at or (datetime.now(timezone.utc) - timedelta(seconds=span_extent + 60))

    spans: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for session in sessions:
        values = {**session["values"], "session": session["session_id"], "trace": session["trace_id"]}
        for template in session["span_templates"]:
            spans.append(_render_span(template, session, values, origin, target_agent))

        if session["feedback"] is not None:
            annotations.append({"kind": "feedback", "session_id": session["session_id"], "value": session["feedback"]})
            if session["helpfulness_score"] is not None:
                annotations.append(
                    {
                        "kind": "label",
                        "session_id": session["session_id"],
                        "name": HELPFULNESS_LABEL,
                        "value_type": "numeric",
                        "value": session["helpfulness_score"],
                    }
                )
    return spans, annotations


def _render_span(
    template: dict[str, Any],
    session: dict[str, Any],
    values: dict[str, Any],
    origin: datetime,
    target_agent: str,
) -> dict[str, Any]:
    start = origin + timedelta(seconds=session["base_offset"] + template["start_offset_seconds"])
    span: dict[str, Any] = {
        "span_id": f"{session['id_prefix']}-{template['suffix']}",
        "trace_id": session["trace_id"],
        "session_id": session["session_id"],
        "name": template["name"],
        "kind": template["kind"],
        "status": template["status"],
        "started_at": start.isoformat(),
        "input": _substitute(template.get("input"), values),
        "output": _substitute(template.get("output"), values),
        "attributes": {
            **_substitute(template.get("attributes") or {}, values),
            AGENT_NAME_ATTRIBUTE: target_agent,
        },
    }
    if "parent_suffix" in template:
        span["parent_span_id"] = f"{session['id_prefix']}-{template['parent_suffix']}"
    duration = template.get("duration_seconds")
    if duration is not None:
        span["ended_at"] = (start + timedelta(seconds=duration)).isoformat()
    return span


def _substitute(value: Any, values: dict[str, Any]) -> Any:
    """Recursively expand ``{placeholder}`` templates in strings."""
    if isinstance(value, str):
        return value.format(**values)
    if isinstance(value, dict):
        return {key: _substitute(item, values) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute(item, values) for item in value]
    return value


def _aware_datetime(raw: str) -> datetime:
    value = datetime.fromisoformat(raw)
    if value.tzinfo is None:
        raise argparse.ArgumentTypeError("--started-at must include a UTC offset, e.g. 2026-08-26T12:00:00+00:00")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--workspace", default="default")
    parser.add_argument(
        "--target-agent",
        default="demo-agent",
        help="Value written to gen_ai.agent.name on every span. Must match the Analyst's target agent.",
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--source", help="Override the fixture's Intake source name.")
    parser.add_argument(
        "--started-at",
        type=_aware_datetime,
        help=(
            "Timestamp the first span starts at (ISO-8601 with offset). Defaults to the origin of a corpus already "
            "in Intake, or else to placing the newest span just before now."
        ),
    )
    parser.add_argument("--skip-annotations", action="store_true", help="Post spans only.")
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
