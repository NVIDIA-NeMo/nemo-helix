# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render explicitly selected Gym records or standalone ATIF as private offline HTML.

This is a reader, not a provider validator, converter, or case-to-run matcher.
Only the named input is read. All recorded content is escaped text, including
URLs, Markdown, tool arguments, and image references. Requires only Python 3.11+.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any

MAX_SOURCE_BYTES = 32 * 1024 * 1024
MAX_SELECTED_BYTES = 8 * 1024 * 1024
MAX_RECORD_BYTES = 2 * 1024 * 1024
MAX_ROWS = 200
MAX_DEPTH = 64
MAX_VALUES = 20_000
MAX_REPORT_BYTES = 32 * 1024 * 1024
ROLES = {
    "case": "Case specification",
    "source": "Source provenance",
    "control": "Task control",
    "agent": "Actual-agent evidence",
}
STYLE = """
:root { color-scheme: light; font: 16px/1.55 system-ui, sans-serif; color: #182b37; background: #f3f6f8; }
* { box-sizing: border-box; } body { margin: 0; overflow-wrap: anywhere; } a { color: #145e91; }
header, main { max-width: 1120px; margin: auto; padding: 24px; } header { padding-bottom: 0; }
h1 { margin: 0 0 8px; font-size: 2rem; } h2 { font-size: 1.4rem; } h3 { margin-bottom: 8px; }
p { margin: 8px 0; } .muted { color: #506371; } .badge { font-weight: 650; color: #315b50; }
article { background: white; padding: 24px; margin: 24px 0; border: 1px solid #cad6df; border-radius: 10px; }
section { margin: 24px 0; } .message { border-left: 3px solid #7899b2; padding: 8px 16px; margin: 12px 0; }
.notice { padding: 10px 14px; background: #eef3f7; border-left: 3px solid #54748a; }
.problem { padding: 10px 14px; background: #fff4dc; border-left: 3px solid #ab6800; }
pre, .text { white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; margin: 8px 0; }
pre { font: .88rem/1.55 ui-monospace, monospace; background: #f3f6f8; padding: 12px; border-radius: 4px; }
dl { margin: 8px 0; } dt { font-weight: 650; overflow-wrap: anywhere; } dd { margin: 0 0 12px 16px; }
details { border: 1px solid #d9e2e8; padding: 10px 14px; border-radius: 6px; margin: 12px 0; }
summary { cursor: pointer; font-weight: 600; } nav ul { display: flex; flex-wrap: wrap; gap: 8px 20px; padding: 0; }
nav li { list-style: none; } .provenance { font-size: .85rem; } .top { float: right; }
@media (max-width: 600px) { header, main { padding: 16px; } article { padding: 16px; } dd { margin-left: 8px; } }
@media print { body { background: white; } article { break-before: page; } nav, .top { display: none; } }
"""


class ReviewError(ValueError):
    """An input/output error whose message contains no recorded content."""


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, indent=2)


def text(value: Any) -> str:
    return f'<div class="text">{escape(value)}</div>'


def details(label: str, body: str) -> str:
    return f"<details><summary>{escape(label)}</summary>{body}</details>"


def fields(value: Any) -> str:
    """Readable scalar fields, with formatted structures instead of lossy flattening."""
    if not isinstance(value, dict):
        return f"<pre>{escape(pretty(value))}</pre>"
    if not value:
        return '<p class="muted">None recorded.</p>'
    return (
        "<dl>"
        + "".join(
            f"<dt>{escape(key)}</dt><dd>"
            + (
                f"<pre>{escape(pretty(item))}</pre>"
                if isinstance(item, (dict, list))
                else text(pretty(item) if item is None else item)
            )
            + "</dd>"
            for key, item in value.items()
        )
        + "</dl>"
    )


def remainder(value: dict, used: set[str], label: str = "Additional recorded fields") -> str:
    rest = {key: item for key, item in value.items() if key not in used}
    return details(label, fields(rest)) if rest else ""


def section(name: str, title: str, body: str) -> str:
    return f'<section data-section="{name}"><h3>{escape(title)}</h3>{body}</section>'


class Renderer:
    def __init__(self) -> None:
        self.issues = 0

    def problem(self, message: str) -> str:
        self.issues += 1
        return f'<p class="problem">{escape(message)}</p>'

    def content(self, value: Any) -> str:
        if isinstance(value, str):
            return text(value)
        if not isinstance(value, list):
            return self.problem("Missing or unsupported message content; recorded value follows.") + fields(value)
        body = ""
        for part in value:
            if (
                isinstance(part, dict)
                and part.get("type") in ("text", "input_text", "output_text")
                and isinstance(part.get("text"), str)
            ):
                body += text(part["text"]) + remainder(part, {"type", "text"})
            else:
                body += self.problem("Unsupported content shown as metadata only; media and references are not loaded.")
                body += fields(part)
        return body or '<p class="muted">Empty content recorded.</p>'

    def messages(self, value: Any, *, input_string: bool = False) -> str:
        if input_string and isinstance(value, str):
            return '<div class="message"><h4>user</h4>' + text(value) + "</div>"
        if not isinstance(value, list):
            return self.problem("Missing or malformed message list.") + fields(value)
        body = ""
        for item in value:
            if not isinstance(item, dict):
                body += self.problem("Malformed message item.") + fields(item)
                continue
            kind = item.get("type", "message")
            if kind == "message":
                label = item.get("role", "Unspecified role")
                rendered = self.content(item.get("content")) + remainder(item, {"type", "role", "content"})
            elif kind == "function_call":
                label = "Tool call: " + str(item.get("name", "Unspecified tool"))
                rendered = fields({key: val for key, val in item.items() if key not in {"type", "name"}})
            elif kind == "function_call_output":
                label = "Tool result"
                rendered = fields({key: val for key, val in item.items() if key not in {"type", "output"}})
                rendered += self.content(item.get("output"))
            elif kind == "reasoning":
                label = "Recorded reasoning summary"
                rendered = fields({key: val for key, val in item.items() if key != "type"})
            else:
                label = "Unsupported message type"
                rendered = self.problem(
                    "This item is shown as structured data, not interpreted as a message."
                ) + fields(item)
            body += f'<div class="message"><h4>{escape(label)}</h4>{rendered}</div>'
        return body or '<p class="notice">No messages recorded.</p>'

    def gym(self, record: dict, role: str) -> str:
        params = record.get("responses_create_params")
        if isinstance(params, dict):
            request = ""
            if "instructions" in params:
                request += "<h4>Instructions</h4>" + self.content(params["instructions"])
            request += self.messages(params.get("input"), input_string=True)
            request += remainder(params, {"instructions", "input"}, "Request configuration")
        else:
            request = self.problem("No usable responses_create_params request recorded.") + fields(params)
        body = section("request", "Model input", request)
        result_keys = {"reward", "rewards", "error", "errors", "failure_reason", "status"}
        identity_keys = {"id", "task_id", "case_id", "_ng_task_index", "_ng_rollout_index"}
        capture_keys = {"ng_trajectory", "ng_model_call_capture", "ng_agent_observations", "atif_conversion"}
        used = {"responses_create_params", "response"} | result_keys | identity_keys | capture_keys
        body += section(
            "grading",
            "Case fields — grading and other metadata",
            fields({k: v for k, v in record.items() if k not in used}),
        )
        identity = {k: v for k, v in record.items() if k in identity_keys}
        if identity:
            body += section("identity", "Recorded identifiers", fields(identity))
        response = record.get("response")
        if "response" not in record or response is None:
            message = "No response recorded. This record does not establish execution."
            rendered = self.problem(message) if role in {"agent", "control"} else f'<p class="notice">{message}</p>'
        elif isinstance(response, dict):
            rendered = fields({k: v for k, v in response.items() if k != "output"})
            rendered += self.messages(response.get("output"))
        else:
            rendered = self.problem("Malformed response; recorded value follows.") + fields(response)
        body += section("response", "Recorded response", rendered)
        results = {k: v for k, v in record.items() if k in result_keys}
        body += section("results", "Reported rewards and errors", fields(results))
        capture = {k: v for k, v in record.items() if k in capture_keys}
        if capture:
            body += section(
                "capture",
                "Additional capture and conversion metadata",
                self.problem(
                    "Additional capture is shown as structured data, not rendered as a transcript. Referenced files are not followed."
                )
                + fields(capture),
            )
        return body

    def atif(self, record: dict) -> str:
        version = record.get("schema_version")
        if version not in tuple(f"ATIF-v1.{minor}" for minor in range(8)):
            return self.problem("Unsupported or missing ATIF schema_version; use the raw record below.")
        body = fields(
            {
                k: v
                for k, v in record.items()
                if k in {"schema_version", "session_id", "trajectory_id", "agent", "notes"}
            }
        )
        steps = record.get("steps")
        if not isinstance(steps, list) or not steps:
            body += self.problem("Missing or malformed ATIF steps.") + fields(steps)
        else:
            for ordinal, step in enumerate(steps, 1):
                if not isinstance(step, dict):
                    body += self.problem(f"Malformed step at position {ordinal}.") + fields(step)
                    continue
                step_label = (
                    f"Step {step['step_id']}" if "step_id" in step else f"Step at position {ordinal} (ID missing)"
                )
                label = f"{step_label} · {step.get('source', 'Unspecified source')}"
                rendered = self.content(step.get("message"))
                if step.get("is_copied_context") is True:
                    rendered = '<p class="notice">Copied context (as recorded).</p>' + rendered
                if "reasoning_content" in step:
                    rendered += details("Recorded reasoning", fields(step["reasoning_content"]))
                calls = step.get("tool_calls")
                if calls is not None:
                    if not isinstance(calls, list):
                        rendered += self.problem("Malformed tool_calls list.") + fields(calls)
                    else:
                        for call in calls:
                            if isinstance(call, dict):
                                rendered += (
                                    "<h4>Tool call: "
                                    + escape(call.get("function_name", "Unspecified tool"))
                                    + "</h4>"
                                    + fields(call)
                                )
                            else:
                                rendered += self.problem("Malformed tool call.") + fields(call)
                observation = step.get("observation")
                if observation is not None:
                    rendered += "<h4>Tool results / observations</h4>"
                    if isinstance(observation, dict) and isinstance(observation.get("results"), list):
                        for result in observation["results"]:
                            if isinstance(result, dict):
                                rendered += fields({k: v for k, v in result.items() if k != "content"})
                                if "content" in result and result["content"] is not None:
                                    rendered += self.content(result["content"])
                            else:
                                rendered += self.problem("Malformed observation result.") + fields(result)
                        rendered += remainder(observation, {"results"})
                    else:
                        rendered += self.problem("Malformed observation.") + fields(observation)
                rendered += remainder(
                    step, {"step_id", "source", "message", "reasoning_content", "tool_calls", "observation"}
                )
                body += f'<div class="message"><h3>{escape(label)}</h3>{rendered}</div>'
        body += section("results", "Final metrics (as recorded)", fields(record.get("final_metrics")))
        if record.get("continued_trajectory_ref") or record.get("subagent_trajectories"):
            body += self.problem(
                "Continuation references are not followed. Embedded subagents remain structured data below, outside the rendered transcript."
            )
        body += section(
            "metadata",
            "Provenance, conversion limits, and other recorded fields",
            fields(
                {
                    k: v
                    for k, v in record.items()
                    if k
                    not in {"schema_version", "session_id", "trajectory_id", "agent", "notes", "steps", "final_metrics"}
                }
            ),
        )
        return body

    def record(self, raw: bytes | None, number: int, format_name: str, role: str, missing: str | None = None) -> str:
        title = f"Row {number}" if format_name == "gym-jsonl" else "ATIF trajectory"
        body = f'<article id="record-{number}" data-record="{number}"><a class="top" href="#top">Back to index</a><h2>{title}</h2>'
        body += f'<p class="badge">{escape(ROLES[role])} · caller-supplied label</p>'
        if missing:
            return body + self.problem(missing) + "</article>"
        assert raw is not None
        body += f'<p class="provenance">Selected record SHA-256: {hashlib.sha256(raw).hexdigest()}</p>'
        try:
            source = raw.decode("utf-8")
        except UnicodeDecodeError:
            return (
                body
                + self.problem("Malformed record: invalid UTF-8. See the original input for the retained bytes.")
                + "</article>"
            )
        try:
            record = json.loads(source, object_pairs_hook=unique_object, parse_constant=reject_constant)
            limit = readability_limit(record)
            if limit:
                body += self.problem(limit)
            elif not isinstance(record, dict):
                body += self.problem("Unsupported record: expected a JSON object.")
            else:
                body += self.gym(record, role) if format_name == "gym-jsonl" else self.atif(record)
        except (ValueError, RecursionError):
            body += self.problem(
                "Malformed JSON record (including duplicate keys, non-finite numbers, or excessive nesting)."
            )
        body += details("Raw selected record", f"<pre>{escape(source)}</pre>")
        return body + "</article>"


def readability_limit(record: Any) -> str | None:
    """Bound expansion before pretty-printing nested or very numerous values."""
    pending = [(record, 0)]
    count = 0
    while pending:
        value, depth = pending.pop()
        count += 1
        if depth > MAX_DEPTH or count > MAX_VALUES:
            return (
                "Unsupported record structure: exceeds 64 nesting levels or 20,000 values. Only raw evidence is shown."
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite JSON number")
        if isinstance(value, dict):
            pending.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            pending.extend((item, depth + 1) for item in value)
    return None


def unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def reject_constant(_value: str) -> None:
    raise ValueError("Non-finite JSON number")


def select_rows(value: str) -> list[int]:
    rows: set[int] = set()
    for token in value.split(","):
        if not re.fullmatch(r"[1-9][0-9]*(?:-[1-9][0-9]*)?", token):
            raise argparse.ArgumentTypeError("rows must be positive physical line numbers, e.g. 1,3-5")
        parts = token.split("-")
        try:
            start, end = int(parts[0]), int(parts[-1])
        except ValueError:
            raise argparse.ArgumentTypeError("row number is too large") from None
        if end < start or end - start >= MAX_ROWS:
            raise argparse.ArgumentTypeError(f"select at most {MAX_ROWS} rows in ascending ranges")
        rows.update(range(start, end + 1))
        if len(rows) > MAX_ROWS:
            raise argparse.ArgumentTypeError(f"select at most {MAX_ROWS} rows")
    return sorted(rows)


def read_records(path: Path, rows: list[int] | None) -> list[tuple[int, bytes | None, str | None]]:
    try:
        if not stat.S_ISREG(path.stat().st_mode):
            raise ReviewError("Input must be a regular file.")
        with path.open("rb") as stream:
            source = stream.read(MAX_SOURCE_BYTES + 1)
    except FileNotFoundError:
        return [(number, None, "Missing input file; no evidence was read.") for number in (rows or [1])]
    if len(source) > MAX_SOURCE_BYTES:
        raise ReviewError("Input exceeds 32 MiB; select a smaller source file.")
    if rows is None:
        records = [(1, source, None)]
    else:
        selected = set(rows)
        found = {number: raw for number, raw in enumerate(io.BytesIO(source), 1) if number in selected}
        records = [
            (
                number,
                found.get(number),
                None if number in found else "Missing selected row; physical line is outside the input.",
            )
            for number in rows
        ]
    if any(raw is not None and len(raw) > MAX_RECORD_BYTES for _, raw, _ in records):
        raise ReviewError("A selected record exceeds 2 MiB.")
    if sum(len(raw) for _, raw, _ in records if raw is not None) > MAX_SELECTED_BYTES:
        raise ReviewError("Selected records exceed 8 MiB; select fewer rows.")
    return records


def document(
    path: Path, records: list[tuple[int, bytes | None, str | None]], format_name: str, role: str
) -> tuple[str, int]:
    renderer = Renderer()
    parts = []
    size = 0
    for number, raw, missing in records:
        part = renderer.record(raw, number, format_name, role, missing)
        size += len(part.encode("utf-8", errors="xmlcharrefreplace"))
        if size > MAX_REPORT_BYTES:
            raise ReviewError("Rendered records exceed 32 MiB; select fewer rows.")
        parts.append(part)
    body = "".join(parts)
    links = "".join(
        f'<li><a href="#record-{number}">{"Row " + str(number) if format_name == "gym-jsonl" else "ATIF trajectory"}</a></li>'
        for number, _, _ in records
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Generated case review</title><style>{STYLE}</style></head>
<body><header id="top"><p class="badge">EVAL AUTHOR · LOCAL REVIEW</p><h1>Generated case review</h1>
<p><a href="{escape(path.as_uri())}">Original input</a>: {escape(path)}</p>
<p>{len(records)} selected record(s) · {renderer.issues} issue(s) · {escape(format_name)}</p>
<p class="muted">This snapshot contains selected source data. Keep it as private as the source.</p>
<p class="notice">Labels are supplied by the caller. Rendering does not validate a task, prove execution or success,
or associate a case with a run. Request and response remain in their recorded order and may overlap.
Missing capture is not proof that an action did not happen. ATIF schema validity is not checked.</p>
<nav aria-label="Selected records"><ul>{links}</ul></nav></header><main>{body}</main></body></html>"""
    return page, renderer.issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--format", choices=("gym-jsonl", "atif"), required=True)
    parser.add_argument("--rows", type=select_rows, help="Gym physical lines, e.g. 1,3-5; required for JSONL")
    parser.add_argument("--evidence-role", choices=tuple(ROLES), required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New private directory; existing paths are refused"
    )
    args = parser.parse_args()
    if (args.format == "gym-jsonl") != (args.rows is not None):
        parser.error("--rows is required for gym-jsonl and cannot be used with atif")
    try:
        path = args.input.expanduser().absolute()
        records = read_records(path, args.rows)
        page, issues = document(path, records, args.format, args.evidence_role)
        output = args.output_dir.expanduser().absolute()
        # mkdir is exclusive, including dangling symlinks. Never change an existing directory.
        output.mkdir(mode=0o700)
        report = output / "report.html"
        descriptor = os.open(report, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(page.encode("utf-8", errors="xmlcharrefreplace"))
    except (ReviewError, OSError) as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": str(exc)
                    if isinstance(exc, ReviewError)
                    else f"Local file operation failed ({type(exc).__name__}).",
                }
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "attention" if issues else "ok",
                "report": str(report),
                "records": len(records),
                "issues": issues,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
