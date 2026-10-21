# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static Markdown and HTML reports. Nothing here is served or fetched."""

from __future__ import annotations

import html
import json
from pathlib import Path

from config import Step, Sweep
from records import history_path, read_json, step_dir
from redact import redact

_GROUPS = (
    ("jobs", ("exit", "idle")),
    ("files", ("hammer",)),
    ("secrets", ("hammer",)),
    ("auth", ("hammer",)),
)


class ReportError(ValueError):
    """The local bundle cannot produce a report."""


def write_report(sweep: Sweep, run_dir: Path, output_dir: Path) -> None:
    """Write ``report.md``, ``report.html``, and the SVG charts next to the Markdown."""
    if not run_dir.is_dir():
        raise ReportError(f"local bundle {run_dir} does not exist")
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = _meta(run_dir)
    sections: list[str] = [_overview(sweep, meta)]
    charts: dict[str, str] = {}
    for group, kinds in _GROUPS:
        steps = [step for step in sweep.steps if _in_group(step, group, kinds)]
        chart_name = f"{group}.svg"
        chart = _chart(group, [_step_bar(run_dir, step) for step in steps])
        charts[chart_name] = chart
        sections.append(_section(run_dir, group, steps, chart_name))
    markdown = redact("\n".join(sections) + "\n")
    document = _html(markdown, charts)
    (output_dir / "report.md").write_text(markdown)
    (output_dir / "report.html").write_text(document)
    for name, chart in charts.items():
        (output_dir / name).write_text(chart)


def report_files(output_dir: Path) -> list[Path]:
    return sorted(path for path in output_dir.iterdir() if path.is_file())


def _meta(run_dir: Path) -> dict[str, object]:
    path = run_dir / "meta.json"
    if not path.is_file():
        return {}
    return read_json(path)


def _overview(sweep: Sweep, meta: dict[str, object]) -> str:
    rows = [
        f"# Stress report `{sweep.run_key}`",
        "",
        f"- Context: `{_text(meta.get('context'), 'unknown')}`",
        f"- Base URL: `{_text(meta.get('base_url'), 'unknown')}`",
        f"- Profile: `{_text(meta.get('profile'), 'unknown')}`",
        f"- TTL before active: `{_text(meta.get('ttl_seconds_before_active'), 'unknown')}`",
        f"- TTL active: `{_text(meta.get('ttl_seconds_active'), 'unknown')}`",
        f"- Worker digest: `{_text(meta.get('worker_digest'), 'unknown')}`",
        f"- Controller: `{_text(meta.get('controller_version'), 'unknown')}`",
        "",
        "## Pod sizes",
        "",
        "| Kind | Requests | Limits |",
        "| --- | --- | --- |",
    ]
    for kind, pod in sweep.pods.items():
        rows.append(
            f"| {kind} | {pod.cpu_request} CPU, {pod.memory_request} | {pod.cpu_limit} CPU, {pod.memory_limit} |"
        )
    return "\n".join(rows)


def _in_group(step: Step, group: str, kinds: tuple[str, ...]) -> bool:
    if step.kind not in kinds:
        return False
    if group == "jobs":
        return step.kind in ("exit", "idle")
    return step.service == group


def _section(run_dir: Path, group: str, steps: list[Step], chart_name: str) -> str:
    lines = ["", f"## {group}", "", f"![{group}]({chart_name})", ""]
    if not steps:
        lines.append("No steps in the loaded sweep.")
        return "\n".join(lines)
    lines.extend(
        [
            "| Step | Kind | Count | Duration | Service | Rate | Accepted | Terminal | Scheduling | Missing | Partial | Code fetch | Achieved |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for step in steps:
        stats = _stats(run_dir, step)
        lines.append(
            "| {id} | {kind} | {count} | {duration} | {service} | {rate} | {accepted} | {terminal} | "
            "{scheduling} | {missing} | {partial} | {fetch} | {achieved} |".format(
                id=step.id,
                kind=step.kind,
                count=step.count,
                duration=step.duration_seconds if step.duration_seconds is not None else "",
                service=step.service or "",
                rate=step.rate_per_second if step.rate_per_second is not None else "",
                accepted=stats["accepted"],
                terminal=stats["terminal"],
                scheduling=stats["scheduling"],
                missing=stats["missing"],
                partial="yes" if stats["partial"] else "no",
                fetch=stats["fetch"],
                achieved=stats["achieved"],
            )
        )
        latency = stats["latency"]
        if isinstance(latency, dict) and latency:
            lines.extend(["", f"### {step.id} latency", ""])
            for operation, value in latency.items():
                lines.append(f"- `{operation}`: {value}")
        excerpt = _excerpt(run_dir, step)
        if excerpt:
            lines.extend(["", f"### {step.id} history", "", excerpt])
    return "\n".join(lines)


def _stats(run_dir: Path, step: Step) -> dict[str, object]:
    directory = step_dir(run_dir, step.id)
    accepted = 0
    terminal: dict[str, int] = {}
    scheduling = 0
    missing = 0
    partial = False
    fetches: list[float] = []
    achieved: list[float] = []
    latency: dict[str, list[float]] = {}
    if not directory.is_dir():
        return _empty_stats()
    for path in sorted(directory.glob("*.record.json")):
        payload = read_json(path)
        name = payload.get("job_name")
        if not isinstance(name, str):
            continue
        if payload.get("submit") != "accepted":
            scheduling += 1
            continue
        accepted += 1
        history = payload.get("status_history")
        statuses = [item for item in history if isinstance(item, str)] if isinstance(history, list) else []
        last = statuses[-1] if statuses else "unknown"
        terminal[last] = terminal.get(last, 0) + 1
        if last == "error" and "active" not in statuses:
            scheduling += 1
        if payload.get("download_succeeded") is not True:
            missing += 1
            partial = True
        history_file = history_path(run_dir, step.id, name)
        if not history_file.is_file():
            partial = True
            continue
        _read_history(history_file, fetches, achieved, latency)
    mean_latency = {operation: _mean(values) for operation, values in latency.items()}
    return {
        "accepted": accepted,
        "terminal": ", ".join(f"{key}={value}" for key, value in sorted(terminal.items())),
        "scheduling": scheduling,
        "missing": missing,
        "partial": partial,
        "fetch": _mean(fetches) if fetches else "",
        "achieved": _mean(achieved) if achieved else "",
        "latency": mean_latency,
    }


def _empty_stats() -> dict[str, object]:
    return {
        "accepted": 0,
        "terminal": "",
        "scheduling": 0,
        "missing": 0,
        "partial": False,
        "fetch": "",
        "achieved": "",
        "latency": {},
    }


def _read_history(path: Path, fetches: list[float], achieved: list[float], latency: dict[str, list[float]]) -> None:
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        fetch = payload.get("code_fetch_seconds")
        if isinstance(fetch, (int, float)) and not isinstance(fetch, bool):
            fetches.append(float(fetch))
        rate = payload.get("achieved_rate")
        if isinstance(rate, (int, float)) and not isinstance(rate, bool):
            achieved.append(float(rate))
        operation = payload.get("operation")
        delay = payload.get("latency")
        if isinstance(operation, str) and isinstance(delay, (int, float)) and not isinstance(delay, bool):
            latency.setdefault(operation, []).append(float(delay))


def _excerpt(run_dir: Path, step: Step) -> str:
    directory = step_dir(run_dir, step.id)
    if not directory.is_dir():
        return ""
    chunks: list[str] = []
    for path in sorted(directory.glob("*.jsonl")):
        lines = path.read_text().splitlines()[:200]
        chunks.extend(lines)
    return "\n".join(chunks)


def _step_bar(run_dir: Path, step: Step) -> tuple[str, float]:
    directory = step_dir(run_dir, step.id)
    if not directory.is_dir():
        return step.id, 0
    return step.id, float(len(list(directory.glob("*.record.json"))))


def _chart(title: str, rows: list[tuple[str, float]]) -> str:
    width = 640
    row_height = 28
    height = 36 + row_height * max(len(rows), 1)
    peak = max((value for _, value in rows), default=1.0) or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img">',
        f"<title>{html.escape(title)}</title>",
    ]
    for index, (label, value) in enumerate(rows):
        y = 28 + index * row_height
        bar = int((width - 180) * (value / peak))
        parts.append(
            f'<text x="8" y="{y + 14}" font-size="12">{html.escape(label)}</text>'
            f'<rect x="140" y="{y}" width="{bar}" height="16" fill="#333"></rect>'
            f'<text x="{148 + bar}" y="{y + 13}" font-size="12">{value:g}</text>'
        )
    parts.append("</svg>\n")
    return "".join(parts)


def _html(markdown: str, charts: dict[str, str]) -> str:
    body = ["<article>"]
    for block in markdown.split("\n\n"):
        body.append(f"<pre>{html.escape(block)}</pre>")
    for chart in charts.values():
        body.append(chart)
    body.append("</article>")
    document = (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><title>Stress report</title>'
        "<style>body{font-family:sans-serif;margin:2rem} pre{white-space:pre-wrap}</style>"
        f"</head><body>{''.join(body)}</body></html>\n"
    )
    if "<script" in document.lower():
        raise ReportError("report HTML must not contain a script")
    return document


def _text(value: object, fallback: str) -> str:
    if value is None:
        return fallback
    return str(value)


def _mean(values: list[float]) -> str:
    if not values:
        return ""
    return f"{sum(values) / len(values):.3f}"
