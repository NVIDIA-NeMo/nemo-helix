# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static Markdown and HTML reports. Nothing here is served or fetched."""

from __future__ import annotations

import html
import json
from pathlib import Path

from config import Step, Sweep
from records import cohort_path, history_path, read_json, step_dir
from redact import redact

_GROUPS = (
    ("exit", ("exit",)),
    ("idle", ("idle",)),
    ("barrier", ("barrier",)),
    ("files", ("hammer",)),
    ("secrets", ("hammer",)),
    ("auth", ("hammer",)),
)
_EXIT_RESULTS = ("Accepted", "Terminal", "Scheduling", "Throughput")
_BARRIER_RESULTS = (
    "Accepted",
    "Ready",
    "Peak running",
    "Submit",
    "Ready time",
    "Release to finalized",
    "Barrier",
    "Timeout",
    "Unknown",
)
_HAMMER_RESULTS = (
    "Accepted",
    "Terminal",
    "Scheduling",
    "Missing",
    "Partial",
    "Zip fetch (s)",
    "Throughput",
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
        configured = [step for step in sweep.steps if _in_group(step, group, kinds)]
        ran = [step for step in configured if _has_records(run_dir, step)]
        chart_name = f"{group}.svg"
        if ran:
            charts[chart_name] = _chart(group, [_chart_row(group, run_dir, step) for step in ran])
        sections.append(_section(sweep, group, ran, configured, chart_name if ran else None, run_dir))
    markdown = redact("\n".join(sections) + "\n")
    document = _html(markdown, charts)
    (output_dir / "report.md").write_text(markdown)
    (output_dir / "report.html").write_text(document)
    for name, chart in charts.items():
        (output_dir / name).write_text(chart)
    for path in output_dir.glob("*.svg"):
        if path.name not in charts:
            path.unlink()


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
        f"- Backend: `{_text(meta.get('backend'), 'unknown')}`",
        f"- Image: `{_text(meta.get('image') or None, 'profile default')}`",
        f"- Auth: `{_text(meta.get('auth'), 'unknown')}`",
        f"- TTL before active: `{_text(meta.get('ttl_seconds_before_active'), 'unknown')}`",
        f"- TTL active: `{_text(meta.get('ttl_seconds_active'), 'unknown')}`",
        f"- Worker digest: `{_text(meta.get('worker_digest'), 'unknown')}`",
        f"- Controller: `{_text(meta.get('controller_version'), 'unknown')}`",
        "",
        "Each section below is one benchmark. Rows in a section use the same metrics. "
        "Count, where a section has it, is the number of jobs that step starts in each workspace. "
        "A run with more than one workspace submits that count to every workspace, round-robin.",
    ]
    return "\n".join(rows)


def _in_group(step: Step, group: str, kinds: tuple[str, ...]) -> bool:
    if step.kind not in kinds:
        return False
    if group in {"exit", "idle", "barrier"}:
        return True
    return step.service == group


def _has_records(run_dir: Path, step: Step) -> bool:
    directory = step_dir(run_dir, step.id)
    return directory.is_dir() and any(directory.glob("*.record.json"))


def _section(
    sweep: Sweep, group: str, steps: list[Step], configured: list[Step], chart_name: str | None, run_dir: Path
) -> str:
    lines = ["", f"## {group}", "", _about(group)]
    if not configured:
        lines.extend(["", "No steps in the loaded sweep."])
        return "\n".join(lines)
    stable, varying = _split_inputs(group, configured)
    parameters = [*_pod_parameters(sweep, group), *stable]
    if parameters:
        lines.extend(["", *_parameter_table(parameters)])
    if not steps:
        lines.extend(["", "No steps in this group have run."])
        return "\n".join(lines)
    if chart_name is not None:
        lines.extend(["", f"![{group}]({chart_name})", "", _chart_caption(group)])
    lines.extend(["", *_metrics(group), ""])
    columns = ("Step", *varying, *_result_columns(group))
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _column in columns) + " |")
    details: list[str] = []
    for step in steps:
        lines.append(_row(group, run_dir, step, varying))
        details.extend(_details(group, run_dir, step))
    if group not in {"exit", "idle", "barrier"}:
        lines.extend(_operation_table(run_dir, steps))
    return "\n".join([*lines, *details])


def _pod_kind(group: str) -> str:
    if group in {"exit", "idle", "barrier"}:
        return group
    return "hammer"


def _pod_parameters(sweep: Sweep, group: str) -> list[tuple[str, str]]:
    pod = sweep.pods.get(_pod_kind(group))
    if pod is None:
        return []
    return [
        ("CPU request", pod.cpu_request),
        ("Memory request", pod.memory_request),
        ("CPU limit", pod.cpu_limit),
        ("Memory limit", pod.memory_limit),
    ]


def _parameter_table(rows: list[tuple[str, str]]) -> list[str]:
    lines = [
        "These inputs are the same for every step in this section.",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
    ]
    lines.extend(f"| {label} | {value} |" for label, value in rows)
    return lines


def _inputs(group: str, step: Step) -> list[tuple[str, str]]:
    if group == "exit":
        return [("Count", str(step.count))]
    if group == "idle":
        return [("Count", str(step.count)), ("Duration", _optional(step.duration_seconds))]
    if group == "barrier":
        return [
            ("Jobs per workspace", str(step.count)),
            ("Workspaces", str(step.workspaces)),
            ("Timeout (s)", _optional(step.timeout_seconds)),
        ]
    rate = "" if step.rate_per_second is None else str(step.rate_per_second)
    return [("Count", str(step.count)), ("Duration", _optional(step.duration_seconds)), ("Rate", rate)]


def _optional(value: int | None) -> str:
    return "" if value is None else str(value)


def _split_inputs(group: str, steps: list[Step]) -> tuple[list[tuple[str, str]], list[str]]:
    rows = [_inputs(group, step) for step in steps]
    if not rows:
        return [], []
    stable: list[tuple[str, str]] = []
    varying: list[str] = []
    for index, (label, _value) in enumerate(rows[0]):
        values = [row[index][1] for row in rows]
        if len(set(values)) == 1:
            stable.append((label, values[0]))
        else:
            varying.append(label)
    return stable, varying


def _result_columns(group: str) -> tuple[str, ...]:
    if group == "barrier":
        return _BARRIER_RESULTS
    if group in {"exit", "idle"}:
        return _EXIT_RESULTS
    return _HAMMER_RESULTS


def _about(group: str) -> str:
    if group == "exit":
        return (
            "Exit runs `/bin/bash -c` with `exit 0`. The job does not download the worker. "
            "This shows how quickly the platform accepts and finishes jobs that do no work."
        )
    if group == "idle":
        return (
            "Idle runs `/bin/bash -c` with `sleep` for the step duration. The job does not download the worker. "
            "The sleep holds the pod so later jobs can pile up in the queue. "
            "Throughput is completed jobs divided by the time from the first submit until the last finish. "
            "If every job runs at once, the ceiling is the count divided by the duration."
        )
    if group == "barrier":
        return (
            "Barrier jobs download the worker zip, write a ready file, and wait. "
            "When every job has written ready, the controller writes a release file into each workspace fileset. "
            "Barrier means the release file was seen and the job exited 0. "
            "Timeout means the job reached its timeout first and exited 1."
        )
    service = group
    return (
        f"The {service} hammer downloads the worker zip, then calls the {service} service at the configured rate "
        "for the step duration. "
        "Zip fetch is the time spent downloading the worker before those calls, and it is separate from service latency."
    )


def _chart_caption(group: str) -> str:
    if group == "barrier":
        return (
            "The bars are the largest number of these jobs observed active while the controller waited for ready files."
        )
    if group in {"exit", "idle"}:
        return "The bars are throughput, in jobs per second."
    return "The bars are throughput, in operations per second."


def _metrics(group: str) -> list[str]:
    if group == "exit":
        lines = [
            "Count is the number of jobs this step starts in each workspace.",
            "Accepted is how many creates the platform accepted.",
            "Terminal counts the last status reported for each accepted job.",
            "Scheduling counts creates the platform refused, plus accepted jobs that reached error without becoming active.",
            "Throughput is completed jobs divided by the time from the first submit until the last job was seen finished, "
            "in jobs per second. The window includes submission and queueing.",
        ]
    elif group == "idle":
        lines = [
            "Count is the number of jobs this step starts in each workspace.",
            "Duration is how many seconds each job sleeps.",
            "Accepted is how many creates the platform accepted.",
            "Terminal counts the last status reported for each accepted job.",
            "Scheduling counts creates the platform refused, plus accepted jobs that reached error without becoming active.",
            "Throughput uses the same window as exit. A long sleep lowers the ceiling, because the jobs stay running together.",
        ]
    elif group == "barrier":
        lines = [
            "Jobs per workspace is the configured count started in each workspace.",
            "Workspaces is how many workspaces received jobs for this step.",
            "Accepted is how many creates the platform accepted.",
            "Ready is how many jobs wrote a ready file before the cohort timed out.",
            "Peak running is the largest number of these jobs observed in the active status during that wait.",
            "Submit is the seconds from the start of the first create until the last create returned.",
            "Ready time is the seconds from the start of submit until every job was ready, or until the timeout. "
            "It includes the submit time.",
            "Release to finalized is the seconds from the release write until every job was terminal and deleted. "
            "The cell says `not released` when some jobs never became ready, so the controller did not write the release.",
            "Barrier counts jobs that saw the release and exited 0.",
            "Timeout counts jobs that exited 1 because the timeout elapsed.",
            "Unknown counts accepted jobs that recorded neither reason.",
            "A step that uses more than one workspace lists the same counts again for each workspace. "
            "Fair sharing would give those workspaces similar accepted and peak counts while the jobs contend.",
        ]
    else:
        lines = [
            "Count is the number of jobs this step starts in each workspace.",
            "Duration is how many seconds each job calls the service.",
            "Rate is the target operations per second inside each job.",
            "Accepted is how many creates the platform accepted.",
            "Terminal counts the last status reported for each accepted job.",
            "Scheduling counts creates the platform refused, plus accepted jobs that reached error without becoming active.",
            "Missing counts accepted jobs whose service-stress result was not copied.",
            "Partial is yes when any job is missing that result or its operation log.",
            "Zip fetch is the mean seconds the bootstrap spent downloading the worker zip.",
            "Throughput is the mean operation rate the workers reported, in operations per second.",
            "The operation table lists the mean calls per second for each operation. Each row is one step.",
        ]
    return [f"- {line}" for line in lines]


def _row(group: str, run_dir: Path, step: Step, varying: list[str]) -> str:
    cells: list[object] = [step.id, *(_input_value(group, step, label) for label in varying)]
    if group == "barrier":
        cells.extend(_barrier_results(run_dir, step))
    else:
        stats = _stats(run_dir, step)
        cells.extend([stats["accepted"], stats["terminal"], stats["scheduling"]])
        if group not in {"exit", "idle"}:
            cells.extend(
                [
                    stats["missing"],
                    "yes" if stats["partial"] else "no",
                    stats["fetch"],
                ]
            )
        cells.append(stats["achieved"])
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _input_value(group: str, step: Step, label: str) -> str:
    for name, value in _inputs(group, step):
        if name == label:
            return value
    return ""


def _operation_table(run_dir: Path, steps: list[Step]) -> list[str]:
    rates = [(_step_rates(run_dir, step)) for step in steps]
    columns: list[str] = []
    for row in rates:
        for name in row:
            if name not in columns:
                columns.append(name)
    if not columns:
        return []
    lines = [
        "",
        "Each cell is the mean calls of that operation per second across the jobs in the step.",
        "",
        "| Step | " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _column in ("Step", *columns)) + " |",
    ]
    for step, row in zip(steps, rates, strict=True):
        cells = [step.id, *(row.get(column, "") for column in columns)]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _step_rates(run_dir: Path, step: Step) -> dict[str, str]:
    rates = _stats(run_dir, step)["operation_rates"]
    if not isinstance(rates, dict):
        return {}
    return {str(key): value for key, value in rates.items() if isinstance(value, str)}


def _details(group: str, run_dir: Path, step: Step) -> list[str]:
    details: list[str] = []
    if group == "barrier":
        details.extend(_workspace_lines(run_dir, step))
    failures = _failures(run_dir, step)
    if failures:
        details.extend(["", f"### {step.id} failures", "", *failures])
    return details


def _stats(run_dir: Path, step: Step) -> dict[str, object]:
    directory = step_dir(run_dir, step.id)
    accepted = 0
    terminal: dict[str, int] = {}
    scheduling = 0
    missing = 0
    partial = False
    fetches: list[float] = []
    achieved: list[float] = []
    operation_rates: dict[str, list[float]] = {}
    shell = step.kind in {"exit", "idle"}
    started_at: list[float] = []
    finished_at: list[float] = []
    completed = 0
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
        if shell:
            if last == "completed":
                completed += 1
                start = _number(payload.get("submitted_at"))
                end = _number(payload.get("finished_at"))
                if start is not None:
                    started_at.append(start)
                if end is not None:
                    finished_at.append(end)
            continue
        if step.kind == "barrier":
            continue
        if payload.get("download_succeeded") is not True:
            missing += 1
            partial = True
        history_file = history_path(run_dir, step.id, name)
        if not history_file.is_file():
            partial = True
            continue
        counts, elapsed = _read_history(history_file, fetches, achieved)
        if elapsed is None and step.duration_seconds is not None:
            elapsed = float(step.duration_seconds)
        if elapsed is not None and elapsed > 0:
            for operation, count in counts.items():
                operation_rates.setdefault(operation, []).append(count / elapsed)
    return {
        "accepted": accepted,
        "terminal": ", ".join(f"{key}={value}" for key, value in sorted(terminal.items())),
        "scheduling": scheduling,
        "missing": missing,
        "partial": partial,
        "fetch": "" if shell else (_mean(fetches) if fetches else ""),
        "achieved": _jobs_per_second(started_at, finished_at, completed) if shell else _ops_per_second(achieved),
        "operation_rates": {operation: _mean(values) for operation, values in operation_rates.items()},
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
        "operation_rates": {},
    }


def _read_history(path: Path, fetches: list[float], achieved: list[float]) -> tuple[dict[str, int], float | None]:
    counts: dict[str, int] = {}
    started: float | None = None
    ended: float | None = None
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
        if isinstance(operation, str):
            counts[operation] = counts.get(operation, 0) + 1
        start = _number(payload.get("started_at"))
        end = _number(payload.get("ended_at"))
        if start is not None and end is not None:
            started = start
            ended = end
    elapsed = ended - started if started is not None and ended is not None and ended > started else None
    return counts, elapsed


def _barrier_results(run_dir: Path, step: Step) -> list[object]:
    reasons = _exit_reasons(run_dir, step)
    cohort = _cohort(run_dir, step)
    released = cohort.get("released")
    finalized = _seconds(cohort.get("release_to_finalized_seconds")) if released is True else "not released"
    if "submit_seconds" not in cohort:
        finalized = ""
    return [
        sum(reasons.values()),
        cohort.get("ready_count", ""),
        cohort.get("peak_active", ""),
        _seconds(cohort.get("submit_seconds")) if "submit_seconds" in cohort else "",
        _seconds(cohort.get("ready_seconds")) if "ready_seconds" in cohort else "",
        finalized,
        reasons["barrier"],
        reasons["timeout"],
        reasons["unknown"],
    ]


def _workspace_lines(run_dir: Path, step: Step) -> list[str]:
    cohort = _cohort(run_dir, step)
    by_workspace = cohort.get("by_workspace")
    if not isinstance(by_workspace, dict) or len(by_workspace) < 2:
        return []
    lines = ["", f"### {step.id} workspaces", ""]
    for name, raw in by_workspace.items():
        if not isinstance(raw, dict):
            continue
        lines.append(
            f"- `{name}`: accepted={raw.get('accepted', 0)} ready={raw.get('ready', 0)} "
            f"peak={raw.get('peak_active', 0)} barrier={raw.get('barrier', 0)} "
            f"timeout={raw.get('timeout', 0)} unknown={raw.get('unknown', 0)}"
        )
    return lines


def _exit_reasons(run_dir: Path, step: Step) -> dict[str, int]:
    reasons = {"barrier": 0, "timeout": 0, "unknown": 0}
    directory = step_dir(run_dir, step.id)
    if not directory.is_dir():
        return reasons
    for path in directory.glob("*.record.json"):
        payload = read_json(path)
        if payload.get("submit") != "accepted":
            continue
        reason = payload.get("exit_reason")
        if reason == "barrier":
            reasons["barrier"] += 1
        elif reason == "timeout":
            reasons["timeout"] += 1
        else:
            reasons["unknown"] += 1
    return reasons


def _cohort(run_dir: Path, step: Step) -> dict[str, object]:
    path = cohort_path(run_dir, step.id)
    if not path.is_file():
        return {}
    return read_json(path)


def _seconds(value: object) -> str:
    number = _number(value)
    if number is None:
        return "unknown"
    return f"{number:.3f}s"


def _failures(run_dir: Path, step: Step) -> list[str]:
    directory = step_dir(run_dir, step.id)
    if not directory.is_dir():
        return []
    lines: list[str] = []
    for path in sorted(directory.glob("*.record.json")):
        payload = read_json(path)
        errors = payload.get("platform_errors")
        if isinstance(errors, list):
            lines.extend(f"- `{payload.get('job_name')}`: {redact(str(item))}" for item in errors)
    return lines


def _chart_row(group: str, run_dir: Path, step: Step) -> tuple[str, float]:
    if group == "barrier":
        value = _cohort(run_dir, step).get("peak_active", 0)
        return step.id, float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0
    return step.id, _leading_float(_stats(run_dir, step)["achieved"])


def _leading_float(value: object) -> float:
    if not isinstance(value, str):
        return 0
    token = value.split(" ", 1)[0]
    try:
        return float(token)
    except ValueError:
        return 0


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


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _jobs_per_second(started: list[float], finished: list[float], completed: int) -> str:
    if completed == 0 or not started or not finished:
        return ""
    span = max(finished) - min(started)
    if span <= 0:
        return ""
    return f"{completed / span:.3f} jobs/s"


def _ops_per_second(values: list[float]) -> str:
    mean = _mean(values)
    if not mean:
        return ""
    return f"{mean} ops/s"


def _mean(values: list[float]) -> str:
    if not values:
        return ""
    return f"{sum(values) / len(values):.3f}"
