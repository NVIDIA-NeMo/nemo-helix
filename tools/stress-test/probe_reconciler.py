# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only probe of how often the jobs reconciler writes each step.

``watch`` polls job status for every job in a workspace whose name starts with a prefix and
appends one JSON line per job per poll. ``analyze`` reads those lines and reports, per job, when
Helix changed the step and its tasks, the longest gap between step writes while the step was not
terminal, and global stretches where no step was written at all.

The reconciler writes a step only when something changed, and the pending TTL runs from
``max(created_at, updated_at)``. A step that sits pending while its task records keep changing
points at a sync that fails after the task upsert. A step whose task records also stop changing
points at a reconciler that did not reach it.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.jobs.client import JobsClient

TERMINAL = {"completed", "error", "cancelled"}


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _iso(value: object) -> str | None:
    if isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=datetime.timezone.utc)
        return value.isoformat()
    return None


def _status(value: object) -> str:
    return str(getattr(value, "value", value))


def _snapshot(jobs: JobsClient, workspace: str, name: str) -> dict[str, object]:
    record: dict[str, object] = {"t": _now(), "job": name}
    try:
        status = jobs.get_job_status(workspace=workspace, name=name).data()
    except Exception as exc:  # noqa: BLE001 - one failed read is one sample
        record["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return record
    record["job_status"] = _status(status.status)
    record["job_created_at"] = _iso(status.created_at)
    steps = []
    for step in status.steps:
        tasks = []
        for task in step.tasks:
            pod = (task.status_details or {}).get("pod") or {}
            containers = pod.get("containers") or []
            terminated = containers[0].get("terminated") if containers and isinstance(containers[0], dict) else None
            tasks.append(
                {
                    "status": _status(task.status),
                    "updated_at": _iso(task.updated_at),
                    "phase": pod.get("phase"),
                    "exit_code": terminated.get("exit_code") if isinstance(terminated, dict) else None,
                }
            )
        steps.append(
            {
                "status": _status(step.status),
                "created_at": _iso(step.created_at),
                "updated_at": _iso(step.updated_at),
                "message": (step.status_details or {}).get("message"),
                "error": (step.error_details or {}).get("message"),
                "tasks": tasks,
            }
        )
    record["steps"] = steps
    return record


def watch(args: argparse.Namespace) -> int:
    jobs = JobsClient.from_client(NemoClient.from_config(context=args.context))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    finished: set[str] = set()
    idle_since: float | None = None
    with out.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        while time.monotonic() - started < args.max_seconds:
            began = time.monotonic()
            try:
                names = sorted(
                    job.name
                    for job in jobs.list_jobs(workspace=args.workspace).items()
                    if job.name.startswith(args.prefix)
                )
            except Exception as exc:  # noqa: BLE001
                handle.write(json.dumps({"t": _now(), "list_error": f"{type(exc).__name__}: {str(exc)[:200]}"}) + "\n")
                handle.flush()
                time.sleep(args.interval)
                continue
            live = [name for name in names if name not in finished]
            for record in pool.map(lambda name: _snapshot(jobs, args.workspace, name), live):
                handle.write(json.dumps(record) + "\n")
                if record.get("job_status") in TERMINAL:
                    finished.add(str(record["job"]))
            handle.write(json.dumps({"t": _now(), "poll": len(live), "seen": len(names)}) + "\n")
            handle.flush()
            if len(finished) >= args.expect and not live:
                break
            if names and not live:
                idle_since = idle_since or time.monotonic()
                if time.monotonic() - idle_since > args.idle_seconds:
                    break
            else:
                idle_since = None
            remaining = args.interval - (time.monotonic() - began)
            if remaining > 0:
                time.sleep(remaining)
    print(f"watched {len(finished)} terminal job(s); records in {out}", file=sys.stderr)
    return 0


def _parse(value: object) -> datetime.datetime | None:
    if not isinstance(value, str):
        return None
    return datetime.datetime.fromisoformat(value)


def analyze(args: argparse.Namespace) -> int:
    per_job: dict[str, list[dict[str, Any]]] = {}
    for line in Path(args.out).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "job" in record and "steps" in record:
            per_job.setdefault(str(record["job"]), []).append(record)
    step_writes: set[datetime.datetime] = set()
    rows = []
    for name in sorted(per_job):
        records = per_job[name]
        step_times: list[datetime.datetime] = []
        task_times: list[datetime.datetime] = []
        statuses: list[str] = []
        first_exited: str | None = None
        first_terminal: str | None = None
        created = None
        last = records[-1]
        for record in records:
            for step in record["steps"]:
                created = created or _parse(step["created_at"])
                updated = _parse(step["updated_at"])
                if updated and updated not in step_times:
                    step_times.append(updated)
                if not statuses or statuses[-1] != step["status"]:
                    statuses.append(step["status"])
                if first_terminal is None and step["status"] in TERMINAL:
                    first_terminal = str(record["t"])
                for task in step["tasks"]:
                    task_updated = _parse(task["updated_at"])
                    if task_updated and task_updated not in task_times:
                        task_times.append(task_updated)
                    if first_exited is None and (
                        task["exit_code"] is not None or task["phase"] in ("Succeeded", "Failed")
                    ):
                        first_exited = str(record["t"])
        step_writes.update(step_times)
        ordered = sorted(step_times)
        gaps = [(b - a).total_seconds() for a, b in zip(ordered, ordered[1:], strict=False)]
        final_step = (last.get("steps") or [{}])[-1]
        rows.append(
            {
                "job": name,
                "statuses": " > ".join(statuses),
                "step_writes": len(ordered),
                "task_writes": len(task_times),
                "max_step_gap_s": round(max(gaps), 1) if gaps else None,
                "created": created.isoformat() if created else None,
                "first_seen_exited": first_exited,
                "first_seen_terminal": first_terminal,
                "final_error": final_step.get("error") if isinstance(final_step, dict) else None,
            }
        )
    for row in rows:
        print(json.dumps(row))
    ordered_writes = sorted(step_writes)
    quiet = sorted(
        ((b - a).total_seconds(), a.isoformat(), b.isoformat())
        for a, b in zip(ordered_writes, ordered_writes[1:], strict=False)
    )[-5:]
    print(json.dumps({"step_writes_total": len(ordered_writes), "longest_quiet_stretches_s": quiet}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    watch_parser = commands.add_parser("watch")
    watch_parser.add_argument("--context", required=True)
    watch_parser.add_argument("--workspace", required=True)
    watch_parser.add_argument("--prefix", required=True)
    watch_parser.add_argument("--out", required=True)
    watch_parser.add_argument("--expect", type=int, required=True, help="Stop after this many jobs are terminal.")
    watch_parser.add_argument("--interval", type=float, default=5.0)
    watch_parser.add_argument("--workers", type=int, default=8)
    watch_parser.add_argument("--max-seconds", type=float, default=3 * 3600)
    watch_parser.add_argument("--idle-seconds", type=float, default=600)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--out", required=True)
    args = parser.parse_args()
    return watch(args) if args.command == "watch" else analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
