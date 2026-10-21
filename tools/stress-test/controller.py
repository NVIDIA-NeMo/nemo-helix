# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit one sweep step at a time and mark it finished only after cleanup."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

from bundle import job_environment
from cluster import Cluster, ClusterWriteError, JobView
from config import Step, Sweep
from names import job_name, run_prefix, step_prefix
from records import marker_matches, marker_path, read_json, record_path, step_dir, write_json

TERMINAL = frozenset({"completed", "error", "cancelled"})
POLL_SECONDS = 2.0


def selected_steps(sweep: Sweep, run_dir: Path, only: str | None) -> list[tuple[Step, bool]]:
    """Steps to execute. ``--step`` replaces that step; a bare run skips finished steps."""
    if only is not None:
        return [(sweep.step(only), True)]
    return [(step, False) for step in sweep.steps if not marker_matches(run_dir, step)]


def execute_steps(
    sweep: Sweep,
    run_dir: Path,
    steps: list[tuple[Step, bool]],
    cluster: Cluster,
    *,
    profile_name: str,
    fileset: str,
    ttl_before: int,
    ttl_active: int,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    poll_seconds: float = POLL_SECONDS,
) -> int:
    """Run steps in order. A step that does not finalize stops the sweep."""
    for step, force in steps:
        finished = execute_step(
            sweep,
            run_dir,
            step,
            cluster,
            force=force,
            profile_name=profile_name,
            fileset=fileset,
            ttl_before=ttl_before,
            ttl_active=ttl_active,
            clock=clock,
            sleep=sleep,
            poll_seconds=poll_seconds,
        )
        if not finished:
            return 1
    return 0


def execute_step(
    sweep: Sweep,
    run_dir: Path,
    step: Step,
    cluster: Cluster,
    *,
    force: bool,
    profile_name: str,
    fileset: str,
    ttl_before: int,
    ttl_active: int,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    poll_seconds: float,
) -> bool:
    """Finalize leftovers, submit the configured count once, then require an empty list."""
    prefix = step_prefix(sweep.run_key, step.id)
    existing = _live(cluster, prefix)
    budget = float(ttl_before + ttl_active)
    if existing:
        if not _finalize(
            run_dir, step, cluster, prefix, [job.name for job in existing], clock, sleep, poll_seconds, budget
        ):
            return False
        if _live(cluster, prefix):
            return False
        if _covers(run_dir, sweep, step) and not force and not _marker_differs(run_dir, step):
            return _commit_marker(run_dir, step, cluster)
        _wipe(run_dir, step.id, cluster)
    elif not force and marker_matches(run_dir, step):
        return True
    else:
        _wipe(run_dir, step.id, cluster)
    if not _submit(
        sweep, run_dir, step, cluster, profile_name, fileset, clock, sleep, poll_seconds, ttl_before, ttl_active
    ):
        return False
    expected = [job_name(sweep.run_key, step.id, index, step.count) for index in range(1, step.count + 1)]
    if not _finalize(run_dir, step, cluster, prefix, expected, clock, sleep, poll_seconds, budget):
        return False
    if _live(cluster, prefix) or not _covers(run_dir, sweep, step):
        return False
    return _commit_marker(run_dir, step, cluster)


def clean_jobs(
    cluster: Cluster,
    prefix: str,
    *,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    budget: float,
    poll_seconds: float = POLL_SECONDS,
) -> bool:
    """Cancel non-terminal jobs, delete them, and fail if any prefix job remains."""
    started = clock()
    while True:
        live = _live(cluster, prefix)
        if not live:
            return True
        for job in live:
            if job.status not in TERMINAL:
                cluster.cancel(job.name)
        live = _live(cluster, prefix)
        if live and all(job.status in TERMINAL for job in live):
            for job in live:
                cluster.delete(job.name)
            return not _live(cluster, prefix)
        if clock() - started >= budget:
            return False
        sleep(poll_seconds)


def clean_results(cluster: Cluster, run_dir: Path, output_dir: Path) -> None:
    """Delete this run's fileset objects, local bundle, and report directory."""
    cluster.remove_prefix("")
    shutil.rmtree(run_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)


def run_job_prefix(run_key: str) -> str:
    return run_prefix(run_key)


def _live(cluster: Cluster, prefix: str) -> list[JobView]:
    return [job for job in cluster.jobs(prefix) if job.name.startswith(prefix)]


def _marker_differs(run_dir: Path, step: Step) -> bool:
    path = marker_path(run_dir, step.id)
    return path.is_file() and not marker_matches(run_dir, step)


def _covers(run_dir: Path, sweep: Sweep, step: Step) -> bool:
    expected = {job_name(sweep.run_key, step.id, index, step.count) for index in range(1, step.count + 1)}
    accepted: set[str] = set()
    directory = step_dir(run_dir, step.id)
    if not directory.is_dir():
        return False
    for path in directory.glob("*.record.json"):
        payload = read_json(path)
        name = payload.get("job_name")
        if payload.get("submit") == "accepted" and isinstance(name, str):
            accepted.add(name)
    return expected <= accepted


def _wipe(run_dir: Path, step_id: str, cluster: Cluster) -> None:
    shutil.rmtree(step_dir(run_dir, step_id), ignore_errors=True)
    cluster.remove_prefix(f"steps/{step_id}/")


def _record(run_dir: Path, step_id: str, name: str, clock: Callable[[], float]) -> dict[str, object]:
    path = record_path(run_dir, step_id, name)
    if path.is_file():
        return read_json(path)
    payload: dict[str, object] = {
        "job_name": name,
        "submit": "accepted",
        "status_history": [],
        "terminal": False,
        "result_registered": False,
        "download_succeeded": False,
        "fileset_copy_succeeded": False,
        "submitted_at": clock(),
    }
    write_json(path, payload)
    return payload


def _save(run_dir: Path, step_id: str, name: str, payload: dict[str, object]) -> None:
    write_json(record_path(run_dir, step_id, name), payload)


def _submitted_at(payload: dict[str, object], fallback: float) -> float:
    value = payload.get("submitted_at")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return fallback


def _note(payload: dict[str, object], status: str) -> None:
    raw = payload.get("status_history")
    history = [item for item in raw if isinstance(item, str)] if isinstance(raw, list) else []
    if not history or history[-1] != status:
        history.append(status)
    payload["status_history"] = history


def _history(payload: dict[str, object]) -> list[str]:
    raw = payload.get("status_history")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str)]


def _wait(
    run_dir: Path,
    step: Step,
    cluster: Cluster,
    prefix: str,
    names: list[str],
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    poll_seconds: float,
    budget: float,
) -> bool:
    wanted = set(names)
    while True:
        live = {job.name: job for job in _live(cluster, prefix) if job.name in wanted}
        pending = False
        now = clock()
        for name in names:
            payload = _record(run_dir, step.id, name, clock)
            job = live.get(name)
            if job is None:
                seen_terminal = any(status in TERMINAL for status in _history(payload))
                if not seen_terminal:
                    pending = True
                    if now >= _submitted_at(payload, now) + budget:
                        _save(run_dir, step.id, name, payload)
                        return False
                _save(run_dir, step.id, name, payload)
                continue
            _note(payload, job.status)
            _save(run_dir, step.id, name, payload)
            if job.status not in TERMINAL:
                pending = True
                if now >= _submitted_at(payload, now) + budget:
                    return False
        if not pending:
            return True
        sleep(poll_seconds)


def _finalize(
    run_dir: Path,
    step: Step,
    cluster: Cluster,
    prefix: str,
    names: list[str],
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    poll_seconds: float,
    budget: float,
) -> bool:
    if not _wait(run_dir, step, cluster, prefix, names, clock, sleep, poll_seconds, budget):
        return False
    live = {job.name: job for job in _live(cluster, prefix) if job.name in set(names)}
    for name in names:
        payload = _record(run_dir, step.id, name, clock)
        job = live.get(name)
        body = cluster.read_result(name) if job is not None and job.status in TERMINAL else None
        payload["terminal"] = job is not None and job.status in TERMINAL
        payload["result_registered"] = body is not None
        if body is None:
            payload["download_succeeded"] = False
            payload["fileset_copy_succeeded"] = False
        else:
            history = step_dir(run_dir, step.id) / f"{name}.jsonl"
            history.parent.mkdir(parents=True, exist_ok=True)
            history.write_bytes(body)
            payload["download_succeeded"] = True
            try:
                cluster.write_file(f"steps/{step.id}/{name}.jsonl", body)
                payload["fileset_copy_succeeded"] = True
            except ClusterWriteError:
                payload["fileset_copy_succeeded"] = False
        _save(run_dir, step.id, name, payload)
        if job is not None and job.status in TERMINAL:
            cluster.delete(name)
    return True


def _submit(
    sweep: Sweep,
    run_dir: Path,
    step: Step,
    cluster: Cluster,
    profile_name: str,
    fileset: str,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    poll_seconds: float,
    ttl_before: int,
    ttl_active: int,
) -> bool:
    environment = job_environment(sweep, step, fileset)
    pod = sweep.pods[step.kind]
    prefix = step_prefix(sweep.run_key, step.id)
    accepted: list[str] = []
    budget = float(ttl_before + ttl_active)
    for index in range(1, step.count + 1):
        name = job_name(sweep.run_key, step.id, index, step.count)
        ok = cluster.create_job(name, environment, pod, profile_name)
        payload: dict[str, object] = {
            "job_name": name,
            "submit": "accepted" if ok else "rejected",
            "status_history": [],
            "terminal": False,
            "result_registered": False,
            "download_succeeded": False,
            "fileset_copy_succeeded": False,
            "submitted_at": clock(),
        }
        _save(run_dir, step.id, name, payload)
        if not ok:
            if accepted:
                _finalize(run_dir, step, cluster, prefix, accepted, clock, sleep, poll_seconds, budget)
            return False
        accepted.append(name)
    return True


def _commit_marker(run_dir: Path, step: Step, cluster: Cluster) -> bool:
    body = dict(step.measurement())
    try:
        encoded = json.dumps(body, indent=2, sort_keys=True).encode() + b"\n"
        cluster.write_file(f"steps/{step.id}/complete.json", encoded)
    except ClusterWriteError:
        return False
    write_json(marker_path(run_dir, step.id), body)
    return True
