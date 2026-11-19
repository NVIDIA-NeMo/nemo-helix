# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit one sweep step at a time and mark it finished only after cleanup."""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Callable

from bundle import barrier_exit_path, barrier_ready_path, barrier_release_path, job_environment
from cluster import Cluster, ClusterWriteError, JobView
from config import Step, Sweep, _step_line
from names import job_name, run_prefix, step_prefix, workspace_names
from records import cohort_path, marker_matches, marker_path, read_json, record_path, step_dir, write_json

TERMINAL = frozenset({"completed", "error", "cancelled"})
POLL_SECONDS = 2.0
log = logging.getLogger("stress")


def selected_steps(sweep: Sweep, run_dir: Path, only: str | None, *, force: bool = False) -> list[tuple[Step, bool]]:
    """Steps to execute. ``--step`` replaces that step; a bare run skips finished steps."""
    if only is not None:
        return [(sweep.step(only), True)]
    if force:
        return [(step, True) for step in sweep.steps]
    return [(step, False) for step in sweep.steps if not marker_matches(run_dir, step)]


def execute_steps(
    sweep: Sweep,
    run_dir: Path,
    steps: list[tuple[Step, bool]],
    cluster: Cluster,
    *,
    profile_name: str,
    provider: str,
    fileset: str,
    ttl_before: int,
    ttl_active: int,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    poll_seconds: float = POLL_SECONDS,
    image: str | None = None,
    clusters: list[Cluster] | None = None,
    workspaces: list[str] | None = None,
    workspace_clusters: Mapping[str, Cluster] | None = None,
) -> int:
    """Run steps in order. A step that does not finalize stops the sweep."""
    total = len(steps)
    if total:
        log.info("running %d step(s): %s", total, ", ".join(step.id for step, _force in steps))
    for index, (step, force) in enumerate(steps, start=1):
        log.info("preparing step %d/%d %s", index, total, _step_line(step))
        finished = execute_step(
            sweep,
            run_dir,
            step,
            cluster,
            force=force,
            profile_name=profile_name,
            provider=provider,
            fileset=fileset,
            ttl_before=ttl_before,
            ttl_active=ttl_active,
            clock=clock,
            sleep=sleep,
            poll_seconds=poll_seconds,
            image=image,
            clusters=clusters,
            workspaces=workspaces,
            workspace_clusters=workspace_clusters,
        )
        log.info("step %s: %s", step.id, step_summary(run_dir, step.id))
        if not finished:
            log.error("error %s: did not finish; rerun to resume", step.id)
            return 1
    return 0


def step_summary(run_dir: Path, step_id: str) -> str:
    """One line of counts from the local records for a step."""
    counts: dict[str, int] = {}
    results = 0
    directory = step_dir(run_dir, step_id)
    records = sorted(directory.glob("*.record.json")) if directory.is_dir() else []
    if not records:
        return "not run"
    for path in records:
        payload = read_json(path)
        if payload.get("submit") != "accepted":
            counts["refused"] = counts.get("refused", 0) + 1
            continue
        history = _history(payload)
        last = history[-1] if history else "unseen"
        counts[last] = counts.get(last, 0) + 1
        if payload.get("result_registered") is True:
            results += 1
    parts = [f"{len(records)} submitted"] + [f"{count} {status}" for status, count in sorted(counts.items())]
    parts.append(f"{results} with results")
    complete = "marker written" if marker_path(run_dir, step_id).is_file() else "no marker"
    return ", ".join(parts) + f"; {complete}"


def execute_step(
    sweep: Sweep,
    run_dir: Path,
    step: Step,
    cluster: Cluster,
    *,
    force: bool,
    profile_name: str,
    provider: str,
    fileset: str,
    ttl_before: int,
    ttl_active: int,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    poll_seconds: float,
    image: str | None = None,
    clusters: list[Cluster] | None = None,
    workspaces: list[str] | None = None,
    workspace_clusters: Mapping[str, Cluster] | None = None,
) -> bool:
    """Finalize leftovers, submit the configured count once, then require an empty list."""
    owners = _owners(sweep, step, cluster, fileset, clusters, workspaces, workspace_clusters)
    spaces = list(owners)
    group = list(owners.values())
    prefix = step_prefix(sweep.run_key, step.id)
    existing = _live_all(group, prefix)
    budget = float(ttl_before + ttl_active)
    workspace_count = len(spaces)
    if existing:
        log.info("%s: finalizing %d job(s) left from an earlier run", step.id, len(existing))
        if not _finalize(
            run_dir,
            step,
            cluster,
            prefix,
            [job.name for job in existing],
            clock,
            sleep,
            poll_seconds,
            budget,
            owners=owners,
        ):
            return False
        if _live_all(group, prefix):
            return False
        if (
            _covers(run_dir, sweep, step, workspace_count)
            and not force
            and not _marker_differs(run_dir, step, workspace_count)
        ):
            return _commit_marker(run_dir, step, group, workspace_count)
        _wipe_all(run_dir, step.id, group)
    elif not force and marker_matches(run_dir, step, workspaces=workspace_count):
        log.debug("%s: already complete; use --step %s to rerun", step.id, step.id)
        return True
    else:
        _wipe_all(run_dir, step.id, group)
    submitted_at = clock()
    if not _submit(
        sweep,
        run_dir,
        step,
        owners,
        profile_name,
        provider,
        clock,
        sleep,
        poll_seconds,
        ttl_before,
        ttl_active,
        image,
    ):
        return False
    submit_seconds = clock() - submitted_at
    release: dict[str, object] = {}
    released_at: float | None = None
    if step.kind == "barrier":
        release = _release_barrier(
            step,
            _assignments(sweep, step, spaces),
            owners,
            prefix=prefix,
            started=submitted_at,
            clock=clock,
            sleep=sleep,
            poll_seconds=poll_seconds,
        )
        if release.get("released") is True:
            released_at = clock()
            total_jobs = step.count * workspace_count
            log.info("waiting for %d job(s) to exit after the barrier in %s", total_jobs, step.id)
    expected = _expected_names(sweep, step, workspace_count)
    if not _finalize(run_dir, step, cluster, prefix, expected, clock, sleep, poll_seconds, budget, owners=owners):
        return False
    if step.kind == "barrier":
        _store_cohort(
            run_dir,
            step,
            group,
            spaces,
            release,
            submit_seconds=submit_seconds,
            release_to_finalized_seconds=None if released_at is None else clock() - released_at,
        )
    if _live_all(group, prefix) or not _covers(run_dir, sweep, step, workspace_count):
        return False
    return _commit_marker(run_dir, step, group, workspace_count)


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


def _marker_differs(run_dir: Path, step: Step, workspaces: int) -> bool:
    path = marker_path(run_dir, step.id)
    return path.is_file() and not marker_matches(run_dir, step, workspaces=workspaces)


def _expected_names(sweep: Sweep, step: Step, workspaces: int) -> list[str]:
    total = step.count * workspaces
    return [job_name(sweep.run_key, step.id, index, total) for index in range(1, total + 1)]


def _assignments(sweep: Sweep, step: Step, spaces: list[str]) -> list[tuple[str, str]]:
    total = step.count * len(spaces)
    return [
        (job_name(sweep.run_key, step.id, index, total), spaces[(index - 1) % len(spaces)])
        for index in range(1, total + 1)
    ]


def _covers(run_dir: Path, sweep: Sweep, step: Step, workspaces: int) -> bool:
    expected = set(_expected_names(sweep, step, workspaces))
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


def _wipe_all(run_dir: Path, step_id: str, clusters: list[Cluster]) -> None:
    shutil.rmtree(step_dir(run_dir, step_id), ignore_errors=True)
    for cluster in clusters:
        cluster.remove_prefix(f"steps/{step_id}/")


def _live_all(clusters: list[Cluster], prefix: str) -> list[JobView]:
    found: list[JobView] = []
    for cluster in clusters:
        found.extend(_live(cluster, prefix))
    return found


def _owners(
    sweep: Sweep,
    step: Step,
    cluster: Cluster,
    fileset: str,
    clusters: list[Cluster] | None,
    workspaces: list[str] | None,
    workspace_clusters: Mapping[str, Cluster] | None,
) -> dict[str, Cluster]:
    if workspaces is not None:
        group = clusters if clusters is not None else [cluster]
        return {space: item for space, item in zip(workspaces, group, strict=True)}
    if workspace_clusters is not None:
        names = workspace_names(sweep.run_key, step.workspaces)
        return {name: workspace_clusters[name] for name in names}
    return {fileset: cluster}


def _owner(owners: dict[str, Cluster], workspace: object, fallback: Cluster) -> Cluster:
    if isinstance(workspace, str) and workspace in owners:
        return owners[workspace]
    return fallback


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
        log.debug("%s: %s", payload.get("job_name"), status)
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
    owners: dict[str, Cluster] | None = None,
) -> bool:
    wanted = set(names)
    group = list(owners.values()) if owners is not None else [cluster]
    while True:
        live = {job.name: job for job in _live_all(group, prefix) if job.name in wanted}
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
                        log.error("error %s: not listed within %.0fs", name, budget)
                        return False
                _save(run_dir, step.id, name, payload)
                continue
            _note(payload, job.status)
            _save(run_dir, step.id, name, payload)
            if job.status not in TERMINAL:
                pending = True
                if now >= _submitted_at(payload, now) + budget:
                    log.error("error %s: still %s after %.0fs", name, job.status, budget)
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
    owners: dict[str, Cluster] | None = None,
) -> bool:
    if not _wait(run_dir, step, cluster, prefix, names, clock, sleep, poll_seconds, budget, owners):
        return False
    group = list(owners.values()) if owners is not None else [cluster]
    live = {job.name: job for job in _live_all(group, prefix) if job.name in set(names)}
    for name in names:
        payload = _record(run_dir, step.id, name, clock)
        job = live.get(name)
        owner = _owner(owners or {}, payload.get("workspace"), cluster)
        shell = step.kind in {"exit", "idle"}
        if step.kind == "barrier":
            _record_barrier_exit(payload, owner, step.id, name, job, clock)
            _save(run_dir, step.id, name, payload)
            if job is not None and job.status in TERMINAL:
                owner.delete(name)
            continue
        body = owner.read_result(name) if not shell and job is not None and job.status in TERMINAL else None
        payload["terminal"] = job is not None and job.status in TERMINAL
        payload["result_registered"] = body is not None
        if shell and job is not None and job.status == "completed":
            payload["download_succeeded"] = False
            payload["fileset_copy_succeeded"] = False
            payload["finished_at"] = clock()
            log.info("stopped %s", name)
        elif body is None:
            payload["download_succeeded"] = False
            payload["fileset_copy_succeeded"] = False
            if job is not None:
                _explain(payload, owner.diagnose(name), job.status)
        else:
            history = step_dir(run_dir, step.id) / f"{name}.jsonl"
            history.parent.mkdir(parents=True, exist_ok=True)
            history.write_bytes(body)
            payload["download_succeeded"] = True
            try:
                owner.write_file(f"steps/{step.id}/{name}.jsonl", body)
                payload["fileset_copy_succeeded"] = True
            except ClusterWriteError:
                payload["fileset_copy_succeeded"] = False
            if payload["fileset_copy_succeeded"]:
                log.info("stopped %s", name)
            else:
                log.error("error %s: fileset copy failed", name)
            log.debug("%s: result saved", name)
        _save(run_dir, step.id, name, payload)
        if job is not None and job.status in TERMINAL:
            owner.delete(name)
    return True


def _explain(payload: dict[str, object], found: dict[str, object], status: str) -> None:
    raw_errors = found.get("errors")
    raw_tail = found.get("log_tail")
    errors = [str(item) for item in raw_errors] if isinstance(raw_errors, list) else []
    tail = [str(item).rstrip() for item in raw_tail] if isinstance(raw_tail, list) else []
    payload["platform_errors"] = errors
    payload["log_tail"] = tail
    reason = "; ".join(errors) if errors else f"{status}, no service-stress result"
    log.error("error %s: %s", payload.get("job_name"), reason)
    for line in tail:
        log.debug("%s: %s", payload.get("job_name"), line)


def _submit(
    sweep: Sweep,
    run_dir: Path,
    step: Step,
    owners: dict[str, Cluster],
    profile_name: str,
    provider: str,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    poll_seconds: float,
    ttl_before: int,
    ttl_active: int,
    image: str | None,
) -> bool:
    pod = sweep.pods[step.kind]
    prefix = step_prefix(sweep.run_key, step.id)
    accepted: list[str] = []
    budget = float(ttl_before + ttl_active)
    assignments = _assignments(sweep, step, list(owners))
    target = f" on {provider}/{profile_name}"
    across = f" across {len(owners)} workspaces" if len(owners) > 1 else ""
    log.info("submitting %d job(s) for %s%s%s", len(assignments), step.id, target, across)
    fallback = next(iter(owners.values()))
    for name, space in assignments:
        environment = job_environment(sweep, step, space)
        owner = owners[space]
        ok = owner.create_job(name, environment, pod, profile_name, provider, image)
        payload: dict[str, object] = {
            "job_name": name,
            "workspace": space,
            "submit": "accepted" if ok else "rejected",
            "status_history": [],
            "terminal": False,
            "result_registered": False,
            "download_succeeded": False,
            "fileset_copy_succeeded": False,
            "submitted_at": clock(),
        }
        _save(run_dir, step.id, name, payload)
        if ok and len(owners) == 1:
            log.info("started %s", name)
        elif ok:
            log.info("started %s in %s", name, space)
        if not ok:
            if accepted:
                _finalize(run_dir, step, fallback, prefix, accepted, clock, sleep, poll_seconds, budget, owners=owners)
            return False
        accepted.append(name)
    if step.kind == "barrier":
        log.info("submitted %d job(s) for %s; waiting for them to reach the barrier", len(accepted), step.id)
    else:
        log.info("waiting for %d job(s) in %s", len(accepted), step.id)
    return True


def _commit_marker(run_dir: Path, step: Step, clusters: list[Cluster], workspaces: int) -> bool:
    body = dict(step.measurement())
    body["workspaces"] = workspaces
    encoded = json.dumps(body, indent=2, sort_keys=True).encode() + b"\n"
    try:
        for cluster in clusters:
            cluster.write_file(f"steps/{step.id}/complete.json", encoded)
    except ClusterWriteError:
        return False
    write_json(marker_path(run_dir, step.id), body)
    return True


def _record_barrier_exit(
    payload: dict[str, object], owner: Cluster, step_id: str, name: str, job: JobView | None, clock: Callable[[], float]
) -> None:
    raw = owner.read_file(barrier_exit_path(step_id, name))
    text = raw.decode().strip() if isinstance(raw, bytes) else ""
    reason = text if text in {"barrier", "timeout"} else "unknown"
    payload["exit_reason"] = reason
    payload["terminal"] = job is not None and job.status in TERMINAL
    if job is not None and job.status in TERMINAL:
        payload["finished_at"] = clock()
    if reason == "barrier" and job is not None and job.status == "completed":
        log.info("stopped %s: barrier", name)
        return
    if reason == "timeout":
        log.error("error %s: timed out waiting for the barrier", name)
        return
    status = job.status if job is not None else "missing"
    log.error("error %s: exit %s (%s)", name, reason, status)


def _release_barrier(
    step: Step,
    assignments: list[tuple[str, str]],
    owners: dict[str, Cluster],
    *,
    prefix: str,
    started: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    poll_seconds: float,
) -> dict[str, object]:
    timeout = float(step.timeout_seconds or 0)
    deadline = started + timeout
    names_by_space: dict[str, list[str]] = {}
    for name, space in assignments:
        names_by_space.setdefault(space, []).append(name)
    peak = 0
    peak_by_space = {space: 0 for space in names_by_space}
    ready_by_space = {space: 0 for space in names_by_space}
    ready_total = 0
    released = False
    while True:
        ready_total = 0
        active_total = 0
        for space, names in names_by_space.items():
            owner = owners[space]
            ready = sum(1 for name in names if owner.read_file(barrier_ready_path(step.id, name)) is not None)
            ready_by_space[space] = ready
            ready_total += ready
            live = {job.name: job.status for job in _live(owner, prefix)}
            active = sum(1 for name in names if live.get(name) == "active")
            peak_by_space[space] = max(peak_by_space[space], active)
            active_total += active
        peak = max(peak, active_total)
        if ready_total == len(assignments):
            for space in names_by_space:
                owners[space].write_file(barrier_release_path(step.id), b"release\n")
            released = True
            log.info("barrier %s: %d/%d ready; releasing", step.id, ready_total, len(assignments))
            break
        if clock() >= deadline:
            log.error(
                "barrier %s: %d/%d ready after %.0fs; not releasing",
                step.id,
                ready_total,
                len(assignments),
                timeout,
            )
            break
        sleep(poll_seconds)
    return {
        "ready_count": ready_total,
        "ready_seconds": clock() - started,
        "released": released,
        "peak_active": peak,
        "ready_by_workspace": ready_by_space,
        "peak_by_workspace": peak_by_space,
    }


def _store_cohort(
    run_dir: Path,
    step: Step,
    clusters: list[Cluster],
    spaces: list[str],
    release: dict[str, object],
    *,
    submit_seconds: float,
    release_to_finalized_seconds: float | None,
) -> None:
    ready_by = release.get("ready_by_workspace")
    peak_by = release.get("peak_by_workspace")
    ready_map = ready_by if isinstance(ready_by, dict) else {}
    peak_map = peak_by if isinstance(peak_by, dict) else {}
    by_workspace: dict[str, dict[str, int]] = {}
    for space in spaces:
        ready = ready_map.get(space, 0)
        peak = peak_map.get(space, 0)
        by_workspace[space] = {
            "accepted": 0,
            "ready": ready if isinstance(ready, int) else 0,
            "peak_active": peak if isinstance(peak, int) else 0,
            "barrier": 0,
            "timeout": 0,
            "unknown": 0,
        }
    directory = step_dir(run_dir, step.id)
    if directory.is_dir():
        for path in directory.glob("*.record.json"):
            payload = read_json(path)
            space = payload.get("workspace")
            if not isinstance(space, str) or space not in by_workspace or payload.get("submit") != "accepted":
                continue
            bucket = by_workspace[space]
            bucket["accepted"] += 1
            reason = payload.get("exit_reason")
            if reason == "barrier":
                bucket["barrier"] += 1
            elif reason == "timeout":
                bucket["timeout"] += 1
            else:
                bucket["unknown"] += 1
    barrier = sum(item["barrier"] for item in by_workspace.values())
    timeout = sum(item["timeout"] for item in by_workspace.values())
    unknown = sum(item["unknown"] for item in by_workspace.values())
    body: dict[str, object] = {
        "workspaces": spaces,
        "jobs_per_workspace": step.count,
        "submit_seconds": submit_seconds,
        "ready_count": release.get("ready_count", 0),
        "ready_seconds": release.get("ready_seconds", 0),
        "released": release.get("released") is True,
        "peak_active": release.get("peak_active", 0),
        "release_to_finalized_seconds": release_to_finalized_seconds,
        "barrier": barrier,
        "timeout": timeout,
        "unknown": unknown,
        "by_workspace": by_workspace,
    }
    write_json(cohort_path(run_dir, step.id), body)
    encoded = json.dumps(body, indent=2, sort_keys=True).encode() + b"\n"
    for cluster in clusters:
        try:
            cluster.write_file(f"steps/{step.id}/cohort.json", encoded)
        except ClusterWriteError:
            log.error("error %s: cohort copy failed", step.id)
    log.info("barrier %s: barrier=%d timeout=%d unknown=%d", step.id, barrier, timeout, unknown)
