# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Snapshot the platform's evaluation state as portable export bundles.

Snapshot (capture) is a read-API export: :func:`snapshot_export` drains each
subject's workspaces through :mod:`evaluation.export` into a JSONL bundle
(``kind: evaluation-export``) — it works against any platform URL and never
touches ClickHouse directly.

Restore lives in :mod:`evaluation.reingest`: an additive, idempotent re-ingest
into fixture-scoped workspaces through the real APIs. Legacy tar bundles
(state-v1..v5) are restorable only from a pre-migration checkout.
"""

import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

import httpx
from evaluation import export
from evaluation.intake_client import build_basic_auth_intake_client
from evaluation.registry import Subject
from nemo_helix import AsyncNeMoHelix

LOCAL_URL = "http://localhost:8080"  # the local NeMo Helix (the default restore/analyze target)
EXPORT_KIND = "evaluation-export"
LEGACY_EXPORT_KIND = "testbed-export"
EXPORT_KINDS = frozenset({EXPORT_KIND, LEGACY_EXPORT_KIND})


def is_export_manifest(manifest: object) -> bool:
    """Return whether a manifest is a current or pre-rename export bundle."""
    return isinstance(manifest, Mapping) and manifest.get("kind") in EXPORT_KINDS


def pick_records(tmp_dir: Path, subjects: list[str]) -> list[Path]:
    """The selected *subjects*' run records present in *tmp_dir* (sorted).

    Scoped by name (``<subject>.run.json``) so a bundle never captures another
    subject's records that happen to sit in the shared ``evaluation/tmp``
    directory. Insight YAMLs never travel in bundles: insights are per-analyze
    output, not state (``analyze`` is fresh by default; ``--update-insights``
    seeds the analyst from the local file).
    """
    names = {f"{s}.run.json" for s in subjects}
    return sorted(p for p in (tmp_dir / n for n in names) if p.is_file())


# manifest key -> env var it is sourced from (CI produce job; parity with the old bundle.py)
_LINEAGE_ENV = {
    "nemo_helix_sha": "GITHUB_SHA",
    "tau2_bench_sha": "TAU2_BENCH_REF",
    "github_run_id": "GITHUB_RUN_ID",
    "num_tasks": "NUM_TASKS",
    "num_trials": "NUM_TRIALS",
    "judge_llm": "TAU2_JUDGE_LLM",
    "reason": "REASON",
}


def build_export_manifest(
    subjects: list[str],
    records: list[Path],
    stats: dict,
    *,
    source_url: str,
    platform_info: dict | None,
    env: Mapping[str, str] | None = None,
) -> dict:
    """Export-bundle manifest; CI lineage fields are merged in only when their env vars are set.

    *stats* is :func:`evaluation.export.export_workspaces`'s return value (per-collection
    counts + span time bounds). *platform_info* is :func:`fetch_platform_info`'s
    best-effort answer (None when the platform didn't say). Laptop snapshots (no CI
    env) keep the lite manifest.
    """
    manifest = {
        "kind": EXPORT_KIND,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_url": source_url,
        "subjects": sorted(subjects),
        "workspaces": sorted(stats["workspaces"]),
        "counts": stats["workspaces"],
        "min_start_time": stats["min_start_time"],
        "max_start_time": stats["max_start_time"],
        "platform_version": (platform_info or {}).get("platform_version"),
        "platform_revision": (platform_info or {}).get("revision"),
        "records": [r.name for r in records],
    }
    if stats.get("selections"):
        manifest["selections"] = stats["selections"]
    env = env or {}
    manifest |= {key: env[var] for key, var in _LINEAGE_ENV.items() if env.get(var)}
    return manifest


def fetch_platform_info(base_url: str) -> dict | None:
    """Best-effort platform identity from ``GET /cluster-info`` (None when it can't say).

    The platform serves ``{"platform_version": ..., "revision": ...}`` on an
    unauthenticated route; recording it pins which platform build minted the
    bundle (a later task's attribute-catalog inversion keys off it).
    """
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/cluster-info", timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        return {"platform_version": data.get("platform_version"), "revision": data.get("revision")}
    except Exception:
        return None


def workspaces_for_subject(subject: Subject) -> list[str]:
    """The intake workspaces a subject's export must capture.

    Benchmark subjects own their realistic workspace plus the ``-oracle`` twin
    (the answer key) unless ``include_rewards`` is off; intake subjects own just
    their workspace. Any other type has no owned-workspace scoping rule, so it
    is a hard error.
    """
    workspace = str(subject.config.get("workspace") or "")
    if not workspace:
        sys.exit(f"snapshot: subject '{subject.name}' has no workspace configured")
    if subject.type == "benchmark":
        if subject.config.get("include_rewards", True):
            return [workspace, f"{workspace}-oracle"]
        return [workspace]
    if subject.type == "intake":
        return [workspace]
    sys.exit(
        f"snapshot: subject '{subject.name}' has type '{subject.type}' — only 'benchmark' and "
        "'intake' subjects own intake workspaces to export (this type has no workspace-scoping rule)"
    )


def _basic_auth_intake_client_for(subject: Subject, source_url: str) -> AsyncNeMoHelix | None:
    """Build this subject's configured basic-auth Intake client, if needed."""
    if subject.config.get("auth") != "basic":
        return None
    config = subject.config
    credentials: dict[str, str] = {}
    for role, key in (("username", "auth_user_env"), ("password", "auth_password_env")):
        env_name = config.get(key)
        value = os.environ.get(str(env_name)) if env_name else None
        if not value:
            env_label = str(env_name) if env_name else key
            sys.exit(f"snapshot: subject '{subject.name}' is missing basic-auth {role} credential: {env_label}")
        credentials[role] = value
    real_prefix = str(config.get("intake_path_prefix", "/api/intake")).rstrip("/") + "/"
    return build_basic_auth_intake_client(
        base_url=source_url,
        username=credentials["username"],
        password=credentials["password"],
        real_prefix=real_prefix,
    )


def backup_records(evaluation_tmp: Path, names: list[str], *, backup_dir: Path | None = None) -> Path:
    """Move the named local files into *backup_dir* (default ``<evaluation_tmp>/backup-<UTC stamp>/``);
    returns that dir.

    The shared no-silent-destruction mechanism: restores park clobbered local
    records here, and fresh-by-default analyze parks the prior insights YAML
    here — nothing in ``evaluation/tmp`` is ever deleted, only moved (no prompt —
    CI-safe). Callers print their own one-liner naming the moved files. A CLI
    invocation that backs up more than once (analyze: the restore's
    clobber-backup, then the fresh-insights move) passes one *backup_dir* so a
    single command yields a single backup dir, not one per stamp.
    """
    if backup_dir is None:
        # Microseconds keep two invocations within the same second from sharing a dir.
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        backup_dir = evaluation_tmp / f"backup-{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.move(evaluation_tmp / name, backup_dir / name)
    return backup_dir


def seed_records(state_tmp: Path, evaluation_tmp: Path, *, backup_dir: Path | None = None) -> tuple[str, list[str]]:
    """Copy the bundle's ``tmp/`` run records beside the local ones; ``(message, seeded names)``.

    Only ``*.run.json`` files travel: insights are per-analyze output, not
    state, so any ``insights_*.yaml`` a pre-cutover bundle still carries is
    ignored (and local insight YAMLs are never touched — the fresh/keep choice
    lives in the analyze CLI). Never destroys local state silently: any local
    record the bundle is about to overwrite is first moved into *backup_dir*
    (default: a fresh ``<evaluation_tmp>/backup-<UTC stamp>/``) with one printed
    line saying so.
    """
    evaluation_tmp.mkdir(parents=True, exist_ok=True)
    bundle_files = sorted(p for p in state_tmp.glob("*.run.json") if p.is_file()) if state_tmp.is_dir() else []
    clobbered = [p.name for p in bundle_files if (evaluation_tmp / p.name).exists()]
    if clobbered:
        backup_dir = backup_records(evaluation_tmp, clobbered, backup_dir=backup_dir)
        print(f"backed up {len(clobbered)} local record(s) to {backup_dir}: {', '.join(clobbered)}")
    for item in bundle_files:
        shutil.copy2(item, evaluation_tmp / item.name)
    seeded = [p.name for p in bundle_files]
    if not seeded:
        return "the bundle carries no run records", seeded
    return f"seeded {len(seeded)} run record(s) from the bundle: {', '.join(seeded)}", seeded


def snapshot_export(
    subjects: list[Subject],
    out: Path,
    tmp_dir: Path,
    *,
    since: datetime.datetime | None,
) -> Path:
    """Export the subjects' workspaces from the read API into one bundle at *out*.

    Bundle layout matches the legacy tar (root ``state/``) but the payload is
    ``export/<workspace>/*.jsonl`` from :func:`evaluation.export.export_workspaces`
    plus the ``tmp/`` run records and a ``kind: evaluation-export`` manifest.
    Source URL is each subject's stanza ``base_url`` (they must agree; the
    CLI's ``--base`` override rewrites every stanza before this is called).
    """
    subject_workspaces: list[tuple[Subject, list[str]]] = []
    claimed_workspaces: dict[str, Subject] = {}
    for subject in subjects:
        for workspace in workspaces_for_subject(subject):
            owner = claimed_workspaces.get(workspace)
            if owner is not None and (owner.config.get("experiment") or subject.config.get("experiment")):
                sys.exit(
                    f"snapshot: workspace '{workspace}' is shared by experiment-scoped subject "
                    f"'{owner.name if owner.config.get('experiment') else subject.name}' and another subject"
                )
        workspaces = [workspace for workspace in workspaces_for_subject(subject) if workspace not in claimed_workspaces]
        if workspaces:
            subject_workspaces.append((subject, workspaces))
            claimed_workspaces.update({workspace: subject for workspace in workspaces})
    # Every selected subject must carry a base_url: a partial miss would silently
    # let the agreement check pass on the configured subset and export the
    # unconfigured subject from the others' platform.
    missing = [s.name for s in subjects if not s.config.get("base_url")]
    if missing:
        sys.exit("snapshot: no base_url configured for subject(s) " + ", ".join(missing) + " (pass --base URL?)")
    urls = {str(s.config["base_url"]) for s in subjects}
    if not urls:
        sys.exit("snapshot: no base_url configured for the selected subjects (pass --base URL?)")
    if len(urls) > 1:
        sys.exit(
            "snapshot: subjects disagree on base_url (" + ", ".join(sorted(urls)) + ") — "
            "snapshot them separately or pass --base URL"
        )
    source_url = urls.pop()
    records = pick_records(tmp_dir, [s.name for s in subjects])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
        state = Path(tmp) / "state"
        (state / "tmp").mkdir(parents=True)
        exported = [
            export.export_workspaces(
                source_url,
                workspaces,
                state,
                since=since,
                client=_basic_auth_intake_client_for(subject, source_url),
                experiment=str(subject.config["experiment"]) if subject.config.get("experiment") else None,
            )
            for subject, workspaces in subject_workspaces
        ]
        min_bounds = [
            datetime.datetime.fromisoformat(stats["min_start_time"])
            for stats in exported
            if stats["min_start_time"] is not None
        ]
        max_bounds = [
            datetime.datetime.fromisoformat(stats["max_start_time"])
            for stats in exported
            if stats["max_start_time"] is not None
        ]
        stats = {
            "workspaces": {
                workspace: counts for result in exported for workspace, counts in result["workspaces"].items()
            },
            "min_start_time": min(min_bounds).isoformat() if min_bounds else None,
            "max_start_time": max(max_bounds).isoformat() if max_bounds else None,
            "selections": {
                workspace: selection
                for result in exported
                for workspace, selection in result.get("selections", {}).items()
            },
        }
        for rec in records:
            shutil.copy2(rec, state / "tmp" / rec.name)
        manifest = build_export_manifest(
            [s.name for s in subjects],
            records,
            stats,
            source_url=source_url,
            platform_info=fetch_platform_info(source_url),
            env=os.environ,
        )
        (state / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        subprocess.run(["tar", "--zstd", "-cf", str(out), "-C", tmp, "state"], check=True)
    print(f"export bundle written to {out}")
    return out
