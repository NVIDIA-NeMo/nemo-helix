# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct one Harbor job from already-redacted evaluation archives.

No Harbor installation is required in the control plane. The export preserves
source job metadata separately and computes only additive trial statistics;
custom metrics and pass@k are left to downstream Harbor analysis.
"""

import hashlib
import json
import logging
import shutil
import tarfile
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath

from pydantic import BaseModel

from scaled_evals.api import artifacts
from scaled_evals.api.redaction import redact_secret_text
from scaled_evals.api.settings import settings
from scaled_evals.archive_validation import (
    BenchmarkArchiveError,
    benchmark_archive_object_key,
    file_sha256,
    harbor_job_uuid,
    validate_archive_id,
    verify_artifact_manifest,
)
from scaled_evals.archive_validation import (
    read_archive_json as _read_json,
)
from scaled_evals.models.benchmark_archives import (
    ArchivedTrial,
    ArchiveFile,
    BenchmarkArchiveArtifact,
    BenchmarkArchiveExportMember,
    BenchmarkArchiveManifest,
    BenchmarkEvidenceCheck,
)

LOG = logging.getLogger(__name__)


def _local_task_labels(source: Path, member: dict) -> dict[str, str]:
    """Scope local task identities to their member without collapsing datasets."""
    paths = set()
    for trial_dir in source.iterdir():
        if not trial_dir.is_dir():
            continue
        for filename in ("config.json", "result.json"):
            path = trial_dir / filename
            if not path.is_file():
                continue
            data = _read_json(path)
            config = data if filename == "config.json" else data.get("config", {})
            task = config.get("task", {})
            if task.get("path") and not task.get("git_url") and not task.get("name"):
                paths.add(task["path"])
    slug = member.get("task_slug") or member["task_id"]
    return {
        path: (
            slug
            if len(paths) == 1
            else (f"{slug}__{PurePosixPath(path).name}__{hashlib.sha256(path.encode()).hexdigest()[:8]}")
        )
        for path in paths
    }


def _rename_local_task(config: dict, labels: dict[str, str]) -> str | None:
    task = config.get("task", {})
    path = task.get("path")
    if path not in labels or task.get("git_url") or task.get("name"):
        return None
    label = labels[path]
    task["path"] = str(PurePosixPath(path).with_name(label))
    return label


def _write_json(path: Path, value: dict | BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    path.write_text(json.dumps(document, indent=2) + "\n")


def _unpack(archive: Path, root: Path, budget: list[int]) -> None:
    """Extract regular files only, with aggregate limits across the whole run."""
    seen: set[str] = set()
    with tarfile.open(archive, mode="r|gz") as source:
        for member in source:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or "\\" in member.name or not name.parts:
                raise BenchmarkArchiveError("unsafe member archive path")
            if member.isdir():
                continue
            if len(name.parts) < 2 or name.parts[0] != "artifacts":
                raise BenchmarkArchiveError("expected an evaluation archive with artifacts/ root")
            name = PurePosixPath(*name.parts[1:])
            if not member.isfile() or name.as_posix() in seen:
                raise BenchmarkArchiveError("member archives must contain unique regular files")
            seen.add(name.as_posix())
            budget[0] += 1
            budget[1] += member.size
            if (
                budget[0] > settings.benchmark_archive_max_files
                or budget[1] > settings.benchmark_archive_max_source_bytes
            ):
                raise BenchmarkArchiveError("benchmark archive exceeds configured file/byte limits")
            target = root.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            body = source.extractfile(member)
            if body is None:
                raise BenchmarkArchiveError("could not read member archive file")
            with body, target.open("wb") as output:
                shutil.copyfileobj(body, output)


def _benchmark_artifacts(
    run_id: str, output: Path, budget: list[int], check_claim: Callable[[], None]
) -> list[BenchmarkArchiveArtifact]:
    prefix = f"benchmark-runs/{run_id}/artifacts/"
    objects = sorted(artifacts.list_objects(prefix), key=lambda item: item["key"])
    entries = []
    seen = set()
    for item in objects:
        check_claim()
        key = item["key"]
        if not key.startswith(prefix):
            raise BenchmarkArchiveError("benchmark artifact is outside the run prefix")
        relative = key[len(prefix) :]
        if relative.endswith("/") and item["size_bytes"] == 0:
            continue
        path = PurePosixPath(relative)
        if (
            not relative
            or not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in relative
            or path.as_posix() != relative
            or relative in seen
        ):
            raise BenchmarkArchiveError("unsafe or duplicate benchmark artifact path")
        seen.add(relative)
        budget[0] += 1
        if (
            budget[0] > settings.benchmark_archive_max_files
            or budget[1] + item["size_bytes"] > settings.benchmark_archive_max_source_bytes
        ):
            raise BenchmarkArchiveError("benchmark archive exceeds configured file/byte limits")
        target = output / "_scaled_evals" / "benchmark" / "artifacts" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        with target.open("xb") as destination:
            for chunk in artifacts.stream_object(key):
                check_claim()
                size += len(chunk)
                budget[1] += len(chunk)
                if size > item["size_bytes"] or budget[1] > settings.benchmark_archive_max_source_bytes:
                    raise BenchmarkArchiveError("benchmark artifact size changed or exceeds limit")
                digest.update(chunk)
                destination.write(chunk)
        if size != item["size_bytes"]:
            raise BenchmarkArchiveError("benchmark artifact size changed; rebuild export")
        entries.append(
            BenchmarkArchiveArtifact(
                object_key=key,
                path=target.relative_to(output).as_posix(),
                size_bytes=size,
                sha256=digest.hexdigest(),
                updated_at=item.get("updated_at"),
            )
        )
    check_claim()
    if objects != sorted(artifacts.list_objects(prefix), key=lambda item: item["key"]):
        raise BenchmarkArchiveError("benchmark artifacts changed during export; rebuild export")
    return entries


def _stats(trials: list[dict]) -> dict:
    stats = {
        "n_completed_trials": len(trials),
        "n_errored_trials": 0,
        "n_running_trials": 0,
        "n_pending_trials": 0,
        "n_cancelled_trials": 0,
        "n_retries": 0,
        "evals": {},
    }
    for trial in trials:
        agent = trial["agent_info"]
        model = (agent.get("model_info") or {}).get("name")
        key = "__".join(v for v in [agent["name"], model, trial.get("source") or "adhoc"] if v)
        group = stats["evals"].setdefault(
            key,
            {
                "n_trials": 0,
                "n_errors": 0,
                "metrics": [],
                "pass_at_k": {},
                "reward_stats": {},
                "exception_stats": {},
            },
        )
        rewards = (trial.get("verifier_result") or {}).get("rewards")
        if rewards is not None:
            group["n_trials"] += 1
            for reward, value in rewards.items():
                group["reward_stats"].setdefault(reward, {}).setdefault(value, []).append(trial["trial_name"])
        exception = trial.get("exception_info")
        if exception:
            group["n_errors"] += 1
            stats["n_errored_trials"] += 1
            kind = exception["exception_type"]
            group["exception_stats"].setdefault(kind, []).append(trial["trial_name"])
            stats["n_cancelled_trials"] += kind == "CancelledError"
        contexts = (
            [trial["agent_result"]]
            if trial.get("agent_result") is not None
            else [
                step["agent_result"] for step in trial.get("step_results") or [] if step.get("agent_result") is not None
            ]
        )
        for context in contexts:
            for field in ("n_input_tokens", "n_cache_tokens", "n_output_tokens", "cost_usd"):
                if context.get(field) is not None:
                    stats[field] = stats.get(field, 0) + context[field]
    # Legacy Harbor readers still use these names.
    stats["n_trials"] = stats["n_completed_trials"]
    stats["n_errors"] = stats["n_errored_trials"]
    return stats


def build_benchmark_archive(
    job: dict,
    *,
    check_claim: Callable[[], None],
    evidence_checks: Sequence[BenchmarkEvidenceCheck] = (),
) -> dict:
    """Assemble one typed, checksummed export without provider lifecycle logic."""
    run_id = validate_archive_id(job["benchmark_run_id"])
    job_uuid = harbor_job_uuid(run_id, job["generation"])
    budget = [0, 0]
    trials: list[dict] = []
    configs: list[dict] = []
    results: list[dict] = []
    manifest = BenchmarkArchiveManifest(
        benchmark_run_id=run_id,
        generation=job["generation"],
        harbor_job_id=job_uuid,
    )
    with tempfile.TemporaryDirectory(prefix="benchmark-archive-") as tmp:
        base = Path(tmp)
        output = base / run_id
        output.mkdir()
        for member in job["members"]:
            check_claim()
            evaluation_id = validate_archive_id(member["id"])
            compressed = base / "member.tar.gz"
            downloaded = 0
            digest = hashlib.sha256()
            with compressed.open("wb") as destination:
                for chunk in artifacts.stream_object(member["archive_object_key"]):
                    check_claim()
                    downloaded += len(chunk)
                    if downloaded > settings.benchmark_archive_max_source_bytes:
                        raise BenchmarkArchiveError("member archive exceeds configured byte limit")
                    digest.update(chunk)
                    destination.write(chunk)
            if downloaded != member["archive_size_bytes"]:
                raise BenchmarkArchiveError(f"archive size changed for {evaluation_id}; rebuild export")
            source = base / "member"
            source.mkdir()
            _unpack(compressed, source, budget)
            integrity = verify_artifact_manifest(source, check_claim)
            labels = _local_task_labels(source, member)
            metadata = output / "_scaled_evals" / "evaluations" / evaluation_id
            metadata.mkdir(parents=True)
            provenance = source / "scaled-evals-provenance.json"
            entry = BenchmarkArchiveExportMember(
                **member,
                archive_sha256=digest.hexdigest(),
                artifact_integrity=integrity,
                provenance=_read_json(provenance) if provenance.is_file() else None,
                provenance_path=(metadata / provenance.name).relative_to(output).as_posix()
                if provenance.is_file()
                else None,
            )
            if (source / "config.json").is_file():
                member_config = _read_json(source / "config.json")
                for task in member_config.get("tasks", []):
                    _rename_local_task({"task": task}, labels)
                configs.append(member_config)
            else:
                entry.missing.append("config.json")
            if (source / "result.json").is_file():
                results.append(_read_json(source / "result.json"))
            else:
                entry.missing.append("result.json")
            for path in sorted(source.iterdir()):
                if path.is_dir() and ((path / "config.json").is_file() or (path / "result.json").is_file()):
                    trial_name = f"{evaluation_id}__{path.name}"
                    target = output / trial_name
                    shutil.move(path, target)
                    entry.trials.append(ArchivedTrial(source=path.name, exported=trial_name))
                    for filename in ("config.json", "result.json"):
                        trial_path = target / filename
                        if not trial_path.is_file():
                            entry.missing.append(f"{path.name}/{filename}")
                            continue
                        original = metadata / path.name / filename
                        original.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(trial_path, original)
                        data = _read_json(trial_path)
                        data["trial_name"] = trial_name
                        if filename == "config.json":
                            _rename_local_task(data, labels)
                            data.update(job_id=str(job_uuid), trials_dir=run_id)
                        if filename == "result.json":
                            data["trial_uri"] = f"{run_id}/{trial_name}"
                            if isinstance(data.get("config"), dict):
                                label = _rename_local_task(data["config"], labels)
                                if label:
                                    data["task_name"] = label
                                    data["task_id"] = {"path": data["config"]["task"]["path"]}
                                data["config"].update(trial_name=trial_name, job_id=str(job_uuid), trials_dir=run_id)
                            trials.append(data)
                        _write_json(trial_path, data)
                else:
                    shutil.move(path, metadata / path.name)
            if not entry.trials:
                entry.missing.append("trial directories")
            manifest.members.append(entry)
            shutil.rmtree(source)
            compressed.unlink()
        manifest.benchmark_artifacts = _benchmark_artifacts(run_id, output, budget, check_claim)
        for check_evidence in evidence_checks:
            manifest.unavailable_benchmark_evidence.extend(
                check_evidence(output, manifest.members, manifest.benchmark_artifacts)
            )
        if not configs or not results:
            raise BenchmarkArchiveError("no Harbor job config/results found in member archives")
        config = dict(configs[0])
        config.update(job_name=run_id, jobs_dir=".", metrics=[])
        for field in ("tasks", "datasets", "agents"):
            values = [value for item in configs for value in item.get(field, [])]
            config[field] = list({json.dumps(value, sort_keys=True): value for value in values}.values())
        _write_json(output / "config.json", config)
        starts = [item["started_at"] for item in results if item.get("started_at")]
        finishes = [item["finished_at"] for item in results if item.get("finished_at")]
        total = sum(item.get("n_total_trials", 0) for item in results)
        manifest.partial = (
            any(m.missing for m in manifest.members)
            or total != len(trials)
            or bool(manifest.unavailable_benchmark_evidence)
        )
        manifest.n_exported_trial_results = len(trials)
        manifest.n_declared_trials = total
        stats = _stats(trials)
        stats["n_retries"] = sum((item.get("stats") or {}).get("n_retries", 0) for item in results)
        _write_json(
            output / "result.json",
            {
                "id": str(job_uuid),
                "started_at": min(starts) if starts else manifest.created_at.isoformat(),
                "finished_at": max(finishes) if finishes else manifest.created_at.isoformat(),
                "updated_at": manifest.created_at.isoformat(),
                "n_total_trials": max(total, len(trials)),
                "stats": stats,
                "trial_results": trials,
            },
        )
        for path in sorted(output.rglob("*")):
            if path.is_file():
                check_claim()
                manifest.files.append(
                    ArchiveFile(
                        path=path.relative_to(output).as_posix(),
                        size_bytes=path.stat().st_size,
                        sha256=file_sha256(path),
                    )
                )
        _write_json(output / "scaled-evals-benchmark-archive.json", manifest)
        archive_path = base / "results.tar.gz"
        check_claim()
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(output, arcname=run_id)
        key = benchmark_archive_object_key(run_id, job["generation"], job["claim_token"])
        sha256 = file_sha256(archive_path)
        check_claim()
        try:
            size = artifacts.upload_file(archive_path, key, content_type="application/gzip")
            check_claim()
        except Exception:
            # Publication has not been attempted yet, so this claim's object is
            # safe to remove even if the upload succeeded but its response failed.
            try:
                artifacts.delete_object(key)
            except Exception as exc:  # noqa: BLE001 - a later reconciliation sweep retries deletion
                LOG.warning("benchmark archive upload cleanup deferred for %s: %s", key, redact_secret_text(str(exc)))
            raise
        return {"object_key": key, "size_bytes": size, "sha256": sha256, "partial": manifest.partial}
