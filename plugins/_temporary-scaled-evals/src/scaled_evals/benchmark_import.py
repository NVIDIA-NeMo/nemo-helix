# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conformance and orchestration helpers for materialized benchmark imports."""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
import tempfile
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, Field

DEFAULT_MAX_PACK_BYTES = 20 * 1024 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_HARBOR_TASK_NAME = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*/[A-Za-z0-9_-][A-Za-z0-9._-]*$")


class ConformanceCheck(BaseModel):
    code: str
    status: Literal["passed", "warning", "failed", "requires_remote_validation"]
    message: str
    subject: str | None = None


class BenchmarkImportValidation(BaseModel):
    schema_version: Literal["scaled-evals-benchmark-import-validation-v1"] = (
        "scaled-evals-benchmark-import-validation-v1"
    )
    valid: bool
    manifest_sha256: str
    manifest_path: str
    tasks: int
    benchmarks: int
    checks: list[ConformanceCheck] = Field(default_factory=list)


def canonical_manifest_sha256(manifest: dict[str, Any]) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_benchmark_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("benchmark import manifest must be a JSON object")
    return payload


def load_import_images(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Load prebuilt image JSON/JSONL while preserving opaque non-error metadata."""
    images: dict[str, dict[str, Any]] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        stripped = text.lstrip()
        payload: Any
        if stripped.startswith("{") or stripped.startswith("["):
            payload = json.loads(text)
        else:
            payload = [json.loads(line) for line in text.splitlines() if line.strip()]
        if isinstance(payload, dict) and payload.get("slug"):
            records = [payload]
        elif (
            isinstance(payload, dict)
            and "images" not in payload
            and "results" not in payload
            and all(isinstance(value, dict) for value in payload.values())
        ):
            records = [{"slug": slug, **value} for slug, value in payload.items()]
        elif isinstance(payload, dict):
            records = payload.get("images") or payload.get("results") or []
        else:
            records = payload
        if not isinstance(records, list):
            raise ValueError(f"image results must contain a list: {path}")
        for record in records:
            if not isinstance(record, dict) or record.get("status") == "failed":
                continue
            slug = str(record.get("slug") or "").strip()
            image_ref = str(record.get("image_ref") or "").strip()
            image_digest = str(record.get("image_digest") or "").strip()
            if slug and image_ref and image_digest:
                image = dict(record)
                image.pop("slug", None)
                image.pop("error", None)
                image["image_ref"] = image_ref
                image["image_digest"] = image_digest
                images[slug] = image
    return images


def write_legacy_import_state(path: Path, detail: dict[str, Any], *, phase: str) -> None:
    """Atomically project durable import state into the historical JSON-list shape."""
    records = []
    for task in detail.get("tasks") or []:
        status = str(task.get("status") or "unknown")
        if phase == "upload":
            status = "existing" if status == "ready" else "created"
        record = {
            "slug": task["slug"],
            "status": status,
            "task_id": task["task_id"],
            "revision": task["task_revision"],
            "import_id": detail["id"],
        }
        if phase == "finalize":
            metadata = dict(task.get("image_metadata") or {})
            for key in {
                "slug",
                "status",
                "task_id",
                "revision",
                "import_id",
                "image_ref",
                "image_digest",
                "build_error",
                "error",
            }:
                metadata.pop(key, None)
            record.update(metadata)
            for key in ("image_ref", "image_digest", "build_error"):
                if task.get(key) is not None:
                    record[key] = task[key]
        records.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(sorted(records, key=lambda item: item["slug"]), indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        temporary.write(encoded)
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def import_id_from_legacy_state(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"legacy state must be a JSON list: {path}")
    import_ids = {row.get("import_id") for row in payload if isinstance(row, dict)}
    import_ids.discard(None)
    if len(import_ids) != 1:
        raise ValueError(f"legacy state does not identify exactly one benchmark import: {path}")
    return str(import_ids.pop())


def _safe_relative_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        return None
    return Path(*pure.parts)


def _check_source(source: object, checks: list[ConformanceCheck]) -> None:
    if not isinstance(source, dict) or not source:
        checks.append(
            ConformanceCheck(
                code="source_provenance",
                status="failed",
                message="manifest source provenance must be a non-empty object",
            )
        )
        return
    source_type = source.get("type")
    pinned = False
    if source_type in {"git", "git_generator"}:
        pinned = bool(re.fullmatch(r"[0-9a-f]{40}", str(source.get("commit") or "")))
    elif source_type == "huggingface":
        pinned = bool(source.get("dataset") and source.get("revision") and source.get("split"))
    elif source_type == "harbor_registry":
        pinned = bool(source.get("reference"))
    elif source_type == "subset":
        pinned = bool(source.get("dataset") and source.get("split") and source.get("task_ids"))
    if pinned:
        checks.append(
            ConformanceCheck(
                code="source_provenance",
                status="passed",
                message=f"source type {source_type!r} carries a reproducible reference",
            )
        )
    else:
        checks.append(
            ConformanceCheck(
                code="source_provenance",
                status="warning",
                message=(
                    f"source type {source_type!r} is described but lacks a complete immutable "
                    "revision; preserve the supplied metadata and pin it before shared publication"
                ),
            )
        )


def _validate_pack(
    path: Path,
    *,
    expected_sha256: str,
    max_pack_bytes: int,
    subject: str,
) -> list[ConformanceCheck]:
    checks: list[ConformanceCheck] = []
    if not path.is_file():
        return [
            ConformanceCheck(
                code="pack_exists",
                status="failed",
                message=f"task pack does not exist: {path}",
                subject=subject,
            )
        ]
    size = path.stat().st_size
    checks.append(
        ConformanceCheck(
            code="pack_size",
            status="passed" if size <= max_pack_bytes else "failed",
            message=f"compressed task pack is {size} bytes (limit {max_pack_bytes})",
            subject=subject,
        )
    )
    if size > max_pack_bytes:
        return checks
    with path.open("rb") as raw:
        actual_sha256 = hashlib.file_digest(raw, "sha256").hexdigest()
    checks.append(
        ConformanceCheck(
            code="pack_sha256",
            status="passed" if actual_sha256 == expected_sha256 else "failed",
            message=f"observed sha256 {actual_sha256}; manifest declares {expected_sha256}",
            subject=subject,
        )
    )
    try:
        with path.open("rb") as raw:
            header = raw.read(10)
        gzip_mtime = int.from_bytes(header[4:8], "little") if header.startswith(b"\x1f\x8b") else -1
        with tarfile.open(path, mode="r:gz") as archive:
            members = archive.getmembers()
            names = [member.name.removeprefix("./") for member in members]
            duplicate_names = sorted({name for name in names if names.count(name) > 1})
            unsafe = []
            special = []
            noncanonical_metadata = []
            for member, name in zip(members, names, strict=True):
                pure = PurePosixPath(name)
                if not name or pure.is_absolute() or ".." in pure.parts:
                    unsafe.append(member.name)
                if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                    special.append(member.name)
                if member.uid != 0 or member.gid != 0 or member.mtime != 0:
                    noncanonical_metadata.append(member.name)
            if duplicate_names or unsafe or special:
                details = []
                if duplicate_names:
                    details.append(f"duplicate members: {duplicate_names[:5]}")
                if unsafe:
                    details.append(f"unsafe paths: {unsafe[:5]}")
                if special:
                    details.append(f"links/special files: {special[:5]}")
                checks.append(
                    ConformanceCheck(
                        code="archive_safety",
                        status="failed",
                        message="; ".join(details),
                        subject=subject,
                    )
                )
            else:
                checks.append(
                    ConformanceCheck(
                        code="archive_safety",
                        status="passed",
                        message="archive has unique safe regular-file/directory members",
                        subject=subject,
                    )
                )
            canonical = gzip_mtime == 0 and not noncanonical_metadata and names == sorted(names)
            checks.append(
                ConformanceCheck(
                    code="deterministic_pack",
                    status="passed" if canonical else "failed",
                    message=(
                        "gzip and tar metadata/order are deterministic"
                        if canonical
                        else "pack must use gzip mtime 0, tar uid/gid/mtime 0, and sorted members"
                    ),
                    subject=subject,
                )
            )
            files = {name: member for member, name in zip(members, names, strict=True) if member.isfile()}
            task_tomls = [name for name in files if re.fullmatch(r"tasks/[^/]+/task\.toml", name)]
            dockerfiles = [name for name in files if name == "Dockerfile"]
            if len(task_tomls) != 1 or len(dockerfiles) != 1:
                checks.append(
                    ConformanceCheck(
                        code="harbor_structure",
                        status="failed",
                        message=("pack must contain one root Dockerfile and exactly one tasks/<name>/task.toml"),
                        subject=subject,
                    )
                )
            else:
                member = archive.extractfile(files[task_tomls[0]])
                if member is None:
                    raise tarfile.ReadError("cannot read task.toml")
                task_config = tomllib.loads(member.read().decode("utf-8"))
                raw_task_name = task_config["task"].get("name") if isinstance(task_config.get("task"), dict) else None
                task_name = raw_task_name if isinstance(raw_task_name, str) else ""
                has_task = bool(_HARBOR_TASK_NAME.fullmatch(task_name)) and ".." not in task_name
                has_verifier = isinstance(task_config.get("verifier"), dict)
                checks.append(
                    ConformanceCheck(
                        code="harbor_structure",
                        status="passed" if has_task else "failed",
                        message=f"Harbor task metadata is present ({task_name})"
                        if has_task
                        else "[task].name must use Harbor's org/name package format",
                        subject=subject,
                    )
                )
                checks.append(
                    ConformanceCheck(
                        code="verifier_separation",
                        status="passed" if has_verifier else "failed",
                        message=(
                            "task declares a Harbor verifier stage"
                            if has_verifier
                            else "task.toml must declare [verifier]"
                        ),
                        subject=subject,
                    )
                )
                docker = archive.extractfile(files["Dockerfile"])
                if docker is None:
                    raise tarfile.ReadError("cannot read Dockerfile")
                docker_text = docker.read().decode("utf-8", errors="replace")
                workdirs = re.findall(r"(?im)^\s*WORKDIR\s+([^\s#]+)", docker_text)
                invalid_workdir = bool(
                    workdirs and (not workdirs[-1].startswith("/") or workdirs[-1] == "/" or "$" in workdirs[-1])
                )
                checks.append(
                    ConformanceCheck(
                        code="workspace_compatibility",
                        status="failed" if invalid_workdir else "passed",
                        message=(
                            f"final WORKDIR {workdirs[-1]!r} is not an absolute stable workspace"
                            if invalid_workdir
                            else "task image has no incompatible final WORKDIR declaration"
                        ),
                        subject=subject,
                    )
                )
    except (OSError, tarfile.TarError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        checks.append(
            ConformanceCheck(
                code="archive_readable",
                status="failed",
                message=f"cannot inspect task pack: {exc}",
                subject=subject,
            )
        )
    return checks


def validate_benchmark_import(
    manifest_path: Path,
    *,
    output_root: Path | None = None,
    max_pack_bytes: int = DEFAULT_MAX_PACK_BYTES,
) -> BenchmarkImportValidation:
    manifest_path = manifest_path.resolve()
    manifest = load_benchmark_manifest(manifest_path)
    root = (output_root or manifest_path.parent.parent).resolve()
    checks: list[ConformanceCheck] = []
    if manifest.get("schema_version") != 1:
        checks.append(
            ConformanceCheck(
                code="manifest_schema",
                status="failed",
                message="only standard benchmark manifest schema_version 1 is supported",
            )
        )
    else:
        checks.append(ConformanceCheck(code="manifest_schema", status="passed", message="manifest schema_version is 1"))
    visibility = manifest.get("visibility")
    checks.append(
        ConformanceCheck(
            code="visibility",
            status="passed" if visibility in {"private", "team", "org", "public"} else "failed",
            message=f"manifest visibility is {visibility!r}",
        )
    )
    _check_source(manifest.get("source"), checks)

    tasks = manifest.get("tasks")
    benchmarks = manifest.get("benchmarks")
    if not isinstance(tasks, list) or not tasks:
        checks.append(ConformanceCheck(code="tasks", status="failed", message="manifest tasks must be non-empty"))
        tasks = []
    if not isinstance(benchmarks, list) or not benchmarks:
        checks.append(
            ConformanceCheck(code="benchmarks", status="failed", message="manifest benchmarks must be non-empty")
        )
        benchmarks = []

    slugs: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            checks.append(ConformanceCheck(code="task_shape", status="failed", message="task is not an object"))
            continue
        slug = str(task.get("slug") or "")
        pack = _safe_relative_path(task.get("pack"))
        digest = str(task.get("pack_sha256") or "")
        if not _SLUG.fullmatch(slug) or slug in slugs or pack is None or not _SHA256.fullmatch(digest):
            checks.append(
                ConformanceCheck(
                    code="task_shape",
                    status="failed",
                    message="task needs a unique valid slug, safe pack path, and lowercase sha256",
                    subject=slug or None,
                )
            )
            continue
        slugs.add(slug)
        checks.extend(
            _validate_pack(
                root / pack,
                expected_sha256=digest,
                max_pack_bytes=max_pack_bytes,
                subject=slug,
            )
        )

    benchmark_slugs: set[str] = set()
    for benchmark in benchmarks:
        if not isinstance(benchmark, dict):
            checks.append(
                ConformanceCheck(code="benchmark_shape", status="failed", message="benchmark is not an object")
            )
            continue
        slug = str(benchmark.get("slug") or "")
        members = benchmark.get("tasks")
        valid = (
            bool(_SLUG.fullmatch(slug))
            and slug not in benchmark_slugs
            and isinstance(members, list)
            and bool(members)
            and len(members) == len(set(members))
            and set(members) <= slugs
        )
        checks.append(
            ConformanceCheck(
                code="benchmark_membership",
                status="passed" if valid else "failed",
                message=(
                    f"benchmark pins {len(members)} unique manifest tasks"
                    if valid
                    else "benchmark needs a unique valid slug and unique known task members"
                ),
                subject=slug or None,
            )
        )
        benchmark_slugs.add(slug)

    checks.append(
        ConformanceCheck(
            code="remote_preparation",
            status="requires_remote_validation",
            message=(
                "image pullability and managed build completion are verified during import; "
                "external signing, admission, and runtime qualification are outside this workflow"
            ),
        )
    )
    return BenchmarkImportValidation(
        valid=not any(check.status == "failed" for check in checks),
        manifest_sha256=canonical_manifest_sha256(manifest),
        manifest_path=str(manifest_path),
        tasks=len(tasks),
        benchmarks=len(benchmarks),
        checks=checks,
    )
