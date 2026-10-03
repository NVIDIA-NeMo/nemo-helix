# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize Harbor dataset images through deployment-managed task revisions."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
import yaml

from scaled_evals.api import artifacts
from scaled_evals.api.build.task_image_identity import (
    TaskImageReference,
    parse_task_image_ref,
    resolve_upstream_image,
)
from scaled_evals.api.repositories.base_repository import Conflict
from scaled_evals.api.repositories.task_repository import TaskRepository
from scaled_evals.api.settings import settings
from scaled_evals.api.utils import make_id
from scaled_evals.harbor_dataset_import import (
    HarborDatasetImageImport,
    build_image_import_context,
)

_SLUG_CHAR = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class DatasetImageImport:
    source_image: str
    source_immutable_image: str
    task_id: str
    task_revision: int
    image_ref: str
    image_digest: str
    runtime_image: str


def effective_image_mode(profile: Mapping[str, Any]) -> str:
    requested = profile.get("dataset_image_mode")
    if requested not in {None, "managed"}:
        raise ValueError("Harbor dataset images must use deployment-managed image imports")
    return "managed"


def dataset_configs(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = next(
        (
            profile.get(key)
            for key in ("config", "harbor_config", "template", "harbor_template")
            if profile.get(key) is not None
        ),
        None,
    )
    if isinstance(raw, str):
        parsed = yaml.safe_load(raw) or {}
    elif raw is None:
        parsed = {key: value for key, value in profile.items() if key not in {"dataset_only", "dataset_image_mode"}}
    else:
        raise ValueError("dataset-only Harbor profile template must be YAML text")
    datasets = parsed.get("datasets") if isinstance(parsed, dict) else None
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("dataset-only Harbor profile requires a non-empty datasets list")
    if not all(isinstance(item, dict) for item in datasets):
        raise ValueError("dataset-only Harbor datasets must be objects")
    return datasets


def resolve_dataset_members(
    datasets: list[dict[str, Any]],
    *,
    harbor_dir: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[dict[str, Any]]:
    resolver = Path(__file__).resolve().parents[1] / "harbor_dataset_resolver.py"
    python = Path(harbor_dir).expanduser() / ".venv" / "bin" / "python"
    with tempfile.TemporaryDirectory(prefix="se-harbor-dataset-") as tmp:
        input_path = Path(tmp) / "input.json"
        output_path = Path(tmp) / "output.json"
        input_path.write_text(json.dumps({"datasets": datasets}))
        completed = runner(
            [str(python), str(resolver), str(input_path), str(output_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "unknown error").strip()
            raise RuntimeError(f"could not resolve Harbor dataset members: {detail[-2000:]}")
        payload = json.loads(output_path.read_text())
    members = payload.get("members")
    if not isinstance(members, list) or not members:
        raise RuntimeError("Harbor dataset resolved no task members")
    return members


def _slug(value: str) -> str:
    return _SLUG_CHAR.sub("-", value.lower()).strip("-") or "task"


def _build_backend() -> tuple[str, dict[str, str]]:
    if settings.image_builder_service_url:
        payload = {"context_path": "."}
        builder_source_commit = settings.image_builder_source_commit.strip()
        if builder_source_commit:
            payload["builder_source_commit"] = builder_source_commit
        return "image_builder_service", payload
    if settings.cloud_build_enabled:
        if settings.object_store_backend != "gcs":
            raise RuntimeError("managed Harbor dataset Cloud Build imports require GCS storage")
        return "cloudbuild", {}
    if settings.buildkit_enabled:
        return "buildkit", {}
    raise RuntimeError("managed Harbor dataset images require an enabled image builder")


def _runtime_image(image_ref: str, image_digest: str) -> str:
    ref: TaskImageReference = parse_task_image_ref(image_ref)
    if settings.sandbox_k8s_task_image_reference_mode == "tag":
        if ref.tag is None:
            raise RuntimeError("tag-mode dataset imports require a managed image tag")
        return ref.normalized_ref
    return ref.digest_ref(image_digest)


def _queue_import_revision(
    conn: psycopg.Connection,
    *,
    member: Mapping[str, Any],
    source_image: str,
) -> tuple[str, int, str]:
    resolved = resolve_upstream_image(source_image)
    imported = HarborDatasetImageImport.parse(resolved.immutable_ref)
    task_dir = Path(str(member["path"]))
    task_name = _slug(str(member["name"]))[:48]
    archive = build_image_import_context(imported, task_dir=task_dir, task_name=task_name)
    pack_sha256 = hashlib.sha256(archive).hexdigest()
    identity = hashlib.sha256(
        json.dumps(
            {
                "source": member.get("source"),
                "ref": member.get("ref"),
                "name": member["name"],
                "source_image": resolved.immutable_ref,
                "pack_sha256": pack_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    slug = f"harbor-{task_name[:28]}-{identity[:24]}"[:63]
    description = f"managed Harbor dataset image import {identity}"
    repo = TaskRepository(conn)
    task = repo.get_by_slug(slug)
    if task is None:
        task_id = make_id("task")
        object_key = f"{task_id}/rev/1/tarball.tar.gz"
        try:
            repo.create_with_initial_revision(
                task_id,
                name=f"Harbor {member['name']} ({source_image})",
                slug=slug,
                description=description,
                visibility="private",
                object_key=object_key,
            )
        except Conflict:
            task = repo.get_by_slug(slug)
            if task is None:
                raise
            task_id = str(task["id"])
    else:
        if task.get("owner_id") is not None or task.get("description") != description:
            raise RuntimeError(f"managed Harbor task slug collision: {slug}")
        task_id = str(task["id"])

    detail = repo.get_detail(task_id)
    assert detail is not None
    if detail.get("tarball_sha256") == pack_sha256 and detail["status"] in {
        "building",
        "ready",
    }:
        return task_id, int(detail["revision"]), resolved.immutable_ref
    if detail["status"] != "uploading":
        revision = repo.create_next_revision(task_id)
        if revision is None:
            raise RuntimeError(f"managed Harbor task disappeared: {task_id}")
        revision_number = revision.revision
        object_key = revision.object_key
    else:
        revision_number = int(detail["revision"])
        object_key = str(detail["tarball_object_key"])

    with tempfile.TemporaryDirectory(prefix="se-harbor-pack-") as tmp:
        archive_path = Path(tmp) / "task-pack.tar.gz"
        archive_path.write_bytes(archive)
        size_bytes = artifacts.upload_file(
            archive_path,
            object_key,
            content_type="application/gzip",
        )
    backend, payload = _build_backend()
    queued = repo.mark_latest_revision_building(
        task_id,
        build_backend=backend,
        build_payload=payload,
        tarball_sha256=pack_sha256,
        expected_revision=revision_number,
        exact_revision=True,
        tarball_size_bytes=size_bytes,
        tenant_storage_quota_bytes=settings.task_pack_tenant_storage_quota_bytes,
    )
    if queued is None:
        raise RuntimeError(f"could not queue managed Harbor task revision {task_id}:{revision_number}")
    if queued.quota_exceeded:
        raise RuntimeError(
            f"managed Harbor task pack exceeds the deployment's tenant storage quota for {task_id}:{revision_number}"
        )
    if queued.status != "uploading":
        detail = repo.get_detail(task_id)
        if not (
            detail
            and int(detail["revision"]) == revision_number
            and detail.get("tarball_sha256") == pack_sha256
            and detail["status"] in {"building", "ready"}
        ):
            raise RuntimeError(f"could not queue managed Harbor task revision {task_id}:{revision_number}")
    return task_id, revision_number, resolved.immutable_ref


def prepare_dataset_images(
    conn: psycopg.Connection,
    *,
    datasets: list[dict[str, Any]],
    harbor_dir: str,
    renew: Callable[[], None] | None = None,
) -> list[DatasetImageImport]:
    members = resolve_dataset_members(datasets, harbor_dir=harbor_dir)
    pending: dict[tuple[str, int], tuple[str, str]] = {}
    for member in members:
        images = member.get("images")
        if not isinstance(images, list) or not images:
            raise RuntimeError(f"Harbor dataset task has no images: {member.get('name')}")
        for source_image in images:
            task_id, revision, immutable = _queue_import_revision(
                conn,
                member=member,
                source_image=str(source_image),
            )
            pending[(task_id, revision)] = (str(source_image), immutable)

    deadline = time.monotonic() + settings.harbor_dataset_image_prepare_timeout_seconds
    repo = TaskRepository(conn)
    while True:
        if renew is not None:
            renew()
        imports: list[DatasetImageImport] = []
        waiting = False
        for (task_id, revision), (source_image, immutable) in pending.items():
            detail = repo.get_detail(task_id)
            if detail is None or int(detail["revision"]) != revision:
                raise RuntimeError(f"managed Harbor task revision changed: {task_id}:{revision}")
            if detail["status"] == "failed":
                raise RuntimeError(
                    f"managed Harbor image build failed for {source_image}: "
                    f"{detail.get('build_error') or 'unknown error'}"
                )
            if detail["status"] != "ready":
                waiting = True
                continue
            image_ref = str(detail["image_ref"])
            image_digest = str(detail["image_digest"])
            imports.append(
                DatasetImageImport(
                    source_image=source_image,
                    source_immutable_image=immutable,
                    task_id=task_id,
                    task_revision=revision,
                    image_ref=image_ref,
                    image_digest=image_digest,
                    runtime_image=_runtime_image(image_ref, image_digest),
                )
            )
        if not waiting:
            return imports
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for managed Harbor dataset task images")
        time.sleep(settings.harbor_dataset_image_poll_interval_seconds)
