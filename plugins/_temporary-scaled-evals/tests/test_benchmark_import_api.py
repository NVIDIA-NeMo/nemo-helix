# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("scaled_evals")


from fastapi import Response
from scaled_evals.api.auth import CurrentPrincipal
from scaled_evals.api.routers.benchmark_imports import (
    create_benchmark_import,
    publish_benchmark_import,
)
from scaled_evals.api.schemas.benchmark_imports import BenchmarkImportCreate
from scaled_evals.benchmark_import import (
    canonical_manifest_sha256,
    import_id_from_legacy_state,
    load_import_images,
    write_legacy_import_state,
)


def test_import_request_defaults_to_public_for_ci() -> None:
    request = BenchmarkImportCreate.model_validate(
        {
            "manifest_sha256": "a" * 64,
            "manifest": {
                "schema_version": 1,
                "catalog_id": "fixture",
                "visibility": "public",
                "source": {"type": "git", "commit": "b" * 40},
                "tasks": [
                    {
                        "name": "Task",
                        "slug": "task",
                        "pack": "packs/task.tar.gz",
                        "pack_sha256": "c" * 64,
                    }
                ],
                "benchmarks": [{"name": "Bench", "slug": "bench", "tasks": ["task"]}],
            },
        }
    )

    assert request.visibility == "public"


def test_image_results_preserve_opaque_metadata(tmp_path: Path) -> None:
    path = tmp_path / "images.jsonl"
    path.write_text(
        json.dumps(
            {
                "slug": "task",
                "status": "signed-rhacs-pending",
                "image_ref": "registry.example/task:one",
                "image_digest": "sha256:" + "d" * 64,
                "pipeline_url": "https://ci.example/job/1",
                "signature_ref": "opaque-external-metadata",
            }
        )
        + "\n"
    )

    images = load_import_images([path])

    assert images["task"]["pipeline_url"] == "https://ci.example/job/1"
    assert images["task"]["signature_ref"] == "opaque-external-metadata"


def test_legacy_state_is_an_atomic_projection_of_import_detail(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    write_legacy_import_state(
        state,
        {
            "id": "bmi_one",
            "tasks": [
                {
                    "slug": "task-b",
                    "task_id": "task_2",
                    "task_revision": 3,
                    "status": "ready",
                    "image_ref": "registry.example/b:one",
                    "image_digest": "sha256:" + "e" * 64,
                    "image_metadata": {"source_commit": "abc"},
                }
            ],
        },
        phase="finalize",
    )

    assert import_id_from_legacy_state(state) == "bmi_one"
    assert json.loads(state.read_text())[0]["source_commit"] == "abc"


def test_resumed_import_attaches_opaque_image_metadata_idempotently() -> None:
    manifest = {
        "schema_version": 1,
        "catalog_id": "fixture",
        "visibility": "public",
        "source": {"type": "git", "commit": "b" * 40},
        "tasks": [
            {
                "name": "Task",
                "slug": "task",
                "pack": "packs/task.tar.gz",
                "pack_sha256": "c" * 64,
            }
        ],
        "benchmarks": [{"name": "Bench", "slug": "bench", "tasks": ["task"]}],
    }
    body = BenchmarkImportCreate.model_validate(
        {
            "manifest_sha256": canonical_manifest_sha256(manifest),
            "manifest": manifest,
            "images": {
                "task": {
                    "image_ref": "registry.example/task:one",
                    "image_digest": "sha256:" + "d" * 64,
                    "pipeline_url": "https://ci.example/job/1",
                }
            },
        }
    )
    now = datetime.now(UTC)
    row = {
        "id": "bmi_one",
        "owner_id": "usr_test",
        "manifest_sha256": body.manifest_sha256,
        "manifest": manifest,
        "visibility": "public",
        "description": None,
        "created_at": now,
        "updated_at": now,
    }
    task = {
        "position": 0,
        "slug": "task",
        "task_id": "task_one",
        "task_revision": 1,
        "pack_path": "packs/task.tar.gz",
        "pack_sha256": "c" * 64,
        "requested_image_ref": "registry.example/task:one",
        "requested_image_digest": "sha256:" + "d" * 64,
        "requested_image_metadata": {"pipeline_url": "https://ci.example/job/1"},
        "status": "uploading",
        "image_ref": None,
        "image_digest": None,
        "build_error": None,
        "tarball_object_key": "task_one/rev/1/tarball.tar.gz",
    }
    db = MagicMock()
    db.benchmark_imports.get_by_identity.return_value = row
    db.benchmark_imports.tasks.return_value = [task]
    db.benchmark_imports.benchmarks.return_value = []

    result = create_benchmark_import(
        body,
        db,
        CurrentPrincipal(owner_type="user", owner_id="usr_test"),
        Response(),
    )

    assert result.id == "bmi_one"
    metadata = db.benchmark_imports.attach_task_image.call_args.kwargs["image_metadata"]
    assert metadata["pipeline_url"] == "https://ci.example/job/1"


def test_public_import_promotes_reused_owned_task_visibility() -> None:
    manifest = {
        "schema_version": 1,
        "catalog_id": "fixture",
        "visibility": "public",
        "source": {"type": "git", "commit": "b" * 40},
        "tasks": [
            {
                "name": "Task",
                "slug": "task",
                "pack": "packs/task.tar.gz",
                "pack_sha256": "c" * 64,
            }
        ],
        "benchmarks": [{"name": "Bench", "slug": "bench", "tasks": ["task"]}],
    }
    body = BenchmarkImportCreate.model_validate(
        {"manifest_sha256": canonical_manifest_sha256(manifest), "manifest": manifest}
    )
    now = datetime.now(UTC)
    row = {
        "id": "bmi_one",
        "owner_id": "usr_test",
        "manifest_sha256": body.manifest_sha256,
        "visibility": "public",
        "description": None,
        "created_at": now,
        "updated_at": now,
    }
    db = MagicMock()
    db.benchmark_imports.get_by_identity.return_value = None
    db.benchmark_imports.create.return_value = row
    db.tasks.get_by_slug.return_value = {
        "id": "task_one",
        "owner_id": "usr_test",
        "visibility": "private",
    }
    db.tasks.get_detail.return_value = {
        "revision": 2,
        "status": "ready",
        "tarball_sha256": "c" * 64,
    }
    db.benchmark_imports.tasks.return_value = [
        {
            "slug": "task",
            "task_id": "task_one",
            "task_revision": 2,
            "pack_path": "packs/task.tar.gz",
            "pack_sha256": "c" * 64,
            "status": "ready",
            "image_ref": "registry.example/task:one",
            "image_digest": "sha256:" + "d" * 64,
            "build_error": None,
            "tarball_object_key": "task_one/rev/2/tarball.tar.gz",
        }
    ]
    db.benchmark_imports.benchmarks.return_value = [
        {
            "slug": "bench",
            "name": "Bench",
            "task_slugs": ["task"],
            "benchmark_id": None,
            "benchmark_revision": None,
        }
    ]

    create_benchmark_import(
        body,
        db,
        CurrentPrincipal(owner_type="user", owner_id="usr_test"),
        Response(),
    )

    db.tasks.update.assert_called_once_with("task_one", visibility="public")


def test_public_import_promotes_reused_benchmark_visibility() -> None:
    now = datetime.now(UTC)
    row = {
        "id": "bmi_one",
        "owner_id": "usr_test",
        "manifest_sha256": "a" * 64,
        "manifest": {"benchmarks": [{"slug": "bench"}]},
        "visibility": "public",
        "description": "CI import",
        "created_at": now,
        "updated_at": now,
    }
    task = {
        "slug": "task",
        "task_id": "task_one",
        "task_revision": 2,
        "pack_path": "packs/task.tar.gz",
        "pack_sha256": "c" * 64,
        "status": "ready",
        "image_ref": "registry.example/task:one",
        "image_digest": "sha256:" + "d" * 64,
        "build_error": None,
        "tarball_object_key": "task_one/rev/2/tarball.tar.gz",
    }
    benchmark = {
        "slug": "bench",
        "name": "Bench",
        "task_slugs": ["task"],
        "benchmark_id": None,
        "benchmark_revision": None,
    }
    db = MagicMock()
    db.benchmark_imports.get.return_value = row
    db.benchmark_imports.tasks.return_value = [task]
    db.benchmark_imports.benchmarks.return_value = [benchmark]
    db.benchmarks.get_by_slug.return_value = {
        "id": "bm_one",
        "visibility": "private",
        "current_revision": 1,
    }
    db.benchmarks.list_tasks.return_value = [{"task_id": "task_one", "task_revision": 2}]

    result = publish_benchmark_import(
        "bmi_one",
        db,
        CurrentPrincipal(owner_type="user", owner_id="usr_test"),
        None,
    )

    assert result.visibility == "public"
    db.benchmarks.update.assert_called_once_with("bm_one", visibility="public")
    db.benchmark_imports.lock_benchmark_slug.assert_called_once_with("bench")
