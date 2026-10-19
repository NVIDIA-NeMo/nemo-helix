# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import tarfile
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

try:
    from scaled_evals.api.auth import CurrentPrincipal
    from scaled_evals.api.routers import benchmark_imports as benchmark_imports_router
    from scaled_evals.api.routers.benchmark_imports import publish_benchmark_import
    from scaled_evals.api.schemas.benchmark_imports import BenchmarkImportCreate
    from scaled_evals.benchmark_import import (
        _validate_pack,
        import_id_from_legacy_state,
        load_import_images,
        write_legacy_import_state,
    )
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


def test_import_schema_and_compatibility_state_preserve_portable_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = BenchmarkImportCreate.model_validate(
        {
            "manifest_sha256": "a" * 64,
            "manifest": {
                "schema_version": 1,
                "catalog_id": "fixture",
                "visibility": "public",
                "source": {"type": "git", "commit": "b" * 40},
                "nullable_extension": None,
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
    assert "nullable_extension" in request.manifest.model_dump(mode="json")
    observed_manifest: dict = {}
    monkeypatch.setattr(
        benchmark_imports_router,
        "canonical_manifest_sha256",
        lambda manifest: observed_manifest.update(manifest) or "b" * 64,
    )
    with pytest.raises(HTTPException, match="manifest digest"):
        benchmark_imports_router.create_benchmark_import(
            request,
            MagicMock(),
            CurrentPrincipal(owner_type="USER", owner_id="owner"),
            MagicMock(),
        )
    assert "nullable_extension" in observed_manifest

    images_path = tmp_path / "images.jsonl"
    images_path.write_text(
        json.dumps(
            {
                "slug": "task",
                "status": "ready",
                "image_ref": "registry.example/task:one",
                "image_digest": "sha256:" + "d" * 64,
                "source_commit": "abc",
            }
        )
        + "\n"
    )
    images = load_import_images([images_path])
    assert images["task"]["source_commit"] == "abc"

    state_path = tmp_path / "state.json"
    write_legacy_import_state(
        state_path,
        {
            "id": "bmi_one",
            "tasks": [
                {
                    "slug": "task",
                    "task_id": "task_1",
                    "task_revision": 2,
                    "status": "ready",
                    "image_ref": images["task"]["image_ref"],
                    "image_digest": images["task"]["image_digest"],
                    "image_metadata": {"source_commit": images["task"]["source_commit"]},
                }
            ],
        },
        phase="finalize",
    )
    assert import_id_from_legacy_state(state_path) == "bmi_one"
    assert json.loads(state_path.read_text())[0]["source_commit"] == "abc"

    oversized = tmp_path / "oversized.tar.gz"
    oversized.write_bytes(b"x")
    checks = _validate_pack(oversized, expected_sha256="0" * 64, max_pack_bytes=0, subject="task")
    assert [(check.code, check.status) for check in checks] == [("pack_size", "failed")]

    directory_task_toml = tmp_path / "directory-task-toml.tar.gz"
    with tarfile.open(directory_task_toml, "w:gz") as archive:
        task_dir = tarfile.TarInfo("tasks/task/task.toml")
        task_dir.type = tarfile.DIRTYPE
        archive.addfile(task_dir)
        dockerfile = b"FROM scratch\n"
        docker_member = tarfile.TarInfo("Dockerfile")
        docker_member.size = len(dockerfile)
        archive.addfile(docker_member, BytesIO(dockerfile))
    with directory_task_toml.open("rb") as raw:
        digest = hashlib.file_digest(raw, "sha256").hexdigest()
    checks = _validate_pack(
        directory_task_toml,
        expected_sha256=digest,
        max_pack_bytes=directory_task_toml.stat().st_size,
        subject="task",
    )
    assert any(check.code == "harbor_structure" and check.status == "failed" for check in checks)


def test_publish_rejects_benchmark_slug_owned_by_another_principal() -> None:
    db = MagicMock()
    db.benchmark_imports.get.return_value = {
        "id": "bmi_one",
        "manifest": {"benchmarks": [{"slug": "bench"}]},
        "visibility": "public",
    }
    db.benchmark_imports.tasks.return_value = [
        {
            "slug": "task",
            "status": "ready",
            "task_id": "task_one",
            "task_revision": 1,
        }
    ]
    db.benchmark_imports.benchmarks.return_value = [{"slug": "bench", "name": "Bench", "task_slugs": ["task"]}]
    db.benchmarks.get_by_slug.return_value = {
        "id": "bm_other",
        "owner_id": "other",
    }

    with pytest.raises(HTTPException) as exc_info:
        publish_benchmark_import(
            "bmi_one",
            db,
            CurrentPrincipal(owner_type="USER", owner_id="owner"),
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["error"]["code"] == "benchmark_slug_owned_by_another_user"
    db.benchmarks.update.assert_not_called()
    db.benchmarks.create_next_revision.assert_not_called()
