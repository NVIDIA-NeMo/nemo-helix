# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export contracts requested in review: typed provenance and verifiable bytes."""

import hashlib
import json
from pathlib import Path
from uuid import UUID

import pytest

pytest.importorskip("scaled_evals")

from scaled_evals.benchmark_archive import BenchmarkArchiveError
from scaled_evals.models.benchmark_archives import BenchmarkArchiveManifest
from test_benchmark_archives import build, harbor_files, tar_bytes


def with_file_manifest(files: dict) -> dict:
    entries = []
    for path, value in files.items():
        body = value if isinstance(value, bytes) else json.dumps(value).encode()
        entries.append({"path": path, "size_bytes": len(body), "sha256": f"sha256:{hashlib.sha256(body).hexdigest()}"})
    return {**files, "scaled-evals-manifest.json": {"schema_version": "scaled-evals-artifacts-v1", "files": entries}}


def test_typed_manifest_preserves_complete_public_provenance(monkeypatch, tmp_path: Path) -> None:
    source = harbor_files()
    provenance = {
        "schema_version": "scaled-evals-provenance-v2",
        "evaluation_id": "ev_1",
        "config": {
            "framework": "harbor",
            "framework_version": "0.20.0",
            "framework_adapter_version": "adapter-v1",
            "agent_bundle": {"bundle_id": "ab_1", "agent_version": "1.2", "image_digest": "sha256:" + "a" * 64},
        },
        "runtime": {"name": "sandbox_k8s", "runner_image_ref": "registry.example/runner:1"},
        "execution_inputs": {"profiles": {"framework": {"config_sha256": "b" * 64}}},
        "future_public_metadata": {"runner_source_revision": "c" * 40},
    }
    source["scaled-evals-provenance.json"] = provenance
    files = build(monkeypatch, tmp_path, [tar_bytes(with_file_manifest(source))])
    manifest = BenchmarkArchiveManifest.model_validate_json(files["bmr_1/scaled-evals-benchmark-archive.json"])
    member = manifest.members[0]
    assert member.provenance == provenance
    assert member.artifact_integrity == "verified"
    assert json.loads(files[f"bmr_1/{member.provenance_path}"]) == provenance
    assert str(manifest.harbor_job_id) == json.loads(files["bmr_1/result.json"])["id"]
    indexed = {entry.path for entry in manifest.files}
    assert indexed == {
        path.removeprefix("bmr_1/") for path in files if not path.endswith("scaled-evals-benchmark-archive.json")
    }
    for entry in manifest.files:
        body = files[f"bmr_1/{entry.path}"]
        assert entry.size_bytes == len(body)
        assert entry.sha256 == hashlib.sha256(body).hexdigest()


def test_legacy_member_explicitly_reports_missing_integrity_manifest(monkeypatch, tmp_path: Path) -> None:
    files = build(monkeypatch, tmp_path, [tar_bytes(harbor_files())])
    manifest = BenchmarkArchiveManifest.model_validate_json(files["bmr_1/scaled-evals-benchmark-archive.json"])
    assert manifest.members[0].artifact_integrity == "manifest_missing"


@pytest.mark.parametrize("change", ["same_size", "missing", "extra", "unsafe_path", "duplicate"])
def test_producer_manifest_rejects_changed_inputs(monkeypatch, tmp_path: Path, change: str) -> None:
    original = harbor_files()
    source = with_file_manifest(original)
    target = "task__same/verifier/reward.txt"
    if change == "same_size":
        source[target] = b"0"
        assert len(source[target]) == len(original[target])
    elif change == "missing":
        del source[target]
    elif change == "extra":
        source["unexpected.txt"] = b"extra"
    elif change == "unsafe_path":
        source["scaled-evals-manifest.json"]["files"][0]["path"] = "../escape"
    else:
        source["scaled-evals-manifest.json"]["files"].append(source["scaled-evals-manifest.json"]["files"][0])
    with pytest.raises(BenchmarkArchiveError, match="manifest|checksum"):
        build(monkeypatch, tmp_path, [tar_bytes(source)])
    assert not (tmp_path / "result.tar.gz").exists()


def test_harbor_job_uuid_is_stable_per_export_generation() -> None:
    from scaled_evals.archive_validation import harbor_job_uuid

    first = harbor_job_uuid("bmr_prod123", "generation-one")
    assert isinstance(first, UUID)
    assert first == harbor_job_uuid("bmr_prod123", "generation-one")
    assert first != harbor_job_uuid("bmr_prod123", "generation-two")
    assert first != harbor_job_uuid("bmr_other", "generation-one")


@pytest.mark.parametrize("identifier", ["../escape", "with/slash", "with\\slash", "/absolute", "", ".", "space id"])
def test_archive_ids_are_path_components_not_magic_test_ids(identifier: str) -> None:
    from scaled_evals.archive_validation import validate_archive_id

    with pytest.raises(BenchmarkArchiveError, match="identifier"):
        validate_archive_id(identifier)
    assert validate_archive_id("bmr_production-123") == "bmr_production-123"


def test_manifest_rejects_unknown_fields_and_invalid_counts(monkeypatch, tmp_path: Path) -> None:
    from pydantic import ValidationError

    files = build(monkeypatch, tmp_path, [tar_bytes(harbor_files())])
    data = json.loads(files["bmr_1/scaled-evals-benchmark-archive.json"])
    with pytest.raises(ValidationError):
        BenchmarkArchiveManifest.model_validate({**data, "memberz": []})
    with pytest.raises(ValidationError):
        BenchmarkArchiveManifest.model_validate({**data, "n_declared_trials": -1})
    # The additional v1 fields do not make earlier exports unreadable.
    data.pop("harbor_job_id")
    data.pop("files")
    for member in data["members"]:
        member.pop("artifact_integrity")
        member.pop("provenance")
        member.pop("provenance_path")
    legacy = BenchmarkArchiveManifest.model_validate(data)
    assert legacy.harbor_job_id is None
    assert legacy.members[0].artifact_integrity == "manifest_missing"


def test_export_core_has_no_switchyard_dependency() -> None:
    import ast

    import scaled_evals.benchmark_archive as module

    tree = ast.parse(Path(module.__file__).read_text())
    assert not any(isinstance(node, ast.ImportFrom) and "switchyard" in (node.module or "") for node in ast.walk(tree))
