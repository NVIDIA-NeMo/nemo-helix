# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport-agnostic characterization of the artifact-storage contract.

These tests pin the BEHAVIOR that must survive the migration of ``scaled_evals.api.s3``
from raw object storage onto the Files service:

* secret-file / ``.env`` redaction and exclusion on the write path,
* per-artifact sha256 manifest correctness,
* artifact round-trip (put/list/read/delete) through the public ``s3`` functions,
* ``results.tar.gz`` bundling (manifest member + traversal guard + size limits),
* task-pack size probing used by the finalize quota guardrails.

They drive the CURRENT boto3-backed implementation through a complete in-memory S3
fake installed at the ``s3._client`` seam (the same seam ``test_object_store.py`` uses),
so no network, RustFS, or GCS is required. After the Files migration, the SAME assertions
must pass against the Files-backed implementation, proving equivalence.
"""

from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Iterator
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pytest

# Skip the whole module only when the optional plugin package is genuinely absent (default
# sync). Gate on find_spec so a BROKEN import inside an installed package still fails
# collection loudly instead of being masked as "not installed".
if find_spec("scaled_evals") is None:
    pytest.skip("scaled-evals plugin not installed", allow_module_level=True)

from botocore.exceptions import ClientError
from scaled_evals.api import s3
from scaled_evals.api.settings import settings

# ---------------------------------------------------------------------------
# In-memory S3 fake — a complete-enough boto3 client for the s3 module.
# ---------------------------------------------------------------------------


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def close(self) -> None:  # closing() context manager support
        return None

    def iter_chunks(self, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i : i + chunk_size]


class _Paginator:
    def __init__(self, store: dict[str, bytes]) -> None:
        self._store = store

    def paginate(self, *, Bucket: str, Prefix: str) -> Iterator[dict[str, Any]]:  # noqa: N803
        contents = [
            {"Key": key, "Size": len(body), "LastModified": datetime(2026, 1, 1, tzinfo=UTC)}
            for key, body in sorted(self._store.items())
            if key.startswith(Prefix)
        ]
        yield {"Contents": contents}


class _InMemoryS3:
    """Records objects in a dict; supports the boto3 calls s3.py makes on the S3 path."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str | None = None) -> None:  # noqa: N803
        self.store[Key] = bytes(Body)

    def upload_file(self, filename: str, Bucket: str, Key: str, ExtraArgs: dict | None = None) -> None:  # noqa: N803
        self.store[Key] = Path(filename).read_bytes()

    def upload_fileobj(self, fileobj: Any, Bucket: str, Key: str, ExtraArgs: dict | None = None) -> None:  # noqa: N803
        self.store[Key] = fileobj.read()

    def download_file(self, Bucket: str, Key: str, filename: str) -> None:  # noqa: N803
        self._require(Key)
        Path(filename).write_bytes(self.store[Key])

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        self._require(Key)
        return {"Body": _Body(self.store[Key])}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        self._require(Key)
        return {"ContentLength": len(self.store[Key])}

    def delete_object(self, *, Bucket: str, Key: str) -> None:  # noqa: N803
        self.store.pop(Key, None)

    def get_paginator(self, name: str) -> _Paginator:
        assert name == "list_objects_v2"
        return _Paginator(self.store)

    def head_bucket(self, *, Bucket: str) -> None:  # noqa: N803
        return None

    def _require(self, key: str) -> None:
        if key not in self.store:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": key}}, "GetObject")


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> _InMemoryS3:
    fake = _InMemoryS3()
    monkeypatch.setattr(s3, "_client", lambda *_a, **_k: fake)
    monkeypatch.setattr(s3, "_bucket", lambda: "scaled-evals")
    monkeypatch.setattr(s3, "_using_gcs", lambda: False)
    return fake


# ---------------------------------------------------------------------------
# Redaction + exclusion on the write path.
# ---------------------------------------------------------------------------


def test_sync_excludes_secret_files_and_redacts_text(store: _InMemoryS3, tmp_path: Path) -> None:
    root = tmp_path / "job"
    root.mkdir()
    (root / ".env").write_text("SECRET=topsecret\n")
    (root / "nested").mkdir()
    (root / "nested" / "target.env").write_text("TOKEN=abc\n")
    (root / "result.json").write_text(json.dumps({"ok": True}))

    count = s3.sync_directory_to_prefix(root, s3.evaluation_artifact_prefix("ev1"))

    keys = set(store.store)
    # No secret/.env file is ever persisted.
    assert not any(key.endswith(".env") for key in keys)
    # The manifest plus the one non-secret artifact were written.
    assert any(key.endswith(s3.ARTIFACT_MANIFEST_PATH) for key in keys)
    assert any(key.endswith("result.json") for key in keys)
    assert count == 2  # 1 artifact + manifest


def test_manifest_records_sha256_for_each_artifact(store: _InMemoryS3, tmp_path: Path) -> None:
    import hashlib

    root = tmp_path / "job"
    root.mkdir()
    payload = b'{"metric": 0.9}'
    (root / "score.json").write_bytes(payload)

    s3.sync_directory_to_prefix(root, s3.evaluation_artifact_prefix("ev2"))

    manifest_key = next(k for k in store.store if k.endswith(s3.ARTIFACT_MANIFEST_PATH))
    manifest = json.loads(store.store[manifest_key])
    assert manifest["schema_version"] == "scaled-evals-artifacts-v1"
    entry = next(e for e in manifest["files"] if e["path"] == "score.json")
    assert entry["size_bytes"] == len(payload)
    assert entry["sha256"] == f"sha256:{hashlib.sha256(payload).hexdigest()}"


# ---------------------------------------------------------------------------
# Artifact round-trip through the public functions.
# ---------------------------------------------------------------------------


def test_json_and_text_object_roundtrip(store: _InMemoryS3) -> None:
    key = s3.evaluation_artifact_key("ev3", "provenance.json")
    s3.put_json_object(key, {"a": 1})
    assert s3.read_json_object(key) == {"a": 1}

    log_key = s3.evaluation_live_log_key("ev3", 1)
    s3.put_text_object(log_key, "hello\n")
    assert s3.read_text_object_if_exists(log_key) == "hello\n"
    assert s3.read_text_object_if_exists(s3.evaluation_live_log_key("ev3", 99)) is None


def test_object_exists_and_delete(store: _InMemoryS3) -> None:
    key = s3.evaluation_artifact_key("ev4", "x.txt")
    assert s3.object_exists(key) is False
    s3.put_text_object(key, "y")
    assert s3.object_exists(key) is True
    assert s3.object_size(key) == 1
    s3.delete_object(key)
    assert s3.object_exists(key) is False


def test_list_objects_returns_metadata(store: _InMemoryS3) -> None:
    prefix = s3.evaluation_artifact_prefix("ev5")
    s3.put_text_object(f"{prefix}a.txt", "aa")
    s3.put_text_object(f"{prefix}b.txt", "bbb")
    listed = {item["key"]: item["size_bytes"] for item in s3.list_objects(prefix)}
    assert listed == {f"{prefix}a.txt": 2, f"{prefix}b.txt": 3}


def test_unsafe_artifact_paths_are_rejected() -> None:
    # Traversal, doubled/dot segments, and backslashes are rejected outright.
    for bad in ("../escape", "a/../../b", "a//b", "a/./b", "back\\slash"):
        with pytest.raises(ValueError):
            s3.evaluation_artifact_key("ev6", bad)


def test_leading_slash_is_normalized_not_rejected() -> None:
    # A leading slash is stripped (not treated as an absolute-path escape), so the
    # artifact still lands under the evaluation's prefix. Pin this so the Files-backed
    # implementation preserves the same normalization.
    assert s3.evaluation_artifact_key("ev6", "/abs") == f"{s3.evaluation_artifact_prefix('ev6')}abs"


# ---------------------------------------------------------------------------
# results.tar.gz bundling.
# ---------------------------------------------------------------------------


def test_build_archive_from_directory_bundles_manifest_and_redacts(store: _InMemoryS3, tmp_path: Path) -> None:
    root = tmp_path / "job"
    root.mkdir()
    (root / ".env").write_text("SECRET=nope\n")
    (root / "out.json").write_text(json.dumps({"v": 1}))

    result = s3.build_evaluation_archive_from_directory("ev7", root)

    assert result["object_key"] == s3.evaluation_archive_key("ev7")
    archive_bytes = store.store[result["object_key"]]
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
        names = set(tar.getnames())
    assert f"artifacts/{s3.ARTIFACT_MANIFEST_PATH}" in names
    assert "artifacts/out.json" in names
    # Redacted/secret files never enter the bundle.
    assert not any(name.endswith(".env") for name in names)


def test_archive_enforces_max_files(store: _InMemoryS3, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "job"
    root.mkdir()
    for i in range(3):
        (root / f"f{i}.json").write_text("{}")
    monkeypatch.setattr(settings, "evaluation_archive_max_files", 2)
    with pytest.raises(s3.ArchiveBuildError):
        s3.build_evaluation_archive_from_directory("ev8", root)


def test_missing_root_sync_is_noop(store: _InMemoryS3, tmp_path: Path) -> None:
    assert s3.sync_directory_to_prefix(tmp_path / "does-not-exist", s3.evaluation_artifact_prefix("ev9")) == 0


# ---------------------------------------------------------------------------
# Task-pack size probe (finalize quota guardrail input).
# ---------------------------------------------------------------------------


def test_upload_file_and_size_probe(store: _InMemoryS3, tmp_path: Path) -> None:
    pack = tmp_path / "tarball.tar.gz"
    pack.write_bytes(b"x" * 4096)
    key = "task_abc/rev/1/tarball.tar.gz"
    size = s3.upload_file(pack, key, content_type="application/gzip")
    assert size == 4096
    assert s3.object_size(key) == 4096
    assert s3.object_exists(key) is True


def test_download_object_roundtrip(store: _InMemoryS3, tmp_path: Path) -> None:
    key = "task_abc/rev/1/tarball.tar.gz"
    store.store[key] = b"packbytes"
    dest = tmp_path / "out.tar.gz"
    s3.download_object(key, str(dest))
    assert dest.read_bytes() == b"packbytes"
