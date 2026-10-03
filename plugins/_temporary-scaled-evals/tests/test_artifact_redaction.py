# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify uploaded JSON artifacts remain valid after secret redaction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

try:
    from scaled_evals.api import artifacts
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


def test_json_artifact_upload_redacts_without_corrupting_json_or_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "job"
    root.mkdir()
    observation = json.dumps({"output": "password=synthetic-value", "exit_code": 0})
    source = json.dumps({"observation": observation, "reward": 1.0}, indent=2).encode()
    (root / "trajectory.json").write_bytes(source)
    (root / "events.jsonl").write_text('{"password":"synthetic-value"}\n{"step":2}\n')

    # Capture what the Files-backed transport would upload. Regular files go through
    # upload_file (from a local staged path); the manifest is written via put_bytes.
    uploaded: dict[str, bytes] = {}

    def fake_upload_file(path: Path, object_key: str, *, content_type: str) -> int:
        data = Path(path).read_bytes()
        uploaded[object_key] = data
        return len(data)

    def fake_put_bytes(object_key: str, body: bytes, *, content_type: str | None = None) -> None:
        uploaded[object_key] = body

    monkeypatch.setattr(artifacts._files_backend, "upload_file", fake_upload_file)
    monkeypatch.setattr(artifacts._files_backend, "put_bytes", fake_put_bytes)

    assert artifacts.sync_directory_to_prefix(root, "evaluations/ev_json/artifacts/") == 3

    trajectory = uploaded["evaluations/ev_json/artifacts/trajectory.json"]
    assert "synthetic-value" not in trajectory.decode()
    parsed = json.loads(trajectory)
    assert json.loads(parsed["observation"])["output"] == "password=<redacted>"
    assert parsed["reward"] == 1.0
    assert (root / "trajectory.json").read_bytes() == source
    assert [json.loads(line) for line in uploaded["evaluations/ev_json/artifacts/events.jsonl"].splitlines()] == [
        {"password": "<redacted>"},
        {"step": 2},
    ]
    manifest = json.loads(uploaded["evaluations/ev_json/artifacts/" + artifacts.ARTIFACT_MANIFEST_PATH])
    for item in manifest["files"]:
        body = uploaded["evaluations/ev_json/artifacts/" + item["path"]]
        assert item["sha256"] == "sha256:" + hashlib.sha256(body).hexdigest()
        assert item["size_bytes"] == len(body)
