# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify uploaded JSON artifacts remain valid after secret redaction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:
    from scaled_evals.api import s3
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


def test_json_artifact_upload_redacts_without_corrupting_json_or_source(tmp_path: Path) -> None:
    root = tmp_path / "job"
    root.mkdir()
    observation = json.dumps({"output": "password=synthetic-value", "exit_code": 0})
    source = json.dumps({"observation": observation, "reward": 1.0}, indent=2).encode()
    (root / "trajectory.json").write_bytes(source)
    (root / "events.jsonl").write_text('{"password":"synthetic-value"}\n{"step":2}\n')
    uploaded: dict[str, bytes] = {}
    client = MagicMock()
    client.upload_file.side_effect = lambda filename, _bucket, key: uploaded.update({key: Path(filename).read_bytes()})

    with patch.object(s3, "_client", return_value=client):
        assert s3.sync_directory_to_prefix(root, "evaluations/ev_json/artifacts/") == 3

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
    manifest = json.loads(client.put_object.call_args_list[0].kwargs["Body"])
    for item in manifest["files"]:
        body = uploaded["evaluations/ev_json/artifacts/" + item["path"]]
        assert item["sha256"] == "sha256:" + hashlib.sha256(body).hexdigest()
        assert item["size_bytes"] == len(body)
