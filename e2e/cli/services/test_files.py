# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for files/filesets CLI commands."""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

import pytest
from nhx.testing import NemoRun, assert_exit_0

pytestmark = [pytest.mark.timeout(600)]


def test_files_lifecycle(workspace: str, nemo_run: NemoRun) -> None:
    """Full files/filesets lifecycle via CLI: fileset create/list/get/update/delete,
    file upload (flat + nested + directory), list, download, delete."""
    fileset_name = f"e2e-cli-files-{uuid.uuid4().hex[:8]}"

    result = nemo_run("files", "filesets", "create", fileset_name, workspace=workspace)
    assert_exit_0(result, "filesets create failed")
    assert json.loads(result.stdout).get("name") == fileset_name

    result = nemo_run("files", "filesets", "list", workspace=workspace)
    assert_exit_0(result, "filesets list failed")
    assert any(f["name"] == fileset_name for f in json.loads(result.stdout).get("data", []))

    result = nemo_run("files", "filesets", "get", fileset_name, workspace=workspace)
    assert_exit_0(result, "filesets get failed")
    assert json.loads(result.stdout).get("name") == fileset_name

    result = nemo_run(
        "files",
        "filesets",
        "update",
        fileset_name,
        "--description",
        "updated description",
        workspace=workspace,
    )
    assert_exit_0(result, "filesets update failed")
    assert json.loads(result.stdout).get("description") == "updated description"

    with tempfile.TemporaryDirectory() as tmpdir:
        flat_content = b"Hello from e2e CLI test!"
        flat_local = Path(tmpdir) / "flat.txt"
        flat_local.write_bytes(flat_content)
        result = nemo_run(
            "files",
            "upload",
            str(flat_local),
            fileset_name,
            "--remote-path",
            "flat.txt",
            workspace=workspace,
        )
        assert_exit_0(result, "files upload (flat) failed")

        nested_content = b"Nested file content"
        nested_local = Path(tmpdir) / "nested.txt"
        nested_local.write_bytes(nested_content)
        result = nemo_run(
            "files",
            "upload",
            str(nested_local),
            fileset_name,
            "--remote-path",
            "folder/subfolder/nested.txt",
            workspace=workspace,
        )
        assert_exit_0(result, "files upload (nested) failed")

        upload_dir = Path(tmpdir) / "upload"
        upload_dir.mkdir()
        (upload_dir / "dir_a.txt").write_text("content_a")
        (upload_dir / "sub").mkdir()
        (upload_dir / "sub" / "dir_b.txt").write_text("content_b")
        result = nemo_run(
            "files",
            "upload",
            str(upload_dir) + "/",
            fileset_name,
            "--remote-path",
            "dir/",
            workspace=workspace,
        )
        assert_exit_0(result, "files upload (dir) failed")

        result = nemo_run("files", "list", fileset_name, workspace=workspace)
        assert_exit_0(result, "files list failed")
        data = json.loads(result.stdout)
        items = data if isinstance(data, list) else data.get("data", data)
        paths = {item.get("path") for item in items if isinstance(item, dict)}
        assert "flat.txt" in paths
        assert "folder/subfolder/nested.txt" in paths
        assert "dir/dir_a.txt" in paths
        assert "dir/sub/dir_b.txt" in paths
        flat_item = next(i for i in items if i.get("path") == "flat.txt")
        assert flat_item.get("size") == len(flat_content)

        download_flat = Path(tmpdir) / "downloaded_flat.txt"
        result = nemo_run(
            "files",
            "download",
            fileset_name,
            "--remote-path",
            "flat.txt",
            "--output",
            str(download_flat),
            workspace=workspace,
        )
        assert_exit_0(result, "files download (flat) failed")
        assert download_flat.read_bytes() == flat_content

        download_nested = Path(tmpdir) / "downloaded_nested.txt"
        result = nemo_run(
            "files",
            "download",
            fileset_name,
            "--remote-path",
            "folder/subfolder/nested.txt",
            "--output",
            str(download_nested),
            workspace=workspace,
        )
        assert_exit_0(result, "files download (nested) failed")
        assert download_nested.read_bytes() == nested_content

        download_dir = Path(tmpdir) / "downloaded_dir"
        download_dir.mkdir()
        result = nemo_run(
            "files",
            "download",
            fileset_name,
            "--remote-path",
            "dir/",
            "--output",
            str(download_dir) + "/",
            workspace=workspace,
        )
        assert_exit_0(result, "files download (dir) failed")
        assert (download_dir / "dir_a.txt").read_text() == "content_a"
        assert (download_dir / "sub" / "dir_b.txt").read_text() == "content_b"

        result = nemo_run(
            "files",
            "delete",
            fileset_name,
            "--remote-path",
            "flat.txt",
            workspace=workspace,
        )
        assert_exit_0(result, "files delete failed")

        result = nemo_run("files", "list", fileset_name, workspace=workspace)
        assert_exit_0(result, "files list failed")
        data = json.loads(result.stdout)
        items = data if isinstance(data, list) else data.get("data", data)
        assert "flat.txt" not in {i.get("path") for i in items if isinstance(i, dict)}

    result = nemo_run("files", "filesets", "delete", fileset_name, workspace=workspace)
    assert_exit_0(result, "filesets delete failed")

    result = nemo_run("files", "filesets", "list", workspace=workspace)
    assert_exit_0(result, "filesets list failed")
    assert all(f["name"] != fileset_name for f in json.loads(result.stdout).get("data", []))
