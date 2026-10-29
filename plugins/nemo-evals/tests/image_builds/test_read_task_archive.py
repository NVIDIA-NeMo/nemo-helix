# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``read_task_archive``: the archive's listing, plus the parsed compose file for a task that has one."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nemo_evals.api.task_definitions.harbor import HarborArchiveSource
from nemo_evals.harbor import archive
from nemo_evals.image_builds.planning import ImagePlanError
from nemo_evals.image_builds.runner import read_task_archive

pytest.importorskip("harbor")

SOURCE = HarborArchiveSource(fileset_ref="default/files#task/task_archive", files_hash="a" * 64)


def _serve(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, files: dict[str, str]) -> None:
    """Make the Files download return a packed task directory holding ``files`` besides a valid task's own."""
    root = tmp_path / "task"
    for name, text in {
        "task.toml": "",
        "instruction.md": "Do it",
        "environment/Dockerfile": "FROM python:3.12-slim\n",
        "tests/test.sh": "exit 0",
        **files,
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    packed = tmp_path / "task.tar.gz"
    archive.pack_task(root, packed)

    async def download(client, ref, output, *, limit, expected_digest):
        """Stand in for the Files download: write the packed task as the downloaded archive."""
        output.write(packed.read_bytes())

    monkeypatch.setattr("nemo_evals.image_builds.runner.download_verified_async", download)


async def test_a_compose_task_is_read_with_its_parsed_compose_file(tmp_path, monkeypatch) -> None:
    compose = "services:\n  cache:\n    image: ${REGISTRY:-docker.io}/library/redis:7\n  api:\n    build: ./api\n"
    _serve(
        monkeypatch,
        tmp_path,
        {
            "environment/docker-compose.yaml": compose,
            "environment/.env": "REGISTRY=mirror.example.com\n",
            "environment/api/Dockerfile": "FROM python:3.12-slim\n",
        },
    )

    result = await read_task_archive(MagicMock(), SOURCE)

    assert result.compose == compose
    assert result.compose_task is not None
    assert [
        (name, getattr(service.source, "reference", None)) for name, service in result.compose_task.services.items()
    ] == [
        ("cache", "mirror.example.com/library/redis:7"),
        ("api", None),
    ]
    assert result.compose_task.main.entrypoint is None  # planning leaves main's Dockerfile unread


async def test_a_task_without_a_compose_file_is_only_listed(tmp_path, monkeypatch) -> None:
    _serve(monkeypatch, tmp_path, {})
    result = await read_task_archive(MagicMock(), SOURCE)
    assert result.compose is None and result.compose_task is None
    assert result.has("environment/Dockerfile")


async def test_a_compose_file_the_pod_cannot_run_is_an_invalid_compose_file(tmp_path, monkeypatch) -> None:
    _serve(
        monkeypatch,
        tmp_path,
        {"environment/docker-compose.yaml": "services:\n  db:\n    image: redis\n    privileged: true\n"},
    )
    with pytest.raises(ImagePlanError) as error:
        await read_task_archive(MagicMock(), SOURCE)
    assert error.value.code == "invalid_compose_file"
    assert "compose_unsupported" in error.value.detail
