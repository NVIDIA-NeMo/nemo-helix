# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The job-owned copy of a registered agent's Ethos files: taken at submit, dropped when the run is done."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from nemo_evaluator.filesets import FilesetRef
from nemo_evaluator.jobs.agent_files_snapshot import (
    discard_registered_agent_files,
    discard_registered_agent_files_sync,
    snapshot_registered_agent_files,
)
from nemo_helix_plugin.client.client import NemoClient
from pytest_mock import MockerFixture


def _files(mocker: MockerFixture) -> Any:
    files = mocker.Mock()
    files.create_fileset = AsyncMock()
    files.upload_file = AsyncMock()
    files.delete_fileset = AsyncMock()
    mocker.patch("nemo_evaluator.jobs.agent_files_snapshot.AsyncFilesetFileSystem")
    return files


def _fake_download(tree: dict[str, str]):
    async def download(ref: FilesetRef, destination: str, **_: Any) -> Path:
        root = Path(destination) / ref.root
        for relative, content in tree.items():
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).write_text(content)
        return root

    return download


async def test_snapshot_copies_every_file_into_a_fresh_fileset(mocker: MockerFixture) -> None:
    files = _files(mocker)
    mocker.patch(
        "nemo_evaluator.jobs.agent_files_snapshot._download_fileset_ref",
        side_effect=_fake_download({"skills/a/SKILL.md": "a", "prompts/system.md": "s"}),
    )

    ref = await snapshot_registered_agent_files(files, workspace="dev", agent_name="calc")

    name = ref.root.removeprefix("dev/")
    assert name.startswith("agent-files-") and len(name) == len("agent-files-") + 12
    assert sorted((c.kwargs["path"], c.kwargs["content"]) for c in files.upload_file.await_args_list) == [
        ("prompts/system.md", b"s"),
        ("skills/a/SKILL.md", b"a"),
    ]
    assert {c.kwargs["name"] for c in files.upload_file.await_args_list} == {name}
    files.delete_fileset.assert_not_awaited()


async def test_a_failed_copy_deletes_the_half_made_snapshot(mocker: MockerFixture) -> None:
    files = _files(mocker)
    mocker.patch(
        "nemo_evaluator.jobs.agent_files_snapshot._download_fileset_ref",
        side_effect=_fake_download({"skills/a/SKILL.md": "a"}),
    )
    files.upload_file.side_effect = RuntimeError("storage unavailable")

    with pytest.raises(RuntimeError, match="storage unavailable"):
        await snapshot_registered_agent_files(files, workspace="dev", agent_name="calc")

    deleted = files.delete_fileset.await_args.kwargs
    assert deleted["workspace"] == "dev" and deleted["name"] == files.create_fileset.await_args.kwargs["body"].name


async def test_discard_logs_rather_than_fails_the_job(mocker: MockerFixture, caplog: pytest.LogCaptureFixture) -> None:
    files = _files(mocker)
    files.delete_fileset.side_effect = RuntimeError("gone already")

    with caplog.at_level("WARNING"):
        await discard_registered_agent_files(files, FilesetRef(root="dev/agent-files-0123abcd4567"))

    assert any("agent-files-0123abcd4567" in r.message for r in caplog.records)


def test_discard_from_the_evaluation_step_uses_the_sync_client_it_was_given(mocker: MockerFixture) -> None:
    files_client = mocker.Mock()
    mocker.patch("nemo_evaluator.jobs.agent_files_snapshot.FilesClient.from_client", return_value=files_client)

    discard_registered_agent_files_sync(mocker.Mock(spec=NemoClient), FilesetRef(root="dev/agent-files-0123abcd4567"))

    files_client.delete_fileset.assert_called_once_with(workspace="dev", name="agent-files-0123abcd4567")
