# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import httpx
import pytest
from nemo_evaluator.jobs.environment_stage import EnvironmentStageJob
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import LocalJobResults
from pytest_mock import MockerFixture


def _context(tmp_path: Path) -> JobContext:
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    return JobContext(
        workspace="dev",
        job_id="job-1",
        storage=StoragePaths(ephemeral=tmp_path / "ephemeral", persistent=persistent),
        results=LocalJobResults(tmp_path / "results"),
    )


def _task_client(mocker: MockerFixture) -> NemoClient:
    return NemoClient(
        base_url="http://platform.test",
        workspace="dev",
        http_client=mocker.Mock(spec=httpx.Client),
    )


def test_stages_environment_at_fixed_persistent_path(tmp_path: Path, mocker: MockerFixture) -> None:
    ctx = _context(tmp_path)
    task_client = _task_client(mocker)

    def download_contents(*, client: NemoClient, workspace: str, fileset: str, destination: Path) -> None:
        assert client is task_client
        assert workspace == "shared"
        assert fileset == "custom-gym"
        Path(destination, "nemo-environment.yaml").write_text("format: wheels-v1\n")

    download_fileset_contents = mocker.patch(
        "nemo_evaluator.jobs.environment_stage._download_fileset_contents",
        side_effect=download_contents,
    )

    result = EnvironmentStageJob().run(
        {"environment": "shared/custom-gym"},
        ctx=ctx,
        client=task_client,
    )

    download_fileset_contents.assert_called_once_with(
        client=task_client,
        fileset="custom-gym",
        workspace="shared",
        destination=ctx.storage.persistent / ".environment-staging",
    )
    assert (ctx.storage.persistent / "environment" / "nemo-environment.yaml").is_file()
    assert (ctx.storage.persistent / "workspace").is_dir()
    assert not (ctx.storage.persistent / ".environment-staging").exists()
    assert result["path"] == str(ctx.storage.persistent / "environment")


def test_staged_fileset_preserves_wheels_tree(tmp_path: Path, mocker: MockerFixture) -> None:
    ctx = _context(tmp_path)
    task_client = _task_client(mocker)
    wheel_name = "xmltodict-1.0.4-py3-none-any.whl"
    config_rel = Path("resources_servers") / "structeval" / "configs" / "structeval_nonrenderable.yaml"

    def download_contents(*, client: NemoClient, workspace: str, fileset: str, destination: Path) -> None:
        assert client is task_client
        assert workspace == "dev"
        assert fileset == "structeval-wheels"
        root = destination
        (root / "nemo-environment.yaml").write_text("format: wheels-v1\n")
        config = root / config_rel
        config.parent.mkdir(parents=True)
        config.write_text("structeval: {}\n")
        wheels = root / "wheels"
        wheels.mkdir()
        (wheels / wheel_name).write_bytes(b"wheel")

    mocker.patch(
        "nemo_evaluator.jobs.environment_stage._download_fileset_contents",
        side_effect=download_contents,
    )

    EnvironmentStageJob().run({"environment": "dev/structeval-wheels"}, ctx=ctx, client=task_client)

    environment = ctx.storage.persistent / "environment"
    assert (environment / "nemo-environment.yaml").is_file()
    assert (environment / config_rel).is_file()
    assert (environment / "wheels" / wheel_name).read_bytes() == b"wheel"
    assert not (ctx.storage.persistent / ".environment-staging").exists()


def test_failed_download_removes_partial_staging_without_replacing_environment(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    ctx = _context(tmp_path)
    environment = ctx.storage.persistent / "environment"
    environment.mkdir()
    (environment / "existing.txt").write_text("complete")
    task_client = _task_client(mocker)

    def fail_download(*, client: NemoClient, workspace: str, fileset: str, destination: Path) -> None:
        assert client is task_client
        assert workspace == "dev"
        assert fileset == "custom-gym"
        Path(destination, "partial.txt").write_text("partial")
        raise RuntimeError("download failed")

    mocker.patch(
        "nemo_evaluator.jobs.environment_stage._download_fileset_contents",
        side_effect=fail_download,
    )

    with pytest.raises(RuntimeError, match="download failed"):
        EnvironmentStageJob().run(
            {"environment": "custom-gym"},
            ctx=ctx,
            client=task_client,
        )

    assert (environment / "existing.txt").read_text() == "complete"
    assert not (ctx.storage.persistent / ".environment-staging").exists()
    assert (ctx.storage.persistent / "workspace").is_dir()


def test_run_passes_the_declared_typed_client(tmp_path: Path, mocker: MockerFixture) -> None:
    """The sync job entrypoint uses the typed client declared by the job."""
    received: dict[str, object] = {}

    def download_contents(*, client: object, workspace: str, fileset: str, destination: Path) -> None:
        received["client"] = client
        Path(destination, "nemo-environment.yaml").write_text("format: wheels-v1\n")

    mocker.patch(
        "nemo_evaluator.jobs.environment_stage._download_fileset_contents",
        side_effect=download_contents,
    )
    client = _task_client(mocker)
    ctx = _context(tmp_path)

    result = EnvironmentStageJob().run(
        {"environment": "shared/custom-gym"},
        ctx=ctx,
        client=client,
    )

    assert result["status"] == "completed"
    assert received["client"] is client
