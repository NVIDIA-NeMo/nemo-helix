# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import httpx
import pytest
from nemo_evaluator.jobs.environment_stage import EnvironmentStageJob
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import LocalJobResults
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


def test_a_registered_gym_agent_stages_its_files_and_package_on_top_of_the_environment(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """One environment tree: the user's package first, the agent's Ethos files and generated package added to it."""
    ctx = _context(tmp_path)
    task_client = _task_client(mocker)

    def download_contents(*, client: NemoClient, workspace: str, fileset: str, destination: Path) -> None:
        if fileset == "custom-gym":
            Path(destination, "nemo-environment.yaml").write_text(
                "format: wheels-v1\nconfig_paths: [resources_servers/g/configs/g.yaml]\nmetadata: {name: g}\n"
            )
            Path(destination, "resources_servers", "g", "configs").mkdir(parents=True)
            Path(destination, "resources_servers", "g", "configs", "g.yaml").write_text("g: {}\n")
            Path(destination, "wheels").mkdir()
            Path(destination, "wheels", "g-1.0-py3-none-any.whl").write_bytes(b"")
        else:
            assert (workspace, fileset) == ("dev", "calc-ethos")
            Path(destination, "skills", "a").mkdir(parents=True)
            Path(destination, "skills", "a", "SKILL.md").write_text("# a")

    mocker.patch("nemo_evaluator.jobs.environment_stage._download_fileset_contents", side_effect=download_contents)
    downloaded: list = []
    mocker.patch(
        "nemo_evaluator.jobs.gym_registered_agent_package.download_wheels",
        side_effect=lambda reqs, cons, dest, pv, plat: (
            downloaded.append(list(reqs)) or (dest / "nemo_fabric-0.3.0-py3-none-any.whl").write_bytes(b"")
        ),
    )

    result = EnvironmentStageJob().run(
        {
            "environment": "shared/custom-gym",
            "agent_files": "dev/calc-ethos",
            "gym_registered_agent": {
                "agent": "dev/calc",
                "resolved_config": {"harness": {"adapter_id": "x"}, "skills": {"paths": ["skills/a"]}},
                "requirements": ["nemo-fabric[deepagents,relay]==0.3.0"],
            },
        },
        ctx=ctx,
        client=task_client,
    )

    root = ctx.storage.persistent / "environment"
    assert (root / "resources_servers" / "g" / "configs" / "g.yaml").is_file()
    assert (root / "responses_api_agents" / "nemo_registered_agent" / "app.py").is_file()
    assert (
        root
        / "responses_api_agents"
        / "nemo_registered_agent"
        / "agents"
        / "registered_calc"
        / "skills"
        / "a"
        / "SKILL.md"
    ).is_file()
    assert (root / "wheels" / "nemo_fabric-0.3.0-py3-none-any.whl").is_file()
    assert downloaded == [["nemo-fabric[deepagents,relay]==0.3.0"]]
    assert not (ctx.storage.persistent / ".agent-files-staging").exists()
    assert result["agent_files"] == "dev/calc-ethos" and result["environment"] == "shared/custom-gym"


def test_stage_spec_needs_something_to_stage() -> None:
    from nemo_evaluator.jobs.environment_stage import EnvironmentStageSpec

    with pytest.raises(ValueError, match="nothing to stage"):
        EnvironmentStageSpec()
    with pytest.raises(ValueError, match="agent_files"):
        EnvironmentStageSpec(environment="ws/env", agent_files="ws/ethos")
