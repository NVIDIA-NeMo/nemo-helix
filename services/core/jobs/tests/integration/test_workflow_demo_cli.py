# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named demo workflows through real in-process Files, Secrets and Jobs APIs."""

from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NemoHTTPError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nemo_helix_plugin.jobs.types import HelixJobResponse, UpdateHelixJobRequest
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest
from nhx.core.files.service import FilesService
from nhx.core.jobs.service import JobsService
from nhx.core.secrets.service import SecretsService
from nhx.testing import ClientContext, create_test_client
from pydantic import SecretStr

from demo.workflows import ROOT, DemoSpec, Workflow, create_run, download_run, trigger_run

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def demo_client() -> Iterator[NemoClient]:
    with create_test_client(JobsService, FilesService, SecretsService, client_type=ClientContext) as context:
        SecretsClient.from_client(context.client).create_secret(
            workspace="default", body=HelixSecretCreateRequest(name="demo-model-key", value=SecretStr("test-value"))
        ).data()
        yield context.client


def create(client: NemoClient, workflow: Workflow = Workflow.LAYERS) -> HelixJobResponse:
    return create_run(
        client,
        workflow,
        "demo-" + uuid4().hex[:12],
        workspace="default",
        profile="default",
        image="test-image:local",
        secret="demo-model-key" if workflow == Workflow.MONTAGE else None,
    )


def test_named_layers_runs_are_isolated_and_seeded(demo_client: NemoClient) -> None:
    first, second = create(demo_client), create(demo_client)
    first_spec, second_spec = (DemoSpec.model_validate(job.spec["demo"]) for job in (first, second))
    assert set(first_spec.filesets.values()).isdisjoint(second_spec.filesets.values())
    files_client = FilesClient.from_client(demo_client)
    assert (
        files_client.download_file(workspace="default", name=first.name + "-input", path="message.txt").read()
        == b"Hello from the continuous DAG demo!\n"
    )
    prior = files_client.get_fileset(workspace="default", name=second.name + "-input").data().commit_revision
    result = trigger_run(
        demo_client, first.name, workspace="default", role="input", input_path=None, steps=[], request_id="unused"
    )
    assert first.attempt_id is not None
    assert first.attempt_id in result
    assert files_client.get_fileset(workspace="default", name=second.name + "-input").data().commit_revision == prior
    assert len(first.platform_spec.steps) == 19


def test_montage_seeds_its_own_photo_and_accepts_second_photo(demo_client: NemoClient) -> None:
    job = create(demo_client, Workflow.MONTAGE)
    files_client = FilesClient.from_client(demo_client)
    before = files_client.get_fileset(workspace="default", name=job.name + "-input").data().commit_revision
    trigger_run(
        demo_client,
        job.name,
        workspace="default",
        role="input",
        input_path=ROOT / "demo/opencv/fixtures/birthday-cake.jpg",
        steps=[],
        request_id="unused",
    )
    assert {
        entry.path for entry in files_client.list_files(workspace="default", name=job.name + "-input").data().data
    } == {"sutro-tower.jpg", "birthday-cake.jpg"}
    assert files_client.get_fileset(workspace="default", name=job.name + "-input").data().commit_revision != before
    assert len(job.platform_spec.steps) == 3
    assert job.platform_spec.steps[0].environment is None
    environment = job.platform_spec.steps[-1].environment
    assert environment is not None and environment[0].from_secret is not None
    assert environment[0].from_secret.name == "demo-model-key"


def test_trigger_uses_captured_attempt_metadata_after_job_edit(demo_client: NemoClient) -> None:
    first, second = create(demo_client), create(demo_client)
    jobs_client = JobsClient.from_client(demo_client)
    files_client = FilesClient.from_client(demo_client)
    names = [job.name + "-input" for job in (first, second)]
    revisions = [files_client.get_fileset(workspace="default", name=name).data().commit_revision for name in names]
    jobs_client.update_job(
        workspace="default",
        name=first.name,
        body=UpdateHelixJobRequest(spec=second.spec, platform_spec=first.platform_spec),
    ).data()
    trigger_run(
        demo_client, first.name, workspace="default", role="input", input_path=None, steps=[], request_id="unused"
    )
    assert files_client.get_fileset(workspace="default", name=names[0]).data().commit_revision != revisions[0]
    assert files_client.get_fileset(workspace="default", name=names[1]).data().commit_revision == revisions[1]


def test_manual_trigger_is_attempt_scoped_and_repeated_signal_coalesces(demo_client: NemoClient) -> None:
    job = create(demo_client)
    jobs_client = JobsClient.from_client(demo_client)
    for _ in range(2):
        trigger_run(
            demo_client,
            job.name,
            workspace="default",
            role="input",
            input_path=None,
            steps=["worker-03", "worker-04"],
            request_id="repeat-this-signal",
        )
    attempt = jobs_client.get_job_attempt(workspace="default", job=job.name, attempt="current").data()
    assert attempt.id == job.attempt_id
    assert attempt.status_details["_workflow_requests"] == {
        "worker-03": "repeat-this-signal",
        "worker-04": "repeat-this-signal",
    }
    assert len(list(jobs_client.list_job_attempts(workspace="default", job=job.name).items())) == 1


def test_layers_configuration_upload_uses_expected_filename(demo_client: NemoClient, tmp_path: Path) -> None:
    job = create(demo_client)
    settings = tmp_path / "new-settings.txt"
    settings.write_text("partitions=10; batches=2\n")
    trigger_run(
        demo_client, job.name, workspace="default", role="config", input_path=settings, steps=[], request_id="unused"
    )
    assert (
        FilesClient.from_client(demo_client)
        .download_file(workspace="default", name=job.name + "-config", path="settings.txt")
        .read()
        == settings.read_bytes()
    )


@pytest.mark.parametrize("failure", ["unknown-node", "missing-file", "output-trigger", "mixed-trigger"])
def test_invalid_triggers_leave_input_revision_unchanged(demo_client: NemoClient, failure: str, tmp_path: Path) -> None:
    job = create(demo_client)
    files_client = FilesClient.from_client(demo_client)
    before = files_client.get_fileset(workspace="default", name=job.name + "-input").data().commit_revision
    with pytest.raises(ValueError):
        trigger_run(
            demo_client,
            job.name,
            workspace="default",
            role="output" if failure == "output-trigger" else "input",
            input_path=tmp_path / "missing.txt" if failure in {"missing-file", "mixed-trigger"} else None,
            steps=["missing-node"]
            if failure == "unknown-node"
            else ["worker-03"]
            if failure == "mixed-trigger"
            else [],
            request_id="invalid",
        )
    assert files_client.get_fileset(workspace="default", name=job.name + "-input").data().commit_revision == before


def test_duplicate_run_rejected_before_creating_more_filesets(demo_client: NemoClient) -> None:
    job = create(demo_client)
    files_client = FilesClient.from_client(demo_client)
    before = {fileset.id for fileset in files_client.list_filesets(workspace="default").items()}
    with pytest.raises(ValueError, match="already exists"):
        create_run(
            demo_client,
            Workflow.LAYERS,
            job.name,
            workspace="default",
            profile="default",
            image="test-image",
            secret=None,
        )
    assert {fileset.id for fileset in files_client.list_filesets(workspace="default").items()} == before


def test_download_uses_run_output_fileset(demo_client: NemoClient, tmp_path: Path) -> None:
    job = create(demo_client)
    FilesClient.from_client(demo_client).upload_file(
        workspace="default", name=job.name + "-output", path="report/report.txt", content=b"published report\n"
    ).data()
    download_run(demo_client, job.name, workspace="default", output=tmp_path)
    assert (tmp_path / "output/report/report.txt").read_bytes() == b"published report\n"


def test_terminal_attempt_cannot_be_triggered(demo_client: NemoClient) -> None:
    from nemo_helix_plugin.entities.client import EntitiesClient
    from nhx.common.entities.client import SyncEntityClient
    from nhx.core.jobs.entities import HelixJobAttempt

    job = create(demo_client)
    store = SyncEntityClient(EntitiesClient.from_client(demo_client))
    assert job.attempt_id is not None
    attempt = store.get_by_id(HelixJobAttempt, job.attempt_id)
    attempt.status = HelixJobStatus.CANCELLED
    store.update(attempt)
    with pytest.raises(ValueError, match="terminal"):
        trigger_run(
            demo_client, job.name, workspace="default", role="input", input_path=None, steps=[], request_id="unused"
        )


def test_preexisting_fileset_rejected_before_partial_creation(demo_client: NemoClient) -> None:
    files_client = FilesClient.from_client(demo_client)
    name = "demo-" + uuid4().hex[:12]
    files_client.create_fileset(workspace="default", body=CreateFilesetRequest(name=name + "-output")).data()
    before = {fileset.id for fileset in files_client.list_filesets(workspace="default").items()}
    with pytest.raises(ValueError, match="already exists"):
        create_run(
            demo_client, Workflow.LAYERS, name, workspace="default", profile="default", image="test-image", secret=None
        )
    assert {fileset.id for fileset in files_client.list_filesets(workspace="default").items()} == before


@pytest.mark.parametrize("secret", [None, "missing-model-key"])
def test_missing_montage_secret_creates_no_filesets(demo_client: NemoClient, secret: str | None) -> None:
    files_client = FilesClient.from_client(demo_client)
    before = {fileset.id for fileset in files_client.list_filesets(workspace="default").items()}
    with pytest.raises((ValueError, NemoHTTPError)):
        create_run(
            demo_client,
            Workflow.MONTAGE,
            "demo-" + uuid4().hex[:12],
            workspace="default",
            profile="default",
            image="test-image",
            secret=secret,
        )
    assert {fileset.id for fileset in files_client.list_filesets(workspace="default").items()} == before
