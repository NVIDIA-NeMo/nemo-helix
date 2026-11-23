# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The builder's remote backend on a kind cluster, with a stand-in for a remote build service.

No build service is called. The remote-build step runs the release's own ``nhx-builder-tasks`` image, and its
program is ``e2e/resources/builder_remote/remote_build_step.py``, which this module puts in a ConfigMap that the
step's Jobs profile mounts. That program runs the public step driver around a service that ends each build as a file
in its build context says. Everything else is the platform's own: the backend's check and its one-step job, Jobs'
Kubernetes launcher, the driver, the completion route and settling.

The platform must be upgraded with ``e2e/k8s/values/builder-remote.yaml``, on top of ``builder.yaml``, as the
``kind-builder-e2e`` CI job does once its in-cluster tests have passed. The test creates that ConfigMap, so it needs
``kubectl`` and:

- ``NHX_E2E_KUBE_CONTEXT``: the cluster's kubectl context. Required: the test never uses the current context.
- ``NHX_E2E_KUBE_NAMESPACE``: the release namespace, where the remote-build step runs; ``nemo-helix`` by default.

No image is pushed. The builder records the digest a remote build's step reports and reads no registry for it, so
the stand-in reports a digest the test makes up.

Run it with ``--feature builder-remote``, against a platform with auth off: the values file accepts no completion
that nothing can verify, so with auth on and no workload token exchange the builder refuses every submit. The module
fails on a platform with auth on, before it submits anything.
"""

import json
import os
import subprocess
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pytest
from nemo_builder_plugin.client import BuilderClient
from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.run.remote import FAILURE_MARKER, POLL_SECONDS, STOPPED
from nemo_builder_plugin.schema import BuildSet, BuildSpec, FileSetSource
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.oidc import discover_nhx_config
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from nhx.testing.e2e.jobs import cleanup_platform_job, wait_for_platform_job

from e2e.resources.builder_remote.remote_build_step import ATTRIBUTES, BUILD_ID_PREFIX, OUTCOME_FILE

pytestmark = [
    pytest.mark.builder_e2e,
    pytest.mark.feature("builder-remote"),
    pytest.mark.platform("kubernetes"),
    pytest.mark.timeout(1800),
]

#: The stand-in step program, and the ConfigMap the remote profile mounts it from (builder-remote.yaml).
STEP_PROGRAM = Path(__file__).parent / "resources" / "builder_remote" / "remote_build_step.py"
STEP_CONFIGMAP = "nhx-e2e-remote-build-step"
#: The one step the remote backend compiles.
STEP_NAME = "remote-build"
#: How long a row may take to settle. Shorter than builder-remote.yaml's build_timeout_seconds, so a build that
#: was cancelled isn't one that timed out.
ROW_DEADLINE_SECONDS = 300
JOB_DEADLINE_SECONDS = 240
#: How long a build is watched before its job is cancelled: a little over two of the driver's polls of the service,
#: so a build that is still there is one the stand-in is holding, not one a step is about to fail.
HOLD_SECONDS = 2 * POLL_SECONDS + 5


@dataclass(frozen=True)
class Kube:
    """Runs kubectl in the release namespace, on the named context only."""

    context: str
    namespace: str

    def _kubectl(self, *args: str, stdin: str | None = None, timeout: int = 60) -> subprocess.CompletedProcess[str]:
        """Run kubectl and return how it ended. It reads ``stdin`` or nothing, never the caller's own input."""
        return subprocess.run(
            ["kubectl", "--context", self.context, "--namespace", self.namespace, *args],
            input=stdin,
            stdin=None if stdin is not None else subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def put_configmap(self, name: str, file: Path) -> None:
        """Create or replace the ConfigMap ``name``, holding ``file`` under its own name."""
        source = f"--from-file={file.name}={file}"
        rendered = self._kubectl("create", "configmap", name, source, "--dry-run=client", "-o", "yaml")
        assert rendered.returncode == 0, f"could not render ConfigMap {name}: {_ended(rendered)}"
        applied = self._kubectl("apply", "-f", "-", stdin=rendered.stdout)
        assert applied.returncode == 0, f"could not apply ConfigMap {name}: {_ended(applied)}"


@dataclass(frozen=True)
class Workspace:
    """Where this module's builds are submitted."""

    name: str
    #: Every job this module created, for teardown.
    jobs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Submitted:
    job: str
    #: The one image's row name.
    image: str
    #: ``<registry>/<repository>``, as the row was written at submit: where the deployment's service publishes.
    destination: str
    #: What the build context told the stand-in to do.
    outcome: dict[str, str]


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} is required; see this module's docstring")
    return value


def _ended(result: subprocess.CompletedProcess[str]) -> str:
    return f"exit {result.returncode}\n--- stdout\n{result.stdout}\n--- stderr\n{result.stderr}"


def _job_account(client: NemoClient, workspace: str, job: str) -> str:
    """What Jobs says of ``job``, for a failure message. Each task carries the end of its log, which is all that
    is left of a finished step once its pod is deleted."""
    try:
        status = JobsClient.from_client(client).get_job_status(workspace=workspace, name=job).data()
    except Exception as exc:
        return f"job {job}: its status could not be read ({type(exc).__name__}: {exc})"
    return f"job {job}: {status.model_dump_json()[:4000]}"


def _submit(client: NemoClient, workspace: Workspace, build_set: str, outcome: dict[str, str]) -> Submitted:
    """Submit one image whose build context tells the stand-in how its build ends."""
    files = FilesClient.from_client(client)
    fileset = f"context-{build_set}"
    files.create_fileset(workspace=workspace.name, body=CreateFilesetRequest(name=fileset))
    # The driver refuses a context without a Dockerfile, though nothing here builds it.
    files.upload_file(workspace=workspace.name, name=fileset, path="Dockerfile", content=b"FROM scratch\n")
    files.upload_file(workspace=workspace.name, name=fileset, path=OUTCOME_FILE, content=json.dumps(outcome).encode())

    submitted = (
        BuilderClient.from_client(client)
        .submit_build(
            workspace=workspace.name,
            body=BuildSet(name=build_set, revision=1, build_specs=[BuildSpec(source=FileSetSource(fileset=fileset))]),
        )
        .data()
    )
    workspace.jobs.append(submitted.job)
    assert len(submitted.images) == 1, f"submit returned rows {[image.name for image in submitted.images]}"
    row = submitted.images[0]
    if row.provenance.backend != "remote":
        pytest.fail(
            f"{row.name} was submitted under the {row.provenance.backend!r} backend: the platform is not building "
            "through the remote backend; upgrade it with e2e/k8s/values/builder-remote.yaml"
        )
    destination = f"{row.registry}/{row.repository}"
    return Submitted(job=submitted.job, image=row.name, destination=destination, outcome=outcome)


def _wait_row(
    client: NemoClient, workspace: Workspace, build: Submitted, want: Literal["ready", "failed"]
) -> ContainerImage:
    """The image's row once it is ``want``. Fails at once if it settles the other way."""
    builder = BuilderClient.from_client(client)
    deadline = time.monotonic() + ROW_DEADLINE_SECONDS
    while True:
        # Reading a row settles it, once its job has ended without completing it.
        row = builder.get_container_image(workspace=workspace.name, name=build.image).data()
        if row.status == want:
            return row
        if row.status in ("ready", "failed"):
            pytest.fail(
                f"{build.image} is {row.status} ({row.status_detail}), not {want}\n"
                f"{_job_account(client, workspace.name, build.job)}"
            )
        if time.monotonic() > deadline:
            pytest.fail(
                f"{build.image} is still {row.status} after {ROW_DEADLINE_SECONDS}s, not {want}\n"
                f"{_job_account(client, workspace.name, build.job)}"
            )
        time.sleep(5)


@pytest.fixture(scope="module")
def kube() -> Kube:
    return Kube(
        context=_required_env("NHX_E2E_KUBE_CONTEXT"),
        namespace=os.environ.get("NHX_E2E_KUBE_NAMESPACE") or "nemo-helix",
    )


@pytest.fixture(scope="module")
def step_program(kube: Kube) -> None:
    """Put the stand-in step program where the remote profile mounts it from."""
    kube.put_configmap(STEP_CONFIGMAP, STEP_PROGRAM)


@pytest.fixture(scope="module")
def remote_workspace(
    request: pytest.FixtureRequest,
    client: NemoClient,
    step_program: None,  # noqa: ARG001
) -> Iterator[Workspace]:
    """A workspace for this module's builds, on a platform with auth off that holds the step's program.

    Every submit takes it, so none is made before the ConfigMap exists: a remote-build step whose ConfigMap is
    missing never starts, and its row would wait until the test gave up.
    """
    assert not discover_nhx_config(str(client.base_url)).auth_enabled, (
        "the platform runs with auth on: e2e/k8s/values/builder-remote.yaml accepts no completion that nothing can "
        "verify, so without workload token exchange the builder refuses every remote submit. Run this module "
        "against a platform with auth off"
    )
    workspaces = WorkspacesClient.from_client(client)
    workspace = Workspace(name=f"e2e-builder-remote-{uuid.uuid4().hex[:8]}")
    workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace.name)).data()
    failed_before = request.session.testsfailed
    yield workspace
    # Best effort; the cluster is discarded after the run. After a failure the jobs are left as they are: cancelling
    # one deletes its step's pod, and that pod's log is what the CI job collects next.
    if request.session.testsfailed == failed_before:
        for job in workspace.jobs:
            try:
                cleanup_platform_job(client, job, workspace.name, timeout=120)
            except Exception:
                pass
    try:
        workspaces.delete_workspace(name=workspace.name).data()
    except Exception:
        pass


@pytest.fixture(scope="module")
def failed_build(client: NemoClient, remote_workspace: Workspace) -> Submitted:
    """A build the stand-in reports failed, for a reason no other run gives."""
    outcome = {"outcome": "failed", "reason": f"stand-in-refused-{uuid.uuid4().hex[:8]}"}
    return _submit(client, remote_workspace, "remote-failed", outcome)


def test_a_built_image_is_completed_with_its_digest_and_the_services_handle(
    client: NemoClient, remote_workspace: Workspace
) -> None:
    """The one-step job's step completes its row ``ready``, at the digest and with the handle the service reports."""
    # Made up, and of no image: the builder records the digest the step reports, and reads no registry for it.
    digest = "sha256:" + uuid.uuid4().hex * 2
    image_id = f"e2e-image-{uuid.uuid4().hex[:8]}"
    outcome = {"outcome": "ready", "digest": digest, "image_id": image_id}
    build = _submit(client, remote_workspace, "remote-ready", outcome)

    row = _wait_row(client, remote_workspace, build, "ready")

    assert row.status_detail is None
    assert row.digest == digest
    assert row.image_ref == f"{build.destination}@{digest}"
    external = row.external
    assert external is not None, f"{build.image} is ready without the build service's handle"
    assert external.build_id and external.build_id.startswith(BUILD_ID_PREFIX), external
    assert external.image_id == image_id
    assert external.attributes == ATTRIBUTES
    provenance = row.provenance
    assert provenance.backend == "remote"
    assert provenance.job == f"{remote_workspace.name}/{build.job}"
    assert provenance.job_uid and provenance.job_attempt_id, f"{build.image} doesn't record its build job"

    # The job the remote backend compiled: one step, which ended well.
    job = wait_for_platform_job(
        client,
        build.job,
        remote_workspace.name,
        timeout=JOB_DEADLINE_SECONDS,
        image_pull_timeout=JOB_DEADLINE_SECONDS,
        poll_interval=2,
    )
    account = _job_account(client, remote_workspace.name, build.job)
    assert job.status == "completed", account
    status = JobsClient.from_client(client).get_job_status(workspace=remote_workspace.name, name=build.job).data()
    assert [step.name for step in status.steps] == [STEP_NAME], account


def test_a_failed_builds_reason_reaches_its_row(
    client: NemoClient, remote_workspace: Workspace, failed_build: Submitted
) -> None:
    """What the service said of a build that failed is on the image's row, from the end of the step's log."""
    row = _wait_row(client, remote_workspace, failed_build, "failed")

    detail = row.status_detail or ""
    account = _job_account(client, remote_workspace.name, failed_build.job)
    assert detail.startswith("not built: its build job ended ("), f"{detail}\n{account}"
    reason = f"the build service reports it failed: {failed_build.outcome['reason']}"
    assert reason in detail, f"{detail}\n{account}"
    assert row.digest is None and row.external is None


def test_cancelling_the_job_fails_its_row(client: NemoClient, remote_workspace: Workspace) -> None:
    """A build whose job is cancelled while the stand-in holds it ends ``failed``, well before its timeout."""
    jobs = JobsClient.from_client(client)
    builder = BuilderClient.from_client(client)
    build = _submit(client, remote_workspace, "remote-cancel", {"outcome": "hang"})
    job = wait_for_platform_job(
        client,
        build.job,
        remote_workspace.name,
        timeout=JOB_DEADLINE_SECONDS,
        image_pull_timeout=JOB_DEADLINE_SECONDS,
        poll_interval=2,
        status_to_check="active",
    )
    assert job.status == "active", _job_account(client, remote_workspace.name, build.job)

    # Held: the step neither completes the image nor fails it. A read settles a row whose job has ended, so a step
    # that died by itself shows here, as a row no longer pending, and is not taken for one that was cancelled.
    held_until = time.monotonic() + HOLD_SECONDS
    while time.monotonic() < held_until:
        row = builder.get_container_image(workspace=remote_workspace.name, name=build.image).data()
        assert row.status == "pending", (
            f"{build.image} is {row.status} ({row.status_detail}) before its job was cancelled\n"
            f"{_job_account(client, remote_workspace.name, build.job)}"
        )
        time.sleep(5)
    job = jobs.get_job(workspace=remote_workspace.name, name=build.job).data()
    assert job.status == "active", _job_account(client, remote_workspace.name, build.job)

    jobs.cancel_job(workspace=remote_workspace.name, name=build.job)

    row = _wait_row(client, remote_workspace, build, "failed")
    detail = row.status_detail or ""
    # Not the step's own reason, nor how the job ended: a cancelled step may keep no log, and may end in error.
    assert detail.startswith("not built: its build job ended ("), detail
    assert "timed out after" not in detail, detail
    assert row.digest is None
    # But whatever the step did say of the image, where Jobs kept the end of its log, is that it was told to stop:
    # a build that failed some other way just as it was cancelled is not this one. Jobs may have kept none, so this
    # can pass having read nothing; that the stand-in's held build ends on that line is tested without a cluster,
    # in plugins/nemo-builder/tests/test_remote_driver.py.
    ended = jobs.get_job_status(workspace=remote_workspace.name, name=build.job).data()
    said = [
        line
        for step in ended.steps
        for task in step.tasks
        for line in (task.error_stack or "").split("\n")
        if FAILURE_MARKER in line
    ]
    assert all(STOPPED in line for line in said), f"{said}\n{detail}"
