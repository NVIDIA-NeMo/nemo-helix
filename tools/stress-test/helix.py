# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved-context client for the stress controller. The worker does not import this module."""

from __future__ import annotations

from bundle import RESULT_NAME, WORKER_PATH, digest, job_request
from cluster import ClusterWriteError, JobView
from config import PodSize, Step
from job_profile import ProfileError, ProfileFacts, validate_profile
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.config.config import ConfigParams, get_context
from nemo_helix_plugin.client.errors import NemoHTTPError
from nemo_helix_plugin.client.oidc import discover_nhx_config
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.endpoints import ExecutionProfile
from nemo_helix_plugin.jobs.execution_profiles import KubernetesJobExecutionProfile
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest


class ContextError(RuntimeError):
    """The saved context cannot be used for a stress run."""


class HelixCluster:
    """Jobs and files operations against one workspace and one run fileset."""

    def __init__(self, client: NemoClient, workspace: str, fileset: str) -> None:
        self.client = client
        self.workspace = workspace
        self.fileset = fileset
        self._jobs = JobsClient.from_client(client)
        self._files = FilesClient.from_client(client)

    def create_job(self, name: str, environment: dict[str, str], pod: PodSize, profile_name: str) -> bool:
        body = job_request(name, environment, pod, profile_name)
        try:
            self._jobs.create_job(workspace=self.workspace, body=body).data()
        except NemoHTTPError:
            return False
        return True

    def jobs(self, prefix: str) -> list[JobView]:
        found: list[JobView] = []
        for job in self._jobs.list_jobs(workspace=self.workspace).items():
            if job.name.startswith(prefix):
                found.append(JobView(name=job.name, status=_status_text(job.status)))
        return found

    def cancel(self, name: str) -> None:
        try:
            self._jobs.cancel_job(workspace=self.workspace, name=name).data()
        except NemoHTTPError:
            return

    def delete(self, name: str) -> None:
        try:
            self._jobs.delete_job(workspace=self.workspace, name=name)
        except NemoHTTPError as exc:
            if exc.status_code != 404:
                raise

    def read_result(self, name: str) -> bytes | None:
        try:
            return self._jobs.download_job_result(workspace=self.workspace, job=name, name=RESULT_NAME).read()
        except NemoHTTPError:
            return None

    def write_file(self, path: str, data: bytes) -> None:
        try:
            self._files.upload_file(workspace=self.workspace, name=self.fileset, path=path, content=data).data()
        except NemoHTTPError as exc:
            raise ClusterWriteError(path) from exc

    def remove_prefix(self, prefix: str) -> None:
        listed = self._files.list_files(workspace=self.workspace, name=self.fileset).data()
        for item in listed.data:
            if prefix and not item.path.startswith(prefix):
                continue
            try:
                self._files.delete_file(workspace=self.workspace, name=self.fileset, path=item.path).data()
            except NemoHTTPError as exc:
                if exc.status_code != 404:
                    raise

    def sync_worker(self, payload: bytes) -> str:
        """Upload the worker zip when the fileset copy is missing or different."""
        current = digest(payload)
        try:
            remote = self._files.download_file(workspace=self.workspace, name=self.fileset, path=WORKER_PATH).read()
        except NemoHTTPError as exc:
            if exc.status_code != 404:
                raise ContextError(f"cannot read {WORKER_PATH}") from exc
            remote = b""
        if digest(remote) != current:
            try:
                self._files.upload_file(
                    workspace=self.workspace, name=self.fileset, path=WORKER_PATH, content=payload
                ).data()
            except NemoHTTPError as exc:
                raise ContextError(f"cannot publish {WORKER_PATH}") from exc
        return current


def open_cluster(
    *,
    context: str | None,
    workspace: str,
    profile_name: str,
    steps: list[Step],
) -> tuple[HelixCluster, ProfileFacts, str]:
    """Connect, require auth, and require a Kubernetes profile."""
    overrides: ConfigParams | None = {"current_context": context} if context else None
    try:
        saved = get_context(overrides=overrides)
    except (OSError, ValueError) as exc:
        raise ContextError(str(exc)) from exc
    client = NemoClient.from_config(context=saved.context_name)
    discovery = discover_nhx_config(str(client.base_url))
    if not discovery.auth_enabled:
        raise ContextError(f"auth is disabled at {client.base_url}")
    jobs = JobsClient.from_client(client)
    try:
        profiles = jobs.get_execution_profiles().data()
    except NemoHTTPError as exc:
        raise ContextError("cannot read execution profiles") from exc
    facts = _facts(profiles, profile_name)
    validate_profile(facts, steps)
    WorkspacesClient.from_client(client).create_workspace(body=CreateWorkspaceRequest(name=workspace), exist_ok=True)
    FilesClient.from_client(client).create_fileset(
        workspace=workspace, body=CreateFilesetRequest(name=workspace), exist_ok=True
    )
    return HelixCluster(client, workspace, workspace), facts, str(client.base_url)


def _facts(profiles: list[ExecutionProfile], name: str) -> ProfileFacts:
    matches = [item for item in profiles if item.profile == name]
    if len(matches) != 1:
        raise ProfileError(f"profile {name!r} was not found")
    profile = matches[0]
    if not isinstance(profile, KubernetesJobExecutionProfile):
        raise ProfileError(f"profile {name!r} backend is {profile.backend!r}; expected kubernetes_job")
    return ProfileFacts(
        name=profile.profile,
        backend=profile.backend,
        task_image=profile.config.default_task_image,
        ttl_seconds_before_active=profile.config.ttl_seconds_before_active,
        ttl_seconds_active=profile.config.ttl_seconds_active,
    )


def _status_text(status: HelixJobStatus) -> str:
    return status.value
