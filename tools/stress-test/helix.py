# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved-context client for the stress controller. The worker does not import this module."""

from __future__ import annotations

import json
import logging
from typing import Any

from bundle import RESULT_NAME, WORKER_PATH, digest, job_request
from cluster import ClusterWriteError, JobView
from config import PodSize
from job_profile import ProfileCandidate, ProfileFacts, choose_profile
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.config.config import ConfigParams, get_context
from nemo_helix_plugin.client.errors import NemoHTTPError
from nemo_helix_plugin.client.oidc import discover_nhx_config
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.endpoints import ExecutionProfile
from nemo_helix_plugin.jobs.schemas import HelixJobStatus, HelixJobStatusResponse
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from redact import redact

LOG_TAIL = 20
log = logging.getLogger("stress")


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

    def create_job(
        self,
        name: str,
        environment: dict[str, str],
        pod: PodSize,
        profile_name: str,
        provider: str = "cpu",
        image: str | None = None,
    ) -> bool:
        body = job_request(name, environment, pod, profile_name, provider, image)
        try:
            self._jobs.create_job(workspace=self.workspace, body=body).data()
        except NemoHTTPError as exc:
            log.error("error %s: refused, HTTP %s %s", name, exc.status_code, redact(str(exc.detail)))
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

    def diagnose(self, name: str) -> dict[str, object]:
        found: dict[str, object] = {}
        try:
            status = self._jobs.get_job_status(workspace=self.workspace, name=name).data()
        except NemoHTTPError as exc:
            found["errors"] = [f"cannot read job status: HTTP {exc.status_code}"]
        else:
            found["errors"] = _error_messages(status)
        try:
            page = self._jobs.list_job_logs(workspace=self.workspace, name=name, query_params={"tail": LOG_TAIL})
            found["log_tail"] = [redact(entry.message) for entry in page.page().items]
        except NemoHTTPError:
            found["log_tail"] = []
        return found

    def read_file(self, path: str) -> bytes | None:
        try:
            return self._files.download_file(workspace=self.workspace, name=self.fileset, path=path).read()
        except NemoHTTPError as exc:
            if exc.status_code == 404:
                return None
            raise ContextError(f"cannot read {path}") from exc

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
        log.info("bundled %s (%d bytes)", WORKER_PATH, len(payload))
        try:
            remote = self._files.download_file(workspace=self.workspace, name=self.fileset, path=WORKER_PATH).read()
        except NemoHTTPError as exc:
            if exc.status_code != 404:
                raise ContextError(f"cannot read {WORKER_PATH}") from exc
            remote = b""
        if digest(remote) != current:
            log.info("sent %s to the %s fileset", WORKER_PATH, self.fileset)
            try:
                self._files.upload_file(
                    workspace=self.workspace, name=self.fileset, path=WORKER_PATH, content=payload
                ).data()
            except NemoHTTPError as exc:
                raise ContextError(f"cannot publish {WORKER_PATH}") from exc
        else:
            log.info("%s already matches the %s fileset", WORKER_PATH, self.fileset)
        return current


def open_cluster(
    *,
    context: str | None,
    workspace: str,
    profile_name: str,
) -> tuple[HelixCluster, ProfileFacts, str, bool]:
    """Connect and resolve ``profile_name``. Auth may be off; the caller decides which steps run."""
    overrides: ConfigParams | None = {"current_context": context} if context else None
    try:
        saved = get_context(overrides=overrides)
    except (OSError, ValueError) as exc:
        raise ContextError(str(exc)) from exc
    client = NemoClient.from_config(context=saved.context_name)
    discovery = discover_nhx_config(str(client.base_url))
    jobs = JobsClient.from_client(client)
    try:
        profiles = jobs.get_execution_profiles().data()
    except NemoHTTPError as exc:
        raise ContextError("cannot read execution profiles") from exc
    facts = choose_profile(_candidates(profiles), profile_name)
    WorkspacesClient.from_client(client).create_workspace(body=CreateWorkspaceRequest(name=workspace), exist_ok=True)
    FilesClient.from_client(client).create_fileset(
        workspace=workspace, body=CreateFilesetRequest(name=workspace), exist_ok=True
    )
    return HelixCluster(client, workspace, workspace), facts, str(client.base_url), discovery.auth_enabled


def _candidates(profiles: list[ExecutionProfile]) -> list[ProfileCandidate]:
    return [
        ProfileCandidate(
            name=item.profile,
            provider=item.provider,
            backend=item.backend,
            task_image=item.config.default_task_image,
            ttl_seconds_before_active=item.config.ttl_seconds_before_active,
            ttl_seconds_active=item.config.ttl_seconds_active,
        )
        for item in profiles
    ]


def _status_text(status: HelixJobStatus) -> str:
    return status.value


def _error_messages(status: HelixJobStatusResponse) -> list[str]:
    details: list[dict[str, Any] | None] = [status.error_details]
    for step in status.steps:
        details.append(step.error_details)
        details.extend(task.error_details for task in step.tasks)
    messages: list[str] = []
    for item in details:
        if not item:
            continue
        text = redact(str(item.get("message") or json.dumps(item, sort_keys=True, default=str)))
        if text not in messages:
            messages.append(text)
    return messages
