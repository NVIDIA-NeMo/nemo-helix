# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK resources for the auditor plugin.

Mounted on :class:`~nemo_helix.NeMoHelix` as ``client.auditor`` via the
``nemo.sdk`` entry-point in :file:`pyproject.toml`. Exposes:

- ``client.auditor.plugin_status()`` — service healthz check.
- ``client.auditor.configs.{create,list,get,update,delete}`` — ``AuditConfig`` CRUD.
- ``client.auditor.targets.{create,list,get,update,delete}`` — ``AuditTarget`` CRUD.
- ``client.auditor.submit(config=..., target=..., workspace=...)`` — submit a K8s
  audit job and return an :class:`~nemo_auditor.sdk_resources.job_resources.AuditorJobResource`
  handle. Call ``.wait_until_done()`` on the handle to block until the job completes,
  then ``.download_artifacts()`` to fetch the garak report tarball.
- ``client.auditor.list_jobs(workspace=...)`` — list submitted audit jobs.
- ``client.auditor.get_job(job_name, workspace=...)`` — fetch a single audit job.
- ``client.auditor.run(config=..., target=..., workspace=...)`` — in-process
  audit using :class:`~nemo_auditor.jobs.audit.AuditJob`. Delegates to
  :meth:`~nemo_helix_plugin.scheduler.NemoJobScheduler.run_local`, which
  constructs a tempdir-backed :class:`~nemo_helix_plugin.job_context.JobContext`
  and writes report artifacts via
  :class:`~nemo_helix_plugin.job_results.LocalJobResults`.
"""

from __future__ import annotations

import asyncio

from nemo_auditor.entities import AuditConfig, AuditTarget
from nemo_auditor.jobs.audit import AuditInputSpec, AuditJob
from nemo_auditor.sdk_resources.configs import _AsyncConfigResource, _ConfigResource
from nemo_auditor.sdk_resources.job_resources import AsyncAuditorJobResource, AuditorJobResource
from nemo_auditor.sdk_resources.targets import _AsyncTargetResource, _TargetResource
from nemo_helix import AsyncNeMoHelix, NeMoHelix
from nemo_helix_plugin.entities import parse_qualified_name
from nemo_helix_plugin.scheduler import NemoJobScheduler
from nemo_helix_plugin.sdk import NemoPluginSDKResources


class AuditorPluginResource:
    """Sync SDK namespace mounted as ``client.auditor``."""

    def __init__(self, platform: NeMoHelix) -> None:
        self._platform = platform
        self._http_client = platform._client
        self._configs: _ConfigResource | None = None
        self._targets: _TargetResource | None = None

    def plugin_status(self) -> dict[str, object]:
        response = self._http_client.get(self._url("/v1/healthz"))
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError("Auditor plugin status response must be a JSON object.")
        return {str(key): value for key, value in payload.items()}

    @property
    def configs(self) -> _ConfigResource:
        if self._configs is None:
            self._configs = _ConfigResource(self)
        return self._configs

    @property
    def targets(self) -> _TargetResource:
        if self._targets is None:
            self._targets = _TargetResource(self)
        return self._targets

    def submit(
        self,
        *,
        config: AuditConfig | str,
        target: AuditTarget | str,
        workspace: str | None = None,
        max_probe_retries: int = 0,
        fail_job_on_retries_exhausted: bool = True,
    ) -> AuditorJobResource:
        """Submit an audit job to the K8s executor via the plugin job endpoint.

        Returns an :class:`~nemo_auditor.sdk_resources.job_resources.AuditorJobResource`
        handle. Call ``.wait_until_done()`` to block until the job completes, then
        ``.download_artifacts()`` to fetch the garak report tarball.
        """
        ws = workspace or "default"
        spec = AuditInputSpec(
            config=config,
            target=target,
            max_probe_retries=max_probe_retries,
            fail_job_on_retries_exhausted=fail_job_on_retries_exhausted,
        )
        response = self._http_client.post(
            self._url(f"/v2/workspaces/{ws}/jobs/audit"),
            json={"spec": spec.model_dump(mode="json")},
        )
        response.raise_for_status()
        return AuditorJobResource(job_name=response.json()["name"], platform=self._platform, workspace=ws)

    def list_jobs(
        self,
        *,
        workspace: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        """List audit jobs in the workspace with basic pagination."""
        ws = workspace or "default"
        response = self._http_client.get(
            self._url(f"/v2/workspaces/{ws}/jobs/audit"),
            params={"page": page, "page_size": page_size},
        )
        response.raise_for_status()
        return response.json()

    def get_job(self, job_name: str, *, workspace: str | None = None) -> dict:
        """Fetch a single audit job by name."""
        ws = workspace or "default"
        response = self._http_client.get(
            self._url(f"/v2/workspaces/{ws}/jobs/audit/{job_name}"),
        )
        response.raise_for_status()
        return response.json()

    def run(
        self,
        *,
        config: AuditConfig | str,
        target: AuditTarget | str,
        workspace: str | None = None,
    ) -> dict:
        """Run an audit locally, in-process — no jobs-service submission.

        ``config`` / ``target`` accept either an inline pydantic entity or a
        ``"name"`` / ``"workspace/name"`` string referencing one in the entity
        store. Name strings are resolved through ``self.configs.get`` /
        ``self.targets.get`` before the spec is handed to the scheduler, so
        the scheduler always sees inline entities and ``AuditJob.to_spec``
        becomes a no-op.
        """
        ws = workspace or "default"
        resolved_config = self._resolve_config(config, default_workspace=ws)
        resolved_target = self._resolve_target(target, default_workspace=ws)
        spec = AuditInputSpec(config=resolved_config, target=resolved_target)
        return NemoJobScheduler().run_local(
            AuditJob,
            spec.model_dump(mode="json"),
            workspace=ws,
            sdk=self._platform,
        )

    def _resolve_config(self, value: AuditConfig | str, *, default_workspace: str) -> AuditConfig:
        if isinstance(value, AuditConfig):
            return value
        ws, name = parse_qualified_name(value, default_workspace=default_workspace)
        return self.configs.get(workspace=ws, name=name)

    def _resolve_target(self, value: AuditTarget | str, *, default_workspace: str) -> AuditTarget:
        if isinstance(value, AuditTarget):
            return value
        ws, name = parse_qualified_name(value, default_workspace=default_workspace)
        return self.targets.get(workspace=ws, name=name)

    def _url(self, path: str) -> str:
        return str(self._platform.base_url).rstrip("/") + "/apis/auditor" + path


class AsyncAuditorPluginResource:
    """Async SDK namespace mounted as ``client.auditor``."""

    def __init__(self, platform: AsyncNeMoHelix) -> None:
        self._platform = platform
        self._http_client = platform._client
        self._configs: _AsyncConfigResource | None = None
        self._targets: _AsyncTargetResource | None = None

    async def plugin_status(self) -> dict[str, object]:
        response = await self._http_client.get(self._url("/v1/healthz"))
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError("Auditor plugin status response must be a JSON object.")
        return {str(key): value for key, value in payload.items()}

    @property
    def configs(self) -> _AsyncConfigResource:
        if self._configs is None:
            self._configs = _AsyncConfigResource(self)
        return self._configs

    @property
    def targets(self) -> _AsyncTargetResource:
        if self._targets is None:
            self._targets = _AsyncTargetResource(self)
        return self._targets

    async def submit(
        self,
        *,
        config: AuditConfig | str,
        target: AuditTarget | str,
        workspace: str | None = None,
        max_probe_retries: int = 0,
        fail_job_on_retries_exhausted: bool = True,
    ) -> AsyncAuditorJobResource:
        """Async twin of :meth:`AuditorPluginResource.submit`."""
        ws = workspace or "default"
        spec = AuditInputSpec(
            config=config,
            target=target,
            max_probe_retries=max_probe_retries,
            fail_job_on_retries_exhausted=fail_job_on_retries_exhausted,
        )
        response = await self._http_client.post(
            self._url(f"/v2/workspaces/{ws}/jobs/audit"),
            json={"spec": spec.model_dump(mode="json")},
        )
        response.raise_for_status()
        return AsyncAuditorJobResource(job_name=response.json()["name"], platform=self._platform, workspace=ws)

    async def list_jobs(
        self,
        *,
        workspace: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        """Async twin of :meth:`AuditorPluginResource.list_jobs`."""
        ws = workspace or "default"
        response = await self._http_client.get(
            self._url(f"/v2/workspaces/{ws}/jobs/audit"),
            params={"page": page, "page_size": page_size},
        )
        response.raise_for_status()
        return response.json()

    async def get_job(self, job_name: str, *, workspace: str | None = None) -> dict:
        """Async twin of :meth:`AuditorPluginResource.get_job`."""
        ws = workspace or "default"
        response = await self._http_client.get(
            self._url(f"/v2/workspaces/{ws}/jobs/audit/{job_name}"),
        )
        response.raise_for_status()
        return response.json()

    async def run(
        self,
        *,
        config: AuditConfig | str,
        target: AuditTarget | str,
        workspace: str | None = None,
    ) -> dict:
        """Async twin of :meth:`AuditorPluginResource.run`.

        ``NemoJobScheduler.run_local`` is sync and itself calls
        ``asyncio.run`` to drive ``to_spec``, so we push it onto a worker
        thread to keep the caller's event loop free — same pattern as
        :class:`nemo_evaluator.sdk._executor._AsyncEvaluatorPluginExecutor.run_local`.
        """
        ws = workspace or "default"
        resolved_config = await self._resolve_config(config, default_workspace=ws)
        resolved_target = await self._resolve_target(target, default_workspace=ws)
        spec = AuditInputSpec(config=resolved_config, target=resolved_target)
        scheduler = NemoJobScheduler()
        return await asyncio.to_thread(
            scheduler.run_local,
            AuditJob,
            spec.model_dump(mode="json"),
            workspace=ws,
            async_sdk=self._platform,
        )

    async def _resolve_config(self, value: AuditConfig | str, *, default_workspace: str) -> AuditConfig:
        if isinstance(value, AuditConfig):
            return value
        ws, name = parse_qualified_name(value, default_workspace=default_workspace)
        return await self.configs.get(workspace=ws, name=name)

    async def _resolve_target(self, value: AuditTarget | str, *, default_workspace: str) -> AuditTarget:
        if isinstance(value, AuditTarget):
            return value
        ws, name = parse_qualified_name(value, default_workspace=default_workspace)
        return await self.targets.get(workspace=ws, name=name)

    def _url(self, path: str) -> str:
        return str(self._platform.base_url).rstrip("/") + "/apis/auditor" + path


auditor_sdk_resources = NemoPluginSDKResources(
    sync_resource=AuditorPluginResource,
    async_resource=AsyncAuditorPluginResource,
)
