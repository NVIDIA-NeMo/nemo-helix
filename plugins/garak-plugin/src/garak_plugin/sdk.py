# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK resources for the garak_plugin plugin.

Mounted on :class:`~nemo_helix.NeMoHelix` as ``client.garak_plugin`` via the
``nemo.sdk`` entry-point in :file:`pyproject.toml`. Exposes:

- ``client.garak_plugin.plugin_status()`` — service healthz check.
- ``client.garak_plugin.configs.{create,list,get,update,delete}`` — ``ScanConfig`` CRUD.
- ``client.garak_plugin.targets.{create,list,get,update,delete}`` — ``ScanTarget`` CRUD.
- ``client.garak_plugin.submit(config=..., target=..., workspace=...)`` — submit a K8s
  scan job and return an :class:`~garak_plugin.sdk_resources.job_resources.GarakPluginJobResource`
  handle. Call ``.wait_until_done()`` on the handle to block until the job completes,
  then ``.download_artifacts()`` to fetch the garak report tarball.
- ``client.garak_plugin.list_jobs(workspace=...)`` — list submitted scan jobs.
- ``client.garak_plugin.get_job(job_name, workspace=...)`` — fetch a single scan job.
- ``client.garak_plugin.run(config=..., target=..., workspace=...)`` — in-process
  scan using :class:`~garak_plugin.jobs.scan.ScanJob`.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from garak_plugin.entities import ScanConfig, ScanTarget
from garak_plugin.jobs.scan import ScanInputSpec, ScanJob
from garak_plugin.sdk_resources.configs import _AsyncConfigResource, _ConfigResource
from garak_plugin.sdk_resources.job_resources import AsyncGarakPluginJobResource, GarakPluginJobResource
from garak_plugin.sdk_resources.targets import _AsyncTargetResource, _TargetResource
from nemo_helix import AsyncNeMoHelix, NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.entities import parse_qualified_name
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import LocalJobResults
from nemo_helix_plugin.sdk import NemoPluginSDKResources


def _local_job_context(*, workspace: str, job_name: str) -> JobContext:
    root = Path(tempfile.mkdtemp(prefix="garak-plugin-", suffix=f"-{job_name}"))
    storage = StoragePaths(ephemeral=root / "ephemeral", persistent=root / "persistent")
    storage.ephemeral.mkdir(parents=True, exist_ok=True)
    storage.persistent.mkdir(parents=True, exist_ok=True)
    return JobContext(
        workspace=workspace,
        storage=storage,
        results=LocalJobResults(root=storage.persistent / "results"),
    )


class GarakPluginResource:
    """Sync SDK namespace mounted as ``client.garak_plugin``."""

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
            raise TypeError("Garak Plugin status response must be a JSON object.")
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
        config: ScanConfig | str,
        target: ScanTarget | str,
        workspace: str | None = None,
        max_probe_retries: int = 0,
        fail_job_on_retries_exhausted: bool = True,
    ) -> GarakPluginJobResource:
        """Submit a scan job to the K8s executor via the plugin job endpoint.

        Returns an :class:`~garak_plugin.sdk_resources.job_resources.GarakPluginJobResource`
        handle. Call ``.wait_until_done()`` to block until the job completes, then
        ``.download_artifacts()`` to fetch the garak report tarball.
        """
        ws = workspace or "default"
        spec = ScanInputSpec(
            config=config,
            target=target,
            max_probe_retries=max_probe_retries,
            fail_job_on_retries_exhausted=fail_job_on_retries_exhausted,
        )
        response = self._http_client.post(
            self._url(f"/v2/workspaces/{ws}/jobs/scan"),
            json={"spec": spec.model_dump(mode="json")},
        )
        response.raise_for_status()
        return GarakPluginJobResource(job_name=response.json()["name"], platform=self._platform, workspace=ws)

    def list_jobs(
        self,
        *,
        workspace: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        """List scan jobs in the workspace with basic pagination."""
        ws = workspace or "default"
        response = self._http_client.get(
            self._url(f"/v2/workspaces/{ws}/jobs/scan"),
            params={"page": page, "page_size": page_size},
        )
        response.raise_for_status()
        return response.json()

    def get_job(self, job_name: str, *, workspace: str | None = None) -> dict:
        """Fetch a single scan job by name."""
        ws = workspace or "default"
        response = self._http_client.get(
            self._url(f"/v2/workspaces/{ws}/jobs/scan/{job_name}"),
        )
        response.raise_for_status()
        return response.json()

    def run(
        self,
        *,
        config: ScanConfig | str,
        target: ScanTarget | str,
        workspace: str | None = None,
    ) -> dict:
        """Run a scan locally, in-process — no jobs-service submission.

        ``config`` / ``target`` accept either an inline pydantic entity or a
        ``"name"`` / ``"workspace/name"`` string referencing one in the entity
        store. Name strings are resolved through ``self.configs.get`` /
        ``self.targets.get`` before the spec is handed to the scheduler, so
        the scheduler always sees inline entities and ``ScanJob.to_spec``
        becomes a no-op.
        """
        ws = workspace or "default"
        resolved_config = self._resolve_config(config, default_workspace=ws)
        resolved_target = self._resolve_target(target, default_workspace=ws)
        spec = ScanInputSpec(config=resolved_config, target=resolved_target)
        return ScanJob().run(
            spec.model_dump(mode="json"),
            ctx=_local_job_context(workspace=ws, job_name=ScanJob.name),
            sdk=client_from_platform(self._platform, NemoClient),
        )

    def _resolve_config(self, value: ScanConfig | str, *, default_workspace: str) -> ScanConfig:
        if isinstance(value, ScanConfig):
            return value
        ws, name = parse_qualified_name(value, default_workspace=default_workspace)
        return self.configs.get(workspace=ws, name=name)

    def _resolve_target(self, value: ScanTarget | str, *, default_workspace: str) -> ScanTarget:
        if isinstance(value, ScanTarget):
            return value
        ws, name = parse_qualified_name(value, default_workspace=default_workspace)
        return self.targets.get(workspace=ws, name=name)

    def _url(self, path: str) -> str:
        return str(self._platform.base_url).rstrip("/") + "/apis/garak-plugin" + path


class AsyncGarakPluginResource:
    """Async SDK namespace mounted as ``client.garak_plugin``."""

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
            raise TypeError("Garak Plugin status response must be a JSON object.")
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
        config: ScanConfig | str,
        target: ScanTarget | str,
        workspace: str | None = None,
        max_probe_retries: int = 0,
        fail_job_on_retries_exhausted: bool = True,
    ) -> AsyncGarakPluginJobResource:
        """Async twin of :meth:`GarakPluginResource.submit`."""
        ws = workspace or "default"
        spec = ScanInputSpec(
            config=config,
            target=target,
            max_probe_retries=max_probe_retries,
            fail_job_on_retries_exhausted=fail_job_on_retries_exhausted,
        )
        response = await self._http_client.post(
            self._url(f"/v2/workspaces/{ws}/jobs/scan"),
            json={"spec": spec.model_dump(mode="json")},
        )
        response.raise_for_status()
        return AsyncGarakPluginJobResource(job_name=response.json()["name"], platform=self._platform, workspace=ws)

    async def list_jobs(
        self,
        *,
        workspace: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        """Async twin of :meth:`GarakPluginResource.list_jobs`."""
        ws = workspace or "default"
        response = await self._http_client.get(
            self._url(f"/v2/workspaces/{ws}/jobs/scan"),
            params={"page": page, "page_size": page_size},
        )
        response.raise_for_status()
        return response.json()

    async def get_job(self, job_name: str, *, workspace: str | None = None) -> dict:
        """Async twin of :meth:`GarakPluginResource.get_job`."""
        ws = workspace or "default"
        response = await self._http_client.get(
            self._url(f"/v2/workspaces/{ws}/jobs/scan/{job_name}"),
        )
        response.raise_for_status()
        return response.json()

    async def run(
        self,
        *,
        config: ScanConfig | str,
        target: ScanTarget | str,
        workspace: str | None = None,
    ) -> dict:
        """Async twin of :meth:`GarakPluginResource.run`.

        The job body is sync and may use blocking filesystem/subprocess work,
        so the async resource runs it on a worker thread.
        """
        ws = workspace or "default"
        resolved_config = await self._resolve_config(config, default_workspace=ws)
        resolved_target = await self._resolve_target(target, default_workspace=ws)
        spec = ScanInputSpec(config=resolved_config, target=resolved_target)
        return await asyncio.to_thread(
            ScanJob().run,
            spec.model_dump(mode="json"),
            ctx=_local_job_context(workspace=ws, job_name=ScanJob.name),
            async_sdk=client_from_platform(self._platform, AsyncNemoClient),
        )

    async def _resolve_config(self, value: ScanConfig | str, *, default_workspace: str) -> ScanConfig:
        if isinstance(value, ScanConfig):
            return value
        ws, name = parse_qualified_name(value, default_workspace=default_workspace)
        return await self.configs.get(workspace=ws, name=name)

    async def _resolve_target(self, value: ScanTarget | str, *, default_workspace: str) -> ScanTarget:
        if isinstance(value, ScanTarget):
            return value
        ws, name = parse_qualified_name(value, default_workspace=default_workspace)
        return await self.targets.get(workspace=ws, name=name)

    def _url(self, path: str) -> str:
        return str(self._platform.base_url).rstrip("/") + "/apis/garak-plugin" + path


garak_plugin_sdk_resources = NemoPluginSDKResources(
    sync_resource=GarakPluginResource,
    async_resource=AsyncGarakPluginResource,
)
