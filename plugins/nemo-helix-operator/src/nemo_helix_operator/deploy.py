# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""helix-operator's deploy routes: ``helm upgrade`` the Helix release with the chart from an uploaded checkout."""

import asyncio
import io
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from nemo_helix_operator.config import HelixOperatorConfig
from nemo_helix_operator.schemas import BuildSource
from nemo_helix_plugin.authz import AuthzScope, CallerKind, PermissionSet, path_rule, perm
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NemoHTTPError, NemoTransportError
from nemo_helix_plugin.dependencies import get_nemo_client
from nemo_helix_plugin.files.client import AsyncFilesClient
from pydantic import BaseModel, ConfigDict, Field, field_validator

scope = AuthzScope("helix-operator")

# ponytail: an upgrade runs as a child of this process. If the pod restarts mid-upgrade, the release stays
# pending-upgrade and every later deploy gets 409 until someone runs `helm rollback`; running helm as a Jobs job
# is the upgrade path.
_upgrades: dict[int, asyncio.Task[tuple[int, str]]] = {}


class DeployPerms(PermissionSet, namespace="helix-operator.deployments"):
    CREATE = perm("Upgrade the Helix release with new images")
    READ = perm("Read a Helix release revision")


def split_image(ref: str) -> tuple[str, str]:
    """Return ``(repository, tag)`` of ``<repository>:<tag>``; a registry port isn't a tag."""
    repository, colon, tag = ref.rpartition(":")
    if not colon or not repository or not tag or "/" in tag or "@" in ref:
        raise ValueError(f"{ref!r} isn't `<repository>:<tag>`")
    return repository, tag


class CreateDeploymentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: BuildSource = Field(description="The uploaded checkout whose Helm chart is installed.")
    api_image: str = Field(description="`<repository>:<tag>` for the api deployment.")
    core_image: str | None = Field(default=None, description="`<repository>:<tag>` for core; defaults to api_image.")
    rollback_on_failure: bool = Field(default=False, description="Roll back on its own if the upgrade fails.")

    @field_validator("api_image", "core_image")
    @classmethod
    def _image(cls, ref: str | None) -> str | None:
        if ref is not None:
            split_image(ref)
        return ref


class HelixDeployment(BaseModel):
    release: str
    revision: int
    status: str = Field(description="Helm's status, such as pending-upgrade, deployed, superseded or failed.")
    api_image: str | None = None
    core_image: str | None = None
    detail: str | None = None


async def _helm(*args: str) -> tuple[int, str]:
    home = Path(tempfile.gettempdir()) / "helm"
    env = {**os.environ, **{f"HELM_{kind}_HOME": str(home / kind.lower()) for kind in ("CACHE", "CONFIG", "DATA")}}
    process = await asyncio.create_subprocess_exec(
        "helm", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env
    )
    output, _ = await process.communicate()
    return process.returncode or 0, output.decode(errors="replace")


def _target() -> tuple[str, str]:
    namespace = os.environ.get("POD_NAMESPACE")
    if not namespace:
        raise HTTPException(status_code=503, detail="POD_NAMESPACE is unset: deploy runs only in the operator pod")
    return HelixOperatorConfig.get().release_name, namespace


def extract_chart(archive: bytes, chart_path: str, into: Path) -> Path:
    """Extract only ``chart_path`` from ``archive`` into ``into`` and return the chart's folder."""
    prefix = chart_path.strip("/") + "/"
    try:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(into, members=[m for m in tar.getmembers() if m.name.startswith(prefix)], filter="data")
    except tarfile.TarError as exc:
        raise HTTPException(status_code=422, detail=f"Can't extract the chart: {exc}") from exc
    chart = into / chart_path.strip("/")
    if not (chart / "Chart.yaml").is_file():
        raise HTTPException(status_code=422, detail=f"The archive has no {chart_path}/Chart.yaml")
    return chart


async def _upgrade(work: Path, args: list[str]) -> tuple[int, str]:
    try:
        return await _helm(*args)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _image_refs(values: dict) -> tuple[str | None, str | None]:
    def ref(component: str) -> str | None:
        image = values.get(component, {}).get("image", {})
        return f"{image['repository']}:{image['tag']}" if image.get("repository") and image.get("tag") else None

    return ref("api"), ref("core")


def deploy_router() -> APIRouter:
    router = APIRouter()

    @router.post("/deployments", response_model=HelixDeployment, status_code=202)
    @scope.write
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[DeployPerms.CREATE])
    async def create_deployment(
        workspace: str, body: CreateDeploymentRequest, client: AsyncNemoClient = Depends(get_nemo_client)
    ) -> HelixDeployment:
        """Start ``helm upgrade`` of the Helix release with the uploaded chart and these images."""
        release, namespace = _target()
        config = HelixOperatorConfig.get()
        code, output = await _helm("status", release, "-n", namespace, "-o", "json")
        if code:
            raise HTTPException(status_code=502, detail=f"helm status failed: {output.strip()}")
        current = json.loads(output)
        status = current["info"]["status"]
        if status.startswith("pending") or any(not task.done() for task in _upgrades.values()):
            raise HTTPException(status_code=409, detail=f"{release} is already being upgraded ({status})")
        try:
            download = await AsyncFilesClient.from_client(client).download_file(
                workspace=workspace, name=body.source.fileset, path=body.source.archive
            )
            archive = await download.read()
        except NemoHTTPError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        except NemoTransportError as exc:
            raise HTTPException(status_code=502, detail=f"Couldn't reach the Helix API: {exc}") from exc

        work = Path(tempfile.mkdtemp(prefix="helix-deploy-"))
        try:
            chart = extract_chart(archive, config.chart_path, work)
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise
        core_image = body.core_image or body.api_image
        args = ["upgrade", release, str(chart), "-n", namespace, "--reset-then-reuse-values"]
        for component, ref in (("api", body.api_image), ("core", core_image)):
            repository, tag = split_image(ref)
            args += ["--set-string", f"{component}.image.repository={repository}"]
            args += ["--set-string", f"{component}.image.tag={tag}"]
        args += ["--wait", "--timeout", config.deploy_timeout]
        if body.rollback_on_failure:
            args.append("--atomic")
        revision = current["version"] + 1
        _upgrades[revision] = asyncio.create_task(_upgrade(work, args))
        return HelixDeployment(
            release=release,
            revision=revision,
            status="pending-upgrade",
            api_image=body.api_image,
            core_image=core_image,
        )

    @router.get("/deployments/{revision}", response_model=HelixDeployment)
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[DeployPerms.READ])
    async def get_deployment(workspace: str, revision: int) -> HelixDeployment:
        """Read a revision of the Helix release from Helm."""
        release, namespace = _target()
        code, output = await _helm("history", release, "-n", namespace, "-o", "json", "--max", "256")
        if code:
            raise HTTPException(status_code=502, detail=f"helm history failed: {output.strip()}")
        row = next((row for row in json.loads(output) if row["revision"] == revision), None)
        if row is None:
            task = _upgrades.get(revision)
            if task is None:
                raise HTTPException(status_code=404, detail=f"{release} has no revision {revision}")
            if not task.done():
                return HelixDeployment(release=release, revision=revision, status="pending-upgrade")
            # Helm failed before recording the revision, for example because the chart didn't render.
            detail = task.result()[1][-2000:]
            return HelixDeployment(release=release, revision=revision, status="failed", detail=detail)
        code, values = await _helm("get", "values", release, "-n", namespace, "--revision", str(revision), "-o", "json")
        api_image, core_image = _image_refs(json.loads(values)) if not code else (None, None)
        return HelixDeployment(
            release=release,
            revision=revision,
            status=row["status"],
            api_image=api_image,
            core_image=core_image,
            detail=row.get("description"),
        )

    return router
