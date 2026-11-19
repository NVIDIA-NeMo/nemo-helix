# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""helix-operator's routes: build an uploaded checkout with the builder, and read the build."""

from typing import ClassVar, Literal

from fastapi import APIRouter, Depends, HTTPException
from nemo_builder_plugin.client import AsyncBuilderClient
from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.schema import BuildOutput, BuildSet, BuildSpec, FileSetSource
from nemo_helix_operator.config import HelixOperatorConfig
from nemo_helix_operator.deploy import deploy_router
from nemo_helix_operator.schemas import CreateBuildRequest, HelixBuild, SourceInfo
from nemo_helix_operator.source import DOCKERFILE_PATH
from nemo_helix_operator.testrun import e2e_router
from nemo_helix_plugin.authz import AuthzScope, CallerKind, PermissionSet, path_rule, perm
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NemoHTTPError, NemoTransportError
from nemo_helix_plugin.dependencies import get_nemo_client
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nemo_helix_plugin.service import NemoService, RouterSpec
from pydantic import ValidationError

scope = AuthzScope("helix-operator")


class BuildPerms(PermissionSet, namespace="helix-operator.builds"):
    CREATE = perm("Build an nhx-api image from an uploaded checkout")
    READ = perm("Read an nhx-api image build")


def _build(image: ContainerImage, status: Literal["failed"] | None = None, detail: str | None = None) -> HelixBuild:
    # The image record doesn't keep the output tag, so the spec is named after the commit to rebuild it.
    tag = f"{image.provenance.spec}-r{image.provenance.revision}"
    ref = f"{image.registry}/{image.repository}:{tag}" if image.status == "ready" else None
    return HelixBuild(
        name=image.name,
        status=status or image.status,
        tag=tag,
        api_image=ref,
        core_image=ref,
        digest=image.digest,
        job=image.provenance.job,
        detail=detail or image.status_detail,
    )


class HelixOperatorService(NemoService):
    name: ClassVar[str] = "helix-operator"
    dependencies: ClassVar[list[str]] = []

    def get_routers(self) -> list[RouterSpec]:
        return [
            RouterSpec(_build_router(), tag="Helix Operator", prefix="/v2/workspaces/{workspace}"),
            RouterSpec(deploy_router(), tag="Helix Operator", prefix="/v2/workspaces/{workspace}"),
            RouterSpec(e2e_router(), tag="Helix Operator", prefix="/v2/workspaces/{workspace}"),
        ]


def _build_router() -> APIRouter:
    router = APIRouter()

    @router.post("/builds", response_model=HelixBuild, status_code=201)
    @scope.write
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[BuildPerms.CREATE])
    async def create_build(
        workspace: str, body: CreateBuildRequest, client: AsyncNemoClient = Depends(get_nemo_client)
    ) -> HelixBuild:
        """Build the uploaded checkout into an nhx-api image."""
        config = HelixOperatorConfig.get()
        builder = AsyncBuilderClient.from_client(client)
        try:
            download = await AsyncFilesClient.from_client(client).download_file(
                workspace=workspace, name=body.source.fileset, path="source.json"
            )
            source = SourceInfo.model_validate_json(await download.read())
            # ponytail: only the first page of 100 images is read; a stale page makes the builder refuse the
            # revision with 409, and paging by revision is the upgrade.
            images = (
                await builder.list_container_images(
                    workspace=workspace, query_params={"build_set": config.build_set, "page_size": 100}
                )
            ).data()
            revision = max((image.provenance.revision for image in images), default=0) + 1
            build_set = BuildSet(
                name=config.build_set,
                revision=revision,
                build_specs=[
                    BuildSpec(
                        name=source.commit,
                        source=FileSetSource(fileset=body.source.fileset, archive=body.source.archive),
                        dockerfile=DOCKERFILE_PATH,
                        platform=config.platform,
                        output=BuildOutput(repository=config.output_repository, tag=f"{source.commit}-r{revision}"),
                    )
                ],
            )
            submitted = (await builder.submit_build(workspace=workspace, body=build_set)).data()
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except NemoHTTPError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        except NemoTransportError as exc:
            raise HTTPException(status_code=502, detail=f"Couldn't reach the Helix API: {exc}") from exc
        return _build(submitted.images[0])

    @router.get("/builds/{name}", response_model=HelixBuild)
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[BuildPerms.READ])
    async def get_build(workspace: str, name: str, client: AsyncNemoClient = Depends(get_nemo_client)) -> HelixBuild:
        """Read a build; a pending one whose job failed is reported failed."""
        try:
            image = (
                await AsyncBuilderClient.from_client(client).get_container_image(workspace=workspace, name=name)
            ).data()
            if image.status == "pending":
                job_workspace, _, job_name = image.provenance.job.partition("/")
                job = (
                    await AsyncJobsClient.from_client(client).get_job_status(workspace=job_workspace, name=job_name)
                ).data()
                if job.status in (HelixJobStatus.ERROR, HelixJobStatus.CANCELLED):
                    return _build(image, status="failed", detail=f"Build job {job.status.value}: {job.error_details}")
        except NemoHTTPError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        except NemoTransportError as exc:
            raise HTTPException(status_code=502, detail=f"Couldn't reach the Helix API: {exc}") from exc
        return _build(image)

    return router
