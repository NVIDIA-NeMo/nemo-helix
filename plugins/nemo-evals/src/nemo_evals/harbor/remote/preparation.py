# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image preparation contract: the state M4 stores per task and the call that advances it.

Contract module (M1 story 5). ``prepare_task_images`` has M2 story 0's MVP body: it reads the newest
``ready`` image build of the task revision (#2881) or pins the task's prebuilt images, and returns
``ready`` or ``failed`` in one call, never ``pending``. M3 story 7 replaces the body; types change only
additively, and ``FakePreparer`` (tests/conftest.py) changes with them.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, Self, get_args

import httpx
from nemo_evals.api.task_definitions.harbor import ResolvedHarborTaskDefinition
from nemo_evals.config import get_config
from nemo_evals.harbor.remote.profile import ResolvedHarborSandboxPolicy, ResourceClass
from nemo_evals.harbor.remote.registry import ImageDigestUnavailableError, resolve_image_digest
from nemo_evals.image_builds.builder import builder_name
from nemo_evals.image_builds.models import BuildImage, ImageBuild
from nemo_evals.image_builds.planning import MAIN_SERVICE, VERIFIER_ROLE, ImagePlanError, parse_task_config, plan_images
from nemo_evals.image_builds.runner import read_task_archive
from nemo_evals.jobs.agent_spec import ResolvedTask
from nemo_evals.jobs.harbor_task_spec import PreparedHarborEnvironment, PreparedImage
from nemo_evals.taskset_runs.specs import MissingPrebuiltImagePolicy
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.evals.client import AsyncEvaluatorClient
from nemo_helix_plugin.files.client import AsyncFilesClient
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

ImageRole = Literal["agent", "verifier"]
"""``agent`` is ``image_builds``' ``BuildImage.role == "main"`` (Harbor's name for the agent container).
M3 story 8b adds ``service``."""

PreparationErrorCode = Literal[
    # Raised by image_builds.planning today (ImagePlanError, #2881)
    "invalid_task_archive",
    "windows_unsupported",
    "invalid_compose_file",
    "registry_not_allowed",
    "main_dockerfile_missing",
    "dockerfile_missing",
    # Task shape (M3 story 1)
    "invalid_task_config",
    "shared_verifier_unsupported",
    "multi_step_separate_verifier_unsupported",
    "verifier_dockerfile_missing",
    "multi_container_not_yet_supported",
    "gpu_unsupported",
    "unsupported_gpu_type",
    "gpu_request_exceeds_policy",
    "gpu_unsupported_by_backend",
    # Prebuilt images (M3 story 2)
    "image_unavailable",
    # Building (M3 stories 4-5)
    "base_image_unavailable",
    "build_failed",
    "build_timeout",
]
"""Stable strings: they end up on member records and in ``runs tasks`` output. Add codes; never rename one."""

_ROLE_BY_BUILD_ROLE: dict[str, ImageRole] = {MAIN_SERVICE: "agent", VERIFIER_ROLE: "verifier"}


class PreparationErrorInfo(BaseModel):
    """Why a task's images can't be prepared. Terminal: the member fails with ``code``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: PreparationErrorCode
    detail: str = Field(description="Human-readable reason, for logs and `runs tasks`.")
    role: ImageRole | None = Field(default=None, description="The image that failed; None for task-level errors.")


class ImagePreparation(BaseModel):
    """Progress of one image (agent or verifier). M3 stories 4-5 add optional ``staged`` / ``build`` fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: ImageRole
    preparation_key: str = Field(min_length=1)
    source: Literal["prebuilt", "built"]
    prepared: PreparedImage | None = Field(default=None, description="Set once verified; never replaced.")


class TaskPreparation(BaseModel):
    """Everything the next ``prepare_task_images`` call needs. Stored on the member by M4."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["pending", "ready", "failed"]
    resource_class: ResourceClass | None = None
    images: dict[ImageRole, ImagePreparation] = Field(default_factory=dict)
    environment: PreparedHarborEnvironment | None = None
    error: PreparationErrorInfo | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if (self.status == "ready") != (self.environment is not None):
            raise ValueError("environment is set exactly when status is 'ready'")
        if (self.status == "failed") != (self.error is not None):
            raise ValueError("error is set exactly when status is 'failed'")
        if self.status == "ready" and "agent" not in self.images:
            raise ValueError("a ready task has an agent image")
        for role, image in self.images.items():
            if image.role != role:
                raise ValueError(f"images[{role!r}] has role {image.role!r}")
        return self


@dataclass(frozen=True)
class PreparationClients:
    """Clients acting for the run's caller (M4 builds them with ``clients_for(run)``).

    M3 may add fields with defaults; it never removes one.
    """

    client: AsyncNemoClient  # build reads and submits on the caller's behalf
    files: AsyncFilesClient  # archive reads and build-context staging
    registry: httpx.AsyncClient | None = None  # anonymous tag lookups; None opens one per lookup


class ImagePreparer(Protocol):
    """One bounded step of preparation for one task. Never waits; safe to repeat after a crash.

    ``recorded=None`` is step A: decide images and keys, call no build backend.
    Otherwise step B: advance each unprepared image. Transient errors return ``recorded`` unchanged.
    """

    async def __call__(
        self,
        task: ResolvedTask,
        recorded: TaskPreparation | None,
        *,
        policy: ResolvedHarborSandboxPolicy,
        missing_prebuilt_image: MissingPrebuiltImagePolicy,
        caller: str,
        clients: PreparationClients,
        deadline: datetime,
    ) -> TaskPreparation: ...


async def prepare_task_images(
    task: ResolvedTask,
    recorded: TaskPreparation | None,
    *,
    policy: ResolvedHarborSandboxPolicy,
    missing_prebuilt_image: MissingPrebuiltImagePolicy,
    caller: str,
    clients: PreparationClients,
    deadline: datetime,
) -> TaskPreparation:
    """MVP: one call returns ``ready`` or ``failed``, never ``pending``.

    - Every image prebuilt (the task names a ``docker_image``): pin each tag to its digest.
    - Every image from a Dockerfile: use the newest ``ready`` build of the task revision (#2881).
    - Mixed, compose or GPU tasks: ``failed``. #2881's runner fails a whole build when any role is prebuilt.

    ``missing_prebuilt_image`` is not applied yet: a tag that doesn't resolve fails with ``image_unavailable``.
    """
    if recorded is not None and recorded.status != "pending":
        return recorded
    if not isinstance(task.spec, ResolvedHarborTaskDefinition):
        return _failed("invalid_task_config", f"task {task.id} is not a Harbor task")
    harbor = get_config().runs.harbor
    allowed_registries = list(harbor.allowed_registries) if harbor is not None else []
    try:
        return await _prepare(task.spec, clients=clients, allowed_registries=allowed_registries)
    except ImagePlanError as exc:
        code = exc.code if exc.code in get_args(PreparationErrorCode) else "invalid_task_config"
        return _failed(code, exc.detail)  # ty: ignore[invalid-argument-type]


async def _prepare(
    definition: ResolvedHarborTaskDefinition, *, clients: PreparationClients, allowed_registries: list[str]
) -> TaskPreparation:
    task_config = parse_task_config(definition.config)
    if task_config.environment.gpus:
        return _failed("gpu_unsupported", "GPU tasks aren't supported on Kubernetes runs yet")
    archive = await read_task_archive(clients.files, definition.source)
    if archive.compose is not None:
        return _failed("multi_container_not_yet_supported", "compose tasks aren't supported on Kubernetes runs yet")
    roles = plan_images(definition.config, archive, allowed_registries=allowed_registries)
    # plan_images treats an empty list as "no narrowing"; for runs it means no prebuilt registry is allowed.
    if not allowed_registries and (prebuilt := next((role for role in roles if role.source == "prebuilt"), None)):
        return _failed(
            "registry_not_allowed",
            f"{prebuilt.role} image {prebuilt.image} is prebuilt, and evals.runs.harbor.allowed_registries is empty",
            role=_ROLE_BY_BUILD_ROLE[prebuilt.role],
        )

    if all(role.source == "prebuilt" for role in roles):
        images: dict[ImageRole, ImagePreparation] = {}
        for role in roles:
            image_role = _ROLE_BY_BUILD_ROLE[role.role]
            assert role.image is not None
            try:
                image_ref = await resolve_image_digest(role.image, http_client=clients.registry)
            except ImageDigestUnavailableError as exc:
                return _failed("image_unavailable", str(exc), role=image_role)
            prepared = PreparedImage(image_ref=image_ref, preparation_key=image_ref.rpartition("@")[2])
            images[image_role] = ImagePreparation(
                role=image_role, preparation_key=prepared.preparation_key, source="prebuilt", prepared=prepared
            )
    elif all(role.source == "dockerfile" for role in roles):
        build = await _newest_ready_build(clients.client, definition)
        if build is None:
            return _failed(
                "image_unavailable",
                f"no ready image build for {definition.provenance.entity_name}#"
                f"{definition.provenance.revision_digest}; run `nemo evals runs submit` without "
                "--no-build-images, or POST …/image-builds",
            )
        images = {}
        for role in roles:
            image_role = _ROLE_BY_BUILD_ROLE[role.role]
            prepared = _built(build, role, definition)
            if prepared is None:
                return _failed("image_unavailable", f"build {build.id} has no ready {role.role} image", role=image_role)
            images[image_role] = ImagePreparation(
                role=image_role, preparation_key=prepared.preparation_key, source="built", prepared=prepared
            )
    else:
        return _failed("image_unavailable", "mixed prebuilt and Dockerfile images aren't supported yet")

    agent = images["agent"].prepared
    verifier = images["verifier"].prepared if "verifier" in images else None
    assert agent is not None
    environment = PreparedHarborEnvironment(
        task_revision_digest=definition.provenance.revision_digest,
        archive_digest=definition.source.files_hash,
        agent=agent,
        verifier=verifier,
    )
    return TaskPreparation(status="ready", resource_class="cpu", images=images, environment=environment)


async def _newest_ready_build(client: AsyncNemoClient, definition: ResolvedHarborTaskDefinition) -> ImageBuild | None:
    """The newest ``ready`` build of the task revision, or None. Builds list newest first."""
    workspace, _, name = definition.provenance.entity_name.partition("/")
    evaluator = AsyncEvaluatorClient.from_client(client)
    try:
        response = await evaluator.list_task_image_builds(
            workspace=workspace,
            name=name,
            revision=definition.provenance.revision_digest,
            query_params={"page": 1, "page_size": 100},
        )
    except NotFoundError:
        return None
    for item in response.page().items:
        build = ImageBuild.model_validate(item.model_dump(mode="json"))
        if build.status == "ready":
            return build
    return None


def _built(build: ImageBuild, role: BuildImage, definition: ResolvedHarborTaskDefinition) -> PreparedImage | None:
    image = next((i for i in build.images if i.role == role.role and i.status == "ready" and i.image_ref), None)
    if image is None or image.image_ref is None:
        return None
    assert role.context_path is not None and role.dockerfile is not None
    return PreparedImage(
        image_ref=image.image_ref,
        # Builder's per-archive name, without the revision suffix of its ContainerImage record.
        preparation_key=builder_name(
            archive_digest=definition.source.files_hash, context_path=role.context_path, dockerfile=role.dockerfile
        ),
        builder_image=image.builder_image,
        build_job=image.build_job,
        source_archive_digest=definition.source.files_hash,
    )


def _failed(code: PreparationErrorCode, detail: str, *, role: ImageRole | None = None) -> TaskPreparation:
    return TaskPreparation(status="failed", error=PreparationErrorInfo(code=code, detail=detail, role=role))
