# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image preparation contract: the state M4 stores per task and the calls that advance it.

Contract module (M1 story 5). ``ImageBuildPreparer`` is M2 story 0's MVP: ``check`` rejects what a
Kubernetes run can't take, then one call reads the task revision's image build through ``ImageBuildService``
(starting or repairing it when needed) and returns ``ready`` or ``failed``, never ``pending``.
TODO(M3 story 7): store the pending build and poll it instead of failing. Types change only additively,
and ``FakePreparer`` (tests/conftest.py) changes with them.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, Self, get_args

from nemo_evals.api.task_definitions.harbor import ResolvedHarborTaskDefinition
from nemo_evals.harbor.remote.profile import ResolvedHarborSandboxPolicy, ResourceClass
from nemo_evals.image_builds.models import ImageBuild, ImageBuildRequest
from nemo_evals.image_builds.planning import MAIN_SERVICE, VERIFIER_ROLE, ImagePlanError, parse_task_config
from nemo_evals.image_builds.runner import read_task_archive
from nemo_evals.image_builds.service import ImageBuildForbiddenError, ImageBuildJobConflictError, ImageBuildService
from nemo_evals.jobs.agent_spec import ResolvedTask
from nemo_evals.jobs.harbor_task_spec import PreparedHarborEnvironment, PreparedImage
from nemo_evals.revisions import RevisionNotFoundError
from nemo_evals.task_identity import UnsupportedTaskKindError
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NemoHTTPError, PermissionDeniedError
from nemo_helix_plugin.files.client import AsyncFilesClient
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

ImageRole = Literal["agent", "verifier"]
"""``agent`` is ``image_builds``' ``BuildImage.role == "main"`` (Harbor's name for the agent container).
TODO(M3 story 8b): add ``service``."""

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
    """Progress of one image (agent or verifier).

    TODO(M3 stories 4-5): add optional ``staged`` / ``build`` fields.
    """

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


class ImagePreparationUnavailableError(Exception):
    """A service preparation depends on failed transiently; the submit can be retried as is."""


@dataclass(frozen=True)
class PreparationClients:
    """Clients acting for the run's caller. M3 may add fields with defaults.

    TODO(M4): build them with ``clients_for(run)``.
    """

    client: AsyncNemoClient  # reads and Jobs on the caller's behalf
    files: AsyncFilesClient  # task archive reads
    image_builds: Callable[[str], ImageBuildService] | None = None  # the build service for a task's workspace


class ImagePreparer(Protocol):
    """Prepares one task's images in bounded steps. Never waits; safe to repeat after a crash."""

    async def check(
        self, task: ResolvedTask, *, policy: ResolvedHarborSandboxPolicy, clients: PreparationClients
    ) -> PreparationErrorInfo | None:
        """Why a run can't take the task, or None. No side effects: submit checks every task before any build."""
        ...

    async def __call__(
        self,
        task: ResolvedTask,
        recorded: TaskPreparation | None,
        *,
        policy: ResolvedHarborSandboxPolicy,
        caller: str,
        clients: PreparationClients,
        deadline: datetime,
    ) -> TaskPreparation:
        """Advance a checked task's preparation by one step.

        ``recorded=None`` is step A: decide images and keys. Otherwise step B: advance each unprepared image.
        Transient errors return ``recorded`` unchanged, or raise ``ImagePreparationUnavailableError`` before any
        state is recorded.
        """
        ...


class ImageBuildPreparer:
    """M2 story 0's preparer: images come from the task revision's image build (#2881, #2986)."""

    async def check(
        self, task: ResolvedTask, *, policy: ResolvedHarborSandboxPolicy, clients: PreparationClients
    ) -> PreparationErrorInfo | None:
        """Reject non-Harbor, GPU and compose tasks.

        Build roles can't show compose: a compose file whose only service is ``main`` builds just ``main``.
        So the archive is still read, for its compose file only.
        """
        if not isinstance(task.spec, ResolvedHarborTaskDefinition):
            return PreparationErrorInfo(code="invalid_task_config", detail=f"task {task.id} is not a Harbor task")
        try:
            task_config = parse_task_config(task.spec.config)
            if task_config.environment.gpus:
                return PreparationErrorInfo(
                    code="gpu_unsupported", detail="GPU tasks aren't supported on Kubernetes runs yet"
                )
            archive = await read_task_archive(clients.files, task.spec.source)
        except ImagePlanError as exc:
            return _plan_error(exc)
        if archive.compose is not None:
            return PreparationErrorInfo(
                code="multi_container_not_yet_supported", detail="compose tasks aren't supported on Kubernetes runs yet"
            )
        return None

    async def __call__(
        self,
        task: ResolvedTask,
        recorded: TaskPreparation | None,
        *,
        policy: ResolvedHarborSandboxPolicy,
        caller: str,
        clients: PreparationClients,
        deadline: datetime,
    ) -> TaskPreparation:
        """MVP: one call returns ``ready`` or ``failed``, never ``pending``.

        Raises:
            ImagePreparationUnavailableError: The Jobs service failed while starting or repairing a build.
        """
        if recorded is not None and recorded.status != "pending":
            return recorded
        if not isinstance(task.spec, ResolvedHarborTaskDefinition):
            return _failed("invalid_task_config", f"task {task.id} is not a Harbor task")
        if clients.image_builds is None:
            raise ValueError("PreparationClients.image_builds is required to prepare Harbor task images")
        definition = task.spec
        workspace, _, name = definition.provenance.entity_name.partition("/")
        try:
            build = await _current_build(
                clients.image_builds(workspace), workspace, name, definition.provenance.revision_digest
            )
        except (RevisionNotFoundError, UnsupportedTaskKindError) as exc:
            return _failed("invalid_task_config", str(exc))
        except ImagePlanError as exc:
            return TaskPreparation(status="failed", error=_plan_error(exc))
        except (ImageBuildForbiddenError, PermissionDeniedError) as exc:
            return _failed("image_unavailable", f"could not start an image build: {exc}")
        except ImageBuildJobConflictError as exc:
            return _failed("image_unavailable", f"could not start an image build: {exc}")
        except NemoHTTPError as exc:
            raise ImagePreparationUnavailableError(f"could not start an image build for task {task.id}: {exc}") from exc
        if build is None:
            return _failed("invalid_task_config", f"task {definition.provenance.entity_name} no longer exists")
        return _from_build(build, definition)


prepare_task_images = ImageBuildPreparer()
"""The deployment's preparer."""


async def _current_build(builds: ImageBuildService, workspace: str, name: str, revision: str) -> ImageBuild | None:
    """The revision's newest build, after ``request_build`` starts or repairs it; None when the task is gone.

    A ready or failed build is returned as is, so a broken task fails here instead of starting attempt after
    attempt. Only a missing, pending or running build goes through ``request_build``: that starts the first
    attempt, recreates a lost Job, or replaces a build whose Job ended without a result. ``request_build``
    checks build permission first, so a caller without it still learns that someone else's build is running.
    """
    page = await builds.list_builds(workspace, name, revision, page_size=1)
    if page is None:
        return None
    newest = page.data[0] if page.data else None
    if newest is not None and newest.status in ("ready", "failed"):
        return newest
    try:
        return await builds.request_build(workspace, name, revision, ImageBuildRequest())
    except ImageBuildForbiddenError:
        if newest is None:
            raise
        return newest


def _from_build(build: ImageBuild, definition: ResolvedHarborTaskDefinition) -> TaskPreparation:
    target = f"{definition.provenance.entity_name}#{definition.provenance.revision_digest}"
    if build.status == "failed":
        return _failed("build_failed", f"image build {build.id} of {target} failed: {build.error or 'no reason given'}")
    if build.status != "ready":
        return _failed(
            "image_unavailable",
            f"image build {build.id} of {target} is {build.status}; resubmit when it's ready "
            "(`nemo evals runs submit` waits for it)",
        )
    images: dict[ImageRole, ImagePreparation] = {}
    for image in build.images:
        role = _ROLE_BY_BUILD_ROLE.get(image.role)
        if role is None:
            return _failed(
                "multi_container_not_yet_supported", f"image build {build.id} of {target} has a {image.role} image"
            )
        if image.status != "ready" or image.image_ref is None:
            return _failed("image_unavailable", f"image build {build.id} has no ready {image.role} image", role=role)
        built = image.source == "dockerfile"
        prepared = PreparedImage(
            image_ref=image.image_ref,
            preparation_key=image.image_ref.rpartition("@")[2],
            builder_image=image.builder_image,
            build_job=image.build_job,
            source_archive_digest=definition.source.files_hash if built else None,
        )
        source: Literal["prebuilt", "built"] = "built" if built else "prebuilt"
        images[role] = ImagePreparation(
            role=role, preparation_key=prepared.preparation_key, source=source, prepared=prepared
        )
    agent = images["agent"].prepared if "agent" in images else None
    if agent is None:
        return _failed("image_unavailable", f"image build {build.id} of {target} has no main image")
    environment = PreparedHarborEnvironment(
        task_revision_digest=definition.provenance.revision_digest,
        archive_digest=definition.source.files_hash,
        agent=agent,
        verifier=images["verifier"].prepared if "verifier" in images else None,
    )
    return TaskPreparation(status="ready", resource_class="cpu", images=images, environment=environment)


def _plan_error(exc: ImagePlanError) -> PreparationErrorInfo:
    code = exc.code if exc.code in get_args(PreparationErrorCode) else "invalid_task_config"
    return PreparationErrorInfo(code=code, detail=exc.detail)  # ty: ignore[invalid-argument-type]


def _failed(code: PreparationErrorCode, detail: str, *, role: ImageRole | None = None) -> TaskPreparation:
    return TaskPreparation(status="failed", error=PreparationErrorInfo(code=code, detail=detail, role=role))
