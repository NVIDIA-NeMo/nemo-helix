# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image preparation contract: the state M4 stores per task and the calls that advance it.

Contract module (M1 story 5). ``ImageBuildPreparer`` is M2 story 0's MVP: ``check`` rejects what a
Kubernetes run can't take, then one call reads the task revision's image build through ``ImageBuildService``
(read-only: it never starts one) and returns ``ready`` or ``failed``, never ``pending``.
TODO(M3 story 7): store the pending build and poll it instead of failing. Types change only additively,
and ``FakePreparer`` (tests/conftest.py) changes with them.
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, Self, get_args

from harbor.models.task.verifier_mode import resolve_effective_verifier_env_config
from nemo_evals.agent_layers.models import AgentLayer
from nemo_evals.api.task_definitions.harbor import ResolvedHarborTaskDefinition
from nemo_evals.harbor.compose.models import ComposeError, ComposeTask
from nemo_evals.harbor.compose.spec import check_compose_task
from nemo_evals.harbor.remote.profile import ResolvedHarborSandboxPolicy, ResourceClass
from nemo_evals.harbor.remote.sandbox_policy import compose_resources_violation, task_resources_violation
from nemo_evals.image_builds.models import ImageBuild
from nemo_evals.image_builds.planning import (
    MAIN_SERVICE,
    SERVICE_TYPE_PREFIX,
    VERIFIER_TYPE,
    ImagePlanError,
    parse_task_config,
)
from nemo_evals.image_builds.runner import read_task_archive
from nemo_evals.image_builds.service import ImageBuildService
from nemo_evals.jobs.agent_spec import ResolvedTask
from nemo_evals.jobs.harbor_task_spec import PreparedHarborEnvironment, PreparedImage
from nemo_evals.revisions import RevisionNotFoundError
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NemoHTTPError, PermissionDeniedError
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError
from nemo_helix_plugin.files.client import AsyncFilesClient
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

ImageType = Literal["agent", "verifier"]
"""``agent`` is ``image_builds``' ``BuildImage.type == "main"`` (Harbor's name for the agent container)."""

ErrorImageType = Literal["agent", "verifier", "service"]
"""The image an error names. ``service`` is one of the task's compose services, which the error's detail names.
Service images are recorded only in ``PreparedHarborEnvironment.compose_images``: a variable number of them can't be
keyed by type, and how their progress is stored is for when preparation can be pending."""

PreparationErrorCode = Literal[
    # Raised by image_builds.planning today (ImagePlanError, #2881)
    "invalid_task_archive",
    "windows_unsupported",
    "invalid_compose_file",
    "registry_not_allowed",
    "main_dockerfile_missing",
    "dockerfile_missing",
    "unsupported_compose_build_option",
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
    "resources_exceed_policy",
    # Prebuilt images (M3 story 2)
    "image_unavailable",
    # Building (M3 stories 4-5)
    "base_image_unavailable",
    "build_failed",
    "build_timeout",
]
"""Stable strings: they end up on member records and in ``runs tasks`` output. Add codes; never rename one."""

_PREPARATION_TYPE_BY_BUILD_TYPE: dict[str, ImageType] = {MAIN_SERVICE: "agent", VERIFIER_TYPE: "verifier"}


class PreparationErrorInfo(BaseModel):
    """Why a task's images can't be prepared. Terminal: the member fails with ``code``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: PreparationErrorCode
    detail: str = Field(description="Human-readable reason, for logs and `runs tasks`.")
    image_type: ErrorImageType | None = Field(
        default=None, description="The image that failed; None for task-level errors."
    )


class ImagePreparation(BaseModel):
    """Progress of one image (agent or verifier).

    TODO(M3 stories 4-5): add optional ``staged`` / ``build`` fields.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: ImageType
    preparation_key: str = Field(min_length=1)
    source: Literal["prebuilt", "built"]
    prepared: PreparedImage | None = Field(default=None, description="Set once verified; never replaced.")


class TaskPreparation(BaseModel):
    """Everything the next ``prepare_task_images`` call needs. Stored on the member by M4."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["pending", "ready", "failed"]
    resource_class: ResourceClass | None = None
    images: dict[ImageType, ImagePreparation] = Field(default_factory=dict)
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
        for image_type, image in self.images.items():
            if image.type != image_type:
                raise ValueError(f"images[{image_type!r}] has type {image.type!r}")
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
    #: The task's ready agent layer for the run's agent, by ``(workspace, name, revision)``; None skips layers.
    agent_layer: Callable[[str, str, str], Awaitable[AgentLayer | None]] | None = None


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
        """Reject non-Harbor, GPU (agent or separate verifier) and compose tasks, and tasks whose sandboxes exceed
        the profile's ceilings.

        Build roles can't show compose: a compose file whose only service is ``main`` builds just ``main``.
        So the archive is still read, for its compose file only.
        """
        if not isinstance(task.spec, ResolvedHarborTaskDefinition):
            return PreparationErrorInfo(code="invalid_task_config", detail=f"task {task.id} is not a Harbor task")
        try:
            task_config = parse_task_config(task.spec.config)
            verifier = resolve_effective_verifier_env_config(task_config, None)
            if task_config.environment.gpus or (verifier is not None and verifier.gpus):
                return PreparationErrorInfo(
                    code="gpu_unsupported", detail="GPU tasks aren't supported on Kubernetes runs yet"
                )
            if (violation := task_resources_violation(task_config, policy)) is not None:
                return PreparationErrorInfo(code="resources_exceed_policy", detail=violation)
            archive = await read_task_archive(clients.files, task.spec.source)
        except ImagePlanError as exc:
            return _plan_error(exc)
        except NemoHTTPError as exc:
            raise ImagePreparationUnavailableError(f"could not read task {task.id}'s archive: {exc}") from exc
        if archive.compose_task is not None:
            return _check_compose(archive.compose_task, policy)
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

        Reads the revision's newest build for the deployment's default platform and never starts one, so it needs
        no Builder permission: a task whose newest build isn't ``ready`` fails until someone builds it.

        Raises:
            ImagePreparationUnavailableError: Reading the task's builds failed transiently.
        """
        if recorded is not None and recorded.status != "pending":
            return recorded
        if not isinstance(task.spec, ResolvedHarborTaskDefinition):
            return _failed("invalid_task_config", f"task {task.id} is not a Harbor task")
        if clients.image_builds is None:
            raise ValueError("PreparationClients.image_builds is required to prepare Harbor task images")
        definition = task.spec
        workspace, _, name = definition.provenance.entity_name.partition("/")
        revision = definition.provenance.revision_digest
        # A run takes images for the deployment's default platform: another platform's build, ready or failed,
        # is no answer.
        try:
            build = await clients.image_builds(workspace).newest_build(workspace, name, revision)
        except NemoEntityNotFoundError:
            return _failed("invalid_task_config", f"task {definition.provenance.entity_name} no longer exists")
        except RevisionNotFoundError as exc:
            return _failed("invalid_task_config", str(exc))
        except PermissionDeniedError as exc:
            return _failed("image_unavailable", f"could not read the image builds: {exc}")
        except NemoHTTPError as exc:
            raise ImagePreparationUnavailableError(f"could not read the image builds of task {task.id}: {exc}") from exc
        if build is None:
            return _failed(
                "image_unavailable",
                f"{definition.provenance.entity_name}#{revision} has no image build; build it first "
                "(`nemo evals runs submit` builds and waits)",
            )
        preparation = _from_build(build, definition)
        if clients.agent_layer is None or preparation.environment is None:
            return preparation
        try:
            layer = await clients.agent_layer(workspace, name, revision)
        except (NemoHTTPError, NemoEntityNotFoundError, RevisionNotFoundError) as exc:
            # A layer only saves the trial its install: without one the task still runs, installing in setup.
            logger.warning("Could not read the agent layer of task %s; its trials install the agent: %s", task.id, exc)
            return preparation
        return _with_agent_layer(preparation, layer) if layer is not None else preparation


def _with_agent_layer(preparation: TaskPreparation, layer: AgentLayer) -> TaskPreparation:
    """``preparation`` with its agent image replaced by ``layer``'s, the same image with the agent installed."""
    image = layer.image
    if image is None or image.status != "ready" or image.image_ref is None or preparation.environment is None:
        return preparation
    agent = PreparedImage(
        image_ref=image.image_ref,
        preparation_key=image.image_ref.rpartition("@")[2],
        builder_image=image.builder_image,
        build_job=image.build_job,
        source_archive_digest=preparation.environment.agent.source_archive_digest,
        runtime_config=image.runtime_config,
    )
    images = dict(preparation.images)
    images["agent"] = ImagePreparation(
        type="agent", preparation_key=agent.preparation_key, source="built", prepared=agent
    )
    environment = preparation.environment.model_copy(update={"agent": agent, "agent_preinstalled": True})
    return preparation.model_copy(update={"images": images, "environment": environment})


def _check_compose(task: ComposeTask, policy: ResolvedHarborSandboxPolicy) -> PreparationErrorInfo | None:
    """Reject a compose task that can't run as a pod, or whose services declare more than the policy allows.

    The parser's own message is the detail, so users see which key of which file is the problem.
    """
    try:
        services = check_compose_task(task)
    except ComposeError as exc:
        return PreparationErrorInfo(code="invalid_compose_file", detail=str(exc))
    if (violation := compose_resources_violation(services, policy)) is not None:
        return PreparationErrorInfo(code="resources_exceed_policy", detail=violation)
    return None


prepare_task_images = ImageBuildPreparer()
"""The deployment's preparer."""


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
    images: dict[ImageType, ImagePreparation] = {}
    compose_images: dict[str, PreparedImage] = {}
    for image in build.images:
        image_type = _PREPARATION_TYPE_BY_BUILD_TYPE.get(image.type)
        if image.status != "ready" or image.image_ref is None:
            return _failed(
                "image_unavailable",
                f"image build {build.id} has no ready {image.type} image",
                image_type=image_type or "service",
            )
        built = image.source == "dockerfile"
        prepared = PreparedImage(
            image_ref=image.image_ref,
            preparation_key=image.image_ref.rpartition("@")[2],
            builder_image=image.builder_image,
            build_job=image.build_job,
            source_archive_digest=definition.source.files_hash if built else None,
            runtime_config=image.runtime_config,
        )
        if image_type is None:
            # The task Job looks up a service's image by the service's own name.
            compose_images[image.type.removeprefix(SERVICE_TYPE_PREFIX)] = prepared
            continue
        source: Literal["prebuilt", "built"] = "built" if built else "prebuilt"
        images[image_type] = ImagePreparation(
            type=image_type, preparation_key=prepared.preparation_key, source=source, prepared=prepared
        )
    agent = images["agent"].prepared if "agent" in images else None
    if agent is None:
        return _failed("image_unavailable", f"image build {build.id} of {target} has no main image")
    environment = PreparedHarborEnvironment(
        task_revision_digest=definition.provenance.revision_digest,
        archive_digest=definition.source.files_hash,
        agent=agent,
        verifier=images["verifier"].prepared if "verifier" in images else None,
        compose_images=compose_images,
    )
    return TaskPreparation(status="ready", resource_class="cpu", images=images, environment=environment)


def _plan_error(exc: ImagePlanError) -> PreparationErrorInfo:
    code = exc.code if exc.code in get_args(PreparationErrorCode) else "invalid_task_config"
    return PreparationErrorInfo(code=code, detail=exc.detail)  # ty: ignore[invalid-argument-type]


def _failed(code: PreparationErrorCode, detail: str, *, image_type: ErrorImageType | None = None) -> TaskPreparation:
    """A failed preparation with ``code`` and ``detail``; ``image_type`` names the image that failed, if one did."""
    return TaskPreparation(status="failed", error=PreparationErrorInfo(code=code, detail=detail, image_type=image_type))
