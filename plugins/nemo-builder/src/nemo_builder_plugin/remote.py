# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The remote backend: a build service outside this cluster builds and publishes each image.

It names no service. It compiles one job step, which runs the image the deployment names; that image holds the
service's client (see ``run/remote.py`` for the driver such a step program runs).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePosixPath

from nemo_builder_plugin.backend import BackendRejectedError
from nemo_builder_plugin.config import BuilderConfig, RemoteConfig
from nemo_builder_plugin.plan import BuildPlan, Destination, PlannedImage
from nemo_builder_plugin.schema import ImageSource
from nemo_builder_plugin.steps import REMOTE_CREDENTIAL_ENV, ROOT_DOCKERFILE, RemoteBuildImage, RemoteBuildStepConfig
from nemo_helix_plugin.auth.workload_identity import (
    WorkloadIdentityConfigError,
    is_workload_identity_token_exchange_enabled,
)
from nemo_helix_plugin.jobs.endpoints import ExecutionProfile
from nemo_helix_plugin.jobs.providers import ContainerSpec, CPUExecutionProvider
from nemo_helix_plugin.jobs.spec import (
    HelixJobEnvironmentVariable,
    HelixJobSecretEnvironmentVariableRef,
    HelixJobSpec,
    HelixJobStepSpec,
)

STEP_NAME = "remote-build"

#: How long past its last build timeout a step needs, to cancel the build and say why.
CANCEL_MARGIN_SECONDS = 120


def is_root_dockerfile(dockerfile: str) -> bool:
    """Whether ``dockerfile`` is the one at the context's root, however it is spelt (``./Dockerfile`` too)."""
    return PurePosixPath(dockerfile).as_posix() == ROOT_DOCKERFILE


def _token_exchange() -> bool:
    try:
        return is_workload_identity_token_exchange_enabled()
    except WorkloadIdentityConfigError:
        # The auth config can't be read here. Jobs, which can, refuses the job if the secret can't be fetched.
        return False


class RemoteBackend:
    """:class:`~nemo_builder_plugin.backend.Backend` for builds a service outside this cluster runs."""

    name = "remote"

    def __init__(self, config: BuilderConfig, profiles: Sequence[ExecutionProfile]) -> None:
        self._config = config
        self._profiles = profiles

    @property
    def _remote(self) -> RemoteConfig:
        return self._config.remote

    def _placement(self) -> tuple[str, str]:
        remote = self._remote
        if not remote.registry or not remote.repository:
            raise BackendRejectedError(
                "builder.remote.registry and builder.remote.repository are not both configured; "
                "there is nowhere the build service publishes"
            )
        return remote.registry, remote.repository

    def _check_deployment(self) -> None:
        """Refuse every request while the step couldn't run: rows are written before the job is created."""
        self._placement()
        remote = self._remote
        name = remote.profile
        profile = next((p for p in self._profiles if (p.provider, p.profile) == ("cpu", name)), None)
        if profile is None:
            raise BackendRejectedError(f"Jobs has no execution profile cpu/{name} to run the remote-build step on")
        if remote.image is None and not profile.config.default_task_image:
            # Jobs would fall back to the platform's own tasks image, which holds no build service's client.
            raise BackendRejectedError(
                f"builder.remote.image is unset and Jobs execution profile cpu/{name} names no default_task_image; "
                "there is no image that holds the remote-build step"
            )
        # Jobs ends a step at the profile's limit. It counts from when the job was created, so the time a build
        # waits for a slot is in it too; that part an operator sizes, and this checks only what can't be enough.
        needed = remote.build_timeout_seconds * remote.max_specs_per_set + CANCEL_MARGIN_SECONDS
        if profile.config.ttl_seconds_active < needed:
            raise BackendRejectedError(
                f"Jobs execution profile cpu/{name} ends a step after {profile.config.ttl_seconds_active}s "
                f"(ttl_seconds_active), which is less than the {needed}s one build set may take "
                "(builder.remote.build_timeout_seconds for each of max_specs_per_set images, and a margin)"
            )
        if self._config.default_platform not in remote.platforms:
            raise BackendRejectedError(
                f"builder.default_platform is {self._config.default_platform}, which is not one of "
                f"builder.remote.platforms ({', '.join(remote.platforms)}); a request that names no platform "
                "could never build"
            )
        if remote.credential_secret is not None and _token_exchange():
            # The launcher fetches a step's secrets as the job's principal, which token exchange doesn't give it.
            raise BackendRejectedError(
                "builder.remote.credential_secret is set, but this platform uses workload token exchange, under "
                f"which Jobs cannot fetch a step's platform secrets; mount the credential on Jobs execution profile "
                f"cpu/{name} and unset credential_secret"
            )

    def check(self, plan: BuildPlan) -> None:
        """Refuse what no remote service can build, what this deployment's doesn't, and a deployment that can't."""
        self._check_deployment()
        remote = self._remote
        if len(plan.images) > remote.max_specs_per_set:
            raise BackendRejectedError(
                f"build_specs: {len(plan.images)} images in one set, and this deployment's build service takes "
                f"{remote.max_specs_per_set}"
            )
        for image in plan.images:
            spec = image.spec
            if isinstance(spec.source, ImageSource):
                raise BackendRejectedError(
                    f"{image.name}: source.image: the build service publishes what it builds, so it cannot copy an "
                    "image unchanged"
                )
            if spec.output is not None:
                raise BackendRejectedError(
                    f"{image.name}: output: the build service publishes every image to one repository, told apart "
                    "by digest; a request cannot name a repository or a tag"
                )
            if spec.mounts:
                raise BackendRejectedError(f"{image.name}: mounts: the build service takes one build context")
            if image.platform not in remote.platforms:
                raise BackendRejectedError(
                    f"{image.name}: platform: the build service builds {', '.join(remote.platforms)}, "
                    f"not {image.platform}"
                )
            if not is_root_dockerfile(spec.dockerfile) and not remote.allow_dockerfile_path:
                raise BackendRejectedError(
                    f"{image.name}: dockerfile: the build service builds the Dockerfile at the context's root, "
                    f"not {spec.dockerfile!r}"
                )

    def destination(self, image: PlannedImage) -> Destination:
        """The service's one repository, whatever the workspace: an image is told apart by its digest."""
        registry, repository = self._placement()
        # No tag is pushed. The field is required, so it holds the row's name, which is always a valid tag; the
        # in-cluster tag would put the workspace's name in it, and not every workspace's name can be in a tag.
        return Destination(registry=registry, repository=repository, system_tag=image.name)

    def compile(self, plan: BuildPlan) -> HelixJobSpec:
        registry, repository = self._placement()
        remote = self._remote
        environment = (
            [
                HelixJobEnvironmentVariable(
                    name=REMOTE_CREDENTIAL_ENV,
                    from_secret=HelixJobSecretEnvironmentVariableRef(name=remote.credential_secret),
                )
            ]
            if remote.credential_secret is not None
            else None
        )
        # Named only when the deployment names one: unset, Jobs resolves the profile's `default_task_image`.
        container = (
            ContainerSpec(image=remote.image, command=list(remote.command))
            if remote.image is not None
            else ContainerSpec(command=list(remote.command))
        )
        # No time limit of its own: the step enforces each image's timeout itself, and Jobs counts a step's limit
        # from the job's creation, so one sized to the build would end every build that had waited for a slot. The
        # profile's limit, which `_check_deployment` holds above the build's, ends a step that hangs.
        step = HelixJobStepSpec(
            name=STEP_NAME,
            executor=CPUExecutionProvider(
                # Explicit though it is the default: the Jobs client serializes with `exclude_unset`.
                provider="cpu",
                profile=remote.profile,
                container=container,
            ),
            environment=environment,
            # A step can't read the platform's config, so everything it needs travels here.
            config=RemoteBuildStepConfig(
                images=[
                    RemoteBuildImage(
                        image=image.name,
                        source=image.source,
                        dockerfile=image.spec.dockerfile,
                        platform=image.platform,
                    )
                    for image in plan.images
                ],
                registry=registry,
                repository=repository,
                build_timeout_seconds=remote.build_timeout_seconds,
                credential_in_environment=remote.credential_secret is not None,
            ).model_dump(),
        )
        return HelixJobSpec(steps=[step])
