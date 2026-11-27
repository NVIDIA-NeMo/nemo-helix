# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The remote backend: a build service outside this cluster builds and publishes each image.

It names no service. It compiles one job step that runs the deployment's step image, which holds the service's
client and runs :func:`nemo_builder_plugin.run.remote.step_main`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePosixPath

from nemo_builder_plugin.backend import BackendRejectedError
from nemo_builder_plugin.config import BuilderConfig, RemoteConfig
from nemo_builder_plugin.identity import compose_system_tag, validate_repository
from nemo_builder_plugin.plan import BuildPlan, Destination, PlannedImage
from nemo_builder_plugin.steps import REMOTE_CREDENTIAL_ENV, RemoteBuildImage, RemoteBuildStepConfig
from nemo_helix_plugin.jobs.endpoints import ExecutionProfile
from nemo_helix_plugin.jobs.providers import ContainerSpec, CPUExecutionProvider
from nemo_helix_plugin.jobs.spec import (
    HelixJobEnvironmentVariable,
    HelixJobSecretEnvironmentVariableRef,
    HelixJobSpec,
    HelixJobStepSpec,
)

STEP_NAME = "remote-build"


class RemoteBackend:
    """:class:`~nemo_builder_plugin.backend.Backend` for builds a service outside this cluster runs."""

    def __init__(self, config: BuilderConfig, profiles: Sequence[ExecutionProfile]) -> None:
        self._config = config
        self._profiles = profiles

    def _remote(self) -> RemoteConfig:
        if self._config.remote is None:
            raise BackendRejectedError("builder.backend is remote, but builder.remote is not configured")
        return self._config.remote

    def check(self, plan: BuildPlan) -> None:
        """Refuse a deployment whose step can't run, and what the service can't build: it takes one context with
        a root Dockerfile, and publishes to its one repository."""
        remote = self._remote()
        if not any((p.provider, p.profile) == ("cpu", remote.profile) for p in self._profiles):
            raise BackendRejectedError(f"Jobs has no execution profile cpu/{remote.profile} to run a build step on")
        for image in plan.images:
            spec = image.spec
            if image.upstream is not None:
                refusal = "source.image: the build service builds from a context; it cannot copy an image"
            elif spec.output is not None:
                refusal = "output: the build service publishes to one repository, so a request cannot name one or a tag"
            elif spec.mounts:
                refusal = "mounts: the build service takes one build context"
            elif image.platform not in remote.platforms:
                refusal = f"platform: the build service builds {', '.join(remote.platforms)}, not {image.platform}"
            elif PurePosixPath(spec.dockerfile).as_posix() != "Dockerfile":
                refusal = f"dockerfile: the build service builds the context's root Dockerfile, not {spec.dockerfile!r}"
            else:
                continue
            raise BackendRejectedError(f"{image.name}: {refusal}")

    def destination(self, image: PlannedImage) -> Destination:
        """Under the workspace's own path, as in-cluster builds are, unless the service publishes to one
        repository only (`per_workspace: false`). Either way with a per-row system tag."""
        remote = self._remote()
        repository = remote.repository
        if remote.per_workspace:
            parts = (remote.repository, image.workspace, image.build_set, image.spec.name)
            repository = validate_repository("/".join(part for part in parts if part))
        return Destination(
            registry=remote.registry,
            repository=repository,
            system_tag=compose_system_tag(image.workspace, image.name),
        )

    def compile(self, plan: BuildPlan) -> HelixJobSpec:
        """One step. The credential reaches it only from a platform secret, which submit refuses a caller who
        can't read."""
        remote = self._remote()
        environment = None
        if remote.credential_secret is not None:
            secret = HelixJobSecretEnvironmentVariableRef(name=remote.credential_secret)
            environment = [HelixJobEnvironmentVariable(name=REMOTE_CREDENTIAL_ENV, from_secret=secret)]
        config = RemoteBuildStepConfig(
            images=[
                RemoteBuildImage(
                    name=image.name,
                    source=image.source,
                    platform=image.platform,
                    registry=image.placed.registry,
                    repository=image.placed.repository,
                    tag=image.placed.system_tag,
                )
                for image in plan.images
            ],
            build_timeout_seconds=remote.build_timeout_seconds,
            settings=remote.settings,
            uses_credential=remote.credential_secret is not None,
        )
        step = HelixJobStepSpec(
            name=STEP_NAME,
            executor=CPUExecutionProvider(
                # Explicit though it is the default: the Jobs client serializes with `exclude_unset`.
                provider="cpu",
                profile=remote.profile,
                container=ContainerSpec(image=remote.image, command=list(remote.command)),
            ),
            environment=environment,
            config=config.model_dump(),
        )
        return HelixJobSpec(steps=[step])
