# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The execution backend: build in this cluster, in a sandbox that holds no credential."""

from __future__ import annotations

from collections.abc import Sequence

from nemo_builder_plugin.backend import BackendRejectedError
from nemo_builder_plugin.compile import compile_build_set
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.identity import compose_system_tag, validate_repository
from nemo_builder_plugin.plan import BuildPlan, Destination, PlannedImage
from nemo_helix_plugin.jobs.endpoints import ExecutionProfile
from nemo_helix_plugin.jobs.execution_profiles import (
    KubernetesJobExecutionProfile,
    KubernetesJobExecutionProfileConfig,
)
from nemo_helix_plugin.jobs.spec import HelixJobSpec


class ExecutionBackend:
    """:class:`~nemo_builder_plugin.backend.Backend` for builds in this cluster."""

    def __init__(self, config: BuilderConfig, profiles: Sequence[ExecutionProfile]) -> None:
        """``profiles`` are the platform's Jobs execution profiles, which say where the build's steps run."""
        self._config = config
        self._profiles = profiles

    def _registry(self) -> str:
        if not self._config.registry:
            raise BackendRejectedError("builder.registry is not configured; there is nowhere to publish the result")
        return self._config.registry

    def _sandbox_image(self) -> str:
        if not self._config.sandbox.image:
            raise BackendRejectedError(
                "builder.sandbox.image is not configured; there is no kaniko image to build with"
            )
        return self._config.sandbox.image

    def _profile(self, name: str) -> KubernetesJobExecutionProfileConfig:
        profile = next((p for p in self._profiles if (p.provider, p.profile) == ("cpu", name)), None)
        if profile is None:
            raise BackendRejectedError(f"Jobs has no execution profile cpu/{name} to run a build step on")
        if not isinstance(profile, KubernetesJobExecutionProfile):
            raise BackendRejectedError(
                f"Jobs execution profile cpu/{name} runs on {profile.backend}, but builds need kubernetes_job: "
                "each runs its Dockerfiles in a pod of its own"
            )
        return profile.config

    def _work_profile(self) -> KubernetesJobExecutionProfileConfig:
        """The fetch step's profile, whose work volume and nodes the sandbox uses.

        The sandbox mounts the volume ``fetch`` writes the contexts to and ``push`` reads the layouts from, in the
        namespace the build step runs in, so the three profiles must agree on both.
        """
        names = (self._config.fetch_profile, self._config.control_profile, self._config.push_profile)
        fetch, control, push = (self._profile(name) for name in names)
        if not fetch.storage.pvc_name:
            raise BackendRejectedError(f"Jobs execution profile cpu/{names[0]} names no work volume (storage.pvc_name)")
        if not fetch.namespace == control.namespace == push.namespace:
            raise BackendRejectedError(
                "Jobs execution profiles " + ", ".join(f"cpu/{name}" for name in names) + " must name one namespace"
            )
        if push.storage.pvc_name != fetch.storage.pvc_name:
            raise BackendRejectedError(
                f"Jobs execution profiles cpu/{names[0]} and cpu/{names[2]} must name one work volume (storage.pvc_name)"
            )
        return fetch

    def check(self, plan: BuildPlan) -> None:
        """Refuse every request while a setting with no default is unset, or the build's profiles disagree."""
        self._registry()
        self._sandbox_image()
        if self._config.sandbox.provider == "opensandbox" and self._config.sandbox.opensandbox is None:
            raise BackendRejectedError(
                "builder.sandbox.opensandbox is not configured; the opensandbox provider has no server to build on"
            )
        self._work_profile()

    def destination(self, image: PlannedImage) -> Destination:
        """The deployment's registry, under the submitting workspace, with a per-row system tag.

        The workspace in the path keeps each workspace's images under its own repositories.
        """
        spec = image.spec
        tail = spec.output.repository if spec.output else "/".join(p for p in (image.build_set, spec.name) if p)
        repository = "/".join(part for part in (self._config.repository_prefix, image.workspace, tail) if part)
        # Checked whole: the workspace is a platform name, and `NAME_PATTERN` admits spellings
        # (`@`, `+`, `..`) that a repository component does not.
        validate_repository(repository)
        return Destination(
            registry=self._registry(),
            repository=repository,
            system_tag=compose_system_tag(image.workspace, image.name),
        )

    def compile(self, plan: BuildPlan) -> HelixJobSpec:
        return compile_build_set(
            plan,
            config=self._config,
            registry=self._registry(),
            sandbox_image=self._sandbox_image(),
            work_profile=self._work_profile(),
        )
