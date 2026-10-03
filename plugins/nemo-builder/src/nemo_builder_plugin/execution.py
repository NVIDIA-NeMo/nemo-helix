# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The execution backend: build in this cluster, in a sandbox that holds no credential."""

from __future__ import annotations

from nemo_builder_plugin.backend import BackendRefused
from nemo_builder_plugin.compile import compile_build_set
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.identity import compose_system_tag, validate_repository
from nemo_builder_plugin.plan import BuildPlan, Destination, PlannedImage
from nemo_helix_plugin.jobs.spec import HelixJobSpec


class ExecutionBackend:
    """:class:`~nemo_builder_plugin.backend.Backend` for builds in this cluster."""

    def __init__(self, config: BuilderConfig) -> None:
        self._config = config

    def _registry(self) -> str:
        if not self._config.registry:
            raise BackendRefused("builder.registry is not configured; there is nowhere to publish the result")
        return self._config.registry

    def _sandbox_image(self) -> str:
        if not self._config.sandbox_image:
            raise BackendRefused("builder.sandbox_image is not configured; there is no kaniko image to build with")
        return self._config.sandbox_image

    def check(self, plan: BuildPlan) -> None:
        """Refuse every request while a setting with no default is unset."""
        self._registry()
        self._sandbox_image()

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
        )
