# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The execution backend: build in this cluster, in a sandbox that holds no credential.

The whole backend in one place: where it publishes, what job it compiles, how its signatures are
checked, and what the credential broker may hand its steps.

- **Destination: the deployment's.** One registry, and a path under the submitting workspace --
  ``<prefix>/<workspace>/<output.repository>``, or ``<prefix>/<workspace>/<set>/<spec>`` when the
  spec names none -- plus a system tag that is unique per row.
- **Publisher: the push step**, with a registry token the broker scopes to this job's destinations.
- **Signer: the broker**, over a digest it reads from the registry itself, with a key no step can
  read. So its signatures are verified against the deployment's public key.
"""

from __future__ import annotations

from collections.abc import Mapping

from nemo_builder_plugin.backend import BackendRefused, StepEntitlement
from nemo_builder_plugin.compile import compile_build_set
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.identity import compose_system_tag, validate_repository
from nemo_builder_plugin.plan import BuildPlan, Destination, PlannedImage
from nemo_builder_plugin.signing import DeploymentKeyPolicy
from nemo_helix_plugin.jobs.spec import HelixJobSpec

#: The step name `compile` gives the push step, and the one role this backend's broker serves.
PUSH_STEP = "push"


class ExecutionBackend:
    """:class:`~nemo_builder_plugin.backend.Backend` for builds in this cluster."""

    def __init__(self, config: BuilderConfig) -> None:
        self._config = config

    # --- What the deployment must have -------------------------------------------------------

    def _registry(self) -> str:
        if not self._config.registry:
            raise BackendRefused("builder.registry is not configured; there is nowhere to publish the result")
        return self._config.registry

    def _sandbox_image(self) -> str:
        if not self._config.sandbox_image:
            raise BackendRefused("builder.sandbox_image is not configured; there is no kaniko image to build with")
        return self._config.sandbox_image

    def _broker(self) -> str:
        if not self._config.credential_broker:
            raise BackendRefused(
                "builder.credential_broker is not configured. Every credential the push step uses, "
                "and every signature, comes from the broker; without one nothing could publish."
            )
        return self._config.credential_broker

    def _public_key(self) -> str:
        if not self._config.signing_public_key:
            raise BackendRefused(
                "builder.signing_public_key is not configured. A row goes `ready` only on a signature "
                "verified against it, so a build nothing could verify is refused rather than run."
            )
        return self._config.signing_public_key

    # --- The interface -----------------------------------------------------------------------

    def check(self, plan: BuildPlan) -> None:
        """Refuse every request while a setting with no default is unset. Nothing else is refused."""
        self._registry()
        self._sandbox_image()
        self._broker()
        self._public_key()

    def destination(self, image: PlannedImage) -> Destination:
        """The deployment's registry, under the submitting workspace, with a per-row system tag.

        **The workspace component is the isolation between tenants.** The broker grants a push
        step only repositories under its own job's workspace, so a path composed here can never be
        one another workspace's build could write. ``output.repository`` is checked component by
        component at the schema, so it cannot climb out of the workspace's part of the path.
        """
        spec = image.spec
        tail = spec.output.repository if spec.output else f"{image.build_set}/{spec.name}"
        repository = "/".join(part for part in (self._config.repository_prefix, image.workspace, tail) if part)
        # Checked whole, because the workspace is a platform name and `NAME_PATTERN` admits
        # spellings -- `@`, `+`, `..` -- that a repository component does not.
        validate_repository(repository)
        return Destination(
            registry=self._registry(),
            repository=repository,
            # Per image, not per set: two images in one repository must not push one tag.
            system_tag=compose_system_tag(image.workspace, image.build_set, image.revision, image.index),
        )

    def compile(self, plan: BuildPlan) -> HelixJobSpec:
        return compile_build_set(
            plan,
            config=self._config,
            registry=self._registry(),
            broker=self._broker(),
            sandbox_image=self._sandbox_image(),
        )

    def signature_policy(self) -> DeploymentKeyPolicy:
        """The deployment's key."""
        return DeploymentKeyPolicy(self._public_key())

    def entitlements(self) -> Mapping[str, StepEntitlement]:
        """The push step: pull and push on its job's pending destinations, and signing them.

        ``fetch`` and ``build`` get nothing. ``fetch`` reads Files as the submitter and holds no
        registry credential; ``build`` orchestrates the sandbox, which must reach nothing.
        """
        return {
            PUSH_STEP: StepEntitlement(
                step=PUSH_STEP,
                profile=self._config.push_profile,
                service_account=self._config.push_service_account,
                destinations=("pull", "push"),
                sign=True,
            )
        }
