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
from nemo_builder_plugin.schema import MAX_BUILD_SPECS, ImageSource
from nemo_builder_plugin.steps import REMOTE_CREDENTIAL_ENV, ROOT_DOCKERFILE, RemoteBuildImage, RemoteBuildStepConfig
from nemo_helix_plugin.auth import platform_auth_enabled
from nemo_helix_plugin.auth.workload_identity import (
    WorkloadIdentityConfigError,
    is_workload_identity_token_exchange_enabled,
)
from nemo_helix_plugin.jobs.endpoints import ExecutionProfile
from nemo_helix_plugin.jobs.execution_profiles import KubernetesJobExecutionProfile
from nemo_helix_plugin.jobs.providers import ContainerSpec, CPUExecutionProvider
from nemo_helix_plugin.jobs.spec import (
    HelixJobEnvironmentVariable,
    HelixJobSecretEnvironmentVariableRef,
    HelixJobSpec,
    HelixJobStepSpec,
)

STEP_NAME = "remote-build"
#: The Jobs backends that can run the step, which is a container that reads the job's ID from its environment
#: and, with `credential_secret`, a platform secret. Kubernetes alone for now, as for the in-cluster build: Jobs'
#: Docker backend names a step's container for its job without the workspace, so one build set built in two
#: workspaces would share a container.
STEP_BACKENDS = ("kubernetes_job",)

#: How long past its last build timeout a step needs, to cancel the build and say why.
CANCEL_MARGIN_SECONDS = 120


def is_root_dockerfile(dockerfile: str) -> bool:
    """Whether ``dockerfile`` is the one at the context's root, however it is spelt (``./Dockerfile`` too)."""
    return PurePosixPath(dockerfile).as_posix() == ROOT_DOCKERFILE


class RemoteBackend:
    """:class:`~nemo_builder_plugin.backend.Backend` for builds a service outside this cluster runs."""

    name = "remote"
    #: The builder holds no credential for the service, so a completion's digest, build and runtime
    #: configuration are its caller's word. What stands for them is that the caller is a step of the job the
    #: submit created.
    binds_rows_to_job = True

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
        if profile.backend not in STEP_BACKENDS:
            raise BackendRejectedError(
                f"Jobs execution profile cpu/{name} runs on {profile.backend}, which can't run the remote-build "
                f"step; it needs one of: {', '.join(STEP_BACKENDS)}"
            )
        if remote.image is None and not profile.config.default_task_image:
            # Jobs would fall back to the platform's own tasks image, which holds no build service's client.
            raise BackendRejectedError(
                f"builder.remote.image is unset and Jobs execution profile cpu/{name} names no default_task_image; "
                "there is no image that holds the remote-build step"
            )
        # Jobs ends a step at the profile's limit. It counts from when the job was created, so the time a build
        # waits for a slot is in it too; that part an operator sizes, and this checks only what can't be enough.
        # As many images as one set can hold: what the deployment allows, and no more than any request may name.
        images = min(remote.max_specs_per_set, MAX_BUILD_SPECS)
        needed = remote.build_timeout_seconds * images + CANCEL_MARGIN_SECONDS
        if profile.config.ttl_seconds_active < needed:
            raise BackendRejectedError(
                f"Jobs execution profile cpu/{name} ends a step after {profile.config.ttl_seconds_active}s "
                f"(ttl_seconds_active), which is less than the {needed}s one build set may take "
                f"(builder.remote.build_timeout_seconds times {images}, the images one set may hold "
                f"(builder.remote.max_specs_per_set, at most {MAX_BUILD_SPECS}), plus a {CANCEL_MARGIN_SECONDS}s "
                "margin)"
            )
        if self._config.default_platform not in remote.platforms:
            raise BackendRejectedError(
                f"builder.default_platform is {self._config.default_platform}, which is not one of "
                f"builder.remote.platforms ({', '.join(remote.platforms)}); a request that names no platform "
                "could never build"
            )
        self._check_completion(profile)

    def _check_completion(self, profile: ExecutionProfile) -> None:
        """Refuse every request where the build's completion would be refused: no build is made that could
        not then be recorded.

        With platform auth on and workload token exchange off, the platform takes a request's identity from
        headers that anything able to reach its API can set. Nothing then shows that a completion's caller is
        the build's own step, and no secret the builder could hand that step is out of such a caller's reach.
        With auth off no caller is checked at all, and the backend is not refused.

        Whether the platform uses token exchange is read before the operator's setting is looked at, and a
        platform whose auth configuration can't be read is refused whatever that setting is: the completion
        route reads it first too, admits no one where it can't, and would put the build's step off until
        it gave up. The setting accepts a platform known to verify no one, not one that may be verifying.

        Where a step is verified, the deployment is refused too while its step runs in a namespace that is not
        its own (:meth:`_check_step_namespace`).
        """
        if not platform_auth_enabled():
            return
        try:
            verifiable = is_workload_identity_token_exchange_enabled()
        except WorkloadIdentityConfigError as exc:
            raise BackendRejectedError(
                "the platform's auth configuration can't be read, so nothing says whether the completion of a "
                "remote build could be verified, and none would be let through"
            ) from exc
        if verifiable:
            self._check_step_namespace(profile)
        elif not self._remote.allow_unverified_completion:
            raise BackendRejectedError(
                "platform auth is on without workload token exchange, so nothing shows which job a caller "
                "completing an image is a step of, and a remote build could not be recorded. Turn workload "
                "token exchange on, or set builder.remote.allow_unverified_completion, which accepts that "
                "anything able to reach the platform's API can make a pending image ready at a digest of its "
                "choosing"
            )

    def _check_step_namespace(self, profile: ExecutionProfile) -> None:
        """Refuse every request while the step runs where a step of the in-cluster build may create pods.

        Under workload token exchange, a completion's caller is shown to be the build's step by the platform's
        record of its pod. That record is as sure as the step's namespace is closed: the check holds as far as
        nothing but Jobs creates pods there. The build step of an in-cluster build may create pods in its own
        namespace (`control_profile`), so a remote-build profile in that namespace is refused while Jobs lists
        that profile. What else may create pods where the step runs is the deployment's to see to: the
        builder can't read it.

        A profile that names no namespace runs its steps in the namespace of Jobs' own controller, which Jobs
        takes from that process's environment and the builder can't read. So the two profiles are known to be
        apart only where both name a namespace. Two that name none share one, and one that names none may
        share the namespace the other names: both are refused.
        """
        name = self._config.control_profile
        control = next((p for p in self._profiles if (p.provider, p.profile) == ("cpu", name)), None)
        if not isinstance(control, KubernetesJobExecutionProfile) or not isinstance(
            profile, KubernetesJobExecutionProfile
        ):
            return
        # As Jobs reads them: an empty namespace is one left unset.
        step_namespace = profile.config.namespace or None
        control_namespace = control.config.namespace or None
        if (step_namespace is None) != (control_namespace is None):
            unnamed = self._remote.profile if step_namespace is None else name
            raise BackendRejectedError(
                f"Jobs execution profile cpu/{unnamed} names no namespace, so its steps run in the namespace of "
                "Jobs' own controller, which the builder can't read. Nothing then shows that the remote-build "
                f"step (cpu/{self._remote.profile}) runs apart from the steps of cpu/{name}, which may create "
                "pods where they run. Under workload token exchange the builder relies on nothing but Jobs "
                "creating pods where the remote-build step runs. Name a namespace on both profiles, and give "
                "the remote-build step one of its own, where only Jobs creates pods"
            )
        if control_namespace == step_namespace:
            raise BackendRejectedError(
                f"Jobs execution profile cpu/{self._remote.profile} runs the remote-build step in the namespace of "
                f"cpu/{name}, whose steps may create pods there. Under workload token exchange the builder relies "
                "on nothing but Jobs creating pods where the remote-build step runs. "
                + (
                    # Naming one on the step alone would only be refused again, for the other's sake.
                    "Neither profile names a namespace: name one on both, and give the remote-build step one of "
                    "its own, where only Jobs creates pods"
                    if step_namespace is None
                    else "Run the remote-build step in a namespace of its own, where only Jobs creates pods"
                )
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
                        role=image.spec.role,
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
