# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The execution backend's compiler: one :class:`BuildPlan` in, one three-step ``HelixJobSpec`` out."""

from __future__ import annotations

from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.plan import BuildPlan
from nemo_builder_plugin.steps import (
    REGISTRY_PASSWORD_ENV,
    REGISTRY_USERNAME_ENV,
    SIGNING_KEY_ENV,
    ContextSource,
    FetchStepConfig,
    PushImage,
    PushStepConfig,
    SandboxGroup,
    SandboxImage,
    SandboxSpec,
    SuperviseStepConfig,
)
from nemo_helix_plugin.jobs.constants import (
    DEFAULT_JOB_STORAGE_PATH,
    PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
)
from nemo_helix_plugin.jobs.execution_profiles import KubernetesJobExecutionProfileConfig
from nemo_helix_plugin.jobs.providers import ContainerSpec, CPUExecutionProvider
from nemo_helix_plugin.jobs.spec import (
    HelixJobEnvironmentVariable,
    HelixJobSecretEnvironmentVariableRef,
    HelixJobSpec,
    HelixJobStepSpec,
)

#: Where `fetch` and `push` mount the work volume. Jobs mounts it only in a step whose environment
#: sets `PERSISTENT_JOB_STORAGE_PATH_ENVVAR`, at that variable's value.
WORK_MOUNT = DEFAULT_JOB_STORAGE_PATH


def _fetch_sources(plan: BuildPlan) -> list[ContextSource]:
    whole_filesets = {
        image.source.fileset
        for image in plan.images
        if image.source.archive is None and image.source.context_path is None
    }

    sources: list[ContextSource] = []
    for image in plan.images:
        source = image.source
        if source.archive is not None:
            # Unpacked whole, once, whichever directory of it each image builds from.
            source = ContextSource(fileset=source.fileset, archive=source.archive)
        elif source.context_path is not None and source.fileset in whole_filesets:
            continue  # the whole-fileset download covers this subtree
        if source not in sources:
            sources.append(source)
    return sources


def _fetch_step(plan: BuildPlan, config: BuilderConfig) -> HelixJobStepSpec:
    return HelixJobStepSpec(
        name="fetch",
        executor=CPUExecutionProvider(
            # Explicit though it is the default: the Jobs client serializes with `exclude_unset`,
            # which would drop the discriminator, and the server would reject the spec.
            provider="cpu",
            profile=config.fetch_profile,
            container=ContainerSpec(command=["nhx-build", "fetch"]),
        ),
        environment=[HelixJobEnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value=WORK_MOUNT)],
        config=FetchStepConfig(sources=_fetch_sources(plan)).model_dump(),
    )


def _build_step(
    plan: BuildPlan,
    config: BuilderConfig,
    *,
    sandbox_image: str,
    work_profile: KubernetesJobExecutionProfileConfig,
) -> HelixJobStepSpec:
    """Holds no credential and no work volume: it declares no ``environment``, so Jobs mounts none.

    Its sandboxes mount the fetch step's work volume, on that step's nodes.
    """
    groups = [
        SandboxGroup(
            source=source,
            images=[
                SandboxImage(image=image.name, platform=image.spec.platform, dockerfile=image.spec.dockerfile)
                for image in images
            ],
        )
        for source, images in plan.groups()
    ]

    return HelixJobStepSpec(
        name="build",
        executor=CPUExecutionProvider(
            provider="cpu",  # see _fetch_step
            profile=config.control_profile,
            container=ContainerSpec(command=["nhx-build", "supervise"]),
        ),
        config=SuperviseStepConfig(
            sandbox=SandboxSpec(
                image=sandbox_image,
                provider=config.sandbox.provider,
                opensandbox=config.sandbox.opensandbox,
                work_pvc=work_profile.storage.pvc_name,
                node_selector=work_profile.node_selector,
                dns_nameservers=config.sandbox.dns_nameservers,
                egress_allow=config.sandbox.egress_allow,
                cpu=config.sandbox.cpu,
                memory=config.sandbox.memory,
            ),
            groups=groups,
        ).model_dump(),
    )


def _push_step(plan: BuildPlan, config: BuilderConfig, *, registry: str) -> HelixJobStepSpec:
    """The one step with a credential: the workspace's registry credential and signing key, read as the submitter.

    A secret the deployment turned off is left out, and the step told not to expect it.
    """
    images = [
        PushImage(
            image=image.name,
            # Always pushed: the one tag every image has, which no request can name.
            system_ref=image.placed.system_ref,
            caller_ref=image.caller_ref,
        )
        for image in plan.images
    ]

    return HelixJobStepSpec(
        name="push",
        executor=CPUExecutionProvider(
            provider="cpu",  # see _fetch_step
            profile=config.push_profile,
            container=ContainerSpec(command=["nhx-build", "push"]),
        ),
        environment=[
            HelixJobEnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value=WORK_MOUNT),
            # Platform secrets in the job's workspace. Jobs refuses the job if the submitter can't read them.
            *(
                HelixJobEnvironmentVariable(name=name, from_secret=HelixJobSecretEnvironmentVariableRef(name=secret))
                for name, secret in (
                    (REGISTRY_USERNAME_ENV, config.registry_username_secret),
                    (REGISTRY_PASSWORD_ENV, config.registry_password_secret),
                    (SIGNING_KEY_ENV, config.signing_key_secret),
                )
                if secret is not None
            ),
        ],
        config=PushStepConfig(
            images=images,
            registry=registry,
            plain_http=config.registry_plain_http,
            log_in=config.registry_username_secret is not None,
            sign=config.signing_key_secret is not None,
        ).model_dump(),
    )


def compile_build_set(
    plan: BuildPlan,
    *,
    config: BuilderConfig,
    registry: str,
    sandbox_image: str,
    work_profile: KubernetesJobExecutionProfileConfig,
) -> HelixJobSpec:
    """Compile a checked, placed plan into the job that builds it. ``work_profile`` is the fetch step's profile."""
    return HelixJobSpec(
        steps=[
            _fetch_step(plan, config),
            _build_step(plan, config, sandbox_image=sandbox_image, work_profile=work_profile),
            _push_step(plan, config, registry=registry),
        ]
    )
