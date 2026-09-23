# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The build compiler: one :class:`BuildPlan` in, one three-step ``PlatformJobSpec`` out.

**A pure function.** No I/O, no writes, no clients. That is not stylistic -- it is what makes
the entire submit path testable without a cluster, a registry, or a database, which in a project
whose end-to-end loop costs minutes is the only fast feedback loop there is. It validates nothing
either: the plan it is handed has already been checked, so compiling is a projection.

**It synthesizes the whole spec, ``executor`` included, and accepts none of it from the
request.** ``PlatformJobStepSpec.executor`` is a caller-facing field in Jobs, so a compiler that
passed one through would undo the property the entire design rests on: that *how* a build runs is
the system's decision. A caller picks a Dockerfile and a fileset.

**The trust split is three strings.** ``build-fetch``, ``build-control`` and ``build-push`` are
execution profile names, and each resolves to a different ServiceAccount in
``plugins/nemo-builder/deploy/10-serviceaccounts.yaml``. That is the entire mechanism -- Jobs
already lets each step name its own profile, so per-step identity needed no schema change at all.

**The work volume is requested by declaring an env var, and one step deliberately does not.**
The Kubernetes backend mounts the shared PVC only for steps whose environment contains
``NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH``, using that variable's *value* as the mount path. ``fetch``
and ``push`` declare it. ``build`` does not -- so the step that orchestrates untrusted code
cannot read a context even by mistake. Absence is the control. The platform applies the per-job
``subPath``, so inside a step pod that mount is already the job's own slice.

**It emits no paths.** Configs name filesets and images; each step finds them through
:class:`~nemo_builder_plugin.steps.WorkLayout`.
"""

from __future__ import annotations

from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.plan import BuildPlan
from nemo_builder_plugin.steps import (
    CREDENTIAL_ENVVAR,
    ContextSource,
    FetchStepConfig,
    PushImage,
    PushStepConfig,
    SandboxGroup,
    SandboxImage,
    SandboxSpec,
    SigningConfig,
    SuperviseStepConfig,
)
from nemo_platform_plugin.jobs.constants import (
    DEFAULT_JOB_STORAGE_PATH,
    PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
)
from nemo_platform_plugin.jobs.providers import ContainerSpec, CPUExecutionProvider
from nemo_platform_plugin.jobs.spec import (
    PlatformJobEnvironmentVariable,
    PlatformJobSecret,
    PlatformJobSecretEnvironmentVariableRef,
    PlatformJobSpec,
    PlatformJobStepSpec,
)

#: Where the work volume is mounted in `fetch` and `push`, and therefore the root of their
#: `WorkLayout`. Setting it is also what asks the Kubernetes backend for the mount.
WORK_MOUNT = DEFAULT_JOB_STORAGE_PATH


def _fetch_sources(plan: BuildPlan) -> list[ContextSource]:
    """What to download, with overlapping requests collapsed.

    Two specs sharing a source cause one download, not two. The case worth writing down is the
    *nested* one: a set that wants both a whole fileset and a subtree of it. ``fetch`` copies a
    fileset with its paths intact and a subtree sits inside it (``WorkLayout.context``), so a
    whole-fileset download already contains every subtree of it. Emitting both would fetch the
    overlap twice, so a whole request absorbs the subtree requests for its fileset.

    Order is first-appearance, which keeps the compiled document stable for a given request.
    """
    whole_filesets = {image.source.fileset for image in plan.images if image.source.context_path is None}

    sources: list[ContextSource] = []
    for image in plan.images:
        source = image.source
        if source.context_path is not None and source.fileset in whole_filesets:
            continue  # the whole-fileset download covers this subtree
        if source not in sources:
            sources.append(source)
    return sources


def _fetch_step(plan: BuildPlan, config: BuilderConfig) -> PlatformJobStepSpec:
    """Trusted. Holds a Files client. No registry credential, no pod RBAC, runs no caller code."""
    return PlatformJobStepSpec(
        name="fetch",
        executor=CPUExecutionProvider(
            # `provider` is explicit even though "cpu" is its default. The Jobs client serializes
            # with `exclude_unset`, so a defaulted discriminator is DROPPED from the wire payload
            # and the server then rejects the union with "Unable to extract tag using
            # discriminator 'provider'". `test_the_compiled_spec_survives_exclude_unset` guards it.
            provider="cpu",
            profile=config.fetch_profile,
            container=ContainerSpec(command=["nmp-build", "fetch"]),
        ),
        # Declaring this is what asks for the work volume.
        environment=[PlatformJobEnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value=WORK_MOUNT)],
        config=FetchStepConfig(sources=_fetch_sources(plan)).model_dump(),
    )


def _build_step(plan: BuildPlan, config: BuilderConfig) -> PlatformJobStepSpec:
    """Trusted control plane for an untrusted pod. Holds NO credential of any kind.

    Note what is absent: no ``environment`` entry requesting the work volume, and no registry,
    tag or secret name anywhere in its config. The sandbox does not push, so the step
    orchestrating it is never told where the trusted step intends to write.
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

    return PlatformJobStepSpec(
        name="build",
        executor=CPUExecutionProvider(
            provider="cpu",  # see _fetch_step
            profile=config.control_profile,
            container=ContainerSpec(command=["nmp-build", "supervise"]),
        ),
        config=SuperviseStepConfig(
            sandbox=SandboxSpec(
                image=config.sandbox_image,
                namespace=config.namespace,
                work_pvc=config.work_pvc,
                runtime_class=config.runtime_class,
                registry_mirror=config.registry_mirror,
                node_selector=config.node_selector,
                dns_nameservers=config.sandbox_dns_nameservers,
                cpu=config.sandbox_cpu,
                memory=config.sandbox_memory,
            ),
            groups=groups,
        ).model_dump(),
    )


def _push_step(plan: BuildPlan, config: BuilderConfig) -> PlatformJobStepSpec:
    """Trusted. Holds the registry credential and the signing key. Runs no caller code."""
    images = [
        PushImage(
            image=image.name,
            push_secret=plan.push_secret,
            # The caller's tag AND the system tag. A build that pushed only the first would be
            # invisible to the reconciler, which resolves the second.
            tags=[image.caller_ref, image.system_ref],
        )
        for image in plan.images
    ]

    return PlatformJobStepSpec(
        name="push",
        executor=CPUExecutionProvider(
            provider="cpu",  # see _fetch_step
            profile=config.push_profile,
            container=ContainerSpec(command=["nmp-build", "push"]),
        ),
        environment=[
            PlatformJobEnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value=WORK_MOUNT),
            # The credential appears exactly once in the whole document, on the last step.
            PlatformJobEnvironmentVariable(
                name=CREDENTIAL_ENVVAR,
                from_secret=PlatformJobSecretEnvironmentVariableRef(name=plan.push_secret),
            ),
        ],
        config=PushStepConfig(
            signing=SigningConfig(key=plan.signing_key, storage=config.signature_storage),
            images=images,
            # The operator's registry, not any destination a spec named. See the field.
            credential_registry=config.default_registry,
            insecure=config.registry_insecure,
        ).model_dump(),
    )


def compile_build_set(plan: BuildPlan, *, config: BuilderConfig) -> PlatformJobSpec:
    """Compile a resolved plan into the job that builds it."""
    return PlatformJobSpec(
        steps=[_fetch_step(plan, config), _build_step(plan, config), _push_step(plan, config)],
        # Declared on the job, consumed by exactly one step.
        secrets=[PlatformJobSecret(name=plan.push_secret)],
    )
