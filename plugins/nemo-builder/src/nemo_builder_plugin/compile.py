# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The build compiler: one :class:`BuildPlan` in, one ``HelixJobSpec`` out -- three steps, or two.

**A pure function.** No I/O, no writes, no clients. That is not stylistic -- it is what makes
the entire submit path testable without a cluster, a registry, or a database, which in a project
whose end-to-end loop costs minutes is the only fast feedback loop there is. It validates nothing
either: the plan it is handed has already been checked, so compiling is a projection.

**It synthesizes the whole spec, ``executor`` included, and accepts none of it from the
request.** ``HelixJobStepSpec.executor`` is a caller-facing field in Jobs, so a compiler that
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

**It emits no secrets.** The Jobs launcher resolves a job's secrets as the submitting principal,
and the one credential a build needs is the operator's, so nothing here goes through it. ``push``
is told the name of a Kubernetes Secret and reads it as its own ServiceAccount.

**A set of copies compiles to two steps.** A copy is pulled by ``fetch`` and published by
``push``; nothing about it executes, so there is no sandbox to create and no ``build`` step. A
deployment whose only use of this system is importing published images then never exercises the
pod-create grant at all.
"""

from __future__ import annotations

from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.plan import BuildPlan
from nemo_builder_plugin.schema import DEFAULT_DOCKERFILE
from nemo_builder_plugin.steps import (
    ContextSource,
    FetchDockerfile,
    FetchImport,
    FetchStepConfig,
    PushImage,
    PushStepConfig,
    SandboxGroup,
    SandboxImage,
    SandboxSpec,
    SigningConfig,
    SuperviseStepConfig,
)
from nemo_helix_plugin.jobs.constants import (
    DEFAULT_JOB_STORAGE_PATH,
    PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
)
from nemo_helix_plugin.jobs.providers import ContainerSpec, CPUExecutionProvider
from nemo_helix_plugin.jobs.spec import HelixJobEnvironmentVariable, HelixJobSpec, HelixJobStepSpec

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
    contexts = [image.source for image in plan.images if image.source is not None]
    whole_filesets = {source.fileset for source in contexts if source.context_path is None}

    sources: list[ContextSource] = []
    for source in contexts:
        if source.context_path is not None and source.fileset in whole_filesets:
            continue  # the whole-fileset download covers this subtree
        if source not in sources:
            sources.append(source)
    return sources


def _fetch_dockerfiles(plan: BuildPlan) -> list[FetchDockerfile]:
    """The fetched Dockerfiles the runtime layer is appended to: each one once, however many
    specs build from it. Empty without a layer."""
    if plan.runtime_layer is None:
        return []
    dockerfiles: list[FetchDockerfile] = []
    for image in plan.images:
        if image.source is None:
            continue
        entry = FetchDockerfile(source=image.source, dockerfile=image.spec.dockerfile or DEFAULT_DOCKERFILE)
        if entry not in dockerfiles:
            dockerfiles.append(entry)
    return dockerfiles


def _fetch_imports(plan: BuildPlan) -> list[FetchImport]:
    return [
        FetchImport(
            image=image.name,
            ref=image.pinned_upstream,
            platform=image.spec.platform,
            mode="copy" if plan.is_copy(image) else "derive",
        )
        for image in plan.imports
    ]


def _fetch_step(plan: BuildPlan, config: BuilderConfig) -> HelixJobStepSpec:
    """Trusted. Holds a Files client. No registry credential, no pod RBAC, runs no caller code.

    It also pulls imports, anonymously: an import's source is a public, allowlisted registry, and
    the only credential in this job is the push credential, which stays on ``push``.
    """
    return HelixJobStepSpec(
        name="fetch",
        executor=CPUExecutionProvider(
            # `provider` is explicit even though "cpu" is its default. The Jobs client serializes
            # with `exclude_unset`, so a defaulted discriminator is DROPPED from the wire payload
            # and the server then rejects the union with "Unable to extract tag using
            # discriminator 'provider'". `test_the_compiled_spec_survives_exclude_unset` guards it.
            provider="cpu",
            profile=config.fetch_profile,
            container=ContainerSpec(command=["nhx-build", "fetch"]),
        ),
        # Declaring this is what asks for the work volume.
        environment=[HelixJobEnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value=WORK_MOUNT)],
        config=FetchStepConfig(
            sources=_fetch_sources(plan),
            imports=_fetch_imports(plan),
            runtime_layer=plan.runtime_layer,
            dockerfiles=_fetch_dockerfiles(plan),
        ).model_dump(),
    )


def _build_step(plan: BuildPlan, config: BuilderConfig) -> HelixJobStepSpec | None:
    """Trusted control plane for an untrusted pod. Holds NO credential of any kind.

    Note what is absent: no ``environment`` entry requesting the work volume, and no registry,
    tag or secret name anywhere in its config. The sandbox does not push, so the step
    orchestrating it is never told where the trusted step intends to write.

    None when there is nothing to build: a set of copies.
    """
    groups = [
        SandboxGroup(
            source=source,
            images=[
                SandboxImage(
                    image=image.name,
                    platform=image.spec.platform,
                    # A derived import's context is one Dockerfile that `fetch` wrote.
                    dockerfile=image.spec.dockerfile or DEFAULT_DOCKERFILE,
                )
                for image in images
            ],
        )
        for source, images in plan.groups()
    ]
    if not groups:
        return None

    return HelixJobStepSpec(
        name="build",
        executor=CPUExecutionProvider(
            provider="cpu",  # see _fetch_step
            profile=config.control_profile,
            container=ContainerSpec(command=["nhx-build", "supervise"]),
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


def _push_step(plan: BuildPlan, config: BuilderConfig) -> HelixJobStepSpec:
    """Trusted. Holds the registry credential and the signing key. Runs no caller code."""
    images = [
        PushImage(
            image=image.name,
            # The caller's tag AND the system tag. A build that pushed only the first would be
            # invisible to the reconciler, which resolves the second.
            tags=[image.caller_ref, image.system_ref],
            expected_digest=image.upstream_manifest_digest if plan.is_copy(image) else None,
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
        # Declaring this is what asks for the work volume. There is no credential here: `push`
        # reads its own, from the Secret its config names.
        environment=[HelixJobEnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value=WORK_MOUNT)],
        config=PushStepConfig(
            signing=SigningConfig(key=plan.signing_key, storage=config.signature_storage),
            images=images,
            registry=plan.registry,
            credential_secret=plan.push_credential_secret,
            namespace=config.namespace,
            insecure=config.registry_insecure,
        ).model_dump(),
    )


def compile_build_set(plan: BuildPlan, *, config: BuilderConfig) -> HelixJobSpec:
    """Compile a resolved plan into the job that builds it.

    Every import in ``plan`` must already carry its resolved upstream manifest
    (:meth:`BuildPlan.with_upstream_manifests`); an unresolved one raises rather than compiling a
    job that pulls something other than what the row will be checked against.
    """
    build = _build_step(plan, config)
    steps = [_fetch_step(plan, config), *([build] if build else []), _push_step(plan, config)]
    return HelixJobSpec(steps=steps)
