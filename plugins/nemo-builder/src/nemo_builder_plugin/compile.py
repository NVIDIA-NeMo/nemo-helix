# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The build compiler: one ``BuildSet`` in, one three-step ``PlatformJobSpec`` out.

**A pure function.** No I/O, no writes, no clients. That is not stylistic -- it is what makes
the entire submit path testable without a cluster, a registry, or a database, which in a project
whose end-to-end loop costs minutes is the only fast feedback loop there is.

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
cannot read a context even by mistake. Absence is the control.

The per-job ``subPath`` is applied by the platform (``jobs/<workspace>/<job_id>``), so inside a
step pod the mount path is already that job's own slice and nothing here has to know a job id.
``supervise`` reproduces the same layout when it mounts the volume into a sandbox, using
``job_storage_subpath`` rather than spelling it out.
"""

from __future__ import annotations

from dataclasses import dataclass

from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.schema import BuildSet
from nemo_builder_plugin.steps import (
    FetchSource,
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

#: Where the work volume is mounted in `fetch` and `push`. Inside the pod this is already the
#: job's own slice -- the platform applies `jobs/<workspace>/<job_id>` as the subPath.
WORK_MOUNT = DEFAULT_JOB_STORAGE_PATH

#: Mount points inside the sandbox. It sees a context and an output directory and nothing else.
SANDBOX_CONTEXT_MOUNT = "/ctx"
SANDBOX_OUTPUT_MOUNT = "/out"

#: The environment variable the push step reads its registry credential from. The jobs launcher
#: resolves `from_secret` in-pod, as the submitting principal; the value never enters this spec.
PUSH_CREDENTIAL_ENVVAR = "NMP_REGISTRY_AUTH"


class BuildCompileError(ValueError):
    """The request cannot be compiled for this deployment."""


def job_name_for(build_set: BuildSet) -> str:
    """``<set>-<revision>``. Deterministic from the request, which is what lets concurrent
    submitters settle on create-or-get against the unique name index rather than on a
    read-then-write that races across replicas."""
    return f"{build_set.name}-{build_set.revision}"


def image_name_for(job_name: str, index: int) -> str:
    """``<job>-<index>`` -- the ``ContainerImage`` row name.

    The index is the spec's position in ``build_specs``: a tracking handle, not a label. It is
    stable within a revision and means nothing across revisions. Deliberately not derived from
    ``output.repository``, which is arbitrary, caller-owned, and may be longer than a name may be.
    """
    return f"{job_name}-{index}"


def resolve_destinations(build_set: BuildSet, config: BuilderConfig) -> list[tuple[str, str]]:
    """``(registry_host, repository)`` per spec. Raises if a spec has no registry anywhere.

    The two are kept apart deliberately. `registry` is a HOST -- it is what a registry client
    opens a connection to -- and `repository` is a path within it. An earlier version carried
    `<host>/<project>/<repo>` in one string, which reads fine in a log and produces the URL
    `https://host/project/repo/v2/...` the moment anything tries to resolve it.
    """
    destinations: list[tuple[str, str]] = []
    for spec in build_set.build_specs:
        registry = spec.output.registry or config.default_registry
        if not registry:
            raise BuildCompileError(
                f"build spec {spec.name!r} names no registry and this deployment has no "
                "default_registry configured. Set builder.default_registry, or name a registry "
                "on the output."
            )
        repository = spec.output.repository
        # The prefix applies only to the deployment default; a spec that named its own registry
        # names its own full path too.
        if not spec.output.registry and config.repository_prefix:
            repository = f"{config.repository_prefix.strip('/')}/{repository}"
        destinations.append((registry, repository))
    return destinations


@dataclass(frozen=True, slots=True)
class _Resolved:
    """The settings that have no safe default, once proven present.

    Returned rather than merely asserted so the rest of the compiler works with `str`, not
    `str | None`. The alternative is an `assert` in every function that touches one, which is
    the same check written four times and dropped entirely under `python -O`.
    """

    push_secret: str
    signing_key: str


def _require_configured(config: BuilderConfig) -> _Resolved:
    """Reject an unconfigured deployment at COMPILE time, not run time.

    A deployment missing its credential or its signing key should fail the submit with an error
    the caller can act on, rather than produce a job that dies in a pod twenty minutes later
    with a message only an operator can read.
    """
    if not config.execution_enabled:
        raise BuildCompileError(
            "the execution backend is disabled on this deployment (builder.execution_enabled). "
            "Existing images remain readable; no new builds will be accepted."
        )
    if not config.push_secret:
        raise BuildCompileError("builder.push_secret is not configured; nothing could publish the result")
    if not config.signing_key:
        raise BuildCompileError(
            "builder.signing_key is not configured. Signing is required for everything this "
            "system builds, so an unconfigured key fails the compile rather than publishing "
            "unsigned output."
        )
    return _Resolved(push_secret=config.push_secret, signing_key=config.signing_key)


def _fetch_sources(build_set: BuildSet) -> list[FetchSource]:
    """What to download, with overlapping requests collapsed.

    Two specs sharing a source cause one download, not two -- that much is obvious. The case
    worth writing down is the *nested* one: a set that wants both the whole fileset and a subtree
    of it. ``context_path`` selects a subtree, so a root download already contains every subtree
    of the same fileset, and emitting both would fetch the overlap twice and write it twice into
    the same directory. So a root request absorbs the subtree requests for its fileset.

    Order is first-appearance, which keeps the compiled document stable for a given request --
    worth having when the thing you diff to understand a failure is the spec itself.
    """
    whole_filesets = {s.source.fileset for s in build_set.build_specs if s.source.context_path is None}

    sources: list[FetchSource] = []
    seen: set[tuple[str, str | None]] = set()
    for spec in build_set.build_specs:
        fileset, context_path = spec.source.fileset, spec.source.context_path
        if context_path is not None and fileset in whole_filesets:
            continue  # the root download covers this subtree
        key = (fileset, context_path)
        if key in seen:
            continue
        seen.add(key)
        sources.append(FetchSource(fileset=fileset, context_path=context_path))
    return sources


def _fetch_step(build_set: BuildSet, config: BuilderConfig) -> PlatformJobStepSpec:
    """Trusted. Holds a Files client. No registry credential, no pod RBAC, runs no caller code."""
    return PlatformJobStepSpec(
        name="fetch",
        executor=CPUExecutionProvider(
            # `provider` is explicit even though "cpu" is its default. The Jobs client serializes
            # with `exclude_unset`, so a defaulted discriminator is DROPPED from the wire payload
            # and the server then rejects the union with "Unable to extract tag using
            # discriminator 'provider'". Measured, not theorised -- it is what the first real
            # submit returned. `test_the_compiled_spec_survives_exclude_unset` guards it.
            provider="cpu",
            profile=config.fetch_profile,
            container=ContainerSpec(command=["nmp-build", "fetch"]),
        ),
        # Declaring this is what asks for the work volume.
        environment=[PlatformJobEnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value=WORK_MOUNT)],
        config=FetchStepConfig(dest=f"{WORK_MOUNT}/context", sources=_fetch_sources(build_set)).model_dump(),
    )


def _build_step(build_set: BuildSet, config: BuilderConfig, job_name: str) -> PlatformJobStepSpec:
    """Trusted control plane for an untrusted pod. Holds NO credential of any kind.

    Note what is absent: no ``environment`` entry requesting the work volume, and no registry,
    tag or secret name anywhere in its config. The sandbox does not push, so the step
    orchestrating it is never told where the trusted step intends to write.
    """
    groups: list[SandboxGroup] = []
    for (fileset, context_path), indices in build_set.groups():
        context_sub_path = f"context/{fileset}"
        if context_path:
            context_sub_path = f"{context_sub_path}/{context_path}"
        groups.append(
            SandboxGroup(
                context_sub_path=context_sub_path,
                output_sub_path="out",
                images=[
                    SandboxImage(
                        image=image_name_for(job_name, index),
                        platform=build_set.build_specs[index].platform,
                        context=SANDBOX_CONTEXT_MOUNT,
                        dockerfile=build_set.build_specs[index].dockerfile,
                        layout=f"{SANDBOX_OUTPUT_MOUNT}/{image_name_for(job_name, index)}",
                    )
                    for index in indices
                ],
            )
        )

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


def _push_step(
    build_set: BuildSet,
    config: BuilderConfig,
    resolved: _Resolved,
    job_name: str,
    system_tag: str,
    destinations: list[tuple[str, str]],
) -> PlatformJobStepSpec:
    """Trusted. Holds the registry credential and the signing key. Runs no caller code."""

    images = [
        PushImage(
            image=image_name_for(job_name, index),
            layout=f"{WORK_MOUNT}/out/{image_name_for(job_name, index)}",
            push_secret=resolved.push_secret,
            # The caller's tag AND the system tag. A build that pushed only the first would be
            # invisible to the reconciler, which resolves the second.
            tags=[
                f"{destinations[index][0]}/{destinations[index][1]}:{spec.output.tag}",
                f"{destinations[index][0]}/{destinations[index][1]}:{system_tag}",
            ],
        )
        for index, spec in enumerate(build_set.build_specs)
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
                name=PUSH_CREDENTIAL_ENVVAR,
                from_secret=PlatformJobSecretEnvironmentVariableRef(name=resolved.push_secret),
            ),
        ],
        config=PushStepConfig(
            signing=SigningConfig(key=resolved.signing_key, storage=config.signature_storage),
            images=images,
            insecure=config.registry_insecure,
        ).model_dump(),
    )


def compile_build_set(
    build_set: BuildSet,
    *,
    config: BuilderConfig,
    system_tag: str,
) -> PlatformJobSpec:
    """Compile a ``BuildSet`` into the job that builds it.

    ``system_tag`` is passed in rather than composed here: it is generated once at submit and
    handed to both the ``ContainerImage`` rows and this function, because composing it twice is a
    drift bug whose only symptom is a reconciler resolving a tag nothing ever pushed.
    """
    resolved = _require_configured(config)
    destinations = resolve_destinations(build_set, config)
    job_name = job_name_for(build_set)

    return PlatformJobSpec(
        steps=[
            _fetch_step(build_set, config),
            _build_step(build_set, config, job_name),
            _push_step(build_set, config, resolved, job_name, system_tag, destinations),
        ],
        # Declared on the job, consumed by exactly one step.
        secrets=[PlatformJobSecret(name=resolved.push_secret)],
    )
