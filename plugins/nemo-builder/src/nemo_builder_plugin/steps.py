# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The step config contracts -- what the compiler puts in each step's ``config``.

This module is the interface freeze. The compiler writes these; the three step binaries read
them. Freezing it here means the step binaries can be written without waiting for the submit
path, which is the single highest-leverage thing in the plan: those two workstreams share a data
structure and nothing else.

**How a step actually receives this.** The jobs controller serialises ``HelixJobStepSpec.config``
to a ConfigMap, mounts it, and points ``NEMO_JOB_STEP_CONFIG_FILE_PATH`` at the file. In the pod,
``nhx.common.jobs.config.get_task_config()`` reads that path and returns the dict. So a step does
``FetchStepConfig.model_validate(get_task_config())`` and nothing else.

**Read these by what each one does not carry.**

- ``fetch`` is told what to download. It is not told any registry, tag, or credential.
- ``supervise`` is told how to build. It is **not told where anything will be pushed** -- no
  registry, no tags, no secret name. The sandbox does not push, so the step orchestrating it has
  no reason to know where the trusted step intends to write, and not telling it means a
  compromise of it learns nothing about the destination.
- ``push`` is told where to publish and how to sign. It is not told anything about a fileset, a
  context, or a Dockerfile -- it publishes bytes it did not produce and cannot reproduce.

**No config carries a path.** Configs name things -- a fileset, an image -- and every step
derives where those things live from one :class:`WorkLayout`.

No step config carries a credential *value*. ``push_secret`` is a name that the jobs launcher
resolves in-pod against the Secrets service, as the submitting principal. The value never enters
this structure, the Job object, or etcd.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field

#: The environment variable `push` reads the registry credential from. The compiler wires it with
#: `from_secret`, so the jobs launcher resolves the value in-pod and it never enters a job spec.
CREDENTIAL_ENVVAR = "NHX_REGISTRY_AUTH"


class ContextSource(BaseModel):
    """One build context: a whole fileset, or a subtree of one."""

    fileset: str
    context_path: str | None = None


# ---------------------------------------------------------------------------
# The work volume
# ---------------------------------------------------------------------------


def _relative(part: str) -> PurePosixPath:
    """A caller-supplied path component, refused if it could leave the directory it is joined to.

    ``PurePosixPath("a") / "/etc"`` is ``/etc``: an absolute component does not nest, it
    replaces. So a fileset name or ``context_path`` that is absolute, or climbs with ``..``,
    would point a step outside this job's directory.
    """
    path = PurePosixPath(part)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{part!r} is not a relative path inside the work volume")
    return path


@dataclass(frozen=True, slots=True)
class WorkLayout:
    """Where everything lives in one job's slice of the work volume. The only definition of it.

    Rooted wherever the caller sees that slice:

    - ``fetch`` and ``push`` mount the slice, and root it at their mount path.
    - ``supervise`` mounts nothing, but writes the sandbox's ``subPath``\\ s, so it roots it at
      the slice's path *within* the volume.
    - The sandbox sees the same layout again under its own root, with only two parts of it
      mounted: its own context, read-only, and the output directory.

    So what ``fetch`` writes, what the sandbox mounts and builds, and what ``push`` reads are one
    function of the same names, evaluated under different roots. They cannot disagree.
    """

    root: PurePosixPath

    def fileset(self, name: str) -> PurePosixPath:
        """Where ``fetch`` copies a fileset, with its entry paths unchanged."""
        return self.root / "context" / _relative(name)

    def context(self, source: ContextSource) -> PurePosixPath:
        """A build context: its fileset's directory, or a subtree inside it.

        Inside, not beside: that is what lets one whole-fileset download serve every subtree of
        the same fileset.
        """
        path = self.fileset(source.fileset)
        return path / _relative(source.context_path) if source.context_path else path

    def context_hash_file(self, source: ContextSource) -> PurePosixPath:
        """Beside the context rather than in it, where it would become part of the build."""
        context = self.context(source)
        return context.with_name(f"{context.name}.nhx-context-hash")

    @property
    def outputs(self) -> PurePosixPath:
        """Where the sandbox writes OCI layouts and ``push`` reads them."""
        return self.root / "out"

    def output(self, image: str) -> PurePosixPath:
        """One image's OCI layout. ``image`` is the ``ContainerImage`` row name."""
        return self.outputs / _relative(image)


# ---------------------------------------------------------------------------
# 1. fetch
# ---------------------------------------------------------------------------


class FetchStepConfig(BaseModel):
    """Trusted. Holds a Files client. Holds no registry credential. Runs no caller code."""

    sources: list[ContextSource] = Field(
        min_length=1,
        description=(
            "Deduplicated across the set: two specs sharing a source cause one download, and a "
            "whole fileset absorbs requests for its own subtrees."
        ),
    )


# ---------------------------------------------------------------------------
# 2. supervise  (and the sandbox it creates, which is not a step)
# ---------------------------------------------------------------------------


class SandboxSpec(BaseModel):
    """How to compose the untrusted pod.

    Every field here comes from operator config and **none of it from the request**. A caller
    picks a Dockerfile and a fileset; it does not pick egress, the executor image, the runtime
    class, or the service account. That is Requirement 2 expressed as a type.
    """

    image: str = Field(description="The kaniko executor image.")
    namespace: str
    work_pvc: str = Field(description="Claim the sandbox mounts for context and output.")
    service_account: str = Field(
        default="",
        description="Empty, always. The sandbox runs with automountServiceAccountToken: false.",
    )
    runtime_class: str | None = Field(
        default=None,
        description=(
            "`gvisor` on a cluster that offers it, unset where none exists. Sandboxed isolation "
            "is a cluster fact, so the operator states it rather than the design assuming it."
        ),
    )
    registry_mirror: str | None = Field(
        default=None,
        description=(
            "Pull-through mirror for every FROM. Unset in the PoC -- the mirror is `M2-1` and "
            "not built. Wired here so the flag path exists and the day it lands is config, not "
            "code. While it is unset, base images resolve upstream and Requirement 9 is unmet."
        ),
    )
    node_selector: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "PoC-only in practice: while the work volume is ReadWriteOnce, every build pod must "
            "land on one node. Production wants RWX and an empty selector."
        ),
    )
    dns_nameservers: list[str] = Field(
        default_factory=lambda: ["8.8.8.8", "1.1.1.1"],
        description=(
            "The sandbox uses `dnsPolicy: None` with these, NOT cluster DNS. Measured reason: "
            "GKE's NodeLocal DNSCache answers on a LINK-LOCAL address, so the egress denial that "
            "closes the metadata server closes cluster DNS with it -- and allowing kube-dns back "
            "reopens the Pod CIDR, because that allow is by construction an exception to the "
            "policy's `except` list. A sandbox needs to resolve pypi.org, not "
            "kubernetes.default.svc, so pointing it at public resolvers is strictly more closed."
        ),
    )
    cpu: str = "2"
    memory: str = "8Gi"


class SandboxImage(BaseModel):
    """One kaniko invocation inside a group.

    No tags and no registry: see the module docstring.
    """

    image: str = Field(description="`ContainerImage.name` -- the row this invocation is for.")
    platform: str
    dockerfile: str = Field(description="Dockerfile path, relative to the group's context.")


class SandboxGroup(BaseModel):
    """One sandbox pod. All images here share one source, so they can share one context mount.

    Where that context and the outputs are mounted is ``supervise``'s to decide, from the
    :class:`WorkLayout`: it owns the pod. The compiler could not write the ``subPath``\\ s anyway
    -- they include the job id, and the job does not exist yet when the compiler runs.
    """

    source: ContextSource
    images: list[SandboxImage] = Field(min_length=1)


class SuperviseStepConfig(BaseModel):
    """Trusted control plane for an untrusted pod. Holds NO credential of any kind.

    It does not mount the work volume, so it cannot read a context even by mistake.

    Images inside a group are built **sequentially**, one kaniko invocation each with
    ``--cleanup``. Not a throughput choice: they share one container root and measurably cannot
    overlap. Parallelism across a set comes from multiple groups, which are separate pods.
    """

    sandbox: SandboxSpec
    groups: list[SandboxGroup] = Field(min_length=1)


# ---------------------------------------------------------------------------
# 3. push
# ---------------------------------------------------------------------------


class SigningConfig(BaseModel):
    """Operator config. A caller cannot choose whether its image is signed, or with what."""

    key: str = Field(description="cosign key reference, e.g. `k8s://ns/secret` or `gcpkms://...`.")
    storage: Literal["tag", "referrers"] = Field(
        default="tag",
        description="Pinned to the OLDEST verifier we serve, not to what the registry supports.",
    )


class PushImage(BaseModel):
    """One OCI layout to publish. Its path is :meth:`WorkLayout.output` of ``image``."""

    image: str = Field(description="`ContainerImage.name` -- the row this will satisfy.")
    push_secret: str = Field(description="Secret NAME. Resolved in-pod; the value is never here.")
    tags: list[str] = Field(
        min_length=2,
        description=(
            "The caller's tag and the system tag, in that order. Both, always: the caller's tag "
            "is what a human uses and the system tag is what the reconciler resolves, and a "
            "build that pushed only the first would be invisible to the control plane."
        ),
    )


class PushStepConfig(BaseModel):
    """Trusted. Holds the registry credential and the signing key. Runs no caller code.

    It publishes bytes it did not produce. That is the point, and it is also why this step must
    treat the OCI layout as hostile input: it is the only thing standing between a sandbox's
    output and the registry credential.
    """

    signing: SigningConfig
    images: list[PushImage] = Field(min_length=1)
    credential_registry: str | None = Field(
        default=None,
        description=(
            "The one registry host a bare `user:password` credential is presented to: the "
            "deployment's `default_registry`. Never derived from the images' destinations, "
            "because a spec may name its own registry, and binding the credential to whatever "
            "host a caller names would hand it to that host. A dockerconfigjson credential "
            "ignores this -- it already says which hosts it is for."
        ),
    )
    insecure: bool = Field(
        default=False,
        description=(
            "Push over plain HTTP. Dev registries only -- it means the bytes and the credential "
            "cross the network unprotected, and the digest the registry reports is unauthenticated."
        ),
    )
