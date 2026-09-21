# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The step config contracts -- what the compiler puts in each step's ``config``.

This module is the interface freeze. The compiler writes these; the three step binaries read
them. Freezing it here means the step binaries can be written without waiting for the submit
path, which is the single highest-leverage thing in the plan: those two workstreams share a data
structure and nothing else.

**How a step actually receives this.** The jobs controller serialises ``PlatformJobStepSpec.config``
to a ConfigMap, mounts it, and points ``NEMO_JOB_STEP_CONFIG_FILE_PATH`` at the file. In the pod,
``nmp.common.jobs.config.get_task_config()`` reads that path and returns the dict. So a step does
``FetchStepConfig.model_validate(get_task_config())`` and nothing else.

**Read these by what each one does not carry.**

- ``fetch`` is told where to put things. It is not told any registry, tag, or credential.
- ``supervise`` is told how to build. It is **not told where anything will be pushed** -- no
  registry, no tags, no secret name. The sandbox does not push, so the step orchestrating it has
  no reason to know where the trusted step intends to write, and not telling it means a
  compromise of it learns nothing about the destination.
- ``push`` is told where to publish and how to sign. It is not told anything about a fileset, a
  context, or a Dockerfile -- it publishes bytes it did not produce and cannot reproduce.

No step config carries a credential *value*. ``push_secret`` is a name that the jobs launcher
resolves in-pod against the Secrets service, as the submitting principal. The value never enters
this structure, the Job object, or etcd.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. fetch
# ---------------------------------------------------------------------------


class FetchSource(BaseModel):
    """One (fileset, context_path) pair to download. Deduplicated by the compiler."""

    fileset: str
    context_path: str | None = None


class FetchStepConfig(BaseModel):
    """Trusted. Holds a Files client. Holds no registry credential. Runs no caller code."""

    dest: str = Field(description="Directory on the work volume to write contexts under.")
    sources: list[FetchSource] = Field(
        min_length=1,
        description=(
            "Deduplicated across the set: two specs sharing a fileset cause one download, not "
            "two. Each lands at `<dest>/<fileset>/`."
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
            "PoC-only in practice: the work volume is ReadWriteOnce on this cluster, so every "
            "build pod must land on one node. Production wants RWX and an empty selector."
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
    context: str = Field(description="Absolute path to the build context inside the sandbox.")
    dockerfile: str = Field(description="Dockerfile path, relative to `context`.")
    layout: str = Field(description="Absolute path to write the OCI layout to.")


class SandboxGroup(BaseModel):
    """One sandbox pod. All images here share one source, so they can share one context mount.

    Both paths are **relative to the job's own storage slice**, not to the PVC root. The compiler
    cannot write an absolute subPath because it does not know the job id -- the job does not
    exist yet when it runs. ``supervise`` prefixes ``job_storage_subpath(workspace, job_id)``
    from its own environment, which is the platform's own helper rather than a string this
    design spells out twice.
    """

    context_sub_path: str = Field(description="This group's context, mounted read-only. Relative to the job slice.")
    output_sub_path: str = Field(description="Where OCI layouts go, mounted read-write. Relative to the job slice.")
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
    """One OCI layout to publish."""

    image: str = Field(description="`ContainerImage.name` -- the row this will satisfy.")
    layout: str = Field(description="Absolute path to the OCI layout on the work volume.")
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
