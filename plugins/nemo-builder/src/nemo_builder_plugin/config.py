# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator configuration for the builder.

Everything about *how* a build runs lives here rather than in the request. A caller names a
fileset, a Dockerfile and a destination; it does not name the executor image, the namespace, the
service accounts, the egress posture, or whether its image gets signed. That split is
Requirement 2, and putting it in a config class rather than in the request schema is how it is
enforced instead of merely documented.

**Unconfigured means the submit fails, not that the build fails.** A deployment missing its
registry or its signing key should be rejected at compile time with a clear error, not produce a
job that dies in a pod twenty minutes later. The fields that cannot have a safe default are
therefore ``None`` here and checked by the compiler.

Env prefix is ``NEMO_BUILDER_`` -- note that plugin ``NemoConfig`` uses ``NEMO_``, while core
services use ``NMP_``. Some prose in the repo says ``NMP_`` for plugins; the code is the
authority and it computes ``NEMO_<SAFE_NAME>_``.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from nemo_platform_plugin.config import NemoConfig
from pydantic import Field, field_validator


class BuilderConfig(NemoConfig):
    """Configuration for in-cluster container image builds."""

    plugin_name: ClassVar[str] = "builder"
    plugin_description: ClassVar[str] = "In-cluster container image builds for NeMo Platform."

    # --- The kill switch -------------------------------------------------

    execution_enabled: bool = Field(
        default=True,
        description=(
            "When False, POST /builds refuses with a clear error and no job is created. The "
            "REQUEST SURFACE stays up -- reads still work, rows are still readable -- so "
            "disabling builds does not look like an outage to a consumer that is only resolving "
            "digests. An operator turns this off to stop running untrusted code without "
            "rolling back a deployment."
        ),
    )

    # --- Where builds run ------------------------------------------------

    namespace: str = Field(
        default="nmp-builds",
        description="The `baseline`-enforcing namespace. See plugins/nemo-builder/deploy/.",
    )
    work_pvc: str = Field(
        default="nmp-build-work",
        description="Work volume claim. fetch writes it, the sandbox reads and writes it, push reads it.",
    )
    sandbox_image: str = Field(
        default="gcr.io/kaniko-project/executor:debug",
        description=(
            "The kaniko executor. `:debug` rather than `:latest` because it ships a shell, and "
            "the shell is what sequences a group's builds inside one sandbox -- the plain image "
            "has no way to run two invocations.\n\n"
            "RFC 001 wants a pinned first-party fork here (`M2-3`): "
            "upstream kaniko was archived by Google on 2025-06-03, so this is a standing "
            "third-party dependency whose advisory obligation currently has no owner."
        ),
    )
    runtime_class: str | None = Field(
        default=None,
        description=(
            "`gvisor` where the cluster offers a sandboxed runtime. Unset here: gVisor exists on "
            "the lab cluster but its nodes are tainted and in separate pools, which fights the "
            "single-node pinning the ReadWriteOnce work volume requires. Measured to work with "
            "kaniko, so this is a scheduling constraint rather than a capability one."
        ),
    )
    registry_mirror: str | None = Field(
        default=None,
        description=(
            "Pull-through mirror every FROM resolves against. Unset: the mirror is `M2-1` and "
            "not built. While it is unset Requirement 9 is unmet -- base images resolve upstream, "
            "and nothing at the network bounds what a Dockerfile may pull FROM."
        ),
    )
    node_selector: dict[str, str] = Field(
        default_factory=lambda: {"nmp.nvidia.com/build-node": "true"},
        description=(
            "Pins every build pod to one node. Required only because this cluster has no RWX "
            "StorageClass; see deploy/40-work-volume.yaml. Empty it once the work volume is RWX."
        ),
    )
    sandbox_cpu: str = Field(
        default="2",
        description=(
            "CPU request for the sandbox. Sizing is a property of the DEPLOYMENT, not of the "
            "request -- a caller cannot make its build bigger by asking.\n\n"
            "Worth knowing why this is settable at all: `CPUKubernetesJobBackend.schedule` "
            "forwards only a step's `container`, so `executor.resources` reaches nothing and a "
            "Jobs step is sized by its profile. The sandbox escapes that because the build step "
            "writes its pod spec directly -- a small unplanned benefit of this architecture."
        ),
    )
    sandbox_memory: str = Field(default="8Gi", description="Memory request for the sandbox. See `sandbox_cpu`.")
    sandbox_dns_nameservers: list[str] = Field(
        default_factory=lambda: ["8.8.8.8", "1.1.1.1"],
        description=(
            "The sandbox resolves against these, not cluster DNS. See deploy/README.md for the "
            "measurement: cluster DNS here is link-local, and re-allowing it reopens the Pod CIDR."
        ),
    )

    # --- Execution profiles ----------------------------------------------

    fetch_profile: str = Field(default="build-fetch")
    control_profile: str = Field(default="build-control")
    push_profile: str = Field(default="build-push")

    # --- Publishing -------------------------------------------------------

    default_registry: str | None = Field(
        default=None,
        description=(
            "Registry HOST used when a BuildOutput omits one -- `us-central1-docker.pkg.dev`, "
            "not `us-central1-docker.pkg.dev/project/repo`. A host and a repository path are "
            "different things, and conflating them produces a reference that looks right in a "
            "log and builds the URL `https://host/project/repo/v2/...` when anything tries to "
            "resolve it. Validated below rather than trusted.\n\n"
            "No safe default; unset fails the compile."
        ),
    )
    repository_prefix: str = Field(
        default="",
        description=(
            "Path prepended to every `BuildOutput.repository`. This is where a registry's "
            "project/repo path belongs -- for GAR, `<project>/<artifact-repo>`. Kept separate "
            "from `default_registry` so `ContainerImage.registry` stays a host that a registry "
            "client can actually connect to."
        ),
    )

    @field_validator("default_registry")
    @classmethod
    def _registry_is_a_host(cls, value: str | None) -> str | None:
        """A registry is a host, optionally with a port. It is not a path.

        Checked here because the failure is otherwise silent until a reconciler builds a URL,
        and at that point the error reads as a registry problem rather than a config one.
        """
        if value and "/" in value:
            raise ValueError(
                f"default_registry must be a host without a path, got {value!r}. "
                "Put the project/repository path in `repository_prefix` instead."
            )
        return value

    push_secret: str | None = Field(
        default=None,
        description=(
            "Name of the Secrets entry holding the registry credential. Resolved in-pod by the "
            "jobs launcher as the submitting principal -- the value never enters a job spec, "
            "and never enters a pod that ran a Dockerfile."
        ),
    )
    signing_key: str | None = Field(
        default=None,
        description=(
            "cosign key reference. `k8s://<ns>/<secret>` for a static key, `gcpkms://...` for "
            "KMS, where the private half never materialises in a pod. Unset FAILS THE COMPILE "
            "rather than publishing unsigned: Requirement 8 makes signing a MUST on everything "
            "this system builds, and a MUST with a fallback is a default."
        ),
    )
    registry_username: str = Field(
        default="",
        description=(
            "Read-only registry credential for the RECONCILER, which resolves digests and checks "
            "for signatures. Separate from `push_secret` on purpose: this one never writes, so "
            "it should be scoped to pulls. For GAR the pair is `oauth2accesstoken` plus an "
            "access token.\n\n"
            "PoC-only shape. A production deployment resolves this from the Secrets service as "
            "the platform, rather than from operator config."
        ),
    )
    registry_password: str = Field(default="", description="See `registry_username`.")
    registry_insecure: bool = Field(
        default=False,
        description=(
            "Reach the registry over plain HTTP, for an in-cluster dev registry. Off by default "
            "and never a fallback: a registry reached over HTTP cannot establish that the digest "
            "it reports is the digest anyone else would see, which is the whole point of reading "
            "identity from a registry rather than from the build."
        ),
    )
    reconcile_interval_seconds: float = Field(
        default=10.0,
        description="How often the reconciler looks for `pending` rows. It queries rather than watching.",
    )

    signature_storage: Literal["tag", "referrers"] = Field(
        default="tag",
        description=(
            "Pin to the oldest verifier served, not to the registry. Measured: GAR implements "
            "referrers fully AND accepts the legacy `.sig` tag, so the registry is not the "
            "constraint -- the consumer is. A v2 verifier reading a v3 referrers-stored "
            "signature reports 'no signatures found', i.e. UNSIGNED, not 'format mismatch'."
        ),
    )
