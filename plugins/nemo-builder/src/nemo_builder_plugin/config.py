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
services use ``NHX_``. Some prose in the repo says ``NHX_`` for plugins; the code is the
authority and it computes ``NEMO_<SAFE_NAME>_``.
"""

from __future__ import annotations

import re
from typing import ClassVar, Literal

from nemo_builder_plugin.identity import ImageIdentityError, validate_registry_host, validate_repository
from nemo_helix_plugin.config import NemoConfig
from pydantic import BaseModel, Field, field_validator

#: A runtime name, as `BuildSet.runtime` accepts it.
_RUNTIME_NAME = re.compile(r"[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?")
#: A Dockerfile instruction that starts a new stage. See `RuntimeLayer`.
_FROM_LINE = re.compile(r"^\s*FROM\s", re.IGNORECASE | re.MULTILINE)


class RuntimeLayer(BaseModel):
    """One runtime's contract, as Dockerfile text appended to every image built for it.

    A runtime -- the evaluator's Harbor sandbox, a deployed agent -- runs images under rules an
    image built for Docker Desktop does not meet: a non-root user, a read-only root filesystem,
    tools its harness expects. This closes the gap in the build, once per runtime, instead of in
    every Dockerfile. It belongs to a runtime rather than a cluster: one cluster can host several,
    and Astra already does.
    """

    version: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$",
        description="Bumped whenever `dockerfile` changes. Recorded on every row as `<name>@<version>`.",
    )
    dockerfile: str = Field(
        min_length=1,
        description=(
            "Instructions appended after the image's own. No `FROM`: a new stage would discard "
            "the image it is meant to adapt, so the result would not be derived from it at all.\n\n"
            "Prefer pinned versions, or tools copied from a pinned image, over installing from the "
            "network: an unpinned install makes a derived image a function of the day it was built."
        ),
    )

    @field_validator("dockerfile")
    @classmethod
    def _no_new_stage(cls, value: str) -> str:
        if _FROM_LINE.search(value):
            raise ValueError("a runtime layer must not contain FROM; it would start a new stage")
        return value


class BuilderConfig(NemoConfig):
    """Configuration for in-cluster container image builds."""

    plugin_name: ClassVar[str] = "builder"
    plugin_description: ClassVar[str] = "In-cluster container image builds for NeMo Helix."

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
        default="nhx-builds",
        description="The `baseline`-enforcing namespace. See plugins/nemo-builder/deploy/.",
    )
    work_pvc: str = Field(
        default="nhx-build-work",
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
            "`gvisor` where the cluster offers a sandboxed runtime. Unset by default: where gVisor "
            "nodes are tainted and in separate pools, as on GKE, that fights the single-node "
            "pinning the ReadWriteOnce work volume requires. Measured to work with kaniko, so "
            "this is a scheduling constraint rather than a capability one."
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
        default_factory=lambda: {"nhx.nvidia.com/build-node": "true"},
        description=(
            "Pins every build pod to one node. Required only while the work volume is "
            "ReadWriteOnce; see deploy/40-work-volume.yaml. Empty it once the work volume is RWX."
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
            "The sandbox resolves against these, not cluster DNS. See deploy/30-networkpolicy.yaml: "
            "where cluster DNS is link-local, re-allowing it reopens the Pod CIDR."
        ),
    )

    # --- Execution profiles ----------------------------------------------

    fetch_profile: str = Field(default="build-fetch")
    control_profile: str = Field(default="build-control")
    push_profile: str = Field(default="build-push")

    # --- Publishing -------------------------------------------------------

    registry: str | None = Field(
        default=None,
        description=(
            "The registry HOST every image is published to -- `us-central1-docker.pkg.dev`, not "
            "`us-central1-docker.pkg.dev/project/repo`. One per deployment, and a caller cannot "
            "name another: whatever runs a built image has to pull it, and one registry means one "
            "pull credential for every workload that does.\n\n"
            "A host and a repository path are different things, and conflating them produces a "
            "reference that looks right in a log and builds the URL "
            "`https://host/project/repo/v2/...` when anything tries to resolve it. Validated below "
            "rather than trusted.\n\n"
            "No safe default; unset fails the submit."
        ),
    )
    repository_prefix: str = Field(
        default="",
        description=(
            "Path every image is published under, ahead of the submitting workspace: images land "
            "at `<repository_prefix>/<workspace>/<output.repository>`. This is where a "
            "registry's project/repo path belongs -- for GAR, `<project>/<artifact-repo>`. Kept "
            "separate from `registry` so `ContainerImage.registry` stays a host that a registry "
            "client can actually connect to."
        ),
    )

    @field_validator("registry")
    @classmethod
    def _registry_is_a_host(cls, value: str | None) -> str | None:
        """A registry is a host, optionally with a port. It is not a path.

        Checked here because the failure is otherwise silent until a reconciler builds a URL,
        and at that point the error reads as a registry problem rather than a config one.
        """
        if value is None:
            return None
        if "/" in value:
            raise ValueError(
                f"registry must be a host without a path, got {value!r}. "
                "Put the project/repository path in `repository_prefix` instead."
            )
        try:
            # Lowercased here because every reference compared against it is normalized, and a
            # host is case-insensitive: `Registry.example.com` would otherwise never match.
            return validate_registry_host(value)
        except ImageIdentityError as exc:
            raise ValueError(f"registry must be a host, optionally with a port, got {value!r}") from exc

    @field_validator("repository_prefix")
    @classmethod
    def _prefix_is_a_repository_path(cls, value: str) -> str:
        """Normalized once here, so the composed path has exactly one `/` between its parts."""
        value = value.strip("/")
        return validate_repository(value) if value else value

    push_credential_secret: str | None = Field(
        default=None,
        description=(
            "Name of the Kubernetes Secret, in `namespace`, holding the credential `push` "
            "publishes with: a `kubernetes.io/dockerconfigjson` Secret, as `kubectl create secret "
            "docker-registry` makes. It needs write access to `registry` under "
            "`repository_prefix`, and nothing more.\n\n"
            "The operator's credential, delivered the way the signing key is: `push` reads it "
            "through the Kubernetes API as `nhx-build-push`, the only ServiceAccount granted `get` "
            "on it (deploy/20-rbac.yaml). Deliberately NOT a Secrets-service entry. The Jobs "
            "launcher resolves those as the submitting principal, so an operator credential "
            "delivered that way would have to be readable by everyone who can submit a build.\n\n"
            "No safe default; unset fails the submit."
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
    # --- Imports and runtimes -------------------------------------------

    import_registries: list[str] = Field(
        default_factory=list,
        description=(
            "Registry hosts an `ImageSource` may name, e.g. `docker.io`. Checked at submit, before "
            "any row exists, against the NORMALIZED reference -- so `alpine@sha256:...` is "
            "`docker.io`. Empty, the default, refuses every import.\n\n"
            "This is what bounds what the signing key covers on the import path: every import is "
            "published and signed, so the list is the operator's statement of whose images it is "
            "willing to sign for."
        ),
    )
    runtime_layers: dict[str, RuntimeLayer] = Field(
        default_factory=dict,
        description=(
            "Runtime name -> the layer appended to every image built for it. A `BuildSet` names "
            "one in `runtime`, or none. With none, an import is a byte-for-byte copy; with one, it "
            "is `FROM <upstream>@<digest>` plus the layer. Supply as JSON through "
            "`NEMO_BUILDER_RUNTIME_LAYERS`, or in the config file."
        ),
    )

    @field_validator("import_registries")
    @classmethod
    def _import_registries_are_hosts(cls, value: list[str]) -> list[str]:
        """Each a host on the reference grammar, lowercased, and Docker Hub's aliases folded in.

        Normalized the way references are, because they are compared with normalized references:
        an entry that never matches anything -- `" docker.io"`, `Index.Docker.io` -- would refuse
        imports it was written to allow, with nothing saying why.
        """
        hosts: list[str] = []
        for host in value:
            try:
                normalized = validate_registry_host(host)
            except ImageIdentityError as exc:
                raise ValueError(f"import_registries entries must be registry hosts, got {host!r}") from exc
            hosts.append("docker.io" if normalized in ("index.docker.io", "registry-1.docker.io") else normalized)
        return hosts

    @field_validator("runtime_layers")
    @classmethod
    def _runtime_names(cls, value: dict[str, RuntimeLayer]) -> dict[str, RuntimeLayer]:
        for name in value:
            if not _RUNTIME_NAME.fullmatch(name):
                raise ValueError(f"invalid runtime name {name!r}; use lowercase letters, digits and '-'")
        return value

    registry_username: str = Field(
        default="",
        description=(
            "Read-only registry credential for the RECONCILER, which resolves digests and checks "
            "for signatures. Separate from `push_credential_secret` on purpose: this one never "
            "writes, so it should be scoped to pulls. For GAR the pair is `oauth2accesstoken` "
            "plus an access token.\n\n"
            "Supply both from a Kubernetes Secret as `NEMO_BUILDER_REGISTRY_USERNAME` and "
            "`NEMO_BUILDER_REGISTRY_PASSWORD` -- environment variables override the config file -- "
            "rather than in the file itself, which is a ConfigMap. See deploy/local/platform.yaml."
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
        gt=0,
        description="How often the reconciler looks for `pending` rows. It queries rather than watching.",
    )

    signature_storage: Literal["tag", "referrers"] = Field(
        default="tag",
        description=(
            "Pin to the oldest verifier served, not to the registry. Measured: GAR implements "
            "referrers fully AND accepts the legacy `.sig` tag, so the registry is not the "
            "constraint -- the consumer is. A v2 verifier reading a v3 referrers-stored "
            "signature reports 'no signatures found', i.e. UNSIGNED, not 'format mismatch'.\n\n"
            "Only `tag` is implemented: `push` runs cosign v2, which writes the legacy tag, and the "
            "reconciler looks there. `referrers` is refused rather than accepted and ignored -- "
            "recorded on every row while the signature sat somewhere else, it would make each row "
            "misreport where its signature is."
        ),
    )

    @field_validator("signature_storage")
    @classmethod
    def _only_tag_storage_is_implemented(cls, value: str) -> str:
        if value != "tag":
            raise ValueError(
                "signature_storage 'referrers' is not implemented: push writes cosign's legacy "
                "`.sig` tag and the reconciler looks only there. Use 'tag'."
            )
        return value
