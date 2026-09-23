# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``ContainerImage`` -- the only entity in this design.

Created **at submit, in ``pending``, before anything is built.** That inversion is the whole
simplification: there is no orphan to detect. An earlier design wrote the row *after* the push,
so a pod dying in between left a pushed image with nothing pointing at it and needed a sweep over
terminal jobs to find it. Creating the row first means the pointer always exists and the only
question is what state it is in.

**Exactly one writer of the observed half.** The submit route writes the desired fields --
``registry``, ``repository``, ``provenance``, ``platform`` -- and never touches them again. The
reconciler writes ``status``, ``status_detail`` and everything under the *Observed* marker, and
is the only thing that does. **The build pod never touches this entity at all**: it pushes and
exits, so nothing that ran a caller's Dockerfile can write control-plane identity.

**Write-once on ``ready``.** ``digest``, ``manifest_digest`` and ``tag`` are set exactly once, at
the ``pending -> ready`` transition. A rebuild creates a *new* row rather than mutating one a
consumer may have pinned. ``ready`` is terminal; so is ``failed``.

No migration is required to add this type. The entity store is one generic table keyed on
``entity_type`` with the payload in a JSON column, so a new ``NemoEntity`` subclass is storable
and filterable the moment it is declared. There is also no secondary-index mechanism, which
settles a question RFC 001's plan left open: the digest index it wanted is not merely expensive,
it has nowhere to live yet.
"""

from __future__ import annotations

from typing import Annotated, Literal

from nemo_builder_plugin.identity import DIGEST_PATTERN
from nemo_helix_plugin.entity import NemoEntity
from nemo_helix_plugin.refs import ENTITY_REF_PATTERN
from pydantic import BaseModel, Field


class RegisteredOrigin(BaseModel):
    """Registered through ``POST /container-images``. Nothing here built it.

    Not reachable in the PoC -- the registration route is `M1-10` and out of scope -- but the arm
    exists so the discriminator is stable and the reconciler can already say "never reconcile
    this", which is the behaviour that makes the route safe to add later.
    """

    type: Literal["registered"] = "registered"
    image_ref: str = Field(
        description=(
            "The reference the caller supplied, recorded AS SUPPLIED. Redundant against `digest` "
            "on a good day and the only useful field on a bad one: when the two disagree later, "
            "this says what was asked for and the digest says what was got, so a tag that has "
            "since moved is visible as the difference."
        )
    )


class JobOrigin(BaseModel):
    """A build job produces it."""

    type: Literal["job"] = "job"
    build_set: str = Field(
        description=(
            "The `BuildSet.name` this row came from. Stored rather than parsed back out of "
            "`ContainerImage.name`, which is `<set>-<revision>-<index>` and whose last two "
            "components are both numeric."
        )
    )
    revision: int = Field(
        ge=1,
        description=(
            "The caller's ordinal for `build_set`. The only place a resubmission precondition "
            "can read that outlives the job: a HelixJob disappears on explicit delete and "
            "goes invisible to the non-terminal query the moment it completes, so a precondition "
            "read against jobs is blind in two ways that a read against this row is not."
        ),
    )
    job: str = Field(
        pattern=ENTITY_REF_PATTERN,
        description="`workspace/name` of the build job, on the platform's own reference pattern.",
    )
    system_tag: str = Field(
        description=(
            "The tag this build pushes alongside the caller's, and the one the reconciler "
            "resolves. It sits on this arm because it only exists on this arm -- a registered "
            "image pushes nothing -- and a type says that better than a nullable field plus a "
            "sentence about when it is null.\n\n"
            "Stored rather than re-derived. The redundancy is deliberate: this records the tag "
            "this build ACTUALLY PUSHED, so a later change to how the tag is composed leaves "
            "already-written rows resolving correctly instead of re-deriving a string nobody "
            "pushed."
        )
    )


Origin = Annotated[RegisteredOrigin | JobOrigin, Field(discriminator="type")]


class Provenance(BaseModel):
    """Where an image came from. Written by one writer at one moment, and never touched again."""

    backend: Literal["execution", "external", "managed", "registered"] = Field(
        description=(
            "Which backend satisfied this, or `registered` for a row no backend produced. Not "
            "recoverable from `built_by`, which only distinguishes job from registration -- "
            "`execution`, `external` and `managed` are all jobs.\n\n"
            "It is NOT a quality ordering. `managed` breaks that in both directions at once: it "
            "emits SLSA provenance the execution mode cannot, and runs caller-authored RUN under "
            "a cloud identity the execution mode denies it. A consumer reads this to learn WHICH "
            "guarantees hold, never how many."
        )
    )
    built_by: Origin


class Signature(BaseModel):
    """What the reconciler OBSERVED in the registry -- not what the push step intended.

    Written once at the ``pending -> ready`` transition, from a registry read, on the same
    footing as ``digest``. A row that says it is signed means someone can go and find the
    signature *there*.

    **It does not mean anyone checked who made it.** v1 is a presence check. The reconciler is
    not the thing standing between a bad signature and a running pod -- the cluster's own
    verifier is, at admission, against its own trust root. What this adds is a pipeline
    correctness check that catches signing having silently not happened, and catching that needs
    no public key.
    """

    storage: Literal["tag", "referrers"] = Field(
        description=(
            "Where the artifact was found. A cosign signature lives either at the legacy "
            "`sha256-<hex>.sig` tag or as an OCI 1.1 referrer, and cosign v3 defaults to "
            "referrers. A verifier that reads only the legacy tag then finds nothing and reports "
            "'unsigned image' rather than 'format mismatch' -- so the storage mode is pinned to "
            "the oldest verifier we serve, not to what the registry supports."
        )
    )
    verified_against: str | None = Field(
        default=None,
        description=(
            "ALWAYS None in v1, and the field exists to say so. Populating it would claim a "
            "trusted key was checked, which a presence check does not do -- anything able to "
            "write to the repository can place a well-formed signature made with any key. The "
            "field is here now so that the day verification lands is a population rather than a "
            "migration."
        ),
    )


class ContainerImage(NemoEntity, entity_type="container_image"):
    """One image: desired state at submit, observed state after reconcile.

    ``name`` is ``<job-name>-<index>``, where the index is the spec's position in
    ``build_specs``. That makes the name deterministic from the request -- which is what lets
    concurrent submitters settle on create-or-get against the unique name index rather than a
    read-then-write that races across replicas. The index is positional: stable within a
    revision, meaningless across revisions.

    Not consulted at pull time. A workload names an image as a string and the kubelet pulls it.
    This row is what a dispatcher resolves *before* it writes that string, and what an auditor
    reads afterwards.
    """

    # --- Desired. Written once by the submit route; never touched again. ---

    status: Literal["pending", "ready", "failed"] = "pending"
    status_detail: str | None = None

    registry: str = Field(description="Registry host the image lives on.")
    repository: str = Field(description="Repository path within the registry.")
    provenance: Provenance
    platform: str = Field(
        default="linux/amd64",
        pattern=r"^[a-z0-9]+/[a-z0-9]+(/[a-z0-9]+)?$",
    )

    # --- Observed. Absent until `status == "ready"`, then immutable. ---

    digest: str | None = Field(
        default=None,
        pattern=DIGEST_PATTERN,
        description="What the reference resolved to, read from the registry. Never reported by a build.",
    )
    manifest_digest: str | None = Field(
        default=None,
        pattern=DIGEST_PATTERN,
        description=(
            "The per-platform manifest inside an index. Equal to `digest` when the reference "
            "resolves to a plain manifest, which is the usual case under this execution mode."
        ),
    )
    tag: str | None = Field(
        default=None,
        description="The tag the reconciler actually resolved to reach `digest`.",
    )
    source_digest: str | None = Field(
        default=None,
        description=(
            "Hash of the context the fetch step downloaded, carried out of the image config "
            "blob as a label. ADVISORY, never identity: the build plane reports it, which is "
            "why it can never be the thing a consumer pins."
        ),
    )
    signature: Signature | None = Field(
        default=None,
        description=(
            "None on a registered row, always -- signing an image this system did not build is "
            "forbidden. A consumer on a verification-enforcing cluster reads that None to know "
            "the row will be refused at admission before it schedules anything."
        ),
    )

    @property
    def image_ref(self) -> str | None:
        """``<registry>/<repository>@<digest>`` -- a projection, not a stored field.

        Stored, it would be a second place the digest lives, and two places disagree eventually.
        """
        if self.digest is None:
            return None
        return f"{self.registry}/{self.repository}@{self.digest}"
