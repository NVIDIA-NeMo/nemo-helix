# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend interface.

A deployment runs one backend. The submit route, the signature route and the credential broker all
load it from the same configuration. A backend only declares: it never writes a row, reads a
registry, or holds a credential.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.plan import BuildPlan, Destination, PlannedImage
from nemo_helix_plugin.jobs.spec import HelixJobSpec


class BackendRefused(ValueError):
    """The backend cannot build this request, or cannot build at all (409).

    The submit path catches this before a plain ``ValueError``, which it reports as a malformed
    request (400).
    """


@dataclass(frozen=True, slots=True)
class StepEntitlement:
    """What the credential broker may give one step, and how it recognizes the step's pod.

    The pod is recognized by what Jobs set on it, which the step cannot change: its step and profile
    labels, and its ServiceAccount.
    """

    #: The pod's ``job_step_name`` label.
    step: str
    #: The pod's ``job_execution_profile`` label.
    profile: str
    #: The ServiceAccount the profile runs as.
    service_account: str
    #: Registry actions on the job's ``pending`` destinations. Never ``delete``.
    destinations: tuple[Literal["pull", "push"], ...] = ()
    #: Whether the step may ask the broker to sign its job's ``pending`` rows.
    sign: bool = False


@dataclass(frozen=True, slots=True)
class SignedPayload:
    """What a verified signature's payload says. Nothing outside the payload is trusted."""

    #: ``<registry>/<repository>`` the signer found the image at.
    reference: str
    #: The manifest digest the signature covers.
    digest: str
    annotations: Mapping[str, str] = field(default_factory=dict)


class SignatureRefused(Exception):
    """A signature that does not verify, or does not belong to the row it was delivered for."""


class SignaturePolicy(Protocol):
    """How a backend's signatures are checked: verified against a key, then tied to one row."""

    @property
    def trust_root(self) -> str:
        """An id for the key signatures are verified against, recorded on every ``ready`` row."""
        ...

    def verify(self, payload: bytes, signature: bytes) -> None:
        """Raise :class:`SignatureRefused` unless ``signature`` is the trust root's, over ``payload``."""
        ...

    def bind(self, row: ContainerImage, payload: SignedPayload) -> dict[str, str]:
        """Tie a verified payload to ``row``, or raise :class:`SignatureRefused`.

        Returns the claims it checked, which the row records as ``signature.signer``.
        """
        ...


class Backend(Protocol):
    """The deployment's backend."""

    def check(self, plan: BuildPlan) -> None:
        """Raise :class:`BackendRefused` for a request this backend cannot build, or if it is misconfigured."""
        ...

    def destination(self, image: PlannedImage) -> Destination:
        """Where ``image`` is published. Raises ``ValueError`` if it cannot be composed."""
        ...

    def compile(self, plan: BuildPlan) -> HelixJobSpec:
        """The build job for a checked and placed plan."""
        ...

    def signature_policy(self) -> SignaturePolicy:
        """How completion signatures are verified and tied to a row."""
        ...

    def entitlements(self) -> Mapping[str, StepEntitlement]:
        """What the broker may give each step, by step name. A step not listed gets nothing."""
        ...
