# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Completion: a row goes ``ready`` on a verified signature, and on nothing else (Requirement 6).

**What is delivered.** The cosign simple-signing payload and its signature, exactly as the signer
produced them -- the same bytes a verifier finds in the registry beside the image. The courier,
the build's last step, sends nothing else. The digest, the repository and the claims that bind it
to a row are all read out of the payload *after* the signature verifies, so nothing unsigned is
trusted.

**Why a courier is enough.** The signature is made by the credential broker -- or a backend's own
signer -- over a digest *it* read from the registry, with a key the build plane cannot read. So the
step that carries it can withhold the signature, and then the row fails, but it cannot forge one. What
the original requirement protected -- *never trust a value reported out of a build pod* -- still
holds: the value is read by a party the build plane cannot impersonate, and carried under a
signature it cannot produce.

**Verification, in order:**

1. The signature verifies under the backend's trust root.
2. The payload names a manifest digest.
3. The payload's repository is the row's destination.
4. The payload's claims tie it to this row, under the backend's signature policy.
5. The row is ``pending``.

Then one write, conditional on the version the row was read at, records the digest and the
signature with the trust root that verified it. A signature for a row already ``ready`` with the
same digest is accepted again -- an evicted step that reruns still completes -- and one naming
a different digest for a settled row is refused.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass
from typing import Protocol

from nemo_builder_plugin.backend import SignaturePolicy, SignatureRefused, SignedPayload
from nemo_builder_plugin.entities import ContainerImage, Signature
from nemo_builder_plugin.identity import DIGEST_PATTERN
from nemo_helix_plugin.entities import EntityConflictError
from pydantic import BaseModel, ConfigDict, Field

#: `critical.type` of every cosign simple-signing payload.
SIMPLE_SIGNING_TYPE = "cosign container image signature"

#: A payload is a few hundred bytes of JSON; one of 64 KiB is not a payload.
_MAX_PAYLOAD = 64 * 1024

_DIGEST = re.compile(DIGEST_PATTERN)


class SignatureDelivery(BaseModel):
    """What the build's last step delivers for one row: a signed payload and its signature."""

    model_config = ConfigDict(extra="forbid")

    payload: str = Field(
        max_length=2 * _MAX_PAYLOAD,
        description="base64 of the cosign simple-signing payload, byte for byte as it was signed.",
    )
    signature: str = Field(max_length=4096, description="base64 of the signature over those bytes.")


class CompletionConflict(Exception):
    """A signature for a row that has already settled, and disagrees with how (409)."""


def _b64(value: str, what: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SignatureRefused(f"the {what} is not base64") from exc


def read_payload(payload: bytes) -> SignedPayload:
    """What a verified simple-signing payload says. Raises :class:`SignatureRefused` if it is not one."""
    if len(payload) > _MAX_PAYLOAD:
        raise SignatureRefused("the payload is too large to be a simple-signing payload")
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, ValueError) as exc:
        raise SignatureRefused("the payload is not JSON") from exc
    critical = document.get("critical") if isinstance(document, dict) else None
    if not isinstance(critical, dict) or critical.get("type") != SIMPLE_SIGNING_TYPE:
        raise SignatureRefused(f"the payload is not a {SIMPLE_SIGNING_TYPE!r}")
    identity, image = critical.get("identity"), critical.get("image")
    reference = identity.get("docker-reference") if isinstance(identity, dict) else None
    digest = image.get("docker-manifest-digest") if isinstance(image, dict) else None
    if not isinstance(reference, str) or not reference:
        raise SignatureRefused("the payload names no repository")
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise SignatureRefused("the payload names no sha256 manifest digest")
    optional = document.get("optional")
    annotations = (
        {name: value for name, value in optional.items() if isinstance(value, str)}
        if isinstance(optional, dict)
        else {}
    )
    return SignedPayload(reference=reference, digest=digest, annotations=annotations)


@dataclass(frozen=True, slots=True)
class Completion:
    """What a verified signature lets the control plane write on a row."""

    digest: str
    signature: Signature


def check_signature(row: ContainerImage, delivered: SignatureDelivery, policy: SignaturePolicy) -> Completion:
    """Steps 1 to 4: verify, then read, then tie to ``row``. Raises :class:`SignatureRefused`."""
    payload, signature = _b64(delivered.payload, "payload"), _b64(delivered.signature, "signature")

    policy.verify(payload, signature)
    signed = read_payload(payload)
    destination = f"{row.registry}/{row.repository}"
    if signed.reference != destination:
        raise SignatureRefused(f"the signature is for {signed.reference}, not {row.name}'s {destination}")
    claims = policy.bind(row, signed)
    return Completion(
        digest=signed.digest,
        signature=Signature(verified_against=policy.trust_root, signer=claims),
    )


class _Rows(Protocol):
    """The two entity operations completion needs."""

    async def get(self, entity_type: type[ContainerImage], *, name: str, workspace: str) -> ContainerImage: ...

    async def update(self, entity: ContainerImage) -> ContainerImage: ...


async def complete(
    rows: _Rows,
    *,
    workspace: str,
    name: str,
    delivered: SignatureDelivery,
    policy: SignaturePolicy,
) -> ContainerImage:
    """Verify the signature ``delivered`` for the row ``name`` and, if it holds, write the row ``ready``.

    Raises :class:`SignatureRefused` (the signature is not this row's), :class:`CompletionConflict`
    (the row settled differently), or the store's not-found error.

    **Read, decide, write -- conditionally, and twice at most.** The write carries the version the
    row was read at. If something wrote in between -- the failure sweep failing it, or a twin of
    this delivery completing it -- the write is refused, and the row is read and decided again:
    ``ready`` with this digest is success, anything else a conflict.
    """
    for _ in range(2):
        row = await rows.get(ContainerImage, name=name, workspace=workspace)
        completion = check_signature(row, delivered, policy)
        if row.status == "ready":
            if row.digest == completion.digest:
                return row
            raise CompletionConflict(
                f"{name} is already ready as {row.digest}; this signature names {completion.digest}"
            )
        if row.status == "failed":
            raise CompletionConflict(f"{name} has already failed ({row.status_detail}); ready is not reachable from it")

        row.digest = completion.digest
        row.signature = completion.signature
        row.status = "ready"
        row.status_detail = None
        try:
            return await rows.update(row)
        except EntityConflictError:
            continue
    raise CompletionConflict(f"{name} kept changing while its signature was being recorded")
