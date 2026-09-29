# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What this deployment's key signs, and how the control plane checks it.

Two halves of one contract, kept in one module so they cannot drift:

- :func:`signature_annotations` is what the credential broker attaches when it signs an image:
  every annotation taken from the row, none from the step that asked.
- :class:`DeploymentKeyPolicy` is how the control plane checks the result: the signature must
  verify under the deployment's public key, and the annotations must name *this* row.

**Why the annotations matter as much as the key.** A signature proves the broker signed a digest.
It does not, by itself, say for which row: the broker signs for every workspace. The annotations
are what make the signature for one row useless for another -- the workspace, the row,
the job and the request digest are all signed, so a courier holding a genuine signature for one
image cannot present it for a different one.

ci-toolkit's broker is the counter-example: it lets a caller's annotations through beside three it
controls, so a signature there says only "some GitLab job asked". Here every annotation is the
broker's own.
"""

from __future__ import annotations

import hashlib

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from nemo_builder_plugin.backend import SignatureRefused, SignedPayload
from nemo_builder_plugin.entities import ContainerImage

#: Every annotation this system writes is under this prefix.
ANNOTATION_PREFIX = "nhx.nvidia.com/"


def signature_annotations(row: ContainerImage) -> dict[str, str]:
    """What a signature says about the image -- all of it from the row, none from the caller."""
    origin = row.provenance
    return {
        f"{ANNOTATION_PREFIX}workspace": row.workspace,
        f"{ANNOTATION_PREFIX}image": row.name,
        f"{ANNOTATION_PREFIX}build-set": origin.build_set,
        f"{ANNOTATION_PREFIX}revision": str(origin.revision),
        f"{ANNOTATION_PREFIX}job": origin.job,
        f"{ANNOTATION_PREFIX}request-digest": origin.request_digest,
    }


def key_fingerprint(public_key: ec.EllipticCurvePublicKey | rsa.RSAPublicKey) -> str:
    """``key:sha256:<hex>`` of the key's SubjectPublicKeyInfo: the trust root's id on every row.

    Derived from the key rather than configured beside it, so a row can never name a key other than
    the one that verified it.
    """
    spki = public_key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return f"key:sha256:{hashlib.sha256(spki).hexdigest()}"


class DeploymentKeyPolicy:
    """The signature policy for a row this deployment's broker signed.

    **The key**: the public half of the broker's cosign key, from ``builder.signing_public_key``.
    ECDSA with SHA-256 is what ``cosign generate-key-pair`` makes; RSA with PKCS #1 v1.5 and
    SHA-256 is what a KMS key usually is.

    **The binding**: every annotation :func:`signature_annotations` writes must be in the payload,
    with this row's value. Annotations beyond those are ignored, never recorded.
    """

    def __init__(self, public_key_pem: str) -> None:
        key = serialization.load_pem_public_key(public_key_pem.encode())
        if not isinstance(key, ec.EllipticCurvePublicKey | rsa.RSAPublicKey):
            raise ValueError("the signing public key must be an EC or RSA key")
        self._key = key
        self._trust_root = key_fingerprint(key)

    @property
    def trust_root(self) -> str:
        return self._trust_root

    def verify(self, payload: bytes, signature: bytes) -> None:
        try:
            if isinstance(self._key, ec.EllipticCurvePublicKey):
                self._key.verify(signature, payload, ec.ECDSA(hashes.SHA256()))
            else:
                self._key.verify(signature, payload, padding.PKCS1v15(), hashes.SHA256())
        except InvalidSignature as exc:
            raise SignatureRefused(f"the signature does not verify under {self._trust_root}") from exc

    def bind(self, row: ContainerImage, payload: SignedPayload) -> dict[str, str]:
        expected = signature_annotations(row)
        mismatched = sorted(name for name, value in expected.items() if payload.annotations.get(name) != value)
        if mismatched:
            # Names only, not values: a mismatch is either a bug or someone presenting another
            # row's signature, and neither needs the other row's details echoed back.
            raise SignatureRefused(f"the signature does not name {row.name}: {', '.join(mismatched)} differ")
        return expected
