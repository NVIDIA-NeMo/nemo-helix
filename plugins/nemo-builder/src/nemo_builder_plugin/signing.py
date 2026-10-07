# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a build's signatures say, where they're stored, and the key that makes them.

Signatures are in cosign's format, stored where cosign looks, so `cosign verify --key` checks them.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.registry import OCI_MANIFEST, ManifestNotFound, Registry

ANNOTATION_PREFIX = "nhx.nvidia.com/"
SIMPLE_SIGNING_MEDIA_TYPE = "application/vnd.dev.cosign.simplesigning.v1+json"
SIGNATURE_ANNOTATION = "dev.cosignproject.cosign/signature"
OCI_CONFIG = "application/vnd.oci.image.config.v1+json"


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


def simple_signing_payload(repository: str, digest: str, annotations: Mapping[str, str]) -> bytes:
    """What cosign signs for an image: its repository and digest, then ``annotations``."""
    claims = {
        "critical": {
            "identity": {"docker-reference": repository},
            "image": {"docker-manifest-digest": digest},
            "type": "cosign container image signature",
        },
        "optional": dict(annotations),
    }
    return _json(claims)


def signature_tag(digest: str) -> str:
    """The tag cosign keeps ``digest``'s signatures under: ``sha256-<hex>.sig``."""
    algorithm, _, hex_digest = digest.partition(":")
    return f"{algorithm}-{hex_digest}.sig"


def attach_signature(registry: Registry, repository: str, digest: str, payload: bytes, signature: bytes) -> None:
    """Add a signature to the image cosign keeps ``digest``'s signatures in, beside any already there."""
    tag = signature_tag(digest)
    try:
        existing = registry.manifest(repository, tag)[1].get("layers", [])
    except ManifestNotFound:
        existing = []
    layers = [layer for layer in existing if isinstance(layer, dict) and isinstance(layer.get("digest"), str)]
    layers.append(
        {
            "mediaType": SIMPLE_SIGNING_MEDIA_TYPE,
            "size": len(payload),
            "digest": registry.put_blob(repository, payload),
            "annotations": {SIGNATURE_ANNOTATION: base64.b64encode(signature).decode()},
        }
    )
    config = _json(
        {"architecture": "", "os": "", "rootfs": {"type": "layers", "diff_ids": [layer["digest"] for layer in layers]}}
    )
    manifest = {
        "schemaVersion": 2,
        "mediaType": OCI_MANIFEST,
        "config": {"mediaType": OCI_CONFIG, "size": len(config), "digest": registry.put_blob(repository, config)},
        "layers": layers,
    }
    registry.put_manifest(repository, tag, _json(manifest), OCI_MANIFEST)


def key_fingerprint(public_key: ec.EllipticCurvePublicKey | rsa.RSAPublicKey) -> str:
    """``key:sha256:<hex>`` of the key's SubjectPublicKeyInfo, which names the key without revealing it."""
    spki = public_key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return f"key:sha256:{hashlib.sha256(spki).hexdigest()}"


class SigningKey:
    """The private key a build signs with, from PEM: EC or RSA, as `cosign verify --key` checks."""

    def __init__(self, pem: bytes, password: bytes | None = None) -> None:
        key = serialization.load_pem_private_key(pem, password=password)
        if not isinstance(key, ec.EllipticCurvePrivateKey | rsa.RSAPrivateKey):
            raise ValueError("the signing key must be an EC or RSA key")
        self._key = key
        self._fingerprint = key_fingerprint(key.public_key())

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def sign(self, payload: bytes) -> bytes:
        # ECDSA, or RSA PKCS #1 v1.5, over SHA-256: what `cosign verify --key` checks.
        if isinstance(self._key, ec.EllipticCurvePrivateKey):
            return self._key.sign(payload, ec.ECDSA(hashes.SHA256()))
        return self._key.sign(payload, padding.PKCS1v15(), hashes.SHA256())


def sign_image(registry: Registry, row: ContainerImage, digest: str, key: SigningKey) -> None:
    """Sign ``digest`` as ``row``'s image, with every annotation from the row, and store it where cosign looks."""
    payload = simple_signing_payload(f"{row.registry}/{row.repository}", digest, signature_annotations(row))
    attach_signature(registry, row.repository, digest, payload, key.sign(payload))


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
