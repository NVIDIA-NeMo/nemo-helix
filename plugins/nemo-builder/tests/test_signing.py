# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a build's signatures say, how they are stored, and the key that makes them."""

from __future__ import annotations

import base64
import json

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa
from nemo_builder_plugin.entities import ContainerImage, Provenance
from nemo_builder_plugin.registry import OCI_MANIFEST, Registry
from nemo_builder_plugin.signing import (
    OCI_CONFIG,
    SIGNATURE_ANNOTATION,
    SIMPLE_SIGNING_MEDIA_TYPE,
    SigningKey,
    attach_signature,
    key_fingerprint,
    sign_image,
    signature_annotations,
    signature_tag,
    simple_signing_payload,
)

EC_KEY = ec.generate_private_key(ec.SECP256R1())
RSA_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PAYLOAD = b'{"critical":{"type":"cosign container image signature"}}'
DIGEST = "sha256:" + "d" * 64
AUTHORIZATION = "Bearer registry-token"


def _verify(key: ec.EllipticCurvePrivateKey | rsa.RSAPrivateKey, payload: bytes, signature: bytes) -> None:
    """What `cosign verify --key` checks: ECDSA, or RSA PKCS #1 v1.5, over SHA-256. Raises if it doesn't verify."""
    public = key.public_key()
    if isinstance(public, ec.EllipticCurvePublicKey):
        public.verify(signature, payload, ec.ECDSA(hashes.SHA256()))
    else:
        public.verify(signature, payload, padding.PKCS1v15(), hashes.SHA256())


def _row() -> ContainerImage:
    return ContainerImage(
        name="demo-1-0",
        workspace="default",
        registry="reg.example.com",
        repository="default/team/app",
        provenance=Provenance(build_set="demo", revision=1, job="default/demo-1", request_digest="sha256:" + "a" * 64),
    )


class TestAnnotations:
    def test_every_annotation_comes_from_the_row(self) -> None:
        assert signature_annotations(_row()) == {
            "nhx.nvidia.com/workspace": "default",
            "nhx.nvidia.com/image": "demo-1-0",
            "nhx.nvidia.com/build-set": "demo",
            "nhx.nvidia.com/revision": "1",
            "nhx.nvidia.com/job": "default/demo-1",
            "nhx.nvidia.com/request-digest": "sha256:" + "a" * 64,
        }


def _private_pem(
    key: ec.EllipticCurvePrivateKey | rsa.RSAPrivateKey | ed25519.Ed25519PrivateKey, password: bytes | None = None
) -> bytes:
    encryption = serialization.BestAvailableEncryption(password) if password else serialization.NoEncryption()
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, encryption)


class TestTheSigningKey:
    @pytest.mark.parametrize("key", [EC_KEY, RSA_KEY], ids=["ecdsa", "rsa"])
    def test_what_it_signs_verifies_under_its_public_half(
        self, key: ec.EllipticCurvePrivateKey | rsa.RSAPrivateKey
    ) -> None:
        signing = SigningKey(_private_pem(key))
        _verify(key, PAYLOAD, signing.sign(PAYLOAD))

    def test_it_is_named_by_its_public_halfs_fingerprint(self) -> None:
        fingerprint = SigningKey(_private_pem(EC_KEY)).fingerprint
        assert fingerprint == key_fingerprint(EC_KEY.public_key())
        assert fingerprint.startswith("key:sha256:") and len(fingerprint) == len("key:sha256:") + 64

    def test_an_encrypted_key_needs_its_password(self) -> None:
        pem = _private_pem(EC_KEY, b"secret")
        with pytest.raises(TypeError):
            SigningKey(pem)
        assert SigningKey(pem, b"secret").fingerprint == key_fingerprint(EC_KEY.public_key())

    def test_a_key_cosign_does_not_check_is_refused(self) -> None:
        with pytest.raises(ValueError, match="EC or RSA"):
            SigningKey(_private_pem(ed25519.Ed25519PrivateKey.generate()))


class TestTheCosignFormat:
    def test_the_payload_names_the_image_and_carries_the_annotations(self) -> None:
        payload = simple_signing_payload("reg.example.com/ws/app", DIGEST, {"nhx.nvidia.com/job": "ws/demo-1"})
        assert json.loads(payload) == {
            "critical": {
                "identity": {"docker-reference": "reg.example.com/ws/app"},
                "image": {"docker-manifest-digest": DIGEST},
                "type": "cosign container image signature",
            },
            "optional": {"nhx.nvidia.com/job": "ws/demo-1"},
        }

    def test_signatures_are_kept_under_the_digests_sig_tag(self) -> None:
        assert signature_tag(DIGEST) == "sha256-" + "d" * 64 + ".sig"

    def test_a_signature_is_stored_as_a_layer_cosign_reads(self, oci_registry) -> None:
        attach_signature(
            Registry("reg.example.com", AUTHORIZATION, http=oci_registry.client()), "ws/app", DIGEST, PAYLOAD, b"sig"
        )
        manifest = oci_registry.manifest("ws/app", signature_tag(DIGEST))
        assert (manifest["mediaType"], manifest["config"]["mediaType"]) == (OCI_MANIFEST, OCI_CONFIG)
        [layer] = manifest["layers"]
        assert layer["mediaType"] == SIMPLE_SIGNING_MEDIA_TYPE and layer["size"] == len(PAYLOAD)
        assert layer["annotations"] == {SIGNATURE_ANNOTATION: base64.b64encode(b"sig").decode()}
        assert oci_registry.blobs[("ws/app", layer["digest"])] == PAYLOAD

    def test_another_signature_is_kept_beside_the_ones_already_there(self, oci_registry) -> None:
        registry = Registry("reg.example.com", AUTHORIZATION, http=oci_registry.client())
        attach_signature(registry, "ws/app", DIGEST, b"first", b"one")
        attach_signature(registry, "ws/app", DIGEST, b"second", b"two")
        layers = oci_registry.manifest("ws/app", signature_tag(DIGEST))["layers"]
        assert [oci_registry.blobs[("ws/app", layer["digest"])] for layer in layers] == [b"first", b"second"]


class TestSigningAnImage:
    def test_the_digest_is_signed_as_the_rows_image_with_the_rows_annotations(self, oci_registry) -> None:
        row = _row()
        sign_image(
            Registry(row.registry, AUTHORIZATION, http=oci_registry.client()),
            row,
            DIGEST,
            SigningKey(_private_pem(EC_KEY)),
        )
        [layer] = oci_registry.manifest(row.repository, signature_tag(DIGEST))["layers"]
        payload = oci_registry.blobs[(row.repository, layer["digest"])]
        assert payload == simple_signing_payload(f"{row.registry}/{row.repository}", DIGEST, signature_annotations(row))
        _verify(EC_KEY, payload, base64.b64decode(layer["annotations"][SIGNATURE_ANNOTATION]))
