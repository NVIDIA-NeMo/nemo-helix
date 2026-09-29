# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What this deployment's key signs, and how a signature is checked and tied to one row."""

from __future__ import annotations

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from nemo_builder_plugin.backend import SignatureRefused, SignedPayload
from nemo_builder_plugin.entities import ContainerImage, Provenance
from nemo_builder_plugin.signing import DeploymentKeyPolicy, key_fingerprint, signature_annotations

EC_KEY = ec.generate_private_key(ec.SECP256R1())
RSA_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PAYLOAD = b'{"critical":{"type":"cosign container image signature"}}'


def _pem(key: ec.EllipticCurvePrivateKey | rsa.RSAPrivateKey) -> str:
    return (
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )


def _row(name: str = "demo-1-0", *, workspace: str = "default", job: str = "default/demo-1") -> ContainerImage:
    return ContainerImage(
        name=name,
        workspace=workspace,
        registry="reg.example.com",
        repository=f"{workspace}/team/app",
        provenance=Provenance(
            build_set="demo",
            revision=1,
            job=job,
            system_tag=f"{workspace}--demo-1-0",
            request_digest="sha256:" + "a" * 64,
        ),
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


class TestVerify:
    def test_an_ecdsa_signature_from_the_deployments_key_verifies(self) -> None:
        """What `cosign generate-key-pair` makes."""
        policy = DeploymentKeyPolicy(_pem(EC_KEY))
        policy.verify(PAYLOAD, EC_KEY.sign(PAYLOAD, ec.ECDSA(hashes.SHA256())))

    def test_an_rsa_signature_verifies_too(self) -> None:
        """What a KMS key usually is."""
        policy = DeploymentKeyPolicy(_pem(RSA_KEY))
        policy.verify(PAYLOAD, RSA_KEY.sign(PAYLOAD, padding.PKCS1v15(), hashes.SHA256()))

    def test_another_keys_signature_is_refused(self) -> None:
        other = ec.generate_private_key(ec.SECP256R1())
        with pytest.raises(SignatureRefused, match="does not verify"):
            DeploymentKeyPolicy(_pem(EC_KEY)).verify(PAYLOAD, other.sign(PAYLOAD, ec.ECDSA(hashes.SHA256())))

    def test_a_signature_over_other_bytes_is_refused(self) -> None:
        signature = EC_KEY.sign(PAYLOAD, ec.ECDSA(hashes.SHA256()))
        with pytest.raises(SignatureRefused):
            DeploymentKeyPolicy(_pem(EC_KEY)).verify(PAYLOAD + b" ", signature)

    def test_the_trust_root_is_named_by_the_keys_own_fingerprint(self) -> None:
        policy = DeploymentKeyPolicy(_pem(EC_KEY))
        assert policy.trust_root == key_fingerprint(EC_KEY.public_key())
        assert policy.trust_root.startswith("key:sha256:") and len(policy.trust_root) == len("key:sha256:") + 64


class TestBind:
    def _payload(self, row: ContainerImage, **changes: str) -> SignedPayload:
        annotations = signature_annotations(row) | {f"nhx.nvidia.com/{k}": v for k, v in changes.items()}
        return SignedPayload(
            reference=f"{row.registry}/{row.repository}", digest="sha256:" + "b" * 64, annotations=annotations
        )

    def test_a_signature_naming_this_row_binds_and_says_what_was_checked(self) -> None:
        row = _row()
        assert DeploymentKeyPolicy(_pem(EC_KEY)).bind(row, self._payload(row)) == signature_annotations(row)

    @pytest.mark.parametrize("claim", ["workspace", "image", "job", "request-digest"])
    def test_a_signature_for_another_row_does_not(self, claim: str) -> None:
        """A genuine signature for one image, presented for another, is refused."""
        row = _row()
        with pytest.raises(SignatureRefused, match=claim):
            DeploymentKeyPolicy(_pem(EC_KEY)).bind(row, self._payload(row, **{claim: "something-else"}))

    def test_a_missing_annotation_is_a_mismatch(self) -> None:
        row = _row()
        payload = self._payload(row)
        stripped = SignedPayload(
            reference=payload.reference,
            digest=payload.digest,
            annotations={k: v for k, v in payload.annotations.items() if not k.endswith("/job")},
        )
        with pytest.raises(SignatureRefused, match="job"):
            DeploymentKeyPolicy(_pem(EC_KEY)).bind(row, stripped)

    def test_extra_annotations_are_ignored_and_not_recorded(self) -> None:
        row = _row()
        payload = self._payload(row)
        extra = SignedPayload(
            reference=payload.reference, digest=payload.digest, annotations={**payload.annotations, "note": "x"}
        )
        assert "note" not in DeploymentKeyPolicy(_pem(EC_KEY)).bind(row, extra)
