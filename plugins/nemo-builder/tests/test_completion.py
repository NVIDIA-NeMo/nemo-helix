# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Completion by signature: what makes a row `ready`, and everything that must not.

The signatures here are what cosign produces -- a simple-signing payload, signed with an ECDSA key --
so a test that passes is one the broker's real output would pass too. Most tests are a courier
trying something: another row's signature, a forged digest, a signature from another key.
"""

from __future__ import annotations

import base64
import json

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from nemo_builder_plugin.backend import SignatureRefused
from nemo_builder_plugin.completion import (
    SIMPLE_SIGNING_TYPE,
    CompletionConflict,
    SignatureDelivery,
    check_signature,
    complete,
    read_payload,
)
from nemo_builder_plugin.entities import ContainerImage, Provenance
from nemo_builder_plugin.signing import DeploymentKeyPolicy, key_fingerprint, signature_annotations
from nemo_helix_plugin.entities import EntityConflictError

KEY = ec.generate_private_key(ec.SECP256R1())
POLICY = DeploymentKeyPolicy(
    KEY.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
)
DIGEST = "sha256:" + "d" * 64


def _row(name: str = "build-1-0", *, workspace: str = "ws-a") -> ContainerImage:
    return ContainerImage(
        name=name,
        workspace=workspace,
        registry="reg.example.com",
        repository=f"{workspace}/app",
        provenance=Provenance(
            build_set="build",
            revision=1,
            job=f"{workspace}/build-1",
            system_tag=f"{workspace}--build-1-0",
            request_digest="sha256:" + "a" * 64,
        ),
    )


def _payload(row: ContainerImage, *, digest: str = DIGEST, reference: str | None = None, **optional: str) -> bytes:
    """A cosign simple-signing payload, as `cosign sign` writes it with `--output-payload`."""
    document = {
        "critical": {
            "identity": {"docker-reference": reference or f"{row.registry}/{row.repository}"},
            "image": {"docker-manifest-digest": digest},
            "type": SIMPLE_SIGNING_TYPE,
        },
        "optional": signature_annotations(row) | optional,
    }
    return json.dumps(document).encode()


def _signed(payload: bytes, *, key: ec.EllipticCurvePrivateKey = KEY) -> SignatureDelivery:
    signature = key.sign(payload, ec.ECDSA(hashes.SHA256()))
    return SignatureDelivery(payload=base64.b64encode(payload).decode(), signature=base64.b64encode(signature).decode())


class TestCheckSignature:
    def test_the_brokers_signature_for_this_row_completes_it(self) -> None:
        row = _row()
        completion = check_signature(row, _signed(_payload(row)), POLICY)
        assert completion.digest == DIGEST
        assert completion.signature.verified_against == key_fingerprint(KEY.public_key())
        assert completion.signature.signer == signature_annotations(row)

    def test_a_signature_from_another_key_is_refused(self) -> None:
        row = _row()
        with pytest.raises(SignatureRefused, match="does not verify"):
            check_signature(row, _signed(_payload(row), key=ec.generate_private_key(ec.SECP256R1())), POLICY)

    def test_a_payload_changed_after_signing_is_refused(self) -> None:
        """The courier swaps the digest in the payload it carries. The signature no longer covers it."""
        row = _row()
        genuine = _signed(_payload(row))
        forged = SignatureDelivery(
            payload=base64.b64encode(_payload(row, digest="sha256:" + "e" * 64)).decode(), signature=genuine.signature
        )
        with pytest.raises(SignatureRefused, match="does not verify"):
            check_signature(row, forged, POLICY)

    def test_another_rows_genuine_signature_is_refused(self) -> None:
        """A real signature for one image, delivered for another."""
        other = _row("build-1-1")
        with pytest.raises(SignatureRefused, match="does not name build-1-0"):
            check_signature(_row(), _signed(_payload(other, reference="reg.example.com/ws-a/app")), POLICY)

    def test_another_workspaces_signature_is_refused_even_for_a_row_of_the_same_name(self) -> None:
        theirs = _row(workspace="ws-b")
        with pytest.raises(SignatureRefused):
            check_signature(_row(), _signed(_payload(theirs)), POLICY)

    def test_a_signature_for_another_repository_is_refused(self) -> None:
        row = _row()
        with pytest.raises(SignatureRefused, match="not build-1-0's"):
            check_signature(row, _signed(_payload(row, reference="reg.example.com/ws-a/other")), POLICY)

    @pytest.mark.parametrize(
        ("payload", "reason"),
        [
            (b"not json", "not JSON"),
            (b'{"critical": {"type": "something else"}}', "not a"),
            (json.dumps({"critical": {"type": SIMPLE_SIGNING_TYPE, "image": {}}}).encode(), "no repository"),
        ],
    )
    def test_a_signed_document_that_is_not_a_payload_is_refused(self, payload: bytes, reason: str) -> None:
        with pytest.raises(SignatureRefused, match=reason):
            check_signature(_row(), _signed(payload), POLICY)

    def test_a_signature_that_is_not_base64_is_refused(self) -> None:
        with pytest.raises(SignatureRefused, match="not base64"):
            check_signature(_row(), SignatureDelivery(payload="%%%", signature="AAAA"), POLICY)

    def test_a_non_sha256_digest_is_refused(self) -> None:
        row = _row()
        with pytest.raises(SignatureRefused, match="sha256"):
            read_payload(_payload(row, digest="md5:abc"))


class FakeRows:
    """One row, versioned the way the entity store versions it: a stale write is a conflict."""

    def __init__(self, row: ContainerImage) -> None:
        self.row = row
        self.version = 1
        self.writes = 0
        self.interfere: ContainerImage | None = None

    async def get(self, entity_type: type[ContainerImage], *, name: str, workspace: str) -> ContainerImage:
        assert (name, workspace) == (self.row.name, self.row.workspace)
        copy = self.row.model_copy(deep=True)
        copy._db_version = self.version
        return copy

    async def update(self, entity: ContainerImage) -> ContainerImage:
        if self.interfere is not None:
            # Someone else writes between this delivery's read and its write.
            self.row, self.interfere = self.interfere, None
            self.version += 1
        if entity.db_version != self.version:
            raise EntityConflictError("version mismatch")
        self.writes += 1
        self.version += 1
        self.row = entity
        return entity


async def _complete(rows: FakeRows, delivered: SignatureDelivery) -> ContainerImage:
    return await complete(rows, workspace="ws-a", name="build-1-0", delivered=delivered, policy=POLICY)


class TestComplete:
    @pytest.mark.asyncio
    async def test_a_verified_signature_makes_the_row_ready_in_one_write(self) -> None:
        rows = FakeRows(_row())
        ready = await _complete(rows, _signed(_payload(rows.row)))
        assert (ready.status, ready.digest) == ("ready", DIGEST)
        assert ready.signature is not None and ready.signature.verified_against == POLICY.trust_root
        assert rows.writes == 1

    @pytest.mark.asyncio
    async def test_the_same_signature_again_is_accepted_and_writes_nothing(self) -> None:
        """An evicted step that reruns, or a retry after a lost response, still completes."""
        rows = FakeRows(_row())
        delivered = _signed(_payload(rows.row))
        await _complete(rows, delivered)
        again = await _complete(rows, delivered)
        assert again.status == "ready" and rows.writes == 1

    @pytest.mark.asyncio
    async def test_a_signature_for_another_digest_on_a_ready_row_is_a_conflict(self) -> None:
        rows = FakeRows(_row())
        await _complete(rows, _signed(_payload(rows.row)))
        with pytest.raises(CompletionConflict, match="already ready"):
            await _complete(rows, _signed(_payload(rows.row, digest="sha256:" + "e" * 64)))

    @pytest.mark.asyncio
    async def test_a_failed_row_is_never_made_ready(self) -> None:
        row = _row()
        row.status, row.status_detail = "failed", "build job ended as error"
        rows = FakeRows(row)
        with pytest.raises(CompletionConflict, match="already failed"):
            await _complete(rows, _signed(_payload(row)))
        assert rows.writes == 0

    @pytest.mark.asyncio
    async def test_a_sweep_that_fails_the_row_first_wins(self) -> None:
        """The write is conditional: re-read, and the row has failed, so the signature is refused."""
        rows = FakeRows(_row())
        failed = _row()
        failed.status, failed.status_detail = "failed", "build job ended as cancelled"
        rows.interfere = failed
        with pytest.raises(CompletionConflict, match="already failed"):
            await _complete(rows, _signed(_payload(rows.row)))

    @pytest.mark.asyncio
    async def test_a_twin_delivery_that_completes_first_is_success(self) -> None:
        rows = FakeRows(_row())
        delivered = _signed(_payload(rows.row))
        twin = await _complete(FakeRows(_row()), delivered)
        rows.interfere = twin
        assert (await _complete(rows, delivered)).status == "ready"

    @pytest.mark.asyncio
    async def test_a_refused_signature_writes_nothing(self) -> None:
        rows = FakeRows(_row())
        with pytest.raises(SignatureRefused):
            await _complete(rows, _signed(_payload(rows.row), key=ec.generate_private_key(ec.SECP256R1())))
        assert rows.writes == 0 and rows.row.status == "pending"
