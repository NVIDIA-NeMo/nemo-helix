# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator configuration: what is refused when the platform starts, rather than at every build."""

from __future__ import annotations

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from nemo_builder_plugin.config import BuilderConfig
from pydantic import ValidationError


def _pem(key: ec.EllipticCurvePrivateKey | ed25519.Ed25519PrivateKey) -> str:
    return (
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )


class TestRegistry:
    def test_a_registry_with_a_path_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="without a path"):
            BuilderConfig(registry="us-central1-docker.pkg.dev/proj/repo")

    def test_a_registry_host_is_lowercased(self) -> None:
        assert BuilderConfig(registry="Reg.Example.com:5000").registry == "reg.example.com:5000"

    def test_a_registry_is_reached_over_https_by_default(self) -> None:
        config = BuilderConfig(registry="reg.example.com")
        assert config.registry_plain_http is False
        assert BuilderConfig(registry="https://reg.example.com").registry_plain_http is False

    def test_an_http_registry_is_stored_as_its_host(self) -> None:
        config = BuilderConfig(registry="http://registry.ns.svc.cluster.local:5000")
        assert config.registry == "registry.ns.svc.cluster.local:5000"
        assert config.registry_plain_http is True

    def test_an_http_registry_can_come_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEMO_BUILDER_REGISTRY", "http://registry.local:5000")
        config = BuilderConfig()
        assert (config.registry, config.registry_plain_http) == ("registry.local:5000", True)

    def test_another_scheme_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="http:// or https://"):
            BuilderConfig(registry="oci://reg.example.com")

    def test_the_prefix_is_normalized_and_checked(self) -> None:
        assert BuilderConfig(repository_prefix="/proj/artifacts/").repository_prefix == "proj/artifacts"
        with pytest.raises(ValidationError, match="repository"):
            BuilderConfig(repository_prefix="proj/../elsewhere")

    def test_a_registry_is_not_assumed_to_narrow_scope(self) -> None:
        assert BuilderConfig().registry_narrows_scope is False


class TestTheBrokerAndTheTrustRoot:
    def test_the_broker_is_an_http_url(self) -> None:
        config = BuilderConfig(credential_broker="http://nhx-build-broker.nhx-build-broker.svc:8080/")
        assert config.credential_broker == "http://nhx-build-broker.nhx-build-broker.svc:8080"

    @pytest.mark.parametrize("bad", ["nhx-build-broker:8080", "ftp://broker", "http://", "http://b?x=1"])
    def test_anything_else_is_refused(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="credential_broker"):
            BuilderConfig(credential_broker=bad)

    def test_the_public_key_is_parsed_when_the_platform_starts(self) -> None:
        pem = _pem(ec.generate_private_key(ec.SECP256R1()))
        assert BuilderConfig(signing_public_key=pem).signing_public_key == pem

    def test_a_key_that_is_not_one_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="PEM public key"):
            BuilderConfig(signing_public_key="-----BEGIN PUBLIC KEY-----\ntruncated\n-----END PUBLIC KEY-----\n")

    def test_a_key_cosign_does_not_sign_with_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="EC or RSA"):
            BuilderConfig(signing_public_key=_pem(ed25519.Ed25519PrivateKey.generate()))
