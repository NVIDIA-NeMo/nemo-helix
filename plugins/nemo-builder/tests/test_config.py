# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator configuration: what is refused when the platform starts, rather than at every build."""

from __future__ import annotations

import pytest
from nemo_builder_plugin.config import BuilderConfig
from pydantic import ValidationError


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


class TestThePushStepsSecrets:
    def test_each_has_a_name_a_workspace_can_create_it_under(self) -> None:
        config = BuilderConfig()
        names = (config.registry_username_secret, config.registry_password_secret, config.signing_key_secret)
        assert names == ("builder-registry-username", "builder-registry-password", "builder-signing-key")

    def test_an_empty_name_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="signing_key_secret"):
            BuilderConfig(signing_key_secret="")

    def test_each_can_be_turned_off(self) -> None:
        config = BuilderConfig(registry_username_secret=None, registry_password_secret=None, signing_key_secret=None)
        assert config.registry_username_secret is config.registry_password_secret is config.signing_key_secret is None

    def test_half_a_credential_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="set together"):
            BuilderConfig(registry_password_secret=None)


class TestTheSandbox:
    def test_the_section_can_come_from_one_environment_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEMO_BUILDER_SANDBOX", '{"image": "kaniko:1", "dns_nameservers": ["10.0.0.53"]}')
        sandbox = BuilderConfig().sandbox
        assert (sandbox.image, sandbox.dns_nameservers) == ("kaniko:1", ["10.0.0.53"])

    def test_sandboxes_are_plain_pods_by_default(self) -> None:
        assert BuilderConfig().sandbox.provider == "kubernetes_pod"

    def test_a_provider_the_builder_does_not_have_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="provider"):
            BuilderConfig.model_validate({"sandbox": {"provider": "docker"}})

    def test_a_misspelt_setting_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="imgae"):
            BuilderConfig.model_validate({"sandbox": {"imgae": "kaniko:1"}})
