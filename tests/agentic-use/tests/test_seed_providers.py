# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the declarative provider-seeding module."""

from __future__ import annotations

import json
import textwrap
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest
from seed_providers import (
    DEFAULT_HTTP_TIMEOUT_SEC,
    ProviderSeedResult,
    ProviderSpec,
    SeedResult,
    VirtualModelSpec,
    _create_provider,
    _create_secret,
    _create_virtual_model,
    _wait_for_provider_discovery,
    load_manifest,
    seed_all,
)


@pytest.fixture()
def manifest_path(tmp_path: Path) -> Path:
    """Write a minimal providers.yaml and return its path."""
    p = tmp_path / "providers.yaml"
    p.write_text(
        textwrap.dedent("""\
            providers:
              - name: build-provider
                host_url: https://integrate.api.nvidia.com
                secret_name: build-key
                from_env: BUILD_KEY

              - name: inference-provider
                host_url: https://inference-api.nvidia.com
                secret_name: inf-key
                from_env: INF_KEY
                wait_for_discovery: true
                discovery_timeout_sec: 5
        """)
    )
    return p


class TestLoadManifest:
    def test_loads_two_providers(self, manifest_path: Path) -> None:
        specs = load_manifest(manifest_path)
        assert len(specs) == 2
        assert specs[0].name == "build-provider"
        assert specs[1].wait_for_discovery is True
        assert specs[1].discovery_timeout_sec == 5

    def test_defaults(self, manifest_path: Path) -> None:
        specs = load_manifest(manifest_path)
        assert specs[0].wait_for_discovery is False
        assert specs[0].discovery_timeout_sec == 120

    def test_missing_providers_key(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("something_else: true\n")
        with pytest.raises(ValueError, match="providers"):
            load_manifest(bad)

    def test_duplicate_from_env_is_valid(self, tmp_path: Path) -> None:
        p = tmp_path / "dup.yaml"
        p.write_text(
            textwrap.dedent("""\
                providers:
                  - name: a
                    host_url: https://a.example.com
                    secret_name: key-a
                    from_env: SHARED_KEY
                  - name: b
                    host_url: https://b.example.com
                    secret_name: key-b
                    from_env: SHARED_KEY
            """)
        )
        specs = load_manifest(p)
        assert specs[0].from_env == specs[1].from_env == "SHARED_KEY"


class TestCreateSecret:
    @patch("nemo_helix_plugin.secrets.client.SecretsClient.from_client")
    def test_creates_secret(self, mock_from_client: MagicMock) -> None:
        client = MagicMock()
        mock_secrets = mock_from_client.return_value

        _create_secret(client, "ws", "my-secret", "val")

        mock_from_client.assert_called_once_with(client)
        mock_secrets.create_secret.assert_called_once()
        kwargs = mock_secrets.create_secret.call_args.kwargs
        body = kwargs["body"]
        assert isinstance(body, HelixSecretCreateRequest)
        assert body.name == "my-secret"
        assert body.value.get_secret_value() == "val"
        assert kwargs["workspace"] == "ws"

    @patch("nemo_helix_plugin.secrets.client.SecretsClient.from_client")
    def test_ignores_conflict(self, mock_from_client: MagicMock) -> None:
        mock_from_client.return_value.create_secret.side_effect = Exception("409 Conflict: already exists")

        _create_secret(MagicMock(), "ws", "my-secret", "val")

    @patch("nemo_helix_plugin.secrets.client.SecretsClient.from_client")
    def test_raises_on_other_error(self, mock_from_client: MagicMock) -> None:
        mock_from_client.return_value.create_secret.side_effect = RuntimeError("network error")

        with pytest.raises(RuntimeError, match="network error"):
            _create_secret(MagicMock(), "ws", "my-secret", "val")


class TestCreateProvider:
    @patch("nemo_helix_plugin.models.client.ModelsClient.from_client")
    def test_creates_provider(self, mock_from_client: MagicMock) -> None:
        spec = ProviderSpec(name="p", host_url="https://x.com", secret_name="s", from_env="E")

        _create_provider(MagicMock(), "ws", spec)

        kwargs = mock_from_client.return_value.create_provider.call_args.kwargs
        assert kwargs["workspace"] == "ws"
        assert kwargs["body"].name == "p"
        assert kwargs["body"].host_url == "https://x.com"
        assert kwargs["body"].api_key_secret_name == "s"

    @patch("nemo_helix_plugin.models.client.ModelsClient.from_client")
    def test_ignores_conflict(self, mock_from_client: MagicMock) -> None:
        mock_from_client.return_value.create_provider.side_effect = Exception("409 conflict")
        spec = ProviderSpec(name="p", host_url="https://x.com", secret_name="s", from_env="E")

        _create_provider(MagicMock(), "ws", spec)


class TestWaitForDiscovery:
    @patch("nemo_helix_plugin.models.client.ModelsClient.from_client")
    def test_returns_true_when_models_discovered(self, mock_from_client: MagicMock) -> None:
        provider_obj = MagicMock()
        provider_obj.served_models = [MagicMock(model_entity_id="ns/my-model")]
        mock_from_client.return_value.get_provider.return_value.data.return_value = provider_obj

        spec = ProviderSpec(
            name="p",
            host_url="https://x.com",
            secret_name="s",
            from_env="E",
            wait_for_discovery=True,
            discovery_timeout_sec=1,
        )
        assert _wait_for_provider_discovery(MagicMock(), "ws", spec) is True

    @patch("nemo_helix_plugin.models.client.ModelsClient.from_client")
    def test_returns_false_on_timeout(self, mock_from_client: MagicMock) -> None:
        provider_obj = MagicMock()
        provider_obj.served_models = []
        mock_from_client.return_value.get_provider.return_value.data.return_value = provider_obj

        spec = ProviderSpec(
            name="p",
            host_url="https://x.com",
            secret_name="s",
            from_env="E",
            wait_for_discovery=True,
            discovery_timeout_sec=0,
        )
        assert _wait_for_provider_discovery(MagicMock(), "ws", spec) is False


class TestCreateVirtualModel:
    @patch("urllib.request.urlopen")
    def test_uses_bounded_timeout(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value.__enter__.return_value.read.return_value = b'{"id":"vm-1"}'
        spec = VirtualModelSpec(
            name="vm-test",
            models=[{"model": "ws/model-a", "backend_format": "OPENAI_CHAT"}],
            request_middleware=[],
            response_middleware=[],
        )

        _create_virtual_model("http://localhost:8080", "ws", spec)

        assert mock_urlopen.call_args.kwargs["timeout"] == DEFAULT_HTTP_TIMEOUT_SEC

    @patch("urllib.request.urlopen")
    def test_post_url_and_method(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value.__enter__.return_value.read.return_value = b'{"id":"vm-1"}'
        spec = VirtualModelSpec(
            name="vm-test",
            models=[{"model": "ws/m", "backend_format": "OPENAI_CHAT"}],
            request_middleware=[],
            response_middleware=[],
        )

        _create_virtual_model("http://localhost:8080/", "ws", spec)

        req = mock_urlopen.call_args.args[0]
        assert req.full_url == "http://localhost:8080/apis/entities/v2/workspaces/ws/entities/virtual_model"
        assert req.get_method() == "POST"
        assert req.get_header("Content-type") == "application/json"

    @patch("urllib.request.urlopen")
    def test_request_body_shape(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value.__enter__.return_value.read.return_value = b'{"id":"vm-1"}'
        spec = VirtualModelSpec(
            name="vm-test",
            models=[{"model": "ws/m", "backend_format": "OPENAI_CHAT"}],
            request_middleware=[{"name": "nemo-switchyard", "config_type": "passthrough", "config": {}}],
            response_middleware=[],
        )

        _create_virtual_model("http://localhost:8080", "ws", spec)

        sent = json.loads(mock_urlopen.call_args.args[0].data)
        assert sent["name"] == "vm-test"
        assert sent["data"]["models"] == [{"model": "ws/m", "backend_format": "OPENAI_CHAT"}]
        assert sent["data"]["request_middleware"] == [
            {"name": "nemo-switchyard", "config_type": "passthrough", "config": {}}
        ]
        assert sent["data"]["response_middleware"] == []
        assert sent["data"]["post_response_middleware"] == []
        assert sent["data"]["default_model_entity"] is None
        assert sent["data"]["override_proxy"] is None
        assert sent["data"]["project"] is None

    @patch("urllib.request.urlopen")
    def test_ignores_409_conflict(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://localhost:8080/...",
            code=409,
            msg="Conflict",
            hdrs={},
            fp=BytesIO(b'{"detail":"already exists"}'),
        )
        spec = VirtualModelSpec(name="vm-test", models=[], request_middleware=[], response_middleware=[])

        _create_virtual_model("http://localhost:8080", "ws", spec)


class TestSeedAll:
    @patch("nemo_helix_plugin.models.client.ModelsClient.from_client")
    @patch("nemo_helix_plugin.secrets.client.SecretsClient.from_client")
    @patch("nemo_helix_plugin.client.client.NemoClient")
    def test_skips_unset_env_vars(
        self,
        mock_client_cls: MagicMock,
        mock_secrets_from_client: MagicMock,
        mock_models_from_client: MagicMock,
        manifest_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("BUILD_KEY", raising=False)
        monkeypatch.delenv("INF_KEY", raising=False)

        result = seed_all(manifest_path, base_url="http://localhost:8080")
        assert result.ok
        assert all(p.status == "skipped" for p in result.providers)
        mock_secrets_from_client.return_value.create_secret.assert_not_called()
        mock_models_from_client.return_value.create_provider.assert_not_called()

    @patch("nemo_helix_plugin.models.client.ModelsClient.from_client")
    @patch("nemo_helix_plugin.secrets.client.SecretsClient.from_client")
    @patch("nemo_helix_plugin.client.client.NemoClient")
    def test_seeds_providers(
        self,
        mock_client_cls: MagicMock,
        mock_secrets_from_client: MagicMock,
        mock_models_from_client: MagicMock,
        manifest_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("BUILD_KEY", "nvapi-xxx")
        monkeypatch.setenv("INF_KEY", "sk-yyy")

        provider_obj = MagicMock()
        provider_obj.served_models = [MagicMock(model_entity_id="ns/m")]
        mock_models = mock_models_from_client.return_value
        mock_models.get_provider.return_value.data.return_value = provider_obj

        result = seed_all(manifest_path, base_url="http://localhost:8080")
        assert result.ok
        assert len(result.providers) == 2
        assert result.providers[0].status == "ok"
        assert result.providers[1].status == "ok"
        assert mock_secrets_from_client.return_value.create_secret.call_count == 2
        assert mock_models.create_provider.call_count == 2

    @patch("nemo_helix_plugin.models.client.ModelsClient.from_client")
    @patch("nemo_helix_plugin.secrets.client.SecretsClient.from_client")
    @patch("nemo_helix_plugin.client.client.NemoClient")
    def test_partial_env_skips_missing(
        self,
        mock_client_cls: MagicMock,
        mock_secrets_from_client: MagicMock,
        mock_models_from_client: MagicMock,
        manifest_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("BUILD_KEY", "nvapi-xxx")
        monkeypatch.delenv("INF_KEY", raising=False)

        result = seed_all(manifest_path, base_url="http://localhost:8080")
        assert result.ok
        assert result.providers[0].status == "ok"
        assert result.providers[1].status == "skipped"

    @patch("nemo_helix_plugin.models.client.ModelsClient.from_client")
    @patch("nemo_helix_plugin.secrets.client.SecretsClient.from_client")
    @patch("nemo_helix_plugin.client.client.NemoClient")
    def test_error_on_create_marks_status(
        self,
        mock_client_cls: MagicMock,
        mock_secrets_from_client: MagicMock,
        mock_models_from_client: MagicMock,
        manifest_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("BUILD_KEY", "nvapi-xxx")
        monkeypatch.delenv("INF_KEY", raising=False)

        mock_secrets_from_client.return_value.create_secret.side_effect = RuntimeError("kaboom")

        result = seed_all(manifest_path, base_url="http://localhost:8080")
        assert not result.ok
        assert result.providers[0].status == "error"

    @patch("urllib.request.urlopen")
    @patch("nemo_helix_plugin.client.client.NemoClient")
    def test_skips_vm_when_dep_provider_failed(
        self,
        mock_client_cls: MagicMock,
        mock_urlopen: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        manifest = tmp_path / "providers.yaml"
        manifest.write_text(
            textwrap.dedent("""\
                providers:
                  - name: dep-provider
                    host_url: https://example.com
                    secret_name: s
                    from_env: DEP_KEY
                virtual_models:
                  - name: vm-gated
                    depends_on_provider: dep-provider
                    models: [{model: "default/x", backend_format: "OPENAI_CHAT"}]
                    request_middleware: []
                    response_middleware: []
            """)
        )
        monkeypatch.delenv("DEP_KEY", raising=False)

        result = seed_all(manifest, base_url="http://localhost:8080")

        assert result.providers[0].status == "skipped"
        assert len(result.virtual_models) == 1
        assert result.virtual_models[0].status == "skipped"
        assert "depends_on_provider" in (result.virtual_models[0].message or "")
        assert result.ok
        mock_urlopen.assert_not_called()

    def test_seed_result_summary(self) -> None:
        r = SeedResult()
        r.providers.append(ProviderSeedResult(name="a", status="ok"))
        r.providers.append(ProviderSeedResult(name="b", status="skipped", message="env unset"))
        s = r.summary()
        assert "a: ok" in s
        assert "b: skipped (env unset)" in s
