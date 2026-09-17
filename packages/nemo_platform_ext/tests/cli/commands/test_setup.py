# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the nemo setup command."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import httpx
import nemo_platform_ext.cli.commands.setup as setup_commands
import pytest
import typer
from click.core import ParameterSource
from click.exceptions import Exit as ClickExit
from nemo_platform import APIConnectionError, APIStatusError, APITimeoutError
from nemo_platform.resources.inference.providers import ProvidersResource
from nemo_platform_ext.cli.commands.setup import (
    _AGENT_API_READINESS_POLL_INTERVAL,
    _AGENT_DEPLOY_POLL_INTERVAL,
    _AGENT_MARKERS,
    _CONTROLLER_HEALTH_RETRY_DELAY,
    _KNOWN_PROVIDERS_BY_NAME,
    _MODEL_DISCOVERY_POLL_INTERVAL,
    _POST_START_REACHABLE_DELAY,
    _POST_START_REACHABLE_RETRIES,
    _PROBE_CONFIGS,
    _SERVICE_STARTUP_POLL_INTERVAL,
    _SERVICE_STARTUP_TIMEOUT_SECONDS,
    KNOWN_PROVIDERS,
    ONBOARDING_PATHS,
    KeyValidationResult,
    ModelPair,
    _agent_config_path,
    _agent_exists,
    _agents_api_ready,
    _agents_plugin_available,
    _auto_setup,
    _bootstrap_config_if_missing,
    _check_controller_health,
    _check_ollama_running,
    _check_platform_reachable,
    _check_platform_reachable_with_retries,
    _configure_local_connection,
    _create_provider,
    _deploy_demo_agent,
    _detect_coding_agents,
    _detect_startup_port_conflict,
    _ensure_port_available_for_start,
    _filter_agents_by_scope,
    _find_project_root,
    _is_preferred_vendor,
    _kill_existing_services,
    _last_startup_service,
    _load_persisted_data_dir,
    _load_skills_with_warnings,
    _maybe_deploy_agent,
    _maybe_install_skills,
    _maybe_start_services,
    _model_parameter_size,
    _order_candidates_by_size,
    _parse_csv_flag,
    _pick_default_chat_entity,
    _print_onboarding,
    _probe_model_entity,
    _prompt_custom_provider,
    _register_provider_interactive,
    _render_onboarding_card,
    _require_supported_python,
    _resolve_provider_for_url,
    _resolve_setup_workspace,
    _run_auto_mode,
    _run_interactive_mode,
    _save_data_dir,
    _select_model_pair,
    _select_usable_model_pair,
    _services_log_suggests_port_conflict,
    _start_services_background,
    _validate_api_key,
    _verify_platform_health,
    _wait_for_models,
    _wait_for_platform,
    setup_command,
)
from nemo_platform_ext.cli.commands.skills import registry as skills_registry
from nemo_platform_ext.cli.commands.skills.base import Scope, Skill
from nemo_platform_ext.cli.commands.skills.registry import UnsupportedAgentError
from nemo_platform_ext.config.config import Config
from nemo_platform_ext.config.models import (
    DEFAULT_BASE_URL,
    Cluster,
    ConfigFile,
    ConfigParams,
    Context,
    ContextDefinition,
    NoAuthUser,
    OAuthUser,
)
from nemo_platform_ext.local.process import PortConflict
from nemo_platform_ext.ui.prompts import UserCancelled
from nemo_platform_plugin.client.errors import NotFoundError
from nemo_platform_plugin.secrets.types import PlatformSecretCreateRequest, PlatformSecretUpdateRequest
from pydantic import SecretStr

SETUP_MOD = "nemo_platform_ext.cli.commands.setup"


@pytest.fixture(autouse=True)
def _silence_telemetry():
    """Silence stray telemetry side effects across this module.

    These are direct-call unit tests with no telemetry intent. Several exercise
    the real setup wrappers (`_create_provider`, `_wait_for_models`,
    `_deploy_demo_agent`, `_auto_setup`), which call the real `emit_event`. With
    telemetry enabled by default that constructs a `TelemetryHandler` and
    schedules `_flush_events`, whose orphaned coroutine surfaces later as a
    "coroutine ... was never awaited" RuntimeWarning (blamed on whichever
    mock-heavy test triggers GC, not the emitting one). Patch emit_event at module
    scope so no handler is ever built. No test in this file asserts on emit_event
    (the telemetry assertions live in test_onboarding_events.py), so this weakens
    nothing.
    """
    with patch("nemo_platform_ext.cli.telemetry.emit.emit_event"):
        yield


# ---------------------------------------------------------------------------
# KnownProvider catalog tests
# ---------------------------------------------------------------------------


class TestKnownProviderCatalog:
    def test_catalog_not_empty(self):
        assert len(KNOWN_PROVIDERS) > 0

    def test_all_providers_have_required_fields(self):
        for p in KNOWN_PROVIDERS:
            assert p.name
            assert p.label
            assert p.description
            assert p.host_url

    def test_by_name_lookup_matches_catalog(self):
        for p in KNOWN_PROVIDERS:
            assert _KNOWN_PROVIDERS_BY_NAME[p.name] is p

    def test_ollama_does_not_require_api_key(self):
        ollama = _KNOWN_PROVIDERS_BY_NAME["ollama"]
        assert not ollama.requires_api_key
        assert ollama.env_var is None

    def test_anthropic_has_custom_auth_header(self):
        anthropic = _KNOWN_PROVIDERS_BY_NAME["anthropic"]
        assert anthropic.auth_header_format is not None
        assert "X-Api-Key" in anthropic.auth_header_format
        assert anthropic.default_extra_headers is not None
        assert "anthropic-version" in anthropic.default_extra_headers

    def test_openai_uses_bearer_auth(self):
        openai = _KNOWN_PROVIDERS_BY_NAME["openai"]
        assert openai.auth_header_format is None
        assert openai.requires_api_key

    def test_frozen_dataclass(self):
        p = KNOWN_PROVIDERS[0]
        with pytest.raises(AttributeError):
            p.name = "changed"


# ---------------------------------------------------------------------------
# URL resolution
# ---------------------------------------------------------------------------


class TestResolveProviderForUrl:
    def test_exact_match(self):
        result = _resolve_provider_for_url("https://api.openai.com/v1")
        assert result is not None
        assert result.name == "openai"

    def test_trailing_slash_stripped(self):
        result = _resolve_provider_for_url("https://api.openai.com/v1/")
        assert result is not None
        assert result.name == "openai"

    def test_unknown_url_returns_none(self):
        result = _resolve_provider_for_url("https://my-custom-llm.example.com/v1")
        assert result is None

    def test_each_known_provider_resolves(self):
        for p in KNOWN_PROVIDERS:
            assert _resolve_provider_for_url(p.host_url) is p


# ---------------------------------------------------------------------------
# Platform reachability
# ---------------------------------------------------------------------------


class TestCheckPlatformReachable:
    def test_reachable_via_status(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with (
            patch(f"{SETUP_MOD}.httpx_tls_config_from_env", return_value={"verify": "/tmp/custom-ca.pem"}),
            patch(f"{SETUP_MOD}.httpx.get", return_value=mock_resp) as mock_get,
        ):
            assert _check_platform_reachable("http://localhost:8080") is True
        mock_get.assert_called_once_with(
            "http://localhost:8080/status",
            timeout=5.0,
            verify="/tmp/custom-ca.pem",
        )

    def test_uses_context_certificate_authority_when_env_unset(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(f"{SETUP_MOD}.httpx.get", return_value=mock_resp) as mock_get,
        ):
            assert (
                _check_platform_reachable(
                    "https://nemo.example.com",
                    certificate_authority="/ctx/ca.pem",
                )
                is True
            )

        mock_get.assert_called_once_with(
            "https://nemo.example.com/status",
            timeout=5.0,
            verify="/ctx/ca.pem",
        )

    def test_env_certificate_authority_overrides_context_certificate_authority(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with (
            patch.dict("os.environ", {"NMP_CLIENT_SSL_CERT_FILE": "/env/ca.pem"}, clear=True),
            patch(f"{SETUP_MOD}.httpx.get", return_value=mock_resp) as mock_get,
        ):
            assert (
                _check_platform_reachable(
                    "https://nemo.example.com",
                    certificate_authority="/ctx/ca.pem",
                )
                is True
            )

        mock_get.assert_called_once_with(
            "https://nemo.example.com/status",
            timeout=5.0,
            verify="/env/ca.pem",
        )

    def test_reachable_via_cluster_info_when_status_missing(self):
        status_resp = MagicMock()
        status_resp.status_code = 404
        cluster_resp = MagicMock()
        cluster_resp.status_code = 200
        calls: list[tuple[str, object]] = []

        def _get(url, **kwargs):
            calls.append((url, kwargs.get("verify")))
            if url.endswith("/status"):
                return status_resp
            if url.endswith("/cluster-info"):
                return cluster_resp
            raise AssertionError(f"unexpected url: {url}")

        with (
            patch(f"{SETUP_MOD}.httpx_tls_config_from_env", return_value={"verify": "/tmp/custom-ca.pem"}),
            patch(f"{SETUP_MOD}.httpx.get", side_effect=_get),
        ):
            assert _check_platform_reachable("https://nemo-platform-freeplay.dev.aire.nvidia.com") is True
        assert calls == [
            ("https://nemo-platform-freeplay.dev.aire.nvidia.com/status", "/tmp/custom-ca.pem"),
            ("https://nemo-platform-freeplay.dev.aire.nvidia.com/cluster-info", "/tmp/custom-ca.pem"),
        ]

    def test_unreachable_when_all_probes_fail(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch(f"{SETUP_MOD}.httpx.get", return_value=mock_resp):
            assert _check_platform_reachable("http://localhost:8080") is False

    def test_unreachable(self):
        with patch(f"{SETUP_MOD}.httpx.get", side_effect=Exception("conn refused")):
            assert _check_platform_reachable("http://localhost:8080") is False

    def test_non_200_status_without_cluster_info_fallback(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        with patch(f"{SETUP_MOD}.httpx.get", return_value=mock_resp):
            assert _check_platform_reachable("http://localhost:8080") is False


# ---------------------------------------------------------------------------
# Platform reachability with retries
# ---------------------------------------------------------------------------


class TestCheckPlatformReachableWithRetries:
    """After startup the health endpoint can flip-flop while controllers initialize."""

    def test_succeeds_on_first_try(self):
        with patch(f"{SETUP_MOD}._check_platform_reachable", return_value=True):
            assert _check_platform_reachable_with_retries("http://localhost:8080") is True

    def test_succeeds_after_transient_failures(self):
        with (
            patch(f"{SETUP_MOD}._check_platform_reachable", side_effect=[False, False, True]),
            patch(f"{SETUP_MOD}._pause") as mock_pause,
        ):
            assert _check_platform_reachable_with_retries("http://localhost:8080", retries=3) is True
        assert mock_pause.call_count == 2

    def test_fails_after_all_retries_exhausted(self):
        with (
            patch(f"{SETUP_MOD}._check_platform_reachable", return_value=False),
            patch(f"{SETUP_MOD}._pause"),
        ):
            assert _check_platform_reachable_with_retries("http://localhost:8080", retries=3) is False

    def test_uses_configured_delay_between_retries(self):
        with (
            patch(f"{SETUP_MOD}._check_platform_reachable", side_effect=[False, True]),
            patch(f"{SETUP_MOD}._pause") as mock_pause,
        ):
            _check_platform_reachable_with_retries("http://localhost:8080", retries=2, delay=4.0)
        mock_pause.assert_called_once_with(4.0)

    def test_no_pause_after_final_failed_attempt(self):
        with (
            patch(f"{SETUP_MOD}._check_platform_reachable", return_value=False),
            patch(f"{SETUP_MOD}._pause") as mock_pause,
        ):
            _check_platform_reachable_with_retries("http://localhost:8080", retries=2)
        assert mock_pause.call_count == 1

    def test_defaults_match_constants(self):
        assert _POST_START_REACHABLE_RETRIES == 6
        assert _POST_START_REACHABLE_DELAY == 2.0

    def test_passes_certificate_authority_to_single_probe(self):
        with patch(f"{SETUP_MOD}._check_platform_reachable", return_value=True) as mock_reachable:
            assert (
                _check_platform_reachable_with_retries(
                    "https://nemo.example.com",
                    certificate_authority="/ctx/ca.pem",
                )
                is True
            )

        mock_reachable.assert_called_once_with(
            "https://nemo.example.com",
            certificate_authority="/ctx/ca.pem",
        )


# ---------------------------------------------------------------------------
# Ollama detection
# ---------------------------------------------------------------------------


class TestCheckOllamaRunning:
    def test_ollama_running(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("nemo_platform_ext.cli.commands.setup.httpx.get", return_value=mock_resp):
            assert _check_ollama_running("http://localhost:11434/v1") is True

    def test_ollama_not_running(self):
        with patch("nemo_platform_ext.cli.commands.setup.httpx.get", side_effect=Exception("conn refused")):
            assert _check_ollama_running("http://localhost:11434/v1") is False


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------


def _not_found_error() -> NotFoundError:
    """Build a NotFoundError backed by a mock 404 response for `get_secret`."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 404
    resp.json.side_effect = ValueError("no body")
    resp.text = "not found"
    resp.reason_phrase = "Not Found"
    return NotFoundError(resp)


def _make_mock_secrets_client(*, secret_exists: bool = False) -> MagicMock:
    """Build a mock typed SecretsClient with get/create/update as MagicMocks."""
    secrets = MagicMock()
    secrets.get_secret = MagicMock()
    secrets.create_secret = MagicMock()
    secrets.update_secret = MagicMock()
    if secret_exists:
        secrets.get_secret.return_value = MagicMock()
    else:
        secrets.get_secret.side_effect = _not_found_error()
    return secrets


def _make_mock_client(*, provider_exists: bool = False, secret_exists: bool = False) -> MagicMock:
    """Build a mock NeMoPlatform client with configurable provider/secret state.

    The typed secrets client returned by ``client_from_platform`` (patched via
    the ``_patch_secrets_client`` fixture) is stashed on ``client.mock_secrets``
    so tests can assert on ``get_secret``/``create_secret``/``update_secret``.
    """
    client = MagicMock()
    client.inference.providers = MagicMock(spec=ProvidersResource)
    client.mock_secrets = _make_mock_secrets_client(secret_exists=secret_exists)
    if provider_exists:
        client.inference.providers.retrieve.return_value = MagicMock()
    else:
        client.inference.providers.retrieve.side_effect = Exception("not found")
    return client


@pytest.fixture(autouse=True)
def _patch_secrets_client():
    """Route ``client_from_platform(client, SecretsClient)`` to ``client.mock_secrets``.

    The secrets helpers in setup.py obtain a typed ``SecretsClient`` via
    ``client_from_platform``. Real adaptation would run against a MagicMock and
    blow up in ``raise_for_status``, so tests patch it to return the mock secrets
    client attached to the mock platform client (falling back to a fresh mock for
    plain ``MagicMock()`` clients that lack ``mock_secrets``).
    """

    def _resolve(client, _client_cls):
        secrets = getattr(client, "mock_secrets", None)
        if isinstance(secrets, MagicMock):
            return secrets
        fallback = _make_mock_secrets_client()
        client.mock_secrets = fallback
        return fallback

    with patch(f"{SETUP_MOD}.client_from_platform", side_effect=_resolve):
        yield


# ---------------------------------------------------------------------------
# Provider creation
# ---------------------------------------------------------------------------


class TestCreateProvider:
    """Tests for _create_provider -- ensures only SDK-supported kwargs are passed."""

    def _make_client(self):
        client = MagicMock()
        client.inference.providers = MagicMock(spec=ProvidersResource)
        return client

    def test_anthropic_provider_kwargs(self):
        """Provider creation keeps auth templating in the dedicated field."""
        client = self._make_client()
        _create_provider(
            client,
            name="anthropic",
            host_url="https://api.anthropic.com",
            secret_name="anthropic-api-key",
            workspace="default",
            auth_header_format="X-Api-Key: {{ auth_secret }}",
            default_extra_headers={"anthropic-version": "2023-06-01"},
        )
        call_kwargs = client.inference.providers.create.call_args.kwargs
        assert call_kwargs["api_key_secret_name"] == "anthropic-api-key"
        assert call_kwargs["auth_header_format"] == "X-Api-Key: {{ auth_secret }}"
        assert "required_extra_headers" not in call_kwargs
        assert call_kwargs["default_extra_headers"] == {"anthropic-version": "2023-06-01"}

    def test_no_auth_header_format_skips_auth_fields(self):
        """Providers using default Bearer auth do not send auth overrides."""
        client = self._make_client()
        _create_provider(
            client,
            name="openai",
            host_url="https://api.openai.com/v1",
            secret_name="openai-api-key",
            workspace="default",
        )
        call_kwargs = client.inference.providers.create.call_args.kwargs
        assert "required_extra_headers" not in call_kwargs
        assert "auth_header_format" not in call_kwargs

    def test_provider_without_secret(self):
        """Providers without secrets (e.g. Ollama) should omit api_key_secret_name."""
        client = self._make_client()
        _create_provider(
            client,
            name="ollama",
            host_url="http://localhost:11434/v1",
            secret_name=None,
            workspace="default",
        )
        call_kwargs = client.inference.providers.create.call_args.kwargs
        assert "api_key_secret_name" not in call_kwargs


# ---------------------------------------------------------------------------
# Auto setup
# ---------------------------------------------------------------------------

_VALID_KEY_RESULT = KeyValidationResult(passed=True, message="")


@pytest.fixture(autouse=False)
def _bypass_key_validation():
    """Stub out API key validation so tests focused on registration logic are not affected."""
    with patch(
        "nemo_platform_ext.cli.commands.setup._validate_api_key",
        return_value=_VALID_KEY_RESULT,
    ):
        yield


@pytest.mark.usefixtures("_bypass_key_validation")
class TestAutoSetup:
    def test_no_env_vars_returns_none(self):
        client = _make_mock_client()
        with patch.dict("os.environ", {}, clear=True):
            assert _auto_setup(client, "default") is None

    def test_openai_key_creates_provider(self):
        client = _make_mock_client()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test123"}, clear=True):
            result = _auto_setup(client, "default")
        assert result == "openai"
        client.mock_secrets.create_secret.assert_called_once()
        client.inference.providers.create.assert_called_once()
        create_kwargs = client.inference.providers.create.call_args
        assert create_kwargs.kwargs["name"] == "openai"
        assert create_kwargs.kwargs["host_url"] == "https://api.openai.com/v1"

    def test_nvidia_key_creates_build_provider(self):
        client = _make_mock_client()
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "nvapi-test"}, clear=True):
            result = _auto_setup(client, "default")
        assert result == "nvidia-build"
        create_kwargs = client.inference.providers.create.call_args
        assert create_kwargs.kwargs["name"] == "nvidia-build"

    def test_nemo_default_key_with_url(self):
        client = _make_mock_client()
        env = {
            "NEMO_DEFAULT_INFERENCE_KEY": "my-key",
            "NEMO_DEFAULT_INFERENCE_BASE_URL": "https://api.openai.com/v1",
        }
        with patch.dict("os.environ", env, clear=True):
            result = _auto_setup(client, "default")
        assert result == "openai"
        create_kwargs = client.inference.providers.create.call_args
        assert create_kwargs.kwargs["name"] == "openai"

    def test_nemo_default_key_with_custom_url(self):
        client = _make_mock_client()
        env = {
            "NEMO_DEFAULT_INFERENCE_KEY": "my-key",
            "NEMO_DEFAULT_INFERENCE_BASE_URL": "https://my-custom-llm.example.com/v1",
        }
        with patch.dict("os.environ", env, clear=True):
            result = _auto_setup(client, "default")
        assert result == "my-custom-llm-example-com"
        create_kwargs = client.inference.providers.create.call_args
        assert create_kwargs.kwargs["name"] == "my-custom-llm-example-com"
        assert create_kwargs.kwargs["host_url"] == "https://my-custom-llm.example.com/v1"

    def test_existing_provider_updated_not_recreated(self):
        client = _make_mock_client(provider_exists=True, secret_exists=True)
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test123"}, clear=True):
            result = _auto_setup(client, "default")
        assert result == "openai"
        client.inference.providers.create.assert_not_called()
        client.inference.providers.update.assert_called_once()

    def test_priority_order(self):
        """NEMO_DEFAULT_INFERENCE_KEY takes priority over OPENAI_API_KEY."""
        client = _make_mock_client()
        env = {
            "NEMO_DEFAULT_INFERENCE_KEY": "nemo-key",
            "NEMO_DEFAULT_INFERENCE_BASE_URL": "https://api.anthropic.com",
            "OPENAI_API_KEY": "sk-test123",
        }
        with patch.dict("os.environ", env, clear=True):
            result = _auto_setup(client, "default")
        assert result == "anthropic"
        create_kwargs = client.inference.providers.create.call_args
        assert create_kwargs.kwargs["name"] == "anthropic"

    def test_anthropic_auto_setup_maps_auth_header(self):
        """Auto-setup persists the Anthropic auth template without exposing the key."""
        client = _make_mock_client()
        api_key = "sk-ant-test"
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": api_key}, clear=True):
            result = _auto_setup(client, "default")
        assert result == "anthropic"
        call_kwargs = client.inference.providers.create.call_args.kwargs
        assert call_kwargs["name"] == "anthropic"
        assert call_kwargs["api_key_secret_name"] == "anthropic-api-key"
        assert call_kwargs["auth_header_format"] == "X-Api-Key: {{ auth_secret }}"
        assert "required_extra_headers" not in call_kwargs
        assert api_key not in str(call_kwargs)


# ---------------------------------------------------------------------------
# Config model tests
# ---------------------------------------------------------------------------


class TestDefaultModelConfig:
    def test_context_definition_has_default_model(self):
        ctx = ContextDefinition(name="test", cluster="c", user="u")
        assert ctx.default_model is None

        ctx2 = ContextDefinition(name="test", cluster="c", user="u", default_model="my-model")
        assert ctx2.default_model == "my-model"

    def test_context_definition_has_optional_fast_model(self):
        ctx = ContextDefinition(name="test", cluster="c", user="u")
        assert ctx.fast_model is None

        configured = ContextDefinition(
            name="test",
            cluster="c",
            user="u",
            fast_model="workspace/fast",
        )
        assert configured.fast_model == "workspace/fast"

    def test_config_params_accepts_default_model(self):
        params: ConfigParams = {
            "default_model": "workspace/openai-gpt-4o",
            "fast_model": "workspace/openai-gpt-4o-mini",
        }
        assert params["default_model"] == "workspace/openai-gpt-4o"
        assert params["fast_model"] == "workspace/openai-gpt-4o-mini"

    def test_ensure_context_applies_default_model(self):
        config_file = ConfigFile(
            current_context="test",
            clusters=[{"name": "test-cluster", "base_url": "http://localhost:8080"}],
            users=[{"name": "test-user", "type": "no-auth"}],
            contexts=[{"name": "test", "cluster": "test-cluster", "user": "test-user"}],
        )
        params: ConfigParams = {
            "default_model": "workspace/my-model",
            "fast_model": "workspace/fast",
        }
        _, _, ctx_def = config_file.ensure_context("test", params)
        assert ctx_def.default_model == "workspace/my-model"
        assert ctx_def.fast_model == "workspace/fast"

    def test_context_has_default_model(self):
        ctx = Context(
            context_name="test",
            cluster=Cluster(name="c", base_url="http://localhost:8080"),
            workspace="default",
            default_model="workspace/gpt-4o",
            fast_model="workspace/gpt-4o-mini",
            preferences={},
        )
        assert ctx.default_model == "workspace/gpt-4o"
        assert ctx.fast_model == "workspace/gpt-4o-mini"

    def test_default_model_fills_missing_fast_role(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NEMO_DEFAULT_MODEL", raising=False)
        monkeypatch.delenv("NEMO_FAST_MODEL", raising=False)
        config_file = ConfigFile(
            current_context="test",
            clusters=[{"name": "test-cluster", "base_url": "http://localhost:8080"}],
            users=[{"name": "test-user", "type": "no-auth"}],
            contexts=[
                {
                    "name": "test",
                    "cluster": "test-cluster",
                    "user": "test-user",
                    "default_model": "workspace/legacy-model",
                }
            ],
        )

        context = Config.create(tmp_path / "config.yaml", config_file).resolve()

        assert context.default_model == "workspace/legacy-model"
        assert context.fast_model == "workspace/legacy-model"

    def test_explicit_model_pair_resolves_independently(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NEMO_DEFAULT_MODEL", raising=False)
        monkeypatch.delenv("NEMO_FAST_MODEL", raising=False)
        config_file = ConfigFile(
            current_context="test",
            clusters=[{"name": "test-cluster", "base_url": "http://localhost:8080"}],
            users=[{"name": "test-user", "type": "no-auth"}],
            contexts=[
                {
                    "name": "test",
                    "cluster": "test-cluster",
                    "user": "test-user",
                    "default_model": "workspace/default",
                    "fast_model": "workspace/fast",
                }
            ],
        )

        context = Config.create(tmp_path / "config.yaml", config_file).resolve()

        assert context.default_model == "workspace/default"
        assert context.fast_model == "workspace/fast"

    def test_model_environment_overrides_resolve_independently(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NEMO_DEFAULT_MODEL", "workspace/env-default")
        monkeypatch.setenv("NEMO_FAST_MODEL", "workspace/env-fast")
        config_file = ConfigFile(
            current_context="test",
            clusters=[{"name": "test-cluster", "base_url": "http://localhost:8080"}],
            users=[{"name": "test-user", "type": "no-auth"}],
            contexts=[
                {
                    "name": "test",
                    "cluster": "test-cluster",
                    "user": "test-user",
                    "default_model": "workspace/config-default",
                    "fast_model": "workspace/config-fast",
                }
            ],
        )

        context = Config.create(tmp_path / "config.yaml", config_file).resolve()

        assert context.default_model == "workspace/env-default"
        assert context.fast_model == "workspace/env-fast"


# ---------------------------------------------------------------------------
# Service startup helpers
# ---------------------------------------------------------------------------


class TestKillExistingServices:
    """``_kill_existing_services`` delegates to ``_process.stop_instance``."""

    def test_delegates_to_stop_instance(self):
        with patch(f"{SETUP_MOD}.stop_instance") as mock_stop:
            mock_stop.return_value = MagicMock(stopped_pids=[])
            _kill_existing_services("http://localhost:8080")
        mock_stop.assert_called_once()
        _, kwargs = mock_stop.call_args
        assert kwargs["timeout"] == 2.0
        assert kwargs["force"] is True


@pytest.fixture
def maybe_start_preflight_mocks():
    """Shared mocks for ``_maybe_start_services`` preflight tests."""
    with (
        patch(f"{SETUP_MOD}._check_platform_reachable", return_value=False),
        patch(f"{SETUP_MOD}.importlib.util.find_spec", return_value=MagicMock()),
        patch(f"{SETUP_MOD}._start_services_background") as mock_start,
        patch(f"{SETUP_MOD}.prompt_choice", return_value="yes"),
        patch(f"{SETUP_MOD}._prompt_data_dir", return_value="/tmp/data"),
        # Docker preflight is covered separately; keep other start-path tests focused.
        patch(f"{SETUP_MOD}.require_docker_for_default_local"),
    ):
        yield mock_start


class TestMaybeStartServices:
    def test_skips_when_running_and_start_not_requested(self):
        with patch(f"{SETUP_MOD}._check_platform_reachable", return_value=True):
            _maybe_start_services("http://localhost:8080", auto=False, start_services=None)

    def test_skips_when_running_and_explicitly_false(self):
        with patch(f"{SETUP_MOD}._check_platform_reachable", return_value=True):
            _maybe_start_services("http://localhost:8080", auto=False, start_services=False)

    def test_returns_remote_choice_without_starting_services(self):
        with (
            patch(f"{SETUP_MOD}._check_platform_reachable", return_value=False),
            patch(f"{SETUP_MOD}.prompt_choice", return_value="remote") as mock_prompt,
            patch(f"{SETUP_MOD}._start_services_background") as mock_start,
        ):
            result = _maybe_start_services("http://localhost:8080", auto=False, start_services=None)

        assert result == "connect_remote"
        assert mock_prompt.call_args.kwargs["options"] == [
            ("yes", "Yes, start services now"),
            ("remote", "No, I want to connect to a remote Platform instance"),
            ("manual", "No, I'll start them myself"),
        ]
        mock_start.assert_not_called()

    def test_start_myself_choice_keeps_existing_exit(self, capsys):
        with (
            patch(f"{SETUP_MOD}._check_platform_reachable", return_value=False),
            patch(f"{SETUP_MOD}.prompt_choice", return_value="manual"),
            pytest.raises(ClickExit),
        ):
            _maybe_start_services("http://localhost:8080", auto=False, start_services=None)

        assert "Start the platform first" in capsys.readouterr().err

    def test_unreachable_remote_url_selects_remote_connection_without_local_prompt(self):
        with (
            patch(f"{SETUP_MOD}._check_platform_reachable", return_value=False),
            patch(f"{SETUP_MOD}.prompt_choice") as mock_prompt,
        ):
            result = _maybe_start_services("https://remote.example.com", auto=False, start_services=None)

        assert result == "connect_remote"
        mock_prompt.assert_not_called()

    def test_reachable_remote_prompts_continue(self):
        with (
            patch(f"{SETUP_MOD}._check_platform_reachable", return_value=True),
            patch(f"{SETUP_MOD}.prompt_choice", return_value="continue") as mock_prompt,
        ):
            result = _maybe_start_services("https://remote.example.com", auto=False, start_services=None)

        assert result == "ready"
        mock_prompt.assert_called_once()
        assert (
            mock_prompt.call_args.kwargs["message"]
            == "Platform reachable at remote.example.com (https://remote.example.com). What would you like to do?"
        )

    def test_reachable_remote_start_local(self):
        with (
            patch(f"{SETUP_MOD}._check_platform_reachable", return_value=True),
            patch(f"{SETUP_MOD}.prompt_choice", return_value="local"),
        ):
            result = _maybe_start_services("https://remote.example.com", auto=False, start_services=None)

        assert result == "start_local"

    def test_reachable_remote_change_remote(self):
        with (
            patch(f"{SETUP_MOD}._check_platform_reachable", return_value=True),
            patch(f"{SETUP_MOD}.prompt_choice", return_value="change"),
        ):
            result = _maybe_start_services("https://remote.example.com", auto=False, start_services=None)

        assert result == "connect_remote"

    def test_reachable_remote_auto_skips_prompt(self):
        with (
            patch(f"{SETUP_MOD}._check_platform_reachable", return_value=True),
            patch(f"{SETUP_MOD}.prompt_choice") as mock_prompt,
        ):
            result = _maybe_start_services("https://remote.example.com", auto=True, start_services=None)

        assert result == "ready"
        mock_prompt.assert_not_called()

    def test_rejects_start_services_for_remote_url(self):
        with (
            patch(f"{SETUP_MOD}._check_platform_reachable", return_value=False),
            patch(f"{SETUP_MOD}._start_services_background") as mock_start,
            pytest.raises(typer.BadParameter, match="local Platform URL"),
        ):
            _maybe_start_services("https://remote.example.com", auto=False, start_services=True)

        mock_start.assert_not_called()

    def test_restarts_when_running_and_start_services_true(self):
        reachable_calls = [True, True, False, True]

        with (
            patch(f"{SETUP_MOD}._check_platform_reachable", side_effect=reachable_calls),
            patch(f"{SETUP_MOD}._kill_existing_services") as mock_kill,
            patch(f"{SETUP_MOD}._start_services_background") as mock_start,
            patch(f"{SETUP_MOD}._wait_for_platform", return_value=True),
            patch(f"{SETUP_MOD}._prompt_data_dir", return_value="/tmp/test-data") as mock_db_prompt,
            patch(f"{SETUP_MOD}.check_port_available_for_start", return_value=None),
            patch(f"{SETUP_MOD}._ensure_port_available_for_start", wraps=_ensure_port_available_for_start) as mock_port,
            patch(f"{SETUP_MOD}.require_docker_for_default_local"),
            patch(f"{SETUP_MOD}._pause"),
        ):
            mock_start.return_value = MagicMock(pid=999)
            _maybe_start_services("http://localhost:8080", auto=False, start_services=True)
        mock_kill.assert_called_once()
        mock_port.assert_called_once()
        mock_start.assert_called_once_with("http://localhost:8080", data_dir="/tmp/test-data")
        mock_db_prompt.assert_called_once()

    def test_restarts_exits_when_port_still_occupied_after_kill(self, capsys):
        """Port preflight runs after kill/wait and blocks spawn when the port stays busy."""
        reachable_calls = [True, True, False, True]
        conflict = PortConflict(kind="foreign", port=8080)

        with (
            patch(f"{SETUP_MOD}._check_platform_reachable", side_effect=reachable_calls),
            patch(f"{SETUP_MOD}._kill_existing_services") as mock_kill,
            patch(f"{SETUP_MOD}._start_services_background") as mock_start,
            patch(f"{SETUP_MOD}.check_port_available_for_start", return_value=conflict),
            patch(f"{SETUP_MOD}._prompt_data_dir", return_value="/tmp/test-data"),
            patch(f"{SETUP_MOD}.require_docker_for_default_local"),
            patch(f"{SETUP_MOD}._pause"),
            pytest.raises(ClickExit),
        ):
            _maybe_start_services("http://localhost:8080", auto=False, start_services=True)
        mock_kill.assert_called_once()
        mock_start.assert_not_called()
        captured = capsys.readouterr()
        assert "already in use" in captured.err
        assert "services.log" not in captured.err

    def test_exits_early_when_services_extra_missing(self, maybe_start_preflight_mocks, capsys):
        """Preflight aborts with install hint before spawning a subprocess."""
        with (
            patch(f"{SETUP_MOD}.importlib.util.find_spec", return_value=None),
            pytest.raises(ClickExit),
        ):
            _maybe_start_services("http://localhost:8080", auto=False, start_services=True)
        maybe_start_preflight_mocks.assert_not_called()
        captured = capsys.readouterr()
        assert "nemo-platform[all]" in captured.err

    def test_exits_early_when_port_occupied(self, maybe_start_preflight_mocks, capsys):
        """Preflight aborts with port hint on stderr before spawning a subprocess."""
        conflict = PortConflict(kind="foreign", port=8080)
        with (
            patch(f"{SETUP_MOD}.check_port_available_for_start", return_value=conflict),
            pytest.raises(ClickExit),
        ):
            _maybe_start_services("http://localhost:8080", auto=False, start_services=True)
        maybe_start_preflight_mocks.assert_not_called()
        captured = capsys.readouterr()
        assert "already in use" in captured.err
        assert "EADDRINUSE" in captured.err
        assert "lsof" in captured.err
        assert "services.log" not in captured.err

    def test_early_exit_names_eaddrinuse_when_startup_log_has_bind_failure(
        self, maybe_start_preflight_mocks, capsys, tmp_path
    ):
        """Post-spawn bind failures should not collapse to a generic early-exit message."""
        dead = MagicMock(pid=999)
        dead.poll.return_value = 1
        log = tmp_path / "services.log"
        log.write_text("OSError: [Errno 98] Address already in use\n", encoding="utf-8")
        with (
            patch(f"{SETUP_MOD}.check_port_available_for_start", return_value=None),
            patch(f"{SETUP_MOD}._wait_for_platform", return_value=False),
            patch(f"{SETUP_MOD}.log_path_for", return_value=log),
            patch(f"{SETUP_MOD}.probe_docker", return_value=MagicMock(available=True)),
            patch(f"{SETUP_MOD}._pause"),
            pytest.raises(ClickExit),
        ):
            maybe_start_preflight_mocks.return_value = dead
            _maybe_start_services("http://localhost:8080", auto=False, start_services=True)
        captured = capsys.readouterr()
        assert "Port 8080" in captured.err
        assert "EADDRINUSE" in captured.err
        assert "lsof -i :8080" in captured.err
        assert "Service process exited early" not in captured.err

    def test_allows_start_when_port_free(self, maybe_start_preflight_mocks):
        with (
            patch(f"{SETUP_MOD}.check_port_available_for_start", return_value=None),
            patch(f"{SETUP_MOD}._wait_for_platform", return_value=True),
            patch(f"{SETUP_MOD}._pause"),
        ):
            maybe_start_preflight_mocks.return_value = MagicMock(pid=999)
            _maybe_start_services("http://localhost:8080", auto=False, start_services=True)
        maybe_start_preflight_mocks.assert_called_once()

    def test_early_exit_prints_docker_hint_when_daemon_unavailable(self, maybe_start_preflight_mocks, capsys):
        dead = MagicMock(pid=999)
        dead.poll.return_value = 3
        wait = MagicMock(return_value=False)
        with (
            patch(f"{SETUP_MOD}.check_port_available_for_start", return_value=None),
            patch(f"{SETUP_MOD}._wait_for_platform", wait),
            patch(f"{SETUP_MOD}.probe_docker", return_value=MagicMock(available=False)),
            patch(f"{SETUP_MOD}.log_path_for", return_value=MagicMock(__str__=lambda self: "/tmp/services.log")),
            patch(f"{SETUP_MOD}._pause"),
            pytest.raises(ClickExit),
        ):
            maybe_start_preflight_mocks.return_value = dead
            _maybe_start_services("http://localhost:8080", auto=False, start_services=True)
        wait.assert_called_once()
        assert wait.call_args.kwargs.get("proc") is dead
        captured = capsys.readouterr()
        assert "exited early (exit code 3)" in captured.err
        assert "Check /tmp/services.log for details." in captured.err
        assert "Docker is required for this default local setup" in captured.err

    def test_docker_preflight_blocks_before_spawn(self, maybe_start_preflight_mocks, capsys):
        """Default-local Docker gate exits before start_background (NVBug 6537617)."""
        with (
            patch(f"{SETUP_MOD}.check_port_available_for_start", return_value=None),
            patch(
                f"{SETUP_MOD}.require_docker_for_default_local",
                side_effect=typer.Exit(1),
            ),
            pytest.raises(ClickExit),
        ):
            _maybe_start_services("http://localhost:8080", auto=False, start_services=True)
        maybe_start_preflight_mocks.assert_not_called()

    def test_readiness_timeout_does_not_hint_docker_without_evidence(
        self, maybe_start_preflight_mocks, capsys, tmp_path
    ):
        alive = MagicMock(pid=999)
        alive.poll.return_value = None
        log = tmp_path / "services.log"
        log.write_text("starting auth\n", encoding="utf-8")
        with (
            patch(f"{SETUP_MOD}.check_port_available_for_start", return_value=None),
            patch(f"{SETUP_MOD}._wait_for_platform", return_value=False),
            patch(f"{SETUP_MOD}.probe_docker", return_value=MagicMock(available=False)),
            patch(f"{SETUP_MOD}.log_path_for", return_value=log),
            patch(f"{SETUP_MOD}._pause"),
            pytest.raises(ClickExit),
        ):
            maybe_start_preflight_mocks.return_value = alive
            _maybe_start_services("http://localhost:8080", auto=False, start_services=True)
        captured = capsys.readouterr()
        assert "did not become ready" in captured.err
        assert "Docker does not appear to be available" not in captured.err

    def test_docker_hint_from_services_log_markers(self, maybe_start_preflight_mocks, capsys, tmp_path):
        alive = MagicMock(pid=999)
        alive.poll.return_value = None
        log = tmp_path / "services.log"
        log.write_text("Skipping executor 'local-docker': Docker daemon is unavailable\n", encoding="utf-8")
        with (
            patch(f"{SETUP_MOD}.check_port_available_for_start", return_value=None),
            patch(f"{SETUP_MOD}._wait_for_platform", return_value=False),
            patch(f"{SETUP_MOD}.probe_docker", return_value=MagicMock(available=True)),
            patch(f"{SETUP_MOD}.log_path_for", return_value=log),
            patch(f"{SETUP_MOD}._pause"),
            pytest.raises(ClickExit),
        ):
            maybe_start_preflight_mocks.return_value = alive
            _maybe_start_services("http://localhost:8080", auto=False, start_services=True)
        captured = capsys.readouterr()
        assert "Docker is required for this default local setup" in captured.err

    def test_startup_port_conflict_prefers_live_port_probe(self, tmp_path):
        log = tmp_path / "services.log"
        log.write_text("", encoding="utf-8")
        conflict = PortConflict(kind="foreign", port=9090)
        with patch(f"{SETUP_MOD}.check_port_available_for_start", return_value=conflict):
            assert _detect_startup_port_conflict("http://localhost:9090", log) is conflict

    def test_startup_port_conflict_detects_log_marker(self, tmp_path):
        log = tmp_path / "services.log"
        log.write_text("RuntimeError: EADDRINUSE while binding\n", encoding="utf-8")
        with patch(f"{SETUP_MOD}.check_port_available_for_start", return_value=None):
            conflict = _detect_startup_port_conflict("http://localhost:9090", log)
        assert conflict == PortConflict(kind="foreign", port=9090)

    def test_services_log_suggests_port_conflict(self, tmp_path):
        log = tmp_path / "services.log"
        log.write_text("uvicorn failed: address already in use\n", encoding="utf-8")
        assert _services_log_suggests_port_conflict(log) is True


class TestRemoteConnection:
    def test_prompts_again_until_platform_is_reachable(self, capsys):
        with (
            patch(
                f"{SETUP_MOD}.prompt_text",
                side_effect=["https://unreachable.example.com", "https://remote.example.com/"],
            ),
            patch(
                f"{SETUP_MOD}._check_platform_reachable_with_retries",
                side_effect=[False, True],
            ),
        ):
            base_url = setup_commands._prompt_remote_base_url()

        assert base_url == "https://remote.example.com"
        assert "Unable to connect" in capsys.readouterr().err

    def test_persists_remote_url_and_workspace_in_active_context(self, tmp_path, monkeypatch):
        config_path = tmp_path / "config.yaml"
        monkeypatch.setenv("NMP_CONFIG_FILE", str(config_path))
        cli_context = MagicMock()
        cli_context.overrides = {}
        cli_context.get_sdk_context.return_value = Context(
            context_name="default",
            cluster=Cluster(name="default-cluster", base_url="http://localhost:8080"),
            user=NoAuthUser(name="default-user"),
            workspace="default",
            preferences={},
        )

        setup_commands._configure_remote_connection(
            cli_context,
            "https://remote.example.com",
            "shared-workspace",
        )

        context = Config.load(config_path=config_path).resolve()
        assert str(context.cluster.base_url) == "https://remote.example.com/"
        assert context.workspace == "shared-workspace"
        cli_context.reset_sdk_context.assert_called_once_with()

    def test_updates_selected_context_and_runtime_url_override(self, tmp_path, monkeypatch):
        config_path = tmp_path / "config.yaml"
        monkeypatch.setenv("NMP_CONFIG_FILE", str(config_path))
        Config.write({"base_url": "https://default.example.com"})
        Config.write({"base_url": "https://old-dev.example.com"}, context_name="dev")

        cli_context = MagicMock()
        cli_context.overrides = {
            "current_context": "dev",
            "base_url": "https://stale-override.example.com",
        }
        cli_context.get_sdk_context.return_value = Context(
            context_name="dev",
            cluster=Cluster(name="dev-cluster", base_url="https://stale-override.example.com"),
            user=NoAuthUser(name="dev-user"),
            workspace="default",
            preferences={},
        )

        setup_commands._configure_remote_connection(
            cli_context,
            "https://new-dev.example.com",
            "shared-workspace",
        )

        config_file = Config.load(config_path=config_path).get_config_file()
        default_cluster = next(cluster for cluster in config_file.clusters if cluster.name == "default-cluster")
        dev_cluster = next(cluster for cluster in config_file.clusters if cluster.name == "dev-cluster")
        assert str(default_cluster.base_url) == "https://default.example.com/"
        assert str(dev_cluster.base_url) == "https://new-dev.example.com/"
        assert cli_context.overrides["base_url"] == "https://new-dev.example.com"

    def test_local_connection_preserves_remote_context_and_uses_no_auth(self, tmp_path, monkeypatch):
        config_path = tmp_path / "config.yaml"
        monkeypatch.setenv("NMP_CONFIG_FILE", str(config_path))
        Config.write(
            {
                "base_url": "https://remote.example.com",
                "access_token": "remote-token",
                "refresh_token": "remote-refresh-token",
                "current_context": "dev",
            },
            context_name="dev",
        )
        cli_context = MagicMock()
        cli_context.overrides = {
            "current_context": "dev",
            "base_url": "https://remote.example.com",
        }

        _configure_local_connection(cli_context, "local-workspace")

        config = Config.load(config_path=config_path)
        config_file = config.get_config_file()
        assert config_file.current_context == "local"
        remote = next(context for context in config_file.contexts if context.name == "dev")
        remote_cluster = next(cluster for cluster in config_file.clusters if cluster.name == remote.cluster)
        remote_user = next(user for user in config_file.users if user.name == remote.user)
        assert str(remote_cluster.base_url) == "https://remote.example.com/"
        assert isinstance(remote_user, OAuthUser)

        local = config.resolve()
        assert local.context_name == "local"
        assert str(local.cluster.base_url) == "http://localhost:8080/"
        assert local.workspace == "local-workspace"
        assert isinstance(local.user, NoAuthUser)
        assert cli_context.overrides == {
            "current_context": "local",
            "base_url": "http://localhost:8080",
        }
        cli_context.reset_sdk_context.assert_called_once_with()

    def test_local_connection_does_not_overwrite_existing_local_context(self, tmp_path, monkeypatch):
        config_path = tmp_path / "config.yaml"
        monkeypatch.setenv("NMP_CONFIG_FILE", str(config_path))
        Config.write(
            {
                "base_url": "https://remote.example.com",
                "access_token": "remote-token",
                "current_context": "local",
            },
            context_name="local",
        )
        cli_context = MagicMock()
        cli_context.overrides = {}

        _configure_local_connection(cli_context, "local-workspace")

        config_file = Config.load(config_path=config_path).get_config_file()
        original = next(context for context in config_file.contexts if context.name == "local")
        original_cluster = next(cluster for cluster in config_file.clusters if cluster.name == original.cluster)
        assert str(original_cluster.base_url) == "https://remote.example.com/"
        assert config_file.current_context == "local-2"
        assert cli_context.overrides == {
            "current_context": "local-2",
            "base_url": "http://localhost:8080",
        }

    def test_preserves_active_workspace_when_flag_not_explicit(self):
        ctx = MagicMock(spec=typer.Context)
        ctx.get_parameter_source.return_value = ParameterSource.DEFAULT
        cli_context = MagicMock()
        cli_context.get_sdk_context.return_value = Context(
            context_name="default",
            cluster=Cluster(name="default-cluster", base_url="http://localhost:8080"),
            user=NoAuthUser(name="default-user"),
            workspace="team-a",
            preferences={},
        )

        assert _resolve_setup_workspace(ctx, cli_context, "default") == "team-a"
        ctx.get_parameter_source.assert_called_once_with("workspace")

    def test_uses_explicit_workspace_flag_over_active_context(self):
        ctx = MagicMock(spec=typer.Context)
        ctx.get_parameter_source.return_value = ParameterSource.COMMANDLINE
        cli_context = MagicMock()
        cli_context.get_sdk_context.return_value = Context(
            context_name="default",
            cluster=Cluster(name="default-cluster", base_url="http://localhost:8080"),
            user=NoAuthUser(name="default-user"),
            workspace="team-a",
            preferences={},
        )

        assert _resolve_setup_workspace(ctx, cli_context, "shared-workspace") == "shared-workspace"

    def test_authenticates_when_context_has_no_credentials(self):
        cli_context = MagicMock()
        cli_context.get_sdk_context.return_value = Context(
            context_name="default",
            cluster=Cluster(name="remote", base_url="https://remote.example.com"),
            user=NoAuthUser(name="default-user"),
            workspace="default",
            preferences={},
        )

        with patch("nemo_platform_ext.cli.commands.auth._login_with_oidc", return_value=True) as mock_login:
            setup_commands._ensure_platform_auth(cli_context)

        mock_login.assert_called_once_with(cli_context, selected_context="default")
        cli_context.reset_sdk_context.assert_called_once_with()

    def test_reauthenticates_stored_context_credentials_for_new_remote(self):
        cli_context = MagicMock()
        context = Context(
            context_name="default",
            cluster=Cluster(name="remote", base_url="https://remote.example.com"),
            user=OAuthUser(name="default-user", token=SecretStr("token")),
            workspace="default",
            preferences={},
        )
        cli_context.get_sdk_context.return_value = context

        with patch("nemo_platform_ext.cli.commands.auth._login_with_oidc", return_value=True) as mock_login:
            setup_commands._ensure_platform_auth(cli_context)

        mock_login.assert_called_once_with(cli_context, selected_context="default")
        cli_context.reset_sdk_context.assert_called_once_with()

    def test_reuses_runtime_access_token_override(self, monkeypatch):
        monkeypatch.setenv("NMP_ACCESS_TOKEN", "runtime-token")
        cli_context = MagicMock()
        cli_context.get_sdk_context.return_value = Context(
            context_name="default",
            cluster=Cluster(name="remote", base_url="https://remote.example.com"),
            user=OAuthUser(name="default-user", token=SecretStr("runtime-token")),
            workspace="default",
            preferences={},
        )

        with patch("nemo_platform_ext.cli.commands.auth._login_with_oidc") as mock_login:
            setup_commands._ensure_platform_auth(cli_context)

        mock_login.assert_not_called()

    def test_clears_stale_credentials_when_remote_auth_is_disabled(self, tmp_path, monkeypatch):
        config_path = tmp_path / "config.yaml"
        monkeypatch.setenv("NMP_CONFIG_FILE", str(config_path))
        monkeypatch.delenv("NMP_ACCESS_TOKEN", raising=False)
        Config.write(
            {
                "base_url": "https://remote.example.com",
                "access_token": "old-cluster-token",
            }
        )
        cli_context = MagicMock()
        cli_context.get_sdk_context.return_value = Config.load(config_path=config_path).resolve()

        with patch("nemo_platform_ext.cli.commands.auth._login_with_oidc", return_value=False):
            setup_commands._ensure_platform_auth(cli_context)

        assert isinstance(Config.load(config_path=config_path).resolve().user, NoAuthUser)
        cli_context.reset_sdk_context.assert_called_once_with()

    def test_platform_request_headers_include_context_token(self):
        cli_context = MagicMock()
        cli_context.get_sdk_context.return_value = Context(
            context_name="default",
            cluster=Cluster(name="remote", base_url="https://remote.example.com"),
            user=OAuthUser(name="default-user", token=SecretStr("remote-token")),
            workspace="default",
            preferences={},
        )

        assert setup_commands._platform_request_headers(cli_context) == {"Authorization": "Bearer remote-token"}


class TestLocalDataDirHelpers:
    """Tests for the XDG-default data-dir helpers used by `nemo setup`."""

    def test_load_persisted_returns_none_when_no_config_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NMP_CONFIG_FILE", str(tmp_path / "missing.yaml"))
        assert _load_persisted_data_dir() is None

    def test_save_then_load_roundtrips_data_dir(self, tmp_path, monkeypatch):
        config_path = tmp_path / "config.yaml"
        monkeypatch.setenv("NMP_CONFIG_FILE", str(config_path))
        _save_data_dir("/custom/data/dir")
        assert _load_persisted_data_dir() == "/custom/data/dir"

    def test_start_services_background_forwards_data_dir(self):
        """The setup.py wrapper should pass *data_dir* through to the shared
        process lifecycle module unchanged. The actual ``NMP_DATA_DIR``
        environment injection lives in ``services._process`` and is covered
        by its own tests."""
        with patch(f"{SETUP_MOD}.start_background") as mock_start:
            mock_start.return_value = MagicMock(pid=42)
            _start_services_background("http://localhost:9090", data_dir="/chosen/data/dir")
        mock_start.assert_called_once()
        args, kwargs = mock_start.call_args
        config = args[0]
        assert kwargs["data_dir"] == "/chosen/data/dir"
        assert config.port == 9090

    def test_auto_mode_skips_prompt_and_uses_persisted(self, tmp_path, monkeypatch):
        """`--auto` must not prompt but should still honor any persisted data dir."""
        config_path = tmp_path / "config.yaml"
        monkeypatch.setenv("NMP_CONFIG_FILE", str(config_path))
        _save_data_dir("/persisted/from/setup")

        with (
            patch(f"{SETUP_MOD}._check_platform_reachable", return_value=False),
            patch(f"{SETUP_MOD}._kill_existing_services"),
            patch(f"{SETUP_MOD}.check_port_available_for_start", return_value=None),
            patch(f"{SETUP_MOD}._start_services_background") as mock_start,
            patch(f"{SETUP_MOD}._wait_for_platform", return_value=True),
            patch(f"{SETUP_MOD}._prompt_data_dir") as mock_prompt,
            patch(f"{SETUP_MOD}.require_docker_for_default_local"),
            patch(f"{SETUP_MOD}._pause"),
        ):
            mock_start.return_value = MagicMock(pid=999)
            _maybe_start_services("http://localhost:8080", auto=True, start_services=True)
        mock_prompt.assert_not_called()
        mock_start.assert_called_once_with("http://localhost:8080", data_dir="/persisted/from/setup")

    def test_bootstrap_seeds_cluster_when_only_data_dir_persisted(self, tmp_path, monkeypatch):
        """Reproduces the user-reported crash where ``_prompt_data_dir`` writes a
        partial config (just ``local_services.data_dir``) before bootstrap runs.

        ``_bootstrap_config_if_missing`` used to short-circuit on
        ``config_path.exists()``, leaving the file without any cluster. Later
        steps (`_save_model_pair` → ``Config.write`` → ``ensure_context``)
        then raised ``Cluster 'default-cluster' does not exist and no base_url
        provided to create it!`` when the user picked a default model.

        After the fix, bootstrap recognizes a partial config (clusters empty)
        and seeds the cluster + context while preserving the data dir.
        """
        config_path = tmp_path / "config.yaml"
        monkeypatch.setenv("NMP_CONFIG_FILE", str(config_path))

        # Simulate the partial-config state that ``_prompt_data_dir`` leaves
        # behind when run on a fresh install.
        _save_data_dir("/chosen/data/dir")
        assert config_path.exists()
        from nemo_platform_ext.config.config import Config

        before = Config.load(config_path=config_path).get_config_file()
        assert before.clusters == []  # exactly the user's broken state
        assert before.local_services is not None
        assert before.local_services.data_dir == "/chosen/data/dir"

        # Bootstrap should now seed the cluster (was a no-op pre-fix).
        _bootstrap_config_if_missing("http://localhost:8080", "default")

        after = Config.load(config_path=config_path).get_config_file()
        assert len(after.clusters) == 1
        assert str(after.clusters[0].base_url).rstrip("/") == "http://localhost:8080"
        # Data dir was preserved across the bootstrap write.
        assert after.local_services is not None
        assert after.local_services.data_dir == "/chosen/data/dir"

    def test_bootstrap_is_noop_when_cluster_already_seeded(self, tmp_path, monkeypatch):
        """Idempotency check: bootstrap should not rewrite an already-seeded config."""
        config_path = tmp_path / "config.yaml"
        monkeypatch.setenv("NMP_CONFIG_FILE", str(config_path))

        # First seed.
        _bootstrap_config_if_missing("http://localhost:8080", "default")
        mtime_before = config_path.stat().st_mtime_ns

        # Second call should be a no-op (no rewrite, same mtime).
        _bootstrap_config_if_missing("http://other:9090", "other")
        mtime_after = config_path.stat().st_mtime_ns
        assert mtime_after == mtime_before

        from nemo_platform_ext.config.config import Config

        after = Config.load(config_path=config_path).get_config_file()
        # Original cluster URL preserved (not overwritten by the second call).
        assert str(after.clusters[0].base_url).rstrip("/") == "http://localhost:8080"


class TestMaybeDeployAgentPluginCheck:
    def test_skips_without_prompting_when_plugin_missing(self):
        """When plugin is not available, user should never be prompted."""
        with (
            patch(f"{SETUP_MOD}._agents_plugin_available", return_value=False),
            patch(f"{SETUP_MOD}.prompt_choice") as mock_prompt,
        ):
            _maybe_deploy_agent("http://localhost:8080", "default", auto=False, deploy_agent=None)
        mock_prompt.assert_not_called()

    def test_prompts_when_plugin_available(self):
        with (
            patch(f"{SETUP_MOD}._agents_plugin_available", return_value=True),
            patch(f"{SETUP_MOD}.prompt_choice", return_value="no") as mock_prompt,
        ):
            _maybe_deploy_agent("http://localhost:8080", "default", auto=False, deploy_agent=None)
        mock_prompt.assert_called_once()


# ---------------------------------------------------------------------------
# Skills installation helpers
# ---------------------------------------------------------------------------


class TestDetectCodingAgents:
    def test_detects_cursor_directory(self, tmp_path):
        (tmp_path / ".cursor").mkdir()
        (tmp_path / ".git").mkdir()
        with patch("nemo_platform_ext.cli.commands.setup.Path.cwd", return_value=tmp_path):
            detected = _detect_coding_agents()
        agent_names = [name for _, name in detected]
        assert "cursor" in agent_names

    def test_detects_agents_md(self, tmp_path):
        (tmp_path / "AGENTS.md").touch()
        (tmp_path / ".git").mkdir()
        with patch("nemo_platform_ext.cli.commands.setup.Path.cwd", return_value=tmp_path):
            detected = _detect_coding_agents()
        agent_names = [name for _, name in detected]
        assert "codex" in agent_names

    def test_detects_multiple_agents(self, tmp_path):
        (tmp_path / "AGENTS.md").touch()
        (tmp_path / ".cursor").mkdir()
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".git").mkdir()
        with patch("nemo_platform_ext.cli.commands.setup.Path.cwd", return_value=tmp_path):
            detected = _detect_coding_agents()
        agent_names = [name for _, name in detected]
        assert "codex" in agent_names
        assert "cursor" in agent_names
        assert "claude" in agent_names

    def test_returns_empty_when_none_found(self, tmp_path):
        (tmp_path / ".git").mkdir()
        with patch("nemo_platform_ext.cli.commands.setup.Path.cwd", return_value=tmp_path):
            detected = _detect_coding_agents()
        assert detected == []


class TestAgentMarkers:
    def test_all_markers_have_agent_names(self):
        for marker, agent_name in _AGENT_MARKERS:
            assert marker
            assert agent_name


# ---------------------------------------------------------------------------
# Skills installation helpers
# ---------------------------------------------------------------------------


class TestParseCsvFlag:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, None),
            ("codex", ["codex"]),
            ("codex,cursor,claude", ["codex", "cursor", "claude"]),
            (" codex , cursor ", ["codex", "cursor"]),
            ("", None),
            (",,,", None),
        ],
    )
    def test_parse(self, raw, expected):
        assert _parse_csv_flag(raw) == expected


class TestFilterAgentsByScope:
    def test_project_scope_keeps_all_supported_agents(self):
        kept, skipped = _filter_agents_by_scope(["claude", "cursor", "codex"], Scope.PROJECT)
        assert set(kept) == {"claude", "cursor", "codex"}
        assert skipped == []

    def test_user_scope_skips_cursor(self):
        # Cursor only supports PROJECT scope; should be filtered out at USER scope.
        kept, skipped = _filter_agents_by_scope(["claude", "cursor", "codex"], Scope.USER)
        assert "cursor" not in kept
        assert {"claude", "codex"}.issubset(set(kept))
        assert any(name == "cursor" for name, _ in skipped)
        cursor_reason = next(reason for name, reason in skipped if name == "cursor")
        assert "user" in cursor_reason

    def test_user_scope_keeps_codex_and_claude(self):
        kept, _ = _filter_agents_by_scope(["claude", "codex"], Scope.USER)
        assert set(kept) == {"claude", "codex"}


class TestLoadSkillsWithWarnings:
    def test_captures_plugin_warnings(self):
        # Stub load_skills so it emits a warning via the registry logger; the
        # capture handler should pick it up and return it as a structured message.
        def _stub() -> dict:
            skills_registry.logger.warning("Skipping plugin 'fake': boom")
            return {}

        with patch("nemo_platform_ext.cli.commands.setup.load_skills", side_effect=_stub):
            skills, warnings = _load_skills_with_warnings()
        assert skills == {}
        assert any("Skipping plugin 'fake'" in msg for msg in warnings)

    def test_handler_cleaned_up_on_exception(self):
        handler_count_before = len(skills_registry.logger.handlers)

        def _boom() -> dict:
            raise RuntimeError("kaboom")

        with patch("nemo_platform_ext.cli.commands.setup.load_skills", side_effect=_boom):
            with pytest.raises(RuntimeError):
                _load_skills_with_warnings()
        # Even on exception, the capture handler is removed.
        assert len(skills_registry.logger.handlers) == handler_count_before

    def test_warnings_captured_even_when_logger_level_above_warning(self):
        """If an upstream caller raised the registry logger to ERROR, plugin
        warnings should still reach the preview."""

        def _stub() -> dict:
            skills_registry.logger.warning("Skipping plugin 'quiet': nothing here")
            return {}

        original = skills_registry.logger.level
        skills_registry.logger.setLevel(logging.ERROR)
        try:
            with patch("nemo_platform_ext.cli.commands.setup.load_skills", side_effect=_stub):
                _, warnings = _load_skills_with_warnings()
            assert any("Skipping plugin 'quiet'" in m for m in warnings)
            # Logger level is restored to ERROR after the call.
            assert skills_registry.logger.level == logging.ERROR
        finally:
            skills_registry.logger.setLevel(original)


class TestMaybeInstallSkills:
    @staticmethod
    def _skill(name: str, plugin: str | None = None) -> Skill:
        return Skill(
            name=name,
            description=f"{name} desc",
            version="0.1",
            content=f"# {name}",
            raw=f"---\nname: {name}\n---\n# {name}",
            source_plugin=plugin,
        )

    def _patched_install(self, tmp_path, **call_kwargs):
        """Run _maybe_install_skills with detection and project root pinned, capturing installer calls."""
        # alpha is built-in (source_plugin=None → "nemo-platform"); beta comes
        # from a plugin so source filtering can be exercised independently.
        skills = {
            "alpha": self._skill("alpha"),
            "beta": self._skill("beta", plugin="example-plugin"),
        }

        recorded_calls: list[tuple[str, Scope, dict]] = []

        class _StubInstaller:
            def __init__(self, name: str, scopes: list[Scope]):
                self.name = name
                self.supported_scopes = scopes

            def install(self, scope, project_root, sk):
                recorded_calls.append((self.name, scope, sk))
                return []

            def get_install_path(self, scope, project_root, skill_name):
                return project_root / "stub" / skill_name

        installers = {
            "codex": _StubInstaller("codex", [Scope.PROJECT, Scope.USER]),
            "cursor": _StubInstaller("cursor", [Scope.PROJECT]),
            "claude": _StubInstaller("claude", [Scope.PROJECT, Scope.USER]),
        }

        with (
            patch(
                "nemo_platform_ext.cli.commands.setup._detect_coding_agents",
                return_value=[("AGENTS.md", "codex"), (".cursor", "cursor"), (".claude", "claude")],
            ),
            patch(
                "nemo_platform_ext.cli.commands.setup._load_skills_with_warnings",
                return_value=(skills, []),
            ),
            patch(
                "nemo_platform_ext.cli.commands.setup._find_project_root",
                return_value=tmp_path,
            ),
            patch(
                "nemo_platform_ext.cli.commands.setup.get_installer",
                side_effect=lambda name: installers[name],
            ),
        ):
            _maybe_install_skills(**call_kwargs)
        return recorded_calls

    @pytest.mark.parametrize(
        "call_kwargs",
        [
            # --install-skills=False is an explicit opt-out
            {"install_skills": False},
            # --install-skills=None (unset) is a noop without an explicit opt-in
            {"install_skills": None},
            # Filter flags alone don't opt in; --install-skills is the master switch.
            {"install_skills": None, "skills_scope": Scope.USER},
            {"install_skills": None, "skills_agents": ["codex"]},
            {"install_skills": None, "skills_from": ["nemo-platform"]},
        ],
        ids=[
            "install_skills_false",
            "install_skills_none",
            "skills_scope_only",
            "skills_agents_only",
            "skills_from_only",
        ],
    )
    def test_noop_when_install_skills_not_true(self, tmp_path, call_kwargs):
        calls = self._patched_install(tmp_path, auto=True, **call_kwargs)
        assert calls == []

    def test_auto_with_install_skills_true_installs_all_detected_at_project(self, tmp_path):
        calls = self._patched_install(tmp_path, auto=True, install_skills=True)
        installed_agents = {name for name, _, _ in calls}
        assert installed_agents == {"codex", "cursor", "claude"}
        for _name, scope, _sk in calls:
            assert scope == Scope.PROJECT

    def test_skills_agents_flag_filters_install_loop(self, tmp_path):
        calls = self._patched_install(
            tmp_path,
            auto=True,
            install_skills=True,
            skills_agents=["codex"],
        )
        assert {name for name, _, _ in calls} == {"codex"}

    def test_skills_scope_user_filters_out_cursor(self, tmp_path):
        calls = self._patched_install(
            tmp_path,
            auto=True,
            install_skills=True,
            skills_scope=Scope.USER,
        )
        installed_agents = {name for name, _, _ in calls}
        # Cursor doesn't support USER scope; should be skipped, not installed.
        assert "cursor" not in installed_agents
        assert {"codex", "claude"}.issubset(installed_agents)
        for _name, scope, _sk in calls:
            assert scope == Scope.USER

    def test_skills_from_builtin_only(self, tmp_path):
        """--skills-from nemo-platform installs only built-in skills."""
        calls = self._patched_install(
            tmp_path,
            auto=True,
            install_skills=True,
            skills_from=["nemo-platform"],
        )
        for _name, _scope, sk in calls:
            assert set(sk.keys()) == {"alpha"}

    def test_skills_from_plugin_only(self, tmp_path):
        """--skills-from example-plugin installs only plugin-provided skills."""
        calls = self._patched_install(
            tmp_path,
            auto=True,
            install_skills=True,
            skills_from=["example-plugin"],
        )
        for _name, _scope, sk in calls:
            assert set(sk.keys()) == {"beta"}

    def test_skills_from_multiple_sources_unions(self, tmp_path):
        """Passing several sources installs the union of their skills."""
        calls = self._patched_install(
            tmp_path,
            auto=True,
            install_skills=True,
            skills_from=["nemo-platform", "example-plugin"],
        )
        for _name, _scope, sk in calls:
            assert set(sk.keys()) == {"alpha", "beta"}

    def test_unknown_skills_from_raises_bad_parameter(self, tmp_path):
        """A typo in --skills-from fails fast with a clear typer.BadParameter."""
        with pytest.raises(typer.BadParameter) as exc_info:
            self._patched_install(
                tmp_path,
                auto=True,
                install_skills=True,
                skills_from=["does-not-exist"],
            )
        assert "Unknown skill source" in str(exc_info.value.message) or "Unknown skill source" in str(exc_info.value)

    def test_all_install_failures_raise_exit(self, tmp_path):
        """If every agent's install raises, --auto should exit non-zero so CI catches it."""
        skills = {"alpha": self._skill("alpha")}

        class _AlwaysFailInstaller:
            name = "x"
            supported_scopes = [Scope.PROJECT, Scope.USER]

            def install(self, scope, project_root, sk):
                raise RuntimeError("disk full")

            def get_install_path(self, scope, project_root, skill_name):
                return project_root / "stub" / skill_name

        failing = _AlwaysFailInstaller()
        installers = {"codex": failing, "claude": failing}

        with (
            patch(
                "nemo_platform_ext.cli.commands.setup._detect_coding_agents",
                return_value=[("AGENTS.md", "codex"), (".claude", "claude")],
            ),
            patch(
                "nemo_platform_ext.cli.commands.setup._load_skills_with_warnings",
                return_value=(skills, []),
            ),
            patch(
                "nemo_platform_ext.cli.commands.setup._find_project_root",
                return_value=tmp_path,
            ),
            patch(
                "nemo_platform_ext.cli.commands.setup.get_installer",
                side_effect=lambda name: installers[name],
            ),
            pytest.raises(typer.Exit) as exc_info,
        ):
            _maybe_install_skills(auto=True, install_skills=True)
        assert exc_info.value.exit_code == 1

    def test_skills_agents_overrides_empty_detection(self, tmp_path):
        """An explicit --skills-agents must install even when no markers exist."""
        skills = {"alpha": self._skill("alpha")}
        recorded: list[tuple[str, Scope, dict]] = []

        class _StubInstaller:
            supported_scopes = [Scope.PROJECT, Scope.USER]

            def __init__(self, name: str):
                self.name = name

            def install(self, scope, project_root, sk):
                recorded.append((self.name, scope, sk))
                return []

            def get_install_path(self, scope, project_root, skill_name):
                return project_root / "stub" / skill_name

        with (
            patch(
                "nemo_platform_ext.cli.commands.setup._detect_coding_agents",
                return_value=[],
            ),
            patch(
                "nemo_platform_ext.cli.commands.setup._load_skills_with_warnings",
                return_value=(skills, []),
            ),
            patch(
                "nemo_platform_ext.cli.commands.setup._find_project_root",
                return_value=tmp_path,
            ),
            patch(
                "nemo_platform_ext.cli.commands.setup.get_installer",
                side_effect=lambda name: _StubInstaller(name),
            ),
        ):
            _maybe_install_skills(
                auto=True,
                install_skills=True,
                skills_agents=["codex"],
            )

        assert [name for name, _, _ in recorded] == ["codex"]

    def test_unknown_skills_agents_raises_unsupported_agent(self, tmp_path):
        """Typos in --skills-agents fail fast via UnsupportedAgentError (before any install work)."""
        # Don't even need to patch detection — validation happens up-front.
        with pytest.raises(UnsupportedAgentError):
            _maybe_install_skills(
                auto=True,
                install_skills=True,
                skills_agents=["copex"],  # typo
            )

    def test_partial_install_failure_does_not_raise(self, tmp_path):
        """Mixed success/failure shouldn't fail the run; only total failure should."""
        skills = {"alpha": self._skill("alpha")}

        class _SuccessInstaller:
            supported_scopes = [Scope.PROJECT, Scope.USER]

            def install(self, scope, project_root, sk):
                return []

            def get_install_path(self, scope, project_root, skill_name):
                return project_root / "stub" / skill_name

        class _FailInstaller:
            supported_scopes = [Scope.PROJECT, Scope.USER]

            def install(self, scope, project_root, sk):
                raise RuntimeError("nope")

            def get_install_path(self, scope, project_root, skill_name):
                return project_root / "stub" / skill_name

        installers = {"codex": _SuccessInstaller(), "claude": _FailInstaller()}

        with (
            patch(
                "nemo_platform_ext.cli.commands.setup._detect_coding_agents",
                return_value=[("AGENTS.md", "codex"), (".claude", "claude")],
            ),
            patch(
                "nemo_platform_ext.cli.commands.setup._load_skills_with_warnings",
                return_value=(skills, []),
            ),
            patch(
                "nemo_platform_ext.cli.commands.setup._find_project_root",
                return_value=tmp_path,
            ),
            patch(
                "nemo_platform_ext.cli.commands.setup.get_installer",
                side_effect=lambda name: installers[name],
            ),
        ):
            # Should NOT raise — one agent succeeded.
            _maybe_install_skills(auto=True, install_skills=True)


# ---------------------------------------------------------------------------
# Agent deployment helpers
# ---------------------------------------------------------------------------


class TestAgentsPluginAvailable:
    def test_returns_true_when_importable(self):
        with patch("nemo_platform_ext.cli.commands.setup.importlib.util.find_spec", return_value=MagicMock()):
            assert _agents_plugin_available() is True

    def test_returns_false_when_missing(self):
        def side_effect(name):
            if name == "nemo_agents_plugin":
                return None
            return MagicMock()

        with patch("nemo_platform_ext.cli.commands.setup.importlib.util.find_spec", side_effect=side_effect):
            assert _agents_plugin_available() is False


class TestFindProjectRoot:
    def test_finds_git_root(self, tmp_path):
        (tmp_path / ".git").mkdir()
        subdir = tmp_path / "a" / "b"
        subdir.mkdir(parents=True)
        with patch("nemo_platform_ext.cli.commands.setup.Path.cwd", return_value=subdir):
            root = _find_project_root()
        assert root == tmp_path

    def test_falls_back_to_cwd(self, tmp_path):
        with patch("nemo_platform_ext.cli.commands.setup.Path.cwd", return_value=tmp_path):
            root = _find_project_root()
        assert root == tmp_path


# ---------------------------------------------------------------------------
# Provider idempotency
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_bypass_key_validation")
class TestProviderIdempotency:
    """Test that setup correctly upserts secrets and providers on re-runs.

    The bug: when a provider already exists but the secret is new (or its
    value has changed), setup skips the provider entirely and never
    re-associates the secret.  Model discovery then fails.
    """

    # -- interactive path --

    def test_existing_secret_gets_updated_with_new_value(self):
        """When the secret already exists, its value should be updated."""
        client = _make_mock_client(secret_exists=True, provider_exists=False)
        _register_provider_interactive(
            client,
            provider_name="nvidia-build",
            host_url="https://integrate.api.nvidia.com",
            api_key="new-key-value",
            workspace="default",
        )
        client.mock_secrets.update_secret.assert_called_once()
        update_call = client.mock_secrets.update_secret.call_args
        assert update_call.kwargs["name"] == "nvidia-build-api-key"
        assert update_call.kwargs["workspace"] == "default"
        assert isinstance(update_call.kwargs["body"], PlatformSecretUpdateRequest)
        assert update_call.kwargs["body"].value.get_secret_value() == "new-key-value"
        client.mock_secrets.create_secret.assert_not_called()
        client.inference.providers.create.assert_called_once()
        client.inference.providers.update.assert_not_called()

    def test_existing_provider_gets_updated_with_secret(self):
        """When the provider exists, it should be updated to point at the secret."""
        client = _make_mock_client(secret_exists=False, provider_exists=True)
        _register_provider_interactive(
            client,
            provider_name="nvidia-build",
            host_url="https://integrate.api.nvidia.com",
            api_key="my-key",
            workspace="default",
        )
        client.mock_secrets.create_secret.assert_called_once()
        create_call = client.mock_secrets.create_secret.call_args
        assert create_call.kwargs["workspace"] == "default"
        assert isinstance(create_call.kwargs["body"], PlatformSecretCreateRequest)
        assert create_call.kwargs["body"].name == "nvidia-build-api-key"
        assert create_call.kwargs["body"].value.get_secret_value() == "my-key"
        client.mock_secrets.update_secret.assert_not_called()
        client.inference.providers.update.assert_called_once()
        call_kwargs = client.inference.providers.update.call_args.kwargs
        assert call_kwargs["api_key_secret_name"] == "nvidia-build-api-key"
        assert call_kwargs["host_url"] == "https://integrate.api.nvidia.com"
        client.inference.providers.create.assert_not_called()

    def test_fresh_install_creates_both(self):
        """When neither exists, both secret and provider are created (regression guard)."""
        client = _make_mock_client(secret_exists=False, provider_exists=False)
        _register_provider_interactive(
            client,
            provider_name="openai",
            host_url="https://api.openai.com/v1",
            api_key="sk-test",
            workspace="default",
        )
        client.mock_secrets.create_secret.assert_called_once()
        client.inference.providers.create.assert_called_once()
        client.mock_secrets.update_secret.assert_not_called()
        client.inference.providers.update.assert_not_called()

    def test_existing_provider_updated_with_extra_headers(self):
        """Provider update repairs Anthropic auth without exposing the API key."""
        client = _make_mock_client(secret_exists=False, provider_exists=True)
        api_key = "sk-ant-test"
        _register_provider_interactive(
            client,
            provider_name="anthropic",
            host_url="https://api.anthropic.com",
            api_key=api_key,
            workspace="default",
            auth_header_format="X-Api-Key: {{ auth_secret }}",
            default_extra_headers={"anthropic-version": "2023-06-01"},
        )
        call_kwargs = client.inference.providers.update.call_args.kwargs
        assert call_kwargs["api_key_secret_name"] == "anthropic-api-key"
        assert call_kwargs["auth_header_format"] == "X-Api-Key: {{ auth_secret }}"
        assert call_kwargs["required_extra_headers"] is None
        assert api_key not in str(call_kwargs)
        assert call_kwargs["default_extra_headers"] == {"anthropic-version": "2023-06-01"}

    def test_auto_setup_updates_existing_anthropic_auth(self):
        """Auto setup repairs an existing Anthropic provider's auth template."""
        client = _make_mock_client(provider_exists=True, secret_exists=True)
        api_key = "sk-ant-updated"
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": api_key}, clear=True):
            result = _auto_setup(client, "default")
        assert result == "anthropic"
        call_kwargs = client.inference.providers.update.call_args.kwargs
        assert call_kwargs["api_key_secret_name"] == "anthropic-api-key"
        assert call_kwargs["auth_header_format"] == "X-Api-Key: {{ auth_secret }}"
        assert call_kwargs["required_extra_headers"] is None
        assert api_key not in str(call_kwargs)

    # -- auto path --

    def test_auto_setup_updates_existing_provider_secret_binding(self):
        """In auto mode, existing provider should be updated with the secret."""
        client = _make_mock_client(provider_exists=True, secret_exists=False)
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-new-key"}, clear=True):
            result = _auto_setup(client, "default")
        assert result == "openai"
        client.mock_secrets.create_secret.assert_called_once()
        client.inference.providers.update.assert_called_once()
        call_kwargs = client.inference.providers.update.call_args.kwargs
        assert call_kwargs["api_key_secret_name"] == "openai-api-key"

    def test_auto_setup_updates_existing_secret_value(self):
        """In auto mode, existing secret should be updated with the new key value."""
        client = _make_mock_client(provider_exists=True, secret_exists=True)
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "nvapi-new"}, clear=True):
            result = _auto_setup(client, "default")
        assert result == "nvidia-build"
        client.mock_secrets.update_secret.assert_called_once()
        update_call = client.mock_secrets.update_secret.call_args
        assert update_call.kwargs["name"] == "nvidia-build-api-key"
        assert update_call.kwargs["workspace"] == "default"
        assert isinstance(update_call.kwargs["body"], PlatformSecretUpdateRequest)
        assert update_call.kwargs["body"].value.get_secret_value() == "nvapi-new"
        client.mock_secrets.create_secret.assert_not_called()


# ---------------------------------------------------------------------------
# Interactive setup regression — provider sync timeout vs model picker
# ---------------------------------------------------------------------------


class TestInteractiveModelPairSelection:
    _MOD = "nemo_platform_ext.cli.commands.setup"

    def test_run_scopes_picker_to_registered_provider(self):
        client = MagicMock()
        cli_context = MagicMock()
        model_pair = ModelPair(
            default="default/claude-sonnet-4-6",
            fast="default/claude-haiku-4-5-20251001",
        )

        with (
            patch(
                f"{self._MOD}._interactive_collect_provider",
                return_value=("anthropic", "https://api.anthropic.com", None, None, None),
            ),
            patch(f"{self._MOD}._register_provider_interactive"),
            patch(
                f"{self._MOD}._wait_for_models",
                return_value=[
                    "default/claude-sonnet-4-6",
                    "default/claude-haiku-4-5-20251001",
                ],
            ),
            patch(f"{self._MOD}._select_model_pair", return_value=model_pair) as select_model_pair,
            patch(f"{self._MOD}._save_model_pair"),
            patch(f"{self._MOD}._maybe_install_skills"),
            patch(f"{self._MOD}._maybe_deploy_agent", return_value=False),
            patch(f"{self._MOD}._print_onboarding"),
        ):
            _run_interactive_mode(
                cli_context,
                client,
                "default",
                "http://localhost:8080",
                install_skills=False,
                deploy_agent=False,
            )

        select_model_pair.assert_called_once_with(client, "default", provider_name="anthropic")

    def test_skips_default_model_picker_when_new_provider_has_no_models(self):
        """When the new provider is still syncing, setup should not show a misleading picker."""
        client = MagicMock()
        cli_context = MagicMock()

        with (
            patch(
                f"{self._MOD}._interactive_collect_provider",
                return_value=("my-ollama-custom", "http://localhost:11434/v1", "ollama", None, None),
            ),
            patch(f"{self._MOD}._register_provider_interactive"),
            patch(f"{self._MOD}._wait_for_models", return_value=[]),
            patch(f"{self._MOD}._select_model_pair") as mock_select_model_pair,
            patch(f"{self._MOD}._save_model_pair"),
            patch(f"{self._MOD}._maybe_install_skills"),
            patch(f"{self._MOD}._maybe_deploy_agent"),
            patch(f"{self._MOD}._print_onboarding"),
            patch(f"{self._MOD}.console") as mock_console,
        ):
            _run_interactive_mode(
                cli_context,
                client,
                "default",
                "http://localhost:8080",
                install_skills=False,
                deploy_agent=False,
            )

        mock_select_model_pair.assert_not_called()
        printed_lines = [call.args[0] for call in mock_console.print.call_args_list if call.args]
        assert any("No models discovered yet (provider may still be syncing)" in line for line in printed_lines)
        assert any("Step 5: Choose agent models" in line for line in printed_lines)
        assert any("No agent models set for this provider yet." in line for line in printed_lines)
        assert any("Run [cyan]nemo setup[/cyan] again after models sync" in line for line in printed_lines)

    def test_warns_when_only_existing_provider_models_are_available(self):
        """If other providers have models, explain that the new provider is not yet represented."""
        client = MagicMock()
        cli_context = MagicMock()

        with (
            patch(
                f"{self._MOD}._interactive_collect_provider",
                return_value=("my-ollama-custom", "http://localhost:11434/v1", "ollama", None, None),
            ),
            patch(f"{self._MOD}._register_provider_interactive"),
            patch(f"{self._MOD}._wait_for_models", return_value=[]),
            patch(
                f"{self._MOD}._get_all_model_choices",
                return_value=[
                    (
                        "default/nvidia-nemotron-3.5-lightning-30b-a3b",
                        "nvidia-nemotron-3.5-lightning-30b-a3b (nvidia-build)",
                    )
                ],
            ),
            patch(f"{self._MOD}._select_model_pair") as mock_select_model_pair,
            patch(f"{self._MOD}._save_model_pair"),
            patch(f"{self._MOD}._maybe_install_skills"),
            patch(f"{self._MOD}._maybe_deploy_agent"),
            patch(f"{self._MOD}._print_onboarding"),
            patch(f"{self._MOD}.console") as mock_console,
        ):
            _run_interactive_mode(
                cli_context,
                client,
                "default",
                "http://localhost:8080",
                install_skills=False,
                deploy_agent=False,
            )

        mock_select_model_pair.assert_not_called()
        printed_lines = [call.args[0] for call in mock_console.print.call_args_list if call.args]
        assert any(
            "Models from existing providers are available, but not from 'my-ollama-custom' yet." in line
            for line in printed_lines
        )

    def test_picker_only_includes_models_from_selected_provider(self):
        """Models from another provider in the workspace must not enter the picker."""
        client = MagicMock()
        provider_a = MagicMock()
        provider_a.name = "nvidia-build"
        provider_a.served_models = [
            MagicMock(model_entity_id="default/nvidia-nemotron-3.5-lightning-30b-a3b"),
        ]
        provider_b = MagicMock()
        provider_b.name = "my-ollama-custom"
        provider_b.served_models = [
            MagicMock(model_entity_id="default/qwen2.5:1.5b"),
        ]
        client.inference.providers.list.return_value = MagicMock(data=[provider_a, provider_b])

        with patch(
            f"{self._MOD}.prompt_select",
            side_effect=["default/qwen2.5:1.5b", "default/qwen2.5:1.5b"],
        ) as mock_prompt_select:
            result = _select_model_pair(client, "default", provider_name="my-ollama-custom")

        assert result == ModelPair(
            default="default/qwen2.5:1.5b",
            fast="default/qwen2.5:1.5b",
        )
        expected_choices = [
            (
                "default/qwen2.5:1.5b",
                "qwen2.5:1.5b (my-ollama-custom)",
            ),
        ]
        assert mock_prompt_select.call_count == 2
        assert all(call.kwargs["choices"] == expected_choices for call in mock_prompt_select.call_args_list)
        default_call, fast_call = mock_prompt_select.call_args_list
        assert default_call.args[0] == "Choose your default model (used for quality-critical agent work):"
        assert default_call.kwargs["default"] == "default/qwen2.5:1.5b"
        assert default_call.kwargs["hint"] == "Press Enter to accept the default."
        assert fast_call.args[0] == "Choose your fast model (used for latency-sensitive agent work):"
        assert fast_call.kwargs["default"] == "default/qwen2.5:1.5b"
        assert fast_call.kwargs["hint"] == "Press Enter to reuse the default model."

    def test_picker_has_no_unverified_default(self):
        client = MagicMock()
        choices = [
            ("default/ai21labs-jamba-1-5-large-instruct", "Jamba"),
            ("default/nvidia/nemotron-nano-9b-v2", "Nemotron Nano"),
        ]

        with (
            patch(f"{self._MOD}._get_all_model_choices", return_value=choices),
            patch(f"{self._MOD}._select_usable_model_pair", return_value=None),
            patch(
                f"{self._MOD}.prompt_select",
                side_effect=["default/nvidia/nemotron-nano-9b-v2", "default/nvidia/nemotron-nano-9b-v2"],
            ) as mock_prompt_select,
        ):
            result = _select_model_pair(client, "default", provider_name="nvidia-build")

        assert result == ModelPair(
            default="default/nvidia/nemotron-nano-9b-v2",
            fast="default/nvidia/nemotron-nano-9b-v2",
        )
        default_call, fast_call = mock_prompt_select.call_args_list
        assert default_call.kwargs["default"] is None
        assert default_call.kwargs["hint"] == "Choose a model you can access."
        assert fast_call.kwargs["default"] == "default/nvidia/nemotron-nano-9b-v2"

    def test_returns_none_when_only_specialist_models_are_available(self):
        client = MagicMock()
        provider = MagicMock()
        provider.name = "nvidia-build"
        provider.served_models = [
            MagicMock(model_entity_id="default/adept-fuyu-8b"),
            MagicMock(model_entity_id="default/nvidia-nv-embedqa-e5-v5"),
            MagicMock(model_entity_id="default/llama-3.1-nemoguard-8b-content-safety"),
        ]
        client.inference.providers.list.return_value = MagicMock(data=[provider])

        with (
            patch(f"{self._MOD}.prompt_select") as mock_prompt_select,
            patch(f"{self._MOD}.console.print") as mock_print,
        ):
            result = _select_model_pair(client, "default", provider_name="nvidia-build")

        assert result is None
        mock_prompt_select.assert_not_called()
        printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        assert "No usable chat models discovered yet" in printed


class TestPickDefaultChatEntity:
    def test_skips_embedding_vision_guard_and_rerank(self):
        assert (
            _pick_default_chat_entity(
                [
                    "default/adept-fuyu-8b",
                    "default/nvidia-nv-embedqa-e5-v5",
                    "default/llama-nemoguard-8b",
                    "default/nv-rerankqa-mistral-4b",
                    "default/nvidia-nemotron-3.5-lightning-30b-a3b",
                ]
            )
            == "default/nvidia-nemotron-3.5-lightning-30b-a3b"
        )

    def test_returns_none_when_nothing_qualifies(self):
        assert _pick_default_chat_entity(["default/adept-fuyu-8b", "default/nv-embedqa"]) is None


def _probe_error(status_code: int, text: str = "", body: object | None = None) -> APIStatusError:
    """Build the SDK error the gateway raises for a non-2xx probe response."""
    request = httpx.Request("POST", "http://localhost:8080/v1/chat/completions")
    return APIStatusError("probe failed", response=httpx.Response(status_code, text=text, request=request), body=body)


def _client_answering(outcomes: dict[str, object]) -> MagicMock:
    """Return a client whose gateway probe answers per model entity ID.

    A value that is an exception is raised; anything else is returned as the
    successful response.
    """
    client = MagicMock()

    def post(trailing_uri: str, *, workspace: str, body: dict) -> object:
        outcome = outcomes[str(body["model"])]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client.with_options.return_value.inference.gateway.openai.post.side_effect = post
    return client


class TestModelSizeOrdering:
    @pytest.mark.parametrize(
        ("entity_id", "expected"),
        [
            ("default/meta-llama-3-3-70b-instruct", 70.0),
            ("default/nvidia-nemotron-3.5-lightning-30b-a3b", 30.0),
            ("default/openai-gpt-oss-120b", 120.0),
            ("default/meta-llama-3-1-8b-instruct", 8.0),
            ("default/nvidia-nemotron-nano-1.5b-v2", 1.5),
            ("default/ai21labs-jamba-1-5-large-instruct", None),
        ],
    )
    def test_parses_parameter_count_from_entity_name(self, entity_id, expected):
        assert _model_parameter_size(entity_id) == expected

    def test_orders_nvidia_first_then_by_size(self):
        candidates = [
            "default/meta-llama-3-3-70b-instruct",
            "default/nvidia-nemotron-3.5-lightning-30b-a3b",
            "default/ai21labs-jamba-1-5-large-instruct",
            "default/nvidia-nemotron-nano-9b-v2",
        ]

        assert _order_candidates_by_size(candidates, largest_first=True) == [
            "default/nvidia-nemotron-3.5-lightning-30b-a3b",
            "default/nvidia-nemotron-nano-9b-v2",
            "default/meta-llama-3-3-70b-instruct",
            "default/ai21labs-jamba-1-5-large-instruct",
        ]
        assert _order_candidates_by_size(candidates, largest_first=False) == [
            "default/nvidia-nemotron-nano-9b-v2",
            "default/nvidia-nemotron-3.5-lightning-30b-a3b",
            "default/meta-llama-3-3-70b-instruct",
            "default/ai21labs-jamba-1-5-large-instruct",
        ]

    def test_prefers_nvidia_slash_ids(self):
        nvidia = "default/nvidia/mistral-nemo-minitron-8b-instruct"
        meta = "default/meta/llama-3.3-70b-instruct"

        assert _is_preferred_vendor("default/nvidia/llama-3.1-8b-instruct") is True
        assert _order_candidates_by_size([meta, nvidia], largest_first=True) == [nvidia, meta]


class TestModelProbe:
    @pytest.mark.parametrize(
        ("outcome", "expected"),
        [
            (MagicMock(), True),
            (_probe_error(404, body={"detail": "Function not found for account"}), False),
            (_probe_error(403), False),
            (_probe_error(502, '{"detail": "Backend returned 404: Function ... not found for account"}'), False),
            (_probe_error(503, "upstream busy"), False),
        ],
    )
    def test_classifies_gateway_responses(self, outcome, expected):
        client = _client_answering({"default/model": outcome})

        assert _probe_model_entity(client.with_options(), "default", "default/model") is expected

    def test_retries_a_404_until_the_route_is_published(self):
        """The gateway 404s a discovered model until its VirtualModel exists."""
        client = MagicMock()
        client.inference.gateway.openai.post.side_effect = [_probe_error(404), _probe_error(404), MagicMock()]

        with patch(f"{SETUP_MOD}._pause"):
            usable = _probe_model_entity(client, "default", "default/model")

        assert usable is True
        assert client.inference.gateway.openai.post.call_count == 3

    def test_does_not_retry_an_entitlement_404(self):
        client = MagicMock()
        client.inference.gateway.openai.post.side_effect = _probe_error(
            404, body={"detail": "Function not found for account"}
        )

        assert _probe_model_entity(client, "default", "default/model") is False
        assert client.inference.gateway.openai.post.call_count == 1

    def test_skips_timeouts_so_later_candidates_can_be_tried(self):
        client = _client_answering(
            {"default/model": APITimeoutError(request=httpx.Request("POST", "http://localhost:8080"))}
        )

        assert _probe_model_entity(client.with_options(), "default", "default/model") is False

    def test_reraises_when_the_gateway_is_unreachable(self):
        client = _client_answering(
            {"default/model": APIConnectionError(request=httpx.Request("POST", "http://localhost:8080"))}
        )

        with pytest.raises(APIConnectionError):
            _probe_model_entity(client.with_options(), "default", "default/model")

    def test_sends_a_short_chat_request_for_the_candidate(self):
        client = _client_answering({"default/model": MagicMock()})
        gateway = client.with_options()

        _probe_model_entity(gateway, "default", "default/model")

        gateway.inference.gateway.openai.post.assert_called_once()
        _, kwargs = gateway.inference.gateway.openai.post.call_args
        assert kwargs["workspace"] == "default"
        assert kwargs["body"]["model"] == "default/model"
        assert kwargs["body"]["messages"][0]["role"] == "user"


class TestUsableModelPairSelection:
    @pytest.fixture(autouse=True)
    def _no_sleep(self) -> Iterator[None]:
        """Probes pause between 404 retries; selection tests should not sleep."""
        with patch(f"{SETUP_MOD}._pause"):
            yield

    def test_picks_largest_usable_as_default_and_smallest_as_fast(self):
        entity_ids = [
            "default/nvidia-nemotron-3.5-lightning-30b-a3b",
            "default/nvidia-nemotron-nano-9b-v2",
        ]
        client = _client_answering({entity_id: MagicMock() for entity_id in entity_ids})

        assert _select_usable_model_pair(client, "default", entity_ids) == ModelPair(
            default="default/nvidia-nemotron-3.5-lightning-30b-a3b",
            fast="default/nvidia-nemotron-nano-9b-v2",
        )

    def test_skips_a_model_the_account_cannot_serve(self):
        entity_ids = [
            "default/ai21labs-jamba-1-5-large-instruct",
            "default/meta-llama-3-3-70b-instruct",
        ]
        client = _client_answering(
            {
                "default/meta-llama-3-3-70b-instruct": _probe_error(
                    502, '{"detail": "Backend returned 404: Function not found for account"}'
                ),
                "default/ai21labs-jamba-1-5-large-instruct": MagicMock(),
            }
        )

        assert _select_usable_model_pair(client, "default", entity_ids) == ModelPair(
            default="default/ai21labs-jamba-1-5-large-instruct",
            fast="default/ai21labs-jamba-1-5-large-instruct",
        )

    def test_saves_nothing_when_no_candidate_serves_a_request(self):
        entity_ids = ["default/one-8b-instruct", "default/two-70b-instruct"]
        client = _client_answering({entity_id: _probe_error(404) for entity_id in entity_ids})

        assert _select_usable_model_pair(client, "default", entity_ids) is None

    def test_stops_probing_after_the_attempt_budget(self):
        entity_ids = [f"default/model-{index}b-instruct" for index in range(1, 12)]
        client = _client_answering({entity_id: _probe_error(404) for entity_id in entity_ids})

        assert _select_usable_model_pair(client, "default", entity_ids) is None
        post = client.with_options.return_value.inference.gateway.openai.post
        probed = {str(probe.kwargs["body"]["model"]) for probe in post.call_args_list}
        assert len(probed) == setup_commands._MODEL_PROBE_MAX_ATTEMPTS

    def test_skips_a_timed_out_candidate_and_keeps_probing(self):
        entity_ids = [
            "default/nvidia-nemotron-3-ultra-550b-a55b",
            "default/nvidia-nemotron-nano-9b-v2",
        ]
        timeout = APITimeoutError(request=httpx.Request("POST", "http://localhost:8080"))
        client = _client_answering(
            {
                "default/nvidia-nemotron-3-ultra-550b-a55b": timeout,
                "default/nvidia-nemotron-nano-9b-v2": MagicMock(),
            }
        )

        assert _select_usable_model_pair(client, "default", entity_ids) == ModelPair(
            default="default/nvidia-nemotron-nano-9b-v2",
            fast="default/nvidia-nemotron-nano-9b-v2",
        )

    def test_waits_out_the_reconcile_lag_before_rejecting_candidates(self):
        """Cold start: the first probe 404s because the model has no route yet."""
        entity_ids = ["default/nvidia-nemotron-nano-9b-v2"]
        client = MagicMock()
        client.with_options.return_value.inference.gateway.openai.post.side_effect = [
            _probe_error(404),
            MagicMock(),
        ]

        assert _select_usable_model_pair(client, "default", entity_ids) == ModelPair(
            default="default/nvidia-nemotron-nano-9b-v2",
            fast="default/nvidia-nemotron-nano-9b-v2",
        )

    def test_each_candidate_gets_a_route_readiness_window(self):
        large = "default/nvidia-model-70b-instruct"
        small = "default/nvidia-model-8b-instruct"
        client = MagicMock()
        attempts = {large: 0, small: 0}
        clock = {"now": 0.0}

        def post(trailing_uri: str, *, workspace: str, body: dict) -> object:
            entity_id = str(body["model"])
            attempts[entity_id] += 1
            if entity_id == large:
                if attempts[large] == setup_commands._MODEL_ROUTE_MAX_RETRIES + 1:
                    clock["now"] = setup_commands._MODEL_ROUTE_READY_SECONDS + 1
                raise _probe_error(404)
            if attempts[entity_id] == 1:
                raise _probe_error(404)
            return MagicMock()

        client.with_options.return_value.inference.gateway.openai.post.side_effect = post

        with patch(f"{SETUP_MOD}.time.monotonic", side_effect=lambda: clock["now"]):
            assert _select_usable_model_pair(client, "default", [large, small]) == ModelPair(
                default=small,
                fast=small,
            )

        assert attempts[large] == setup_commands._MODEL_ROUTE_MAX_RETRIES + 1
        assert attempts[small] == 2
        assert clock["now"] > setup_commands._MODEL_ROUTE_READY_SECONDS

    def test_stops_probing_after_the_time_budget(self):
        entity_ids = [f"default/model-{index}b-instruct" for index in range(1, 6)]
        timeout = APITimeoutError(request=httpx.Request("POST", "http://localhost:8080"))
        client = _client_answering({entity_id: timeout for entity_id in entity_ids})
        clock = iter([0.0, 0.0, 0.0, 91.0])

        with patch(f"{SETUP_MOD}.time.monotonic", side_effect=lambda: next(clock)):
            assert _select_usable_model_pair(client, "default", entity_ids) is None

        post = client.with_options.return_value.inference.gateway.openai.post
        assert post.call_count == 1

    def test_saves_nothing_when_the_gateway_is_unreachable(self):
        entity_ids = ["default/nvidia-nemotron-nano-9b-v2", "default/meta-llama-3-3-70b-instruct"]
        unreachable = APIConnectionError(request=httpx.Request("POST", "http://localhost:8080"))
        client = _client_answering({entity_id: unreachable for entity_id in entity_ids})

        assert _select_usable_model_pair(client, "default", entity_ids) is None

    def test_ignores_non_chat_entities(self):
        entity_ids = ["default/nvidia-nv-embedqa-e5-v5", "default/adept-fuyu-8b"]
        client = _client_answering({})

        assert _select_usable_model_pair(client, "default", entity_ids) is None
        client.with_options.return_value.inference.gateway.openai.post.assert_not_called()


class TestAutoModelPairSelection:
    def test_entity_discovery_filters_other_workspace_providers(self):
        client = MagicMock()
        openai = MagicMock()
        openai.name = "openai"
        openai.served_models = [MagicMock(model_entity_id="default/gpt-4.1")]
        anthropic = MagicMock()
        anthropic.name = "anthropic"
        anthropic.served_models = [MagicMock(model_entity_id="default/claude-sonnet-4-6")]
        client.inference.providers.list.return_value = MagicMock(data=[openai, anthropic])

        result = setup_commands._get_all_model_entity_ids(
            client,
            "default",
            provider_name="anthropic",
        )

        assert result == ["default/claude-sonnet-4-6"]

    def test_explicit_default_and_fast_models_are_saved_independently(self):
        cli_context = MagicMock()
        with (
            patch.dict(
                "os.environ",
                {
                    "NEMO_DEFAULT_MODEL": "default/quality",
                    "NEMO_FAST_MODEL": "default/fast",
                },
                clear=True,
            ),
            patch(f"{SETUP_MOD}._auto_setup", return_value="anthropic"),
            patch(f"{SETUP_MOD}._get_all_model_entity_ids", return_value=["default/discovered"]),
            patch(f"{SETUP_MOD}._save_model_pair") as save_pair,
            patch(f"{SETUP_MOD}._maybe_install_skills"),
            patch(f"{SETUP_MOD}._maybe_deploy_agent"),
            patch(f"{SETUP_MOD}._verify_platform_health", return_value=True),
        ):
            _run_auto_mode(
                cli_context,
                MagicMock(),
                "default",
                "http://localhost:8080",
                install_skills=False,
                deploy_agent=False,
            )

        save_pair.assert_called_once_with(cli_context, ModelPair(default="default/quality", fast="default/fast"))

    def test_discovered_default_is_scoped_to_auto_configured_provider(self):
        cli_context = MagicMock()
        client = MagicMock()
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(f"{SETUP_MOD}._auto_setup", return_value="anthropic"),
            patch(
                f"{SETUP_MOD}._get_all_model_entity_ids",
                return_value=["default/claude-sonnet-4-6"],
            ) as get_model_ids,
            patch(f"{SETUP_MOD}._save_model_pair") as save_pair,
            patch(f"{SETUP_MOD}._maybe_install_skills"),
            patch(f"{SETUP_MOD}._maybe_deploy_agent"),
            patch(f"{SETUP_MOD}._verify_platform_health", return_value=True),
        ):
            _run_auto_mode(
                cli_context,
                client,
                "default",
                "http://localhost:8080",
                install_skills=False,
                deploy_agent=False,
            )

        get_model_ids.assert_called_once_with(client, "default", provider_name="anthropic")
        save_pair.assert_called_once_with(
            cli_context,
            ModelPair(
                default="default/claude-sonnet-4-6",
                fast="default/claude-sonnet-4-6",
            ),
        )

    def test_first_discovered_model_is_default_and_fast_fallback(self):
        cli_context = MagicMock()
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(f"{SETUP_MOD}._auto_setup", return_value="openai"),
            patch(
                f"{SETUP_MOD}._get_all_model_entity_ids",
                return_value=["default/a-model", "default/z-model"],
            ),
            patch(f"{SETUP_MOD}._save_model_pair") as save_pair,
            patch(f"{SETUP_MOD}._maybe_install_skills"),
            patch(f"{SETUP_MOD}._maybe_deploy_agent"),
            patch(f"{SETUP_MOD}._verify_platform_health", return_value=True),
        ):
            _run_auto_mode(
                cli_context,
                MagicMock(),
                "default",
                "http://localhost:8080",
                install_skills=False,
                deploy_agent=False,
            )

        save_pair.assert_called_once_with(
            cli_context,
            ModelPair(default="default/a-model", fast="default/a-model"),
        )

    def test_skips_unusable_first_discovered_model(self):
        cli_context = MagicMock()
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(f"{SETUP_MOD}._auto_setup", return_value="nvidia-build"),
            patch(
                f"{SETUP_MOD}._get_all_model_entity_ids",
                return_value=[
                    "default/adept-fuyu-8b",
                    "default/nvidia-nv-embedqa-e5-v5",
                    "default/nvidia-nemotron-3.5-lightning-30b-a3b",
                ],
            ),
            patch(f"{SETUP_MOD}._save_model_pair") as save_pair,
            patch(f"{SETUP_MOD}._maybe_install_skills"),
            patch(f"{SETUP_MOD}._maybe_deploy_agent"),
            patch(f"{SETUP_MOD}._verify_platform_health", return_value=True),
        ):
            _run_auto_mode(
                cli_context,
                MagicMock(),
                "default",
                "http://localhost:8080",
                install_skills=False,
                deploy_agent=False,
            )

        save_pair.assert_called_once_with(
            cli_context,
            ModelPair(
                default="default/nvidia-nemotron-3.5-lightning-30b-a3b",
                fast="default/nvidia-nemotron-3.5-lightning-30b-a3b",
            ),
        )

    def test_does_not_save_default_when_only_unusable_models_qualify(self):
        cli_context = MagicMock()
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(f"{SETUP_MOD}._auto_setup", return_value="nvidia-build"),
            patch(
                f"{SETUP_MOD}._get_all_model_entity_ids",
                return_value=[
                    "default/adept-fuyu-8b",
                    "default/nvidia-nv-embedqa-e5-v5",
                    "default/llama-3.1-nemoguard-8b-content-safety",
                ],
            ),
            patch(f"{SETUP_MOD}._save_model_pair") as save_pair,
            patch(f"{SETUP_MOD}._maybe_install_skills"),
            patch(f"{SETUP_MOD}._maybe_deploy_agent"),
            patch(f"{SETUP_MOD}._verify_platform_health", return_value=True),
        ):
            _run_auto_mode(
                cli_context,
                MagicMock(),
                "default",
                "http://localhost:8080",
                install_skills=False,
                deploy_agent=False,
            )

        save_pair.assert_not_called()

    def test_fast_override_without_default_warns_and_is_not_saved(self):
        cli_context = MagicMock()
        with (
            patch.dict("os.environ", {"NEMO_FAST_MODEL": "default/fast"}, clear=True),
            patch(f"{SETUP_MOD}._MODEL_DISCOVERY_MAX_ROUNDS", 1),
            patch(f"{SETUP_MOD}._MODEL_DISCOVERY_ROUND_SECONDS", 0),
            patch(f"{SETUP_MOD}._auto_setup", return_value="openai"),
            patch(f"{SETUP_MOD}._get_all_model_entity_ids", return_value=[]),
            patch(f"{SETUP_MOD}._save_model_pair") as save_pair,
            patch(f"{SETUP_MOD}._maybe_install_skills"),
            patch(f"{SETUP_MOD}._maybe_deploy_agent"),
            patch(f"{SETUP_MOD}._verify_platform_health", return_value=True),
            patch(f"{SETUP_MOD}.console.print") as print_message,
        ):
            _run_auto_mode(
                cli_context,
                MagicMock(),
                "default",
                "http://localhost:8080",
                install_skills=False,
                deploy_agent=False,
            )

        save_pair.assert_not_called()
        assert any("NEMO_FAST_MODEL is ignored" in call.args[0] for call in print_message.call_args_list)
        printed = " ".join(str(c) for c in print_message.call_args_list)
        assert "Setup complete with warnings" in printed
        assert "[green]Setup complete![/green]" not in printed

    def test_setup_complete_is_unqualified_when_default_model_is_set(self):
        cli_context = MagicMock()
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(f"{SETUP_MOD}._auto_setup", return_value="openai"),
            patch(f"{SETUP_MOD}._get_all_model_entity_ids", return_value=["default/a-model"]),
            patch(f"{SETUP_MOD}._save_model_pair"),
            patch(f"{SETUP_MOD}._maybe_install_skills"),
            patch(f"{SETUP_MOD}._maybe_deploy_agent"),
            patch(f"{SETUP_MOD}._verify_platform_health", return_value=True),
            patch(f"{SETUP_MOD}.console.print") as print_message,
        ):
            _run_auto_mode(
                cli_context,
                MagicMock(),
                "default",
                "http://localhost:8080",
                install_skills=False,
                deploy_agent=False,
            )

        printed = " ".join(str(c) for c in print_message.call_args_list)
        assert "[green]Setup complete![/green]" in printed
        assert "Setup complete with warnings" not in printed

    def test_discovered_models_are_probed_before_being_saved(self):
        cli_context = MagicMock()
        client = MagicMock()
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(f"{SETUP_MOD}._auto_setup", return_value="nvidia-build"),
            patch(f"{SETUP_MOD}._get_all_model_entity_ids", return_value=["default/a-model"]),
            patch(
                f"{SETUP_MOD}._select_usable_model_pair",
                return_value=ModelPair(default="default/a-model", fast="default/a-model"),
            ) as select_pair,
            patch(f"{SETUP_MOD}._save_model_pair") as save_pair,
            patch(f"{SETUP_MOD}._maybe_install_skills"),
            patch(f"{SETUP_MOD}._maybe_deploy_agent"),
            patch(f"{SETUP_MOD}._verify_platform_health", return_value=True),
        ):
            _run_auto_mode(
                cli_context,
                client,
                "default",
                "http://localhost:8080",
                install_skills=False,
                deploy_agent=False,
            )

        select_pair.assert_called_once_with(client, "default", ["default/a-model"])
        save_pair.assert_called_once_with(cli_context, ModelPair(default="default/a-model", fast="default/a-model"))

    def test_no_model_is_saved_when_none_serve_a_request(self):
        cli_context = MagicMock()
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(f"{SETUP_MOD}._auto_setup", return_value="nvidia-build"),
            patch(
                f"{SETUP_MOD}._get_all_model_entity_ids",
                return_value=["default/ai21labs-jamba-1-5-large-instruct"],
            ),
            patch(f"{SETUP_MOD}._select_usable_model_pair", return_value=None),
            patch(f"{SETUP_MOD}._save_model_pair") as save_pair,
            patch(f"{SETUP_MOD}._maybe_install_skills"),
            patch(f"{SETUP_MOD}._maybe_deploy_agent"),
            patch(f"{SETUP_MOD}._verify_platform_health", return_value=True),
            patch(f"{SETUP_MOD}.console.print") as print_message,
        ):
            _run_auto_mode(
                cli_context,
                MagicMock(),
                "default",
                "http://localhost:8080",
                install_skills=False,
                deploy_agent=False,
            )

        save_pair.assert_not_called()
        printed = " ".join(str(c) for c in print_message.call_args_list)
        assert "none of the discovered models served a test request" in printed

    def test_explicit_default_model_is_not_probed(self):
        cli_context = MagicMock()
        with (
            patch.dict("os.environ", {"NEMO_DEFAULT_MODEL": "default/quality"}, clear=True),
            patch(f"{SETUP_MOD}._auto_setup", return_value="nvidia-build"),
            patch(f"{SETUP_MOD}._get_all_model_entity_ids", return_value=["default/a-model"]),
            patch(f"{SETUP_MOD}._select_usable_model_pair") as select_pair,
            patch(f"{SETUP_MOD}._save_model_pair") as save_pair,
            patch(f"{SETUP_MOD}._maybe_install_skills"),
            patch(f"{SETUP_MOD}._maybe_deploy_agent"),
            patch(f"{SETUP_MOD}._verify_platform_health", return_value=True),
        ):
            _run_auto_mode(
                cli_context,
                MagicMock(),
                "default",
                "http://localhost:8080",
                install_skills=False,
                deploy_agent=False,
            )

        select_pair.assert_not_called()
        save_pair.assert_called_once_with(cli_context, ModelPair(default="default/quality", fast="default/quality"))


# ---------------------------------------------------------------------------
# API key validation
# ---------------------------------------------------------------------------


def _http_response(status_code: int, json_body: dict | None = None, text: str = "") -> MagicMock:
    """Build a mock httpx response with explicit JSON/text, not MagicMock defaults."""
    mock_resp = MagicMock(status_code=status_code)
    mock_resp.is_success = 200 <= status_code < 300
    if json_body is None:
        mock_resp.json.side_effect = ValueError("no json")
        mock_resp.text = text
    else:
        mock_resp.json.return_value = json_body
        mock_resp.text = text
    return mock_resp


class TestValidateApiKey:
    """Tests for _validate_api_key — probes provider to detect bad credentials."""

    _MOD = "nemo_platform_ext.cli.commands.setup"

    @pytest.mark.parametrize(
        "provider_name,status_code,expected_passed",
        [
            ("nvidia-build", 400, True),
            ("nvidia-build", 401, False),
            ("nvidia-build", 403, False),
            ("nvidia-build", 200, True),
            ("openai", 200, True),
            ("openai", 401, False),
        ],
    )
    def test_http_responses(self, provider_name, status_code, expected_passed):
        host_url = _KNOWN_PROVIDERS_BY_NAME[provider_name].host_url
        mock_resp = _http_response(status_code, {"data": [{"id": "nvidia/nemotron-3-nano-30b-a3b"}]})
        with patch(f"{self._MOD}.httpx.request", return_value=mock_resp):
            result = _validate_api_key(provider_name, host_url, "test-key")
        assert result.passed is expected_passed

    def test_nvidia_build_lists_catalog_then_posts_preferred_nvidia_chat_model(self):
        catalog = _http_response(
            200,
            {
                "data": [
                    {"id": "ai21labs/jamba-1.5-large-instruct"},
                    {"id": "nvidia/nv-embedqa-e5-v5"},
                    {"id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"},
                ]
            },
        )
        chat_ok = _http_response(200, {})

        def side_effect(method, url, **kwargs):
            if method == "GET":
                assert url == "https://integrate.api.nvidia.com/v1/models"
                return catalog
            assert method == "POST"
            assert url == "https://integrate.api.nvidia.com/v1/chat/completions"
            assert kwargs["json"]["model"] == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
            return chat_ok

        with patch(f"{self._MOD}.httpx.request", side_effect=side_effect) as mock_req:
            result = _validate_api_key("nvidia-build", "https://integrate.api.nvidia.com", "test-key")
        assert result.passed is True
        assert result.message == ""
        assert mock_req.call_count == 2

    def test_nvidia_build_skips_gone_model_and_accepts_next(self):
        catalog = _http_response(
            200,
            {
                "data": [
                    {"id": "nvidia/nemotron-3-super-120b-a12b"},
                    {"id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"},
                ]
            },
        )
        gone = _http_response(
            410,
            {"detail": "The model 'nvidia/nemotron-3-super-120b-a12b' has reached its end of life."},
        )
        chat_ok = _http_response(200, {})
        posts: list[str] = []

        def side_effect(method, url, **kwargs):
            if method == "GET":
                return catalog
            posts.append(kwargs["json"]["model"])
            if kwargs["json"]["model"] == "nvidia/nemotron-3-super-120b-a12b":
                return gone
            return chat_ok

        with patch(f"{self._MOD}.httpx.request", side_effect=side_effect):
            result = _validate_api_key("nvidia-build", "https://integrate.api.nvidia.com", "test-key")
        assert result.passed is True
        assert result.message == ""
        assert posts == [
            "nvidia/nemotron-3-super-120b-a12b",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        ]

    def test_rejected_key_does_not_blame_probe_target(self):
        catalog = _http_response(200, {"data": [{"id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"}]})
        rejected = _http_response(401, {"detail": "Invalid API key"}, text='{"detail":"Invalid API key"}')

        def side_effect(method, url, **kwargs):
            return catalog if method == "GET" else rejected

        with patch(f"{self._MOD}.httpx.request", side_effect=side_effect):
            result = _validate_api_key("nvidia-build", "https://integrate.api.nvidia.com", "bad-key")
        assert result.passed is False
        assert "credentials" in result.message.lower()
        assert "Invalid API key" in result.message
        assert "unavailable" not in result.message

    @pytest.mark.parametrize("status_code", [404, 405, 410, 429, 500, 502])
    def test_non_2xx_non_rejection_returns_warning(self, status_code):
        mock_resp = _http_response(status_code)
        with patch(f"{self._MOD}.httpx.request", return_value=mock_resp):
            result = _validate_api_key("openai", "https://api.openai.com/v1", "test-key")
        assert result.passed is True
        assert f"HTTP {status_code}" in result.message
        assert "credentials" not in result.message.lower()
        assert "API key" not in result.message

    def test_gone_probe_includes_upstream_detail_and_does_not_fail(self):
        detail = (
            "The model 'nvidia/nemotron-3-nano-30b-a3b' has reached its end of life "
            "on 2026-09-01T09:00:00Z and is no longer available."
        )
        catalog = _http_response(200, {"data": [{"id": "nvidia/nemotron-3-nano-30b-a3b"}]})
        gone = _http_response(410, {"title": "Gone", "status": 410, "detail": detail})

        def side_effect(method, url, **kwargs):
            return catalog if method == "GET" else gone

        with patch(f"{self._MOD}.httpx.request", side_effect=side_effect):
            result = _validate_api_key("nvidia-build", "https://integrate.api.nvidia.com", "test-key")
        assert result.passed is True
        assert "HTTP 410" in result.message
        assert "end of life" in result.message
        assert "credentials" not in result.message.lower()
        assert "API key" not in result.message

    @pytest.mark.parametrize(
        "provider_name,api_key,side_effect,expected_passed",
        [
            ("nvidia-build", "k", httpx.TimeoutException("timeout"), True),
            ("nvidia-build", "k", httpx.ConnectError("refused"), True),
            ("custom-thing", "k", None, True),
            ("nvidia-build", None, None, True),
        ],
        ids=["timeout", "connect-error", "unknown-provider", "no-api-key"],
    )
    def test_skip_and_error_paths(self, provider_name, api_key, side_effect, expected_passed):
        with patch(f"{self._MOD}.httpx.request", side_effect=side_effect) as mock_req:
            result = _validate_api_key(provider_name, "https://example.com", api_key)
        assert result.passed is expected_passed
        if api_key is None or provider_name not in _PROBE_CONFIGS:
            mock_req.assert_not_called()

    def test_custom_auth_header_format(self):
        """Anthropic-style 'X-Api-Key: {{ auth_secret }}' header should be constructed."""
        mock_resp = MagicMock(status_code=400)
        mock_resp.is_success = False
        with patch(f"{self._MOD}.httpx.request", return_value=mock_resp) as mock_req:
            _validate_api_key(
                "nvidia-build",
                "https://integrate.api.nvidia.com",
                "my-secret-key",
                auth_header_format="X-Api-Key: {{ auth_secret }}",
            )
        _, kwargs = mock_req.call_args
        assert kwargs["headers"]["X-Api-Key"] == "my-secret-key"
        assert "Authorization" not in kwargs["headers"]

    def test_extra_headers_included(self):
        """default_extra_headers should be merged into the probe request."""
        mock_resp = MagicMock(status_code=400)
        mock_resp.is_success = False
        extra = {"anthropic-version": "2023-06-01"}
        with patch(f"{self._MOD}.httpx.request", return_value=mock_resp) as mock_req:
            _validate_api_key(
                "nvidia-build",
                "https://integrate.api.nvidia.com",
                "test-key",
                default_extra_headers=extra,
            )
        _, kwargs = mock_req.call_args
        assert kwargs["headers"]["anthropic-version"] == "2023-06-01"


class TestValidateApiKeyIntegration:
    """Tests that _validate_api_key gates model discovery in interactive and auto flows."""

    _MOD = "nemo_platform_ext.cli.commands.setup"

    def test_interactive_invalid_key_blocks_discovery(self):
        """When key validation fails, _wait_for_models must not be called."""
        with (
            patch(
                f"{self._MOD}._interactive_collect_provider",
                return_value=("nvidia-build", "https://integrate.api.nvidia.com", "bad-key", None, None),
            ),
            patch(f"{self._MOD}._register_provider_interactive"),
            patch(
                f"{self._MOD}._validate_api_key",
                return_value=KeyValidationResult(passed=False, message="API key rejected"),
            ),
            patch(f"{self._MOD}._wait_for_models") as mock_wait,
            patch(f"{self._MOD}.console"),
            pytest.raises(ClickExit) as exc_info,
        ):
            cli_ctx = MagicMock()
            _run_interactive_mode(
                cli_ctx,
                _make_mock_client(),
                "default",
                "http://localhost:8080",
                None,
                None,
            )
        assert exc_info.value.exit_code == 1
        mock_wait.assert_not_called()

    def test_auto_invalid_key_raises_exit(self):
        """When key validation fails in auto mode, _auto_setup should raise typer.Exit(1)."""
        client = _make_mock_client()
        with (
            patch.dict("os.environ", {"NVIDIA_API_KEY": "bad-key"}, clear=True),
            patch(
                f"{self._MOD}._validate_api_key",
                return_value=KeyValidationResult(passed=False, message="API key rejected"),
            ),
            patch(f"{self._MOD}.console"),
            pytest.raises(ClickExit) as exc_info,
        ):
            _auto_setup(client, "default")
        assert exc_info.value.exit_code == 1

    def test_auto_gone_probe_still_registers_provider(self):
        """HTTP 410 on the chat probe must not abort provider registration or blame the key."""
        client = _make_mock_client()
        catalog = _http_response(200, {"data": [{"id": "nvidia/nemotron-3-nano-30b-a3b"}]})
        gone = _http_response(
            410,
            {
                "title": "Gone",
                "status": 410,
                "detail": "The model 'nvidia/nemotron-3-nano-30b-a3b' has reached its end of life.",
            },
        )

        def side_effect(method, url, **kwargs):
            return catalog if method == "GET" else gone

        with (
            patch.dict("os.environ", {"NVIDIA_API_KEY": "nvapi-good"}, clear=True),
            patch(f"{self._MOD}.httpx.request", side_effect=side_effect),
            patch(f"{self._MOD}.console.print") as print_message,
        ):
            result = _auto_setup(client, "default")
        assert result == "nvidia-build"
        client.inference.providers.create.assert_called_once()
        printed = " ".join(str(call) for call in print_message.call_args_list)
        assert "Check the value of $NVIDIA_API_KEY" not in printed
        assert "Could not validate API key" not in printed
        assert "end of life" in printed

    def test_auto_rejected_key_still_points_at_env_var(self):
        client = _make_mock_client()
        catalog = _http_response(200, {"data": [{"id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"}]})
        rejected = _http_response(401, {"detail": "Unauthorized"})

        def side_effect(method, url, **kwargs):
            return catalog if method == "GET" else rejected

        with (
            patch.dict("os.environ", {"NVIDIA_API_KEY": "bad-key"}, clear=True),
            patch(f"{self._MOD}.httpx.request", side_effect=side_effect),
            patch(f"{self._MOD}.console.print") as print_message,
            pytest.raises(ClickExit) as exc_info,
        ):
            _auto_setup(client, "default")
        assert exc_info.value.exit_code == 1
        client.inference.providers.create.assert_not_called()
        printed = " ".join(str(call) for call in print_message.call_args_list)
        assert "Check the value of $NVIDIA_API_KEY" in printed

    def test_auto_warning_escapes_rich_markup_in_upstream_detail(self):
        client = _make_mock_client()
        catalog = _http_response(200, {"data": [{"id": "nvidia/nemotron-3-nano-30b-a3b"}]})
        gone = _http_response(410, {"detail": "invalid value at [/messages/0]"})

        def side_effect(method, url, **kwargs):
            return catalog if method == "GET" else gone

        with (
            patch.dict("os.environ", {"NVIDIA_API_KEY": "nvapi-good"}, clear=True),
            patch(f"{self._MOD}.httpx.request", side_effect=side_effect),
            patch(f"{self._MOD}.console.print") as print_message,
        ):
            result = _auto_setup(client, "default")
        assert result == "nvidia-build"
        printed = " ".join(str(call.args[0]) for call in print_message.call_args_list if call.args)
        assert r"\[/messages/0]" in printed


# ---------------------------------------------------------------------------
# Polling constants — tighter intervals for localhost responsiveness
# ---------------------------------------------------------------------------


@pytest.fixture
def spinner_console():
    """Patch console with a mock status spinner context manager."""
    mock_status = MagicMock()
    with patch("nemo_platform_ext.cli.commands.setup.console") as mock_console:
        mock_console.status.return_value.__enter__ = MagicMock(return_value=mock_status)
        mock_console.status.return_value.__exit__ = MagicMock(return_value=False)
        yield mock_console, mock_status


class TestPollingConstants:
    @pytest.mark.parametrize(
        "constant",
        [
            _SERVICE_STARTUP_POLL_INTERVAL,
            _MODEL_DISCOVERY_POLL_INTERVAL,
            _AGENT_DEPLOY_POLL_INTERVAL,
            _AGENT_API_READINESS_POLL_INTERVAL,
        ],
    )
    def test_poll_intervals_are_at_most_one_second(self, constant):
        assert constant <= 1


# ---------------------------------------------------------------------------
# Progress spinner tests — _wait_for_platform
# ---------------------------------------------------------------------------


class TestWaitForPlatformSpinner:
    _MOD = "nemo_platform_ext.cli.commands.setup"

    def test_shows_spinner_while_waiting(self, spinner_console):
        """console.status() should be used as a context manager during polling."""
        mock_console, _ = spinner_console
        with (
            patch(f"{self._MOD}._pause"),
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 0.5, 1, 1]),
            patch(f"{self._MOD}._check_platform_reachable", side_effect=[False, True]),
        ):
            result = _wait_for_platform("http://localhost:8080")

        assert result is True
        mock_console.status.assert_called_once()

    def test_spinner_updates_with_elapsed_time(self, spinner_console):
        """status.update() should include elapsed seconds."""
        _, mock_status = spinner_console
        with (
            patch(f"{self._MOD}._pause"),
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 2, 4, 4, 6, 6]),
            patch(f"{self._MOD}._check_platform_reachable", side_effect=[False, False, True]),
        ):
            _wait_for_platform("http://localhost:8080")

        update_texts = [c.args[0] for c in mock_status.update.call_args_list]
        assert any("2s" in t for t in update_texts), f"Expected elapsed time in updates: {update_texts}"

    def test_spinner_stops_on_timeout(self, spinner_console):
        """Spinner exits cleanly and function returns False on timeout."""
        with (
            patch(f"{self._MOD}._pause"),
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 0, 200, 200]),
            patch(f"{self._MOD}._check_platform_reachable", return_value=False),
        ):
            result = _wait_for_platform("http://localhost:8080", timeout=120)

        assert result is False

    def test_uses_reduced_http_timeout(self, spinner_console):
        """Health check should use a short HTTP timeout (<=1.5s) for fast cycles."""
        with (
            patch(f"{self._MOD}._pause"),
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 0.5, 1]),
            patch(f"{self._MOD}._check_platform_reachable", side_effect=[True]) as mock_reachable,
        ):
            _wait_for_platform("http://localhost:8080")

        _, kwargs = mock_reachable.call_args
        assert kwargs.get("timeout", 999) <= 1.5

    def test_spinner_shows_last_loaded_service(self, spinner_console, tmp_path):
        """When a log_path is provided, the spinner should include the service name."""
        _, mock_status = spinner_console
        log = tmp_path / "services.log"
        log.write_text("[STARTUP] service:auth: 500ms\n[STARTUP] service:guardrails: 40000ms\n")

        with (
            patch(f"{self._MOD}._pause"),
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 5, 10, 10]),
            patch(f"{self._MOD}._check_platform_reachable", side_effect=[False, True]),
        ):
            result = _wait_for_platform("http://localhost:8080", log_path=log)

        assert result is True
        update_texts = [c.args[0] for c in mock_status.update.call_args_list]
        assert any("guardrails" in t for t in update_texts), f"Expected service name in updates: {update_texts}"

    def test_spinner_works_without_log_path(self, spinner_console):
        """Spinner should work normally when no log_path is provided."""
        _, mock_status = spinner_console
        with (
            patch(f"{self._MOD}._pause"),
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 2, 4, 4]),
            patch(f"{self._MOD}._check_platform_reachable", side_effect=[False, True]),
        ):
            result = _wait_for_platform("http://localhost:8080", log_path=None)

        assert result is True
        update_texts = [c.args[0] for c in mock_status.update.call_args_list]
        assert all("loaded" not in t for t in update_texts)

    def test_returns_false_immediately_when_process_exits(self, spinner_console):
        """Do not wait the full timeout when the service process already exited."""
        dead = MagicMock()
        dead.poll.return_value = 3
        with (
            patch(f"{self._MOD}._pause") as pause,
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 1, 2]),
            patch(f"{self._MOD}._check_platform_reachable") as reachable,
        ):
            result = _wait_for_platform("http://localhost:8080", timeout=120, proc=dead)

        assert result is False
        reachable.assert_not_called()
        pause.assert_not_called()


# ---------------------------------------------------------------------------
# _last_startup_service tests
# ---------------------------------------------------------------------------


class TestLastStartupService:
    def test_parses_last_service(self, tmp_path):
        log = tmp_path / "services.log"
        log.write_text(
            "[STARTUP] service:auth: 500ms\n[STARTUP] service:files: 3000ms\n[STARTUP] service:guardrails: 40000ms\n"
        )
        assert _last_startup_service(log) == "guardrails"

    def test_returns_empty_for_missing_file(self, tmp_path):
        assert _last_startup_service(tmp_path / "nonexistent.log") == ""

    def test_returns_empty_for_none(self):
        assert _last_startup_service(None) == ""

    def test_returns_empty_for_no_startup_lines(self, tmp_path):
        log = tmp_path / "services.log"
        log.write_text("some other log content\n")
        assert _last_startup_service(log) == ""


# ---------------------------------------------------------------------------
# Default timeout constant
# ---------------------------------------------------------------------------


class TestDefaultTimeout:
    def test_default_startup_timeout_is_240(self):
        assert _SERVICE_STARTUP_TIMEOUT_SECONDS == 240


# ---------------------------------------------------------------------------
# Progress spinner tests — _wait_for_models
# ---------------------------------------------------------------------------


class TestWaitForModelsSpinner:
    _MOD = "nemo_platform_ext.cli.commands.setup"

    def test_shows_spinner_during_model_discovery(self, spinner_console):
        """console.status() should be active during model discovery polling."""
        mock_console, _ = spinner_console
        client = MagicMock()
        provider = MagicMock()
        provider.served_models = []
        client.inference.providers.retrieve.return_value = provider

        with (
            patch(f"{self._MOD}._pause"),
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 1, 2, 100]),
        ):
            _wait_for_models(client, "nvidia-build", "default", round_seconds=5, max_rounds=1)

        mock_console.status.assert_called()

    def test_spinner_updates_with_elapsed_time(self, spinner_console):
        """status.update() should include elapsed seconds during model polling."""
        _, mock_status = spinner_console
        client = MagicMock()
        model = MagicMock()
        model.model_entity_id = "default/nvidia-llama"
        client.inference.providers.retrieve.side_effect = [
            MagicMock(served_models=[]),
            MagicMock(served_models=[model]),
        ]

        with (
            patch(f"{self._MOD}._pause"),
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 0, 1, 2, 3]),
        ):
            result = _wait_for_models(client, "nvidia-build", "default", round_seconds=30, max_rounds=1)

        assert result == ["default/nvidia-llama"]
        update_texts = [c.args[0] for c in mock_status.update.call_args_list]
        assert any("s)" in t for t in update_texts), f"Expected elapsed time in updates: {update_texts}"

    def test_continues_polling_when_served_models_lack_entity_ids(self, spinner_console):
        """Polling should continue when served_models exist but have no entity IDs."""
        client = MagicMock()
        model_without_id = MagicMock(spec=[])
        model_with_id = MagicMock()
        model_with_id.model_entity_id = "default/nvidia-llama"
        client.inference.providers.retrieve.side_effect = [
            MagicMock(served_models=[model_without_id]),
            MagicMock(served_models=[model_with_id]),
        ]

        with (
            patch(f"{self._MOD}._pause"),
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 0, 1, 2, 3]),
        ):
            result = _wait_for_models(client, "nvidia-build", "default", round_seconds=30, max_rounds=1)

        assert result == ["default/nvidia-llama"]


# ---------------------------------------------------------------------------
# Early exit tests — _wait_for_models
# ---------------------------------------------------------------------------


class TestWaitForModelsEarlyExit:
    _MOD = "nemo_platform_ext.cli.commands.setup"

    def test_exits_early_on_non_compliant_provider(self, spinner_console):
        """Returns [] immediately when provider status_message indicates non-compliant."""
        mock_console, _ = spinner_console
        client = MagicMock()
        provider = MagicMock()
        provider.served_models = []
        provider.status = "READY"
        provider.status_message = "Non-OpenAI compliant endpoint, model entity routing disabled"
        client.inference.providers.retrieve.return_value = provider

        with (
            patch(f"{self._MOD}._pause"),
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 0, 0]),
        ):
            result = _wait_for_models(
                client,
                "bad-provider",
                "default",
                host_url="https://inference.nvidia.com",
                round_seconds=30,
                max_rounds=2,
            )

        assert result == []
        assert client.inference.providers.retrieve.call_count == 1
        mock_console.print.assert_any_call(
            "\n  [yellow]![/yellow] Provider 'bad-provider' (https://inference.nvidia.com) "
            "returned a non-OpenAI compliant response from GET /v1/models."
        )

    def test_exits_early_on_error_status(self, spinner_console):
        """Returns [] immediately when provider status is ERROR."""
        _mock_console, _ = spinner_console
        client = MagicMock()
        provider = MagicMock()
        provider.served_models = []
        provider.status = "ERROR"
        provider.status_message = "Provider discovery failed: Gateway error (HTTP 502)"
        client.inference.providers.retrieve.return_value = provider

        with (
            patch(f"{self._MOD}._pause"),
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 0, 0]),
        ):
            result = _wait_for_models(
                client,
                "broken-provider",
                "default",
                host_url="https://example.com",
                round_seconds=30,
                max_rounds=2,
            )

        assert result == []
        assert client.inference.providers.retrieve.call_count == 1

    def test_exits_early_on_lost_status(self, spinner_console):
        """Returns [] immediately when provider status is LOST."""
        client = MagicMock()
        provider = MagicMock()
        provider.served_models = []
        provider.status = "LOST"
        provider.status_message = "Provider discovery permanently failed."
        client.inference.providers.retrieve.return_value = provider

        with (
            patch(f"{self._MOD}._pause"),
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 0, 0]),
        ):
            result = _wait_for_models(client, "lost-provider", "default", round_seconds=30, max_rounds=2)

        assert result == []

    def test_no_false_positive_on_created_status(self, spinner_console):
        """CREATED status with no status_message should NOT trigger early exit."""
        client = MagicMock()
        provider = MagicMock()
        provider.served_models = []
        provider.status = "CREATED"
        provider.status_message = None
        client.inference.providers.retrieve.return_value = provider

        # monotonic calls: start, deadline, while-check, elapsed, while-check, elapsed, while-check(exit)
        with (
            patch(f"{self._MOD}._pause"),
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 1, 1, 2, 2, 100]),
        ):
            result = _wait_for_models(client, "new-provider", "default", round_seconds=5, max_rounds=1)

        assert result == []
        assert client.inference.providers.retrieve.call_count > 1

    def test_models_found_before_non_compliant(self, spinner_console):
        """If models appear on the same poll, served_models are checked first."""
        client = MagicMock()
        model = MagicMock()
        model.model_entity_id = "default/my-model"
        provider = MagicMock()
        provider.served_models = [model]
        provider.status = "READY"
        provider.status_message = None
        client.inference.providers.retrieve.return_value = provider

        with (
            patch(f"{self._MOD}._pause"),
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 0, 0]),
        ):
            result = _wait_for_models(client, "good-provider", "default", round_seconds=30, max_rounds=1)

        assert result == ["default/my-model"]

    def test_host_url_omitted_in_warning(self, spinner_console):
        """When host_url is empty the warning should not include a parenthetical."""
        mock_console, _ = spinner_console
        client = MagicMock()
        provider = MagicMock()
        provider.served_models = []
        provider.status = "READY"
        provider.status_message = "Non-OpenAI compliant endpoint, model entity routing disabled"
        client.inference.providers.retrieve.return_value = provider

        with (
            patch(f"{self._MOD}._pause"),
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 0, 0]),
        ):
            _wait_for_models(client, "bad-provider", "default", round_seconds=30, max_rounds=1)

        printed = [str(c) for c in mock_console.print.call_args_list]
        assert not any("()" in p for p in printed)


# ---------------------------------------------------------------------------
# Direct agent API helper TLS
# ---------------------------------------------------------------------------


class TestAgentApiTLS:
    _MOD = "nemo_platform_ext.cli.commands.setup"

    def test_agent_exists_uses_context_certificate_authority_when_env_unset(self):
        resp = MagicMock()
        resp.status_code = 200
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(f"{self._MOD}.httpx.get", return_value=resp) as mock_get,
        ):
            assert (
                _agent_exists(
                    "https://nemo.example.com",
                    "default",
                    certificate_authority="/ctx/ca.pem",
                )
                is True
            )

        mock_get.assert_called_once_with(
            "https://nemo.example.com/apis/agents/v2/workspaces/default/agents/calculator-agent",
            headers=None,
            timeout=10.0,
            verify="/ctx/ca.pem",
        )

    def test_agents_api_ready_uses_context_certificate_authority_when_env_unset(self):
        resp = MagicMock()
        resp.status_code = 200
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(f"{self._MOD}.httpx.get", return_value=resp) as mock_get,
        ):
            assert (
                _agents_api_ready(
                    "https://nemo.example.com",
                    "default",
                    certificate_authority="/ctx/ca.pem",
                )
                is True
            )

        mock_get.assert_called_once_with(
            "https://nemo.example.com/apis/agents/v2/workspaces/default/agents",
            headers=None,
            timeout=3.0,
            verify="/ctx/ca.pem",
        )

    def test_deploy_demo_agent_uses_context_certificate_authority_for_httpx_calls(self, tmp_path, spinner_console):
        config = tmp_path / "calculator-agent.yml"
        config.write_text("llms: {}\n", encoding="utf-8")
        exists_resp = MagicMock()
        exists_resp.status_code = 404
        create_resp = MagicMock()
        create_resp.status_code = 200
        create_resp.raise_for_status = MagicMock()
        deploy_resp = MagicMock()
        deploy_resp.status_code = 200
        deploy_resp.raise_for_status = MagicMock()
        deploy_resp.json.return_value = {"name": "calculator-agent-abc12345"}
        status_resp = MagicMock()
        status_resp.status_code = 200
        status_resp.json.return_value = {"status": "running"}

        with (
            patch.dict("os.environ", {}, clear=True),
            patch(f"{self._MOD}.httpx.get", side_effect=[exists_resp, status_resp]) as mock_get,
            patch(f"{self._MOD}.httpx.post", side_effect=[create_resp, deploy_resp]) as mock_post,
            patch(f"{self._MOD}._pause"),
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 1, 2]),
        ):
            assert (
                _deploy_demo_agent(
                    "https://nemo.example.com",
                    "default",
                    config,
                    default_model="m",
                    certificate_authority="/ctx/ca.pem",
                )
                is True
            )

        assert [call.kwargs.get("verify") for call in mock_get.call_args_list] == ["/ctx/ca.pem", "/ctx/ca.pem"]
        assert [call.kwargs.get("verify") for call in mock_post.call_args_list] == ["/ctx/ca.pem", "/ctx/ca.pem"]


# ---------------------------------------------------------------------------
# Progress spinner tests — _deploy_demo_agent
# ---------------------------------------------------------------------------


class TestDeployDemoAgentSpinner:
    _MOD = "nemo_platform_ext.cli.commands.setup"

    def test_agent_config_path_finds_package_local_yaml(self, monkeypatch, tmp_path):
        """Packaged wheels bundle calculator-agent.yml inside the calculator_agent package."""
        package_dir = tmp_path / "calculator_agent"
        package_dir.mkdir()
        (package_dir / "__init__.py").write_text("", encoding="utf-8")
        config = package_dir / "calculator-agent.yml"
        config.write_text("llms: {}\n", encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "calculator_agent", raising=False)

        assert _agent_config_path() == config

    def test_agent_config_path_finds_namespace_package_yaml(self, monkeypatch, tmp_path):
        """calculator_agent is an implicit namespace package, so __file__ may be None."""
        package_dir = tmp_path / "calculator_agent"
        package_dir.mkdir()
        config = package_dir / "calculator-agent.yml"
        config.write_text("llms: {}\n", encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "calculator_agent", raising=False)

        assert _agent_config_path() == config

    def _mock_deploy_responses(self, *, status_sequence):
        """Build httpx response mocks for create + deployment status polling.

        The deploy POST returns the full entity including its unique ``name``
        so the polling loop can GET that specific deployment by name.
        """
        create_resp = MagicMock()
        create_resp.status_code = 200
        create_resp.raise_for_status = MagicMock()

        deploy_resp = MagicMock()
        deploy_resp.status_code = 200
        deploy_resp.raise_for_status = MagicMock()
        deploy_resp.json.return_value = {
            "name": "calculator-agent-abc12345",
            "agent": "calculator-agent",
            "status": "pending",
        }

        status_resps = []
        for s in status_sequence:
            r = MagicMock()
            r.status_code = 200
            r.json.return_value = {"name": "calculator-agent-abc12345", "agent": "calculator-agent", "status": s}
            status_resps.append(r)

        return [create_resp, deploy_resp] + status_resps

    def test_shows_spinner_during_deployment_wait(self, tmp_path, spinner_console):
        """console.status() should be active while polling deployment status."""
        config = tmp_path / "calculator-agent.yml"
        config.write_text("llms: {}\n")

        responses = self._mock_deploy_responses(status_sequence=["pending", "running"])
        mock_console, _ = spinner_console

        with (
            patch(f"{self._MOD}.httpx.get", side_effect=responses[2:]),
            patch(f"{self._MOD}.httpx.post", side_effect=responses[:2]),
            patch(f"{self._MOD}._agent_exists", return_value=False),
            patch(f"{self._MOD}._pause"),
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 1, 2, 3]),
        ):
            result = _deploy_demo_agent("http://localhost:8080", "default", config, default_model="m")

        assert result is True
        mock_console.status.assert_called()

    def test_spinner_updates_with_elapsed_time(self, tmp_path, spinner_console):
        """status.update() should include elapsed seconds during deploy polling."""
        config = tmp_path / "calculator-agent.yml"
        config.write_text("llms: {}\n")

        responses = self._mock_deploy_responses(status_sequence=["pending", "pending", "running"])
        _, mock_status = spinner_console

        with (
            patch(f"{self._MOD}.httpx.get", side_effect=responses[2:]),
            patch(f"{self._MOD}.httpx.post", side_effect=responses[:2]),
            patch(f"{self._MOD}._agent_exists", return_value=False),
            patch(f"{self._MOD}._pause"),
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 1, 2, 3, 4, 5]),
        ):
            _deploy_demo_agent("http://localhost:8080", "default", config, default_model="m")

        update_texts = [c.args[0] for c in mock_status.update.call_args_list]
        assert any("s)" in t for t in update_texts), f"Expected elapsed time in updates: {update_texts}"

    def test_uses_reduced_http_timeout(self, tmp_path, spinner_console):
        """Deployment status GET should use a short HTTP timeout (<=3s)."""
        config = tmp_path / "calculator-agent.yml"
        config.write_text("llms: {}\n")

        responses = self._mock_deploy_responses(status_sequence=["running"])

        with (
            patch(f"{self._MOD}.httpx.get", side_effect=responses[2:]) as mock_get,
            patch(f"{self._MOD}.httpx.post", side_effect=responses[:2]),
            patch(f"{self._MOD}._agent_exists", return_value=False),
            patch(f"{self._MOD}._pause"),
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 1, 2]),
        ):
            _deploy_demo_agent("http://localhost:8080", "default", config, default_model="m")

        get_calls = mock_get.call_args_list
        for c in get_calls:
            assert c.kwargs.get("timeout", 999) <= 3.0

    def test_polls_specific_deployment_by_name(self, tmp_path, spinner_console):
        """The poll must GET the specific deployment, not list all deployments."""
        config = tmp_path / "calculator-agent.yml"
        config.write_text("llms: {}\n")

        responses = self._mock_deploy_responses(status_sequence=["running"])

        with (
            patch(f"{self._MOD}.httpx.get", side_effect=responses[2:]) as mock_get,
            patch(f"{self._MOD}.httpx.post", side_effect=responses[:2]),
            patch(f"{self._MOD}._agent_exists", return_value=False),
            patch(f"{self._MOD}._pause"),
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 1, 2]),
        ):
            result = _deploy_demo_agent("http://localhost:8080", "default", config, default_model="m")

        assert result is True
        url = mock_get.call_args_list[0].args[0]
        assert "/deployments/calculator-agent-abc12345" in url

    def test_expands_default_model_placeholder_on_create(self, tmp_path, spinner_console):
        """The built-in YAML uses ``${NEMO_DEFAULT_MODEL}``; resolve before POST
        because the agents service has no user context to resolve it itself.
        Regression for AIRCORE-601.
        """
        config = tmp_path / "calculator-agent.yml"
        config.write_text("llms:\n  agent:\n    _type: openai\n    model_name: ${NEMO_DEFAULT_MODEL}\n")

        responses = self._mock_deploy_responses(status_sequence=["running"])

        with (
            patch(f"{self._MOD}.httpx.get", side_effect=responses[2:]),
            patch(f"{self._MOD}.httpx.post", side_effect=responses[:2]) as mock_post,
            patch(f"{self._MOD}._agent_exists", return_value=False),
            patch(f"{self._MOD}._pause"),
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 1, 2]),
        ):
            _deploy_demo_agent(
                "http://localhost:8080",
                "default",
                config,
                default_model="nvidia-nemotron-3-super-v3",
            )

        create_call = mock_post.call_args_list[0]
        sent_config = create_call.kwargs["json"]["config"]
        assert sent_config["llms"]["agent"]["model_name"] == "nvidia-nemotron-3-super-v3"


# ---------------------------------------------------------------------------
# `_maybe_deploy_agent` guards
# ---------------------------------------------------------------------------


class TestMaybeDeployAgentGuards:
    _MOD = "nemo_platform_ext.cli.commands.setup"

    def test_skips_deploy_when_default_model_missing(self):
        """No default model selected → skip deploy so the agent service never
        stores an unresolved ``${NEMO_DEFAULT_MODEL}``. Regression for AIRCORE-601.
        """
        with (
            patch(f"{self._MOD}._agents_plugin_available", return_value=True),
            patch(f"{self._MOD}._deploy_demo_agent") as mock_deploy,
        ):
            result = _maybe_deploy_agent(
                "http://localhost:8080",
                "default",
                auto=True,
                deploy_agent=True,
                default_model=None,
            )

        mock_deploy.assert_not_called()
        assert result is False

    def test_returns_false_when_plugin_unavailable(self):
        with patch(f"{self._MOD}._agents_plugin_available", return_value=False):
            result = _maybe_deploy_agent(
                "http://localhost:8080", "default", auto=True, deploy_agent=True, default_model="m"
            )
        assert result is False

    def test_returns_false_in_auto_mode_without_explicit_flag(self):
        with patch(f"{self._MOD}._agents_plugin_available", return_value=True):
            result = _maybe_deploy_agent(
                "http://localhost:8080", "default", auto=True, deploy_agent=None, default_model="m"
            )
        assert result is False

    def test_returns_false_when_config_path_missing(self):
        with (
            patch(f"{self._MOD}._agents_plugin_available", return_value=True),
            patch(f"{self._MOD}._agent_config_path", return_value=None),
        ):
            result = _maybe_deploy_agent(
                "http://localhost:8080", "default", auto=True, deploy_agent=True, default_model="m"
            )
        assert result is False

    def test_returns_true_on_successful_deploy(self, spinner_console):
        with (
            patch(f"{self._MOD}._agents_plugin_available", return_value=True),
            patch(f"{self._MOD}._agent_config_path", return_value=MagicMock()),
            patch(f"{self._MOD}._agents_api_ready", return_value=True),
            patch(f"{self._MOD}._deploy_demo_agent", return_value=True),
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 1]),
        ):
            result = _maybe_deploy_agent(
                "http://localhost:8080", "default", auto=False, deploy_agent=True, default_model="m"
            )
        assert result is True

    def test_returns_false_when_deploy_raises(self, spinner_console):
        with (
            patch(f"{self._MOD}._agents_plugin_available", return_value=True),
            patch(f"{self._MOD}._agent_config_path", return_value=MagicMock()),
            patch(f"{self._MOD}._agents_api_ready", return_value=True),
            patch(f"{self._MOD}._deploy_demo_agent", side_effect=RuntimeError("boom")),
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 1]),
        ):
            result = _maybe_deploy_agent(
                "http://localhost:8080", "default", auto=False, deploy_agent=True, default_model="m"
            )
        assert result is False


# ---------------------------------------------------------------------------
# Progress spinner tests — agents API readiness in _maybe_deploy_agent
# ---------------------------------------------------------------------------


class TestAgentsApiReadinessSpinner:
    _MOD = "nemo_platform_ext.cli.commands.setup"

    def test_shows_spinner_while_waiting_for_agents_api(self, spinner_console):
        """console.status() should be active while waiting for agents API readiness."""
        mock_console, _ = spinner_console
        with (
            patch(f"{self._MOD}._agents_plugin_available", return_value=True),
            patch(f"{self._MOD}._agent_config_path", return_value=MagicMock()),
            patch(f"{self._MOD}._agents_api_ready", side_effect=[False, True]),
            patch(f"{self._MOD}._deploy_demo_agent", return_value=True),
            patch(f"{self._MOD}._pause"),
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 1, 2, 3]),
        ):
            _maybe_deploy_agent("http://localhost:8080", "default", auto=False, deploy_agent=True, default_model="m")

        mock_console.status.assert_called()

    def test_uses_poll_interval_constant(self, spinner_console):
        """Should use _AGENT_API_READINESS_POLL_INTERVAL, not a hardcoded value."""
        with (
            patch(f"{self._MOD}._agents_plugin_available", return_value=True),
            patch(f"{self._MOD}._agent_config_path", return_value=MagicMock()),
            patch(f"{self._MOD}._agents_api_ready", side_effect=[False, True]),
            patch(f"{self._MOD}._deploy_demo_agent", return_value=True),
            patch(f"{self._MOD}._pause") as mock_pause,
            patch(f"{self._MOD}.time.monotonic", side_effect=[0, 0, 1, 2, 3]),
        ):
            _maybe_deploy_agent("http://localhost:8080", "default", auto=False, deploy_agent=True, default_model="m")

        pause_values = [c.args[0] for c in mock_pause.call_args_list]
        assert all(v == _AGENT_API_READINESS_POLL_INTERVAL for v in pause_values)


# ---------------------------------------------------------------------------
# Custom provider prompt
# ---------------------------------------------------------------------------


class TestPromptCustomProvider:
    _MOD = "nemo_platform_ext.cli.commands.setup"

    def test_prompt_passes_validator(self):
        """The provider name prompt should include a validator."""
        with (
            patch(f"{self._MOD}.prompt_text", return_value="my-provider") as mock_prompt,
            patch(f"{self._MOD}.prompt_password", return_value=""),
        ):
            _prompt_custom_provider()
        name_call = mock_prompt.call_args_list[0]
        assert name_call.kwargs.get("validator") is not None

    def test_prompt_passes_hint(self):
        """The provider name prompt should include a naming hint."""
        with (
            patch(f"{self._MOD}.prompt_text", return_value="my-provider") as mock_prompt,
            patch(f"{self._MOD}.prompt_password", return_value=""),
        ):
            _prompt_custom_provider()
        name_call = mock_prompt.call_args_list[0]
        hint = name_call.kwargs.get("hint")
        assert hint is not None
        assert "lowercase" in hint.lower()


# ---------------------------------------------------------------------------
# setup_command entry-path helpers
# ---------------------------------------------------------------------------


def _make_setup_command_ctx(
    *,
    base_url: str = "http://localhost:8080",
    workspace: str = "default",
    certificate_authority: str | None = None,
    workspace_source: ParameterSource = ParameterSource.DEFAULT,
) -> tuple[MagicMock, MagicMock]:
    """Build a typer Context + CLIContext pair for invoking ``setup_command``."""
    ctx = MagicMock(spec=typer.Context)
    ctx.get_parameter_source.return_value = workspace_source
    cli_context = MagicMock()
    cli_context.overrides = {}
    cli_context.get_base_url.return_value = base_url
    cli_context.get_sdk_context.return_value = Context(
        context_name="default",
        cluster=Cluster(name="default-cluster", base_url=base_url, certificate_authority=certificate_authority),
        user=NoAuthUser(name="default-user"),
        workspace=workspace,
        preferences={},
    )
    cli_context.get_client.return_value.workspaces.retrieve.return_value = MagicMock()
    ctx.obj = cli_context
    return ctx, cli_context


@contextmanager
def _patch_setup_command(
    *,
    interactive: bool = True,
    maybe_start_services: object | None = None,
    remote_url: str | None = None,
    include_config_write: bool = False,
    auto_mode: bool = False,
) -> Iterator[SimpleNamespace]:
    """Patch the common ``setup_command`` entry path and yield named mocks.

    When *remote_url* is set, also patches the connect-remote branch helpers and
    defaults ``_maybe_start_services`` to return ``\"connect_remote\"``.
    """
    maybe_start_kwargs: dict[str, object] = {}
    if maybe_start_services is None:
        if remote_url is not None:
            maybe_start_kwargs["return_value"] = "connect_remote"
    elif isinstance(maybe_start_services, list):
        maybe_start_kwargs["side_effect"] = maybe_start_services
    elif isinstance(maybe_start_services, MagicMock):
        maybe_start_kwargs["new"] = maybe_start_services
    else:
        maybe_start_kwargs["return_value"] = maybe_start_services

    with ExitStack() as stack:
        mocks = SimpleNamespace(
            is_interactive=stack.enter_context(patch(f"{SETUP_MOD}.is_interactive", return_value=interactive)),
            maybe_start_services=stack.enter_context(patch(f"{SETUP_MOD}._maybe_start_services", **maybe_start_kwargs)),
            check_reachable=stack.enter_context(
                patch(f"{SETUP_MOD}._check_platform_reachable_with_retries", return_value=True)
            ),
            bootstrap=stack.enter_context(patch(f"{SETUP_MOD}._bootstrap_config_if_missing")),
            run_interactive=stack.enter_context(patch(f"{SETUP_MOD}._run_interactive_mode")),
            prompt_remote=None,
            configure_remote=None,
            configure_local=stack.enter_context(patch(f"{SETUP_MOD}._configure_local_connection")),
            ensure_auth=None,
            config_write=None,
            run_auto=None,
        )
        if remote_url is not None:
            mocks.prompt_remote = stack.enter_context(
                patch(f"{SETUP_MOD}._prompt_remote_base_url", return_value=remote_url)
            )
            mocks.configure_remote = stack.enter_context(patch(f"{SETUP_MOD}._configure_remote_connection"))
            mocks.ensure_auth = stack.enter_context(patch(f"{SETUP_MOD}._ensure_platform_auth"))
        if include_config_write:
            mocks.config_write = stack.enter_context(patch(f"{SETUP_MOD}.Config.write"))
        if auto_mode:
            mocks.run_auto = stack.enter_context(patch(f"{SETUP_MOD}._run_auto_mode"))
        yield mocks


# ---------------------------------------------------------------------------
# Python version guard
# ---------------------------------------------------------------------------


class TestRequireSupportedPython:
    def test_accepts_3_12_and_3_13(self):
        for version in ((3, 12, 11, "final", 0), (3, 13, 5, "final", 0)):
            with patch.object(setup_commands.sys, "version_info", version):
                _require_supported_python()

    def test_rejects_3_14_and_3_11(self, capsys):
        for version in ((3, 14, 0, "final", 0), (3, 11, 13, "final", 0)):
            with (
                patch.object(setup_commands.sys, "version_info", version),
                pytest.raises(typer.Exit) as exc_info,
            ):
                _require_supported_python()
            assert exc_info.value.exit_code == 1
            err = capsys.readouterr().err
            assert "Unsupported Python" in err
            assert "Python 3.12-3.13" in err
            assert "--python 3.13" in err

    def test_setup_command_exits_before_starting_services(self):
        ctx, _cli_context = _make_setup_command_ctx()
        with (
            _patch_setup_command() as mocks,
            patch.object(setup_commands.sys, "version_info", (3, 14, 0, "final", 0)),
            pytest.raises(typer.Exit) as exc_info,
        ):
            setup_command(ctx, auto=False)
        assert exc_info.value.exit_code == 1
        mocks.maybe_start_services.assert_not_called()


# ---------------------------------------------------------------------------
# Non-TTY early exit guard
# ---------------------------------------------------------------------------


class TestNonTtyEarlyExit:
    """setup_command must exit(1) when stdin is not a TTY and --auto is not passed."""

    def test_exits_when_non_tty_without_auto(self):
        ctx, _cli_context = _make_setup_command_ctx()
        with (
            _patch_setup_command(interactive=False),
            pytest.raises(typer.Exit) as exc_info,
        ):
            setup_command(ctx, auto=False)
        assert exc_info.value.exit_code == 1

    def test_proceeds_when_non_tty_with_auto(self):
        ctx, _cli_context = _make_setup_command_ctx()
        with _patch_setup_command(interactive=False, auto_mode=True):
            setup_command(ctx, auto=True)

    def test_proceeds_when_tty_without_auto(self):
        ctx, _cli_context = _make_setup_command_ctx()
        with _patch_setup_command():
            setup_command(ctx, auto=False)

    def test_cancelling_initial_connection_prompt_exits_cleanly(self, capsys):
        ctx, _cli_context = _make_setup_command_ctx()
        with (
            _patch_setup_command(maybe_start_services=MagicMock(side_effect=UserCancelled)),
            pytest.raises(typer.Exit) as exc_info,
        ):
            setup_command(ctx, auto=False)

        assert exc_info.value.exit_code == 0
        assert "Setup cancelled" in capsys.readouterr().err

    def test_resume_runs_idempotent_setup_path(self, capsys):
        ctx, _cli_context = _make_setup_command_ctx()
        with _patch_setup_command() as mocks:
            setup_command(ctx, resume=True)

        assert "Retrying setup using the normal idempotent setup path" in capsys.readouterr().err
        mocks.run_interactive.assert_called_once()


class TestSetupCommandRemoteFlow:
    def test_remote_choice_connects_before_continuing_setup(self):
        ctx, cli_context = _make_setup_command_ctx()
        with _patch_setup_command(remote_url="https://remote.example.com") as mocks:
            setup_command(ctx)

        mocks.prompt_remote.assert_called_once_with(
            default_url="http://localhost:8080",
            certificate_authority=None,
        )
        mocks.configure_remote.assert_called_once_with(cli_context, "https://remote.example.com", "default")
        mocks.ensure_auth.assert_called_once_with(cli_context)
        assert mocks.run_interactive.call_args.args[3] == "https://remote.example.com"

    def test_saved_certificate_authority_flows_through_setup_probes(self):
        ctx, _cli_context = _make_setup_command_ctx(
            base_url="https://nemo.example.com",
            certificate_authority="/ctx/ca.pem",
        )
        with _patch_setup_command(maybe_start_services="ready") as mocks:
            setup_command(ctx)

        mocks.maybe_start_services.assert_called_once_with(
            "https://nemo.example.com",
            False,
            None,
            timeout=_SERVICE_STARTUP_TIMEOUT_SECONDS,
            certificate_authority="/ctx/ca.pem",
        )
        mocks.check_reachable.assert_called_once_with(
            "https://nemo.example.com",
            certificate_authority="/ctx/ca.pem",
        )
        assert mocks.run_interactive.call_args.kwargs["certificate_authority"] == "/ctx/ca.pem"

    def test_remote_choice_preserves_active_workspace_when_flag_omitted(self):
        ctx, cli_context = _make_setup_command_ctx(workspace="team-a")
        with _patch_setup_command(remote_url="https://remote.example.com") as mocks:
            setup_command(ctx)

        mocks.configure_remote.assert_called_once_with(cli_context, "https://remote.example.com", "team-a")
        assert mocks.bootstrap.call_args_list == [
            call("https://remote.example.com", "default"),
            call("https://remote.example.com", "team-a"),
        ]
        assert mocks.run_interactive.call_args.args[2] == "team-a"

    def test_remote_choice_uses_explicit_workspace_flag(self):
        ctx, cli_context = _make_setup_command_ctx(
            workspace="team-a",
            workspace_source=ParameterSource.COMMANDLINE,
        )
        with _patch_setup_command(remote_url="https://remote.example.com") as mocks:
            setup_command(ctx, workspace="shared-workspace")

        mocks.configure_remote.assert_called_once_with(cli_context, "https://remote.example.com", "shared-workspace")
        assert mocks.run_interactive.call_args.args[2] == "shared-workspace"

    def test_start_local_persists_default_url_and_starts_services(self):
        ctx, cli_context = _make_setup_command_ctx(base_url="https://remote.example.com")
        with _patch_setup_command(
            maybe_start_services=["start_local", "ready"],
        ) as mocks:
            setup_command(ctx)

        assert mocks.maybe_start_services.call_count == 2
        assert mocks.maybe_start_services.call_args_list[1] == call(
            DEFAULT_BASE_URL,
            False,
            start_services=True,
            timeout=_SERVICE_STARTUP_TIMEOUT_SECONDS,
            certificate_authority=None,
        )
        mocks.configure_local.assert_called_once_with(cli_context, "default")
        assert mocks.run_interactive.call_args.args[3] == DEFAULT_BASE_URL


# ---------------------------------------------------------------------------
# Controller health check
# ---------------------------------------------------------------------------


def _status_response(*, healthy: bool = True, controller_status: dict | None = None):
    """Build a mock httpx.Response for GET /status."""
    body = {
        "status": "healthy" if healthy else "degraded",
        "services": {"ready": ["auth", "models"], "not_ready": []},
        "controllers": {
            "healthy": healthy,
            "status": controller_status if controller_status is not None else {},
        },
    }
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = body
    return resp


class TestCheckControllerHealth:
    def test_healthy_controllers(self):
        resp = _status_response(healthy=True, controller_status={"models_controller": True})
        with patch(f"{SETUP_MOD}.httpx.get", return_value=resp):
            ok, msg = _check_controller_health("http://localhost:8080")
        assert ok is True
        assert msg == ""

    def test_uses_context_certificate_authority_when_env_unset(self):
        resp = _status_response(healthy=True, controller_status={"models_controller": True})
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(f"{SETUP_MOD}.httpx.get", return_value=resp) as mock_get,
        ):
            ok, msg = _check_controller_health(
                "https://nemo.example.com",
                certificate_authority="/ctx/ca.pem",
            )

        assert ok is True
        assert msg == ""
        mock_get.assert_called_once_with(
            "https://nemo.example.com/status",
            timeout=5.0,
            verify="/ctx/ca.pem",
        )

    def test_env_certificate_authority_overrides_context_certificate_authority(self):
        resp = _status_response(healthy=True, controller_status={"models_controller": True})
        with (
            patch.dict("os.environ", {"NMP_CLIENT_SSL_CERT_FILE": "/env/ca.pem"}, clear=True),
            patch(f"{SETUP_MOD}.httpx.get", return_value=resp) as mock_get,
        ):
            ok, msg = _check_controller_health(
                "https://nemo.example.com",
                certificate_authority="/ctx/ca.pem",
            )

        assert ok is True
        assert msg == ""
        mock_get.assert_called_once_with(
            "https://nemo.example.com/status",
            timeout=5.0,
            verify="/env/ca.pem",
        )

    def test_unhealthy_controller(self):
        # A controller shows up as unhealthy immediately (before it has registered
        # its control loop), so an in-progress startup can legitimately look
        # unhealthy on the first call — this must retry once before giving up,
        # same as the empty-status_map case below.
        resp = _status_response(healthy=False, controller_status={"models_controller": False})
        with (
            patch(f"{SETUP_MOD}.httpx.get", return_value=resp),
            patch(f"{SETUP_MOD}._pause") as mock_pause,
        ):
            ok, msg = _check_controller_health("http://localhost:8080")
        assert ok is False
        assert "models_controller" in msg
        mock_pause.assert_called_once_with(_CONTROLLER_HEALTH_RETRY_DELAY)

    def test_unhealthy_controller_recovers_after_retry(self):
        starting_resp = _status_response(healthy=False, controller_status={"models_controller": False})
        healthy_resp = _status_response(healthy=True, controller_status={"models_controller": True})
        with (
            patch(f"{SETUP_MOD}.httpx.get", side_effect=[starting_resp, healthy_resp]),
            patch(f"{SETUP_MOD}._pause") as mock_pause,
        ):
            ok, msg = _check_controller_health("http://localhost:8080")
        assert ok is True
        assert msg == ""
        mock_pause.assert_called_once_with(_CONTROLLER_HEALTH_RETRY_DELAY)

    def test_empty_controllers_retries_then_succeeds(self):
        empty_resp = _status_response(healthy=True, controller_status={})
        populated_resp = _status_response(healthy=True, controller_status={"models_controller": True})
        with (
            patch(f"{SETUP_MOD}.httpx.get", side_effect=[empty_resp, populated_resp]),
            patch(f"{SETUP_MOD}._pause") as mock_pause,
        ):
            ok, msg = _check_controller_health("http://localhost:8080")
        assert ok is True
        assert msg == ""
        mock_pause.assert_called_once_with(_CONTROLLER_HEALTH_RETRY_DELAY)

    def test_empty_controllers_retries_then_still_empty(self):
        empty_resp = _status_response(healthy=True, controller_status={})
        with (
            patch(f"{SETUP_MOD}.httpx.get", return_value=empty_resp),
            patch(f"{SETUP_MOD}._pause"),
        ):
            ok, msg = _check_controller_health("http://localhost:8080")
        assert ok is False
        assert "no controllers" in msg.lower()

    def test_endpoint_unreachable(self):
        with patch(f"{SETUP_MOD}.httpx.get", side_effect=httpx.ConnectError("refused")):
            ok, msg = _check_controller_health("http://localhost:8080")
        assert ok is False
        assert "could not reach" in msg.lower()

    def test_non_200_response(self):
        resp = MagicMock()
        resp.status_code = 500
        with patch(f"{SETUP_MOD}.httpx.get", return_value=resp):
            ok, _ = _check_controller_health("http://localhost:8080")
        assert ok is False

    def test_status_not_published_on_hosted_deployment(self):
        status_resp = MagicMock()
        status_resp.status_code = 404
        cluster_resp = MagicMock()
        cluster_resp.status_code = 200

        def _get(url, **kwargs):
            if url.endswith("/status"):
                return status_resp
            if url.endswith("/cluster-info"):
                return cluster_resp
            raise AssertionError(f"unexpected url: {url}")

        with patch(f"{SETUP_MOD}.httpx.get", side_effect=_get):
            ok, msg = _check_controller_health("https://nemo-platform-freeplay.dev.aire.nvidia.com")
        assert ok is True
        assert "does not publish /status" in msg

    def test_status_404_without_cluster_info_is_unhealthy(self):
        status_resp = MagicMock()
        status_resp.status_code = 404
        cluster_resp = MagicMock()
        cluster_resp.status_code = 404

        def _get(url, **kwargs):
            if url.endswith("/status"):
                return status_resp
            if url.endswith("/cluster-info"):
                return cluster_resp
            raise AssertionError(f"unexpected url: {url}")

        with patch(f"{SETUP_MOD}.httpx.get", side_effect=_get):
            ok, msg = _check_controller_health("http://localhost:8080")
        assert ok is False
        assert "404" in msg

    def test_invalid_json_response(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("not json")
        with patch(f"{SETUP_MOD}.httpx.get", return_value=resp):
            ok, msg = _check_controller_health("http://localhost:8080")
        assert ok is False
        assert "invalid json" in msg.lower()

    def test_malformed_controllers_payload(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"status": "healthy", "controllers": None}
        with patch(f"{SETUP_MOD}.httpx.get", return_value=resp):
            ok, msg = _check_controller_health("http://localhost:8080")
        assert ok is False
        assert "invalid" in msg.lower()

    def test_malformed_status_list_treated_as_empty(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"status": "healthy", "controllers": {"healthy": True, "status": [1, 2]}}
        with (
            patch(f"{SETUP_MOD}.httpx.get", return_value=resp),
            patch(f"{SETUP_MOD}._pause"),
        ):
            ok, msg = _check_controller_health("http://localhost:8080")
        assert ok is False
        assert "no controllers" in msg.lower()


class TestVerifyPlatformHealth:
    def test_healthy_returns_true(self):
        with patch(f"{SETUP_MOD}._check_controller_health", return_value=(True, "")) as mock_health:
            result = _verify_platform_health("http://localhost:8080")
        assert result is True
        mock_health.assert_called_once_with("http://localhost:8080", certificate_authority=None)

    def test_passes_certificate_authority_to_controller_health(self):
        with patch(f"{SETUP_MOD}._check_controller_health", return_value=(True, "")) as mock_health:
            result = _verify_platform_health("https://nemo.example.com", certificate_authority="/ctx/ca.pem")

        assert result is True
        mock_health.assert_called_once_with("https://nemo.example.com", certificate_authority="/ctx/ca.pem")

    def test_missing_status_endpoint_returns_true_with_info(self):
        with (
            patch(
                f"{SETUP_MOD}._check_controller_health",
                return_value=(True, "Hosted deployment does not publish /status."),
            ),
            patch(f"{SETUP_MOD}.console") as mock_console,
        ):
            result = _verify_platform_health("https://nemo-platform-freeplay.dev.aire.nvidia.com")
        assert result is True
        printed = " ".join(str(c) for c in mock_console.print.call_args_list)
        assert "does not publish /status" in printed
        assert "could not be verified" not in printed
        assert "yellow" not in printed.lower()

    def test_unhealthy_prints_red_error(self):
        with (
            patch(f"{SETUP_MOD}._check_controller_health", return_value=(False, "models_controller unhealthy")),
            patch(f"{SETUP_MOD}.console") as mock_console,
        ):
            result = _verify_platform_health("http://localhost:8080")
        assert result is False
        printed = " ".join(str(c) for c in mock_console.print.call_args_list)
        assert "red" in printed.lower()
        assert "nemo services run" in printed

    def test_empty_after_retry_prints_yellow_warning(self):
        with (
            patch(f"{SETUP_MOD}._check_controller_health", return_value=(False, "no controllers reported status")),
            patch(f"{SETUP_MOD}.console") as mock_console,
        ):
            result = _verify_platform_health("http://localhost:8080")
        assert result is False
        printed = " ".join(str(c) for c in mock_console.print.call_args_list)
        assert "yellow" in printed.lower() or "nemo services status" in printed


class TestRenderOnboardingCard:
    def test_optimize_card_includes_note(self):
        with patch(f"{SETUP_MOD}.console") as mock_console:
            _render_onboarding_card("optimize")

        panel = mock_console.print.call_args_list[0].args[0]
        assert "agent's project directory" in panel.renderable

    def test_optimize_card_uses_published_agents_docs_url(self):
        with patch(f"{SETUP_MOD}.console") as mock_console:
            _render_onboarding_card("optimize")

        panel = mock_console.print.call_args_list[0].args[0]
        content = panel.renderable
        assert "https://docs.nvidia.com/nemo-platform/documentation/agents" in content
        assert "https://docs.nvidia.com/nemo-platform/agents" not in content

    def test_explore_card_contains_skill_prompt_and_docs(self):
        with patch(f"{SETUP_MOD}.console") as mock_console:
            _render_onboarding_card("explore")

        panel = mock_console.print.call_args_list[0].args[0]
        content = panel.renderable
        assert "What can I do with NeMo Platform?" in content
        assert "docs.nvidia.com/nemo-platform" in content

    def test_unknown_value_is_silently_skipped(self):
        with patch(f"{SETUP_MOD}.console") as mock_console:
            _render_onboarding_card("nonexistent")

        mock_console.print.assert_not_called()

    def test_all_paths_render_without_error(self):
        for path in ONBOARDING_PATHS:
            with patch(f"{SETUP_MOD}.console") as mock_console:
                _render_onboarding_card(path.value)

            panel = mock_console.print.call_args_list[0].args[0]
            assert hasattr(panel, "renderable")


class TestPrintOnboarding:
    def test_shows_setup_complete_and_choice(self):
        with (
            patch(f"{SETUP_MOD}._verify_platform_health", return_value=True),
            patch(f"{SETUP_MOD}.prompt_choice", return_value="optimize") as mock_choice,
            patch(f"{SETUP_MOD}._render_onboarding_card") as mock_card,
            patch(f"{SETUP_MOD}.console") as mock_console,
        ):
            _print_onboarding("http://localhost:8080", "nvidia-build", "default/some-model")

        printed = " ".join(str(c) for c in mock_console.print.call_args_list)
        assert "Setup complete" in printed
        assert "nvidia-build" in printed
        mock_choice.assert_called_once()
        mock_card.assert_called_once_with("optimize")

    def test_explore_path_renders_card(self):
        with (
            patch(f"{SETUP_MOD}._verify_platform_health", return_value=True),
            patch(f"{SETUP_MOD}.prompt_choice", return_value="explore"),
            patch(f"{SETUP_MOD}._render_onboarding_card") as mock_card,
            patch(f"{SETUP_MOD}.console"),
        ):
            _print_onboarding("http://localhost:8080", "nvidia-build", None)

        mock_card.assert_called_once_with("explore")

    def test_unhealthy_platform_exits(self):
        with (
            patch(f"{SETUP_MOD}._verify_platform_health", return_value=False),
            pytest.raises((typer.Exit, SystemExit)),
        ):
            _print_onboarding("http://localhost:8080", "nvidia-build", None)

    def test_default_model_shown_when_present(self):
        with (
            patch(f"{SETUP_MOD}._verify_platform_health", return_value=True),
            patch(f"{SETUP_MOD}.prompt_choice", return_value="explore"),
            patch(f"{SETUP_MOD}._render_onboarding_card"),
            patch(f"{SETUP_MOD}.console") as mock_console,
        ):
            _print_onboarding("http://localhost:8080", "nvidia-build", "default/llama-3-3")

        printed = " ".join(str(c) for c in mock_console.print.call_args_list)
        assert "llama-3-3" in printed

    def test_demo_agent_shown_when_deployed(self):
        with (
            patch(f"{SETUP_MOD}._verify_platform_health", return_value=True),
            patch(f"{SETUP_MOD}.prompt_choice", return_value="explore"),
            patch(f"{SETUP_MOD}._render_onboarding_card"),
            patch(f"{SETUP_MOD}.console") as mock_console,
        ):
            _print_onboarding("http://localhost:8080", "nvidia-build", "m", demo_deployed=True)

        printed = " ".join(str(c) for c in mock_console.print.call_args_list)
        assert "calculator-agent" in printed

    def test_demo_agent_hidden_when_not_deployed(self):
        with (
            patch(f"{SETUP_MOD}._verify_platform_health", return_value=True),
            patch(f"{SETUP_MOD}.prompt_choice", return_value="explore"),
            patch(f"{SETUP_MOD}._render_onboarding_card"),
            patch(f"{SETUP_MOD}.console") as mock_console,
        ):
            _print_onboarding("http://localhost:8080", "nvidia-build", "m", demo_deployed=False)

        printed = " ".join(str(c) for c in mock_console.print.call_args_list)
        assert "calculator-agent" not in printed
