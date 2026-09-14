# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_agents_plugin.utils.

Covers:
- inject_gateway_url: IGW URL construction, setdefault semantics, explicit
  base_url override, NEMO_BASE_URL env var, non-openai LLMs left unchanged,
  original config dict not mutated.
- inject_fabric_gateway_url: shared and harness-specific Fabric models are
  bound to the IGW without overriding explicit endpoints.
- merge_agent_config: per-section merge semantics for component dicts vs.
  scalar/dict sections, workflow ownership, and input immutability.
- temp_injected_config: temp file written to same directory as source,
  injected content is correct, file is deleted on context exit, and
  ``extra_config`` agent merge works alongside gateway URL injection.
- EvaluateAgentSpec workspace and agent field semantics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import httpx
import pytest
import yaml
from nemo_agents_plugin.jobs.evaluate_agent import EvaluateAgentJob, EvaluateAgentSpec
from nemo_agents_plugin.refs import AgentRef
from nemo_agents_plugin.utils import (
    get_internal_base_url,
    inject_default_model,
    inject_fabric_gateway_url,
    inject_gateway_url,
    merge_agent_config,
    preflight_validate_llm_models,
    rebase_optimize_outputs,
    rebind_intake_ingest_workspace,
    temp_injected_config,
    validate_llm_models,
)
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.errors import NotFoundError as ClientNotFoundError
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.refs import EndpointURL, FilesetRef, LocalDir
from nemo_platform_plugin.run_dependencies import LocalRunError

# ---------------------------------------------------------------------------
# inject_gateway_url
# ---------------------------------------------------------------------------


class TestInjectGatewayUrl:
    _BASE_CONFIG = {
        "llms": {
            "llm": {"_type": "openai", "api_key": "not-used", "model_name": "my-model"},
        }
    }

    def test_injects_base_url_when_absent(self) -> None:
        result = inject_gateway_url(self._BASE_CONFIG, "default", base_url="http://platform:8080")
        assert result["llms"]["llm"]["base_url"] == (
            "http://platform:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
        )

    def test_injects_api_key_when_absent(self) -> None:
        config = {"llms": {"llm": {"_type": "openai", "model_name": "x"}}}
        result = inject_gateway_url(config, "default", base_url="http://platform:8080")
        assert result["llms"]["llm"]["api_key"] == "not-used"

    def test_does_not_override_existing_base_url(self) -> None:
        config = {
            "llms": {
                "llm": {
                    "_type": "openai",
                    "base_url": "http://my-server/v1",
                    "model_name": "x",
                }
            }
        }
        result = inject_gateway_url(config, "default", base_url="http://platform:8080")
        assert result["llms"]["llm"]["base_url"] == "http://my-server/v1"

    def test_does_not_override_existing_api_key(self) -> None:
        config = {
            "llms": {
                "llm": {
                    "_type": "openai",
                    "api_key": "real-key",
                    "model_name": "x",
                }
            }
        }
        result = inject_gateway_url(config, "default", base_url="http://platform:8080")
        assert result["llms"]["llm"]["api_key"] == "real-key"

    def test_workspace_appears_in_url(self) -> None:
        result = inject_gateway_url(self._BASE_CONFIG, "production", base_url="http://platform:8080")
        assert "/workspaces/production/" in result["llms"]["llm"]["base_url"]

    def test_nim_type_is_injected(self) -> None:
        config = {"llms": {"llm": {"_type": "nim", "model_name": "x"}}}
        result = inject_gateway_url(config, "default", base_url="http://platform:8080")
        assert "base_url" in result["llms"]["llm"]

    def test_unknown_type_is_not_injected(self) -> None:
        config = {"llms": {"llm": {"_type": "huggingface", "model_name": "x"}}}
        result = inject_gateway_url(config, "default", base_url="http://platform:8080")
        assert "base_url" not in result["llms"]["llm"]

    def test_original_config_not_mutated(self) -> None:
        original = {"llms": {"llm": {"_type": "openai", "model_name": "x"}}}
        inject_gateway_url(original, "default", base_url="http://platform:8080")
        assert "base_url" not in original["llms"]["llm"]

    def test_multiple_llms_all_injected(self) -> None:
        config = {
            "llms": {
                "llm_a": {"_type": "openai", "model_name": "a"},
                "llm_b": {"_type": "nim", "model_name": "b"},
            }
        }
        result = inject_gateway_url(config, "default", base_url="http://platform:8080")
        assert "base_url" in result["llms"]["llm_a"]
        assert "base_url" in result["llms"]["llm_b"]

    def test_empty_llms_section(self) -> None:
        config: dict = {"llms": {}}
        result = inject_gateway_url(config, "default", base_url="http://platform:8080")
        assert result["llms"] == {}

    def test_no_llms_key(self) -> None:
        config: dict = {"workflow": {"_type": "react_agent"}}
        result = inject_gateway_url(config, "default", base_url="http://platform:8080")
        assert "llms" not in result

    def test_explicit_base_url_takes_precedence_over_nemo_base_url_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEMO_BASE_URL", "http://env-override:9999")
        result = inject_gateway_url(self._BASE_CONFIG, "default", base_url="http://explicit:8080")
        assert result["llms"]["llm"]["base_url"].startswith("http://explicit:8080")

    def test_trailing_slash_stripped_from_base_url(self) -> None:
        result = inject_gateway_url(self._BASE_CONFIG, "default", base_url="http://platform:8080/")
        assert "//apis" not in result["llms"]["llm"]["base_url"]


class TestInjectFabricGatewayUrl:
    def test_injects_shared_and_harness_model_base_url(self) -> None:
        config = {
            "models": {"default": {"provider": "openai", "model": "test-model"}},
            "harnesses": {
                "hermes": {
                    "kind": "hermes",
                    "model": {"provider": "nvidia", "model": "test-harness-model"},
                },
                "claude": {
                    "kind": "claude",
                    "model": {"provider": "anthropic", "model": "anthropic/claude-sonnet-4-5"},
                },
            },
        }

        result = inject_fabric_gateway_url(config, "test-workspace", base_url="http://platform:8080")
        expected_url = "http://platform:8080/apis/inference-gateway/v2/workspaces/test-workspace/openai/-/v1"

        assert result["models"]["default"]["base_url"] == expected_url
        assert result["harnesses"]["hermes"]["model"]["base_url"] == expected_url
        assert "base_url" not in result["harnesses"]["claude"]["model"]
        assert "settings" not in result["models"]["default"]
        assert "settings" not in result["harnesses"]["hermes"]["model"]
        assert "settings" not in config["models"]["default"]
        assert "settings" not in config["harnesses"]["hermes"]["model"]

    def test_preserves_explicit_top_level_endpoint_and_input(self) -> None:
        config = {
            "models": {
                "default": {
                    "provider": "openai",
                    "model": "test-model",
                    "base_url": "http://explicit:8080/v1",
                }
            }
        }

        result = inject_fabric_gateway_url(config, "test-workspace", base_url="http://platform:8080")

        assert result["models"]["default"]["base_url"] == "http://explicit:8080/v1"
        assert config["models"]["default"]["base_url"] == "http://explicit:8080/v1"

    def test_promotes_legacy_settings_endpoint(self) -> None:
        config = {
            "models": {
                "default": {
                    "provider": "openai",
                    "model": "test-model",
                    "settings": {"base_url": "http://legacy:8080/v1"},
                }
            }
        }

        result = inject_fabric_gateway_url(config, "test-workspace", base_url="http://platform:8080")

        assert result["models"]["default"]["base_url"] == "http://legacy:8080/v1"
        assert "base_url" not in result["models"]["default"]["settings"]
        assert "base_url" not in config["models"]["default"]
        assert config["models"]["default"]["settings"]["base_url"] == "http://legacy:8080/v1"

    def test_rebinds_atif_ingest_workspace(self) -> None:
        config = {
            "telemetry": {
                "atif": {
                    "storage": [
                        {
                            "type": "http",
                            "endpoint": "http://127.0.0.1:8080/apis/intake/v2/workspaces/default/ingest/atif",
                        }
                    ]
                }
            }
        }

        result = inject_fabric_gateway_url(config, "test-workspace", base_url="http://platform:8080")

        assert result["telemetry"]["atif"]["storage"][0]["endpoint"] == (
            "http://127.0.0.1:8080/apis/intake/v2/workspaces/test-workspace/ingest/atif"
        )
        assert config["telemetry"]["atif"]["storage"][0]["endpoint"] == (
            "http://127.0.0.1:8080/apis/intake/v2/workspaces/default/ingest/atif"
        )


class TestRebindIntakeIngestWorkspace:
    def test_leaves_non_intake_and_non_http_endpoints_alone(self) -> None:
        config = {
            "telemetry": {
                "atif": {
                    "storage": [
                        {"type": "http", "endpoint": "http://collector:4318/v1/traces"},
                        {"type": "file", "endpoint": "http://x/apis/intake/v2/workspaces/default/ingest/atif"},
                    ]
                }
            }
        }

        rebind_intake_ingest_workspace(config, "test-workspace")

        storage = config["telemetry"]["atif"]["storage"]
        assert storage[0]["endpoint"] == "http://collector:4318/v1/traces"
        assert storage[1]["endpoint"] == "http://x/apis/intake/v2/workspaces/default/ingest/atif"

    def test_ignores_intake_pattern_outside_the_path(self) -> None:
        endpoint = "http://collector:4318/v1/traces?forward=/apis/intake/v2/workspaces/default/ingest/atif"
        config = {"telemetry": {"atif": {"storage": [{"type": "http", "endpoint": endpoint}]}}}

        rebind_intake_ingest_workspace(config, "test-workspace")

        assert config["telemetry"]["atif"]["storage"][0]["endpoint"] == endpoint

    def test_preserves_query_and_fragment_when_rewriting_the_path(self) -> None:
        config = {
            "telemetry": {
                "atif": {
                    "storage": [
                        {
                            "type": "http",
                            "endpoint": (
                                "http://127.0.0.1:8080/apis/intake/v2/workspaces/default/ingest/atif?retries=2#frag"
                            ),
                        }
                    ]
                }
            }
        }

        rebind_intake_ingest_workspace(config, "test-workspace")

        assert config["telemetry"]["atif"]["storage"][0]["endpoint"] == (
            "http://127.0.0.1:8080/apis/intake/v2/workspaces/test-workspace/ingest/atif?retries=2#frag"
        )

    def test_rebinds_workspace_containing_regex_metacharacters(self) -> None:
        config = {
            "telemetry": {
                "atif": {
                    "storage": [
                        {
                            "type": "http",
                            "endpoint": "http://127.0.0.1:8080/apis/intake/v2/workspaces/default/ingest/atif",
                        }
                    ]
                }
            }
        }

        rebind_intake_ingest_workspace(config, r"ws\1+x")

        assert config["telemetry"]["atif"]["storage"][0]["endpoint"] == (
            "http://127.0.0.1:8080/apis/intake/v2/workspaces/ws\\1+x/ingest/atif"
        )

    @pytest.mark.parametrize(
        "telemetry",
        [
            {},
            {"atif": None},
            {"atif": {}},
            {"atif": {"storage": None}},
            {"atif": {"storage": [{"type": "http"}]}},
        ],
    )
    def test_tolerates_absent_or_malformed_storage(self, telemetry: dict[str, Any]) -> None:
        config = {"telemetry": telemetry}

        assert rebind_intake_ingest_workspace(config, "test-workspace") is config


class TestGetInternalBaseUrl:
    def test_returns_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NEMO_INTERNAL_BASE_URL", raising=False)
        monkeypatch.delenv("NMP_INTERNAL_BASE_URL", raising=False)
        assert get_internal_base_url() is None

    def test_reads_nmp_internal_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NEMO_INTERNAL_BASE_URL", raising=False)
        monkeypatch.setenv("NMP_INTERNAL_BASE_URL", "http://nmp-api:8080/")
        assert get_internal_base_url() == "http://nmp-api:8080"

    def test_nemo_internal_base_url_takes_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEMO_INTERNAL_BASE_URL", "http://nemo:8080")
        monkeypatch.setenv("NMP_INTERNAL_BASE_URL", "http://nmp:8080")
        assert get_internal_base_url() == "http://nemo:8080"


# ---------------------------------------------------------------------------
# inject_default_model
# ---------------------------------------------------------------------------


class TestInjectDefaultModel:
    """Resolve ``${NEMO_DEFAULT_MODEL}`` placeholders against the SDK
    context's :func:`get_default_model`.

    The function is the deploy/eval-time companion to the placeholder in
    builtin agent YAMLs (``model_name: ${NEMO_DEFAULT_MODEL}``): configs
    ship without baking in an upstream model name, and the placeholder is
    resolved at the call site by reading the user's active SDK context.
    Tests stub :func:`get_default_model` directly so they don't depend on
    a live ``~/.config/nemo`` profile.
    """

    @staticmethod
    def _patch_default_model(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
        """Stub ``utils.get_default_model`` to return *value* for this test."""
        import nemo_agents_plugin.utils as utils_mod

        monkeypatch.setattr(utils_mod, "get_default_model", lambda: value)

    def test_braced_form_replaced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_default_model(monkeypatch, "nemotron-3-super-v3")
        config = {"llms": {"llm": {"_type": "openai", "model_name": "${NEMO_DEFAULT_MODEL}"}}}
        result = inject_default_model(config)
        assert result["llms"]["llm"]["model_name"] == "nemotron-3-super-v3"

    def test_bare_form_replaced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_default_model(monkeypatch, "nemotron-3-super-v3")
        config = {"llms": {"llm": {"_type": "openai", "model_name": "$NEMO_DEFAULT_MODEL"}}}
        result = inject_default_model(config)
        assert result["llms"]["llm"]["model_name"] == "nemotron-3-super-v3"

    def test_embedded_placeholder_replaced_in_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The placeholder is a substring substitution, not a whole-value match."""
        self._patch_default_model(monkeypatch, "x")
        config = {"meta": {"label": "model=${NEMO_DEFAULT_MODEL}"}}
        result = inject_default_model(config)
        assert result["meta"]["label"] == "model=x"

    def test_unrelated_env_var_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only ``NEMO_DEFAULT_MODEL`` is in the substitution map; other tokens stay."""
        self._patch_default_model(monkeypatch, "x")
        config = {"llms": {"llm": {"model_name": "${OTHER_VAR}"}}}
        result = inject_default_model(config)
        assert result["llms"]["llm"]["model_name"] == "${OTHER_VAR}"

    def test_longer_prefix_token_not_matched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``$NEMO_DEFAULT_MODEL_FOO`` is a different identifier — leave it alone."""
        self._patch_default_model(monkeypatch, "x")
        config = {"llms": {"llm": {"model_name": "$NEMO_DEFAULT_MODEL_FOO"}}}
        result = inject_default_model(config)
        assert result["llms"]["llm"]["model_name"] == "$NEMO_DEFAULT_MODEL_FOO"

    def test_replaces_in_nested_dicts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_default_model(monkeypatch, "x")
        config = {"a": {"b": {"c": "${NEMO_DEFAULT_MODEL}"}}}
        result = inject_default_model(config)
        assert result["a"]["b"]["c"] == "x"

    def test_replaces_in_lists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_default_model(monkeypatch, "x")
        config = {"items": ["${NEMO_DEFAULT_MODEL}", "plain", {"k": "${NEMO_DEFAULT_MODEL}"}]}
        result = inject_default_model(config)
        assert result["items"] == ["x", "plain", {"k": "x"}]

    def test_non_string_leaves_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_default_model(monkeypatch, "x")
        config: dict[str, Any] = {"max_tokens": 1024, "enabled": True, "ratios": [0.1, 0.2]}
        result = inject_default_model(config)
        assert result == {"max_tokens": 1024, "enabled": True, "ratios": [0.1, 0.2]}

    def test_no_placeholder_is_noop_in_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Configs without the placeholder round-trip through unchanged."""
        self._patch_default_model(monkeypatch, "x")
        config = {"llms": {"llm": {"model_name": "explicit-model"}}}
        snapshot = yaml.safe_dump(config)
        result = inject_default_model(config)
        assert yaml.safe_dump(result) == snapshot

    def test_original_config_not_mutated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``expand_env_vars`` returns new containers — the input dict stays put."""
        self._patch_default_model(monkeypatch, "x")
        original = {"llms": {"llm": {"model_name": "${NEMO_DEFAULT_MODEL}"}}}
        snapshot = yaml.safe_dump(original)
        inject_default_model(original)
        assert yaml.safe_dump(original) == snapshot

    def test_no_default_model_returns_input_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``get_default_model() is None`` short-circuits to ``return config``.

        The placeholder is left in place so a downstream consumer that
        does know which model to use (e.g. an explicit override on the
        deployment record) can still substitute it. The function does
        NOT log a warning here — the intent is silent fallthrough so
        partial configs with an unresolved placeholder don't spam logs.
        """
        self._patch_default_model(monkeypatch, None)
        config = {"llms": {"llm": {"model_name": "${NEMO_DEFAULT_MODEL}"}}}
        result = inject_default_model(config)
        assert result is config
        assert result["llms"]["llm"]["model_name"] == "${NEMO_DEFAULT_MODEL}"

    def test_no_default_model_does_not_warn(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Silent fallthrough: no warning when the SDK context has no default model."""
        self._patch_default_model(monkeypatch, None)
        config = {"llms": {"llm": {"model_name": "${NEMO_DEFAULT_MODEL}"}}}
        with caplog.at_level("WARNING"):
            inject_default_model(config)
        assert not any("NEMO_DEFAULT_MODEL" in rec.message for rec in caplog.records)

    def test_uses_sdk_context_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The resolved value comes from ``nemo_platform.config.get_context().default_model``."""
        self._patch_default_model(monkeypatch, "from-sdk")
        config = {"llms": {"llm": {"model_name": "${NEMO_DEFAULT_MODEL}"}}}
        result = inject_default_model(config)
        assert result["llms"]["llm"]["model_name"] == "from-sdk"


# ---------------------------------------------------------------------------
# temp_injected_config
# ---------------------------------------------------------------------------


class TestTempInjectedConfig:
    def _write_config(self, path: Path, config: dict) -> None:
        path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def test_yields_path_in_same_directory(self, tmp_path: Path) -> None:
        config_path = tmp_path / "my-agent.yaml"
        self._write_config(config_path, {"llms": {"llm": {"_type": "openai", "model_name": "x"}}})

        with temp_injected_config(config_path, "default", base_url="http://p:8080") as injected:
            assert injected.parent == tmp_path

    def test_injected_file_contains_base_url(self, tmp_path: Path) -> None:
        config_path = tmp_path / "agent.yaml"
        self._write_config(config_path, {"llms": {"llm": {"_type": "openai", "model_name": "x"}}})

        with temp_injected_config(config_path, "default", base_url="http://p:8080") as injected:
            loaded = yaml.safe_load(injected.read_text())
            assert "base_url" in loaded["llms"]["llm"]
            assert "/workspaces/default/" in loaded["llms"]["llm"]["base_url"]

    def test_temp_file_deleted_after_context(self, tmp_path: Path) -> None:
        config_path = tmp_path / "agent.yaml"
        self._write_config(config_path, {"llms": {}})

        with temp_injected_config(config_path, "default", base_url="http://p:8080") as injected:
            tmp_file = injected
            assert tmp_file.exists()

        assert not tmp_file.exists()

    def test_temp_file_deleted_even_on_exception(self, tmp_path: Path) -> None:
        config_path = tmp_path / "agent.yaml"
        self._write_config(config_path, {"llms": {}})

        tmp_file: Path | None = None
        with pytest.raises(RuntimeError):
            with temp_injected_config(config_path, "default", base_url="http://p:8080") as injected:
                tmp_file = injected
                raise RuntimeError("boom")

        assert tmp_file is not None
        assert not tmp_file.exists()

    def test_original_file_not_modified(self, tmp_path: Path) -> None:
        config_path = tmp_path / "agent.yaml"
        original_content = {"llms": {"llm": {"_type": "openai", "model_name": "x"}}}
        self._write_config(config_path, original_content)

        with temp_injected_config(config_path, "default", base_url="http://p:8080"):
            pass

        loaded = yaml.safe_load(config_path.read_text())
        assert "base_url" not in loaded["llms"]["llm"]

    def test_existing_base_url_preserved(self, tmp_path: Path) -> None:
        config_path = tmp_path / "agent.yaml"
        self._write_config(
            config_path,
            {"llms": {"llm": {"_type": "openai", "base_url": "http://custom/v1", "model_name": "x"}}},
        )

        with temp_injected_config(config_path, "default", base_url="http://p:8080") as injected:
            loaded = yaml.safe_load(injected.read_text())
            assert loaded["llms"]["llm"]["base_url"] == "http://custom/v1"

    def test_defaults_injected_when_key_absent(self, tmp_path: Path) -> None:
        config_path = tmp_path / "optimize.yaml"
        self._write_config(config_path, {"llms": {"llm": {"_type": "openai", "model_name": "x"}}})

        defaults = {"workflow": {"_type": "chat_completion", "llm_name": "llm"}}
        with temp_injected_config(config_path, "default", base_url="http://p:8080", defaults=defaults) as injected:
            loaded = yaml.safe_load(injected.read_text())
            assert loaded["workflow"] == {"_type": "chat_completion", "llm_name": "llm"}

    def test_defaults_do_not_override_existing_keys(self, tmp_path: Path) -> None:
        config_path = tmp_path / "agent.yaml"
        self._write_config(
            config_path,
            {
                "llms": {"llm": {"_type": "openai", "model_name": "x"}},
                "workflow": {"_type": "react_agent", "llm_name": "llm"},
            },
        )

        defaults = {"workflow": {"_type": "chat_completion", "llm_name": "llm"}}
        with temp_injected_config(config_path, "default", base_url="http://p:8080", defaults=defaults) as injected:
            loaded = yaml.safe_load(injected.read_text())
            assert loaded["workflow"]["_type"] == "react_agent"


# ---------------------------------------------------------------------------
# EvaluateAgentSpec workspace field
# ---------------------------------------------------------------------------


class TestEvaluateAgentSpecWorkspace:
    def test_workspace_defaults_to_default(self) -> None:
        cfg = EvaluateAgentSpec(eval_config="/tmp/eval.yml")
        assert cfg.workspace == "default"

    def test_workspace_can_be_set(self) -> None:
        cfg = EvaluateAgentSpec(eval_config="/tmp/eval.yml", workspace="production")
        assert cfg.workspace == "production"


# ---------------------------------------------------------------------------
# EvaluateAgentSpec.agent — single union-typed field replacing the old
# (agent, agent_endpoint) XOR pair.  The shape (URL vs name) is detected at
# resolve time, not declared up-front by the caller.
# ---------------------------------------------------------------------------


class TestEvaluateAgentSpecAgent:
    def test_agent_defaults_to_none(self) -> None:
        cfg = EvaluateAgentSpec(eval_config="/tmp/eval.yml")
        assert cfg.agent is None

    def test_agent_accepts_bare_name(self) -> None:
        """A bare name round-trips as a plain string; the resolver classifies it later.

        The ``# ty: ignore`` here (and on the sibling tests below) covers
        a checker blind spot: the StrRef Pydantic core schema accepts a
        bare ``str`` and wraps it in the union arm at runtime, but ``ty``
        only sees the union annotation.  The runtime assertion is the
        actual behaviour we care about.
        """
        cfg = EvaluateAgentSpec(eval_config="/tmp/eval.yml", agent="calculator")  # ty: ignore[invalid-argument-type]
        assert cfg.agent == "calculator"
        assert isinstance(cfg.agent, str)

    def test_agent_accepts_ws_qualified_name(self) -> None:
        cfg = EvaluateAgentSpec(eval_config="/tmp/eval.yml", agent="prod/calculator")  # ty: ignore[invalid-argument-type]
        assert cfg.agent == "prod/calculator"

    def test_agent_accepts_http_url(self) -> None:
        cfg = EvaluateAgentSpec(eval_config="/tmp/eval.yml", agent="http://localhost:8080")  # ty: ignore[invalid-argument-type]
        assert cfg.agent == "http://localhost:8080"

    def test_agent_accepts_https_url(self) -> None:
        cfg = EvaluateAgentSpec(eval_config="/tmp/eval.yml", agent="https://api.example.com/v1/agent")  # ty: ignore[invalid-argument-type]
        assert cfg.agent == "https://api.example.com/v1/agent"

    def test_no_agent_endpoint_field_after_union_merge(self) -> None:
        """The union-typed ``agent`` replaces both old fields; ``agent_endpoint`` is gone."""
        assert "agent_endpoint" not in EvaluateAgentSpec.model_fields


class TestResolveEndpoint:
    """``EvaluateAgentJob._resolve_endpoint`` projects the union to a single URL.

    The classifier uses ``"://"`` as the URL marker — the same shape rule
    the LJ-3 ``classify_input`` parser applies.  These tests cover the
    three branches the production code handles: ``None`` (no override),
    URL (passthrough), and bare name (gateway URL construction).
    """

    def test_none_returns_none(self) -> None:
        """``None`` means "no override" — keep whatever the eval config declares."""
        assert EvaluateAgentJob._resolve_endpoint(None, workspace="default") is None

    def test_url_passthrough(self) -> None:
        """URLs are forwarded verbatim to ``nat eval --endpoint``."""
        url = EndpointURL("http://localhost:8080")
        assert EvaluateAgentJob._resolve_endpoint(url, workspace="default") == "http://localhost:8080"

    def test_bare_name_resolves_against_workspace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NMP_BASE_URL", "http://platform:8080")
        ref = AgentRef("calculator")
        endpoint = EvaluateAgentJob._resolve_endpoint(ref, workspace="default")
        assert endpoint == "http://platform:8080/apis/agents/v2/workspaces/default/agents/calculator/-"

    def test_ws_qualified_name_overrides_workspace_arg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``"prod/calculator"`` wins over the spec's ``workspace`` field."""
        monkeypatch.setenv("NMP_BASE_URL", "http://platform:8080")
        ref = AgentRef("prod/calculator")
        endpoint = EvaluateAgentJob._resolve_endpoint(ref, workspace="default")
        assert endpoint == "http://platform:8080/apis/agents/v2/workspaces/prod/agents/calculator/-"

    def test_plain_str_value_treated_as_url_when_it_has_a_scheme(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Robust to model-validate output: ``str`` values still classify by shape."""
        monkeypatch.setenv("NMP_BASE_URL", "http://platform:8080")
        # Pydantic emits the value as a ``str`` subclass via the StrRef hook,
        # but the resolver must not depend on isinstance checks — the parser
        # is shape-based.
        cfg = EvaluateAgentSpec.model_validate({"eval_config": "x", "agent": "https://example.com"})
        assert EvaluateAgentJob._resolve_endpoint(cfg.agent, workspace="default") == "https://example.com"


# ---------------------------------------------------------------------------
# EvaluateAgentSpec.output — same union pattern as ``agent``: local
# directory path *or* NeMo Platform fileset reference.  The shape (path vs name)
# is detected at resolve time.
# ---------------------------------------------------------------------------


class TestEvaluateAgentSpecOutput:
    def test_output_defaults_to_none(self) -> None:
        cfg = EvaluateAgentSpec(eval_config="/tmp/eval.yml")
        assert cfg.output is None

    def test_output_accepts_local_path(self) -> None:
        cfg = EvaluateAgentSpec(eval_config="/tmp/eval.yml", output="./eval-out")  # ty: ignore[invalid-argument-type]
        assert cfg.output == "./eval-out"
        assert isinstance(cfg.output, str)

    def test_output_accepts_absolute_path(self) -> None:
        cfg = EvaluateAgentSpec(eval_config="/tmp/eval.yml", output="/tmp/eval-out")  # ty: ignore[invalid-argument-type]
        assert cfg.output == "/tmp/eval-out"

    def test_output_accepts_fileset_name(self) -> None:
        cfg = EvaluateAgentSpec(eval_config="/tmp/eval.yml", output="eval-results")  # ty: ignore[invalid-argument-type]
        assert cfg.output == "eval-results"

    def test_output_accepts_ws_qualified_fileset(self) -> None:
        cfg = EvaluateAgentSpec(eval_config="/tmp/eval.yml", output="prod/eval-results")  # ty: ignore[invalid-argument-type]
        assert cfg.output == "prod/eval-results"

    def test_no_output_base_dir_field_after_rename(self) -> None:
        """The renamed field replaces the old ``output_base_dir``; keep this loud."""
        assert "output_base_dir" not in EvaluateAgentSpec.model_fields
        assert "output" in EvaluateAgentSpec.model_fields


class TestResolveOutput:
    """``EvaluateAgentJob._resolve_output`` chooses local-write vs upload-on-exit.

    The context-manager shape lets the same call site handle both arms
    of the union — ``LocalDir`` yields a real on-disk path and is a
    no-op on exit; ``FilesetRef`` yields a tempdir and uploads the
    captured tree to the named fileset on a clean exit.  Failures
    inside the ``with`` block intentionally skip the upload to avoid
    polluting filesets with broken outputs.
    """

    def test_none_yields_persistent_results(self, tmp_path: Path, ctx: JobContext) -> None:
        """No-output fallback writes under ``ctx.storage.persistent / "results"``."""
        job = EvaluateAgentJob()
        with job._resolve_output(None, workspace="default", sdk=None, ctx=ctx) as base:
            assert base == ctx.storage.persistent / "results"
            assert base.is_dir()

    def test_local_dir_is_yielded_and_created(self, tmp_path: Path, ctx: JobContext) -> None:
        local = tmp_path / "out" / "nested"
        job = EvaluateAgentJob()
        with job._resolve_output(LocalDir(str(local)), workspace="default", sdk=None, ctx=ctx) as base:
            assert base == local
            assert base.exists()

    def test_local_dir_expanduser(self, tmp_path: Path, ctx: JobContext, monkeypatch: pytest.MonkeyPatch) -> None:
        """``~`` expands relative to ``$HOME`` so users can pass ``~/foo``."""
        monkeypatch.setenv("HOME", str(tmp_path))
        job = EvaluateAgentJob()
        with job._resolve_output(LocalDir("~/eval-out"), workspace="default", sdk=None, ctx=ctx) as base:
            assert base == tmp_path / "eval-out"
            assert base.exists()

    def test_local_dir_relative_path_is_resolved(
        self, tmp_path: Path, ctx: JobContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Relative ``./eval-out`` is resolved against the caller's CWD.

        ``nat eval`` later runs with ``cwd=injected_path.parent``, so a
        bare relative path would otherwise land inside the eval YAML's
        directory.  The resolver pins the output to an absolute location
        before the subprocess gets a chance to chdir.
        """
        monkeypatch.chdir(tmp_path)
        job = EvaluateAgentJob()
        with job._resolve_output(LocalDir("./eval-out"), workspace="default", sdk=None, ctx=ctx) as base:
            assert base.is_absolute()
            assert base == (tmp_path / "eval-out").resolve()
            assert base.exists()

    def test_fileset_ref_uploads_on_clean_exit(
        self, tmp_path: Path, ctx: JobContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Successful exit forwards the staged tempdir to the fileset uploader."""
        captured: dict[str, object] = {}

        def fake_upload(
            local_dir: Path,
            *,
            fileset: str,
            workspace: str,
            sdk: object,
        ) -> None:
            captured["local_dir"] = local_dir
            captured["fileset"] = fileset
            captured["workspace"] = workspace
            captured["sdk"] = sdk

        monkeypatch.setattr(EvaluateAgentJob, "_upload_to_fileset", staticmethod(fake_upload))
        sentinel_sdk = _platform_sdk_stub()
        job = EvaluateAgentJob()
        with job._resolve_output(
            FilesetRef("eval-results"),
            workspace="default",
            sdk=sentinel_sdk,
            ctx=ctx,
        ) as base:
            (base / "summary.json").write_text("{}")

        assert captured["fileset"] == "eval-results"
        assert captured["workspace"] == "default"
        assert isinstance(captured["local_dir"], Path)
        assert captured["sdk"] is sentinel_sdk

    def test_fileset_ref_workspace_qualified_name_wins(
        self, tmp_path: Path, ctx: JobContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``"prod/eval-results"`` overrides the spec-level ``workspace``."""
        captured: dict[str, object] = {}

        def fake_upload(
            local_dir: Path,
            *,
            fileset: str,
            workspace: str,
            sdk: object,
        ) -> None:
            del local_dir, sdk
            captured["fileset"] = fileset
            captured["workspace"] = workspace

        monkeypatch.setattr(EvaluateAgentJob, "_upload_to_fileset", staticmethod(fake_upload))
        job = EvaluateAgentJob()
        with job._resolve_output(
            FilesetRef("prod/eval-results"),
            workspace="default",
            sdk=_platform_sdk_stub(),
            ctx=ctx,
        ) as _:
            pass

        assert captured["workspace"] == "prod"
        assert captured["fileset"] == "eval-results"

    def test_fileset_ref_skips_upload_on_exception(
        self, tmp_path: Path, ctx: JobContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Don't pollute the fileset with partial outputs from a crashed run."""
        called: list[bool] = []

        def fake_upload(
            local_dir: Path,
            *,
            fileset: str,
            workspace: str,
            sdk: object,
        ) -> None:
            del local_dir, fileset, workspace, sdk
            called.append(True)

        monkeypatch.setattr(EvaluateAgentJob, "_upload_to_fileset", staticmethod(fake_upload))
        job = EvaluateAgentJob()
        with pytest.raises(RuntimeError, match="boom"):
            with job._resolve_output(
                FilesetRef("eval-results"),
                workspace="default",
                sdk=_platform_sdk_stub(),
                ctx=ctx,
            ) as base:
                (base / "partial.json").write_text("{}")
                raise RuntimeError("boom")

        assert called == []

    def test_fileset_ref_tempdir_cleaned_up(
        self, tmp_path: Path, ctx: JobContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The staging tempdir is removed regardless of upload outcome."""

        def fake_upload(
            local_dir: Path,
            *,
            fileset: str,
            workspace: str,
            sdk: object,
        ) -> None:
            del fileset, workspace, sdk
            assert local_dir.exists()

        monkeypatch.setattr(EvaluateAgentJob, "_upload_to_fileset", staticmethod(fake_upload))
        job = EvaluateAgentJob()
        captured_path: Path | None = None
        with job._resolve_output(
            FilesetRef("eval-results"),
            workspace="default",
            sdk=_platform_sdk_stub(),
            ctx=ctx,
        ) as base:
            captured_path = base

        assert captured_path is not None
        assert not captured_path.exists()

    def test_fileset_ref_without_sdk_raises(self, tmp_path: Path, ctx: JobContext) -> None:
        """A fileset target with no injected SDK errors before the subprocess runs.

        The user gets an actionable message — including how to switch to
        a local-directory output — instead of silently dropping the
        evaluation artifacts on the floor.
        """
        job = EvaluateAgentJob()
        with pytest.raises(LocalRunError, match="sdk: NeMoPlatform"):
            with job._resolve_output(FilesetRef("eval-results"), workspace="default", sdk=None, ctx=ctx):
                pass

    def test_upload_to_fileset_delegates_to_typed_file_manager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``_upload_to_fileset`` uploads through a typed Files manager."""
        manager = Mock()
        manager_cls = Mock(return_value=manager)
        monkeypatch.setattr("nemo_agents_plugin.jobs.evaluate_agent.client_from_platform", Mock())
        monkeypatch.setattr("nemo_agents_plugin.jobs.evaluate_agent.FilesetFileSystem", Mock())
        monkeypatch.setattr("nemo_agents_plugin.jobs.evaluate_agent.FilesetFileManager", manager_cls)

        sdk = Mock()
        EvaluateAgentJob._upload_to_fileset(
            Path("/tmp/eval-out"),
            fileset="eval-results",
            workspace="prod",
            sdk=sdk,
        )

        manager_cls.assert_called_once()
        assert manager_cls.call_args.kwargs["fileset_name"] == "eval-results"
        assert manager_cls.call_args.kwargs["workspace"] == "prod"
        assert manager_cls.call_args.kwargs["ensure_fileset_exists"] is True
        manager.validate_storage.assert_called_once_with()
        manager.upload.assert_called_once_with(local_path=Path("/tmp/eval-out"), remote_path="")


# ---------------------------------------------------------------------------
# merge_agent_config — composes a stored agent definition with an optimize
# config so trials run the agent's workflow locally with the optimize side's
# tuning knobs and eval/optimizer settings.
# ---------------------------------------------------------------------------


class TestMergeAgentConfig:
    """Per-section merge semantics for the agent + optimize compose step.

    The shape we care about end-to-end: an agent contributes the workflow,
    tools, telemetry, and base LLM specs; the optimize config contributes
    eval/optimizer/judge LLMs and any tuning overrides on shared LLM keys.
    The component-dict sections (``llms``, ``functions``, ...) merge
    per-key so that overriding ``llms.llm`` swaps the whole spec rather
    than interleaving fields from two unrelated definitions.
    """

    _AGENT: dict[str, Any] = {
        "functions": {
            "wiki": {"_type": "wiki_search"},
            "clock": {"_type": "current_datetime"},
        },
        "llms": {
            "llm": {"_type": "openai", "model_name": "nemotron-30b", "temperature": 0.0},
        },
        "workflow": {
            "_type": "react_agent",
            "tool_names": ["wiki", "clock"],
            "llm_name": "llm",
        },
        "general": {
            "telemetry": {
                "tracing": {
                    "nemo_trace": {"_type": "nemo_files", "batch_size": 128},
                },
            },
        },
    }

    _OPTIMIZE: dict[str, Any] = {
        "llms": {
            "llm": {
                "_type": "openai",
                "model_name": "nemotron-30b",
                "temperature": 0.0,
                "optimizable_params": ["temperature", "top_p"],
                "search_space": {
                    "temperature": {"low": 0.0, "high": 0.8, "step": 0.2},
                    "top_p": {"low": 0.5, "high": 1.0, "step": 0.1},
                },
            },
            "judge_llm": {
                "_type": "openai",
                "model_name": "nemotron-120b",
                "max_tokens": 1024,
            },
        },
        "eval": {
            "general": {"max_concurrency": 4, "output_dir": "eval/react-agent"},
            "evaluators": {"accuracy": {"_type": "tunable_rag_evaluator", "llm_name": "judge_llm"}},
        },
        "optimizer": {
            "output_path": "optimizer_results/react-agent",
            "numeric": {"enabled": True, "n_trials": 3},
        },
    }

    def test_workflow_comes_from_agent(self) -> None:
        """The agent's workflow is the whole point of the merge — never replaced."""
        merged = merge_agent_config(self._AGENT, self._OPTIMIZE)
        assert merged["workflow"]["_type"] == "react_agent"
        assert merged["workflow"]["tool_names"] == ["wiki", "clock"]

    def test_functions_from_agent_preserved(self) -> None:
        merged = merge_agent_config(self._AGENT, self._OPTIMIZE)
        assert merged["functions"]["wiki"] == {"_type": "wiki_search"}
        assert merged["functions"]["clock"] == {"_type": "current_datetime"}

    def test_shared_llm_key_is_replaced_wholesale(self) -> None:
        """``optimizable_params`` and ``search_space`` must land on the
        same LLM the workflow invokes — swap the spec rather than deep
        merge so partial agent fields don't leak through.
        """
        merged = merge_agent_config(self._AGENT, self._OPTIMIZE)
        llm = merged["llms"]["llm"]
        assert llm["optimizable_params"] == ["temperature", "top_p"]
        assert "search_space" in llm
        assert llm["model_name"] == "nemotron-30b"

    def test_optimize_only_llm_added(self) -> None:
        """Judge LLM is optimize-only and rides through unchanged."""
        merged = merge_agent_config(self._AGENT, self._OPTIMIZE)
        assert merged["llms"]["judge_llm"]["model_name"] == "nemotron-120b"

    def test_agent_only_llm_preserved(self) -> None:
        """An LLM that exists only on the agent side stays put."""
        agent = {**self._AGENT, "llms": {"helper_llm": {"_type": "openai", "model_name": "x"}}}
        merged = merge_agent_config(agent, {"llms": {"judge_llm": {"_type": "openai", "model_name": "y"}}})
        assert "helper_llm" in merged["llms"]
        assert "judge_llm" in merged["llms"]

    def test_eval_section_from_optimize_only(self) -> None:
        merged = merge_agent_config(self._AGENT, self._OPTIMIZE)
        assert merged["eval"]["general"]["max_concurrency"] == 4
        assert "accuracy" in merged["eval"]["evaluators"]

    def test_optimizer_section_from_optimize_only(self) -> None:
        merged = merge_agent_config(self._AGENT, self._OPTIMIZE)
        assert merged["optimizer"]["output_path"] == "optimizer_results/react-agent"

    def test_telemetry_from_agent_preserved(self) -> None:
        """``general.telemetry`` rides through so trials emit traces."""
        merged = merge_agent_config(self._AGENT, self._OPTIMIZE)
        tracing = merged["general"]["telemetry"]["tracing"]
        assert tracing["nemo_trace"]["_type"] == "nemo_files"
        assert tracing["nemo_trace"]["batch_size"] == 128

    def test_general_keys_deep_merged_when_both_sides_define_them(self) -> None:
        """Optimize-side ``general`` additions land alongside agent telemetry."""
        agent = {"general": {"telemetry": {"tracing": {"trace": {"_type": "nemo_files"}}}}}
        optimize = {"general": {"telemetry": {"tracing": {"otel": {"_type": "otel"}}}}}
        merged = merge_agent_config(agent, optimize)
        assert "trace" in merged["general"]["telemetry"]["tracing"]
        assert "otel" in merged["general"]["telemetry"]["tracing"]

    def test_inputs_not_mutated(self) -> None:
        """``merge_agent_config`` never mutates either side's input dict."""
        agent_copy = {"llms": {"llm": {"_type": "openai", "model_name": "a"}}}
        optimize_copy = {"llms": {"llm": {"_type": "openai", "model_name": "b"}}}
        merge_agent_config(agent_copy, optimize_copy)
        assert agent_copy["llms"]["llm"]["model_name"] == "a"
        assert optimize_copy["llms"]["llm"]["model_name"] == "b"

    def test_optimize_workflow_is_warned_about_and_dropped(self, caplog: pytest.LogCaptureFixture) -> None:
        """If the optimize side declares a workflow that disagrees with the
        agent's, the agent's wins and we log a warning so the user
        notices the inert override.
        """
        agent = {"workflow": {"_type": "react_agent", "llm_name": "llm"}}
        optimize = {"workflow": {"_type": "chat_completion", "llm_name": "llm"}}
        with caplog.at_level("WARNING"):
            merged = merge_agent_config(agent, optimize)
        assert merged["workflow"]["_type"] == "react_agent"
        assert any("workflow" in rec.message for rec in caplog.records)

    def test_lists_are_replaced_not_concatenated(self) -> None:
        """Lists in NAT configs are ordered references; concatenation would
        produce nonsense (e.g. ``tool_names: [wiki, clock, foo]`` from a
        partial override).  Replace wholesale instead.
        """
        agent = {"general": {"some_list": [1, 2, 3]}}
        optimize = {"general": {"some_list": [9]}}
        merged = merge_agent_config(agent, optimize)
        assert merged["general"]["some_list"] == [9]


# ---------------------------------------------------------------------------
# temp_injected_config — extra_config branch (agent merge + IGW injection
# end-to-end on disk, exercising the path the optimize job actually hits).
# ---------------------------------------------------------------------------


class TestTempInjectedConfigErrorHandling:
    def test_invalid_yaml_syntax_raises_clear_error(self, tmp_path: Path) -> None:
        """Malformed YAML should raise YAMLError with helpful context."""
        config_path = tmp_path / "bad.yaml"
        config_path.write_text("invalid: yaml: : syntax", encoding="utf-8")

        with pytest.raises(yaml.YAMLError, match=f"Failed to parse config file.*{config_path.name}"):
            with temp_injected_config(config_path, "default", base_url="http://p:8080"):
                pass

    def test_nonexistent_file_raises_clear_error(self, tmp_path: Path) -> None:
        """Missing config file should raise RuntimeError with helpful context."""
        config_path = tmp_path / "nonexistent.yaml"

        with pytest.raises(RuntimeError, match=f"Failed to read config file.*{config_path.name}"):
            with temp_injected_config(config_path, "default", base_url="http://p:8080"):
                pass

    def test_directory_traversal_in_output_base_raises_clear_error(self, tmp_path: Path) -> None:
        """Directory traversal in rebased paths should raise ValueError."""
        config_path = tmp_path / "optimize.yaml"
        config_path.write_text(yaml.safe_dump({"eval": {"general": {"output_dir": "../../etc"}}}), encoding="utf-8")

        with pytest.raises(ValueError, match="directory traversal"):
            with temp_injected_config(config_path, "default", base_url="http://p:8080", output_base=tmp_path / "out"):
                pass


class TestTempInjectedConfigWithExtraConfig:
    def _write(self, path: Path, config: dict) -> None:
        path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def test_extra_config_workflow_lands_in_injected_yaml(self, tmp_path: Path) -> None:
        """Agent's workflow (passed as ``extra_config``) is merged under the optimize YAML."""
        config_path = tmp_path / "optimize.yaml"
        self._write(
            config_path,
            {
                "llms": {"llm": {"_type": "openai", "model_name": "x", "optimizable_params": ["temperature"]}},
                "optimizer": {"output_path": "out"},
            },
        )

        agent_config = {
            "workflow": {"_type": "react_agent", "llm_name": "llm", "tool_names": ["t"]},
            "functions": {"t": {"_type": "current_datetime"}},
            "llms": {"llm": {"_type": "openai", "model_name": "from-agent"}},
        }

        with temp_injected_config(
            config_path,
            "default",
            base_url="http://p:8080",
            extra_config=agent_config,
        ) as injected:
            loaded = yaml.safe_load(injected.read_text())

        assert loaded["workflow"]["_type"] == "react_agent"
        assert loaded["functions"]["t"] == {"_type": "current_datetime"}
        # Optimize side wins on shared LLM key.
        assert loaded["llms"]["llm"]["model_name"] == "x"
        assert loaded["llms"]["llm"]["optimizable_params"] == ["temperature"]
        # Gateway URL still gets injected on top of the merged config.
        assert "/workspaces/default/" in loaded["llms"]["llm"]["base_url"]


class TestRebaseOptimizeOutputs:
    def test_repoints_eval_and_optimizer_outputs(self, tmp_path: Path) -> None:
        config = yaml.safe_load(
            """
eval:
  general:
    output_dir: eval/calculator
optimizer:
  output_path: optimizer_results/calculator
""".strip()
        )

        rebased = rebase_optimize_outputs(config, tmp_path / "results")

        assert rebased["eval"]["general"]["output_dir"] == str(tmp_path / "results" / "eval" / "calculator")
        assert rebased["optimizer"]["output_path"] == str(tmp_path / "results" / "optimizer_results" / "calculator")
        assert config["eval"]["general"]["output_dir"] == "eval/calculator"
        assert config["optimizer"]["output_path"] == "optimizer_results/calculator"

    def test_skips_missing_sections(self, tmp_path: Path) -> None:
        config = {"llms": {}}
        assert rebase_optimize_outputs(config, tmp_path / "results") == {"llms": {}}

    def test_rejects_directory_traversal_in_eval_output(self, tmp_path: Path) -> None:
        """Directory traversal attempts in eval.general.output_dir should be rejected."""
        config = yaml.safe_load(
            """
eval:
  general:
    output_dir: ../../etc/passwd
""".strip()
        )

        with pytest.raises(ValueError, match="directory traversal"):
            rebase_optimize_outputs(config, tmp_path / "results")

    def test_rejects_directory_traversal_in_optimizer_output(self, tmp_path: Path) -> None:
        """Directory traversal attempts in optimizer.output_path should be rejected."""
        config = yaml.safe_load(
            """
optimizer:
  output_path: ../../../sensitive/data
""".strip()
        )

        with pytest.raises(ValueError, match="directory traversal"):
            rebase_optimize_outputs(config, tmp_path / "results")

    def test_rejects_path_escaping_base_directory(self, tmp_path: Path) -> None:
        """Paths that would escape base directory after resolution should be rejected."""
        config = yaml.safe_load(
            """
eval:
  general:
    output_dir: legitimate/../../../etc
""".strip()
        )

        with pytest.raises(ValueError, match="directory traversal|escape"):
            rebase_optimize_outputs(config, tmp_path / "results")

    def test_absolute_path_stripped_to_relative(self, tmp_path: Path) -> None:
        """Absolute-looking paths have leading slashes stripped and become relative."""
        config = yaml.safe_load(
            """
eval:
  general:
    output_dir: /tmp/eval/results
optimizer:
  output_path: /opt/optimizer/output
""".strip()
        )

        rebased = rebase_optimize_outputs(config, tmp_path / "results")

        # Leading slashes stripped, paths are relative to output_base
        assert rebased["eval"]["general"]["output_dir"] == str(tmp_path / "results" / "tmp" / "eval" / "results")
        assert rebased["optimizer"]["output_path"] == str(tmp_path / "results" / "opt" / "optimizer" / "output")

    def test_empty_string_output_dir_becomes_default(self, tmp_path: Path) -> None:
        """Empty string paths are replaced with 'output' default."""
        config = yaml.safe_load(
            """
eval:
  general:
    output_dir: ""
""".strip()
        )

        rebased = rebase_optimize_outputs(config, tmp_path / "results")

        # Empty string normalized to 'output'
        assert rebased["eval"]["general"]["output_dir"] == str(tmp_path / "results" / "output")

    def test_whitespace_only_path_becomes_default(self, tmp_path: Path) -> None:
        """Whitespace-only paths are normalized to default 'output'."""
        config = yaml.safe_load(
            """
optimizer:
  output_path: "   "
""".strip()
        )

        rebased = rebase_optimize_outputs(config, tmp_path / "results")

        assert rebased["optimizer"]["output_path"] == str(tmp_path / "results" / "output")

    def test_paths_with_special_characters(self, tmp_path: Path) -> None:
        """Paths with special characters (hyphens, underscores, dots) work correctly."""
        config = yaml.safe_load(
            """
eval:
  general:
    output_dir: eval-results_v2.0/run-123
optimizer:
  output_path: optimizer.out/results_2024-01-01
""".strip()
        )

        rebased = rebase_optimize_outputs(config, tmp_path / "results")

        assert rebased["eval"]["general"]["output_dir"] == str(tmp_path / "results" / "eval-results_v2.0" / "run-123")
        assert rebased["optimizer"]["output_path"] == str(tmp_path / "results" / "optimizer.out" / "results_2024-01-01")

    def test_single_component_path(self, tmp_path: Path) -> None:
        """Single-component paths (no slashes) work correctly."""
        config = yaml.safe_load(
            """
eval:
  general:
    output_dir: results
""".strip()
        )

        rebased = rebase_optimize_outputs(config, tmp_path / "results")

        assert rebased["eval"]["general"]["output_dir"] == str(tmp_path / "results" / "results")

    def test_multiple_slashes_collapsed(self, tmp_path: Path) -> None:
        """Multiple consecutive slashes are collapsed to single separators."""
        config = yaml.safe_load(
            """
eval:
  general:
    output_dir: eval///results//output
""".strip()
        )

        rebased = rebase_optimize_outputs(config, tmp_path / "results")

        # Multiple slashes collapsed
        assert rebased["eval"]["general"]["output_dir"] == str(tmp_path / "results" / "eval" / "results" / "output")

    def test_leading_and_trailing_slashes_stripped(self, tmp_path: Path) -> None:
        """Leading and trailing slashes are properly stripped."""
        config = yaml.safe_load(
            """
eval:
  general:
    output_dir: /eval/results/
optimizer:
  output_path: ///optimizer///output///
""".strip()
        )

        rebased = rebase_optimize_outputs(config, tmp_path / "results")

        assert rebased["eval"]["general"]["output_dir"] == str(tmp_path / "results" / "eval" / "results")
        assert rebased["optimizer"]["output_path"] == str(tmp_path / "results" / "optimizer" / "output")

    def test_both_eval_and_optimizer_paths_rebased_independently(self, tmp_path: Path) -> None:
        """Both eval and optimizer paths are rebased without interfering with each other."""
        config = yaml.safe_load(
            """
eval:
  general:
    output_dir: eval/run1
optimizer:
  output_path: optimizer/trial5
llms:
  llm:
    _type: openai
""".strip()
        )

        rebased = rebase_optimize_outputs(config, tmp_path / "results")

        assert rebased["eval"]["general"]["output_dir"] == str(tmp_path / "results" / "eval" / "run1")
        assert rebased["optimizer"]["output_path"] == str(tmp_path / "results" / "optimizer" / "trial5")
        # Other config sections preserved
        assert rebased["llms"]["llm"]["_type"] == "openai"

    def test_non_string_output_dir_skipped_gracefully(self, tmp_path: Path) -> None:
        """Non-string output_dir values are skipped without error."""
        config = yaml.safe_load(
            """
eval:
  general:
    output_dir: null
""".strip()
        )

        # Should not raise, just skip the None value
        rebased = rebase_optimize_outputs(config, tmp_path / "results")
        assert rebased["eval"]["general"]["output_dir"] is None

    def test_nested_config_structure_preserved(self, tmp_path: Path) -> None:
        """Complex nested config structure is preserved during rebasing."""
        config = yaml.safe_load(
            """
eval:
  general:
    output_dir: eval/results
    max_concurrency: 4
    other_setting: true
  evaluators:
    accuracy:
      _type: rag_evaluator
optimizer:
  output_path: optimizer/results
  numeric:
    enabled: true
    n_trials: 10
llms:
  llm:
    _type: openai
    model_name: test-model
""".strip()
        )

        rebased = rebase_optimize_outputs(config, tmp_path / "results")

        # Paths rebased
        assert rebased["eval"]["general"]["output_dir"] == str(tmp_path / "results" / "eval" / "results")
        assert rebased["optimizer"]["output_path"] == str(tmp_path / "results" / "optimizer" / "results")

        # Other fields preserved
        assert rebased["eval"]["general"]["max_concurrency"] == 4
        assert rebased["eval"]["general"]["other_setting"] is True
        assert rebased["eval"]["evaluators"]["accuracy"]["_type"] == "rag_evaluator"
        assert rebased["optimizer"]["numeric"]["enabled"] is True
        assert rebased["optimizer"]["numeric"]["n_trials"] == 10
        assert rebased["llms"]["llm"]["_type"] == "openai"


# ---------------------------------------------------------------------------
# validate_llm_models
# ---------------------------------------------------------------------------


def _make_not_found_error(message: str) -> ClientNotFoundError:
    response = httpx.Response(
        404,
        request=httpx.Request("GET", "http://platform/apis/inference-gateway/v2/workspaces/default/virtual-models/x"),
        json={"detail": message},
    )
    return ClientNotFoundError(response)


class _RecordingVirtualModels:
    """Stub for the typed VirtualModels client that records lookup calls.

    ``missing`` is the set of ``model_name`` values that should raise
    :class:`nemo_platform_plugin.client.errors.NotFoundError`; everything else
    returns a sentinel object. ``side_effect`` overrides this to raise an
    arbitrary exception on every call (used to exercise the soft-fail branch).
    """

    def __init__(
        self,
        *,
        missing: set[str] | None = None,
        side_effect: Exception | None = None,
    ) -> None:
        self.missing = missing or set()
        self.side_effect = side_effect
        self.calls: list[dict[str, str]] = []

    def get_virtual_model(self, *, name: str, workspace: str) -> object:
        self.calls.append({"name": name, "workspace": workspace})
        if self.side_effect is not None:
            raise self.side_effect
        if name in self.missing:
            raise _make_not_found_error(f"VirtualModel {name!r} not found")
        return object()


class _StubInference:
    def __init__(self, virtual_models: _RecordingVirtualModels) -> None:
        self.virtual_models = virtual_models


class _StubSDKWithVirtualModels:
    def __init__(self, virtual_models: _RecordingVirtualModels) -> None:
        self.virtual_models = virtual_models
        self.inference = _StubInference(virtual_models)


def _platform_sdk_stub() -> NeMoPlatform:
    return cast(NeMoPlatform, object())


def _virtual_model_sdk(virtual_models: _RecordingVirtualModels) -> NeMoPlatform:
    return cast(NeMoPlatform, _StubSDKWithVirtualModels(virtual_models))


@pytest.fixture(autouse=True)
def _adapt_virtual_model_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agents_plugin import utils as utils_module

    real_client_from_platform = utils_module.client_from_platform

    def _client_from_platform(sdk: object, client_cls: type[object]) -> object:
        if isinstance(sdk, _StubSDKWithVirtualModels):
            return sdk.virtual_models
        return cast(Any, real_client_from_platform)(sdk, client_cls)

    monkeypatch.setattr(utils_module, "client_from_platform", _client_from_platform)


class TestValidateLLMModels:
    def test_happy_path_dedupes_repeated_model_names(self) -> None:
        """Same ``model_name`` under multiple LLM keys -> one lookup call."""
        config = {
            "llms": {
                "judge_llm": {"_type": "openai", "model_name": "shared-model"},
                "agent_llm": {"_type": "openai", "model_name": "shared-model"},
            }
        }
        vms = _RecordingVirtualModels()
        sdk = _virtual_model_sdk(vms)

        validate_llm_models(config, workspace="default", sdk=sdk)

        assert vms.calls == [{"name": "shared-model", "workspace": "default"}]

    def test_strips_workspace_qualifier_before_lookup(self) -> None:
        """A ``{workspace}/model`` value is stripped to bare ``model`` before lookup.

        Mirrors IGW's OpenAI proxy. ``nemo setup`` stores the qualified form as
        the default, so ``${NEMO_DEFAULT_MODEL}`` resolves to e.g.
        ``default/nvidia-nemotron`` — the VM route takes workspace as a separate
        path segment, so the bare name is what must reach the SDK.
        """
        config = {
            "llms": {
                "agent_llm": {"_type": "openai", "model_name": "default/nvidia-nemotron"},
            }
        }
        vms = _RecordingVirtualModels()
        sdk = _virtual_model_sdk(vms)

        validate_llm_models(config, workspace="default", sdk=sdk)

        assert vms.calls == [{"name": "nvidia-nemotron", "workspace": "default"}]

    def test_qualified_name_routes_to_its_own_workspace(self) -> None:
        """A qualified ``ws/model`` looks up ``model`` in ``ws``, not the path workspace.

        ``parse_qualified_name`` splits on the first ``/``: a qualifier always
        wins over the fallback workspace, so a cross-workspace reference resolves
        where it points. Bare names fall back to the path workspace.
        """
        config = {
            "llms": {
                "bare": {"_type": "openai", "model_name": "plain-model"},
                "other_ws": {"_type": "openai", "model_name": "prod/foo"},
            }
        }
        vms = _RecordingVirtualModels()
        sdk = _virtual_model_sdk(vms)

        validate_llm_models(config, workspace="default", sdk=sdk)

        calls = sorted((c["workspace"], c["name"]) for c in vms.calls)
        assert calls == [("default", "plain-model"), ("prod", "foo")]

    def test_qualified_and_bare_forms_dedupe_to_one_lookup(self) -> None:
        """``default/foo`` and ``foo`` normalize to the same name -> one lookup.

        Dedup happens *after* stripping, so the same model spelled both ways
        across LLM keys costs a single lookup.
        """
        config = {
            "llms": {
                "qualified": {"_type": "openai", "model_name": "default/foo"},
                "bare": {"_type": "openai", "model_name": "foo"},
            }
        }
        vms = _RecordingVirtualModels()
        sdk = _virtual_model_sdk(vms)

        validate_llm_models(config, workspace="default", sdk=sdk)

        assert vms.calls == [{"name": "foo", "workspace": "default"}]

    def test_bare_workspace_prefix_normalizes_to_empty_and_is_skipped(self) -> None:
        """A ``model_name`` of exactly ``{workspace}/`` strips to '' and is skipped.

        The empty string is not a routable model name; it must not reach the SDK.
        """
        config = {
            "llms": {
                "empty_after_strip": {"_type": "openai", "model_name": "default/"},
            }
        }
        vms = _RecordingVirtualModels()
        sdk = _virtual_model_sdk(vms)

        validate_llm_models(config, workspace="default", sdk=sdk)

        assert vms.calls == []

    def test_distinct_model_names_each_get_one_call(self) -> None:
        config = {
            "llms": {
                "agent_llm": {"_type": "openai", "model_name": "model-a"},
                "judge_llm": {"_type": "nim", "model_name": "model-b"},
            }
        }
        vms = _RecordingVirtualModels()
        sdk = _virtual_model_sdk(vms)

        validate_llm_models(config, workspace="ws", sdk=sdk)

        names = sorted(call["name"] for call in vms.calls)
        assert names == ["model-a", "model-b"]
        assert all(call["workspace"] == "ws" for call in vms.calls)

    def test_missing_model_raises_value_error_with_actionable_message(self) -> None:
        config = {
            "llms": {
                "judge_llm": {"_type": "openai", "model_name": "missing-model"},
            }
        }
        vms = _RecordingVirtualModels(missing={"missing-model"})
        sdk = _virtual_model_sdk(vms)

        with pytest.raises(ValueError) as exc_info:
            validate_llm_models(config, workspace="default", sdk=sdk)

        message = str(exc_info.value)
        # Names the missing model + the YAML key + the workspace, and points
        # the user at concrete recovery steps.
        assert "'default/missing-model'" in message
        assert "llms.judge_llm.model_name" in message
        assert "'default'" in message
        assert "NEMO_DEFAULT_MODEL" in message
        assert "nemo inference virtual-models" in message

    def test_multiple_missing_models_listed_in_single_error(self) -> None:
        config = {
            "llms": {
                "judge_llm": {"_type": "openai", "model_name": "missing-1"},
                "agent_llm": {"_type": "nim", "model_name": "missing-2"},
            }
        }
        vms = _RecordingVirtualModels(missing={"missing-1", "missing-2"})
        sdk = _virtual_model_sdk(vms)

        with pytest.raises(ValueError) as exc_info:
            validate_llm_models(config, workspace="default", sdk=sdk)

        message = str(exc_info.value)
        assert "'default/missing-1'" in message
        assert "'default/missing-2'" in message

    def test_non_igw_llm_types_are_skipped(self) -> None:
        """LLMs whose ``_type`` doesn't route through IGW aren't validatable here."""
        config = {
            "llms": {
                # bedrock / unknown types bypass the gateway and have no
                # corresponding VirtualModel — skip rather than spuriously
                # raise NotFound.
                "bedrock_llm": {"_type": "aws_bedrock", "model_name": "anthropic.claude"},
                "openai_llm": {"_type": "openai", "model_name": "real-model"},
            }
        }
        vms = _RecordingVirtualModels()
        sdk = _virtual_model_sdk(vms)

        validate_llm_models(config, workspace="ws", sdk=sdk)

        assert [call["name"] for call in vms.calls] == ["real-model"]

    def test_unexpanded_env_var_placeholder_is_skipped(self) -> None:
        """Names like ``${NEMO_DEFAULT_MODEL}`` that survived expansion are skipped."""
        config = {
            "llms": {
                # User set neither NEMO_DEFAULT_MODEL nor a literal — the
                # placeholder fell through expand_env_vars unchanged.  Don't
                # confuse users by claiming "VirtualModel '${NEMO_DEFAULT_MODEL}'
                # not found"; the expand_env_vars warning already explained.
                "judge_llm": {"_type": "openai", "model_name": "${NEMO_DEFAULT_MODEL}"},
                "real_llm": {"_type": "openai", "model_name": "real-model"},
            }
        }
        vms = _RecordingVirtualModels()
        sdk = _virtual_model_sdk(vms)

        validate_llm_models(config, workspace="ws", sdk=sdk)

        assert [call["name"] for call in vms.calls] == ["real-model"]

    def test_partial_placeholder_is_skipped(self) -> None:
        """Even an embedded ``$VAR`` (no braces) is treated as unexpanded."""
        config = {
            "llms": {
                "judge_llm": {"_type": "openai", "model_name": "prefix-$SUFFIX"},
            }
        }
        vms = _RecordingVirtualModels()
        sdk = _virtual_model_sdk(vms)

        validate_llm_models(config, workspace="ws", sdk=sdk)

        assert vms.calls == []

    def test_missing_model_name_is_skipped(self) -> None:
        """LLM blocks without a ``model_name`` aren't validatable; don't crash."""
        config = {
            "llms": {
                "broken_llm": {"_type": "openai"},  # no model_name at all
                "real_llm": {"_type": "openai", "model_name": "real-model"},
            }
        }
        vms = _RecordingVirtualModels()
        sdk = _virtual_model_sdk(vms)

        validate_llm_models(config, workspace="ws", sdk=sdk)

        assert [call["name"] for call in vms.calls] == ["real-model"]

    def test_empty_llms_block_is_noop(self) -> None:
        config: dict[str, Any] = {"llms": {}}
        vms = _RecordingVirtualModels()
        sdk = _virtual_model_sdk(vms)

        validate_llm_models(config, workspace="ws", sdk=sdk)

        assert vms.calls == []

    def test_missing_llms_key_is_noop(self) -> None:
        """Configs that don't declare any LLMs (e.g. trace-only fragments)."""
        config: dict[str, Any] = {"general": {"telemetry": {}}}
        vms = _RecordingVirtualModels()
        sdk = _virtual_model_sdk(vms)

        validate_llm_models(config, workspace="ws", sdk=sdk)

        assert vms.calls == []

    def test_soft_fails_on_non_not_found_exception(self, caplog: pytest.LogCaptureFixture) -> None:
        """Network/auth/5xx errors log a warning but don't gate the run."""
        config = {
            "llms": {
                "judge_llm": {"_type": "openai", "model_name": "some-model"},
            }
        }
        vms = _RecordingVirtualModels(side_effect=RuntimeError("connection refused"))
        sdk = _virtual_model_sdk(vms)

        with caplog.at_level("WARNING"):
            # Must not raise — the underlying eval/optimize call will surface
            # the real error if the model truly isn't reachable.
            validate_llm_models(config, workspace="ws", sdk=sdk)

        assert any("Could not validate LLM" in record.message for record in caplog.records)

    def test_non_dict_llm_entry_is_skipped(self) -> None:
        """Defensive: malformed YAML where an llm value isn't a dict."""
        config = {
            "llms": {
                "broken": "not-a-dict",
                "real": {"_type": "openai", "model_name": "real-model"},
            }
        }
        vms = _RecordingVirtualModels()
        sdk = _virtual_model_sdk(vms)

        validate_llm_models(config, workspace="ws", sdk=sdk)

        assert [call["name"] for call in vms.calls] == ["real-model"]


class TestPreflightValidateLLMModels:
    """Covers the eval/optimize-side helper.

    :func:`validate_llm_models` is exhaustively tested above; this class
    exercises only the wrapper-specific behaviour: YAML loading, env-var
    expansion, the optional ``agent_config`` merge, and the ``sdk=None``
    no-op short-circuit.
    """

    def _write(self, tmp_path: Path, body: str) -> Path:
        path = tmp_path / "config.yml"
        path.write_text(body)
        return path

    def test_happy_path_loads_yaml_and_validates(self, tmp_path: Path) -> None:
        config_path = self._write(
            tmp_path,
            "llms:\n  judge_llm:\n    _type: openai\n    model_name: real-model\n",
        )
        vms = _RecordingVirtualModels()
        sdk = _virtual_model_sdk(vms)

        preflight_validate_llm_models(config_path, workspace="ws", sdk=sdk)

        assert vms.calls == [{"name": "real-model", "workspace": "ws"}]

    def test_sdk_none_is_noop(self, tmp_path: Path) -> None:
        """``sdk=None`` short-circuits without reading the file."""
        # File doesn't even need to exist — sdk=None should bail before I/O.
        config_path = tmp_path / "does-not-exist.yml"
        # Should not raise FileNotFoundError.
        preflight_validate_llm_models(config_path, workspace="ws", sdk=None)

    def test_expands_env_vars_before_validation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``${NEMO_DEFAULT_MODEL}`` is expanded against the env before lookup."""
        monkeypatch.setenv("NEMO_DEFAULT_MODEL", "expanded-model")
        config_path = self._write(
            tmp_path,
            "llms:\n  judge_llm:\n    _type: openai\n    model_name: ${NEMO_DEFAULT_MODEL}\n",
        )
        vms = _RecordingVirtualModels()
        sdk = _virtual_model_sdk(vms)

        preflight_validate_llm_models(config_path, workspace="ws", sdk=sdk)

        # The expanded name reached the SDK; the literal placeholder did not.
        assert vms.calls == [{"name": "expanded-model", "workspace": "ws"}]

    def test_merges_agent_config_before_validation(self, tmp_path: Path) -> None:
        """``agent_config`` is merged under the YAML so its LLMs are validated too.

        Mirrors the optimize job's path where the agent's stored workflow
        contributes its own LLM block, and the optimize-side YAML adds the
        judge — both should be checked in one pass.
        """
        # Optimize-side YAML defines just the judge.
        config_path = self._write(
            tmp_path,
            "llms:\n  judge_llm:\n    _type: openai\n    model_name: judge-model\n",
        )
        # Agent-side stored config defines the agent's runtime LLM.
        agent_config: dict[str, Any] = {
            "llms": {
                "llm": {"_type": "openai", "model_name": "agent-model"},
            },
        }
        vms = _RecordingVirtualModels()
        sdk = _virtual_model_sdk(vms)

        preflight_validate_llm_models(
            config_path,
            workspace="ws",
            sdk=sdk,
            agent_config=agent_config,
        )

        names = sorted(call["name"] for call in vms.calls)
        assert names == ["agent-model", "judge-model"]

    def test_missing_model_propagates_validate_llm_models_error(self, tmp_path: Path) -> None:
        """Underlying ``validate_llm_models`` ``ValueError`` flows through unchanged."""
        config_path = self._write(
            tmp_path,
            "llms:\n  judge_llm:\n    _type: openai\n    model_name: missing-model\n",
        )
        vms = _RecordingVirtualModels(missing={"missing-model"})
        sdk = _virtual_model_sdk(vms)

        with pytest.raises(ValueError) as exc_info:
            preflight_validate_llm_models(config_path, workspace="default", sdk=sdk)

        # Sanity-check the message shape; full message coverage lives in
        # TestValidateLLMModels.
        assert "'default/missing-model'" in str(exc_info.value)
