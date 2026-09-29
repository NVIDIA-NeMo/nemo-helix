# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Handing ``env_secrets`` and ``env_vars`` to a Harbor agent."""

from __future__ import annotations

import logging
import re

import pytest
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_env import (
    HARBOR_SENSITIVE_ENV_KEY_RE,
    harbor_env_templates,
    validate_harbor_env,
    warn_unscrubbed_secret_keys,
)
from nemo_evaluator_sdk.resolvers import LocalSecretResolver
from nemo_evaluator_sdk.values.common import SecretRef

REF = SecretRef("my-workspace/probe-api-key")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("MY_WORKSPACE_PROBE_API_KEY", "PROBE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_templates_name_the_source_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_WORKSPACE_PROBE_API_KEY", "a")
    assert harbor_env_templates({"LLM_API_KEY": REF}, LocalSecretResolver()) == {
        "LLM_API_KEY": "${MY_WORKSPACE_PROBE_API_KEY}"
    }


def test_missing_secret_is_named_with_its_key() -> None:
    with pytest.raises(ValueError) as excinfo:
        harbor_env_templates({"LLM_API_KEY": REF}, LocalSecretResolver())
    assert str(excinfo.value) == (
        "env_secrets['LLM_API_KEY'] -> secret 'my-workspace/probe-api-key' is not found. Make sure to set the "
        "MY_WORKSPACE_PROBE_API_KEY (or PROBE_API_KEY) env var before launching evaluation locally."
    )


@pytest.mark.parametrize(
    ("env_vars", "message"),
    [
        ({"X": "${WORKER_VAR}"}, "looks like a ${NAME} template"),
        ({"TOKENIZERS_PARALLELISM": "false"}, "Harbor treats keys matching"),
        ({"MAX_TOKENS": "4096"}, "Harbor treats keys matching"),
        ({"AUTH_ENABLED": "1"}, "Harbor treats keys matching"),
        ({"max_tokens": "4096"}, "Harbor treats keys matching"),
        ({"MODEL_CREDS": "sk-not-a-real-key-0123456789"}, "look like plaintext credentials"),
        ({"LLM_API_KEY": "x"}, "appear in both env_vars and env_secrets"),
    ],
)
def test_invalid_env_vars_are_refused(env_vars: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        validate_harbor_env(env_vars, {"LLM_API_KEY": REF})


def test_non_secret_env_vars_are_accepted() -> None:
    validate_harbor_env({"FABRIC_LOG": "debug", "AGENT_MODE": "fast", "AWS_REGION": "us-east-1"}, {})


def test_sensitive_key_regex_matches_harbor() -> None:
    harbor_env = pytest.importorskip("harbor.utils.env")
    assert HARBOR_SENSITIVE_ENV_KEY_RE.pattern == harbor_env._SENSITIVE_KEY_RE.pattern
    assert HARBOR_SENSITIVE_ENV_KEY_RE.flags == harbor_env._SENSITIVE_KEY_RE.flags


def test_accepted_env_vars_round_trip_through_harbor_agent_config() -> None:
    """Harbor masks values under sensitive keys, which would refuse every resume; accepted keys aren't."""
    trial_config = pytest.importorskip("harbor.models.trial.config")
    env_vars = {"FABRIC_LOG": "debug", "AGENT_MODE": "fast"}
    validate_harbor_env(env_vars, {})
    original = trial_config.AgentConfig(name="oracle", env={**env_vars, "LLM_API_KEY": "${PROBE_API_KEY}"})
    assert trial_config.AgentConfig.model_validate(original.model_dump(mode="json")) == original


def test_unscrubbed_secret_keys_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        warn_unscrubbed_secret_keys({"MODEL_ENDPOINT_X": REF, "MY_PASSWORD": REF, "X_AUTH": REF})
    assert "MODEL_ENDPOINT_X" in caplog.text
    assert "MY_PASSWORD" not in caplog.text and "X_AUTH" not in caplog.text
