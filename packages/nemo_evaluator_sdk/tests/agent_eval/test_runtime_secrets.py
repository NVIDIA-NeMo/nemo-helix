# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolving ``env_secrets`` to values for harnesses handed values, such as Gym."""

from __future__ import annotations

import os

import pytest
from nemo_evaluator_sdk.agent_eval.runtimes import secrets as runtime_secrets
from nemo_evaluator_sdk.resolver_protocols import MissingSecretError
from nemo_evaluator_sdk.resolvers import LocalSecretResolver
from nemo_evaluator_sdk.values.common import SecretRef

REF = SecretRef("my-workspace/probe-api-key")


class _EnvOnlyResolver:
    def env_var_for(self, secret_ref: SecretRef, env_name: str) -> str:
        if os.environ.get(env_name):
            return env_name
        raise MissingSecretError(f"missing {secret_ref.root}")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("MY_WORKSPACE_PROBE_API_KEY", "PROBE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_values_resolve_from_the_found_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROBE_API_KEY", "b")
    assert runtime_secrets.env_secret_values({"LLM_API_KEY": REF}, LocalSecretResolver()) == {"LLM_API_KEY": "b"}


def test_a_missing_value_uses_the_resolver_message() -> None:
    with pytest.raises(MissingSecretError, match=r"env_secrets\['LLM_API_KEY'\] -> secret .* is not found"):
        runtime_secrets.env_secret_values({"LLM_API_KEY": REF}, LocalSecretResolver())


def test_env_secret_values_reads_named_variable_and_refuses_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "key")
    assert runtime_secrets.env_secret_values({"LLM_API_KEY": REF}, _EnvOnlyResolver()) == {"LLM_API_KEY": "key"}
    monkeypatch.setenv("LLM_API_KEY", "")
    with pytest.raises(MissingSecretError, match="missing my-workspace/probe-api-key"):
        runtime_secrets.env_secret_values({"LLM_API_KEY": REF}, _EnvOnlyResolver())


def test_env_secret_vars_prefixes_and_chains_the_source_error() -> None:
    source_error = MissingSecretError("missing my-workspace/probe-api-key")

    class MissingSource:
        def env_var_for(self, secret_ref: SecretRef, env_name: str) -> str:
            raise source_error

    with pytest.raises(MissingSecretError) as excinfo:
        runtime_secrets.env_secret_vars({"LLM_API_KEY": REF}, MissingSource())
    assert str(excinfo.value) == "env_secrets['LLM_API_KEY'] -> missing my-workspace/probe-api-key"
    assert excinfo.value.__cause__ is source_error
