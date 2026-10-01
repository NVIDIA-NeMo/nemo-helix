# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolving ``env_secrets`` to values for harnesses handed values, such as Gym."""

from __future__ import annotations

import pytest
from nemo_evaluator_sdk.agent_eval.runtimes.secrets import resolve_env_secrets
from nemo_evaluator_sdk.resolvers import LocalSecretResolver
from nemo_evaluator_sdk.values.common import SecretRef

REF = SecretRef("my-workspace/probe-api-key")


class _ValueOnlyResolver:
    async def resolve_secret(self, secret_ref: SecretRef) -> str | None:
        return None


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("MY_WORKSPACE_PROBE_API_KEY", "PROBE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.asyncio
async def test_values_resolve_from_the_found_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROBE_API_KEY", "b")
    assert await resolve_env_secrets({"LLM_API_KEY": REF}, LocalSecretResolver()) == {"LLM_API_KEY": "b"}


@pytest.mark.asyncio
async def test_a_missing_value_uses_the_resolver_message() -> None:
    with pytest.raises(ValueError, match=r"env_secrets\['LLM_API_KEY'\] -> secret .* is not found"):
        await resolve_env_secrets({"LLM_API_KEY": REF}, LocalSecretResolver())


@pytest.mark.asyncio
async def test_value_resolution_names_the_resolver_when_it_has_no_message() -> None:
    with pytest.raises(ValueError, match="could not be resolved by _ValueOnlyResolver"):
        await resolve_env_secrets({"LLM_API_KEY": REF}, _ValueOnlyResolver())
