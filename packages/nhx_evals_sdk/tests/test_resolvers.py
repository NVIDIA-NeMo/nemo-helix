# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local secret lookup order: workspace-prefixed names first, then the bare secret name."""

from __future__ import annotations

import logging

import pytest
from nhx_evals_sdk.resolver_protocols import EnvSecretSource, MissingSecretError
from nhx_evals_sdk.resolvers import LocalSecretResolver, _candidate_env_names
from nhx_evals_sdk.values.common import SecretRef

WS_REF = SecretRef("my-workspace/probe-api-key")
BARE_REF = SecretRef("probe-api-key")
PREFIXED = "MY_WORKSPACE_PROBE_API_KEY"
BARE = "PROBE_API_KEY"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in {
        *_candidate_env_names(WS_REF.root),
        *_candidate_env_names(BARE_REF.root),
        *_candidate_env_names("9ws/9key"),
        *_candidate_env_names("9key"),
    }:
        monkeypatch.delenv(name, raising=False)


def test_local_resolver_is_an_env_secret_source() -> None:
    assert isinstance(LocalSecretResolver(), EnvSecretSource)


def test_prefixed_name_is_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PREFIXED, "a")
    assert LocalSecretResolver().env_var_for(WS_REF) == PREFIXED


def test_qualified_ref_falls_back_to_the_bare_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BARE, "b")
    assert LocalSecretResolver().env_var_for(WS_REF) == BARE


def test_prefixed_wins_over_bare_and_logs_both_names_without_values(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(PREFIXED, "value-a")
    monkeypatch.setenv(BARE, "value-b")
    with caplog.at_level(logging.INFO, logger="nhx_evals_sdk.resolvers"):
        assert LocalSecretResolver().env_var_for(WS_REF) == PREFIXED
    assert PREFIXED in caplog.text and BARE in caplog.text
    assert "value-a" not in caplog.text and "value-b" not in caplog.text


def test_bare_ref_has_no_prefixed_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PREFIXED, "a")
    with pytest.raises(MissingSecretError):
        LocalSecretResolver().env_var_for(BARE_REF)
    monkeypatch.setenv(BARE, "b")
    assert LocalSecretResolver().env_var_for(BARE_REF) == BARE


def test_empty_values_count_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PREFIXED, "")
    monkeypatch.setenv(BARE, "b")
    assert LocalSecretResolver().env_var_for(WS_REF) == BARE


def test_digit_leading_names_get_the_underscore_variant_in_both_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("_9KEY", "b")
    assert LocalSecretResolver().env_var_for(SecretRef("9ws/9key")) == "_9KEY"
    monkeypatch.setenv("_9WS_9KEY", "a")
    assert LocalSecretResolver().env_var_for(SecretRef("9ws/9key")) == "_9WS_9KEY"


def test_bare_fallback_off_keeps_qualified_refs_to_their_prefixed_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BARE, "b")
    resolver = LocalSecretResolver(bare_fallback=False)
    with pytest.raises(MissingSecretError):
        resolver.env_var_for(WS_REF)
    # A bare ref's own names are its bare names, so it resolves whatever the flag.
    assert resolver.env_var_for(BARE_REF) == BARE


@pytest.mark.asyncio
async def test_resolve_secret_reads_the_found_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PREFIXED, "value-a")
    monkeypatch.setenv(BARE, "value-b")
    assert await LocalSecretResolver().resolve_secret(WS_REF) == "value-a"
    monkeypatch.delenv(PREFIXED)
    assert await LocalSecretResolver().resolve_secret(WS_REF) == "value-b"
    monkeypatch.delenv(BARE)
    assert await LocalSecretResolver().resolve_secret(WS_REF) is None


def test_missing_secret_error_names_what_to_export() -> None:
    with pytest.raises(MissingSecretError) as bare_error:
        LocalSecretResolver().env_var_for(BARE_REF)
    assert str(bare_error.value) == (
        "secret 'probe-api-key' is not found. Make sure to set the PROBE_API_KEY env var "
        "before launching evaluation locally."
    )
    with pytest.raises(MissingSecretError) as workspace_error:
        LocalSecretResolver().env_var_for(WS_REF)
    assert str(workspace_error.value) == (
        "secret 'my-workspace/probe-api-key' is not found. Make sure to set the "
        "MY_WORKSPACE_PROBE_API_KEY (or PROBE_API_KEY) env var before launching evaluation locally."
    )
    with pytest.raises(MissingSecretError) as strict_error:
        LocalSecretResolver(bare_fallback=False).env_var_for(WS_REF)
    assert "(or" not in str(strict_error.value)
