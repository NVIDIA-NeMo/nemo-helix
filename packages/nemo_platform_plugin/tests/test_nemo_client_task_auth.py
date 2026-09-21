# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the plugin-side (zero-nmp-common) NemoClient task path.

Guards the three PR-800 review findings against the DefaultNemoClientProvider
and the public ``get_task_nemo_client`` API:

1. task clients must carry job-creator on-behalf-of delegation;
2. workload-identity must bootstrap bearer-token auth (no trusted X-NMP-* headers);
3. ``get_nemo_client(as_service=...)`` must stay *undelegated* (background
   controllers rely on that), so the task path is a distinct entry point.
"""

from __future__ import annotations

import json

import pytest
from nemo_platform_plugin import client_provider as cp
from nemo_platform_plugin.client_provider import (
    DefaultNemoClientProvider,
    get_async_task_nemo_client,
    get_nemo_client,
    get_task_nemo_client,
    set_nemo_client_provider,
)

CREATOR = {
    "id": "user:alice@acme.com",
    "email": "alice@acme.com",
    "groups": ["team-a", "team-b"],
}


@pytest.fixture(autouse=True)
def _force_default_provider(monkeypatch):
    # Pin the built-in default provider so the public helpers don't pick up an
    # entry-point-registered platform provider from the environment.
    set_nemo_client_provider(DefaultNemoClientProvider())
    monkeypatch.setenv("NMP_BASE_URL", "http://platform:8080")
    monkeypatch.delenv("NMP_WORKLOAD_IDENTITY_TOKEN_FILE", raising=False)
    yield
    set_nemo_client_provider(None)


# ---------------------------------------------------------------------------
# Claim 1 — task clients preserve creator delegation
# ---------------------------------------------------------------------------
def test_task_client_delegates_to_job_creator(monkeypatch):
    monkeypatch.setenv("NMP_PRINCIPAL", json.dumps(CREATOR))
    headers = get_task_nemo_client("evaluator")._default_headers

    assert headers["X-NMP-Internal"] == "true"
    assert headers["X-NMP-Principal-Id"] == "service:evaluator"
    assert headers["X-NMP-Actor-Aliases"] == "service:evaluator"
    assert headers["X-NMP-Principal-On-Behalf-Of"] == "user:alice@acme.com"
    assert headers["X-NMP-Principal-On-Behalf-Of-Email"] == "alice@acme.com"
    assert headers["X-NMP-Principal-On-Behalf-Of-Groups"] == "team-a,team-b"


def test_task_client_collapses_already_delegated_creator(monkeypatch):
    # If the creator principal was itself delegated, the acting identity wins.
    delegated = {
        "id": "service:jobs",
        "on_behalf_of": "user:bob@acme.com",
        "on_behalf_of_email": "bob@acme.com",
        "on_behalf_of_groups": ["team-z"],
    }
    monkeypatch.setenv("NMP_PRINCIPAL", json.dumps(delegated))
    headers = get_task_nemo_client("evaluator")._default_headers
    assert headers["X-NMP-Principal-Id"] == "service:evaluator"
    assert headers["X-NMP-Principal-On-Behalf-Of"] == "user:bob@acme.com"
    assert headers["X-NMP-Principal-On-Behalf-Of-Email"] == "bob@acme.com"
    assert headers["X-NMP-Principal-On-Behalf-Of-Groups"] == "team-z"


def test_task_client_without_principal_warns_and_stays_service(monkeypatch, caplog):
    monkeypatch.delenv("NMP_PRINCIPAL", raising=False)
    with caplog.at_level("WARNING"):
        headers = get_task_nemo_client("evaluator")._default_headers
    assert headers["X-NMP-Principal-Id"] == "service:evaluator"
    assert "X-NMP-Principal-On-Behalf-Of" not in headers
    assert "without on-behalf-of delegation" in caplog.text


async def test_async_task_client_delegates(monkeypatch):
    monkeypatch.setenv("NMP_PRINCIPAL", json.dumps(CREATOR))
    headers = get_async_task_nemo_client("evaluator")._default_headers
    assert headers["X-NMP-Principal-Id"] == "service:evaluator"
    assert headers["X-NMP-Principal-On-Behalf-Of"] == "user:alice@acme.com"


def test_general_as_service_stays_undelegated(monkeypatch):
    # Contrast: the generic entry point must NOT silently delegate, so
    # background controllers can act as an unscoped service principal.
    monkeypatch.setenv("NMP_PRINCIPAL", json.dumps(CREATOR))
    headers = get_nemo_client(as_service="evaluator")._default_headers
    assert headers["X-NMP-Principal-Id"] == "service:evaluator"
    assert "X-NMP-Principal-On-Behalf-Of" not in headers


# ---------------------------------------------------------------------------
# Claim 2 — workload identity bootstraps bearer auth, drops trusted headers
# ---------------------------------------------------------------------------
class _FakeExchangeProvider:
    def get_access_token(self) -> str:
        return "exchanged-token"

    async def get_access_token_async(self) -> str:
        return "exchanged-token"


@pytest.fixture
def _stub_workload_exchange(monkeypatch):
    captured = {}

    def _fake(*, base_url, subject_token_file):
        captured["base_url"] = base_url
        captured["subject_token_file"] = str(subject_token_file)
        return _FakeExchangeProvider()

    monkeypatch.setattr(
        "nemo_platform_plugin.client.oidc_factory.resolve_workload_exchange_provider",
        _fake,
    )
    return captured


def test_task_client_uses_workload_identity(monkeypatch, tmp_path, _stub_workload_exchange):
    token_file = tmp_path / "token"
    token_file.write_text("subject-token")
    monkeypatch.setenv("NMP_WORKLOAD_IDENTITY_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("NMP_PRINCIPAL", json.dumps(CREATOR))  # must be ignored in WI mode

    client = get_task_nemo_client("evaluator")

    # Bearer exchange wired up...
    assert isinstance(client._auth, _FakeExchangeProvider)
    assert _stub_workload_exchange["subject_token_file"] == str(token_file)
    assert _stub_workload_exchange["base_url"] == "http://platform:8080"
    # ...and NO trusted principal headers are sent (they'd be stripped/rejected
    # at a workload-identity trust boundary).
    assert "X-NMP-Principal-Id" not in client._default_headers
    assert client._default_headers.get("X-NMP-Internal") == "true"


async def test_async_task_client_uses_workload_identity(monkeypatch, tmp_path, _stub_workload_exchange):
    token_file = tmp_path / "token"
    token_file.write_text("subject-token")
    monkeypatch.setenv("NMP_WORKLOAD_IDENTITY_TOKEN_FILE", str(token_file))

    client = get_async_task_nemo_client("evaluator")
    assert isinstance(client._auth, _FakeExchangeProvider)
    assert "X-NMP-Principal-Id" not in client._default_headers


def test_default_provider_directly(monkeypatch):
    monkeypatch.setenv("NMP_PRINCIPAL", json.dumps(CREATOR))
    provider = DefaultNemoClientProvider()
    assert isinstance(provider, cp.NemoClientProvider)
    headers = provider.get_task_nemo_client("evaluator")._default_headers
    assert headers["X-NMP-Principal-On-Behalf-Of"] == "user:alice@acme.com"
