# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the model connectivity preflight (probe + validate + launch-time guard)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from nemo_agent_hardener_plugin.jobs import run as run_module
from nemo_agent_hardener_plugin.jobs.errors import CATEGORY_MODEL_UNAVAILABLE, AgentHardenerRunError
from nemo_agent_hardener_plugin.model_config import ModelChoice, WarGameModels
from nemo_agent_hardener_plugin.model_preflight import probe_models, validate_choice


def _client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_probe_lists_available_models() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "b/model"}, {"id": "a/model"}]})

    result = probe_models("https://x/v1", "key", client=_client(handler))
    assert result.reachable and result.auth_ok
    assert result.available == ["a/model", "b/model"]  # sorted


def test_probe_auth_failure() -> None:
    result = probe_models("https://x/v1", "bad", client=_client(lambda _r: httpx.Response(401)))
    assert result.reachable and not result.auth_ok


def test_probe_no_model_list_is_soft_pass() -> None:
    result = probe_models("https://x/v1", "key", client=_client(lambda _r: httpx.Response(404)))
    assert result.reachable and result.auth_ok and not result.list_supported


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_a_provider_side_error_is_not_reported_as_bad_credentials(status: int) -> None:
    """A 5xx/429 used to set auth_ok=False, telling the user to go rotate a perfectly good key."""
    result = probe_models("https://x/v1", "key", client=_client(lambda _r: httpx.Response(status)))
    assert result.reachable and result.auth_ok and not result.status_ok

    verdict = validate_choice("a/model", "https://x/v1", "key", client=_client(lambda _r: httpx.Response(status)))
    assert not verdict.ok and verdict.reason == "provider_error"
    assert f"HTTP {status}" in verdict.detail


@pytest.mark.parametrize("status", [401, 403])
def test_only_401_403_are_credential_failures(status: int) -> None:
    verdict = validate_choice("a/model", "https://x/v1", "bad", client=_client(lambda _r: httpx.Response(status)))
    assert not verdict.ok and verdict.reason == "auth"


def test_validate_unknown_model_returns_available() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "real/model"}]})

    v = validate_choice("typo/model", "https://x/v1", "key", client=_client(handler))
    assert not v.ok and v.reason == "unknown_model" and v.available == ["real/model"]


def test_validate_known_model_ok() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "real/model"}]})

    v = validate_choice("real/model", "https://x/v1", "key", client=_client(handler))
    assert v.ok


def test_preflight_raises_model_unavailable_with_list(monkeypatch: pytest.MonkeyPatch) -> None:
    # A configured analysis model that the endpoint doesn't serve → classified failure listing real models.
    monkeypatch.setattr(
        run_module,
        "validate_choice",
        lambda model, base_url, key: SimpleNamespace(
            ok=False, reason="unknown_model", available=["real/a", "real/b"], detail=""
        ),
    )
    models = WarGameModels(analysis=ModelChoice(model="typo", base_url="https://x/v1"))
    with pytest.raises(AgentHardenerRunError) as exc:
        run_module._preflight_models(models, sdk=None, workspace="default", default_key="k")
    assert exc.value.category == CATEGORY_MODEL_UNAVAILABLE
    assert "real/a" in str(exc.value)


def test_preflight_skips_default_only_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    # No model/base_url set → nothing probed (defaults are known-good), so validate is never called.
    called = {"n": 0}
    monkeypatch.setattr(
        run_module,
        "validate_choice",
        lambda *a, **k: called.__setitem__("n", called["n"] + 1) or SimpleNamespace(ok=True),
    )
    run_module._preflight_models(WarGameModels(), sdk=None, workspace="default", default_key="k")
    assert called["n"] == 0
