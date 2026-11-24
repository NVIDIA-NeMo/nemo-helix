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


def test_probe_sends_the_credential_over_https() -> None:
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["authorization"] = req.headers.get("authorization")
        return httpx.Response(200, json={"data": []})

    probe_models("https://x/v1", "secret-key", client=_client(handler))
    assert captured["authorization"] == "Bearer secret-key"


def test_probe_never_sends_the_credential_over_plain_http() -> None:
    """An http:// endpoint would carry the key in the clear; withhold it instead."""
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["authorization"] = req.headers.get("authorization")
        return httpx.Response(200, json={"data": []})

    result = probe_models("http://attacker.example/v1", "secret-key", client=_client(handler))
    assert captured["authorization"] is None
    # Still probed for reachability — withholding the credential isn't the same as refusing to try.
    assert result.reachable


def test_a_withheld_credential_explains_itself_instead_of_blaming_the_key() -> None:
    """A bare 'HTTP 401' here would send the user off to rotate a perfectly good credential."""
    result = probe_models("http://x/v1", "secret-key", client=_client(lambda _r: httpx.Response(401)))

    assert not result.auth_ok
    assert "withheld" in result.detail and "http://" in result.detail

    verdict = validate_choice("a/model", "http://x/v1", "secret-key", client=_client(lambda _r: httpx.Response(401)))
    assert not verdict.ok and verdict.reason == "auth"
    assert "withheld" in verdict.detail


def test_a_plain_http_endpoint_needing_no_credential_still_validates() -> None:
    """Withholding the header must not break an open endpoint — the probe still runs."""

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "real/model"}]})

    verdict = validate_choice("real/model", "http://x/v1", "secret-key", client=_client(handler))
    assert verdict.ok


def test_probe_allows_the_credential_over_plain_http_localhost() -> None:
    """A local dev inference server has no network hop to leak the credential over."""
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["authorization"] = req.headers.get("authorization")
        return httpx.Response(200, json={"data": []})

    probe_models("http://localhost:8000/v1", "secret-key", client=_client(handler))
    assert captured["authorization"] == "Bearer secret-key"


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


def _split_handler(models: httpx.Response, chat: httpx.Response) -> Any:
    def handler(req: httpx.Request) -> httpx.Response:
        return chat if req.method == "POST" else models

    return handler


def test_a_key_rejected_by_chat_fails_even_when_models_lists_fine() -> None:
    handler = _split_handler(httpx.Response(200, json={"data": [{"id": "m"}]}), httpx.Response(401))
    verdict = validate_choice("m", "https://x/v1", "bad", client=_client(handler))
    assert not verdict.ok and verdict.reason == "auth"


def test_chat_probe_is_one_token_with_the_key() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"data": [{"id": "m"}]})

    assert validate_choice("m", "https://x/v1", "k", client=_client(handler)).ok
    post = next(r for r in seen if r.method == "POST")
    assert post.url.path == "/v1/chat/completions"
    assert post.headers["authorization"] == "Bearer k"
    assert b'"max_tokens":1' in post.content.replace(b" ", b"")


def test_missing_models_route_defers_to_the_chat_probe() -> None:
    ok = _split_handler(httpx.Response(404), httpx.Response(200, json={}))
    assert validate_choice("m", "https://x/v1", "k", client=_client(ok)).ok

    bad = _split_handler(httpx.Response(404), httpx.Response(404))
    verdict = validate_choice("m", "https://x/v1", "k", client=_client(bad))
    assert not verdict.ok and verdict.reason == "unknown_model"

    denied = _split_handler(httpx.Response(404), httpx.Response(403))
    assert validate_choice("m", "https://x/v1", "k", client=_client(denied)).reason == "auth"


@pytest.mark.parametrize("status", [400, 422])
def test_a_refused_probe_shape_still_passes_the_key(status: int) -> None:
    """A reasoning model can refuse max_tokens=1 with a 400; the key and model were still accepted."""
    chat = httpx.Response(status, json={"error": {"message": "max_tokens too small"}})
    handler = _split_handler(httpx.Response(200, json={"data": [{"id": "m"}]}), chat)
    verdict = validate_choice("m", "https://x/v1", "k", client=_client(handler))
    assert verdict.ok
    assert "max_tokens too small" in verdict.detail


def test_a_transient_chat_error_is_retried_then_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin import model_preflight

    monkeypatch.setattr(model_preflight.time, "sleep", lambda _s: None)
    posts: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        posts.append(1)
        return httpx.Response(503) if len(posts) < 3 else httpx.Response(200, json={})

    assert validate_choice("m", "https://x/v1", "k", client=_client(handler)).ok
    assert len(posts) == 3


def test_a_persistent_transient_error_fails_with_the_server_message(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin import model_preflight

    monkeypatch.setattr(model_preflight.time, "sleep", lambda _s: None)
    chat = httpx.Response(429, json={"error": {"message": "slow down"}})
    verdict = validate_choice("m", "https://x/v1", "k", client=_client(_split_handler(httpx.Response(404), chat)))
    assert verdict.reason == "provider_error" and "slow down" in verdict.detail


def test_auth_detail_carries_the_server_message() -> None:
    chat = httpx.Response(403, json={"error": {"message": "key lacks access to m"}})
    verdict = validate_choice("m", "https://x/v1", "k", client=_client(_split_handler(httpx.Response(404), chat)))
    assert verdict.reason == "auth" and "key lacks access to m" in verdict.detail


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
