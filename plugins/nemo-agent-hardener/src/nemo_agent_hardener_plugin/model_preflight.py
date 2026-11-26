# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model connectivity preflight — probe an OpenAI-compatible endpoint before a costly war-game.

A war-game spins up a Docker sandbox and runs for minutes; a mistyped model name or a wrong
``base_url``/key should fail in seconds, not after the sandbox is up. :func:`probe_models` lists the
models a credential can reach (``GET {base_url}/models``); :func:`validate_choice` turns that into a
verdict, and — crucially — hands back the *available* models so the caller can show the user what they
*can* use instead of what they typed. Providers that don't implement ``/models`` fall through
to a one-token ``/chat/completions`` probe, which is also what proves the key can actually generate.

The same helper backs both the interactive Studio "Test connection" and the launch-time preflight.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx
from nemo_helix_plugin.client.retry import is_retryable, retry_delay
from nemo_helix_plugin.client.types import RetryPolicy

_PROBE_TIMEOUT_S = 10.0
#: Brief provider hiccups (429, 502-504) a one-token probe retries so they don't fail a launch or "Test
#: connection": three attempts, 1s then 2s apart unless the provider sends ``Retry-After``.
PROBE_RETRY = RetryPolicy(max_retries=2, backoff_base=1.0, respect_retry_after_headers=True)
#: A probe is interactive, so a provider's ``Retry-After`` is honored only up to this.
_MAX_RETRY_AFTER_S = 5.0
#: Hosts where plaintext http:// carries the credential no further than this machine, so sending it
#: is safe. Anything else over http:// would put the key on the wire in the clear.
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of listing an endpoint's models. ``list_supported`` is False when it has no ``/models``.

    ``auth_ok`` is strictly about 401/403. Any other error status leaves it True and clears
    ``status_ok`` instead — a provider 500 or 429 is the provider's problem, not a bad credential.
    """

    reachable: bool
    auth_ok: bool
    available: list[str] = field(default_factory=list)
    list_supported: bool = True
    status_ok: bool = True
    detail: str = ""


@dataclass(frozen=True)
class Validation:
    """A model choice's verdict. ``available`` is populated so the UI/error can offer real options."""

    ok: bool
    reason: str = ""  # "", "auth", "unreachable", "provider_error", "unknown_model"
    available: list[str] = field(default_factory=list)
    detail: str = ""


def _auth_headers(base_url: str, api_key: str | None) -> tuple[dict[str, str], bool]:
    """Bearer header for *api_key*, and whether it was withheld.

    A plaintext endpoint would put the key on the wire in the clear, so the header is withheld
    rather than the probe refused: an endpoint that needs no credential still validates cleanly.
    """
    parsed = urlsplit(base_url)
    withheld = bool(api_key) and parsed.scheme != "https" and parsed.hostname not in _LOOPBACK_HOSTS
    return ({"Authorization": f"Bearer {api_key}"} if api_key and not withheld else {}), withheld


def probe_models(base_url: str, api_key: str | None, *, client: httpx.Client | None = None) -> ProbeResult:
    """List the models reachable at ``{base_url}/models`` with *api_key* (best-effort, never raises)."""
    url = base_url.rstrip("/") + "/models"
    headers, withheld_credential = _auth_headers(base_url, api_key)
    owns = client is None
    client = client or httpx.Client(timeout=_PROBE_TIMEOUT_S)
    try:
        resp = client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        return ProbeResult(reachable=False, auth_ok=False, detail=str(exc) or exc.__class__.__name__)
    finally:
        if owns:
            client.close()
    if resp.status_code in (401, 403):
        # Naming the withheld credential matters: otherwise this reads as "your key is bad" and
        # sends the user off to rotate a perfectly good one, when the fix is the http:// URL.
        detail = (
            f"HTTP {resp.status_code} — the credential was withheld because {base_url} is plaintext "
            "http://; use https:// (or localhost) to send it"
            if withheld_credential
            else f"HTTP {resp.status_code}"
        )
        return ProbeResult(reachable=True, auth_ok=False, detail=detail)
    if resp.status_code == 404:
        # No OpenAI-compatible model list — reachable, but we can't enumerate. Soft pass.
        return ProbeResult(reachable=True, auth_ok=True, list_supported=False, detail="endpoint has no /models")
    if resp.status_code >= 400:
        # Reachable and the credential wasn't rejected — the provider itself is erroring (5xx, 429, ...).
        return ProbeResult(reachable=True, auth_ok=True, status_ok=False, detail=f"HTTP {resp.status_code}")
    try:
        data = resp.json().get("data", [])
        ids = sorted(str(m["id"]) for m in data if isinstance(m, dict) and m.get("id"))
    except (ValueError, KeyError, TypeError):
        return ProbeResult(reachable=True, auth_ok=True, list_supported=False, detail="unparseable /models response")
    return ProbeResult(reachable=True, auth_ok=True, available=ids)


def _probe_chat(
    model: str, base_url: str, api_key: str | None, client: httpx.Client | None, retry: RetryPolicy
) -> Validation:
    """One-token ``POST /chat/completions``: the only check that the key can actually use *model*.

    ``/models`` is often public or key-agnostic, so listing alone can pass a key the endpoint rejects.
    """
    headers, withheld = _auth_headers(base_url, api_key)
    body = {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}
    url = base_url.rstrip("/") + "/chat/completions"
    owns = client is None
    client = client or httpx.Client(timeout=_PROBE_TIMEOUT_S)
    try:
        resp = client.post(url, headers=headers, json=body)
        for attempt in range(retry.max_retries):
            if not is_retryable(retry, resp):
                break
            time.sleep(min(retry_delay(retry, attempt, resp), _MAX_RETRY_AFTER_S))
            resp = client.post(url, headers=headers, json=body)
    except httpx.HTTPError as exc:
        return Validation(ok=False, reason="unreachable", detail=str(exc) or exc.__class__.__name__)
    finally:
        if owns:
            client.close()
    status = resp.status_code
    if status < 400:
        return Validation(ok=True)
    detail = f"HTTP {status}: {_server_message(resp)}".rstrip(": ")
    if status in (401, 403):
        suffix = " (credential withheld: endpoint is plaintext http://)" if withheld else ""
        return Validation(ok=False, reason="auth", detail=detail + suffix)
    if status == 404:
        return Validation(ok=False, reason="unknown_model", detail=detail)
    if status in (400, 422):
        # The key and model got past auth; only the probe's shape was refused (e.g. a reasoning model
        # that wants more than one token), which says nothing against the choice.
        return Validation(ok=True, detail=f"probe request refused, key accepted ({detail})")
    return Validation(ok=False, reason="provider_error", detail=detail)


def _server_message(resp: httpx.Response) -> str:
    """The ``error.message`` of an OpenAI-style error body, else the start of the raw body."""
    try:
        body = resp.json()
    except ValueError:
        body = None
    error = body.get("error") if isinstance(body, dict) else None
    message = error.get("message") if isinstance(error, dict) else error
    return str(message or resp.text)[:200]


def validate_choice(
    model: str | None,
    base_url: str,
    api_key: str | None,
    *,
    client: httpx.Client | None = None,
    retry: RetryPolicy = PROBE_RETRY,
) -> Validation:
    """Validate one model group's (model, base_url, key): reachable, authorized, and the model exists.

    A model that isn't in the reachable list fails with ``reason="unknown_model"`` and the available
    list, so the caller can present real choices. A model that passes the list (or whose endpoint has no
    ``/models``) must still answer a one-token chat call with the key. With no model to test and no
    ``/models``, reachability is all that can be confirmed. *retry* governs re-sending that chat call.
    """
    result = probe_models(base_url, api_key, client=client)
    if not result.reachable:
        return Validation(ok=False, reason="unreachable", detail=result.detail)
    if not result.auth_ok:
        return Validation(ok=False, reason="auth", detail=result.detail)
    if not result.status_ok:
        return Validation(ok=False, reason="provider_error", detail=result.detail)
    if result.list_supported and model and model not in result.available:
        return Validation(ok=False, reason="unknown_model", available=result.available)
    detail = "" if result.list_supported else result.detail
    if model:
        chat = _probe_chat(model, base_url, api_key, client, retry)
        if not chat.ok:
            return Validation(ok=False, reason=chat.reason, available=result.available, detail=chat.detail)
        detail = chat.detail or detail
    return Validation(ok=True, available=result.available, detail=detail)
