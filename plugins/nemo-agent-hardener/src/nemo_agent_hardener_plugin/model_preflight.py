# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model connectivity preflight — probe an OpenAI-compatible endpoint before a costly war-game.

A war-game spins up a Docker sandbox and runs for minutes; a mistyped model name or a wrong
``base_url``/key should fail in seconds, not after the sandbox is up. :func:`probe_models` lists the
models a credential can reach (``GET {base_url}/models``); :func:`validate_choice` turns that into a
verdict, and — crucially — hands back the *available* models so the caller can show the user what they
*can* use instead of what they typed. Providers that don't implement ``/models`` are a soft pass
(reachable, list unknown) rather than a hard failure.

The same helper backs both the interactive Studio "Test connection" and the launch-time preflight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

_PROBE_TIMEOUT_S = 10.0


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


def probe_models(base_url: str, api_key: str | None, *, client: httpx.Client | None = None) -> ProbeResult:
    """List the models reachable at ``{base_url}/models`` with *api_key* (best-effort, never raises)."""
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
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
        return ProbeResult(reachable=True, auth_ok=False, detail=f"HTTP {resp.status_code}")
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


def validate_choice(
    model: str | None, base_url: str, api_key: str | None, *, client: httpx.Client | None = None
) -> Validation:
    """Validate one model group's (model, base_url, key): reachable, authorized, and the model exists.

    A model that isn't in the reachable list fails with ``reason="unknown_model"`` and the available
    list, so the caller can present real choices. If the endpoint has no ``/models`` we can't verify the
    name — treat it as a pass (reachability + auth already confirmed).
    """
    result = probe_models(base_url, api_key, client=client)
    if not result.reachable:
        return Validation(ok=False, reason="unreachable", detail=result.detail)
    if not result.auth_ok:
        return Validation(ok=False, reason="auth", detail=result.detail)
    if not result.status_ok:
        return Validation(ok=False, reason="provider_error", detail=result.detail)
    if not result.list_supported:
        return Validation(ok=True, detail=result.detail)
    if model and model not in result.available:
        return Validation(ok=False, reason="unknown_model", available=result.available)
    return Validation(ok=True, available=result.available)
