# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NemoClient factory for task containers and services — the plugin-side
interface for building authenticated
:class:`~nemo_platform_plugin.client.client.NemoClient` /
:class:`~nemo_platform_plugin.client.client.AsyncNemoClient` handles.

This is the :class:`~nemo_platform_plugin.client.client.NemoClient` sibling of
:mod:`nemo_platform_plugin.sdk_provider`.  Plugin authors call
:func:`get_nemo_client` / :func:`get_async_nemo_client` here instead of
importing from ``nmp.common``.  This keeps ``nemo-platform-plugin`` free of any
``nmp-common`` dependency while still allowing the platform to register a richer
provider (URL routing, shared HTTP clients, OTEL headers, workload identity,
...) when ``nmp-common`` is installed.

Lookup order for the provider
-----------------------------

1. **Explicit override** — set via :func:`set_nemo_client_provider` (for tests).
2. **Entry-point discovery** — scans the ``nemo.client_provider`` group.
   When ``nmp-common`` is installed in the image (platform deployment), its
   provider is picked up automatically.
3. **Built-in default** — :class:`DefaultNemoClientProvider`, an env-var-based
   implementation that reads ``NMP_BASE_URL`` and ``NMP_PRINCIPAL``.  Works for
   local development and gateway-routed task containers.

For user-facing / CLI usage, prefer ``NemoClient.from_config()`` which reads
``~/.config/nmp/config.yaml`` and wires up OIDC token refresh / workload
identity token exchange.
"""

from __future__ import annotations

import json
import logging
import os
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from nemo_platform_plugin.client.auth import TokenProvider
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.client.constants import (
    WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR,
    is_workload_identity_token_file_set,
)

logger = logging.getLogger(__name__)

_INTERNAL_REQUEST_HEADER = "X-NMP-Internal"
_NMP_PRINCIPAL_ENVVAR = "NMP_PRINCIPAL"


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class NemoClientProvider(Protocol):
    """Contract for building authenticated NemoClient handles.

    Implementations live outside this module — the default is below;
    ``nmp-common`` ships a richer one registered via entry-point.
    """

    def get_nemo_client(
        self,
        *,
        as_service: str | None = None,
        internal: bool = False,
        on_behalf_of: str | None = None,
        workspace: str | None = None,
    ) -> NemoClient:
        """Build a sync NemoClient for the current service context."""

    def get_async_nemo_client(
        self,
        *,
        as_service: str | None = None,
        internal: bool = False,
        on_behalf_of: str | None = None,
        workspace: str | None = None,
    ) -> AsyncNemoClient:
        """Build an async NemoClient for the current service context."""

    def get_task_nemo_client(
        self,
        service_name: str,
        *,
        workspace: str | None = None,
    ) -> NemoClient:
        """Build a sync NemoClient for use inside a task container.

        Mirrors ``nmp.common.sdk_factory.get_task_sdk``: authenticate as
        ``service:{service_name}`` while acting on behalf of the job creator
        (read from ``NMP_PRINCIPAL``), or bootstrap workload-identity bearer-token
        exchange when ``NMP_WORKLOAD_IDENTITY_TOKEN_FILE`` is set.
        """

    def get_async_task_nemo_client(
        self,
        service_name: str,
        *,
        workspace: str | None = None,
    ) -> AsyncNemoClient:
        """Async counterpart of :meth:`get_task_nemo_client`."""


# ---------------------------------------------------------------------------
# Default provider (env-var based, zero nmp-common dependency)
# ---------------------------------------------------------------------------


def _read_principal_from_env() -> dict[str, Any] | None:
    """Read and parse ``NMP_PRINCIPAL`` from the environment.

    Returns ``None`` when the variable is absent or empty.  Raises
    :class:`ValueError` on malformed JSON so task containers surface the same
    error as ``nmp.common``.
    """
    raw = os.environ.get(_NMP_PRINCIPAL_ENVVAR)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {_NMP_PRINCIPAL_ENVVAR}: {exc}") from exc
    if not isinstance(data, dict) or not data.get("id"):
        return None
    return data


def _build_headers(
    *,
    as_service: str | None = None,
    internal: bool = False,
    on_behalf_of: str | None = None,
) -> dict[str, str]:
    """Build X-NMP-* headers from env vars and explicit parameters."""
    headers: dict[str, str] = {}

    if internal:
        headers[_INTERNAL_REQUEST_HEADER] = "true"

    if as_service is not None:
        headers["X-NMP-Principal-Id"] = f"service:{as_service}"
    else:
        principal = _read_principal_from_env()
        if principal is not None:
            headers["X-NMP-Principal-Id"] = principal["id"]
            if principal.get("email"):
                headers["X-NMP-Principal-Email"] = principal["email"]
            if principal.get("groups"):
                headers["X-NMP-Principal-Groups"] = ",".join(principal["groups"])
            if principal.get("on_behalf_of"):
                headers["X-NMP-Principal-On-Behalf-Of"] = principal["on_behalf_of"]
                if principal.get("on_behalf_of_email"):
                    headers["X-NMP-Principal-On-Behalf-Of-Email"] = principal["on_behalf_of_email"]
                if principal.get("on_behalf_of_groups"):
                    headers["X-NMP-Principal-On-Behalf-Of-Groups"] = ",".join(principal["on_behalf_of_groups"])

    if on_behalf_of is not None:
        # An explicit override wins over any on-behalf-of delegation carried by
        # the env principal.  Drop the principal's stale sub-headers so we don't
        # ship a mismatched delegated identity (correct id but wrong
        # email/groups) -- mirrors nmp.common.sdk_factory._get_default_headers.
        headers.pop("X-NMP-Principal-On-Behalf-Of-Email", None)
        headers.pop("X-NMP-Principal-On-Behalf-Of-Groups", None)
        headers["X-NMP-Principal-On-Behalf-Of"] = on_behalf_of

    return headers


def _effective_on_behalf_of(principal: dict[str, Any]) -> tuple[str, list[str], str | None]:
    """Collapse an env principal to its acting identity (id, groups, email).

    Mirrors :pyattr:`nmp.common.auth.Principal.effective_principal`: if the job
    creator's principal is itself delegated, the ``on_behalf_of`` identity wins;
    otherwise the principal's own identity is used.
    """
    if principal.get("on_behalf_of"):
        return (
            principal["on_behalf_of"],
            list(principal.get("on_behalf_of_groups") or []),
            principal.get("on_behalf_of_email"),
        )
    return principal["id"], list(principal.get("groups") or []), principal.get("email")


def _build_task_headers(service_name: str) -> dict[str, str]:
    """Headers for a task container: service principal + creator delegation.

    Wire-equivalent to ``get_task_sdk(as_service=service_name)`` in the
    non-workload-identity path -- ``service:{service_name}`` plus the full
    ``X-NMP-Principal-On-Behalf-Of*`` set derived from ``NMP_PRINCIPAL``.
    """
    headers: dict[str, str] = {
        _INTERNAL_REQUEST_HEADER: "true",
        "X-NMP-Principal-Id": f"service:{service_name}",
    }
    principal = _read_principal_from_env()
    if principal is None:
        logger.warning(
            "NMP_PRINCIPAL not set; task NemoClient will authenticate as service:%s without on-behalf-of delegation",
            service_name,
        )
        return headers
    obo_id, obo_groups, obo_email = _effective_on_behalf_of(principal)
    headers["X-NMP-Principal-On-Behalf-Of"] = obo_id
    if obo_groups:
        headers["X-NMP-Principal-On-Behalf-Of-Groups"] = ",".join(obo_groups)
    if obo_email:
        headers["X-NMP-Principal-On-Behalf-Of-Email"] = obo_email
    return headers


def _workload_identity_auth(base_url: str) -> TokenProvider:
    """Build a workload-identity token-exchange auth provider.

    Only call when :func:`is_workload_identity_token_file_set` is true.
    """
    from nemo_platform_plugin.client.oidc_factory import resolve_workload_exchange_provider

    token_file = os.environ[WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR]
    return resolve_workload_exchange_provider(base_url=base_url, subject_token_file=Path(token_file))


def _base_url() -> str:
    return os.environ.get("NMP_BASE_URL", "http://localhost:8080")


class DefaultNemoClientProvider:
    """Env-var-based provider that ships with the plugin package.

    Reads ``NMP_BASE_URL`` (default ``http://localhost:8080``) and
    ``NMP_PRINCIPAL`` — both are set by the jobs backend before launching task
    containers.  No ``nmp-common`` imports, so it works standalone.
    """

    def get_nemo_client(
        self,
        *,
        as_service: str | None = None,
        internal: bool = False,
        on_behalf_of: str | None = None,
        workspace: str | None = None,
    ) -> NemoClient:
        headers = _build_headers(as_service=as_service, internal=internal, on_behalf_of=on_behalf_of)
        return NemoClient(base_url=_base_url(), workspace=workspace, default_headers=headers or None)

    def get_async_nemo_client(
        self,
        *,
        as_service: str | None = None,
        internal: bool = False,
        on_behalf_of: str | None = None,
        workspace: str | None = None,
    ) -> AsyncNemoClient:
        headers = _build_headers(as_service=as_service, internal=internal, on_behalf_of=on_behalf_of)
        return AsyncNemoClient(base_url=_base_url(), workspace=workspace, default_headers=headers or None)

    def get_task_nemo_client(
        self,
        service_name: str,
        *,
        workspace: str | None = None,
    ) -> NemoClient:
        base_url = _base_url()
        if is_workload_identity_token_file_set():
            return NemoClient(
                base_url=base_url,
                workspace=workspace,
                auth=_workload_identity_auth(base_url),
                default_headers={_INTERNAL_REQUEST_HEADER: "true"},
            )
        return NemoClient(
            base_url=base_url,
            workspace=workspace,
            default_headers=_build_task_headers(service_name),
        )

    def get_async_task_nemo_client(
        self,
        service_name: str,
        *,
        workspace: str | None = None,
    ) -> AsyncNemoClient:
        base_url = _base_url()
        if is_workload_identity_token_file_set():
            return AsyncNemoClient(
                base_url=base_url,
                workspace=workspace,
                auth=_workload_identity_auth(base_url),
                default_headers={_INTERNAL_REQUEST_HEADER: "true"},
            )
        return AsyncNemoClient(
            base_url=base_url,
            workspace=workspace,
            default_headers=_build_task_headers(service_name),
        )


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------

_cached_provider: NemoClientProvider | None = None


def set_nemo_client_provider(provider: NemoClientProvider | None) -> None:
    """Override the provider (primarily for tests).

    Pass ``None`` to clear the override and fall back to entry-point discovery
    on the next call.
    """
    global _cached_provider
    _cached_provider = provider


def _resolve_provider() -> NemoClientProvider:
    """Resolve the provider once: explicit override → entry-point → default."""
    global _cached_provider
    if _cached_provider is not None:
        return _cached_provider

    # Scan entry-points.  nmp-common registers a provider; the nemo-platform
    # bundle inherits the same entry-point, so identical registrations are
    # legitimate duplicates.  A duplicate name pointing elsewhere is a
    # conflicting registration and must not depend on metadata ordering.
    eps = {}
    for ep in sorted(
        entry_points(group="nemo.client_provider"), key=lambda candidate: (candidate.name, candidate.value)
    ):
        existing = eps.get(ep.name)
        if existing is not None and existing.value != ep.value:
            targets = ", ".join(sorted((existing.value, ep.value)))
            raise RuntimeError(
                f"Conflicting NemoClient providers registered under 'nemo.client_provider' with name {ep.name!r}: "
                f"{targets}. Provider names must resolve to a single target."
            )
        eps[ep.name] = ep

    if len(eps) > 1:
        names = ", ".join(sorted(eps))
        raise RuntimeError(
            f"Multiple NemoClient providers registered under 'nemo.client_provider': {names}. "
            "Only the platform (nmp-common) should register a provider."
        )

    if eps:
        ep = next(iter(eps.values()))
        try:
            obj = ep.load()
            if isinstance(obj, type):
                obj = obj()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load or construct NemoClient provider {ep.name!r} from entry-point target {ep.value!r}."
            ) from exc
        if not isinstance(obj, NemoClientProvider):
            raise RuntimeError(
                f"NemoClient provider {ep.name!r} from entry-point target {ep.value!r} "
                "does not satisfy NemoClientProvider."
            )
        logger.debug("Using NemoClient provider from entry-point %r", ep.name)
        _cached_provider = obj
        return obj

    # Fall back to the built-in default only when no provider is registered.
    logger.debug("No entry-point NemoClient provider found; using DefaultNemoClientProvider")
    _cached_provider = DefaultNemoClientProvider()
    return _cached_provider


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_nemo_client(
    *,
    as_service: str | None = None,
    internal: bool = False,
    on_behalf_of: str | None = None,
    workspace: str | None = None,
) -> NemoClient:
    """Build a sync NemoClient for the current service context.

    Delegates to the resolved :class:`NemoClientProvider`.  Under the built-in
    default this reads ``NMP_BASE_URL`` (default ``http://localhost:8080``) and
    ``NMP_PRINCIPAL`` from the environment; under the platform provider it
    additionally routes service URLs, reuses the shared HTTP client, and injects
    OTEL headers.

    Args:
        as_service: If provided, authenticate as ``service:{as_service}``.
            If ``None``, propagate the principal read from ``NMP_PRINCIPAL``.
        internal: Mark requests as internal (service-to-service).
        on_behalf_of: Principal ID to act on behalf of.
        workspace: Default workspace used to fill ``{workspace}`` path params.
    """
    return _resolve_provider().get_nemo_client(
        as_service=as_service,
        internal=internal,
        on_behalf_of=on_behalf_of,
        workspace=workspace,
    )


def get_async_nemo_client(
    *,
    as_service: str | None = None,
    internal: bool = False,
    on_behalf_of: str | None = None,
    workspace: str | None = None,
) -> AsyncNemoClient:
    """Async counterpart of :func:`get_nemo_client`.

    Used by middleware and controllers that run inside the platform service
    process and need an async client.
    """
    return _resolve_provider().get_async_nemo_client(
        as_service=as_service,
        internal=internal,
        on_behalf_of=on_behalf_of,
        workspace=workspace,
    )


def get_task_nemo_client(service_name: str, *, workspace: str | None = None) -> NemoClient:
    """Build a sync NemoClient for use inside a task container.

    NemoClient counterpart of ``nmp.common.sdk_factory.get_task_sdk``.  Reads the
    job creator's principal from ``NMP_PRINCIPAL`` and authenticates as
    ``service:{service_name}`` while acting on behalf of that creator, or --
    when ``NMP_WORKLOAD_IDENTITY_TOKEN_FILE`` is set -- bootstraps
    workload-identity bearer-token exchange instead of trusted ``X-NMP-*``
    principal headers.

    Use this from task containers rather than ``get_nemo_client(as_service=...)``,
    which authenticates as an *undelegated* service principal.
    """
    return _resolve_provider().get_task_nemo_client(service_name, workspace=workspace)


def get_async_task_nemo_client(service_name: str, *, workspace: str | None = None) -> AsyncNemoClient:
    """Async counterpart of :func:`get_task_nemo_client`."""
    return _resolve_provider().get_async_task_nemo_client(service_name, workspace=workspace)


def get_forwarding_headers(client: NemoClient | AsyncNemoClient) -> dict[str, str]:
    """Return default headers that should be forwarded to nested platform calls.

    This is the NemoClient equivalent of
    :func:`nemo_platform_plugin.sdk_provider.get_forwarding_headers`. Guardrails
    middleware uses it to propagate service-principal and tracing headers into
    model calls made by cached LangChain clients.
    """
    return dict(client.default_headers)
