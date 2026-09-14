# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK factory for task containers — the plugin-side interface for getting
authenticated :class:`~nemo_platform.NeMoPlatform` handles.

Plugin authors use :func:`get_task_sdk` in their ``__main__.py`` entrypoints
instead of importing from ``nmp.common.sdk_factory``.  This keeps the
``nemo-platform-plugin`` package free of ``nmp-common`` dependencies while
still allowing the platform to register a richer provider (with URL routing,
shared HTTP clients, OTEL headers, etc.) when ``nmp-common`` is installed.

Lookup order for the provider
-----------------------------

1. **Explicit override** — set via :func:`set_sdk_provider` (for tests).
2. **Entry-point discovery** — scans the ``nemo.sdk_provider`` group.
   When ``nmp-common`` is installed in the image (platform deployment), its
   provider is picked up automatically.
3. **Built-in default** — :class:`DefaultSDKProvider`, an env-var-based
   implementation that reads ``NMP_BASE_URL`` and ``NMP_PRINCIPAL``.  Works
   for local development and gateway-routed task containers.

Usage from a plugin ``__main__.py``::

    from nemo_platform_plugin.sdk_provider import get_task_sdk
    from nemo_platform_plugin.tasks.dispatcher import run_task

    sdk = get_task_sdk("evaluator")
    sys.exit(run_task(EvaluateJob, sdk=sdk))
"""

from __future__ import annotations

import json
import logging
import os
from importlib.metadata import entry_points
from typing import Any, Protocol, TypeVar, runtime_checkable

from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.client.constants import is_workload_identity_token_file_set

_SDKT = TypeVar("_SDKT", NeMoPlatform, AsyncNeMoPlatform)

logger = logging.getLogger(__name__)

# Header name the platform uses to mark internal (service-to-service) requests.
_INTERNAL_REQUEST_HEADER = "X-NMP-Internal"

# Environment variable the jobs backend writes with the job creator's
# principal (JSON-serialised).
_NMP_PRINCIPAL_ENVVAR = "NMP_PRINCIPAL"


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SDKProvider(Protocol):
    """Contract for building authenticated SDK handles.

    Implementations live outside this module — the default is below;
    ``nmp-common`` ships a richer one registered via entry-point.
    """

    def get_task_sdk(self, service_name: str) -> NeMoPlatform:
        """Build an SDK for use inside a platform-spawned task container.

        The returned client authenticates as ``service:{service_name}`` and,
        when ``NMP_PRINCIPAL`` is set, acts on behalf of the job creator.
        """

    def get_async_task_sdk(self, service_name: str) -> AsyncNeMoPlatform:
        """Async counterpart of :meth:`get_task_sdk` for use inside a task container.

        Authenticates as ``service:{service_name}`` and, when ``NMP_PRINCIPAL``
        is set, acts on behalf of the job creator with the *full* delegated
        identity (on-behalf-of id, email, and groups) — wire-identical to
        :meth:`get_task_sdk`, just async.
        """

    def get_platform_sdk(
        self,
        *,
        as_service: str | None = None,
        internal: bool = False,
        on_behalf_of: str | None = None,
    ) -> NeMoPlatform:
        """Build a general-purpose sync SDK handle.

        Lower-level than :meth:`get_task_sdk` — callers choose their own
        auth mode.
        """

    def get_async_platform_sdk(
        self,
        *,
        as_service: str | None = None,
        internal: bool = False,
        on_behalf_of: str | None = None,
    ) -> AsyncNeMoPlatform:
        """Build a general-purpose async SDK handle.

        Used by middleware and controllers that run inside the platform
        service process.
        """


# ---------------------------------------------------------------------------
# Default provider (env-var based, zero nmp-common dependency)
# ---------------------------------------------------------------------------


def _read_principal_from_env() -> dict[str, Any] | None:
    """Read and parse ``NMP_PRINCIPAL`` from the environment.

    Returns ``None`` when the variable is absent or empty.  Raises
    :class:`ValueError` on malformed JSON (matches ``nmp.common``
    behaviour so task containers surface the same error).
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


def _on_behalf_of_headers(principal: dict[str, Any]) -> dict[str, str]:
    """Derive ``X-NMP-Principal-On-Behalf-Of*`` headers from a principal dict.

    Mirrors the header logic in ``nmp.common.sdk_factory._get_default_headers``
    so the default provider is wire-compatible.
    """
    # When the principal has an on_behalf_of field, use the effective principal
    # (the on-behalf-of identity).  Otherwise, use the principal itself.
    if principal.get("on_behalf_of"):
        effective_id = principal["on_behalf_of"]
        effective_email = principal.get("on_behalf_of_email")
        effective_groups = principal.get("on_behalf_of_groups") or []
    else:
        effective_id = principal["id"]
        effective_email = principal.get("email")
        effective_groups = principal.get("groups") or []

    headers: dict[str, str] = {"X-NMP-Principal-On-Behalf-Of": effective_id}
    if effective_email:
        headers["X-NMP-Principal-On-Behalf-Of-Email"] = effective_email
    if effective_groups:
        headers["X-NMP-Principal-On-Behalf-Of-Groups"] = ",".join(effective_groups)
    return headers


def _workload_identity_headers(*, internal: bool) -> dict[str, str]:
    return {_INTERNAL_REQUEST_HEADER: "true"} if internal else {}


class DefaultSDKProvider:
    """Env-var-based provider that ships with the plugin package.

    Reads ``NMP_BASE_URL`` (default ``http://localhost:8080``) and
    ``NMP_PRINCIPAL`` — both are set by the jobs backend before launching
    task containers.  No ``nmp-common`` imports.
    """

    def get_task_sdk(self, service_name: str) -> NeMoPlatform:
        if is_workload_identity_token_file_set():
            return NeMoPlatform(
                base_url=self._base_url(),
                default_headers=_workload_identity_headers(internal=True),
            )

        headers: dict[str, str] = {
            "X-NMP-Principal-Id": f"service:{service_name}",
            _INTERNAL_REQUEST_HEADER: "true",
        }

        principal = _read_principal_from_env()
        if principal is not None:
            headers.update(_on_behalf_of_headers(principal))
        else:
            logger.warning(
                "%s not set; task SDK will authenticate as service:%s without on-behalf-of delegation",
                _NMP_PRINCIPAL_ENVVAR,
                service_name,
            )

        return NeMoPlatform(
            base_url=self._base_url(),
            default_headers=headers,
        )

    def get_async_task_sdk(self, service_name: str) -> AsyncNeMoPlatform:
        # Async mirror of get_task_sdk: identical headers (service principal,
        # internal marker, and full on-behalf-of id/email/groups), async client.
        if is_workload_identity_token_file_set():
            return AsyncNeMoPlatform(
                base_url=self._base_url(),
                default_headers=_workload_identity_headers(internal=True),
            )

        headers: dict[str, str] = {
            "X-NMP-Principal-Id": f"service:{service_name}",
            _INTERNAL_REQUEST_HEADER: "true",
        }

        principal = _read_principal_from_env()
        if principal is not None:
            headers.update(_on_behalf_of_headers(principal))
        else:
            logger.warning(
                "%s not set; async task SDK will authenticate as service:%s without on-behalf-of delegation",
                _NMP_PRINCIPAL_ENVVAR,
                service_name,
            )

        return AsyncNeMoPlatform(
            base_url=self._base_url(),
            default_headers=headers,
        )

    def _make_sdk(
        self,
        cls: type[_SDKT],
        *,
        as_service: str | None = None,
        internal: bool = False,
        on_behalf_of: str | None = None,
    ) -> _SDKT:
        if as_service is None and on_behalf_of is None and is_workload_identity_token_file_set():
            headers = _workload_identity_headers(internal=internal)
            return cls(base_url=self._base_url(), default_headers=headers or None)

        headers = self._build_headers(as_service=as_service, internal=internal, on_behalf_of=on_behalf_of)
        return cls(base_url=self._base_url(), default_headers=headers or None)

    def get_platform_sdk(
        self,
        *,
        as_service: str | None = None,
        internal: bool = False,
        on_behalf_of: str | None = None,
    ) -> NeMoPlatform:
        return self._make_sdk(NeMoPlatform, as_service=as_service, internal=internal, on_behalf_of=on_behalf_of)

    def get_async_platform_sdk(
        self,
        *,
        as_service: str | None = None,
        internal: bool = False,
        on_behalf_of: str | None = None,
    ) -> AsyncNeMoPlatform:
        return self._make_sdk(AsyncNeMoPlatform, as_service=as_service, internal=internal, on_behalf_of=on_behalf_of)

    @staticmethod
    def _build_headers(
        *,
        as_service: str | None = None,
        internal: bool = False,
        on_behalf_of: str | None = None,
    ) -> dict[str, str]:
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
                    headers.update(_on_behalf_of_headers(principal))

        if on_behalf_of is not None:
            headers.pop("X-NMP-Principal-On-Behalf-Of-Email", None)
            headers.pop("X-NMP-Principal-On-Behalf-Of-Groups", None)
            headers["X-NMP-Principal-On-Behalf-Of"] = on_behalf_of

        return headers

    @staticmethod
    def _base_url() -> str:
        return os.environ.get("NMP_BASE_URL", "http://localhost:8080")


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------

_cached_provider: SDKProvider | None = None


def set_sdk_provider(provider: SDKProvider | None) -> None:
    """Override the provider (primarily for tests).

    Pass ``None`` to clear the override and fall back to entry-point
    discovery on the next call.
    """
    global _cached_provider
    _cached_provider = provider


def _resolve_provider() -> SDKProvider:
    """Resolve the provider once: explicit override → entry-point → default."""
    global _cached_provider
    if _cached_provider is not None:
        return _cached_provider

    # Scan entry-points.  nmp-common registers a provider; the nemo-platform
    # bundle inherits the same entry-point, so identical registrations are
    # legitimate duplicates.  A duplicate name pointing elsewhere is a
    # conflicting registration and must not depend on metadata ordering.
    eps = {}
    for ep in sorted(entry_points(group="nemo.sdk_provider"), key=lambda candidate: (candidate.name, candidate.value)):
        existing = eps.get(ep.name)
        if existing is not None and existing.value != ep.value:
            targets = ", ".join(sorted((existing.value, ep.value)))
            raise RuntimeError(
                f"Conflicting SDK providers registered under 'nemo.sdk_provider' with name {ep.name!r}: "
                f"{targets}. Provider names must resolve to a single target."
            )
        eps[ep.name] = ep

    if len(eps) > 1:
        names = ", ".join(sorted(eps))
        raise RuntimeError(
            f"Multiple SDK providers registered under 'nemo.sdk_provider': {names}. "
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
                f"Failed to load or construct SDK provider {ep.name!r} from entry-point target {ep.value!r}."
            ) from exc
        if not isinstance(obj, SDKProvider):
            raise RuntimeError(
                f"SDK provider {ep.name!r} from entry-point target {ep.value!r} does not satisfy SDKProvider."
            )
        logger.debug("Using SDK provider from entry-point %r", ep.name)
        _cached_provider = obj
        return obj

    # Fall back to the built-in default only when no provider is registered.
    logger.debug("No entry-point SDK provider found; using DefaultSDKProvider")
    _cached_provider = DefaultSDKProvider()
    return _cached_provider


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_task_sdk(service_name: str) -> NeMoPlatform:
    """Build an authenticated SDK for use inside a platform task container.

    Equivalent to the legacy ``nmp.common.sdk_factory.get_task_sdk`` but
    without requiring ``nmp-common`` as a dependency.

    Args:
        service_name: Plugin/service name used as the service principal
            (e.g. ``"evaluator"``).

    Returns:
        A :class:`~nemo_platform.NeMoPlatform` handle with internal +
        on-behalf-of headers set.
    """
    return _resolve_provider().get_task_sdk(service_name)


def get_async_task_sdk(service_name: str) -> AsyncNeMoPlatform:
    """Async counterpart of :func:`get_task_sdk` for use inside a task container.

    For a (synchronous) job ``run`` that needs to drive an async helper — e.g. an entity-store
    write — without fabricating its own client. Authenticates as ``service:{service_name}`` and, when
    ``NMP_PRINCIPAL`` is set, on behalf of the job creator with the full delegated identity
    (on-behalf-of id, email, and groups) — wire-identical to :func:`get_task_sdk`.

    A dedicated provider method (not a wrapper over :func:`get_async_platform_sdk`) so each provider
    mirrors its own sync :meth:`SDKProvider.get_task_sdk` exactly; the platform provider routes URLs
    and reuses its shared async client, the default provider uses env-var headers.
    """
    return _resolve_provider().get_async_task_sdk(service_name)


def get_platform_sdk(
    *,
    as_service: str | None = None,
    internal: bool = False,
    on_behalf_of: str | None = None,
) -> NeMoPlatform:
    """Build a general-purpose sync SDK handle.

    Lower-level than :func:`get_task_sdk` — callers choose their own auth
    mode.  Useful for plugins that don't use :func:`run_task` (e.g.
    ``safe-synthesizer``).
    """
    return _resolve_provider().get_platform_sdk(
        as_service=as_service,
        internal=internal,
        on_behalf_of=on_behalf_of,
    )


def get_async_platform_sdk(
    *,
    as_service: str | None = None,
    internal: bool = False,
    on_behalf_of: str | None = None,
) -> AsyncNeMoPlatform:
    """Build a general-purpose async SDK handle.

    Used by middleware and controllers that run inside the platform
    service process and need an async client.
    """
    return _resolve_provider().get_async_platform_sdk(
        as_service=as_service,
        internal=internal,
        on_behalf_of=on_behalf_of,
    )


def get_forwarding_headers(sdk: NeMoPlatform | AsyncNeMoPlatform) -> dict[str, str]:
    """Extract the platform headers an SDK would send on outbound requests.

    Returns the ``X-NMP-*`` and trace-propagation headers that *sdk* was
    configured with.  Use this when forwarding identity and observability
    context through a non-SDK HTTP client (e.g. LangChain's ``ChatNVIDIA``).

    The returned dict is a shallow copy — callers may mutate it freely.

    Args:
        sdk: A :class:`~nemo_platform.NeMoPlatform` or
            :class:`~nemo_platform.AsyncNeMoPlatform` instance.  For
            per-request headers, pass a request-scoped SDK built with
            ``sdk.with_options(set_default_headers=...)``.
    """
    # _custom_headers holds the headers passed at construction time
    # (service principal, internal marker, on-behalf-of, OTEL, etc.)
    # — everything the platform SDK factory injects.
    return dict(sdk._custom_headers)
