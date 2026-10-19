# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default and immutable httpx clients for typed NeMo Helix clients.

Typed clients reuse their underlying httpx client when callers derive scoped
clients via ``with_options()`` / ``from_client()``. Those clients must be
created with their required transport-level configuration and then left alone.

Caller-specific request configuration, including auth headers, belongs on the
typed client or on a separate explicit httpx client. Mutating a shared httpx
client after it has been handed out is a bug because that state can leak into
derived clients or requests. These wrappers make those bugs fail immediately.
"""

from types import MappingProxyType
from typing import Any, NoReturn

import httpx
from httpx._types import CookieTypes, HeaderTypes

DEFAULT_TIMEOUT = httpx.Timeout(timeout=60, connect=5.0)
DEFAULT_CONNECTION_LIMITS = httpx.Limits(max_connections=100, max_keepalive_connections=20)


def _with_platform_defaults(kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    kwargs.setdefault("limits", DEFAULT_CONNECTION_LIMITS)
    kwargs.setdefault("follow_redirects", True)
    return kwargs


class DefaultHttpxClient(httpx.Client):
    """Sync httpx client with the platform default timeout, limits and redirects."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**_with_platform_defaults(kwargs))


class DefaultAsyncHttpxClient(httpx.AsyncClient):
    """Async httpx client with the platform default timeout, limits and redirects."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**_with_platform_defaults(kwargs))


_IMMUTABLE_CLIENT_ATTRS = {
    "_base_url",
    "_cookies",
    "_event_hooks",
    "_headers",
    "_params",
    "_timeout",
    "_transport",
    "_mounts",
    "_trust_env",
    "base_url",
    "cookies",
    "event_hooks",
    "follow_redirects",
    "headers",
    "max_redirects",
    "params",
    "timeout",
    "trust_env",
}


def _raise_immutable_client_mutation_error() -> NoReturn:
    raise TypeError(
        "This HTTP client is immutable. Pass per-client options as typed client "
        "constructor arguments, or pass a separate httpx client configured for that use case."
    )


class _ImmutableHeaders(httpx.Headers):
    def __setitem__(self, key: str, value: str) -> None:
        _raise_immutable_client_mutation_error()

    def __delitem__(self, key: str) -> None:
        _raise_immutable_client_mutation_error()

    def clear(self) -> None:
        _raise_immutable_client_mutation_error()

    def pop(self, key: str, default: object = None) -> str:
        _raise_immutable_client_mutation_error()

    def popitem(self) -> tuple[str, str]:
        _raise_immutable_client_mutation_error()

    def setdefault(self, key: str, default: str = "") -> str:
        _raise_immutable_client_mutation_error()

    def update(self, headers: HeaderTypes | None = None) -> None:
        _raise_immutable_client_mutation_error()


class _ImmutableCookies(httpx.Cookies):
    def extract_cookies(self, response: httpx.Response) -> None:
        # httpx normally persists response cookies on the client. Shared clients
        # should not carry request/session state between derived client handles.
        pass

    def set(self, name: str, value: str, domain: str = "", path: str = "/") -> None:
        _raise_immutable_client_mutation_error()

    def delete(
        self,
        name: str,
        domain: str | None = None,
        path: str | None = None,
    ) -> None:
        _raise_immutable_client_mutation_error()

    def clear(self, domain: str | None = None, path: str | None = None) -> None:
        _raise_immutable_client_mutation_error()

    def update(self, cookies: CookieTypes | None = None) -> None:
        _raise_immutable_client_mutation_error()

    def __setitem__(self, name: str, value: str) -> None:
        _raise_immutable_client_mutation_error()

    def __delitem__(self, name: str) -> None:
        _raise_immutable_client_mutation_error()


class ImmutableHttpClientMixin:
    """Mixin for httpx client subclasses that are immutable after construction."""

    _immutable_http_client_frozen = False

    def __setattr__(self, name: str, value: object) -> None:
        if self._immutable_http_client_frozen and name in _IMMUTABLE_CLIENT_ATTRS:
            raise AttributeError(
                "This HTTP client is immutable. Pass a separate httpx client "
                "when client-level configuration needs to differ."
            )
        super().__setattr__(name, value)

    def _freeze_http_client(self) -> None:
        # Assignment blocking is not enough because these attributes are mutable
        # containers. Replace them with immutable versions before sharing the
        # client through typed client clones or plugin adapters.
        self._headers = _ImmutableHeaders(self._headers)
        self._cookies = _ImmutableCookies(self._cookies)
        self._event_hooks = MappingProxyType(
            {hook_name: tuple(hooks) for hook_name, hooks in self._event_hooks.items()}
        )
        self._mounts = MappingProxyType(self._mounts)
        self._immutable_http_client_frozen = True


class ImmutableHttpxClient(ImmutableHttpClientMixin, httpx.Client):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._freeze_http_client()


class ImmutableAsyncHttpxClient(ImmutableHttpClientMixin, httpx.AsyncClient):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._freeze_http_client()


class ImmutableDefaultHttpxClient(ImmutableHttpClientMixin, DefaultHttpxClient):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._freeze_http_client()


class ImmutableDefaultAsyncHttpxClient(ImmutableHttpClientMixin, DefaultAsyncHttpxClient):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._freeze_http_client()
