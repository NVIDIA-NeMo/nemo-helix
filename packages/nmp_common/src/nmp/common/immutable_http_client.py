# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutability helpers for SDK-owned HTTP clients.

NeMo Platform SDK instances reuse their underlying httpx client when callers
derive scoped SDKs via ``with_options()`` or pass the SDK into typed plugin
clients. Those clients must be created with their required transport-level
configuration and then left alone.

Caller-specific request configuration, including auth headers, belongs on the
SDK instance or on a separate explicit client. Mutating an SDK-owned client
after it has been handed out is a bug because that state can leak into derived
SDKs or requests. These wrappers make those bugs fail immediately.
"""

from types import MappingProxyType
from typing import NoReturn

import httpx
from httpx._types import CookieTypes, HeaderTypes
from nemo_platform import DefaultAsyncHttpxClient, DefaultHttpxClient

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
        "SDK HTTP clients are immutable. Pass per-SDK options as SDK constructor "
        "arguments, or pass a separate httpx client configured for that use case."
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
        # httpx normally persists response cookies on the client. SDK clients
        # should not carry request/session state between derived SDK handles.
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
                "SDK HTTP clients are immutable. Pass a separate httpx client "
                "when client-level configuration needs to differ."
            )
        super().__setattr__(name, value)

    def _freeze_http_client(self) -> None:
        # Assignment blocking is not enough because these attributes are mutable
        # containers. Replace them with immutable versions before sharing the
        # client through SDK clones or plugin adapters.
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
