# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional convenience layer: turn endpoint methods into client methods.

Plugin authors define endpoints once in a collection class, then use
``method()`` to bridge them onto a client class::

    class _ExampleMethods:
        hello = method(ExampleEndpoints.hello)
        create_item = method(ExampleEndpoints.create_item)

    class ExampleClient(_ExampleMethods, NemoClient): pass
    class AsyncExampleClient(_ExampleMethods, AsyncNemoClient): pass

    client = ExampleClient(base_url="...", workspace="default")
    resp = client.hello(name="alice")  # NemoResponse[HelloResponse]

The descriptor dispatches sync vs async based on the client type.
The ``method()`` function is overloaded so that the return type of the
bound callable matches what ``send()`` returns for each response-type
marker (``BinaryContent``, ``Stream[T]``, ``Paginated[T]``, plain model).

Note: ``ty`` shows ``Unknown |`` on the method types due to unannotated
class attributes (astral-sh/ty#3254). The types themselves are correct
and ``pyright`` resolves them cleanly.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar, overload

from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.response import (
    AsyncNemoBinaryResponse,
    AsyncNemoPaginatedResponse,
    AsyncNemoStreamResponse,
    NemoBinaryResponse,
    NemoPaginatedResponse,
    NemoResponse,
    NemoStreamResponse,
)
from nemo_helix_plugin.client.types import (
    BinaryContent,
    ModelT,
    P,
    Paginated,
    PreparedRequest,
    ResponseT,
    StrategyT,
    Stream,
)

# TypeVar for the sync return type of the bound callable.
SyncReturnT = TypeVar("SyncReturnT")
# TypeVar for the async return type of the bound callable.
AsyncReturnT = TypeVar("AsyncReturnT")


class EndpointMethod(Generic[P, SyncReturnT, AsyncReturnT]):
    """Descriptor that binds an endpoint to a client instance.

    When accessed on a :class:`NemoClient`, returns a sync callable
    with return type ``SyncReturnT``.
    When accessed on an :class:`AsyncNemoClient`, returns an async callable
    with return type ``AsyncReturnT``.

    The type parameters are set by the ``method()`` overloads to match
    the response type that ``send()`` returns for each endpoint marker.
    """

    # Copied from the endpoint in __init__ so help() and autodoc describe the
    # endpoint. Declared here so the descriptor's introspection surface is part of
    # its type rather than something callers have to discover at runtime.
    __wrapped__: Callable[P, PreparedRequest]
    __name__: str
    __qualname__: str
    __doc__: str | None
    __module__: str

    def __init__(self, endpoint_fn: Callable[P, PreparedRequest]) -> None:
        self._endpoint_fn = endpoint_fn
        # Per-owning-class callable stubs handed out on *class*-level access, so
        # unittest.mock / inspect can classify each endpoint (sync vs coroutine)
        # and autospec it as callable. Keyed and cached by objtype so repeated
        # class access is stable. See __get__.
        self._class_stubs: dict[type | None, Callable[..., object]] = {}
        # Carry the endpoint's name, docstring, and annotations onto the descriptor
        # so help() and autodoc describe the endpoint rather than the descriptor.
        # Set directly rather than via functools.update_wrapper, which expects a
        # callable wrapper; a descriptor is not one, and which would also copy
        # __dict__ and with it the endpoint's __isabstractmethod__ marker.
        #
        # inspect.signature(SomeClient.method) works because class-level access
        # returns a functools.wraps'd stub (see __get__), not the raw descriptor.
        self.__wrapped__ = endpoint_fn
        for attr in functools.WRAPPER_ASSIGNMENTS:
            try:
                setattr(self, attr, getattr(endpoint_fn, attr))
            except AttributeError:
                pass

    def _class_level_stub(self, objtype: type | None) -> Callable[..., object]:
        """Callable stub returned on class-level access, matched to ``objtype``.

        ``unittest.mock`` (both ``Mock(spec=...)`` and ``create_autospec``) reads
        each attribute off the *class* and classifies it with ``callable()`` and
        ``asyncio.iscoroutinefunction()``. A bare descriptor is neither callable
        nor a coroutine function, so every endpoint on an async client would be
        mocked as a sync ``MagicMock`` and could not be awaited. This hands back a
        real function -- ``async def`` for async clients, ``def`` for sync -- that
        wraps the endpoint (so ``inspect.signature`` works and autospec validates
        call signatures), but refuses to run unbound: endpoints only mean anything
        against a client instance.
        """
        cached = self._class_stubs.get(objtype)
        if cached is not None:
            return cached
        is_async = objtype is not None and issubclass(objtype, AsyncNemoClient)
        if is_async:

            @functools.wraps(self._endpoint_fn)
            async def stub(*args: object, **kwargs: object) -> object:
                raise TypeError(f"{self.__name__} must be called on a client instance, not the class")
        else:

            @functools.wraps(self._endpoint_fn)
            def stub(*args: object, **kwargs: object) -> object:
                raise TypeError(f"{self.__name__} must be called on a client instance, not the class")

        # __isabstractmethod__ rides along in the endpoint's __dict__ via wraps;
        # drop it so the stub is never mistaken for an abstract member.
        stub.__dict__.pop("__isabstractmethod__", None)
        self._class_stubs[objtype] = stub
        return stub

    @property
    def endpoint(self) -> Callable[P, PreparedRequest]:
        """The endpoint function this descriptor binds."""
        return self._endpoint_fn

    @overload
    def __get__(self, obj: None, objtype: type | None = None) -> EndpointMethod[P, SyncReturnT, AsyncReturnT]: ...
    @overload
    def __get__(self, obj: NemoClient, objtype: type | None = None) -> Callable[P, SyncReturnT]: ...
    @overload
    def __get__(self, obj: AsyncNemoClient, objtype: type | None = None) -> Callable[P, Awaitable[AsyncReturnT]]: ...

    def __get__(self, obj: NemoClient | AsyncNemoClient | None, objtype: type | None = None) -> object:
        if obj is None:
            # Class-level access. Anything that inspects a client class rather than
            # an instance -- Mock(spec=...), autospec, inspect, help(), autodoc --
            # lands here. Hand back a callable stub matched to the owning client
            # type so mock classifies sync vs async correctly and autospec sees a
            # callable. The raw descriptor is still reachable via
            # inspect.getattr_static, which never invokes __get__.
            return self._class_level_stub(objtype)
        if isinstance(obj, AsyncNemoClient):

            @functools.wraps(self._endpoint_fn)
            async def async_bound(*args: P.args, **kwargs: P.kwargs) -> AsyncReturnT:
                return await obj.send(self._endpoint_fn(*args, **kwargs))  # type: ignore[return-value]

            return async_bound

        @functools.wraps(self._endpoint_fn)
        def sync_bound(*args: P.args, **kwargs: P.kwargs) -> SyncReturnT:
            return obj.send(self._endpoint_fn(*args, **kwargs))  # type: ignore[return-value]

        return sync_bound


# ---------------------------------------------------------------------------
# method() overloads — one per response-type marker
# ---------------------------------------------------------------------------


@overload
def method(
    endpoint_fn: Callable[P, PreparedRequest[BinaryContent]],
) -> EndpointMethod[P, NemoBinaryResponse, AsyncNemoBinaryResponse]: ...


@overload
def method(
    endpoint_fn: Callable[P, PreparedRequest[Stream[ModelT]]],
) -> EndpointMethod[P, NemoStreamResponse[ModelT], AsyncNemoStreamResponse[ModelT]]: ...


@overload
def method(
    endpoint_fn: Callable[P, PreparedRequest[Paginated[ModelT, StrategyT]]],
) -> EndpointMethod[
    P,
    NemoPaginatedResponse[ModelT, StrategyT],
    AsyncNemoPaginatedResponse[ModelT, StrategyT],
]: ...


@overload
def method(
    endpoint_fn: Callable[P, PreparedRequest[None]],
) -> EndpointMethod[P, NemoResponse[None], NemoResponse[None]]: ...


@overload
def method(
    endpoint_fn: Callable[P, PreparedRequest[ResponseT]],
) -> EndpointMethod[P, NemoResponse[ResponseT], NemoResponse[ResponseT]]: ...


def method(endpoint_fn: Callable[P, PreparedRequest]) -> EndpointMethod:
    """Create an :class:`EndpointMethod` descriptor from an endpoint function.

    The return type of the bound callable is determined by the endpoint's
    response-type marker via overloads, matching the dispatch in ``send()``.

    Usage::

        class _MyMethods:
            create_item = method(MyEndpoints.create_item)   # NemoResponse[Item]
            list_items = method(MyEndpoints.list_items)      # NemoPaginatedResponse[Item]
            download = method(MyEndpoints.download)          # NemoBinaryResponse
    """
    return EndpointMethod(endpoint_fn)
