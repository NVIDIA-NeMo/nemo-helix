# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``method()`` descriptor that binds endpoints onto client classes.

Instance-level dispatch is exercised throughout the client and Models suites. What
is pinned here is *class*-level access, which nothing else touches and which every
introspection tool performs: ``Mock(spec=SomeClient)``, ``help()``, ``pydoc``,
autodoc, and anything walking ``dir()``.
"""

from __future__ import annotations

import asyncio
import inspect
import pydoc
from unittest.mock import AsyncMock, MagicMock, create_autospec

import pytest
from nemo_helix_plugin.client.method import EndpointMethod
from nemo_helix_plugin.models.client import AsyncModelsClient, ModelsClient

BASE = "http://test:8000"


def _descriptor(name: str) -> EndpointMethod:
    return inspect.getattr_static(AsyncModelsClient, name)


# ---------------------------------------------------------------------------
# Class-level access
# ---------------------------------------------------------------------------


def test_class_level_access_returns_a_typed_callable_stub() -> None:
    """``__get__`` with no instance hands back a callable stub matched to the class.

    The stub exists so ``unittest.mock`` / ``inspect`` can classify each endpoint
    (sync vs coroutine) and autospec it as callable. The raw descriptor is still
    reachable via ``inspect.getattr_static``, which never invokes ``__get__``.
    """
    assert callable(ModelsClient.create_model)
    assert callable(AsyncModelsClient.create_model)
    assert not inspect.iscoroutinefunction(ModelsClient.create_model)
    assert inspect.iscoroutinefunction(AsyncModelsClient.create_model)
    # The descriptor itself is reachable without triggering __get__.
    assert isinstance(inspect.getattr_static(ModelsClient, "create_model"), EndpointMethod)


def test_class_level_stub_refuses_to_run_unbound() -> None:
    """The stub is for introspection only; calling it without an instance errors.

    ty types class-level access as the ``EndpointMethod`` descriptor (the
    ``__get__(obj=None)`` overload cannot distinguish the sync vs async owning
    class, so it cannot describe the callable stub returned at runtime). These
    calls exercise that runtime stub deliberately, hence the suppressions.
    """
    with pytest.raises(TypeError):
        ModelsClient.create_model(workspace="w", body=None)  # ty: ignore[call-non-callable]
    with pytest.raises(TypeError):
        asyncio.run(AsyncModelsClient.create_model(workspace="w", body=None))  # ty: ignore[call-non-callable]


def test_mock_spec_against_a_client_class_builds() -> None:
    """``Mock(spec=...)`` reads every attribute off the class to classify it.

    This is the failure that surfaced the bug: the models controller test suite
    could not spec a mock against its own client.
    """
    mock = MagicMock(spec=AsyncModelsClient)

    assert callable(mock.create_model)
    # The async client's endpoints spec as awaitables, not sync MagicMocks --
    # this is the whole point of the typed async client.
    assert isinstance(mock.create_model, AsyncMock)
    with pytest.raises(AttributeError):
        mock.create_modle  # noqa: B018  a typo must not be silently mockable


def test_pydoc_lists_endpoints_with_their_docstrings() -> None:
    """Endpoint docs reach help() through the copied ``__doc__``.

    pydoc swallows per-member errors, so this does not discriminate the
    class-access fix; what it pins is the attribute copying.
    """
    # plain() strips pydoc's backspace-overstrike bolding.
    rendered = pydoc.plain(pydoc.render_doc(ModelsClient))

    assert "delete_deployment" in rendered
    assert "Delete a deployment" in rendered


def test_create_autospec_yields_awaitable_endpoint_stubs() -> None:
    """autospec walks the class and gets callable, correctly-async stubs.

    Sync clients autospec to callable MagicMocks; async clients to AsyncMocks, so
    ``await auto.create_model(...)`` works and signature validation is enforced.
    """
    auto_sync = create_autospec(ModelsClient)
    assert callable(auto_sync.create_model)
    assert not isinstance(auto_sync.create_model, AsyncMock)

    auto_async = create_autospec(AsyncModelsClient)
    assert callable(auto_async.create_model)
    assert isinstance(auto_async.create_model, AsyncMock)


# ---------------------------------------------------------------------------
# Identity carried from the endpoint
# ---------------------------------------------------------------------------


def test_descriptor_carries_endpoint_identity() -> None:
    descriptor = _descriptor("delete_deployment")

    assert descriptor.__name__ == "delete_deployment"
    assert descriptor.__doc__ == descriptor.endpoint.__doc__
    assert descriptor.__doc__  # the endpoint really does carry one
    assert descriptor.endpoint.__name__ == "delete_deployment"


def test_sync_descriptor_unwraps_to_its_endpoint() -> None:
    """On a sync client the endpoint *is* the unwrap target, so nothing indirects."""
    descriptor = inspect.getattr_static(ModelsClient, "delete_deployment")

    assert descriptor.__wrapped__ is descriptor.endpoint


def test_async_descriptor_unwraps_to_a_coroutine_function() -> None:
    """``inspect.unwrap`` has to land on something awaitable for the async client.

    Python 3.14's ``unittest.mock`` classifies a spec'd attribute with ``getattr_static`` +
    ``inspect.unwrap`` instead of ``getattr``, so it never sees the async stub that ``__get__``
    hands out. Unwrapping has to terminate at a coroutine function or every endpoint on an async
    client specs as a sync ``MagicMock`` and cannot be awaited. The endpoint itself stays reachable
    via ``.endpoint``.
    """
    descriptor = _descriptor("delete_deployment")

    assert asyncio.iscoroutinefunction(inspect.unwrap(descriptor))
    assert descriptor.endpoint.__name__ == "delete_deployment"


def test_signature_is_reachable_at_class_level() -> None:
    """The class-level stub is a real function, so ``signature()`` works directly.

    Typed as the descriptor at class level (see the stub note above), so the
    ``signature()`` argument is suppressed; at runtime it is a real function.
    """
    signature = inspect.signature(AsyncModelsClient.create_model)  # ty: ignore[invalid-argument-type]

    assert set(signature.parameters) == {"workspace", "body", "exist_ok"}
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in signature.parameters.values())


def test_descriptor_does_not_leak_the_endpoint_abstractmethod_marker() -> None:
    """Endpoints are ``@abstractmethod`` stubs; that marker must not ride along.

    ``functools.update_wrapper`` would copy ``__dict__`` and with it
    ``__isabstractmethod__``, which would make any ABCMeta-based client class
    uninstantiable. The attributes are copied one by one to avoid exactly that.
    """
    descriptor = _descriptor("create_model")

    assert getattr(descriptor.endpoint, "__isabstractmethod__", False) is True
    assert not getattr(descriptor, "__isabstractmethod__", False)
    assert not getattr(AsyncModelsClient, "__abstractmethods__", frozenset())
    ModelsClient(base_url=BASE, workspace="default")  # constructs


# ---------------------------------------------------------------------------
# Instance-level dispatch (unchanged, but nothing states it outright)
# ---------------------------------------------------------------------------


def test_sync_client_binds_a_plain_callable() -> None:
    bound = ModelsClient(base_url=BASE, workspace="default").create_model

    assert callable(bound)
    assert not inspect.iscoroutinefunction(bound)


def test_async_client_binds_a_coroutine_function() -> None:
    bound = AsyncModelsClient(base_url=BASE, workspace="default").create_model

    assert inspect.iscoroutinefunction(bound)


def test_bound_method_exposes_the_real_signature() -> None:
    """Instance access hands back the wrapped function, so signature() works there."""
    signature = inspect.signature(ModelsClient(base_url=BASE, workspace="default").get_model)

    assert set(signature.parameters) == {"workspace", "name", "query_params"}
