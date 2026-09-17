# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test doubles for the agent-hardener plugin unit tests.

The records/manifest/sdk/events modules talk to the entity store through the typed
:class:`EntitiesClient` via :func:`client_from_platform`. Unit tests fake the entity store
as a ``SimpleNamespace(entities=...)`` shape, so ``client_from_platform`` must be patched
at each consuming module's boundary to route the typed-client calls back onto that fake.
This keeps the per-test ``entities`` fakes (and the capturing run-service doubles) unchanged.

The typed client method calls are translated to the fake's shape here, once:
create_entity/get_entity_by_name/update_entity_by_name/list_entities -> entities.create/
get_entity_by_name/update_entity_by_name/list, with the request ``body``/``query_params``
flattened back to the ``data``/``sort``/``page_size`` kwargs the fakes expect.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from nemo_agent_hardener_plugin import sdk as sdk_module
from nemo_agent_hardener_plugin.api.v2 import events as events_module
from nemo_agent_hardener_plugin.jobs import manifest as manifest_module
from nemo_agent_hardener_plugin.jobs import records as records_module


def _data(body: Any) -> Any:
    """Flatten the request body (EntityCreateInput/EntityUpdate) to its data dict."""
    return body.data


def _ok(value: Any) -> Any:
    """Typed-client responses expose ``.data()``; wrap the fake so call sites can call it."""
    return SimpleNamespace(data=lambda: value)


def _page(value: Any) -> Any:
    """Typed paginated responses expose ``.page().items``; surface the fake's iteration through it."""
    return SimpleNamespace(page=lambda: SimpleNamespace(items=iter(value)))


def _fake_entities_client(platform: Any, _client_cls: Any) -> Any:
    """Build a typed-client-shaped stub that delegates to the fake ``platform.entities`` namespace."""
    entities = platform.entities
    return SimpleNamespace(
        create_entity=lambda *, entity_type, workspace, body: _ok(
            entities.create(entity_type, workspace=workspace, data=_data(body))
        ),
        get_entity_by_name=lambda *, name, entity_type, workspace: _ok(
            entities.get_entity_by_name(name=name, entity_type=entity_type, workspace=workspace)
        ),
        update_entity_by_name=lambda *, name, entity_type, workspace, body: _ok(
            entities.update_entity_by_name(name=name, entity_type=entity_type, workspace=workspace, data=_data(body))
        ),
        list_entities=lambda *, entity_type, workspace, query_params=None: _page(
            entities.list(
                entity_type,
                sort=query_params.get("sort") if query_params else None,
                workspace=workspace,
                page_size=query_params.get("page_size") if query_params else None,
            )
        ),
    )


@pytest.fixture(autouse=True)
def _fake_entities_client_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route typed-client entity calls onto the fake ``entities`` namespace in every consuming module."""
    for mod in (records_module, manifest_module, events_module):
        monkeypatch.setattr(mod, "client_from_platform", _fake_entities_client)
    monkeypatch.setattr(sdk_module, "client_from_platform", _fake_entities_client)
