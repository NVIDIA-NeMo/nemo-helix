# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for the Entities service.

Wraps the endpoint functions from :mod:`nemo_helix_plugin.entities.endpoints`
as direct methods via the ``method()`` descriptor, following the Files/Secrets
template.

Usage::

    client = EntitiesClient(base_url="...", workspace="default")
    entity = client.get_entity_by_name(entity_type="model", name="llama").data()
    for e in client.list_entities(entity_type="model").items():
        ...
"""

from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.method import method
from nemo_helix_plugin.entities import endpoints


class _EntitiesMethods:
    create_entity = method(endpoints.create_entity)
    list_entities = method(endpoints.list_entities)
    get_entity_by_name = method(endpoints.get_entity_by_name)
    update_entity_by_name = method(endpoints.update_entity_by_name)
    delete_entity_by_name = method(endpoints.delete_entity_by_name)
    get_entity_by_id = method(endpoints.get_entity_by_id)


class EntitiesClient(_EntitiesMethods, NemoClient):
    """Sync client for the Entities service API."""


class AsyncEntitiesClient(_EntitiesMethods, AsyncNemoClient):
    """Async client for the Entities service API."""
