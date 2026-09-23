# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed sync and async clients for Inference Gateway VirtualModel CRUD."""

from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.method import method
from nemo_helix_plugin.virtual_models import endpoints


class _VirtualModelsMethods:
    create_virtual_model = method(endpoints.create_virtual_model)
    list_virtual_models = method(endpoints.list_virtual_models)
    get_virtual_model = method(endpoints.get_virtual_model)
    update_virtual_model = method(endpoints.update_virtual_model)
    delete_virtual_model = method(endpoints.delete_virtual_model)


class VirtualModelsClient(_VirtualModelsMethods, NemoClient):
    """Sync client for Inference Gateway VirtualModel CRUD."""


class AsyncVirtualModelsClient(_VirtualModelsMethods, AsyncNemoClient):
    """Async client for Inference Gateway VirtualModel CRUD."""
