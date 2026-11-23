# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the Model Entity refs a Compass run is given."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol

from nemo_helix_plugin.agents.plugin_managed_agents import PluginManagedAgentInputError
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.client.response import NemoResponse
from nemo_helix_plugin.models.types import ModelEntity
from nemo_helix_plugin.nooa_model_client import supported_backend_format


class ModelLookupClient(Protocol):
    def get_model(self, *, name: str, workspace: str | None = None) -> Awaitable[NemoResponse[ModelEntity]]: ...


async def resolve_model_ref(client: ModelLookupClient, ref: str, *, field: str, workspace: str) -> str:
    """Return ``<workspace>/<name>`` for an existing Model Entity Compass can call.

    A bare name resolves against the provided workspace.

    Raises:
        PluginManagedAgentInputError: *ref* is malformed, names no Model Entity,
            or names one Compass cannot call. The caller can fix each of these.
        NemoHTTPError: Any other failure to read the model, such as a denial,
            with the status the Models service returned.
        NemoTransportError: The Models service could not be reached.
    """
    match ref.split("/"):
        case [name]:
            model_workspace = workspace
        case [model_workspace, name] if model_workspace and name:
            pass
        case _:
            raise PluginManagedAgentInputError(
                f"{field} must be a Model Entity name or '<workspace>/<name>' ref, got {ref!r}."
            )

    try:
        model_entity = (await client.get_model(name=name, workspace=model_workspace)).data()
    except NotFoundError as exc:
        raise PluginManagedAgentInputError(f"{field} '{model_workspace}/{name}' is not a known Model Entity.") from exc

    # Ensure the model is one NOOA clients can communicate with
    try:
        supported_backend_format(model_entity)
    except ValueError as exc:
        raise PluginManagedAgentInputError(f"{field}: {exc}") from exc

    return f"{model_workspace}/{name}"
