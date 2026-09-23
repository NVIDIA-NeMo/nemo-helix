# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from datetime import datetime

from ..._models import BaseModel
from .middleware_call import MiddlewareCall
from .virtual_model_inference_config import VirtualModelInferenceConfig

__all__ = ["VirtualModel"]


class VirtualModel(BaseModel):
    """Logical inference route.

    Maps a user-facing model name to an optional default model entity and
    defines ordered middleware pipelines for the request, response, and
    post-response phases.

    When a caller sets ``model: "workspace/my-virtual-model"`` in an inference
    request, IGW resolves the ``VirtualModel`` instead of a ``ModelEntity``
    directly. If ``default_model_entity`` is set, IGW writes it into
    ``request["model"]`` before the request middleware pipeline runs. Middleware
    may mutate ``request["model"]`` freely. After the pipeline completes, IGW
    reads ``request["model"]``, resolves it to a ``ModelProvider`` via the
    ``ModelCache``, and proxies.

    The ``ModelProviderReconciler`` auto-creates a passthrough ``VirtualModel``
    for each discovered model (same workspace and name as the ``ModelEntity``,
    empty middleware lists, ``default_model_entity`` pointing to that entity).
    All existing inference requests continue to work without changes.
    """

    id: str

    created_at: datetime

    created_by: Optional[str] = None

    db_version: int
    """Database version of the entity for optimistic locking."""

    entity_id: str
    """Alias for id for backwards compatibility."""

    parent: str
    """Parent entity ID for nested entities."""

    updated_at: datetime

    updated_by: Optional[str] = None

    workspace: str
    """Workspace identifier"""

    autoprovisioned: Optional[bool] = None
    """Marks this VirtualModel as controller-managed.

    The Models controller will delete it once no ModelProvider serves the matching
    entity. Setting this manually opts the VirtualModel into that cleanup behavior.
    """

    default_model_entity: Optional[str] = None

    guardrail_config_ids: Optional[List[str]] = None
    """System-managed.

    Guardrail configs applied by this VirtualModel's middleware, as "workspace/name"
    references. Derived from the middleware pipelines on every write and ignored if
    supplied on a create or update body. Filter on it with filter[guardrail_config].
    """

    models: Optional[List[VirtualModelInferenceConfig]] = None

    name: Optional[str] = None
    """Entity name within the workspace"""

    override_proxy: Optional[str] = None

    post_response_middleware: Optional[List[MiddlewareCall]] = None

    project: Optional[str] = None
    """The name of the project associated with this entity."""

    request_middleware: Optional[List[MiddlewareCall]] = None

    response_middleware: Optional[List[MiddlewareCall]] = None
