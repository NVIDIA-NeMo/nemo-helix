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

from __future__ import annotations

from typing import Iterable
from typing_extensions import Required, TypedDict

from .middleware_call_param import MiddlewareCallParam
from .virtual_model_inference_config_param import VirtualModelInferenceConfigParam

__all__ = ["VirtualModelCreateParams"]


class VirtualModelCreateParams(TypedDict, total=False):
    workspace: str

    name: Required[str]
    """Name of the virtual model within the workspace. Must be unique per workspace."""

    autoprovisioned: bool
    """Marks this VirtualModel as controller-managed.

    The Models controller will delete it once no ModelProvider serves the matching
    entity. Setting this manually opts the VirtualModel into that cleanup behavior.
    """

    default_model_entity: str
    """Model entity to route to, in "workspace/name" format.

    Written into request["model"] before the request middleware pipeline runs. If
    omitted, a request middleware plugin must handle backend routing itself. Set to
    null to clear an existing value.
    """

    models: Iterable[VirtualModelInferenceConfigParam]
    """Model entity references used by this VirtualModel.

    A per-entry backend_format overrides the referenced ModelEntity backend_format
    when IGW resolves the backend format for a request.
    """

    override_proxy: str
    """
    Plugin-provided proxy implementation for IGW to use instead of its default
    aiohttp proxy. Format: "plugin-name.proxy-name". Leave unset to use the default
    IGW proxy. Set to null to clear an existing value.
    """

    post_response_middleware: Iterable[MiddlewareCallParam]
    """
    Ordered list of middleware plugins invoked after the response has been returned
    to the caller. Intended for fire-and-forget work (logging, analytics) that must
    not block or modify the response.
    """

    request_middleware: Iterable[MiddlewareCallParam]
    """Ordered list of middleware plugins applied before proxying to the backend.

    Each entry is a MiddlewareCall with a "name" (plugin identifier) and optional
    "config_type" and "config_id" fields that reference a stored plugin
    configuration.
    """

    response_middleware: Iterable[MiddlewareCallParam]
    """
    Ordered list of middleware plugins applied after the backend response is
    received, before returning it to the caller.
    """
