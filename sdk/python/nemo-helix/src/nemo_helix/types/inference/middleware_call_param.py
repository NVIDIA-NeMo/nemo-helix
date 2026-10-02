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

from typing import Dict
from typing_extensions import Required, TypedDict

__all__ = ["MiddlewareCallParam"]


class MiddlewareCallParam(TypedDict, total=False):
    """One entry in a VirtualModel middleware pipeline.

    Declares which plugin to invoke and how to resolve its configuration.
    Exactly one of ``config`` (inline dict) or ``config_id`` (entity reference)
    should be provided. ``config_type`` is always required regardless of which
    is used — it is the discriminator that tells IGW (and the plugin) which
    config schema applies.

    Attributes:
        name: The entry-point key of the plugin to invoke
            (e.g. ``"nemo-switchyard"``). Must match the plugin's
            ``nemo.inference_middleware`` entry-point key.
        config_type: Always required. Maps to the ``entity_type`` of the plugin's
            config ``NemoEntity`` subclass (e.g. ``"routellm_config"``). Used by
            IGW to call :meth:`~NemoInferenceMiddleware.validate_middleware_config`
            with the right discriminator, and by the plugin to dispatch to the
            correct schema when it supports multiple config types.
        config: Inline config dict. Mutually exclusive with ``config_id``.
        config_id: ``"workspace/name"`` reference to a stored config entity.
            Mutually exclusive with ``config``. IGW resolves this by calling
            :meth:`~NemoInferenceMiddleware.get_middleware_config` on the plugin.
    """

    config_type: Required[str]

    name: Required[str]

    config: Dict[str, object]

    config_id: str
