# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated ``client.auditor`` SDK namespace; removed one release after the plugin rename."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

from garak_plugin.sdk import AsyncGarakPluginResource, GarakPluginResource
from nemo_helix_plugin.sdk import NemoPluginSDKResources

_MESSAGE = "`client.auditor` is deprecated and will be removed; use `client.garak_plugin` instead."


def _deprecated(factory: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def build(platform: Any) -> Any:
        warnings.warn(_MESSAGE, DeprecationWarning, stacklevel=3)
        return factory(platform)

    return build


auditor_sdk_resources = NemoPluginSDKResources(
    sync_resource=_deprecated(GarakPluginResource),
    async_resource=_deprecated(AsyncGarakPluginResource),
)
