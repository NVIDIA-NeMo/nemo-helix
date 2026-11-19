# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_observability_plugin.adapters.base import ProviderAdapter
from nemo_observability_plugin.adapters.mlflow import MLflowAdapter
from nemo_observability_plugin.adapters.phoenix import PhoenixAdapter
from nemo_observability_plugin.config import ProviderName

ADAPTERS: dict[ProviderName, ProviderAdapter] = {
    "mlflow": MLflowAdapter(),
    "phoenix": PhoenixAdapter(),
}


def adapter_for(provider: ProviderName) -> ProviderAdapter:
    return ADAPTERS[provider]
