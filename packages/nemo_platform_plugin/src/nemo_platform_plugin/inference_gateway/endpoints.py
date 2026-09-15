# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for narrow Inference Gateway provider calls."""

from __future__ import annotations

from abc import abstractmethod

from nemo_platform_plugin.client.endpoint import get
from nemo_platform_plugin.client.types import BinaryContent

_PROVIDER = "/apis/inference-gateway/v2/workspaces/{workspace}/provider/{name}/-"


@get(_PROVIDER + "/v1/models")
@abstractmethod
def get_provider_models_raw(*, workspace: str | None = None, name: str) -> BinaryContent: ...
