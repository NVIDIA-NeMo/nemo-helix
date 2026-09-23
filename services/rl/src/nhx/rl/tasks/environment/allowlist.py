# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image-bundled Gym adapter allowlist for adapter-wheels-v1 bootstrap."""

from __future__ import annotations

IMAGE_ADAPTER_ALLOWLIST: dict[str, str] = {
    "verifiers_agent": "responses_api_agents/verifiers_agent",
}

DEFAULT_ADAPTER_AGENT = "verifiers_agent"
