# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Auditor plugin."""

from __future__ import annotations

from typing import ClassVar

from nemo_helix_plugin.config import NemoConfig


class AuditorPluginConfig(NemoConfig):
    plugin_name: ClassVar[str] = "auditor"
    plugin_description: ClassVar[str] = "Auditor plugin configuration"

    # "default" is registered out of the box on every runtime (Docker and Kubernetes),
    # so audit jobs work without extra config in CI/k8s. Deployments that redirect
    # cpu/default to a non-container backend (e.g. local dev's subprocess translation
    # workaround, see packages/nhx_platform/config/local.yaml) should register a
    # dedicated container-backed profile and point this at it instead.
    job_executor_profile: str = "default"


def get_config() -> AuditorPluginConfig:
    return AuditorPluginConfig.get()
