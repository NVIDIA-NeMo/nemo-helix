# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``config.py`` plugin config module."""

from __future__ import annotations

from garak_plugin.config import GarakPluginConfig


def test_default_job_executor_profile() -> None:
    """ "default" is registered out of the box on every runtime (Docker and Kubernetes),
    so scan jobs work without extra config in CI/k8s."""
    cfg = GarakPluginConfig()
    assert cfg.job_executor_profile == "default"


def test_job_executor_profile_override() -> None:
    cfg = GarakPluginConfig(job_executor_profile="garak-plugin")
    assert cfg.job_executor_profile == "garak-plugin"
