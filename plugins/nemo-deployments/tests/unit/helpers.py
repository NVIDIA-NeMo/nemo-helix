# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

from nemo_deployments_plugin.entities import Container, Deployment, DeploymentConfig, Volume
from nemo_helix_plugin.entity_client import NemoPaginationInfo

NOW = datetime.now(timezone.utc)


def make_deployment_config(name: str = "cfg1", workspace: str = "default") -> DeploymentConfig:
    cfg = DeploymentConfig(
        name=name,
        workspace=workspace,
        containers=[Container(name="main", image="nginx")],
    )
    cfg._id = f"id-{name}"
    cfg._created_at = NOW
    return cfg


def make_deployment(name: str = "dep1", workspace: str = "default") -> Deployment:
    dep = Deployment(name=name, workspace=workspace, deployment_config="cfg1", status="PENDING")
    dep._id = f"id-{name}"
    dep._created_at = NOW
    return dep


def make_volume(name: str = "vol1", workspace: str = "default") -> Volume:
    vol = Volume(name=name, workspace=workspace)
    vol._id = f"id-{name}"
    vol._created_at = NOW
    return vol


def list_response(items: list[Any]) -> MagicMock:
    resp = MagicMock()
    resp.data = items
    resp.pagination = NemoPaginationInfo(
        page=1,
        page_size=20,
        current_page_size=len(items),
        total_pages=1,
        total_results=len(items),
    )
    return resp
