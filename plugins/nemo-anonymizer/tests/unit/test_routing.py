# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import httpx
from anonymizer.config.anonymizer_config import AnonymizerConfig
from anonymizer.config.replace_strategies import Redact
from nemo_anonymizer_plugin.app.input import AnonymizerInputSpec
from nemo_anonymizer_plugin.app.task_config import AnonymizerRequest
from nemo_anonymizer_plugin.sdk.resources import AnonymizerResource
from nemo_anonymizer_plugin.service import AnonymizerService
from nemo_helix_plugin.client.client import NemoClient


def test_service_mounts_run_job_at_generated_collection_path() -> None:
    paths = {
        route.path
        for spec in AnonymizerService().get_routers()
        for route in spec.router.routes
        if hasattr(route, "path")
    }

    assert "/jobs/run" in paths
    assert "/run/jobs/run" not in paths


def test_sdk_run_uses_run_job_collection_path() -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        return httpx.Response(200, json={"name": "anonymizer-run-1"})

    platform = NemoClient(
        base_url="https://platform.test",
        workspace="default",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    resource = AnonymizerResource(platform)
    request = AnonymizerRequest(
        config=AnonymizerConfig(replace=Redact()),
        data=AnonymizerInputSpec(source="inputs#records.csv"),
    )

    resource.run(request)
    resource.get_job_resource("anonymizer-run-1")

    assert calls == [
        ("POST", "https://platform.test/apis/anonymizer/v2/workspaces/default/jobs/run"),
        ("GET", "https://platform.test/apis/anonymizer/v2/workspaces/default/jobs/run/anonymizer-run-1"),
    ]


def test_service_mounts_entity_labels_route() -> None:
    paths = {
        route.path
        for spec in AnonymizerService().get_routers()
        for route in spec.router.routes
        if hasattr(route, "path")
    }

    assert "/entity-labels" in paths


async def test_entity_labels_route_returns_default_labels() -> None:
    from anonymizer import DEFAULT_ENTITY_LABELS
    from nemo_anonymizer_plugin.app.entity_labels import list_entity_labels

    result = await list_entity_labels(workspace="default")

    assert result.data == list(DEFAULT_ENTITY_LABELS)
    assert result.data
