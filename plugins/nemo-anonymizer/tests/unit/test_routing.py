# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from anonymizer.config.anonymizer_config import AnonymizerConfig
from anonymizer.config.replace_strategies import Redact
from nemo_anonymizer_plugin.app.input import AnonymizerInputSpec
from nemo_anonymizer_plugin.app.task_config import AnonymizerRequest
from nemo_anonymizer_plugin.sdk.resources import AnonymizerResource
from nemo_anonymizer_plugin.service import AnonymizerService


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

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, str]:
            return {"name": "anonymizer-run-1"}

    class Client:
        def post(self, url: str, **kwargs: object) -> Response:
            calls.append(("POST", url))
            return Response()

        def get(self, url: str, **kwargs: object) -> Response:
            calls.append(("GET", url))
            return Response()

    platform = SimpleNamespace(
        base_url="https://platform.test",
        workspace="default",
        default_headers={},
        _client=Client(),
    )
    resource = AnonymizerResource(platform)  # type: ignore[arg-type]
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
