# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``GET /jobs/{name}`` on a ``job_route_factory`` collection when the name belongs elsewhere.

The Jobs service resolves a name across every source in the workspace, so a collection's GET
route can be handed another collection's record (e.g. a taskset run's ``evals.harbor-task`` Job
read through the agent-evaluate route). That must be a 404, not a 500 from rendering the foreign
spec, while records of this collection keep rendering, including ones whose ``source`` predates a
rename.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from nemo_helix_plugin.dependencies import get_entity_client, get_nemo_client
from nemo_helix_plugin.jobs.api_factory import job_route_factory
from pydantic import BaseModel
from starlette.testclient import TestClient


class _WidgetSpec(BaseModel):
    color: str


def _fake_compiler(workspace, original_spec, transformed_spec, entity_client, job_name, sdk):
    return {"steps": []}


def _job_record(*, name: str, source: str, spec: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"job-{name}",
        name=name,
        description=None,
        workspace="default",
        source=source,
        created_at=None,
        updated_at=None,
        spec=spec,
        status=None,
        status_details=None,
        error_details=None,
        ownership=None,
        custom_fields=None,
    )


_RECORDS = {
    "widget-1": _job_record(name="widget-1", source="widgets.paint", spec={"color": "red"}),
    # Another collection's Job in the same workspace, e.g. a taskset run's task Job.
    "run-1-task-0": _job_record(name="run-1-task-0", source="evals.harbor-task", spec={"task_ref": "t"}),
    # This collection's Job, tagged with its source from before a rename.
    "widget-old": _job_record(name="widget-old", source="old-widgets.paint", spec={"color": "blue"}),
}


@pytest.fixture
def client() -> Iterator[TestClient]:
    async def _get_job(*, name: str, workspace: str) -> Any:
        return SimpleNamespace(data=lambda: _RECORDS[name])

    jobs_client = SimpleNamespace(get_job=_get_job)
    app = FastAPI()
    app.include_router(
        job_route_factory(
            service_name="widgets.paint",
            job_type="Widget",
            job_input=_WidgetSpec,
            platform_job_config_compiler=_fake_compiler,
        ),
        prefix="/apis/widgets/v2/workspaces/{workspace}",
    )
    app.dependency_overrides[get_nemo_client] = lambda: SimpleNamespace()
    app.dependency_overrides[get_entity_client] = lambda: SimpleNamespace()
    with patch("nemo_helix_plugin.jobs.api_factory.AsyncJobsClient.from_client", return_value=jobs_client):
        yield TestClient(app)


@pytest.mark.parametrize(
    ("name", "color"),
    [("widget-1", "red"), ("widget-old", "blue")],
    ids=["own-source", "pre-rename-source"],
)
def test_get_renders_this_collections_job(client: TestClient, name: str, color: str) -> None:
    resp = client.get(f"/apis/widgets/v2/workspaces/default/jobs/{name}")

    assert resp.status_code == 200, resp.text
    assert resp.json()["spec"] == {"color": color}


def test_get_another_collections_job_is_404(client: TestClient) -> None:
    resp = client.get("/apis/widgets/v2/workspaces/default/jobs/run-1-task-0")

    assert resp.status_code == 404, resp.text
    assert resp.json()["detail"] == "Job 'run-1-task-0' not found"
