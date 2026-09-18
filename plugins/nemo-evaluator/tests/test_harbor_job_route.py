# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise HTTP submission through the real transformer and compiler."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_evaluator.api.schemas import HarborTaskDefinition, TaskRef
from nemo_evaluator.api.task_definitions.harbor import HarborArchiveSource, HarborTaskHash
from nemo_evaluator.entities import TaskEntity, TaskRevisionEntity, TasksetEntity, TasksetRevisionEntity
from nemo_evaluator.jobs.agent_evaluate import AgentEvalJob
from nemo_evaluator.revisions import publish_revision
from nemo_platform_plugin.dependencies import get_entity_client, get_sdk_client
from nemo_platform_plugin.jobs.execution_profiles import SubprocessJobExecutionProfile
from nemo_platform_plugin.jobs.routes import add_job_routes


@pytest.mark.parametrize("direct", [False, True])
async def test_post_pins_sources_in_canonical_and_compiled_job(entity_store, monkeypatch, direct):
    task = TaskEntity(
        name="checkout",
        workspace="default",
        spec=HarborTaskDefinition(
            kind="harbor",
            harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
            source=HarborArchiveSource(
                fileset_ref="default/files#v1/task/files",
                files_hash="a" * 64,
            ),
        ),
    )
    await entity_store.create(task)
    revision, _, _ = await publish_revision(entity_store, entity_store, task, TaskRevisionEntity)
    suite = TasksetEntity(
        name="suite", workspace="default", tasks=[TaskRef(f"default/checkout#{revision.content_hash}")]
    )
    await entity_store.create(suite)
    suite_revision, _, _ = await publish_revision(entity_store, entity_store, suite, TasksetRevisionEntity)
    captured = []

    class Jobs:
        async def get_execution_profiles(self):
            response = MagicMock()
            response.data.return_value = [SubprocessJobExecutionProfile(provider="subprocess", profile="harbor-test")]
            return response

        async def create_job(self, *, workspace, body):
            captured.append(body)
            response = MagicMock()
            response.data.return_value = SimpleNamespace(
                id="job-1",
                name="harbor-job",
                workspace=workspace,
                description=None,
                created_at=datetime(2026, 1, 1),
                updated_at=datetime(2026, 1, 1),
                spec=body.spec,
                status="created",
                status_details=None,
                error_details=None,
                ownership=None,
                custom_fields=None,
            )
            return response

    jobs = Jobs()
    monkeypatch.setattr("nemo_platform_plugin.jobs.api_factory.client_from_platform", lambda *args: jobs)
    monkeypatch.setattr("nemo_evaluator.jobs.agent_evaluate.client_from_platform", lambda *args: jobs)
    app = FastAPI()
    app.include_router(add_job_routes(AgentEvalJob), prefix="/apis/evaluator/v2/workspaces/{workspace}")
    app.dependency_overrides[get_entity_client] = lambda: entity_store
    app.dependency_overrides[get_sdk_client] = lambda: MagicMock()
    public_tasks = ["default/checkout"] if direct else "default/suite"
    response = TestClient(app).post(
        "/apis/evaluator/v2/workspaces/default/agent-evaluate/jobs",
        json={
            "profile": "harbor-test",
            "spec": {"tasks": public_tasks, "target": {"kind": "harbor", "agent_name": "oracle"}},
        },
    )
    assert response.status_code == 201, response.text
    body = captured[0]
    expected = (
        {"kind": "harbor-task-list", "task_refs": [f"default/checkout#{revision.content_hash}"]}
        if direct
        else {"kind": "harbor-taskset", "taskset_ref": f"default/suite#{suite_revision.content_hash}"}
    )
    assert body.spec["tasks"] == expected
    assert response.json()["spec"]["tasks"] == expected
    assert body.platform_spec.steps[0].config["tasks"] == expected
    assert body.platform_spec.steps[0].executor.provider == "subprocess"
    assert body.platform_spec.steps[0].executor.profile == "harbor-test"
