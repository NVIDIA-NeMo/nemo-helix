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
from nemo_evaluator_sdk.agent_eval.tasks import SemanticView
from nemo_helix_plugin.dependencies import get_entity_client, get_sdk_client
from nemo_helix_plugin.jobs.execution_profiles import SubprocessJobExecutionProfile
from nemo_helix_plugin.jobs.routes import add_job_routes


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("invalid", [None, "missing-member", "missing-revision", "duplicate-metric", "invalid-view"])
async def test_post_snapshots_tasks_before_creating_job(entity_store, monkeypatch, direct, invalid):
    """Exercise the HTTP submission route, ensuring invalid selections fail and valid jobs embed snapshots
    without downloads.
    """
    task = TaskEntity(
        name="checkout",
        workspace="default",
        spec=HarborTaskDefinition(
            kind="harbor",
            native_task_id="task",
            harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
            source=HarborArchiveSource(
                fileset_ref="default/files#v1/task/files",
                files_hash="a" * 64,
            ),
        ),
    )
    if invalid in {"duplicate-metric", "invalid-view"}:
        from nemo_evaluator.jobs.metric_resolution import to_inline
        from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
        from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
        from nemo_evaluator_sdk.metrics.runner_rewards import HarborRewardMetric

        if invalid == "duplicate-metric":
            task.spec.metrics = [to_inline(bundle_metric(HarborRewardMetric(), CloudpickleMetricBundlePackager()))]
        else:
            task.spec.views = {
                "bad": SemanticView.model_validate(
                    {"reducer": "mean", "signals": [{"metric": "harbor_reward", "output": "missing"}]}
                )
            }
    await entity_store.create(task)
    revision, _, _ = await publish_revision(entity_store, entity_store, task, TaskRevisionEntity)
    suite = TasksetEntity(
        name="suite", workspace="default", tasks=[TaskRef(f"default/checkout#{revision.content_hash}")]
    )
    await entity_store.create(suite)
    await publish_revision(entity_store, entity_store, suite, TasksetRevisionEntity)
    if invalid == "missing-member":
        await entity_store.delete(TaskEntity, task.name, workspace=task.workspace)
    elif invalid == "missing-revision":
        await entity_store.delete(
            TaskRevisionEntity, revision.name, workspace=revision.workspace, parent=revision.parent
        )
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

    def forbidden(*args, **kwargs):
        pytest.fail("Submission attempted Harbor archive download")

    from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient

    monkeypatch.setattr(AsyncFilesClient, "download_file", forbidden)
    monkeypatch.setattr(FilesClient, "download_file", forbidden)

    jobs = Jobs()
    monkeypatch.setattr("nemo_helix_plugin.jobs.api_factory.client_from_platform", lambda *args: jobs)
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
    if invalid:
        assert response.status_code == 422, response.text
        assert captured == []
        return
    assert response.status_code == 201, response.text
    body = captured[0]
    assert isinstance(task.spec, HarborTaskDefinition)
    expected = [
        {
            "id": task.spec.native_task_id,
            "spec": {
                **task.spec.model_dump(mode="json"),
                "provenance": {"entity_name": "default/checkout", "revision_digest": revision.content_hash},
            },
            "metadata": [],
        }
    ]
    assert body.spec["tasks"] == expected
    assert response.json()["spec"]["tasks"] == expected
    assert body.platform_spec.steps[0].config["tasks"] == expected
    assert body.platform_spec.steps[0].executor.provider == "subprocess"
    assert body.platform_spec.steps[0].executor.profile == "harbor-test"
