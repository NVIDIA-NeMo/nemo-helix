# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared run-service inputs and published Harbor records for API tests."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from nemo_evals.api.fields import TaskRef
from nemo_evals.api.service.taskset_run_service import TasksetRunService
from nemo_evals.api.task_definitions.harbor import HarborTaskDefinition
from nemo_evals.config import EvaluatorConfig
from nemo_evals.entities import TaskEntity, TaskRevisionEntity, TasksetEntity, TasksetRevisionEntity
from nemo_evals.jobs import agent_files_snapshot, registered_agent_resolution
from nemo_evals.revisions import publish_revision
from nemo_evals.taskset_runs.entities import TasksetRunEntity
from nemo_evals.taskset_runs.store import TasksetRunStore
from nemo_helix_plugin.agents.client import AsyncAgentsClient
from nemo_helix_plugin.client.client import AsyncNemoClient

NOW = datetime(2026, 10, 14, 12, tzinfo=timezone.utc)


@pytest.fixture
def request_body():
    return {
        "spec": {"tasks": "default/suite", "target": {"kind": "harbor"}},
        "execution": {},
        "idempotency_key": "ci-run",
    }


@pytest.fixture
async def async_client():
    def reject(request):
        raise AssertionError(f"Unexpected network request: {request.method} {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(reject)) as http_client:
        async with AsyncNemoClient(base_url="http://platform.test", http_client=http_client) as client:
            yield client


@pytest.fixture
def run_store(entity_store, files_client):
    return TasksetRunStore._with_entity_adapter(entity_store, files_client)


@pytest.fixture
def authorizer():
    return AsyncMock()


@pytest.fixture
def run_config(harbor_config):
    """Operator config with Kubernetes taskset runs enabled."""
    config = EvaluatorConfig()
    return config.model_copy(update={"runs": config.runs.model_copy(update={"harbor": harbor_config})})


@pytest.fixture
def run_preparer(fake_preparer):
    """Every task gets prebuilt images at once, as the MVP preparer does for ready builds."""
    fake_preparer.ready_at_once = True
    return fake_preparer


@pytest.fixture
def service(
    run_store,
    entity_store,
    async_client,
    authorizer,
    run_config,
    run_preparer,
    fake_task_jobs,
    fake_evaluations,
    sandbox_platform,
):
    return TasksetRunService(
        run_store,
        entity_store,
        async_client,
        authorizer,
        run_config,
        clock=lambda: NOW,
        platform=sandbox_platform,
        preparer=run_preparer,
        task_jobs=fake_task_jobs,
        evaluations=fake_evaluations,
    )


@pytest.fixture
def seed_run(entity_store):
    """Return an async factory for persisted runs without submit or Files operations."""

    async def seed(name="run-selected", *, workspace="default", caller="user:alice", **state):
        """Persist name/workspace under caller, applying supplied domain state; return the stored record."""
        entity_store.caller = caller
        data = {
            "name": name,
            "workspace": workspace,
            "taskset_ref": "default/suite#" + "a" * 64,
            "plan_ref": "default/unavailable-plan#plan.json",
            "plan_digest": "b" * 64,
            "request_digest": "c" * 64,
            "accepted_at": NOW,
            "timeout_seconds": 3600,
            "execution_deadline_at": NOW + timedelta(hours=1),
            "finalization_deadline_at": NOW + timedelta(hours=2),
        }
        data.update(state)
        return await entity_store.create(TasksetRunEntity.model_validate(data))

    return seed


@pytest.fixture
async def published(entity_store):
    definition = HarborTaskDefinition.model_validate(
        {
            "kind": "harbor",
            "native_task_id": "task-1",
            "instruction": "Solve the task.",
            "source": {"fileset_ref": "default/harbor-tasks#task-1/task_archive", "files_hash": "a" * 64},
            "harbor_hash": {"digest": "b" * 64, "harbor_version": "0.20.0"},
            # Sized, so submit's warnings don't include the default-resources notice.
            "config": {"environment": {"cpus": 1, "memory_mb": 1024}},
        }
    )
    task = await entity_store.create(TaskEntity(name="task-1", workspace="default", spec=definition))
    task_revision, _, _ = await publish_revision(entity_store, entity_store, task, TaskRevisionEntity)
    head = await entity_store.create(
        TasksetEntity(
            name="suite", workspace="default", tasks=[TaskRef(f"default/task-1#{task_revision.content_hash}")]
        )
    )
    revision, head, _ = await publish_revision(entity_store, entity_store, head, TasksetRevisionEntity, tags={"stable"})
    return head, revision


@pytest.fixture
def registered_agent(monkeypatch, files_client, request_body):
    """Use real agent translation and snapshot copying over fake Agents and Files reads."""
    config = {
        "config_format": "nemo-agents-spec-v1",
        "name": "coder",
        "default_harness": "deepagents",
        "harnesses": {"deepagents": {"kind": "deepagents", "settings": {"deepagents": {}}}},
        "models": {"default": {"provider": "openai", "model": "openai/gpt-5.4"}},
        "skills": {"paths": ["skills/coding"]},
    }
    agents = SimpleNamespace(
        get_agent=AsyncMock(
            return_value=SimpleNamespace(
                data=lambda: SimpleNamespace(config_format="nemo-agents-spec-v1", config=config)
            )
        )
    )
    monkeypatch.setenv("NHX_BASE_URL", "http://platform.test")
    monkeypatch.setattr(
        registered_agent_resolution,
        "client_from_platform",
        lambda client, kind: agents if kind is AsyncAgentsClient else files_client,
    )
    monkeypatch.setattr(files_client, "get_fileset", AsyncMock(return_value=object()))

    async def download(ref, destination, **kwargs):
        assert ref.root == "agents/coder-ethos"
        root = Path(destination) / "files"
        skill = root / "skills/coding/SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text("# Coding instructions\n")
        return root

    monkeypatch.setattr(agent_files_snapshot, "_download_fileset_ref", download)
    monkeypatch.setattr(agent_files_snapshot, "AsyncFilesetFileSystem", lambda **kwargs: object())
    request_body["spec"]["target"]["source"] = {"agent": "agents/coder"}
    return agents
