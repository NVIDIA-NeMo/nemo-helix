# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticated transport through revision resolution, tree materialization and the public evaluator."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

pytest.importorskip("harbor")
from nemo_evaluator.api.schemas import HarborTaskDefinition, TaskRef, TasksetRef
from nemo_evaluator.api.task_definitions.harbor import HarborArchiveSource, HarborTaskHash
from nemo_evaluator.entities import TaskEntity, TaskRevisionEntity, TasksetEntity, TasksetRevisionEntity
from nemo_evaluator.harbor.archive import pack_task
from nemo_evaluator.jobs.agent_evaluate import AgentEvalJob, AsyncAgentEvalJob
from nemo_evaluator.jobs.agent_spec import AgentEvalInputSpec, AgentEvalSpec, HarborRunnerTarget
from nemo_evaluator.jobs.metric_resolution import to_inline
from nemo_evaluator.revisions import publish_revision
from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import HarborAgentTaskRunner, HarborRewardMetric
from nemo_evaluator_sdk.agent_eval.tasks import SemanticReducer, SemanticView, ViewSignal
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput
from nemo_evaluator_sdk.execution.metric_execution import run_sync
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import LocalJobResults


@pytest.fixture(params=["direct", "taskset"])
def stored_packages(tmp_path, entity_store, request):
    """Publish two ordered task archives and return submission snapshots with an authenticated fake HTTP transport."""
    refs = []
    objects = {}
    for entity_name, folder, task_id in [
        ("first", "z-folder", "commerce/checkout"),
        ("second", "a-folder", "commerce/search"),
    ]:
        root = tmp_path / folder
        root.mkdir()
        for path, content in {
            "task.toml": f'[task]\nname = "{task_id}"\n',
            "instruction.md": "Fix it",
            "environment/Dockerfile": "FROM ubuntu",
            "tests/test.sh": "exit 0",
        }.items():
            dest = root / path
            dest.parent.mkdir(exist_ok=True, parents=True)
            dest.write_text(content)
        archive_path = tmp_path / f"{entity_name}.tar.gz"
        digest = pack_task(root, archive_path)
        objects[f"{entity_name}/task_archive"] = archive_path.read_bytes()
        task = TaskEntity(
            name=entity_name,
            workspace="default",
            spec=HarborTaskDefinition(
                kind="harbor",
                native_task_id=task_id,
                instruction="Fix it",
                source=HarborArchiveSource(fileset_ref=f"default/files#{entity_name}/task_archive", files_hash=digest),
                harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
            ),
        )
        run_sync(lambda: entity_store.create(task))
        revision, _, _ = run_sync(lambda: publish_revision(entity_store, entity_store, task, TaskRevisionEntity))
        refs.append(TaskRef(f"default/{entity_name}#{revision.content_hash}"))
    requests = []

    def payload(entity):
        return {
            "entity_type": entity.__entity_type__,
            "id": entity.id,
            "name": entity.name,
            "workspace": entity.workspace,
            "parent": entity.parent,
            "data": entity._get_data_fields(),
            "created_at": entity.created_at.isoformat(),
            "updated_at": entity.updated_at.isoformat(),
            "db_version": entity.db_version,
        }

    def handler(request):
        """Serve fake Files and Entity API requests while checking the worker's workspace and authentication
        headers.
        """
        requests.append(request)
        assert request.headers["x-nhx-principal-id"] == "service:harbor-test"
        assert request.headers["authorization"] == "Bearer test-token"
        assert "/workspaces/default/" in request.url.path
        if "/apis/files/" in request.url.path:
            return httpx.Response(200, stream=httpx.ByteStream(objects[request.url.path.split("/-/", 1)[1]]))
        rest = request.url.path.rsplit("/entities/", 1)[1].split("/")
        entities = [e for e in entity_store.entities.values() if e.__entity_type__ == rest[0]]
        if len(rest) == 2:
            return httpx.Response(200, json=payload(next(e for e in entities if e.name == rest[1])))
        filters = json.loads(request.url.params["filter"])["$and"]
        values = {field: operation["$eq"] for item in filters for field, operation in item.items()}
        entities = [
            e for e in entities if e.parent == values["parent"] and e.content_hash == values["data.content_hash"]
        ]
        return httpx.Response(
            200,
            json={
                "data": [payload(e) for e in entities],
                "pagination": {
                    "page": 1,
                    "page_size": 1,
                    "current_page_size": len(entities),
                    "total_pages": 1,
                    "total_results": len(entities),
                },
            },
        )

    selectors = refs
    if request.param == "taskset":
        suite = TasksetEntity(name="submission-suite", workspace="default", tasks=refs)
        run_sync(lambda: entity_store.create(suite))
        run_sync(lambda: publish_revision(entity_store, entity_store, suite, TasksetRevisionEntity))
        selectors = TasksetRef("default/submission-suite")
    spec = run_sync(
        lambda: AgentEvalJob.to_spec(
            AgentEvalInputSpec(tasks=selectors, target=HarborRunnerTarget()),
            workspace="default",
            entity_client=entity_store,
            async_sdk=None,
            is_local=False,
        )
    )
    assert isinstance(spec, AgentEvalSpec)
    restored = AgentEvalSpec.model_validate_json(spec.model_dump_json())
    assert all(task.spec.kind == "harbor" for task in restored.tasks)
    return restored.tasks, handler, requests


@pytest.mark.parametrize("transport", ["sync", "async"])
def test_worker_passes_verified_ordered_tasks_to_public_evaluator(
    tmp_path, stored_packages, monkeypatch, entity_store, transport
):
    """Delete stored entities after submission and verify the worker uses snapshots and authenticated archive
    downloads.
    """
    source, handler, requests = stored_packages
    # Execution must not depend on task/taskset heads or revisions after submission.
    entity_store.entities.clear()
    sync_http = httpx.Client(transport=httpx.MockTransport(handler))
    async_http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    headers = {"X-NHX-Principal-Id": "service:harbor-test", "Authorization": "Bearer test-token"}
    sdk = NemoClient(
        http_client=sync_http, base_url="http://platform.test", workspace="default", default_headers=headers
    )
    async_sdk = AsyncNemoClient(
        http_client=async_http, base_url="http://platform.test", workspace="default", default_headers=headers
    )
    # Isolate onto a real async transport that records requests, as the worker normally does.
    monkeypatch.setattr("nemo_evaluator.jobs.utils.httpx", SimpleNamespace(AsyncClient=lambda **_: async_http))
    ctx = JobContext(
        workspace="default",
        job_id="bridge",
        storage=StoragePaths(ephemeral=tmp_path / "ephemeral", persistent=tmp_path / "persistent"),
        results=LocalJobResults(root=tmp_path / "results"),
    )
    evaluator = MagicMock()
    job_type = AgentEvalJob if transport == "sync" else AsyncAgentEvalJob
    monkeypatch.setattr(job_type, "_build_evaluator", lambda *args: evaluator)
    # Stop after the public run boundary; persistence is covered by the existing worker tests.
    evaluator.run_sync.side_effect = RuntimeError("captured public evaluator")
    with pytest.raises(RuntimeError, match="captured public evaluator"):
        config = {"tasks": [task.model_dump(mode="json") for task in source], "target": {"kind": "harbor"}}
        if transport == "sync":
            AgentEvalJob().run(config, ctx=ctx, client=sdk)
        else:
            AsyncAgentEvalJob().run(config, ctx=ctx, async_client=async_sdk)
    tasks = evaluator.run_sync.call_args.kwargs["tasks"]
    assert [task.id for task in tasks] == ["commerce/checkout", "commerce/search"]
    assert [Path(task.metadata["harbor_task_dir"]).name for task in tasks] == ["z-folder", "a-folder"]
    assert all(Path(task.metadata["harbor_dataset_path"]).is_absolute() for task in tasks)
    assert all(isinstance(task.metrics[0], HarborRewardMetric) for task in tasks)
    assert not any("/apis/entities/" in request.url.path for request in requests)
    assert sum("/apis/files/" in request.url.path for request in requests) == 2
    sync_http.close()
    run_sync(async_http.aclose)


@pytest.mark.parametrize("offline", [False, True])
def test_worker_executes_additional_metric_and_view(tmp_path, stored_packages, monkeypatch, offline):
    """Reorder task snapshots and verify custom scoring stays attached to its task in live and offline runs."""
    source, handler, _ = stored_packages
    async_http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("nemo_evaluator.jobs.utils.httpx", SimpleNamespace(AsyncClient=lambda **_: async_http))
    metric = to_inline(
        bundle_metric(
            ExactMatchMetric(reference="Fix it", candidate="{{inputs.instruction}}"), CloudpickleMetricBundlePackager()
        )
    )
    views = {
        "quality": SemanticView(
            reducer=SemanticReducer.MEAN,
            signals=[
                ViewSignal(metric="harbor_reward", output="grade"),
                ViewSignal(metric="exact-match", output="exact-match"),
            ],
        )
    }
    source[0].spec.metrics = [metric]
    source[0].spec.views = views
    # Reorder whole tasks: each definition retains its own scoring.
    source.reverse()

    async def trials(self, tasks, config=None):
        return [
            AgentEvalTrial(
                id=f"trial-{task.id}",
                task_id=task.id,
                status=AgentEvalTrialStatus.COMPLETED,
                output=AgentOutput(output_text="yes"),
                metadata={
                    "harbor_primary_reward_key": "grade",
                    "reward": 0.5,
                    "reward_details": {"grade": 0.5, "secondary": 0.25},
                },
            )
            for task in tasks
        ]

    monkeypatch.setattr(HarborAgentTaskRunner, "run_tasks", trials)
    monkeypatch.setattr("nemo_evaluator.jobs.agent_evaluate.persist_agent_eval_result", lambda *args, **kwargs: None)
    ctx = JobContext(
        workspace="default",
        job_id="scoring",
        storage=StoragePaths(ephemeral=tmp_path / "ephemeral", persistent=tmp_path / "persistent"),
        results=LocalJobResults(root=tmp_path / "results"),
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        client = NemoClient(
            http_client=transport,
            base_url="http://platform.test",
            workspace="default",
            default_headers={"X-NHX-Principal-Id": "service:harbor-test", "Authorization": "Bearer test-token"},
        )
        result = AgentEvalJob().run(
            {
                "tasks": [task.model_dump(mode="json") for task in source],
                **(
                    {
                        "trials": [
                            trial.model_dump(mode="json")
                            for trial in run_sync(
                                lambda: trials(
                                    None,
                                    [SimpleNamespace(id="commerce/checkout"), SimpleNamespace(id="commerce/search")],
                                )
                            )
                        ]
                    }
                    if offline
                    else {"target": {"kind": "harbor", "reward_key": "grade"}}
                ),
            },
            ctx=ctx,
            client=client,
        )
    assert result["status"] == "completed"
    summary_path = next((tmp_path / "persistent").rglob("summary.json"))
    summary = json.loads(summary_path.read_text())
    scores = {score["name"]: score for score in summary["scores"]["scores"]}
    assert scores["harbor_reward.grade"]["mean"] == 0.5
    assert scores["harbor_reward.secondary"]["mean"] == 0.25
    assert scores["exact-match.exact-match"]["mean"] == 1.0
    assert scores["exact-match.exact-match"]["count"] == 1
    assert scores["view.quality"]["mean"] == 0.75
