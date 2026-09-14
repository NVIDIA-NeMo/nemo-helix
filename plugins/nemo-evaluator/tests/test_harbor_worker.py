# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticated transport through revision resolution, tree materialization and the public evaluator."""

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

pytest.importorskip("harbor")
from nemo_evaluator.api.schemas import HarborTaskDefinition, TaskRef
from nemo_evaluator.api.task_definitions.harbor import HarborTreeSource
from nemo_evaluator.entities import TaskEntity, TaskRevisionEntity
from nemo_evaluator.harbor.manifest import inspect_tree
from nemo_evaluator.harbor.tasks import PinnedHarborTaskList
from nemo_evaluator.jobs.agent_evaluate import AgentEvalJob
from nemo_evaluator.revisions import publish_revision
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import HarborRewardMetric
from nemo_evaluator_sdk.agent_eval.taskset_sources import digest_harbor_tree
from nemo_evaluator_sdk.execution.metric_execution import run_sync
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import LocalJobResults


@pytest.fixture
def stored_packages(tmp_path, entity_store):
    refs = []
    objects = {}
    for entity_name, folder, task_id in [
        ("first", "z-folder", "commerce/checkout"),
        ("second", "a-folder", "commerce/search"),
    ]:
        root = tmp_path / entity_name
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
        manifest = inspect_tree(root, task_dir=folder).to_bytes()
        objects[f"{entity_name}/manifest.json"] = manifest
        for path in root.rglob("*"):
            if path.is_file():
                objects[f"{entity_name}/files/{path.relative_to(root).as_posix()}"] = path.read_bytes()
        task = TaskEntity(
            name=entity_name,
            workspace="default",
            spec=HarborTaskDefinition(
                kind="harbor",
                tree=HarborTreeSource(
                    root_ref=f"default/files#{entity_name}/files",
                    manifest_ref=f"default/files#{entity_name}/manifest.json",
                    manifest_digest=hashlib.sha256(manifest).hexdigest(),
                    tree_digest=digest_harbor_tree(root),
                ),
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
        requests.append(request)
        assert request.headers["x-nmp-principal-id"] == "service:harbor-test"
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

    return PinnedHarborTaskList(task_refs=refs), handler, requests


@pytest.mark.parametrize("transport", ["sync", "async", "both"])
def test_worker_passes_verified_ordered_tasks_to_public_evaluator(tmp_path, stored_packages, monkeypatch, transport):
    source, handler, requests = stored_packages
    sync_http = httpx.Client(transport=httpx.MockTransport(handler))
    async_http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    headers = {"X-NMP-Principal-Id": "service:harbor-test", "Authorization": "Bearer test-token"}
    sdk = NemoClient(
        http_client=sync_http, base_url="http://platform.test", workspace="default", default_headers=headers
    )
    async_sdk = AsyncNemoClient(
        http_client=async_http, base_url="http://platform.test", workspace="default", default_headers=headers
    )
    # Isolate onto a real async transport that records requests, as the worker normally does.
    monkeypatch.setattr("nemo_evaluator.jobs.utils.httpx.AsyncClient", lambda: async_http)
    if transport == "both":
        sync_http.close()  # Selecting sync when both are supplied now fails.
    ctx = JobContext(
        workspace="default",
        job_id="bridge",
        storage=StoragePaths(ephemeral=tmp_path / "ephemeral", persistent=tmp_path / "persistent"),
        results=LocalJobResults(root=tmp_path / "results"),
    )
    evaluator = MagicMock()
    monkeypatch.setattr(AgentEvalJob, "_build_evaluator", lambda *args: evaluator)
    # Stop after the public run boundary; persistence is covered by the existing worker tests.
    evaluator.run_sync.side_effect = RuntimeError("captured public evaluator")
    with pytest.raises(RuntimeError, match="captured public evaluator"):
        AgentEvalJob().run(
            {"tasks": source.model_dump(mode="json"), "target": {"kind": "harbor"}},
            ctx=ctx,
            sdk=sdk if transport != "async" else None,
            async_sdk=async_sdk if transport != "sync" else None,
        )
    tasks = evaluator.run_sync.call_args.kwargs["tasks"]
    assert [task.id for task in tasks] == ["commerce/checkout", "commerce/search"]
    assert [Path(task.metadata["harbor_task_dir"]).name for task in tasks] == ["z-folder", "a-folder"]
    assert all(Path(task.metadata["harbor_dataset_path"]).is_absolute() for task in tasks)
    assert all(isinstance(task.metrics[0], HarborRewardMetric) for task in tasks)
    assert any("/apis/entities/" in request.url.path for request in requests)
    assert sum("/apis/files/" in request.url.path for request in requests) == 10
    sync_http.close()
    run_sync(async_http.aclose)


async def test_adapter_returns_receipt_from_same_verified_packages(
    tmp_path, stored_packages, entity_store, monkeypatch
):
    from nemo_evaluator.entities import TasksetEntity, TasksetRevisionEntity
    from nemo_evaluator.harbor.taskset_source import factory
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import HarborTasksetLoader
    from nemo_evaluator_sdk.agent_eval.taskset_sources import digest_harbor_tree
    from nemo_platform import AsyncNeMoPlatform

    source, handler, requests = stored_packages
    suite = TasksetEntity(name="suite", workspace="default", tasks=source.task_refs)
    await entity_store.create(suite)
    revision, _, _ = await publish_revision(entity_store, entity_store, suite, TasksetRevisionEntity)
    monkeypatch.setattr(HarborTasksetLoader, "load", lambda *args: pytest.fail("adapter must not load SDK tasks"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        sdk = AsyncNeMoPlatform(
            base_url="http://platform.test",
            workspace="default",
            http_client=http_client,
            default_headers={"X-NMP-Principal-Id": "service:harbor-test", "Authorization": "Bearer test-token"},
        )
        adapter = factory(client=sdk, workspace="default")
        uri = f"nemo-evaluator-taskset://default/suite#{revision.content_hash}"
        receipt = await adapter.materialize(uri, destination_root=tmp_path / "adapter")
    assert receipt.source_uri == uri
    assert receipt.revision_digest == revision.content_hash
    assert receipt.member_digests == {
        "commerce/checkout": source.task_refs[0].root.split("#")[1],
        "commerce/search": source.task_refs[1].root.split("#")[1],
    }
    assert receipt.materialization_digest == digest_harbor_tree(receipt.materialized_root)
    assert receipt.materialized_root.is_relative_to(tmp_path / "adapter")
    assert sum("/apis/files/" in request.url.path for request in requests) == 10
