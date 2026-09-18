# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import tarfile
from contextlib import asynccontextmanager, contextmanager
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from nemo_evaluator.api.schemas import TaskInput
from nemo_evaluator.api.service.task_service import TaskService
from nemo_evaluator.api.task_definitions.harbor import HarborArchiveSource, HarborTaskDefinition, HarborTaskHash
from nemo_evaluator.harbor.publication import publish_harbor_task_archive
from nemo_evaluator.sdk.harbor import upload_harbor_dataset, upload_harbor_task
from nemo_platform_plugin.client.errors import NemoHTTPError

pytest.importorskip("harbor")


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "task"
    for name, text in {
        "task.toml": "",
        "instruction.md": "Do it",
        "environment/Dockerfile": "FROM ubuntu",
        "tests/test.sh": "exit 0",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


@pytest.fixture
def files():
    objects = {}
    client = Mock()
    headers = {}

    def with_headers(value):
        headers.update(value)
        return client

    def upload(*, workspace, name, path, content):
        assert "If-None-Match" not in headers
        data = b"".join(content)
        assert len(data) == int(headers["Content-Length"])
        objects[path] = data
        return Mock()

    def download(*, workspace, name, path):
        @contextmanager
        def stream(**kwargs):
            yield iter([objects[path]])

        return Mock(stream=stream)

    client.with_headers.side_effect = with_headers
    client.upload_file.side_effect = upload
    client.download_file.side_effect = download
    return client, objects


def test_publish_reuses_verified_archive(root, files):
    client, objects = files
    first = publish_harbor_task_archive(root, files_client=client, fileset_ref="default/harbor-tasks")
    second = publish_harbor_task_archive(root, files_client=client, fileset_ref="default/harbor-tasks")
    assert first == second
    assert len(objects) == 1
    assert next(iter(objects)).endswith("/task_archive")
    assert first.source.files_hash == hashlib.sha256(next(iter(objects.values()))).hexdigest()
    assert client.download_file.call_count == 2


def test_publication_replaces_existing_bytes_and_verifies(root, files):
    client, objects = files
    publish_harbor_task_archive(root, files_client=client, fileset_ref="default/harbor-tasks")
    key = next(iter(objects))
    objects[key] = b"tampered"
    result = publish_harbor_task_archive(root, files_client=client, fileset_ref="default/harbor-tasks")
    assert hashlib.sha256(objects[key]).hexdigest() == result.source.files_hash


def test_publication_rejects_corrupted_readback(root, files):
    client, objects = files
    upload = client.upload_file.side_effect

    def corrupt(**kwargs):
        result = upload(**kwargs)
        objects[kwargs["path"]] = b"corrupted"
        return result

    client.upload_file.side_effect = corrupt
    with pytest.raises(ValueError, match="checksum"):
        publish_harbor_task_archive(root, files_client=client, fileset_ref="default/harbor-tasks")


@pytest.mark.parametrize("dataset", [False, True])
def test_upload_only_has_no_entity_calls(root, files, monkeypatch, dataset):
    client, objects = files
    monkeypatch.setattr("nemo_evaluator.sdk.harbor.FilesClient.from_client", lambda _: client)
    monkeypatch.setattr("nemo_evaluator.sdk.harbor.EvaluatorClient.from_client", lambda _: pytest.fail("No DB calls"))
    if dataset:
        receipt = upload_harbor_dataset(root, client=Mock(), register=False)
        assert len(receipt.members) == 1
        assert receipt.taskset_ref is None
        member = receipt.members[0]
        fileset_name, prefix = "harbor-tasksets", "task/task/"
    else:
        member = upload_harbor_task(root, client=Mock(), register=False)
        fileset_name, prefix = "harbor-tasks", "task/"
    assert member.task_ref is None and member.task_name is None
    assert member.native_name == root.name
    assert next(iter(objects)).startswith(prefix)
    assert member.definition.source.fileset_ref.startswith(f"default/{fileset_name}#{prefix}")


@pytest.mark.parametrize("bad", ["step", "metadata", "checksum"])
async def test_registration_rejects_before_persistence(root, entity_store, bad):
    if bad == "step":
        (root / "task.toml").write_text("[[steps]]\nname = '../../outside'\n")
    elif bad == "metadata":
        (root / "instruction.md").write_bytes(b"x" * (1024**2 + 1))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", format=tarfile.USTAR_FORMAT) as archive:
        archive.add(root, arcname="task")
    data = buffer.getvalue()

    @asynccontextmanager
    async def stream(**kwargs):
        async def chunks():
            yield data

        yield chunks()

    client = Mock(download_file=AsyncMock(return_value=Mock(stream=stream)))
    service = TaskService(entity_store, AsyncMock(), client)
    definition = HarborTaskDefinition(
        kind="harbor",
        native_task_id="task",
        source=HarborArchiveSource(
            fileset_ref="default/files#task_archive",
            files_hash="0" * 64 if bad == "checksum" else hashlib.sha256(data).hexdigest(),
        ),
        harbor_hash=HarborTaskHash(digest="a" * 64, harbor_version="future"),
    )
    before = len(entity_store.entities)
    with pytest.raises(ValueError):
        await service.create_task("bad", TaskInput(spec=definition), workspace="default")
    assert len(entity_store.entities) == before


def test_registration_uses_published_ordinal_not_latest():
    from types import SimpleNamespace

    from nemo_evaluator.api.schemas import Task
    from nemo_evaluator.sdk.harbor import _digest, _register
    from nemo_evaluator.sdk.task_resources import EvaluatorTasksResource

    spec = HarborTaskDefinition(
        kind="harbor",
        native_task_id="task",
        source=HarborArchiveSource(fileset_ref="default/files#task_archive", files_hash="a" * 64),
        harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
    )
    value = TaskInput(spec=spec)
    intended = _digest(value)
    resource = Mock(spec=EvaluatorTasksResource)
    resource.retrieve.side_effect = NemoHTTPError(httpx.Response(404))
    resource.create.return_value = Task.model_construct(name="task", spec=spec, revision=2)
    resource.list_revisions.return_value = SimpleNamespace(
        data=[SimpleNamespace(revision=3, content_hash="f" * 64), SimpleNamespace(revision=2, content_hash=intended)],
        pagination=SimpleNamespace(total_pages=1),
    )
    assert _register(resource, "task", value, "default", False) == f"default/task#{intended}"
    resource.retrieve.assert_called_once()


def test_registration_reconciles_lost_response_by_intended_digest():
    from types import SimpleNamespace

    from nemo_evaluator.sdk.harbor import _digest, _register
    from nemo_evaluator.sdk.task_resources import EvaluatorTasksResource
    from nemo_platform_plugin.client.errors import NemoTransportError

    spec = HarborTaskDefinition(
        kind="harbor",
        native_task_id="task",
        source=HarborArchiveSource(fileset_ref="default/files#task_archive", files_hash="a" * 64),
        harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
    )
    value = TaskInput(spec=spec)
    intended = _digest(value)
    resource = Mock(spec=EvaluatorTasksResource)
    resource.retrieve.side_effect = NemoHTTPError(httpx.Response(404))
    resource.create.side_effect = NemoTransportError(httpx.ReadError("response lost"))
    resource.list_revisions.side_effect = [
        SimpleNamespace(
            data=[SimpleNamespace(revision=3, content_hash="f" * 64)], pagination=SimpleNamespace(total_pages=2)
        ),
        SimpleNamespace(
            data=[SimpleNamespace(revision=2, content_hash=intended)], pagination=SimpleNamespace(total_pages=2)
        ),
    ]
    assert _register(resource, "task", value, "default", False) == f"default/task#{intended}"
    assert resource.create.call_count == 1


def test_member_failure_does_not_publish_partial_suite(root, files, monkeypatch):
    from nemo_evaluator.sdk.harbor import HarborUploadError

    client, objects = files
    monkeypatch.setattr("nemo_evaluator.sdk.harbor.FilesClient.from_client", lambda _: client)
    register = Mock(side_effect=ValueError("registration failed"))
    monkeypatch.setattr("nemo_evaluator.sdk.harbor._register", register)
    monkeypatch.setattr("nemo_evaluator.sdk.harbor.EvaluatorClient.from_client", lambda _: Mock())
    with pytest.raises(HarborUploadError) as caught:
        upload_harbor_dataset(root, client=Mock(), taskset_name="suite")
    assert len(caught.value.upload_details.members) == 1
    assert caught.value.upload_details.taskset_ref is None
    assert len(objects) == 1
    assert register.call_count == 1


@pytest.mark.parametrize("outcome", ["created", "existing", "response_lost"])
def test_taskset_registration_uses_server_membership_order(outcome):
    from types import SimpleNamespace

    from nemo_evaluator.api.schemas import TaskRef, Taskset, TasksetInput
    from nemo_evaluator.sdk.harbor import _digest, _register
    from nemo_evaluator.sdk.taskset_resources import EvaluatorTasksetsResource
    from nemo_platform_plugin.client.errors import NemoTransportError

    # Renaming local folders a -> z and b -> a reverses their DB reference order.
    refs = [TaskRef("default/z#" + "a" * 64), TaskRef("default/a#" + "b" * 64)]
    request = TasksetInput(tasks=refs)
    stored = Taskset.model_construct(name="suite", tasks=sorted(refs, key=lambda ref: ref.root), revision=1)
    digest = _digest(stored)
    resource = Mock(spec=EvaluatorTasksetsResource)
    if outcome == "existing":
        resource.retrieve.return_value = stored
    else:
        resource.retrieve.side_effect = NemoHTTPError(httpx.Response(404))
        if outcome == "response_lost":
            resource.create.side_effect = NemoTransportError(httpx.ReadError("response lost"))
        else:
            resource.create.return_value = stored
    resource.list_revisions.return_value = SimpleNamespace(
        data=[SimpleNamespace(revision=1, content_hash=digest)], pagination=SimpleNamespace(total_pages=1)
    )
    assert _register(resource, "suite", request, "default", False) == f"default/suite#{digest}"
    if outcome == "existing":
        resource.create.assert_not_called()


@pytest.mark.parametrize("name", ["task_archive", "contents"])
def test_publication_root_does_not_collide_with_staging(root, files, name):
    renamed = root.with_name(name)
    root.rename(renamed)
    client, objects = files
    definition = publish_harbor_task_archive(renamed, files_client=client, fileset_ref="default/harbor-tasks")
    assert definition.instruction == "Do it"
    assert next(iter(objects)).startswith(f"{name}/")
    with tarfile.open(fileobj=io.BytesIO(next(iter(objects.values()))), mode="r:gz") as archive:
        assert archive.getnames()[0] == name


@pytest.mark.parametrize("name", ["task_archive", "contents"])
async def test_registration_root_does_not_collide_with_download(root, entity_store, name):
    from nemo_evaluator.harbor.archive import pack_task

    renamed = root.with_name(name)
    root.rename(renamed)
    archive_path = root.parent / "download.tar.gz"
    digest = pack_task(renamed, archive_path)
    data = archive_path.read_bytes()

    @asynccontextmanager
    async def stream(**kwargs):
        async def chunks():
            yield data

        yield chunks()

    client = Mock(download_file=AsyncMock(return_value=Mock(stream=stream)))
    service = TaskService(entity_store, AsyncMock(), client)
    definition = HarborTaskDefinition(
        kind="harbor",
        native_task_id=name,
        source=HarborArchiveSource(fileset_ref="default/files#task_archive", files_hash=digest),
        harbor_hash=HarborTaskHash(digest="a" * 64, harbor_version="test"),
    )
    task, published = await service.create_task("stored", TaskInput(spec=definition), workspace="default")
    assert published
    assert task.spec.instruction == "Do it"
