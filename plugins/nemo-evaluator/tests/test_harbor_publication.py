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
from nemo_evaluator.sdk.harbor import upload_harbor_dataset
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


def test_upload_only_has_no_entity_calls(root, files, monkeypatch):
    client, objects = files
    monkeypatch.setattr("nemo_evaluator.sdk.harbor.FilesClient.from_client", lambda _: client)
    monkeypatch.setattr("nemo_evaluator.sdk.harbor.EvaluatorClient.from_client", lambda _: pytest.fail("No DB calls"))
    receipt = upload_harbor_dataset(root, client=Mock(), register=False)
    assert len(receipt.members) == 1
    assert receipt.taskset_ref is None and receipt.members[0].task_ref is None
    assert next(iter(objects)).startswith("task/task/")


def test_upload_only_preserves_mapped_registration_name(root, files, monkeypatch):
    client, _ = files
    monkeypatch.setattr("nemo_evaluator.sdk.harbor.FilesClient.from_client", lambda _: client)
    receipt = upload_harbor_dataset(
        root,
        client=Mock(),
        register=False,
        member_names={"task": "renamed-task"},
    )
    assert receipt.members[0].native_name == "task"
    assert receipt.members[0].task_name == "renamed-task"
    assert receipt.members[0].task_ref is None


def test_register_uploaded_dataset_creates_tasks_and_taskset(monkeypatch):
    from types import SimpleNamespace

    from nemo_evaluator.api.schemas import Task, TaskRef, Taskset, TasksetInput
    from nemo_evaluator.sdk.harbor import (
        HarborDatasetReceipt,
        HarborUploadReceipt,
        _digest,
        register_harbor_dataset,
    )
    from nemo_evaluator.sdk.task_resources import EvaluatorTasksResource
    from nemo_evaluator.sdk.taskset_resources import EvaluatorTasksetsResource

    definition = HarborTaskDefinition(
        kind="harbor",
        source=HarborArchiveSource(fileset_ref="default/files#task_archive", files_hash="a" * 64),
        harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
    )
    task = Task.model_construct(name="renamed-task", workspace="default", spec=definition, metadata=[], revision=1)
    task_digest = _digest(TaskInput(spec=definition))
    task_ref = TaskRef(f"default/renamed-task#{task_digest}")
    taskset_input = TasksetInput(tasks=[task_ref])
    taskset = Taskset.model_construct(
        name="suite",
        workspace="default",
        tasks=[task_ref],
        files_ref=None,
        description=None,
        metadata=[],
        revision=1,
    )
    taskset_digest = _digest(taskset_input)

    def page(digest):
        return SimpleNamespace(
            data=[SimpleNamespace(revision=1, content_hash=digest)],
            pagination=SimpleNamespace(total_pages=1),
        )

    task_retrieve = Mock(side_effect=[NemoHTTPError(httpx.Response(404)), task, task])
    task_create = Mock(return_value=task)
    taskset_retrieve = Mock(side_effect=[NemoHTTPError(httpx.Response(404)), taskset])
    taskset_create = Mock(return_value=taskset)
    monkeypatch.setattr(EvaluatorTasksResource, "__init__", lambda self, client: None)
    monkeypatch.setattr(EvaluatorTasksResource, "retrieve", task_retrieve)
    monkeypatch.setattr(EvaluatorTasksResource, "create", task_create)
    monkeypatch.setattr(EvaluatorTasksResource, "list_revisions", Mock(return_value=page(task_digest)))
    monkeypatch.setattr(EvaluatorTasksetsResource, "__init__", lambda self, client: None)
    monkeypatch.setattr(EvaluatorTasksetsResource, "retrieve", taskset_retrieve)
    monkeypatch.setattr(EvaluatorTasksetsResource, "create", taskset_create)
    monkeypatch.setattr(EvaluatorTasksetsResource, "list_revisions", Mock(return_value=page(taskset_digest)))
    monkeypatch.setattr("nemo_evaluator.sdk.harbor.EvaluatorClient.from_client", lambda _: Mock())

    receipt = HarborDatasetReceipt(
        members=[HarborUploadReceipt(native_name="task", task_name="renamed-task", definition=definition)]
    )
    result = register_harbor_dataset(receipt, client=Mock(), workspace="default", taskset_name="suite")

    assert result is receipt
    assert result.taskset_ref is not None and result.taskset_ref.root == f"default/suite#{taskset_digest}"
    assert [member.task_ref for member in result.members] == [task_ref]
    task_create.assert_called_once()
    taskset_create.assert_called_once()


def test_register_uploaded_dataset_rehydrates_reused_taskset_members(monkeypatch):
    from types import SimpleNamespace

    from nemo_evaluator.api.schemas import Task, TaskRef, Taskset, TasksetInput
    from nemo_evaluator.sdk.harbor import (
        HarborDatasetReceipt,
        HarborUploadReceipt,
        _digest,
        register_harbor_dataset,
    )
    from nemo_evaluator.sdk.task_resources import EvaluatorTasksResource
    from nemo_evaluator.sdk.taskset_resources import EvaluatorTasksetsResource

    def definition(seed):
        return HarborTaskDefinition(
            kind="harbor",
            source=HarborArchiveSource(fileset_ref=f"default/files#{seed}", files_hash=seed * 64),
            harbor_hash=HarborTaskHash(digest=seed * 64, harbor_version="0.20.0"),
        )

    local_definition = definition("a")
    reused_definition = definition("b")
    suite_definition = definition("c")
    reused_task = Task.model_construct(
        name="local-task", workspace="default", spec=reused_definition, metadata=[], revision=1
    )
    suite_task = Task.model_construct(
        name="suite-task", workspace="default", spec=suite_definition, metadata=[], revision=1
    )
    reused_ref = TaskRef(f"default/local-task#{_digest(TaskInput(spec=reused_definition))}")
    suite_ref = TaskRef(f"default/suite-task#{_digest(TaskInput(spec=suite_definition))}")
    taskset = Taskset.model_construct(
        name="suite",
        workspace="default",
        tasks=[suite_ref],
        files_ref=None,
        description=None,
        metadata=[],
        revision=1,
    )
    taskset_digest = _digest(TasksetInput(tasks=[suite_ref]))

    tasks_by_name = {"local-task": reused_task, "suite-task": suite_task}
    task_retrieve = Mock(side_effect=lambda name, **_: tasks_by_name[name])
    task_create = Mock()
    taskset_create = Mock()
    monkeypatch.setattr(EvaluatorTasksResource, "__init__", lambda self, client: None)
    monkeypatch.setattr(EvaluatorTasksResource, "retrieve", task_retrieve)
    monkeypatch.setattr(EvaluatorTasksResource, "create", task_create)
    monkeypatch.setattr(
        EvaluatorTasksResource,
        "list_revisions",
        Mock(
            side_effect=lambda name, **_: SimpleNamespace(
                data=[SimpleNamespace(revision=1, content_hash=reused_ref.root.rsplit("#", 1)[1])],
                pagination=SimpleNamespace(total_pages=1),
            )
        ),
    )
    monkeypatch.setattr(EvaluatorTasksetsResource, "__init__", lambda self, client: None)
    monkeypatch.setattr(EvaluatorTasksetsResource, "retrieve", Mock(return_value=taskset))
    monkeypatch.setattr(EvaluatorTasksetsResource, "create", taskset_create)
    monkeypatch.setattr(
        EvaluatorTasksetsResource,
        "list_revisions",
        Mock(
            return_value=SimpleNamespace(
                data=[SimpleNamespace(revision=1, content_hash=taskset_digest)],
                pagination=SimpleNamespace(total_pages=1),
            )
        ),
    )
    monkeypatch.setattr("nemo_evaluator.sdk.harbor.EvaluatorClient.from_client", lambda _: Mock())

    receipt = HarborDatasetReceipt(
        members=[HarborUploadReceipt(native_name="local-task", task_name="local-task", definition=local_definition)]
    )
    result = register_harbor_dataset(receipt, client=Mock(), workspace="default", taskset_name="suite")

    assert result.taskset_ref is not None and result.taskset_ref.root == f"default/suite#{taskset_digest}"
    assert [(member.task_name, member.task_ref) for member in result.members] == [("suite-task", suite_ref)]
    assert result.members[0].definition == suite_definition
    task_create.assert_not_called()
    taskset_create.assert_not_called()


def test_reuse_registration_recovers_create_conflict():
    from types import SimpleNamespace

    from nemo_evaluator.api.schemas import Task
    from nemo_evaluator.sdk.harbor import _digest, _register
    from nemo_evaluator.sdk.task_resources import EvaluatorTasksResource
    from nemo_platform_plugin.client.errors import ConflictError

    intended = HarborTaskDefinition(
        kind="harbor",
        source=HarborArchiveSource(fileset_ref="default/files#intended", files_hash="a" * 64),
        harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
    )
    existing = HarborTaskDefinition(
        kind="harbor",
        source=HarborArchiveSource(fileset_ref="default/files#existing", files_hash="c" * 64),
        harbor_hash=HarborTaskHash(digest="d" * 64, harbor_version="0.20.0"),
    )
    stored = Task.model_construct(name="task", workspace="default", spec=existing, metadata=[], revision=1)
    digest = _digest(stored)
    resource = Mock(spec=EvaluatorTasksResource)
    resource.retrieve.side_effect = [NemoHTTPError(httpx.Response(404)), stored]
    resource.create.side_effect = ConflictError(httpx.Response(409))
    resource.list_revisions.return_value = SimpleNamespace(
        data=[SimpleNamespace(revision=1, content_hash=digest)],
        pagination=SimpleNamespace(total_pages=1),
    )

    assert (
        _register(resource, "task", TaskInput(spec=intended), "default", False, reuse_existing=True)
        == f"default/task#{digest}"
    )


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
    assert len(caught.value.receipt.members) == 1
    assert caught.value.receipt.taskset_ref is None
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
        source=HarborArchiveSource(fileset_ref="default/files#task_archive", files_hash=digest),
        harbor_hash=HarborTaskHash(digest="a" * 64, harbor_version="test"),
    )
    task, published = await service.create_task("stored", TaskInput(spec=definition), workspace="default")
    assert published
    assert task.spec.instruction == "Do it"
