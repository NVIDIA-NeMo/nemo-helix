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

pytest.importorskip("harbor")


@pytest.fixture
def root(tmp_path):
    """Create a minimal on-disk Harbor task that can be packaged and registered."""
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
    """Provide an in-memory Files client that checks upload length and supports archive readback."""
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
    """Corrupt uploaded bytes before readback and verify publication detects the checksum mismatch."""
    client, objects = files
    upload = client.upload_file.side_effect

    def corrupt(**kwargs):
        result = upload(**kwargs)
        objects[kwargs["path"]] = b"corrupted"
        return result

    client.upload_file.side_effect = corrupt
    with pytest.raises(ValueError, match="checksum"):
        publish_harbor_task_archive(root, files_client=client, fileset_ref="default/harbor-tasks")


@pytest.mark.parametrize("bad", ["step", "metadata", "checksum"])
async def test_registration_rejects_before_persistence(root, entity_store, bad):
    """Feed unsafe or corrupted archives to registration and verify no entities are persisted."""
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
    """Verify task folder names cannot collide with the temporary download file or extraction directory."""
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
    assert isinstance(task.spec, HarborTaskDefinition)
    assert task.spec.instruction == "Do it"


def _resource(monkeypatch, files_client):
    from nemo_evaluator.sdk.task_resources import EvaluatorTasksResource
    from nemo_helix_plugin.evaluator.client import EvaluatorClient

    monkeypatch.setattr("nemo_evaluator.sdk.task_resources.FilesClient.from_client", lambda _: files_client)
    return EvaluatorTasksResource(EvaluatorClient(base_url="http://test", workspace="default"))


@pytest.mark.parametrize("instruction", ["", " \n", "<!-- SPDX-License-Identifier: Apache-2.0 -->\n"])
def test_prepare_rejects_blank_instruction(root, files, monkeypatch, instruction):
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import discover_harbor_tasks

    task = discover_harbor_tasks(root)[0]
    (root / "instruction.md").write_text(instruction)  # Edited after discovery; publication recaptures.
    client, objects = files
    with pytest.raises(ValueError, match="instruction is empty"):
        _resource(monkeypatch, client).prepare(task)
    assert not objects


@pytest.mark.parametrize("name", ["foo#rev", "foo/bar", "", "a" * 256])
def test_source_write_invalid_name_before_upload(root, files, monkeypatch, name):
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import discover_harbor_tasks

    client, objects = files
    with pytest.raises(ValueError):
        _resource(monkeypatch, client).replace(name, task=discover_harbor_tasks(root)[0])
    assert not objects


def test_prepare_preserves_instruction_and_default_storage(root, files, monkeypatch):
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import discover_harbor_tasks

    prompt = "<!-- SPDX-License-Identifier: Apache-2.0 -->\n Do it \n"
    (root / "instruction.md").write_text(prompt)
    client, objects = files
    prepared = _resource(monkeypatch, client).prepare(discover_harbor_tasks(root)[0])
    assert prepared.spec.instruction == prompt
    assert prepared.spec.source.fileset_ref.startswith("default/harbor-tasks#task/")
    assert len(objects) == 1


def test_failed_task_write_preserves_prepared_input(root, files, monkeypatch):
    from nemo_evaluator.sdk.task_preparation import TaskPublicationError
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import discover_harbor_tasks
    from nemo_helix_plugin.client.errors import NemoHTTPError

    client, objects = files
    resource = _resource(monkeypatch, client)
    error = NemoHTTPError(httpx.Response(403, request=httpx.Request("PUT", "http://test")))
    monkeypatch.setattr(resource._client, "replace_task", Mock(side_effect=error))
    with pytest.raises(TaskPublicationError) as raised:
        resource.replace("task", task=discover_harbor_tasks(root)[0])
    assert raised.value.__cause__ is error
    assert isinstance(raised.value.prepared_task.spec, HarborTaskDefinition)
    assert raised.value.prepared_task.spec.native_task_id == "task"
    assert len(objects) == 1
    # Prepared-input retry does not wrap the error or upload again.
    with pytest.raises(NemoHTTPError):
        resource.replace("task", task=raised.value.prepared_task)
    assert client.upload_file.call_count == 1


@pytest.fixture
def async_files(files):
    _, objects = files
    async_client = Mock()
    async_client.create_fileset = AsyncMock(return_value=Mock())
    async_client.with_headers.return_value = async_client

    async def upload(**kwargs):
        objects[kwargs["path"]] = b"".join([chunk async for chunk in kwargs["content"]])
        return Mock()

    async def download(**kwargs):
        @asynccontextmanager
        async def stream(**_):
            async def chunks():
                yield objects[kwargs["path"]]

            yield chunks()

        return Mock(stream=stream)

    async_client.upload_file = AsyncMock(side_effect=upload)
    async_client.download_file = AsyncMock(side_effect=download)
    return async_client, objects


async def test_async_archive_matches_sync_and_verifies(root, files, async_files):
    from nemo_evaluator.harbor.publication import publish_harbor_task_archive_async

    client, objects = files
    expected = publish_harbor_task_archive(root, files_client=client, fileset_ref="default/harbor-tasks")
    objects.clear()  # Async readback must use bytes uploaded by the async path.
    async_client, _ = async_files
    actual = await publish_harbor_task_archive_async(
        root, files_client=async_client, fileset_ref="default/harbor-tasks"
    )
    assert actual == expected
    async_client.upload_file.assert_awaited_once()
    async_client.download_file.assert_awaited_once()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("method", ["create", "replace"])
@pytest.mark.parametrize(
    "workspace,fileset_ref,expected_storage",
    [
        (None, None, "default/harbor-tasks"),
        ("other", None, "other/harbor-tasks"),
        ("other", "storage/custom", "storage/custom"),
    ],
)
async def test_source_write_creates_fileset_and_preserves_prepared_retry(
    root, files, async_files, monkeypatch, asynchronous, method, workspace, fileset_ref, expected_storage
):
    from nemo_evaluator.sdk.task_preparation import TaskPublicationError
    from nemo_evaluator.sdk.task_resources import AsyncEvaluatorTasksResource
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import discover_harbor_tasks
    from nemo_helix_plugin.evaluator.client import AsyncEvaluatorClient

    client, objects = async_files if asynchronous else files
    if asynchronous:
        monkeypatch.setattr("nemo_evaluator.sdk.task_resources.AsyncFilesClient.from_client", lambda _: client)
        resource = AsyncEvaluatorTasksResource(AsyncEvaluatorClient(base_url="http://test", workspace="default"))
    else:
        resource = _resource(monkeypatch, client)
    error = RuntimeError("registration failed")
    write = AsyncMock(side_effect=error) if asynchronous else Mock(side_effect=error)
    monkeypatch.setattr(resource._client, f"{method}_task", write)

    async def invoke(task, **options):
        result = getattr(resource, method)("task", task=task, workspace=workspace, **options)
        return await result if asynchronous else result

    with pytest.raises(TaskPublicationError) as raised:
        await invoke(discover_harbor_tasks(root)[0], fileset_ref=fileset_ref)
    assert raised.value.__cause__ is error
    assert isinstance(raised.value.prepared_task.spec, HarborTaskDefinition)
    assert raised.value.prepared_task.spec.source.fileset_ref.startswith(f"{expected_storage}#task/")
    storage_workspace, storage_name = expected_storage.split("/")
    client.create_fileset.assert_called_once()
    kwargs = client.create_fileset.call_args.kwargs
    assert kwargs["workspace"] == storage_workspace
    assert kwargs["body"].name == storage_name
    assert kwargs["exist_ok"] is True
    assert write.call_args.kwargs["workspace"] == (workspace or "default")
    assert len(objects) == 1
    for operation in (client.upload_file, client.download_file):
        assert operation.call_args.kwargs["workspace"] == storage_workspace
        assert operation.call_args.kwargs["name"] == storage_name
    # A ready-input retry must propagate the original error without another Files operation.
    with pytest.raises(RuntimeError) as retried:
        await invoke(raised.value.prepared_task)
    assert retried.value is error
    for operation in (client.create_fileset, client.upload_file, client.download_file):
        operation.assert_called_once()
        if asynchronous:
            operation.assert_awaited_once()


async def test_async_archive_cancellation_does_not_cleanup_live_work(root, monkeypatch):
    import asyncio
    import threading

    from nemo_evaluator.harbor import publication

    entered = threading.Event()
    release = threading.Event()
    seen = []
    original = publication.capture_task

    def slow_capture(source, parent):
        entered.set()
        release.wait(timeout=10)
        seen.append(parent.exists())
        return original(source, parent)

    monkeypatch.setattr(publication, "capture_task", slow_capture)
    client = Mock()
    work = asyncio.create_task(
        publication.publish_harbor_task_archive_async(root, files_client=client, fileset_ref="default/harbor-tasks")
    )
    while not entered.is_set():
        await asyncio.sleep(0.01)
    work.cancel()
    await asyncio.sleep(0.02)
    assert not work.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await work
    assert seen == [True]
    client.create_fileset.assert_not_called()


@pytest.mark.parametrize("field", ["id", "intent", "inputs", "reference", "file"])
def test_discovered_source_drift_rejected_before_writes(root, files, field):
    from nemo_evaluator.sdk.task_preparation import prepare_task
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import discover_harbor_tasks

    task = discover_harbor_tasks(root)[0]
    if field == "file":
        (root / "instruction.md").write_text("Changed instruction")
    else:
        setattr(task, field, "changed" if field in {"id", "intent"} else {"changed": True})
    client, objects = files
    with pytest.raises(ValueError):
        prepare_task(task, files_client=client, workspace="default")
    assert not objects
    client.create_fileset.assert_not_called()


def test_discovered_task_publishes_current_payload(root, files):
    from nemo_evaluator.sdk.task_preparation import prepare_task
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import discover_harbor_tasks

    task = discover_harbor_tasks(root)[0]
    (root / "tests/test.sh").write_text("echo current")
    task.metadata["harbor_task_dir"] = "/ignored"
    client, objects = files
    prepared = prepare_task(task, files_client=client, workspace="default")
    assert prepared.spec.kind == "harbor"
    assert len(objects) == 1
    assert prepared.metadata == []


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_discovered_views_with_runner_selected_reward(root, files, async_files, asynchronous):
    from nemo_evaluator.sdk.task_preparation import prepare_task, prepare_task_async
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import discover_harbor_tasks
    from nemo_evaluator_sdk.agent_eval.tasks import SemanticReducer, SemanticView, ViewSignal

    task = discover_harbor_tasks(root)[0]
    task.views["quality"] = SemanticView(
        reducer=SemanticReducer.MEAN, signals=[ViewSignal(metric="harbor_reward", output="success")]
    )
    if asynchronous:
        prepared = await prepare_task_async(task, files_client=async_files[0], workspace="default")
    else:
        prepared = prepare_task(task, files_client=files[0], workspace="default")
    assert prepared.spec.views == task.views
    assert prepared.spec.kind == "harbor"
