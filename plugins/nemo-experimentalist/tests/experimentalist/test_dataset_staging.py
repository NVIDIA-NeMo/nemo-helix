# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from nemo_experimentalist_plugin.entities import Dataset, DatasetRef, Task
from nemo_experimentalist_plugin.experimentalist.components.dataset_staging import (
    distribute_insight_suite_tasks,
    stage_eval_author_inputs,
    stage_task_template,
)
from nemo_experimentalist_plugin.experimentalist.components.evaluator.harbor import HarborDataset
from nemo_platform_plugin.client.client import AsyncNemoClient


def _write_tree(root: Path, content: str) -> None:
    tests = root / "task-1" / "tests"
    tests.mkdir(parents=True)
    (tests / "test.sh").write_text(content, encoding="utf-8")


class _MemoryDataset(Dataset):
    def add_tasks(self, tasks: list[Task]) -> None:
        self.tasks.extend(tasks)


@dataclass(frozen=True)
class _FilesetItem:
    path: str


@dataclass(frozen=True)
class _FilesetListing:
    data: list[_FilesetItem]


class _FilesetResponse:
    def __init__(self, listing: _FilesetListing) -> None:
        self._listing = listing

    def data(self) -> _FilesetListing:
        return self._listing


class _FilesetDownload:
    async def read(self) -> bytes:
        return b"fileset content"


class _RecordingFilesClient:
    def __init__(self, paths: list[str]) -> None:
        self._paths = paths
        self.downloaded_paths: list[str] = []

    async def list_files(self, *, workspace: str, name: str, query_params: object) -> _FilesetResponse:
        return _FilesetResponse(_FilesetListing([_FilesetItem(path) for path in self._paths]))

    async def download_file(self, *, workspace: str, name: str, path: str) -> _FilesetDownload:
        self.downloaded_paths.append(path)
        return _FilesetDownload()


def test_distribute_insight_suite_tasks_uses_a_30_70_validation_train_split() -> None:
    insight_suite = _MemoryDataset(id="insight", tasks=[Task(id=f"task-{index}") for index in range(10)])
    train_dataset = _MemoryDataset(id="train")
    validation_dataset = _MemoryDataset(id="validation")

    distribute_insight_suite_tasks(insight_suite, train_dataset, validation_dataset)

    assert [task.id for task in validation_dataset.list_tasks()] == ["task-0", "task-1", "task-2"]
    assert [task.id for task in train_dataset.list_tasks()] == [
        "task-3",
        "task-4",
        "task-5",
        "task-6",
        "task-7",
        "task-8",
        "task-9",
    ]


def test_harbor_dataset_add_tasks_copies_task_directories(tmp_path: Path) -> None:
    source_root = tmp_path / "insight-suite"
    destination_root = tmp_path / "train"
    (source_root / "insight-task" / "task.toml").parent.mkdir(parents=True)
    (source_root / "insight-task" / "task.toml").write_text("", encoding="utf-8")
    (destination_root / "insight-task" / "task.toml").parent.mkdir(parents=True)
    (destination_root / "insight-task" / "stale.txt").write_text("old", encoding="utf-8")
    (destination_root / "insight-task" / "task.toml").write_text("", encoding="utf-8")
    insight_suite = HarborDataset.from_path(source_root)
    train_dataset = HarborDataset.from_path(destination_root)

    train_dataset.add_tasks(list(insight_suite.list_tasks()))

    assert (destination_root / "insight-task" / "task.toml").is_file()
    assert not (destination_root / "insight-task" / "stale.txt").exists()
    assert [task.id for task in train_dataset.list_tasks()] == ["insight-task"]
    assert Path(train_dataset.get_task("insight-task").uri.removeprefix("file://")) == destination_root / "insight-task"


def test_harbor_dataset_allows_an_empty_split_only_when_requested(tmp_path: Path) -> None:
    """Check that an empty split is accepted only for an Insight-generated suite."""
    with pytest.raises(ValueError, match="contains no Harbor task directories"):
        HarborDataset.from_path(tmp_path)

    dataset = HarborDataset.from_path(tmp_path, allow_empty=True)

    assert dataset.list_tasks() == []


def test_harbor_dataset_add_tasks_preserves_existing_task_when_replacement_is_invalid(tmp_path: Path) -> None:
    source_root = tmp_path / "insight-suite"
    destination_root = tmp_path / "train"
    source_task = source_root / "task-1"
    destination_task = destination_root / "task-1"
    source_task.mkdir(parents=True)
    destination_task.mkdir(parents=True)
    (source_task / "task.toml").write_text("", encoding="utf-8")
    (destination_task / "task.toml").write_text("", encoding="utf-8")
    (destination_task / "stale.txt").write_text("keep", encoding="utf-8")
    insight_suite = HarborDataset.from_path(source_root)
    train_dataset = HarborDataset.from_path(destination_root)
    (source_task / "task.toml").write_text("not valid = [", encoding="utf-8")

    with pytest.raises(tomllib.TOMLDecodeError):
        train_dataset.add_tasks(list(insight_suite.list_tasks()))

    assert (destination_task / "stale.txt").read_text(encoding="utf-8") == "keep"
    assert (destination_task / "task.toml").is_file()


@pytest.mark.asyncio
async def test_stage_eval_author_inputs_isolates_all_mutable_sources(tmp_path: Path) -> None:
    source = tmp_path / "source"
    train = source / "train"
    validation = source / "validation"
    template = source / "task-template"
    _write_tree(train, "train source")
    _write_tree(validation, "validation source")
    _write_tree(template, "template source")

    experiment = tmp_path / "experiment"
    staged = await stage_eval_author_inputs(
        experiment,
        train_dataset=DatasetRef(uri=str(train), metadata={"id": "train"}),
        validation_dataset=DatasetRef(uri=str(validation), metadata={"id": "validation"}),
        task_template=DatasetRef(uri=template.as_uri(), metadata={"id": "template"}),
        client=cast(AsyncNemoClient, object()),
        workspace="default",
    )

    staged_train = Path(staged.train_dataset.uri)
    staged_validation = Path(staged.validation_dataset.uri)
    staged_template = Path(staged.task_template.uri)
    (staged_train / "task-1" / "tests" / "test.sh").write_text("curated train", encoding="utf-8")
    (staged_validation / "task-1" / "tests" / "test.sh").write_text("curated validation", encoding="utf-8")
    generated = staged_template.parent / "generated-task"
    generated.mkdir()

    assert (train / "task-1" / "tests" / "test.sh").read_text(encoding="utf-8") == "train source"
    assert (validation / "task-1" / "tests" / "test.sh").read_text(encoding="utf-8") == "validation source"
    assert not (template.parent / "generated-task").exists()
    assert staged_train == experiment / "dataset" / "train"
    assert staged_validation == experiment / "dataset" / "validation"
    assert staged_template == experiment / "dataset" / "task-template"
    assert staged.train_dataset.metadata == {"id": "train"}
    assert staged.validation_dataset.metadata == {"id": "validation"}
    assert staged.task_template.metadata == {"id": "template"}


@pytest.mark.asyncio
async def test_stage_eval_author_inputs_keeps_train_and_validation_isolated_when_source_is_shared(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared"
    template = tmp_path / "template"
    _write_tree(shared, "shared source")
    _write_tree(template, "template source")
    shared_ref = DatasetRef(uri=str(shared))

    staged = await stage_eval_author_inputs(
        tmp_path / "experiment",
        train_dataset=shared_ref,
        validation_dataset=shared_ref,
        task_template=DatasetRef(uri=str(template)),
        client=cast(AsyncNemoClient, object()),
        workspace="default",
    )
    staged_train_test = Path(staged.train_dataset.uri) / "task-1" / "tests" / "test.sh"
    staged_validation_test = Path(staged.validation_dataset.uri) / "task-1" / "tests" / "test.sh"
    staged_train_test.write_text("train only", encoding="utf-8")

    assert staged_validation_test.read_text(encoding="utf-8") == "shared source"
    assert (shared / "task-1" / "tests" / "test.sh").read_text(encoding="utf-8") == "shared source"


@pytest.mark.asyncio
async def test_stage_eval_author_inputs_reuses_dataset_destinations_but_refreshes_template(tmp_path: Path) -> None:
    source = tmp_path / "source"
    template = tmp_path / "template"
    experiment = tmp_path / "experiment"
    _write_tree(source, "source")
    _write_tree(template, "template")
    staged = await stage_eval_author_inputs(
        experiment,
        train_dataset=DatasetRef(uri=str(source)),
        validation_dataset=DatasetRef(uri=str(source)),
        task_template=DatasetRef(uri=str(template)),
        client=cast(AsyncNemoClient, object()),
        workspace="default",
    )
    (Path(staged.train_dataset.uri) / "task-1" / "tests" / "test.sh").write_text("curated", encoding="utf-8")
    (template / "task-1" / "tests" / "test.sh").write_text("updated template", encoding="utf-8")

    restaged = await stage_eval_author_inputs(
        experiment,
        train_dataset=DatasetRef(uri=str(source)),
        validation_dataset=DatasetRef(uri=str(source)),
        task_template=DatasetRef(uri=str(template)),
        client=cast(AsyncNemoClient, object()),
        workspace="default",
    )

    staged_test = Path(restaged.train_dataset.uri) / "task-1" / "tests" / "test.sh"
    assert staged_test.read_text(encoding="utf-8") == "curated"
    staged_template_test = Path(restaged.task_template.uri) / "task-1" / "tests" / "test.sh"
    assert staged_template_test.read_text(encoding="utf-8") == "updated template"


@pytest.mark.asyncio
async def test_stage_task_template_hydrates_and_refreshes_fileset_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []
    content = "first fileset template"

    class _Response:
        def __init__(self, body: object) -> None:
            self._body = body

        def data(self) -> object:
            return self._body

    class _Download:
        async def read(self) -> bytes:
            return content.encode()

    class _File:
        path = "task-1/tests/test.sh"

    class _Files:
        data = [_File()]

    class FakeFiles:
        async def list_files(self, *, workspace: str, name: str, query_params: object) -> _Response:
            calls.append({"op": "list", "workspace": workspace, "name": name, "query_params": query_params})
            return _Response(_Files())

        async def download_file(self, *, workspace: str, name: str, path: str) -> _Download:
            calls.append({"op": "download", "workspace": workspace, "name": name, "path": path})
            return _Download()

    from nemo_experimentalist_plugin.experimentalist.components import dataset_staging

    monkeypatch.setattr(dataset_staging.AsyncFilesClient, "from_client", lambda _client: FakeFiles())
    client = cast(AsyncNemoClient, object())
    ref = DatasetRef(uri="fileset://workspace-a/task-template", metadata={"id": "template-fileset"})

    first = await stage_task_template(tmp_path / "experiment", ref, client=client, workspace="workspace-a")
    content = "second fileset template"
    second = await stage_task_template(tmp_path / "experiment", ref, client=client, workspace="workspace-a")

    expected_path = tmp_path / "experiment" / "dataset" / "task-template"
    assert first.uri == str(expected_path)
    assert second.uri == str(expected_path)
    assert second.metadata == {"id": "template-fileset"}
    assert (expected_path / "task-1" / "tests" / "test.sh").read_text(encoding="utf-8") == content
    assert calls == [
        {"op": "list", "workspace": "workspace-a", "name": "task-template", "query_params": None},
        {"op": "download", "workspace": "workspace-a", "name": "task-template", "path": "task-1/tests/test.sh"},
        {"op": "list", "workspace": "workspace-a", "name": "task-template", "query_params": None},
        {"op": "download", "workspace": "workspace-a", "name": "task-template", "path": "task-1/tests/test.sh"},
    ]


async def test_stage_task_template_honors_fileset_fragment_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    class _Response:
        def __init__(self, body: object) -> None:
            self._body = body

        def data(self) -> object:
            return self._body

    class _Download:
        async def read(self) -> bytes:
            return b"fragment-scoped template"

    class _File:
        path = "task-root/tests/test.sh"

    class _Files:
        data = [_File()]

    class FakeFiles:
        async def list_files(self, *, workspace: str, name: str, query_params: object) -> _Response:
            calls.append({"op": "list", "workspace": workspace, "name": name, "query_params": query_params})
            return _Response(_Files())

        async def download_file(self, *, workspace: str, name: str, path: str) -> _Download:
            calls.append({"op": "download", "workspace": workspace, "name": name, "path": path})
            return _Download()

    from nemo_experimentalist_plugin.experimentalist.components import dataset_staging

    monkeypatch.setattr(dataset_staging.AsyncFilesClient, "from_client", lambda _client: FakeFiles())
    client = AsyncNemoClient(base_url="http://test")
    ref = DatasetRef(uri="fileset://workspace-a/task-template#task-root", metadata={"id": "template-fileset"})

    try:
        staged = await stage_task_template(tmp_path / "experiment", ref, client=client, workspace="workspace-a")
    finally:
        await client.close()

    expected_path = tmp_path / "experiment" / "dataset" / "task-template"
    assert staged.uri == str(expected_path)
    assert staged.metadata == {"id": "template-fileset"}
    assert (expected_path / "tests" / "test.sh").read_text(encoding="utf-8") == "fragment-scoped template"
    assert calls == [
        {"op": "list", "workspace": "workspace-a", "name": "task-template", "query_params": {"path": "task-root"}},
        {"op": "download", "workspace": "workspace-a", "name": "task-template", "path": "task-root/tests/test.sh"},
    ]


async def test_stage_task_template_rejects_fileset_path_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nemo_experimentalist_plugin.experimentalist.components import dataset_staging

    files_client = _RecordingFilesClient(["../escape.txt"])
    monkeypatch.setattr(dataset_staging.AsyncFilesClient, "from_client", lambda _client: files_client)
    client = AsyncNemoClient(base_url="http://test")
    ref = DatasetRef(uri="fileset://workspace-a/task-template", metadata={"id": "template-fileset"})

    try:
        with pytest.raises(ValueError, match="escapes staging destination"):
            await stage_task_template(tmp_path / "experiment", ref, client=client, workspace="workspace-a")
    finally:
        await client.close()

    assert files_client.downloaded_paths == []
    assert not (tmp_path / "experiment" / "dataset" / "escape.txt").exists()


async def test_stage_task_template_rejects_prefixed_fileset_path_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nemo_experimentalist_plugin.experimentalist.components import dataset_staging

    files_client = _RecordingFilesClient(["task-root/../../escape.txt"])
    monkeypatch.setattr(dataset_staging.AsyncFilesClient, "from_client", lambda _client: files_client)
    client = AsyncNemoClient(base_url="http://test")
    ref = DatasetRef(uri="fileset://workspace-a/task-template#task-root", metadata={"id": "template-fileset"})

    try:
        with pytest.raises(ValueError, match="escapes staging destination"):
            await stage_task_template(tmp_path / "experiment", ref, client=client, workspace="workspace-a")
    finally:
        await client.close()

    assert files_client.downloaded_paths == []
    assert not (tmp_path / "experiment" / "dataset" / "escape.txt").exists()


async def test_stage_task_template_rejects_fileset_paths_outside_fragment_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nemo_experimentalist_plugin.experimentalist.components import dataset_staging

    files_client = _RecordingFilesClient(["other-root/tests/test.sh"])
    monkeypatch.setattr(dataset_staging.AsyncFilesClient, "from_client", lambda _client: files_client)
    client = AsyncNemoClient(base_url="http://test")
    ref = DatasetRef(uri="fileset://workspace-a/task-template#task-root", metadata={"id": "template-fileset"})

    try:
        with pytest.raises(ValueError, match="outside requested prefix"):
            await stage_task_template(tmp_path / "experiment", ref, client=client, workspace="workspace-a")
    finally:
        await client.close()

    assert files_client.downloaded_paths == []
