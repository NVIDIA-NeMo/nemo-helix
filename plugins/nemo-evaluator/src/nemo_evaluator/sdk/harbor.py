# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Upload local Harbor tasks/datasets and optionally register immutable evaluator pins."""

from pathlib import Path

from nemo_evaluator.api.schemas import Task, TaskInput, TaskRef, Taskset, TasksetInput, TasksetRef
from nemo_evaluator.api.task_definitions.harbor import HarborTaskDefinition, validate_archive_path
from nemo_evaluator.entities import TaskEntity, TasksetEntity
from nemo_evaluator.harbor.archive import capture_task, private_directory, validate_native_task_inputs
from nemo_evaluator.harbor.publication import publish_harbor_task_archive
from nemo_evaluator.revisions import head_digest
from nemo_evaluator.sdk.task_resources import EvaluatorTasksResource
from nemo_evaluator.sdk.taskset_resources import EvaluatorTasksetsResource
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.client.errors import NemoClientError, NemoHTTPError
from nemo_platform_plugin.evaluator.client import EvaluatorClient
from nemo_platform_plugin.files.client import FilesClient
from pydantic import BaseModel, Field


class HarborUploadReceipt(BaseModel):
    native_name: str
    definition: HarborTaskDefinition
    task_name: str | None = None
    task_ref: TaskRef | None = None


class HarborDatasetReceipt(BaseModel):
    members: list[HarborUploadReceipt] = Field(default_factory=list)
    taskset_ref: TasksetRef | None = None


class HarborUploadError(RuntimeError):
    """Completed artifacts/pins remain available for inspection and retry."""

    def __init__(self, message: str, receipt: HarborDatasetReceipt):
        super().__init__(message)
        self.receipt = receipt


def _digest(value: TaskInput | Task | TasksetInput | Taskset) -> str:
    if isinstance(value, (Task, TaskInput)):
        entity = TaskEntity(name="comparison", workspace="default", spec=value.spec, metadata=value.metadata)
    else:
        entity = TasksetEntity(
            name="comparison",
            workspace="default",
            # Match the server's canonical membership order.
            tasks=sorted(value.tasks, key=lambda ref: ref.root),
            files_ref=value.files_ref,
            description=value.description,
            metadata=value.metadata,
        )
    return head_digest(entity)


def _register(
    resource: EvaluatorTasksResource | EvaluatorTasksetsResource,
    name: str,
    value: TaskInput | TasksetInput,
    workspace: str,
    replace: bool,
) -> str:
    intended = _digest(value)
    try:
        existing = resource.retrieve(name, workspace=workspace)
    except NemoHTTPError as exc:
        if exc.status_code != 404:
            raise
        existing = None
    if existing is not None and _digest(existing) == intended:
        published = existing
    else:
        if existing is not None and not replace:
            raise ValueError(f"Different content already exists at {workspace}/{name}; use replace=True explicitly")
        try:
            if isinstance(resource, EvaluatorTasksResource) and isinstance(value, TaskInput):
                operation = resource.replace if replace else resource.create
                published = operation(name, workspace=workspace, task=value)
            elif isinstance(resource, EvaluatorTasksetsResource) and isinstance(value, TasksetInput):
                operation = resource.replace if replace else resource.create
                published = operation(name, workspace=workspace, taskset=value)
            else:
                raise TypeError("Mismatched registration resource")
        except NemoClientError:
            # A lost response or create race is only successful if history contains intended content.
            page = 1
            while True:
                history = resource.list_revisions(name, workspace=workspace, page=page)
                if any(item.content_hash == intended for item in history.data):
                    return f"{workspace}/{name}#{intended}"
                if history.pagination is None or page >= history.pagination.total_pages:
                    raise
                page += 1
    if _digest(published) != intended:
        raise ValueError("Publication response differs from intended content")
    page = 1
    while True:
        history = resource.list_revisions(name, workspace=workspace, page=page)
        for revision in history.data:
            if revision.revision == published.revision:
                if revision.content_hash != intended:
                    raise ValueError("Published revision content mismatch")
                return f"{workspace}/{name}#{revision.content_hash}"
        if history.pagination is None or page >= history.pagination.total_pages:
            raise ValueError("Published revision missing from history")
        page += 1


def upload_harbor_task(
    task_path: Path | str,
    *,
    client: NemoClient,
    workspace: str = "default",
    task_name: str | None = None,
    fileset_ref: str | None = None,
    path_prefix: str = "",
    register: bool = True,
    replace: bool = False,
) -> HarborUploadReceipt:
    root = Path(task_path)
    name = task_name or root.name
    TaskRef(f"{workspace}/{name}")
    definition = publish_harbor_task_archive(
        root,
        files_client=FilesClient.from_client(client),
        fileset_ref=fileset_ref or f"{workspace}/harbor-tasks",
        path_prefix=path_prefix,
    )
    receipt = HarborUploadReceipt(native_name=root.name, definition=definition)
    if register:
        try:
            pin = _register(
                EvaluatorTasksResource(EvaluatorClient.from_client(client)),
                name,
                TaskInput(spec=definition),
                workspace,
                replace,
            )
            receipt.task_name, receipt.task_ref = name, TaskRef(pin)
        except Exception as exc:
            raise HarborUploadError(
                f"Task registration failed: {name}", HarborDatasetReceipt(members=[receipt])
            ) from exc
    return receipt


def upload_harbor_dataset(
    dataset_path: Path | str,
    *,
    client: NemoClient,
    workspace: str = "default",
    taskset_name: str | None = None,
    fileset_ref: str | None = None,
    dataset_prefix: str | None = None,
    register: bool = True,
    replace: bool = False,
    member_names: dict[str, str] | None = None,
) -> HarborDatasetReceipt:
    root = Path(dataset_path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Dataset must be a real directory")
    if register:
        if not taskset_name:
            raise ValueError("taskset_name is required when register=True")
        TasksetRef(f"{workspace}/{taskset_name}")
    prefix = root.name if dataset_prefix is None else dataset_prefix
    if prefix:
        validate_archive_path(prefix)
    roots = [root] if (root / "task.toml").is_file() else sorted(root.iterdir())
    members = []
    for path in roots:
        if path.name == "task_template":
            import warnings

            warnings.warn(f"Excluded template directory: {path}", stacklevel=2)
            continue
        if path.is_symlink() or not path.is_dir() or not (path / "task.toml").is_file():
            raise ValueError(f"Unhandled dataset entry: {path}; package shared runtime files inside each task")
        members.append(path)
    if not members:
        raise ValueError("Dataset has no tasks")
    names = member_names or {}
    if names.keys() - {path.name for path in members}:
        raise ValueError("Member-name mapping contains unknown tasks")
    result = HarborDatasetReceipt()
    # Capture every task before remote writes so all metadata, names and structure are preflighted.
    with private_directory() as staging:
        snapshots = []
        ids: set[str] = set()
        db_names: set[str] = set()
        for index, path in enumerate(members):
            parent = staging / str(index)
            parent.mkdir()
            snapshot = capture_task(path, parent)
            native = validate_native_task_inputs(snapshot)
            name = names.get(path.name, path.name)
            TaskRef(f"{workspace}/{name}")
            if native.task_id.casefold() in ids or name.casefold() in db_names:
                raise ValueError("Duplicate runtime or DB task identity")
            ids.add(native.task_id.casefold())
            db_names.add(name.casefold())
            snapshots.append((snapshot, name))
        try:
            for snapshot, name in snapshots:
                receipt = upload_harbor_task(
                    snapshot,
                    client=client,
                    workspace=workspace,
                    task_name=name,
                    fileset_ref=fileset_ref or f"{workspace}/harbor-tasksets",
                    path_prefix=prefix,
                    register=False,
                )
                result.members.append(receipt)
            if register:
                assert taskset_name is not None
                resource = EvaluatorTasksResource(EvaluatorClient.from_client(client))
                for receipt, (_, name) in zip(result.members, snapshots, strict=True):
                    receipt.task_ref = TaskRef(
                        _register(resource, name, TaskInput(spec=receipt.definition), workspace, replace)
                    )
                    receipt.task_name = name
                taskset = TasksetInput(
                    tasks=sorted(
                        (member.task_ref for member in result.members if member.task_ref is not None),
                        key=lambda ref: ref.root,
                    )
                )
                result.taskset_ref = TasksetRef(
                    _register(
                        EvaluatorTasksetsResource(EvaluatorClient.from_client(client)),
                        taskset_name,
                        taskset,
                        workspace,
                        replace,
                    )
                )
        except Exception as exc:
            raise HarborUploadError(
                "Dataset upload/registration incomplete; completed members are retained", result
            ) from exc
    return result
