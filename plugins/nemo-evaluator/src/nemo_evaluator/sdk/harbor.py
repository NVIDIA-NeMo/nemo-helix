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
from nemo_platform_plugin.client.errors import ConflictError, NemoClientError, NemoHTTPError
from nemo_platform_plugin.evaluator.client import EvaluatorClient
from nemo_platform_plugin.files.client import FilesClient
from pydantic import BaseModel, Field


class HarborUploadReceipt(BaseModel):
    native_name: str
    definition: HarborTaskDefinition
    task_name: str | None = Field(
        default=None,
        description="Target database name, available before registration so upload and registration can be separate.",
    )
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
    *,
    reuse_existing: bool = False,
) -> str:
    intended = _digest(value)
    try:
        existing = resource.retrieve(name, workspace=workspace)
    except NemoHTTPError as exc:
        if exc.status_code != 404:
            raise
        existing = None
    if existing is not None and (reuse_existing or _digest(existing) == intended):
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
        except ConflictError:
            if reuse_existing:
                published = resource.retrieve(name, workspace=workspace)
            else:
                # A create race is only successful if history contains intended content.
                page = 1
                while True:
                    history = resource.list_revisions(name, workspace=workspace, page=page)
                    if any(item.content_hash == intended for item in history.data):
                        return f"{workspace}/{name}#{intended}"
                    if history.pagination is None or page >= history.pagination.total_pages:
                        raise
                    page += 1
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
    published_digest = _digest(published)
    if not reuse_existing and published_digest != intended:
        raise ValueError("Publication response differs from intended content")
    page = 1
    while True:
        history = resource.list_revisions(name, workspace=workspace, page=page)
        for revision in history.data:
            if revision.revision == published.revision:
                if revision.content_hash != published_digest:
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
    receipt = HarborUploadReceipt(native_name=root.name, definition=definition, task_name=name)
    if register:
        try:
            pin = _register(
                EvaluatorTasksResource(EvaluatorClient.from_client(client)),
                name,
                TaskInput(spec=definition),
                workspace,
                replace,
            )
            receipt.task_ref = TaskRef(pin)
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


def register_harbor_dataset(
    receipt: HarborDatasetReceipt,
    *,
    client: NemoClient,
    taskset_name: str,
    workspace: str = "default",
) -> HarborDatasetReceipt:
    """Register an uploaded dataset, reusing existing Tasks and Tasksets by name.

    The returned receipt always describes the exact pinned members of the registered or reused
    Taskset. Completed member pins remain on ``receipt`` if registration fails partway through.
    """
    TasksetRef(f"{workspace}/{taskset_name}")
    if not receipt.members:
        raise ValueError("Dataset receipt has no tasks")

    evaluator_client = EvaluatorClient.from_client(client)
    tasks = EvaluatorTasksResource(evaluator_client)
    tasksets = EvaluatorTasksetsResource(evaluator_client)

    def retrieve_harbor_task(ref: TaskRef) -> tuple[str, HarborTaskDefinition]:
        address, separator, revision = ref.root.partition("#")
        task_workspace, task_name = address.split("/", 1)
        task = tasks.retrieve(
            task_name,
            workspace=task_workspace,
            revision=revision if separator else None,
        )
        if not isinstance(task.spec, HarborTaskDefinition):
            raise ValueError(f"Taskset member {ref.root} is not a Harbor task")
        return task.name, task.spec

    try:
        refs: list[TaskRef] = []
        for member in receipt.members:
            name = member.task_name or member.native_name
            member.task_name = name
            member.task_ref = TaskRef(
                _register(
                    tasks,
                    name,
                    TaskInput(spec=member.definition),
                    workspace,
                    False,
                    reuse_existing=True,
                )
            )
            _, member.definition = retrieve_harbor_task(member.task_ref)
            refs.append(member.task_ref)

        taskset_ref = TasksetRef(
            _register(
                tasksets,
                taskset_name,
                TasksetInput(tasks=sorted(refs, key=lambda ref: ref.root)),
                workspace,
                False,
                reuse_existing=True,
            )
        )
        _, _, revision = taskset_ref.root.partition("#")
        taskset = tasksets.retrieve(taskset_name, workspace=workspace, revision=revision)

        attempted = {member.task_ref.root: member for member in receipt.members if member.task_ref is not None}
        effective_members = []
        for task_ref in taskset.tasks:
            task_name, definition = retrieve_harbor_task(task_ref)
            member = attempted.get(task_ref.root)
            if member is None:
                member = HarborUploadReceipt(
                    native_name=task_name,
                    definition=definition,
                    task_name=task_name,
                    task_ref=task_ref,
                )
            else:
                member.definition = definition
            effective_members.append(member)
        receipt.members = effective_members
        receipt.taskset_ref = taskset_ref
    except Exception as exc:
        raise HarborUploadError("Dataset registration incomplete; completed members are retained", receipt) from exc
    return receipt
