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


class HarborUploadDetails(BaseModel):
    """Result of publishing one local Harbor task package.

    Archive publication always populates ``native_name`` and ``definition``.
    The registration fields are populated only after the corresponding
    Evaluator Task has been registered successfully.
    """

    native_name: str = Field(description="Name of the task's source directory in the Harbor dataset.")
    definition: HarborTaskDefinition = Field(
        description="Verified Harbor task definition that points to the published archive."
    )
    task_name: str | None = Field(
        default=None,
        description="Name of the registered Evaluator Task, which may differ from the native Harbor name.",
    )
    task_ref: TaskRef | None = Field(
        default=None,
        description="Immutable reference to the registered Evaluator Task revision; None when not registered.",
    )


class HarborDatasetUploadDetails(BaseModel):
    """Result of publishing a local Harbor dataset and optionally registering it.

    ``members`` retains every task completed before an upload or registration
    failure, allowing callers to inspect partial progress through
    ``HarborUploadError.upload_details``.
    """

    members: list[HarborUploadDetails] = Field(
        default_factory=list,
        description="Per-task receipts in deterministic source-directory order, including partial progress.",
    )
    taskset_ref: TasksetRef | None = Field(
        default=None,
        description="Immutable reference to the registered Taskset revision; None when not registered.",
    )


class HarborUploadError(RuntimeError):
    """Completed artifacts/pins remain available for inspection and retry."""

    def __init__(self, message: str, upload_details: HarborDatasetUploadDetails):
        super().__init__(message)
        self.upload_details = upload_details


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


def _upload_harbor_archive(
    task_path: Path,
    *,
    client: NemoClient,
    fileset_ref: str,
    path_prefix: str,
) -> HarborUploadDetails:
    """Publish verified archive bytes without registering a task entity."""
    definition = publish_harbor_task_archive(
        task_path,
        files_client=FilesClient.from_client(client),
        fileset_ref=fileset_ref,
        path_prefix=path_prefix,
    )
    return HarborUploadDetails(native_name=task_path.name, definition=definition)


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
) -> HarborUploadDetails:
    """Upload, verify, and register one local Harbor task.

    The task directory is captured while quiescent, packaged as a deterministic
    compressed archive, uploaded, and verified by downloading and inspecting the
    stored bytes. By default, the archive is stored at
    ``<workspace>/harbor-tasks#[<path_prefix>/]<source-directory>/<sha256>/task_archive``.

    When registration is enabled, this function creates an Evaluator Task from
    the verified archive definition. It reuses an existing Task with identical
    content. If the existing content differs, registration fails unless
    ``replace=True``. The returned receipt includes the immutable, digest-pinned
    reference for the exact registered revision.

    Args:
        task_path: Path to a valid native Harbor task directory. The directory
            name is retained as ``HarborUploadDetails.native_name`` and in the
            archive's object path.
        client: Authenticated NeMo Platform client used for Files and Evaluator
            operations.
        workspace: Workspace in which to register the Task and, unless
            ``fileset_ref`` is set, store its archive.
        task_name: Evaluator Task name. Defaults to the source directory name
            and may differ from that native name.
        fileset_ref: Destination in ``workspace/fileset`` form. Defaults to
            ``<workspace>/harbor-tasks``.
        path_prefix: Optional path inserted before the source directory name in
            the destination Fileset.
        register: Whether to register an Evaluator Task after publishing the
            archive. When false, no Evaluator API calls are made.
        replace: Whether registration may replace an existing Task whose content
            differs. This option has no effect when ``register`` is false.

    Returns:
        A receipt containing the native directory name and verified Harbor task
        definition. When ``register`` is true, ``task_name`` and ``task_ref`` are
        also populated; otherwise, both are ``None``.

    Raises:
        ValueError: If a path, name, package, archive, or stored readback is
            invalid.
        HarborUploadError: If Task registration fails after the archive is
            published, including when different Task content exists and
            ``replace`` is false. The exception's ``upload_details.members[0]`` retains
            the completed archive details for inspection or retry.

    With ``register=False``, callers can register later by passing
    ``TaskInput(spec=receipt.definition)`` to
    ``EvaluatorTasksResource.create``. Registration through this helper also
    reconciles uncertain create responses and verifies the exact revision pin.
    """
    root = Path(task_path)
    name = task_name or root.name
    TaskRef(f"{workspace}/{name}")
    upload_details = _upload_harbor_archive(
        root,
        client=client,
        fileset_ref=fileset_ref or f"{workspace}/harbor-tasks",
        path_prefix=path_prefix,
    )
    if register:
        try:
            pin = _register(
                EvaluatorTasksResource(EvaluatorClient.from_client(client)),
                name,
                TaskInput(spec=upload_details.definition),
                workspace,
                replace,
            )
            upload_details.task_name, upload_details.task_ref = name, TaskRef(pin)
        except Exception as exc:
            raise HarborUploadError(
                f"Task registration failed: {name}", HarborDatasetUploadDetails(members=[upload_details])
            ) from exc
    return upload_details


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
) -> HarborDatasetUploadDetails:
    """Upload, verify, and optionally register a local Harbor dataset.

    ``dataset_path`` may identify one Harbor task directory or a dataset
    directory whose immediate child directories are Harbor tasks. Before making
    remote writes, this function captures and validates every task, rejects
    duplicate runtime or registered identities, and sorts members by source
    directory name. A ``task_template`` child is excluded with a warning; every
    other dataset entry must be a task directory containing ``task.toml``.

    Each task is packaged as a deterministic compressed archive, uploaded, and
    verified by downloading and inspecting the stored bytes. By default,
    archives are grouped at
    ``<workspace>/harbor-tasksets#[<dataset_prefix>/]<source-directory>/<sha256>/task_archive``.
    This Fileset prefix only groups uploaded objects; registered Taskset
    membership comes from the immutable Task references stored in the Taskset.

    When registration is enabled, each verified archive is registered as an
    Evaluator Task before the Taskset is registered. Existing entities with
    identical content are reused. If existing content differs, registration
    fails unless ``replace=True``. Every successful registration is resolved to
    and verified against its exact digest-pinned revision.

    Args:
        dataset_path: Path to one native Harbor task or to a directory containing
            native Harbor task directories.
        client: Authenticated NeMo Platform client used for Files and Evaluator
            operations.
        workspace: Workspace in which to register Tasks and the Taskset and,
            unless ``fileset_ref`` is set, store their archives.
        taskset_name: Name of the Evaluator Taskset to register. Required when
            ``register`` is true and unused otherwise.
        fileset_ref: Destination in ``workspace/fileset`` form. Defaults to
            ``<workspace>/harbor-tasksets``.
        dataset_prefix: Optional path inserted before each source directory name
            in the destination Fileset. Defaults to the dataset directory name;
            use an empty string to omit the prefix.
        register: Whether to register each Evaluator Task and their Taskset after
            publishing all archives. When false, no Evaluator API calls are made.
        replace: Whether registration may replace existing Tasks or a Taskset
            whose content differs. This option has no effect when ``register`` is
            false.
        member_names: Mapping from source directory names to Evaluator Task
            names. Unmapped members retain their source directory names.

    Returns:
        A receipt whose ``members`` are in deterministic source-directory order.
        Each member always contains its native name and verified Harbor
        definition. When ``register`` is true, members also contain registered
        Task names and immutable references, and ``taskset_ref`` contains the
        immutable Taskset reference. Registration fields are ``None`` when
        registration is disabled.

    Raises:
        ValueError: If the dataset, a member package, a destination path, an
            entity name, or a name mapping is invalid; if the dataset is empty;
            or if runtime or registered task identities are duplicated.
        HarborUploadError: If an upload or registration fails after preflight.
            The exception's ``upload_details`` retains all completed uploads, exact
            Task revision pins, and the Taskset pin when available. This includes
            registration conflicts when ``replace`` is false.

    Completed remote operations are not rolled back after a failure. Use
    ``HarborUploadError.upload_details`` to inspect partial progress or safely retry;
    identical archives and registered entity content are reused.
    """
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
    result = HarborDatasetUploadDetails()
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
            name = names[path.name] if path.name in names else path.name
            TaskRef(f"{workspace}/{name}")
            if native.task_id.casefold() in ids or name.casefold() in db_names:
                raise ValueError("Duplicate runtime or DB task identity")
            ids.add(native.task_id.casefold())
            db_names.add(name.casefold())
            snapshots.append((snapshot, name))
        try:
            for snapshot, _ in snapshots:
                receipt = _upload_harbor_archive(
                    snapshot,
                    client=client,
                    fileset_ref=fileset_ref or f"{workspace}/harbor-tasksets",
                    path_prefix=prefix,
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
