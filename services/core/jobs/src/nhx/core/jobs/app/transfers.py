# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate Task-local transfers without collecting output or claiming file paths."""

from datetime import UTC, datetime, timedelta

from httpx import HTTPError
from nemo_helix_plugin.client.errors import HTTPClientError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.jobs.providers import DistributedGPUExecutionProvider
from nemo_helix_plugin.jobs.spec import HelixJobStepSpec
from nemo_helix_plugin.jobs.transfers import StepTransferState, TaskTransferProgress, TransferAction
from nhx.common.entities.client import EntityClient, EntityValidationError
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.entities import HelixJobAttempt, HelixJobStep, HelixJobTask, get_step_spec_name

_FINISHED = {"finished", "uploaded", "upload_failed"}
_UPLOADED = {"uploaded", "upload_failed"}
_PHASE_ORDER = {
    "pulling": 0,
    "ready": 1,
    "running": 2,
    "finished": 3,
    "uploaded": 4,
    "pull_failed": 4,
    "upload_failed": 4,
}


def execution_spec(attempt: HelixJobAttempt, step: HelixJobStep) -> HelixJobStepSpec:
    return attempt.get_step_spec(get_step_spec_name(step.config, step.name) or step.name)


def participants(spec: HelixJobStepSpec) -> int:
    if isinstance(spec.executor, DistributedGPUExecutionProvider):
        return spec.executor.resources.num_nodes
    return 1


def action(state: StepTransferState, progress: TaskTransferProgress) -> TransferAction:
    if state.phase == "aborted":
        return "abort"
    if state.phase == "done":
        return "done"
    if state.phase == "workload":
        if state.outcome is not None and progress.phase in {"ready", "running"}:
            return "stop"
        if progress.phase == "ready":
            return "run"
    if state.phase == "publishing" and progress.phase == "finished":
        return "upload_success" if state.outcome == "success" else "upload_failure"
    return "wait"


async def coordinate(
    store: EntityClient,
    files: AsyncFilesClient,
    attempt: HelixJobAttempt,
    step: HelixJobStep,
    tasks: list[HelixJobTask],
    *,
    backend_failed: bool = False,
    allow_commit: bool = True,
) -> HelixJobStep:
    spec = execution_spec(attempt, step)
    state = step.transfer_state.model_copy(deep=True) if step.transfer_state else StepTransferState()
    before = state.model_dump(mode="json")
    progress = [task.transfer for task in tasks if task.transfer is not None]
    if state.phase in {"done", "aborted"}:
        return step
    if any(p.phase == "pull_failed" for p in progress):
        state.phase = "aborted"
        state.errors = [p.error or "input pull failed" for p in progress if p.phase == "pull_failed"]
    elif backend_failed:
        state.lost = True
        if state.phase == "staging":
            state.phase = "aborted"
            state.errors = ["Task exited before input staging completed"]
        elif state.outcome is None:
            state.outcome = "failure"
    if state.phase == "staging" and len(progress) >= participants(spec) and all(p.phase == "ready" for p in progress):
        state.phase = "workload"
    if state.phase == "workload":
        finished = [p for p in progress if p.phase in _FINISHED]
        if any(p.exit_code != 0 and not p.stopped for p in finished):
            state.outcome = "failure"
        elif state.outcome is None and any(p.leader and p.exit_code == 0 for p in finished):
            state.outcome = "success"
        live = [task for task in tasks if not task.status.is_terminal()]
        all_finished = len(finished) >= participants(spec) or (
            state.lost and all(task.transfer is not None and task.transfer.phase in _FINISHED for task in live)
        )
        if all_finished:
            state.outcome = state.outcome or "success"
            state.phase = "publishing"
    if state.phase == "publishing":
        uploaders = [
            task
            for task in tasks
            if not task.status.is_terminal() or task.transfer is not None and task.transfer.phase in _UPLOADED
        ]
        finished_uploads = bool(uploaders) and all(
            task.transfer is not None and task.transfer.phase in _UPLOADED for task in uploaders
        )
        if state.lost and not uploaders:
            state.phase = "done"
            state.errors = ["All remaining uploaders or their storage were lost"]
        if finished_uploads:
            if state.lost:
                state.errors = ["Task storage or uploader was lost before publication completed"]
            destinations = {entry.fileset_id for entry in spec.filesets.push if entry.when in {"always", state.outcome}}
            pending = False
            for destination in sorted(destinations):
                if destination in state.commits:
                    continue
                results = [
                    result
                    for task in uploaders
                    if task.transfer is not None
                    for result in task.transfer.uploads
                    if result.fileset_id == destination
                ]
                if state.lost or len(results) != len(uploaders) or not all(result.success for result in results):
                    message = f"Publication failed for Fileset {destination}"
                    if message not in state.errors:
                        state.errors.append(message)
                    continue
                count = state.commit_attempts.get(destination, 0)
                if count >= 3:
                    message = f"Commit failed for Fileset {destination} after three attempts"
                    if message not in state.errors:
                        state.errors.append(message)
                    continue
                if not allow_commit:
                    pending = True
                    continue
                now = datetime.now(UTC)
                if destination in state.retry_at and state.retry_at[destination] > now:
                    pending = True
                    continue
                # Persist the attempt before remote I/O. A lost response can repeat the
                # commit, but a controller restart cannot reset its recorded budget.
                state.commit_attempts[destination] = count + 1
                step.transfer_state = state
                step = await store.update(step)
                try:
                    current = (await files.get_fileset_by_id(workspace=step.workspace, fileset_id=destination)).data()
                    if current.id != destination:
                        raise ValueError("Fileset metadata did not match the selected identity")
                    committed = (await files.commit_fileset(workspace=step.workspace, name="id:" + current.id)).data()
                    if committed.id != destination or not committed.commit_revision:
                        raise ValueError("Commit did not acknowledge the selected Fileset identity/revision")
                    state.commits[destination] = committed.commit_revision
                except HTTPClientError as exc:
                    if exc.status_code < 500 and exc.status_code not in {408, 409, 429}:
                        state.commit_attempts[destination] = 3
                        state.errors.append(
                            f"Commit denied or invalid for Fileset {destination}: HTTP {exc.status_code}"
                        )
                    else:
                        state.retry_at[destination] = now + timedelta(seconds=1 if count == 0 else 5)
                        pending = True
                except (ValueError, OSError, HTTPError):
                    state.retry_at[destination] = now + timedelta(seconds=1 if count == 0 else 5)
                    pending = True
            if not pending:
                state.phase = "done"
    if step.transfer_state is None or state.model_dump(mode="json") != before:
        step.transfer_state = state
        step.status_details = {**step.status_details, "fileset_transfers": state.model_dump(mode="json")}
        step = await store.update(step)
    return step


async def report(
    store: EntityClient,
    files: AsyncFilesClient,
    attempt: HelixJobAttempt,
    step: HelixJobStep,
    task_name: str,
    progress: TaskTransferProgress,
    tasks: list[HelixJobTask],
) -> HelixJobTask:
    spec = execution_spec(attempt, step)
    if not spec.filesets.enabled:
        raise EntityValidationError("Task transfer progress requires Step filesets configuration")
    allowed = {entry.fileset_id for entry in spec.filesets.push}
    if any(result.fileset_id not in allowed for result in progress.uploads):
        raise EntityValidationError("Upload result references an unconfigured Fileset")
    if len({result.fileset_id for result in progress.uploads}) != len(progress.uploads):
        raise EntityValidationError("Upload results must contain one result per destination")
    task = next((item for item in tasks if item.name == task_name), None)
    if step.status.is_terminal() or step.status in {HelixJobStatus.CANCELLING, HelixJobStatus.PAUSING}:
        if task is None:
            task = HelixJobTask(name=task_name, step_id=step.id, workspace=step.workspace)
        return task.model_copy(update={"transfer_action": "abort"})
    if task is not None and progress.runtime_id and progress.runtime_id in task.retired_transfer_runtimes:
        return task.model_copy(update={"transfer_action": "abort"})
    new_runtime = (
        progress.phase == "pulling"
        and progress.runtime_id
        and (task is None or task.transfer is None or progress.runtime_id != task.transfer.runtime_id)
    )
    if (
        new_runtime
        and step.transfer_state is not None
        and (
            step.transfer_state.phase in {"done", "aborted"}
            or step.transfer_state.lost
            or task is None
            and step.transfer_state.phase != "staging"
            or task is not None
            and task.transfer is not None
            and task.transfer.runtime_id != progress.runtime_id
        )
    ):
        # A backend-native restart runs a new launcher in the same logical Task
        # slot. Retire old reports before staging that new workload generation.
        for previous in tasks:
            if previous.transfer is not None and previous.transfer.runtime_id:
                previous.retired_transfer_runtimes.append(previous.transfer.runtime_id)
            previous.transfer = None
            previous.status = HelixJobStatus.ERROR
            await store.update(previous)
        step.transfer_state = None
        step = await store.update(step)
        if task is not None:
            task = await store.get_by_id(HelixJobTask, task.id)
    if task is not None and task.transfer is not None and progress.runtime_id != task.transfer.runtime_id:
        if progress.phase != "pulling" or step.transfer_state is not None and step.transfer_state.phase != "staging":
            return task.model_copy(update={"transfer_action": "abort"})
    if task is None:
        task = await store.create(
            HelixJobTask(
                name=task_name,
                step_id=step.id,
                workspace=step.workspace,
                status=HelixJobStatus.ACTIVE,
                transfer=progress,
            )
        )
        tasks.append(task)
    elif task.transfer is None or _PHASE_ORDER[progress.phase] >= _PHASE_ORDER[task.transfer.phase]:
        task.transfer = progress
        task.status = HelixJobStatus.ACTIVE
        task = await store.update(task)
        tasks = [task if item.id == task.id else item for item in tasks]
    refreshed = []
    for item in tasks:
        refreshed.append(await store.get_by_id(HelixJobTask, item.id))
    step = await coordinate(store, files, attempt, step, refreshed)
    assert step.transfer_state is not None
    assert task.transfer is not None
    return task.model_copy(update={"transfer_action": action(step.transfer_state, task.transfer)})
