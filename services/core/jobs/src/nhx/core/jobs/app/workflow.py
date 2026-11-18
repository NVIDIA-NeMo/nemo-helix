# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One controller evaluation of a workflow's immutable attempt."""

import hashlib
from datetime import datetime, timezone
from uuid import uuid4

from nemo_helix_plugin.client.errors import NemoClientError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.types import FilesetOutput
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nemo_helix_plugin.jobs.workflows import (
    WorkflowNodeState,
    WorkflowNodeView,
    WorkflowSignals,
    WorkflowState,
    WorkflowView,
    workflow_order,
)
from nhx.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError
from nhx.core.jobs.entities import STEP_SPEC_NAME_CONFIG_KEY, HelixJob, HelixJobAttempt, HelixJobStep

WORKFLOW_REQUESTS = "_workflow_requests"
WORKFLOW_CONTROL = "_workflow_control"


def _has_eligible_pending(attempt: HelixJobAttempt, state: WorkflowState) -> bool:
    return any(
        node.last_poll_error is None
        and node.requested_run_revision != node.last_attempted_revision
        and all(state.nodes[parent].freshness == "clean" for parent in spec.depends_on or [])
        for spec in attempt.platform_spec.steps
        for node in [state.nodes[spec.name]]
    )


def workflow_view(attempt: HelixJobAttempt) -> WorkflowView | None:
    config = attempt.platform_spec.workflow
    state = attempt.workflow_state
    if config is None or state is None:
        return None
    return WorkflowView(
        mode=config.mode,
        pass_number=state.pass_number,
        control=state.control,
        waiting_for_changes=config.mode == "continuous"
        and not attempt.status.is_terminal()
        and not state.pass_open
        and state.control == "running"
        and not _has_eligible_pending(attempt, state),
        degraded=any(node.last_poll_error or node.last_outcome == "error" for node in state.nodes.values()),
        nodes=[
            WorkflowNodeView(
                name=step.name,
                depends_on=step.depends_on or [],
                freshness=node.freshness,
                scheduling_state=node.scheduling_state,
                reason=node.reason,
                blocked_by=node.blocked_by,
                requested_run_revision=node.requested_run_revision,
                active_execution_id=node.active_step_id,
                latest_execution_id=node.latest_step_id,
                successful_execution_id=node.successful_step_id,
                last_outcome=node.last_outcome,
                has_pending_run=node.requested_run_revision != node.last_attempted_revision,
                last_inputs_checked_at=node.last_inputs_checked_at,
                last_poll_attempt_at=node.last_poll_attempt_at,
                last_poll_error=node.last_poll_error,
            )
            for step in attempt.platform_spec.steps
            for node in [state.nodes[step.name]]
        ],
    )


def workflow_step_held(attempt: HelixJobAttempt, step: HelixJobStep) -> bool:
    if attempt.platform_spec.workflow is None:
        return False
    state = attempt.workflow_state
    if state is None or state.control != "running" or attempt.status != HelixJobStatus.ACTIVE:
        return True
    name = step.config[STEP_SPEC_NAME_CONFIG_KEY]
    return state.nodes[name].active_step_id != step.id


async def reconcile_workflow(
    store: EntityClient, files: AsyncFilesClient, job: HelixJob, attempt: HelixJobAttempt
) -> HelixJobAttempt:
    config = attempt.platform_spec.workflow
    if config is None or attempt.status.is_terminal():
        return attempt
    state = attempt.workflow_state or WorkflowState(
        nodes={step.name: WorkflowNodeState() for step in attempt.platform_spec.steps}
    )
    request = attempt.status_details.get(WORKFLOW_CONTROL)
    if request == "pause":
        state.control = "pausing"
    elif request == "resume":
        state.control = "running"
        state.paused_at = None
    elif request == "cancel":
        state.control = "cancelling"
    attempt.status_details.pop(WORKFLOW_CONTROL, None)
    if state.control == "paused":
        return attempt

    now = datetime.now(timezone.utc)
    active: dict[str, HelixJobStep] = {}
    for name, node in state.nodes.items():
        if node.active_step_id is None:
            continue
        step = await store.get_by_id(HelixJobStep, node.active_step_id)
        if not step.status.is_terminal():
            active[name] = step
            continue
        node.active_step_id = None
        node.latest_step_id = step.id
        node.last_outcome = (
            "completed"
            if step.status == HelixJobStatus.COMPLETED
            else "error"
            if step.status == HelixJobStatus.ERROR
            else "cancelled"
        )
        node.scheduling_state = "idle"
        node.reason = None
        node.decision_pass = state.pass_number
        if step.status == HelixJobStatus.COMPLETED:
            node.successful_inputs = step.workflow_inputs
            node.successful_revision = step.workflow_revision
            node.successful_step_id = step.id
        elif config.mode == "run_to_completion" and step.status == HelixJobStatus.ERROR:
            state.control = "failing"

    if state.control in ("pausing", "cancelling", "failing"):
        running = False
        for name, step in active.items():
            node = state.nodes[name]
            if step.status in (HelixJobStatus.CREATED, HelixJobStatus.QUEUED):
                node.scheduling_state = "held"
                if state.control != "pausing":
                    step.status = HelixJobStatus.CANCELLED
                    await store.update(step)
                    node.active_step_id = None
                continue
            running = True
            if state.control == "cancelling" and step.status != HelixJobStatus.CANCELLING:
                step.status = HelixJobStatus.CANCELLING
                await store.update(step)
        if not running:
            if state.control == "pausing":
                state.control = "paused"
                state.paused_at = now
                attempt.status = HelixJobStatus.PAUSED
            else:
                attempt.status = HelixJobStatus.CANCELLED if state.control == "cancelling" else HelixJobStatus.ERROR
                state.stopped_at = now
        else:
            attempt.status = (
                HelixJobStatus.PAUSING
                if state.control == "pausing"
                else HelixJobStatus.CANCELLING
                if state.control == "cancelling"
                else HelixJobStatus.ACTIVE
            )
        attempt.workflow_state = state
        return await store.update(attempt)

    signals = WorkflowSignals.model_validate(attempt.status_details.get(WORKFLOW_REQUESTS, {})).root
    observations: dict[str, FilesetOutput] = {}
    for spec in attempt.platform_spec.steps:
        node = state.nodes[spec.name]
        signal = signals.get(spec.name)
        if signal is not None and signal != node.last_invalidation:
            node.last_invalidation = signal
            node.requested_run_revision = str(uuid4())
        node.last_poll_attempt_at = now
        vector: dict[str, str] = {}
        try:
            for reference in spec.inputs or []:
                if reference.fileset_id not in observations:
                    observations[reference.fileset_id] = (
                        await files.get_fileset_by_id(workspace=job.workspace, fileset_id=reference.fileset_id)
                    ).data()
                token = observations[reference.fileset_id].commit_revision
                if token is None:
                    raise ValueError(f"Input {reference.fileset_id} has no commit_revision")
                vector[reference.fileset_id] = token
        except (NemoClientError, ValueError) as error:
            node.last_poll_error = str(error)
            node.reason = "input_unavailable"
            continue
        if node.observed_inputs is not None and node.observed_inputs != vector:
            node.requested_run_revision = str(uuid4())
        node.observed_inputs = vector
        node.last_inputs_checked_at = now
        node.last_poll_error = None
        if node.reason == "input_unavailable":
            node.reason = None

    if not state.pass_open:
        if not _has_eligible_pending(attempt, state):
            attempt.workflow_state = state
            return await store.update(attempt)
        state.pass_number += 1
        state.pass_open = True

    # Persist requested revisions before creating executions so restart can adopt their stable keys.
    attempt.workflow_state = state
    attempt = await store.update(attempt)
    assert attempt.workflow_state is not None
    state = attempt.workflow_state

    order = workflow_order({step.name: step.depends_on or [] for step in attempt.platform_spec.steps})
    specs = {step.name: step for step in attempt.platform_spec.steps}
    for name in order:
        if state.control == "failing":
            break
        spec = specs[name]
        node = state.nodes[name]
        if name in active:
            step = active[name]
            if step.status in (HelixJobStatus.CREATED, HelixJobStatus.QUEUED):
                if step.workflow_revision != node.requested_run_revision:
                    step.status = HelixJobStatus.CANCELLED
                    await store.update(step)
                    node.active_step_id = None
                else:
                    node.scheduling_state = "waiting_capacity"
                    node.reason = "awaiting_dispatch"
                    continue
            else:
                node.scheduling_state = "running"
                node.reason = None
                continue
        if node.decision_pass == state.pass_number:
            continue
        parents = [state.nodes[parent] for parent in spec.depends_on or []]
        node.blocked_by = [
            parent
            for parent in spec.depends_on or []
            if state.nodes[parent].decision_pass != state.pass_number or state.nodes[parent].last_outcome != "completed"
        ]
        if node.blocked_by:
            node.scheduling_state = "waiting_dependencies"
            node.reason = "dependencies"
            if all(parent.decision_pass == state.pass_number for parent in parents):
                node.decision_pass = state.pass_number
            continue
        if node.last_poll_error:
            node.scheduling_state = "idle"
            node.reason = "input_unavailable"
            node.decision_pass = state.pass_number
            if config.mode == "run_to_completion":
                state.control = "failing"
            continue
        if node.freshness == "clean":
            node.decision_pass = state.pass_number
            node.last_outcome = "completed"
            node.scheduling_state = "idle"
            node.reason = "unchanged_inputs"
            continue
        if node.last_attempted_revision == node.requested_run_revision:
            node.decision_pass = state.pass_number
            node.scheduling_state = "idle"
            node.reason = "awaiting_invalidation"
            continue
        key = "wf-" + hashlib.sha256(f"{attempt.id}/{name}/{node.requested_run_revision}".encode()).hexdigest()[:24]
        try:
            step = await store.get(HelixJobStep, key, workspace=job.workspace, parent=attempt.id)
        except EntityNotFoundError:
            try:
                step = await store.create(
                    HelixJobStep(
                        name=key,
                        workspace=job.workspace,
                        attempt_id=attempt.id,
                        config={**spec.config, STEP_SPEC_NAME_CONFIG_KEY: name},
                        workflow_revision=node.requested_run_revision,
                        workflow_inputs=node.observed_inputs,
                    )
                )
            except EntityConflictError:
                step = await store.get(HelixJobStep, key, workspace=job.workspace, parent=attempt.id)
        node.active_step_id = step.id
        node.latest_step_id = step.id
        node.last_attempted_revision = node.requested_run_revision
        node.scheduling_state = "ready"
        node.reason = "changed_inputs" if node.successful_revision else "initial_run"

    if all(node.decision_pass == state.pass_number for node in state.nodes.values()):
        state.pass_open = False
        if config.mode == "run_to_completion":
            attempt.status = (
                HelixJobStatus.ERROR
                if state.control == "failing" or any(node.last_outcome == "error" for node in state.nodes.values())
                else HelixJobStatus.COMPLETED
            )
            state.stopped_at = now
    if not attempt.status.is_terminal():
        attempt.status = HelixJobStatus.ACTIVE
    attempt.workflow_state = state
    return await store.update(attempt)
