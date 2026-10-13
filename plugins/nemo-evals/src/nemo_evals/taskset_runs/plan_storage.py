# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal Files operations; only TasksetRunStore owns this lifecycle."""

import logging
import secrets
from dataclasses import dataclass
from hashlib import sha256

from filesets import parse_fileset_ref
from nemo_evals.taskset_runs.specs import TasksetRunPlan
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest

logger = logging.getLogger(__name__)


class PlanStorageError(RuntimeError):
    """A plan write failed; its chained cause preserves the original Files error."""


class PlanDigestMismatchError(RuntimeError):
    """Reject bytes that differ from the plan accepted by the run.

    ``plan_ref`` identifies the downloaded file; ``expected`` and ``actual``
    retain its recorded and observed SHA-256 digests for integrity diagnostics.
    """

    def __init__(self, plan_ref: str, expected: str, actual: str) -> None:
        """Capture the file ref and expected/actual digests in a readable error.

        Args:
            plan_ref: Qualified file ref whose bytes failed verification.
            expected: SHA-256 recorded on the accepted run.
            actual: SHA-256 computed from the downloaded bytes.
        """
        self.plan_ref = plan_ref
        self.expected = expected
        self.actual = actual
        super().__init__(f"plan digest mismatch for {plan_ref!r}: expected {expected}, found {actual}")


@dataclass(frozen=True)
class StoredPlan:
    """Return the binding needed to attach uploaded bytes to a run record.

    This pair lets readers detect changes before parsing the stored plan.

    Attributes:
        ref: Workspace-qualified file ref of the uploaded plan
            (``workspace/<run_id>-plan-<6 hex>#plan.json``); becomes the run's ``plan_ref``.
        digest: SHA-256 hex of the exact uploaded bytes, not a later reserialization;
            becomes the run's ``plan_digest``.
    """

    ref: str
    digest: str


async def store_plan(files_client: AsyncFilesClient, plan: TasksetRunPlan) -> StoredPlan:
    """Upload plan.json to a fresh run-specific fileset and return its ref/digest.

    Roll back a failed upload only after a successful create. A failed create,
    including a lost reply followed by 409, provides no ownership proof.

    Args:
        files_client: Authorized Files client for the plan's workspace.
        plan: Resolved plan to serialize once and upload as UTF-8 JSON.

    Returns:
        Qualified file ref and SHA-256 of the uploaded bytes. A fresh fileset
        per attempt keeps concurrent losers from deleting a winner's plan.

    Raises:
        PlanStorageError: Fileset creation or upload failed, with the original
            error retained as its cause even when rollback also fails.
    """
    name = f"{plan.run_id}-plan-{secrets.token_hex(3)}"
    body = plan.model_dump_json().encode("utf-8")
    ref = f"{plan.workspace}/{name}#plan.json"
    try:
        await files_client.create_fileset(
            workspace=plan.workspace,
            body=CreateFilesetRequest(name=name, description=f"Frozen plan for {plan.workspace}/{plan.run_id}."),
            exist_ok=False,
        )
    except Exception as exc:
        raise PlanStorageError(f"failed to create plan fileset {plan.workspace}/{name}") from exc
    try:
        await files_client.upload_file(workspace=plan.workspace, name=name, path="plan.json", content=body)
    except Exception as exc:
        await delete_fileset_ref(files_client, ref)
        raise PlanStorageError(f"failed to upload plan to {ref}") from exc
    return StoredPlan(ref=ref, digest=sha256(body).hexdigest())


async def read_verified_bytes(files_client: AsyncFilesClient, plan_ref: str, *, expected_digest: str) -> bytes:
    """Verify stored bytes before parsing so changed JSON cannot pass as a plan.

    Args:
        files_client: Authorized client used to download the referenced file.
        plan_ref: Workspace-qualified fileset ref with a nonempty file fragment.
        expected_digest: SHA-256 bound to the run at acceptance.

    Returns:
        The downloaded bytes, only after their SHA-256 matches the recorded value.

    Raises:
        ValueError: The ref is malformed or lacks a nonempty file fragment.
        PlanDigestMismatchError: The downloaded bytes do not match the digest.
        Exception: Files download/read failures propagate without being rewrapped.
    """
    workspace, name, path = parse_fileset_ref(plan_ref, workspace_fallback=None)
    if "#" not in plan_ref or not path:
        raise ValueError(f"plan reference must include a nonempty file fragment: {plan_ref!r}")
    response = await files_client.download_file(workspace=workspace, name=name, path=path)
    body = await response.read()
    actual = sha256(body).hexdigest()
    if actual != expected_digest:
        raise PlanDigestMismatchError(plan_ref, expected_digest, actual)
    return body


async def load_plan(files_client: AsyncFilesClient, plan_ref: str, *, expected_digest: str) -> TasksetRunPlan:
    """Reconstruct a fresh plan from verified bytes without resolving inputs again.

    Args:
        files_client: Authorized client for the stored plan.
        plan_ref: Qualified file ref recorded on the run.
        expected_digest: SHA-256 recorded alongside that ref.

    Returns:
        A newly validated plan with independently reconstructed nested models.

    Raises:
        PlanDigestMismatchError: Integrity verification failed before JSON parsing.
        ValueError: The ref or plan JSON is invalid, including schema validation.
        Exception: Files download/read failures propagate.
    """
    body = await read_verified_bytes(files_client, plan_ref, expected_digest=expected_digest)
    return TasksetRunPlan.model_validate_json(body)


async def delete_fileset_ref(files_client: AsyncFilesClient, ref: str) -> None:
    """Clean up owned storage without replacing a request's original outcome.

    Args:
        files_client: Authorized client used to delete the fileset.
        ref: Qualified ref to a request-owned fileset proven safe to discard.
            Any file fragment is ignored: deletion removes the whole fileset.

    Returns:
        None. Malformed refs and ordinary deletion failures are logged as possible
        leaks; cancellation propagates. Callers must establish exclusive ownership
        because this helper does not check for references from winning runs.
    """
    try:
        workspace, name, _ = parse_fileset_ref(ref, workspace_fallback=None)
        await files_client.delete_fileset(workspace=workspace, name=name)
    except Exception:
        logger.warning("Failed to clean up fileset %s; storage may be leaked", ref, exc_info=True)
