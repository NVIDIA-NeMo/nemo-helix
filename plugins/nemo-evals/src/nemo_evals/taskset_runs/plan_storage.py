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
    """Downloaded bytes differ from the digest bound to the run at acceptance."""

    def __init__(self, plan_ref: str, expected: str, actual: str) -> None:
        self.plan_ref = plan_ref
        self.expected = expected
        self.actual = actual
        super().__init__(f"plan digest mismatch for {plan_ref!r}: expected {expected}, found {actual}")


@dataclass(frozen=True)
class StoredPlan:
    """Reference and SHA-256 of the exact uploaded bytes."""

    ref: str
    digest: str


async def store_plan(files_client: AsyncFilesClient, plan: TasksetRunPlan) -> StoredPlan:
    """Upload plan.json to a fresh run-specific fileset and return its ref/digest.

    Roll back a failed upload only after a successful create. A failed create,
    including a lost reply followed by 409, provides no ownership proof.
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
    """Download and verify raw bytes before parsing."""
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
    """Return a validated plan only after its downloaded bytes match the recorded digest."""
    body = await read_verified_bytes(files_client, plan_ref, expected_digest=expected_digest)
    return TasksetRunPlan.model_validate_json(body)


async def delete_fileset_ref(files_client: AsyncFilesClient, ref: str) -> None:
    """Best-effort deletion of a proven-unreferenced, request-owned fileset; log leaked refs."""
    try:
        workspace, name, _ = parse_fileset_ref(ref, workspace_fallback=None)
        await files_client.delete_fileset(workspace=workspace, name=name)
    except Exception:
        logger.warning("Failed to clean up fileset %s; storage may be leaked", ref, exc_info=True)
