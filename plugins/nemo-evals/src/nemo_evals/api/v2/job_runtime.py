# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service-owned operations for one authenticated native Evals Job.

The signed workload binding comes from the server-side delegation record. A user,
non-Job workload, or another Job cannot select the operation's Job or workspace.
The service owns storage and trial attributes; outbound business calls still
represent the Job creator and retain that creator's authorization checks.
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Response
from nemo_evals.authz import scope
from nemo_evals.entities import AgentEvalResultEntity, EvaluateResultEntity
from nemo_evals.gateway.tokens import TOKEN_LIFETIME_SEC, TrialClaims
from nemo_evals.image_builds.builder import HttpImageBuilder
from nemo_evals.image_builds.models import TASK_IMAGE_BUILD_JOB_SOURCE, TaskImageBuildEntity
from nemo_evals.image_builds.runner import TaskImageBuildSpec, run_task_image_build
from nemo_helix_plugin.auth import (
    RuntimeAuthClient,
    RuntimePrincipal,
    current_auth_client,
    current_auth_context,
    current_workload_token_claims,
)
from nemo_helix_plugin.auth.access_keys.client import AsyncAccessKeysClient
from nemo_helix_plugin.auth.access_keys.types import (
    AccessKeyCreateRequest,
    AccessKeyCreateResponse,
    AccessKeyRevokeResponse,
)
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NemoHTTPError
from nemo_helix_plugin.client_provider import get_async_nemo_client
from nemo_helix_plugin.client_provider import get_nemo_client as sync_service_client
from nemo_helix_plugin.dependencies import get_nemo_client
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.types import HelixJobResponse

router = APIRouter()
_PREFIX = "/job-runtime/{job}"
_AGENT_SOURCE = "nemo-evals.agent-evaluate"
_ROW_SOURCE = "nemo-evals"


async def _job(
    workspace: str, job: str, auth: RuntimeAuthClient, client: AsyncNemoClient, sources: set[str]
) -> HelixJobResponse:
    if auth.auth_enabled:
        claims = current_workload_token_claims()
        binding = claims.get("nhx_workload") if claims is not None else None
        if (
            claims is None
            or not auth.principal.is_delegated
            or not isinstance(binding, dict)
            or binding.get("workspace") != workspace
            or binding.get("job") != job
        ):
            raise HTTPException(403, "A workload credential bound to this Job is required")
    current = (await AsyncJobsClient.from_client(client).get_job(workspace=workspace, name=job)).data()
    if current.source not in sources:
        raise HTTPException(403, "The Job does not own this Evals operation")
    return current


@router.post(_PREFIX + "/task-image-build", response_model=TaskImageBuildEntity)
@scope.write
@path_rule(callers=[CallerKind.SERVICE_PRINCIPAL], permissions=[])
async def execute_task_build(
    workspace: str,
    job: str,
    auth: RuntimeAuthClient = Depends(current_auth_client),
    client: AsyncNemoClient = Depends(get_nemo_client),
) -> TaskImageBuildEntity:
    current = await _job(workspace, job, auth, client, {TASK_IMAGE_BUILD_JOB_SOURCE})
    spec = TaskImageBuildSpec.model_validate(current.spec)
    if spec.workspace != workspace:
        raise HTTPException(403, "Build workspace differs from its Job")
    async with get_async_nemo_client(as_service="evals", internal=True) as storage:
        return await run_task_image_build(
            spec,
            entities=EntityClient(AsyncEntitiesClient.from_client(storage)),
            files=AsyncFilesClient.from_client(client),
            builder=HttpImageBuilder(client, workspace=workspace),
        )


@router.post(_PREFIX + "/result", status_code=204)
@scope.write
@path_rule(callers=[CallerKind.SERVICE_PRINCIPAL], permissions=[])
async def save_result(
    workspace: str,
    job: str,
    body: AgentEvalResultEntity | EvaluateResultEntity,
    auth: RuntimeAuthClient = Depends(current_auth_client),
    client: AsyncNemoClient = Depends(get_nemo_client),
) -> Response:
    current = await _job(workspace, job, auth, client, {_ROW_SOURCE, _AGENT_SOURCE, "nemo-evals.retrieve-eval"})
    expected = AgentEvalResultEntity if current.source == _AGENT_SOURCE else EvaluateResultEntity
    if not isinstance(body, expected) or body.workspace != workspace or body.name != job or body.job_id != job:
        raise HTTPException(403, "The result must belong to the producing Job")
    async with get_async_nemo_client(as_service="evals", internal=True) as storage:
        await EntityClient(AsyncEntitiesClient.from_client(storage)).save(body)
    return Response(status_code=204)


def _creator() -> RuntimePrincipal:
    context = current_auth_context()
    if context is None:
        raise HTTPException(403, "An authenticated Job creator is required")
    return context.to_principal()


def _trial_policy(current: HelixJobResponse, auth: RuntimeAuthClient, claims: TrialClaims):
    # Use the same route/model discovery as the runner; never trust the
    # workload to widen the persisted Job's model or closed-book policy.
    from nemo_evals.jobs.agent_evaluate import _trial_model_policy
    from nemo_evals.jobs.agent_spec import AgentEvalSpec

    with sync_service_client(as_service="evals", internal=True, on_behalf_of=_creator()) as client:
        policy = _trial_model_policy(AgentEvalSpec.model_validate(current.spec), client)
    return claims.model_copy(update={"model_access": policy.model_access, "models": policy.models})


@router.post(_PREFIX + "/trial-keys", response_model=AccessKeyCreateResponse)
@scope.write
@path_rule(callers=[CallerKind.SERVICE_PRINCIPAL], permissions=[])
async def issue_trial_key(
    workspace: str,
    job: str,
    body: TrialClaims,
    auth: RuntimeAuthClient = Depends(current_auth_client),
    client: AsyncNemoClient = Depends(get_nemo_client),
) -> AccessKeyCreateResponse:
    current = await _job(workspace, job, auth, client, {_AGENT_SOURCE})
    if body.intake_workspace != workspace:
        raise HTTPException(403, "Trial workspace differs from its Job")
    claims = await asyncio.to_thread(_trial_policy, current, auth, body)
    async with get_async_nemo_client(as_service="evals", internal=True, on_behalf_of=_creator()) as issuer:
        try:
            return (
                await AsyncAccessKeysClient.from_client(issuer).create_access_key(
                    body=AccessKeyCreateRequest(
                        name=f"nemo-evals-trial {claims.session}"[:128],
                        description="Inference Gateway key for one evaluation trial; revoked when the trial ends.",
                        expires_in_seconds=TOKEN_LIFETIME_SEC,
                        scope=["inference"],
                        service_attributes=claims.service_attributes(),
                    )
                )
            ).data()
        except NemoHTTPError as exc:
            raise HTTPException(exc.status_code, "Trial credential issuance was refused") from exc


@router.delete(_PREFIX + "/trial-keys/{jti}", response_model=AccessKeyRevokeResponse)
@scope.write
@path_rule(callers=[CallerKind.SERVICE_PRINCIPAL], permissions=[])
async def revoke_trial_key(
    workspace: str,
    job: str,
    jti: str,
    auth: RuntimeAuthClient = Depends(current_auth_client),
    client: AsyncNemoClient = Depends(get_nemo_client),
) -> AccessKeyRevokeResponse:
    await _job(workspace, job, auth, client, {_AGENT_SOURCE})
    async with get_async_nemo_client(as_service="evals", internal=True, on_behalf_of=_creator()) as issuer:
        return (await AsyncAccessKeysClient.from_client(issuer).revoke_access_key(jti=jti)).data()
