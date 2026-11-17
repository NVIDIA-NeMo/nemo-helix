# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job-bound Evals operations: workloads never receive platform-service credentials."""

from abc import abstractmethod

from nemo_evals.entities import AgentEvalResultEntity, EvaluateResultEntity
from nemo_evals.gateway.tokens import TrialClaims
from nemo_evals.image_builds.models import TaskImageBuildEntity
from nemo_helix_plugin.auth.access_keys.types import AccessKeyCreateResponse, AccessKeyRevokeResponse
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.endpoint import delete, post
from nemo_helix_plugin.client.method import method

_PREFIX = "/apis/evals/v2/workspaces/{workspace}/job-runtime/{job}"


@post(_PREFIX + "/task-image-build")
@abstractmethod
def execute_build(*, workspace: str, job: str) -> TaskImageBuildEntity: ...


@post(_PREFIX + "/result")
@abstractmethod
def save_result(*, workspace: str, job: str, body: AgentEvalResultEntity | EvaluateResultEntity) -> None: ...


@post(_PREFIX + "/trial-keys")
@abstractmethod
def issue_trial_key(*, workspace: str, job: str, body: TrialClaims) -> AccessKeyCreateResponse: ...


@delete(_PREFIX + "/trial-keys/{jti}")
@abstractmethod
def revoke_trial_key(*, workspace: str, job: str, jti: str) -> AccessKeyRevokeResponse: ...


class EvalsRuntimeClient(AsyncNemoClient):
    execute_build = method(execute_build)
    save_result = method(save_result)
    issue_trial_key = method(issue_trial_key)
    revoke_trial_key = method(revoke_trial_key)
