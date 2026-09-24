# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import time
import uuid

import httpx
import pytest

from tests.auth_idp.common import (
    managed_workload_workspace_get_command,
    nhx_api_image,
    require_capability,
    runtime_tls_config,
)
from tests.auth_idp.helpers import grant_workspace_role
from tests.auth_idp.runtime_contract import AuthIdpRuntime, JsonObject

pytestmark = [
    pytest.mark.auth_idp,
    pytest.mark.auth_idp_runtime,
    pytest.mark.e2e,
]

DEPLOYMENTS_REQUEST_TIMEOUT_SECONDS = 10.0
DEPLOYMENTS_WAIT_TIMEOUT_SECONDS = 240.0
DEPLOYMENTS_DELETE_TIMEOUT_SECONDS = 30.0
DEPLOYMENTS_WAIT_SLEEP_SECONDS = 2.0
DEPLOYMENT_FAILURE_STATUSES = {"FAILED", "LOST"}


def _deployment_api_base(auth_idp_runtime: AuthIdpRuntime, workspace: str) -> str:
    return f"{auth_idp_runtime.gateway_base_url}/apis/deployments/v2/workspaces/{workspace}"


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _request_json(
    method: str,
    url: str,
    *,
    token: str,
    auth_idp_runtime: AuthIdpRuntime,
    json_body: JsonObject | None = None,
) -> httpx.Response:
    tls_config = runtime_tls_config(auth_idp_runtime)
    response = httpx.request(
        method,
        url,
        json=json_body,
        headers=_headers(token),
        timeout=DEPLOYMENTS_REQUEST_TIMEOUT_SECONDS,
        **tls_config,
    )
    if not 200 <= response.status_code < 300:
        raise AssertionError(f"{method} {url} returned {response.status_code}: {response.text}")
    return response


def _delete_best_effort(auth_idp_runtime: AuthIdpRuntime, url: str, *, token: str) -> None:
    tls_config = runtime_tls_config(auth_idp_runtime)
    with contextlib.suppress(Exception):
        httpx.delete(
            url,
            headers=_headers(token),
            timeout=DEPLOYMENTS_REQUEST_TIMEOUT_SECONDS,
            **tls_config,
        )


def _wait_for_deployment_status(
    auth_idp_runtime: AuthIdpRuntime,
    *,
    workspace: str,
    deployment_name: str,
    token: str,
    timeout: float = DEPLOYMENTS_WAIT_TIMEOUT_SECONDS,
) -> JsonObject:
    url = f"{_deployment_api_base(auth_idp_runtime, workspace)}/deployments/{deployment_name}"
    deadline = time.monotonic() + timeout
    tls_config = runtime_tls_config(auth_idp_runtime)
    last_payload: JsonObject | None = None
    last_text = ""

    while time.monotonic() < deadline:
        response = httpx.get(
            url,
            headers=_headers(token),
            timeout=DEPLOYMENTS_REQUEST_TIMEOUT_SECONDS,
            **tls_config,
        )
        if response.status_code == 404:
            last_text = response.text
            time.sleep(DEPLOYMENTS_WAIT_SLEEP_SECONDS)
            continue
        if not 200 <= response.status_code < 300:
            raise AssertionError(f"GET {url} returned {response.status_code}: {response.text}")

        payload = response.json()
        last_payload = payload
        status = payload.get("status")
        if status == "SUCCEEDED":
            return payload
        if status in DEPLOYMENT_FAILURE_STATUSES:
            raise AssertionError(
                "Deployment failed while verifying workload token exchange: "
                f"status={status!r} exit_code={payload.get('exit_code')!r} "
                f"message={payload.get('status_message')!r} details={payload.get('error_details')!r}"
            )
        time.sleep(DEPLOYMENTS_WAIT_SLEEP_SECONDS)

    raise AssertionError(
        "Timed out waiting for deployment workload token exchange verification to succeed: "
        f"last_payload={last_payload!r} last_text={last_text!r}"
    )


def _wait_for_deployment_deleted(
    auth_idp_runtime: AuthIdpRuntime, *, workspace: str, deployment_name: str, token: str
) -> None:
    url = f"{_deployment_api_base(auth_idp_runtime, workspace)}/deployments/{deployment_name}"
    _wait_for_resource_deleted(
        auth_idp_runtime,
        url,
        token=token,
        resource_description=f"Deployment {deployment_name}",
    )


def _wait_for_resource_deleted(
    auth_idp_runtime: AuthIdpRuntime,
    url: str,
    *,
    token: str,
    resource_description: str,
) -> None:
    deadline = time.monotonic() + DEPLOYMENTS_DELETE_TIMEOUT_SECONDS
    tls_config = runtime_tls_config(auth_idp_runtime)
    while time.monotonic() < deadline:
        response = httpx.get(
            url,
            headers=_headers(token),
            timeout=DEPLOYMENTS_REQUEST_TIMEOUT_SECONDS,
            **tls_config,
        )
        if response.status_code == 404:
            return
        time.sleep(DEPLOYMENTS_WAIT_SLEEP_SECONDS)

    raise AssertionError(
        f"{resource_description} was not deleted within {DEPLOYMENTS_DELETE_TIMEOUT_SECONDS} seconds: {url}"
    )


def test_provider_workload_deployment_runs_with_managed_obo(
    auth_idp_case,
    auth_idp_runtime,
    auth_idp_workspace,
):
    require_capability(auth_idp_case, "workspace_rbac")
    require_capability(auth_idp_case, "workload_deployment")
    require_capability(auth_idp_case, "managed_workload_deployment_obo")

    e2e_setup_client = auth_idp_runtime.e2e_setup_client()
    for principal in auth_idp_runtime.workload_role_principals():
        grant_workspace_role(
            e2e_setup_client,
            workspace=auth_idp_workspace,
            principal=principal,
            roles=["Viewer", "Editor", "JobRunner"],
        )

    suffix = uuid.uuid4().hex[:8]
    config_name = f"workload-dep-cfg-{suffix}"
    deployment_name = f"workload-dep-{suffix}"
    base = _deployment_api_base(auth_idp_runtime, auth_idp_workspace)
    workload_provider_token = auth_idp_runtime.workload_platform_token().access_token
    setup_token = auth_idp_runtime.e2e_setup_token().access_token
    runtime_config = auth_idp_runtime.deployment_workload_runtime_config()
    deployment_config_body: JsonObject = {
        "name": config_name,
        "restartPolicy": "Never",
        "containers": [
            {
                "name": "workload-workspace-get",
                "image": nhx_api_image(),
                "command": ["sh", "-c"],
                "args": [managed_workload_workspace_get_command(task_config={"workspace": auth_idp_workspace})],
                "env": [dict(item) for item in runtime_config.env],
            }
        ],
        "workloadIdentity": {
            "enabled": True,
            "workloadKind": "auth_idp_deployment_e2e",
            "workloadId": deployment_name,
            "tokenExpirationSeconds": 900,
        },
    }
    if runtime_config.config_files:
        deployment_config_body["configFiles"] = [dict(item) for item in runtime_config.config_files]

    try:
        _request_json(
            "POST",
            f"{base}/deployment-configs",
            token=workload_provider_token,
            auth_idp_runtime=auth_idp_runtime,
            json_body=deployment_config_body,
        )
        _request_json(
            "POST",
            f"{base}/deployments",
            token=workload_provider_token,
            auth_idp_runtime=auth_idp_runtime,
            json_body={
                "name": deployment_name,
                "deployment_config": config_name,
                "desired_state": "READY",
            },
        )

        completed = _wait_for_deployment_status(
            auth_idp_runtime,
            workspace=auth_idp_workspace,
            deployment_name=deployment_name,
            token=setup_token,
        )

        assert completed["status"] == "SUCCEEDED"
        assert completed["exit_code"] == 0
    finally:
        _delete_best_effort(auth_idp_runtime, f"{base}/deployments/{deployment_name}", token=setup_token)
        _wait_for_deployment_deleted(
            auth_idp_runtime,
            workspace=auth_idp_workspace,
            deployment_name=deployment_name,
            token=setup_token,
        )
        config_url = f"{base}/deployment-configs/{config_name}"
        _delete_best_effort(auth_idp_runtime, config_url, token=setup_token)
        _wait_for_resource_deleted(
            auth_idp_runtime,
            config_url,
            token=setup_token,
            resource_description=f"DeploymentConfig {config_name}",
        )
