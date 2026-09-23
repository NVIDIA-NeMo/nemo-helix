# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for task-side auth propagation via ``NHX_PRINCIPAL``.

These tests cover the runtime half of the jobs auth propagation story:

- the task receives ``NHX_PRINCIPAL``
- ``get_task_sdk(as_service=...)`` converts that into service + on-behalf-of headers
- downstream services authorize based on the delegated user's permissions
"""

from __future__ import annotations

import json
import os
from typing import Protocol

import pytest
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest
from nhx.core.secrets.service import SecretsService
from nhx.testing import (
    TEST_ADMIN_EMAIL,
    ClientContext,
    SDKTestClientAdapter,
    as_user,
    create_test_client,
    grant_workspace_role,
    short_unique_name,
    unique_email,
)
from pydantic import SecretStr


class _SecretAccessTask(Protocol):
    def run(self, *, http_client) -> str: ...


def _secret_access_task_module() -> _SecretAccessTask:
    class _Task:
        @staticmethod
        def run(*, http_client) -> str:
            from nhx.common.sdk_factory import get_task_sdk

            workspace = os.environ["NEMO_JOB_WORKSPACE"]
            secret_name = os.environ["NEMO_TEST_SECRET_NAME"]

            task_sdk = get_task_sdk(as_service="jobs", http_client=http_client)
            result = client_from_platform(task_sdk, SecretsClient).access_secret(
                name=secret_name,
                workspace=workspace,
            )
            return result.data().value

    return _Task()


class TestTaskRuntimeAuthPropagation:
    def test_task_sdk_accesses_secret_on_behalf_of_creator(self):
        workspace = short_unique_name("task-obo")
        secret_name = short_unique_name("secret")
        secret_value = "task-visible-secret"
        creator_email = unique_email("creator")

        with create_test_client(
            SecretsService,
            auth_enabled=True,
            access_log=True,
            client_type=ClientContext,
            workspaces=[workspace],
        ) as ctx:
            admin_sdk = as_user(ctx.sdk, TEST_ADMIN_EMAIL)
            client_from_platform(admin_sdk, SecretsClient).create_secret(
                body=HelixSecretCreateRequest(name=secret_name, value=SecretStr(secret_value)),
                workspace=workspace,
            )
            grant_workspace_role(
                admin_sdk,
                workspace=workspace,
                principal=creator_email,
                roles=["Viewer"],
            )

            ctx.access_log.clear()
            with (
                pytest.MonkeyPatch.context() as monkeypatch,
            ):
                monkeypatch.setenv("NEMO_JOB_WORKSPACE", workspace)
                monkeypatch.setenv("NEMO_TEST_SECRET_NAME", secret_name)
                monkeypatch.setenv(
                    "NHX_PRINCIPAL",
                    json.dumps(
                        {
                            "id": creator_email,
                            "email": creator_email,
                            "groups": [],
                        }
                    ),
                )
                secret = _secret_access_task_module().run(http_client=SDKTestClientAdapter(ctx.test_client))

            assert secret == secret_value

            request = ctx.access_log.assert_has_request(
                method="GET",
                path_contains=f"/apis/secrets/v2/workspaces/{workspace}/secrets/{secret_name}/access",
                principal_id="service:jobs",
            )
            assert request.on_behalf_of == creator_email

    def test_task_sdk_denies_secret_access_when_creator_lacks_permission(self):
        workspace = short_unique_name("task-deny")
        secret_name = short_unique_name("secret")
        creator_email = unique_email("creator")

        with create_test_client(
            SecretsService,
            auth_enabled=True,
            access_log=True,
            client_type=ClientContext,
            workspaces=[workspace],
        ) as ctx:
            admin_sdk = as_user(ctx.sdk, TEST_ADMIN_EMAIL)
            client_from_platform(admin_sdk, SecretsClient).create_secret(
                body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("secret-value")),
                workspace=workspace,
            )

            ctx.access_log.clear()
            with (
                pytest.MonkeyPatch.context() as monkeypatch,
                pytest.raises(PermissionDeniedError),
            ):
                monkeypatch.setenv("NEMO_JOB_WORKSPACE", workspace)
                monkeypatch.setenv("NEMO_TEST_SECRET_NAME", secret_name)
                monkeypatch.setenv(
                    "NHX_PRINCIPAL",
                    json.dumps(
                        {
                            "id": creator_email,
                            "email": creator_email,
                            "groups": [],
                        }
                    ),
                )
                _secret_access_task_module().run(http_client=SDKTestClientAdapter(ctx.test_client))

            request = ctx.access_log.assert_has_request(
                method="GET",
                path_contains=f"/apis/secrets/v2/workspaces/{workspace}/secrets/{secret_name}/access",
                principal_id="service:jobs",
            )
            assert request.on_behalf_of == creator_email
