# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real platform middleware and caller-scoped source checks for taskset run submission."""

from nemo_agents_plugin.service import AgentsService
from nemo_evals.service import EvaluatorPluginService
from nhx.core.entities.service import EntitiesService
from nhx.core.files.service import FilesService
from nhx.platform_runner.plugin_adapter import NemoServiceAdapter
from nhx.testing import ClientContext, create_test_client

PATH = "/apis/evals/v2/workspaces/destination/agent-evaluate/runs"


class TestEvalsService(NemoServiceAdapter):
    """Mount the real plugin routes in the platform authorization harness."""

    __test__ = False

    def __init__(self):
        super().__init__(EvaluatorPluginService())


class TestAgentsService(NemoServiceAdapter):
    """Register the agent source-read binding used by run submission."""

    __test__ = False

    def __init__(self):
        super().__init__(AgentsService())


def test_taskset_run_real_authorization_denies_create_service_and_source(tmp_path, request_body):
    with create_test_client(
        TestEvalsService,
        TestAgentsService,
        EntitiesService,
        FilesService,
        client_type=ClientContext,
        auth_enabled=True,
        tmp_dir=tmp_path,
        workspaces=["destination", "source"],
        projects=[],
        auth_bundle_cache_seconds=0,
        auth_policy_data_refresh_interval=3600,
        access_log=True,
    ) as ctx:
        admin = {"X-NHX-Principal-Id": "service:integration-test"}
        user = {"X-NHX-Principal-Id": "run-submitter@example.com"}
        denied = ctx.test_client.post(PATH, json=request_body, headers=user)
        assert denied.status_code == 403, denied.text
        direct_service = ctx.test_client.post(PATH, json=request_body, headers=admin)
        assert direct_service.status_code == 403, direct_service.text
        grant = ctx.test_client.post(
            "/apis/auth/v2/iam/role-bindings?wait_role_propagation=false",
            headers=admin,
            json={"principal": user["X-NHX-Principal-Id"], "workspace": "destination", "role": "Owner"},
        )
        assert grant.status_code == 200, grant.text
        request_body["spec"]["tasks"] = "source/suite"
        # The resolver's service identity has the wildcard grant; the caller's source
        # check must still reject the same cross-workspace read before any write.
        probe = ctx.test_client.get(
            "/apis/entities/v2/workspaces/source/entities/taskset/suite",
            headers={
                "X-NHX-Principal-Id": "service:evals",
                "X-NHX-Internal": "true",
            },
        )
        assert probe.status_code == 404, probe.text
        assert ctx.access_log is not None
        ctx.access_log.clear()
        source_denied = ctx.test_client.post(PATH, json=request_body, headers=user)
        assert source_denied.status_code == 403, source_denied.text
        assert not [
            entry
            for entry in ctx.access_log.requests
            if entry.method in {"POST", "PUT"} and ("/apis/files/" in entry.path or "/apis/entities/" in entry.path)
        ]
