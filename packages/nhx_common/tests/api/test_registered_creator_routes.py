# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard the registered creator-snapshot response surfaces."""

import pytest
from fastapi import APIRouter, FastAPI
from nemo_agents_plugin.api.v2 import deployments as agent_deployments
from nemo_deployments_plugin.api.v2 import deployments as plugin_deployments
from nemo_deployments_plugin.api.v2 import status as plugin_status
from nhx.common.api.creator_snapshot_projection import collect_creator_snapshot_routes
from nhx.core.jobs.api.v2.jobs import endpoints as jobs
from nhx.core.models.api.v2 import deployments as model_deployments
from nhx.core.models.api.v2 import providers as model_providers


@pytest.mark.parametrize(
    ("router", "expected_count"),
    [
        (model_providers.router, 5),
        (model_deployments.router, 7),
        (jobs.router, 2),
        (agent_deployments.router, 3),
        (plugin_deployments.router, 3),
        (plugin_status.router, 1),
    ],
)
def test_known_creator_response_routes_are_registered(router: APIRouter, expected_count: int) -> None:
    app = FastAPI()
    app.include_router(router, prefix="/apis/test")

    registrations = collect_creator_snapshot_routes(app)

    assert len(registrations) == expected_count
    assert all(registration.path_regex.pattern.startswith("^/apis/test/") for registration in registrations)
