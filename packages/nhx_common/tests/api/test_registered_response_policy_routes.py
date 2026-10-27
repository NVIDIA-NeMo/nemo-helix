# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard the registered auth-context response surfaces."""

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.routing import iter_route_contexts
from nemo_agents_plugin.api.v2 import deployments as agent_deployments
from nemo_deployments_plugin.api.v2 import deployments as plugin_deployments
from nemo_deployments_plugin.api.v2 import status as plugin_status
from nemo_helix_plugin.api.response_policy import PolicyRoute
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
def test_known_auth_context_response_routes_are_registered(router: APIRouter, expected_count: int) -> None:
    app = FastAPI()
    app.include_router(router, prefix="/apis/test")

    registrations = [
        context for context in iter_route_contexts(app.router.routes) if isinstance(context.original_route, PolicyRoute)
    ]

    assert len(registrations) == expected_count
    assert all(registration.path.startswith("/apis/test/") for registration in registrations)
