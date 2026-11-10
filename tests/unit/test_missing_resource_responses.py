# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Missing-resource response contracts (NVBug 7074797), without a running platform."""

import importlib
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute

ROUTES = (
    [
        ("nhx.core.entities.api.v2.workspaces.endpoints", "router", name)
        for name in (
            "get_workspace",
            "update_workspace",
            "delete_workspace",
            "get_workspace_deletion",
            "update_workspace_member",
            "remove_workspace_member",
        )
    ]
    + [
        ("nhx.core.entities.api.v2.entities.endpoints", "router", name)
        for name in ("get_entity_by_name", "update_entity_by_name", "delete_entity_by_name", "get_entity_by_id")
    ]
    + [("nhx.core.entities.api.v2.projects.endpoints", "router", name) for name in ("update_project", "delete_project")]
    + [
        (f"nhx.core.models.api.v2.{module}", router, name)
        for module, router, names in (
            ("models", "router", ("update_model", "delete_model")),
            ("adapters", "router", ("update_adapter", "delete_adapter")),
            ("prompts", "router", ("update_prompt", "delete_prompt")),
            ("providers", "router", ("delete_provider",)),
            ("deployment_configs", "router", ("get_deployment_config_version",)),
            ("deployments", "router", ("get_deployment_version",)),
        )
        for name in names
    ]
    + [
        ("nhx.core.files.api.v2.filesets.endpoints", "router", name)
        for name in ("delete_fileset", "update_fileset_metadata")
    ]
    + [("nhx.core.secrets.api.v2.secrets.endpoints", "router", name) for name in ("update_secret", "delete_secret")]
    + [("nhx.core.auth.api.v2.iam.endpoints", "router", name) for name in ("get_role_binding", "revoke_role_binding")]
    + [
        (f"nhx.intake.spans.api.{module}", "router", name)
        for module, name in (
            ("traces", "get_trace"),
            ("spans", "get_span"),
            ("evaluator_results", "get_evaluator_result"),
        )
    ]
    + [
        ("nhx.core.jobs.api.v2.jobs.endpoints", "router", name)
        for name in ("get_attempt_step", "get_attempt_step_task", "update_attempt_step_task")
    ]
)


@pytest.mark.parametrize(("module", "router_name", "endpoint"), ROUTES, ids=[r[2] for r in ROUTES])
def test_missing_resource_is_documented(module, router_name, endpoint):
    router = getattr(importlib.import_module(module), router_name)
    route = next(r for r in router.routes if isinstance(r, APIRoute) and r.endpoint.__name__ == endpoint)
    app = FastAPI()
    app.router.routes.append(route)
    for method in route.methods:
        responses = app.openapi()["paths"][route.path][method.lower()]["responses"]
        assert responses["404"]["description"]
        # Existing validation and success declarations must survive the addition.
        assert "422" in responses
        assert str(route.status_code or 200) in responses
        if module != "nhx.core.jobs.api.v2.jobs.endpoints":
            assert responses["404"]["content"]["application/json"]["schema"]["$ref"].endswith("/ErrorResponse")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint", ["get_workspace", "update_workspace", "delete_workspace", "get_workspace_deletion"]
)
async def test_workspace_missing_returns_404(endpoint):
    from nhx.core.entities.api.v2.workspaces import endpoints

    repository = AsyncMock()
    repository.get_workspace_by_name.return_value = None
    repository.update_workspace.return_value = None
    kwargs = {"name": "missing", "repository": repository}
    if endpoint == "update_workspace":
        kwargs.update(workspace_data=Mock(description="updated"), auth_client=Mock())
    elif endpoint == "delete_workspace":
        kwargs["entity_repository"] = AsyncMock()
    elif endpoint == "get_workspace_deletion":
        kwargs["auth_client"] = Mock()
    with pytest.raises(HTTPException) as exc:
        await getattr(endpoints, endpoint)(**kwargs)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["get_attempt_step", "get_attempt_step_task", "update_attempt_step_task"])
async def test_attempt_route_missing_parent_returns_404(endpoint):
    from nhx.core.jobs.api.v2.jobs import endpoints

    dispatcher = AsyncMock()
    dispatcher.get_attempt_step_by_name.return_value = None
    dispatcher.resolve_attempt.return_value = None
    kwargs = dict(job="job", attempt="attempt", name="missing", workspace="default", dispatcher=dispatcher)
    if endpoint != "get_attempt_step":
        kwargs["step"] = "missing-step"
    if endpoint == "update_attempt_step_task":
        kwargs["update"] = Mock()
    with pytest.raises(HTTPException) as exc:
        await getattr(endpoints, endpoint)(**kwargs)
    assert exc.value.status_code == 404
    dispatcher.create_or_update_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_put_upserts_task_when_parent_exists():
    from nhx.core.jobs.api.v2.jobs import endpoints

    dispatcher = AsyncMock()
    job_entity = Mock(name="job")
    step_entity = Mock()
    dispatcher.resolve_attempt.return_value = (job_entity, Mock())
    dispatcher.get_attempt_step_by_name.return_value = step_entity
    update = Mock()
    result = await endpoints.update_attempt_step_task(
        job="job",
        attempt="attempt",
        step="step",
        name="new-task",
        workspace="default",
        update=update,
        dispatcher=dispatcher,
    )
    assert result is dispatcher.create_or_update_task.return_value
    dispatcher.create_or_update_task.assert_awaited_once_with(
        job_entity.name,
        "new-task",
        "default",
        update,
        step_entity,
    )


@pytest.mark.asyncio
async def test_provider_put_upserts_missing_provider_without_404():
    from nhx.common.auth.models import Principal
    from nhx.core.models.api.v2 import providers

    service = AsyncMock()
    provider = Mock()
    service.upsert_model_provider.return_value = provider
    request = Mock(api_key_secret_name=None, model_deployment_id=None)
    result = await providers.upsert_provider(
        workspace="default",
        name="new",
        request=request,
        service=service,
        auth_client=Mock(principal=Principal(id="test")),
        secrets=AsyncMock(),
    )
    assert result is provider
    route = next(r for r in providers.router.routes if isinstance(r, APIRoute) and r.name == "upsert_provider")
    assert 404 not in route.responses


@pytest.mark.asyncio
async def test_missing_authorization_transaction_is_400_not_404():
    from nhx.core.auth.oidc_broker import routes

    store = AsyncMock()
    store.get.return_value = None
    with pytest.raises(HTTPException) as exc:
        await routes.continue_token_authorization("missing", store)
    assert exc.value.status_code == 400
    route = next(
        r for r in routes.router.routes if isinstance(r, APIRoute) and r.endpoint is routes.continue_token_authorization
    )
    assert 404 not in route.responses
