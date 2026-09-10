# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for persisted agent-session lifecycle helpers."""

from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemo_agents_plugin.entities import AgentDeployment, AgentSession, Endpoint, SessionStatus
from nemo_agents_plugin.session_lifecycle import cleanup_fabric_runtime, session_expiration_is_due
from nemo_deployments_plugin.endpoint_transport import EndpointTransport


@pytest.mark.parametrize(
    ("at", "expected"),
    [
        (datetime(2026, 8, 31, 16, 59, tzinfo=UTC), False),
        (datetime(2026, 8, 31, 17, 0, tzinfo=UTC), True),
    ],
)
def test_session_expiration_normalizes_non_utc_deadline(at: datetime, expected: bool) -> None:
    session = AgentSession(
        name="session-one",
        workspace="default",
        deployment_id="deployment-id",
        expires_at=datetime(2026, 8, 31, 12, 0, tzinfo=timezone(-timedelta(hours=5))),
    )

    assert session_expiration_is_due(session, at=at) is expected


def _gateway_routed_deployment() -> AgentDeployment:
    deployment = AgentDeployment(
        name="dep",
        workspace="default",
        agent="calc",
        status="running",
        endpoint="",
        deployment_mode="k8s",
        endpoints=[Endpoint(name="http", url="https://default--dep--http.openshell.localhost:8080/")],
    )
    deployment._id = "dep-id"
    return deployment


@pytest.mark.asyncio
async def test_cleanup_dials_a_gateway_routed_deployment_through_its_gateway() -> None:
    """Session cleanup reaches an openshell deployment via the gateway address, Host preserved."""
    deployment = _gateway_routed_deployment()
    session = AgentSession(name="s", workspace="default", deployment_id="dep-id", status=SessionStatus.CLOSED)
    session._id = "sess/1"
    entity_client = AsyncMock()
    entity_client.find_one = AsyncMock(return_value=deployment)
    transport = EndpointTransport(
        connect_base="https://openshell.openshell.svc.cluster.local:8080", ca_cert="/tls/ca.crt"
    )
    tls_context = object()

    response = MagicMock(status_code=204)
    client = MagicMock()
    client.delete = AsyncMock(return_value=response)
    client_cm = MagicMock()
    client_cm.__aenter__ = AsyncMock(return_value=client)
    client_cm.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("nemo_agents_plugin.session_lifecycle.get_deployment_transport", return_value=transport),
        patch.object(EndpointTransport, "client_kwargs", return_value={"verify": tls_context}),
        patch("nemo_agents_plugin.session_lifecycle.httpx.AsyncClient", return_value=client_cm) as client_cls,
    ):
        await cleanup_fabric_runtime(entity_client, session)

    assert client_cls.call_args.kwargs["verify"] is tls_context
    client.delete.assert_awaited_once_with(
        "https://openshell.openshell.svc.cluster.local:8080/v1/sessions/sess%2F1",
        headers={"Host": "default--dep--http.openshell.localhost:8080"},
    )
