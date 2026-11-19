# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session-wide deletion through the synchronous and asynchronous Intake clients."""

import httpx
import pytest
from nemo_helix_plugin.client.errors import ConflictError, NotFoundError, PermissionDeniedError
from nemo_helix_plugin.intake import endpoints
from nemo_helix_plugin.intake.client import AsyncIntakeClient, IntakeClient


def test_delete_session_endpoint_shape() -> None:
    request = endpoints.delete_session(workspace="team-a", id="session-a")
    assert request.method == "DELETE"
    assert request.path_template == "/apis/intake/v2/workspaces/{workspace}/sessions/{id}"
    assert request.path_params == {"workspace": "team-a", "id": "session-a"}
    assert request.response_type is None


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize(
    ("status", "error"), [(204, None), (403, PermissionDeniedError), (404, NotFoundError), (409, ConflictError)]
)
@pytest.mark.asyncio
async def test_delete_session_wire_behavior(use_async: bool, status: int, error: type[Exception] | None) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if status == 204:
            return httpx.Response(status)
        return httpx.Response(status, json={"detail": "Deletion rejected"})

    transport = httpx.MockTransport(handler)
    if use_async:
        async with httpx.AsyncClient(transport=transport) as async_http_client:
            async_client = AsyncIntakeClient(base_url="http://test", workspace="team-a", http_client=async_http_client)
            if error:
                with pytest.raises(error):
                    await async_client.delete_session(id="session-a")
            else:
                async_response = await async_client.delete_session(id="session-a")
                assert async_response.data() is None
    else:
        with httpx.Client(transport=transport) as http_client:
            client = IntakeClient(base_url="http://test", workspace="team-a", http_client=http_client)
            if error:
                with pytest.raises(error):
                    client.delete_session(id="session-a")
            else:
                response = client.delete_session(id="session-a")
                assert response.data() is None
    assert len(requests) == 1
    assert requests[0].method == "DELETE"
    assert requests[0].url.path == "/apis/intake/v2/workspaces/team-a/sessions/session-a"
    assert requests[0].content == b""
