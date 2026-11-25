# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``compass.analyst``: its context contract and how ``prepare`` resolves a run."""

from __future__ import annotations

import json

import httpx
import pytest
from nemo_compass_plugin.analyst import CompassAnalyst, CompassContext, CompassPreparedContext
from nemo_compass_plugin.config import CompassConfig
from nemo_helix_plugin.agents.plugin_managed_agents import (
    CONFIGURED_DEFAULT_MODEL,
    CONFIGURED_FAST_MODEL,
    HELIX_DEFAULT_SCHEMA_KEY,
    PluginManagedAgentInputError,
    PreparedRun,
)
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nemo_helix_plugin.config import clear_nemo_config_override, set_nemo_config_override
from nhx.testing import mock_async_nemo_client

_ETHOS_PATH = "/apis/files/v2/workspaces/team/filesets/support-bot-ethos/-/ETHOS.md"


class _Platform:
    """The Models and Files APIs, as the caller of a Compass run sees them."""

    def __init__(
        self,
        *,
        models: dict[str, str | None] | None = None,
        ethos: str | None = None,
        ethos_status: int = 200,
    ) -> None:
        # ``workspace/name`` -> backend format (None is the OpenAI default).
        self.models = models if models is not None else {"team/big": None, "team/small": None}
        self.ethos = ethos
        self.ethos_status = ethos_status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.startswith("/apis/models/v2/workspaces/"):
            # /apis/models/v2/workspaces/{workspace}/models/{name}
            workspace, _, name = path.removeprefix("/apis/models/v2/workspaces/").split("/")
            ref = f"{workspace}/{name}"
            if ref not in self.models:
                return httpx.Response(404, json={"detail": f"Model {ref} not found"})
            return httpx.Response(200, json=_model_entity(workspace, name, self.models[ref]))
        if path == _ETHOS_PATH:
            if self.ethos_status != 200:
                return httpx.Response(self.ethos_status, json={"detail": "nope"})
            if self.ethos is None:
                return httpx.Response(404, json={"detail": "no such fileset"})
            return httpx.Response(200, content=self.ethos.encode())
        return httpx.Response(404, json={"detail": f"unexpected {request.method} {path}"})


def _model_entity(workspace: str, name: str, backend_format: str | None) -> dict[str, object]:
    entity: dict[str, object] = {
        "id": f"model-{name}",
        "name": name,
        "workspace": workspace,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    if backend_format is not None:
        entity["backend_format"] = backend_format
    return entity


async def _prepare_run(platform: _Platform, **context: object) -> PreparedRun[CompassPreparedContext]:
    fields = {"agent": "support-bot", "default_model": "big", **context}
    client = mock_async_nemo_client(platform, AsyncNemoClient, workspace="team")
    prepared = await CompassAnalyst.prepare(CompassContext.model_validate(fields), workspace="team", client=client)
    assert prepared.subject == "support-bot"
    return prepared


async def _prepare(platform: _Platform, **context: object) -> CompassPreparedContext:
    return (await _prepare_run(platform, **context)).context


def test_the_context_schema_is_flat_and_hints_both_models() -> None:
    schema = CompassAnalyst.context_schema()

    assert "$ref" not in json.dumps(schema)
    assert schema["required"] == ["agent", "default_model"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["default_model"][HELIX_DEFAULT_SCHEMA_KEY] == CONFIGURED_DEFAULT_MODEL
    assert schema["properties"]["fast_model"][HELIX_DEFAULT_SCHEMA_KEY] == CONFIGURED_FAST_MODEL
    assert "ethos" not in schema["properties"], "the Ethos is read by prepare, not passed by the caller"


async def test_prepare_qualifies_the_models_and_reads_the_agents_ethos() -> None:
    context = await _prepare(_Platform(ethos="# Ethos\n\nBe kind."), fast_model="small", evaluation_id="eval-1")

    assert context == CompassPreparedContext(
        agent="support-bot",
        default_model="team/big",
        fast_model="team/small",
        ethos="# Ethos\n\nBe kind.",
        evaluation_id="eval-1",
    )


async def test_the_fast_model_defaults_to_the_default_model() -> None:
    context = await _prepare(_Platform())

    assert context.fast_model == "team/big"


async def test_a_qualified_ref_may_name_a_model_in_another_workspace() -> None:
    context = await _prepare(_Platform(models={"shared/big": None}), default_model="shared/big")

    assert context.default_model == "shared/big"


@pytest.mark.parametrize(
    ("models", "default_model", "message"),
    [
        pytest.param({}, "big", "default_model 'team/big' is not a known Model Entity", id="unknown"),
        pytest.param({}, "a/b/c", "must be a Model Entity name or '<workspace>/<name>' ref", id="malformed"),
        pytest.param({"team/big": "SYSTEMONE"}, "big", "default_model: Model 'team/big' has unsupported", id="format"),
    ],
)
async def test_a_model_the_run_cannot_use_is_the_callers_to_fix(
    models: dict[str, str | None], default_model: str, message: str
) -> None:
    with pytest.raises(PluginManagedAgentInputError, match=message):
        await _prepare(_Platform(models=models), default_model=default_model)


@pytest.mark.parametrize("ethos", [None, "  \n"], ids=["no-fileset", "blank"])
async def test_an_agent_without_an_ethos_is_analyzed_without_one(ethos: str | None) -> None:
    context = await _prepare(_Platform(ethos=ethos))

    assert context.ethos is None


async def test_an_ethos_the_caller_may_not_read_fails_the_run() -> None:
    """Silently analyzing without an Ethos the caller expected would be worse than refusing."""
    with pytest.raises(PermissionDeniedError):
        await _prepare(_Platform(ethos_status=403))


@pytest.mark.parametrize("profile", ["default", "cpu-cluster"])
async def test_every_run_uses_the_configured_job_profile(profile: str) -> None:
    config = CompassConfig()
    config.analysis.job_profile = profile
    set_nemo_config_override(config)
    try:
        prepared = await _prepare_run(_Platform())
    finally:
        clear_nemo_config_override(CompassConfig)

    assert prepared.job_profile == profile
