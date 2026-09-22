# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for ``nemo guardrail configs`` against the in-process Guardrails service.

Guardrail configs live in the entity store, so the ASGI app runs without an LLM;
``nemo guardrail check`` needs a model behind it and is covered wire-level only.

``HF_HUB_OFFLINE`` is set before the ``GuardrailsService`` import because it
transitively imports ``nemoguardrails``, which reaches HuggingFace at import
time unless told to stay offline.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from nemo_guardrails_plugin.cli import GuardrailCLI
from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_plugin.client.client import NemoClient
from nmp.testing import SDKTestClientAdapter, create_test_client
from typer.testing import CliRunner

# Force offline before the service import so a developer shell with HF_HUB_OFFLINE=0 cannot hit the Hub.
os.environ["HF_HUB_OFFLINE"] = "1"

from nmp.guardrails.service import GuardrailsService  # noqa: E402

app = GuardrailCLI().get_cli()


@pytest.fixture(scope="module")
def asgi_client() -> Iterator[NemoClient]:
    """Typed client whose transport is the in-process Guardrails ASGI app."""
    with create_test_client(GuardrailsService, client_type=TestClient) as test_client:
        yield NemoClient(
            base_url=str(test_client.base_url), workspace="default", http_client=SDKTestClientAdapter(test_client)
        )


@pytest.fixture
def run(asgi_client: NemoClient):
    def _run(args: list[str], **kwargs):
        state = CLIContext(overrides={"base_url": "http://test", "output_format": "json"}, _client=asgi_client)
        return CliRunner().invoke(app, args, obj=state, **kwargs)

    return _run


def _name() -> str:
    return f"cfg-{uuid.uuid4().hex[:8]}"


def test_configs_lifecycle(run) -> None:
    name = _name()
    data = {
        "models": [{"type": "main", "engine": "nim", "model": "meta/llama-3.1-8b-instruct"}],
        "instructions": [{"type": "general", "content": "You are a helpful AI assistant."}],
    }

    result = run(["configs", "create", name, "--description", "demo", "--data", json.dumps(data)])
    assert result.exit_code == 0, result.output
    created = json.loads(result.stdout)
    assert created["name"] == name
    assert created["workspace"] == "default"
    assert created["description"] == "demo"
    # The service normalizes RailsConfig with its defaults, so compare the fields we sent.
    assert created["data"]["models"][0]["model"] == "meta/llama-3.1-8b-instruct"
    assert created["data"]["instructions"][0]["content"] == "You are a helpful AI assistant."

    result = run(["configs", "get", name])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["id"] == created["id"]

    result = run(["configs", "list", "--filter.name", name])
    assert result.exit_code == 0, result.output
    listed = json.loads(result.stdout)
    assert [item["name"] for item in listed["data"]] == [name]
    assert listed["pagination"]["total_results"] == 1

    result = run(["configs", "update", name, "--description", "changed"])
    assert result.exit_code == 0, result.output
    updated = json.loads(result.stdout)
    assert updated["description"] == "changed"
    assert updated["data"]["models"][0]["model"] == "meta/llama-3.1-8b-instruct"

    result = run(["configs", "delete", name])
    assert result.exit_code == 0, result.output
    assert "Deleted successfully" in result.stdout

    result = run(["configs", "get", name])
    assert result.exit_code == 3
    assert "Not found" in result.stderr


def test_configs_create_exist_ok_returns_existing(run) -> None:
    name = _name()
    first = run(["configs", "create", name, "--description", "first"])
    assert first.exit_code == 0, first.output

    conflict = run(["configs", "create", name, "--description", "second"])
    assert conflict.exit_code == 3
    assert "Conflict" in conflict.stderr

    result = run(["configs", "create", name, "--description", "second", "--exist-ok"])
    assert result.exit_code == 0, result.output
    existing = json.loads(result.stdout)
    assert existing["id"] == json.loads(first.stdout)["id"]
    assert existing["description"] == "first"


def test_configs_create_from_stdin(run) -> None:
    name = _name()
    payload = {"name": name, "description": "piped"}
    result = run(["configs", "create", "--input-file", "-"], input=json.dumps(payload))
    assert result.exit_code == 0, result.output
    created = json.loads(result.stdout)
    assert created["name"] == name
    assert created["description"] == "piped"


def test_configs_list_all_pages(run) -> None:
    prefix = f"page-{uuid.uuid4().hex[:6]}"
    names = sorted(f"{prefix}-{i}" for i in range(3))
    for name in names:
        assert run(["configs", "create", name]).exit_code == 0

    filter_expr = json.dumps({"name": {"$like": f"{prefix}%"}})
    result = run(["configs", "list", "--filter", filter_expr, "--page-size", "1", "--sort", "name"])
    assert result.exit_code == 0, result.output
    page = json.loads(result.stdout)
    assert [item["name"] for item in page["data"]] == names[:1]
    assert page["pagination"]["total_pages"] == 3
    assert "More pages" in result.stderr

    result = run(["configs", "list", "--filter", filter_expr, "--page-size", "1", "--sort", "name", "--all-pages"])
    assert result.exit_code == 0, result.output
    all_pages = json.loads(result.stdout)
    assert [item["name"] for item in all_pages["data"]] == names
    assert all_pages["pagination"]["total_results"] == 3
    assert "More pages" not in result.stderr
