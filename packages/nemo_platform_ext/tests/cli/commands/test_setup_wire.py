# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire-level tests for the ``nemo setup`` platform calls against scripted typed clients.

Each test drives the real setup helpers with a ``SetupClients`` bundle whose
``NemoClient`` transport records the HTTP requests, so the assertions pin the
exact method, path, and JSON body setup sends for secrets, providers, model
discovery, and the gateway probe.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from unittest.mock import patch

import httpx
import pytest
from nemo_platform_ext.cli.commands.setup import (
    KeyValidationResult,
    ModelPair,
    SetupClients,
    _auto_setup,
    _create_provider,
    _get_all_model_choices,
    _get_all_model_entity_ids,
    _register_provider_interactive,
    _select_usable_model_pair,
    _wait_for_models,
)
from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_ext.cli.telemetry.events import TaskStatusEnum
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.inference_gateway.client import InferenceGatewayClient
from nemo_platform_plugin.models.client import ModelsClient
from nemo_platform_plugin.secrets.client import SecretsClient
from pydantic import ValidationError

SETUP_MOD = "nemo_platform_ext.cli.commands.setup"

SECRETS_BASE = "/apis/secrets/v2/workspaces/default/secrets"
PROVIDERS_BASE = "/apis/models/v2/workspaces/default/providers"
# The gateway's ``{trailing_uri:path}`` segment is sent percent-encoded by the typed
# client (its own endpoint tests pin that form); uvicorn decodes it before routing.
PROBE_PATH = "/apis/inference-gateway/v2/workspaces/default/openai/-/v1%2Fchat%2Fcompletions"

SECRET = {
    "name": "openai-api-key",
    "workspace": "default",
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
}


def provider_json(name: str, host_url: str, *, served: Sequence[str] = (), status: str = "READY") -> dict:
    return {
        "id": f"provider-{name}",
        "name": name,
        "workspace": "default",
        "host_url": host_url,
        "status": status,
        "status_message": "",
        "served_models": [{"model_entity_id": entity_id, "served_model_name": entity_id} for entity_id in served],
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }


def provider_page(providers: list[dict], *, page: int = 1, total_pages: int = 1) -> dict:
    return {
        "data": providers,
        "pagination": {
            "page": page,
            "page_size": 10,
            "current_page_size": len(providers),
            "total_pages": total_pages,
            "total_results": len(providers) * total_pages,
        },
    }


Outcome = httpx.Response | Exception


@dataclass
class Recorder:
    """Records requests and answers each with the next scripted response (or raises it)."""

    responses: list[Outcome]
    requests: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError(f"unexpected request {request.method} {request.url}")
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]

    def calls(self) -> list[tuple[str, str]]:
        # raw_path keeps percent-encoding, so the assertions see the wire form of the path.
        return [(request.method, request.url.raw_path.decode().split("?", 1)[0]) for request in self.requests]


def make_clients(recorder: Callable[[httpx.Request], httpx.Response]) -> SetupClients:
    """Build the typed-client bundle exactly as ``setup_command`` does, over a recording transport."""
    client = NemoClient(
        base_url="http://test",
        workspace="default",
        http_client=httpx.Client(transport=httpx.MockTransport(recorder)),
    )
    state = CLIContext(overrides={"base_url": "http://test"}, _client=client)
    return SetupClients.from_context(state)


def body_of(request: httpx.Request) -> dict:
    return json.loads(request.content)


@pytest.fixture(autouse=True)
def _quiet(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[None]:
    config_file = tmp_path / "config.yaml"
    config_file.touch()
    monkeypatch.setenv("NMP_CONFIG_FILE", str(config_file))
    monkeypatch.delenv("NMP_ACCESS_TOKEN", raising=False)
    with (
        patch("nemo_platform_ext.cli.telemetry.emit.emit_event"),
        patch(f"{SETUP_MOD}._pause"),
        patch(f"{SETUP_MOD}._validate_api_key", return_value=KeyValidationResult(passed=True, message="")),
    ):
        yield


def test_setup_clients_share_the_cli_client_transport_and_workspace() -> None:
    recorder = Recorder([])
    clients = make_clients(recorder)

    assert isinstance(clients.models, ModelsClient)
    assert isinstance(clients.secrets, SecretsClient)
    assert isinstance(clients.gateway, InferenceGatewayClient)
    assert {c.base_url for c in (clients.models, clients.secrets, clients.gateway)} == {"http://test"}
    assert {c.workspace for c in (clients.models, clients.secrets, clients.gateway)} == {"default"}


class TestAutoSetupWire:
    def test_fresh_install_creates_secret_then_provider(self) -> None:
        recorder = Recorder(
            [
                httpx.Response(404, json={"detail": "not found"}),  # GET secret
                httpx.Response(201, json=SECRET),  # POST secret
                httpx.Response(404, json={"detail": "not found"}),  # GET provider
                httpx.Response(201, json=provider_json("openai", "https://api.openai.com/v1")),  # POST provider
            ]
        )
        clients = make_clients(recorder)

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True):
            assert _auto_setup(clients, "default") == "openai"

        assert recorder.calls() == [
            ("GET", f"{SECRETS_BASE}/openai-api-key"),
            ("POST", SECRETS_BASE),
            ("GET", f"{PROVIDERS_BASE}/openai"),
            ("POST", PROVIDERS_BASE),
        ]
        assert body_of(recorder.requests[1]) == {"name": "openai-api-key", "value": "sk-test"}
        assert body_of(recorder.requests[3]) == {
            "name": "openai",
            "host_url": "https://api.openai.com/v1",
            "api_key_secret_name": "openai-api-key",
        }

    def test_existing_provider_is_upserted_with_auth_template(self) -> None:
        recorder = Recorder(
            [
                httpx.Response(200, json={**SECRET, "name": "anthropic-api-key"}),  # GET secret
                httpx.Response(200, json={**SECRET, "name": "anthropic-api-key"}),  # PATCH secret
                httpx.Response(200, json=provider_json("anthropic", "https://api.anthropic.com")),  # GET provider
                httpx.Response(200, json=provider_json("anthropic", "https://api.anthropic.com")),  # PUT provider
            ]
        )
        clients = make_clients(recorder)

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant"}, clear=True):
            assert _auto_setup(clients, "default") == "anthropic"

        assert recorder.calls() == [
            ("GET", f"{SECRETS_BASE}/anthropic-api-key"),
            ("PATCH", f"{SECRETS_BASE}/anthropic-api-key"),
            ("GET", f"{PROVIDERS_BASE}/anthropic"),
            ("PUT", f"{PROVIDERS_BASE}/anthropic"),
        ]
        assert body_of(recorder.requests[1]) == {"value": "sk-ant"}
        # The legacy required_extra_headers auth path is explicitly cleared as JSON null.
        assert body_of(recorder.requests[3]) == {
            "host_url": "https://api.anthropic.com",
            "api_key_secret_name": "anthropic-api-key",
            "auth_header_format": "X-Api-Key: {{ auth_secret }}",
            "required_extra_headers": None,
            "default_extra_headers": {"anthropic-version": "2023-06-01"},
        }

    def test_secret_value_never_leaves_in_provider_body(self) -> None:
        recorder = Recorder(
            [
                httpx.Response(404, json={"detail": "not found"}),
                httpx.Response(201, json={**SECRET, "name": "nvidia-build-api-key"}),
                httpx.Response(404, json={"detail": "not found"}),
                httpx.Response(201, json=provider_json("nvidia-build", "https://integrate.api.nvidia.com")),
            ]
        )
        clients = make_clients(recorder)

        with patch.dict("os.environ", {"NVIDIA_API_KEY": "nvapi-secret"}, clear=True):
            _auto_setup(clients, "default")

        assert b"nvapi-secret" not in recorder.requests[3].content


class TestRegisterProviderWire:
    def test_provider_without_key_sends_minimal_body(self) -> None:
        recorder = Recorder(
            [
                httpx.Response(404, json={"detail": "not found"}),  # GET provider
                httpx.Response(201, json=provider_json("ollama", "http://localhost:11434/v1")),  # POST provider
            ]
        )
        clients = make_clients(recorder)

        _register_provider_interactive(
            clients,
            provider_name="ollama",
            host_url="http://localhost:11434/v1",
            api_key=None,
            workspace="default",
        )

        assert recorder.calls() == [("GET", f"{PROVIDERS_BASE}/ollama"), ("POST", PROVIDERS_BASE)]
        assert body_of(recorder.last) == {"name": "ollama", "host_url": "http://localhost:11434/v1"}

    def test_locally_invalid_provider_name_still_emits_error_telemetry(self) -> None:
        recorder = Recorder([])  # any request would be an assertion failure
        clients = make_clients(recorder)

        with patch("nemo_platform_ext.cli.telemetry.emit.emit_event") as emit_event:
            with pytest.raises(ValidationError):
                _create_provider(
                    clients,
                    name="not a valid name",
                    host_url="http://localhost:11434/v1",
                    secret_name=None,
                    workspace="default",
                )

        statuses = [call.args[0].task_status for call in emit_event.call_args_list]
        assert statuses == [TaskStatusEnum.ERROR]
        assert recorder.requests == []

    def test_explicit_workspace_is_used_over_client_default(self) -> None:
        recorder = Recorder(
            [
                httpx.Response(404, json={"detail": "not found"}),
                httpx.Response(201, json={**SECRET, "workspace": "team"}),
                httpx.Response(404, json={"detail": "not found"}),
                httpx.Response(201, json={**provider_json("openai", "https://api.openai.com/v1"), "workspace": "team"}),
            ]
        )
        clients = make_clients(recorder)

        _register_provider_interactive(
            clients,
            provider_name="openai",
            host_url="https://api.openai.com/v1",
            api_key="sk-test",
            workspace="team",
        )

        assert recorder.calls() == [
            ("GET", "/apis/secrets/v2/workspaces/team/secrets/openai-api-key"),
            ("POST", "/apis/secrets/v2/workspaces/team/secrets"),
            ("GET", "/apis/models/v2/workspaces/team/providers/openai"),
            ("POST", "/apis/models/v2/workspaces/team/providers"),
        ]


class TestModelDiscoveryWire:
    def test_wait_for_models_polls_the_provider_by_name(self) -> None:
        recorder = Recorder(
            [
                httpx.Response(200, json=provider_json("openai", "https://api.openai.com/v1", status="CREATED")),
                httpx.Response(200, json=provider_json("openai", "https://api.openai.com/v1", served=["default/gpt"])),
            ]
        )
        clients = make_clients(recorder)

        with patch(f"{SETUP_MOD}.time.monotonic", side_effect=[0, 0, 0, 1, 2, 3]):
            models = _wait_for_models(clients, "openai", "default", round_seconds=30, max_rounds=1)

        assert models == ["default/gpt"]
        assert recorder.calls() == [("GET", f"{PROVIDERS_BASE}/openai")] * 2

    def test_wait_for_models_stops_on_error_status(self, capsys) -> None:
        recorder = Recorder(
            [
                httpx.Response(
                    200, json={**provider_json("bad", "https://x"), "status": "ERROR", "status_message": "boom"}
                )
            ]
        )
        clients = make_clients(recorder)

        with patch(f"{SETUP_MOD}.time.monotonic", side_effect=[0, 0, 0, 0]):
            assert _wait_for_models(clients, "bad", "default", round_seconds=30, max_rounds=2) == []

        assert len(recorder.requests) == 1
        assert "Provider 'bad' is in ERROR state." in capsys.readouterr().err

    def test_entity_ids_walk_every_page_and_scope_to_provider(self) -> None:
        recorder = Recorder(
            [
                httpx.Response(
                    200,
                    json=provider_page(
                        [provider_json("openai", "https://api.openai.com/v1", served=["default/gpt-4.1"])],
                        page=1,
                        total_pages=2,
                    ),
                ),
                httpx.Response(
                    200,
                    json=provider_page(
                        [provider_json("anthropic", "https://api.anthropic.com", served=["default/claude"])],
                        page=2,
                        total_pages=2,
                    ),
                ),
            ]
        )
        clients = make_clients(recorder)

        assert _get_all_model_entity_ids(clients, "default", provider_name="anthropic") == ["default/claude"]

        assert recorder.calls() == [("GET", PROVIDERS_BASE)] * 2
        assert recorder.requests[0].url.params.get("page") is None
        assert recorder.requests[1].url.params["page"] == "2"

    def test_model_choices_label_with_provider_name(self) -> None:
        recorder = Recorder(
            [
                httpx.Response(
                    200,
                    json=provider_page([provider_json("openai", "https://api.openai.com/v1", served=["default/gpt"])]),
                )
            ]
        )
        clients = make_clients(recorder)

        assert _get_all_model_choices(clients, "default") == [("default/gpt", "gpt (openai)")]


class TestModelProbeWire:
    def test_probe_posts_a_short_chat_completion_per_candidate(self) -> None:
        recorder = Recorder([httpx.Response(200, json={"choices": []}), httpx.Response(200, json={"choices": []})])
        clients = make_clients(recorder)
        entity_ids = ["default/nvidia-model-70b-instruct", "default/nvidia-model-8b-instruct"]

        pair = _select_usable_model_pair(clients, "default", entity_ids)

        assert pair == ModelPair(default="default/nvidia-model-70b-instruct", fast="default/nvidia-model-8b-instruct")
        assert recorder.calls() == [("POST", PROBE_PATH)] * 2
        assert body_of(recorder.requests[0]) == {
            "model": "default/nvidia-model-70b-instruct",
            "messages": [{"role": "user", "content": "Respond with 'OK'"}],
            "max_tokens": 16,
        }
        assert body_of(recorder.requests[1])["model"] == "default/nvidia-model-8b-instruct"

    def test_probe_retries_a_route_404_then_accepts(self) -> None:
        recorder = Recorder(
            [
                httpx.Response(404, json={"detail": "no route"}),
                httpx.Response(200, json={"choices": []}),
            ]
        )
        clients = make_clients(recorder)

        pair = _select_usable_model_pair(clients, "default", ["default/nvidia-model-8b-instruct"])

        assert pair == ModelPair(default="default/nvidia-model-8b-instruct", fast="default/nvidia-model-8b-instruct")
        assert recorder.calls() == [("POST", PROBE_PATH)] * 2

    def test_probe_does_not_retry_server_errors(self, capsys) -> None:
        """The probe client runs with a zero-retry policy, so a 503 costs exactly one request."""
        recorder = Recorder([httpx.Response(503, text="upstream busy")])
        clients = make_clients(recorder)

        assert _select_usable_model_pair(clients, "default", ["default/nvidia-model-8b-instruct"]) is None
        assert len(recorder.requests) == 1
        assert "Skipping nvidia-model-8b-instruct (HTTP 503)" in capsys.readouterr().err

    def test_probe_skips_a_timed_out_candidate(self, capsys) -> None:
        recorder = Recorder(
            [
                httpx.ReadTimeout("slow"),
                httpx.Response(200, json={"choices": []}),
            ]
        )
        clients = make_clients(recorder)
        entity_ids = ["default/nvidia-model-70b-instruct", "default/nvidia-model-8b-instruct"]

        pair = _select_usable_model_pair(clients, "default", entity_ids)

        assert pair == ModelPair(default="default/nvidia-model-8b-instruct", fast="default/nvidia-model-8b-instruct")
        assert "Skipping nvidia-model-70b-instruct (timed out)" in capsys.readouterr().err

    def test_unreachable_gateway_selects_nothing(self, capsys) -> None:
        recorder = Recorder([httpx.ConnectError("refused")])
        clients = make_clients(recorder)

        assert _select_usable_model_pair(clients, "default", ["default/nvidia-model-8b-instruct"]) is None
        assert len(recorder.requests) == 1
        assert "Could not verify models" in capsys.readouterr().err
