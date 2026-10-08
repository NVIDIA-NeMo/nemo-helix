# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from deepagents.backends import FilesystemBackend
from deepagents.middleware.skills import _list_skills_with_errors
from nemo_agents_plugin.agent_config import load_agent_config
from nemo_agents_plugin.fabric.translator import translate_agent_config
from nemo_helix_plugin.client.client import NemoClient
from nemo_studio_assistant import register
from nemo_studio_assistant.fabric_compat import (
    apply_deepagents_mcp_env_compatibility,
    apply_deepagents_skill_path_compatibility,
    apply_platform_skill_translation_compatibility,
    virtualize_skill_sources,
)
from nemo_studio_assistant.mcp_server import create_server
from nhx.testing import mock_nemo_client

AGENT_ROOT = Path(__file__).parents[1]
ETHOS_ROOT = AGENT_ROOT.parent / "nemo-studio-assistant-ethos"
EVAL_DATA_PATH = AGENT_ROOT / "src/nemo_studio_assistant/nemo-studio-assistant-eval-data.json"
DOCKERFILE_PATH = AGENT_ROOT / "Dockerfile.fabric-local"
REAL_PREFLIGHT_GUARDRAIL_MODEL = register._preflight_guardrail_model
SKILL_PATHS = ["skills"]

pytestmark = pytest.mark.unit

EvalCall = tuple[str, str, str | None]
EVAL_CASE_COVERAGE: dict[str, tuple[str, tuple[EvalCall, ...], list[object]]] = {
    "Show the details for the default workspace without making any changes.": (
        "details for the default workspace",
        (("workspaces", "retrieve", '{"name":"default"}'),),
        ["default"],
    ),
    "List all workspaces on the platform": (
        "list of workspaces",
        (("workspaces", "list", None),),
        [["default", "research"]],
    ),
    "List the available models and inference providers using the platform API.": (
        "list of available models and inference providers",
        (("models", "list", None), ("inference.providers", "list", None)),
        [["model-a"], ["provider-a"]],
    ),
}


def _load_eval_cases() -> list[dict[str, str]]:
    raw_cases = json.loads(EVAL_DATA_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw_cases, list)
    cases: list[dict[str, str]] = []
    for raw_case in raw_cases:
        assert isinstance(raw_case, dict)
        input_message = raw_case.get("input_message")
        expected_output = raw_case.get("expected_output")
        assert isinstance(input_message, str)
        assert isinstance(expected_output, str)
        cases.append({"input_message": input_message, "expected_output": expected_output})
    return cases


EVAL_CASES = _load_eval_cases()


@pytest.fixture(autouse=True)
def _reset_api_error_streaks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(register, "_api_error_streaks", {})
    monkeypatch.setattr(register, "_guardrail_check_failures", {})
    monkeypatch.setattr(register, "_preflighted_guardrail_models", set())
    monkeypatch.setattr(register, "_guardrail_deployment_results", {})
    monkeypatch.setattr(register, "_preflight_guardrail_model", lambda *_args, **_kwargs: None)


@pytest.fixture(autouse=True)
def _restore_fabric_compat_patches(monkeypatch: pytest.MonkeyPatch) -> None:
    # The fabric_compat shims patch module globals for the agent's process lifetime. Record the
    # originals so pytest restores them and later tests in the worker see unpatched Fabric code.
    from nemo_agents_plugin.fabric import translator
    from nemo_fabric_adapters.deepagents import adapter

    monkeypatch.setattr(translator, "_skills_config", translator._skills_config)
    monkeypatch.setattr(adapter, "resolve_skills", adapter.resolve_skills)
    monkeypatch.setattr(adapter, "_mcp_connection", adapter._mcp_connection)
    monkeypatch.setattr(adapter.common_utils, "capability_plan", adapter.common_utils.capability_plan)


def test_agent_config_translates_to_fabric_deepagents() -> None:
    apply_platform_skill_translation_compatibility()
    config = load_agent_config(AGENT_ROOT / "agent.yaml")
    translated = translate_agent_config(config)

    assert config.default_harness == "deepagents"
    assert config.models["default"].model == "${NEMO_DEFAULT_MODEL}"
    assert translated.harness.adapter_id == "nvidia.fabric.langchain.deepagents"
    assert translated.models["default"].provider == "nvidia"
    assert translated.mcp is not None
    assert translated.mcp.servers["nemo_studio"].url == "nemo-studio-assistant-mcp"
    assert translated.mcp.servers["nemo_studio"].env == {
        "NHX_BASE_URL": "http://host.docker.internal:8080",
        "NHX_WORKSPACE": "default",
    }
    assert translated.skills is not None
    assert translated.skills.paths == ["/skills"]


def test_canonical_registration_config_translates_to_same_runtime() -> None:
    source = translate_agent_config(load_agent_config(AGENT_ROOT / "agent.yaml"))
    registered = translate_agent_config(load_agent_config(ETHOS_ROOT / "agent.yaml"))

    assert registered.harness == source.harness
    assert registered.models == source.models
    assert registered.instructions == source.instructions
    assert registered.mcp == source.mcp


def test_agent_telemetry_exports_atif_traces_to_intake() -> None:
    for config_path in (AGENT_ROOT / "agent.yaml", ETHOS_ROOT / "agent.yaml"):
        config = load_agent_config(config_path)

        assert config.telemetry.enabled is True
        assert config.telemetry.provider == "relay"
        translated = translate_agent_config(config)
        assert translated.relay is not None
        assert translated.relay.project == "nemo-studio-assistant"
        assert translated.relay.observability is not None
        assert translated.relay.observability.model_dump(exclude_none=True)["atif"] == {
            "enabled": True,
            "filename_template": "trajectory-{session_id}.atif.json",
            "output_directory": "./artifacts/relay",
            "agent_name": "nemo-studio-assistant",
            "model_name": "${NEMO_DEFAULT_MODEL}",
            "storage": [
                {
                    "type": "http",
                    "endpoint": "http://127.0.0.1:8080/apis/intake/v2/workspaces/default/ingest/atif",
                    "headers": {},
                    "header_env": {},
                    "timeout_millis": 3000,
                }
            ],
        }


def test_every_configured_skill_is_packaged() -> None:
    config = load_agent_config(AGENT_ROOT / "agent.yaml")

    assert config.skills is not None
    assert config.skills.paths == SKILL_PATHS
    skill_files = sorted((AGENT_ROOT / "skills").glob("*/SKILL.md"))
    assert len(skill_files) == 9
    assert all(skill_file.is_file() for skill_file in skill_files)
    assert all(skill_file.read_text(encoding="utf-8").startswith("---\n") for skill_file in skill_files)

    registered = load_agent_config(ETHOS_ROOT / "agent.yaml")
    assert registered.skills is not None
    assert registered.skills.paths == SKILL_PATHS
    assert len(list((ETHOS_ROOT / "skills").glob("*/SKILL.md"))) == 9
    assert config.environment.workspace == "."
    assert registered.environment.workspace == "."


def test_deepagents_runtime_can_load_packaged_skill_library() -> None:
    backend = FilesystemBackend(root_dir=AGENT_ROOT, virtual_mode=True)

    skills, error = _list_skills_with_errors(backend, "skills")

    assert error is None
    assert {skill["name"] for skill in skills} == {
        "auditor",
        "benchmark-execution",
        "entities",
        "evals",
        "files",
        "guardrails",
        "inference",
        "secrets",
        "workspace",
    }


def test_fabric_compatibility_resolves_packaged_skills_in_virtual_mode() -> None:
    from nemo_fabric_adapter_contract.models import (
        AgentConfig,
        AgentSkillConfig,
        ArtifactManifest,
        ControlLocation,
        EnvironmentHandle,
        EnvironmentOwnership,
        RuntimeContext,
    )
    from nemo_fabric_adapters.deepagents import adapter

    apply_deepagents_skill_path_compatibility()

    runtime_context = RuntimeContext(
        runtime_id="test-runtime",
        invocation_id="test-invocation",
        request_id="test-request",
        environment=EnvironmentHandle(
            environment_id="test-environment",
            provider="local",
            control_location=ControlLocation.EXTERNAL_CONTROL,
            workspace=".",
            ownership=EnvironmentOwnership.CALLER_OWNED,
        ),
        artifacts=ArtifactManifest(),
    )
    config = AgentConfig(skills=AgentSkillConfig(paths=["/skills"]))
    backend = adapter.resolve_backend(runtime_context, str(AGENT_ROOT))
    skill_sources = adapter.resolve_skills(config)

    assert isinstance(backend, FilesystemBackend)
    assert skill_sources is not None
    assert skill_sources == ["/skills"]
    skills, error = _list_skills_with_errors(backend, skill_sources[0])
    assert error is None
    assert {skill["name"] for skill in skills} == {
        "auditor",
        "benchmark-execution",
        "entities",
        "evals",
        "files",
        "guardrails",
        "inference",
        "secrets",
        "workspace",
    }


def test_fabric_absolute_skill_source_is_virtualized_under_workspace() -> None:
    assert virtualize_skill_sources(
        ["/tmp/nemo/skills", "relative-skills", "/outside/skills"],
        Path("/tmp/nemo"),
    ) == ["/skills", "relative-skills", "/outside/skills"]


def test_fabric_stdio_mcp_inherits_platform_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_fabric_adapters.deepagents import adapter

    monkeypatch.setenv("NHX_BASE_URL", "http://platform:8080")
    monkeypatch.setenv("NHX_WORKSPACE", "default")
    apply_deepagents_mcp_env_compatibility()

    connection = adapter._mcp_connection(
        "nemo_studio",
        SimpleNamespace(
            transport="stdio",
            url="nemo-studio-assistant-mcp",
            args=[],
            env={"CUSTOM": "value"},
        ),
    )

    assert connection["env"]["NHX_BASE_URL"] == "http://platform:8080"
    assert connection["env"]["NHX_WORKSPACE"] == "default"
    assert connection["env"]["CUSTOM"] == "value"


def test_fabric_image_loads_startup_compatibility_shims() -> None:
    dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "ENV PYTHONPATH=/workspace" in dockerfile
    assert (AGENT_ROOT / "sitecustomize.py").is_file()


def test_platform_translation_virtualizes_packaged_skill_root() -> None:
    apply_platform_skill_translation_compatibility()

    translated = translate_agent_config(load_agent_config(AGENT_ROOT / "agent.yaml"))

    assert translated.skills is not None
    assert translated.skills.paths == ["/skills"]


def test_guardrails_skill_is_generic_sdk_workflow_and_copies_match() -> None:
    source = AGENT_ROOT / "src/nemo_studio_assistant/skills/guardrails/SKILL.md"
    packaged = AGENT_ROOT / "skills/guardrails/SKILL.md"
    registered = ETHOS_ROOT / "skills/guardrails/SKILL.md"
    skill_text = source.read_text(encoding="utf-8")

    assert packaged.read_text(encoding="utf-8") == skill_text
    assert registered.read_text(encoding="utf-8") == skill_text
    assert "guardrail.configs" in skill_text
    assert "| List/read backend models | `models` | `list` / `retrieve` |" in skill_text
    assert "inference.virtual_models" in skill_text
    assert "inference.gateway.model" in skill_text
    assert 'status: "blocked"' in skill_text
    assert "unvalidated config" in skill_text
    assert "nemo guardrail" not in skill_text
    assert "harmful-message check" not in skill_text
    assert "default/mock-llm" not in skill_text


def test_mcp_server_exposes_expected_tools() -> None:
    tools = asyncio.run(create_server().list_tools())

    assert {tool.name for tool in tools} == {
        "deploy_guardrail",
        "nemo_api",
        "check_status",
        "select_agent",
        "select_model",
        "select_dataset_file",
        "select_eval_config",
        "job_progress",
        "studio_link",
        "ask_user_question",
    }
    deploy_tool = next(tool for tool in tools if tool.name == "deploy_guardrail")
    blocked_schema = deploy_tool.parameters["properties"]["blocked_message"]
    assert "actual end-user request" in blocked_schema["description"]
    assert "Never pass predicted refusal text" in blocked_schema["description"]


_TIMESTAMP = "2026-01-01T00:00:00Z"
_WORKSPACES = "/apis/entities/v2/workspaces"
_MODELS = "/apis/models/v2/workspaces/default"
_GUARDRAILS = "/apis/guardrails/v2/workspaces/default"
_INFERENCE_GATEWAY = "/apis/inference-gateway/v2/workspaces/default"
_PAGINATION = {"page": 1, "page_size": 10, "current_page_size": 0, "total_pages": 1, "total_results": 0}

Route = Callable[[httpx.Request], httpx.Response]


def _page(items: list[dict[str, Any]]) -> httpx.Response:
    pagination = {**_PAGINATION, "current_page_size": len(items), "total_results": len(items)}
    return httpx.Response(200, json={"data": items, "pagination": pagination})


def _workspace(name: str) -> dict[str, Any]:
    return {
        "id": f"ws-{name}",
        "name": name,
        "description": f"{name} workspace",
        "created_at": _TIMESTAMP,
        "updated_at": _TIMESTAMP,
    }


def _entity(name: str, **fields: Any) -> dict[str, Any]:
    return {
        "id": f"id-{name}",
        "name": name,
        "workspace": "default",
        "created_at": _TIMESTAMP,
        "updated_at": _TIMESTAMP,
        **fields,
    }


def _not_found(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(404, json={"detail": "not found"})


class _FakeHelix:
    """An in-memory platform answering typed-client requests by ``(method, path)``."""

    def __init__(self, routes: dict[tuple[str, str], Route] | None = None) -> None:
        self.routes: dict[tuple[str, str], Route] = dict(routes or {})
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.routes.get((request.method, request.url.path), _not_found)(request)

    def client(self, workspace: str = "default") -> NemoClient:
        return mock_nemo_client(self.handle, NemoClient, workspace=workspace)

    def calls(self, method: str, path: str) -> list[httpx.Request]:
        return [request for request in self.requests if (request.method, request.url.path) == (method, path)]

    def mutations(self) -> list[httpx.Request]:
        return [request for request in self.requests if request.method != "GET"]


def _json_route(payload: Any, status: int = 200) -> Route:
    return lambda _request: httpx.Response(status, json=payload)


def _ok_helix(method: str, path: str, payload: Any = None, status: int = 200) -> _FakeHelix:
    return _FakeHelix({(method, path): _json_route(payload if payload is not None else {}, status)})


def _guardrail_workflow_client(check_statuses: list[str]) -> tuple[NemoClient, dict[str, Any]]:
    state: dict[str, Any] = {"configs": {}, "virtual_models": {}, "checks": []}
    statuses = iter(check_statuses)

    def get_config(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, json=state["configs"][name]) if name in state["configs"] else _not_found(request)

    def create_config(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        state["configs"][body["name"]] = {**_entity(body["name"]), **body, "data": body.get("data") or {}}
        return httpx.Response(201, json=state["configs"][body["name"]])

    def get_virtual_model(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[-1]
        if name not in state["virtual_models"]:
            return _not_found(request)
        return httpx.Response(200, json=state["virtual_models"][name])

    def create_virtual_model(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        state["virtual_models"][body["name"]] = {**_entity(body["name"]), **body}
        return httpx.Response(201, json=state["virtual_models"][body["name"]])

    def check(request: httpx.Request) -> httpx.Response:
        state["checks"].append(json.loads(request.content))
        return httpx.Response(200, json={"status": next(statuses)})

    def list_models(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": f"default/{name}"} for name in state["virtual_models"]]})

    platform = _FakeHelix()
    platform.routes = {
        ("POST", f"{_GUARDRAILS}/checks"): check,
        ("POST", f"{_GUARDRAILS}/configs"): create_config,
        ("GET", f"{_GUARDRAILS}/configs/no-fruit"): get_config,
        ("GET", f"{_INFERENCE_GATEWAY}/virtual-models/guarded-model"): get_virtual_model,
        ("POST", f"{_INFERENCE_GATEWAY}/virtual-models"): create_virtual_model,
        ("GET", f"{_INFERENCE_GATEWAY}/openai/-/v1/models"): list_models,
    }
    state["platform"] = platform
    return platform.client(), state


def _install_client(monkeypatch: pytest.MonkeyPatch, client: NemoClient, workspace: str = "default") -> None:
    monkeypatch.setattr(register, "_clients", {workspace: client})


def test_deploy_guardrail_uses_one_approval_and_reports_milestones(monkeypatch: pytest.MonkeyPatch) -> None:
    session_id = "90a877d5-19f6-49a8-bf09-d0020ae0833a"
    client, state = _guardrail_workflow_client(["blocked", "success"])
    callbacks: list[tuple[str, dict[str, object]]] = []

    def callback(
        _session_id: str,
        tool_name: str,
        arguments: dict[str, object],
        **_kwargs: object,
    ) -> dict[str, object]:
        callbacks.append((tool_name, arguments))
        return {"behavior": "allow"} if tool_name == "approval_prompt" else {"status": "rendered"}

    _install_client(monkeypatch, client)
    monkeypatch.setattr(register, "_call_studio_tool", callback)

    response = json.loads(
        register.deploy_guardrail(
            policy="Do not discuss fruit.",
            config_name="no-fruit",
            backend_model="default/model-a",
            virtual_model_name="guarded-model",
            blocked_message="Tell me about bananas.",
            allowed_message="Tell me about the moon.",
            studio_session_id=session_id,
            deployment_run_id="2a37cb64-6e28-4674-94a7-a122cdf0d08f",
            workspace="default",
        )
    )

    assert response == {
        "status": "success",
        "config": "default/no-fruit",
        "virtual_model": "default/guarded-model",
        "validation": {"blocked_message": "blocked", "allowed_message": "success"},
        "chat_model": "default/guarded-model",
        "studio_link": (
            "[Chat with VirtualModel default/guarded-model]"
            "(/workspaces/default/virtual-models?virtualModel=guarded-model&tab=chat)"
        ),
        "routable": True,
        "warning": None,
    }
    assert [name for name, _ in callbacks].count("approval_prompt") == 1
    assert [arguments["step"] for name, arguments in callbacks if name == "assistant_activity"] == [
        "config_ready",
        "blocked_check",
        "allowed_check",
        "virtual_model_ready",
    ]
    assert [check["guardrails"] for check in state["checks"]] == [{"config_id": "default/no-fruit"}] * 2
    assert [check["messages"][0]["content"] for check in state["checks"]] == [
        "Tell me about bananas.",
        "Tell me about the moon.",
    ]
    assert set(state["virtual_models"]) == {"guarded-model"}


def test_deploy_guardrail_preserves_created_virtual_model_when_routing_is_delayed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, state = _guardrail_workflow_client(["blocked", "success"])
    callbacks: list[tuple[str, dict[str, object]]] = []

    def callback(
        _session: str,
        tool_name: str,
        arguments: dict[str, object],
        **_kwargs: object,
    ) -> dict[str, object]:
        callbacks.append((tool_name, arguments))
        return {"behavior": "allow"} if tool_name == "approval_prompt" else {"status": "rendered"}

    _install_client(monkeypatch, client)
    monkeypatch.setattr(register, "_call_studio_tool", callback)
    monkeypatch.setattr(register, "_routable_virtual_model", lambda *_args: False)

    response = json.loads(
        register.deploy_guardrail(
            policy="Do not discuss fruit.",
            config_name="no-fruit",
            backend_model="default/model-a",
            virtual_model_name="guarded-model",
            blocked_message="Tell me about bananas.",
            allowed_message="Tell me about the moon.",
            studio_session_id="90a877d5-19f6-49a8-bf09-d0020ae0833a",
            deployment_run_id="2a37cb64-6e28-4674-94a7-a122cdf0d08f",
            workspace="default",
        )
    )

    assert response["status"] == "success"
    assert response["virtual_model"] == "default/guarded-model"
    assert response["chat_model"] == "default/guarded-model"
    assert response["routable"] is False
    assert response["studio_link"].endswith("virtualModel=guarded-model&tab=chat)")
    assert "was created and verified, but routing is still propagating" in response["warning"]
    assert "Open the chat link in a moment; do not redeploy it" in response["warning"]
    pending_activity = callbacks[-1][1]
    assert pending_activity["step"] == "virtual_model_pending"
    assert pending_activity["status"] == "completed"
    assert set(state["virtual_models"]) == {"guarded-model"}


def test_deploy_guardrail_returns_original_result_for_duplicate_run(monkeypatch: pytest.MonkeyPatch) -> None:
    client, state = _guardrail_workflow_client(["blocked", "success"])
    callbacks: list[str] = []

    def callback(
        _session_id: str,
        tool_name: str,
        _arguments: dict[str, object],
        **_kwargs: object,
    ) -> dict[str, object]:
        callbacks.append(tool_name)
        return {"behavior": "allow"} if tool_name == "approval_prompt" else {"status": "rendered"}

    _install_client(monkeypatch, client)
    monkeypatch.setattr(register, "_call_studio_tool", callback)
    arguments = {
        "policy": "Do not discuss fruit.",
        "config_name": "no-fruit",
        "backend_model": "default/model-a",
        "virtual_model_name": "guarded-model",
        "blocked_message": "Tell me about bananas.",
        "allowed_message": "Tell me about the moon.",
        "studio_session_id": "90a877d5-19f6-49a8-bf09-d0020ae0833a",
        "deployment_run_id": "2a37cb64-6e28-4674-94a7-a122cdf0d08f",
        "workspace": "default",
    }

    first = register.deploy_guardrail(**arguments)
    second = register.deploy_guardrail(**arguments)

    assert second == first
    assert callbacks.count("approval_prompt") == 1
    assert len(state["checks"]) == 2
    assert set(state["virtual_models"]) == {"guarded-model"}


def test_deploy_guardrail_caches_setup_failure_instead_of_leaving_run_in_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_client(_workspace: str) -> None:
        raise RuntimeError("client setup failed")

    monkeypatch.setattr(register, "_get_client", fail_client)
    monkeypatch.setattr(register, "_report_workflow_activity", lambda *_args, **_kwargs: None)
    arguments = {
        "policy": "Do not discuss fruit.",
        "config_name": "no-fruit",
        "backend_model": "default/model-a",
        "virtual_model_name": "guarded-model",
        "blocked_message": "Tell me about bananas.",
        "allowed_message": "Tell me about the moon.",
        "studio_session_id": "90a877d5-19f6-49a8-bf09-d0020ae0833a",
        "deployment_run_id": "2a37cb64-6e28-4674-94a7-a122cdf0d08f",
        "workspace": "default",
    }

    first = register.deploy_guardrail(**arguments)
    second = register.deploy_guardrail(**arguments)

    assert second == first
    assert "RuntimeError: client setup failed" in first
    assert "duplicate guardrail deployment attempt" not in second


def test_matches_fields_allows_server_normalized_nested_defaults() -> None:
    actual = {
        "default_model_entity": "default/model-a",
        "request_middleware": [
            {
                "name": "nemo-guardrails",
                "config_type": "guardrail_config",
                "config_id": "default/no-fruit",
                "config": None,
            }
        ],
        "id": "virtual-model-generated-id",
    }
    expected = {
        "default_model_entity": "default/model-a",
        "request_middleware": [
            {
                "name": "nemo-guardrails",
                "config_type": "guardrail_config",
                "config_id": "default/no-fruit",
            }
        ],
    }

    assert register._matches_fields(actual, expected)
    assert not register._matches_fields(
        actual,
        {
            **expected,
            "request_middleware": [
                {
                    **expected["request_middleware"][0],
                    "config_id": "default/different-config",
                }
            ],
        },
    )


def test_deploy_guardrail_stops_before_virtual_model_when_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, state = _guardrail_workflow_client(["success"])
    callbacks: list[tuple[str, dict[str, object]]] = []

    def callback(
        _session_id: str,
        tool_name: str,
        arguments: dict[str, object],
        **_kwargs: object,
    ) -> dict[str, object]:
        callbacks.append((tool_name, arguments))
        return {"behavior": "allow"} if tool_name == "approval_prompt" else {"status": "rendered"}

    _install_client(monkeypatch, client)
    monkeypatch.setattr(register, "_call_studio_tool", callback)

    response = json.loads(
        register.deploy_guardrail(
            policy="Do not discuss fruit.",
            config_name="no-fruit",
            backend_model="default/model-a",
            virtual_model_name="guarded-model",
            blocked_message="Tell me about bananas.",
            allowed_message="Tell me about the moon.",
            studio_session_id="90a877d5-19f6-49a8-bf09-d0020ae0833a",
            deployment_run_id="2a37cb64-6e28-4674-94a7-a122cdf0d08f",
            workspace="default",
        )
    )

    assert response["status"] == "partial"
    assert response["config"] == "default/no-fruit"
    assert response["virtual_model"] is None
    assert "expected guardrail check status 'blocked'" in response["error"]
    assert state["virtual_models"] == {}
    assert callbacks[-1][1]["step"] == "failed"
    assert callbacks[-1][1]["status"] == "failed"


@pytest.mark.parametrize(
    "blocked_message",
    [
        "I'm sorry, I can't discuss fruit.",
        "I’m sorry, I cannot discuss fruit.",
        "We have to decline to discuss fruit.",
        "Unable to help with fruit.",
    ],
)
def test_deploy_guardrail_rejects_refusal_like_probe_before_any_platform_action(
    monkeypatch: pytest.MonkeyPatch,
    blocked_message: str,
) -> None:
    callbacks: list[str] = []
    monkeypatch.setattr(register, "_clients", {})
    monkeypatch.setattr(
        register,
        "_call_studio_tool",
        lambda _session, tool_name, *_args, **_kwargs: callbacks.append(tool_name),
    )

    response = json.loads(
        register.deploy_guardrail(
            policy="Do not discuss fruit.",
            config_name="no-fruit",
            backend_model="default/model-a",
            virtual_model_name="guarded-model",
            blocked_message=blocked_message,
            allowed_message="Tell me about the moon.",
            studio_session_id="90a877d5-19f6-49a8-bf09-d0020ae0833a",
            deployment_run_id="2a37cb64-6e28-4674-94a7-a122cdf0d08f",
            workspace="default",
        )
    )

    assert response == {
        "status": "failed",
        "error": ("blocked_message must be an actual policy-violating user request, not refusal-like assistant output"),
    }
    assert register._clients == {}
    assert callbacks == []


def test_deploy_guardrail_rejects_identical_probe_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(register, "_clients", {})

    response = json.loads(
        register.deploy_guardrail(
            policy="Do not discuss fruit.",
            config_name="no-fruit",
            backend_model="default/model-a",
            virtual_model_name="guarded-model",
            blocked_message="Tell me about bananas.",
            allowed_message="tell me about bananas.",
            studio_session_id="90a877d5-19f6-49a8-bf09-d0020ae0833a",
            deployment_run_id="2a37cb64-6e28-4674-94a7-a122cdf0d08f",
            workspace="default",
        )
    )

    assert response["status"] == "failed"
    assert response["error"] == ("blocked_message and allowed_message must be different representative user requests")
    assert register._clients == {}


def test_deploy_guardrail_selects_backend_inside_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _state = _guardrail_workflow_client(["blocked", "success"])
    callback_names: list[str] = []

    def callback(
        _session_id: str,
        tool_name: str,
        _arguments: dict[str, object],
        **_kwargs: object,
    ) -> dict[str, object]:
        callback_names.append(tool_name)
        if tool_name == "select_model":
            return {"status": "submitted", "model": "default/model-a"}
        if tool_name == "approval_prompt":
            return {"behavior": "allow"}
        return {"status": "rendered"}

    _install_client(monkeypatch, client)
    monkeypatch.setattr(register, "_call_studio_tool", callback)

    response = json.loads(
        register.deploy_guardrail(
            policy="Do not discuss fruit.",
            config_name="no-fruit",
            virtual_model_name="guarded-model",
            blocked_message="Tell me about bananas.",
            allowed_message="Tell me about the moon.",
            studio_session_id="90a877d5-19f6-49a8-bf09-d0020ae0833a",
            deployment_run_id="2a37cb64-6e28-4674-94a7-a122cdf0d08f",
            workspace="default",
        )
    )

    assert response["status"] == "success"
    assert callback_names[:2] == ["select_model", "approval_prompt"]


def test_studio_link_passes_workspace_to_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    callback_args: dict[str, object] = {}

    def callback(*args: object, **kwargs: object) -> dict[str, str]:
        callback_args.update({"args": args, "kwargs": kwargs})
        return {"href": "/workspaces/default/guardrails"}

    monkeypatch.setattr(register, "_call_studio_tool", callback)

    response = json.loads(
        register.studio_link(
            "90a877d5-19f6-49a8-bf09-d0020ae0833a",
            "guardrails",
            workspace="default",
        )
    )

    assert response == {"href": "/workspaces/default/guardrails"}
    assert callback_args["kwargs"] == {"workspace": "default", "studio_base_url": None}


def test_studio_link_failure_is_non_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> dict[str, str]:
        raise RuntimeError("callback unavailable")

    monkeypatch.setattr(register, "_call_studio_tool", fail)

    response = json.loads(
        register.studio_link(
            "90a877d5-19f6-49a8-bf09-d0020ae0833a",
            "guardrails",
            workspace="default",
        )
    )

    assert response["status"] == "unavailable"
    assert "Continue with the verified task result" in response["message"]
    assert "callback unavailable" in response["message"]


def test_read_only_nemo_api_calls_platform_without_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = _ok_helix("GET", _WORKSPACES, {"data": [_workspace("default")], "pagination": _PAGINATION})
    _install_client(monkeypatch, platform.client())

    response = json.loads(register.nemo_api("workspaces", "list", workspace="default"))

    assert [item["name"] for item in response["data"]] == ["default"]
    assert [request.method for request in platform.requests] == ["GET"]


def test_nemo_api_accepts_object_params(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = _ok_helix("GET", f"{_WORKSPACES}/default", _workspace("default"))
    _install_client(monkeypatch, platform.client())

    response = json.loads(register.nemo_api("workspaces", "retrieve", params={"name": "default"}, workspace="default"))

    assert response["name"] == "default"
    assert response["description"] == "default workspace"


def test_nemo_api_maps_params_onto_path_query_and_body(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = _FakeHelix(
        {
            ("GET", f"{_MODELS}/models"): lambda _request: _page([_entity("model-a")]),
            ("POST", f"{_WORKSPACES}"): _json_route(_workspace("demo"), 201),
        }
    )
    _install_client(monkeypatch, platform.client())

    listed = json.loads(register.nemo_api("models", "list", params={"page_size": 5}, workspace="default"))
    assert [item["name"] for item in listed["data"]] == ["model-a"]
    assert dict(platform.requests[0].url.params) == {"page_size": "5"}

    monkeypatch.setattr(register, "_call_studio_tool", lambda *_args, **_kwargs: {"behavior": "allow"})
    created = json.loads(
        register.nemo_api(
            "workspaces",
            "create",
            params={"name": "demo", "description": "A demo"},
            studio_session_id="90a877d5-19f6-49a8-bf09-d0020ae0833a",
            workspace="default",
        )
    )
    assert created["name"] == "demo"
    assert json.loads(platform.requests[1].content) == {"name": "demo", "description": "A demo"}


def test_nemo_api_maps_fileset_params_onto_file_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    path = "/apis/files/v2/workspaces/default/filesets/harbor/-/verify.txt"
    uploaded = {"file_ref": "ref", "file_url": "url", "path": "verify.txt", "size": 4}
    platform = _FakeHelix(
        {
            ("PUT", path): _json_route(uploaded),
            ("GET", path): lambda _request: httpx.Response(200, content=b"text"),
        }
    )
    _install_client(monkeypatch, platform.client())
    monkeypatch.setattr(register, "_call_studio_tool", lambda *_args, **_kwargs: {"behavior": "allow"})

    upload = json.loads(
        register.nemo_api(
            "files",
            "upload_content",
            params={"content": "text", "remote_path": "verify.txt", "fileset": "harbor"},
            studio_session_id="90a877d5-19f6-49a8-bf09-d0020ae0833a",
            workspace="default",
        )
    )
    download = json.loads(
        register.nemo_api(
            "files",
            "download_content",
            params={"remote_path": "verify.txt", "fileset": "harbor"},
            studio_session_id="90a877d5-19f6-49a8-bf09-d0020ae0833a",
            workspace="default",
        )
    )

    assert upload["path"] == "verify.txt"
    assert platform.calls("PUT", path)[0].content == b"text"
    assert download == "text"


def test_invalid_guardrail_config_action_returns_exact_path_before_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approval_requested = False

    def approve(*_args: object, **_kwargs: object) -> dict[str, str]:
        nonlocal approval_requested
        approval_requested = True
        return {"behavior": "allow"}

    platform = _FakeHelix()
    _install_client(monkeypatch, platform.client())
    monkeypatch.setattr(register, "_call_studio_tool", approve)
    response = register.nemo_api(
        "guardrail",
        "create",
        '{"name": "fruit-blocker"}',
        studio_session_id="90a877d5-19f6-49a8-bf09-d0020ae0833a",
        workspace="default",
    )

    assert "resource='guardrail.configs'" in response
    assert "resource='guardrail' is only for action='check'" in response
    assert approval_requested is False
    assert platform.requests == []


def test_repeated_api_path_errors_trip_api_circuit_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_client(monkeypatch, _FakeHelix().client())
    session_id = "90a877d5-19f6-49a8-bf09-d0020ae0833a"

    first = register.nemo_api("guardrails", "list", studio_session_id=session_id, workspace="default")
    second = register.nemo_api("input_guardrail", "list", studio_session_id=session_id, workspace="default")
    third = register.nemo_api("guardrail_policy", "list", studio_session_id=session_id, workspace="default")

    assert first.startswith("Error: ApiPathError:")
    assert second.startswith("Error: ApiPathError:")
    assert third.startswith("Error circuit breaker: 3 consecutive nemo_api calls failed")
    assert "Stop retrying or guessing API paths and parameters" in third


def test_repeated_parameter_errors_trip_api_circuit_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = _FakeHelix()
    _install_client(monkeypatch, platform.client())

    first = register.nemo_api("guardrail", "check", '{"config_id":"bad"}', workspace="default")
    second = register.nemo_api("guardrail", "check", '{"input":"bad"}', workspace="default")
    third = register.nemo_api("guardrail", "check", '{"config":"bad"}', workspace="default")

    assert first.startswith("Guardrail validation stopped: ValidationError:")
    assert second.startswith("Guardrail validation stopped: ValidationError:")
    assert third.startswith("Guardrail validation stopped: 3 validation attempts failed")
    assert "do not attach it to a VirtualModel" in third
    assert platform.requests == []


def test_guardrail_check_uses_workspace_from_params_when_outer_argument_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform = _FakeHelix({("POST", f"{_GUARDRAILS}/checks"): _json_route({"status": "blocked"})})
    _install_client(monkeypatch, platform.client())

    response = json.loads(
        register.nemo_api(
            "guardrail",
            "check",
            '{"workspace":"default","model":"default/model","messages":[]}',
        )
    )

    assert response["status"] == "blocked"
    assert json.loads(platform.requests[0].content)["model"] == "default/model"


def test_guardrail_check_missing_workspace_counts_as_validation_failure() -> None:
    response = register.nemo_api(
        "guardrail",
        "check",
        '{"model":"default/model","messages":[]}',
        studio_session_id="90a877d5-19f6-49a8-bf09-d0020ae0833a",
    )

    assert response.startswith("Guardrail validation stopped: which workspace")
    assert "Validation failure 1 of 3" in response


def test_guardrail_check_rejects_unexpected_status(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = _FakeHelix({("POST", f"{_GUARDRAILS}/checks"): _json_route({"status": "unknown"})})
    _install_client(monkeypatch, platform.client())

    response = register.nemo_api(
        "guardrail",
        "check",
        '{"model":"default/model","messages":[]}',
        workspace="default",
    )

    assert "unexpected status 'unknown'" in response
    assert "Do not attach this unvalidated guardrail" in response


def test_guardrail_model_preflight_calls_chat_completion() -> None:
    platform = _ok_helix("POST", f"{_INFERENCE_GATEWAY}/model/model-a/-/v1/chat/completions", {"choices": []})
    client = platform.client()

    REAL_PREFLIGHT_GUARDRAIL_MODEL(
        client,
        "default",
        {"model": "default/model-a"},
        "session-a",
    )
    REAL_PREFLIGHT_GUARDRAIL_MODEL(
        client,
        "default",
        {"model": "default/model-a"},
        "session-a",
    )

    assert len(platform.requests) == 1
    assert json.loads(platform.requests[0].content) == {
        "model": "default/model-a",
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 1,
        "temperature": 0,
    }


def test_guardrail_model_preflight_surfaces_provider_failure() -> None:
    platform = _ok_helix(
        "POST",
        f"{_INFERENCE_GATEWAY}/model/model-a/-/v1/chat/completions",
        {"detail": "provider offline"},
        status=503,
    )

    with pytest.raises(register.ModelPreflightError, match="model 'default/model-a' is unavailable") as exc_info:
        REAL_PREFLIGHT_GUARDRAIL_MODEL(
            platform.client(),
            "default",
            {"model": "default/model-a"},
            "session-a",
        )

    assert "provider offline" in str(exc_info.value)


def test_successful_api_call_resets_api_error_circuit_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = _FakeHelix({("GET", f"{_GUARDRAILS}/configs"): lambda _request: _page([_entity("fruit-blocker")])})
    _install_client(monkeypatch, platform.client())
    register._api_error_streaks["workspace:default"] = 2

    response = json.loads(register.nemo_api("guardrail.configs", "list", workspace="default"))

    assert [item["name"] for item in response["data"]] == ["fruit-blocker"]
    assert register._api_error_streaks == {}


def test_guardrail_check_does_not_request_mutation_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    approval_requested = False

    def approve(*_args: object, **_kwargs: object) -> dict[str, str]:
        nonlocal approval_requested
        approval_requested = True
        return {"behavior": "allow"}

    platform = _FakeHelix({("POST", f"{_GUARDRAILS}/checks"): _json_route({"status": "blocked"})})
    _install_client(monkeypatch, platform.client())
    monkeypatch.setattr(register, "_call_studio_tool", approve)

    response = json.loads(
        register.nemo_api(
            "guardrail",
            "check",
            '{"model":"default/demo-unguarded","messages":[]}',
            workspace="default",
        )
    )

    assert response["status"] == "blocked"
    assert approval_requested is False


def test_every_eval_case_has_deterministic_coverage() -> None:
    assert len(EVAL_CASES) == 3
    assert {case["input_message"] for case in EVAL_CASES} == EVAL_CASE_COVERAGE.keys()


@pytest.mark.parametrize("case", EVAL_CASES, ids=lambda case: case["input_message"])
def test_eval_case_uses_expected_read_only_tool_paths(monkeypatch: pytest.MonkeyPatch, case: dict[str, str]) -> None:
    expected_output, calls, expected_results = EVAL_CASE_COVERAGE[case["input_message"]]
    assert case["expected_output"] == expected_output

    platform = _FakeHelix(
        {
            ("GET", f"{_WORKSPACES}/default"): _json_route(_workspace("default")),
            ("GET", _WORKSPACES): lambda _request: _page([_workspace("default"), _workspace("research")]),
            ("GET", f"{_MODELS}/models"): lambda _request: _page([_entity("model-a")]),
            ("GET", f"{_MODELS}/providers"): lambda _request: _page([_entity("provider-a", host_url="http://x")]),
        }
    )
    _install_client(monkeypatch, platform.client())

    results = [
        json.loads(register.nemo_api(resource, action, params, workspace="default"))
        for resource, action, params in calls
    ]

    def names(result: Any) -> Any:
        if isinstance(result, dict) and "data" in result:
            return [item["name"] for item in result["data"]]
        return result["name"] if isinstance(result, dict) else result

    assert [names(result) for result in results] == expected_results
    assert platform.mutations() == []


def test_mutating_nemo_api_requires_studio_session(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = _ok_helix("POST", _WORKSPACES, _workspace("demo"), status=201)
    _install_client(monkeypatch, platform.client())

    response = register.nemo_api("workspaces", "create", '{"name": "demo"}', workspace="default")

    assert response.startswith("Denied:")
    assert platform.requests == []


def test_mutating_nemo_api_uses_studio_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    session_id = "90a877d5-19f6-49a8-bf09-d0020ae0833a"
    approvals: list[tuple[str, str, dict[str, object], dict[str, object]]] = []

    def approve(session: str, tool_name: str, arguments: dict[str, object], **kwargs: object) -> dict[str, str]:
        approvals.append((session, tool_name, arguments, kwargs))
        return {"behavior": "allow"}

    platform = _ok_helix("POST", _WORKSPACES, _workspace("demo"), status=201)
    _install_client(monkeypatch, platform.client())
    monkeypatch.setattr(register, "_call_studio_tool", approve)

    response = json.loads(
        register.nemo_api("workspaces", "create", '{"name": "demo"}', studio_session_id=session_id, workspace="default")
    )

    assert response["name"] == "demo"
    assert json.loads(platform.requests[0].content) == {"name": "demo"}
    assert approvals[0][0:2] == (session_id, "approval_prompt")
    assert approvals[0][2]["input"] == {
        "resource": "workspaces",
        "action": "create",
        "params": '{"name": "demo"}',
        "workspace": "default",
    }
    assert approvals[0][3]["workspace"] == "default"


def test_mutating_nemo_api_uses_edited_approved_input(monkeypatch: pytest.MonkeyPatch) -> None:
    session_id = "90a877d5-19f6-49a8-bf09-d0020ae0833a"

    def approve(_session: str, _tool_name: str, arguments: dict[str, object], **_kwargs: object) -> dict[str, object]:
        input_value = arguments["input"]
        assert isinstance(input_value, dict)
        updated_input = dict(input_value)
        updated_input["params"] = '{"name": "approved"}'
        return {"behavior": "allow", "updatedInput": updated_input}

    platform = _ok_helix("POST", _WORKSPACES, _workspace("approved"), status=201)
    _install_client(monkeypatch, platform.client())
    monkeypatch.setattr(register, "_call_studio_tool", approve)

    response = json.loads(
        register.nemo_api(
            "workspaces", " CREATE ", '{"name": "original"}', studio_session_id=session_id, workspace="default"
        )
    )

    assert response["name"] == "approved"
    assert json.loads(platform.requests[0].content) == {"name": "approved"}


def test_mutating_nemo_api_revalidates_approved_guardrail_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight_calls: list[tuple[str, dict[str, object] | None]] = []

    def approve(_session: str, _tool_name: str, arguments: dict[str, object], **_kwargs: object) -> dict[str, object]:
        input_value = arguments["input"]
        assert isinstance(input_value, dict)
        return {
            "behavior": "allow",
            "updatedInput": {
                **input_value,
                "resource": "guardrail",
                "action": "check",
                "params": '{"model":"default/model-a","messages":[]}',
            },
        }

    platform = _FakeHelix(
        {
            ("POST", _WORKSPACES): _json_route(_workspace("original"), 201),
            ("POST", f"{_GUARDRAILS}/checks"): _json_route({"status": "unknown"}),
        }
    )
    _install_client(monkeypatch, platform.client())
    monkeypatch.setattr(register, "_call_studio_tool", approve)
    monkeypatch.setattr(
        register,
        "_preflight_guardrail_model",
        lambda _client, workspace, params, _key: preflight_calls.append((workspace, params)),
    )

    response = register.nemo_api(
        "workspaces",
        "create",
        '{"name":"original"}',
        studio_session_id="90a877d5-19f6-49a8-bf09-d0020ae0833a",
        workspace="default",
    )

    assert preflight_calls == [("default", {"model": "default/model-a", "messages": []})]
    assert platform.calls("POST", _WORKSPACES) == []
    assert "unexpected status 'unknown'" in response
    assert "Do not attach this unvalidated guardrail" in response


def test_mutating_nemo_api_rejects_approved_workspace_change(monkeypatch: pytest.MonkeyPatch) -> None:
    def approve(_session: str, _tool_name: str, arguments: dict[str, object], **_kwargs: object) -> dict[str, object]:
        input_value = arguments["input"]
        assert isinstance(input_value, dict)
        updated_input = dict(input_value)
        updated_input["workspace"] = "other"
        return {"behavior": "allow", "updatedInput": updated_input}

    platforms = {"default": _ok_helix("POST", _WORKSPACES), "other": _ok_helix("POST", _WORKSPACES)}
    monkeypatch.setattr(
        register, "_clients", {workspace: platform.client(workspace) for workspace, platform in platforms.items()}
    )
    monkeypatch.setattr(register, "_call_studio_tool", approve)

    response = register.nemo_api(
        "workspaces",
        "create",
        '{"name": "demo"}',
        studio_session_id="90a877d5-19f6-49a8-bf09-d0020ae0833a",
        workspace="default",
    )

    assert response == "Denied: mutation approval cannot change the request workspace"
    assert all(platform.requests == [] for platform in platforms.values())


@pytest.mark.parametrize(("field", "value"), [("resource", []), ("action", 7)])
def test_mutating_nemo_api_rejects_non_string_approved_paths(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    def approve(_session: str, _tool_name: str, arguments: dict[str, object], **_kwargs: object) -> dict[str, object]:
        input_value = arguments["input"]
        assert isinstance(input_value, dict)
        updated_input = dict(input_value)
        updated_input[field] = value
        return {"behavior": "allow", "updatedInput": updated_input}

    monkeypatch.setattr(register, "_call_studio_tool", approve)

    response = register.nemo_api(
        "workspaces",
        "create",
        '{"name": "demo"}',
        studio_session_id="90a877d5-19f6-49a8-bf09-d0020ae0833a",
        workspace="default",
    )

    assert response == f"Error: ValueError: approved {field} must be a string"


def test_mutating_nemo_api_validates_params_before_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    approval_requested = False

    def approve(*_args: object, **_kwargs: object) -> dict[str, str]:
        nonlocal approval_requested
        approval_requested = True
        return {"behavior": "allow"}

    monkeypatch.setattr(register, "_call_studio_tool", approve)

    response = register.nemo_api(
        "workspaces",
        "create",
        '["not", "an", "object"]',
        studio_session_id="90a877d5-19f6-49a8-bf09-d0020ae0833a",
        workspace="default",
    )

    assert response == "Error: ValueError: params must decode to a JSON object"
    assert approval_requested is False


def test_nemo_api_rejects_non_object_params(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_client(monkeypatch, _FakeHelix().client())

    response = register.nemo_api("workspaces", "list", '["not", "an", "object"]', workspace="default")

    assert response == "Error: ValueError: params must decode to a JSON object"


def test_nemo_api_uses_request_workspace_for_each_client(monkeypatch: pytest.MonkeyPatch) -> None:
    created_workspaces: list[str] = []
    seen_paths: list[str] = []

    def build_client(*, base_url: str, workspace: str) -> NemoClient:
        created_workspaces.append(workspace)

        def handle(request: httpx.Request) -> httpx.Response:
            seen_paths.append(request.url.path)
            return _page([_entity(f"model-in-{workspace}", workspace=workspace)])

        return mock_nemo_client(handle, NemoClient, workspace=workspace)

    monkeypatch.setenv("NHX_BASE_URL", "http://platform:8080")
    monkeypatch.setattr(register, "_clients", {})
    monkeypatch.setattr(register, "build_direct_nemo_client", build_client)

    first = json.loads(register.nemo_api("models", "list", workspace="first"))
    second = json.loads(register.nemo_api("models", "list", workspace="second"))

    assert first["data"][0]["name"] == "model-in-first"
    assert second["data"][0]["name"] == "model-in-second"
    assert created_workspaces == ["first", "second"]
    assert seen_paths == [
        "/apis/models/v2/workspaces/first/models",
        "/apis/models/v2/workspaces/second/models",
    ]


def test_nemo_api_requires_request_workspace() -> None:
    response = register.nemo_api("models", "list")

    assert response == "Clarification required: which workspace should this operation use?"


def test_get_client_prefers_platform_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    created_with: dict[str, object] = {}

    def fake_client(**kwargs: object) -> object:
        created_with.update(kwargs)
        return object()

    monkeypatch.setenv("NHX_BASE_URL", "http://platform:8080")
    monkeypatch.setenv("NEMO_BASE_URL", "http://model-gateway:8000")
    monkeypatch.setattr(register, "build_direct_nemo_client", fake_client)
    monkeypatch.setattr(register, "_clients", {})

    register._get_client("default")

    assert created_with == {"workspace": "default", "base_url": "http://platform:8080"}


def test_get_client_reads_the_active_cli_context_without_a_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    created_with: dict[str, object] = {}

    def fake_client(**kwargs: object) -> object:
        created_with.update(kwargs)
        return object()

    monkeypatch.delenv("NHX_BASE_URL", raising=False)
    monkeypatch.delenv("NEMO_BASE_URL", raising=False)
    monkeypatch.setattr(register, "build_nemo_client", fake_client)
    monkeypatch.setattr(register, "_clients", {})

    register._get_client("default")

    assert created_with == {"workspace": "default"}


_JOB_STATUS = {
    "id": "job-id",
    "name": "job-1",
    "status": "completed",
    "status_details": {},
    "error_details": None,
    "steps": [],
    "created_at": _TIMESTAMP,
    "updated_at": _TIMESTAMP,
}


@pytest.mark.parametrize(
    ("service", "path", "payload"),
    [
        ("evals", "/apis/evals/v2/workspaces/default/evaluate/jobs/job-1/status", _JOB_STATUS),
        ("data_designer", "/apis/data-designer/v2/workspaces/default/jobs/create/job-1/status", _JOB_STATUS),
        ("auditor", "/apis/auditor/v2/workspaces/default/jobs/audit/job-1", {"name": "job-1", "status": "completed"}),
        ("customization.automodel", "/apis/jobs/v2/workspaces/default/jobs/job-1/status", _JOB_STATUS),
    ],
)
def test_check_status_reads_the_service_job(
    monkeypatch: pytest.MonkeyPatch, service: str, path: str, payload: dict[str, Any]
) -> None:
    platform = _ok_helix("GET", path, payload)
    _install_client(monkeypatch, platform.client())

    response = json.loads(register.check_status(service, "job-1", workspace="default"))

    assert response["name"] == "job-1"
    assert response["status"] == "completed"
    assert [request.url.path for request in platform.requests] == [path]


def test_check_status_reports_platform_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_client(monkeypatch, _FakeHelix().client())

    response = register.check_status("evals", "missing", workspace="default")

    assert response.startswith("Error: NotFoundError:")


def test_studio_callback_url_validates_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NHX_BASE_URL", "http://platform:8080")

    with pytest.raises(ValueError, match="valid Studio session"):
        register._studio_callback_url("../../bad")


def test_studio_callback_url_uses_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NHX_BASE_URL", "http://platform:8080")
    session_id = "90a877d5-19f6-49a8-bf09-d0020ae0833a"

    url = register._studio_callback_url(session_id, workspace="demo")

    assert url == f"http://platform:8080/studio/api/assistant/mcp/{session_id}?workspace=demo"


def test_ask_user_question_rejects_invalid_payload() -> None:
    response = register.ask_user_question("90a877d5-19f6-49a8-bf09-d0020ae0833a", "{}")

    assert response == "Error: `questions` must be a non-empty JSON array of question objects."


def test_ask_user_question_accepts_structured_model_input(monkeypatch: pytest.MonkeyPatch) -> None:
    questions = [
        {
            "question": "Which workspace?",
            "header": "Workspace",
            "options": [{"label": "default", "description": "Use default"}],
        }
    ]
    callback_input: dict[str, object] = {}

    def callback(_session_id: str, _tool_name: str, arguments: dict[str, object]) -> dict[str, object]:
        callback_input.update(arguments)
        return {"behavior": "allow", "updatedInput": {"workspace": "default"}}

    monkeypatch.setattr(register, "_call_studio_tool", callback)

    response = register.ask_user_question("90a877d5-19f6-49a8-bf09-d0020ae0833a", questions)

    assert json.loads(response) == {"workspace": "default"}
    assert callback_input == {"tool_name": "AskUserQuestion", "input": {"questions": questions}}
