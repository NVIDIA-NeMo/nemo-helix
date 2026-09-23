# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Studio Assistant tool implementations exposed through MCP."""

import json
import os
import re
import uuid
from typing import Annotated, Any
from urllib.parse import quote, urlencode

import httpx
from nemo_helix import NeMoHelix
from pydantic import BaseModel, Field

DEFAULT_WORKSPACE = "default"
STUDIO_CALLBACK_PATH = "/studio/api/assistant/mcp/{session_id}"
STUDIO_CALLBACK_TIMEOUT_SECONDS = 3600.0
_READ_ONLY_SDK_ACTIONS = frozenset({"check", "get", "get_logs", "get_status", "list", "read", "retrieve", "search"})
_API_ERROR_LIMIT = 3
_GUARDRAIL_CHECK_FAILURE_LIMIT = 3
_REFUSAL_LIKE_PROBE_RE = re.compile(
    r"^\s*(?:(?:i(?:'m| am)|we(?:'re| are))\s+sorry[,;:]?\s*(?:but\s+)?)?"
    r"(?:(?:i|we)\s+(?:can't|cannot|won't|will not|am unable to|are unable to)|"
    r"(?:i|we)\s+(?:must|have to)\s+(?:decline|refuse)|"
    r"(?:unable to|can't|cannot|won't|refuse to|decline to))\b",
    re.IGNORECASE,
)

_clients: dict[str, NeMoHelix] = {}
_api_error_streaks: dict[str, int] = {}
_guardrail_check_failures: dict[str, int] = {}
_preflighted_guardrail_models: set[tuple[str, str, str]] = set()
_guardrail_deployment_results: dict[tuple[str, str], str | None] = {}


class SDKPathError(ValueError):
    """An invalid public SDK resource or action selected by the agent."""


class ModelPreflightError(RuntimeError):
    """The model selected for a guardrail check is not currently reachable."""


class GuardrailWorkflowError(RuntimeError):
    """A deterministic guardrail deployment gate failed."""


def _active_workspace() -> str:
    return os.environ.get("NHX_WORKSPACE") or DEFAULT_WORKSPACE


def _get_client(workspace: str) -> NeMoHelix:
    request_workspace = workspace.strip()
    if not request_workspace:
        raise ValueError("workspace is required")
    if request_workspace not in _clients:
        base_url = os.environ.get("NHX_BASE_URL") or os.environ.get("NEMO_BASE_URL")
        kwargs: dict[str, Any] = {"workspace": request_workspace}
        if base_url:
            kwargs["base_url"] = base_url
        _clients[request_workspace] = NeMoHelix(**kwargs)
    return _clients[request_workspace]


def _serialize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {key: _serialize(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    return str(obj)


def _public_members(value: Any) -> list[str]:
    return sorted(name for name in dir(value) if not name.startswith("_"))


def _resolve_resource(client: NeMoHelix, resource_path: str) -> Any:
    current = client
    for part in resource_path.split("."):
        if not part or part.startswith("_"):
            raise SDKPathError(f"Invalid SDK resource path: {resource_path!r}")
        if not hasattr(current, part):
            available = ", ".join(_public_members(current)) or "none"
            raise SDKPathError(
                f"Invalid SDK resource {resource_path!r}: {part!r} does not exist. "
                f"Available members at this level: {available}."
            )
        current = getattr(current, part)
    return current


def _resolve_sdk_method(resource: Any, resource_path: str, action: str) -> Any:
    if not action or action.startswith("_"):
        raise SDKPathError(f"Invalid SDK action: {action!r}")
    if not hasattr(resource, action) or not callable(method := getattr(resource, action)):
        available = ", ".join(_public_members(resource)) or "none"
        guardrail_hint = (
            " Guardrail config CRUD uses resource='guardrail.configs'; resource='guardrail' is only for action='check'."
            if resource_path == "guardrail"
            else ""
        )
        raise SDKPathError(
            f"Invalid SDK action {action!r} for resource {resource_path!r}. "
            f"Available members: {available}.{guardrail_hint}"
        )
    return method


def _call_sdk_method(resource: Any, resource_path: str, action: str, params: dict[str, Any] | None = None) -> Any:
    method = _resolve_sdk_method(resource, resource_path, action)
    return method(**params) if params else method()


def _api_error_key(workspace: str, studio_session_id: str | None) -> str:
    return studio_session_id or f"workspace:{workspace}"


def _record_api_error(key: str, exc: Exception) -> str:
    count = _api_error_streaks.get(key, 0) + 1
    _api_error_streaks[key] = count
    if count >= _API_ERROR_LIMIT:
        return (
            f"Error circuit breaker: {count} consecutive nemo_api calls failed. "
            "Stop retrying or guessing SDK paths and parameters; report this failure to the user. "
            f"Last error: {exc}"
        )
    return f"Error: {type(exc).__name__}: {exc}"


def _is_guardrail_check(resource: str, action: str) -> bool:
    return resource.strip() == "guardrail" and action.strip().lower() == "check"


def _record_guardrail_check_failure(key: str, message: str) -> str:
    count = _guardrail_check_failures.get(key, 0) + 1
    _guardrail_check_failures[key] = count
    prefix = "Guardrail validation stopped"
    if count >= _GUARDRAIL_CHECK_FAILURE_LIMIT:
        return (
            f"{prefix}: {count} validation attempts failed. Stop retrying or selecting another model. "
            "Report any guardrail configuration already created as a partial success, do not attach it to a "
            f"VirtualModel, and tell the user how to restore model connectivity. Last error: {message}"
        )
    return (
        f"{prefix}: {message} Do not attach this unvalidated guardrail to a VirtualModel. "
        f"Validation failure {count} of {_GUARDRAIL_CHECK_FAILURE_LIMIT}."
    )


def _normalize_workspace(workspace: str | None, params: dict[str, Any] | None) -> str | None:
    if workspace is not None and workspace.strip():
        return workspace.strip()
    params_workspace = params.get("workspace") if params else None
    if isinstance(params_workspace, str) and params_workspace.strip():
        return params_workspace.strip()
    return None


def _guardrail_model_route(model: Any, workspace: str) -> tuple[str, str]:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("guardrail check params must include a model")
    qualified_model = model.strip()
    if "/" not in qualified_model:
        return qualified_model, qualified_model
    model_workspace, model_name = qualified_model.split("/", 1)
    if not model_workspace or not model_name:
        raise ValueError("guardrail check model must be '<workspace>/<model>'")
    if model_workspace != workspace:
        raise ValueError(
            f"guardrail check model workspace {model_workspace!r} does not match request workspace {workspace!r}"
        )
    return qualified_model, model_name


def _preflight_guardrail_model(
    client: NeMoHelix,
    workspace: str,
    params: dict[str, Any] | None,
    error_key: str,
) -> None:
    if params is None:
        raise ValueError("guardrail check params are required")
    qualified_model, model_name = _guardrail_model_route(params.get("model"), workspace)
    cache_key = (error_key, workspace, qualified_model)
    if cache_key in _preflighted_guardrail_models:
        return
    try:
        client.inference.gateway.model.post(
            workspace=workspace,
            name=model_name,
            trailing_uri="v1/chat/completions",
            body={
                "model": qualified_model,
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 1,
                "temperature": 0,
            },
        )
    except Exception as exc:
        raise ModelPreflightError(
            f"model {qualified_model!r} is unavailable ({type(exc).__name__}: {exc}). "
            "Restore its provider connection or select a reachable model"
        ) from exc
    _preflighted_guardrail_models.add(cache_key)


def _optional_retrieve(method: Any, name: str, workspace: str) -> Any | None:
    try:
        return method(name, workspace=workspace)
    except Exception as exc:
        if getattr(exc, "status_code", None) == 404:
            return None
        raise


def _guardrail_config_data(policy: str) -> dict[str, Any]:
    return {
        "rails": {"input": {"flows": ["self check input"]}},
        "prompts": [
            {
                "task": "self_check_input",
                "content": (
                    "Check whether the user message violates this policy:\n"
                    f"{policy}\n\n"
                    "User message: {{ user_input }}\n\n"
                    "Should this message be blocked (Yes or No)?\nAnswer:"
                ),
            }
        ],
    }


def _guardrail_virtual_model_data(
    workspace: str,
    backend_model: str,
    config_name: str,
) -> dict[str, Any]:
    return {
        "default_model_entity": backend_model,
        "models": [{"model": backend_model, "backend_format": "OPENAI_CHAT"}],
        "request_middleware": [
            {
                "name": "nemo-guardrails",
                "config_type": "guardrail_config",
                "config_id": f"{workspace}/{config_name}",
            }
        ],
        "response_middleware": [],
        "post_response_middleware": [],
    }


def _contains_expected_fields(value: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(value, dict) and all(
            key in value and _contains_expected_fields(value[key], expected_value)
            for key, expected_value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(value, list)
            and len(value) == len(expected)
            and all(
                _contains_expected_fields(actual_item, expected_item)
                for actual_item, expected_item in zip(value, expected, strict=True)
            )
        )
    return value == expected


def _matches_fields(value: Any, expected: dict[str, Any]) -> bool:
    return _contains_expected_fields(_serialize(value), expected)


def _report_workflow_activity(
    studio_session_id: str,
    step: str,
    *,
    status: str = "completed",
    detail: str | None = None,
    workspace: str,
) -> None:
    try:
        _call_studio_tool(
            studio_session_id,
            "assistant_activity",
            {
                "tool_name": "deploy_guardrail",
                "status": status,
                "step": step,
                "detail": detail,
            },
            workspace=workspace,
        )
    except Exception:
        # Activity is presentation-only and must never change the platform result.
        return


def _guardrail_check(
    client: NeMoHelix,
    *,
    workspace: str,
    backend_model: str,
    config_name: str,
    message: str,
    expected_status: str,
) -> dict[str, Any]:
    result = client.guardrail.check(
        workspace=workspace,
        model=backend_model,
        messages=[{"role": "user", "content": message}],
        guardrails={"config_id": f"{workspace}/{config_name}"},
        max_tokens=50,
        temperature=0,
    )
    serialized = _serialize(result)
    status = serialized.get("status") if isinstance(serialized, dict) else None
    if status != expected_status:
        raise GuardrailWorkflowError(
            f"expected guardrail check status {expected_status!r} for {message!r}, received {status!r}"
        )
    return serialized


def _routable_virtual_model(client: NeMoHelix, workspace: str, virtual_model_name: str) -> bool:
    expected = f"{workspace}/{virtual_model_name}"
    models = client.inference.gateway.openai.v1.models.list(workspace=workspace)
    for model in models:
        value = _serialize(model)
        if isinstance(value, dict) and value.get("id") == expected:
            return True
    return False


def _validate_guardrail_probe_messages(blocked_message: str, allowed_message: str) -> None:
    normalized_blocked_message = blocked_message.replace("’", "'").strip()
    if _REFUSAL_LIKE_PROBE_RE.match(normalized_blocked_message):
        raise ValueError(
            "blocked_message must be an actual policy-violating user request, not refusal-like assistant output"
        )
    if normalized_blocked_message.casefold() == allowed_message.strip().casefold():
        raise ValueError("blocked_message and allowed_message must be different representative user requests")


def _validated_session_id(session_id: str) -> str:
    try:
        return str(uuid.UUID(session_id))
    except (TypeError, ValueError) as exc:
        raise ValueError("A valid Studio session id is required") from exc


def _studio_callback_url(
    session_id: str,
    *,
    workspace: str | None = None,
    studio_base_url: str | None = None,
) -> str:
    base_url = (os.environ.get("NHX_BASE_URL") or os.environ.get("NEMO_BASE_URL") or "").rstrip("/")
    if not base_url:
        raise RuntimeError("NHX_BASE_URL is required for Studio UI callbacks")

    query = {
        key: value
        for key, value in {
            "workspace": workspace or _active_workspace(),
            "studio_base_url": studio_base_url,
        }.items()
        if value
    }
    url = f"{base_url}{STUDIO_CALLBACK_PATH.format(session_id=_validated_session_id(session_id))}"
    return f"{url}?{urlencode(query)}" if query else url


def _json_rpc_content(payload: dict[str, Any]) -> dict[str, Any]:
    error = payload.get("error")
    if isinstance(error, dict):
        raise RuntimeError(str(error.get("message") or "Studio callback failed"))
    result = payload.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("Studio callback response did not include a result")
    content = result.get("content")
    if not isinstance(content, list) or not content:
        return {}
    first = content[0]
    if not isinstance(first, dict) or not isinstance(first.get("text"), str):
        return {}
    decoded = json.loads(first["text"])
    if not isinstance(decoded, dict):
        raise RuntimeError("Studio callback tool returned a non-object result")
    return decoded


def _parse_studio_callback_response(response: httpx.Response) -> dict[str, Any]:
    response.raise_for_status()
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        for line in reversed(response.text.splitlines()):
            if line.startswith("data:"):
                payload = json.loads(line.removeprefix("data:").strip())
                if isinstance(payload, dict):
                    return _json_rpc_content(payload)
        raise RuntimeError("Studio callback stream ended without a tool result")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Studio callback returned a non-object response")
    return _json_rpc_content(payload)


def _call_studio_tool(
    session_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    workspace: str | None = None,
    studio_base_url: str | None = None,
) -> dict[str, Any]:
    body = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    with httpx.Client(timeout=STUDIO_CALLBACK_TIMEOUT_SECONDS) as client:
        response = client.post(
            _studio_callback_url(session_id, workspace=workspace, studio_base_url=studio_base_url), json=body
        )
    return _parse_studio_callback_response(response)


def _request_mutation_approval(
    studio_session_id: str | None,
    *,
    tool_name: str,
    tool_input: dict[str, Any],
    workspace: str,
) -> dict[str, Any] | str:
    if studio_session_id is None:
        return "Denied: mutating operations require a Studio session id and explicit approval"
    try:
        session_id = _validated_session_id(studio_session_id)
    except ValueError:
        return "Denied: mutating operations require a valid Studio session id and explicit approval"
    approval = _call_studio_tool(
        session_id,
        "approval_prompt",
        {"tool_name": tool_name, "input": tool_input},
        workspace=workspace,
    )
    if approval.get("behavior") != "allow":
        return f"Denied by user: {approval.get('message') or 'operation was not approved'}"
    updated_input = approval.get("updatedInput")
    return updated_input if isinstance(updated_input, dict) else tool_input


def deploy_guardrail(
    policy: str,
    config_name: str,
    virtual_model_name: str,
    blocked_message: Annotated[
        str,
        Field(
            description=(
                "An actual end-user request that directly violates the policy and must be blocked. "
                "Never pass predicted refusal text or assistant output."
            )
        ),
    ],
    allowed_message: Annotated[
        str,
        Field(description="A distinct, policy-compliant end-user request that must remain allowed."),
    ],
    studio_session_id: str,
    deployment_run_id: Annotated[
        str,
        Field(
            description=(
                "The Studio-provided id for this user request. Pass the exact value from the Studio context; "
                "it prevents duplicate deployment attempts."
            )
        ),
    ],
    backend_model: str | None = None,
    workspace: str | None = None,
    description: str | None = None,
) -> str:
    """Create, validate, and deploy one input guardrail with one approval.

    This is the deterministic fast path for a new input-only self-check policy.
    It refuses to overwrite resources, requires both a blocked and allowed check,
    and reports the resulting VirtualModel's current routing status without
    blocking the assistant while gateway propagation completes.
    """
    try:
        deployment_key = (_validated_session_id(studio_session_id), str(uuid.UUID(deployment_run_id)))
    except ValueError:
        return json.dumps(
            {"status": "failed", "error": "valid studio_session_id and deployment_run_id values are required"}
        )
    if deployment_key in _guardrail_deployment_results:
        previous_result = _guardrail_deployment_results[deployment_key]
        if previous_result is not None:
            return previous_result
        return json.dumps(
            {
                "status": "failed",
                "error": "duplicate guardrail deployment attempt prevented; report the original result and stop",
            }
        )
    _guardrail_deployment_results[deployment_key] = None

    def finish(payload: dict[str, Any], *, indent: int | None = None) -> str:
        result = json.dumps(payload, indent=indent)
        _guardrail_deployment_results[deployment_key] = result
        return result

    requested_workspace = workspace.strip() if workspace and workspace.strip() else None
    if requested_workspace is None:
        return finish({"status": "failed", "error": "workspace is required"})

    values = {
        "policy": policy.strip(),
        "config_name": config_name.strip(),
        "virtual_model_name": virtual_model_name.strip(),
        "blocked_message": blocked_message.strip(),
        "allowed_message": allowed_message.strip(),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        return finish({"status": "failed", "error": f"required values are empty: {', '.join(missing)}"})
    try:
        _validate_guardrail_probe_messages(values["blocked_message"], values["allowed_message"])
    except ValueError as exc:
        return finish({"status": "failed", "error": str(exc)})

    selected_backend_model = backend_model.strip() if backend_model and backend_model.strip() else None
    if selected_backend_model is None:
        try:
            selection = _call_studio_tool(
                studio_session_id,
                "select_model",
                {
                    "title": "Select the guardrail model",
                    "description": "Choose the backend model for policy checks and the guarded VirtualModel.",
                    "output_key": "model",
                    "display_label": "Backend model",
                    "required_message": "Select a model to deploy this guardrail.",
                    "submit_label": "Use model",
                },
                workspace=requested_workspace,
            )
        except Exception as exc:
            return finish({"status": "failed", "error": f"model selection failed: {type(exc).__name__}: {exc}"})
        selected = selection.get("model")
        if not isinstance(selected, str) or not selected.strip():
            return finish({"status": "failed", "error": "a backend model was not selected"})
        selected_backend_model = selected.strip()
    values["backend_model"] = selected_backend_model

    try:
        model_workspace, model_name = values["backend_model"].split("/", 1)
    except ValueError:
        return finish({"status": "failed", "error": "backend_model must be '<workspace>/<model>'"})
    if not model_workspace or not model_name or model_workspace != requested_workspace:
        return finish(
            {
                "status": "failed",
                "error": (
                    f"backend_model workspace must match {requested_workspace!r}; received {values['backend_model']!r}"
                ),
            }
        )

    validation: dict[str, str] = {}
    config_ready = False
    virtual_model_created = False
    chat_model: str | None = None
    virtual_model_link: str | None = None

    try:
        client = _get_client(requested_workspace)
        config_data = _guardrail_config_data(values["policy"])
        virtual_model_data = _guardrail_virtual_model_data(
            requested_workspace,
            values["backend_model"],
            values["config_name"],
        )
        config_description = (
            description.strip() if description and description.strip() else f"Input guardrail: {values['policy']}"
        )
        existing_config = _optional_retrieve(
            client.guardrail.configs.retrieve,
            values["config_name"],
            requested_workspace,
        )
        if existing_config is not None and not _matches_fields(existing_config, {"data": config_data}):
            raise GuardrailWorkflowError(
                f"GuardrailConfig {requested_workspace}/{values['config_name']} already exists with different data"
            )

        existing_virtual_model = _optional_retrieve(
            client.inference.virtual_models.retrieve,
            values["virtual_model_name"],
            requested_workspace,
        )
        if existing_virtual_model is not None and not _matches_fields(existing_virtual_model, virtual_model_data):
            raise GuardrailWorkflowError(
                f"VirtualModel {requested_workspace}/{values['virtual_model_name']} already exists with different data"
            )

        mutation_input = {
            "workspace": requested_workspace,
            **values,
            "description": config_description,
            "create_config": existing_config is None,
            "create_virtual_model": existing_virtual_model is None,
        }
        if existing_config is None or existing_virtual_model is None:
            approved_input = _request_mutation_approval(
                studio_session_id,
                tool_name="deploy_guardrail",
                tool_input=mutation_input,
                workspace=requested_workspace,
            )
            if isinstance(approved_input, str):
                return finish({"status": "denied", "error": approved_input})
            if approved_input != mutation_input:
                return finish(
                    {
                        "status": "denied",
                        "error": "Approval edits are not supported for this workflow; submit the edited request again",
                    }
                )

        if existing_config is None:
            client.guardrail.configs.create(
                workspace=requested_workspace,
                name=values["config_name"],
                description=config_description,
                data=config_data,
            )
        config_ready = True
        _report_workflow_activity(
            studio_session_id,
            "config_ready",
            detail=f"GuardrailConfig {requested_workspace}/{values['config_name']} is ready.",
            workspace=requested_workspace,
        )

        _guardrail_check(
            client,
            workspace=requested_workspace,
            backend_model=values["backend_model"],
            config_name=values["config_name"],
            message=values["blocked_message"],
            expected_status="blocked",
        )
        validation["blocked_message"] = "blocked"
        _report_workflow_activity(
            studio_session_id,
            "blocked_check",
            detail="The representative policy-violating message was blocked.",
            workspace=requested_workspace,
        )

        _guardrail_check(
            client,
            workspace=requested_workspace,
            backend_model=values["backend_model"],
            config_name=values["config_name"],
            message=values["allowed_message"],
            expected_status="success",
        )
        validation["allowed_message"] = "success"
        _report_workflow_activity(
            studio_session_id,
            "allowed_check",
            detail="The representative allowed message passed.",
            workspace=requested_workspace,
        )

        if existing_virtual_model is None:
            client.inference.virtual_models.create(
                workspace=requested_workspace,
                name=values["virtual_model_name"],
                **virtual_model_data,
            )
        created_virtual_model = client.inference.virtual_models.retrieve(
            values["virtual_model_name"], workspace=requested_workspace
        )
        if not _matches_fields(created_virtual_model, virtual_model_data):
            raise GuardrailWorkflowError("VirtualModel readback did not match the approved deployment")
        virtual_model_created = True
        chat_model = f"{requested_workspace}/{values['virtual_model_name']}"
        virtual_model_query = urlencode({"virtualModel": values["virtual_model_name"], "tab": "chat"})
        virtual_model_path = f"/workspaces/{quote(requested_workspace, safe='')}/virtual-models?{virtual_model_query}"
        virtual_model_link = f"[Chat with VirtualModel {chat_model}]({virtual_model_path})"
        routing_warning: str | None = None
        if not _routable_virtual_model(client, requested_workspace, values["virtual_model_name"]):
            routing_warning = (
                f"VirtualModel {chat_model} was created and verified, but routing is still propagating. "
                "Open the chat link in a moment; do not redeploy it."
            )
            _report_workflow_activity(
                studio_session_id,
                "virtual_model_pending",
                detail=routing_warning,
                workspace=requested_workspace,
            )
        else:
            _report_workflow_activity(
                studio_session_id,
                "virtual_model_ready",
                detail=f"VirtualModel {chat_model} is routable and ready for chat.",
                workspace=requested_workspace,
            )
        return finish(
            {
                "status": "success",
                "config": f"{requested_workspace}/{values['config_name']}",
                "virtual_model": chat_model,
                "validation": validation,
                "chat_model": chat_model,
                "studio_link": virtual_model_link,
                "routable": routing_warning is None,
                "warning": routing_warning,
            },
            indent=2,
        )
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        _report_workflow_activity(
            studio_session_id,
            "failed",
            status="failed",
            detail=detail,
            workspace=requested_workspace,
        )
        return finish(
            {
                "status": "partial" if config_ready else "failed",
                "config": f"{requested_workspace}/{values['config_name']}" if config_ready else None,
                "virtual_model": (
                    f"{requested_workspace}/{values['virtual_model_name']}" if virtual_model_created else None
                ),
                "validation": validation,
                "chat_model": chat_model,
                "studio_link": virtual_model_link,
                "routable": False if virtual_model_created else None,
                "error": detail,
            },
            indent=2,
        )


def _parse_nemo_api_params(params: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if params is None or params == "":
        return None
    parsed_params: Any
    if isinstance(params, str):
        parsed_params = json.loads(params)
    elif isinstance(params, dict):
        parsed_params = dict(params)
    else:
        raise ValueError("params must decode to a JSON object")
    if not isinstance(parsed_params, dict):
        raise ValueError("params must decode to a JSON object")
    return parsed_params


def nemo_api(
    resource: str,
    action: str,
    params: str | dict[str, Any] | None = None,
    studio_session_id: str | None = None,
    workspace: str | None = None,
) -> str:
    """Call a NeMo Helix SDK method; writes require explicit Studio approval.

    ``resource`` is a dot-separated SDK path such as ``workspaces``,
    ``inference.providers``, ``files.filesets``, ``evaluation.metric_jobs``,
    ``guardrail.configs``, ``secrets``, ``models`` or ``datasets``. Use
    ``resource='guardrail.configs'`` for guardrail config CRUD and
    ``resource='guardrail', action='check'`` only for standalone checks. ``action`` is the
    SDK method name. ``params`` is an optional JSON object or JSON object string
    containing keyword arguments. Pass the active request workspace for every operation.
    Pass the Studio session id from the user context for create, update, delete,
    submit, upload, cancel, or other mutating actions.
    """
    parsed_params: dict[str, Any] | None = None
    is_guardrail_check = _is_guardrail_check(resource, action)
    try:
        parsed_params = _parse_nemo_api_params(params)
        workspace = _normalize_workspace(workspace, parsed_params)
        error_key = _api_error_key(workspace or "", studio_session_id)
        if is_guardrail_check and _guardrail_check_failures.get(error_key, 0) >= _GUARDRAIL_CHECK_FAILURE_LIMIT:
            return _record_guardrail_check_failure(
                error_key,
                "the validation retry limit was already reached for this request",
            )
        if workspace is None:
            message = "which workspace should this operation use?"
            if is_guardrail_check:
                return _record_guardrail_check_failure(error_key, message)
            return f"Clarification required: {message}"
        normalized_action = action.strip().lower()
        client = _get_client(workspace)
        sdk_resource = _resolve_resource(client, resource)
        _resolve_sdk_method(sdk_resource, resource, normalized_action)
        if is_guardrail_check:
            _preflight_guardrail_model(client, workspace, parsed_params, error_key)
        if normalized_action not in _READ_ONLY_SDK_ACTIONS:
            approved_input = _request_mutation_approval(
                studio_session_id,
                tool_name="nemo_api",
                tool_input={"resource": resource, "action": action, "params": params, "workspace": workspace},
                workspace=workspace,
            )
            if isinstance(approved_input, str):
                return approved_input
            if approved_input.get("workspace", workspace) != workspace:
                return "Denied: mutation approval cannot change the request workspace"
            approved_resource = approved_input.get("resource", resource)
            approved_action = approved_input.get("action", action)
            if not isinstance(approved_resource, str):
                raise ValueError("approved resource must be a string")
            if not isinstance(approved_action, str):
                raise ValueError("approved action must be a string")
            resource = approved_resource
            action = approved_action
            approved_params = approved_input.get("params", params)
            if approved_params != params:
                params = approved_params
                parsed_params = _parse_nemo_api_params(params)
            normalized_action = action.strip().lower()
            is_guardrail_check = _is_guardrail_check(resource, action)
            sdk_resource = _resolve_resource(client, resource)
            _resolve_sdk_method(sdk_resource, resource, normalized_action)
            if is_guardrail_check:
                _preflight_guardrail_model(client, workspace, parsed_params, error_key)
        result = _call_sdk_method(sdk_resource, resource, normalized_action, parsed_params)
        if is_guardrail_check:
            serialized_result = _serialize(result)
            status = serialized_result.get("status") if isinstance(serialized_result, dict) else None
            if status not in {"blocked", "success"}:
                raise ValueError(f"guardrail check returned unexpected status {status!r}")
            _guardrail_check_failures.pop(error_key, None)
        _api_error_streaks.pop(error_key, None)
        return json.dumps(_serialize(result), indent=2, default=str)
    except Exception as exc:
        error_key = _api_error_key(workspace or "", studio_session_id)
        if is_guardrail_check:
            return _record_guardrail_check_failure(error_key, f"{type(exc).__name__}: {exc}")
        return _record_api_error(error_key, exc)


def select_agent(
    studio_session_id: str, title: str = "Select agent", description: str = "", default_agent: str | None = None
) -> str:
    """Render Studio's agent selector and return the selected agent."""
    return json.dumps(
        _call_studio_tool(
            studio_session_id,
            "select_agent",
            {"title": title, "description": description, "default_agent": default_agent},
        )
    )


def select_model(
    studio_session_id: str,
    title: str = "Select model",
    description: str = "",
    default_model: str | None = None,
    output_key: str = "model",
    display_label: str | None = None,
    field_label: str | None = None,
    placeholder: str | None = None,
    required_message: str | None = None,
    submit_label: str | None = None,
) -> str:
    """Render Studio's model selector and return the selected model."""
    return json.dumps(
        _call_studio_tool(
            studio_session_id,
            "select_model",
            {
                "title": title,
                "description": description,
                "default_model": default_model,
                "output_key": output_key,
                "display_label": display_label,
                "field_label": field_label,
                "placeholder": placeholder,
                "required_message": required_message,
                "submit_label": submit_label,
            },
        )
    )


def select_dataset_file(
    studio_session_id: str,
    title: str = "Select dataset",
    description: str = "",
    accepted_file_types: list[str] | None = None,
) -> str:
    """Render Studio's fileset/dataset picker and return the selected file."""
    return json.dumps(
        _call_studio_tool(
            studio_session_id,
            "select_dataset_file",
            {"title": title, "description": description, "accepted_file_types": accepted_file_types or []},
        )
    )


def select_eval_config(
    studio_session_id: str,
    title: str = "Select evaluation config",
    description: str = "",
    agent: str | None = None,
    default_agent: str | None = None,
    accepted_file_types: list[str] | None = None,
) -> str:
    """Render Studio's evaluation-config picker and return the selected config."""
    return json.dumps(
        _call_studio_tool(
            studio_session_id,
            "select_eval_config",
            {
                "title": title,
                "description": description,
                "agent": agent,
                "default_agent": default_agent,
                "accepted_file_types": accepted_file_types or [],
            },
        )
    )


def job_progress(
    studio_session_id: str,
    job_name: str,
    job_type: str | None = None,
    source: str | None = None,
    title: str | None = None,
    description: str | None = None,
    workspace: str | None = None,
) -> str:
    """Render a Studio progress card for a platform job."""
    return json.dumps(
        _call_studio_tool(
            studio_session_id,
            "job_progress",
            {
                "job_name": job_name,
                "job_type": job_type,
                "source": source,
                "title": title,
                "description": description,
                "workspace": workspace,
            },
        )
    )


def studio_link(
    studio_session_id: str,
    destination: str,
    name: str | None = None,
    label: str | None = None,
    workspace: str | None = None,
    studio_base_url: str | None = None,
) -> str:
    """Build an optional feature-flag-aware link to a NeMo Studio page.

    Link rendering is presentation-only. If Studio cannot build the link, return
    a non-fatal tool result so completed platform work can still be summarized.
    """
    try:
        result = _call_studio_tool(
            studio_session_id,
            "studio_link",
            {"destination": destination, "name": name, "label": label},
            workspace=workspace,
            studio_base_url=studio_base_url,
        )
    except Exception as exc:
        return json.dumps(
            {
                "status": "unavailable",
                "message": (
                    "Optional Studio link could not be created. Continue with the verified task result. "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )
    return json.dumps(result)


def ask_user_question(studio_session_id: str, questions: str | list[dict[str, Any]]) -> str:
    """Render one or more Studio multiple-choice questions."""
    if isinstance(questions, str):
        try:
            parsed = json.loads(questions)
        except (json.JSONDecodeError, TypeError) as exc:
            return f"Error: `questions` must be a JSON array or JSON array string: {exc}"
    else:
        parsed = questions
    if not isinstance(parsed, list) or not parsed or not all(isinstance(question, dict) for question in parsed):
        return "Error: `questions` must be a non-empty JSON array of question objects."
    approval = _call_studio_tool(
        studio_session_id, "approval_prompt", {"tool_name": "AskUserQuestion", "input": {"questions": parsed}}
    )
    if approval.get("behavior") != "allow":
        return f"User declined to answer: {approval.get('message') or 'no selection made'}"
    return json.dumps(approval.get("updatedInput") or {})


def check_status(service: str, job_name: str, workspace: str | None = None) -> str:
    """Check an evaluation, customization, audit, or Data Designer job."""
    try:
        if workspace is None or not workspace.strip():
            return "Clarification required: which workspace should this status check use?"
        client = _get_client(workspace)
        if service == "evaluator":
            result = _resolve_resource(client, service).get_job_resource(job_name).get_job_status()
            return json.dumps(_serialize(result), indent=2, default=str)
        if service == "data_designer":
            result = _resolve_resource(client, service).get_job_resource(job_name).get_job_status()
            return json.dumps(_serialize(result), indent=2, default=str)
        if service == "auditor":
            result = _resolve_resource(client, service).get_job(job_name)
            return json.dumps(_serialize(result), indent=2, default=str)
        if service.startswith("customization."):
            jobs = _resolve_resource(client, f"{service}.jobs")
            result = jobs.get_job_resource(job_name).get_status()
            return json.dumps(_serialize(result), indent=2, default=str)

        svc = getattr(client, service)
        last_error: Exception | None = None
        for sub_resource in ("metric_jobs", "benchmark_jobs", "jobs"):
            resource = getattr(svc, sub_resource, None)
            if resource is None or not hasattr(resource, "get_status"):
                continue
            try:
                return json.dumps(_serialize(resource.get_status(name=job_name)), indent=2, default=str)
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            return f"Error: {type(last_error).__name__}: {last_error}"
        return f"Error: no status method found for service '{service}'"
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"
