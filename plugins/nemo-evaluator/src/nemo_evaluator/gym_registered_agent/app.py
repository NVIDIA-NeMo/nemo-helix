# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Gym agent that runs a registered platform agent's Fabric config.

The evaluator's staging step copies this file into a Gym environment package next to the
agent-instance config it generates. It runs inside the sandboxed Gym host, so it imports only what
that host has: ``nemo_gym`` and the ``nemo-fabric`` wheels staged beside it. The rollout plumbing
mirrors Gym's own ``nemo_fabric_agent``; the difference is that the registered agent's config is
run as registered, with only its default model rebound to Gym's policy model server.
"""

import json
import logging
import os
import re
import tempfile
from asyncio import Semaphore
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, ClassVar, Optional
from uuid import uuid4

from fastapi import Body, Request
from nemo_fabric import Fabric, FabricConfig, RunRequest  # ty: ignore[unresolved-import]
from nemo_gym.base_resources_server import (  # ty: ignore[unresolved-import]
    NEMO_GYM_MCP_METADATA_KEY,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (  # ty: ignore[unresolved-import]
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef  # ty: ignore[unresolved-import]
from nemo_gym.global_config import SKILLS_REF_KEY_NAME, get_first_server_config_dict  # ty: ignore[unresolved-import]
from nemo_gym.openai_utils import (  # ty: ignore[unresolved-import]
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)
from nemo_gym.server_utils import get_response_json, raise_for_status  # ty: ignore[unresolved-import]
from pydantic import ConfigDict, Field, PositiveInt

try:  # Gym renamed this item type after 0.5.0.
    from nemo_gym.openai_utils import (  # ty: ignore[unresolved-import]
        NeMoGymResponseFunctionCallOutput as FunctionCallOutput,  # ty: ignore[unresolved-import]
    )
except ImportError:  # pragma: no cover - depends on the Gym version in the host image
    from nemo_gym.openai_utils import NeMoGymFunctionCallOutput as FunctionCallOutput  # ty: ignore[unresolved-import]

LOG = logging.getLogger(__name__)

#: The variable the composed config reads the policy model server's key from.
POLICY_MODEL_API_KEY_ENV = "NEMO_GYM_FABRIC_MODEL_API_KEY"  # pragma: allowlist secret
_ENV_TEMPLATE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def compose_fabric_config(
    registered: dict[str, Any],
    *,
    model_name: str,
    model_base_url: str,
    model_api_key: str,
    workspace: str,
    system_prompt: Optional[str],
    mcp_servers: dict[str, Any],
    skills: list[str],
    environ: dict[str, str],
    timeout_seconds: Optional[float] = None,
) -> dict[str, Any]:
    """The registered agent's config, made runnable for one Gym rollout.

    Everything the agent is stays: identity, harness, instructions, tools, MCP servers, skills, the
    models other than the default. What changes is what Gym owns: the default model becomes the
    policy model server, the workspace is the rollout's, Gym's rollout MCP server and skills are
    added, and a stdio MCP server's ``${NAME}`` env templates are filled from this process. An agent
    that declares no ``runtime.timeout_seconds`` gets ``timeout_seconds`` as its task deadline.
    """
    config = deepcopy(registered)
    environment = config.setdefault("environment", {})
    environment["workspace"] = workspace
    environment.setdefault("env", {})[POLICY_MODEL_API_KEY_ENV] = model_api_key
    if timeout_seconds is not None:
        config.setdefault("runtime", {}).setdefault("timeout_seconds", timeout_seconds)

    models = config.setdefault("models", {})
    default = dict(models.get("default") or {})
    default.update(
        {
            "provider": "openai-compatible",
            "model": model_name,
            "base_url": model_base_url,
            "api_key_env": POLICY_MODEL_API_KEY_ENV,
        }
    )
    models["default"] = default

    if system_prompt:
        instructions = config.setdefault("instructions", {})
        system = dict(instructions.get("system") or {})
        authored = system.get("content")
        parts = [part for part in (authored, system_prompt) if isinstance(part, str) and part]
        system.update({"content": "\n\n".join(parts), "mode": "replace"})
        instructions["system"] = system

    servers = (config.get("mcp") or {}).get("servers") or {}
    for server in servers.values():
        if not isinstance(server, dict) or server.get("transport") != "stdio":
            continue
        for key, value in list((server.get("env") or {}).items()):
            match = _ENV_TEMPLATE.match(value) if isinstance(value, str) else None
            if match and match.group(1) in environ:
                server["env"][key] = environ[match.group(1)]
    if mcp_servers:
        config.setdefault("mcp", {})["servers"] = {**servers, **mcp_servers}

    if skills:
        skill_config = config.setdefault("skills", {})
        skill_config["paths"] = list(dict.fromkeys([*(skill_config.get("paths") or []), *skills]))
    return config


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _extract_request_input(body_input: Any) -> tuple[Any, Optional[str]]:
    if body_input is None:
        return "", None
    if isinstance(body_input, str):
        return body_input, None
    messages: list[dict[str, Any]] = []
    instruction_parts: list[str] = []
    for item in body_input:
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
        text = _content_text(content)
        if role in {"system", "developer"}:
            if text:
                instruction_parts.append(text)
            continue
        messages.append(item.model_dump(mode="json", exclude_none=True) if hasattr(item, "model_dump") else dict(item))
    if len(messages) == 1 and messages[0].get("role") == "user":
        request_input: Any = _content_text(messages[0].get("content"))
    else:
        request_input = messages
    return request_input, "\n\n".join(instruction_parts) or None


def _skill_paths(skills_root: Optional[str]) -> list[str]:
    if not skills_root:
        return []
    root = Path(skills_root).resolve()
    if not root.is_dir():
        raise ValueError(f"skills_ref path is not a directory: {root}")
    if (root / "SKILL.md").is_file():
        return [str(root)]
    paths = [str(path) for path in sorted(root.iterdir()) if path.is_dir() and (path / "SKILL.md").is_file()]
    if not paths:
        raise ValueError(f"skills_ref path contains no SKILL.md directories: {root}")
    return paths


def _mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_mapping"):
        value = value.to_mapping()
    return dict(value) if isinstance(value, dict) else {}


def _usage_value(usage: dict[str, Any], *names: str) -> int:
    for name in names:
        value = usage.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return 0


def _normalized_usage(result: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    usage = _mapping(result.get("usage"))
    if not usage:
        output_usage = _mapping(output.get("usage"))
        usage = _mapping(output_usage.get("total")) or output_usage
    return {**_mapping(usage.get("extensions")), **_mapping(usage.get("metadata")), **usage}


def _turns_used(output: dict[str, Any]) -> int:
    usage = _mapping(output.get("usage"))
    for value in (output.get("turns_used"), output.get("num_turns"), output.get("api_calls"), usage.get("api_calls")):
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    messages = output.get("messages")
    if isinstance(messages, list):
        turns = sum(1 for m in messages if isinstance(m, dict) and m.get("role") in {"ai", "assistant"})
        if turns:
            return turns
    return 1


def fabric_output_items(output: dict[str, Any], response_text: str) -> list[Any]:
    """Fabric's messages, as the Responses API items Gym records for the rollout."""
    items: list[Any] = []
    messages = output.get("messages")
    if isinstance(messages, list) and messages:
        pending: list[str] = []
        for message in messages[:-1]:
            if not isinstance(message, dict):
                continue
            role = message.get("role") or message.get("type")
            if role in {"human", "user", "system", "developer"}:
                continue
            if role == "tool":
                call_id = pending.pop(0) if pending else str(message.get("tool_call_id") or message.get("id") or "")
                items.append(
                    FunctionCallOutput(
                        id=f"fco_{uuid4().hex}",
                        call_id=call_id,
                        output=_content_text(message.get("content")),
                        status="completed",
                        type="function_call_output",
                    )
                )
                continue
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                items.append(
                    NeMoGymResponseReasoningItem(
                        id=f"rs_{uuid4().hex}", summary=[NeMoGymSummary(text=reasoning, type="summary_text")]
                    )
                )
            text = _content_text(message.get("content"))
            if text:
                items.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg_{uuid4().hex}",
                        content=[NeMoGymResponseOutputText(text=text, annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                )
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
                call_id = str(tool_call.get("id") or tool_call.get("call_id") or f"call_{uuid4().hex}")
                pending.append(call_id)
                arguments = tool_call.get("args", tool_call.get("arguments", function.get("arguments")))
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True)
                items.append(
                    NeMoGymResponseFunctionToolCall(
                        id=f"fc_{uuid4().hex}",
                        call_id=call_id,
                        name=str(tool_call.get("name") or function.get("name") or ""),
                        arguments=arguments,
                        status="completed",
                    )
                )
    items.append(
        NeMoGymResponseOutputMessage(
            id=f"msg_{uuid4().hex}",
            content=[NeMoGymResponseOutputText(text=response_text, annotations=[])],
            role="assistant",
            status="completed",
            type="message",
        )
    )
    return items


class NeMoRegisteredAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    fabric_config: dict[str, Any]
    fabric_config_base_dir: Optional[str] = None
    model: str = "gym-policy-model"
    model_api_key: str = "local"  # pragma: allowlist secret
    concurrency: PositiveInt = 32
    #: The agent's task deadline when its own config declares none; Fabric clocks it on the invoke,
    #: so adapter startup and tool installation do not count against the agent.
    timeout: PositiveInt = 600
    system_prompt: Optional[str] = None


class NeMoRegisteredAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class NeMoRegisteredAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    fabric_result: dict[str, Any] = Field(default_factory=dict)


class NeMoRegisteredAgent(SimpleResponsesAPIAgent):
    ray_enabled: ClassVar[bool] = False  # a plain assignment is a pydantic field on the Gym 0.5 base class
    config: NeMoRegisteredAgentConfig
    sem: Semaphore | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    def _resources_server_base_url(self) -> str:
        server = get_first_server_config_dict(self.server_client.global_config_dict, self.config.resources_server.name)
        return self.server_client._build_server_base_url(server)

    def _rollout_mcp_servers(self, seed_response: dict[str, Any]) -> dict[str, Any]:
        metadata = seed_response.get(NEMO_GYM_MCP_METADATA_KEY)
        if not isinstance(metadata, dict):
            return {}
        name = str(metadata.get("server_name") or self.config.resources_server.name)
        url_path = str(metadata.get("url_path") or "/mcp")
        server: dict[str, Any] = {
            "transport": "streamable-http",
            "url": f"{self._resources_server_base_url().rstrip('/')}/{url_path.lstrip('/')}",
        }
        headers = metadata.get("headers")
        if isinstance(headers, dict) and headers:
            server["custom_headers"] = {str(key): str(value) for key, value in headers.items()}
        return {name: server}

    async def _create_response(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        mcp_servers: Optional[dict[str, Any]] = None,
        skills_path: Optional[str] = None,
        rollout_id: Optional[str] = None,
    ) -> tuple[NeMoGymResponse, dict[str, Any]]:
        request_input, input_system = _extract_request_input(body.input)
        system_prompt = (
            "\n\n".join(p for p in (self.config.system_prompt, body.instructions, input_system) if p) or None
        )
        model_base_url = self.resolve_model_base_url(self.config.model_server.name, rollout_id)
        with tempfile.TemporaryDirectory(prefix="nemo_registered_agent_") as workspace:
            composed = compose_fabric_config(
                self.config.fabric_config,
                model_name=self.config.model,
                model_base_url=model_base_url,
                model_api_key=self.config.model_api_key,
                workspace=workspace,
                system_prompt=system_prompt,
                mcp_servers=mcp_servers or {},
                skills=_skill_paths(skills_path),
                environ=dict(os.environ),
                timeout_seconds=self.config.timeout,
            )
            result = await Fabric().run(
                FabricConfig.from_mapping(composed),
                base_dir=self.config.fabric_config_base_dir or workspace,
                request=RunRequest(
                    input=request_input,
                    request_id=rollout_id or f"request-{uuid4().hex}",
                    context={"nemo_gym_rollout_id": rollout_id} if rollout_id else {},
                ),
            )
        result_mapping = result.to_mapping()
        if result.status != "succeeded":
            error = result.error.message if result.error is not None else "unknown Fabric failure"
            raise RuntimeError(f"registered agent invocation failed: {error}")
        output = _mapping(result.output)
        usage = _normalized_usage(result_mapping, output)
        input_tokens = _usage_value(usage, "input_tokens", "prompt_tokens", "inputTokens")
        output_tokens = _usage_value(usage, "output_tokens", "completion_tokens", "outputTokens")
        response_text = output.get("response", output.get("output"))
        if not isinstance(response_text, str):
            response_text = (
                "" if response_text is None else json.dumps(response_text, ensure_ascii=False, sort_keys=True)
            )
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=str(output.get("model") or self.config.model),
            object="response",
            output=fabric_output_items(output, response_text),
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(
                    cached_tokens=_usage_value(usage, "cached_input_tokens", "cache_read_input_tokens")
                ),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(
                    reasoning_tokens=_usage_value(usage, "reasoning_tokens", "reasoning_output_tokens")
                ),
                total_tokens=_usage_value(usage, "total_tokens") or input_tokens + output_tokens,
            ),
        )
        return response, json.loads(json.dumps(result_mapping, default=str))

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        response, _ = await self._create_response(body, rollout_id=request.path_params.get("rollout_id"))
        return response

    async def run(self, request: Request, body: NeMoRegisteredAgentRunRequest) -> NeMoRegisteredAgentVerifyResponse:
        if self.sem is None:  # pragma: no cover
            raise RuntimeError("registered agent concurrency control is not initialized")
        async with self.sem:
            seed_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=request.cookies,
            )
            await raise_for_status(seed_response)
            seed_json = await get_response_json(seed_response)
            skills_path = ((body.model_extra or {}).get(SKILLS_REF_KEY_NAME) or {}).get("path")
            agent_response, fabric_result = await self._create_response(
                body.responses_create_params,
                mcp_servers=self._rollout_mcp_servers(seed_json),
                skills_path=skills_path,
                rollout_id=self.rollout_id_from_run(body),
            )
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": agent_response.model_dump(mode="json")},
                cookies=seed_response.cookies,
            )
            await raise_for_status(verify_response)
            verify_json = await get_response_json(verify_response)
            fabric_output = _mapping(fabric_result.get("output"))
            finished = (
                bool(fabric_output["completed"])
                if "completed" in fabric_output
                else not fabric_output.get("failed", False)
            )
            return NeMoRegisteredAgentVerifyResponse.model_validate(
                verify_json
                | {
                    "turns_used": _turns_used(fabric_output),
                    "finished_naturally": finished,
                    "fabric_result": fabric_result,
                }
            )


if __name__ == "__main__":
    NeMoRegisteredAgent.run_webserver()
