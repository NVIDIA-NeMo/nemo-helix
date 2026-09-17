# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect installed harness registration without executing emitted shell commands.

Readers recognize config encodings, not an agent-name allowlist. An unfamiliar
writer is unverified, not unsupported. Even verified registration payloads need
separate native-agent execution evidence.
"""

import importlib
import inspect
import json
import re
import shlex
import tempfile
import tomllib
from pathlib import Path
from typing import Any


def registration_payload(command: str) -> dict[str, Any] | None:
    """Read literal echo or quoted-heredoc JSON/TOML; never evaluate shell text."""
    try:
        words = shlex.split(command)
    except ValueError:
        return None
    payload = None
    if (
        len(words) == 4
        and words[0] == "echo"
        and words[2] in {">", ">>"}
        and command.startswith(f"echo {shlex.quote(words[1])} ")
    ):
        payload = words[1]
    else:
        # Quoted delimiters suppress shell expansion. Require an exact closing
        # delimiter so additional commands cannot masquerade as a config write.
        match = re.fullmatch(r"cat\s+>>?\s+[^\n]+\s+<<'([A-Za-z_]\w*)'\n(.*)\n\1\n?", command, re.DOTALL)
        header = shlex.split(command.split("\n", 1)[0])
        if match and len(header) == 4 and header[:2] in (["cat", ">"], ["cat", ">>"]):
            payload = match[2]
    if payload is None:
        return None
    for parse in (json.loads, tomllib.loads):
        try:
            parsed = parse(payload)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def registered_server(payload: dict[str, Any], name: str) -> dict[str, Any] | None:
    """Normalize map- and list-shaped MCP registration encodings."""
    servers = payload.get("mcp_servers", payload.get("mcpServers"))
    if isinstance(servers, dict):
        server = servers.get(name)
        return server if isinstance(server, dict) else None
    if isinstance(servers, list):
        matches = [server for server in servers if isinstance(server, dict) and server.get("name") == name]
        return matches[0] if len(matches) == 1 else None
    return None


def probe_registration(agent: str | None, servers: list[Any]) -> dict[str, Any]:
    """Use Harbor's factory for built-ins/custom agents; do not run/install them.

    Construction imports trusted provider/custom code. Unknown APIs, unavailable
    agents and constructor failures are limitations, not proof of incompatibility.
    """
    result: dict[str, Any] = {
        "agent": agent,
        "status": "unverified",
        "scope": "registration_payload",
        "execution_verified": False,
        "reason": "agent_not_selected",
    }
    if not agent:
        return result
    try:
        factory = importlib.import_module("harbor.agents.factory").AgentFactory
        config_model = importlib.import_module("harbor.models.trial.config").AgentConfig
        config = config_model.model_validate({"name": agent})
        with tempfile.TemporaryDirectory(prefix="trace-mcp-registration-") as temporary:
            provider = factory.create_agent_from_config(config, logs_dir=Path(temporary), mcp_servers=servers)
            writer = getattr(provider, "_build_register_mcp_servers_command", None)
            if not callable(writer):
                result["reason"] = "registration_writer_unavailable"
                return result
            signature = inspect.signature(writer)
            # Some writers take the destination config path instead of owning it.
            kwargs = (
                {"config_path": str(Path(temporary) / "config.toml")} if "config_path" in signature.parameters else {}
            )
            signature.bind(**kwargs)
            command = writer(**kwargs)
        payload = registration_payload(command) if isinstance(command, str) else None
        if payload is None:
            result["reason"] = "registration_encoding_unrecognized"
            return result
        if not any(key in payload for key in ("mcp_servers", "mcpServers")):
            result["reason"] = "registration_encoding_unrecognized"
            return result
        for expected in servers:
            registered = registered_server(payload, expected.name)
            if (
                registered is None
                or registered.get("command") != expected.command
                or registered.get("args", []) != expected.args
                or registered.get("transport", registered.get("type", "stdio")) != "stdio"
                or "url" in registered
            ):
                result.update(status="unsupported", reason="registration_contract_mismatch")
                return result
        result.update(status="verified", reason="registration_contract_retained")
    except Exception as error:
        # Never print provider exception text: it may include local secrets.
        result.update(reason="provider_probe_unavailable", error_type=type(error).__name__)
    return result
