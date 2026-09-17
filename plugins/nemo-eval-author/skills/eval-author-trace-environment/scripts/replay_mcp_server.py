#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve a strict trace-derived MCP scenario over newline-delimited stdio."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCENARIO_SCHEMA = "nemo.eval_author.trace_environment_mcp_scenario.v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _load_scenario(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCENARIO_SCHEMA:
        raise ValueError("scenario does not match the trace-derived MCP contract")
    if payload.get("unknown_call_policy") != "error" or not isinstance(payload.get("tools"), list):
        raise ValueError("scenario must fail closed and provide a tools list")
    return payload


def _tool_index(scenario: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tools: dict[str, dict[str, Any]] = {}
    for entry in scenario["tools"]:
        definition = entry.get("definition") if isinstance(entry, dict) else None
        name = definition.get("name") if isinstance(definition, dict) else None
        if not isinstance(name, str) or not name or name in tools or not isinstance(entry.get("cases"), list):
            raise ValueError("scenario contains an invalid or duplicate tool")
        tools[name] = entry
    return tools


def _audit(path: Path | None, event: dict[str, Any]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as stream:
        stream.write(_canonical_json(event) + "\n")


def _dispatch(
    request: dict[str, Any], scenario: dict[str, Any], tools: dict[str, dict[str, Any]], audit_path: Path | None
) -> dict[str, Any] | None:
    request_id = request.get("id")
    method = request.get("method")
    if request_id is None:
        _audit(audit_path, {"method": method, "notification": True})
        return None
    if method == "initialize":
        params = request.get("params")
        requested = params.get("protocolVersion") if isinstance(params, dict) else None
        protocol = requested if isinstance(requested, str) and requested else "2024-11-05"
        result = {
            "protocolVersion": protocol,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "trace-derived-replay", "version": "1"},
        }
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": [entry["definition"] for entry in scenario["tools"]]}
    elif method == "tools/call":
        params = request.get("params")
        name = params.get("name") if isinstance(params, dict) else None
        arguments = params.get("arguments", {}) if isinstance(params, dict) else None
        entry = tools.get(name) if isinstance(name, str) else None
        matched = None
        if entry is not None and isinstance(arguments, dict):
            key = _canonical_json(arguments)
            matched = next(
                (case for case in entry["cases"] if _canonical_json(case.get("arguments")) == key),
                None,
            )
        _audit(audit_path, {"method": method, "tool": name, "arguments": arguments, "matched": matched is not None})
        if matched is None:
            result = {
                "content": [{"type": "text", "text": "No recorded fixture matched this request."}],
                "isError": True,
            }
        else:
            result = matched["result"]
    else:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "Method not found"}}
    _audit(audit_path, {"method": method, "completed": True})
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=Path(__file__).with_name("scenario.json"))
    parser.add_argument("--audit-log", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        scenario = _load_scenario(args.scenario)
        tools = _tool_index(scenario)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("request must be an object")
                response = _dispatch(request, scenario, tools, args.audit_log)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Invalid request: {error}"},
                }
            if response is not None:
                print(_canonical_json(response), flush=True)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"trace-derived MCP fixture failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
