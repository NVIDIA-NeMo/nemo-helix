# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serves Studio's pre-#3857 `generate_data_designer_job_request` tool over MCP stdio."""

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

import anyio
import jsonschema
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

TOOL_PATH = Path(__file__).with_name("generate_data_designer_job_request.json")
TOOL_DEF = json.loads(TOOL_PATH.read_text())
INPUT_SCHEMA = TOOL_DEF["inputSchema"]
COLUMN_SCHEMA = INPUT_SCHEMA["properties"]["job_request"]["properties"]["spec"]["properties"]["config"]["properties"][
    "columns"
]["items"]
KNOWN_COLUMN_KEYS = set(COLUMN_SCHEMA["properties"])
KNOWN_MODEL_CONFIG_KEYS = {"alias", "model", "provider", "inference_parameters", "skip_health_check"}

REJECTION_PREFIX = "TOOL CALL REJECTED:"
ENTITY_NAME = re.compile(r"^[a-z][a-z0-9\-@.+_]{0,61}[a-z0-9@.+_]$")
JINJA_REFERENCE = re.compile(r"\{\{-?\s*([A-Za-z_]\w*)")
JINJA_BUILTINS = {"loop", "range", "true", "false", "none", "True", "False", "None", "not"}
UNBUILDABLE_COLUMN_TYPES = {"seed-dataset", "custom", "embedding"}
LLM_COLUMN_TYPES = {"llm-text", "llm-code", "llm-judge", "llm-structured"}
REQUIRED_BY_TYPE = {
    "expression": ["expr"],
    "sampler": ["sampler_type", "params"],
    "llm-text": ["prompt", "model_alias"],
    "llm-code": ["prompt", "model_alias", "code_lang"],
    "llm-judge": ["prompt", "model_alias", "scores"],
    "llm-structured": ["prompt", "model_alias", "output_format"],
}

PLATFORM_URL = os.environ.get("NHX_BASE_URL", "http://localhost:8080").rstrip("/")
PLATFORM_WORKSPACE = "default"

server = Server("data-designer-job-request")


def _get_json(path: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{PLATFORM_URL}{path}", timeout=5) as response:
        return json.loads(response.read())


def _platform_names(path: str, key: str) -> set[str]:
    names: set[str] = set()
    for item in _get_json(path)["data"]:
        qualified = str(item[key])
        names.add(qualified)
        names.add(qualified.removeprefix(f"{PLATFORM_WORKSPACE}/"))
        names.add(f"{PLATFORM_WORKSPACE}/{qualified}")
    return names


def _model_config_platform_errors(model_configs: list[Any]) -> list[str]:
    try:
        providers = _platform_names(f"/apis/models/v2/workspaces/{PLATFORM_WORKSPACE}/providers?page_size=1000", "name")
        models = _platform_names(f"/apis/inference-gateway/v2/workspaces/{PLATFORM_WORKSPACE}/openai/-/v1/models", "id")
    except OSError as error:
        return [f"model_configs: could not reach the platform to check providers and models ({error})"]
    errors: list[str] = []
    for index, model_config in enumerate(model_configs):
        if not isinstance(model_config, dict):
            continue
        provider = model_config.get("provider")
        model = model_config.get("model")
        if isinstance(provider, str) and provider not in providers:
            errors.append(f"model_configs[{index}]: provider {provider!r} is not registered in workspace 'default'")
        if isinstance(model, str) and model not in models:
            errors.append(f"model_configs[{index}]: model {model!r} is not served by the inference gateway")
    return errors


def _schema_errors(arguments: dict[str, Any]) -> list[str]:
    validator = jsonschema.Draft202012Validator(INPUT_SCHEMA)
    return [
        f"{'/'.join(str(p) for p in error.absolute_path) or '<root>'}: {error.message}"
        for error in validator.iter_errors(arguments)
    ]


def _column_errors(index: int, column: dict[str, Any], earlier: list[str], aliases: set[str]) -> list[str]:
    label = f"columns[{index}] ({column.get('name', '?')})"
    column_type = column.get("column_type")
    errors = [f"{label}: unknown field {key!r}" for key in column if key not in KNOWN_COLUMN_KEYS]
    if column_type in UNBUILDABLE_COLUMN_TYPES:
        errors.append(f"{label}: column_type {column_type!r} cannot be built from a description")
    errors.extend(
        f"{label}: {column_type} column requires {field!r}"
        for field in REQUIRED_BY_TYPE.get(str(column_type), [])
        if field not in column
    )
    if column_type in LLM_COLUMN_TYPES and "model_alias" in column and column["model_alias"] not in aliases:
        errors.append(f"{label}: no model is configured with alias {column['model_alias']!r}")
    for field in ("expr", "prompt", "system_prompt"):
        template = column.get(field)
        if not isinstance(template, str):
            continue
        for reference in JINJA_REFERENCE.findall(template):
            if reference not in earlier and reference not in JINJA_BUILTINS:
                errors.append(f"{label}: {field} references {{{{ {reference} }}}}, which is not an earlier column")
    params = column.get("params")
    if column.get("sampler_type") == "subcategory" and isinstance(params, dict):
        if params.get("category") not in earlier:
            errors.append(f"{label}: subcategory parent {params.get('category')!r} is not an earlier column")
    return errors


def validate_job_request(arguments: dict[str, Any]) -> list[str]:
    errors = _schema_errors(arguments)
    job_request = arguments.get("job_request")
    if not isinstance(job_request, dict):
        return errors
    name = job_request.get("name")
    if isinstance(name, str) and (not ENTITY_NAME.match(name) or "--" in name):
        errors.append(f"job_request.name {name!r} does not match the entity name pattern (lowercase, 2-63 chars)")
    spec = job_request.get("spec")
    if not isinstance(spec, dict):
        return errors
    config = spec.get("config")
    if not isinstance(config, dict):
        return errors
    num_records = spec.get("num_records")
    if isinstance(num_records, float) and not num_records.is_integer():
        errors.append("spec.num_records must be an integer")
    model_configs = config.get("model_configs") or []
    aliases = {str(mc["alias"]) for mc in model_configs if isinstance(mc, dict) and "alias" in mc}
    if isinstance(model_configs, list) and model_configs:
        errors.extend(_model_config_platform_errors(model_configs))
    for index, model_config in enumerate(model_configs):
        if isinstance(model_config, dict):
            errors.extend(
                f"model_configs[{index}]: unknown field {key!r}"
                for key in model_config
                if key not in KNOWN_MODEL_CONFIG_KEYS
            )
    earlier: list[str] = []
    for index, column in enumerate(config.get("columns") or []):
        if not isinstance(column, dict):
            continue
        errors.extend(_column_errors(index, column, earlier, aliases))
        if isinstance(column.get("name"), str):
            earlier.append(column["name"])
    return errors


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [types.Tool(name=TOOL_DEF["name"], description=TOOL_DEF["description"], inputSchema=INPUT_SCHEMA)]


@server.call_tool(validate_input=False)
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    if name != TOOL_DEF["name"]:
        raise ValueError(f"unknown tool {name!r}")
    errors = validate_job_request(arguments)
    if errors:
        # Returned as plain text: an MCP isError result aborts the DeepAgents run instead of reaching the model.
        rejection = f"{REJECTION_PREFIX} 422 Unprocessable Entity:\n" + "\n".join(f"- {error}" for error in errors)
        return [types.TextContent(type="text", text=rejection)]
    job_request = arguments["job_request"]
    columns = [column["name"] for column in job_request["spec"]["config"]["columns"]]
    summary = {
        "status": "accepted",
        "name": job_request.get("name"),
        "num_records": job_request["spec"]["num_records"],
        "columns": columns,
    }
    return [types.TextContent(type="text", text=json.dumps(summary))]


async def _serve() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    anyio.run(_serve)


if __name__ == "__main__":
    main()
