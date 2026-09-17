# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scoped replay, reviewed schema and actual Harbor registration regressions."""

import argparse
import asyncio
import importlib
import inspect
import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path, PurePosixPath
from unittest.mock import MagicMock

import pytest
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.codex import Codex
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import TaskConfig
from harbor.models.trial.config import AgentConfig
from harbor.models.trial.paths import EnvironmentPaths
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

# Importlib-mode collection does not add the tests directory to sys.path.
# Load this standalone test helper module by path, as it loads the skill scripts.
_HELPERS_SPEC = spec_from_file_location(
    "trace_environment_fixture_helpers", Path(__file__).with_name("test_trace_environment.py")
)
assert _HELPERS_SPEC is not None and _HELPERS_SPEC.loader is not None
_helpers = module_from_spec(_HELPERS_SPEC)
_HELPERS_SPEC.loader.exec_module(_helpers)
te = _helpers._TRACE_ENVIRONMENT
_candidate = _helpers._candidate
_fixture_atif = _helpers._fixture_atif
_fixture_workspace = _helpers._fixture_workspace
_plan_tool_calls = _helpers._plan_tool_calls
_resolve_tool_access = _helpers._resolve_tool_access
_review_privacy = _helpers._review_privacy
_run = _helpers._run
_write_json = _helpers._write_json

fc = importlib.import_module("fixture_compiler")
registration = importlib.import_module("mock_registration")


@pytest.mark.parametrize("import_mode", ["prepend", "importlib"])
def test_standalone_collection_without_test_directory_on_pythonpath(tmp_path, monkeypatch, import_mode):
    # Explicit plugins load even when entry-point plugin autoload is disabled.
    monkeypatch.setenv("PYTEST_PLUGINS", "unavailable_parent_only_pytest_plugin")
    test_path = Path(__file__).resolve()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(test_path.parents[1] / "pyproject.toml"),
            "-p",
            "no:cacheprovider",
            "--collect-only",
            f"--import-mode={import_mode}",
            str(test_path),
            "-q",
        ],
        cwd=tmp_path,
        env={
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"}
        }
        | {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "tests collected" in result.stdout


def _inventory(trace=None):
    return fc.derive_tool_call_inventory(trace or _fixture_atif(), safe_atif_sha256="sha256:synthetic")


@pytest.mark.parametrize("kind", ["oversized", "symlink", "fifo"])
def test_inventory_rejects_unsafe_trace_before_open(tmp_path, monkeypatch, kind):
    if kind == "fifo" and not hasattr(os, "mkfifo"):
        pytest.skip("FIFO regression requires POSIX")
    task_dir = _fixture_workspace(tmp_path)
    safe_path = task_dir / "safe/trace.atif.json"
    if kind == "oversized":
        with safe_path.open("wb") as stream:
            stream.truncate(te.MAX_CANONICAL_BYTES + 1)
    else:
        safe_path.unlink()
        if kind == "symlink":
            safe_path.symlink_to(task_dir / "private/source.atif.json")
        else:
            os.mkfifo(safe_path)
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path != safe_path, "invalid trace must be rejected before opening it"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(te.ContractError, match="safe ATIF (exceeds|must be a retained regular file)"):
        te._current_tool_call_inventory(task_dir)


def _overrides(inventory):
    tool = inventory["tools"][0]
    return {
        "schema": fc.OVERRIDES_SCHEMA,
        "inventory_sha256": fc.json_sha256(inventory),
        "reviewer_kind": "human",
        "entries": [
            {
                "tool_id": tool["tool_id"],
                "input_schema": {
                    "type": "object",
                    "properties": {"account_id": {"type": "string"}},
                    "required": ["account_id"],
                },
                "note": "Reviewed the recorded object arguments; this describes the replay interface, not the original software.",
                "evidence": [
                    {key: call[key] for key in ("trajectory_path", "step_id", "tool_call_id")} for call in tool["calls"]
                ],
            }
        ],
    }


def test_root_does_not_borrow_child_definition():
    trace = _fixture_atif()
    child = deepcopy(trace)
    trace["agent"]["tool_definitions"] = []
    trace["subagent_trajectories"] = [child]
    inventory = _inventory(trace)
    assert len(inventory["tools"]) == 6
    assert [tool["definition_status"] for tool in inventory["tools"]] == ["missing"] * 3 + ["complete"] * 3
    assert len({tool["tool_id"] for tool in inventory["tools"]}) == 6


def test_server_scopes_and_aliases_keep_same_name_calls_separate():
    trace = _fixture_atif()
    definitions = []
    calls = []
    results = []
    for server in ("first", "second"):
        definition = deepcopy(trace["agent"]["tool_definitions"][0])
        definition["extra"] = {"tool_scope": {"server": server}}
        definitions.append(definition)
        call = deepcopy(trace["steps"][1]["tool_calls"][0])
        call.update(tool_call_id=server, extra=definition["extra"])
        calls.append(call)
        results.append({"source_call_id": server, "content": server})
    trace["agent"]["tool_definitions"] = definitions
    trace["steps"][1].update(tool_calls=calls, observation={"results": results})
    inventory = _inventory(trace)
    plan = fc.derive_tool_call_plan(inventory)
    assert [tool["mock_support"] for tool in plan["tools"]] == ["exact_replay"] * 2
    requested = {
        "schema": fc.DECISIONS_SCHEMA,
        "decisions": [
            {
                "name": tool["name"],
                "tool_id": tool["tool_id"],
                "access": "mock",
                "adapter": "mcp",
                "note": "Use only recorded outputs.",
            }
            for tool in plan["tools"]
        ],
    }
    access = fc.resolve_tool_access(inventory, plan, requested, reviewer_kind="human")
    scenario = fc.build_mcp_scenario(fc.build_call_fixtures(inventory, access))
    assert len({tool["definition"]["name"] for tool in scenario["tools"]}) == 2
    assert [tool["cases"][0]["result"]["content"][0]["text"] for tool in scenario["tools"]] == ["first", "second"]
    del requested["decisions"][0]["tool_id"]
    with pytest.raises(ValueError, match="scoped tool_id"):
        fc.resolve_tool_access(inventory, plan, requested, reviewer_kind="human")


@pytest.mark.parametrize(
    "schema,reason",
    [
        ({"type": "array"}, "input_schema_invalid"),
        ({"type": "object", "required": "account_id"}, "input_schema_invalid"),
        ({"type": "object", "properties": {"account_id": {"type": "integer"}}}, "arguments_schema_mismatch"),
        ({"type": "object", "$ref": "https://example.invalid/schema.json"}, "input_schema_reference_unavailable"),
        ({"type": "object", "$schema": "https://example.invalid/dialect"}, "input_schema_dialect_unsupported"),
    ],
)
def test_invalid_schema_or_arguments_fail_closed(schema, reason):
    trace = _fixture_atif()
    trace["agent"]["tool_definitions"][0]["inputSchema"] = schema
    tool = fc.derive_tool_call_plan(_inventory(trace))["tools"][0]
    assert tool["mock_support"] == "unsupported"
    assert reason in tool["reason_codes"]


def test_local_schema_references_are_validated():
    trace = _fixture_atif()
    trace["agent"]["tool_definitions"][0]["inputSchema"] = {
        "type": "object",
        "$defs": {"id": {"type": "string"}},
        "properties": {"account_id": {"$ref": "#/$defs/id"}},
        "required": ["account_id"],
    }
    assert fc.derive_tool_call_plan(_inventory(trace))["tools"][0]["mock_support"] == "exact_replay"


@pytest.mark.parametrize(
    "case,reason",
    [
        ("missing_definition", "tool_definition_missing"),
        ("incomplete_definition", "tool_definition_incomplete"),
        ("ambiguous_definition", "tool_definition_ambiguous"),
        ("missing_arguments", "arguments_missing_or_non_object"),
        ("string_arguments", "arguments_missing_or_non_object"),
        ("no_observation", "observation_pairing_incomplete"),
        ("duplicate_observation", "observation_pairing_incomplete"),
        ("non_text_result", "non_text_result_unsupported"),
        ("conflicting_results", "conflicting_results_for_identical_arguments"),
        ("omitted_image_input", "redacted_value_required"),
        ("omitted_image_output", "redacted_value_required"),
        ("redacted_description", "redacted_value_required"),
        ("redacted_schema", "redacted_value_required"),
    ],
)
def test_each_replay_rejection_reason_blocks_mock_access(case, reason):
    trace = _fixture_atif()
    definition = trace["agent"]["tool_definitions"][0]
    step = trace["steps"][1]
    call = step["tool_calls"][0]
    result = step["observation"]["results"][0]
    # Isolate one surface so each reason must be independently necessary.
    trace["agent"]["tool_definitions"] = [definition]
    step["tool_calls"] = [call]
    step["observation"]["results"] = [result]
    if case == "missing_definition":
        trace["agent"]["tool_definitions"] = []
    elif case == "incomplete_definition":
        del definition["inputSchema"]
    elif case == "ambiguous_definition":
        trace["agent"]["tool_definitions"].append({"name": definition["name"], "inputSchema": {"type": "object"}})
    elif case == "missing_arguments":
        del call["arguments"]
    elif case == "string_arguments":
        call["arguments"] = "not an object"
    elif case == "no_observation":
        step["observation"]["results"] = []
    elif case == "duplicate_observation":
        step["observation"]["results"].append(deepcopy(result))
    elif case == "non_text_result":
        result["content"] = {"status": "active"}
    elif case == "conflicting_results":
        later = deepcopy(step)
        later["step_id"] = 3
        later["tool_calls"][0]["tool_call_id"] = "different-call"
        later["observation"]["results"][0] = {"source_call_id": "different-call", "content": "different output"}
        trace["steps"].append(later)
    elif case == "omitted_image_input":
        call["arguments"]["account_id"] = "<omitted:image>"
    elif case == "omitted_image_output":
        result["content"] = "<omitted:image>"
    elif case == "redacted_description":
        definition["description"] = "<redacted:secret>"
    elif case == "redacted_schema":
        definition["inputSchema"]["description"] = "<redacted:secret>"
    inventory = _inventory(trace)
    plan = fc.derive_tool_call_plan(inventory)
    assert plan["tools"][0]["reason_codes"] == [reason]
    assert plan["tools"][0]["mock_support"] == "unsupported"
    if reason == "redacted_value_required":
        assert reason in inventory["tools"][0]["uncertainties"]
    requested = {
        "schema": fc.DECISIONS_SCHEMA,
        "decisions": [
            {"name": definition["name"], "access": "mock", "adapter": "mcp", "note": "This must be rejected."}
        ],
    }
    with pytest.raises(ValueError, match=reason):
        fc.resolve_tool_access(inventory, plan, requested, reviewer_kind="human")


def test_generation_refuses_privacy_blockers_after_contextual_review(tmp_path):
    trace = _fixture_atif()
    trace["steps"][0]["message"] = [{"type": "image", "source": {"media_type": "image/png", "path": "screen.png"}}]
    task_dir = _fixture_workspace(tmp_path, trace=trace)
    _plan_tool_calls(task_dir)
    assert (
        _resolve_tool_access(task_dir, {"account.lookup": "mock", "account.update": "none", "account.inspect": "none"})[
            0
        ]
        == 0
    )
    _review_privacy(task_dir)
    summary = json.loads((task_dir / "summary.json").read_text())
    assert summary["privacy"]["contextual_review_complete"] and summary["privacy"]["blocking_reasons"]
    code, result = _run("generate-mock-tool-calls", "--task-dir", str(task_dir))
    assert code == 1 and "blocked by unresolved privacy" in result["error"]
    assert not (task_dir / "task/environment/tool-call-fixtures").exists()
    assert not (task_dir / "private/tool-call-generation.json").exists()


def test_reviewed_override_enables_recorded_calls_without_modifying_trace():
    trace = _fixture_atif()
    trace["agent"]["tool_definitions"] = []
    original = deepcopy(trace)
    inventory = _inventory(trace)
    overrides = _overrides(inventory)
    plan = fc.derive_tool_call_plan(inventory, overrides)
    assert plan["tools"][0]["mock_support"] == "exact_replay"
    assert plan["tools"][0]["definition_provenance"]["kind"] == "reviewed_override"
    assert plan["tools"][1]["mock_support"] == "unsupported"
    assert trace == original
    assert inventory["tools"][0]["definition_status"] == "missing"
    overrides["entries"][0]["evidence"][0]["trajectory_path"] = "$.other"
    with pytest.raises(ValueError, match="scoped call reference"):
        fc.derive_tool_call_plan(inventory, overrides)


def test_override_cli_retains_provenance_and_detects_tampering(tmp_path):
    trace = _fixture_atif()
    trace["agent"]["tool_definitions"] = []
    task_dir = _fixture_workspace(tmp_path, trace=trace)
    assert _run("inventory-tool-calls", "--task-dir", str(task_dir))[0] == 0
    inventory = json.loads((task_dir / "private/tool-call-inventory.json").read_text())
    requested = tmp_path / "overrides.json"
    _write_json(requested, _overrides(inventory))
    before = (task_dir / "safe/trace.atif.json").read_bytes()
    code, result = _run("plan-tool-call-access", "--task-dir", str(task_dir), "--schema-overrides", str(requested))
    assert code == 0, result
    assert (task_dir / "safe/trace.atif.json").read_bytes() == before
    assert (
        _resolve_tool_access(task_dir, {"account.lookup": "mock", "account.update": "none", "account.inspect": "none"})[
            0
        ]
        == 0
    )
    _review_privacy(task_dir)
    code, result = _run("generate-mock-tool-calls", "--task-dir", str(task_dir))
    assert code == 0, result
    generated = json.loads((task_dir / "task/environment/tool-call-fixtures/call-fixtures.json").read_text())
    assert generated["tools"][0]["definition_provenance"]["kind"] == "reviewed_override"
    assert generated["tools"][0]["cases"][0]["output"] == '{"status":"active"}'
    te._validate_tool_call_pipeline(task_dir, require_decisions=True)
    retained = task_dir / "private/tool-schema-overrides.json"
    override = json.loads(retained.read_text())
    override["entries"][0]["note"] = "changed"
    _write_json(retained, override)
    with pytest.raises(te.ContractError, match="differs"):
        te._current_tool_call_plan(task_dir)


@pytest.mark.parametrize("partial", ["absent", "inventory", "plan"])
def test_candidate_requires_decisions_but_pending_and_no_candidate_allow_partial(tmp_path, partial):
    task_dir = _fixture_workspace(tmp_path)
    if partial != "absent":
        assert _run("inventory-tool-calls", "--task-dir", str(task_dir))[0] == 0
    if partial == "plan":
        assert _run("plan-tool-call-access", "--task-dir", str(task_dir))[0] == 0
    te._validate_tool_call_pipeline(task_dir)
    with pytest.raises(te.ContractError, match="complete reviewed"):
        te._validate_tool_call_pipeline(task_dir, require_decisions=True)
    code, result = _run("finalize", "--task-dir", str(task_dir), "--status", "candidate")
    assert code == 1 and "complete reviewed" in result["error"]
    _candidate(task_dir, status="no_candidate")
    code, result = _run(
        "finalize", "--task-dir", str(task_dir), "--status", "no_candidate", "--reason", "Insufficient evidence."
    )
    assert code == 0, result


def test_unresolved_calls_cannot_disappear_from_access_denominator():
    trace = _fixture_atif()
    del trace["steps"][1]["tool_calls"][0]["function_name"]
    inventory = _inventory(trace)
    with pytest.raises(ValueError, match="unresolved"):
        fc.resolve_tool_access(
            inventory,
            fc.derive_tool_call_plan(inventory),
            {"schema": fc.DECISIONS_SCHEMA, "decisions": []},
            reviewer_kind="human",
        )


def _generated_mock(tmp_path):
    task_dir = _fixture_workspace(tmp_path)
    _plan_tool_calls(task_dir)
    assert (
        _resolve_tool_access(task_dir, {"account.lookup": "mock", "account.update": "none", "account.inspect": "none"})[
            0
        ]
        == 0
    )
    _review_privacy(task_dir)
    code, result = _run("generate-mock-tool-calls", "--task-dir", str(task_dir))
    assert code == 0, result
    return task_dir


def _registered_mock(tmp_path, agent_name="codex"):
    task_dir = _generated_mock(tmp_path)
    fixture_dir = task_dir / "task/environment/tool-call-fixtures"
    config = TaskConfig.model_validate(tomllib.loads((fixture_dir / "integration.toml").read_text()))
    server = config.environment.mcp_servers[0]
    # Relocate only the container prefix for this host protocol integration test.
    server.command = str(fixture_dir / "launch-replay.sh")
    provider_agent = AgentFactory.create_agent_from_config(
        AgentConfig(name=agent_name), logs_dir=tmp_path / "harbor-logs", mcp_servers=[server]
    )
    config_dir = tmp_path / "harness-config"
    config_dir.mkdir(mode=0o700)
    writer = getattr(provider_agent, "_build_register_mcp_servers_command")
    kwargs = (
        {"config_path": str(config_dir / "config.toml")}
        if "config_path" in inspect.signature(writer).parameters
        else {}
    )
    command = writer(**kwargs)
    assert isinstance(command, str)
    # Execute only these installed, inspected writers in this synthetic test.
    # Production preflight never executes emitted shell commands.
    assert agent_name in {"codex", "claude-code", "vibe"}
    subprocess.run(
        ["bash", "-c", command],
        env={**os.environ, "CODEX_HOME": str(config_dir), "CLAUDE_CONFIG_DIR": str(config_dir)},
        check=True,
    )
    config_file = config_dir / (".claude.json" if agent_name == "claude-code" else "config.toml")
    payload = (
        json.loads(config_file.read_text()) if config_file.suffix == ".json" else tomllib.loads(config_file.read_text())
    )
    registered = registration.registered_server(payload, server.name)
    assert registered is not None
    assert registered["command"] == server.command
    assert registered.get("args", []) == []
    return config_dir, {key: registered[key] for key in ("command", "args") if key in registered}


@pytest.mark.parametrize("agent_name", ["codex", "claude-code", "vibe"])
def test_harbor_registered_mock_is_discoverable_and_callable(tmp_path, agent_name):
    """Use Harbor's real registration and the MCP SDK, not hand-written JSON-RPC."""
    _, registered = _registered_mock(tmp_path, agent_name)

    async def exercise():
        params = StdioServerParameters(**registered, env={"TRACE_TOOL_CALL_AUDIT_LOG": str(tmp_path / "audit.jsonl")})
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert [tool.name for tool in tools.tools] == ["account.lookup"]
                result = await session.call_tool("account.lookup", {"account_id": "synthetic-001"})
                assert not result.isError
                assert isinstance(result.content[0], TextContent)
                assert result.content[0].text == '{"status":"active"}'
                for name, args in [("account.lookup", {"account_id": "unseen"}), ("unseen", {})]:
                    assert (await session.call_tool(name, args)).isError

    asyncio.run(asyncio.wait_for(exercise(), timeout=20))


@pytest.mark.parametrize(
    "agent_name", ["codex", "claude-code", "vibe", "harbor.agents.installed.claude_code:ClaudeCode"]
)
def test_registration_probe_uses_provider_factory_not_agent_allowlist(agent_name):
    servers = TaskConfig.model_validate(tomllib.loads(te._fixture_integration())).environment.mcp_servers
    result = registration.probe_registration(agent_name, servers)
    assert result["status"] == "verified", result
    assert result["scope"] == "registration_payload" and not result["execution_verified"]


@pytest.mark.parametrize("agent_name", [None, "unknown", "nop"])
def test_unknown_or_unprobed_registration_is_not_unsupported(agent_name):
    servers = TaskConfig.model_validate(tomllib.loads(te._fixture_integration())).environment.mcp_servers
    result = registration.probe_registration(agent_name, servers)
    assert result["status"] == "unverified"
    assert not result["execution_verified"]


@pytest.mark.parametrize("command", [None, "unrecognized-writer", "echo not-json >> config"])
def test_unrecognized_writer_is_unverified(monkeypatch, command):
    monkeypatch.setattr(Codex, "_build_register_mcp_servers_command", lambda self: command)
    servers = TaskConfig.model_validate(tomllib.loads(te._fixture_integration())).environment.mcp_servers
    assert registration.probe_registration("codex", servers)["status"] == "unverified"


@pytest.mark.parametrize("change", [{"command": "wrong executable"}, {"args": ["--wrong"]}, {"transport": "sse"}])
def test_demonstrated_registration_mismatch_is_unsupported(monkeypatch, change):
    server = {"command": "/opt/tool-call-fixtures/launch-replay.sh", "args": [], **change}
    payload = json.dumps({"mcpServers": {"trace-tool-call-replay": server}})
    monkeypatch.setattr(
        Codex, "_build_register_mcp_servers_command", lambda self: f"echo {shlex.quote(payload)} > config"
    )
    servers = TaskConfig.model_validate(tomllib.loads(te._fixture_integration())).environment.mcp_servers
    assert registration.probe_registration("codex", servers)["status"] == "unsupported"


@pytest.mark.parametrize(
    "agent_name, status",
    [
        (None, "unverified"),
        ("unknown", "unverified"),
        ("claude-code", "verified"),
        ("vibe", "verified"),
        ("codex", "unsupported"),
    ],
)
def test_runtime_reports_registration_separately_from_execution(tmp_path, monkeypatch, agent_name, status):
    if status == "unsupported":
        monkeypatch.setattr(
            Codex, "_build_register_mcp_servers_command", lambda self: "echo '{\"mcpServers\": {}}' > config"
        )
    task_dir = _generated_mock(tmp_path)
    config_path = task_dir / "task/task.toml"
    config_path.write_text(
        '[environment]\nnetwork_mode = "no-network"\n'
        '[verifier]\nenvironment_mode = "separate"\nnetwork_mode = "no-network"\n'
        '[verifier.environment]\nnetwork_mode = "no-network"\n' + te._fixture_integration()
    )
    (task_dir / "task/instruction.md").write_text("Exercise the synthetic mock.\n")
    (task_dir / "task/tests").mkdir()
    (task_dir / "task/tests/test.sh").write_text("#!/bin/sh\nexit 0\n")
    result = te._check_runtime(argparse.Namespace(task_dir=task_dir, mock_agent=agent_name))
    assert result["mock_integration"]["registration"]["status"] == status
    assert result["mock_integration"]["execution"]["status"] == "unverified"
    assert not result["execution_verified"]
    assert next(check for check in result["checks"] if check["name"] == "mock_agent_registration")["passed"] == (
        status != "unsupported"
    )
    if status == "unsupported":
        assert not result["valid"]


def test_probe_never_executes_unrecognized_shell(monkeypatch, tmp_path):
    marker = tmp_path / "must-not-exist"
    monkeypatch.setattr(Codex, "_build_register_mcp_servers_command", lambda self: f"touch {shlex.quote(str(marker))}")
    servers = TaskConfig.model_validate(tomllib.loads(te._fixture_integration())).environment.mcp_servers
    assert registration.probe_registration("codex", servers)["status"] == "unverified"
    assert not marker.exists()


@pytest.mark.parametrize(
    "command",
    [
        "echo '",
        "echo '{}' > config && touch elsewhere",
        "cat >> config <<EOF\n{}\nEOF",
        "cat >> config <<'EOF'\n{}\nEOF\ntouch elsewhere",
        'echo "{\\"mcp_servers\\": {\\"$(untrusted)\\": {}}}" > config',
    ],
)
def test_registration_reader_rejects_unrecognized_or_expanding_shell(command):
    assert registration.registration_payload(command) is None


def test_mock_agent_requires_task_scope():
    result = te._check_runtime(argparse.Namespace(task_dir=None, mock_agent="vibe"))
    assert not result["valid"]
    assert any(check["name"] == "mock_agent_scope" and not check["passed"] for check in result["checks"])


@pytest.mark.skipif(
    os.environ.get("TRACE_FIXTURE_LIVE_CODEX") != "1", reason="requires explicit live Codex/model authorization"
)
def test_codex_agent_calls_generated_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run Harbor's actual Codex agent with a constrained host process bridge."""
    _, registered = _registered_mock(tmp_path)
    auth_source = Path(os.environ["TRACE_FIXTURE_CODEX_AUTH"])
    model = os.environ["TRACE_FIXTURE_CODEX_MODEL"]
    agent_logs = tmp_path / "agent"
    secrets_dir = tmp_path / "harbor-secrets"
    config = TaskConfig.model_validate(
        {"environment": {"mcp_servers": [{"name": "trace-tool-call-replay", "transport": "stdio", **registered}]}}
    )
    agent = Codex(
        logs_dir=agent_logs,
        model_name=model,
        mcp_servers=config.environment.mcp_servers,
        extra_env={"CODEX_AUTH_JSON_PATH": str(auth_source)},
    )
    # Map provider-owned container paths to this private host test directory.
    # Registration, auth setup, execution, and cleanup still run through Codex.run.
    monkeypatch.setattr(agent, "_REMOTE_CODEX_HOME", PurePosixPath(tmp_path / "harbor-home"))
    monkeypatch.setattr(agent, "_REMOTE_CODEX_SECRETS_DIR", PurePosixPath(secrets_dir))
    monkeypatch.setattr(EnvironmentPaths, "agent_dir", PurePosixPath(agent_logs))

    async def exec_host(command: str, *, env=None, cwd=None, **kwargs) -> ExecResult:
        # Harbor's installed agent assumes container isolation. This bridge is
        # deliberately stricter on the host; never run its sandbox-bypass flag.
        command = command.replace("--dangerously-bypass-approvals-and-sandbox", "--sandbox read-only")
        result = await asyncio.to_thread(
            subprocess.run,
            ["bash", "-c", command],
            env={**os.environ, **(env or {})},
            cwd=cwd or tmp_path,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert isinstance(result.stdout, str) and isinstance(result.stderr, str)
        if "codex exec" in command:
            (tmp_path / "codex-stderr.log").write_text(result.stderr)
        return ExecResult(stdout=result.stdout, stderr=result.stderr, return_code=result.returncode)

    async def upload_file(source_path: Path, target_path: str) -> None:
        target = Path(target_path)
        assert target.is_relative_to(tmp_path)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copyfile(source_path, target)
        target.chmod(0o600)

    environment = MagicMock(spec=BaseEnvironment)
    environment.default_user = None
    environment.exec = exec_host
    environment.upload_file = upload_file
    audit = Path(te.TOOL_CALL_AUDIT_LOG)
    audit_offset = audit.stat().st_size if audit.exists() else 0
    prompt = (
        "Test the registered trace-tool-call-replay MCP server. Use its account.lookup tool with "
        'account_id="synthetic-001", then use the same tool with account_id="unseen". '
        "Do not read files or use shell tools. Report the successful result and whether the unseen input returned an error."
    )
    try:
        asyncio.run(agent.run(prompt, environment, AgentContext()))
        assert audit.is_file(), f"No native mock call occurred; inspect {tmp_path}"
        with audit.open("rb") as stream:
            stream.seek(audit_offset)
            audit_bytes = stream.read()
        (tmp_path / "live-audit.jsonl").write_bytes(audit_bytes)
        events = [json.loads(line) for line in audit_bytes.splitlines()]
        calls = [event for event in events if event.get("method") == "tools/call" and "matched" in event]
        assert [(call["arguments"], call["matched"]) for call in calls] == [
            ({"account_id": "synthetic-001"}, True),
            ({"account_id": "unseen"}, False),
        ]
        # Audit alone is agent-local evidence. Also require native MCP events from
        # the actual agent, not a shell invocation of the fixture script.
        agent_events = [
            json.loads(line) for line in (agent_logs / "codex.txt").read_text().splitlines() if line.startswith("{")
        ]
        completed = [event["item"] for event in agent_events if event.get("type") == "item.completed"]
        native = [item for item in completed if item.get("type") == "mcp_tool_call"]
        assert len(native) == 2, f"Expected two native tool calls; inspect {tmp_path}"
        assert all(item["server"] == "trace-tool-call-replay" for item in native)
        assert native[0]["result"]["content"] == [{"type": "text", "text": '{"status":"active"}'}]
        assert native[0]["status"] == "completed" and native[1]["status"] == "failed"
        assert not any(item.get("type") == "command_execution" for item in completed)
    finally:
        (secrets_dir / "auth.json").unlink(missing_ok=True)
