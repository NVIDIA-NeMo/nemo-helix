# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for _prepare_aut_config_for_runtime IGW routing logic."""

import json
import stat
import subprocess
import sys
from importlib import util
from pathlib import Path
from typing import cast

import nat_runner
import pytest
import yaml
from nat_runner import (
    _agent_log_has_workflow_error,
    _agent_model_for_backend,
    _agent_model_from_env,
    _build_aut_agent_cmd,
    _build_claude_code_agent_cmd,
    _build_codex_agent_cmd,
    _build_cursor_agent_cmd,
    _build_workflow_agent_cmd,
    _extract_usage_metrics,
    _normalize_secret,
    _prepare_aut_config_for_runtime,
    _prepare_workflow_for_runtime,
    run_agent_phase,
    run_task,
    run_verify_phase,
)

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
TRACE_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "trace_outputs"

spec = util.spec_from_file_location("nat_trace_export", SCRIPTS_DIR / "nat_trace_export.py")
assert spec is not None
assert spec.loader is not None
nat_trace_export = util.module_from_spec(spec)
sys.modules[spec.name] = nat_trace_export
spec.loader.exec_module(nat_trace_export)


def _run_trace_export_cli(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["nat_trace_export.py", *args])
    assert nat_trace_export.main() == 0


def _load_valid_atif(path: Path) -> dict[str, object]:
    from nat.atif import ATIFTrajectory

    payload = json.loads(path.read_text(encoding="utf-8"))
    ATIFTrajectory.model_validate(payload)
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)


def _trajectory_steps(trajectory: dict[str, object]) -> list[dict[str, object]]:
    steps = trajectory["steps"]
    assert isinstance(steps, list)
    return cast(list[dict[str, object]], steps)


def _trajectory_tool_steps(trajectory: dict[str, object]) -> list[dict[str, object]]:
    return [step for step in _trajectory_steps(trajectory) if step.get("tool_calls")]


@pytest.fixture()
def aut_config(tmp_path: Path) -> Path:
    """Write a minimal AUT-style YAML config and return the path."""
    config = {
        "llms": {
            "agent": {
                "_type": "openai",
                "api_key": "not-used",
                "model_name": "aws-anthropic-claude-opus-4-5",
            }
        }
    }
    p = tmp_path / "agent.yml"
    with p.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return p


class TestIGWRouting:
    """Verify _prepare_aut_config_for_runtime injects IGW routing via inject_gateway_url."""

    def test_injects_igw_base_url(self, aut_config: Path, tmp_path: Path) -> None:
        result = _prepare_aut_config_for_runtime(aut_config, tmp_path)
        cfg = yaml.safe_load(result.read_text())
        expected = "http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
        assert cfg["llms"]["agent"]["base_url"] == expected

    def test_preserves_entity_model_name(self, aut_config: Path, tmp_path: Path) -> None:
        result = _prepare_aut_config_for_runtime(aut_config, tmp_path)
        cfg = yaml.safe_load(result.read_text())
        assert cfg["llms"]["agent"]["model_name"] == "aws-anthropic-claude-opus-4-5"

    def test_overrides_model_name_with_nat_model(self, aut_config: Path, tmp_path: Path) -> None:
        result = _prepare_aut_config_for_runtime(aut_config, tmp_path, nat_model="custom-model")
        cfg = yaml.safe_load(result.read_text())
        assert cfg["llms"]["agent"]["model_name"] == "custom-model"

    def test_resolves_default_model_placeholder_from_host_context(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nemo_agents_plugin import utils as agent_utils

        monkeypatch.setattr(agent_utils, "get_default_model", lambda: "nvidia-nemotron-3-nano-30b-a3b")
        config = {
            "llms": {
                "agent": {
                    "_type": "openai",
                    "api_key": "not-used",
                    "model_name": "${NEMO_DEFAULT_MODEL}",
                }
            }
        }
        config_path = tmp_path / "agent.yml"
        with config_path.open("w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        result = _prepare_aut_config_for_runtime(config_path, tmp_path)
        cfg = yaml.safe_load(result.read_text())

        assert cfg["llms"]["agent"]["model_name"] == "nvidia-nemotron-3-nano-30b-a3b"
        assert "${NEMO_DEFAULT_MODEL}" not in result.read_text()

    def test_custom_workspace(self, aut_config: Path, tmp_path: Path) -> None:
        result = _prepare_aut_config_for_runtime(aut_config, tmp_path, workspace="staging")
        cfg = yaml.safe_load(result.read_text())
        assert "/workspaces/staging/" in cfg["llms"]["agent"]["base_url"]

    def test_preserves_explicit_base_url(self, tmp_path: Path) -> None:
        """inject_gateway_url uses setdefault -- explicit base_url is preserved."""
        config = {
            "llms": {
                "agent": {
                    "_type": "openai",
                    "api_key": "real-key",
                    "base_url": "http://my-custom-endpoint/v1",
                    "model_name": "my-model",
                }
            }
        }
        p = tmp_path / "explicit.yml"
        with p.open("w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        result = _prepare_aut_config_for_runtime(p, tmp_path)
        cfg = yaml.safe_load(result.read_text())
        assert cfg["llms"]["agent"]["base_url"] == "http://my-custom-endpoint/v1"
        assert cfg["llms"]["agent"]["api_key"] == "real-key"

    def test_writes_rewritten_to_output_dir(self, aut_config: Path, tmp_path: Path) -> None:
        result = _prepare_aut_config_for_runtime(aut_config, tmp_path)
        assert result.parent == tmp_path
        assert result.name == "aut.runtime.yml"

    def test_custom_base_url_and_port(self, aut_config: Path, tmp_path: Path) -> None:
        result = _prepare_aut_config_for_runtime(
            aut_config,
            tmp_path,
            nhx_base_url="http://myhost:9090",
        )
        cfg = yaml.safe_load(result.read_text())
        assert cfg["llms"]["agent"]["base_url"].startswith("http://myhost:9090/")

    def test_trailing_slash_stripped(self, aut_config: Path, tmp_path: Path) -> None:
        result = _prepare_aut_config_for_runtime(
            aut_config,
            tmp_path,
            nhx_base_url="http://localhost:8080/",
        )
        cfg = yaml.safe_load(result.read_text())
        assert "//apis" not in cfg["llms"]["agent"]["base_url"]

    def test_non_openai_llm_not_injected(self, tmp_path: Path) -> None:
        """LLMs with _type != openai/nim are left untouched."""
        config = {
            "llms": {
                "agent": {
                    "_type": "huggingface",
                    "model_name": "some-model",
                }
            }
        }
        p = tmp_path / "hf.yml"
        with p.open("w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        result = _prepare_aut_config_for_runtime(p, tmp_path)
        cfg = yaml.safe_load(result.read_text())
        assert "base_url" not in cfg["llms"]["agent"]


def test_parse_candidate_params_normalizes_supported_values() -> None:
    params = nat_runner._parse_candidate_params(
        json.dumps(
            {
                "intelligence": "high",
                "speed": "fast",
                "max_budget_usd": "1.25",
                "config": {"model_reasoning_summary": "auto"},
            }
        )
    )

    assert params == {
        "intelligence": "high",
        "speed": "fast",
        "max_budget_usd": 1.25,
        "config": {"model_reasoning_summary": "auto"},
    }


def test_parse_candidate_params_rejects_unsupported_keys() -> None:
    with pytest.raises(ValueError, match="unsupported key"):
        nat_runner._parse_candidate_params('{"ignored": true}')


def test_parse_candidate_params_rejects_wrong_types() -> None:
    with pytest.raises(ValueError, match="'sandbox' must be a string"):
        nat_runner._parse_candidate_params('{"sandbox": false}')


class TestUsageMetrics:
    def test_extracts_cursor_camel_case_usage(self) -> None:
        metrics = _extract_usage_metrics(
            json.dumps(
                {
                    "type": "result",
                    "usage": {
                        "inputTokens": 100,
                        "outputTokens": 20,
                        "cacheReadTokens": 300,
                        "cacheWriteTokens": 4,
                    },
                }
            )
        )

        assert metrics["prompt_tokens"] == 100
        assert metrics["completion_tokens"] == 20
        assert metrics["cache_read_tokens"] == 300
        assert metrics["cache_creation_tokens"] == 4
        assert metrics["total_tokens"] == 424

    def test_extracts_zero_usage_when_usage_envelope_is_present(self) -> None:
        metrics = _extract_usage_metrics(
            json.dumps(
                {
                    "type": "result",
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    },
                    "num_turns": 1,
                }
            )
        )

        assert metrics["prompt_tokens"] == 0
        assert metrics["completion_tokens"] == 0
        assert metrics["cache_read_tokens"] == 0
        assert metrics["cache_creation_tokens"] == 0
        assert metrics["total_tokens"] == 0
        assert metrics["num_turns"] == 1

    def test_extracts_codex_cached_input_tokens(self) -> None:
        metrics = _extract_usage_metrics(
            json.dumps(
                {
                    "type": "result",
                    "usage": {
                        "input_tokens": 195742,
                        "output_tokens": 4012,
                        "cached_input_tokens": 125312,
                    },
                }
            )
        )

        assert metrics["prompt_tokens"] == 70430
        assert metrics["completion_tokens"] == 4012
        assert metrics["cache_read_tokens"] == 125312
        assert metrics["cache_creation_tokens"] is None
        assert metrics["total_tokens"] == 199754


class TestAgentBackends:
    """Verify backend-specific command construction and metadata helpers."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, ""),
            ("", ""),
            ("  ", ""),
            ("null", ""),
            (" NULL ", ""),
            ("None", ""),
            (" none ", ""),
            ("actual-secret", "actual-secret"),
            (" actual-secret ", "actual-secret"),
        ],
    )
    def test_normalize_secret(self, value: str | None, expected: str) -> None:
        assert _normalize_secret(value) == expected

    def test_build_claude_code_agent_cmd_uses_unified_agent_model(self) -> None:
        cmd = _build_claude_code_agent_cmd(
            "/tmp/instruction.md", {"permission_mode": "bypassPermissions", "max_budget_usd": 1.5}
        )
        script = cmd[2]

        assert cmd[:2] == ["bash", "-c"]
        assert "claude -p" in script
        assert "AGENT_MODEL" in script
        assert "CLAUDE_MODEL" not in script
        assert "--output-format json" in script
        assert "--permission-mode bypassPermissions" in script
        assert "--max-budget-usd 1.5" in script
        assert "CLAUDE_CONFIG_DIR=/logs/agent/sessions" in script
        assert "nat_trace_export.py convert-claude-session" in script
        assert "--projects-dir /logs/agent/sessions/projects" in script
        assert "--output /logs/agent/trajectory.json" in script
        assert "--instruction /tmp/instruction.md" in script

    def test_build_aut_agent_cmd_recreates_existing_agent_when_config_is_present(self) -> None:
        cmd = _build_aut_agent_cmd("/tmp/instruction.md")
        script = cmd[2]

        assert 'if [ -n "${EFFECTIVE_AUT_AGENT_CONFIG}" ]; then' in script
        assert 'nemo agents undeploy --agent "${AUT_AGENT_NAME}"' in script
        assert 'nemo agents delete "${AUT_AGENT_NAME}"' in script
        assert 'nemo agents create --name "${AUT_AGENT_NAME}" --agent-config "${EFFECTIVE_AUT_AGENT_CONFIG}"' in script

    def test_build_aut_agent_cmd_exports_atif_trace_artifacts(self) -> None:
        cmd = _build_aut_agent_cmd("/tmp/instruction.md")
        script = cmd[2]

        assert cmd[:2] == ["bash", "-c"]
        assert "nat_trace_export.py invoke-aut" in script
        assert '--endpoint "$dep_endpoint"' in script
        assert "--instruction /tmp/instruction.md" in script
        assert "--output-dir /logs/agent" in script
        assert "/generate/full" not in script

    def test_build_workflow_agent_cmd_exports_atif_trace_artifacts(self) -> None:
        cmd = _build_workflow_agent_cmd("/tmp/nat_workflow.yml", "/tmp/instruction.md")
        script = cmd[2]

        assert cmd[:2] == ["bash", "-c"]
        assert "/app/.venv/bin/nat run" in script
        assert "--config_file /tmp/nat_workflow.yml" in script
        assert '--input "$(cat /tmp/instruction.md)"' in script
        assert "nat_trace_export.py convert-jsonl" in script
        assert "--input /logs/agent/intermediate_steps.jsonl" in script
        assert "--output /logs/agent/trajectory.json" in script
        assert "NAT telemetry exporter did not create /logs/agent/intermediate_steps.jsonl" in script

    def test_prepare_workflow_for_runtime_injects_file_telemetry(self, tmp_path: Path) -> None:
        workflow = tmp_path / "workflow.yml"
        workflow.write_text(
            """
llms:
  agent:
    _type: openai
    base_url: http://localhost:8080/v1
    model_name: nvidia/llama-3.1-nemotron-70b-instruct
workflow:
  _type: react_agent
  llm_name: agent
""".strip(),
            encoding="utf-8",
        )

        runtime_workflow = _prepare_workflow_for_runtime(
            workflow,
            tmp_path,
            nhx_base_url="http://platform:8080",
            nat_model="custom-model",
        )
        config = yaml.safe_load(runtime_workflow.read_text(encoding="utf-8"))

        trace_config = config["general"]["telemetry"]["tracing"]["agentic_use_file_trace"]
        assert config["llms"]["agent"]["base_url"] == "http://platform:8080/v1"
        assert config["llms"]["agent"]["model_name"] == "custom-model"
        assert trace_config == {
            "_type": "file",
            "output_path": "/logs/agent/intermediate_steps.jsonl",
            "project": "agentic-use",
            "mode": "overwrite",
            "cleanup_on_init": True,
        }

    def test_nat_trace_export_assembles_atif_stream(self) -> None:
        events = [
            {
                "step_id": 1,
                "source": "user",
                "message": "Do the thing.",
            },
            {
                "step_id": 2,
                "source": "agent",
                "message": "Done.",
            },
            {"output": "final answer"},
            {
                "schema_version": "ATIF-v1.7",
                "session_id": "session-1",
                "agent": {"name": "nat-agent", "version": "0.0.0"},
            },
        ]

        trajectory, final_result = nat_trace_export._assemble_atif_stream(events)

        assert final_result == {"output": "final answer"}
        assert trajectory is not None
        assert trajectory["session_id"] == "session-1"
        assert len(trajectory["steps"]) == 2

    def test_nat_trace_export_assembles_legacy_sse_delta_stream(self) -> None:
        events = [
            {"value": " The"},
            {"value": " task"},
            {"value": " is"},
            {"value": " complete."},
        ]

        assert nat_trace_export._assemble_legacy_stream(events) == "The task is complete."

    def test_nat_trace_export_assembles_legacy_sse_cumulative_stream(self) -> None:
        events = [
            {"value": "partial answer"},
            {"value": "complete final answer"},
        ]

        assert nat_trace_export._assemble_legacy_stream(events) == "complete final answer"

    def test_invoke_aut_fallback_parses_legacy_sse_stream(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        requested_urls: list[str] = []

        def fake_read_http_response(url: str, _payload: bytes, _timeout: int) -> str:
            requested_urls.append(url)
            if url.endswith("/generate/atif"):
                raise nat_trace_export.urllib.error.HTTPError(url, 404, "not found", hdrs={}, fp=None)
            assert url.endswith("/generate/full?filter_steps=none")
            return "\n\n".join(
                [
                    'data: {"value": " The"}',
                    'data: {"value": " task"}',
                    'data: {"value": " is"}',
                    'data: {"value": " complete."}',
                ]
            )

        monkeypatch.setattr(nat_trace_export, "_read_http_response", fake_read_http_response)

        rc = nat_trace_export._invoke_aut("http://aut.example", "Finish the task.", tmp_path, timeout=10)

        assert rc == 0
        assert requested_urls == [
            "http://aut.example/generate/atif",
            "http://aut.example/generate/full?filter_steps=none",
        ]
        assert json.loads((tmp_path / "final_message.json").read_text(encoding="utf-8")) == {
            "result": "The task is complete."
        }
        assert (tmp_path / "final_message.txt").read_text(encoding="utf-8") == "The task is complete.\n"
        assert json.loads((tmp_path / "legacy_stream_events.json").read_text(encoding="utf-8")) == [
            {"value": " The"},
            {"value": " task"},
            {"value": " is"},
            {"value": " complete."},
        ]

    def test_build_aut_agent_cmd_keeps_diagnostics_best_effort(self) -> None:
        cmd = _build_aut_agent_cmd("/tmp/instruction.md")
        script = cmd[2]

        assert "set +e" in script
        assert "cleanup_rc=$?" in script
        assert 'exit "$cleanup_rc"' in script
        assert ">/tmp/aut_deployments.list.json 2>&1" in script
        assert ">/logs/agent/aut_deployments.list.json" not in script
        assert "mkdir -p /logs/agent/nat_subprocess_logs 2>/dev/null || true" in script

    def test_build_aut_agent_cmd_has_valid_shell_syntax(self) -> None:
        script = _build_aut_agent_cmd("/tmp/instruction.md")[2]

        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_build_codex_agent_cmd_runs_headlessly_with_artifacts(self) -> None:
        cmd = _build_codex_agent_cmd("/tmp/instruction.md", {"intelligence": "high", "speed": "fast"})
        script = cmd[2]

        assert cmd[:2] == ["bash", "-c"]
        assert nat_runner.CODEX_AGENT_SCRIPT_TEMPLATE_PATH.is_file()
        assert "codex exec" in script
        assert "--sandbox danger-full-access" in script
        assert "--ignore-user-config" in script
        assert "--cd /app" in script
        assert "/tmp/codex-task" not in script
        assert "--output-schema" not in script
        assert "--output-last-message /logs/agent/final_message.txt" in script
        assert "codex_structured_prompt.md" not in script
        assert "output_files" not in script
        assert "final_message.txt" in script
        assert "nat_trace_export.py convert-codex-jsonl" in script
        assert "--input /logs/agent/nat_agent.log" in script
        assert "--output /logs/agent/trajectory.json" in script
        assert "--final-message /logs/agent/final_message.txt" in script
        assert "--dangerously-bypass-approvals-and-sandbox" not in script
        assert "--json" in script
        assert "AGENT_MODEL" in script
        assert "CODEX_HOME" in script
        assert 'CODEX_HOME="/home/harbor/codex-benchmark-home"' in script
        assert 'rm -rf "$CODEX_HOME"' in script
        assert "/home/harbor/.codex" not in script
        assert "/tmp/codex_host_auth.json" in script
        assert "CODEX_MODEL" not in script
        assert "-c 'model_reasoning_effort=\"high\"'" in script
        assert "-c 'service_tier=\"fast\"'" in script
        assert "- < /tmp/instruction.md" in script
        assert "@@INSTRUCTION_CONTAINER@@" not in script
        assert "@@CODEX_CONFIG_ARGS@@" not in script

    def test_codex_agent_script_template_has_valid_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(nat_runner.CODEX_AGENT_SCRIPT_TEMPLATE_PATH)], check=True)

    def test_build_cursor_agent_cmd_runs_headlessly_with_artifacts(self) -> None:
        cmd = _build_cursor_agent_cmd("/tmp/instruction.md", {"sandbox": "enabled", "mode": "plan"})
        script = cmd[2]

        assert cmd[:2] == ["bash", "-c"]
        assert "cursor-agent" in script
        assert "--print" in script
        assert "--output-format stream-json" in script
        assert "--output-format json" not in script
        assert "--force" not in script
        assert "--sandbox enabled" in script
        assert "--mode plan" in script
        assert "--workspace /app" in script
        assert "nat_trace_export.py convert-cursor-jsonl" in script
        assert "--input /logs/agent/nat_agent.log" in script
        assert "--output /logs/agent/trajectory.json" in script
        assert "--final-message /logs/agent/final_message.json" in script
        assert "AGENT_MODEL" in script
        assert "CURSOR_MODEL" not in script
        assert "/tmp/instruction.md" in script

    def test_build_cursor_agent_cmd_preserves_default_sandbox_when_unset(self) -> None:
        cmd = _build_cursor_agent_cmd("/tmp/instruction.md", {})
        script = cmd[2]

        assert "cursor-agent" in script
        assert "--sandbox" not in script
        assert "--workspace /app" in script

    @pytest.mark.parametrize(
        ("agent_backend", "expected"),
        [
            ("claude-code", "agent-model"),
            ("codex", "agent-model"),
            ("cursor-agent", "agent-model"),
            ("workflow", "agent-model"),
            ("aut", "agent-model"),
        ],
    )
    def test_agent_model_for_backend(self, agent_backend: str, expected: str) -> None:
        assert (
            _agent_model_for_backend(
                agent_backend=agent_backend,
                agent_model="agent-model",
            )
            == expected
        )

    def test_agent_model_for_backend_defaults_optional_direct_agent_models(self) -> None:
        assert (
            _agent_model_for_backend(
                agent_backend="codex",
                agent_model=None,
            )
            == "default"
        )
        assert (
            _agent_model_for_backend(
                agent_backend="cursor-agent",
                agent_model=None,
            )
            == "default"
        )
        assert (
            _agent_model_for_backend(
                agent_backend="claude-code",
                agent_model=None,
            )
            == "default"
        )
        assert (
            _agent_model_for_backend(
                agent_backend="aut",
                agent_model=None,
            )
            == "unknown"
        )

    def test_agent_model_from_env_prefers_unified_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NAT_AGENT_MODEL", "unified")
        monkeypatch.setenv("CODEX_MODEL", "legacy-codex")

        assert _agent_model_from_env("codex") == "unified"

    def test_agent_model_from_env_reads_backend_legacy_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CURSOR_MODEL", "legacy-cursor")

        assert _agent_model_from_env("cursor-agent") == "legacy-cursor"

    def test_agent_model_from_env_reads_legacy_nat_model_for_platform_backends(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NAT_MODEL", "legacy-nat")

        assert _agent_model_from_env("workflow") == "legacy-nat"

    def test_run_agent_phase_scopes_direct_agent_secrets(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        (task_dir / "instruction.md").write_text("Do the thing.")
        output_dir = tmp_path / "out"
        state_dir = tmp_path / "state"
        workspace_dir = tmp_path / "workspace"

        monkeypatch.setenv("OPENAI_API_KEY", " openai-secret ")
        monkeypatch.setenv("CURSOR_API_KEY", "cursor-secret")
        monkeypatch.setenv("INFERENCE_NVIDIA_API_KEY", "inference-secret")
        captured_env: dict[str, str] = {}

        def fake_docker_run(*_args: object, **kwargs: object) -> subprocess.CompletedProcess:
            env = cast(dict[str, str], kwargs["env"])
            captured_env.update(env)
            return subprocess.CompletedProcess(args=["docker"], returncode=0, stdout="", stderr="")

        monkeypatch.setattr(nat_runner, "_docker_run", fake_docker_run)

        assert run_agent_phase(
            task_dir,
            "task-image",
            output_dir,
            nvidia_api_key="nvidia-secret",
            anthropic_api_key="anthropic-secret",
            anthropic_base_url="https://anthropic.example",
            nhx_base_url="http://localhost:8080",
            agent_model=None,
            agent_params={},
            codex_auth_json=None,
            timeout=10,
            agent_backend="codex",
            aut_agent_name="test-agent",
            aut_agent_config=None,
            aut_seed_providers=True,
            state_dir=state_dir,
            workspace_dir=workspace_dir,
        )

        assert captured_env["OPENAI_API_KEY"] == "openai-secret"
        assert captured_env["AGENTIC_USE_WORKSPACE_DIR"] == "/app/workspace"
        assert "CURSOR_API_KEY" not in captured_env
        assert "NVIDIA_API_KEY" not in captured_env
        assert "ANTHROPIC_API_KEY" not in captured_env
        assert "INFERENCE_NVIDIA_API_KEY" not in captured_env

    def test_run_agent_phase_mounts_explicit_codex_auth_read_only(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        codex_home = tmp_path / ".codex"
        codex_home.mkdir()
        codex_auth = codex_home / "auth.json"
        codex_auth.write_text("{}")

        task_dir = tmp_path / "task"
        task_dir.mkdir()
        (task_dir / "instruction.md").write_text("Do the thing.")
        output_dir = tmp_path / "out"
        state_dir = tmp_path / "state"
        workspace_dir = tmp_path / "workspace"
        captured_mounts: list[tuple[str, str]] = []

        def fake_docker_run(*_args: object, **kwargs: object) -> subprocess.CompletedProcess:
            mounts = cast(list[tuple[str, str]], kwargs["mounts"])
            captured_mounts.extend(mounts)
            return subprocess.CompletedProcess(args=["docker"], returncode=0, stdout="", stderr="")

        monkeypatch.setattr(nat_runner, "_docker_run", fake_docker_run)

        assert run_agent_phase(
            task_dir,
            "task-image",
            output_dir,
            nvidia_api_key="nvidia-secret",
            anthropic_api_key="anthropic-secret",
            anthropic_base_url="https://anthropic.example",
            nhx_base_url="http://localhost:8080",
            agent_model=None,
            agent_params={},
            codex_auth_json=codex_auth,
            timeout=10,
            agent_backend="codex",
            aut_agent_name="test-agent",
            aut_agent_config=None,
            aut_seed_providers=True,
            state_dir=state_dir,
            workspace_dir=workspace_dir,
        )

        assert (str(codex_auth), "/tmp/codex_host_auth.json:ro") in captured_mounts
        assert (str(workspace_dir), "/app/workspace") in captured_mounts
        assert (str(codex_home), "/tmp/codex_host_home") not in captured_mounts

    def test_run_agent_phase_relaxes_bind_mount_directory_permissions(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        (task_dir / "instruction.md").write_text("Do the thing.")
        output_dir = tmp_path / "out"
        state_dir = tmp_path / "state"
        workspace_dir = tmp_path / "workspace"
        state_dir.mkdir()
        workspace_dir.mkdir()
        state_dir.chmod(0o775)
        workspace_dir.chmod(0o775)

        monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
        captured_mounts: list[tuple[str, str]] = []

        def fake_docker_run(*_args: object, **kwargs: object) -> subprocess.CompletedProcess:
            mounts = cast(list[tuple[str, str]], kwargs["mounts"])
            captured_mounts.extend(mounts)
            return subprocess.CompletedProcess(args=["docker"], returncode=0, stdout="", stderr="")

        monkeypatch.setattr(nat_runner, "_docker_run", fake_docker_run)

        assert run_agent_phase(
            task_dir,
            "task-image",
            output_dir,
            nvidia_api_key="",
            anthropic_api_key="",
            anthropic_base_url="https://anthropic.example",
            nhx_base_url="http://localhost:8080",
            agent_model=None,
            agent_params={},
            codex_auth_json=None,
            timeout=10,
            agent_backend="codex",
            aut_agent_name="test-agent",
            aut_agent_config=None,
            aut_seed_providers=True,
            state_dir=state_dir,
            workspace_dir=workspace_dir,
        )

        agent_log_dir = output_dir / "agent"
        assert stat.S_IMODE(agent_log_dir.stat().st_mode) == 0o777
        assert stat.S_IMODE(state_dir.stat().st_mode) == 0o777
        assert stat.S_IMODE(workspace_dir.stat().st_mode) == 0o777
        assert (str(agent_log_dir), "/logs/agent") in captured_mounts
        assert (str(state_dir), "/data") in captured_mounts
        assert (str(workspace_dir), "/app/workspace") in captured_mounts

    def test_run_verify_phase_mounts_shared_evaluator_harness(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        task_dir = tmp_path / "task"
        tests_dir = task_dir / "tests"
        tests_dir.mkdir(parents=True)
        (tests_dir / "test_outputs.py").write_text("def test_ok(): pass\n")
        output_dir = tmp_path / "out"
        (output_dir / "agent").mkdir(parents=True)
        state_dir = tmp_path / "state"
        workspace_dir = tmp_path / "workspace"
        captured_mounts: list[tuple[str, str]] = []

        def fake_docker_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
            mounts = kwargs["mounts"]
            assert isinstance(mounts, list)
            captured_mounts.extend(cast(list[tuple[str, str]], mounts))
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        monkeypatch.setattr(nat_runner, "_docker_run", fake_docker_run)

        passed, _stdout = run_verify_phase(
            task_dir,
            "task-image",
            output_dir,
            nhx_base_url="http://localhost:8080",
            state_dir=state_dir,
            workspace_dir=workspace_dir,
            agent_backend="codex",
            agent_model="default",
        )

        assert passed is True
        assert (str(nat_runner.SHARED_DIR), "/app/tests/agentic-use/shared:ro") in captured_mounts
        assert (
            str(nat_runner.REPO_ROOT / "packages" / "nemo_evaluator_sdk" / "src"),
            "/app/packages/nemo_evaluator_sdk/src:ro",
        ) in captured_mounts

    def test_run_task_records_agent_metadata(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        task_dir = tmp_path / "tasks" / "task-a"
        task_dir.mkdir(parents=True)
        (task_dir / "instruction.md").write_text("Do the thing.")
        jobs_dir = tmp_path / "jobs"

        monkeypatch.setattr(nat_runner, "TASKS_DIR", tmp_path / "tasks")
        monkeypatch.setattr(nat_runner, "_docker_image_exists", lambda _image: True)
        monkeypatch.setattr(nat_runner, "run_verify_phase", lambda *args, **kwargs: (True, "ok"))

        result = run_task(
            "task-a",
            jobs_dir=jobs_dir,
            nvidia_api_key="",
            anthropic_api_key="",
            anthropic_base_url="https://anthropic.example",
            nhx_base_url="http://localhost:8080",
            agent_model="gpt-test",
            agent_params={"intelligence": "high", "speed": "fast"},
            codex_auth_json=None,
            agent_timeout=10,
            skip_build=True,
            build_only=False,
            skip_agent=True,
            agent_backend="codex",
            aut_agent_name="nemo-agent",
            aut_agent_config=None,
            aut_seed_providers=True,
            smoke_workspace=None,
            candidate_id="codex-gpt-test",
            provenance={"commit_short": "abc123"},
        )

        assert result["agent_backend"] == "codex"
        assert result["agent_model"] == "gpt-test"
        assert result["candidate_params"] == {"intelligence": "high", "speed": "fast"}
        assert result["candidate_id"] == "codex-gpt-test"
        result_file = next(jobs_dir.rglob("result.json"))
        payload = json.loads(result_file.read_text())
        assert payload["agent_backend"] == "codex"
        assert payload["agent_model"] == "gpt-test"
        assert payload["candidate_params"] == {"intelligence": "high", "speed": "fast"}
        assert payload["candidate_id"] == "codex-gpt-test"

    def test_run_task_build_only_skips_agent_and_verify(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        task_dir = tmp_path / "tasks" / "task-a"
        task_dir.mkdir(parents=True)
        (task_dir / "instruction.md").write_text("Do the thing.")
        jobs_dir = tmp_path / "jobs"
        built_tags: list[str] = []

        monkeypatch.setattr(nat_runner, "TASKS_DIR", tmp_path / "tasks")
        monkeypatch.setattr(nat_runner, "build_task_image", lambda _task_dir, tag: built_tags.append(tag))
        monkeypatch.setattr(
            nat_runner,
            "run_agent_phase",
            lambda *args, **kwargs: pytest.fail("agent phase should be skipped"),
        )
        monkeypatch.setattr(
            nat_runner,
            "run_verify_phase",
            lambda *args, **kwargs: pytest.fail("verify phase should be skipped"),
        )

        result = run_task(
            "task-a",
            jobs_dir=jobs_dir,
            nvidia_api_key="",
            anthropic_api_key="",
            anthropic_base_url="https://anthropic.example",
            nhx_base_url="http://localhost:8080",
            agent_model=None,
            agent_params=None,
            codex_auth_json=None,
            agent_timeout=10,
            skip_build=False,
            build_only=True,
            skip_agent=False,
            agent_backend="codex",
            aut_agent_name="nemo-agent",
            aut_agent_config=None,
            aut_seed_providers=True,
            smoke_workspace=None,
        )

        assert built_tags == ["nhx-nat-task-a:latest"]
        assert result["build"] == "ok"
        assert result["agent"] == "skipped"
        assert result["verify"] == "skipped"
        assert result["reward"] is None

    def test_run_agent_phase_normalizes_platform_agent_secrets(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        (task_dir / "instruction.md").write_text("Do the thing.")
        output_dir = tmp_path / "out"
        state_dir = tmp_path / "state"
        workspace_dir = tmp_path / "workspace"

        monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
        monkeypatch.setenv("CURSOR_API_KEY", "cursor-secret")
        monkeypatch.setenv("INFERENCE_NVIDIA_API_KEY", " none ")
        # Also neutralize the NVIDIA_INFERENCE_API_KEY alias so the test is
        # deterministic regardless of host env. nat_runner falls back to this
        # var when INFERENCE_NVIDIA_API_KEY is unset/null.
        monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", " none ")
        captured_env: dict[str, str] = {}

        def fake_docker_run(*_args: object, **kwargs: object) -> subprocess.CompletedProcess:
            env = cast(dict[str, str], kwargs["env"])
            captured_env.update(env)
            return subprocess.CompletedProcess(args=["docker"], returncode=0, stdout="", stderr="")

        monkeypatch.setattr(nat_runner, "_docker_run", fake_docker_run)

        assert run_agent_phase(
            task_dir,
            "task-image",
            output_dir,
            nvidia_api_key=" nvidia-secret ",
            anthropic_api_key=" null ",
            anthropic_base_url="https://anthropic.example",
            nhx_base_url="http://localhost:8080",
            agent_model=None,
            agent_params={},
            codex_auth_json=None,
            timeout=10,
            agent_backend="aut",
            aut_agent_name="test-agent",
            aut_agent_config=None,
            aut_seed_providers=True,
            state_dir=state_dir,
            workspace_dir=workspace_dir,
        )

        assert captured_env["NVIDIA_API_KEY"] == "nvidia-secret"
        assert "ANTHROPIC_API_KEY" not in captured_env
        assert "OPENAI_API_KEY" not in captured_env
        assert "CURSOR_API_KEY" not in captured_env
        assert "INFERENCE_NVIDIA_API_KEY" not in captured_env


class TestTraceExportCliFixtures:
    """Exercise checked-in trace examples through the converter CLI surface."""

    def test_convert_nat_intermediate_steps_fixture(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        output_path = tmp_path / "trajectory.json"

        _run_trace_export_cli(
            monkeypatch,
            [
                "convert-jsonl",
                "--input",
                str(TRACE_FIXTURES_DIR / "nat_intermediate_steps.jsonl"),
                "--output",
                str(output_path),
                "--session-id",
                "nat-fixture-session",
            ],
        )

        trajectory = _load_valid_atif(output_path)
        assert trajectory["session_id"] == "nat-fixture-session"
        assert [step["source"] for step in _trajectory_steps(trajectory)] == ["user", "agent"]

    def test_convert_claude_session_fixture(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        instruction_path = tmp_path / "instruction.md"
        final_message_path = tmp_path / "final_message.json"
        output_path = tmp_path / "trajectory.json"
        instruction_path.write_text("Run the failing command.", encoding="utf-8")
        final_message_path.write_text(json.dumps({"result": "The command failed."}), encoding="utf-8")

        _run_trace_export_cli(
            monkeypatch,
            [
                "convert-claude-session",
                "--projects-dir",
                str(TRACE_FIXTURES_DIR / "claude" / "projects"),
                "--output",
                str(output_path),
                "--instruction",
                str(instruction_path),
                "--final-message",
                str(final_message_path),
            ],
        )

        trajectory = _load_valid_atif(output_path)
        tool_steps = _trajectory_tool_steps(trajectory)
        assert trajectory["agent"] == {"name": "claude-code", "version": "0.0.0"}
        assert _trajectory_steps(trajectory)[0]["message"] == "Run the failing command."
        assert cast(list[dict[str, object]], tool_steps[0]["tool_calls"])[0]["function_name"] == "Bash"
        assert "exit code 1" in str(cast(dict[str, object], tool_steps[0]["observation"])["results"])

    def test_convert_codex_jsonl_fixture(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        instruction_path = tmp_path / "instruction.md"
        final_message_path = tmp_path / "final_message.txt"
        output_path = tmp_path / "trajectory.json"
        instruction_path.write_text("Inspect the repo.", encoding="utf-8")
        final_message_path.write_text("Finished inspection.\n", encoding="utf-8")

        _run_trace_export_cli(
            monkeypatch,
            [
                "convert-codex-jsonl",
                "--input",
                str(TRACE_FIXTURES_DIR / "codex_exec.jsonl"),
                "--output",
                str(output_path),
                "--instruction",
                str(instruction_path),
                "--final-message",
                str(final_message_path),
            ],
        )

        trajectory = _load_valid_atif(output_path)
        tool_steps = _trajectory_tool_steps(trajectory)
        assert trajectory["agent"] == {"name": "codex", "version": "0.0.0"}
        assert _trajectory_steps(trajectory)[-1]["message"] == "Finished inspection."
        assert cast(list[dict[str, object]], tool_steps[0]["tool_calls"])[0]["function_name"] == "shell"
        assert "Evaluator" in str(cast(dict[str, object], tool_steps[0]["observation"])["results"])

    def test_convert_cursor_stream_fixture(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        instruction_path = tmp_path / "instruction.md"
        final_message_path = tmp_path / "final_message.json"
        output_path = tmp_path / "trajectory.json"
        instruction_path.write_text("List files.", encoding="utf-8")

        _run_trace_export_cli(
            monkeypatch,
            [
                "convert-cursor-jsonl",
                "--input",
                str(TRACE_FIXTURES_DIR / "cursor_stream.jsonl"),
                "--output",
                str(output_path),
                "--instruction",
                str(instruction_path),
                "--final-message",
                str(final_message_path),
            ],
        )

        trajectory = _load_valid_atif(output_path)
        final_message = json.loads(final_message_path.read_text(encoding="utf-8"))
        tool_steps = _trajectory_tool_steps(trajectory)
        tool_call = cast(list[dict[str, object]], tool_steps[0]["tool_calls"])[0]
        assert trajectory["agent"] == {"name": "cursor-agent", "version": "0.0.0"}
        assert final_message == {"result": "Done listing files."}
        assert tool_call["tool_call_id"] == "cursor-call-1"
        assert cast(dict[str, object], tool_call["arguments"])["command"] == "ls"
        assert "README.md" in str(cast(dict[str, object], tool_steps[0]["observation"])["results"])


class TestAgentLogWorkflowErrors:
    def test_detects_workflow_error_json_payload(self) -> None:
        agent_log = """
intermediate_data: {"event": "tool_start"}
{"code":"workflow_error","message":"Error in react_agent workflow","details":"RuntimeError"}
"""

        assert _agent_log_has_workflow_error(agent_log)

    def test_ignores_successful_messages_payload(self) -> None:
        agent_log = '{"messages":[{"role":"assistant","content":"completed"}]}'

        assert not _agent_log_has_workflow_error(agent_log)


class TestAutValidation:
    """Verify that nat_runner.main() rejects AUT mode when required args are missing."""

    @pytest.fixture(autouse=True)
    def _fake_api_key(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-key-for-test")

    def test_rejects_aut_without_agent_name(self, monkeypatch, tmp_path, capsys):
        config = tmp_path / "agent.yml"
        config.write_text("llms: {}\n")
        monkeypatch.setattr(
            "sys.argv",
            [
                "nat_runner.py",
                "workspace-basic-mcp",
                "--agent-backend",
                "aut",
                "--aut-agent-config",
                str(config),
            ],
        )
        monkeypatch.delenv("AUT_AGENT_NAME", raising=False)
        result = nat_runner.main()
        assert result == 1
        assert "requires --aut-agent-name" in capsys.readouterr().err

    def test_rejects_aut_without_agent_config(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv",
            [
                "nat_runner.py",
                "workspace-basic-mcp",
                "--agent-backend",
                "aut",
                "--aut-agent-name",
                "test-agent",
            ],
        )
        monkeypatch.delenv("AUT_AGENT_CONFIG", raising=False)
        result = nat_runner.main()
        assert result == 1
        assert "requires --aut-agent-config" in capsys.readouterr().err


class TestBuildOnlySummary:
    def test_main_reports_build_summary_in_build_only_mode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        task_dir = tmp_path / "tasks" / "task-a"
        task_dir.mkdir(parents=True)
        (task_dir / "instruction.md").write_text("Do the thing.")
        jobs_dir = tmp_path / "jobs"

        monkeypatch.setattr(nat_runner, "TASKS_DIR", tmp_path / "tasks")
        monkeypatch.setattr(nat_runner, "build_task_image", lambda _task_dir, _tag: None)
        monkeypatch.setattr(nat_runner, "_capture_image_digest", lambda _image: "sha256:test")
        monkeypatch.setattr(
            nat_runner,
            "_capture_repo_provenance",
            lambda: {
                "commit_short": "abc123",
                "branch": "test",
                "commit_dirty": False,
            },
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "nat_runner.py",
                "task-a",
                "--agent-backend",
                "codex",
                "--build-only",
                "--jobs-dir",
                str(jobs_dir),
            ],
        )

        result = nat_runner.main()

        captured = capsys.readouterr()
        assert result == 0
        assert "BUILD SUMMARY: 1/1 task images ready" in captured.out
        assert "tasks passed" not in captured.out
