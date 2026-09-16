# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AUT (agent-under-test) backend — the canonical ``nat_runner`` execution path.

Instead of a task-local ``nat run`` workflow, it drives a **deployed platform
agent**: inside the task image it creates the agent from ``--aut-agent-config``
(if needed), optionally seeds inference providers, deploys + health-checks it,
then invokes it via ``nat_trace_export.py invoke-aut``. The agent is user-
supplied (``--aut-agent-name`` + ``--aut-agent-config``); see
``aut_agent.workspace.example.yml`` for a task-capable starting point.
"""

from __future__ import annotations

import os
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from nemo_evaluator_sdk.agent_eval.runtimes.environment import AgentEnvironmentProvider, EnvRunSpec
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, RunnerInfo

from .platform_runtime import (
    DEFAULT_LOCAL_NHX_BASE_URL,
    DEFAULT_TIMEOUT_SEC,
    INSTRUCTION_CONTAINER_PATH,
    NAT_TRACE_EXPORT_SCRIPT_CONTAINER_PATH,
    REPO_ROOT,
    AgenticRunLayout,
    PlatformDockerEnvironmentProvider,
    base_container_env,
    docker_socket_mounts,
    resolve_run_layout,
    run_agent_then_verify,
    task_agent_timeout_sec,
)

RUNTIME_NAME = "aut"
AUT_CONFIG_CONTAINER_PATH = "/tmp/aut_agent.yml"


@dataclass(frozen=True)
class AutConfig:
    """Configuration for :class:`NatAutRuntime`."""

    aut_agent_name: str
    aut_agent_config: Path | None = None
    aut_seed_providers: bool = True
    aut_health_wait_seconds: int = int(os.environ.get("NAT_AUT_HEALTH_WAIT_SECONDS", "60"))
    agent_model: str | None = None
    nhx_base_url: str = DEFAULT_LOCAL_NHX_BASE_URL
    nvidia_api_key: str | None = None
    inference_nvidia_api_key: str | None = None
    anthropic_api_key: str | None = None
    timeout_sec: int = DEFAULT_TIMEOUT_SEC
    run_verify: bool = False
    docker_extra_args: list[str] = field(default_factory=list)


def build_aut_agent_cmd(instruction_container: str) -> list[str]:
    """``bash -c`` command that creates/deploys/invokes the agent-under-test."""
    return [
        "bash",
        "-c",
        textwrap.dedent(f"""\
            set -euo pipefail
            EFFECTIVE_AUT_AGENT_CONFIG="${{AUT_AGENT_CONFIG}}"
            if [ -n "${{AUT_AGENT_CONFIG}}" ] && [ -n "${{NVIDIA_API_KEY:-}}" -o -n "${{ANTHROPIC_API_KEY:-}}" ]; then
              /app/.venv/bin/python -c "from pathlib import Path; import os; text = Path(os.environ['AUT_AGENT_CONFIG']).read_text(); text = text.replace('\\${{NVIDIA_API_KEY}}', os.environ.get('NVIDIA_API_KEY', '')); text = text.replace('\\${{ANTHROPIC_API_KEY}}', os.environ.get('ANTHROPIC_API_KEY', '')); Path('/tmp/aut_agent.resolved.yml').write_text(text)"
              EFFECTIVE_AUT_AGENT_CONFIG="/tmp/aut_agent.resolved.yml"
            fi
            if /app/.venv/bin/nemo agents get "${{AUT_AGENT_NAME}}" >/tmp/aut_get_before.log 2>&1; then
              if [ -n "${{EFFECTIVE_AUT_AGENT_CONFIG}}" ]; then
                echo "AUT '${{AUT_AGENT_NAME}}' already exists; recreating from AUT_AGENT_CONFIG."
                /app/.venv/bin/nemo agents undeploy --agent "${{AUT_AGENT_NAME}}" >/tmp/aut_undeploy_before_recreate.log 2>&1 || true
                /app/.venv/bin/nemo agents delete "${{AUT_AGENT_NAME}}" >/tmp/aut_delete_before_recreate.log 2>&1 || true
                /app/.venv/bin/nemo agents create --name "${{AUT_AGENT_NAME}}" --agent-config "${{EFFECTIVE_AUT_AGENT_CONFIG}}" >/tmp/aut_create.log 2>&1
              else
                echo "AUT '${{AUT_AGENT_NAME}}' already exists."
              fi
            else
              if [ -z "${{EFFECTIVE_AUT_AGENT_CONFIG}}" ]; then
                echo "AUT agent '${{AUT_AGENT_NAME}}' not found and no AUT_AGENT_CONFIG was provided." >&2
                echo "Set --aut-agent-config so the runner can create the agent." >&2
                cp /tmp/aut_get_before.log /logs/agent/aut_get_before.log 2>/dev/null || true
                exit 1
              fi
              /app/.venv/bin/nemo agents create --name "${{AUT_AGENT_NAME}}" --agent-config "${{EFFECTIVE_AUT_AGENT_CONFIG}}" >/tmp/aut_create.log 2>&1
            fi
            if [ "${{AUT_SEED_PROVIDERS:-1}}" = "1" ]; then
              /app/.venv/bin/python /app/tests/agentic-use/seed_providers.py \\
                --manifest /app/tests/agentic-use/providers.yaml \\
                --base-url "${{NHX_BASE_URL:-http://localhost:8080}}" \\
                2>&1 | tee /tmp/aut_provider_seed.log
            fi
            collect_aut_diagnostics() {{
              set +e
              /app/.venv/bin/nemo agents deployments list >/tmp/aut_deployments.list.json 2>&1
              cp /tmp/aut_deployments.list.json /logs/agent/aut_deployments.list.json 2>/dev/null || true
              cp /tmp/aut_create.log /logs/agent/aut_create.log 2>/dev/null || true
              cp /tmp/aut_get_before.log /logs/agent/aut_get_before.log 2>/dev/null || true
              cp /tmp/aut_provider_seed.log /logs/agent/aut_provider_seed.log 2>/dev/null || true
              cp /tmp/nhx-api.log /logs/agent/nhx-api.log 2>/dev/null || true
              return 0
            }}
            cleanup() {{
              cleanup_rc=$?
              set +e
              collect_aut_diagnostics
              /app/.venv/bin/nemo agents undeploy --agent "${{AUT_AGENT_NAME}}" >/tmp/aut_undeploy_after.log 2>&1 || true
              cp /tmp/aut_undeploy_after.log /logs/agent/aut_undeploy_after.log 2>/dev/null || true
              cp /tmp/nat_agent.log /logs/agent/nat_agent.log 2>/dev/null || true
              exit "$cleanup_rc"
            }}
            trap cleanup EXIT
            /app/.venv/bin/nemo agents undeploy --agent "${{AUT_AGENT_NAME}}" >/tmp/aut_undeploy.log 2>&1 || true
            /app/.venv/bin/nemo agents deploy --agent "${{AUT_AGENT_NAME}}"
            /app/.venv/bin/nemo agents deployments wait --agent "${{AUT_AGENT_NAME}}"
            dep_endpoint=$(
              /app/.venv/bin/nemo agents deployments list 2>/dev/null | /app/.venv/bin/python -c "import json,sys; data=json.load(sys.stdin).get('data', []); match=next((d.get('endpoint') for d in data if d.get('agent') == '${{AUT_AGENT_NAME}}' and d.get('status') == 'running' and d.get('endpoint')), ''); print(match)"
            )
            if [ -z "$dep_endpoint" ]; then
              echo "No running AUT deployment endpoint found for agent '${{AUT_AGENT_NAME}}'." >&2
              exit 1
            fi
            health_wait="${{AUT_HEALTH_WAIT_SECONDS:-60}}"
            aut_healthy=0
            for i in $(seq 1 "$health_wait"); do
              if curl -sf "$dep_endpoint/health" >/dev/null 2>&1; then
                aut_healthy=1
                break
              fi
              sleep 1
            done
            if [ "$aut_healthy" -ne 1 ]; then
              echo "AUT deployment did not become healthy within $health_wait seconds: $dep_endpoint" >&2
              exit 1
            fi
            set +e
            /app/.venv/bin/python {NAT_TRACE_EXPORT_SCRIPT_CONTAINER_PATH} invoke-aut \\
              --endpoint "$dep_endpoint" \\
              --instruction {instruction_container} \\
              --output-dir /logs/agent \\
              --timeout "${{AUT_INVOKE_HTTP_TIMEOUT:-600}}" \\
              2>&1 | tee /tmp/nat_agent.log
            rc=${{PIPESTATUS[0]}}
            set -e
            if [ $rc -ne 0 ]; then
              collect_aut_diagnostics
            fi
            exit $rc
        """),
    ]


def prepare_aut_config_for_runtime(
    config_path: Path,
    output_dir: Path,
    *,
    nat_model: str | None = None,
    nhx_base_url: str = DEFAULT_LOCAL_NHX_BASE_URL,
    workspace: str = "default",
) -> Path:
    """Prepare an AUT agent config for IGW-routed container runtime."""
    from nemo_agents_plugin.utils import inject_gateway_url

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if nat_model:
        for llm_cfg in config.get("llms", {}).values():
            if isinstance(llm_cfg, dict) and llm_cfg.get("_type") in ("openai", "nim"):
                llm_cfg["model_name"] = nat_model
                break
    config = inject_gateway_url(config, workspace, base_url=nhx_base_url)
    rewritten = output_dir / "aut.runtime.yml"
    rewritten.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False), encoding="utf-8")
    return rewritten


class NatAutRuntime:
    """Run agentic-use tasks via a deployed platform agent-under-test (an ``AgentTaskRunner``)."""

    def __init__(self, config: AutConfig, *, environment: AgentEnvironmentProvider | None = None) -> None:
        if not config.aut_agent_name:
            raise ValueError("NatAutRuntime requires aut_agent_name")
        self.config = config
        self.environment = environment or PlatformDockerEnvironmentProvider()

    def runner_info(self) -> RunnerInfo:
        """Identify this runtime and the settings that shape its results (the AgentTaskRunner contract).

        Enumerates fields rather than dumping ``AutConfig``: it also carries ``nvidia_api_key``,
        ``inference_nvidia_api_key`` and ``anthropic_api_key``, and this is persisted with the run.
        """
        return RunnerInfo(
            name="example_nat_aut",
            kind="runner",
            config={
                "aut_agent_name": self.config.aut_agent_name,
                "aut_agent_config": str(self.config.aut_agent_config) if self.config.aut_agent_config else None,
                "aut_seed_providers": self.config.aut_seed_providers,
                "aut_health_wait_seconds": self.config.aut_health_wait_seconds,
                "agent_model": self.config.agent_model,
                "nhx_base_url": self.config.nhx_base_url,
                "timeout_sec": self.config.timeout_sec,
            },
        )

    async def run_tasks(
        self,
        tasks: Sequence[AgentEvalTask],
        config: AgentEvalRunConfig | None = None,
    ) -> Sequence[AgentEvalTrial]:
        return [await self._run_task(task, config) for task in tasks]

    async def _run_task(self, task: AgentEvalTask, config: AgentEvalRunConfig | None) -> AgentEvalTrial:
        layout = resolve_run_layout(task, config)
        agent_model = self.config.agent_model or "unknown"
        handle = await self.environment.prepare(task, config)
        return await run_agent_then_verify(
            handle,
            task=task,
            layout=layout,
            spec=self._agent_run_spec(task, layout),
            runtime_name=RUNTIME_NAME,
            agent_model=agent_model,
            run_verify=self.config.run_verify,
            nhx_base_url=self.config.nhx_base_url,
            verify_timeout_sec=self.config.timeout_sec + 120,
            docker_extra_args=list(self.config.docker_extra_args),
        )

    def _agent_run_spec(self, task: AgentEvalTask, layout: AgenticRunLayout) -> EnvRunSpec:
        task_dir = Path(str(task.metadata["task_dir"]))
        task_timeout = task_agent_timeout_sec(task_dir) or 0
        timeout_sec = max(self.config.timeout_sec, task_timeout)

        env = base_container_env(self.config.nhx_base_url, timeout_sec=timeout_sec)
        if self.config.nvidia_api_key:
            env["NVIDIA_API_KEY"] = self.config.nvidia_api_key
        if self.config.anthropic_api_key:
            env["ANTHROPIC_API_KEY"] = self.config.anthropic_api_key
        if self.config.aut_seed_providers and self.config.inference_nvidia_api_key:
            env["INFERENCE_NVIDIA_API_KEY"] = self.config.inference_nvidia_api_key
        if self.config.agent_model:
            env["NAT_MODEL"] = self.config.agent_model
        env["AUT_AGENT_NAME"] = self.config.aut_agent_name
        env["AUT_SEED_PROVIDERS"] = "1" if self.config.aut_seed_providers else "0"
        env["AUT_HEALTH_WAIT_SECONDS"] = str(self.config.aut_health_wait_seconds)

        mounts: list[tuple[str, str]] = [
            (str(layout.instruction_path), INSTRUCTION_CONTAINER_PATH),
            (str(layout.agent_log_dir), "/logs/agent"),
            (str(layout.workspace_dir), "/app/workspace"),
            (str(layout.state_dir), "/data"),
        ]

        if self.config.aut_agent_config is not None:
            aut_config_path = Path(self.config.aut_agent_config)
            if not aut_config_path.is_absolute():
                aut_config_path = (REPO_ROOT / aut_config_path).resolve()
            if not aut_config_path.exists():
                raise FileNotFoundError(f"AUT config not found: {aut_config_path}")
            aut_config_host = prepare_aut_config_for_runtime(
                aut_config_path,
                layout.agent_log_dir,
                nat_model=self.config.agent_model,
                nhx_base_url=self.config.nhx_base_url,
            )
            env["AUT_AGENT_CONFIG"] = AUT_CONFIG_CONTAINER_PATH
            mounts.append((str(aut_config_host), AUT_CONFIG_CONTAINER_PATH))
        else:
            env["AUT_AGENT_CONFIG"] = ""

        mounts += docker_socket_mounts()

        return EnvRunSpec(
            command=build_aut_agent_cmd(INSTRUCTION_CONTAINER_PATH),
            env=env,
            mounts=mounts,
            timeout=timeout_sec + 120,
            extra_args=list(self.config.docker_extra_args),
        )


__all__ = ["AutConfig", "NatAutRuntime", "build_aut_agent_cmd", "prepare_aut_config_for_runtime"]
