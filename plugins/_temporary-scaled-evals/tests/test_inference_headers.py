# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import yaml

try:
    from scaled_evals.dispatch.inference_headers import (
        INFERENCE_HEADERS_ENV,
        inference_header_runner_env,
        with_default_anthropic_custom_headers,
        with_default_inference_priority,
    )
    from scaled_evals.dispatch.sandbox_k8s import render_harbor_config
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


def test_default_inference_priority_is_case_insensitive_and_preserves_other_headers() -> None:
    assert with_default_inference_priority({"x-inference-priority": "interactive", "X-Existing": "retained"}) == {
        "X-Existing": "retained",
        "X-Inference-Priority": "batch",
    }
    assert (
        with_default_anthropic_custom_headers("X-Existing: retained\nx-INFERENCE-priority: interactive")
        == "X-Existing: retained\nX-Inference-Priority: batch"
    )

    env = inference_header_runner_env({"X-Existing": "retained"})
    assert json.loads(env["CODEX_GATEWAY_HTTP_HEADERS_JSON"]) == {
        "X-Existing": "retained",
        "X-Inference-Priority": "batch",
    }
    assert env["CODEX_GATEWAY_HTTP_HEADERS_TOML"] == ('{"X-Existing"="retained", "X-Inference-Priority"="batch"}')


def test_render_harbor_config_forces_priority_for_codex_claude_and_terminus() -> None:
    rendered = render_harbor_config(
        """
environment:
  import_path: sandbox_k8s.harbor:K8sSandboxEnvironment
  kwargs:
    namespace: evals
    image: registry.example/task:dev
agents:
  - name: codex
    env:
      CODEX_GATEWAY_HTTP_HEADERS_JSON: '{"x-inference-priority":"interactive","X-Codex":"yes"}'
  - import_path: harbor.agents.installed.claude_code:ClaudeCode
    env:
      ANTHROPIC_CUSTOM_HEADERS: |-
        X-Claude: yes
        X-Inference-Priority: interactive
  - name: terminus-2
    kwargs:
      llm_call_kwargs:
        extra_body:
          metadata: retained
        extra_headers:
          x-inference-priority: interactive
          X-Terminus: "yes"
  - name: opencode
    model_name: openai/test-model
    kwargs:
      opencode_config:
        provider:
          openai:
            options:
              headers:
                x-inference-priority: interactive
                X-OpenCode: "yes"
  - name: openclaw
    model_name: openai/test-model
    kwargs:
      openclaw_config:
        models:
          providers:
            openai:
              headers:
                X-Inference-Priority: low
                X-OpenClaw: "yes"
  - name: openhands
    env:
      LLM_EXTRA_HEADERS: '{"x-inference-priority":"interactive","X-OpenHands":"yes"}'
  - name: hermes
  - name: pi
  - name: langgraph
  - import_path: harbor.agents.installed.matrix_command_agent:MatrixCommandAgent
""",
        {},
        job_name="ev_priority",
    )

    agents = yaml.safe_load(rendered)["agents"]
    codex_headers = json.loads(agents[0]["env"]["CODEX_GATEWAY_HTTP_HEADERS_JSON"])
    assert codex_headers == {"X-Codex": "yes", "X-Inference-Priority": "batch"}
    assert agents[0]["env"]["CODEX_GATEWAY_HTTP_HEADERS_TOML"] == ('{"X-Codex"="yes", "X-Inference-Priority"="batch"}')
    assert agents[1]["env"]["ANTHROPIC_CUSTOM_HEADERS"] == ("X-Claude: yes\nX-Inference-Priority: batch")
    assert agents[2]["kwargs"]["llm_call_kwargs"] == {
        "extra_body": {"metadata": "retained"},
        "extra_headers": {"X-Terminus": "yes", "X-Inference-Priority": "batch"},
    }
    assert agents[3]["kwargs"]["opencode_config"]["provider"]["openai"]["options"]["headers"] == {
        "X-OpenCode": "yes",
        "X-Inference-Priority": "batch",
    }
    assert agents[4]["kwargs"]["openclaw_config"]["models"]["providers"]["openai"]["headers"] == {
        "X-OpenClaw": "yes",
        "X-Inference-Priority": "batch",
    }
    assert json.loads(agents[5]["env"]["LLM_EXTRA_HEADERS"]) == {
        "X-OpenHands": "yes",
        "X-Inference-Priority": "batch",
    }
    for agent in agents:
        assert json.loads(agent["env"][INFERENCE_HEADERS_ENV])["X-Inference-Priority"] == "batch"
