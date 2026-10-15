# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The POC must reach a real Harbor environment while keeping its API key out of config."""

import json
import os
from unittest.mock import AsyncMock, MagicMock

from nemo_evals.jobs import harbor_opensandbox_poc as poc
from nemo_evals.jobs.agent_compiler import compile_agent_eval_job
from nemo_evals.jobs.agent_evaluate import AsyncAgentEvalJob
from nemo_evals.jobs.agent_spec import AgentEvalSpec, HarborRunnerTarget
from nemo_evals.tasks import harbor_task_dispatch
from nemo_helix_plugin.job_context import JobContext
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborRuntimeConfig


def test_poc_changes_harbor_compilation_only_when_enabled(monkeypatch):
    spec = AgentEvalSpec.model_construct(tasks=[], target=HarborRunnerTarget())
    monkeypatch.setattr("nemo_evals.jobs.agent_compiler.get_qualified_image", lambda _: "registry/nhx-tasks:nightly")
    monkeypatch.delenv(poc.POC_ENABLED, raising=False)
    normal = compile_agent_eval_job(spec)
    assert normal.steps[-1].executor.container.image == "registry/nhx-tasks:nightly"
    monkeypatch.setenv(poc.POC_ENABLED, "1")
    monkeypatch.setenv(poc.POC_WORKER_IMAGE, "registry/nhx-tasks:branch")
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "server.default.svc:80")
    compiled = compile_agent_eval_job(spec)
    assert compiled.steps[-1].executor.container.image == "registry/nhx-tasks:branch"
    assert compiled.steps[-1].executor.container.command == poc.POC_COMMAND
    assert {item.name: item.value for item in compiled.steps[-1].environment}[
        "OPENSANDBOX_DOMAIN"
    ] == "server.default.svc:80"


def test_runtime_selects_nemo_environment_without_serializing_key(monkeypatch, tmp_path):
    monkeypatch.setenv(poc.POC_ENABLED, "1")
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "server.default.svc:80")
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "private-worker-value")
    ctx = MagicMock(spec=JobContext)
    ctx.job_id = "poc-job"
    original = HarborRuntimeConfig(jobs_dir=tmp_path)
    configured = poc.configure_runtime(original, ctx)
    assert configured.environment.import_path == poc.POC_ENVIRONMENT
    assert configured.environment.kwargs["metadata"]["nemo-job"] == "poc-job"
    assert configured.environment.kwargs["use_server_proxy"] is True
    assert configured.n_concurrent_trials == 1
    assert "private-worker-value" not in json.dumps(configured.model_dump(mode="json"))
    assert original.environment is None


async def test_poc_submission_keeps_kubernetes_cpu_executor(monkeypatch):
    """Exercise the submit compiler's Harbor backend gate, beyond the lower-level step builder."""
    monkeypatch.setenv(poc.POC_ENABLED, "1")
    monkeypatch.setenv(poc.POC_WORKER_IMAGE, "registry/nhx-tasks:branch")
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "server.default.svc:80")
    subprocess_resolver = AsyncMock(side_effect=AssertionError("POC must not require host Docker"))
    monkeypatch.setattr(AsyncAgentEvalJob, "_resolve_harbor_subprocess_executor", subprocess_resolver)
    spec = AgentEvalSpec.model_construct(tasks=[], target=HarborRunnerTarget())
    compiled = await AsyncAgentEvalJob._compile(spec, async_sdk=None, profile="default")
    executor = compiled.steps[-1].executor
    assert executor.provider == "cpu"
    assert executor.container.image == "registry/nhx-tasks:branch"
    assert executor.container.command == poc.POC_COMMAND
    subprocess_resolver.assert_not_awaited()


def test_dispatcher_aliases_jobs_injected_secret(monkeypatch):
    monkeypatch.setenv(poc.POC_ENABLED, "1")
    monkeypatch.delenv("OPENSANDBOX_API_KEY", raising=False)
    monkeypatch.setenv("OPEN_SANDBOX_API_KEY", "injected-at-runtime")
    monkeypatch.setattr(harbor_task_dispatch, "run_async_task_main", lambda *args, **kwargs: 0)
    assert harbor_task_dispatch.main() == 0
    assert os.environ["OPENSANDBOX_API_KEY"] == "injected-at-runtime"
