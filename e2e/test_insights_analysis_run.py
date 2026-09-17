# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for Insights analysis runs backed by ``agents.execute`` jobs.

This is the platform-side replacement for running the Analyst from an
operator's shell: everything here goes through the ``analysis-runs`` API, so
passing it means the path works on a remotely-deployed platform, where nobody
can run ``nemo agents analyst run`` locally.

The Analyst's model is mocked, deliberately. What is under test is the wiring —
run recorded, job submitted under the run's name, Fabric runs the inline
Analyst, the ``insights.analysis`` extension persists the change-set and saves
the report — not whether a real model reaches a good conclusion. The Analyst is
a Nooa CodeAct agent, so one mocked ``execute_python`` tool call carrying a
``return_result(...)`` cell drives a complete, deterministic run.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import pytest
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.inference_middleware import BackendFormat
from nemo_platform_plugin.jobs.client import JobsClient
from nemo_platform_plugin.models.client import ModelsClient
from nemo_platform_plugin.models.types import CreateModelEntityRequest
from nmp.testing import MockProviderResponse, add_mock_provider, wait_for_model_entity

from e2e.agents_deploy_helpers import unique_name

# Subprocess jobs, so the run needs no Docker: the Analyst executes in-process
# inside a Fabric local environment, and the mock provider stands in for the
# model, so nothing here needs a container or a real inference provider.
pytestmark = [
    pytest.mark.timeout(900),
    pytest.mark.e2e_config(
        "e2e/configs/local-subprocess.yaml",
        harness={"backend": "subprocess"},
    ),
]

REPORT_RESULT_NAME = "analysis-report"
# Written by the execute-agent job; the only place an adapter failure is recorded.
FABRIC_RUN_RESULT_NAME = "fabric_run_result"
FABRIC_ERROR_RESULT_NAME = "fabric_error"
JOB_TIMEOUT_SECONDS = 600.0

ANALYST_SUMMARY = "Filed one insight from the deterministic e2e change-set."
INSIGHT_TITLE = "Knowledge search returns no documents and the agent answers anyway"
INSIGHT_DESCRIPTION = (
    "The retrieval tool returns an empty document set, and the agent produces a "
    "confident answer instead of saying it could not find supporting context."
)
TRACE_REF = "trace-insights-analysis-run-e2e"
# Sent inline on the run so the whole ethos chain — request, harness settings,
# adapter, prompt — is exercised by a real job. We can't assert this makes it
# in to the model prompt here, but we at least ensure it isn't rejected anywhere
# along the way.
ETHOS = "# Ethos\n\nAnswer only from retrieved context; say so when there is none."


def _return_result_cell() -> str:
    """The Python cell the mocked model 'writes' to end the CodeAct run.

    ``return_result`` is the Analyst's single terminal call: Nooa validates the
    value against ``AnalystResult`` and ends the run, so one cell is a whole
    analysis.
    """
    return (
        "return_result(result={"
        f"'summary': {ANALYST_SUMMARY!r}, "
        "'new_insights': [{"
        f"'title': {INSIGHT_TITLE!r}, "
        f"'description': {INSIGHT_DESCRIPTION!r}, "
        "'status': 'open', "
        f"'trace_refs': [{TRACE_REF!r}]"
        "}], "
        "'updated_insights': []})"
    )


def _execute_python_response(model: str, code: str) -> dict[str, Any]:
    """A chat completion that calls the CodeAct ``execute_python`` tool."""
    return {
        "id": "chatcmpl-insights-analysis-run-e2e",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_analyst_execute",
                            "type": "function",
                            "function": {
                                "name": "execute_python",
                                "arguments": json.dumps({"code": code}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
                "index": 0,
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _plain_response(model: str, content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-insights-analysis-run-e2e-fast",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _mock_analyst_models(sdk: NeMoPlatform, workspace: str) -> tuple[str, str]:
    """Register the Analyst's default/fast Model Entity pair against a mock provider.

    The pair must be workspace-qualified Model Entity refs — the Analyst
    resolves them through the platform, not as raw provider model ids.
    """
    default_model = unique_name("analyst-default")
    fast_model = unique_name("analyst-fast")

    provider = add_mock_provider(
        sdk,
        workspace=workspace,
        name=unique_name("analyst-provider"),
        mock_response_body_by_model={
            # The default model drives the CodeAct loop and ends it on the
            # first turn.
            f"{workspace}/{default_model}": [
                MockProviderResponse(response_body=_execute_python_response(default_model, _return_result_cell()))
            ],
            # The fast model is only used for context summarization, which a
            # single-turn run never reaches. It still has to resolve.
            f"{workspace}/{fast_model}": [
                MockProviderResponse(response_body=_plain_response(fast_model, "unused summary"))
            ],
        },
        served_models={default_model: default_model, fast_model: fast_model},
    )

    # Create the Model Entities rather than waiting on provider autodiscovery.
    # The models controller reconciles a provider's served_models into entities
    # asynchronously, so relying on it races the run we are about to submit --
    # and the Analyst resolves Model Entities, not served-model ids. This is the
    # same create-then-wait the evaluator e2e does for the same reason.
    models = client_from_platform(sdk, ModelsClient)
    for name in (default_model, fast_model):
        models.create_model(
            workspace=workspace,
            body=CreateModelEntityRequest(
                name=name,
                backend_format=BackendFormat.OPENAI_CHAT,
                model_providers=[f"{workspace}/{provider.name}"],
            ),
            exist_ok=True,
        ).data()
        wait_for_model_entity(sdk, workspace, name)

    return f"{workspace}/{default_model}", f"{workspace}/{fast_model}"


def _job_diagnostics(sdk: NeMoPlatform, workspace: str, job_name: str, prefix: str) -> str:
    """Explain a failed run with the backing job's own error details and logs."""
    parts = [prefix]
    # Fabric reports an adapter failure as a failed *result*, not an exception,
    # so the task exits non-zero with nothing in its stdout and the Analyst's
    # own error lives only in these results.
    for result_name in (FABRIC_ERROR_RESULT_NAME, FABRIC_RUN_RESULT_NAME):
        try:
            parts.append(f"{result_name}: {_download_job_result(sdk, workspace, job_name, result_name)}")
        except Exception as error:
            parts.append(f"Could not fetch {result_name}: {error}")
    try:
        job = sdk.agents.jobs.execute.get(job_name, workspace=workspace)
        for field in ("error_details", "status_details"):
            if detail := job.get(field):
                parts.append(f"{field}: {json.dumps(detail, indent=2, default=str)}")
    except Exception as error:
        parts.append(f"Could not fetch the job: {error}")
    try:
        logs = client_from_platform(sdk, JobsClient).list_job_logs(workspace=workspace, name=job_name)
        for entry in logs.items():
            parts.append(f"  - {entry.message}")
    except Exception as error:
        parts.append(f"Could not fetch job logs: {error}")
    return "\n".join(parts)


def _list_job_results(sdk: NeMoPlatform, workspace: str, job_name: str) -> dict[str, Any]:
    url = f"{str(sdk.base_url).rstrip('/')}/apis/agents/v2/workspaces/{workspace}/jobs/execute/{job_name}/results"
    response = sdk._client.get(url)
    assert response.status_code == 200, f"Failed to list results for {job_name}: {response.text}"
    return response.json()


def _download_job_result(sdk: NeMoPlatform, workspace: str, job_name: str, result_name: str) -> str:
    url = (
        f"{str(sdk.base_url).rstrip('/')}/apis/agents/v2/workspaces/{workspace}"
        f"/jobs/execute/{job_name}/results/{result_name}/download"
    )
    response = sdk._client.get(url)
    assert response.status_code == 200, f"Failed to download {result_name!r} for {job_name}: {response.text}"
    return response.text


def _wait_for_spans(sdk: NeMoPlatform, *, workspace: str, agent_name: str, timeout: float = 120.0) -> list[Any]:
    """Poll Intake: Relay posts as the run ends and ingest is asynchronous."""
    deadline = time.monotonic() + timeout
    while True:
        page = sdk.intake.spans.list(workspace=workspace, filter={"agent_name": agent_name}, page_size=50)
        spans = list(page.data or [])
        if spans or time.monotonic() >= deadline:
            return spans
        time.sleep(2.0)


def _created_insight_id(report: str) -> str:
    """Pull the stored insight id out of the report's change log.

    The Insight is read back by id rather than through ``list_insights``,
    which additionally enriches each row from Intake and so needs ClickHouse
    running. Reading by id keeps this test to the services the analysis-run
    path itself depends on.
    """
    match = re.search(r"- created: .*\[(?P<insight_id>[^\]]+)\]", report)
    assert match is not None, f"No created-insight line in report:\n{report}"
    return match.group("insight_id")


def test_analysis_run_persists_insights_and_saves_its_report(sdk: NeMoPlatform, workspace: str) -> None:
    """One analysis run, end to end, through the supported API surface."""
    target_agent = unique_name("analyzed-agent")
    default_model, fast_model = _mock_analyst_models(sdk, workspace)

    created = sdk.insights.analysis_runs.create(
        workspace=workspace,
        agent=target_agent,
        default_model=default_model,
        fast_model=fast_model,
        ethos=ETHOS,
        timeout_seconds=JOB_TIMEOUT_SECONDS,
    )
    run_name = created.run.name

    # The run is recorded before the job is submitted, and the job takes the
    # run's name — that shared name is the only link between them.
    assert created.run.agent == target_agent
    assert created.job is not None
    assert created.job["name"] == run_name

    final = sdk.insights.analysis_runs.wait(
        workspace=workspace,
        name=run_name,
        timeout=JOB_TIMEOUT_SECONDS,
        poll_interval=2.0,
    )
    assert final.job_status == "completed", _job_diagnostics(
        sdk, workspace, run_name, f"Analysis run {run_name} finished with job status {final.job_status!r}"
    )

    # The run survives as a queryable record, not just as a job.
    listed = sdk.insights.analysis_runs.list_runs(workspace=workspace, agent=target_agent)
    assert run_name in {run.name for run in listed.data}
    assert final.run.default_model == default_model
    assert final.run.fast_model == fast_model

    # The report is the durable record of what the run did — the same result
    # name AnalyzeJob saves, so the two paths stay comparable.
    result_names = {str(result["name"]) for result in _list_job_results(sdk, workspace, run_name)["data"]}
    assert REPORT_RESULT_NAME in result_names, f"Saved results: {sorted(result_names)}"
    report = _download_job_result(sdk, workspace, run_name, REPORT_RESULT_NAME)
    assert ANALYST_SUMMARY in report
    assert INSIGHT_TITLE in report

    # The extension persisted the change-set as a real Insight. The report's
    # change log is what says which one: it is written from what the Insights
    # API returned, so its id only exists if the write landed.
    # Relay carries the Analyst's own trajectory to Intake. Nothing in this
    # test configures telemetry: the agents plugin wires the export, Fabric
    # resolves it, and the adapter activates it.
    spans = _wait_for_spans(sdk, workspace=workspace, agent_name="insights-analyst")
    assert spans, "the Analyst ran but its trajectory never reached Intake"

    insight_id = _created_insight_id(report)
    filed = sdk.insights.insights.get(workspace=workspace, insight_id=insight_id)
    assert filed.title == INSIGHT_TITLE
    assert filed.description == INSIGHT_DESCRIPTION
    assert filed.agent == target_agent
    assert filed.trace_refs == [TRACE_REF]
