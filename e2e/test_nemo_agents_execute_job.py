# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for Fabric-backed agent execution jobs."""

from __future__ import annotations

import io
import json
import tarfile
from typing import Any

import pytest
from nemo_agents_plugin.entities import NEMO_AGENTS_SPEC_CONFIG_FORMAT
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.files.types import CreateFilesetRequest
from nemo_platform_plugin.jobs.client import JobsClient
from nmp.testing import MockProviderResponse, add_mock_provider
from nmp.testing.e2e import wait_for_platform_job

from e2e.agents_deploy_helpers import (
    TEST_AGENT_RESPONSE,
    delete_agent_if_exists,
    mock_backed_fabric_agent_config,
    unique_name,
    wait_for_agent_spans,
)

pytestmark = [pytest.mark.timeout(600)]

# A well-formed reference that can never resolve: ``.invalid`` is reserved by
# RFC 2606 and guaranteed not to exist, so the node fails the pull immediately
# on NXDOMAIN rather than burning the test timeout on registry retries.
_UNRESOLVABLE_IMAGE = "registry.invalid/nmp-e2e/no-such-image:missing"

# Evidence that the failure was about the image. The first three are the
# waiting-state reasons the Kubernetes backend maps to a job error
# (``kubernetes/common.py``); the last two are fragments of the ref itself,
# which the Docker backend's pull error embeds. Matching any one of them keeps
# the assertion specific to the image without pinning either backend's wording.
_IMAGE_FAILURE_MARKERS = (
    "imagepullbackoff",
    "errimagepull",
    "invalidimagename",
    "registry.invalid",
    "no-such-image",
)


def _job_diagnostic_message(sdk: NeMoPlatform, job: Any, workspace: str, prefix: str) -> str:
    parts = [prefix]
    if job.status_details:
        parts.append(f"Status details: {job.status_details}")
    if job.error_details:
        parts.append(f"Error details: {job.error_details}")
    try:
        logs = client_from_platform(sdk, JobsClient).list_job_logs(workspace=workspace, name=job.name)
        entries = list(logs.items())
        if entries:
            parts.append(f"Job logs ({len(entries)} entries):")
            for entry in entries:
                parts.append(f"  - {entry.message}")
    except Exception as error:
        parts.append(f"Could not fetch job logs: {error}")
    return "\n".join(parts)


def _list_execute_job_results(sdk: NeMoPlatform, workspace: str, job_name: str) -> dict[str, Any]:
    return dict(sdk.agents.jobs.execute.list_results(job_name, workspace=workspace))


def _download_execute_job_result(sdk: NeMoPlatform, workspace: str, job_name: str, result_name: str) -> bytes:
    return sdk.agents.jobs.execute.download_result(result_name, job=job_name, workspace=workspace)


def _result_names(results: dict[str, Any]) -> set[str]:
    return {str(result["name"]) for result in results.get("data", [])}


def _tar_member_names(content: bytes) -> set[str]:
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
        return {member.name for member in tar.getmembers()}


def _tar_text_by_suffix(content: bytes, suffix: str) -> str:
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(suffix) or not member.isfile():
                continue
            extracted = tar.extractfile(member)
            assert extracted is not None
            return extracted.read().decode("utf-8")
    raise AssertionError(f"{suffix!r} not found in tarball")


def _tar_contains(member_names: set[str], suffix: str) -> bool:
    return any(name.endswith(suffix) for name in member_names)


def _write_file_tool_call_response(*, model: str, file_path: str, content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-agents-invoke-e2e-tool",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_write_report",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": json.dumps({"file_path": file_path, "content": content}),
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


def _final_chat_completion_response(content: str, model: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-agents-invoke-e2e-final",
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


def _chat_completion_error(message: str) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": "intentional_e2e_error",
        }
    }


def _mock_backed_workspace_agent_config(agent_name: str, model_name: str) -> dict[str, Any]:
    config = mock_backed_fabric_agent_config(agent_name, model_name)
    config["instructions"] = {
        "system": {
            "content": (
                "Use the write_file tool when asked to create files. "
                "Write paths exactly as requested and then summarize what changed."
            )
        },
    }
    return config


def test_fabric_agent_invocation_job_runs_and_saves_results(sdk: NeMoPlatform, workspace: str) -> None:
    agent_name = unique_name("execute-agent")
    job_name = unique_name("execute-job")
    model_name = unique_name("invoke-model")
    fileset_name = unique_name("invoke-inputs")
    generated_report = "Fabric wrote this deterministic e2e report.\n"

    add_mock_provider(
        sdk,
        workspace=workspace,
        name=unique_name("invoke-provider"),
        mock_response_body_by_model={
            f"{workspace}/{model_name}": [
                MockProviderResponse(
                    response_body=_write_file_tool_call_response(
                        model=model_name,
                        file_path="/generated-report.md",
                        content=generated_report,
                    )
                ),
                MockProviderResponse(response_body=_final_chat_completion_response(TEST_AGENT_RESPONSE, model_name)),
            ],
        },
        served_models={model_name: model_name},
    )

    files = client_from_platform(sdk, FilesClient)
    files.create_fileset(body=CreateFilesetRequest(name=fileset_name), workspace=workspace)
    files.upload_file(
        name=fileset_name,
        workspace=workspace,
        path="project/context.txt",
        content=b"This file proves the input workdir was staged.\n",
    )

    sdk.agents.create(
        workspace=workspace,
        name=agent_name,
        config=_mock_backed_workspace_agent_config(agent_name, f"{workspace}/{model_name}"),
        config_format=NEMO_AGENTS_SPEC_CONFIG_FORMAT,
    )

    try:
        sdk.agents.jobs.execute.create(
            name=job_name,
            workspace=workspace,
            spec={
                "agent": agent_name,
                "input": "Answer with the deterministic mock provider response.",
                "workdir": {"base_workdir": f"{fileset_name}#project/"},
            },
        )

        completed_job = wait_for_platform_job(sdk, job_name, workspace, timeout=300)
        assert completed_job.status == "completed", _job_diagnostic_message(
            sdk,
            completed_job,
            workspace,
            f"Execute job failed with status: {completed_job.status}",
        )

        results = _list_execute_job_results(sdk, workspace, job_name)
        assert {
            "input_workdir",
            "output_workdir",
            "output_artifacts",
            "fabric_run_result",
        }.issubset(_result_names(results))

        # The input snapshot contains the original file, but not the agent-generated one
        input_workdir = _download_execute_job_result(sdk, workspace, job_name, "input_workdir")
        input_members = _tar_member_names(input_workdir)
        assert _tar_contains(input_members, "context.txt")
        assert not _tar_contains(input_members, "generated-report.md")

        # The output snapshot contains the original file AND the agent-generated one
        output_workdir = _download_execute_job_result(sdk, workspace, job_name, "output_workdir")
        output_members = _tar_member_names(output_workdir)
        assert _tar_contains(output_members, "context.txt")
        assert _tar_contains(output_members, "generated-report.md")
        assert _tar_text_by_suffix(output_workdir, "generated-report.md") == generated_report

        # The output artifacts contains Fabric-produced files
        artifacts = _download_execute_job_result(sdk, workspace, job_name, "output_artifacts")
        artifacts_members = _tar_member_names(artifacts)
        assert _tar_contains(artifacts_members, "adapter-invocation.json")
        assert _tar_contains(artifacts_members, "stdout.txt")

        # The run result captures platform-normalized Fabric RunResult details
        run_result = json.loads(_download_execute_job_result(sdk, workspace, job_name, "fabric_run_result"))
        assert run_result["status"] == "succeeded"
        assert run_result["runtime_id"].startswith("runtime-")
        assert run_result["invocation_id"]
        assert run_result["request_id"] == job_name
        assert run_result["response"] == TEST_AGENT_RESPONSE
        assert run_result["output"]["event_count"] > 0
        assert run_result["output"]["message_count"] >= 4
        run_result_json = json.dumps(run_result)
        assert "write_file" in run_result_json
        assert "generated-report.md" in run_result_json
        assert TEST_AGENT_RESPONSE in run_result_json

        # Nobody configured an export: the job wires the agent's trajectory to
        # this workspace's Intake, using the platform URL reachable from the
        # task pod and the identity the platform gave the job.
        spans = wait_for_agent_spans(sdk, workspace=workspace, agent_name=agent_name)
        assert spans, "the agent ran but no trajectory reached Intake"
    finally:
        delete_agent_if_exists(sdk, workspace=workspace, name=agent_name)


def test_fabric_agent_invocation_job_saves_failed_run_result_and_partial_outputs(
    sdk: NeMoPlatform,
    workspace: str,
) -> None:
    agent_name = unique_name("execute-agent")
    job_name = unique_name("execute-job")
    model_name = unique_name("invoke-model")
    fileset_name = unique_name("invoke-inputs")
    partial_report = "Fabric wrote this file before the model failed.\n"

    add_mock_provider(
        sdk,
        workspace=workspace,
        name=unique_name("invoke-provider"),
        mock_response_body_by_model={
            f"{workspace}/{model_name}": [
                MockProviderResponse(
                    response_body=_write_file_tool_call_response(
                        model=model_name,
                        file_path="/partial-before-error.txt",
                        content=partial_report,
                    )
                ),
                # Force Fabric to fail
                MockProviderResponse(
                    response_code=500,
                    response_body=_chat_completion_error("intentional e2e model failure"),
                ),
            ],
        },
        served_models={model_name: model_name},
    )

    files = client_from_platform(sdk, FilesClient)
    files.create_fileset(body=CreateFilesetRequest(name=fileset_name), workspace=workspace)
    files.upload_file(
        name=fileset_name,
        workspace=workspace,
        path="project/context.txt",
        content=b"This file proves the failed invocation still saves the input snapshot.\n",
    )

    sdk.agents.create(
        workspace=workspace,
        name=agent_name,
        config=_mock_backed_workspace_agent_config(agent_name, f"{workspace}/{model_name}"),
        config_format=NEMO_AGENTS_SPEC_CONFIG_FORMAT,
    )

    try:
        sdk.agents.jobs.execute.create(
            name=job_name,
            workspace=workspace,
            spec={
                "agent": agent_name,
                "input": "Write the partial file, then continue.",
                "workdir": {"base_workdir": f"{fileset_name}#project/"},
            },
        )

        completed_job = wait_for_platform_job(sdk, job_name, workspace, timeout=300)
        assert completed_job.status == "error", _job_diagnostic_message(
            sdk,
            completed_job,
            workspace,
            f"Execute job unexpectedly finished with status: {completed_job.status}",
        )

        results = _list_execute_job_results(sdk, workspace, job_name)
        assert {
            "input_workdir",
            "output_workdir",
            "output_artifacts",
            "fabric_run_result",
        }.issubset(_result_names(results))
        assert "fabric_error" not in _result_names(results)

        # The input snapshot is as expected
        input_workdir = _download_execute_job_result(sdk, workspace, job_name, "input_workdir")
        assert _tar_contains(_tar_member_names(input_workdir), "context.txt")
        assert not _tar_contains(_tar_member_names(input_workdir), "partial-before-error.txt")

        # We do still get an output snapshot even when Fabric fails, and it does contain
        # the file that Fabric created before failing
        output_workdir = _download_execute_job_result(sdk, workspace, job_name, "output_workdir")
        output_members = _tar_member_names(output_workdir)
        assert _tar_contains(output_members, "context.txt")
        assert _tar_contains(output_members, "partial-before-error.txt")
        assert _tar_text_by_suffix(output_workdir, "partial-before-error.txt") == partial_report

        # The run result includes status=failed and error details
        run_result = json.loads(_download_execute_job_result(sdk, workspace, job_name, "fabric_run_result"))
        assert run_result["status"] == "failed"
        assert run_result["request_id"] == job_name
        assert run_result["runtime_id"].startswith("runtime-")
        assert run_result["invocation_id"]
        assert run_result["error"]["code"] == "deepagents_invocation_failed"
        error_message = run_result["error"]["message"]
        assert "InternalServerError" in error_message
        assert "Error code: 500" in error_message
    finally:
        delete_agent_if_exists(sdk, workspace=workspace, name=agent_name)


# ---------------------------------------------------------------------------
# Per-job execution image
# ---------------------------------------------------------------------------


@pytest.fixture
def container_backed_execute(sdk: NeMoPlatform) -> None:
    """Skip unless ``agents.execute`` actually runs in a container.

    The jobs API rewrites a ``cpu``/``default`` container step into a host
    subprocess whenever a subprocess executor is registered at that profile
    (``translate_cpu_container_steps_to_subprocess``), and that rewrite drops
    ``container.image`` on the floor. Every assertion about which image a job
    ran on is vacuous in that mode, so gate on the precise condition rather
    than on the harness shape: ``container_only`` only proves ``NMP_BASE_URL``
    is set, which an already-running local (subprocess) platform also satisfies.
    """
    profiles = client_from_platform(sdk, JobsClient).list_execution_profiles()
    if any(profile.provider == "subprocess" and profile.profile == "default" for profile in profiles):
        pytest.skip("cpu/default is diverted to the subprocess backend, which discards container.image")


def _register_mock_backed_agent(sdk: NeMoPlatform, workspace: str, *, agent_name: str, model_name: str) -> None:
    """Register a deterministic agent whose single model is served by the mock provider."""
    add_mock_provider(
        sdk,
        workspace=workspace,
        name=unique_name("image-provider"),
        mock_response_body_by_model={
            f"{workspace}/{model_name}": [
                MockProviderResponse(response_body=_final_chat_completion_response(TEST_AGENT_RESPONSE, model_name)),
            ],
        },
        served_models={model_name: model_name},
    )
    sdk.agents.create(
        workspace=workspace,
        name=agent_name,
        config=mock_backed_fabric_agent_config(agent_name, f"{workspace}/{model_name}"),
        config_format=NEMO_AGENTS_SPEC_CONFIG_FORMAT,
    )


@pytest.mark.container_only
def test_execute_job_fails_when_the_requested_image_cannot_be_pulled(
    sdk: NeMoPlatform,
    workspace: str,
    container_backed_execute: None,
) -> None:
    """``spec.image`` reaches the container runtime.

    Failure is the only outcome that can prove this today. A job that silently
    ignored ``image`` would run on the inherited task image and succeed, so a
    passing run says nothing -- and every image this test could legitimately
    ask for is the one it would have inherited anyway. Because this ref cannot
    resolve anywhere, an error whose diagnostics name the image or a pull
    failure can only happen if the string travelled from the request through
    ``compile`` into the pod/container spec.

    The discriminating *positive* test needs an image that holds something the
    default image does not -- a ``nemo agents package`` build carrying its own
    Fabric adapter.
    """
    del container_backed_execute
    agent_name = unique_name("image-agent")
    job_name = unique_name("image-job")
    model_name = unique_name("image-model")

    _register_mock_backed_agent(sdk, workspace, agent_name=agent_name, model_name=model_name)

    try:
        sdk.agents.jobs.execute.create(
            name=job_name,
            workspace=workspace,
            spec={
                "agent": agent_name,
                "input": "This agent never runs; the image cannot be pulled.",
                "image": _UNRESOLVABLE_IMAGE,
            },
        )

        finished_job = wait_for_platform_job(sdk, job_name, workspace, timeout=300)
        diagnostics = _job_diagnostic_message(
            sdk,
            finished_job,
            workspace,
            f"Execute job with an unresolvable image finished as: {finished_job.status}",
        )
        assert finished_job.status == "error", diagnostics
        assert any(marker in diagnostics.lower() for marker in _IMAGE_FAILURE_MARKERS), (
            f"Job failed, but not demonstrably because of the requested image.\n{diagnostics}"
        )
    finally:
        delete_agent_if_exists(sdk, workspace=workspace, name=agent_name)
