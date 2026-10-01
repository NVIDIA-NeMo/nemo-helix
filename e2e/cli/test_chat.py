# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for the chat CLI use-case command."""

from __future__ import annotations

import subprocess
import uuid

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nhx.testing import NemoRun, add_mock_provider, assert_exit_0, wait_for_model_entity
from nhx.testing.utils import wait_for_virtual_model

pytestmark = [pytest.mark.timeout(120)]


def test_chat_lifecycle(client: NemoClient, workspace: str, nemo_run: NemoRun) -> None:
    """Chat sends a prompt to a mock model, streams the response, then exits cleanly.

    Covers the full chat lifecycle:
    - Model entity routing (workspace/model in request body)
    - Provider routing via --provider flag
    - Optional parameters: --system-message, --max-tokens
    """
    run_id = uuid.uuid4().hex[:8]
    model_name = f"e2e-chat-{run_id}"
    expected_content = "Hello from the mock model"

    # The mock provider auto-converts the non-streaming response body to SSE when stream=True.
    provider = add_mock_provider(
        client,
        workspace=workspace,
        name=model_name,
        mock_response_body={
            "id": "chatcmpl-e2e",
            "object": "chat.completion",
            "created": 1677652288,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": expected_content},
                    "finish_reason": "stop",
                }
            ],
        },
        served_models={model_name: model_name},
    )

    # --- Model entity routing (default) ---
    # stdin=DEVNULL causes Prompt.ask to raise EOFError after the first response,
    # which the chat loop catches and turns into a graceful sys.exit(0).
    wait_for_model_entity(client, workspace, model_name)
    wait_for_virtual_model(client, workspace, model_name)
    result = nemo_run(
        "chat",
        "--model",
        f"{workspace}/{model_name}",
        "Say hello",
        workspace=workspace,
        stdin=subprocess.DEVNULL,
    )
    assert_exit_0(result, "chat (model entity route) failed")
    output = result.stdout + result.stderr
    assert expected_content in output, (
        f"Expected response content not found in output (model entity route):\n{output[:1000]}"
    )

    # --- Model entity routing with optional parameters ---
    wait_for_model_entity(client, workspace, model_name)
    wait_for_virtual_model(client, workspace, model_name)
    result = nemo_run(
        "chat",
        "--model",
        f"{workspace}/{model_name}",
        "Say hello again",
        "--system-message",
        "You are a helpful assistant.",
        "--max-tokens",
        "50",
        workspace=workspace,
        stdin=subprocess.DEVNULL,
    )
    assert_exit_0(result, "chat (with options) failed")
    output = result.stdout + result.stderr
    assert expected_content in output, f"Expected response content not found in output (with options):\n{output[:1000]}"

    # --- Provider routing via --provider flag ---
    wait_for_model_entity(client, workspace, model_name)
    result = nemo_run(
        "chat",
        "--model",
        model_name,
        "Say hello via provider",
        "--provider",
        provider.name,
        "--workspace",
        workspace,
        workspace=workspace,
        stdin=subprocess.DEVNULL,
    )
    assert_exit_0(result, "chat (provider route) failed")
    output = result.stdout + result.stderr
    assert expected_content in output, (
        f"Expected response content not found in output (provider route):\n{output[:1000]}"
    )
