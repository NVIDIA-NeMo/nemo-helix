# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small protocol assertion shared by inference, agent invocation and unit tests."""


def require_completion(response: object) -> None:
    if not isinstance(response, dict) or response.get("error"):
        raise ValueError("Response is not a successful chat completion")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Response has no completion choices")
    has_content = False
    for choice in choices:
        if not isinstance(choice, dict) or choice.get("error") or choice.get("finish_reason") == "error":
            raise ValueError("Completion contains a protocol error")
        message = choice.get("message", {})
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str) and content.strip():
            has_content = True
    if not has_content:
        raise ValueError("Completion has no non-empty assistant response")
