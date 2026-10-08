# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility re-exports for LLM judge default prompt helpers."""

from nhx_evals_sdk.values.llm_judge_defaults import (
    DEFAULT_JUDGE_PROMPT_TEMPLATE_WITH_TARGET_MODEL,
    DEFAULT_JUDGE_SYSTEM_PROMPT_TEMPLATE,
    DEFAULT_PROMPT_TEMPLATE,
    LLM_JUDGE_SCORES_CONTEXT_KEY,
    default_judge_prompt_template_chat,
    default_judge_prompt_template_completions,
    default_judge_prompt_template_for_model,
    is_chat_inference,
)

__all__ = [
    "DEFAULT_JUDGE_PROMPT_TEMPLATE_WITH_TARGET_MODEL",
    "DEFAULT_JUDGE_SYSTEM_PROMPT_TEMPLATE",
    "DEFAULT_PROMPT_TEMPLATE",
    "LLM_JUDGE_SCORES_CONTEXT_KEY",
    "default_judge_prompt_template_chat",
    "default_judge_prompt_template_completions",
    "default_judge_prompt_template_for_model",
    "is_chat_inference",
]
