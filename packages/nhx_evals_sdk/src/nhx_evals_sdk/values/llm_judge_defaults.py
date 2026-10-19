# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default prompt templates for LLM judge metric values."""

from typing import Any

from nhx_evals_sdk.values.common import SupportedJobTypes
from nhx_evals_sdk.values.models import Model, ModelRef

DEFAULT_PROMPT_TEMPLATE = "{{item}}"
LLM_JUDGE_SCORES_CONTEXT_KEY = "scores"
DEFAULT_JUDGE_SYSTEM_PROMPT_TEMPLATE = """You are an expert evaluator for answers to user queries. Your task is to assess responses to user queries based on {{ scores.keys() | join(", ") }}
{% if scores | length > 1 %}Scores:{% endif %}
{%- for score_name, score in scores.items() %}
{{ score_name }}{%- if "minimum" in score %} with a score range from {{ score.minimum }} to {{ score.maximum }}{%- endif %}{% if score.description %}: {{score.description}}{% endif %}
{%- if "rubric" in score %}
{%- for rubric in score.rubric %}
* {{ rubric.label }}{% if rubric.description %}: {{rubric.description}}{% endif %}
{%- endfor -%}
{%- endif -%}
{%- endfor -%}
"""
DEFAULT_JUDGE_PROMPT_TEMPLATE_WITH_TARGET_MODEL = "{{sample.output_text}}"


def is_chat_inference(url: str) -> bool:
    """Check if the URL is for chat inference (vs completions)."""
    return "/v1/completions" not in url


def default_judge_prompt_template_chat(job_type: SupportedJobTypes = SupportedJobTypes.ONLINE) -> dict:
    prompt = (
        DEFAULT_JUDGE_PROMPT_TEMPLATE_WITH_TARGET_MODEL
        if job_type == SupportedJobTypes.ONLINE
        else DEFAULT_PROMPT_TEMPLATE
    )
    return {
        "messages": [
            {"role": "system", "content": DEFAULT_JUDGE_SYSTEM_PROMPT_TEMPLATE},
            {"role": "user", "content": prompt},
        ]
    }


def default_judge_prompt_template_completions(job_type: SupportedJobTypes = SupportedJobTypes.ONLINE) -> dict:
    prompt = (
        DEFAULT_JUDGE_PROMPT_TEMPLATE_WITH_TARGET_MODEL
        if job_type == SupportedJobTypes.ONLINE
        else DEFAULT_PROMPT_TEMPLATE
    )
    return {"prompt": f"{DEFAULT_JUDGE_SYSTEM_PROMPT_TEMPLATE}\n{prompt}"}


def _model_uses_chat_prompt_default(model: Model | ModelRef | dict[str, Any]) -> bool:
    """Return whether a model-like value should default to chat prompt format."""
    if isinstance(model, ModelRef):
        return True
    if isinstance(model, Model):
        return is_chat_inference(model.url)

    url = model.get("url")
    if isinstance(url, str):
        return is_chat_inference(url)

    root = model.get("root")
    if isinstance(root, str):
        return True

    raise ValueError("model.url or ModelRef.root is required to infer the default prompt template")


def default_judge_prompt_template_for_model(
    model: Model | ModelRef | dict[str, Any],
    job_type: SupportedJobTypes = SupportedJobTypes.ONLINE,
) -> dict:
    """Return the default judge prompt template for a model-like value."""
    if _model_uses_chat_prompt_default(model):
        return default_judge_prompt_template_chat(job_type)
    return default_judge_prompt_template_completions(job_type)
