# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nmp.core.models.parallelism.utils import detect_reasoning_toggle

NEMOTRON_TEMPLATE = (
    "{%- set enable_thinking = enable_thinking if enable_thinking is defined else true %}\n"
    "{%- if add_generation_prompt %}\n"
    "    {%- if enable_thinking %}<think>{%- endif %}\n"
    "{%- endif %}"
)

LLAMA_TEMPLATE = (
    "{%- for message in messages %}"
    "<|start_header_id|>{{ message['role'] }}<|end_header_id|>{{ message['content'] }}"
    "{%- endfor %}"
)


def test_template_branching_on_enable_thinking() -> None:
    assert detect_reasoning_toggle(NEMOTRON_TEMPLATE) is True


def test_template_without_the_kwarg() -> None:
    assert detect_reasoning_toggle(LLAMA_TEMPLATE) is False


def test_named_template_list() -> None:
    templates = [
        {"name": "default", "template": LLAMA_TEMPLATE},
        {"name": "thinking", "template": NEMOTRON_TEMPLATE},
    ]
    assert detect_reasoning_toggle(templates) is True


def test_named_template_list_without_the_kwarg() -> None:
    assert detect_reasoning_toggle([{"name": "default", "template": LLAMA_TEMPLATE}]) is False


def test_template_mapping() -> None:
    assert detect_reasoning_toggle({"default": LLAMA_TEMPLATE, "thinking": NEMOTRON_TEMPLATE}) is True


def test_absent_template() -> None:
    assert detect_reasoning_toggle(None) is False
