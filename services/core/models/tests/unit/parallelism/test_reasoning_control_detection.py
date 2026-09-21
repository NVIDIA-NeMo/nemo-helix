# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nmp.core.models.parallelism.utils import default_chat_template, detect_reasoning_control

OVERRIDABLE = (
    "{%- set enable_thinking = enable_thinking if enable_thinking is defined else true %}"
    "{%- if add_generation_prompt %}{%- if enable_thinking %}<think>{%- endif %}{%- endif %}"
)

HARDCODED = "{%- set enable_thinking = true %}{%- if enable_thinking %}<think>{%- endif %}"

PLAIN = "{%- for message in messages %}{{ message['role'] }}: {{ message['content'] }}{%- endfor %}"

DEEPSEEK_STYLE = "{%- if thinking %}<think>{%- endif %}{{ messages[0]['content'] }}"

# gpt-oss-style: the effort is interpolated into the system prompt rather than
# branched on as a boolean.
EFFORT_INTERPOLATED = "<|system|>Reasoning: {{ reasoning_effort }}<|end|>{{ messages[0]['content'] }}"

EFFORT_MENTIONED_BUT_FIXED = "<|system|>Reasoning: high<|end|>{# reasoning_effort ignored #}"


def test_template_branching_on_the_kwarg() -> None:
    assert detect_reasoning_control(OVERRIDABLE) is True


def test_template_that_hardcodes_the_kwarg_is_not_a_toggle() -> None:
    assert detect_reasoning_control(HARDCODED) is False


def test_template_without_the_kwarg() -> None:
    assert detect_reasoning_control(PLAIN) is False


def test_alternate_kwarg_spelling() -> None:
    assert detect_reasoning_control(DEEPSEEK_STYLE) is True


def test_effort_interpolated_into_the_prompt() -> None:
    assert detect_reasoning_control(EFFORT_INTERPOLATED) is True


def test_effort_named_in_a_comment_but_never_used() -> None:
    assert detect_reasoning_control(EFFORT_MENTIONED_BUT_FIXED) is False


def test_absent_template_is_a_definite_no() -> None:
    assert detect_reasoning_control(None) is False


def test_named_templates_consult_only_the_default() -> None:
    templates = [
        {"name": "default", "template": PLAIN},
        {"name": "thinking", "template": OVERRIDABLE},
    ]
    assert detect_reasoning_control(templates) is False


def test_named_templates_use_the_default_when_it_toggles() -> None:
    templates = [
        {"name": "default", "template": OVERRIDABLE},
        {"name": "tool_use", "template": PLAIN},
    ]
    assert detect_reasoning_control(templates) is True


def test_named_templates_without_a_default_are_undetermined() -> None:
    assert detect_reasoning_control([{"name": "tool_use", "template": PLAIN}]) is None


def test_mapping_of_named_templates() -> None:
    assert detect_reasoning_control({"default": OVERRIDABLE, "tool_use": PLAIN}) is True


def test_unrenderable_template_is_undetermined() -> None:
    assert detect_reasoning_control("{% if enable_thinking %}{% endfor %}") is None


def test_default_chat_template_resolution() -> None:
    assert default_chat_template(PLAIN) == PLAIN
    assert default_chat_template([{"name": "default", "template": PLAIN}]) == PLAIN
    assert default_chat_template({"default": PLAIN}) == PLAIN
    assert default_chat_template([{"name": "tool_use", "template": PLAIN}]) is None
    assert default_chat_template(None) is None
