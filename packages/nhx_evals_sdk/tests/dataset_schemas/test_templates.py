# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from textwrap import dedent

import pytest
from nhx_evals_sdk.dataset_schemas.templates import infer_required_schema_from_template


def test_infer_required_schema_from_template_ignores_prompt_structure_and_tracks_context_references():
    schema = infer_required_schema_from_template(
        {
            "messages": [
                {"role": "user", "content": "{{ item.user_message | lower }}"},
                {"role": "system", "content": "{{ metadata.request_id }}"},
                {"role": "assistant", "content": "{{ sample.output_text }}"},
            ]
        }
    )

    assert schema["type"] == "object"
    assert sorted(schema["required"]) == ["metadata", "output", "user_message"]
    assert schema["properties"]["user_message"] == {}
    assert schema["properties"]["output"] == {}
    assert schema["properties"]["metadata"]["type"] == "object"
    assert schema["properties"]["metadata"]["required"] == ["request_id"]


def test_infer_required_schema_from_template_rejects_control_flow():
    with pytest.raises(ValueError, match="unsupported Jinja construct"):
        infer_required_schema_from_template("{% for message in item.messages %}{{ message.content }}{% endfor %}")


def test_infer_required_schema_from_template_treats_scores_as_required_by_default():
    schema = infer_required_schema_from_template("Scores: {{ scores.correctness.description }}")

    assert schema["required"] == ["scores"]
    assert schema["properties"]["scores"]["type"] == "object"
    assert schema["properties"]["scores"]["required"] == ["correctness"]


def test_infer_required_schema_from_template_allows_ignored_root_loops():
    schema = infer_required_schema_from_template(
        dedent(
            """
            {% for score_name, score in scores.items() %}
            {{ score_name }}{% if score.description %}: {{ score.description }}{% endif %}
            {% endfor %}
            Question: {{ item.prompt }}
            """
        ).lstrip(),
        ignored_roots={"scores"},
    )

    assert schema["required"] == ["prompt"]


def test_infer_required_schema_from_template_tracks_item_references_inside_ignored_root_loops():
    schema = infer_required_schema_from_template(
        dedent(
            """
            {% for score_name, score in scores.items() %}
            {{ item.prompt }} -> {{ sample.output_text }}
            {% endfor %}
            """
        ).lstrip(),
        ignored_roots={"scores"},
    )

    assert sorted(schema["required"]) == ["output", "prompt"]


def test_infer_required_schema_from_template_supports_common_text_filters():
    schema = infer_required_schema_from_template(
        "Question: {{ input | trim | lower }}\nContext: {{ item.context | replace('\\n', ' ') }}",
    )

    assert sorted(schema["required"]) == ["context", "input"]


def test_infer_required_schema_from_template_supports_function_style_calls():
    schema = infer_required_schema_from_template("Question: {{ upper(input) }}")

    assert schema["required"] == ["input"]


def test_infer_required_schema_from_template_supports_method_style_calls():
    schema = infer_required_schema_from_template("Question: {{ input.upper() }}")

    assert schema["required"] == ["input"]


def test_infer_required_schema_from_template_extracts_call_arguments_and_kwargs():
    schema = infer_required_schema_from_template("A: {{ format(item.a, sample.output_text, x=item.value) }}")

    assert sorted(schema["required"]) == ["a", "output", "value"]


def test_infer_required_schema_from_template_rejects_unsupported_expressions():
    with pytest.raises(ValueError, match="unsupported Jinja expression"):
        infer_required_schema_from_template("{{ item.input ~ item.reference }}")
