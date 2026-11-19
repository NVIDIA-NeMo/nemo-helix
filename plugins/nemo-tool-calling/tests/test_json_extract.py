# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_tool_calling_plugin.json_extract import json_object_in, parse_error, top_level_objects


def test_scanner_ignores_braces_inside_strings() -> None:
    text = 'Before {"prompt": "Review {{ product }} with \\"{\\" chars }"} after {"b": 1}'

    assert top_level_objects(text) == ['{"prompt": "Review {{ product }} with \\"{\\" chars }"}', '{"b": 1}']


def test_scanner_finds_prose_braces_as_their_own_spans() -> None:
    assert top_level_objects('Use {{ column }} then {"a": {"b": 2}}') == ["{{ column }}", '{"a": {"b": 2}}']


def test_scanner_skips_an_unclosed_object() -> None:
    assert top_level_objects('{"a": 1} and {"b": ') == ['{"a": 1}']


def test_comments_and_trailing_commas_are_repaired() -> None:
    text = """{
      // the fix
      "schema_fix": {"edits": [{"path": "/a", "value": 1},], /* none */ },
    }"""

    assert json_object_in(text, keys=["schema_fix"]) == {"schema_fix": {"edits": [{"path": "/a", "value": 1}]}}


def test_the_largest_valid_span_wins_over_prose_braces() -> None:
    text = (
        'The column uses {{ review }}. A small example {"schema_fix": null}. '
        'The answer: {"schema_fix": {"title": "t", "edits": [{"path": "/x", "value": "{{ x }}"}]}}'
    )

    assert json_object_in(text, keys=["schema_fix"]) == {
        "schema_fix": {"title": "t", "edits": [{"path": "/x", "value": "{{ x }}"}]}
    }


def test_a_fenced_block_comes_before_bare_spans() -> None:
    text = 'Draft {"schema_fix": {"title": "draft", "rationale": "longer than the final one"}}\n```json\n{"schema_fix": {"title": "final"}}\n```'

    assert json_object_in(text, keys=["schema_fix"]) == {"schema_fix": {"title": "final"}}


def test_objects_without_an_expected_key_are_rejected() -> None:
    assert json_object_in('{"other": 1} {{ column }}', keys=["schema_fix"]) is None


def test_parse_error_names_the_problem_in_the_largest_candidate() -> None:
    assert parse_error('{"a": 1 "b": 2}').startswith("Expecting ',' delimiter at line 1 column 9")
    assert parse_error("no braces") == "no {...} object in the reply"
