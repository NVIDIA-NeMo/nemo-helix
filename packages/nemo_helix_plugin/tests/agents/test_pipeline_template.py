# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pipeline placeholder templates."""

from __future__ import annotations

import pytest
from nemo_helix_plugin.agents import pipeline_template as tpl

CONTEXT = {
    "input": "the email",
    "steps": {
        "triage": {
            "response": {"verdict": "phishing", "indicators": ["link", "sender"], "a.b": 1},
            "output": {"findings": [{"score": 0.9}]},
        }
    },
    "previous": {"response": "plain text"},
}


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("input", ("input",)),
        ("steps.triage.response.verdict", ("steps", "triage", "response", "verdict")),
        ("steps.triage.output.findings[0].score", ("steps", "triage", "output", "findings", 0, "score")),
        ('steps.triage.response["a.b"]', ("steps", "triage", "response", "a.b")),
        ("steps.my-step.response", ("steps", "my-step", "response")),
    ],
)
def test_parse_path(source: str, expected: tuple[object, ...]) -> None:
    assert tpl.parse_path(source) == expected


def test_spaces_inside_braces_are_optional() -> None:
    assert tpl.render("{{input}}|{{ input }}|{{   input   }}", CONTEXT) == "the email|the email|the email"


def test_render_inserts_json_for_non_strings() -> None:
    rendered = tpl.render("v={{ steps.triage.response.verdict }} i={{ steps.triage.response.indicators }}", CONTEXT)
    assert rendered == 'v=phishing i=["link","sender"]'


def test_render_value_keeps_type_for_a_single_placeholder() -> None:
    assert tpl.render_value(" {{ steps.triage.response.indicators }} ", CONTEXT) == ["link", "sender"]
    assert tpl.render_value("x {{ steps.triage.response.indicators }}", CONTEXT) == 'x ["link","sender"]'


def test_render_tree_renders_nested_strings_only() -> None:
    tree = {"a": "{{ input }}", "b": [{"c": "{{ steps.triage.output.findings[0].score }}"}], "{{ input }}": 3}
    assert tpl.render_tree(tree, CONTEXT) == {"a": "the email", "b": [{"c": 0.9}], "{{ input }}": 3}


def test_raw_block_is_literal() -> None:
    assert (
        tpl.render("{% raw %}{{ name }} {% if %}{% endraw %} {{ input }}", CONTEXT) == "{{ name }} {% if %} the email"
    )


def test_inserted_values_are_not_rendered_again() -> None:
    context = {"input": "{{ secret }}", "secret": "leak"}
    assert tpl.render("{{ input }}", context) == "{{ secret }}"


@pytest.mark.parametrize(
    ("template", "message"),
    [
        ("{{ input | upper }}", "Filters are not supported"),
        ("{{ a + b }}", "Expressions are not supported"),
        ("{% if x %}y{% endif %}", "{% if %} is not supported"),
        ("{% for x in y %}", "{% for %} is not supported"),
        ("{# note #}", "Comments"),
        ("{{ input", "Unclosed placeholder"),
        ("{% raw %}abc", "Unclosed {% raw %}"),
        ("{% raw %}{% raw %}{% endraw %}", "cannot be nested"),
        ("{% endraw %}", "no matching {% raw %}"),
        ("{{ }}", "Empty placeholder"),
        ("{{ 1abc }}", "must start with a name"),
        ("{{ a[x] }}", "list index"),
        ('{{ a["x\\q"] }}', "Invalid escape"),
        ("{%- raw %}x{% endraw %}", "Whitespace control"),
        ("{% raw -%}x{% endraw %}", "Whitespace control"),
        ("{% raw %}x{% endraw -%}", "Whitespace control"),
        ("{% if x -%}", "{% if %} is not supported"),
        ("{{ it's }}", "Expressions are not supported"),
    ],
)
def test_unsupported_syntax_is_rejected(template: str, message: str) -> None:
    with pytest.raises(tpl.TemplateSyntaxError, match=message.replace("{", r"\{").replace("+", r"\+")):
        tpl.parse(template)


def test_missing_key_and_index_raise_render_errors() -> None:
    with pytest.raises(tpl.TemplateRenderError, match="no key 'nope'"):
        tpl.render("{{ steps.triage.response.nope }}", CONTEXT)
    with pytest.raises(tpl.TemplateRenderError, match=r"no item \[5\]"):
        tpl.render("{{ steps.triage.response.indicators[5] }}", CONTEXT)


def test_tree_strings_reports_locations() -> None:
    assert tpl.tree_strings({"a": ["x", {"b": "y"}], "n": 1}, "context") == [
        ("context.a[0]", "x"),
        ("context.a[1].b", "y"),
    ]


def test_format_path_round_trips() -> None:
    path = ("steps", "triage", "response", "a.b", 0)
    assert tpl.parse_path(tpl.format_path(path)) == path


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("a['it\\'s']", ("a", "it's")),
        ('a["say \\"hi\\""]', ("a", 'say "hi"')),
        ('a["x}}y"]', ("a", "x}}y")),
        ('a["\\u00e9\\r\\/"]', ("a", "é\r/")),
    ],
)
def test_quoted_keys_decode_escapes_and_may_contain_braces(source: str, expected: tuple[str, ...]) -> None:
    (placeholder,) = tpl.placeholders("{{ " + source + " }}")
    assert placeholder.path == expected
    assert tpl.parse_path(tpl.format_path(placeholder.path)) == expected


def test_apostrophes_outside_keys_do_not_open_strings() -> None:
    assert [p.dotted for p in tpl.placeholders("{{ input }} it's {{ steps.a.response }} don't")] == [
        "input",
        "steps.a.response",
    ]


@pytest.mark.parametrize("key", ["a\rb", "\u0001", "é", 'q"\\/', "tab\tnl\n"])
def test_format_path_round_trips_every_key(key: str) -> None:
    path = ("steps", "a", "output", key)
    assert tpl.parse_path(tpl.format_path(path)) == path
