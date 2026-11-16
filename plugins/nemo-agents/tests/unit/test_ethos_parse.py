# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for lightweight ETHOS.md parsing."""

from __future__ import annotations

import pytest
from nemo_agents_plugin.ethos import (
    ETHOS_SCHEMA_VERSION,
    ETHOS_SECTION_TITLES,
    RETIRED_SECTION_TITLES,
    known_sections,
    required_sections,
)
from nemo_agents_plugin.ethos_parse import EthosParseError, parse_ethos

_DEFAULT_OVERRIDES = {
    "Role": "help users with IT issues",
    "Framework": "- Resolution: langgraph-nat",
}


def _ethos_md(
    *,
    version: int | None = 1,
    titles: tuple[str, ...] | None = None,
    extra_front: str = "",
    sections: dict[str, str] | None = None,
) -> str:
    titles = titles if titles is not None else ETHOS_SECTION_TITLES
    front_lines = ["name: it-helpdesk", "created_timestamp: '2026-01-02T03:04:05+00:00'", "author: agent-1"]
    if version is not None:
        front_lines.append(f"schema_version: {version}")
    if extra_front:
        front_lines.append(extra_front)
    front = "---\n" + "\n".join(front_lines) + "\n---"

    bodies = {title: f"{title} content" for title in titles}
    bodies.update({k: v for k, v in _DEFAULT_OVERRIDES.items() if k in bodies})
    bodies.update(sections or {})
    body = "\n\n".join(f"## {title}\n\n{bodies[title]}" for title in titles)
    return f"{front}\n\n# Ethos: it-helpdesk\n\n{body}\n"


def test_valid_ethos_parses_to_metadata_and_sections() -> None:
    ethos = parse_ethos(_ethos_md())

    assert ethos.name == "it-helpdesk"
    assert ethos.author == "agent-1"
    assert ethos.role == "help users with IT issues"
    assert ethos.schema_version == 1
    assert ethos.warnings == ()


@pytest.mark.parametrize(
    "front",
    [
        "name: [unfinished",
        "name:\n\tinvalid: indentation",
        "name: *undefined",
        "name: !!python/object:builtins.object {}",
        "created_timestamp: 2026-13-02T03:04:05Z",
        "created_timestamp: 2026-02-30",
        "created_timestamp: '2026-13-02T03:04:05Z'",
    ],
    ids=["sequence", "indentation", "undefined-alias", "unsafe-tag", "timestamp", "date", "quoted-timestamp"],
)
def test_invalid_front_matter_raises_ethos_parse_error(front: str) -> None:
    markdown = _ethos_md(extra_front=front)
    with pytest.raises(EthosParseError):
        parse_ethos(markdown)


def test_missing_required_section_rejected() -> None:
    md = _ethos_md().replace("## Purpose & Outcomes\n\nPurpose & Outcomes content\n\n", "")

    with pytest.raises(EthosParseError, match=r"missing section: ## Purpose & Outcomes"):
        parse_ethos(md)


def test_absent_version_parses_as_v1_and_warns() -> None:
    ethos = parse_ethos(_ethos_md(version=None))

    assert ethos.schema_version == 1
    assert len(ethos.warnings) == 1
    assert "no 'schema_version'" in ethos.warnings[0]


def test_missing_section_is_an_error() -> None:
    titles = tuple(t for t in ETHOS_SECTION_TITLES if t != "Success Criteria")

    with pytest.raises(EthosParseError, match=r"missing section: ## Success Criteria"):
        parse_ethos(_ethos_md(titles=titles))


def test_missing_principles_is_an_error() -> None:
    titles = tuple(t for t in ETHOS_SECTION_TITLES if t != "Principles")

    with pytest.raises(EthosParseError, match=r"missing section: ## Principles"):
        parse_ethos(_ethos_md(titles=titles))


def test_unsupported_future_version_rejected() -> None:
    with pytest.raises(EthosParseError, match=r"schema version 99"):
        parse_ethos(_ethos_md(version=99))


def test_optional_front_matter_fields_are_parsed() -> None:
    ethos = parse_ethos(_ethos_md(extra_front="owner: platform-team\nupdated_timestamp: '2026-03-04T05:06:07+00:00'"))

    assert ethos.owner == "platform-team"
    assert ethos.updated_timestamp is not None
    assert ethos.updated_timestamp.year == 2026


def test_change_scope_levers_recognize_with_approval() -> None:
    change_scope = "\n".join(
        [
            "- System prompt: yes",
            "- Fine-tuning: no",
            "- Model swap (within mode): with-approval",
            "- Tools: **allowed**",
            "- Notes: escalate before broadening destructive capabilities",
        ]
    )

    ethos = parse_ethos(_ethos_md(sections={"Change Scope": change_scope}))

    assert ethos.change_scope_levers == {
        "System prompt": "yes",
        "Fine-tuning": "no",
        "Model swap (within mode)": "with-approval",
    }


@pytest.mark.parametrize(
    ("empty_value", "newline"),
    [("", "\n"), (" \t", "\n"), (" \t", "\r\n")],
    ids=["empty", "horizontal-whitespace", "crlf"],
)
def test_empty_change_scope_lever_does_not_consume_following_line(empty_value: str, newline: str) -> None:
    scope = newline.join([f"- Retrain:{empty_value}", "-\tPrompt edits:\tyes", "- Tools: no"])

    ethos = parse_ethos(_ethos_md(sections={"Change Scope": scope}))

    assert ethos.change_scope_levers == {"Prompt edits": "yes", "Tools": "no"}


def test_schema_drops_optimizer_run_configuration() -> None:
    """Budget was cut: Ethos records durable intent, not per-run limits."""
    assert "Budget" not in ETHOS_SECTION_TITLES


def test_schema_requires_every_section() -> None:
    """The parser requires the full section list."""
    assert required_sections(ETHOS_SCHEMA_VERSION) == ETHOS_SECTION_TITLES
    assert known_sections(ETHOS_SCHEMA_VERSION) == ETHOS_SECTION_TITLES


def test_section_helpers_reject_unsupported_versions() -> None:
    future = ETHOS_SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match=r"between 1 and 1"):
        required_sections(future)
    with pytest.raises(ValueError, match=r"between 1 and 1"):
        known_sections(future)
    with pytest.raises(ValueError, match=r"between 1 and 1"):
        required_sections(0)


def test_retired_headings_are_declared() -> None:
    """Retired AGENT-SPEC headings stay listed so a drop cannot happen silently."""
    assert set(RETIRED_SECTION_TITLES).isdisjoint(ETHOS_SECTION_TITLES)


def test_purpose_merged_into_outcomes() -> None:
    """Purpose became Purpose & Outcomes, so mission and result stay together."""
    assert "Purpose" not in ETHOS_SECTION_TITLES
    assert "Purpose & Outcomes" in ETHOS_SECTION_TITLES


def test_tolerates_leftover_retired_sections() -> None:
    """A file mid-upgrade keeps parsing: retired sections are carried, not rejected."""
    at = ETHOS_SECTION_TITLES.index("Tools") + 1
    titles = ETHOS_SECTION_TITLES[:at] + RETIRED_SECTION_TITLES + ETHOS_SECTION_TITLES[at:]

    ethos = parse_ethos(
        _ethos_md(titles=titles, sections={"Framework": "- Resolution: langgraph-nat"}),
        strict=True,
    )

    assert ethos.warnings == ()
    assert ethos.sections["Framework"] == "- Resolution: langgraph-nat"
    assert ethos.sections["Signals"] == "Signals content"


def test_custom_sections_are_preserved() -> None:
    """Extra headings are part of the contract, not a parse error."""
    titles = ETHOS_SECTION_TITLES + ("Team Runbook",)
    ethos = parse_ethos(
        _ethos_md(titles=titles, sections={"Team Runbook": "Page the on-call before expanding tools."}),
        strict=True,
    )

    assert ethos.sections["Team Runbook"] == "Page the on-call before expanding tools."
    assert ethos.warnings == ()


def test_unknown_front_matter_keys_do_not_fail() -> None:
    """Extra YAML keys are allowed; the parser does not enforce a closed map."""
    ethos = parse_ethos(_ethos_md(extra_front="team: growth"), strict=True)

    assert ethos.name == "it-helpdesk"
    assert ethos.warnings == ()


def test_none_answers_still_require_the_heading() -> None:
    """An honest empty answer keeps the heading; dropping it fails to parse."""
    ethos = parse_ethos(_ethos_md(sections={"Open Questions": "_(none)_", "Vision": "_(none)_"}))

    assert ethos.sections["Open Questions"].strip() == "_(none)_"
    assert ethos.sections["Vision"].strip() == "_(none)_"


def test_framework_is_retired() -> None:
    """Nothing read the section's value; the container label comes from agent.yaml."""
    assert "Framework" not in ETHOS_SECTION_TITLES
    assert "Framework" in RETIRED_SECTION_TITLES


def test_leftover_framework_must_be_resolved() -> None:
    """A leftover Framework heading still has to be resolved."""
    titles = ETHOS_SECTION_TITLES + ("Framework",)
    with pytest.raises(EthosParseError, match="framework section must be resolved"):
        parse_ethos(_ethos_md(titles=titles, sections={"Framework": "_(none)_"}))
