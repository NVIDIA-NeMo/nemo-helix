# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sub-entity (``#fragment``) reference parsing and validation.

A revision is addressed with the platform's standard ``#`` fragment — the same convention filesets
use for a contained file (``workspace/fileset#path``). These tests pin two things: that an absent
fragment means ``latest`` rather than "unpinned", and that a fragment-unaware caller reading a
pinned ref still lands on the right task rather than on one literally named ``task-a#<digest>``.
"""

from __future__ import annotations

import re

import pytest
from nemo_evaluator.api.schemas import (
    LATEST_TAG,
    MetricRef,
    TaskRef,
    TasksetRef,
    parse_subentity_ref,
)
from nemo_helix_plugin.refs import ENTITY_REF_PATTERN, parse_entity_ref
from pydantic import ValidationError

_DIGEST = "a" * 64


# --- Parsing -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("task-a", ("default", "task-a", LATEST_TAG)),
        ("other/task-a", ("other", "task-a", LATEST_TAG)),
        ("task-a#latest", ("default", "task-a", "latest")),
        ("other/task-a#candidate", ("other", "task-a", "candidate")),
        (f"other/task-a#{_DIGEST}", ("other", "task-a", _DIGEST)),
    ],
)
def test_parse_subentity_ref(ref: str, expected: tuple[str, str, str]) -> None:
    assert parse_subentity_ref(ref, "default") == expected


def test_absent_fragment_means_latest_not_unpinned() -> None:
    """A bare ref is "the current revision", which is what makes `#latest` a real default rather
    than a special case the resolver has to guess at."""
    _, _, fragment = parse_subentity_ref("task-a", "default")
    assert fragment == LATEST_TAG


def test_empty_fragment_falls_back_to_latest() -> None:
    """``task-a#`` is degenerate input; treat it as unpinned rather than as a tag named ''."""
    assert parse_subentity_ref("task-a#", "default") == ("default", "task-a", LATEST_TAG)


def test_fragment_is_returned_verbatim() -> None:
    """Parsing does not decide whether a fragment is a tag or a digest — that's resolution's job."""
    _, _, fragment = parse_subentity_ref(f"task-a#{_DIGEST}", "default")
    assert fragment == _DIGEST


# --- Composition with the platform's entity parser ---------------------------


def test_dropping_the_fragment_recovers_the_plain_entity_ref() -> None:
    """Fragment-unaware callers (taskset member existence checks) read a pinned ref by discarding the
    third element, rather than through a second parser that strips ``#`` itself. Keeping one parser
    is what stops evaluator refs and platform refs drifting on what a ``workspace/name`` is."""
    assert parse_subentity_ref(f"other/task-a#{_DIGEST}", "default")[:2] == ("other", "task-a")
    assert parse_subentity_ref("task-a#latest", "default")[:2] == ("default", "task-a")


def test_pinned_and_bare_refs_resolve_to_the_same_task() -> None:
    """The property taskset duplicate-detection relies on: two refs differing only by fragment are
    the same member, and must not both be admitted."""
    assert parse_subentity_ref(f"task-a#{_DIGEST}", "default")[:2] == parse_subentity_ref("task-a", "default")[:2]


def test_the_base_split_is_the_platform_parser() -> None:
    """Not an implementation detail worth pinning for its own sake — it is the guarantee that a
    reference means the same thing to the evaluator as it does to every other plugin."""
    parsed = parse_entity_ref("other/task-a", "default")
    assert parse_subentity_ref("other/task-a", "default")[:2] == (parsed.workspace, parsed.name)


def test_subentity_pattern_is_the_entity_pattern_plus_a_fragment() -> None:
    """The evaluator's ref shape is derived from the platform constant, so widening what counts as a
    ``workspace/name`` widens both at once instead of leaving one behind."""
    assert TaskRef.model_fields["root"].metadata  # the pattern is declared on the field
    for bare in ("task-a", "other/task-a"):
        assert re.fullmatch(ENTITY_REF_PATTERN, bare)
        assert TaskRef(bare).root == bare


# --- Field validation --------------------------------------------------------


@pytest.mark.parametrize("ref", ["task-a", "other/task-a", "task-a#latest", f"other/task-a#{_DIGEST}"])
def test_task_ref_accepts_fragments(ref: str) -> None:
    assert TaskRef(ref).root == ref


@pytest.mark.parametrize("ref", ["task-a#one#two", "task-a#bad/frag", "#latest", "task a#latest"])
def test_task_ref_rejects_malformed_fragments(ref: str) -> None:
    with pytest.raises(ValidationError):
        TaskRef(ref)


def test_metric_ref_still_rejects_fragments() -> None:
    """The fragment pattern is a sibling, not a widening of the shared constant: metrics have no
    revisions, so admitting a fragment would accept input nothing resolves. ``MetricRef`` moves onto
    it when metrics gain revisions — deliberately, at that point."""
    with pytest.raises(ValidationError):
        MetricRef(f"other/thing#{_DIGEST}")


@pytest.mark.parametrize("ref", ["suite", "other/suite", "suite#latest", "suite#blessed", f"other/suite#{_DIGEST}"])
def test_taskset_ref_accepts_fragments(ref: str) -> None:
    """Tasksets are revisioned, so a ref may pin the revision to expand — same shape as ``TaskRef``."""
    assert TasksetRef(ref).root == ref


@pytest.mark.parametrize("ref", ["suite#one#two", "suite#bad/frag", "#latest", "suite name#latest"])
def test_taskset_ref_rejects_malformed_fragments(ref: str) -> None:
    with pytest.raises(ValidationError):
        TasksetRef(ref)
