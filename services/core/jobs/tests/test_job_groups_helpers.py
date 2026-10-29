# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pure job-group helpers: status aggregation, spec expansion, timeout resolution."""

import pytest
from nemo_helix_plugin.jobs.schemas import (
    JobGroupStatus,
    aggregate_job_group_status,
    aggregate_job_group_status_from_counts,
)
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.app.dispatcher import (
    _deep_merge,
    _effective_member_timeout,
    _expand_member_spec,
)
from nhx.core.jobs.app.schemas import HelixJobSpec, HelixJobStepSpec
from nhx.core.jobs.app.test_helpers import TestConstants

# ---------------------------------------------------------------------------
# aggregate_job_group_status
# ---------------------------------------------------------------------------


def test_aggregate_empty_group_is_pending():
    assert aggregate_job_group_status([]) == JobGroupStatus.PENDING


def test_aggregate_all_created_is_pending():
    assert aggregate_job_group_status([HelixJobStatus.CREATED, HelixJobStatus.CREATED]) == JobGroupStatus.PENDING


def test_aggregate_any_active_is_running():
    assert aggregate_job_group_status([HelixJobStatus.CREATED, HelixJobStatus.ACTIVE]) == JobGroupStatus.RUNNING


def test_aggregate_pending_member_is_running_not_pending():
    # A member past CREATED (queued/pending) means the group is underway.
    assert aggregate_job_group_status([HelixJobStatus.PENDING, HelixJobStatus.CREATED]) == JobGroupStatus.RUNNING


def test_aggregate_all_completed_is_completed():
    assert aggregate_job_group_status([HelixJobStatus.COMPLETED, HelixJobStatus.COMPLETED]) == JobGroupStatus.COMPLETED


def test_aggregate_any_error_when_terminal_is_failed():
    assert aggregate_job_group_status([HelixJobStatus.COMPLETED, HelixJobStatus.ERROR]) == JobGroupStatus.FAILED


def test_aggregate_error_but_others_running_is_not_terminal():
    # The group is terminal only when every member is terminal, even if one has already failed.
    assert aggregate_job_group_status([HelixJobStatus.ERROR, HelixJobStatus.ACTIVE]) == JobGroupStatus.RUNNING


def test_aggregate_all_cancelled_is_cancelled():
    assert aggregate_job_group_status([HelixJobStatus.CANCELLED, HelixJobStatus.CANCELLED]) == JobGroupStatus.CANCELLED


def test_aggregate_error_wins_over_cancelled_when_terminal():
    assert aggregate_job_group_status([HelixJobStatus.CANCELLED, HelixJobStatus.ERROR]) == JobGroupStatus.FAILED


def test_aggregate_cancel_in_flight_is_cancelling():
    assert aggregate_job_group_status([HelixJobStatus.CANCELLING, HelixJobStatus.ACTIVE]) == JobGroupStatus.CANCELLING


def test_aggregate_cancelled_member_with_running_peer_is_cancelling():
    # A cancel sweeping the group (one cancelled, others still running) reads as CANCELLING, not terminal.
    assert aggregate_job_group_status([HelixJobStatus.CANCELLED, HelixJobStatus.ACTIVE]) == JobGroupStatus.CANCELLING


@pytest.mark.parametrize("status", JobGroupStatus.terminals())
def test_terminals_are_terminal(status):
    assert status.is_terminal()


@pytest.mark.parametrize("status", [JobGroupStatus.PENDING, JobGroupStatus.RUNNING, JobGroupStatus.CANCELLING])
def test_non_terminals_are_not_terminal(status):
    assert not status.is_terminal()


# ---------------------------------------------------------------------------
# aggregate_job_group_status_from_counts (count-driven; must match the list-based aggregate)
# ---------------------------------------------------------------------------


def test_aggregate_from_counts_empty_group_is_pending():
    assert aggregate_job_group_status_from_counts(0, {}) == JobGroupStatus.PENDING


def test_aggregate_from_counts_treats_unprobed_members_as_created():
    # total 3, only 1 bucket given -> the other 2 are implicitly CREATED -> still RUNNING.
    assert aggregate_job_group_status_from_counts(3, {HelixJobStatus.ACTIVE: 1}) == JobGroupStatus.RUNNING


@pytest.mark.parametrize(
    "total, counts",
    [
        (3, {HelixJobStatus.COMPLETED: 3}),
        (3, {HelixJobStatus.COMPLETED: 2, HelixJobStatus.ERROR: 1}),
        (3, {HelixJobStatus.ERROR: 1}),
        (2, {HelixJobStatus.CANCELLING: 1}),
        (4, {}),
        (2, {HelixJobStatus.ACTIVE: 1}),
        (3, {HelixJobStatus.COMPLETED: 1, HelixJobStatus.CANCELLED: 2}),
        (5, {HelixJobStatus.COMPLETED: 3, HelixJobStatus.ERROR: 1, HelixJobStatus.CANCELLED: 1}),
        (2, {HelixJobStatus.CREATED: 2}),
    ],
)
def test_aggregate_from_counts_matches_list_based(total, counts):
    """The count-driven aggregate must equal the list-based one for the same member multiset.

    This pins the shared-precedence contract: the dispatcher rolls group status up from cheap
    per-status counts, and that must never diverge from deriving it over a full status list.
    """
    representative: list[HelixJobStatus] = []
    for status, count in counts.items():
        representative.extend([status] * count)
    representative.extend([HelixJobStatus.CREATED] * (total - len(representative)))
    assert aggregate_job_group_status_from_counts(total, counts) == aggregate_job_group_status(representative)


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------


def test_deep_merge_recurses_into_nested_dicts():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    patch = {"a": {"y": 9, "z": 5}, "c": 7}
    assert _deep_merge(base, patch) == {"a": {"x": 1, "y": 9, "z": 5}, "b": 3, "c": 7}


def test_deep_merge_replaces_non_dict_values_wholesale():
    assert _deep_merge({"a": [1, 2]}, {"a": [3]}) == {"a": [3]}


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"x": 1}}
    patch = {"a": {"y": 2}}
    _deep_merge(base, patch)
    assert base == {"a": {"x": 1}}
    assert patch == {"a": {"y": 2}}


# ---------------------------------------------------------------------------
# _effective_member_timeout
# ---------------------------------------------------------------------------


def test_timeout_member_override_wins():
    assert _effective_member_timeout(member_override=10, group_default=99) == 10


def test_timeout_falls_back_to_group_default():
    assert _effective_member_timeout(member_override=None, group_default=99) == 99


def test_timeout_none_when_neither_set():
    assert _effective_member_timeout(member_override=None, group_default=None) is None


def test_timeout_zero_override_is_respected():
    # 0 is a real value (disables staleness detection), not "unset".
    assert _effective_member_timeout(member_override=0, group_default=99) == 0


# ---------------------------------------------------------------------------
# _expand_member_spec
# ---------------------------------------------------------------------------


def _two_step_template() -> HelixJobSpec:
    return HelixJobSpec(
        steps=[
            HelixJobStepSpec(name="first", executor=TestConstants.TEST_EXECUTOR, config={"a": 1}),
            HelixJobStepSpec(name="second", executor=TestConstants.TEST_EXECUTOR, config={"nested": {"keep": 1}}),
        ]
    )


def test_expand_merges_config_patch_into_every_step():
    template = _two_step_template()
    member = _expand_member_spec(template, config_patch={"b": 2, "nested": {"add": 3}}, timeout_seconds=None)
    assert member.steps[0].config == {"a": 1, "b": 2, "nested": {"add": 3}}
    assert member.steps[1].config == {"nested": {"keep": 1, "add": 3}, "b": 2}


def test_expand_applies_timeout_to_every_step_lifecycle():
    template = _two_step_template()
    member = _expand_member_spec(template, config_patch={}, timeout_seconds=42)
    assert member.steps[0].lifecycle.staleness_timeout_seconds == 42
    assert member.steps[1].lifecycle.staleness_timeout_seconds == 42


def test_expand_leaves_template_timeout_when_none():
    template = _two_step_template()
    template.steps[0].lifecycle.staleness_timeout_seconds = 7
    member = _expand_member_spec(template, config_patch={}, timeout_seconds=None)
    assert member.steps[0].lifecycle.staleness_timeout_seconds == 7


def test_expand_does_not_mutate_template():
    template = _two_step_template()
    _expand_member_spec(template, config_patch={"b": 2}, timeout_seconds=42)
    assert template.steps[0].config == {"a": 1}
    assert template.steps[0].lifecycle.staleness_timeout_seconds == 0
