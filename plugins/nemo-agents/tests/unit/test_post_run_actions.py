# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolution and listing of ``agents.execute`` post-run action entry points."""

from __future__ import annotations

from importlib.metadata import EntryPoint

import pytest
from nemo_agents_plugin.jobs import post_run_actions
from nemo_agents_plugin.jobs.post_run_actions import (
    NOOP_POST_RUN_ACTION_KIND,
    NoopPostRunAction,
    list_post_run_actions,
    resolve_post_run_action,
    validate_post_run_action_config,
)
from nemo_helix_plugin.agents.post_run_actions import POST_RUN_ACTION_ENTRY_POINT_GROUP
from nemo_helix_plugin.agents.types import PostRunActionInfo

_NOOP_TARGET = "nemo_agents_plugin.jobs.post_run_actions:NoopPostRunAction"
# Any importable class without the display metadata or a config model.
_OTHER_TARGET = "nemo_helix_plugin.agents.post_run_actions:PostRunActionContext"


def _entry_point(name: str, value: str) -> EntryPoint:
    return EntryPoint(name=name, value=value, group=POST_RUN_ACTION_ENTRY_POINT_GROUP)


def _install(monkeypatch: pytest.MonkeyPatch, *entries: EntryPoint) -> None:
    monkeypatch.setattr(post_run_actions, "entry_points", lambda group: list(entries))


def test_the_installed_noop_action_resolves_through_its_entry_point() -> None:
    """``agents.noop`` is registered like any other action, with no loader special case."""
    assert resolve_post_run_action(NOOP_POST_RUN_ACTION_KIND) is NoopPostRunAction


def test_the_noop_action_is_not_resolvable_without_its_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch)

    with pytest.raises(ValueError, match="Unknown post-run action 'agents.noop'"):
        resolve_post_run_action(NOOP_POST_RUN_ACTION_KIND)


def test_a_single_registration_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _entry_point("demo.action", _NOOP_TARGET))

    assert resolve_post_run_action("demo.action") is NoopPostRunAction


def test_the_same_action_declared_twice_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    """The aggregate ``nemo-helix`` wheel re-declares every bundled plugin's
    entry points, so a standard install sees each kind twice — once from the
    plugin distribution, once from the aggregate. Two declarations of one
    implementation are not a conflict.
    """
    _install(
        monkeypatch,
        _entry_point("demo.action", _NOOP_TARGET),
        _entry_point("demo.action", _NOOP_TARGET),
    )

    assert resolve_post_run_action("demo.action") is NoopPostRunAction


def test_two_different_actions_claiming_one_kind_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Genuine ambiguity still fails: nothing can decide which one was meant."""
    _install(
        monkeypatch,
        _entry_point("demo.action", _NOOP_TARGET),
        _entry_point("demo.action", _OTHER_TARGET),
    )

    with pytest.raises(ValueError, match="Conflicting post-run actions"):
        resolve_post_run_action("demo.action")


def test_an_unregistered_kind_is_an_error_naming_the_installed_kinds(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _entry_point("demo.action", _NOOP_TARGET))

    with pytest.raises(ValueError, match=r"Unknown post-run action 'missing.action'\. .*: demo\.action\."):
        resolve_post_run_action("missing.action")


def test_the_installed_compass_action_resolves() -> None:
    """Guards the real environment: insights.analysis is declared twice on disk."""
    resolved = resolve_post_run_action("insights.analysis")

    assert resolved.__name__ == "CompassAnalysisAction"


def test_a_valid_action_config_is_accepted() -> None:
    validate_post_run_action_config("insights.analysis", {"agent": "demo-agent", "workspace": "default"})


def test_an_unknown_action_config_key_is_rejected() -> None:
    """``extra="forbid"`` catches a renamed or typo'd key on the create request."""
    with pytest.raises(ValueError) as excinfo:
        validate_post_run_action_config("insights.analysis", {"agent": "demo", "workspac": "default"})

    message = str(excinfo.value)
    assert "insights.analysis" in message, "the caller supplied a kind, so the error should name it"
    assert "workspac" in message


def test_a_wrongly_typed_action_config_value_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_post_run_action_config("insights.analysis", {"local_only": "yes-please"})


def test_the_noop_action_accepts_an_empty_config() -> None:
    validate_post_run_action_config(NOOP_POST_RUN_ACTION_KIND, {})


def test_the_noop_action_rejects_a_non_empty_config() -> None:
    """Handing config to an action that ignores it is a mistake worth surfacing."""
    with pytest.raises(ValueError):
        validate_post_run_action_config(NOOP_POST_RUN_ACTION_KIND, {"agent": "demo-agent"})


def test_an_action_declaring_no_config_model_is_left_unvalidated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lenient by design: an out-of-tree action predating the contract still creates."""
    _install(monkeypatch, _entry_point("legacy.action", _OTHER_TARGET))

    validate_post_run_action_config("legacy.action", {"anything": "goes"})


def test_validating_an_unregistered_kind_is_still_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch)

    with pytest.raises(ValueError, match="Unknown post-run action"):
        validate_post_run_action_config("nope.action", {})


def test_the_installed_actions_are_listed_with_their_display_metadata() -> None:
    listed = {info.kind: info for info in list_post_run_actions()}

    assert listed[NOOP_POST_RUN_ACTION_KIND] == PostRunActionInfo(
        kind=NOOP_POST_RUN_ACTION_KIND,
        title=NoopPostRunAction.title,
        description=NoopPostRunAction.description,
    )
    assert listed["insights.analysis"].title == "Compass analysis"


def test_listing_collapses_duplicate_declarations_and_sorts_by_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(
        monkeypatch,
        _entry_point("z.action", _NOOP_TARGET),
        _entry_point("a.action", _NOOP_TARGET),
        _entry_point("a.action", _NOOP_TARGET),
    )

    assert [info.kind for info in list_post_run_actions()] == ["a.action", "z.action"]


def test_listing_skips_a_kind_that_cannot_be_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """One broken plugin must not take the whole listing down."""
    _install(
        monkeypatch,
        _entry_point("good.action", _NOOP_TARGET),
        _entry_point("conflicted.action", _NOOP_TARGET),
        _entry_point("conflicted.action", _OTHER_TARGET),
        _entry_point("broken.action", "nemo_agents_plugin.does_not_exist:Action"),
    )

    assert [info.kind for info in list_post_run_actions()] == ["good.action"]


def test_listing_falls_back_when_an_action_declares_no_display_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _entry_point("legacy.action", _OTHER_TARGET))

    assert list_post_run_actions() == [PostRunActionInfo(kind="legacy.action", title="legacy.action", description="")]


def test_listing_scans_the_installed_entry_points_once(monkeypatch: pytest.MonkeyPatch) -> None:
    scans: list[str] = []

    def _entry_points(group: str) -> list[EntryPoint]:
        scans.append(group)
        return [_entry_point("a.action", _NOOP_TARGET), _entry_point("b.action", _NOOP_TARGET)]

    monkeypatch.setattr(post_run_actions, "entry_points", _entry_points)

    assert [info.kind for info in list_post_run_actions()] == ["a.action", "b.action"]
    assert scans == [POST_RUN_ACTION_ENTRY_POINT_GROUP]


def test_validating_against_an_already_resolved_action_skips_the_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """The execute step validates against the class it already resolved from its own image."""
    _install(monkeypatch)

    validate_post_run_action_config("unregistered.action", {}, action=NoopPostRunAction)
    with pytest.raises(ValueError, match="Invalid config for post-run action 'unregistered.action'"):
        validate_post_run_action_config("unregistered.action", {"unexpected": True}, action=NoopPostRunAction)
