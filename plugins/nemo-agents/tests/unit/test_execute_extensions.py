# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolution of trusted ``agents.execute`` extension entry points."""

from __future__ import annotations

from importlib.metadata import EntryPoint

import pytest
from nemo_agents_plugin.jobs import execute_extensions
from nemo_agents_plugin.jobs.execute_extensions import (
    EXECUTE_AGENT_EXTENSION_ENTRY_POINT_GROUP,
    NOOP_EXECUTE_AGENT_EXTENSION_KIND,
    NoopExecuteAgentExtension,
    resolve_execute_agent_extension,
    validate_execute_agent_extension_config,
)

_NOOP_TARGET = "nemo_agents_plugin.jobs.execute_extensions:NoopExecuteAgentExtension"
_OTHER_TARGET = "nemo_agents_plugin.jobs.execute_extensions:ExecuteAgentAfterInvokeContext"


def _entry_point(name: str, value: str) -> EntryPoint:
    return EntryPoint(name=name, value=value, group=EXECUTE_AGENT_EXTENSION_ENTRY_POINT_GROUP)


def _install(monkeypatch: pytest.MonkeyPatch, *entries: EntryPoint) -> None:
    monkeypatch.setattr(execute_extensions, "entry_points", lambda group: list(entries))


def test_the_noop_kind_needs_no_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch)

    assert resolve_execute_agent_extension(NOOP_EXECUTE_AGENT_EXTENSION_KIND) is NoopExecuteAgentExtension


def test_a_single_registration_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _entry_point("demo.extension", _NOOP_TARGET))

    assert resolve_execute_agent_extension("demo.extension") is NoopExecuteAgentExtension


def test_the_same_extension_declared_twice_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    """The aggregate ``nemo-helix`` wheel re-declares every bundled plugin's
    entry points, so a standard install sees each kind twice — once from the
    plugin distribution, once from the aggregate. Two declarations of one
    implementation are not a conflict.
    """
    _install(
        monkeypatch,
        _entry_point("demo.extension", _NOOP_TARGET),
        _entry_point("demo.extension", _NOOP_TARGET),
    )

    assert resolve_execute_agent_extension("demo.extension") is NoopExecuteAgentExtension


def test_two_different_extensions_claiming_one_kind_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Genuine ambiguity still fails: nothing can decide which one was meant."""
    _install(
        monkeypatch,
        _entry_point("demo.extension", _NOOP_TARGET),
        _entry_point("demo.extension", _OTHER_TARGET),
    )

    with pytest.raises(ValueError, match="Conflicting agents.execute extensions"):
        resolve_execute_agent_extension("demo.extension")


def test_an_unregistered_kind_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _entry_point("demo.extension", _NOOP_TARGET))

    with pytest.raises(ValueError, match="Unknown agents.execute extension"):
        resolve_execute_agent_extension("missing.extension")


def test_the_installed_insights_extension_resolves() -> None:
    """Guards the real environment: insights.analysis is declared twice on disk."""
    resolved = resolve_execute_agent_extension("insights.analysis")

    assert resolved.__name__ == "InsightsAnalysisExtension"


def test_a_valid_extension_config_is_accepted() -> None:
    validate_execute_agent_extension_config("insights.analysis", {"agent": "demo-agent", "workspace": "default"})


def test_an_unknown_extension_config_key_is_rejected() -> None:
    """``extra="forbid"`` catches a renamed or typo'd key on the create request."""
    with pytest.raises(ValueError) as excinfo:
        validate_execute_agent_extension_config("insights.analysis", {"agent": "demo", "workspac": "default"})

    message = str(excinfo.value)
    assert "insights.analysis" in message, "the caller supplied a kind, so the error should name it"
    assert "workspac" in message


def test_a_wrongly_typed_extension_config_value_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_execute_agent_extension_config("insights.analysis", {"local_only": "yes-please"})


def test_the_noop_extension_accepts_an_empty_config() -> None:
    validate_execute_agent_extension_config(NOOP_EXECUTE_AGENT_EXTENSION_KIND, {})


def test_the_noop_extension_rejects_a_non_empty_config() -> None:
    """Handing config to an extension that ignores it is a mistake worth surfacing."""
    with pytest.raises(ValueError):
        validate_execute_agent_extension_config(NOOP_EXECUTE_AGENT_EXTENSION_KIND, {"agent": "demo-agent"})


def test_an_extension_declaring_no_config_model_is_left_unvalidated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lenient by design: an out-of-tree extension predating the contract still creates."""
    _install(monkeypatch, _entry_point("legacy.extension", _OTHER_TARGET))

    validate_execute_agent_extension_config("legacy.extension", {"anything": "goes"})


def test_validating_an_unregistered_kind_is_still_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch)

    with pytest.raises(ValueError, match="Unknown agents.execute extension"):
        validate_execute_agent_extension_config("nope.extension", {})
