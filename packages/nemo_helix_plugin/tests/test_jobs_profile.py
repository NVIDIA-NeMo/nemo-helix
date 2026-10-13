# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`nemo_helix_plugin.jobs.profile.stamp_profile`.

The stamper is pure logic — tests use both plain-dict step shapes and
attribute-style shapes (dataclasses) to confirm it works against both
the wire-style TypedDicts produced by the SDK and any Pydantic model-based
executor type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from nemo_helix_plugin.jobs.profile import stamp_profile

# ---------------------------------------------------------------------------
# Fixture shapes — spec-like dataclasses for attribute-style testing
# ---------------------------------------------------------------------------


@dataclass
class _Executor:
    profile: str | None = None


@dataclass
class _Step:
    executor: _Executor = field(default_factory=_Executor)


@dataclass
class _Spec:
    steps: list[_Step] = field(default_factory=list)


def _make_attr_spec(profiles: list[str | None]) -> _Spec:
    return _Spec(steps=[_Step(_Executor(profile=p)) for p in profiles])


def _make_dict_spec(profiles: list[str | None]) -> dict[str, Any]:
    return {"steps": [{"executor": {"profile": p}} for p in profiles]}


# ---------------------------------------------------------------------------
# Attribute-style (dataclass / Pydantic model) steps
# ---------------------------------------------------------------------------


class TestAttrStyle:
    def test_stamps_none_profile(self) -> None:
        spec = _make_attr_spec([None])
        stamp_profile(spec, "research")
        assert spec.steps[0].executor.profile == "research"

    def test_stamps_empty_string_profile(self) -> None:
        spec = _make_attr_spec([""])
        stamp_profile(spec, "research")
        assert spec.steps[0].executor.profile == "research"

    def test_leaves_explicit_per_step_profile_alone(self) -> None:
        spec = _make_attr_spec(["cleanup", None])
        stamp_profile(spec, "research")
        assert spec.steps[0].executor.profile == "cleanup"
        assert spec.steps[1].executor.profile == "research"

    def test_returns_spec_for_chaining(self) -> None:
        spec = _make_attr_spec([None])
        result = stamp_profile(spec, "default")
        assert result is spec

    def test_mixed_steps(self) -> None:
        spec = _make_attr_spec([None, "per-step-override", "", None])
        stamp_profile(spec, "job-level")
        assert [s.executor.profile for s in spec.steps] == [
            "job-level",
            "per-step-override",
            "job-level",
            "job-level",
        ]


# ---------------------------------------------------------------------------
# Dict-style (wire TypedDict) steps
# ---------------------------------------------------------------------------


class TestDictStyle:
    def test_stamps_none_profile(self) -> None:
        spec = _make_dict_spec([None])
        stamp_profile(spec, "research")
        assert spec["steps"][0]["executor"]["profile"] == "research"

    def test_leaves_per_step_override_alone(self) -> None:
        spec = _make_dict_spec(["cleanup", None])
        stamp_profile(spec, "research")
        assert spec["steps"][0]["executor"]["profile"] == "cleanup"
        assert spec["steps"][1]["executor"]["profile"] == "research"

    def test_missing_profile_key_is_stamped(self) -> None:
        # Dict without the key at all — treated as unset.
        spec = {"steps": [{"executor": {}}]}
        stamp_profile(spec, "research")
        assert spec["steps"][0]["executor"]["profile"] == "research"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_missing_steps_raises(self) -> None:
        with pytest.raises(TypeError, match="no 'steps' attribute"):
            stamp_profile(object(), "p")

    def test_step_without_executor_raises(self) -> None:
        spec = {"steps": [{"name": "no-executor"}]}
        with pytest.raises(TypeError, match="no 'executor' field"):
            stamp_profile(spec, "p")
