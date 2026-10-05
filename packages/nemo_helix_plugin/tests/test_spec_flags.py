# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared spec → CLI-flag plumbing.

The module under test is consumed by both the function CLI (this PR)
and — once PR #160 lands — the job CLI. These tests pin the
walker, the overlay/merge contract, and the synthetic-signature
helpers in isolation, so the verb-specific tests in
``test_commands.py`` only need to assert the wiring.
"""

from __future__ import annotations

import inspect
from typing import Optional

import pytest
import typer
from nemo_helix_plugin._spec_flags import (
    UNSET,
    SpecLeafField,
    build_callback_signature,
    build_epilog,
    build_overlay,
    deep_merge,
    kw,
    make_field_param,
    walk_spec_leaves,
)
from pydantic import BaseModel, Field


class _Inner(BaseModel):
    url: str
    timeout_seconds: int = 30


class _Spec(BaseModel):
    name: str
    count: int = 1
    inner: _Inner


class TestWalkSpecLeaves:
    def test_returns_empty_list_for_none_schema(self) -> None:
        assert walk_spec_leaves(None) == []

    def test_emits_one_leaf_per_scalar_field(self) -> None:
        leaves = walk_spec_leaves(_Spec)
        flags = {leaf.flag for leaf in leaves}
        assert flags == {"--name", "--count", "--inner.url", "--inner.timeout-seconds"}

    def test_required_field_has_no_default(self) -> None:
        leaves = {leaf.flag: leaf for leaf in walk_spec_leaves(_Spec)}
        assert leaves["--name"].required is True
        assert leaves["--count"].required is False
        assert leaves["--count"].default == 1

    def test_reserved_param_names_are_skipped(self) -> None:
        # ``count`` is a sensible static-flag name (e.g. for verbs that
        # take their own ``--count``). Reserving it must drop the spec
        # version from the auto-generated set.
        leaves = walk_spec_leaves(_Spec, reserved={"count"})
        flags = {leaf.flag for leaf in leaves}
        assert "--count" not in flags
        assert "--name" in flags

    def test_unsupported_leaf_type_is_skipped(self) -> None:
        # ``list[str]`` is silently dropped — users still pass it via
        # ``--spec`` / ``--spec-file``. The decision is logged at DEBUG
        # so anyone debugging "where's my flag?" can see why.
        class _WithList(BaseModel):
            tags: list[str] = []
            name: str = "x"

        leaves = walk_spec_leaves(_WithList)
        flags = {leaf.flag for leaf in leaves}
        assert flags == {"--name"}

    def test_optional_scalar_unwraps(self) -> None:
        class _OptStr(BaseModel):
            label: Optional[str] = None

        leaves = walk_spec_leaves(_OptStr)
        assert len(leaves) == 1
        assert leaves[0].python_type is str

    def test_metavar_from_field_extra_wins(self) -> None:
        # A per-field override beats every annotation-derived heuristic
        # so a one-off rename doesn't require a type-level marker.
        class _Custom(BaseModel):
            name: str = Field("x", json_schema_extra={"cli_metavar": "USER_NAME"})

        leaf = walk_spec_leaves(_Custom)[0]
        assert leaf.metavar == "USER_NAME"

    def test_mixed_scalar_model_union_keeps_scalar_arm_flag(self) -> None:
        # ``str | SomeModel`` used to be dropped entirely (no shared
        # scalar base across arms). The scalar arm should still surface
        # a flag; the model arm stays reachable via --spec/--spec-file.
        class _Inline(BaseModel):
            value: str

        class _Mixed(BaseModel):
            agent: str | _Inline

        leaves = {leaf.flag: leaf for leaf in walk_spec_leaves(_Mixed)}
        assert leaves["--agent"].python_type is str
        assert leaves["--agent"].partial is True

    def test_mixed_scalar_model_union_reports_unavailable(self) -> None:
        class _Inline(BaseModel):
            value: str

        class _Mixed(BaseModel):
            agent: str | _Inline

        unavailable: list[str] = []
        walk_spec_leaves(_Mixed, unavailable=unavailable)
        assert unavailable == ["agent (other union forms)"]

    def test_two_scalar_arms_plus_model_arm_is_skipped(self) -> None:
        # Ambiguous which scalar flag type to expose, so this remains
        # skipped entirely, same as before.
        class _Inline(BaseModel):
            value: str

        class _Ambiguous(BaseModel):
            field: str | int | _Inline

        leaves = walk_spec_leaves(_Ambiguous)
        assert leaves == []

    def test_fully_unsupported_field_reports_unavailable(self) -> None:
        class _WithList(BaseModel):
            tags: list[str] = []
            name: str = "x"

        unavailable: list[str] = []
        walk_spec_leaves(_WithList, unavailable=unavailable)
        assert unavailable == ["tags"]

    def test_nested_path_to_param_name_is_unique(self) -> None:
        # Two leaves at different paths must not collide on the
        # synthetic param name — the walker disambiguates with a
        # numeric suffix.
        class _A(BaseModel):
            x: str

        class _B(BaseModel):
            x: str

        class _Outer(BaseModel):
            a__x: str = "outer"
            a: _A
            b: _B

        leaves = walk_spec_leaves(_Outer)
        names = [leaf.param_name for leaf in leaves]
        assert len(names) == len(set(names))


class TestOverlayAndMerge:
    def test_unset_values_are_dropped(self) -> None:
        leaves = walk_spec_leaves(_Spec)
        overlay = build_overlay(leaves, {leaf.param_name: UNSET for leaf in leaves})
        assert overlay == {}

    def test_set_values_become_nested_dict(self) -> None:
        leaves = walk_spec_leaves(_Spec)
        by_path = {leaf.path: leaf for leaf in leaves}
        raw = {
            by_path[("name",)].param_name: "Razvan",
            by_path[("inner", "url")].param_name: "https://example.test",
        }
        overlay = build_overlay(leaves, raw)
        assert overlay == {"name": "Razvan", "inner": {"url": "https://example.test"}}

    def test_deep_merge_overlay_wins_at_leaves(self) -> None:
        base = {"name": "from-base", "inner": {"url": "base", "timeout_seconds": 1}}
        overlay = {"inner": {"url": "overlay"}}
        merged = deep_merge(base, overlay)
        assert merged == {"name": "from-base", "inner": {"url": "overlay", "timeout_seconds": 1}}

    def test_deep_merge_does_not_mutate_inputs(self) -> None:
        base = {"a": {"b": 1}}
        overlay = {"a": {"c": 2}}
        merged = deep_merge(base, overlay)
        assert base == {"a": {"b": 1}}
        assert overlay == {"a": {"c": 2}}
        assert merged == {"a": {"b": 1, "c": 2}}


class TestSyntheticSignatureHelpers:
    def test_make_field_param_emits_typer_option_with_flag_name(self) -> None:
        leaf = SpecLeafField(
            path=("name",),
            param_name="name",
            python_type=str,
            default="default-name",
            description="The name.",
            required=False,
        )
        param = make_field_param(leaf, rich_help_panel="Function Spec")
        # Typer reads its option metadata off the parameter default;
        # asserting the flag name keeps the kebab-case rule pinned.
        assert isinstance(param.default, typer.models.OptionInfo)
        assert "--name" in param.default.param_decls

    def test_build_callback_signature_orders_static_then_leaves(self) -> None:
        # Help-output legibility depends on this ordering — the user
        # sees "static flags they always type" before the schema-derived
        # block, not interleaved.
        leaves = walk_spec_leaves(_Spec)
        static = [
            kw("spec", str, typer.Option("{}", "--spec")),
            kw("spec_file", Optional[str], typer.Option(None, "--spec-file")),
        ]
        sig = build_callback_signature(static, leaves, rich_help_panel="Function Spec")
        names = list(sig.parameters)
        assert names[:2] == ["spec", "spec_file"]
        # And every leaf is represented as a keyword-only param.
        for leaf in leaves:
            assert sig.parameters[leaf.param_name].kind is inspect.Parameter.KEYWORD_ONLY


class TestEpilog:
    def test_includes_schema_name_when_flags_present(self) -> None:
        leaves = walk_spec_leaves(_Spec)
        text = build_epilog(schema=_Spec, leaves=leaves, kind="Function")
        assert "_Spec" in text
        assert "Precedence" in text

    def test_falls_back_to_no_flags_message(self) -> None:
        text = build_epilog(schema=_Spec, leaves=[], kind="Function")
        assert "no per-field flags" in text

    def test_unavailable_paths_render_as_extra_note(self) -> None:
        leaves = walk_spec_leaves(_Spec)
        text = build_epilog(schema=_Spec, leaves=leaves, kind="Function", unavailable=["agent (other union forms)"])
        assert "agent (other union forms)" in text
        assert "--spec or --spec-file" in text

    def test_no_unavailable_paths_omits_note(self) -> None:
        leaves = walk_spec_leaves(_Spec)
        text = build_epilog(schema=_Spec, leaves=leaves, kind="Function")
        assert "cannot be represented as CLI flags" not in text

    @pytest.mark.parametrize("kind", ["Function", "Job"])
    def test_kind_is_capitalised_label(self, kind: str) -> None:
        leaves = walk_spec_leaves(_Spec)
        text = build_epilog(schema=_Spec, leaves=leaves, kind=kind)
        assert f"{kind} Spec flags" in text
