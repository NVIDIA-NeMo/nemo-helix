# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`nemo_guardrails_plugin.llmrails_cache`.

Mirrors the source layout: one section per pipeline stage so a regression in
one stage doesn't masquerade as a regression in another.

- §1 :class:`GuardrailConfigSource` — discriminated union and label helpers.
- §2 :func:`stabilize` and :class:`StabilizedRailsConfigCache` — build-input
  transform and its memoization layer; pins the cache-identity determinism
  contract.
- §3 :class:`Pool` — single-key resource pool: build-lock serialization,
  demand-watermark shrink, exclusive-ownership cancellation discard.
- §4 :class:`LLMRailsCache` — outer LRU + pinning + warming + lifecycle.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from nemo_guardrails_plugin.llmrails_cache import (
    MAIN_MODEL_REQUEST_PLACEHOLDER,
    PROVENANCE_HISTORY_LEN,
    EntityGuardrailConfigSource,
    InlineGuardrailConfigSource,
    LLMRailsBuilder,
    LLMRailsCache,
    Pool,
    Provenance,
    StabilizedRailsConfigCache,
    StableRailsConfig,
    _fill_main_model_placeholder,
    _is_missing_model_name_error,
    _resolve_model_target,
    extract_output_rails_streaming_config,
    provenance_of,
    source_has_input_flows,
    source_has_output_flows,
    stabilize,
)
from nemo_platform_plugin.guardrail.types import RailsConfig as PlatformRailsConfig
from nemo_platform_plugin.inference_middleware import OpenAICompatibleInferenceTarget
from nemoguardrails import RailsConfig
from nemoguardrails.rails.llm.config import Model

# =============================================================================
# §1: GuardrailConfigSource — discriminated union + helpers
# =============================================================================


def _rails_dict(rails: dict[str, Any] | None = None) -> PlatformRailsConfig:
    """Bare :class:`PlatformRailsConfig` from a ``rails:`` block, with no models."""
    return PlatformRailsConfig.model_validate({"rails": rails} if rails else {})


def _entity(
    rails: PlatformRailsConfig, *, name: str = "guard", updated_at: str = "2026-01-01T00:00:00Z"
) -> EntityGuardrailConfigSource:
    return EntityGuardrailConfigSource(workspace="ws", name=name, updated_at=updated_at, rails=rails)


class TestProvenanceOf:
    def test_entity_source_label_is_workspace_slash_name_at_updated_at(self) -> None:
        prov = provenance_of(_entity(_rails_dict(), name="guard-A", updated_at="2026-04-30T00:00:00Z"))
        assert prov == Provenance("ws/guard-A@2026-04-30T00:00:00Z")

    def test_inline_source_with_label(self) -> None:
        prov = provenance_of(InlineGuardrailConfigSource(rails=_rails_dict(), label="my-test"))
        assert prov == Provenance("<inline:my-test>")

    def test_inline_source_without_label_falls_back_to_unnamed(self) -> None:
        prov = provenance_of(InlineGuardrailConfigSource(rails=_rails_dict(), label=None))
        assert prov == Provenance("<inline:unnamed>")


class TestSourceFlowPredicates:
    def test_input_flows_true_for_either_arm(self) -> None:
        rails = _rails_dict({"input": {"flows": ["self check input"]}})
        assert source_has_input_flows(_entity(rails)) is True
        assert source_has_input_flows(InlineGuardrailConfigSource(rails=rails)) is True

    def test_output_flows_true_for_either_arm(self) -> None:
        rails = _rails_dict({"output": {"flows": ["self check output"]}})
        assert source_has_output_flows(_entity(rails)) is True
        assert source_has_output_flows(InlineGuardrailConfigSource(rails=rails)) is True

    @pytest.mark.parametrize(
        "rails_dict",
        [
            None,
            {"input": {"flows": []}},
            {"output": {"flows": []}},
        ],
    )
    def test_flows_false_when_absent_or_empty(self, rails_dict: dict[str, Any] | None) -> None:
        source = _entity(_rails_dict(rails_dict))
        assert source_has_input_flows(source) is False
        assert source_has_output_flows(source) is False


class TestExtractOutputRailsStreamingConfig:
    def test_returns_none_when_output_section_absent(self) -> None:
        assert extract_output_rails_streaming_config(_entity(_rails_dict())) is None

    def test_returns_none_when_output_present_but_streaming_absent(self) -> None:
        """Output section with flows but no streaming key: callers treat
        streaming as off — returning a default ``StreamingConfig`` with
        ``enabled=True`` here would silently opt-in every non-streaming
        config to streaming output rails."""
        rails = _rails_dict({"output": {"flows": ["self check output"]}})
        assert extract_output_rails_streaming_config(_entity(rails)) is None

    def test_returns_streaming_config_when_present(self) -> None:
        rails = _rails_dict({"output": {"flows": ["self check output"], "streaming": {"enabled": False}}})
        cfg = extract_output_rails_streaming_config(_entity(rails))
        assert cfg is not None
        assert cfg.enabled is False


class TestSourceEquality:
    """Sources are frozen dataclasses, so equality is structural across all
    fields. The warming dedup in ``on_virtual_model_upserted`` keys on the
    identity triple ``(workspace, name, updated_at)`` — not on source ``==``
    — precisely because :class:`PlatformRailsConfig` is unhashable, which
    makes the source itself unhashable too. These tests pin down the
    structural-equality contract; hashability is intentionally untested.
    """

    def test_entity_source_equality(self) -> None:
        rails = _rails_dict()
        a = EntityGuardrailConfigSource(workspace="ws", name="guard", updated_at="t", rails=rails)
        b = EntityGuardrailConfigSource(workspace="ws", name="guard", updated_at="t", rails=rails)
        assert a == b

    @pytest.mark.parametrize(
        "workspace,name,updated_at",
        [
            ("ws2", "guard", "t"),
            ("ws", "guard2", "t"),
            ("ws", "guard", "t2"),
        ],
        ids=["workspace", "name", "updated_at"],
    )
    def test_entity_source_inequality_by_identity_field(self, workspace: str, name: str, updated_at: str) -> None:
        rails = _rails_dict()
        base = EntityGuardrailConfigSource(workspace="ws", name="guard", updated_at="t", rails=rails)
        other = EntityGuardrailConfigSource(workspace=workspace, name=name, updated_at=updated_at, rails=rails)
        assert base != other

    def test_inline_source_equality(self) -> None:
        rails = _rails_dict()
        a = InlineGuardrailConfigSource(rails=rails, label="x")
        b = InlineGuardrailConfigSource(rails=rails, label="x")
        assert a == b

    def test_inline_source_inequality_by_label(self) -> None:
        rails = _rails_dict()
        a = InlineGuardrailConfigSource(rails=rails, label="x")
        b = InlineGuardrailConfigSource(rails=rails, label="y")
        assert a != b


# =============================================================================
# §2: stabilize() and StabilizedRailsConfigCache
# =============================================================================


def _resolve_target(_model_id: str) -> OpenAICompatibleInferenceTarget:
    return OpenAICompatibleInferenceTarget(
        openai_base_url="http://igw.example/provider/v1",
        model="meta/llama-3.1-8b-instruct",
    )


def _other_resolver(_model_id: str) -> OpenAICompatibleInferenceTarget:
    return OpenAICompatibleInferenceTarget(
        openai_base_url="http://other.example/v1",
        model="other-model",
    )


def _platform_rails(
    *,
    rails_block: dict[str, Any] | None = None,
    models: list[dict[str, Any]] | None = None,
) -> PlatformRailsConfig:
    """Build a minimal :class:`PlatformRailsConfig`. ``models`` defaults to ``[]``
    so the library-side validator (which requires a ``models`` field) is happy.
    """
    payload: dict[str, Any] = {
        "rails": rails_block if rails_block is not None else {"input": {"flows": []}, "output": {"flows": []}},
        "models": models if models is not None else [],
    }
    return PlatformRailsConfig.model_validate(payload)


class TestStabilizeStripsMain:
    def test_main_model_dropped(self) -> None:
        stable = stabilize(
            _platform_rails(
                models=[
                    {"type": "main", "engine": "nim", "model": "ws/llama"},
                    {"type": "content_safety", "engine": "nim", "model": "default/safety"},
                ]
            ),
            _resolve_target,
        )
        types = {m.type for m in stable.rails.models}
        assert "main" not in types
        assert "content_safety" in types

    def test_resolves_base_url_when_unset(self) -> None:
        stable = stabilize(
            _platform_rails(models=[{"type": "content_safety", "engine": "nim", "model": "default/safety"}]),
            _resolve_target,
        )
        action = next(m for m in stable.rails.models if m.type == "content_safety")
        assert action.parameters["base_url"] == "http://igw.example/provider/v1"

    def test_preserves_explicit_base_url(self) -> None:
        stable = stabilize(
            _platform_rails(
                models=[
                    {
                        "type": "content_safety",
                        "engine": "nim",
                        "model": "default/safety",
                        "parameters": {"base_url": "http://override.example/v1"},
                    }
                ]
            ),
            _resolve_target,
        )
        action = next(m for m in stable.rails.models if m.type == "content_safety")
        assert action.parameters["base_url"] == "http://override.example/v1"

    def test_default_headers_defensively_copied(self) -> None:
        """A later mutation of the source dict's headers must not leak into
        the stable rails (the cache reuses the same instance across requests)."""
        source_headers = {"X-Static": "yes"}
        stable = stabilize(
            _platform_rails(
                models=[
                    {
                        "type": "content_safety",
                        "engine": "nim",
                        "model": "default/safety",
                        "parameters": {"default_headers": source_headers},
                    }
                ]
            ),
            _resolve_target,
        )
        action = next(m for m in stable.rails.models if m.type == "content_safety")
        source_headers["X-Mutated"] = "yes"
        assert "X-Mutated" not in action.parameters["default_headers"]


class TestResolveModelTarget:
    """Private-helper coverage for defense-in-depth branches that
    :func:`stabilize`'s upstream validators make unreachable in production.
    Kept because a future upstream relaxation must fail here, not silently
    pass an empty model ID into the resolver.
    """

    def test_raises_when_model_field_is_empty(self) -> None:
        """The upstream ``model_must_be_none_empty`` validator rejects empty
        model IDs before :func:`_resolve_model_target` runs in production.
        This test bypasses that validator via ``model_construct`` to
        exercise the defense-in-depth branch."""
        model = Model.model_construct(type="content_safety", engine="nim", model="", parameters={})
        with pytest.raises(ValueError, match="has no ``model`` field"):
            _resolve_model_target(model, _resolve_target)


class TestStabilizeHashDeterminism:
    """Hash determinism is the new design's correctness lifeline.

    Two sources that produce byte-identical build inputs must hash to the
    same content_hash. Two that differ in any meaningful way must hash to
    different content_hashes. Cosmetic differences (key order in input,
    explicit ``None`` for optional fields) collapse to the same hash.
    """

    def test_hash_is_deterministic(self) -> None:
        rails = _platform_rails(models=[{"type": "content_safety", "engine": "nim", "model": "default/safety"}])
        first = stabilize(rails, _resolve_target)
        second = stabilize(rails, _resolve_target)
        assert first.content_hash == second.content_hash

    def test_hash_collapses_explicit_none_to_absent(self) -> None:
        """``exclude_none=True`` means ``foo: None`` and "field absent" hash
        the same — matches the user-intent that an unspecified optional is
        the same as no constraint at all."""
        without_none = PlatformRailsConfig.model_validate({"rails": {"input": {"flows": []}}, "models": []})
        with_none = PlatformRailsConfig.model_validate(
            {
                "rails": {"input": {"flows": []}},
                "models": [],
                "actions_server_url": None,
            }
        )
        assert (
            stabilize(without_none, _resolve_target).content_hash == stabilize(with_none, _resolve_target).content_hash
        )

    def test_hash_changes_for_meaningful_change(self) -> None:
        with_input = stabilize(
            _platform_rails(rails_block={"input": {"flows": ["custom check"]}}),
            _resolve_target,
        )
        without_input = stabilize(_platform_rails(rails_block={"input": {"flows": []}}), _resolve_target)
        assert with_input.content_hash != without_input.content_hash

    def test_hash_changes_when_resolver_returns_different_url(self) -> None:
        """Provider migrations during the plugin's lifetime cause new
        entity warms to compute new hashes (the URL written into
        ``Model.parameters`` changes); the old pool decays under LRU."""
        rails = _platform_rails(models=[{"type": "content_safety", "engine": "nim", "model": "default/safety"}])
        a = stabilize(rails, _resolve_target)
        b = stabilize(rails, _other_resolver)
        assert a.content_hash != b.content_hash

    def test_hash_invariant_to_model_declaration_order(self) -> None:
        """Two configs with the same models in different order share a hash.

        LLMRails resolves models by ``type`` (not by index), so source-side
        declaration order is semantically irrelevant. ``stabilize`` sorts
        the stripped list before hashing so a cosmetic edit (swap the
        order of two ``content_safety`` entries) doesn't fork the cache.
        Without that sort, ``json.dumps(..., sort_keys=True)`` would only
        sort dict keys and the lists would canonicalize differently.
        """
        first_order = _platform_rails(
            models=[
                {"type": "content_safety", "engine": "nim", "model": "ws/safety-a"},
                {"type": "topic_safety", "engine": "nim", "model": "ws/topic-b"},
                {"type": "embeddings", "engine": "nim", "model": "ws/text-embed"},
            ]
        )
        second_order = _platform_rails(
            models=[
                {"type": "embeddings", "engine": "nim", "model": "ws/text-embed"},
                {"type": "content_safety", "engine": "nim", "model": "ws/safety-a"},
                {"type": "topic_safety", "engine": "nim", "model": "ws/topic-b"},
            ]
        )
        assert (
            stabilize(first_order, _resolve_target).content_hash
            == stabilize(second_order, _resolve_target).content_hash
        )

    def test_cross_source_determinism(self) -> None:
        """The new design's structural guarantee: an entity source and an
        inline source whose ``rails`` are byte-identical hash to the same
        content_hash. The pool selection becomes by content, not by entity
        identity."""
        rails = _platform_rails(rails_block={"input": {"flows": ["custom check"]}})
        # Same platform rails passed in — independent of how the source was framed.
        entity_source = EntityGuardrailConfigSource(workspace="ws", name="g", updated_at="t", rails=rails)
        inline_source = InlineGuardrailConfigSource(rails=rails, label="my-test")
        from_entity = stabilize(entity_source.rails, _resolve_target)
        from_inline = stabilize(inline_source.rails, _resolve_target)
        assert from_entity.content_hash == from_inline.content_hash

    def test_hash_invariant_to_order_when_models_share_type_engine_model(self) -> None:
        """Sort-key total-ordering: two models that share ``(type, engine,
        model)`` but differ in ``parameters`` must canonicalize to the same
        order regardless of declaration order.

        Closes the gap left by sorting on ``(type, engine, model)``: that
        tuple isn't a total order over ``Model`` (two models can compare
        equal under it but differ in ``parameters``), and Python's stable
        sort would let declaration order survive into the hash. The current
        implementation sorts by full canonical model JSON, which is a total
        order.

        Whether two models of the same type are semantically meaningful
        upstream is a separate question — this test only locks in that
        declaration order doesn't fork the cache when they appear.
        """
        first_order = _platform_rails(
            models=[
                {
                    "type": "content_safety",
                    "engine": "nim",
                    "model": "default/safety",
                    "parameters": {"temperature": 0.0},
                },
                {
                    "type": "content_safety",
                    "engine": "nim",
                    "model": "default/safety",
                    "parameters": {"temperature": 0.5},
                },
            ]
        )
        second_order = _platform_rails(
            models=[
                {
                    "type": "content_safety",
                    "engine": "nim",
                    "model": "default/safety",
                    "parameters": {"temperature": 0.5},
                },
                {
                    "type": "content_safety",
                    "engine": "nim",
                    "model": "default/safety",
                    "parameters": {"temperature": 0.0},
                },
            ]
        )
        assert (
            stabilize(first_order, _resolve_target).content_hash
            == stabilize(second_order, _resolve_target).content_hash
        )


class TestStabilizeMainModelTemplate:
    """Pre-validate the ``main`` model entry once during stabilize so the
    per-request hot path doesn't pay a Pydantic validation cost.

    ``main_model_template`` is the new design's surface for
    :func:`build_main_llm` to read static engine / parameters /
    api_key_env_var defaults — keeping it on
    :class:`StableRailsConfig` means the validation is bounded by
    stabilize cache misses (i.e. one validation per unique entity
    revision), not per request."""

    def test_template_is_none_when_no_main_entry(self) -> None:
        stable = stabilize(
            _platform_rails(models=[{"type": "content_safety", "engine": "nim", "model": "default/safety"}]),
            _resolve_target,
        )
        assert stable.main_model_template is None

    def test_template_carries_engine_parameters_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The library RailsConfig validator requires the env var to be set;
        # this isn't part of what we're testing, just a precondition for
        # validation to pass.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-stub")
        stable = stabilize(
            _platform_rails(
                models=[
                    {
                        "type": "main",
                        "engine": "openai",
                        "model": "ws/llama",
                        "api_key_env_var": "OPENAI_API_KEY",
                        "parameters": {"temperature": 0.1},
                    }
                ]
            ),
            _resolve_target,
        )
        assert stable.main_model_template is not None
        assert stable.main_model_template.engine == "openai"
        assert stable.main_model_template.api_key_env_var == "OPENAI_API_KEY"
        assert stable.main_model_template.parameters == {"temperature": 0.1}

    def test_template_extraction_does_not_change_content_hash(self) -> None:
        """Adding template extraction must be observationally invisible to
        cache identity: two stabilize calls of the same source land on the
        same content_hash regardless of whether the main entry is present.
        Otherwise, the entity-vs-inline cross-source determinism guarantee
        breaks — the redesign hinges on it."""
        rails_with_main = _platform_rails(
            models=[
                {"type": "main", "engine": "nim", "model": "ws/llama"},
                {"type": "content_safety", "engine": "nim", "model": "default/safety"},
            ]
        )
        rails_without_main = _platform_rails(
            models=[
                {"type": "content_safety", "engine": "nim", "model": "default/safety"},
            ]
        )
        a = stabilize(rails_with_main, _resolve_target)
        b = stabilize(rails_without_main, _resolve_target)
        assert a.content_hash == b.content_hash

    def test_rejects_duplicate_main_entries(self) -> None:
        """A config with multiple ``main`` entries must raise.

        Stripping all of them before hashing would silently collide
        duplicate-main variants on the same cache key while changing
        which model is actually used. Caller-shape error → ValueError →
        4xx via ``_run_rails``.
        """
        rails = _platform_rails(
            models=[
                {"type": "main", "engine": "nim", "model": "ws/llama"},
                {"type": "main", "engine": "nim", "model": "ws/qwen"},
            ]
        )
        with pytest.raises(ValueError, match="at most one 'main' model"):
            stabilize(rails, _resolve_target)


class TestStabilizeNamelessMainTemplate:
    """A ``main`` entry with no ``model`` name is still kept as
    ``main_model_template`` after stabilization, not dropped."""

    def test_nameless_main_with_base_url_is_kept_as_template(self) -> None:
        """The regression case: a ``main`` entry with no ``model`` but a
        populated ``parameters`` must survive as ``main_model_template`` instead
        of being dropped (which would silently discard the ``base_url``)."""
        rails = _platform_rails(
            models=[
                {"type": "main", "engine": "nim", "parameters": {"base_url": "https://rails.example.com"}},
                {"type": "content_safety", "engine": "nim", "model": "default/safety"},
            ]
        )
        stable = stabilize(rails, _resolve_target)
        assert stable.main_model_template is not None
        assert stable.main_model_template.engine == "nim"
        assert stable.main_model_template.parameters["base_url"] == "https://rails.example.com"
        assert {m.type for m in stable.rails.models} == {"content_safety"}

    def test_nameless_main_with_no_content_is_still_kept_as_template(self) -> None:
        """Even a fully empty ``{"type": "main", "engine": "nim"}`` becomes
        a template (not ``None``) — harmless, since :func:`rails.build_main_llm`
        falls back to the same defaults either way."""
        rails = _platform_rails(
            models=[
                {"type": "main", "engine": "nim"},
                {"type": "content_safety", "engine": "nim", "model": "default/safety"},
            ]
        )
        stable = stabilize(rails, _resolve_target)
        assert stable.main_model_template is not None
        assert stable.main_model_template.engine == "nim"
        assert {m.type for m in stable.rails.models} == {"content_safety"}

    @pytest.mark.parametrize(
        "model_value",
        [None, "", "   "],
        ids=["none", "empty_string", "whitespace_only"],
    )
    def test_nameless_main_wire_shapes_are_all_kept(self, model_value: str | None) -> None:
        """Pin the equivalence of all "missing model name" wire shapes:
        absent, explicit None, empty string, whitespace-only string. All
        four must get a placeholder filled in and stabilize without
        erroring."""
        rails = _platform_rails(
            models=[{"type": "main", "engine": "nim", "model": model_value}],
        )
        stable = stabilize(rails, _resolve_target)
        assert stable.main_model_template is not None
        assert stable.rails.models == []

    @pytest.mark.parametrize(
        "models",
        [
            [{"type": "main", "engine": "nim", "model": "ws/llama"}],
            [{"type": "main", "engine": "nim", "parameters": {"model": "ws/llama"}}],
            [{"type": "main", "engine": "nim", "parameters": {"model_name": "ws/llama"}}],
        ],
        ids=["top_level_model", "parameters_model", "parameters_model_name"],
    )
    def test_named_main_keeps_library_recognized_name(self, models: list[dict[str, Any]]) -> None:
        """A ``main`` entry already named via any library-honoured location
        (top-level ``model``, ``parameters.model``, or ``parameters.model_name``)
        must not get a placeholder filled in — the upstream validator rejects
        dual specification — and must keep that name after stabilize."""
        rails = _platform_rails(models=models)
        stable = stabilize(rails, _resolve_target)
        assert stable.main_model_template is not None
        assert stable.main_model_template.engine == "nim"
        assert stable.main_model_template.model == "ws/llama"

    def test_nameless_main_alongside_named_main_raises(self) -> None:
        """Two ``main`` entries — nameless or not — still trip the
        at-most-one-main invariant; filling in a placeholder doesn't change
        duplicate detection."""
        rails = _platform_rails(
            models=[
                {"type": "main", "engine": "nim"},
                {"type": "main", "engine": "openai", "model": "ws/llama"},
                {"type": "content_safety", "engine": "nim", "model": "default/safety"},
            ]
        )
        with pytest.raises(ValueError, match="at most one 'main' model"):
            stabilize(rails, _resolve_target)

    def test_nameless_main_never_affects_content_hash(self) -> None:
        """``main`` entries are pulled out of the hashed model list
        regardless of whether they name a model, so two configs that
        differ only in a nameless ``main`` entry's presence must share a
        cache key.
        """
        with_nameless_main = _platform_rails(
            models=[
                {"type": "main", "engine": "nim", "parameters": {"base_url": "https://rails.example.com"}},
                {"type": "content_safety", "engine": "nim", "model": "default/safety"},
            ]
        )
        without_main = _platform_rails(models=[{"type": "content_safety", "engine": "nim", "model": "default/safety"}])
        assert (
            stabilize(with_nameless_main, _resolve_target).content_hash
            == stabilize(without_main, _resolve_target).content_hash
        )

    def test_fill_placeholder_only_touches_nameless_main_entries(self) -> None:
        """:func:`_fill_main_model_placeholder` passes non-``main`` entries
        and already-named ``main`` entries through unchanged (including names
        under ``parameters.model`` / ``parameters.model_name``), and only fills
        in a model on genuinely nameless ``main`` entries."""
        content_safety = {"type": "content_safety", "engine": "nim"}
        named_main = {"type": "main", "engine": "nim", "model": "ws/llama"}
        named_via_parameters_model = {
            "type": "main",
            "engine": "nim",
            "parameters": {"model": "ws/llama", "base_url": "https://rails.example.com"},
        }
        named_via_parameters_model_name = {
            "type": "main",
            "engine": "nim",
            "parameters": {"model_name": "ws/llama"},
        }
        nameless_main = {"type": "main", "engine": "nim"}

        assert _fill_main_model_placeholder(content_safety) == content_safety
        assert _fill_main_model_placeholder(named_main) == named_main
        assert _fill_main_model_placeholder(named_via_parameters_model) == named_via_parameters_model
        assert _fill_main_model_placeholder(named_via_parameters_model_name) == named_via_parameters_model_name
        assert _fill_main_model_placeholder(nameless_main) == {
            **nameless_main,
            "model": MAIN_MODEL_REQUEST_PLACEHOLDER,
        }


class TestStabilizeMissingModelNameWrapping:
    """Defense-in-depth: if upstream validation reports a missing model
    name (any entry), wrap the cryptic ``Model name must be specified``
    message with an IGW-aware diagnostic so the 400 reaching the client
    tells the user what's actually wrong.

    ``main`` entries always get a placeholder filled in before validation
    (see :func:`_fill_main_model_placeholder`), so in practice this only
    fires for non-``main`` entries (content-safety, topic-control, etc.)
    that genuinely omit ``model``.
    """

    def test_recognizes_missing_model_name_error(self) -> None:
        """Hand-build a ``ValidationError``-like object whose error list
        carries the upstream missing-model-name message and verify the
        matcher returns True."""

        class _FakeError:
            def errors(self) -> list[dict[str, Any]]:
                return [{"msg": "Model name must be specified in exactly one place: ..."}]

        assert _is_missing_model_name_error(cast(Any, _FakeError())) is True

    def test_matches_regardless_of_loc(self) -> None:
        """The simplified matcher is intentionally agnostic to which
        entry triggered the error — the wrapper message is neutral
        between main and non-main cases. Pins that we no longer scope
        the wrap by ``loc``, so any entry's missing-name error gets
        the friendlier diagnostic."""

        class _FakeError:
            def errors(self) -> list[dict[str, Any]]:
                return [
                    {
                        "loc": ("models", 0),
                        "msg": "Model name must be specified...",
                    }
                ]

        assert _is_missing_model_name_error(cast(Any, _FakeError())) is True

    def test_does_not_match_unrelated_errors(self) -> None:
        """Errors with other messages must surface unchanged so callers
        see the precise upstream diagnostic instead of a misleading
        missing-name wrap."""

        class _FakeError:
            def errors(self) -> list[dict[str, Any]]:
                return [{"msg": "Input should be a valid string"}]

        assert _is_missing_model_name_error(cast(Any, _FakeError())) is False

    def test_does_not_match_empty_error_list(self) -> None:
        """A ``ValidationError`` carrying no errors must return False.
        Pins the ``any(...)`` short-circuit on the empty case."""

        class _FakeError:
            def errors(self) -> list[dict[str, Any]]:
                return []

        assert _is_missing_model_name_error(cast(Any, _FakeError())) is False


class TestStabilizeWithoutModelsField:
    """Pin the IGW Plugin architecture invariant: a platform config that
    omits ``models`` entirely must stabilize cleanly.

    The platform schema makes ``models`` optional because under the
    Plugin model the gateway owns main-LLM routing — a user running only
    task rails (content-safety, topic-control) shouldn't be forced to
    declare an empty ``models: []`` just to satisfy the upstream library
    schema. :func:`stabilize` papers over that mismatch at the
    platform→library boundary."""

    def test_models_field_absent_stabilizes_to_empty_list(self) -> None:
        """``PlatformRailsConfig`` with ``models=None`` (the wire-shape
        default when the field is omitted) must stabilize. Library
        :class:`LibraryRailsConfig` requires the field, so ``stabilize``
        coerces the missing key to ``[]`` before validating."""
        rails = PlatformRailsConfig.model_validate({"rails": {"input": {"flows": ["custom check"]}}})
        assert rails.models is None

        stable = stabilize(rails, _resolve_target)
        assert stable.rails.models == []
        assert stable.main_model_template is None
        assert stable.embedding_model_id is None

    def test_omitted_and_explicit_empty_models_share_content_hash(self) -> None:
        """The two ways to express "no models" must collapse to the same
        cache key — otherwise a single config could fork the LLMRails
        pool depending on whether the wire payload included the empty
        list explicitly."""
        omitted = PlatformRailsConfig.model_validate({"rails": {"input": {"flows": ["custom check"]}}})
        explicit = PlatformRailsConfig.model_validate({"rails": {"input": {"flows": ["custom check"]}}, "models": []})
        assert stabilize(omitted, _resolve_target).content_hash == stabilize(explicit, _resolve_target).content_hash

    def test_task_only_config_stabilizes(self) -> None:
        """Production-shaped config: only task LLMs, no main entry. IGW
        owns main routing, so the user's config has nothing to say about
        the main LLM. Uses a custom-named flow to avoid the library's
        built-in flow-template-binding validators (e.g. ``content safety
        check input`` requires a matching prompt template); those are
        orthogonal to what this test pins."""
        rails = PlatformRailsConfig.model_validate(
            {
                "rails": {"input": {"flows": ["custom check"]}},
                "models": [
                    {"type": "content_safety", "engine": "nim", "model": "default/safety"},
                ],
            }
        )
        stable = stabilize(rails, _resolve_target)
        assert stable.main_model_template is None
        assert {m.type for m in stable.rails.models} == {"content_safety"}


class TestStabilizeEmbeddingModelExtraction:
    def test_returns_none_when_no_embeddings(self) -> None:
        stable = stabilize(_platform_rails(models=[]), _resolve_target)
        assert stable.embedding_model_id is None

    def test_extracts_first_embedding_model(self) -> None:
        stable = stabilize(
            _platform_rails(
                models=[
                    {"type": "content_safety", "engine": "nim", "model": "default/safety"},
                    {"type": "embeddings", "engine": "nim", "model": "default/text-embed"},
                ]
            ),
            _resolve_target,
        )
        assert stable.embedding_model_id == "default/text-embed"


class TestStabilizedRailsConfigCacheConstruction:
    def test_rejects_zero_max_entries(self) -> None:
        with pytest.raises(ValueError, match="max_entries must be >= 1"):
            StabilizedRailsConfigCache(max_entries=0)

    def test_rejects_negative_max_entries(self) -> None:
        with pytest.raises(ValueError, match="max_entries must be >= 1"):
            StabilizedRailsConfigCache(max_entries=-1)


class TestStabilizedRailsConfigCacheEntityHits:
    def test_entity_source_memoized_across_calls(self) -> None:
        """The explicit replacement for the old lazy ``ConfigFactory``
        closure: same ``(workspace, name, updated_at)`` returns the same
        :class:`StableRailsConfig` instance across requests.

        Two distinct source objects with the same identity triple must
        share the cached entry — otherwise the cache would be
        identity-keyed, which the IGW polling path (which rebuilds source
        objects each poll) would never hit.
        """
        cache = StabilizedRailsConfigCache()
        rails = _platform_rails()
        first_source = EntityGuardrailConfigSource(workspace="ws", name="guard", updated_at="t1", rails=rails)
        second_source = EntityGuardrailConfigSource(workspace="ws", name="guard", updated_at="t1", rails=rails)

        first = cache.get_or_compute(first_source, _resolve_target)
        second = cache.get_or_compute(second_source, _resolve_target)
        assert first is second, "entity sources sharing (workspace, name, updated_at) must coalesce"

    def test_distinct_updated_at_misses(self) -> None:
        """An entity revision change misses; the new tuple computes a fresh
        :class:`StableRailsConfig`. The old tuple decays under LRU."""
        cache = StabilizedRailsConfigCache()
        rails = _platform_rails()
        v1 = EntityGuardrailConfigSource(workspace="ws", name="guard", updated_at="t1", rails=rails)
        v2 = EntityGuardrailConfigSource(workspace="ws", name="guard", updated_at="t2", rails=rails)

        first = cache.get_or_compute(v1, _resolve_target)
        second = cache.get_or_compute(v2, _resolve_target)
        assert first is not second

    def test_lru_eviction_drops_oldest(self) -> None:
        cache = StabilizedRailsConfigCache(max_entries=2)
        rails = _platform_rails()
        a = EntityGuardrailConfigSource(workspace="ws", name="a", updated_at="t", rails=rails)
        b = EntityGuardrailConfigSource(workspace="ws", name="b", updated_at="t", rails=rails)
        c = EntityGuardrailConfigSource(workspace="ws", name="c", updated_at="t", rails=rails)

        initial_a = cache.get_or_compute(a, _resolve_target)
        cache.get_or_compute(b, _resolve_target)
        initial_c = cache.get_or_compute(c, _resolve_target)

        # ``a`` is the LRU victim when ``c`` pushes over capacity, so a
        # re-fetch rebuilds a fresh :class:`StableRailsConfig`, distinct
        # from the original. This is the direct observable of eviction —
        # checking ``first_a is second_a`` after a rebuild would still
        # pass under an unbounded cache or the wrong eviction policy.
        rebuilt_a = cache.get_or_compute(a, _resolve_target)
        assert rebuilt_a is not initial_a, "``a`` must be evicted when ``c`` pushes the cache over capacity"
        # ``c`` was among the two most-recent at eviction time, so it still hits.
        assert cache.get_or_compute(c, _resolve_target) is initial_c, "``c`` must survive ``a``'s eviction"

    def test_recent_access_resists_eviction(self) -> None:
        """LRU semantics: a re-touched entry moves to the most-recent end and
        survives subsequent inserts that would evict the oldest."""
        cache = StabilizedRailsConfigCache(max_entries=2)
        rails = _platform_rails()
        a = EntityGuardrailConfigSource(workspace="ws", name="a", updated_at="t", rails=rails)
        b = EntityGuardrailConfigSource(workspace="ws", name="b", updated_at="t", rails=rails)
        c = EntityGuardrailConfigSource(workspace="ws", name="c", updated_at="t", rails=rails)

        first_a = cache.get_or_compute(a, _resolve_target)
        cache.get_or_compute(b, _resolve_target)
        # Touch ``a`` so it becomes most-recent; ``b`` is now LRU.
        touched_a = cache.get_or_compute(a, _resolve_target)
        assert touched_a is first_a
        # Insert ``c``: evicts ``b``, leaves ``a`` intact.
        cache.get_or_compute(c, _resolve_target)
        # ``a`` still hits.
        assert cache.get_or_compute(a, _resolve_target) is first_a


class TestStabilizedRailsConfigCacheInlineBypass:
    def test_inline_source_bypasses_cache(self) -> None:
        """Inline sources have no identity to memoize on, so the cache pays
        the stabilize cost per call. Two calls with the same inline source
        return *equal* stable configs (same content_hash) but distinct
        instances — the cache state is unaffected."""
        cache = StabilizedRailsConfigCache()
        inline = InlineGuardrailConfigSource(rails=_platform_rails(), label="x")

        first = cache.get_or_compute(inline, _resolve_target)
        second = cache.get_or_compute(inline, _resolve_target)
        assert first is not second
        assert first.content_hash == second.content_hash
        assert cache._map == {}, "inline source must not seed the memoization cache"


# =============================================================================
# §3: Pool — single-key resource pool
# =============================================================================


def _fake_rails(name: str = "rails") -> SimpleNamespace:
    """Stand-in for an :class:`LLMRails` instance.

    ``events_history_cache`` and ``explain_info`` are the exact surfaces the
    cache wipes on every lease — model them as on a real instance so the reset
    assertions exercise real behaviour. ``update_llm`` records the last main
    LLM applied for assertions.
    """
    rails = SimpleNamespace(
        name=name,
        events_history_cache={"prefix-A": "prior"},
        explain_info=object(),
        update_llm_calls=[],
    )

    def update_llm(llm: Any) -> None:
        rails.update_llm_calls.append(llm)

    rails.update_llm = update_llm
    return rails


class _FakeBuilder(LLMRailsBuilder):
    """Builder that records every build and lets tests inject latency / errors."""

    def __init__(
        self,
        *,
        delay: float = 0.0,
        raise_on: list[Exception] | None = None,
    ) -> None:
        self.calls = 0
        self.configs: list[RailsConfig] = []
        self._delay = delay
        self._raise_on = list(raise_on or [])

    async def build(self, config: RailsConfig) -> Any:
        self.calls += 1
        self.configs.append(config)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._raise_on:
            raise self._raise_on.pop(0)
        return _fake_rails(name=f"rails-{self.calls}")


class _ConcurrencyProbingBuilder(LLMRailsBuilder):
    """Builder that gates on an :class:`asyncio.Event` and records concurrent-build counts.

    Tests use this to deterministically assert build-time ordering invariants
    (e.g. same-pool builds serialize on ``_build_lock``; different-pool builds
    parallelize). Compared to a delay-based ``_FakeBuilder``, the gating
    semantics avoid wall-clock thresholds that flake on loaded CI hosts.

    :meth:`wait_for_in_flight` is the signal-driven replacement for the
    ``for _ in range(N): await asyncio.sleep(0)`` idiom: it lets a test wait
    until a target number of builds have actually entered the gated section
    (or fail loudly on a real timeout), instead of pumping the loop a
    hand-tuned number of times and hoping that's enough.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.in_flight = 0
        self.peak_in_flight = 0
        self.gate = asyncio.Event()
        # Notified every time ``in_flight`` changes so waiters can re-check.
        # asyncio.Event would only fire once; Condition lets us wake a waiter
        # whose target count is reached without consuming the signal.
        self._inflight_changed = asyncio.Condition()

    async def build(self, config: RailsConfig) -> Any:
        async with self._inflight_changed:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            self.calls += 1
            self._inflight_changed.notify_all()
        try:
            await self.gate.wait()
            return _fake_rails(name=f"rails-{self.calls}")
        finally:
            async with self._inflight_changed:
                self.in_flight -= 1
                self._inflight_changed.notify_all()

    async def wait_for_in_flight(self, n: int, *, timeout: float = 2.0) -> None:
        """Block until at least ``n`` builds have entered the gated section.

        Fails loudly with :class:`asyncio.TimeoutError` if the target isn't
        reached within ``timeout`` seconds — a deterministic timeout is
        recoverable by CI logs, whereas a missed-yield-count silently passes
        with a stale assertion.
        """
        async with asyncio.timeout(timeout):
            async with self._inflight_changed:
                while self.in_flight < n:
                    await self._inflight_changed.wait()


def _stable(content_hash: str = "hash-a") -> StableRailsConfig:
    """Stand-in for a :class:`StableRailsConfig` keyed by ``content_hash``.

    The cache treats ``stable.rails`` opaquely — it stores it on the pool
    and forwards it to the builder. A bare :class:`MagicMock` lets us avoid
    the real :class:`RailsConfig` validator overhead in tests; what the
    cache actually keys by is the hash, which is the only field we need to
    distinguish.
    """
    return StableRailsConfig(
        rails=cast(Any, MagicMock(spec=RailsConfig)),
        content_hash=content_hash,
        embedding_model_id=None,
    )


STABLE_A = _stable("hash-a")
STABLE_B = _stable("hash-b")
STABLE_A_V2 = _stable("hash-a-v2")


class TestPool:
    """``Pool`` is the inner layer — unbounded resource pool for one key.

    Tests here cover the surviving invariants: same-pool build serialization,
    exclusive ownership across acquires, the demand-watermark shrink, and
    failure handling for build/prepare/user-code raises.
    """

    def test_rejects_negative_min_idle(self) -> None:
        """``min_idle < 0`` is meaningless and would let the shrink trim
        run away below zero. Defined out of existence at construction so
        the bug surfaces here, not on the first release."""
        with pytest.raises(ValueError, match="min_idle must be >= 0"):
            Pool(stable=_stable(), min_idle=-1)

    def test_zero_min_idle_is_allowed(self) -> None:
        """``min_idle=0`` is a valid configuration: pool drops to fully
        empty when traffic ceases. Useful when memory matters more than
        next-cold-request latency."""
        pool = Pool(stable=_stable(), min_idle=0)
        assert pool.min_idle == 0

    def test_pool_exposes_config_view_of_stable_rails(self) -> None:
        """``pool.config`` is a read-through view of ``stable.rails`` so the
        builder contract ("give me a RailsConfig") survives the key reshape."""
        stable = _stable()
        pool = Pool(stable=stable)
        assert pool.config is stable.rails

    def test_record_appends_unique_provenance(self) -> None:
        """Provenance is a diagnostics-only field; same-source-twice
        coalesces so a hot pool doesn't fill its bounded buffer with one
        label."""
        pool = Pool(stable=_stable())
        pool.record(Provenance("ws/foo@v1"))
        pool.record(Provenance("ws/foo@v1"))
        pool.record(Provenance("<inline:bar>"))
        assert [p.label for p in pool.provenance_history] == ["ws/foo@v1", "<inline:bar>"]

    def test_record_dedup_is_consecutive_only_not_global(self) -> None:
        """Dedup is most-recent only: an ``A, B, A`` sequence keeps all three.

        A pool alternating between two sources with coincidentally-equal
        content hashes would otherwise compress to just ``[A, B]``, hiding
        the alternation from diagnostic logs. A naive set-based dedup would
        pass :meth:`test_record_appends_unique_provenance` and fail here.
        """
        pool = Pool(stable=_stable())
        pool.record(Provenance("A"))
        pool.record(Provenance("B"))
        pool.record(Provenance("A"))
        assert [p.label for p in pool.provenance_history] == ["A", "B", "A"]

    async def test_acquire_builds_on_miss(self) -> None:
        builder = _FakeBuilder()
        pool = Pool(stable=_stable())

        async with pool.acquire(builder.build) as rails:
            assert rails is not None
            assert builder.calls == 1

    async def test_acquire_reuses_returned_instance(self) -> None:
        """A second acquire on the same pool must reuse — not rebuild."""
        builder = _FakeBuilder()
        pool = Pool(stable=_stable())

        async with pool.acquire(builder.build) as first:
            pass
        async with pool.acquire(builder.build) as second:
            assert second is first
            assert builder.calls == 1

    async def test_concurrent_misses_serialize_on_build_lock(self) -> None:
        """``_build_lock`` is the only thing standing between a thundering-herd
        of cold-key requests and N parallel ``LLMRails.__init__`` calls. Probe
        builder concurrency directly instead of timing-based assertions.

        This is the surviving "throttling" mechanism after dropping
        ``max_size``: not request-level back-pressure, but build-level
        serialization. Without it, N concurrent missers would all build
        in parallel and slam the ``asyncio.to_thread`` worker pool.
        """
        builder = _ConcurrencyProbingBuilder()
        pool = Pool(stable=_stable())

        async def take_and_release() -> None:
            async with pool.acquire(builder.build):
                pass

        task_a = asyncio.create_task(take_and_release())
        task_b = asyncio.create_task(take_and_release())

        # Wait until the first build has actually started, then assert no
        # second build sneaks in while the gate is still closed. ``peak``
        # below covers the whole lifetime, which is the real invariant —
        # this in_flight check is a sanity probe at the gated moment.
        await builder.wait_for_in_flight(1)
        assert builder.in_flight == 1, (
            f"_build_lock should serialize same-pool builds; saw {builder.in_flight} concurrent builds"
        )

        builder.gate.set()
        await asyncio.gather(task_a, task_b)
        assert builder.peak_in_flight == 1

    async def test_grows_under_concurrent_load(self) -> None:
        """Independent concurrent acquires grow the pool to match demand.

        With no ``max_size``, three concurrent holders end up with three
        distinct instances — pool capacity equals concurrent demand. After
        all three release, the shrink trim takes over (verified separately
        in :meth:`test_idle_shrinks_to_min_idle_under_quiet_load`).
        """
        builder = _FakeBuilder()
        pool = Pool(stable=_stable())

        events = [asyncio.Event() for _ in range(3)]
        seen: list[Any] = []
        held = asyncio.Semaphore(0)

        async def hold(idx: int) -> None:
            async with pool.acquire(builder.build) as rails:
                seen.append(rails)
                held.release()
                await events[idx].wait()

        tasks = [asyncio.create_task(hold(i)) for i in range(3)]
        for _ in range(3):
            await held.acquire()

        assert builder.calls == 3
        assert len({id(r) for r in seen}) == 3
        for ev in events:
            ev.set()
        await asyncio.gather(*tasks)

    async def test_idle_shrinks_to_min_idle_under_quiet_load(self) -> None:
        """After a peak-then-quiet transition, idle converges to ``min_idle``.

        The demand-watermark shrink is what keeps unbounded growth bounded
        in practice. Without it, a 100-request burst followed by silence
        would leave 100 idle instances forever; with it, releases drop
        instances in lockstep with ``_leased`` so the pool returns to
        ``min_idle`` after the last release.

        Trace of the scenario below (5 holders, min_idle=1):
        - peak: _leased=5, _idle=0
        - release 1: _leased=4, append → idle=1, 1>max(1,4)=4? no → idle=1
        - release 2: _leased=3, append → idle=2, 2>3? no → idle=2
        - release 3: _leased=2, append → idle=3, 3>2? yes → trim → idle=2
        - release 4: _leased=1, append → idle=3, 3>1? yes → trim → idle=1
        - release 5: _leased=0, append → idle=2, 2>1? yes → trim → idle=1
        """
        builder = _FakeBuilder()
        pool = Pool(stable=_stable(), min_idle=1)

        release_events = [asyncio.Event() for _ in range(5)]
        held = asyncio.Semaphore(0)

        async def hold(idx: int) -> None:
            async with pool.acquire(builder.build):
                held.release()
                await release_events[idx].wait()

        tasks = [asyncio.create_task(hold(i)) for i in range(5)]
        for _ in range(5):
            await held.acquire()

        assert builder.calls == 5

        # Release one at a time so the assertion order matches the trace.
        for ev in release_events:
            ev.set()
        await asyncio.gather(*tasks)

        # All released: leased=0, idle should have converged to min_idle.
        assert pool._leased == 0
        assert len(pool._idle) == 1, f"idle should converge to min_idle=1 after quiet, got {len(pool._idle)}"

    async def test_idle_shrinks_to_zero_when_min_idle_is_zero(self) -> None:
        """``min_idle=0`` lets the pool drop to fully empty after silence.

        Useful for memory-sensitive deployments: every fresh request after
        a quiet period pays one build cost, but no instances are pinned in
        idle when there's no traffic.
        """
        builder = _FakeBuilder()
        pool = Pool(stable=_stable(), min_idle=0)

        async with pool.acquire(builder.build):
            pass
        assert pool._leased == 0
        assert len(pool._idle) == 0

    async def test_idle_keeps_min_idle_after_single_release(self) -> None:
        """Even a single acquire/release cycle leaves ``min_idle`` instances
        warm — the trim only fires when ``len(idle) > max(min_idle, leased)``,
        so a lone release is preserved if it doesn't exceed the floor."""
        builder = _FakeBuilder()
        pool = Pool(stable=_stable(), min_idle=2)

        # Build two via concurrent acquires.
        events = [asyncio.Event() for _ in range(2)]
        held = asyncio.Semaphore(0)

        async def hold(idx: int) -> None:
            async with pool.acquire(builder.build):
                held.release()
                await events[idx].wait()

        tasks = [asyncio.create_task(hold(i)) for i in range(2)]
        for _ in range(2):
            await held.acquire()
        for ev in events:
            ev.set()
        await asyncio.gather(*tasks)

        # Both built and released; min_idle=2 should preserve both.
        assert builder.calls == 2
        assert len(pool._idle) == 2

    async def test_build_failure_does_not_increment_leased(self) -> None:
        """A failed build leaves accounting clean: ``_leased`` is bumped
        only after a successful take/build, so a build raise has nothing
        to roll back."""
        builder = _FakeBuilder(raise_on=[RuntimeError("nim down")])
        pool = Pool(stable=_stable())

        with pytest.raises(RuntimeError, match="nim down"):
            async with pool.acquire(builder.build):
                pass

        assert pool._leased == 0
        assert len(pool._idle) == 0

        # Retry succeeds and makes a fresh instance.
        async with pool.acquire(builder.build):
            assert builder.calls == 2

    async def test_prepare_failure_discards_instance(self) -> None:
        """A ``prepare`` raise must drop the (presumed-half-mutated) instance
        and decrement ``_leased`` so the shrink heuristic sees the right demand."""
        builder = _FakeBuilder()
        pool = Pool(stable=_stable())

        def bad_prepare(_rails: Any) -> None:
            raise RuntimeError("update_llm exploded")

        with pytest.raises(RuntimeError, match="update_llm exploded"):
            async with pool.acquire(builder.build, prepare=bad_prepare):
                pass

        assert pool._leased == 0
        # The discarded instance must NOT be requeued.
        assert len(pool._idle) == 0

        # Next acquire builds fresh.
        async with pool.acquire(builder.build) as rails:
            assert rails is not None
            assert builder.calls == 2

    async def test_prepare_failure_on_reused_instance_discards(self) -> None:
        """Same discard contract when the instance came from the idle queue
        rather than a fresh build. The ``_leased`` accounting is symmetric
        across both paths and shouldn't drift."""
        builder = _FakeBuilder()
        pool = Pool(stable=_stable())

        async with pool.acquire(builder.build) as first:
            pass
        assert builder.calls == 1
        assert len(pool._idle) == 1

        prepare_calls: list[Any] = []

        def bad_prepare(rails: Any) -> None:
            prepare_calls.append(rails)
            raise RuntimeError("reuse-time prepare exploded")

        with pytest.raises(RuntimeError, match="reuse-time prepare exploded"):
            async with pool.acquire(builder.build, prepare=bad_prepare):
                pass

        assert prepare_calls == [first], "prepare ran on the reused instance, not a fresh build"
        assert len(pool._idle) == 0, "the reused-then-discarded instance must not come back"
        assert pool._leased == 0

        async with pool.acquire(builder.build) as second:
            assert second is not first
            assert builder.calls == 2

    async def test_user_exception_returns_instance_to_pool(self) -> None:
        """An exception inside the ``async with`` body must NOT discard — the
        next acquire's prepare wipes per-request state, so reuse is safe.

        The discriminator vs. cancellation is "did the awaitable resolve
        before the body raised?" — for a regular exception the answer is
        yes, so any thread the body was awaiting is already done, and reuse
        is safe. The cancellation case is covered separately in
        :meth:`test_cancellation_discards_instance` because there the
        worker thread can outlive the awaiting coroutine.
        """
        builder = _FakeBuilder()
        pool = Pool(stable=_stable())

        first: Any = None
        try:
            async with pool.acquire(builder.build) as rails:
                first = rails
                raise RuntimeError("user code blew up")
        except RuntimeError:
            pass

        async with pool.acquire(builder.build) as second:
            assert second is first
            assert builder.calls == 1

    async def test_cancellation_discards_instance(self) -> None:
        """A cancelled ``async with`` body MUST drop the rails on the floor.

        A body suspended at ``await asyncio.to_thread(generate_async, ...)``
        whose coroutine is cancelled (client disconnect, request timeout) is
        the one path where the worker thread can still be mutating the
        leased :class:`LLMRails` after release — Python threads ignore
        ``concurrent.futures.Future.cancel()`` once they're running.
        Requeuing here would let the next ``prepare`` wipe race the orphan
        thread's writes and reopen the cross-tenant state-leak surface.
        ``_leased`` must still decrement so demand-watermark math is correct.

        We can't actually launch a thread that survives the lease in a unit
        test (it would leak across tests); instead the body is cancelled
        while suspended on a long ``asyncio.sleep`` — the lease unwinds via
        ``asyncio.CancelledError`` exactly as it would in production.
        """
        builder = _FakeBuilder()
        pool = Pool(stable=_stable())

        held = asyncio.Event()
        first_rails: Any = None

        async def hold() -> None:
            nonlocal first_rails
            async with pool.acquire(builder.build) as rails:
                first_rails = rails
                held.set()
                # Block until cancelled — stands in for an in-flight
                # ``asyncio.to_thread(generate_async, ...)``.
                await asyncio.sleep(3600)

        task = asyncio.create_task(hold())
        await held.wait()
        assert pool._leased == 1
        assert len(pool._idle) == 0

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert pool._leased == 0, "_leased must decrement even on cancellation"
        assert len(pool._idle) == 0, (
            "cancelled lease must NOT requeue rails: a stand-in for the orphan-worker-thread "
            "case where the rails could still be mid-mutation"
        )

        # A fresh acquire builds a new instance — the cancelled one is gone.
        async with pool.acquire(builder.build) as fresh:
            assert fresh is not first_rails
            assert builder.calls == 2

    async def test_generator_exit_discards_instance(self) -> None:
        """An async generator's ``aclose()`` injects :class:`GeneratorExit`
        into the body, distinct from the :class:`asyncio.CancelledError`
        delivered by task cancellation. The discard contract must apply
        in both cases for the same reason: a worker thread spawned inside
        the body could still be mutating ``rails`` after the lease
        unwinds. This is the path the IGW framework takes when it drops a
        streaming response without iterating it (PEP 525) — closing the
        returned iterator without consuming it raises ``GeneratorExit``
        inside the body.

        Without this branch, ``aclose()`` on a never-iterated streaming
        response would re-pool the rails while the next acquirer's
        ``prepare`` wipe could race the worker's writes.
        """
        builder = _FakeBuilder()
        pool = Pool(stable=_stable())

        first_rails: Any = None

        async def hold() -> AsyncGenerator[None, None]:
            # ``AsyncGenerator`` (not ``AsyncIterator``) so the type checker
            # sees the ``aclose()`` method below; bare ``AsyncIterator``
            # only guarantees ``__aiter__`` / ``__anext__``.
            nonlocal first_rails
            async with pool.acquire(builder.build) as rails:
                first_rails = rails
                yield None
                # Body suspended at the next ``yield`` after the consumer
                # calls ``__anext__`` once. ``aclose()`` then injects
                # ``GeneratorExit`` here, mirroring the streaming-response
                # cleanup path in process_response.
                yield None  # pragma: no cover - consumer never reaches here

        gen = hold()
        await gen.__anext__()
        assert pool._leased == 1
        assert len(pool._idle) == 0

        await gen.aclose()

        assert pool._leased == 0, "_leased must decrement even on aclose()"
        assert len(pool._idle) == 0, (
            "aclose() injects GeneratorExit; the rails must be discarded for the same "
            "reason cancellation discards (worker thread may still be mid-mutation)"
        )

        # A fresh acquire builds a new instance — the closed one is gone.
        async with pool.acquire(builder.build) as fresh:
            assert fresh is not first_rails
            assert builder.calls == 2

    async def test_cancellation_discards_only_cancelled_lease(self) -> None:
        """Discard must drop *only* the cancelled lease's instance — a
        sibling lease running concurrently on the same pool must survive
        and re-pool normally on its own release.

        Without this guarantee, a single cancelled request in a busy pool
        could be misread as a global "drop everything" signal and collapse
        unrelated in-flight work, which would convert client disconnects
        into pool stampedes.
        """
        builder = _FakeBuilder()
        pool = Pool(stable=_stable(), min_idle=1)

        survivor_held = asyncio.Event()
        cancelled_held = asyncio.Event()
        survivor_release = asyncio.Event()

        survivor_id: int | None = None
        cancelled_id: int | None = None

        async def survivor() -> None:
            nonlocal survivor_id
            async with pool.acquire(builder.build) as rails:
                survivor_id = id(rails)
                survivor_held.set()
                await survivor_release.wait()

        async def to_cancel() -> None:
            nonlocal cancelled_id
            async with pool.acquire(builder.build) as rails:
                cancelled_id = id(rails)
                cancelled_held.set()
                await asyncio.sleep(3600)

        s_task = asyncio.create_task(survivor())
        c_task = asyncio.create_task(to_cancel())
        await asyncio.gather(survivor_held.wait(), cancelled_held.wait())
        assert pool._leased == 2
        assert builder.calls == 2  # distinct instances per concurrent lease
        assert survivor_id != cancelled_id

        c_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await c_task

        # Survivor is still leased; the cancelled lease's instance is gone.
        assert pool._leased == 1
        assert len(pool._idle) == 0

        survivor_release.set()
        await s_task

        # Survivor was re-pooled normally; the cancelled instance never
        # comes back — pool now holds exactly one idle, which is the
        # survivor's, not a ghost of the cancelled lease.
        assert pool._leased == 0
        assert len(pool._idle) == 1
        assert id(pool._idle[0]) == survivor_id

    async def test_warm_builds_one_instance(self) -> None:
        builder = _FakeBuilder()
        pool = Pool(stable=_stable())

        await pool.warm(builder.build)
        assert builder.calls == 1

        async with pool.acquire(builder.build) as rails:
            assert rails is not None
            assert builder.calls == 1  # warm's instance was reused

    async def test_warm_is_noop_when_pool_already_built(self) -> None:
        builder = _FakeBuilder()
        pool = Pool(stable=_stable())

        async with pool.acquire(builder.build):
            pass
        await pool.warm(builder.build)

        assert builder.calls == 1

    async def test_warm_builds_during_active_lease(self) -> None:
        """Warm builds even when a lease is in flight.

        A leased instance is exclusive to its holder — it provides zero
        warmth to a concurrent acquirer. For long-lived leases
        (streaming, KB-bearing builds) skipping the warm here would
        force the next arrival onto the cold path for the entire stream
        duration. The post-release watermark trim reclaims the extra
        instance if it goes unused (``len(_idle) > max(min_idle,
        _leased)``), so the cost is bounded.
        """
        builder = _FakeBuilder()
        pool = Pool(stable=_stable())

        held = asyncio.Event()
        release = asyncio.Event()

        async def hold() -> None:
            async with pool.acquire(builder.build):
                held.set()
                await release.wait()

        task = asyncio.create_task(hold())
        await held.wait()
        assert builder.calls == 1
        assert pool._leased == 1
        assert len(pool._idle) == 0

        await pool.warm(builder.build)
        assert builder.calls == 2, "warm must build to serve concurrent acquirers"
        assert len(pool._idle) == 1

        release.set()
        await task

        # Watermark trim reclaims the extra on release: with min_idle=1
        # and _leased=0, idle converges back to 1 — the redundant build
        # cost is bounded, not permanent residency.
        assert pool._leased == 0
        assert len(pool._idle) == 1

    async def test_warm_during_lease_serves_concurrent_acquirer_without_cold_build(self) -> None:
        """The actual win: an acquirer arriving between warm and lease
        release reuses the warmed instance instead of paying a cold build.

        Without warming during a lease, this acquirer would hit the slow
        path in ``_take_or_build`` and call ``build`` a third time.
        """
        builder = _FakeBuilder()
        pool = Pool(stable=_stable())

        held = asyncio.Event()
        release = asyncio.Event()

        async def hold() -> None:
            async with pool.acquire(builder.build):
                held.set()
                await release.wait()

        first = asyncio.create_task(hold())
        await held.wait()
        assert builder.calls == 1

        await pool.warm(builder.build)
        assert builder.calls == 2  # warm built one ahead

        # Second acquirer arrives mid-lease — must reuse warm's instance,
        # not pay a third cold build.
        async with pool.acquire(builder.build) as rails:
            assert rails is not None
            assert builder.calls == 2, "concurrent acquirer must reuse warmed instance"

        release.set()
        await first

    async def test_warm_swallows_build_errors(self) -> None:
        """Best-effort: a failed warm just leaves the pool empty for the next acquire."""
        builder = _FakeBuilder(raise_on=[RuntimeError("kb build failed")])
        pool = Pool(stable=_stable())

        await pool.warm(builder.build)

        async with pool.acquire(builder.build) as rails:
            assert rails is not None
            assert builder.calls == 2


# =============================================================================
# §4: LLMRailsCache — keyed lookup, eviction, lifecycle
# =============================================================================


class TestCacheConstruction:
    """Preconditions on ``LLMRailsCache.__init__``.

    Both bounds are precondition-checked at construction so an invalid
    configuration fails loudly *here* rather than producing surprise
    behaviour later (``max_pools=0`` would otherwise log a warning per
    miss and grow without bound; negative ``pool_min_idle`` would underflow
    the trim arithmetic in ``Pool``).
    """

    def test_rejects_zero_max_pools(self) -> None:
        with pytest.raises(ValueError, match="max_pools must be >= 1"):
            LLMRailsCache(builder=_FakeBuilder(), max_pools=0)

    def test_rejects_negative_max_pools(self) -> None:
        with pytest.raises(ValueError, match="max_pools must be >= 1"):
            LLMRailsCache(builder=_FakeBuilder(), max_pools=-1)

    def test_rejects_negative_pool_min_idle(self) -> None:
        with pytest.raises(ValueError, match="pool_min_idle must be >= 0"):
            LLMRailsCache(builder=_FakeBuilder(), pool_min_idle=-1)

    def test_zero_pool_min_idle_is_allowed(self) -> None:
        """Mirrors :class:`Pool`: zero idle floor is a valid memory-saving
        configuration, not a bug."""
        cache = LLMRailsCache(builder=_FakeBuilder(), pool_min_idle=0)
        assert cache._pool_min_idle == 0


class TestCacheLeaseRoutesByKey:
    async def test_independent_keys_build_in_parallel(self) -> None:
        """Different content hashes do not serialize on each other's ``_build_lock``."""
        builder = _ConcurrencyProbingBuilder()
        cache = LLMRailsCache(builder=builder)

        async def lease(stable: StableRailsConfig) -> None:
            async with cache.lease(stable):
                pass

        task_a = asyncio.create_task(lease(STABLE_A))
        task_b = asyncio.create_task(lease(STABLE_B))

        await builder.wait_for_in_flight(2)
        assert builder.peak_in_flight == 2, (
            f"different keys must build in parallel; observed peak={builder.peak_in_flight}"
        )

        builder.gate.set()
        await asyncio.gather(task_a, task_b)

    async def test_same_hash_reuses_pool(self) -> None:
        """Two stable configs sharing a hash must land on the same pool;
        the second lease's ``stable.rails`` is unused. The footgun the
        redesign closes by construction: pool selection is by content,
        not by entity identity."""
        cache = LLMRailsCache(builder=_FakeBuilder())

        first = _stable("hash-shared")
        second = _stable("hash-shared")
        assert first is not second  # distinct instances, same hash

        async with cache.lease(first):
            pass
        async with cache.lease(second):
            pass

        assert len(cache._pools) == 1
        assert "hash-shared" in cache._pools

    async def test_distinct_hashes_create_distinct_pools(self) -> None:
        """Different content hashes produce different pools, regardless of
        how the stable configs were constructed."""
        cache = LLMRailsCache(builder=_FakeBuilder())

        async with cache.lease(STABLE_A):
            pass
        async with cache.lease(STABLE_B):
            pass
        async with cache.lease(STABLE_A_V2):
            pass

        assert set(cache._pools) == {"hash-a", "hash-b", "hash-a-v2"}


class TestCacheProvenance:
    """Provenance is metadata only — recorded for diagnostics, never identity."""

    async def test_lease_records_provenance_on_pool(self) -> None:
        cache = LLMRailsCache(builder=_FakeBuilder())
        prov = Provenance("ws/foo@v1")

        async with cache.lease(STABLE_A, provenance=prov):
            pass

        pool = cache._pools[STABLE_A.content_hash]
        assert list(pool.provenance_history) == [prov]

    async def test_recent_provenance_history_is_bounded(self) -> None:
        """Pool's history is a bounded ring buffer so a long-lived hot
        pool doesn't accumulate unbounded labels."""
        cache = LLMRailsCache(builder=_FakeBuilder())
        labels = [f"ws/foo@v{i}" for i in range(PROVENANCE_HISTORY_LEN + 3)]

        for label in labels:
            async with cache.lease(STABLE_A, provenance=Provenance(label)):
                pass

        pool = cache._pools[STABLE_A.content_hash]
        history = [p.label for p in pool.provenance_history]
        assert len(history) == PROVENANCE_HISTORY_LEN
        # The most recent N labels are kept; the oldest are dropped.
        assert history == labels[-PROVENANCE_HISTORY_LEN:]

    async def test_lease_without_provenance_does_not_record(self) -> None:
        """The provenance arg is optional; tests that don't care needn't pass one."""
        cache = LLMRailsCache(builder=_FakeBuilder())

        async with cache.lease(STABLE_A):
            pass

        pool = cache._pools[STABLE_A.content_hash]
        assert list(pool.provenance_history) == []


class TestCacheLeaseReset:
    """The cache's ``prepare`` callback wipes the per-request leak surfaces.

    ``events_history_cache`` and ``explain_info`` are cross-tenant data
    hazards on a shared instance; ``update_llm`` runs unconditionally —
    including with ``None`` — so a reused instance can't carry the previous
    lease's main LLM into a later lease.
    """

    async def test_clears_events_history_cache(self) -> None:
        cache = LLMRailsCache(builder=_FakeBuilder())

        async with cache.lease(STABLE_A) as rails:
            assert rails.events_history_cache == {}

    async def test_resets_explain_info(self) -> None:
        cache = LLMRailsCache(builder=_FakeBuilder())

        async with cache.lease(STABLE_A) as rails:
            assert rails.explain_info is None

    async def test_update_llm_called_with_main_llm(self) -> None:
        cache = LLMRailsCache(builder=_FakeBuilder())
        main_llm = MagicMock(name="main_llm")

        async with cache.lease(STABLE_A, main_llm=main_llm) as rails:
            # ``rails`` is a ``SimpleNamespace`` from ``_FakeBuilder``, not a
            # real LLMRails — cast to inspect the recording attribute.
            assert cast(Any, rails).update_llm_calls == [main_llm]

    async def test_main_llm_swapped_per_lease(self) -> None:
        """A reused instance gets the new request's main LLM, not the previous one."""
        cache = LLMRailsCache(builder=_FakeBuilder())
        first_llm = MagicMock(name="first_llm")
        second_llm = MagicMock(name="second_llm")

        async with cache.lease(STABLE_A, main_llm=first_llm) as rails:
            first_id = id(rails)
        async with cache.lease(STABLE_A, main_llm=second_llm) as rails:
            assert id(rails) == first_id, "cache should reuse the pooled instance"
            assert cast(Any, rails).update_llm_calls == [first_llm, second_llm]

    async def test_main_llm_none_clears_previous_main_llm(self) -> None:
        """A follow-up lease without ``main_llm`` MUST clear the previous request's LLM.

        Otherwise ``rails.llm`` / ``rails.llm_generation_actions.llm`` /
        the ``"llm"`` action param keep the prior tenant's model and
        rails.generate_async would silently route through it.
        """
        cache = LLMRailsCache(builder=_FakeBuilder())
        first_llm = MagicMock(name="first_llm")

        async with cache.lease(STABLE_A, main_llm=first_llm) as rails:
            first_id = id(rails)
        async with cache.lease(STABLE_A) as rails:
            assert id(rails) == first_id, "cache should reuse the pooled instance"
            # Second update_llm call wipes the prior model rather than leaving it stale.
            assert cast(Any, rails).update_llm_calls == [first_llm, None]

    async def test_history_cache_repopulation_does_not_leak_across_leases(self) -> None:
        """Mid-request mutations to events_history_cache are wiped on next lease."""
        cache = LLMRailsCache(builder=_FakeBuilder())

        async with cache.lease(STABLE_A) as rails:
            rails.events_history_cache["leak-me"] = "tenant-A-data"

        async with cache.lease(STABLE_A) as rails_2:
            assert "leak-me" not in rails_2.events_history_cache


class TestCacheEviction:
    async def test_max_pools_evicts_lru(self) -> None:
        """Adding a third key to a max_pools=2 cache evicts the LRU."""
        cache = LLMRailsCache(builder=_FakeBuilder(), max_pools=2)

        # STABLE_A leased and held — pinned, not evictable.
        ev_release_a = asyncio.Event()
        a_acquired = asyncio.Event()

        async def hold_a() -> None:
            async with cache.lease(STABLE_A):
                a_acquired.set()
                await ev_release_a.wait()

        a_task = asyncio.create_task(hold_a())
        await a_acquired.wait()

        # STABLE_B leased and released — present, not pinned, idle and evictable.
        async with cache.lease(STABLE_B):
            pass

        # Adding STABLE_A_V2 must evict the only evictable pool — STABLE_B.
        async with cache.lease(STABLE_A_V2):
            pass

        assert len(cache._pools) == 2
        assert STABLE_B.content_hash not in cache._pools
        assert STABLE_A.content_hash in cache._pools
        assert STABLE_A_V2.content_hash in cache._pools
        ev_release_a.set()
        await a_task

    async def test_in_use_pool_blocks_concurrent_eviction(self) -> None:
        """A lease in progress must pin its pool against LRU eviction.

        Without the pin, a concurrent ``lease`` for a different key would see
        capacity pressure, scan for an evictable pool, and could drop the
        in-progress one — orphaning the leased instance and breaking the
        next ``release`` into a phantom pool.
        """
        builder = _ConcurrencyProbingBuilder()
        cache = LLMRailsCache(builder=builder, max_pools=1)

        async def lease(stable: StableRailsConfig) -> None:
            async with cache.lease(stable):
                pass

        task_a = asyncio.create_task(lease(STABLE_A))
        await builder.wait_for_in_flight(1)

        # STABLE_B leases; capacity is 1, only STABLE_A exists, STABLE_A is pinned.
        # The cache must grow temporarily over the cap rather than orphan A.
        # STABLE_B's build entering the gated section is the deterministic proof
        # that ``_pin`` succeeded for STABLE_B *and* that STABLE_A wasn't evicted
        # to make room (otherwise A's task would crash, not B's start).
        task_b = asyncio.create_task(lease(STABLE_B))
        await builder.wait_for_in_flight(2)

        assert STABLE_A.content_hash in cache._pools
        assert STABLE_B.content_hash in cache._pools
        builder.gate.set()
        await asyncio.gather(task_a, task_b)

    async def test_recent_lease_resists_eviction(self) -> None:
        """LRU ordering on the outer cache: a re-leased pool moves to MRU and
        survives subsequent inserts that would otherwise evict the oldest.
        """
        cache = LLMRailsCache(builder=_FakeBuilder(), max_pools=2)

        async with cache.lease(STABLE_A):
            pass
        async with cache.lease(STABLE_B):
            pass
        # Re-lease STABLE_A so it becomes the most-recent; STABLE_B is now LRU.
        async with cache.lease(STABLE_A):
            pass
        # Insert STABLE_A_V2: should evict STABLE_B, preserve STABLE_A.
        async with cache.lease(STABLE_A_V2):
            pass

        assert set(cache._pools) == {STABLE_A.content_hash, STABLE_A_V2.content_hash}
        assert STABLE_B.content_hash not in cache._pools, (
            "re-leased STABLE_A should have moved to MRU; STABLE_B was the LRU victim"
        )

    async def test_warm_pins_pool_against_concurrent_eviction(self) -> None:
        """Same invariant via the warm path: an in-progress warm pins its pool."""
        builder = _ConcurrencyProbingBuilder()
        cache = LLMRailsCache(builder=builder, max_pools=1)

        warm_task = cache.warm(STABLE_A)
        await builder.wait_for_in_flight(1)

        async def lease_b() -> None:
            async with cache.lease(STABLE_B):
                pass

        b_task = asyncio.create_task(lease_b())
        await builder.wait_for_in_flight(2)

        # STABLE_A still present even though it has no idle instance yet.
        assert STABLE_A.content_hash in cache._pools
        builder.gate.set()
        await warm_task
        await b_task


class TestCacheBuildFailures:
    async def test_build_failure_allows_retry(self) -> None:
        """A failed build must not pollute pool state; retry is allowed."""
        builder = _FakeBuilder(raise_on=[RuntimeError("nim down")])
        cache = LLMRailsCache(builder=builder)

        with pytest.raises(RuntimeError, match="nim down"):
            async with cache.lease(STABLE_A):
                pass

        async with cache.lease(STABLE_A):
            assert builder.calls == 2

    async def test_warm_failure_is_swallowed(self) -> None:
        builder = _FakeBuilder(raise_on=[RuntimeError("kb build failed")])
        cache = LLMRailsCache(builder=builder)

        await cache.warm(STABLE_A)

        # Pool exists but is empty; subsequent lease retries the build.
        async with cache.lease(STABLE_A):
            assert builder.calls == 2


class TestCacheLifecycle:
    async def test_close_drops_all_pools(self) -> None:
        cache = LLMRailsCache(builder=_FakeBuilder())

        async with cache.lease(STABLE_A):
            pass
        assert len(cache._pools) == 1
        await cache.close()
        assert len(cache._pools) == 0

    async def test_close_during_lease_is_safe(self) -> None:
        """``close`` must not raise even if a lease is mid-flight; the leased
        instance just lands in an orphaned pool that gets garbage-collected.

        The lease entered ``_pin`` before ``close`` flipped ``_closed``, so
        its closed-check passed. Leases started *after* ``close`` raise — see
        :meth:`test_lease_after_close_raises`.
        """
        cache = LLMRailsCache(builder=_FakeBuilder())

        async with cache.lease(STABLE_A):
            await cache.close()  # should not raise

    async def test_lease_after_close_raises(self) -> None:
        """A new lease against a closed cache must fail-fast rather than
        silently rebuild the pool we just dropped."""
        cache = LLMRailsCache(builder=_FakeBuilder())
        await cache.close()

        with pytest.raises(RuntimeError, match="closed"):
            async with cache.lease(STABLE_A):
                pass
        assert cache._pools == {}

    async def test_close_is_idempotent(self) -> None:
        """Double-close must be a no-op so a retried lifecycle hook can't
        double-clear pools or double-cancel warm tasks."""
        cache = LLMRailsCache(builder=_FakeBuilder())
        async with cache.lease(STABLE_A):
            pass

        await cache.close()
        await cache.close()  # must not raise
        assert cache._closed
        assert cache._pools == {}
        assert cache._warm_tasks == set()


class TestCacheWarm:
    """``warm`` is sync (returns the tracked task) and tracks for shutdown.

    Lifecycle hooks fire-and-forget; tests await the same returned task.
    """

    async def test_warm_returns_task_caller_can_await(self) -> None:
        builder = _FakeBuilder()
        cache = LLMRailsCache(builder=builder)

        task = cache.warm(STABLE_A)
        assert isinstance(task, asyncio.Task)
        await task
        assert builder.calls == 1, "warm must build exactly once"

        # The subsequent lease must reuse the warmed instance, not rebuild:
        # that reuse is the whole reason ``warm`` exists. Checking only
        # "lease succeeds" would still pass under a broken warm that
        # silently got evicted or discarded before the lease arrived.
        async with cache.lease(STABLE_A) as rails:
            assert builder.calls == 1, "lease after warm must reuse the warmed instance, not rebuild"
            assert cast(Any, rails).name == "rails-1", "lease must yield the exact instance warm produced"
        assert STABLE_A.content_hash in cache._pools

    async def test_warm_does_not_block_caller(self) -> None:
        """``warm`` returns synchronously even if the build is slow."""
        builder = _ConcurrencyProbingBuilder()
        cache = LLMRailsCache(builder=builder)

        task = cache.warm(STABLE_A)
        assert not task.done()

        builder.gate.set()
        await task

    async def test_completed_warm_is_removed_from_tracking_set(self) -> None:
        """A completed warm task must be removed from the tracking set so it
        doesn't pin its closure forever."""
        cache = LLMRailsCache(builder=_FakeBuilder())

        task = cache.warm(STABLE_A)
        await task
        # add_done_callback fires after the task completes; yield once for it.
        await asyncio.sleep(0)
        assert task not in cache._warm_tasks

    async def test_close_cancels_pending_warm_tasks(self) -> None:
        """``close()`` must cancel pending warm tasks so they don't keep
        building into a torn-down cache, and tests don't see "Task was
        destroyed but it is pending!" warnings."""
        builder = _ConcurrencyProbingBuilder()
        cache = LLMRailsCache(builder=builder)

        task = cache.warm(STABLE_A)
        await builder.wait_for_in_flight(1)
        assert not task.done()

        await cache.close()
        assert task.done()

    async def test_close_drains_already_finished_warms(self) -> None:
        """``close()`` must not block on tasks that have already completed."""
        cache = LLMRailsCache(builder=_FakeBuilder())

        task = cache.warm(STABLE_A)
        await task
        await cache.close()  # should return immediately

    async def test_concurrent_warms_for_same_key_coalesce(self) -> None:
        """Two concurrent ``cache.warm`` calls for the same key build once."""
        builder = _ConcurrencyProbingBuilder()
        cache = LLMRailsCache(builder=builder)

        task1 = cache.warm(STABLE_A)
        task2 = cache.warm(STABLE_A)

        # One build enters the gated section; the other blocks on _build_lock.
        await builder.wait_for_in_flight(1)
        assert builder.in_flight == 1, (
            f"second warm must serialize on _build_lock; saw {builder.in_flight} concurrent builds"
        )

        builder.gate.set()
        await asyncio.gather(task1, task2)

        assert builder.calls == 1, "second warm must reuse the idle instance, not rebuild"
        assert builder.peak_in_flight == 1
        pool = cache._pools[STABLE_A.content_hash]
        assert len(pool._idle) == 1

    async def test_warm_after_close_returns_noop_task(self) -> None:
        """Post-close ``warm()`` returns an awaitable no-op task — no build,
        no pool, no entry in the tracking set.

        Closes the TOCTOU window the pre-fix ``close()`` left open: a warm
        scheduled after the snapshot would otherwise run to completion
        against a soon-to-be-cleared cache, wasting build compute and
        potentially establishing external connections (KB / embedding
        clients) for an instance that's immediately GC'd.

        No-op task vs raise: keeps ``await cache.warm(...)`` uniform across
        lifecycles so callers don't need try/except. Lease still raises —
        see :meth:`TestCacheLifecycle.test_lease_after_close_raises`.
        """
        builder = _FakeBuilder()
        cache = LLMRailsCache(builder=builder)
        await cache.close()

        task = cache.warm(STABLE_A)
        assert isinstance(task, asyncio.Task)
        assert await task is None
        assert builder.calls == 0
        assert cache._warm_tasks == set()
        assert cache._pools == {}
