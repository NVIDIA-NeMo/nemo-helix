# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test: real ``LLMRails`` build from a stabilized rails config.

This file is intentionally separate from ``test_llmrails_cache.py`` (which
uses a fake builder) because it actually calls
``nemoguardrails.rails.llm.llmrails.LLMRails`` to construct an instance —
exercising the full ``__init__`` path against a config that has had its
``main`` model entry stripped by :func:`stabilize`.

The concern this file pins down is the Phase-1 RFC trade-off: cached configs
no longer contain a ``main`` model, so ``_init_llms`` runs with
``self.llm = None`` and downstream ``LLMGenerationActions(llm=None)`` is
constructed before any per-request ``update_llm`` swap. Upstream tolerates
this today (it just stores the ``None`` and uses it lazily at action time).
This regression test catches an upstream change that would break Phase 1's
core invariant.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.base import BaseLanguageModel
from nemo_guardrails_plugin.llmrails_cache import (
    DefaultLLMRailsBuilder,
    LLMRailsCache,
    Provenance,
    stabilize,
)
from nemo_platform_plugin.guardrail.types import RailsConfig as PlatformRailsConfig
from nemo_platform_plugin.inference_middleware import OpenAICompatibleInferenceTarget
from nemoguardrails.integrations.langchain.llm_adapter import LangChainLLMAdapter
from nemoguardrails.rails.llm.llmrails import LLMRails

pytestmark = [pytest.mark.integration]


def _resolve_target(_model_id: str) -> OpenAICompatibleInferenceTarget:
    return OpenAICompatibleInferenceTarget(
        openai_base_url="http://igw.example/provider/v1",
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
    )


def _platform_rails(models: list[dict[str, Any]] | None = None) -> PlatformRailsConfig:
    """Minimal :class:`PlatformRailsConfig` — no flows, so no library prompts required."""
    return PlatformRailsConfig.model_validate(
        {
            "rails": {"input": {"flows": []}, "output": {"flows": []}},
            "models": models if models is not None else [],
        }
    )


def _assert_wraps_langchain_llm(actual: object, expected: BaseLanguageModel) -> None:
    assert isinstance(actual, LangChainLLMAdapter)
    assert actual._llm is expected


# ---------------------------------------------------------------------------
# LLMRails.__init__ tolerates a "main-less" cached config
# ---------------------------------------------------------------------------


class TestRealLLMRailsBuildFromStableConfig:
    """Build a real ``LLMRails`` from :func:`stabilize` output.

    These tests load and parse the entire NeMo Guardrails Colang library on
    every build (Phase 1 has no library cache yet). They are slower than
    the rest of the pool unit tests but each one runs in a few hundred ms
    on a warm interpreter, which is acceptable for a smoke-level guarantee.
    """

    def test_builds_with_no_models_at_all(self) -> None:
        """A config with no ``models`` produces a buildable stable rails config."""
        stable = stabilize(_platform_rails([]), _resolve_target)

        rails = LLMRails(config=stable.rails)

        # No main model in cached config → __init__ leaves self.llm == None.
        # update_llm() at acquire time will set it.
        assert rails.llm is None
        assert rails.llm_generation_actions.llm is None
        # The request action param is therefore not registered yet.
        assert "llm" not in rails.runtime.registered_action_params

    def test_builds_with_only_a_main_model_in_source_config(self) -> None:
        """The most common case: source config has only a ``main`` entry, which
        :func:`stabilize` strips. Nothing left in ``stable.rails.models``, but
        ``LLMRails`` still builds."""
        stable = stabilize(
            _platform_rails([{"type": "main", "engine": "nim", "model": "ws/llama"}]),
            _resolve_target,
        )

        assert stable.rails.models == []
        rails = LLMRails(config=stable.rails)

        assert rails.llm is None
        assert rails.llm_generation_actions.llm is None

    def test_update_llm_seeds_main_after_build(self) -> None:
        """After build, ``update_llm(main_llm)`` must wire the LLM into all
        three places ``LLMRails.update_llm`` documents:
        - ``self.llm``
        - ``self.llm_generation_actions.llm``
        - ``runtime.action_param("llm")``
        """
        stable = stabilize(_platform_rails([]), _resolve_target)
        rails = LLMRails(config=stable.rails)

        main_llm = MagicMock(spec=BaseLanguageModel, name="main_llm")
        rails.update_llm(main_llm)

        _assert_wraps_langchain_llm(rails.llm, main_llm)
        _assert_wraps_langchain_llm(rails.llm_generation_actions.llm, main_llm)
        _assert_wraps_langchain_llm(rails.runtime.registered_action_params["llm"], main_llm)


# ---------------------------------------------------------------------------
# Full pool path with the real DefaultLLMRailsBuilder
# ---------------------------------------------------------------------------


class TestCacheWithRealBuilder:
    """End-to-end: real ``DefaultLLMRailsBuilder`` running through ``LLMRailsCache``.

    Validates that Phase 1's cache plumbing — lease → build → reset → return —
    composes correctly with an actual ``LLMRails`` (not a SimpleNamespace stub).
    """

    async def test_lease_builds_real_llmrails_and_seeds_main_llm(self) -> None:
        stable = stabilize(_platform_rails([]), _resolve_target)
        cache = LLMRailsCache(builder=DefaultLLMRailsBuilder())

        main_llm = MagicMock(spec=BaseLanguageModel, name="main_llm")
        try:
            async with cache.lease(stable, main_llm=main_llm, provenance=Provenance("test")) as rails:
                assert isinstance(rails, LLMRails)
                _assert_wraps_langchain_llm(rails.llm, main_llm)
                _assert_wraps_langchain_llm(rails.llm_generation_actions.llm, main_llm)
                # lease's reset path wiped these.
                assert rails.events_history_cache == {}
                assert rails.explain_info is None
        finally:
            await cache.close()

    async def test_reuse_swaps_in_a_fresh_main_llm(self) -> None:
        """A cached instance reused across two requests must observe the
        second request's ``main_llm``, not the first's."""
        stable = stabilize(_platform_rails([]), _resolve_target)
        cache = LLMRailsCache(builder=DefaultLLMRailsBuilder())

        first_llm = MagicMock(spec=BaseLanguageModel, name="first")
        second_llm = MagicMock(spec=BaseLanguageModel, name="second")

        try:
            async with cache.lease(stable, main_llm=first_llm) as rails_a:
                first_id = id(rails_a)

            async with cache.lease(stable, main_llm=second_llm) as rails_b:
                assert id(rails_b) == first_id, "cache should reuse the pooled instance"
                _assert_wraps_langchain_llm(rails_b.llm, second_llm)
                _assert_wraps_langchain_llm(rails_b.llm_generation_actions.llm, second_llm)
                _assert_wraps_langchain_llm(rails_b.runtime.registered_action_params["llm"], second_llm)
        finally:
            await cache.close()


# ---------------------------------------------------------------------------
# Sanity test: should we ever decide to keep a placeholder main entry instead,
# this test would start failing — making the design choice explicit.
# ---------------------------------------------------------------------------


def test_phase_1_strips_main_from_stable_config_by_design() -> None:
    """Document Phase 1's split: the main model lives outside the cached config.

    If this fails, someone has reintroduced a ``main`` entry into the cached
    config (e.g. via a placeholder approach). That's a valid design choice,
    but it changes both the cache-key story (the placeholder name has to be
    invariant across requests) and the ``update_llm`` contract — so it must
    be a deliberate change, not accidental drift.
    """
    stable = stabilize(
        _platform_rails([{"type": "main", "engine": "nim", "model": "ws/llama"}]),
        _resolve_target,
    )
    main_entries = [m for m in stable.rails.models if m.type == "main"]
    assert main_entries == [], "stabilize should strip the main entry; reintroduce via a separate, intentional change."


# ---------------------------------------------------------------------------
# Additional smoke: a config with a non-main action model also builds.
# ---------------------------------------------------------------------------


def test_builds_with_action_only_models() -> None:
    """A config carrying only non-main entries (``content_safety``, ``topic_safety``)
    must build cleanly. ``_init_llms`` will instantiate those LangChain clients
    against the resolved IGW URL — they don't make network calls at construction
    time, so this remains a pure CPU test."""
    stable = stabilize(
        _platform_rails(
            [
                {
                    "type": "content_safety",
                    "engine": "nim",
                    "model": "default/safety",
                },
            ]
        ),
        _resolve_target,
    )

    # `_init_llms` may raise if the engine "nim" provider can't initialize.
    # If it does, that's a Phase-1 blocker we want to know about NOW, not in prod.
    rails = LLMRails(config=stable.rails)
    assert hasattr(rails, "content_safety_llm")


@pytest.mark.parametrize("colang_version", ["1.0"])
def test_supports_colang_version(colang_version: str) -> None:
    """Pin the Colang versions Phase 1 promises to support.

    Phase 1 only ships Colang 1.0 support. If we add 2.x to this parametrize,
    the upstream ``LLMGenerationActionsV2dotx`` path also needs to tolerate
    ``llm=None`` at ``__init__`` (it does today; verify per upstream bump).
    """
    platform_rails_config = PlatformRailsConfig.model_validate(
        {
            "rails": {"input": {"flows": []}, "output": {"flows": []}},
            "models": [],
            "colang_version": colang_version,
        }
    )
    stable = stabilize(platform_rails_config, _resolve_target)
    rails = LLMRails(config=stable.rails)
    assert rails.config.colang_version == colang_version
