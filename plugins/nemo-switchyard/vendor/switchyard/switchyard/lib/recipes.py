# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-built ``Switchyard`` configurations for common use cases.

Each static factory method returns a ready-to-use ``Switchyard`` with a
sensible chain preset (request processors + backend + response processors
+ translator), so callers don't have to hand-assemble the four roles for
common setups.

Example::

    from switchyard import SwitchyardRecipes, build_switchyard_app
    import uvicorn

    switchyard = SwitchyardRecipes.passthrough_recipe(
        api_key="sk-...",
        base_url="https://api.openai.com/v1",
    )
    uvicorn.run(build_switchyard_app(switchyard), port=4000)
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TYPE_CHECKING

from switchyard.lib.middleware_bundle import MiddlewareBundle
from switchyard.lib.roles import RequestProcessor, ResponseProcessor
from switchyard.lib.switchyard import Switchyard

if TYPE_CHECKING:
    from switchyard.lib.backends.backend_tier import BackendTier
    from switchyard.lib.factories.latency_service import (
        LatencyServiceEndpoint,
    )
    from switchyard.lib.factories.oss_router import OSSRouterConfig
    from switchyard.lib.factories.random_routing import (
        RandomRoutingConfig,
    )
    from switchyard.lib.stats_accumulator import StatsAccumulator


def _bundle_to_switchyard(
    bundle: MiddlewareBundle,
    *,
    extra_request_processors: Sequence[RequestProcessor] = (),
    extra_response_processors: Sequence[ResponseProcessor] = (),
) -> Switchyard:
    """Convert a full bundle to Switchyard, appending processor middleware."""
    if bundle.backend is None or bundle.translator is None:
        missing = [
            slot
            for slot, value in (
                ("backend", bundle.backend),
                ("translator", bundle.translator),
            )
            if value is None
        ]
        raise ValueError(f"Recipe bundle requires {missing} to be set")

    request_processors = [
        *bundle.request_pipeline.processors,
        *extra_request_processors,
    ]
    response_processors = [
        *bundle.response_pipeline.processors,
        *extra_response_processors,
    ]
    return Switchyard(
        request_processors=request_processors or None,
        backend=bundle.backend,
        response_processors=response_processors or None,
        translator=bundle.translator,
    )


class SwitchyardRecipes:
    """Factory methods returning pre-configured ``Switchyard`` instances.

    Each recipe encapsulates a full V2 chain for a specific use case.
    Power users who need bespoke processor chains should drop to the raw
    ``Switchyard(...)`` constructor; recipes are the shortcut for the
    common paths.

    Example::

        switchyard = SwitchyardRecipes.passthrough_recipe(
            api_key="sk-...",
            base_url="https://api.openai.com/v1",
        )
        response = await switchyard.call(request)
    """

    # ------------------------------------------------------------------
    # Passthrough — single OpenAI-compatible backend, default translator
    # ------------------------------------------------------------------

    @staticmethod
    def passthrough_recipe(
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        enable_stats: bool = True,
        extra_request_processors: Sequence[RequestProcessor] = (),
        extra_response_processors: Sequence[ResponseProcessor] = (),
    ) -> Switchyard:
        """Direct proxy to an OpenAI-compatible endpoint.

        Chain: ``OpenAILLMBackend`` → ``DefaultResponseTranslator``

        Serves OpenAI Chat Completions, OpenAI Responses, and Anthropic
        Messages inbound formats simultaneously — the backend translates
        non-OpenAI requests to Chat Completions via
        :class:`ChatRequestTranslationEngine`, and the translator converts
        the response back to the client's original format via
        :class:`ChatResponseTranslationEngine`.

        Thin wrapper around :class:`PassthroughFactory` — the factory is
        the single source of truth for what a passthrough chain contains.
        This recipe just maps kwargs to a :class:`PassthroughConfig` and
        unwraps the resulting bundle into a runnable ``Switchyard``.

        Args:
            api_key: API key for the backend LLM provider.
            base_url: Base URL for the backend LLM API (include ``/v1``).
            timeout: Request timeout in seconds, forwarded to the
                underlying ``OpenAILLMClient``.
            enable_stats: Wire stats processors + ``StatsLLMBackend``.
                Defaults to ``True`` here (recipe path) where embedded
                Python users typically want switchyard's stats endpoint;
                the factory's :class:`PassthroughConfig` defaults to
                ``False`` so config-driven hosts (NeMo Helix IGW) don't
                double-count with their own observability.
        """
        # Local import: avoids pulling the factory (and the OpenAI SDK
        # via its ``OpenAILLMClient`` import inside ``build_backend``)
        # into module load, and avoids a circular import since
        # ``factories.passthrough`` imports from ``foundation.infra``.
        from switchyard.lib.backends.backend_tier import BackendTier
        from switchyard.lib.factories.passthrough.factory import (
            PassthroughConfig,
            PassthroughFactory,
        )

        config = PassthroughConfig(
            tier=BackendTier(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
            ),
            enable_stats=enable_stats,
        )
        return _bundle_to_switchyard(
            MiddlewareBundle.from_factory(PassthroughFactory(), config),
            extra_request_processors=extra_request_processors,
            extra_response_processors=extra_response_processors,
        )

    @staticmethod
    def backend_tier_recipe(
        tier: BackendTier,
        *,
        enable_stats: bool = True,
        stats_accumulator: StatsAccumulator | None = None,
        extra_request_processors: Sequence[RequestProcessor] = (),
        extra_response_processors: Sequence[ResponseProcessor] = (),
    ) -> Switchyard:
        """Direct proxy to one generic backend tier.

        Unlike :meth:`passthrough_recipe`, this honors
        :attr:`BackendTier.backend_format` and :attr:`BackendTier.tuning`, so it
        can build OpenAI-compatible, Anthropic-native, or auto-resolved tiers.
        """
        from switchyard.lib.backends.multi_llm_backend import MultiLLMBackend
        from switchyard.lib.backends.stats_llm_backend import StatsLLMBackend
        from switchyard.lib.processors.stats_request_processor import (
            StatsRequestProcessor,
        )
        from switchyard.lib.processors.stats_response_processor_accumulator import (
            StatsResponseProcessor,
        )
        from switchyard.lib.stats_accumulator import StatsAccumulator
        from switchyard.lib.translators.default_response_translator import (
            DefaultResponseTranslator,
        )

        stats = stats_accumulator or StatsAccumulator()
        request_processors: list[RequestProcessor] = []
        response_processors: list[ResponseProcessor] = []
        if enable_stats:
            request_processors.append(StatsRequestProcessor())
            response_processors.append(StatsResponseProcessor(stats))
        request_processors.extend(extra_request_processors)
        response_processors.extend(extra_response_processors)

        backend = MultiLLMBackend.build_backend(tier)
        if enable_stats:
            backend = StatsLLMBackend(backend, stats)

        return Switchyard(
            request_processors=request_processors or None,
            backend=backend,
            response_processors=response_processors or None,
            translator=DefaultResponseTranslator(),
        )

    # ------------------------------------------------------------------
    # Latency Service — health-aware routing across many endpoints
    # ------------------------------------------------------------------

    @staticmethod
    def latency_service_recipe(
        *,
        latency_service_url: str,
        endpoints: list[LatencyServiceEndpoint],
        poll_interval_s: float = 10.0,
        poll_timeout_s: float = 5.0,
        max_retries: int = 2,
    ) -> Switchyard:
        """Route across many endpoints by Latency Service health verdicts.

        Chain: ``LatencyServiceLLMBackend`` → ``DefaultResponseTranslator``

        The V2 counterpart to V1's ``latency_service_routing_recipe``.
        Designed for Inference Hub deployments where a central Latency
        Service owns heartbeat probing and statistical profiling; a
        background poller inside the backend caches health verdicts so
        the request hot path never blocks on a network call.

        Like ``passthrough_recipe`` this serves OpenAI Chat Completions,
        OpenAI Responses, and Anthropic Messages inbound formats
        simultaneously — the backend translates non-OpenAI requests to
        Chat Completions via :class:`ChatRequestTranslationEngine`, and
        the translator converts the response back via
        :class:`ChatResponseTranslationEngine`.

        Args:
            latency_service_url: Base URL of the Latency Service
                (e.g. ``"http://latency-service.inference-hub.svc:8080"``).
            endpoints: LLM backends to route across.  Each must have a
                unique ``model`` — it's the routing + health-lookup key.
            poll_interval_s: Health poll interval in seconds.
            poll_timeout_s: Timeout for each health API call.
            max_retries: Max retries on a different endpoint per request.
        """
        # Local import to avoid pulling the usage case (and its
        # ``httpx.Client``) into ``SwitchyardRecipes`` at module load
        # time.  Parallels ``passthrough_recipe``'s local ``OpenAILLMClient``
        # import for the same reason.
        from switchyard.lib.config import LatencyServiceBackendConfig
        from switchyard.lib.factories.latency_service import LatencyServiceFactory

        config = LatencyServiceBackendConfig(
            latency_service_url=latency_service_url,
            endpoints=endpoints,
            poll_interval_s=poll_interval_s,
            poll_timeout_s=poll_timeout_s,
            max_retries=max_retries,
        )
        return _bundle_to_switchyard(
            MiddlewareBundle.from_factory(LatencyServiceFactory(), config)
        )

    # ------------------------------------------------------------------
    # Random Routing — weighted-coin routing across two tiers
    # ------------------------------------------------------------------

    @staticmethod
    def random_routing_recipe(
        config: RandomRoutingConfig,
        *,
        rng: random.Random | None = None,
        preset: str | None = None,
        stats_accumulator: StatsAccumulator | None = None,
        pre_routing_request_processors: Sequence[RequestProcessor] = (),
        extra_request_processors: Sequence[RequestProcessor] = (),
        extra_response_processors: Sequence[ResponseProcessor] = (),
    ) -> Switchyard:
        """Weighted-coin routing across two configured tiers.

        Chain: ``RandomRoutingLLMBackend → DefaultResponseTranslator``.

        V2 equivalent of V1's ``claude_code_recipe`` random path, minus
        the RouteLLM / HuggingFace / LiteLLM stack.  The
        :class:`RandomRoutingConfig` bundles both tiers and the coin
        bias; :class:`RandomRoutingLLMBackend` dispatches each tier to
        the appropriate concrete backend (``OpenAILLMBackend`` for
        :attr:`BackendFormat.OPENAI`, ``AnthropicNativeLLMBackend`` for
        :attr:`BackendFormat.ANTHROPIC`) internally.

        Like :meth:`passthrough_recipe` this serves OpenAI Chat
        Completions, OpenAI Responses, and Anthropic Messages inbound
        formats simultaneously — each tier's inner backend runs its own
        :class:`ChatRequestTranslationEngine` normalisation, so the
        picked wire format is decoupled from the inbound client format.

        Migration note: V1 ``router_type="random"`` used an inverted
        ``threshold`` (higher value = more weak traffic).  The V2
        ``RandomRoutingConfig.strong_probability`` knob has the
        intuitive polarity (higher = more strong).  To port a V1
        config: ``strong_probability = 1.0 - v1_threshold``.

        Example — tiers on the same OpenAI-compat backend::

            from switchyard import (
                BackendFormat, BackendTier, RandomRoutingConfig,
                SwitchyardRecipes,
            )

            config = RandomRoutingConfig(
                strong=BackendTier(
                    model="openai/openai/openai/gpt-5.2",
                    backend_format=BackendFormat.OPENAI,
                    api_key=nvidia_api_key,
                    base_url="https://inference-api.nvidia.com/v1",
                ),
                weak=BackendTier(
                    model="openai/nvidia/nvidia/nemotron-3-super-v3",
                    backend_format=BackendFormat.OPENAI,
                    api_key=nvidia_api_key,
                    base_url="https://inference-api.nvidia.com/v1",
                ),
                strong_probability=0.5,
            )
            switchyard = SwitchyardRecipes.random_routing_recipe(config)

        Example — heterogeneous tiers (Anthropic-native strong +
        OpenAI-compat weak)::

            config = RandomRoutingConfig(
                strong=BackendTier(
                    model="claude-opus-4-6",
                    backend_format=BackendFormat.ANTHROPIC,
                    api_key=anthropic_key,
                    base_url="https://api.anthropic.com",
                ),
                weak=BackendTier(
                    model="nvidia/moonshotai/kimi-k2.5",
                    backend_format=BackendFormat.OPENAI,
                    api_key=nvidia_key,
                    base_url="https://inference-api.nvidia.com/v1",
                ),
                strong_probability=0.5,
            )

        Args:
            config: The full routing configuration (both tiers + coin
                bias).  Constructed via :class:`RandomRoutingConfig`.
            rng: Optional pre-seeded :class:`random.Random` for
                deterministic tests or reproducible A/B runs.  ``None``
                creates a fresh per-instance RNG.
            preset: Optional name of the
                :class:`RandomRoutingPresets` factory that produced
                ``config``.  Surfaced back through
                ``GET /v1/routing/stats`` as
                ``_server_config.preset`` so saved ``stats.json``
                files self-document which shipping pair was used
                (``None`` when the router was built from raw flags).
            pre_routing_request_processors: Request processors that must run
                after stats start-time stamping but before the router mutates
                ``request.body["model"]``.
        """
        # Local import: avoids pulling the factory (and the OpenAI /
        # Anthropic SDKs via the multi-tier backend) into module load.
        # Mirrors ``passthrough_recipe``.
        from switchyard.lib.factories.random_routing.factory import (
            RandomRoutingFactory,
        )

        if preset is not None and config.preset != preset:
            config = config.model_copy(update={"preset": preset})

        # Recipe path supports a pre-built ``random.Random`` instance
        # (Python-only ergonomic for tests / reproducible A/B runs).
        # ``RandomRoutingConfig.rng_seed`` is the serializable
        # equivalent for config-driven hosts; the instance kwarg
        # is now converted to rng_seed for the factory path.
        if rng is not None:
            import warnings

            warnings.warn(
                "Passing a pre-built random.Random instance to random_routing_recipe() "
                "is deprecated. Use RandomRoutingConfig.rng_seed instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Get a seed from the pre-built RNG and use the factory path with rng_seed.
            # This ensures stats processors and factory-level wiring are used.
            seed = rng.randint(0, 2**32 - 1)
            config = config.model_copy(update={"rng_seed": seed})

        return _bundle_to_switchyard(
            MiddlewareBundle.from_factory(
                RandomRoutingFactory(
                    stats_accumulator=stats_accumulator,
                    pre_routing_request_processors=pre_routing_request_processors,
                ),
                config,
            ),
            extra_request_processors=extra_request_processors,
            extra_response_processors=extra_response_processors,
        )

    @staticmethod
    def oss_router_recipe(
        config: OSSRouterConfig,
        *,
        stats_accumulator: StatsAccumulator | None = None,
        pre_routing_request_processors: Sequence[RequestProcessor] = (),
        extra_request_processors: Sequence[RequestProcessor] = (),
        extra_response_processors: Sequence[ResponseProcessor] = (),
    ) -> Switchyard:
        """Routing chain steered by an external OSS-router plugin.

        Spawns the plugin subprocess at startup, calls it once per
        request to pick a tier, and dispatches through the existing
        :class:`OSSRouterLLMBackend`. The plugin is shut down with the
        rest of the chain via the standard component-lifecycle hooks.

        Chain shape::

            [StatsRequestProcessor?,
             *pre_routing_request_processors,
             PluginRoutingRequestProcessor]
                ↓
            OSSRouterLLMBackend (+ stats wrapper if enabled)
                ↓
            [StatsResponseProcessor?, *extra_response_processors]
                ↓
            DefaultResponseTranslator

        Example::

            from switchyard import (
                BackendFormat, BackendTier,
                OSSRouterConfig, OSSRouterTier,
                SwitchyardRecipes,
            )

            config = OSSRouterConfig(
                plugin_command=["python", "-u", "examples/oss-routers/echo_router/echo_router.py", "--tier", "weak"],
                tiers=(
                    OSSRouterTier(label="strong", tier=BackendTier(
                        model="openai/openai/gpt-5.2",
                        backend_format=BackendFormat.OPENAI,
                        api_key=nvidia_api_key,
                        base_url="https://integrate.api.nvidia.com/v1",
                    )),
                    OSSRouterTier(label="weak", tier=BackendTier(
                        model="nvidia/moonshotai/kimi-k2.6",
                        backend_format=BackendFormat.OPENAI,
                        api_key=nvidia_api_key,
                        base_url="https://integrate.api.nvidia.com/v1",
                    )),
                ),
                fallback_tier="weak",
            )
            switchyard = SwitchyardRecipes.oss_router_recipe(config)

        Args:
            config: Plugin command + tier list + safety knobs (timeouts,
                fallback tier). See :class:`OSSRouterConfig`.
            stats_accumulator: Optional shared accumulator so an external
                stats endpoint and the chain land in the same bucket.
            pre_routing_request_processors: Processors that run after
                stats start-time stamping but before the plugin call.
            extra_request_processors: Processors appended after the
                plugin call (rare — most callers want pre-routing
                processors).
            extra_response_processors: Processors appended after the
                stats response processor.
        """
        # Local import: pulls in the factory + (transitively) the
        # multi-tier backend and provider SDKs only when this recipe
        # is actually called.
        from switchyard.lib.factories.oss_router.factory import (
            OSSRouterFactory,
        )

        return _bundle_to_switchyard(
            MiddlewareBundle.from_factory(
                OSSRouterFactory(
                    stats_accumulator=stats_accumulator,
                    pre_routing_request_processors=pre_routing_request_processors,
                ),
                config,
            ),
            extra_request_processors=extra_request_processors,
            extra_response_processors=extra_response_processors,
        )
