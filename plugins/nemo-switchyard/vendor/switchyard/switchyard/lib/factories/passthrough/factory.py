# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Passthrough :class:`MiddlewareFactory`.

Single source of truth for what a "passthrough" chain contains —
processors, backend, translator — keyed off a typed
:class:`PassthroughConfig`. The factory exposes four granular
part-builders so consumers only pay for what they use:

* NeMo Helix IGW supplies its own backend and only calls
  :meth:`build_request_pipeline` / :meth:`build_response_pipeline`. The
  OpenAI SDK is never imported on this path; tier config on the config is
  inert.
* Standalone Switchyard (the recipe, the CLI, tests) calls
  :meth:`MiddlewareBundle.from_factory` which fans out to all four
  part-builders to assemble a complete chain.

Importing this module registers ``PassthroughFactory`` under the name
``"passthrough"`` in the process-wide registry. Re-imports are a no-op
(``register(replace=False)`` raises on duplicate names; the module-level
side-effect runs exactly once because Python caches module imports).
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, Field, model_validator

from switchyard.lib.backends.backend_tier import BackendTier
from switchyard.lib.backends.openai_llm_backend import OpenAILLMBackend
from switchyard.lib.processors.stats_request_processor import (
    StatsRequestProcessor,
)
from switchyard.lib.processors.stats_response_processor_accumulator import (
    StatsResponseProcessor,
)
from switchyard.lib.registry import BaseMiddlewareFactory, register
from switchyard.lib.request_pipeline import RequestPipeline
from switchyard.lib.response_pipeline import ResponsePipeline
from switchyard.lib.roles import (
    LLMBackend,
    RequestProcessor,
    ResponseProcessor,
    ResponseTranslator,
)
from switchyard.lib.stats_accumulator import StatsAccumulator
from switchyard.lib.translators.default_response_translator import (
    DefaultResponseTranslator,
)


class PassthroughConfig(BaseModel):
    """Configuration for :class:`PassthroughFactory`.

    Builds a single OpenAI-compatible backend tier. Passthrough by
    contract means a single OpenAI-compatible upstream — native Anthropic
    upstream would be its own factory; multi-format routing is RouteLLM's
    or RandomRouting's job.

    Attributes:
        tier: Backend tier configuration (model, api_key, base_url, timeout,
            tuning). The ``backend_format`` must be OPENAI; Anthropic-native
            upstreams would be a separate factory. **Inert** when the host
            calls only the pipeline part-builders (NeMo Helix IGW path).
        enable_stats: When ``True``, the factory wires a
            :class:`StatsRequestProcessor` + :class:`StatsResponseProcessor`
            pair sharing one :class:`StatsAccumulator`, **and** wraps the
            backend in :class:`StatsLLMBackend` so backend-call counters
            and ``routing_overhead_ms`` are recorded. Default ``False``
            so config-driven hosts (NeMo Helix IGW today) don't double-count
            with their own observability stack.
    """

    tier: BackendTier = Field(default_factory=BackendTier)
    enable_stats: bool = False

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_format(cls, data: Any) -> Any:
        """Handle both legacy (api_key/base_url/timeout direct) and new (tier) formats.

        Backward compatibility: if api_key / base_url / timeout are provided
        directly on the config dict, move them into a tier subobject.
        """
        if not isinstance(data, dict):
            return data
        api_key = data.pop("api_key", None)
        base_url = data.pop("base_url", None)
        timeout = data.pop("timeout", None)
        if api_key is not None or base_url is not None or timeout is not None:
            tier_data = data.get("tier", {})
            if isinstance(tier_data, BackendTier):
                tier_data = tier_data.model_dump()
            if api_key is not None:
                tier_data["api_key"] = api_key
            if base_url is not None:
                tier_data["base_url"] = base_url
            if timeout is not None:
                tier_data["timeout"] = timeout
            data["tier"] = tier_data
        return data


class PassthroughFactory(BaseMiddlewareFactory[PassthroughConfig]):
    """Builds a passthrough chain via granular part-builders.

    Inherits ``return None`` defaults for the optional part-builders
    from :class:`BaseMiddlewareFactory` but overrides them — passthrough
    is a full-chain factory. Pure-middleware factories (RouteLLM,
    FormatTranslate) keep the inherited ``None`` defaults.
    """

    name: ClassVar[str] = "passthrough"
    config_class: ClassVar[type[BaseModel]] = PassthroughConfig

    def validate(self, raw: Any) -> PassthroughConfig:
        """Coerce a dict (or pre-built model) into ``PassthroughConfig``."""
        if isinstance(raw, PassthroughConfig):
            return raw
        if isinstance(raw, BaseModel):
            return PassthroughConfig(**raw.model_dump())
        if isinstance(raw, dict):
            return PassthroughConfig(**raw)
        raise TypeError(
            f"PassthroughFactory.validate() expected dict or PassthroughConfig, "
            f"got {type(raw).__name__}"
        )

    # ------------------------------------------------------------------
    # Part-builders
    # ------------------------------------------------------------------

    def build_request_pipeline(self, config: PassthroughConfig) -> RequestPipeline:
        processors: list[RequestProcessor] = []
        if config.enable_stats:
            # The accumulator is shared with the response processor and
            # with ``StatsLLMBackend``; we re-derive it on each part-builder
            # via the per-call ``_stats_state`` cache below so both
            # processors see the same instance.
            processors.append(StatsRequestProcessor())
        return RequestPipeline(processors)

    def build_response_pipeline(self, config: PassthroughConfig) -> ResponsePipeline:
        processors: list[ResponseProcessor] = []
        if config.enable_stats:
            processors.append(StatsResponseProcessor(self._stats_accumulator(config)))
        return ResponsePipeline(processors)

    def build_backend(self, config: PassthroughConfig) -> LLMBackend:
        # Local import: ``OpenAILLMClient`` pulls in the OpenAI SDK; defer
        # until ``build_backend`` is actually called so the IGW path
        # (which only uses ``build_request_pipeline`` /
        # ``build_response_pipeline``) doesn't pay the import cost.
        from switchyard.lib.llm_client import OpenAILLMClient

        client = OpenAILLMClient(
            api_key=config.tier.api_key,
            base_url=config.tier.base_url,
            timeout=config.tier.timeout,
        )
        backend: LLMBackend = OpenAILLMBackend(client)
        if config.enable_stats:
            from switchyard.lib.backends.stats_llm_backend import (
                StatsLLMBackend,
            )

            backend = StatsLLMBackend(backend, self._stats_accumulator(config))
        return backend

    def build_translator(self, config: PassthroughConfig) -> ResponseTranslator:  # noqa: ARG002
        return DefaultResponseTranslator()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stats_accumulator(self, config: PassthroughConfig) -> StatsAccumulator:
        """Return the stats accumulator shared by this build pass.

        The response processor and ``StatsLLMBackend`` must point at the
        *same* accumulator so token counts, latency, and backend-call
        counters land in one bucket. ``MiddlewareBundle.from_factory``
        calls the part-builders sequentially within a single build pass;
        we cache the accumulator on the factory keyed by the config
        identity so both lookups during that pass return the same
        instance. Subsequent build passes get a fresh accumulator.
        """
        cache: dict[int, StatsAccumulator] = self.__dict__.setdefault(
            "_accumulator_cache", {},
        )
        return cache.setdefault(id(config), StatsAccumulator())


register(PassthroughFactory())
