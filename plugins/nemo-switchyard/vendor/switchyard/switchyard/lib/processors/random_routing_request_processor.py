# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Random-routing :class:`RequestProcessor` for the IGW path.

The routing decision happens in this request processor, decoupled from
backend ownership. Standalone deployments can use this processor with
their own backend; NeMo Helix IGW hosts use it with their backends.

This processor is the IGW counterpart of
:meth:`RandomRoutingLLMBackend._pick_tier` plus the request-mutation
work. It does **not** call any LLM — it just:

1. Flips a coin against ``config.strong_probability``.
2. Picks the corresponding tier (``config.strong`` or ``config.weak``).
3. Rewrites ``request.body["model"]`` to the picked tier's model name.
4. If the picked tier declares a ``backend_format``, stamps
   ``ctx.metadata[CTX_TARGET_FORMAT]`` so the downstream (separate)
   :class:`FormatTranslateRequestProcessor` can translate the request.
5. Records the picked tier label / model on ``ctx.metadata`` for
   logging / stats consumers.

The host's backend (IGW today) makes the actual call. Format translation
is handled by a separate middleware (``TranslateFactory``), allowing
Platform to compose routing and translation independently.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from switchyard.lib.proxy_context import CTX_PROXY_ACTUAL_MODEL

if TYPE_CHECKING:
    from switchyard.lib.chat_request.base import ChatRequest
    from switchyard.lib.factories.random_routing import (
        RandomRoutingConfig,
    )
    from switchyard.lib.proxy_context import ProxyContext

from switchyard.lib.roles import RequestProcessor

log = logging.getLogger(__name__)


class RandomRoutingRequestProcessor(RequestProcessor):
    """Pick a tier per request and stamp routing decision into context.

    Implements weighted-coin routing: ``rng.random() < strong_probability``
    so ``p=0.0`` always picks weak and ``p=1.0`` always picks strong. The
    pre-built ``rng`` argument exists for deterministic tests; production
    callers leave it ``None`` to get a fresh per-instance RNG seeded
    from ``config.rng_seed`` (or system entropy if that's also ``None``).
    """

    def __init__(
        self,
        config: RandomRoutingConfig,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self._config = config
        # Identical seeding logic to ``RandomRoutingLLMBackend.__init__``:
        # explicit instance > config seed > system entropy.
        if rng is not None:
            self._rng = rng
        elif config.rng_seed is not None:
            self._rng = random.Random(config.rng_seed)
        else:
            self._rng = random.Random()

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        use_strong = self._rng.random() < self._config.strong_probability
        tier_label = "strong" if use_strong else "weak"
        chosen_tier = self._config.strong if use_strong else self._config.weak

        # Every concrete ``ChatRequest`` exposes ``body`` via its
        # provider ``TypedDict``; the abstract base doesn't declare it.
        request.body["model"] = chosen_tier.model  # type: ignore[attr-defined]

        ctx.metadata["_random_routing_tier"] = tier_label
        ctx.metadata[CTX_PROXY_ACTUAL_MODEL] = chosen_tier.model

        log.debug(
            "RandomRoutingRequestProcessor: picked tier=%s model=%s",
            tier_label, chosen_tier.model,
        )
        return request
