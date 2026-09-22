# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Return value of :meth:`MiddlewareFactory.build`.

A :class:`MiddlewareBundle` carries everything a factory contributes to a
host's chain: the request pipeline, the response pipeline, and
**optionally** an :class:`LLMBackend` and a :class:`ResponseTranslator`.
Hosts that own their own backend (NeMo Helix IGW today) ignore the optional
slots and slot the request/response pipelines around their own backend.
Standalone callers (the Switchyard CLI, tests, internal recipes) can
ask the factory for a full chain by populating all four slots and then
call :meth:`MiddlewareBundle.to_switchyard` to get a runnable
:class:`Switchyard`.

Why a dataclass and not a 4-tuple:

* Named slots make it obvious which factories ship a backend and which
  don't — important now that we may have format-translating factories
  (SWITCH-167) that own their own backend selection.
* Optional fields give factories room to grow without breaking the
  protocol (e.g. a factory that contributes both pipelines and a
  pre-validated translator is just three populated fields, not a
  signature change).
* Lets us add helpers — ``to_switchyard`` here, lifecycle wiring
  later — on a single type instead of free functions taking a tuple.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel

    from switchyard.lib.registry import MiddlewareFactory
    from switchyard.lib.request_pipeline import RequestPipeline
    from switchyard.lib.response_pipeline import ResponsePipeline
    from switchyard.lib.roles import LLMBackend, ResponseTranslator
    from switchyard.lib.switchyard import Switchyard


@dataclass(frozen=True)
class MiddlewareBundle:
    """Pipelines (and optionally backend + translator) produced by a factory.

    Attributes:
        request_pipeline: Request-side processors. Always present; may be
            empty.
        response_pipeline: Response-side processors. Always present; may
            be empty.
        backend: Optional :class:`LLMBackend`. ``None`` when the host
            (e.g. NeMo Helix IGW) supplies its own backend; populated when the
            factory ships a full chain (e.g. a routing factory that
            picks among its own backends).
        translator: Optional :class:`ResponseTranslator`. ``None`` when
            the host owns translation; populated when the factory ships
            a complete chain.
    """

    request_pipeline: RequestPipeline
    response_pipeline: ResponsePipeline
    backend: LLMBackend | None = None
    translator: ResponseTranslator | None = None

    @classmethod
    def from_factory(
        cls,
        factory: MiddlewareFactory[Any],
        config: BaseModel,
    ) -> MiddlewareBundle:
        """Assemble a bundle by calling all four part-builders on ``factory``.

        Use this from standalone-Switchyard call sites (recipes, the
        Switchyard CLI, tests) that want a full chain. Hosts that only
        need pipelines (NeMo Helix IGW) should call
        :meth:`MiddlewareFactory.build_request_pipeline` /
        :meth:`MiddlewareFactory.build_response_pipeline` directly so
        they don't pay the cost of constructing a backend they're going
        to throw away.
        """
        return cls(
            request_pipeline=factory.build_request_pipeline(config),
            response_pipeline=factory.build_response_pipeline(config),
            backend=factory.build_backend(config),
            translator=factory.build_translator(config),
        )

    def to_switchyard(self) -> Switchyard:
        """Build a runnable :class:`Switchyard` from this bundle.

        Requires both :attr:`backend` and :attr:`translator` to be
        populated — a full chain needs all four roles. Raises
        ``ValueError`` otherwise so misuse fails fast at the call site.
        """
        from switchyard.lib.switchyard import Switchyard

        if self.backend is None or self.translator is None:
            missing = [
                slot
                for slot, value in (
                    ("backend", self.backend),
                    ("translator", self.translator),
                )
                if value is None
            ]
            raise ValueError(
                f"MiddlewareBundle.to_switchyard() requires {missing} to be set"
            )
        return Switchyard(
            request_processors=list(self.request_pipeline.processors) or None,
            backend=self.backend,
            response_processors=list(self.response_pipeline.processors) or None,
            translator=self.translator,
        )
