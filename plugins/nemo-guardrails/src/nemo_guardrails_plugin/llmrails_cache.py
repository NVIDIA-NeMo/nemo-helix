# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end LLMRails cache: source pipeline → stable transform → pool.

Kept in one file because the pipeline has no independent reuse — every
consumer walks the whole chain.

Pipeline:

1. **Source** (:class:`GuardrailConfigSource` = :class:`EntityGuardrailConfigSource` |
   :class:`InlineGuardrailConfigSource`). The IGW resolver boundary sets the discriminator
   exactly once; downstream code reads ``source.rails`` uniformly.
2. **Stable** (:class:`StableRailsConfig` from :func:`stabilize`). Validates
   platform → library, strips the per-request ``main`` model, resolves non-main
   ``base_url``, and hashes canonical JSON for cache identity.
   :class:`StabilizedRailsConfigCache` memoizes the transform by entity identity.
3. **Cache** (:class:`LLMRailsCache` over :class:`Pool`). LRU-bounded
   ``content_hash → Pool``. Pools grow on demand and shrink on quiet.
   Refcount-pinning keeps in-flight pools off the eviction path; per-lease
   reset wipes shared mutable state so tenants can't leak into each other.

The :class:`LLMRailsBuilder` protocol is the seam for swapping in a Phase 2
builder that shares parsed Colang and :class:`KnowledgeBase` across instances.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol, cast

from nemo_platform_plugin.guardrail.types import OutputRailsStreamingConfig
from nemo_platform_plugin.guardrail.types import RailsConfig as PlatformRailsConfig
from nemo_platform_plugin.inference_middleware import OpenAICompatibleInferenceTarget
from nemoguardrails import RailsConfig as LibraryRailsConfig
from nemoguardrails.rails.llm.config import Model
from nemoguardrails.rails.llm.llmrails import LLMRails
from nemoguardrails.types import LLMModel
from pydantic import ValidationError

logger = logging.getLogger(__name__)


# =============================================================================
# Stage 1: discriminated guardrail source
# =============================================================================


@dataclass(frozen=True, slots=True)
class EntityGuardrailConfigSource:
    """Guardrails config resolved from the entity store via ``config_id``.

    Identity fields key :class:`StabilizedRailsConfigCache` memoization and feed
    log labels. Cache identity is the content hash of ``rails``, not these.
    """

    workspace: str
    name: str
    updated_at: str
    rails: PlatformRailsConfig


@dataclass(frozen=True, slots=True)
class InlineGuardrailConfigSource:
    """Guardrails config supplied inline via ``MiddlewareCall.config``.

    No entity metadata, so no stabilize memoization and no warming. ``label``
    is a diagnostic name for logs only.
    """

    rails: PlatformRailsConfig
    label: str | None = None


GuardrailConfigSource = EntityGuardrailConfigSource | InlineGuardrailConfigSource
"""Discriminated union from the IGW resolver methods.

Both arms expose ``.rails: PlatformRailsConfig`` so callers that only need the
payload can read the union directly.
"""


@dataclass(frozen=True, slots=True)
class Provenance:
    """Diagnostic label for a :class:`GuardrailConfigSource`. Not cache identity."""

    label: str


def provenance_of(source: GuardrailConfigSource) -> Provenance:
    """Render the diagnostic label for ``source``.

    :class:`EntityGuardrailConfigSource` → ``"workspace/name@updated_at"``;
    :class:`InlineGuardrailConfigSource` → ``"<inline:label>"`` (``"unnamed"`` if no label).
    """
    if isinstance(source, EntityGuardrailConfigSource):
        return Provenance(f"{source.workspace}/{source.name}@{source.updated_at}")
    return Provenance(f"<inline:{source.label or 'unnamed'}>")


def wire_config_id(source: GuardrailConfigSource) -> str:
    """Stable, externally-meaningful identifier for ``source``.

    Surfaced in ``guardrails_data.config_ids`` and other response payloads.
    Distinct from :func:`provenance_of`: provenance varies per config revision
    (it embeds ``updated_at``) and is intended for logs and diagnostics, while
    downstream consumers depend on ``workspace/name`` being stable across
    revisions for an entity-backed config.

    :class:`EntityGuardrailConfigSource` → ``"workspace/name"``;
    :class:`InlineGuardrailConfigSource` → ``"<inline:label>"`` (``"<inline>"`` if no label).
    """
    if isinstance(source, EntityGuardrailConfigSource):
        return f"{source.workspace}/{source.name}"
    if source.label:
        return f"<inline:{source.label}>"
    return "<inline>"


def source_has_input_flows(source: GuardrailConfigSource) -> bool:
    """Return ``True`` when ``source`` declares at least one input flow."""
    rails = source.rails.rails
    if rails is None or rails.input is None:
        return False
    return bool(rails.input.flows)


def source_has_output_flows(source: GuardrailConfigSource) -> bool:
    """Return ``True`` when ``source`` declares at least one output flow."""
    rails = source.rails.rails
    if rails is None or rails.output is None:
        return False
    return bool(rails.output.flows)


def extract_output_rails_streaming_config(
    source: GuardrailConfigSource,
) -> OutputRailsStreamingConfig | None:
    """Return the output-rails streaming config from ``source``, if present."""
    rails = source.rails.rails
    if rails is None or rails.output is None:
        return None
    return rails.output.streaming


# =============================================================================
# Stage 2: stable build inputs
# =============================================================================


InferenceTargetResolver = Callable[[str], OpenAICompatibleInferenceTarget]
"""Resolve a VirtualModel ID to an OpenAI-compatible IGW target.

Satisfied by
:meth:`~nemo_platform_plugin.inference_middleware.NemoInferenceMiddleware.get_openai_compatible_inference_url_and_model`.
"""


MAIN_MODEL_TYPE = "main"
"""Reserved ``Model.type`` for the per-request main LLM."""


EMBEDDINGS_MODEL_TYPE = "embeddings"
"""Reserved ``Model.type`` for an embeddings model."""


MAIN_MODEL_REQUEST_PLACEHOLDER = "<request>"
"""Placeholder model used as the main model entry in a GuardrailConfig.
``model`` is a required field in the nemoguardrails library schema.
We always populate the actual model name using the incoming request
body's model, so this placeholder is to satisfy the schema validator.
"""


def _has_nonempty_model_name(value: object) -> bool:
    """``True`` when ``value`` is a non-empty (after strip) model name string."""
    return isinstance(value, str) and bool(value.strip())


def _fill_main_model_placeholder(model_config: dict[str, Any]) -> dict[str, Any]:
    """Given a model config from a GuardrailConfig, populates the ``model`` field with
    a placeholder value if the config represents the main model and doesn't already have
    a model name.

    The upstream ``Model`` schema accepts a name in exactly one of three places:
    top-level ``model``, ``parameters.model``, or ``parameters.model_name``. Filling a
    placeholder when either parameters form is already set would trip the library's
    dual-location validator, so those configs are left unchanged.

    This is required to satisfy the nemoguardrails library schema validator.
    We always populate the actual model name using the incoming request
    body's model, so this placeholder is to satisfy the schema validator.
    """
    if model_config.get("type") != MAIN_MODEL_TYPE:
        return model_config
    if _has_nonempty_model_name(model_config.get("model")):
        return model_config

    parameters = model_config.get("parameters")
    if isinstance(parameters, dict) and (
        _has_nonempty_model_name(parameters.get("model")) or _has_nonempty_model_name(parameters.get("model_name"))
    ):
        return model_config

    return {**model_config, "model": MAIN_MODEL_REQUEST_PLACEHOLDER}


def _is_missing_model_name_error(err: ValidationError) -> bool:
    """``True`` when ``err`` reports a missing model name on any entry.

    String-matches the upstream nemoguardrails
    ``Model.model_must_be_none_empty`` message; triggers the IGW-aware
    error wrap at the request boundary.
    """
    return any("Model name must be specified" in (e.get("msg") or "") for e in err.errors())


@dataclass(frozen=True, slots=True)
class StableRailsConfig:
    """Transformed config that uniquely determines an :class:`LLMRails`.

    Per-request data is stripped:

    - The ``main`` model is pulled out of ``rails.models`` (injected at lease
      time via :meth:`LLMRails.update_llm`) and kept pre-validated as
      ``main_model_template`` so :func:`rails.build_main_llm` doesn't re-run
      :class:`Model` validation on the hot path. ``main_model_template`` is
      ``None`` in the production case — under the IGW Plugin model the
      gateway owns main-LLM routing, so configs typically declare only task
      LLMs (content-safety, topic-control, embeddings). The template is
      retained whenever a config declares a ``main`` entry at all, whether
      to pin a specific model (self-check / demo configs) or just to
      override ``engine``/``parameters`` (ex. a custom ``base_url``) while
      leaving the model name to the request body.
    - Non-main ``base_url`` values are resolved against the IGW route table.
    - Static ``default_headers`` are preserved; platform service headers are
      added by the header-aware NIM client at call time.

    ``content_hash`` is sha256 of canonical JSON of ``rails``. Equal hashes
    imply interchangeable :class:`LLMRails` instances.

    ``embedding_model_id`` is extracted as a sub-key for the future KB cache.
    """

    rails: LibraryRailsConfig
    content_hash: str
    embedding_model_id: str | None = None
    main_model_template: Model | None = None


def stabilize(
    rails: PlatformRailsConfig,
    resolver: InferenceTargetResolver,
) -> StableRailsConfig:
    """Validate, strip per-request fields, resolve URLs, compute the hash.

    Always hashes the library :class:`LibraryRailsConfig`, not the platform
    wire shape: two platform payloads that round-trip to the same library
    config must share a cache key, and only the library shape determines
    the build.

    The platform :class:`PlatformRailsConfig` makes ``models`` optional —
    under the IGW Plugin architecture, callers that don't run self-check
    are expected to omit it entirely (IGW owns main-LLM routing; the
    plugin only needs configs for task LLMs like content-safety and
    topic-control). The library :class:`LibraryRailsConfig` schema makes
    ``models`` a required list, so a missing field is coerced to ``[]``
    here at the platform→library boundary. An empty list is a valid
    library config — :class:`LLMRails` logs a one-line "no main LLM"
    warning and proceeds; the per-request main is then injected via
    ``update_llm`` from :func:`rails.build_main_llm`.

    ``main`` entries never need a ``model`` name — IGW always supplies the
    real model from the request body. However, the upstream validator
    requires a non-empty model name on every entry, so nameless ``main``
    entries get a placeholder filled in first. This keeps configs that pin
    other fields on the main entry (e.g. `parameters.base_url`) from being
    lost.

    Non-main models with empty ``model`` are rejected by the library
    ``Model`` validator during ``model_validate`` below, so
    :func:`_resolve_model_target` never sees an empty model ID.
    """
    payload = rails.model_dump(exclude_none=True)
    # See class docstring: ``models`` is required by the library schema but
    # optional under the IGW Plugin architecture. Coerce here so a user can
    # post the minimum-viable config (e.g. just ``{"rails": {...}}``).
    payload.setdefault("models", [])
    payload["models"] = [_fill_main_model_placeholder(m) for m in payload["models"]]

    try:
        library_rails = LibraryRailsConfig.model_validate(payload)
    except ValidationError as exc:
        # Wrap the cryptic upstream error with an IGW-aware message.
        # ``ValueError`` → 400 via ``InferenceMiddleware._run_rails``.
        if _is_missing_model_name_error(exc):
            raise ValueError("Guardrails config has a model entry with a missing `model` name.") from exc
        raise
    stripped: list[Model] = []
    main_template: Model | None = None
    for model in library_rails.models or []:
        if model.type == MAIN_MODEL_TYPE:
            if main_template is not None:
                raise ValueError("guardrails config must contain at most one 'main' model")
            main_template = model.model_copy(deep=True)
            continue
        stripped.append(_resolve_model_target(model, resolver))

    # Sort so source-side declaration order doesn't fork the cache.
    # LLMRails looks up models by ``type``, never by index, so reordering
    # is semantically a no-op. ``json.dumps(sort_keys=True)`` only sorts
    # dict keys, not list elements; sort each model by its full canonical
    # JSON so two models that share ``(type, engine, model)`` but differ
    # in ``parameters`` are ordered deterministically (Python's stable
    # sort would otherwise let declaration order leak into the hash).
    stripped.sort(key=_canonical_model_json)
    canonical = library_rails.model_copy(update={"models": stripped})

    payload = canonical.model_dump(mode="json", exclude_none=True)
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    content_hash = hashlib.sha256(canonical_json).hexdigest()

    return StableRailsConfig(
        rails=canonical,
        content_hash=content_hash,
        embedding_model_id=_find_embedding_model_id(canonical),
        main_model_template=main_template,
    )


def _canonical_model_json(model: Model) -> str:
    """Canonical JSON of ``model`` for total-ordered sorting.

    Used as the sort key on the stripped models list so two configs that
    differ only in declaration order produce byte-identical canonical JSON
    and the same ``content_hash``. ``(type, engine, model)`` alone isn't a
    total order — two models sharing those three but differing in
    ``parameters`` would compare equal and Python's stable sort would let
    declaration order survive into the hash.
    """
    return json.dumps(model.model_dump(mode="json", exclude_none=True), sort_keys=True, separators=(",", ":"))


def _resolve_model_target(model: Model, resolver: InferenceTargetResolver) -> Model:
    """Resolve ``parameters['base_url']`` from ``resolver`` when not already set.

    Mutates ``model.parameters`` in place and returns ``model``. Safe inside
    :func:`stabilize` because ``library_rails`` is freshly built from
    ``model_dump`` and never escapes the function — do not call from
    elsewhere on a ``Model`` held externally.
    """
    base_url = model.parameters.get("base_url")
    if isinstance(base_url, str) and base_url:
        return model
    # ``model_must_be_none_empty`` guarantees ``model`` is a non-empty
    # stripped string post-validation. Re-check explicitly (rather than
    # ``assert``) so a future upstream change fails loudly even under
    # ``python -O``, which strips asserts.
    if not model.model:
        raise ValueError(
            f"Model entry of type {model.type!r} has no ``model`` field; "
            "the upstream ``model_must_be_none_empty`` validator must have changed."
        )
    model.parameters["base_url"] = resolver(model.model).openai_base_url
    return model


def _find_embedding_model_id(rails: LibraryRailsConfig) -> str | None:
    """Return the ``embeddings`` model's ID, if any. Sub-key for the KB cache."""
    for model in rails.models or []:
        if model.type == EMBEDDINGS_MODEL_TYPE:
            return model.model
    return None


class StabilizedRailsConfigCache:
    """Bounded LRU memoization of :class:`StableRailsConfig` by entity identity.

    Entity sources memoize on ``(workspace, name, updated_at)`` so the hot
    path skips Pydantic validation. Inline sources bypass (no identity to
    key on) and pay :func:`stabilize` per request.

    LRU is self-invalidating: a new ``updated_at`` misses, the old tuple
    decays under pressure. The produced :class:`StableRailsConfig` is
    byte-identical to calling :func:`stabilize` directly, so correctness
    doesn't depend on hits. No shutdown required — process-local, GC'd with
    the plugin.
    """

    def __init__(self, *, max_entries: int = 128) -> None:
        if max_entries < 1:
            raise ValueError(f"StabilizedRailsConfigCache.max_entries must be >= 1, got {max_entries}")
        self._map: OrderedDict[tuple[str, str, str], StableRailsConfig] = OrderedDict()
        self._max = max_entries

    def get_or_compute(
        self,
        source: GuardrailConfigSource,
        resolver: InferenceTargetResolver,
    ) -> StableRailsConfig:
        """Return the cached entry or compute it on miss.

        Entity sources memoize by ``(workspace, name, updated_at)``; inline
        sources stabilize every call.
        """
        if isinstance(source, EntityGuardrailConfigSource):
            key = (source.workspace, source.name, source.updated_at)
            hit = self._map.get(key)
            if hit is not None:
                self._map.move_to_end(key)
                return hit
            stable = stabilize(source.rails, resolver)
            self._map[key] = stable
            if len(self._map) > self._max:
                self._map.popitem(last=False)
            return stable
        if isinstance(source, InlineGuardrailConfigSource):
            # Content-fingerprint memoization on inline payloads is possible
            # future work; Phase 1 pays stabilize per request. The outer
            # LLMRailsCache content-hash key already catches identical inline
            # configs at the pool level.
            return stabilize(source.rails, resolver)
        raise TypeError(f"Unsupported guardrail source type: {type(source).__name__}")


# =============================================================================
# Stage 3a: builder seam (LLMRails construction)
# =============================================================================


class LLMRailsBuilder(Protocol):
    """Constructs an :class:`LLMRails` from a cached :class:`LibraryRailsConfig`.

    Implementations must run blocking work off the event loop (typically via
    :func:`asyncio.to_thread`) so the IGW loop stays responsive during builds.
    """

    async def build(self, config: LibraryRailsConfig) -> LLMRails: ...


class DefaultLLMRailsBuilder:
    """Phase 1 builder: ``asyncio.to_thread(LLMRails, config=config)``.

    Pays the full ``LLMRails.__init__`` cost on every build. Phase 2 will swap
    in a builder that shares Colang / KnowledgeBase across instances; the
    pool contract is unchanged.
    """

    async def build(self, config: LibraryRailsConfig) -> LLMRails:
        return await asyncio.to_thread(LLMRails, config=config)


# =============================================================================
# Stage 3b: pool and outer cache
# =============================================================================


CacheKey = str
"""Content hash of the stable build input."""


PROVENANCE_HISTORY_LEN = 5
"""Ring-buffer size for recent :class:`Provenance` labels per pool.

Used only in eviction logs / operator queries to translate a content hash
back to human-meaningful sources.
"""


Build = Callable[[LibraryRailsConfig], Awaitable[LLMRails]]

Prepare = Callable[[LLMRails], None]
"""Per-acquire reset; raising discards the (presumed half-mutated) instance."""


def _completed_warm_task() -> asyncio.Task[None]:
    """Return a no-op already-done task used as the post-close ``warm()`` sentinel.

    Lets callers ``await cache.warm(...)`` uniformly across lifecycles. Not
    tracked in ``_warm_tasks`` since ``close()`` has already drained it.
    """

    async def _noop() -> None:
        return None

    return asyncio.ensure_future(_noop())


class Pool:
    """Resource pool of :class:`LLMRails` instances for one cache key.

    ``LLMRails.__init__`` is expensive (config load, Colang parse, action
    dispatcher setup) and the instance carries per-request mutable state
    (``rails.llm``, ``events_history_cache``, ``explain_info``) that can't
    be shared across concurrent leases. Pooling reuses pre-built instances
    but hands each lease its own, so concurrent requests don't race.

    Grows on demand, shrinks on quiet. No saturation cap — back-pressure is
    the upstream LLM's job. Memory is bounded by:

    - ``_build_lock`` serializes same-pool expansion so by the time misser K
      gets the lock, missers 1..K-1 may have already returned instances that
      K can reuse. Growth tracks true concurrent demand, not request rate.
    - On release, ``_idle`` is FIFO-trimmed to ``max(min_idle, _leased)`` so
      a burst-then-quiet workload reclaims idle instances in lockstep.
    """

    def __init__(self, *, stable: StableRailsConfig, min_idle: int = 1) -> None:
        if min_idle < 0:
            raise ValueError(f"Pool.min_idle must be >= 0, got {min_idle}")
        self.stable = stable
        self.min_idle = min_idle
        self.provenance_history: deque[Provenance] = deque(maxlen=PROVENANCE_HISTORY_LEN)
        self._idle: deque[LLMRails] = deque()
        self._leased = 0
        self._build_lock = asyncio.Lock()

    @property
    def config(self) -> LibraryRailsConfig:
        """Read-through to ``stable.rails`` — the builder's input."""
        return self.stable.rails

    def record(self, provenance: Provenance) -> None:
        """Append ``provenance`` to the bounded history.

        Dedup is most-recent only: a pool serving one source compresses to
        a single label, but a pool alternating between two distinct sources
        sharing a content hash by coincidence still shows ``[A, B, A, B, …]``.
        """
        if self.provenance_history and self.provenance_history[-1] == provenance:
            return
        self.provenance_history.append(provenance)

    @asynccontextmanager
    async def acquire(self, build: Build, prepare: Prepare = lambda _r: None) -> AsyncIterator[LLMRails]:
        """Take one instance for exclusive use; return it on context exit.

        Failure semantics:

        - ``build`` raises: ``_leased`` untouched, exception propagates.
        - ``prepare`` raises: instance discarded, ``_leased`` rolled back.
        - User exception inside ``async with``: instance returned to idle
          (next ``prepare`` wipes per-request state before reuse).
        - ``CancelledError`` / ``GeneratorExit``: instance **discarded**. An
          :func:`asyncio.to_thread` worker started inside the body can keep
          mutating ``rails`` after the awaiting coroutine is cancelled —
          re-pooling would race the next lease's ``prepare`` wipe and
          reopen the cross-tenant leak. ``_leased`` still decrements so
          demand-tracking converges. ``GeneratorExit`` is the path an IGW
          framework ``aclose()`` takes when dropping a streaming response.
        """
        rails = await self._take_or_build(build)
        self._leased += 1
        try:
            prepare(rails)
        except BaseException:
            self._leased -= 1
            raise
        discard = False
        try:
            yield rails
        except (asyncio.CancelledError, GeneratorExit):
            # Don't re-pool: an orphan to_thread worker may still be mutating.
            discard = True
            raise
        finally:
            # Synchronous — no awaits — so cancellation can't tear accounting.
            self._leased -= 1
            if not discard:
                self._idle.append(rails)
            while len(self._idle) > max(self.min_idle, self._leased):
                self._idle.popleft()

    async def warm(self, build: Build) -> None:
        """Ensure at least one idle instance is ready for the next acquire.

        Build errors are logged and swallowed; the next :meth:`acquire`
        retries under the same lock. Lock is held across the build so
        concurrent acquires on the same key see the warmed instance via
        their post-lock recheck — in Phase 1 that means seconds of
        serialization, which Phase 2's shared sub-components will cut.
        """
        async with self._build_lock:
            if self._idle:
                return
            try:
                rails = await build(self.config)
            except Exception:
                # Exception only, so CancelledError propagates for cache.close.
                logger.exception("Failed to warm pool %s", self.stable.content_hash)
                return
            self._idle.append(rails)

    async def _take_or_build(self, build: Build) -> LLMRails:
        if self._idle:
            return self._idle.popleft()
        # Post-lock recheck is load-bearing: an earlier misser may have built
        # *and* released before we got the lock, letting us skip the build.
        async with self._build_lock:
            if self._idle:
                return self._idle.popleft()
            return await build(self.config)


class LLMRailsCache:
    """LRU-bounded keyed lookup of :class:`Pool` with background warming.

    Keyed by ``stable.content_hash`` — the canonical content identity of a
    guardrails config (post-:func:`stabilize`: ``main`` stripped, non-main
    URLs resolved, models sorted). Two sources with byte-identical
    post-stabilize configs share one pool (an entity and an inline payload
    with the same content coalesce); changing any cached field mints a new
    key and a new pool, while changing only the per-request ``main`` model
    does not.

    Pool *creation* is bounded by ``max_pools``; per-pool *capacity* is
    unbounded (pools self-regulate). ``_pins[key] > 0`` (in-progress lease
    or warm) excludes the pool from eviction.
    """

    def __init__(
        self,
        *,
        builder: LLMRailsBuilder,
        max_pools: int = 64,
        pool_min_idle: int = 1,
    ) -> None:
        if max_pools < 1:
            raise ValueError(f"LLMRailsCache.max_pools must be >= 1, got {max_pools}")
        if pool_min_idle < 0:
            raise ValueError(f"LLMRailsCache.pool_min_idle must be >= 0, got {pool_min_idle}")
        self._builder = builder
        self._max_pools = max_pools
        self._pool_min_idle = pool_min_idle

        self._pools: OrderedDict[CacheKey, Pool] = OrderedDict()
        # Refcount of in-progress operations per key; non-zero pins against
        # LRU eviction. Deleted on drop so eviction's "key not in _pins" is trivial.
        self._pins: dict[CacheKey, int] = {}
        self._lock = asyncio.Lock()
        self._warm_tasks: set[asyncio.Task[None]] = set()
        self._closed = False
        # Edge-triggered so sustained capacity pressure logs once per
        # transition into saturation, not on every miss.
        self._all_pinned_active = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def lease(
        self,
        stable: StableRailsConfig,
        *,
        main_llm: LLMModel | None = None,
        provenance: Provenance | None = None,
    ) -> AsyncIterator[LLMRails]:
        """Lease one :class:`LLMRails` for one logical operation.

        Called per request from the InferenceMiddleware handlers: non-streaming
        for the duration of :meth:`LLMRails.generate_async`, streaming for
        the life of the response iterator.

        Keyed by ``stable.content_hash``. Pins the pool against eviction and
        runs the per-request reset before yielding. On a hit the pool's
        stored config is authoritative, so ``stable.rails`` is unused.
        ``provenance`` is diagnostics only.
        """
        async with self._pin(stable, provenance) as pool:
            async with pool.acquire(
                self._builder.build,
                prepare=lambda rails: self._reset(rails, main_llm),
            ) as rails:
                yield rails

    def warm(
        self,
        stable: StableRailsConfig,
        *,
        provenance: Provenance | None = None,
    ) -> asyncio.Task[None]:
        """Schedule a background warm; track the task for shutdown.

        Called from :meth:`on_virtual_model_upserted`, not request handlers.
        Fire-and-forget so the KB build inside ``LLMRails.__init__`` doesn't
        block IGW's VM polling; ``await``-able for tests. A :meth:`lease`
        that arrives before the warm finishes pays the build cost itself.

        Post-close: returns a done no-op task so ``await cache.warm(...)``
        is safe across lifecycles. :meth:`lease` still raises post-close.

        Atomic vs :meth:`close`: the ``_closed`` check and
        ``_warm_tasks.add`` have no intervening ``await``, so a concurrent
        ``close()`` either runs first (warm becomes a no-op) or finds the
        task already tracked and cancels it during shutdown.
        """
        if self._closed:
            return _completed_warm_task()

        async def _run() -> None:
            async with self._pin(stable, provenance) as pool:
                await pool.warm(self._builder.build)

        task = asyncio.create_task(_run())
        self._warm_tasks.add(task)
        task.add_done_callback(self._warm_tasks.discard)
        return task

    async def close(self) -> None:
        """Cancel pending warms and drop all pools. Idempotent.

        Flipping ``_closed`` and snapshotting ``_warm_tasks`` happens under
        one lock so a concurrent :meth:`warm` can't slip a task past the
        snapshot.

        In-flight leases aren't awaited — their ``finally`` releases into a
        now-orphaned pool that's safe to GC. New leases raise via
        :meth:`_pin`.
        """
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            pending = list(self._warm_tasks)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        async with self._lock:
            self._pools.clear()
            self._pins.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _pin(
        self,
        stable: StableRailsConfig,
        provenance: Provenance | None,
    ) -> AsyncIterator[Pool]:
        """Resolve to a pool, refcount-pin it, and unpin on exit.

        Pin increment and closed-check run under ``self._lock`` so concurrent
        evictions/closes can't race an in-flight lease.

        Unpin runs **without** the lock: dict mutations are atomic between
        asyncio steps, and skipping the await means a cancellation between
        the yield and the finally-end can't strand a pin.
        """
        key = stable.content_hash
        async with self._lock:
            if self._closed:
                raise RuntimeError("LLMRailsCache is closed")
            pool = self._pools.get(key)
            if pool is None:
                self._evict_one_if_full()
                pool = Pool(stable=stable, min_idle=self._pool_min_idle)
                self._pools[key] = pool
            else:
                self._pools.move_to_end(key)
            if provenance is not None:
                pool.record(provenance)
            self._pins[key] = self._pins.get(key, 0) + 1
        try:
            yield pool
        finally:
            n = self._pins.get(key, 0) - 1
            if n <= 0:
                self._pins.pop(key, None)
            else:
                self._pins[key] = n

    def _evict_one_if_full(self) -> None:
        """Drop the LRU evictable pool if at capacity. Caller holds ``self._lock``.

        Evictable == ``key not in self._pins``. If all are pinned, grow
        temporarily past the cap rather than drop requests; LRU catches up
        once a pin releases. The "all pinned" warning is edge-triggered so
        sustained pressure logs once per transition.
        """
        if len(self._pools) < self._max_pools:
            return
        # Resolve victim before mutating: deleting during OrderedDict
        # iteration invalidates the iterator.
        victim = next((k for k in self._pools if k not in self._pins), None)
        if victim is not None:
            evicted = self._pools.pop(victim)
            sources = ", ".join(p.label for p in evicted.provenance_history) or "<none>"
            # INFO so operators can see eviction patterns without enabling
            # debug logging — eviction means the working set exceeded
            # max_pools, which is operationally interesting on its own.
            logger.info(
                "Evicted LRU cache entry %s (recent sources: %s)",
                victim,
                sources,
            )
            if self._all_pinned_active:
                self._all_pinned_active = False
                logger.info(
                    "Cache eviction succeeded; recovered from all-pinned state (max_pools=%d)",
                    self._max_pools,
                )
            return
        if not self._all_pinned_active:
            self._all_pinned_active = True
            logger.warning(
                "All %d cached pools are pinned; growing temporarily above max_pools",
                self._max_pools,
            )

    @staticmethod
    def _reset(rails: LLMRails, main_llm: LLMModel | None) -> None:
        """Wipe per-request shared state and apply this request's main LLM."""
        # Prevent leaks by clearing shared state inside a LLMRails instance.
        rails.events_history_cache.clear()
        rails.explain_info = None
        # Inject main_llm. Even if main_llm is None, we want this to persist
        # to prevent subsequent leases without main_llms from reusing the wrong model.
        #
        # ``update_llm`` is typed ``llm: LLMModel`` (no ``None``), but its own
        # isinstance check treats any non-``LLMModel`` value — including
        # ``None`` — as a legacy LangChain LLM to wrap, so passing ``None``
        # here safely clears the previous lease's model instead of reusing it.
        rails.update_llm(cast(LLMModel, main_llm))
