# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Middleware plugin registry for the Inference Gateway.

Handles discovery, loading, lifecycle management, per-VirtualModel config
pre-resolution, and pipeline execution for
:class:`~nemo_platform_plugin.inference_middleware.NemoInferenceMiddleware` plugins
installed via the ``nemo.inference_middleware`` entry-point group.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple

from fastapi import HTTPException
from nemo_platform import AsyncNeMoPlatform
from nemo_platform.types.inference.middleware_call import MiddlewareCall as SDKMiddlewareCall
from nemo_platform.types.inference.virtual_model import VirtualModel as SDKVirtualModel
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.discovery import discover_inference_middleware
from nemo_platform_plugin.inference_middleware import (
    BackendFormat,
    ImmediateResponse,
    InferenceMiddlewareContext,
    InferenceMiddlewareError,
    InferenceRequest,
    InferenceResponse,
    MiddlewareConfigNotFoundError,
    ModelProviderInferenceTarget,
    NemoInferenceMiddleware,
    OpenAICompatibleInferenceTarget,
)
from nemo_platform_plugin.inference_middleware_models import (
    MiddlewareCall,
    VirtualModelInferenceConfig,
)
from nemo_platform_plugin.inference_middleware_models import (
    VirtualModel as PluginVirtualModel,
)
from nmp.common.config import get_platform_config
from nmp.common.entities.utils import parse_entity_ref
from nmp.core.inference_gateway.api.backend_format import resolve_backend_format
from nmp.core.inference_gateway.api.typed_request import parse_typed_request
from nmp.core.inference_gateway.api.typed_response import TypedResponseStream, parse_typed_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# InferenceMiddlewareCacheAccessorImpl
# ---------------------------------------------------------------------------


@dataclass
class InferenceMiddlewareCacheAccessorImpl:
    """Concrete implementation of :class:`~nemo_platform_plugin.inference_middleware.InferenceMiddlewareCacheAccessor`.

    Wraps the IGW's in-memory :class:`~nmp.core.inference_gateway.api.model_cache.ModelCache`
    and :class:`~nmp.core.inference_gateway.api.virtual_model_cache.VirtualModelCache` to
    provide read-only platform state to inference middleware plugins.

    Injected into each :class:`~nemo_platform_plugin.inference_middleware.NemoInferenceMiddleware`
    instance before :meth:`~nemo_platform_plugin.inference_middleware.NemoInferenceMiddleware.on_startup`
    is called.
    """

    _model_cache: Any  # ModelCache — typed as Any to avoid circular import
    _virtual_model_cache: Any  # VirtualModelCache

    def get_model_providers_for_model(self, model_entity_id: str) -> list:
        """Return all ModelProviders currently serving ``model_entity_id`` (``"ws/name"``)."""
        try:
            ref = parse_entity_ref(model_entity_id)
        except ValueError:
            return []
        entity_info = self._model_cache.get_from_model_entity(ref.workspace, ref.name)
        if entity_info is None:
            return []
        return [info.model_provider for _, info in entity_info.model_providers]

    def get_model_entity(self, model_entity_id: str) -> Any | None:
        """Return the :class:`~nmp.core.inference_gateway.api.model_cache.ModelEntityInfo`
        for ``model_entity_id``, or ``None`` if not cached.

        :class:`ModelEntityInfo` structurally satisfies the ``ModelEntity`` Protocol from
        ``nemo_platform_plugin.inference_middleware`` — it exposes ``workspace``, ``name``,
        ``spec``, ``finetuning_type``, and ``providers``.  ``spec`` and
        ``finetuning_type`` are ``None`` until the IGW begins populating them from the
        models service.
        """
        try:
            ref = parse_entity_ref(model_entity_id)
        except ValueError:
            return None
        return self._model_cache.get_from_model_entity(ref.workspace, ref.name)

    def list_model_entities_for_workspace(self, workspace: str | None = None) -> list[str]:
        """Return model entity IDs (``"workspace/name"``) known to IGW.

        Pass ``workspace`` to filter; omit (or pass ``None``) for all workspaces.
        """
        return [
            f"{ws}/{name}"
            for (ws, name) in self._model_cache.model_entity_info_map.keys()
            if workspace is None or ws == workspace
        ]

    def get_virtual_model(self, virtual_model_id: str) -> PluginVirtualModel | None:
        """Return the VirtualModel for ``virtual_model_id`` (``"ws/name"``), or ``None``.

        Converts the cached SDK VirtualModel to the ``nemo_platform_plugin`` VirtualModel type
        so the return value satisfies the ``InferenceMiddlewareCacheAccessor`` Protocol.
        """
        try:
            ref = parse_entity_ref(virtual_model_id)
        except ValueError:
            return None
        sdk_vm = self._virtual_model_cache.get(ref.workspace, ref.name)
        if sdk_vm is None:
            return None
        return _sdk_vm_to_plugin_vm(sdk_vm)

    def list_virtual_models_for_workspace(self, workspace: str) -> list[str]:
        """Return VirtualModel IDs (``"workspace/name"``) in ``workspace``."""
        return [f"{ws}/{name}" for (ws, name) in self._virtual_model_cache.virtual_model_map.keys() if ws == workspace]

    def get_inference_url_and_model(
        self,
        model_entity_id: str,
        append_v1_suffix: bool = True,
    ) -> ModelProviderInferenceTarget:
        """Return the backend provider URL and served model name for ``model_entity_id``.

        Uses the first provider in ModelCache for this entity (consistent with how IGW
        selects providers at proxy time).  Returns the provider's ``host_url`` directly —
        not an IGW loopback — so in-cluster NIM calls don't require extra auth.

        Args:
            model_entity_id: ``"workspace/name"`` of the model entity.
            append_v1_suffix: Append ``/v1`` to the URL. Defaults to ``True``.

        Raises:
            KeyError: If ``model_entity_id`` is not in the ModelCache.
        """
        try:
            ref = parse_entity_ref(model_entity_id)
        except ValueError as exc:
            raise KeyError(f"Invalid model_entity_id format: {model_entity_id!r}") from exc

        entity_info = self._model_cache.get_from_model_entity(ref.workspace, ref.name)
        if entity_info is None or not entity_info.model_providers:
            raise KeyError(f"Model entity {model_entity_id!r} not in ModelCache")

        served_name, provider_info = entity_info.model_providers[0]
        base_url = provider_info.model_provider.host_url.rstrip("/")
        if append_v1_suffix and not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        return ModelProviderInferenceTarget(
            model_provider_gateway_url=base_url,
            served_model_name=served_name,
        )

    def get_backend_format(self, virtual_model_id: str, model_entity_id: str) -> BackendFormat | None:
        """Resolve the backend API format for a ModelEntity in a VirtualModel route.

        A matching VirtualModel ``models`` entry overrides the ModelEntity value.
        Returns ``None`` if neither source defines a valid backend format.

        Raises:
            KeyError: If either ID is invalid or missing from the caches.
        """
        try:
            ref = parse_entity_ref(model_entity_id)
        except ValueError as exc:
            raise KeyError(f"Invalid model_entity_id format: {model_entity_id!r}") from exc

        entity_info = self._model_cache.get_from_model_entity(ref.workspace, ref.name)
        if entity_info is None:
            raise KeyError(f"Model entity {model_entity_id!r} not in ModelCache")

        try:
            vm_ref = parse_entity_ref(virtual_model_id)
        except ValueError as exc:
            raise KeyError(f"Invalid virtual_model_id format: {virtual_model_id!r}") from exc

        virtual_model = self._virtual_model_cache.get(vm_ref.workspace, vm_ref.name)
        if virtual_model is None:
            raise KeyError(f"VirtualModel {virtual_model_id!r} not in VirtualModelCache")

        return resolve_backend_format(entity_info, virtual_model)

    def get_openai_compatible_inference_url_and_model(self, virtual_model_id: str) -> OpenAICompatibleInferenceTarget:
        """Return the OpenAI-compatible IGW URL and VirtualModel ID for ``virtual_model_id``.

        This preserves IGW VirtualModel routing for plugin-owned model calls.

        Raises:
            KeyError: If ``virtual_model_id`` is invalid or not in the VirtualModelCache.
        """
        try:
            ref = parse_entity_ref(virtual_model_id)
        except ValueError as exc:
            raise KeyError(f"Invalid virtual_model_id format: {virtual_model_id!r}") from exc

        virtual_model = self._virtual_model_cache.get(ref.workspace, ref.name)
        if virtual_model is None:
            raise KeyError(f"VirtualModel {virtual_model_id!r} not in VirtualModelCache")

        base_url = get_platform_config().base_url.rstrip("/")
        return OpenAICompatibleInferenceTarget(
            openai_base_url=f"{base_url}/apis/inference-gateway/v2/workspaces/{ref.workspace}/openai/-/v1",
            model=f"{ref.workspace}/{ref.name}",
        )


# ---------------------------------------------------------------------------
# External middleware config reference (per plugin + config type + id)
# ---------------------------------------------------------------------------


class MiddlewareConfigRef(NamedTuple):
    """A unique (plugin, *config_type*, *config_id*) used for de-duplicated fetches.

    *config_id* is the ``"workspace/name"`` string from :class:`MiddlewareCall`.
    """

    plugin_name: str
    config_type: str
    config_id: str


# ---------------------------------------------------------------------------
# ResolvedMiddlewareCall
# ---------------------------------------------------------------------------


@dataclass
class ResolvedMiddlewareCall:
    """A :class:`~nemo_platform_plugin.inference_middleware.MiddlewareCall` with its config
    already resolved (via ``get_middleware_config`` if needed) and validated
    (via ``validate_middleware_config``).

    Pre-resolved at VirtualModel cache build time — never per-request.
    """

    plugin_name: str
    """Entry-point key of the plugin (e.g. ``"nemo-switchyard"``)."""

    config_type: str
    """Discriminator passed to ``process_request`` / ``process_response``."""

    resolved_config: Any
    """Output of ``plugin.validate_middleware_config(config_type, raw_config)``."""


# ---------------------------------------------------------------------------
# PrefetchResult
# ---------------------------------------------------------------------------


@dataclass
class PrefetchResult:
    """Outcome of a single :meth:`MiddlewareRegistry.prefetch_configs` batch.

    Each referenced :class:`MiddlewareConfigRef` ends up in exactly one of three
    sets, letting downstream consumers distinguish "deleted upstream" (act now,
    evict the cache) from "transient failure" (preserve prior state and try again
    next cycle):

    - :attr:`fetched` — the plugin returned a config object. The value is the
      raw object passed to ``validate_middleware_config`` at resolve time.
    - :attr:`missing` — the plugin signalled a definitive deletion by raising
      :class:`~nemo_platform_plugin.inference_middleware.MiddlewareConfigNotFoundError`.
      The diff loop marks these refs as changed so referencing VirtualModels
      are re-resolved (and end up in :attr:`MiddlewareRegistry.broken_vms`).
    - :attr:`transient` — any other failure mode (unknown plugin, plugin not
      implemented, generic exception). The diff loop intentionally leaves the
      cached :attr:`~nmp.core.inference_gateway.api.virtual_model_cache.VirtualModelCache.config_ref_versions`
      entry alone for these refs so a brief outage does not flap the cache.
    """

    fetched: dict[MiddlewareConfigRef, Any] = field(default_factory=dict)
    missing: set[MiddlewareConfigRef] = field(default_factory=set)
    transient: set[MiddlewareConfigRef] = field(default_factory=set)

    def __post_init__(self) -> None:
        assert self.fetched.keys().isdisjoint(self.missing), "fetched and missing overlap"
        assert self.fetched.keys().isdisjoint(self.transient), "fetched and transient overlap"
        assert self.missing.isdisjoint(self.transient), "missing and transient overlap"


# ---------------------------------------------------------------------------
# Module helpers — virtual model cache / middleware config references
# ---------------------------------------------------------------------------


def collect_config_refs(virtual_models: list[SDKVirtualModel]) -> set[MiddlewareConfigRef]:
    """Return the de-duplicated set of external config refs across all *virtual_models*.

    Only includes calls that have both a ``name`` (plugin key) and a ``config_id``
    (entity reference).  Inline ``config`` dicts are skipped.
    """
    refs: set[MiddlewareConfigRef] = set()
    for vm in virtual_models:
        for calls in (
            vm.request_middleware or [],
            vm.response_middleware or [],
            vm.post_response_middleware or [],
        ):
            for call in calls:
                if not call.config_id or not call.name:
                    continue
                refs.add(
                    MiddlewareConfigRef(
                        plugin_name=call.name,
                        config_type=call.config_type or "",
                        config_id=call.config_id,
                    )
                )
    return refs


def vm_uses_any_changed_config(vm: SDKVirtualModel, updated_refs: set[MiddlewareConfigRef]) -> bool:
    """Return ``True`` if *vm* references at least one ref in *updated_refs*."""
    return bool(collect_config_refs([vm]) & updated_refs)


# ---------------------------------------------------------------------------
# MiddlewareRegistry
# ---------------------------------------------------------------------------


@dataclass
class MiddlewareRegistry:
    """Runtime registry of loaded :class:`~nemo_platform_plugin.inference_middleware.NemoInferenceMiddleware` plugins.

    Stores instantiated plugin objects and pre-resolved per-VirtualModel middleware
    configs.  Exposed as a FastAPI global dependency so proxy handlers can reach it.

    Pre-resolved configs are keyed by ``(workspace, name)`` and stored in three parallel
    dicts that mirror ``VirtualModel.request_middleware``,
    ``VirtualModel.response_middleware``, and ``VirtualModel.post_response_middleware``.
    An absent key means "no resolved configs available for this VM" — either the VM has an
    empty middleware list or config resolution failed (and a warning was logged).

    :attr:`broken_vms` is the fail-closed flag set:
    request- or response-phase resolution failures (deleted ``config_id``, validation
    error, unknown plugin, transient fetch failure) add ``(workspace, name)`` here so
    the request-time proxy can return 503 instead of silently bypassing security-critical
    middleware. A successful re-resolve clears the entry. Post-response failures
    intentionally do not flip the flag — they are fire-and-forget telemetry hooks and
    must not take down user-facing traffic.
    """

    plugins: dict[str, NemoInferenceMiddleware] = field(default_factory=dict)

    # Pre-resolved configs keyed by (workspace, name), parallel to the VM's middleware lists
    request_middleware_calls: dict[tuple[str, str], list[ResolvedMiddlewareCall]] = field(default_factory=dict)
    response_middleware_calls: dict[tuple[str, str], list[ResolvedMiddlewareCall]] = field(default_factory=dict)
    post_response_middleware_calls: dict[tuple[str, str], list[ResolvedMiddlewareCall]] = field(default_factory=dict)
    broken_vms: set[tuple[str, str]] = field(default_factory=set)

    async def prefetch_configs(self, refs: set[MiddlewareConfigRef]) -> PrefetchResult:
        """Fetch each *ref* once and partition the results by failure mode.

        Returns a :class:`PrefetchResult` populated with three disjoint sets:

        - ``fetched`` — the plugin returned a config object.
        - ``missing`` — the plugin raised
          :class:`~nemo_platform_plugin.inference_middleware.MiddlewareConfigNotFoundError`,
          signalling a definitive 404 from its upstream store.
        - ``transient`` — any other failure (unknown plugin, ``NotImplementedError``,
          generic exception). The diff loop keeps the prior ``updated_at`` in
          ``VirtualModelCache.config_ref_versions`` for these so a brief outage does
          not flap the cache.
        """
        out = PrefetchResult()

        for middleware_config_ref in sorted(refs, key=lambda r: (r.plugin_name, r.config_type, r.config_id)):
            plugin = self.plugins.get(middleware_config_ref.plugin_name)
            if plugin is None:
                logger.warning(
                    "MiddlewareConfigRef %r / %r / %r skipped — plugin not loaded",
                    middleware_config_ref.plugin_name,
                    middleware_config_ref.config_type,
                    middleware_config_ref.config_id,
                )
                out.transient.add(middleware_config_ref)
                continue
            try:
                raw = await plugin.get_middleware_config(
                    middleware_config_ref.config_type, middleware_config_ref.config_id
                )
            except MiddlewareConfigNotFoundError:
                logger.debug(
                    "MiddlewareConfigRef %r / %r / %r reports deleted upstream - invalidating",
                    middleware_config_ref.plugin_name,
                    middleware_config_ref.config_type,
                    middleware_config_ref.config_id,
                )
                out.missing.add(middleware_config_ref)
                continue
            except NotImplementedError:
                logger.warning(
                    "Plugin %r does not implement get_middleware_config() for ref %r / %r",
                    middleware_config_ref.plugin_name,
                    middleware_config_ref.config_type,
                    middleware_config_ref.config_id,
                )
                out.transient.add(middleware_config_ref)
                continue
            except Exception:
                logger.warning(
                    "Error fetching middleware config for ref %r / %r / %r — treating as transient",
                    middleware_config_ref.plugin_name,
                    middleware_config_ref.config_type,
                    middleware_config_ref.config_id,
                    exc_info=True,
                )
                out.transient.add(middleware_config_ref)
                continue

            out.fetched[middleware_config_ref] = raw

        return out

    async def resolve_configs_for_virtual_model(
        self,
        vm: SDKVirtualModel,
        *,
        prefetch: PrefetchResult,
    ) -> None:
        """Resolve and validate all MiddlewareCall configs for *vm*.

        Each call is resolved from either an inline ``config`` dict or a
        ``config_id`` looked up in *prefetch*, then passed to
        ``plugin.validate_middleware_config``.

        Phases and failure semantics:

        - **request / response** — fail closed. Any error evicts the VM and
          adds it to :attr:`broken_vms` (proxy returns 503).
        - **post_response** — best effort. Failure drops the post-response
          slot but does not mark the VM broken.
        - A fully successful resolve clears any prior broken state. Empty
          middleware lists succeed without plugin interaction.

        :func:`~nmp.core.inference_gateway.api.virtual_model_cache.refresh_virtual_model_cache`
        calls :meth:`notify_upserted` after this method even when the VM is
        marked broken — plugins must not assume resolution succeeded.
        """
        if vm.workspace is None or vm.name is None:
            return

        key: tuple[str, str] = (vm.workspace, vm.name)
        vm_id = f"{vm.workspace}/{vm.name}"

        request_ok, request_resolved = await self._resolve_phase(
            vm_id, "request", vm.request_middleware or [], prefetch
        )
        response_ok, response_resolved = await self._resolve_phase(
            vm_id, "response", vm.response_middleware or [], prefetch
        )

        if not (request_ok and response_ok):
            # Fail closed: any pre-response phase failure means we can no longer
            # serve this VM safely. Evict everything (including any previously
            # successful post-response data) and mark broken. post_response is
            # intentionally not even attempted — its state is moot while the
            # request path is going to 503.
            self.evict(key)
            self.broken_vms.add(key)
            return

        post_ok, post_resolved = await self._resolve_phase(
            vm_id, "post_response", vm.post_response_middleware or [], prefetch
        )

        # All pre-response phases healthy → commit.
        self.request_middleware_calls[key] = request_resolved
        self.response_middleware_calls[key] = response_resolved
        self.broken_vms.discard(key)

        # post_response is best-effort by design (see docstring). Drop the slot
        # on failure so a stale list can't be silently reused, but do not flip
        # the VM to broken.
        if post_ok:
            self.post_response_middleware_calls[key] = post_resolved
        else:
            self.post_response_middleware_calls.pop(key, None)

    async def _resolve_phase(
        self,
        vm_id: str,
        phase: str,
        calls: list[SDKMiddlewareCall],
        prefetch: PrefetchResult,
    ) -> tuple[bool, list[ResolvedMiddlewareCall]]:
        """Resolve a single phase's calls and return ``(ok, resolved)``.

        ``ok`` is ``True`` only if every call resolved cleanly. On failure the
        partial list is discarded — the caller decides whether to evict and/or
        flip the VM to :attr:`broken_vms`.
        """
        resolved: list[ResolvedMiddlewareCall] = []

        for index, call in enumerate(calls):
            plugin_name = call.name
            config_type = call.config_type or ""

            if not plugin_name:
                logger.warning(
                    "VirtualModel %s has a middleware call without plugin name in %s_middleware",
                    vm_id,
                    phase,
                )
                return False, []

            plugin = self.plugins.get(plugin_name)
            if plugin is None:
                logger.warning(
                    "VirtualModel %s references unknown plugin %r in %s_middleware",
                    vm_id,
                    plugin_name,
                    phase,
                )
                return False, []

            if call.config_id is not None:
                mref = MiddlewareConfigRef(
                    plugin_name=plugin_name,
                    config_type=config_type,
                    config_id=call.config_id,
                )
                if mref in prefetch.missing:
                    logger.warning(
                        "Config %r referenced by VirtualModel %s %s_middleware[%d] has been deleted upstream",
                        call.config_id,
                        vm_id,
                        phase,
                        index,
                    )
                    return False, []
                if mref not in prefetch.fetched:
                    # transient (or never observed): treat as a failure for THIS
                    # resolve attempt. The diff loop preserves
                    # `config_ref_versions` for transient refs, so an unchanged
                    # VM whose only churn is a transient ref will not re-enter
                    # `upsert_keys` next cycle — its prior resolved state stays
                    # untouched. A freshly added VM goes to broken until the
                    # ref recovers, which is the right safety posture.
                    logger.warning(
                        "Config %r referenced by VirtualModel %s %s_middleware[%d] is not available "
                        "(transient prefetch failure)",
                        call.config_id,
                        vm_id,
                        phase,
                        index,
                    )
                    return False, []
                raw_config = prefetch.fetched[mref]
            else:
                raw_config = call.config or {}

            try:
                validated = await plugin.validate_middleware_config(config_type, raw_config)
            except (ValueError, InferenceMiddlewareError) as exc:
                logger.warning(
                    "Config validation failed for plugin %r in VirtualModel %s %s_middleware[%d]: %s",
                    plugin_name,
                    vm_id,
                    phase,
                    index,
                    exc,
                )
                return False, []
            except Exception:
                logger.warning(
                    "Unexpected error resolving config for plugin %r in VirtualModel %s %s_middleware[%d]",
                    plugin_name,
                    vm_id,
                    phase,
                    index,
                    exc_info=True,
                )
                return False, []

            resolved.append(
                ResolvedMiddlewareCall(
                    plugin_name=plugin_name,
                    config_type=config_type,
                    resolved_config=validated,
                )
            )

        return True, resolved

    async def notify_upserted(self, vm: SDKVirtualModel) -> None:
        """Call ``on_virtual_model_upserted`` on each plugin referenced by *vm*.

        Errors are logged and swallowed — must not fail the cache refresh.
        """
        if vm.workspace is None or vm.name is None:
            return

        vm_id = f"{vm.workspace}/{vm.name}"
        plugin_vm = _sdk_vm_to_plugin_vm(vm)

        for plugin_name in _referenced_plugins(vm):
            plugin = self.plugins.get(plugin_name)
            if plugin is None:
                continue
            try:
                await plugin.on_virtual_model_upserted(plugin_vm)
            except Exception:
                logger.warning(
                    "Plugin %r raised during on_virtual_model_upserted for %s",
                    plugin_name,
                    vm_id,
                    exc_info=True,
                )

    async def notify_destroyed(self, vm: SDKVirtualModel) -> None:
        """Call ``on_virtual_model_destroyed`` on each plugin referenced by *vm*.

        Errors are logged and swallowed — must not fail the cache refresh.
        """
        if vm.workspace is None or vm.name is None:
            return

        vm_id = f"{vm.workspace}/{vm.name}"
        plugin_vm = _sdk_vm_to_plugin_vm(vm)

        for plugin_name in _referenced_plugins(vm):
            plugin = self.plugins.get(plugin_name)
            if plugin is None:
                continue
            try:
                await plugin.on_virtual_model_destroyed(plugin_vm)
            except Exception:
                logger.warning(
                    "Plugin %r raised during on_virtual_model_destroyed for %s",
                    plugin_name,
                    vm_id,
                    exc_info=True,
                )

    def evict(self, key: tuple[str, str]) -> None:
        """Remove pre-resolved configs for a removed VirtualModel.

        Drops the ``(workspace, name)`` entry from every phase dict. The
        :attr:`broken_vms` membership is intentionally *not* cleared here —
        :meth:`resolve_configs_for_virtual_model` toggles it independently
        based on the next resolution attempt, and the diff loop's removal path
        (in :func:`~nmp.core.inference_gateway.api.virtual_model_cache.refresh_virtual_model_cache`)
        is responsible for discarding it when the VM is gone entirely.
        """
        self.request_middleware_calls.pop(key, None)
        self.response_middleware_calls.pop(key, None)
        self.post_response_middleware_calls.pop(key, None)

    async def shutdown(self) -> None:
        """Call ``on_shutdown`` on all loaded plugins.

        Errors are logged and swallowed — shutdown must complete regardless.
        """
        for plugin_name, plugin in self.plugins.items():
            try:
                await plugin.on_shutdown()
            except Exception:
                logger.warning("Plugin %r raised during on_shutdown", plugin_name, exc_info=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _referenced_plugins(vm: SDKVirtualModel) -> set[str]:
    """Return the set of plugin names referenced in any of *vm*'s middleware lists."""
    names: set[str] = set()
    for calls in (
        vm.request_middleware or [],
        vm.response_middleware or [],
        vm.post_response_middleware or [],
    ):
        for call in calls:
            if call.name:
                names.add(call.name)
    return names


def _sdk_vm_to_plugin_vm(vm: SDKVirtualModel) -> PluginVirtualModel:
    """Convert an SDK VirtualModel to the nemo_platform_plugin VirtualModel for lifecycle hooks.

    Only the fields that plugins care about in lifecycle hooks are mapped.
    """

    def _to_middleware_calls(calls: list | None) -> list[MiddlewareCall]:
        if not calls:
            return []
        return [
            MiddlewareCall(
                name=c.name or "",
                config_type=c.config_type or "",
                config=dict(c.config) if c.config is not None else None,
                config_id=c.config_id,
            )
            for c in calls
        ]

    return PluginVirtualModel(
        name=vm.name or "",
        workspace=vm.workspace or "",
        default_model_entity=vm.default_model_entity,
        models=[
            VirtualModelInferenceConfig(model=m.model, backend_format=m.backend_format)
            for m in (getattr(vm, "models", None) or [])
        ],
        request_middleware=_to_middleware_calls(vm.request_middleware),
        response_middleware=_to_middleware_calls(vm.response_middleware),
        post_response_middleware=_to_middleware_calls(vm.post_response_middleware),
        override_proxy=vm.override_proxy,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


async def load_middleware_plugins(
    model_cache: Any,  # ModelCache
    virtual_model_cache: Any,  # VirtualModelCache
    *,
    plugin_sdk_factory: Callable[[str], AsyncNeMoPlatform] | None = None,
) -> MiddlewareRegistry:
    """Discover ``nemo.inference_middleware`` entry-points, load, and start each plugin.

    For each discovered class:
    1. Instantiate it
    2. Inject the :class:`InferenceMiddlewareCacheAccessorImpl`
    3. Call ``await on_startup()``

    Broken plugins (import failure, ``on_startup`` raising) are logged as warnings and
    excluded from the registry — other plugins continue loading.

    Args:
        model_cache: The IGW's :class:`~nmp.core.inference_gateway.api.model_cache.ModelCache`.
        virtual_model_cache: The IGW's :class:`~nmp.core.inference_gateway.api.virtual_model_cache.VirtualModelCache`.
        plugin_sdk_factory: Optional factory for caller-owned service SDKs keyed by plugin name.

    Returns:
        A :class:`MiddlewareRegistry` containing all successfully loaded plugins.
    """
    accessor = InferenceMiddlewareCacheAccessorImpl(
        _model_cache=model_cache,
        _virtual_model_cache=virtual_model_cache,
    )

    discovered = discover_inference_middleware()
    plugins: dict[str, NemoInferenceMiddleware] = {}

    for name, cls in discovered.items():
        try:
            instance = cls()
            instance._inject_cache(accessor)
            if plugin_sdk_factory is not None:
                sdk = plugin_sdk_factory(name)
                instance._inject_platform_sdk(sdk)
                instance._inject_platform_client(client_from_platform(sdk, AsyncNemoClient))
            await instance.on_startup()
            plugins[name] = instance
            logger.info("Loaded inference middleware plugin %r (%s)", name, cls.__qualname__)
        except Exception:
            logger.warning(
                "Failed to load inference middleware plugin %r — skipping",
                name,
                exc_info=True,
            )

    if not plugins:
        logger.debug("No inference middleware plugins loaded")
    else:
        logger.info("Loaded %d inference middleware plugin(s): %s", len(plugins), ", ".join(sorted(plugins)))

    return MiddlewareRegistry(plugins=plugins)


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


async def execute_request_middleware(
    calls: list[ResolvedMiddlewareCall],
    plugins: dict[str, NemoInferenceMiddleware],
    ctx: InferenceMiddlewareContext,
    request: InferenceRequest,
) -> InferenceRequest | ImmediateResponse:
    """Run the request middleware chain and return the final result.

    Each plugin's ``process_request`` receives the previous plugin's return value.
    The chain short-circuits on the first :class:`~nemo_platform_plugin.inference_middleware.ImmediateResponse`
    — subsequent plugins are not called.

    Args:
        calls: Pre-resolved middleware calls for the request phase.
        plugins: Loaded plugin instances keyed by entry-point name.
        ctx: Per-request context passed to each plugin.
        request: The initial :class:`~nemo_platform_plugin.inference_middleware.InferenceRequest`.

    Returns:
        Either the (possibly mutated) :class:`~nemo_platform_plugin.inference_middleware.InferenceRequest`,
        or an :class:`~nemo_platform_plugin.inference_middleware.ImmediateResponse` if a
        plugin short-circuited the chain.

    Raises:
        :class:`~nemo_platform_plugin.inference_middleware.InferenceMiddlewareError`:
            Re-raised with the plugin's ``status_code`` intact.
        :class:`~fastapi.HTTPException` (503):
            If a required plugin is not loaded in the registry.
        :class:`~fastapi.HTTPException` (500):
            If a plugin raises an unexpected exception.
    """
    result: InferenceRequest | ImmediateResponse = request
    for call in calls:
        if isinstance(result, ImmediateResponse):
            break
        plugin = plugins.get(call.plugin_name)
        if plugin is None:
            raise HTTPException(
                status_code=503,
                detail=f"Plugin '{call.plugin_name}' is not loaded; cannot execute request middleware.",
            )
        try:
            result = await plugin.process_request(ctx, result, call.resolved_config)
            # Re-derive typed_body after each plugin so the next plugin in the
            # chain always receives a fresh, consistent typed view — regardless
            # of whether the plugin mutated in-place, returned a new
            # InferenceRequest, or left typed_body unset.
            if isinstance(result, InferenceRequest):
                result.typed_body = parse_typed_request(result.path, result.body)
        except InferenceMiddlewareError:
            raise
        except Exception:
            logger.exception(
                "Plugin '%s' raised an unexpected error during request middleware",
                call.plugin_name,
            )
            raise InferenceMiddlewareError(
                f"Plugin '{call.plugin_name}' raised an unexpected error during request processing.",
                status_code=500,
            )
    return result


async def execute_response_middleware(
    calls: list[ResolvedMiddlewareCall],
    plugins: dict[str, NemoInferenceMiddleware],
    ctx: InferenceMiddlewareContext,
    response: InferenceResponse,
) -> InferenceResponse:
    """Run the response middleware chain and return the final :class:`~nemo_platform_plugin.inference_middleware.InferenceResponse`.

    Each plugin receives the previous plugin's return value.  Missing plugins are
    logged and skipped (non-fatal); the response passes through unchanged.

    Args:
        calls: Pre-resolved middleware calls for the response phase.
        plugins: Loaded plugin instances.
        ctx: Per-request context passed to each plugin.
        response: The initial :class:`~nemo_platform_plugin.inference_middleware.InferenceResponse`
            wrapping the backend result and response headers.

    Raises:
        :class:`~nemo_platform_plugin.inference_middleware.InferenceMiddlewareError`:
            Re-raised with the plugin's ``status_code`` intact.
        :class:`~nemo_platform_plugin.inference_middleware.InferenceMiddlewareError` (500):
            If a plugin raises an unexpected exception.
    """
    result: InferenceResponse = response
    for call in calls:
        plugin = plugins.get(call.plugin_name)
        if plugin is None:
            logger.warning(
                "Plugin '%s' is not loaded; skipping response middleware call",
                call.plugin_name,
            )
            continue
        try:
            result = await plugin.process_response(ctx, result, call.resolved_config)
        except InferenceMiddlewareError:
            raise
        except Exception:
            logger.exception(
                "Plugin '%s' raised an unexpected error during response middleware",
                call.plugin_name,
            )
            raise InferenceMiddlewareError(
                f"Plugin '{call.plugin_name}' raised an unexpected error during response processing.",
                status_code=500,
            )
    return result


def build_inference_response(
    response_result: Any,
    response_headers: dict[str, str],
    backend_format: BackendFormat | None,
    response_body_annotations: dict[str, Any] | None = None,
) -> InferenceResponse:
    """Build an ``InferenceResponse`` envelope and populate ``typed_body`` when possible."""
    response = InferenceResponse(
        result=response_result,
        headers=response_headers,
        response_body_annotations=dict(response_body_annotations or {}),
    )
    if backend_format is not None:
        if isinstance(response_result, dict):
            response.typed_body = parse_typed_response(backend_format, response_result)
        else:
            # The typed iterator skips chunks the SDK Union doesn't recognise
            # (e.g. Anthropic ``ping`` heartbeats) instead of tearing down the
            # typed view. Raw chunks always come through
            # ``raw_chunks()`` for wire-level serialization.
            typed_stream = TypedResponseStream(backend_format, response_result)
            response.result = typed_stream.raw_chunks()
            response.typed_body = typed_stream

    return response


async def execute_post_response_middleware(
    calls: list[ResolvedMiddlewareCall],
    plugins: dict[str, NemoInferenceMiddleware],
    ctx: InferenceMiddlewareContext,
    response: InferenceResponse,
) -> None:
    """Run post-response middleware fire-and-forget.

    All exceptions are caught, logged, and swallowed — this function must never
    raise.  It is intended to be scheduled via ``asyncio.create_task`` after the
    response has already been sent to the caller.

    Note: only called for non-streaming responses in v1.  Streaming post-response
    middleware is a future enhancement.
    """
    for call in calls:
        plugin = plugins.get(call.plugin_name)
        if plugin is None:
            continue
        try:
            await plugin.process_response(ctx, response, call.resolved_config)
        except Exception:
            logger.warning(
                "Plugin '%s' raised during post_response_middleware (swallowed)",
                call.plugin_name,
                exc_info=True,
            )
