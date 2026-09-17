# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration harness for IGW middleware plugin tests.

Design:

- ``pytest_httpserver`` owns the only real socket. The IGW + Models app
  runs in-process via ``httpx.ASGITransport``. Both the proxy step's
  outbound HTTP and any plugin-side outbound HTTP terminate at the same
  mock-NIM socket.
- Providers are plain (no ``igw-mock-`` prefix); their ``host_url``
  points at the mock NIM so IGW issues real HTTP.
- Assertions read the mock NIM's per-call request log rather than
  threading response IDs through the proxy.

Sync entry points (:meth:`IGWPluginHarness.add_virtual_model`, etc.)
call :func:`asyncio.run` internally and must not run inside a live loop.
Use the ``a``-prefixed siblings (:meth:`aadd_virtual_model`,
:meth:`achat_completions`, :meth:`ause_plugin`) from async tests.

Companion fixtures: :mod:`nmp.core.inference_gateway.testing.fixtures`.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator, Generator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any, cast

from fastapi.testclient import TestClient
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform, omit
from nemo_platform.types.inference import ModelProvider
from nemo_platform.types.inference.middleware_call_param import MiddlewareCallParam
from nemo_platform.types.inference.virtual_model import VirtualModel as SDKVirtualModel
from nemo_platform.types.inference.virtual_model_inference_config_param import VirtualModelInferenceConfigParam
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.discovery import discover_inference_middleware
from nemo_platform_plugin.inference_middleware import NemoInferenceMiddleware
from nemo_platform_plugin.secrets.client import SecretsClient
from nemo_platform_plugin.secrets.types import PlatformSecretCreateRequest
from nmp.common.entities.client import EntityClient
from nmp.common.observability import MARK_INTERNAL_REQUEST_HEADERS
from nmp.common.service.headers import build_downstream_service_headers
from nmp.core.inference_gateway.api.dependencies import (
    global_middleware_registry,
    global_model_cache,
    global_virtual_model_cache,
)
from nmp.core.inference_gateway.api.middleware_registry import (
    InferenceMiddlewareCacheAccessorImpl,
    MiddlewareRegistry,
)
from nmp.core.inference_gateway.api.model_cache import ModelCache, ModelProviderInfo
from nmp.core.inference_gateway.api.virtual_model_cache import (
    VirtualModelCache,
    refresh_virtual_model_cache,
)
from nmp.testing.client import ClientContext
from nmp.testing.mock_chat_completions import (
    BodyPredicate,
    ChatCompletion,
    ChatCompletionStream,
    MockChatCompletionsHandler,
    MockResponse,
    RecordedRequest,
)
from pydantic import SecretStr
from pytest_httpserver import HTTPServer

logger = logging.getLogger(__name__)

DEFAULT_MOCK_CHAT_PATH = "/v1/chat/completions"

DEFAULT_WORKSPACE = "default"
"""Workspace seeded by :func:`~nmp.testing.client.create_test_client` at
module-fixture setup. Exposed on the harness as
:attr:`IGWPluginHarness.workspace` so test bodies don't hardcode
``"default"`` and stay portable if the fixture later issues a per-test
workspace."""


@dataclass
class IGWPluginHarness:
    """Integration harness for IGW middleware plugins.

    Owns a :class:`~nmp.testing.client.ClientContext` (sync + async SDK,
    :class:`TestClient`, :class:`EntityClient`) backed by an in-process
    ASGI IGW + Models app, the mock-NIM :class:`HTTPServer` (the only
    real socket), and a :class:`MockChatCompletionsHandler` pre-mounted
    at ``POST /v1/chat/completions``.

    Construct via the :func:`igw_plugin_harness` pytest fixture.
    """

    sdk: NeMoPlatform
    async_sdk: AsyncNeMoPlatform
    test_client: TestClient
    entity_client: EntityClient

    mock_nim: HTTPServer
    """The only real socket in the test process. Tests can register extra
    routes (e.g. ``/v1/embeddings``) on top of the auto-mounted
    chat-completions handler.

    The auto-mounted handler uses ``expect_request`` (a *permanent*
    matcher). A oneshot matcher for ``/v1/chat/completions`` wins the
    first call; subsequent calls fall through to the permanent
    handler — usually what tests want."""

    handler: MockChatCompletionsHandler
    """The auto-mounted chat-completions handler. Tests interact via
    :meth:`mock_chat_completions` and :meth:`assert_call_count` rather
    than touching this directly."""

    workspace: str
    """Workspace the module-scoped fixture seeded. Reach for this in test
    bodies instead of a literal ``"default"``."""

    _registry: MiddlewareRegistry
    _model_cache: ModelCache
    _vm_cache: VirtualModelCache
    _cache_accessor: InferenceMiddlewareCacheAccessorImpl
    _virtual_models: list[tuple[str, str]]
    """VMs created by this harness, deleted on teardown."""
    _providers: list[tuple[str, str]]
    """Providers created via :meth:`add_provider`, deleted on teardown so
    they can't re-enter the model cache on the next test."""
    _secrets: list[tuple[str, str]]
    """Secrets created via :meth:`create_secret`, deleted on teardown
    after providers."""

    # ------------------------------------------------------------------
    # Builder
    # ------------------------------------------------------------------

    @classmethod
    def _build(
        cls,
        *,
        client_context: ClientContext,
        mock_nim: HTTPServer,
        workspace: str = DEFAULT_WORKSPACE,
        **extra_fields: Any,
    ) -> "IGWPluginHarness":
        """Build a harness around an already-running app.

        Subclasses pass their extra dataclass fields via *extra_fields*
        so one ``cls(...)`` call wires parent + subclass together.
        """
        registry = global_middleware_registry()
        model_cache = global_model_cache()
        vm_cache = global_virtual_model_cache()
        cache_accessor = InferenceMiddlewareCacheAccessorImpl(
            _model_cache=model_cache,
            _virtual_model_cache=vm_cache,
        )

        handler = MockChatCompletionsHandler()
        # Permanent matcher; the handler dispatches by body["model"].
        mock_nim.expect_request(DEFAULT_MOCK_CHAT_PATH, method="POST").respond_with_handler(handler)

        return cls(
            sdk=client_context.sdk,
            async_sdk=client_context.async_sdk,
            test_client=client_context.test_client,
            entity_client=client_context.entity_client,
            mock_nim=mock_nim,
            handler=handler,
            workspace=workspace,
            _registry=registry,
            _model_cache=model_cache,
            _vm_cache=vm_cache,
            _cache_accessor=cache_accessor,
            _virtual_models=[],
            _providers=[],
            _secrets=[],
            **extra_fields,
        )

    def _cleanup(self) -> None:
        """Delete this test's entities, then rebuild the in-memory caches.

        Deletes in FK order: VMs → providers → secrets. Each step
        catches and logs broadly: cleanup runs in ``finally`` after the
        test has already finished, and the goal is to drain every
        tracked entity even if one delete raises. Narrowing to
        ``APIError`` would let a programming error in cleanup code
        strand the remaining deletes — the precise pollution we set
        this up to prevent. Failures are logged with ``exc_info=True``
        so nothing is silently lost.

        **What this does NOT clean up:**

        * Plugins registered persistently at app startup (via the
          ``nemo.inference_middleware`` entry-point group) won't see
          ``on_virtual_model_destroyed`` — only ``registry.evict`` runs.
          Per-VM state in such plugins leaks across tests; register them
          per-test via :meth:`use_plugin` / :meth:`load_plugin` if you
          need clean state.
        * ``registry.broken_vms`` and
          :attr:`VirtualModelCache.config_ref_versions` aren't pruned
          here — they self-heal on the next
          :func:`refresh_virtual_model_cache`. In practice every test
          calls :meth:`add_virtual_model` which triggers a refresh.
        """
        for workspace, name in reversed(self._virtual_models):
            try:
                self.sdk.inference.virtual_models.delete(name=name, workspace=workspace)
            except Exception:  # noqa: BLE001  # see _cleanup docstring
                logger.warning(
                    "Failed to delete VirtualModel %r in workspace %r during harness cleanup",
                    name,
                    workspace,
                    exc_info=True,
                )

        for workspace, name in reversed(self._providers):
            try:
                self.sdk.inference.providers.delete(name=name, workspace=workspace)
            except Exception:  # noqa: BLE001  # see _cleanup docstring
                logger.warning(
                    "Failed to delete ModelProvider %r in workspace %r during harness cleanup",
                    name,
                    workspace,
                    exc_info=True,
                )

        secrets = client_from_platform(self.sdk, SecretsClient)
        for workspace, name in reversed(self._secrets):
            try:
                secrets.delete_secret(name=name, workspace=workspace)
            except Exception:  # noqa: BLE001  # see _cleanup docstring
                logger.warning(
                    "Failed to delete Secret %r in workspace %r during harness cleanup",
                    name,
                    workspace,
                    exc_info=True,
                )

        # Rebuild in-memory caches to match the post-delete entity store.
        for key in self._virtual_models:
            self._registry.evict(key)
        removed = set(self._virtual_models)
        if removed:
            self._vm_cache.rebuild(
                [vm for vm in self._vm_cache.virtual_model_map.values() if (vm.workspace, vm.name) not in removed]
            )
        # Drop deleted providers so the next add_provider fast-path
        # doesn't see ghost ModelProviderInfo rows.
        for workspace, name in self._providers:
            self._model_cache.workspace_name_provider_map.pop((workspace, name), None)
        self._model_cache.rebuild_model_entity_map()

        self._virtual_models.clear()
        self._providers.clear()
        self._secrets.clear()

    # ------------------------------------------------------------------
    # Public conveniences
    # ------------------------------------------------------------------

    @property
    def virtual_models(self) -> tuple[tuple[str, str], ...]:
        """``(workspace, name)`` for every VirtualModel this harness created, in creation order.

        Exposed so a test's own teardown can act on them before :meth:`_cleanup` runs — notably
        to detach a guardrail config from its routes, since the Guardrails service refuses to
        delete a config a VirtualModel still applies.
        """
        return tuple(self._virtual_models)

    @property
    def nim_base_url(self) -> str:
        """OpenAI-compatible base URL (``http://host:port/v1``).

        Pass as ``parameters.base_url``. Both the IGW proxy step and an
        OpenAI client built from this URL hit the auto-mounted handler.
        """
        return self.mock_nim.url_for("/v1")

    @property
    def nim_host_url(self) -> str:
        """Bare ``http://host:port`` — what providers want as ``host_url``."""
        return self.mock_nim.url_for("").rstrip("/")

    # ------------------------------------------------------------------
    # Plugin registration (context-managed)
    # ------------------------------------------------------------------

    def _plugin_sdk(self, name: str) -> AsyncNeMoPlatform:
        headers = dict(self.async_sdk.default_headers)
        headers.update(MARK_INTERNAL_REQUEST_HEADERS)
        headers.update(build_downstream_service_headers(name))
        return self.async_sdk.with_options(set_default_headers=headers)

    @contextmanager
    def use_plugin(
        self,
        name: str,
        plugin: NemoInferenceMiddleware,
        *,
        call_lifecycle: bool = True,
    ) -> Generator[NemoInferenceMiddleware, None, None]:
        """Register *plugin* under *name*; restore the prior entry on exit.

        The cache accessor is injected before yield so plugin cache methods
        work inside the context. Prefer :meth:`load_plugin` for any
        pip-installed plugin so the test exercises its entry-point
        declaration; use this method for workspace-only plugins or to
        substitute a :class:`MagicMock`-spec'd instance.

        With ``call_lifecycle=True`` (the default), ``on_startup`` and
        ``on_shutdown`` each run via :func:`asyncio.run` — i.e. on a
        fresh disposable event loop. Plugins that build loop-bound
        resources in ``on_startup`` (``aiohttp.ClientSession``,
        ``asyncio.Lock``, long-running Tasks) and then use them during
        a request will fail with "attached to a different loop": the
        request runs on yet another loop. Drive those tests from
        ``async def`` and use :meth:`ause_plugin` so both hooks share the
        test's own loop.

        Pass ``call_lifecycle=False`` to skip the hooks — useful for
        ``MagicMock(spec=Plugin)`` or tests that drive the lifecycle
        themselves.
        """
        original_present = name in self._registry.plugins
        original = self._registry.plugins.get(name)

        plugin._inject_cache(self._cache_accessor)
        sdk = self._plugin_sdk(name)
        plugin._inject_platform_sdk(sdk)
        plugin._inject_platform_client(client_from_platform(sdk, AsyncNemoClient))
        if call_lifecycle:
            asyncio.run(plugin.on_startup())
        self._registry.plugins[name] = plugin
        try:
            yield plugin
        finally:
            if original_present:
                assert original is not None
                self._registry.plugins[name] = original
            else:
                self._registry.plugins.pop(name, None)
            if call_lifecycle:
                # Plugin code can raise anything; log rather than raise
                # so a teardown failure doesn't mask the test outcome.
                try:
                    asyncio.run(plugin.on_shutdown())
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "Plugin %r on_shutdown raised during use_plugin teardown",
                        name,
                        exc_info=True,
                    )

    @asynccontextmanager
    async def ause_plugin(
        self,
        name: str,
        plugin: NemoInferenceMiddleware,
        *,
        call_lifecycle: bool = True,
    ) -> AsyncGenerator[NemoInferenceMiddleware, None]:
        """Async variant of :meth:`use_plugin`.

        Both lifecycle hooks run on the test's own loop (the same loop
        the request runs on), so loop-bound resources created in
        ``on_startup`` stay valid through ``on_shutdown``. Use this
        instead of :meth:`use_plugin` for plugins that build long-lived
        loop-bound resources at startup.
        """
        original_present = name in self._registry.plugins
        original = self._registry.plugins.get(name)

        plugin._inject_cache(self._cache_accessor)
        sdk = self._plugin_sdk(name)
        plugin._inject_platform_sdk(sdk)
        plugin._inject_platform_client(client_from_platform(sdk, AsyncNemoClient))
        if call_lifecycle:
            await plugin.on_startup()
        self._registry.plugins[name] = plugin
        try:
            yield plugin
        finally:
            if original_present:
                assert original is not None
                self._registry.plugins[name] = original
            else:
                self._registry.plugins.pop(name, None)
            if call_lifecycle:
                # See use_plugin for why this is a blind catch.
                try:
                    await plugin.on_shutdown()
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "Plugin %r on_shutdown raised during ause_plugin teardown",
                        name,
                        exc_info=True,
                    )

    @contextmanager
    def load_plugin(
        self,
        name: str,
        *,
        call_lifecycle: bool = True,
    ) -> Generator[NemoInferenceMiddleware, None, None]:
        """Load *name* via the ``nemo.inference_middleware`` entry-point group.

        Production-parity: IGW's :func:`load_middleware_plugins` walks
        the same entry-point group, so this exercises the plugin's
        ``pyproject.toml`` declaration and catches misconfigurations
        (missing key, wrong import path) that :meth:`use_plugin` would
        silently gloss over.

        Use :meth:`use_plugin` instead when:

        - The plugin isn't pip-installed (workspace-only plugins).
        - You need a :class:`MagicMock` / :class:`AsyncMock` instance.
        - You need to pre-configure instance state before ``on_startup``.

        ``call_lifecycle`` carries the same loop-binding caveat as
        :meth:`use_plugin` — use :meth:`aload_plugin` for plugins with
        loop-bound startup resources.

        Raises:
            ValueError: If *name* isn't registered in the
                ``nemo.inference_middleware`` entry-point group.
        """
        instance = _instantiate_discovered_plugin(name)
        with self.use_plugin(name, instance, call_lifecycle=call_lifecycle) as plugin:
            yield plugin

    @asynccontextmanager
    async def aload_plugin(
        self,
        name: str,
        *,
        call_lifecycle: bool = True,
    ) -> AsyncGenerator[NemoInferenceMiddleware, None]:
        """Async variant of :meth:`load_plugin`."""
        instance = _instantiate_discovered_plugin(name)
        async with self.ause_plugin(name, instance, call_lifecycle=call_lifecycle) as plugin:
            yield plugin

    # ------------------------------------------------------------------
    # Secret / Provider / VirtualModel creation (refresh hidden)
    # ------------------------------------------------------------------

    def create_secret(
        self,
        *,
        workspace: str,
        name: str,
        value: str,
        description: str | None = None,
    ) -> str:
        """Create a Secret and track it so the harness deletes it on teardown.

        Prefer this over a direct ``SecretsClient.create_secret(...)`` — only
        harness-tracked entities get cleaned up, and an untracked secret
        will leak across tests under module scope (and keep a deleted
        provider's ``api_key_secret_name`` alive on the next refresh).

        Returns *name* so it chains cleanly into
        :meth:`add_provider` (``api_key_secret_name=harness.create_secret(...)``).
        """
        secrets = client_from_platform(self.sdk, SecretsClient)
        secrets.create_secret(
            body=PlatformSecretCreateRequest(name=name, value=SecretStr(value), description=description),
            workspace=workspace,
        )
        self._secrets.append((workspace, name))
        return name

    def add_provider(
        self,
        *,
        workspace: str,
        served_models: Mapping[str, str],
        name: str | None = None,
        host_url: str | None = None,
        enabled_models: Sequence[str] | None = None,
        api_key_secret_name: str | None = None,
    ) -> ModelProvider:
        """Register a real (non-mock) ModelProvider pointing at the mock NIM.

        Call this **before** :meth:`add_virtual_model` for any VM that
        references the provider — an unknown ``default_model_entity``
        silently produces an empty pre-resolved-call list rather than
        raising.

        When *api_key_secret_name* is set, this runs the full
        :func:`refresh_model_cache` so the secret value is resolved via
        the secrets SDK (otherwise the proxy would 424). Without it, the
        method takes a fast path that updates the cache in place.

        Args:
            workspace: Provider workspace.
            served_models: ``model_entity_name`` → ``served_model_name``.
                The served name is what arrives at the upstream — register
                mock handler responses under the same key.
            name: Provider name. Auto-generated if omitted (recommended).
                An explicit duplicate raises ``ConflictError`` so isolation
                breakage fails loudly.
            host_url: Override the default mock-NIM URL.
            enabled_models: Optional enabled-models list for the SDK.
            api_key_secret_name: Existing Secret name to attach as the
                provider's bearer token. Create it via
                :meth:`create_secret` first. Triggers a full cache refresh
                so the secret value is resolved.

        Returns:
            The provider, read back after ``update_status`` so its ``id``
            and ``served_models`` reflect entity-store state.

        Raises:
            ConflictError: If *name* already exists in *workspace*.
        """
        from nmp.testing.utils import short_unique_name

        provider_name = name or short_unique_name("provider")
        host = host_url or self.nim_host_url

        self.sdk.inference.providers.create(
            workspace=workspace,
            name=provider_name,
            host_url=host,
            enabled_models=list(enabled_models) if enabled_models is not None else omit,
            api_key_secret_name=api_key_secret_name if api_key_secret_name is not None else omit,
        )
        # Track right after create so a later raise from update_status /
        # retrieve still leaves the provider eligible for teardown.
        self._providers.append((workspace, provider_name))

        # served_models has to go through update_status — the create path doesn't accept it.
        self.sdk.inference.providers.update_status(
            name=provider_name,
            workspace=workspace,
            served_models=[
                {
                    "model_entity_id": f"{workspace}/{entity_name}",
                    "served_model_name": served_name,
                }
                for entity_name, served_name in served_models.items()
            ],
        )

        # Read authoritative state back so the cache stays in sync with
        # whatever id / served_models shape / timestamps the entity store
        # actually assigned.
        provider = self.sdk.inference.providers.retrieve(name=provider_name, workspace=workspace)

        if api_key_secret_name is not None:
            # Full refresh resolves the secret via the secrets SDK.
            asyncio.run(self._refresh_model_cache())
        else:
            # Fast path: in-place cache update; skips secrets plumbing.
            self._model_cache.update_model_info(ModelProviderInfo(model_provider=provider))
            self._model_cache.rebuild_model_entity_map()

        return provider

    def add_virtual_model(
        self,
        *,
        workspace: str,
        name: str,
        default_model_entity: str | None = None,
        models: Sequence[VirtualModelInferenceConfigParam] = (),
        request_middleware: Sequence[MiddlewareCallParam] = (),
        response_middleware: Sequence[MiddlewareCallParam] = (),
        post_response_middleware: Sequence[MiddlewareCallParam] = (),
    ) -> SDKVirtualModel:
        """Create a VirtualModel and refresh the VM cache so it routes immediately.

        Sync entry — uses :func:`asyncio.run`, so don't call this inside
        a live loop. Use :meth:`aadd_virtual_model` from async tests.

        Call :meth:`add_provider` first for any provider this VM
        references; otherwise middleware-config pre-resolution sees an
        empty model cache.

        ``models`` is the per-VM list of entity refs with optional
        ``backend_format`` overrides. Plugins like ``nemo-switchyard``
        read this in ``on_virtual_model_upserted`` to build their
        routing tables::

            models=[
                {"model": "default/main", "backend_format": "OPENAI_CHAT"},
                {"model": "default/claude", "backend_format": "ANTHROPIC_MESSAGES"},
            ]

        .. note::
            Plugin errors during upsert are swallowed. Both
            ``validate_middleware_config`` and ``on_virtual_model_upserted``
            failures get logged-and-continued by the registry, so a
            plugin that rejects a VM at upsert time won't make this
            method raise — the VM lands in the store with an empty
            phase-list and the rejection surfaces as "no factory
            registered" on the first request. Assert rejections via
            :meth:`chat_completions`, not via this method's return.
        """
        vm = self._create_virtual_model(
            workspace=workspace,
            name=name,
            default_model_entity=default_model_entity,
            models=models,
            request_middleware=request_middleware,
            response_middleware=response_middleware,
            post_response_middleware=post_response_middleware,
        )
        asyncio.run(self._refresh_vm_cache())
        return vm

    async def aadd_virtual_model(
        self,
        *,
        workspace: str,
        name: str,
        default_model_entity: str | None = None,
        models: Sequence[VirtualModelInferenceConfigParam] = (),
        request_middleware: Sequence[MiddlewareCallParam] = (),
        response_middleware: Sequence[MiddlewareCallParam] = (),
        post_response_middleware: Sequence[MiddlewareCallParam] = (),
    ) -> SDKVirtualModel:
        """Async sibling of :meth:`add_virtual_model`."""
        vm = self._create_virtual_model(
            workspace=workspace,
            name=name,
            default_model_entity=default_model_entity,
            models=models,
            request_middleware=request_middleware,
            response_middleware=response_middleware,
            post_response_middleware=post_response_middleware,
        )
        await self._refresh_vm_cache()
        return vm

    def refresh_caches(self) -> None:
        """Refresh model + VM cache (sync). Model cache first because VM
        resolution depends on the served-model topology.

        The model-cache refresh resolves provider secrets via the SDK;
        call this instead of relying on :meth:`add_provider`'s fast path
        when ``api_key_secret_name`` is set.
        """
        asyncio.run(self._refresh_all_caches())

    async def arefresh_caches(self) -> None:
        """Async sibling of :meth:`refresh_caches`."""
        await self._refresh_all_caches()

    async def _refresh_all_caches(self) -> None:
        await self._refresh_model_cache()
        await self._refresh_vm_cache()

    def _create_virtual_model(
        self,
        *,
        workspace: str,
        name: str,
        default_model_entity: str | None,
        models: Sequence[VirtualModelInferenceConfigParam],
        request_middleware: Sequence[MiddlewareCallParam],
        response_middleware: Sequence[MiddlewareCallParam],
        post_response_middleware: Sequence[MiddlewareCallParam],
    ) -> SDKVirtualModel:
        create_kwargs: dict[str, Any] = {
            "workspace": workspace,
            "name": name,
            "request_middleware": list(request_middleware),
            "response_middleware": list(response_middleware),
            "post_response_middleware": list(post_response_middleware),
        }
        if default_model_entity is not None:
            create_kwargs["default_model_entity"] = default_model_entity
        if models:
            # SDK expects an Iterable of VirtualModelInferenceConfigParam
            # (TypedDict). Skip when empty so we don't send an empty list
            # that some validators treat as "explicitly clear models".
            create_kwargs["models"] = list(models)
        vm = self.sdk.inference.virtual_models.create(**create_kwargs)
        self._virtual_models.append((workspace, name))
        return vm

    async def _refresh_vm_cache(self) -> None:
        await refresh_virtual_model_cache(
            self._vm_cache,
            self.async_sdk,
            registry=self._registry,
        )

    async def _refresh_model_cache(self) -> None:
        # Local import keeps module load cheap; refresh_model_cache pulls
        # in secrets SDK plumbing only needed when explicitly invoked.
        from nmp.core.inference_gateway.api.model_cache import (
            model_provider_getter_from_sdk,
            refresh_model_cache,
        )

        await refresh_model_cache(
            model_cache=self._model_cache,
            model_provider_getter=model_provider_getter_from_sdk(self.async_sdk),
            secrets_sdk=self.async_sdk,
            virtual_model_cache=self._vm_cache,
            middleware_registry=self._registry,
        )

    # ------------------------------------------------------------------
    # Mock NIM convenience
    # ------------------------------------------------------------------

    def mock_chat_completions(self, model: str, responses: Sequence[MockResponse]) -> None:
        """Queue *responses* for chat-completion calls with ``body["model"] == model``.

        *model* is the value that arrives at the upstream. For
        ``served_models={"main": "main"}`` on a ``default/main`` entity,
        the upstream sees ``"model": "main"`` — register under ``"main"``.

        Plugin-issued outbound calls (Guardrails rails, etc.) typically
        send the workspace-qualified entity id like ``"default/main"``;
        register a separate queue for those.

        Repeated calls append; the queue is consumed in order and the
        last response is reused once drained. Response bodies built with
        :func:`chat_completion` / :func:`chat_completion_chunk` that
        leave ``model`` unset are auto-stamped with the dispatch key.

        Raises:
            ValueError: If *responses* is empty.
        """
        for resp in responses:
            if isinstance(resp, ChatCompletion) and resp.body.get("model") is None:
                resp.body["model"] = model
            elif isinstance(resp, ChatCompletionStream):
                for chunk in resp.chunks:
                    if isinstance(chunk, dict) and chunk.get("model") is None:
                        chunk["model"] = model
        self.handler.add_responses(model, responses)

    # ------------------------------------------------------------------
    # Convenience callers
    # ------------------------------------------------------------------

    def chat_completions(
        self,
        *,
        workspace: str,
        body: dict[str, Any],
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Call IGW's OpenAI-compatible chat completions endpoint via the SDK."""
        result = self.sdk.inference.gateway.openai.post(
            "v1/chat/completions",
            workspace=workspace,
            body=body,
            extra_headers=dict(extra_headers) if extra_headers is not None else None,
        )
        return _coerce_dict(result)

    async def achat_completions(
        self,
        *,
        workspace: str,
        body: dict[str, Any],
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Async sibling of :meth:`chat_completions`."""
        result = await self.async_sdk.inference.gateway.openai.post(
            "v1/chat/completions",
            workspace=workspace,
            body=body,
            extra_headers=dict(extra_headers) if extra_headers is not None else None,
        )
        return _coerce_dict(result)

    def stream_chat_completions(
        self,
        *,
        workspace: str,
        body: dict[str, Any],
        extra_headers: Mapping[str, str] | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Call the chat-completions endpoint with ``stream=True``.

        Uses :class:`TestClient` directly because the SDK's ``post``
        buffers the full body before returning, defeating streaming.

        Return type depends on ``Content-Type``:

        - ``text/event-stream`` → list of parsed SSE chunks in order
          (``data: [DONE]`` dropped). The normal case: upstream streamed
          and IGW relayed.
        - ``application/json`` → the raw JSON dict. Happens when a plugin
          short-circuits the proxy with :class:`ImmediateResponse` (e.g.
          an input rail blocks before any token streams). Demand SSE
          with ``isinstance(result, list)`` if your assertion requires it.

        Raises:
            httpx.HTTPStatusError: On non-2xx from IGW.
        """
        streaming_body = {**body, "stream": True}
        path = f"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/chat/completions"
        response = self.test_client.post(
            path,
            json=streaming_body,
            headers=dict(extra_headers) if extra_headers else None,
        )
        response.raise_for_status()
        if response.headers.get("content-type", "").startswith("text/event-stream"):
            return _parse_sse_text(response.text)
        return response.json()

    # ------------------------------------------------------------------
    # First-class assertions
    # ------------------------------------------------------------------

    def requests_for(self, model: str) -> list[RecordedRequest]:
        """Return all recorded requests whose ``body["model"] == model``, in order."""
        return [r for r in self.handler.request_log if r.model == model]

    def assert_call_count(self, model: str, n: int) -> None:
        """Assert *model* received exactly *n* requests."""
        actual = self.handler.call_counts.get(model, 0)
        assert actual == n, (
            f"Expected {n} call(s) to model={model!r}, got {actual}. All counts: {dict(self.handler.call_counts)}."
        )

    def assert_called_once(self, model: str) -> None:
        """Assert *model* received exactly one request."""
        self.assert_call_count(model, 1)

    def assert_request_messages_contain(
        self,
        model: str,
        substring: str,
        *,
        index: int = 0,
    ) -> None:
        """Assert the *index*-th request to *model* has *substring* in any message.

        Searches across each message's ``content`` field. Raises
        ``AssertionError`` if no request at *index* or the substring is
        missing.
        """
        recorded_for_model = self.requests_for(model)
        if index < 0 or index >= len(recorded_for_model):
            raise AssertionError(
                f"No request at index {index} for model={model!r}; saw {len(recorded_for_model)} call(s)."
            )
        recorded = recorded_for_model[index]
        haystack = " ".join(str(m.get("content", "")) for m in recorded.body.get("messages", []))
        assert substring in haystack, (
            f"Expected {substring!r} in messages for model={model!r} (call #{index}), "
            f"got messages={recorded.body.get('messages')!r}."
        )

    def assert_call_order(self, models: Sequence[str]) -> None:
        """Assert the recorded sequence of model values equals *models*."""
        actual = list(self.handler.call_order)
        expected = list(models)
        assert actual == expected, f"Expected call order {expected!r}, got {actual!r}."

    def assert_no_calls_to(self, model: str) -> None:
        """Assert *model* received zero requests."""
        actual = self.handler.call_counts.get(model, 0)
        assert actual == 0, f"Expected zero calls to model={model!r}, got {actual}."

    def assert_request_body_for(
        self,
        model: str,
        predicate: BodyPredicate,
        *,
        index: int = 0,
    ) -> None:
        """Assert *predicate(body)* is true for the *index*-th request to *model*.

        Generalises :meth:`assert_request_messages_contain` to any body
        property (tool calls, response_format, embedding inputs,
        plugin-injected fields). The predicate gets the parsed JSON
        body verbatim.

        Raises:
            AssertionError: If no request at *index*, or predicate returns falsy.
        """
        recorded_for_model = self.requests_for(model)
        if index < 0 or index >= len(recorded_for_model):
            raise AssertionError(
                f"No request at index {index} for model={model!r}; saw {len(recorded_for_model)} call(s)."
            )
        recorded = recorded_for_model[index]
        if not predicate(recorded.body):
            raise AssertionError(f"Predicate failed for model={model!r} (call #{index}); body={recorded.body!r}.")

    def assert_request_path_for(
        self,
        model: str,
        path: str,
        *,
        index: int = 0,
    ) -> None:
        """Assert the *index*-th recorded request to *model* arrived on *path*.

        Exact match — include the leading slash. Useful for plugins that
        rewrite ``InferenceRequest.path`` mid-pipeline (e.g. switchyard
        rerouting OpenAI Chat to Anthropic ``v1/messages``); without
        this you only prove the in-memory rewrite happened, not that it
        reached the wire.

        Raises:
            AssertionError: If no request at *index*, or path mismatch.
        """
        recorded_for_model = self.requests_for(model)
        if index < 0 or index >= len(recorded_for_model):
            raise AssertionError(
                f"No request at index {index} for model={model!r}; saw {len(recorded_for_model)} call(s)."
            )
        recorded = recorded_for_model[index]
        if recorded.path != path:
            raise AssertionError(f"Path on call #{index} to model={model!r} was {recorded.path!r}, expected {path!r}.")

    def assert_request_headers_contain(
        self,
        model: str,
        header: str,
        value: str | None = None,
        *,
        index: int = 0,
    ) -> None:
        """Assert the *index*-th request to *model* carries *header*.

        Case-insensitive (HTTP semantics). With *value* unset only
        presence is asserted; with *value* set, an exact match. Use
        :meth:`requests_for` directly for substring or duplicate-header
        assertions.

        Raises:
            AssertionError: If no request at *index*, the header is
                absent, or *value* doesn't match.
        """
        recorded_for_model = self.requests_for(model)
        if index < 0 or index >= len(recorded_for_model):
            raise AssertionError(
                f"No request at index {index} for model={model!r}; saw {len(recorded_for_model)} call(s)."
            )
        recorded = recorded_for_model[index]
        actual = recorded.header(header)
        if actual is None:
            header_names = sorted({name for name, _ in recorded.headers})
            raise AssertionError(
                f"Header {header!r} not present on call #{index} to model={model!r}. Headers seen: {header_names}."
            )
        if value is not None and actual != value:
            raise AssertionError(
                f"Header {header!r} on call #{index} to model={model!r} was {actual!r}, expected {value!r}."
            )

    # ------------------------------------------------------------------
    # Post-response (fire-and-forget) flushing
    # ------------------------------------------------------------------

    async def aflush_post_response(self) -> None:
        """Await every fire-and-forget post-response task IGW has scheduled.

        ``proxy.py`` appends each :func:`execute_post_response_middleware`
        task to ``app.state.pending_post_response_tasks`` (set up by the
        fixture). This drains and awaits them.

        **Loop constraint**: post-response tasks are bound to the loop
        that scheduled them — the request loop. Drive your request from
        ``async def`` via :meth:`achat_completions` so the request and
        flush share a loop. Calling this after a sync
        :meth:`chat_completions` doesn't work: the SDK's transient loop
        is already torn down.

        Exceptions raised by post-response middleware are **not** raised
        from here — they're swallowed in
        :func:`execute_post_response_middleware` (matching production's
        fire-and-forget contract); ``asyncio.gather(return_exceptions=True)``
        ensures one failure doesn't stop the rest of the flush.
        """
        pending = self._pending_post_response_tasks()
        if pending is None:
            raise RuntimeError(
                "Post-response task tracking is not enabled. "
                "The fixture must initialise `app.state.pending_post_response_tasks = []` "
                "before any request runs."
            )
        if not pending:
            return
        in_flight = list(pending)
        pending.clear()
        await asyncio.gather(*in_flight, return_exceptions=True)

    def _pending_post_response_tasks(self) -> list[asyncio.Task[None]] | None:
        # TestClient.app is typed as bare ASGIApp; cast so .state is reachable.
        from fastapi import FastAPI

        app = cast(FastAPI, self.test_client.app)
        return getattr(app.state, "pending_post_response_tasks", None)


def _coerce_dict(value: Any) -> dict[str, Any]:
    """Cast an SDK response (dict or Pydantic model) to ``dict``."""
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(
        f"chat_completions returned {type(value).__name__!r}; expected dict or a Pydantic model. Value: {value!r}"
    )


def _parse_sse_text(text: str) -> list[dict[str, Any]]:
    """Parse a buffered SSE response into chunk dicts.

    Skips ``data: [DONE]`` and silently drops malformed JSON lines,
    matching IGW's own ``_parse_sse_stream`` permissiveness.
    """
    import json

    chunks: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            break
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            chunks.append(parsed)
    return chunks


def _instantiate_discovered_plugin(name: str) -> NemoInferenceMiddleware:
    """Look up *name* in the entry-point group and instantiate.

    Discovery is cached for the process lifetime; tests adding/removing
    entry points dynamically must clear the cache themselves. The
    error on miss points at the two most common fixes (install the
    package, or fall back to :meth:`use_plugin`).
    """
    discovered = discover_inference_middleware()
    cls = discovered.get(name)
    if cls is None:
        available = sorted(discovered)
        raise ValueError(
            f"No plugin registered under entry-point name {name!r} in the "
            f"'nemo.inference_middleware' group. Available: {available}.\n"
            "Install the plugin's package as a test dependency (so its "
            "pyproject.toml entry point is discoverable via importlib.metadata), "
            "or fall back to use_plugin(name, instance) with a directly-instantiated "
            "plugin object."
        )
    return cls()


@dataclass
class IGWLoopbackHarness(IGWPluginHarness):
    """:class:`IGWPluginHarness` plus IGW served on a real ``127.0.0.1`` port.

    Prefer :class:`IGWPluginHarness` for most tests. Reach for this
    harness only when the in-process app needs to be reachable over a
    real socket — e.g. when the plugin calls
    :meth:`~nemo_platform_plugin.inference_middleware.InferenceMiddlewareCacheAccessor.get_openai_compatible_inference_url_and_model`
    and the returned URL must actually work, or when plugin outbound
    HTTP needs to go through IGW's full request pipeline instead of
    landing at the upstream mock.

    Costs a uvicorn thread and an HTTP hop on every plugin-side
    outbound request — hence opt-in.

    .. warning::

        **Two-loop limitation.** This harness drives the FastAPI app
        from two event loops: the TestClient's (for SDK requests via
        ASGI transport) and uvicorn's (for plugin-originated HTTP
        hitting the loopback URL). Implications:

        * The fixture overrides :func:`global_http_client` with a
          per-request :class:`aiohttp.ClientSession` so the proxy step's
          client is created on the loop handling the request.
        * A plugin that builds a long-lived loop-bound resource
          (``aiohttp.ClientSession``, ``asyncio.Lock``, long-running
          ``Task``) in ``on_startup`` and uses it from
          ``process_request`` will fail with "attached to a different
          loop" — the request loop is different. Wire those lazily, or
          test via the plain :class:`IGWPluginHarness` where only one
          loop is in play.
        * Other shared production resources (connection pools, async
          caches, ``asyncio.Queue``) carry the same risk and may need
          per-request dependency overrides too.
    """

    igw_loopback_base_url: str
    """``http://<host>:<port>`` — bare loopback root, no path. Use
    :meth:`igw_openai_loopback_url` for the workspace-scoped variant."""

    def igw_openai_loopback_url(self, workspace: str) -> str:
        """Workspace-scoped OpenAI-compatible loopback URL (includes ``/v1``).

        Pass as ``parameters.base_url`` to route plugin outbound HTTP
        through IGW's openai-compatible proxy for *workspace*.
        """
        return f"{self.igw_loopback_base_url}/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1"


__all__ = [
    "BodyPredicate",
    "DEFAULT_MOCK_CHAT_PATH",
    "IGWLoopbackHarness",
    "IGWPluginHarness",
]
