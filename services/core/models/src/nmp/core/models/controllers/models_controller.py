# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import threading
from logging import getLogger
from typing import Optional

import httpx
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import NotFoundError
from nemo_platform_plugin.models.client import AsyncModelsClient
from nemo_platform_plugin.models.types import ModelDeployment, ModelDeploymentConfig, ModelDeploymentStatus, ModelEntity
from nmp.common.controller import Controller, HeartbeatMixin
from nmp.common.entities.utils import parse_entity_ref
from nmp.common.sdk_factory import get_async_platform_sdk
from nmp.core.models.app import parse_model_name_revision
from nmp.core.models.config import config as models_config
from nmp.core.models.controllers.backends.backends import ServiceBackend
from nmp.core.models.controllers.backends.registry import BackendRegistry
from nmp.core.models.controllers.context import ModelContext
from nmp.core.models.controllers.deployment_reconciler import ModelDeploymentReconciler
from nmp.core.models.controllers.entity_cache import ModelEntityCache
from nmp.core.models.controllers.provider_reconciler import ModelProviderReconciler

logger = getLogger(__name__)

_CONTROLLER_HTTP_TIMEOUT = httpx.Timeout(timeout=60, connect=5.0)
_CONTROLLER_HTTP_LIMITS = httpx.Limits(max_connections=100, max_keepalive_connections=20)

NON_TERMINAL_STATES: list[ModelDeploymentStatus] = [
    ModelDeploymentStatus.CREATED,
    ModelDeploymentStatus.PENDING,
    ModelDeploymentStatus.READY,
    ModelDeploymentStatus.DELETING,
    ModelDeploymentStatus.DELETED,  # Poll DELETED deployments to clean them up after grace period
]


class ModelsController(HeartbeatMixin, Controller):
    """
    Models Controller manages the lifecycle of ModelDeployment objects.

    This controller:
    - Queries Models API for ModelDeployments and ModelProviders in non-terminal states
    - Delegates deployment reconciliation to ModelDeploymentReconciler
    - Delegates provider reconciliation to ModelProviderReconciler
    - Orchestrates the overall controller loop
    """

    def __init__(
        self,
        backend_registry: BackendRegistry,
        stop_signal: threading.Event | None = None,
    ) -> None:
        self._is_healthy = False
        self._backend_registry = backend_registry
        self._stop_signal = stop_signal

        self._loop = asyncio.new_event_loop()
        self._current_task: asyncio.Task | None = None

        # Use service principal for controller - runs in background thread without user context
        self._models_sdk = get_async_platform_sdk(
            as_service="models",
            internal=True,
            http_client=httpx.AsyncClient(
                timeout=_CONTROLLER_HTTP_TIMEOUT,
                limits=_CONTROLLER_HTTP_LIMITS,
                follow_redirects=True,
            ),
        )
        self._models_client = client_from_platform(self._models_sdk, AsyncModelsClient)
        self._service_backends = backend_registry.list_backends()

        # Shared by both reconcilers; re-read at the start of each phase that
        # writes entities so neither phase works from state the other has changed.
        self._entity_cache = ModelEntityCache(models_sdk=self._models_sdk, emit_heartbeat=self.emit_heartbeat)

        # Initialize reconcilers
        self._deployment_reconciler = ModelDeploymentReconciler(
            models_sdk=self._models_sdk,
            backend_registry=backend_registry,
            controller_config=models_config.controller,
            entity_cache=self._entity_cache,
            emit_heartbeat=self.emit_heartbeat,
        )
        self._provider_reconciler = ModelProviderReconciler(
            models_sdk=self._models_sdk,
            controller_config=models_config.controller,
            entity_cache=self._entity_cache,
            emit_heartbeat=self.emit_heartbeat,
        )

        logger.info("Models Controller initialized")
        logger.info(f"Available backends: {', '.join(self._backend_registry.list_backends())}")

    def shutdown(self) -> None:
        """Clean up controller resources.

        Closes the asyncio event loop. Should be called when the controller
        is no longer needed (e.g., during shutdown).
        """
        if self._loop is not None and not self._loop.is_closed():
            try:
                self._loop.run_until_complete(self._models_sdk.close())
            except Exception as e:
                logger.warning(f"Error closing event loop: {e}")
            finally:
                self._loop.close()
                logger.debug("Models controller event loop closed")
        self._backend_registry.shutdown_all_backends()

    def cancel_step(self) -> None:
        """Cancel the currently running async step from another thread.

        Thread-safe. Uses call_soon_threadsafe to schedule cancellation
        on the controller's event loop. This causes the running coroutine
        to raise CancelledError, which step() handles gracefully.
        """
        task = self._current_task
        if task is not None and not task.done() and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(task.cancel)

    @property
    def is_healthy(self) -> bool:
        return self._is_healthy

    def get_service_backend(self) -> ServiceBackend:
        """Get the service backend (single backend enabled at a time; see reconciler for details)."""
        return self._backend_registry.get_backend()

    async def _retrieve_deployment_config(
        self, config_ref: str, config_version: int | str, deployment_workspace: str
    ) -> ModelDeploymentConfig:
        """Retrieve the ModelDeploymentConfig from the API.

        Args:
            config_ref: The config reference (name)
            config_version: The config version
            deployment_workspace: The deployment's workspace (used if config_ref doesn't contain workspace)

        Returns:
            The ModelDeploymentConfig object
        """
        try:
            ref = parse_entity_ref(config_ref, default_workspace=deployment_workspace)
            workspace, name = ref.workspace, ref.name

            logger.debug(f"Fetching ModelDeploymentConfig {workspace}/{name}@{config_version}")
            config = (
                await self._models_client.get_deployment_config_version(
                    name=str(config_version),
                    workspace=workspace,
                    config=name,
                )
            ).data()
            return config
        except Exception as e:
            logger.error(f"Failed to fetch ModelDeploymentConfig {config_ref}@{config_version}: {e}")
            raise

    async def _retrieve_model_entity_for_config(self, config: ModelDeploymentConfig) -> Optional[ModelEntity]:
        """Get model entity from Models API v2 based on deployment config.

        Extracts model information from the config and queries Models API v2 models endpoint.
        If config.model_entity_id is set, it takes precedence over model_spec-derived values.

        Args:
            config: The ModelDeploymentConfig

        Returns:
            Model entity if found and model info available, None otherwise
        """
        try:
            # Prefer explicit model_entity_id when set (e.g. generic llm-nim with entity reference)
            if config.model_entity_id:
                model_workspace, model_name, revision = parse_model_name_revision(
                    model_name=config.model_entity_id,
                )
                if model_workspace and model_name:
                    logger.debug(
                        f"Querying Models API for model entity from config.model_entity_id: "
                        f"{model_workspace}/{model_name}@{revision or 'latest'}"
                    )
                    return await self._retrieve_model_entity(
                        workspace=model_workspace,
                        model_name=model_name,
                        revision=revision,
                    )

            # Extract model information from model_spec when model_entity_id not set
            model_spec = config.model_spec
            if not model_spec:
                logger.debug("No model_spec config, skipping entity store query")
                return None

            # Parse model configuration using unified parsing utility
            model_workspace, model_name, revision = parse_model_name_revision(
                model_namespace=model_spec.model_namespace,
                model_name=model_spec.model_name,
                model_revision=model_spec.model_revision,
            )

            if not model_workspace or not model_name:
                logger.debug(
                    f"Missing model workspace or name in config (workspace={model_workspace}, name={model_name})"
                )
                return None

            logger.debug(f"Querying Models API for model entity from config: {model_workspace}/{model_name}@{revision}")
            model_entity = await self._retrieve_model_entity(
                workspace=model_workspace,
                model_name=model_name,
                revision=revision,
            )

            return model_entity

        except Exception as e:
            logger.warning(f"Error getting model entity for config: {e}")
            return None

    async def _retrieve_model_entity(
        self, workspace: str, model_name: str, revision: Optional[str] = None
    ) -> Optional[ModelEntity]:
        """Get model entity from Models API v2 endpoint.

        Args:
            workspace: Model workspace
            model_name: Model name
            revision: Optional model revision/version

        Returns:
            Model entity if found, None otherwise
        """
        try:
            # Construct full model name with revision if provided
            full_model_name = model_name
            if revision:
                full_model_name = f"{model_name}@{revision}"

            logger.debug(f"Querying Models API for model entity: {workspace}/{full_model_name}")

            if revision or not self._entity_cache.loaded:
                # A revision resolves server-side and does not correspond to an
                # cache key, so it has to be fetched directly.
                model_entity = (
                    await self._models_client.get_model(
                        name=full_model_name,
                        workspace=workspace,
                    )
                ).data()
            else:
                model_entity = self._entity_cache.get(workspace, model_name)
                if model_entity is None:
                    logger.debug(f"No model entity found in Models API: {workspace}/{full_model_name}")
                    return None

            if model_entity:
                logger.debug(f"Successfully retrieved model entity from Models API: {workspace}/{model_name}")
                return model_entity

            # SDK query succeeded but returned None/empty - this shouldn't happen normally
            logger.warning(
                f"Models API query succeeded but returned empty result for model: {workspace}/{full_model_name}"
            )
            return None

        except NotFoundError:
            # Expected for NIMs with baked-in weights that don't have pre-registered model entities
            logger.debug(f"No model entity found in Models API: {workspace}/{full_model_name}")
            return None
        except Exception:
            logger.exception(f"Error querying Models API for model {workspace}/{full_model_name}")
            return None

    async def retrieve_non_terminal_deployments(self) -> list[ModelContext]:
        """Query Models API for all ModelDeployments in non-terminal states.

        Returns a list of ModelContext objects with pre-fetched related data:
        - model_deployment_config (if deployment has config reference)
        - model_provider (if deployment has model_provider_id)
        - model_entity (if available from config)
        """
        logger.debug(f"Querying for deployments in states: {NON_TERMINAL_STATES}")
        all_deployment_graphs: list[ModelContext] = []

        for status in NON_TERMINAL_STATES:
            # Check stop signal between status queries to bail out early
            if self._is_stopping():
                logger.debug("Stop signal received, aborting deployment queries")
                return all_deployment_graphs

            try:
                logger.debug(f"Querying ModelDeployments with status: {status} across all workspaces")
                # SDK returns AsyncPaginator - iterate through all pages
                resp = await self._models_client.list_deployments(
                    workspace="-",  # Cross-workspace query
                    query_params={
                        "filter": json.dumps({"status": status.value}),
                        "all_versions": True,
                        "page_size": 1000,
                    },
                )
                logger.debug(f"Got paginator response for status {status}, iterating...")

                # Collect all deployments from paginator
                deployments = [deployment async for deployment in resp.items()]
                logger.debug(f"Iteration complete for status {status}, got {len(deployments)} deployment(s)")

                if deployments:
                    logger.debug(f"Found {len(deployments)} deployment(s) in {status} state")
                    for dep in deployments:
                        logger.debug(f"  - {dep.workspace}/{dep.name} (status={dep.status})")

                    # Build ModelDeploymentContext for each deployment
                    for deployment in deployments:
                        config = None
                        provider = None
                        entity = None

                        # Fetch config if available
                        if deployment.config and deployment.config_version:
                            try:
                                config = await self._retrieve_deployment_config(
                                    deployment.config, deployment.config_version, deployment.workspace
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to fetch config for deployment {deployment.workspace}/{deployment.name}: {e}"
                                )

                        # Fetch provider if available
                        if deployment.model_provider_id:
                            try:
                                _prov_ref = parse_entity_ref(deployment.model_provider_id)
                                provider_workspace, provider_name = _prov_ref.workspace, _prov_ref.name
                                provider = (
                                    await self._models_client.get_provider(
                                        name=provider_name,
                                        workspace=provider_workspace,
                                    )
                                ).data()
                            except Exception as e:
                                logger.warning(
                                    f"Failed to fetch provider for deployment {deployment.workspace}/{deployment.name}: {e}"
                                )

                        # Fetch model entity if config is available
                        if config:
                            entity = await self._retrieve_model_entity_for_config(config)

                        all_deployment_graphs.append(
                            ModelContext(
                                model_deployment=deployment,
                                model_deployment_config=config,
                                model_provider=provider,
                                model_entity=entity,
                            )
                        )

            except Exception as e:
                logger.warning(f"Error querying deployments with status {status}: {e}")

        return all_deployment_graphs

    async def retrieve_model_providers(self) -> list[ModelContext] | None:
        """Get all model providers from the Models API with pre-fetched related data.

        Returns a list of ModelContext objects with pre-fetched related data:
        - model_deployment (if provider has model_deployment_id)
        - model_deployment_config (if deployment has config reference)
        - model_entity (if available from config or provider)

        Returns None if provider listing failed, so callers can distinguish that
        from a successful empty result and avoid destructive cleanup decisions.
        """
        provider_contexts: list[ModelContext] = []

        try:
            providers = await self._models_client.list_providers(
                workspace="-",  # Cross-workspace query
            )

            async for provider in providers.items():
                deployment = None
                config = None
                entity = None

                # Fetch deployment if provider has model_deployment_id
                if provider.model_deployment_id:
                    try:
                        _depl_ref = parse_entity_ref(provider.model_deployment_id)
                        deployment_workspace, deployment_name = _depl_ref.workspace, _depl_ref.name
                        deployment = (
                            await self._models_client.get_deployment(
                                name=deployment_name,
                                workspace=deployment_workspace,
                            )
                        ).data()

                        # Fetch config if deployment has config reference
                        if deployment and deployment.config and deployment.config_version:
                            try:
                                config = await self._retrieve_deployment_config(
                                    deployment.config, str(deployment.config_version), deployment.workspace
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to fetch config for provider {provider.workspace}/{provider.name}: {e}"
                                )
                    except Exception as e:
                        logger.warning(
                            f"Failed to fetch deployment for provider {provider.workspace}/{provider.name}: {e}"
                        )

                # Fetch model entity if config is available
                if config:
                    entity = await self._retrieve_model_entity_for_config(config)

                provider_contexts.append(
                    ModelContext(
                        model_provider=provider,
                        model_deployment=deployment,
                        model_deployment_config=config,
                        model_entity=entity,
                    )
                )

        except Exception as e:
            logger.warning(f"Error querying providers: {e}")
            return None

        return provider_contexts

    async def retrieve_error_deployments(self) -> list[ModelDeployment]:
        """Query Models API for all ModelDeployments in ERROR state.

        Returns lightweight deployment objects (no config/provider/entity prefetch)
        since GC only needs the deployment itself and its timestamps.
        """
        try:
            resp = await self._models_client.list_deployments(
                workspace="-",
                query_params={
                    "filter": json.dumps({"status": ModelDeploymentStatus.ERROR.value}),
                    "all_versions": True,
                    "page_size": 1000,
                },
            )
            return [deployment async for deployment in resp.items()]
        except Exception:
            logger.warning("Error querying ERROR deployments for GC", exc_info=True)
            return []

    async def async_controller_step(self) -> None:
        """Execute one async iteration of the Models Controller loop.

        Deployment and provider reconciliation both read and write Model Entities,
        and the deployment phase can unlink a provider that the provider phase
        would otherwise re-link. The entity cache is therefore re-read at the
        start of each phase and flushed at the end of it, so each phase sees the
        results of the previous one. Any new phase that writes entities needs the
        same treatment.

        Staged changes are applied even when the step bails out early. Provider
        deletion stages the removal of its entity links, and the provider itself is
        already gone by then, so discarding those would leave the links behind with
        nothing left to trigger another attempt.
        """
        logger.debug("Models controller step starting")

        await self._entity_cache.refresh()
        try:
            deployment_contexts = await self.retrieve_non_terminal_deployments()
            self.emit_heartbeat()

            # Check stop signal before reconciling deployments
            if self._is_stopping():
                logger.debug("Stop signal received, skipping reconciliation")
                return

            if deployment_contexts:
                logger.debug(f"Found {len(deployment_contexts)} total deployment(s) in non-terminal states")
                await self._deployment_reconciler.reconcile_deployments(deployment_contexts)

            known_deployment_ids = {
                f"{ctx.model_deployment.workspace}/{ctx.model_deployment.name}"
                for ctx in deployment_contexts
                if ctx.model_deployment is not None
            }
            await self._deployment_reconciler.reconcile_orphans(known_deployment_ids)
            self.emit_heartbeat()

            # Check stop signal before ERROR GC
            if self._is_stopping():
                logger.debug("Stop signal received, skipping ERROR GC and provider queries")
                return

            error_deployments = await self.retrieve_error_deployments()
            if error_deployments:
                logger.debug(
                    "Found %d ERROR deployment(s) for GC evaluation",
                    len(error_deployments),
                )
                await self._deployment_reconciler.gc_error_deployments(error_deployments)
        finally:
            # Unconditional: a phase that ends early or raises must still apply what
            # it staged, otherwise the changes are stranded and the next refresh
            # refuses to run.
            await self._entity_cache.flush()
        self.emit_heartbeat()

        # Check stop signal before querying providers
        if self._is_stopping():
            logger.debug("Stop signal received, skipping provider queries")
            return

        await self._entity_cache.refresh()
        try:
            provider_contexts = await self.retrieve_model_providers()
            if provider_contexts is None:
                logger.debug("Skipping provider reconciliation because provider listing failed")
                return
            logger.debug("Found %d total model provider(s)", len(provider_contexts))
            self.emit_heartbeat()
            await self._provider_reconciler.reconcile_model_providers(provider_contexts)
        finally:
            await self._entity_cache.flush()

        logger.debug(
            "Models controller step completed: %d deployment(s), %d provider(s)",
            len(deployment_contexts),
            len(provider_contexts),
        )

    def _is_stopping(self) -> bool:
        """Check if the stop signal has been set."""
        return self._stop_signal is not None and self._stop_signal.is_set()

    async def _cancellable_step(self) -> None:
        """Wrapper that captures the current asyncio Task for external cancellation.

        This allows cancel_step() to cancel the running coroutine from another
        thread via call_soon_threadsafe(task.cancel).
        """
        self._current_task = asyncio.current_task()
        try:
            await self.async_controller_step()
        finally:
            self._current_task = None

    def step(self) -> None:
        """Execute one iteration of the Models Controller loop."""
        # Check stop signal before making any API calls
        if self._is_stopping():
            logger.debug("Stop signal received, skipping models controller step")
            return

        logger.debug("Models controller step() called")
        try:
            self._loop.run_until_complete(self._cancellable_step())
            self._is_healthy = True

        except asyncio.CancelledError:
            logger.info("Models controller step cancelled during shutdown")

        except Exception:
            self._is_healthy = False
            logger.exception("Models controller step failed")
            raise
