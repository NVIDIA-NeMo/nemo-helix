# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model deployment reconciliation logic for Models Controller."""

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from logging import getLogger
from typing import Awaitable, Callable, Optional

from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import ConflictError, NotFoundError
from nemo_platform_plugin.models.client import AsyncModelsClient
from nemo_platform_plugin.models.types import (
    CreateModelProviderRequest,
    ModelDeployment,
    ModelDeploymentStatus,
    ModelProvider,
    ModelProviderStatus,
    UpdateModelDeploymentStatusRequest,
    UpsertModelProviderRequest,
)
from nmp.common.entities.utils import parse_entity_ref
from nmp.core.models.config import ControllerConfig
from nmp.core.models.controllers.backends.backends import DeploymentStatusUpdate, ServiceBackend
from nmp.core.models.controllers.backends.common import deleting_elapsed_seconds
from nmp.core.models.controllers.backends.registry import BackendRegistry
from nmp.core.models.controllers.context import ModelContext
from nmp.core.models.controllers.entity_cache import ModelEntityCache

logger = getLogger(__name__)


class RecoveryAction(Enum):
    """Result of checking whether drift recovery should proceed."""

    PROCEED = auto()  # Attempt recovery now
    BACKOFF = auto()  # In backoff period, skip this cycle
    EXHAUSTED = auto()  # Max attempts reached, mark as failed


@dataclass
class DriftRecoveryState:
    """Tracks drift recovery attempts for a single deployment."""

    attempts: int = 0
    last_attempt_at: datetime | None = None


class DriftRecoveryCache:
    """Manages drift recovery state for deployments.

    Handles tracking of recovery attempts, exponential backoff timing,
    and max retry limits for deployments whose backend resources were lost.
    """

    def __init__(self, max_attempts: int, base_delay_seconds: int, max_delay_seconds: int) -> None:
        """Initialize the drift recovery cache.

        Args:
            max_attempts: Maximum recovery attempts before giving up
            base_delay_seconds: Base delay for exponential backoff
            max_delay_seconds: Maximum delay cap for exponential backoff
        """
        self._states: dict[str, DriftRecoveryState] = {}
        self._max_attempts = max_attempts
        self._base_delay_seconds = base_delay_seconds
        self._max_delay_seconds = max_delay_seconds

    @property
    def max_attempts(self) -> int:
        """Maximum recovery attempts before giving up."""
        return self._max_attempts

    def add(self, deployment_id: str) -> None:
        """Add deployment to recovery tracking (idempotent).

        Creates a new DriftRecoveryState if not already present.
        Safe to call multiple times for the same deployment.
        """
        if deployment_id not in self._states:
            self._states[deployment_id] = DriftRecoveryState()

    def remove(self, deployment_id: str) -> None:
        """Remove deployment from recovery tracking.

        Called when deployment recovers successfully (reaches READY).
        Safe to call even if deployment is not being tracked.
        """
        self._states.pop(deployment_id, None)

    def should_recover(self, deployment_id: str) -> RecoveryAction:
        """Check if recovery should proceed for a deployment.

        Returns:
            RecoveryAction.PROCEED: Attempt recovery now
            RecoveryAction.BACKOFF: In backoff period, skip this cycle
            RecoveryAction.EXHAUSTED: Max attempts reached, should mark as failed
        """
        state = self._states.get(deployment_id)
        if state is None:
            return RecoveryAction.PROCEED

        # Check max attempts
        if state.attempts >= self._max_attempts:
            return RecoveryAction.EXHAUSTED

        # Check exponential backoff
        if state.last_attempt_at:
            backoff_seconds = min(
                self._base_delay_seconds * (2**state.attempts),
                self._max_delay_seconds,
            )
            elapsed = (datetime.now(timezone.utc) - state.last_attempt_at).total_seconds()
            if elapsed < backoff_seconds:
                return RecoveryAction.BACKOFF

        return RecoveryAction.PROCEED

    def add_attempt(self, deployment_id: str) -> int:
        """Record a recovery attempt.

        Increments the attempt counter and records the timestamp.
        Automatically adds state if not present.

        Returns:
            The new attempt count after incrementing.
        """
        self.add(deployment_id)
        state = self._states[deployment_id]
        state.attempts += 1
        state.last_attempt_at = datetime.now(timezone.utc)
        return state.attempts

    def get_attempts(self, deployment_id: str) -> int:
        """Get current attempt count for a deployment."""
        state = self._states.get(deployment_id)
        return state.attempts if state else 0


class ModelDeploymentReconciler:
    """
    Handles reconciliation of model deployments with their backing service infrastructure.

    This class manages the lifecycle of ModelDeployment objects by coordinating with
    service backends and managing associated ModelProvider resources.
    """

    def __init__(
        self,
        models_sdk: AsyncNeMoPlatform,
        backend_registry: BackendRegistry,
        controller_config: ControllerConfig,
        entity_cache: ModelEntityCache,
        emit_heartbeat: Callable[[], None],
    ) -> None:
        """Initialize the deployment reconciler.

        Args:
            models_sdk: SDK client for Models API interactions
            backend_registry: Registry of available service backends
            controller_config: Controller configuration containing deployment settings
            entity_cache: Model Entity reads and staged writes for the current phase
            emit_heartbeat: Called as each unit of work finishes so a long pass is
                distinguishable from a stalled one
        """
        self._models_sdk = models_sdk
        self._models_client = client_from_platform(models_sdk, AsyncModelsClient)
        self._backend_registry = backend_registry
        self._controller_config = controller_config
        self._entity_cache = entity_cache
        self._emit_heartbeat = emit_heartbeat
        self._drift_recovery_cache = DriftRecoveryCache(
            max_attempts=controller_config.drift_recovery_max_attempts,
            base_delay_seconds=controller_config.drift_recovery_base_delay_seconds,
            max_delay_seconds=controller_config.drift_recovery_max_delay_seconds,
        )

    def get_service_backend(self) -> ServiceBackend:
        """Get the service backend.

        Currently only one backend (docker or k8s) is enabled at a time. In the future,
        when multiple backends can run in parallel, this may take a deployment id and
        return the backend responsible for that deployment.
        """
        return self._backend_registry.get_backend()

    async def _update_deployment_status(
        self,
        deployment: ModelDeployment,
        *,
        status: ModelDeploymentStatus,
        status_message: str = "",
        model_provider_id: str | None = None,
    ) -> ModelDeployment:
        """Write a controller-owned deployment status update via the typed Models client."""
        return (
            await self._models_client.update_deployment_status(
                name=deployment.name,
                workspace=deployment.workspace,
                body=UpdateModelDeploymentStatusRequest(
                    status=status,
                    status_message=status_message,
                    model_provider_id=model_provider_id,
                ),
                query_params={"version": str(deployment.entity_version)},
            )
        ).data()

    async def reconcile_deployments(self, deployment_contexts: list[ModelContext]) -> None:
        """Process deployments and reconcile their state with backends.

        Maps deployment status to the appropriate backend function and reconciles each deployment.
        Uses pre-fetched data from ModelContext to avoid repeated retrieval calls.

        Args:
            deployment_contexts: List of deployment contexts with pre-fetched related data
        """
        for ctx in deployment_contexts:
            deployment = ctx.model_deployment
            if deployment is None:
                logger.warning("Skipping deployment reconciliation for context with no model_deployment")
                self._emit_heartbeat()
                continue
            model_deployment_id = f"{deployment.workspace}/{deployment.name}"
            try:
                backend = self.get_service_backend()

                match deployment.status:
                    case ModelDeploymentStatus.CREATED:
                        # Lambda needed to bind ctx (the reconcile context bundles
                        # the deployment, config, and model entity).
                        await self._reconcile_individual_deployment(
                            deployment,
                            lambda _dep, _ctx=ctx: backend.create_model_deployment(_ctx),
                            "create",
                            existing_provider=ctx.model_provider,
                        )
                    case ModelDeploymentStatus.PENDING | ModelDeploymentStatus.READY | ModelDeploymentStatus.UNKNOWN:
                        # Check status and handle drift/backend issues. The ctx
                        # carries the config + entity so backends that advance
                        # creation in the status path (k8s vLLM) can compile the
                        # serving objects.
                        status_update = await backend.get_model_deployment_status(ctx)

                        if status_update.status == ModelDeploymentStatus.LOST:
                            # Drift detected - attempt recovery
                            await self._handle_drift_recovery(deployment, ctx, backend)
                            continue
                        elif status_update.status == ModelDeploymentStatus.UNKNOWN:
                            # Backend communication failure - track attempts, eventually error out
                            await self._handle_unknown_status(deployment, status_update)
                            continue
                        elif status_update.status in (ModelDeploymentStatus.READY, ModelDeploymentStatus.ERROR):
                            # Clear recovery state - deployment is healthy or in terminal state
                            self._drift_recovery_cache.remove(model_deployment_id)

                        # Process the status update. ``status_update`` is already
                        # fetched above, so ``_reconcile_individual_deployment``
                        # won't invoke this callable; it's passed only for the
                        # generic signature (bind ctx for type consistency).
                        action = "check status of" if deployment.status == ModelDeploymentStatus.PENDING else "monitor"
                        await self._reconcile_individual_deployment(
                            deployment,
                            lambda _dep, _ctx=ctx: backend.get_model_deployment_status(_ctx),
                            action,
                            existing_provider=ctx.model_provider,
                            status_update=status_update,
                        )
                    case ModelDeploymentStatus.DELETING:
                        await self._reconcile_individual_deployment(
                            deployment,
                            lambda d: backend.delete_model_deployment(
                                d.workspace,
                                d.name,
                                deleting_elapsed_seconds=deleting_elapsed_seconds(d),
                            ),
                            "delete",
                            existing_provider=ctx.model_provider,
                        )
                    case ModelDeploymentStatus.DELETED:
                        # Check if deployment has been in DELETED state long enough to hard-delete
                        await self._handle_deleted_deployment(deployment)
            except Exception as e:
                logger.exception(f"Error processing deployment {model_deployment_id}: {e}")
            finally:
                self._emit_heartbeat()

    async def reconcile_orphans(self, known_deployment_ids: set[str]) -> None:
        """Delete backend deployments that are not in the known set (orphans).

        Uses the same backend as reconcile_deployments (via get_service_backend).
        Lists managed deployment names, diffs against known_deployment_ids, and deletes any
        that exist on the backend but not in the API-backed set.

        Args:
            known_deployment_ids: Set of "workspace/name" from retrieve_non_terminal_deployments().
        """
        backend = self.get_service_backend()
        try:
            backend_names = await backend.list_managed_deployment_names()
        except Exception as e:
            logger.warning("Error listing managed deployments for orphan reconciliation: %s", e, exc_info=True)
            return
        orphans = set(backend_names) - known_deployment_ids
        for deployment_id in orphans:
            try:
                ref = parse_entity_ref(deployment_id)
            except ValueError:
                logger.warning("Invalid deployment id from backend: %r, skipping", deployment_id)
                continue
            workspace, name = ref.workspace, ref.name
            try:
                logger.info("Deleting orphan deployment %s", deployment_id)
                await backend.delete_model_deployment(workspace, name)
            except Exception as e:
                logger.warning(
                    "Failed to delete orphan %s: %s",
                    deployment_id,
                    e,
                    exc_info=True,
                )
            finally:
                self._emit_heartbeat()

    async def gc_error_deployments(self, error_deployments: list[ModelDeployment]) -> None:
        """Garbage-collect backend resources for ERROR deployments past their TTL.

        Queries each deployment's updated_at timestamp and compares against
        the configured error_deployment_ttl_seconds. Deployments past the TTL
        have their backend resources deleted and are transitioned to DELETING,
        which feeds into the normal DELETING -> DELETED -> hard-delete flow.

        Args:
            error_deployments: List of deployments currently in ERROR state
        """
        ttl_seconds = self._controller_config.error_deployment_ttl_seconds
        grace_period = timedelta(seconds=ttl_seconds)
        current_time = datetime.now(timezone.utc)
        backend = self.get_service_backend()

        for deployment in error_deployments:
            deployment_id = f"{deployment.workspace}/{deployment.name}"
            try:
                updated_at = deployment.updated_at
                if updated_at is None:
                    logger.warning(
                        "ERROR deployment has no updated_at, skipping GC",
                        extra={"deployment": deployment_id},
                    )
                    continue

                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)

                time_in_error = current_time - updated_at

                if time_in_error < grace_period:
                    remaining = (grace_period - time_in_error).total_seconds()
                    logger.debug(
                        "ERROR deployment not yet past GC TTL",
                        extra={
                            "deployment": deployment_id,
                            "time_in_error_s": time_in_error.total_seconds(),
                            "remaining_s": remaining,
                        },
                    )
                    continue

                logger.info(
                    "Garbage collecting ERROR deployment past TTL",
                    extra={
                        "deployment": deployment_id,
                        "time_in_error_s": time_in_error.total_seconds(),
                        "ttl_s": ttl_seconds,
                    },
                )

                try:
                    await backend.delete_model_deployment(deployment.workspace, deployment.name)
                except Exception:
                    logger.warning(
                        "Backend delete failed during ERROR GC, proceeding with status transition",
                        extra={"deployment": deployment_id},
                        exc_info=True,
                    )

                provider_cleanup_failed = False
                try:
                    await self._delete_model_provider(deployment)
                except Exception:
                    provider_cleanup_failed = True
                    logger.warning(
                        "Provider cleanup failed during ERROR GC, proceeding with status transition",
                        extra={"deployment": deployment_id},
                        exc_info=True,
                    )

                original_message = deployment.status_message or ""
                gc_message = (
                    f"Backend resources garbage collected after "
                    f"{time_in_error.total_seconds():.0f}s in ERROR state "
                    f"(TTL: {ttl_seconds}s)."
                )
                if provider_cleanup_failed:
                    gc_message = f"{gc_message} Provider cleanup failed."
                if original_message:
                    gc_message = f"{gc_message} Original error: {original_message}"

                await self._update_deployment_status(
                    deployment,
                    status=ModelDeploymentStatus.DELETING,
                    status_message=gc_message,
                )

                logger.info(
                    "ERROR deployment transitioned to DELETING by GC",
                    extra={"deployment": deployment_id},
                )

            except Exception:
                logger.warning(
                    "Failed to garbage collect ERROR deployment",
                    extra={"deployment": deployment_id},
                    exc_info=True,
                )
            finally:
                self._emit_heartbeat()

    async def _reconcile_individual_deployment(
        self,
        deployment: ModelDeployment,
        backend_func: Callable[[ModelDeployment], Awaitable[DeploymentStatusUpdate]],
        action_description: str,
        existing_provider: Optional[ModelProvider] = None,
        status_update: Optional[DeploymentStatusUpdate] = None,
    ) -> None:
        """Generic reconciliation pattern for any deployment state.

        Args:
            deployment: The ModelDeployment to reconcile
            backend_func: A callable that takes deployment and returns DeploymentStatusUpdate
            action_description: Human-readable description of the action (e.g., "create", "delete")
            existing_provider: Optional pre-fetched ModelProvider to avoid retrieval
            status_update: Optional pre-fetched status update. If provided, skips calling backend_func.
        """
        model_deployment_id = f"{deployment.workspace}/{deployment.name}"

        logger.debug(
            f"Reconciling deployment {model_deployment_id} "
            f"(version: {deployment.entity_version}, status: {deployment.status}) - action: {action_description}"
        )

        try:
            # Use pre-fetched status if provided, otherwise call the backend
            if status_update is None:
                status_update = await backend_func(deployment)

            status_msg = (
                f"Backend {action_description} status update for {model_deployment_id}: "
                f"{status_update.status} - {status_update.status_message}"
            )
            if (
                action_description == "monitor"
                and status_update.status == ModelDeploymentStatus.READY
                and not (status_update.status_message or "").strip()
            ):
                # DEBUG level for routine READY monitoring with nothing to report.
                logger.debug(status_msg)
            else:
                # INFO level in all other cases (create/delete/check status, non-READY, READY with a real message, etc.)
                logger.info(status_msg)

            model_provider_id = await self._reconcile_model_provider(deployment, status_update, existing_provider)

            await self._update_deployment_status(
                deployment,
                status=status_update.status,
                status_message=status_update.status_message,
                model_provider_id=model_provider_id,
            )

        except ConflictError as e:
            logger.debug(f"Deployment status conflict: {e}")
            return
        except Exception as e:
            logger.exception(f"Failed to {action_description} deployment {model_deployment_id}: {e}")
            try:
                await self._update_deployment_status(
                    deployment,
                    status=ModelDeploymentStatus.ERROR,
                    status_message=f"Failed to {action_description} deployment {model_deployment_id}",
                )
            except ConflictError as e:
                logger.debug(f"Deployment status conflict: {e}")
                return
            except Exception as update_error:
                logger.exception(f"Failed to update deployment status to ERROR: {update_error}")

    async def _handle_drift_recovery(
        self,
        deployment: ModelDeployment,
        ctx: ModelContext,
        backend: ServiceBackend,
    ) -> None:
        """Handle drift recovery when backend returns LOST status.

        Attempts to recreate backend resources with exponential backoff.
        Sets deployment to ERROR after max attempts exceeded.

        Args:
            deployment: The ModelDeployment that needs recovery
            ctx: ModelContext containing pre-fetched config and entity
            backend: The ServiceBackend to use for recreation
        """
        model_deployment_id = f"{deployment.workspace}/{deployment.name}"
        cache = self._drift_recovery_cache
        max_attempts = cache.max_attempts

        # Ensure state is tracked
        cache.add(model_deployment_id)

        # Check if we should proceed with recovery
        match cache.should_recover(model_deployment_id):
            case RecoveryAction.EXHAUSTED:
                attempts = cache.get_attempts(model_deployment_id)
                logger.error(f"Drift recovery failed for {model_deployment_id} after {attempts} attempts")
                try:
                    await self._update_deployment_status(
                        deployment,
                        status=ModelDeploymentStatus.ERROR,
                        status_message=(
                            f"Drift recovery failed after {attempts} attempts. "
                            f"Backend resources could not be recreated. Manual intervention required."
                        ),
                    )
                except ConflictError as e:
                    logger.debug(f"Deployment status conflict: {e}")
                    return
                except Exception as e:
                    logger.exception(f"Failed to update deployment status to ERROR: {e}")
                return

            case RecoveryAction.BACKOFF:
                logger.debug(f"Drift recovery for {model_deployment_id} in backoff period, skipping this cycle")
                return

            case RecoveryAction.PROCEED:
                pass  # Continue to recovery attempt below

        # Record attempt and get the new count
        attempt_count = cache.add_attempt(model_deployment_id)
        logger.info(f"Attempting drift recovery for {model_deployment_id} (attempt {attempt_count}/{max_attempts})")

        try:
            # Call create_model_deployment to recreate resources
            status_update = await backend.create_model_deployment(ctx)

            # Build recovery message
            recovery_message = (
                f"Recovering deployment - backend resources were recreated "
                f"(attempt {attempt_count}/{max_attempts}). "
                f"{status_update.status_message}"
            )

            await self._update_deployment_status(
                deployment,
                status=status_update.status,
                status_message=recovery_message,
                model_provider_id=None,  # Provider will be recreated when READY
            )

            logger.debug(f"Drift recovery initiated for {model_deployment_id}, status: {status_update.status}")

        except ConflictError as e:
            logger.debug(f"Deployment status conflict: {e}")
            return

        except Exception as e:
            logger.exception(f"Error during drift recovery for {model_deployment_id}: {e}")
            # Update status to PENDING with error info for visibility, but don't set ERROR
            # The next cycle will retry (respecting backoff) and can detect if recovery succeeded
            try:
                await self._update_deployment_status(
                    deployment,
                    status=ModelDeploymentStatus.PENDING,
                    status_message=f"Recovery attempt {attempt_count}/{max_attempts} failed: {e}. Will retry.",
                )
            except Exception:
                pass  # Best effort - don't fail if status update fails

    async def _handle_unknown_status(
        self,
        deployment: ModelDeployment,
        status_update: DeploymentStatusUpdate,
    ) -> None:
        """Handle UNKNOWN status when backend cannot determine deployment state.

        This typically occurs when the backend (e.g., Docker, K8s) is temporarily
        unavailable. Uses the same retry tracking as drift recovery to eventually
        transition to ERROR if the issue persists.

        Args:
            deployment: The ModelDeployment with unknown status
            status_update: The status update from the backend containing error details
        """
        model_deployment_id = f"{deployment.workspace}/{deployment.name}"
        cache = self._drift_recovery_cache
        max_attempts = cache.max_attempts

        # Ensure state is tracked
        cache.add(model_deployment_id)

        # Check if we should proceed or if we've exhausted retries
        match cache.should_recover(model_deployment_id):
            case RecoveryAction.EXHAUSTED:
                attempts = cache.get_attempts(model_deployment_id)
                logger.error(f"Backend communication failed for {model_deployment_id} after {attempts} attempts")
                try:
                    await self._update_deployment_status(
                        deployment,
                        status=ModelDeploymentStatus.ERROR,
                        status_message=(
                            f"Unable to communicate with backend after {attempts} attempts. "
                            f"Last error: {status_update.status_message}. Manual intervention required."
                        ),
                    )
                except ConflictError as e:
                    logger.debug(f"Deployment status conflict: {e}")
                    return
                except Exception as e:
                    logger.exception(f"Failed to update deployment status to ERROR: {e}")
                return

            case RecoveryAction.BACKOFF:
                logger.debug(f"Backend check for {model_deployment_id} in backoff period, skipping this cycle")
                return

            case RecoveryAction.PROCEED:
                pass  # Continue to update status below

        # Record attempt and update status with visibility
        attempt_count = cache.add_attempt(model_deployment_id)
        logger.warning(
            f"Backend returned UNKNOWN for {model_deployment_id} "
            f"(attempt {attempt_count}/{max_attempts}): {status_update.status_message}"
        )

        try:
            await self._update_deployment_status(
                deployment,
                status=ModelDeploymentStatus.UNKNOWN,
                status_message=(
                    f"Unable to determine deployment status (attempt {attempt_count}/{max_attempts}). "
                    f"{status_update.status_message}"
                ),
            )
        except ConflictError as e:
            logger.debug(f"Deployment status conflict: {e}")
            return
        except Exception as e:
            logger.exception(f"Failed to update deployment status: {e}")

    async def _reconcile_model_provider(
        self,
        deployment: ModelDeployment,
        status_update: DeploymentStatusUpdate,
        existing_provider: Optional[ModelProvider] = None,
    ) -> Optional[str]:
        """Manage ModelProvider lifecycle based on ModelDeployment status changes.

        Args:
            deployment: The ModelDeployment object
            status_update: The status update from the backend, containing status and host_url
            existing_provider: Optional pre-fetched ModelProvider to avoid retrieval

        Returns:
            The model_provider_id (workspace/name) if a provider was created, None otherwise
        """
        model_deployment_id = f"{deployment.workspace}/{deployment.name}"

        try:
            if status_update.status == ModelDeploymentStatus.READY:
                return await self._ensure_model_provider(deployment, status_update.host_url, existing_provider)
            elif status_update.status in (ModelDeploymentStatus.DELETING, ModelDeploymentStatus.DELETED):
                await self._delete_model_provider(deployment)
            return None
        except Exception as e:
            logger.warning(f"Failed to manage ModelProvider lifecycle for {model_deployment_id}: {e}")
            return None

    async def _ensure_model_provider(
        self,
        deployment: ModelDeployment,
        host_url: Optional[str],
        existing_provider: Optional[ModelProvider] = None,
    ) -> Optional[str]:
        """Ensure a ModelProvider exists for the deployment.

        Args:
            deployment: The ModelDeployment object
            host_url: The host URL provided by the backend for reaching the deployment
            existing_provider: Optional pre-fetched ModelProvider to avoid retrieval

        Returns:
            The model_provider_id (workspace/name) of the created or existing provider, or None if skipped
        """
        model_deployment_id = f"{deployment.workspace}/{deployment.name}"

        if not host_url:
            logger.warning(
                f"No host_url provided for deployment {model_deployment_id}, skipping ModelProvider creation"
            )
            return None

        if deployment.model_provider_id:
            # model_provider_id has been set, so the model provider should already exist
            # Use pre-fetched provider if available, otherwise fetch it
            try:
                _provider_ref = parse_entity_ref(deployment.model_provider_id)
                provider_workspace, provider_name = _provider_ref.workspace, _provider_ref.name

                if not existing_provider:
                    existing_provider = (
                        await self._models_client.get_provider(
                            name=provider_name,
                            workspace=provider_workspace,
                        )
                    ).data()

                if existing_provider.host_url != host_url:
                    logger.info(
                        f"ModelProvider {deployment.model_provider_id} host_url changed from "
                        f"{existing_provider.host_url} to {host_url}, updating provider"
                    )
                    await self._models_client.upsert_provider(
                        name=provider_name,
                        workspace=provider_workspace,
                        body=UpsertModelProviderRequest(
                            host_url=host_url,
                            description=existing_provider.description,
                            enabled_models=existing_provider.enabled_models,
                            status=ModelProviderStatus.READY,
                        ),
                    )
                else:
                    logger.debug(
                        f"Deployment {model_deployment_id} already has ModelProvider {deployment.model_provider_id}, reusing it"
                    )

                return deployment.model_provider_id
            except NotFoundError:
                logger.warning(
                    f"ModelProvider {deployment.model_provider_id} for deployment {model_deployment_id} not found, will create a new one"
                )
            except ValueError:
                logger.warning(
                    f"Invalid model_provider_id format '{deployment.model_provider_id}' for deployment {model_deployment_id}, will create a new one"
                )

        # This is the first time we're seeing the deployment, so we need to create a new model provider
        # Determine the provider name (handle collisions if necessary)
        provider_name = deployment.name
        provider_workspace = deployment.workspace

        try:
            await self._models_client.get_provider(name=provider_name, workspace=provider_workspace)
            unique_suffix = uuid.uuid4().hex[:8]
            provider_name = f"{deployment.name}_{unique_suffix}"
            logger.info(
                f"ModelProvider {provider_workspace}/{deployment.name} already exists, "
                f"using unique name: {provider_name}"
            )
        except NotFoundError:
            logger.debug(f"Creating ModelProvider {provider_workspace}/{provider_name} for deployment")

        await self._models_client.create_provider(
            workspace=provider_workspace,
            body=CreateModelProviderRequest(
                name=provider_name,
                host_url=host_url,
                description=f"Auto-created provider for deployment {deployment.name}",
                project=deployment.project,
                model_deployment_id=model_deployment_id,
                status=ModelProviderStatus.READY,
            ),
        )

        model_provider_id = f"{provider_workspace}/{provider_name}"
        logger.info(f"Successfully created ModelProvider {model_provider_id}")

        return model_provider_id

    async def _cleanup_model_entities_for_provider(
        self, provider_workspace: str, provider_name: str, provider_id: str
    ) -> None:
        """Remove provider references from all Model Entities that were served by this provider.

        Args:
            provider_workspace: The workspace of the provider being deleted
            provider_name: The name of the provider being deleted
            provider_id: The full provider ID in format "workspace/name"
        """
        try:
            # Get the provider to see what models it was serving
            provider = (await self._models_client.get_provider(name=provider_name, workspace=provider_workspace)).data()

            if not provider.served_models:
                logger.debug(f"Provider {provider_id} has no served_models, no cleanup needed")
                return

            # For each model entity, remove this provider from its model_providers list
            for served_model in provider.served_models:
                try:
                    # Parse model entity ID (format: "workspace/name")
                    _model_ref = parse_entity_ref(served_model.model_entity_id)
                    model_workspace, model_name = _model_ref.workspace, _model_ref.name

                    if not self._entity_cache.loaded:
                        # Without a snapshot, a lookup miss is indistinguishable from
                        # "does not exist", and dropping the unlink here would leave a
                        # dangling provider reference once the provider is deleted.
                        logger.warning(
                            "Model Entity cache holds no snapshot; skipping unlink of %s from %s",
                            provider_id,
                            served_model.model_entity_id,
                        )
                        continue

                    model_entity = self._entity_cache.get(model_workspace, model_name)
                    if model_entity is None:
                        logger.debug(f"Model Entity {served_model.model_entity_id} not found, nothing to unlink")
                        continue

                    if provider_id in (model_entity.model_providers or []):
                        self._entity_cache.stage_provider_unlink(model_workspace, model_name, provider_id)
                        logger.info(f"Removed provider {provider_id} from Model Entity {served_model.model_entity_id}")
                    else:
                        logger.debug(
                            f"Provider {provider_id} not found in Model Entity {served_model.model_entity_id} model_providers"
                        )

                except Exception as e:
                    logger.warning(
                        f"Failed to cleanup Model Entity {served_model.model_entity_id} for provider {provider_id}: {e}"
                    )
                finally:
                    self._emit_heartbeat()

        except NotFoundError:
            logger.debug(f"Provider {provider_id} not found during cleanup, skipping Model Entity cleanup")
        except Exception as e:
            logger.warning(f"Failed to cleanup Model Entities for provider {provider_id}: {e}")

    async def _delete_model_provider(self, deployment: ModelDeployment) -> None:
        """Delete the ModelProvider associated with a deployment.

        Also cleans up references to this provider from all linked Model Entities.

        Args:
            deployment: The ModelDeployment object
        """
        model_deployment_id = f"{deployment.workspace}/{deployment.name}"

        if not deployment.model_provider_id:
            logger.warning(
                f"No model_provider_id set for deployment {model_deployment_id}, skipping ModelProvider deletion"
            )
            return

        try:
            _provider_ref = parse_entity_ref(deployment.model_provider_id)
        except ValueError:
            logger.warning(
                f"Invalid model_provider_id format '{deployment.model_provider_id}' "
                f"for deployment {model_deployment_id}, skipping deletion"
            )
            return
        provider_workspace, provider_name = _provider_ref.workspace, _provider_ref.name

        model_provider_id = f"{provider_workspace}/{provider_name}"

        # Clean up Model Entity references before deleting the provider
        await self._cleanup_model_entities_for_provider(provider_workspace, provider_name, model_provider_id)

        try:
            logger.info(f"Deleting ModelProvider {model_provider_id} for deployment {model_deployment_id}")
            await self._models_client.delete_provider(
                name=provider_name,
                workspace=provider_workspace,
            )
            logger.info(f"Successfully deleted ModelProvider {model_provider_id}")
        except NotFoundError:
            logger.warning(f"ModelProvider {model_provider_id} not found, may have been already deleted")

    async def _handle_deleted_deployment(self, deployment: ModelDeployment) -> None:
        """Handle DELETED deployments by hard-deleting them after the grace period.

        Args:
            deployment: The ModelDeployment in DELETED state
        """
        model_deployment_id = f"{deployment.workspace}/{deployment.name}"

        # Check if deployment has been DELETED long enough to remove from database
        current_time = datetime.now(timezone.utc)

        # Ensure updated_at is timezone-aware (if naive, assume UTC)
        updated_at = deployment.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)

        time_since_deletion = current_time - updated_at
        grace_period = timedelta(seconds=self._controller_config.model_deployment_garbage_collection_ttl_seconds)

        if time_since_deletion >= grace_period:
            logger.info(
                f"Deployment {model_deployment_id} (version {deployment.entity_version}) has been DELETED "
                f"for {time_since_deletion.total_seconds():.0f}s (grace period: {self._controller_config.model_deployment_garbage_collection_ttl_seconds}s), "
                "hard-deleting from database"
            )
            try:
                # Hard-delete this specific version by calling the delete API again on a DELETED deployment
                await self._models_client.delete_deployment_version(
                    name=str(deployment.entity_version),  # version number
                    workspace=deployment.workspace,  # workspace
                    deployment=deployment.name,  # deployment name
                )
                logger.info(
                    f"Successfully hard-deleted deployment {model_deployment_id} version {deployment.entity_version}"
                )
            except NotFoundError:
                logger.debug(f"Deployment {model_deployment_id} already removed from database")
            except Exception as e:
                logger.exception(f"Failed to hard-delete deployment {model_deployment_id}: {e}")
        else:
            remaining_time = (grace_period - time_since_deletion).total_seconds()
            logger.debug(
                f"Deployment {model_deployment_id} (version {deployment.entity_version}) has been DELETED "
                f"for {time_since_deletion.total_seconds():.0f}s, will hard-delete in {remaining_time:.0f}s"
            )
