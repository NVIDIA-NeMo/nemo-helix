# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment state machine reconciliation against DeploymentBackend."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from nemo_deployments_plugin.backends.base import BackendStatusUpdate, DeploymentBackend
from nemo_deployments_plugin.backends.registry import ExecutorNotFoundError, ExecutorRegistry
from nemo_deployments_plugin.config import ControllerConfig
from nemo_deployments_plugin.constants import MANAGED_BY_LABEL
from nemo_deployments_plugin.entities import Deployment, DeploymentConfig, StatusEvent, Volume
from nemo_deployments_plugin.reconciler.drift_recovery import DriftRecoveryCache, DriftRecoveryLimits, RecoveryAction
from nemo_deployments_plugin.reconciler.prerequisite import PrerequisiteResult, prerequisites_met
from nemo_deployments_plugin.reconciler.volume_mounts import VolumeMountResult, volume_mounts_ready
from nemo_helix_plugin.entity_client import NemoEntitiesClient, NemoEntityConflictError, NemoEntityNotFoundError

logger = logging.getLogger(__name__)

_DELETE_COMPLETE_STATUSES = frozenset({"SUCCEEDED", "DELETED"})


def _is_terminal_deleting_timeout_failure(deployment: Deployment) -> bool:
    """Return True when delete reconciliation already recorded a deleting timeout."""
    if deployment.status != "FAILED":
        return False
    details = deployment.error_details
    return isinstance(details, dict) and details.get("reason") == "deleting_timeout"


def deployment_id(deployment: Deployment) -> str:
    return f"{deployment.workspace}/{deployment.name}"


class DeploymentReconciler:
    """Reconciles Deployment entities with backend resources."""

    def __init__(
        self,
        entities: NemoEntitiesClient,
        registry: ExecutorRegistry,
        controller_config: ControllerConfig,
    ) -> None:
        self._entities = entities
        self._registry = registry
        self._controller_config = controller_config
        self._drift_cache = DriftRecoveryCache()
        self._config_cache: dict[tuple[str, str], DeploymentConfig] = {}

    def set_config_cache(self, configs: dict[tuple[str, str], DeploymentConfig]) -> None:
        self._config_cache = configs

    def _drift_limits(self, config: DeploymentConfig) -> DriftRecoveryLimits:
        policy = config.drift_recovery
        ctrl = self._controller_config
        return DriftRecoveryLimits(
            max_attempts=policy.max_attempts if policy.max_attempts is not None else ctrl.drift_recovery_max_attempts,
            initial_delay_seconds=(
                policy.initial_delay_seconds
                if policy.initial_delay_seconds is not None
                else ctrl.drift_recovery_initial_delay_seconds
            ),
            max_delay_seconds=(
                policy.max_delay_seconds
                if policy.max_delay_seconds is not None
                else ctrl.drift_recovery_max_delay_seconds
            ),
        )

    def resolve_backend(self, deployment: Deployment) -> DeploymentBackend:
        return self._registry.resolve(deployment.executor)

    async def _resolve_backend_or_fail(self, deployment: Deployment) -> DeploymentBackend | None:
        """Resolve the deployment executor; mark the deployment FAILED when missing."""
        try:
            return self.resolve_backend(deployment)
        except ExecutorNotFoundError as exc:
            await self._update_deployment_status_failure(deployment, f"No executor available: {exc}")
            return None

    def _try_resolve_backend(self, deployment: Deployment) -> DeploymentBackend | None:
        """Best-effort executor lookup for delete paths; returns None when missing."""
        try:
            return self.resolve_backend(deployment)
        except ExecutorNotFoundError:
            return None

    async def reconcile_one(
        self,
        deployment: Deployment,
        *,
        deployments_by_name: dict[tuple[str, str], Deployment],
        volumes_by_name: dict[tuple[str, str], Volume],
    ) -> None:
        try:
            await self._reconcile_one(
                deployment,
                deployments_by_name=deployments_by_name,
                volumes_by_name=volumes_by_name,
            )
        except NemoEntityConflictError:
            logger.debug("Optimistic lock conflict on deployment %s - retry next cycle.", deployment_id(deployment))

    async def _reconcile_one(
        self,
        deployment: Deployment,
        *,
        deployments_by_name: dict[tuple[str, str], Deployment],
        volumes_by_name: dict[tuple[str, str], Volume],
    ) -> None:
        if deployment.desired_state == "STOPPED" or deployment.status == "DELETING":
            await self._reconcile_delete(deployment)
            return

        config_key = (deployment.workspace, deployment.deployment_config)
        config = self._config_cache.get(config_key)
        if config is None:
            try:
                config = await self._entities.get(
                    DeploymentConfig,
                    name=deployment.deployment_config,
                    workspace=deployment.workspace,
                )
                self._config_cache[config_key] = config
            except NemoEntityNotFoundError:
                await self._update_deployment_status_failure(
                    deployment, f"DeploymentConfig '{deployment.deployment_config}' not found"
                )
                return

        if deployment.status == "PENDING":
            prereq = prerequisites_met(
                deployment,
                deployments_by_name=deployments_by_name,
            )
            if not prereq.met:
                if _prerequisite_failed(prereq, deployments_by_name):
                    await self._update_deployment_status_failure(deployment, prereq.reason)
                    return
                await self._update_deployment_status_pending(deployment, prereq.reason)
                return

            mount_result = volume_mounts_ready(config, deployment.workspace, volumes_by_name)
            if not mount_result.ready:
                if mount_result.blocking_volume and _volume_mount_failed(
                    mount_result, volumes_by_name, deployment.workspace
                ):
                    await self._update_deployment_status_failure(deployment, mount_result.reason)
                    return
                await self._update_deployment_status_pending(deployment, mount_result.reason)
                return

        backend = await self._resolve_backend_or_fail(deployment)
        if backend is None:
            return

        if deployment.status == "PENDING":
            await self._reconcile_create(deployment, config, backend)
            return

        if deployment.status in ("STARTING", "READY", "LOST", "UNKNOWN"):
            status_update = await backend.read_status(workspace=deployment.workspace, name=deployment.name)
            if status_update.status == "LOST":
                await self._handle_drift(deployment, config, backend)
                return
            if status_update.status == "UNKNOWN":
                await self._handle_unknown_status(deployment, status_update)
                return
            if status_update.status == "STARTING":
                timeout_update = self._check_starting_timeout(deployment)
                if timeout_update is not None:
                    await self._update_deployment_status(deployment, timeout_update)
                    return
            if status_update.status in ("READY", "SUCCEEDED"):
                self._drift_cache.remove(deployment_id(deployment))
            await self._update_deployment_status(deployment, status_update)
            return

    async def _reconcile_create(
        self,
        deployment: Deployment,
        config: DeploymentConfig,
        backend: DeploymentBackend,
    ) -> None:
        dep_id = deployment_id(deployment)
        labels = {**config.labels, "managed-by": MANAGED_BY_LABEL}
        try:
            status_update = await backend.create_deployment(
                workspace=deployment.workspace,
                name=deployment.name,
                config_name=config.name,
                labels=labels,
                backend_config=config.backend_config.model_dump(by_alias=True, exclude_none=True),
                auth_context=deployment.auth_context,
            )
        except Exception as exc:
            logger.exception("Failed to create deployment %s", dep_id)
            await self._update_deployment_status_failure(deployment, f"Failed to create deployment: {exc}")
            return
        logger.info("Created deployment %s: %s", dep_id, status_update.status)
        await self._update_deployment_status(deployment, status_update)

    async def _reconcile_delete(self, deployment: Deployment) -> None:
        dep_id = deployment_id(deployment)
        self._drift_cache.remove(dep_id)
        if _is_terminal_deleting_timeout_failure(deployment):
            return
        timeout_update = self._check_deleting_timeout(deployment)
        if timeout_update is not None:
            await self._update_deployment_status(deployment, timeout_update)
            return

        backend = self._try_resolve_backend(deployment)
        if deployment.status != "DELETING":
            await self._update_deployment_status(
                deployment,
                BackendStatusUpdate(status="DELETING", status_message="Stopping deployment"),
            )

        if backend is not None:
            try:
                status_update = await backend.delete_deployment(deployment.workspace, deployment.name)
            except Exception:
                logger.warning("Backend delete failed for %s — will retry", dep_id, exc_info=True)
                return
            if status_update.status not in _DELETE_COMPLETE_STATUSES:
                logger.debug(
                    "Backend delete not complete for %s: %s — %s",
                    dep_id,
                    status_update.status,
                    status_update.status_message,
                )
                return
        else:
            logger.warning("No executor for delete of %s — removing entity only", dep_id)

        try:
            await self._entities.delete(
                Deployment,
                name=deployment.name,
                workspace=deployment.workspace,
                expected_db_version=deployment.db_version,
            )
            logger.info("Deleted deployment entity %s", dep_id)
        except NemoEntityNotFoundError:
            logger.debug("Deployment entity %s already deleted", dep_id)
        except NemoEntityConflictError:
            logger.debug("Optimistic lock conflict deleting deployment %s - retry next cycle.", dep_id)
        except Exception:
            logger.exception("Failed to delete deployment entity %s", dep_id)

    async def _handle_drift(
        self,
        deployment: Deployment,
        config: DeploymentConfig,
        backend: DeploymentBackend,
    ) -> None:
        dep_id = deployment_id(deployment)
        if config.drift_recovery.action == "ignore":
            await self._update_deployment_status(
                deployment,
                BackendStatusUpdate(status="LOST", status_message="Backend resource lost (drift recovery ignored)"),
            )
            return

        if config.restart_policy != "Always":
            await self._update_deployment_status_failure(deployment, "Backend resource lost for non-Always deployment")
            return

        cache = self._drift_cache
        limits = self._drift_limits(config)
        cache.add(dep_id)
        match cache.should_recover(dep_id, limits):
            case RecoveryAction.EXHAUSTED:
                attempts = cache.get_attempts(dep_id)
                await self._update_deployment_status_failure(
                    deployment,
                    f"Drift recovery failed after {attempts} attempts. Manual intervention required.",
                )
                return
            case RecoveryAction.BACKOFF:
                logger.debug("Drift recovery for %s in backoff period", dep_id)
                return
            case RecoveryAction.PROCEED:
                pass

        attempt = cache.add_attempt(dep_id)
        logger.info(
            "Drift recovery for %s (attempt %d/%d)",
            dep_id,
            attempt,
            limits.max_attempts,
        )
        labels = {**config.labels, "managed-by": MANAGED_BY_LABEL}
        try:
            status_update = await backend.create_deployment(
                workspace=deployment.workspace,
                name=deployment.name,
                config_name=config.name,
                labels=labels,
                backend_config=config.backend_config.model_dump(by_alias=True, exclude_none=True),
                auth_context=deployment.auth_context,
            )
        except Exception as exc:
            logger.exception("Drift recovery failed for %s", dep_id)
            await self._update_deployment_status(
                deployment,
                BackendStatusUpdate(
                    status="LOST",
                    status_message=(f"Recovery attempt {attempt}/{limits.max_attempts} failed: {exc}. Will retry."),
                ),
            )
            return
        message = (
            f"Recovering deployment — backend resources recreated "
            f"(attempt {attempt}/{limits.max_attempts}). {status_update.status_message}"
        )
        status_update = status_update.model_copy(update={"status_message": message})
        await self._update_deployment_status(deployment, status_update)

    def _controller_recovery_limits(self) -> DriftRecoveryLimits:
        ctrl = self._controller_config
        return DriftRecoveryLimits(
            max_attempts=ctrl.drift_recovery_max_attempts,
            initial_delay_seconds=ctrl.drift_recovery_initial_delay_seconds,
            max_delay_seconds=ctrl.drift_recovery_max_delay_seconds,
        )

    async def _handle_unknown_status(self, deployment: Deployment, status_update: BackendStatusUpdate) -> None:
        """Handle transient backend communication failures without terminal FAILED."""
        dep_id = deployment_id(deployment)
        cache = self._drift_cache
        limits = self._controller_recovery_limits()
        cache.add(dep_id)
        match cache.should_recover(dep_id, limits):
            case RecoveryAction.EXHAUSTED:
                attempts = cache.get_attempts(dep_id)
                await self._update_deployment_status_failure(
                    deployment,
                    (
                        f"Unable to communicate with backend after {attempts} attempts. "
                        f"Last error: {status_update.status_message}. Manual intervention required."
                    ),
                )
                return
            case RecoveryAction.BACKOFF:
                logger.debug("Backend check for %s in backoff period", dep_id)
                return
            case RecoveryAction.PROCEED:
                pass

        attempt = cache.add_attempt(dep_id)
        logger.warning(
            "Backend returned UNKNOWN for %s (attempt %d/%d): %s",
            dep_id,
            attempt,
            limits.max_attempts,
            status_update.status_message,
        )
        await self._update_deployment_status(
            deployment,
            BackendStatusUpdate(
                status="UNKNOWN",
                status_message=(
                    f"Unable to determine deployment status (attempt {attempt}/{limits.max_attempts}). "
                    f"{status_update.status_message}"
                ),
                error_details=status_update.error_details,
                endpoints=status_update.endpoints,
                exit_code=status_update.exit_code,
            ),
        )

    def _check_starting_timeout(self, deployment: Deployment) -> BackendStatusUpdate | None:
        timeout = self._controller_config.starting_timeout_seconds
        if timeout <= 0:
            return None
        starting_at = _starting_timestamp(deployment)
        if starting_at is None:
            return None
        elapsed = (datetime.now(timezone.utc) - starting_at).total_seconds()
        if elapsed < timeout:
            return None
        elapsed_int = int(elapsed)
        return BackendStatusUpdate(
            status="FAILED",
            status_message=(
                f"Deployment stuck in STARTING for {elapsed_int}s (timeout: {timeout}s). Readiness checks never passed."
            ),
            error_details={
                "reason": "starting_timeout",
                "elapsed_seconds": elapsed_int,
                "timeout_seconds": timeout,
            },
        )

    def _check_deleting_timeout(self, deployment: Deployment) -> BackendStatusUpdate | None:
        timeout = self._controller_config.deleting_timeout_seconds
        if timeout <= 0:
            return None
        deleting_at = _deleting_timestamp(deployment)
        if deleting_at is None:
            return None
        elapsed = (datetime.now(timezone.utc) - deleting_at).total_seconds()
        if elapsed < timeout:
            return None
        elapsed_int = int(elapsed)
        return BackendStatusUpdate(
            status="FAILED",
            status_message=(
                f"Deployment stuck in DELETING for {elapsed_int}s (timeout: {timeout}s). "
                "Backend resources were not removed."
            ),
            error_details={
                "reason": "deleting_timeout",
                "elapsed_seconds": elapsed_int,
                "timeout_seconds": timeout,
            },
        )

    async def _update_deployment_status_pending(self, deployment: Deployment, message: str) -> None:
        if deployment.status == "PENDING" and deployment.status_message == message:
            return
        await self._update_deployment_status(
            deployment,
            BackendStatusUpdate(status="PENDING", status_message=message),
        )

    async def _update_deployment_status_failure(self, deployment: Deployment, message: str) -> None:
        await self._update_deployment_status(
            deployment,
            BackendStatusUpdate(status="FAILED", status_message=message),
        )

    async def _update_deployment_status(self, deployment: Deployment, update: BackendStatusUpdate) -> None:
        if (
            deployment.status == update.status
            and deployment.status_message == update.status_message
            and deployment.endpoints == update.endpoints
            and deployment.exit_code == update.exit_code
            and deployment.error_details == update.error_details
        ):
            return

        deployment.status = update.status
        deployment.status_message = update.status_message
        deployment.endpoints = update.endpoints
        deployment.exit_code = update.exit_code
        deployment.error_details = update.error_details
        deployment.status_history.append(
            StatusEvent(
                status=update.status,
                message=update.status_message,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        )
        await self._save(deployment)

    async def _save(self, deployment: Deployment) -> None:
        await self._entities.update(deployment)


def _starting_timestamp(deployment: Deployment) -> datetime | None:
    for event in reversed(deployment.status_history):
        if event.status == "STARTING" and event.timestamp:
            return datetime.fromisoformat(event.timestamp)
    if deployment.status_history and deployment.status_history[0].timestamp:
        return datetime.fromisoformat(deployment.status_history[0].timestamp)
    return None


def _deleting_timestamp(deployment: Deployment) -> datetime | None:
    for event in reversed(deployment.status_history):
        if event.status == "DELETING" and event.timestamp:
            return datetime.fromisoformat(event.timestamp)
    return None


def _prerequisite_failed(
    result: PrerequisiteResult,
    deployments_by_name: dict[tuple[str, str], Deployment],
) -> bool:
    if result.blocking_prerequisite is None:
        return "failed" in result.reason.lower()
    if result.blocking_workspace is None or result.blocking_name is None:
        return False
    target = deployments_by_name.get((result.blocking_workspace, result.blocking_name))
    return target is not None and target.status == "FAILED"


def _volume_mount_failed(
    result: VolumeMountResult,
    volumes_by_name: dict[tuple[str, str], Volume],
    workspace: str,
) -> bool:
    if result.blocking_volume is None:
        return "failed" in result.reason.lower()
    volume = volumes_by_name.get((workspace, result.blocking_volume))
    return volume is not None and volume.status == "FAILED"
