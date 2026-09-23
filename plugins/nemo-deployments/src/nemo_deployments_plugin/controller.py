# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployments reconcile controller — drives Deployment and Volume state machines."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import ClassVar

from nemo_deployments_plugin.backends.registry import ExecutorRegistry, ExecutorSpec
from nemo_deployments_plugin.config import ControllerConfig, DeploymentsConfig
from nemo_deployments_plugin.entities import Deployment, DeploymentConfig, Volume
from nemo_deployments_plugin.reconciler.deployment_reconciler import DeploymentReconciler
from nemo_deployments_plugin.reconciler.entity_client import list_all_pages
from nemo_deployments_plugin.reconciler.orphan_cleanup import reconcile_orphans
from nemo_deployments_plugin.reconciler.prerequisite import parse_deployment_ref
from nemo_deployments_plugin.reconciler.volume_mounts import collect_volume_mount_names
from nemo_deployments_plugin.reconciler.volume_reconciler import VolumeReconciler
from nemo_deployments_plugin.types import NON_TERMINAL_DEPLOYMENT_STATUSES, NON_TERMINAL_VOLUME_STATUSES
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.controller import NemoController
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entity_client import NemoEntitiesClient, NemoEntityConflictError, NemoEntityNotFoundError
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperator
from nemo_helix_plugin.sdk_provider import get_async_platform_sdk

logger = logging.getLogger(__name__)

_TERMINAL_ORPHAN_STATUSES = ("SUCCEEDED", "FAILED")


class DeploymentsController(NemoController):
    """Reconciles deployments and volumes against registered executor backends."""

    name = "deployments"
    dependencies: ClassVar[list[str]] = ["entities"]

    def __init__(self) -> None:
        self._entities: NemoEntitiesClient | None = None
        self._registry: ExecutorRegistry | None = None
        self._controller_config: ControllerConfig | None = None
        self._deployment_reconciler: DeploymentReconciler | None = None
        self._volume_reconciler: VolumeReconciler | None = None
        self._interval_seconds: float = 5.0
        self._orphan_cleanup_elapsed_seconds: float = 0.0
        self._deployments_list_ok: bool = True
        self._volumes_list_ok: bool = True
        self._terminal_deployments_list_ok: bool = True

    @property
    def is_healthy(self) -> bool:
        return self._deployments_list_ok and self._volumes_list_ok

    @property
    def interval_seconds(self) -> float:
        return self._interval_seconds

    @property
    def entities(self) -> NemoEntitiesClient:
        if self._entities is None:
            raise RuntimeError("DeploymentsController.entities accessed before on_startup()")
        return self._entities

    @property
    def controller_config(self) -> ControllerConfig:
        if self._controller_config is None:
            raise RuntimeError("DeploymentsController.controller_config accessed before on_startup()")
        return self._controller_config

    async def on_startup(self) -> None:
        config = DeploymentsConfig.get()
        self._controller_config = config.controller
        self._interval_seconds = float(config.controller.interval_seconds)

        sdk = get_async_platform_sdk(as_service="deployments", internal=True)
        entities_api = client_from_platform(sdk, AsyncEntitiesClient)
        self._entities = NemoEntitiesClient(entities_api)

        specs = [ExecutorSpec(name=e.name, backend=e.backend, config=e.config) for e in config.executors]
        if specs:
            registry = ExecutorRegistry.from_config(
                sdk,
                specs,
                default_executor=config.default_executor,
            )
        else:
            registry = ExecutorRegistry.empty()
        self._registry = registry

        self._deployment_reconciler = DeploymentReconciler(self.entities, registry, config.controller)
        self._volume_reconciler = VolumeReconciler(self.entities, registry)
        logger.info("DeploymentsController started.")

    async def on_shutdown(self) -> None:
        if self._registry is not None:
            self._registry.shutdown_all()
        logger.info("DeploymentsController shut down.")

    async def reconcile(self) -> None:
        if self._deployment_reconciler is None:
            raise RuntimeError("DeploymentsController.reconcile() called before on_startup()")
        if self._volume_reconciler is None:
            raise RuntimeError("DeploymentsController.reconcile() called before on_startup()")
        if self._registry is None:
            raise RuntimeError("DeploymentsController.reconcile() called before on_startup()")

        deployments = await self._list_deployments()
        if self.stop_requested():
            return

        volumes = await self._list_volumes()
        if self.stop_requested():
            return

        configs = await self._load_configs(deployments)
        self._deployment_reconciler.set_config_cache(configs)

        deployments_by_name = _index_deployments(deployments)
        volumes_by_name = _index_volumes(volumes)
        await self._ensure_volume_refs_loaded(configs, volumes_by_name)
        await self._ensure_prerequisite_refs_loaded(deployments, deployments_by_name)
        if self.stop_requested():
            return

        for volume in volumes:
            if self.stop_requested():
                return
            try:
                await self._volume_reconciler.reconcile_one(volume)
            except NemoEntityConflictError:
                logger.debug("Optimistic lock conflict on volume '%s' — retry next cycle.", volume.name)
            except Exception:
                logger.exception("Error reconciling volume %s/%s", volume.workspace, volume.name)

        for deployment in deployments:
            if self.stop_requested():
                return
            try:
                await self._deployment_reconciler.reconcile_one(
                    deployment,
                    deployments_by_name=deployments_by_name,
                    volumes_by_name=volumes_by_name,
                )
            except NemoEntityConflictError:
                logger.debug("Optimistic lock conflict on deployment '%s' — retry next cycle.", deployment.name)
            except Exception:
                logger.exception("Error reconciling deployment %s/%s", deployment.workspace, deployment.name)

        self._orphan_cleanup_elapsed_seconds += self._interval_seconds
        orphan_interval = self.controller_config.orphan_cleanup_interval_seconds
        if (
            orphan_interval > 0
            and self._orphan_cleanup_elapsed_seconds >= orphan_interval
            and self._deployments_list_ok
            and not self.stop_requested()
        ):
            terminal_deployments = await self._list_terminal_deployments_for_orphan_grace()
            if not self._terminal_deployments_list_ok:
                return
            known_ids = _orphan_protected_ids(
                deployments,
                terminal_deployments,
                self.controller_config.terminal_orphan_grace_seconds,
            )
            await reconcile_orphans(self._registry.all_backends(), known_ids)
            self._orphan_cleanup_elapsed_seconds = 0.0

    async def list_objects(self) -> list:
        raise NotImplementedError("DeploymentsController uses reconcile() override")

    async def reconcile_one(self, obj: object) -> None:
        raise NotImplementedError("DeploymentsController uses reconcile() override")

    async def _list_deployments(self) -> list[Deployment]:
        try:
            deployments = await list_all_pages(
                self.entities,
                Deployment,
                filter_operation=ComparisonOperation(
                    operator=FilterOperator.IN,
                    field="data.status",
                    value=list(NON_TERMINAL_DEPLOYMENT_STATUSES),
                ),
            )
            self._deployments_list_ok = True
            return deployments
        except Exception:
            logger.exception("Failed to list non-terminal deployments")
            self._deployments_list_ok = False
            return []

    async def _list_volumes(self) -> list[Volume]:
        try:
            volumes = await list_all_pages(
                self.entities,
                Volume,
                filter_operation=ComparisonOperation(
                    operator=FilterOperator.IN,
                    field="data.status",
                    value=list(NON_TERMINAL_VOLUME_STATUSES),
                ),
            )
            self._volumes_list_ok = True
            return volumes
        except Exception:
            logger.exception("Failed to list non-terminal volumes")
            self._volumes_list_ok = False
            return []

    async def _list_terminal_deployments_for_orphan_grace(self) -> list[Deployment]:
        grace_seconds = self.controller_config.terminal_orphan_grace_seconds
        if grace_seconds <= 0:
            return []
        try:
            deployments = await list_all_pages(
                self.entities,
                Deployment,
                filter_operation=ComparisonOperation(
                    operator=FilterOperator.IN,
                    field="data.status",
                    value=list(_TERMINAL_ORPHAN_STATUSES),
                ),
            )
            self._terminal_deployments_list_ok = True
            return deployments
        except Exception:
            logger.exception("Failed to list terminal deployments for orphan grace")
            self._terminal_deployments_list_ok = False
            return []

    async def _load_configs(self, deployments: list[Deployment]) -> dict[tuple[str, str], DeploymentConfig]:
        configs: dict[tuple[str, str], DeploymentConfig] = {}
        for deployment in deployments:
            key = (deployment.workspace, deployment.deployment_config)
            if key in configs:
                continue
            try:
                configs[key] = await self.entities.get(
                    DeploymentConfig,
                    name=deployment.deployment_config,
                    workspace=deployment.workspace,
                )
            except Exception:
                logger.warning(
                    "Failed to load DeploymentConfig '%s' in workspace '%s'",
                    deployment.deployment_config,
                    deployment.workspace,
                    exc_info=True,
                )
        return configs

    async def _ensure_volume_refs_loaded(
        self,
        configs: dict[tuple[str, str], DeploymentConfig],
        volumes_by_name: dict[tuple[str, str], Volume],
    ) -> None:
        for (workspace, _config_name), config in configs.items():
            for mount_name in collect_volume_mount_names(config):
                key = (workspace, mount_name)
                if key in volumes_by_name:
                    continue
                try:
                    volumes_by_name[key] = await self.entities.get(
                        Volume,
                        name=mount_name,
                        workspace=workspace,
                    )
                except Exception:
                    logger.debug(
                        "Volume '%s' not yet available in workspace '%s'",
                        mount_name,
                        workspace,
                    )

    async def _ensure_prerequisite_refs_loaded(
        self,
        deployments: list[Deployment],
        deployments_by_name: dict[tuple[str, str], Deployment],
    ) -> None:
        """Load prerequisite deployments from entity store (including terminal states)."""
        for deployment in deployments:
            for prerequisite in deployment.prerequisites:
                try:
                    workspace, name = parse_deployment_ref(prerequisite.deployment_name, deployment.workspace)
                except ValueError:
                    logger.warning(
                        "Invalid prerequisite ref '%s' on deployment '%s' in workspace '%s'",
                        prerequisite.deployment_name,
                        deployment.name,
                        deployment.workspace,
                    )
                    continue
                key = (workspace, name)
                if key in deployments_by_name:
                    continue
                try:
                    dep = await self.entities.get(Deployment, name=name, workspace=workspace)
                except NemoEntityNotFoundError:
                    logger.debug(
                        "Prerequisite deployment '%s' not yet available in workspace '%s'",
                        name,
                        workspace,
                    )
                    continue
                except Exception:
                    logger.warning(
                        "Failed to load prerequisite deployment '%s' in workspace '%s'",
                        name,
                        workspace,
                        exc_info=True,
                    )
                    continue
                deployments_by_name[key] = dep


def _index_volumes(volumes: list[Volume]) -> dict[tuple[str, str], Volume]:
    return {(volume.workspace, volume.name): volume for volume in volumes}


def _index_deployments(deployments: list[Deployment]) -> dict[tuple[str, str], Deployment]:
    return {(deployment.workspace, deployment.name): deployment for deployment in deployments}


def _orphan_protected_ids(
    active_deployments: list[Deployment],
    terminal_deployments: list[Deployment],
    grace_seconds: int,
) -> set[str]:
    known_ids = {f"{deployment.workspace}/{deployment.name}" for deployment in active_deployments}
    if grace_seconds <= 0:
        return known_ids

    now = datetime.now(timezone.utc)
    for deployment in terminal_deployments:
        if _terminal_within_grace(deployment, grace_seconds, now):
            known_ids.add(f"{deployment.workspace}/{deployment.name}")
    return known_ids


def _terminal_within_grace(deployment: Deployment, grace_seconds: int, now: datetime) -> bool:
    if deployment.status not in _TERMINAL_ORPHAN_STATUSES:
        return False
    terminal_at = _terminal_timestamp(deployment)
    if terminal_at is None:
        # Missing history: prefer protecting backend resources over premature orphan delete.
        return True
    return (now - terminal_at).total_seconds() < grace_seconds


def _terminal_timestamp(deployment: Deployment) -> datetime | None:
    for event in reversed(deployment.status_history):
        if event.status in _TERMINAL_ORPHAN_STATUSES and event.timestamp:
            return datetime.fromisoformat(event.timestamp)
    return None
