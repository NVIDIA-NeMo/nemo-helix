# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intake service implementation."""

import asyncio
import logging
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path
from typing import ClassVar, List

from fastapi import FastAPI
from nhx.common.entities.client import EntityClient
from nhx.common.service import RouterConfig, Service
from nhx.common.service.dependencies import service_instance_dependency
from nhx.intake.background_worker import BackgroundWorker
from nhx.intake.config import IntakeConfig, should_provision_local_clickhouse
from nhx.intake.experiments.denormalizer import EvaluationDenormalizer
from nhx.intake.local_clickhouse import (
    DockerUnavailableError,
    LocalClickHouseProvisioningError,
    check_local_clickhouse_data_directory,
    reconcile_local_clickhouse,
    stop_local_clickhouse,
)
from nhx.intake.readiness import CLICKHOUSE_UNAVAILABLE_MESSAGE, LOCAL_CLICKHOUSE_PROVISIONING_MESSAGE
from nhx.intake.repository.clickhouse.evaluation_rollup import ClickHouseEvaluationRollupRepository
from nhx.intake.repository.clickhouse.executor import ClickHouseExecutor, ClickHouseQuery
from nhx.intake.repository.clickhouse.tables import ClickHouseTable
from nhx.intake.spans.clickhouse_client import ClickHouseSettings, ClickHouseSpanClient
from nhx.intake.spans.ingest.atif_offload import atif_ingest_worker_hold, atif_ingest_worker_ready

logger = logging.getLogger(__name__)

# Interruptible pause between failed local ClickHouse reconciles. Tests set this to 0.
LOCAL_CLICKHOUSE_RECONCILE_RETRY_SECONDS = 5.0

# Spawn, not fork: workers must not inherit a copy of this multithreaded server process.
ATIF_INGEST_POOL_START_METHOD = "spawn"
# Long enough that each warm-up task lands on a fresh worker, so startup spawns the full pool.
ATIF_INGEST_WARMUP_HOLD_SECONDS = 0.05


class _LocalClickHouseProvisioner(BackgroundWorker):
    """Retry local ClickHouse reconciliation until it succeeds or Intake shuts down."""

    def __init__(
        self,
        service: "IntakeService",
        settings: ClickHouseSettings,
        *,
        image: str,
        data_dir: Path | None,
    ) -> None:
        super().__init__()
        self._service = service
        self._settings = settings
        self._image = image
        self._data_dir = data_dir
        self._logged_failures: set[str] = set()

    async def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                url = await reconcile_local_clickhouse(
                    self._settings,
                    image=self._image,
                    data_dir=self._data_dir,
                )
            except LocalClickHouseProvisioningError as exc:
                self._record_failure(exc)
                if await self._wait_for_retry():
                    return
                continue
            except Exception as exc:
                logger.exception("Local ClickHouse reconcile failed")
                self._service._readiness_message = str(exc) or CLICKHOUSE_UNAVAILABLE_MESSAGE
                if await self._wait_for_retry():
                    return
                continue
            if self._stopping.is_set():
                await self._stop_unadopted_container()
                return
            try:
                self._service._adopt_reconciled_clickhouse(url, self._settings, self._data_dir)
            except Exception:
                logger.exception("Failed to adopt reconciled local ClickHouse")
                await self._stop_unadopted_container()
                raise
            return

    def _record_failure(self, exc: LocalClickHouseProvisioningError) -> None:
        message = str(exc)
        self._service._readiness_message = message
        if message in self._logged_failures:
            return
        self._logged_failures.add(message)
        log = logger.warning if isinstance(exc, DockerUnavailableError) else logger.error
        log(
            "Local ClickHouse reconciliation failed: %s Retrying until Intake shuts down.",
            exc,
            extra={"service": self._service.name},
        )

    async def _wait_for_retry(self) -> bool:
        """Return True when shutdown was requested during the backoff."""
        try:
            await asyncio.wait_for(
                self._stopping.wait(),
                timeout=LOCAL_CLICKHOUSE_RECONCILE_RETRY_SECONDS,
            )
        except TimeoutError:
            return False
        return True

    async def _stop_unadopted_container(self) -> None:
        try:
            await stop_local_clickhouse(data_dir=self._data_dir)
        except LocalClickHouseProvisioningError:
            logger.warning(
                "Failed to stop local ClickHouse after shutdown interrupted reconciliation",
                exc_info=True,
                extra={"service": self._service.name},
            )


class IntakeService(Service[IntakeConfig]):
    """Intake service for NeMo Helix."""

    dependencies: ClassVar[List[str]] = ["entities", "auth"]

    def __init__(self):
        """Initialize the intake service."""
        super().__init__(name="intake", module_name="nhx.intake")
        # The client is owned by the service lifecycle; it is absent before startup and after shutdown.
        self.clickhouse_client: ClickHouseSpanClient | None = None
        # Runs ATIF mapping off the event loop; absent before startup and after shutdown.
        self.atif_ingest_executor: ProcessPoolExecutor | None = None
        # ATIF requests holding a pool slot (mapping or queued); capped by atif_ingest_max_pending.
        self.atif_ingest_pending = 0
        # Background worker that denormalizes agent/model name fields onto Evaluation entities.
        self.denormalizer: EvaluationDenormalizer | None = None
        self._local_clickhouse_data_dir: Path | None = None
        self._owns_local_clickhouse = False
        self._ready = False
        self._readiness_message = ""
        self._reconciler: _LocalClickHouseProvisioner | None = None
        self._denormalizer_entity_client: EntityClient | None = None

    def _setup_dependencies(self, app: FastAPI) -> None:
        super()._setup_dependencies(app)
        service_instance_dependency(self.name, IntakeService).bind(app, self)

    @property
    def title(self) -> str:
        return "Intake API"

    @property
    def description(self) -> str:
        return "Intake service for ingesting and reading sessions, traces, spans, annotations, and evaluator results"

    @property
    def readiness_message(self) -> str:
        """Return actionable guidance when ClickHouse prevents Intake readiness."""
        return self._readiness_message

    def get_routers(self) -> List[RouterConfig]:
        """Return routers for the intake service."""
        from nhx.intake.api.v2.experiments import endpoints as experiments
        from nhx.intake.spans.api import annotations, evaluator_results, sessions, spans, trace_metrics, traces
        from nhx.intake.spans.ingest import atif, chat_completions, otlp
        from nhx.intake.spans.ingest import spans as span_ingest

        return [
            RouterConfig(spans.router, tag="Spans", description="ClickHouse-backed span read endpoints"),
            # Must precede traces: /traces/metrics would otherwise bind to /traces/{id}.
            RouterConfig(
                trace_metrics.router,
                tag="Traces",
                description="Time-bucketed trace metric rollups",
            ),
            RouterConfig(traces.router, tag="Traces", description="ClickHouse-backed trace summary read endpoints"),
            RouterConfig(sessions.router, tag="Sessions", description="ClickHouse-backed session detail endpoints"),
            RouterConfig(
                evaluator_results.router,
                tag="Evaluator Results",
                description="ClickHouse-backed evaluator_result endpoints",
            ),
            RouterConfig(
                annotations.router,
                tag="Annotations",
                description="Post-hoc annotation endpoints (feedback, labels, notes, metadata)",
            ),
            RouterConfig(
                experiments.router,
                tag="Experiments",
                description="Create, list, get, and delete Evaluations and Experiments",
            ),
            RouterConfig(otlp.router, tag="Ingest", description="OTLP/HTTP trace ingest endpoints"),
            RouterConfig(atif.router, tag="Ingest", description="ATIF trajectory ingest endpoints"),
            RouterConfig(span_ingest.router, tag="Ingest", description="Provider-neutral JSON span ingest endpoint"),
            RouterConfig(
                chat_completions.router,
                tag="Ingest",
                description="OpenAI-compatible chat-completion ingest endpoint",
            ),
        ]

    async def on_startup(self) -> None:
        """Start Intake without blocking the platform lifespan on local ClickHouse."""

        self._local_clickhouse_data_dir = None
        self._owns_local_clickhouse = False
        self._reconciler = None
        cfg = self.service_config or IntakeConfig()
        settings = ClickHouseSettings.from_config(cfg)
        # Start the ingest worker pool first so the spawn cost overlaps the rest of startup.
        self.atif_ingest_executor = self._create_atif_ingest_executor()
        try:
            self._start_rest_of_intake(cfg, settings)
        except BaseException:
            # The lifespan never runs on_shutdown when startup failed, so close the
            # pool here instead of leaking its worker processes.
            executor, self.atif_ingest_executor = self.atif_ingest_executor, None
            await self._stop_atif_ingest_executor(executor)
            raise

    def _start_rest_of_intake(self, cfg: IntakeConfig, settings: ClickHouseSettings) -> None:
        # The denormalizer needs a service-principal entity client (no request context).
        self._denormalizer_entity_client = self.dependency_provider.get_service_entity_client(self.name)
        if self._denormalizer_entity_client is None:
            logger.warning("Entity client unavailable; evaluation denormalizer not started")
        if should_provision_local_clickhouse(cfg.clickhouse_config):
            self._readiness_message = LOCAL_CLICKHOUSE_PROVISIONING_MESSAGE
            self._reconciler = _LocalClickHouseProvisioner(
                self,
                settings,
                image=cfg.clickhouse_config.image,
                data_dir=cfg.clickhouse_config.data_dir,
            )
            self._reconciler.start()
        else:
            self.clickhouse_client = ClickHouseSpanClient(settings)
            self._start_denormalizer(self.clickhouse_client)
        self._ready = True

    def _create_atif_ingest_executor(self) -> ProcessPoolExecutor:
        workers = (self.service_config or IntakeConfig()).atif_ingest_workers
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=get_context(ATIF_INGEST_POOL_START_METHOD))
        try:
            self._warm_atif_ingest_executor(executor, workers)
        except BaseException:
            # Warm-up spawns synchronously; a failed second spawn must not leak the first worker.
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        return executor

    def _warm_atif_ingest_executor(self, executor: ProcessPoolExecutor, workers: int) -> None:
        """Pre-spawn every ingest worker so the first overlapping ingests skip startup latency."""
        tasks = [executor.submit(atif_ingest_worker_hold, ATIF_INGEST_WARMUP_HOLD_SECONDS) for _ in range(workers - 1)]
        tasks.append(executor.submit(atif_ingest_worker_ready))
        for task in tasks:
            task.add_done_callback(self._log_atif_ingest_warmup_failure)

    async def _stop_atif_ingest_executor(self, executor: ProcessPoolExecutor) -> None:
        """Close the pool off the event loop so a draining worker cannot freeze teardown."""
        await asyncio.to_thread(executor.shutdown, True)

    def _log_atif_ingest_warmup_failure(self, future: Future[None]) -> None:
        if future.cancelled():
            return
        error = future.exception()
        if error is not None:
            logger.warning(
                "ATIF ingest worker warm-up failed; the next ingest returns 503 and rebuilds the pool",
                exc_info=error,
                extra={"service": self.name},
            )

    def rebuild_atif_ingest_executor(self, *, broken: ProcessPoolExecutor) -> None:
        """Replace a pool whose worker died; ``ProcessPoolExecutor`` never recovers on its own.

        A no-op when ``broken`` was already replaced by an overlapping failure.
        """
        if self.atif_ingest_executor is not broken:
            return
        logger.error(
            "ATIF ingest worker pool broke; replacing it",
            extra={"service": self.name},
        )
        try:
            replacement = self._create_atif_ingest_executor()
        except (OSError, RuntimeError):
            logger.exception(
                "Could not replace the broken ATIF ingest worker pool; the next ingest retries",
                extra={"service": self.name},
            )
            return
        self.atif_ingest_executor = replacement
        # The broken pool has no live work left; closing it must not block the loop.
        broken.shutdown(wait=False, cancel_futures=True)

    def _adopt_reconciled_clickhouse(self, url: str, settings: ClickHouseSettings, data_dir: Path | None) -> None:
        """Install the reconciled client. No await, so shutdown cannot interleave mid-assign."""

        client = ClickHouseSpanClient(replace(settings, url=url))
        self._local_clickhouse_data_dir = data_dir
        self._owns_local_clickhouse = True
        self.clickhouse_client = client
        self._start_denormalizer(client)

    def _start_denormalizer(self, client: ClickHouseSpanClient) -> None:
        entity_client = self._denormalizer_entity_client
        if entity_client is None or self.denormalizer is not None:
            return
        cfg = self.service_config or IntakeConfig()
        self.denormalizer = EvaluationDenormalizer(
            rollup_repository=ClickHouseEvaluationRollupRepository(ClickHouseExecutor(client)),
            entity_client=entity_client,
            interval_seconds=cfg.denormalization_interval_seconds,
        )
        self.denormalizer.start()

    async def on_shutdown(self) -> None:
        """Stop the ingest pool, close the client, and stop the managed local ClickHouse container."""

        self._ready = False
        self._readiness_message = ""
        executor, self.atif_ingest_executor = self.atif_ingest_executor, None
        if executor is not None:
            # Requests have drained by lifespan shutdown; a straggler mapping still
            # finishes so no offload outlives the service, but off the loop.
            try:
                await self._stop_atif_ingest_executor(executor)
            except Exception:
                logger.exception("ATIF ingest worker pool failed to shut down", extra={"service": self.name})
        reconciler = self._reconciler
        self._reconciler = None
        if reconciler is not None:
            try:
                await reconciler.stop()
            except Exception:
                logger.exception("Local ClickHouse reconciler failed during shutdown")
        # Stop the denormalizer first: its final flush still needs the ClickHouse client below.
        if self.denormalizer is not None:
            await self.denormalizer.stop()
            self.denormalizer = None
        try:
            if self.clickhouse_client is not None:
                await self.clickhouse_client.close()
        finally:
            self.clickhouse_client = None
            try:
                if self._owns_local_clickhouse:
                    await stop_local_clickhouse(data_dir=self._local_clickhouse_data_dir)
            finally:
                self._owns_local_clickhouse = False
                await super().on_shutdown()

    async def is_ready(self) -> bool:
        client = self.clickhouse_client
        if not self._ready or client is None:
            return False

        try:
            if self._owns_local_clickhouse:
                await check_local_clickhouse_data_directory(data_dir=self._local_clickhouse_data_dir)
            executor = ClickHouseExecutor(client)
            await executor.fetch_scalar(
                ClickHouseQuery(
                    name="intake_readiness",
                    statement=f"SELECT 1 AS ready FROM {executor.table(ClickHouseTable.SPANS)} LIMIT 1",
                )
            )
        except Exception:
            if self._readiness_message != CLICKHOUSE_UNAVAILABLE_MESSAGE:
                logger.warning("Intake ClickHouse readiness probe failed", exc_info=True)
            self._readiness_message = CLICKHOUSE_UNAVAILABLE_MESSAGE
            return False

        self._readiness_message = ""
        return self._ready and client is self.clickhouse_client
