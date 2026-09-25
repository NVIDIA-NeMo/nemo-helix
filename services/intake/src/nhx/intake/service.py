# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intake service implementation."""

import logging
from dataclasses import replace
from pathlib import Path
from typing import ClassVar, List

from nhx.common.service import RouterConfig, Service
from nhx.intake.api.v2.experiments import endpoints as experiments
from nhx.intake.config import IntakeConfig, should_provision_local_clickhouse
from nhx.intake.experiments.denormalizer import EvaluationDenormalizer
from nhx.intake.local_clickhouse import (
    DockerUnavailableError,
    LocalClickHouseProvisioningError,
    check_local_clickhouse_data_directory,
    reconcile_local_clickhouse,
    stop_local_clickhouse,
)
from nhx.intake.readiness import CLICKHOUSE_UNAVAILABLE_MESSAGE
from nhx.intake.repository.clickhouse.evaluation_rollup import ClickHouseEvaluationRollupRepository
from nhx.intake.repository.clickhouse.executor import ClickHouseExecutor, ClickHouseQuery
from nhx.intake.repository.clickhouse.tables import ClickHouseTable
from nhx.intake.spans.api import annotations, evaluator_results, sessions, spans, trace_metrics, traces
from nhx.intake.spans.clickhouse_client import ClickHouseSettings, ClickHouseSpanClient
from nhx.intake.spans.ingest import atif, chat_completions, otlp
from nhx.intake.spans.ingest import spans as span_ingest

logger = logging.getLogger(__name__)


class IntakeService(Service[IntakeConfig]):
    """Intake service for NeMo Helix."""

    dependencies: ClassVar[List[str]] = ["entities", "auth"]

    def __init__(self):
        """Initialize the intake service."""
        super().__init__(name="intake", module_name="nhx.intake")
        # The client is owned by the service lifecycle; it is absent before startup and after shutdown.
        self.clickhouse_client: ClickHouseSpanClient | None = None
        # Background worker that denormalizes agent/model name fields onto Evaluation entities.
        self.denormalizer: EvaluationDenormalizer | None = None
        self._local_clickhouse_data_dir: Path | None = None
        self._owns_local_clickhouse = False
        self._ready = False
        self._readiness_message = ""

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
        """Create the trace storage client without requiring ClickHouse to be online."""

        self._local_clickhouse_data_dir = None
        self._owns_local_clickhouse = False
        cfg = self.service_config or IntakeConfig()
        settings = ClickHouseSettings.from_config(cfg)
        if should_provision_local_clickhouse(cfg.clickhouse_config):
            try:
                local_url = await reconcile_local_clickhouse(
                    settings,
                    image=cfg.clickhouse_config.image,
                    data_dir=cfg.clickhouse_config.data_dir,
                )
                settings = replace(settings, url=local_url)
                self._local_clickhouse_data_dir = cfg.clickhouse_config.data_dir
                self._owns_local_clickhouse = True
            except DockerUnavailableError as exc:
                logger.warning(
                    "Skipping local ClickHouse reconciliation: %s ClickHouse-backed endpoints will return 503 until "
                    "ClickHouse is reachable.",
                    exc,
                    extra={"service": self.name, "clickhouse_url": settings.url},
                )
            except LocalClickHouseProvisioningError as exc:
                logger.error(
                    "Local ClickHouse reconciliation failed: %s Intake will continue starting; ClickHouse-backed "
                    "endpoints will return 503 until ClickHouse is reachable.",
                    exc,
                    extra={"service": self.name, "clickhouse_url": settings.url},
                )

        self.clickhouse_client = ClickHouseSpanClient(settings)
        # Start the background denormalizer with a service-principal entity client; startup should fail
        # if the platform cannot build the client for token exchange or trusted-header auth.
        entity_client = self.dependency_provider.get_service_entity_client(self.name)
        self.denormalizer = EvaluationDenormalizer(
            rollup_repository=ClickHouseEvaluationRollupRepository(ClickHouseExecutor(self.clickhouse_client)),
            entity_client=entity_client,
            interval_seconds=cfg.denormalization_interval_seconds,
        )
        self.denormalizer.start()
        self._ready = True

    async def on_shutdown(self) -> None:
        """Close the client and stop the managed local ClickHouse container."""

        self._ready = False
        self._readiness_message = ""
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
