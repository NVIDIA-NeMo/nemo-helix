# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from typing import Annotated

from fastapi import Depends

from scaled_evals.api.agent_bundle_store import AgentBundleStore
from scaled_evals.api.auth import CurrentPrincipal, current_principal
from scaled_evals.api.benchmark_archive_store import BenchmarkArchiveRequestRepository
from scaled_evals.api.benchmark_import_store import BenchmarkImportStore
from scaled_evals.api.benchmark_run_store import BenchmarkRunRequestRepository
from scaled_evals.api.benchmark_store import BenchmarkStore
from scaled_evals.api.config_profile_store import ConfigProfileStore
from scaled_evals.api.evaluation_store import EvaluationGlobalQueries, EvaluationRequestRepository
from scaled_evals.api.resource_access import ResourceAccess
from scaled_evals.api.task_store import TaskStore
from scaled_evals.api.tenancy import IsAdmin


class Database:
    """Catalog stores shared by request facades, without evaluation or run access."""

    @property
    def agent_bundles(self) -> AgentBundleStore:
        return AgentBundleStore()

    @property
    def tasks(self) -> TaskStore:
        return TaskStore()

    @property
    def benchmarks(self) -> BenchmarkStore:
        return BenchmarkStore()

    @property
    def benchmark_imports(self) -> BenchmarkImportStore:
        return BenchmarkImportStore()

    @property
    def config_profiles(self) -> ConfigProfileStore:
        return ConfigProfileStore()


def get_db() -> Database:
    return Database()


class RequestDatabase(Database):
    def __init__(self, access: ResourceAccess) -> None:
        self.access = access

    @property
    def evaluations(self) -> EvaluationRequestRepository:
        return EvaluationRequestRepository(access=self.access)

    @property
    def benchmark_runs(self) -> BenchmarkRunRequestRepository:
        return BenchmarkRunRequestRepository(access=self.access)

    @property
    def benchmark_archives(self) -> BenchmarkArchiveRequestRepository:
        return BenchmarkArchiveRequestRepository(access=self.access)


def get_request_db(
    current: Annotated[CurrentPrincipal, Depends(current_principal)],
    admin: IsAdmin,
) -> RequestDatabase:
    """Bind request stores to the effective owner and the platform admin decision."""
    return RequestDatabase(ResourceAccess(current.owner_id, admin or current.source == "disabled"))


def get_request_stream_database_factory(
    current: Annotated[CurrentPrincipal, Depends(current_principal)],
    admin: IsAdmin,
) -> Callable[[], AbstractContextManager[RequestDatabase]]:
    """Retain request access while opening a fresh store facade for each poll."""
    access = ResourceAccess(current.owner_id, admin or current.source == "disabled")

    @contextmanager
    def request_database() -> Generator[RequestDatabase]:
        yield RequestDatabase(access)

    return request_database


def get_evaluation_global_queries() -> EvaluationGlobalQueries:
    """Read-only cross-owner evaluation queries for explicit global checks and admin reports."""
    return EvaluationGlobalQueries()
