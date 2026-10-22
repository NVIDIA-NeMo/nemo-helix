# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

from scaled_evals.api.agent_bundle_store import AgentBundleStore
from scaled_evals.api.benchmark_archive_store import BenchmarkArchiveStore
from scaled_evals.api.benchmark_import_store import BenchmarkImportStore
from scaled_evals.api.benchmark_run_store import BenchmarkRunStore
from scaled_evals.api.benchmark_store import BenchmarkStore
from scaled_evals.api.config_profile_store import ConfigProfileStore
from scaled_evals.api.evaluation_store import EvaluationStore
from scaled_evals.api.task_store import TaskStore


class Database:
    """Route-facing facade over the entity-backed stores."""

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
    def benchmark_archives(self) -> BenchmarkArchiveStore:
        return BenchmarkArchiveStore()

    @property
    def benchmark_runs(self) -> BenchmarkRunStore:
        return BenchmarkRunStore()

    @property
    def config_profiles(self) -> ConfigProfileStore:
        return ConfigProfileStore()

    @property
    def evaluations(self) -> EvaluationStore:
        return EvaluationStore()


def get_db() -> Database:
    return Database()


@contextmanager
def stream_database() -> Iterator[Database]:
    yield Database()


def get_stream_database_factory() -> Callable[[], AbstractContextManager[Database]]:
    """Return a factory that streaming routes call once per poll."""
    return stream_database
