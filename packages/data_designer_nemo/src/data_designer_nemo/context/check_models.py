# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine context for model probes that have no sync SDK.

``create_data_designer`` needs three engine components: a secret resolver,
seed readers, and a person reader. The real implementations
(:class:`~data_designer_nemo.context.execution.DataDesignerExecutionContext`) all need a
sync ``NeMoPlatform``, and an async caller cannot build one — rebuilding a sync
SDK from an async one is deliberately unsupported (see
:mod:`data_designer_nemo.sdk_translation`).

A model probe needs none of them. ``DataDesigner.check_models`` resolves model
aliases, sends one tiny generation per alias, and reads no data: no seed is
opened, no person dataset is sampled, and no secret is resolved (providers here
authenticate through ``extra_headers``, and the engine skips secret resolution
entirely when a provider's ``api_key`` is unset, which it always is on this
path).

So this context satisfies the same shape ``create_data_designer`` consumes
while supplying components that refuse to do work rather than pretending to.
If the probe ever does reach one of them, that is a change in what
``check_models`` touches, and the loud failure is the point — the alternative
is a probe quietly reading through a half-built context.

The one subtlety is seeds. ``create_resource_provider`` eagerly looks up a
reader for the configured seed source, before any reading happens, and fails if
the registry has no entry for that seed type. So a placeholder is registered for
the config's own seed type — derived from the config, so it can never fall out
of step with the seed types the platform supports.
"""

from __future__ import annotations

import duckdb
from data_designer.engine.resources.person_reader import PersonReader
from data_designer.engine.resources.seed_reader import SeedReader
from data_designer_nemo.errors import NDDInternalError

_UNREACHED = "This component is not used by a model health check and has no sync SDK to work with."


class CheckModelsSecretResolver:
    """Secret resolver placeholder. Raises rather than resolving."""

    def resolve(self, secret: str) -> str:
        raise NDDInternalError(f"Cannot resolve secret {secret!r} during a model health check. {_UNREACHED}")


class CheckModelsSeedReader(SeedReader):
    """Seed reader placeholder registered for one seed type. Raises if read.

    ``get_seed_type`` is normally derived from the reader's generic source
    parameter; here it is supplied directly so a single class can stand in for
    whichever seed type the config happens to use.
    """

    def __init__(self, seed_type: str) -> None:
        self._seed_type = seed_type

    def get_seed_type(self) -> str:
        return self._seed_type

    def get_dataset_uri(self) -> str:
        raise NDDInternalError(f"Cannot read seed data during a model health check. {_UNREACHED}")

    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        raise NDDInternalError(f"Cannot read seed data during a model health check. {_UNREACHED}")


class CheckModelsPersonReader(PersonReader):
    """Person reader placeholder. Raises rather than reading.

    Supplied so the engine does not fall back to its local managed-assets
    reader, which would look for person data on the caller's disk.
    """

    def get_dataset_uri(self, locale: str) -> str:
        raise NDDInternalError(
            f"Cannot read person data for locale {locale!r} during a model health check. {_UNREACHED}"
        )

    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        raise NDDInternalError(f"Cannot read person data during a model health check. {_UNREACHED}")


class DataDesignerCheckModelsContext:
    """Engine context supplying only what a model probe actually needs.

    Deliberately *not* a ``DataDesignerExecutionContext``: it cannot execute a
    workload, and the distinct type keeps it from being passed somewhere that
    expects one.
    """

    def __init__(self, *, seed_type: str | None = None) -> None:
        self._seed_type = seed_type

    def get_secret_resolver(self) -> CheckModelsSecretResolver:
        return CheckModelsSecretResolver()

    def get_seed_readers(self) -> list[SeedReader]:
        if self._seed_type is None:
            return []
        return [CheckModelsSeedReader(self._seed_type)]

    def get_person_reader(self) -> PersonReader | None:
        return CheckModelsPersonReader()


def create_check_models_context(*, seed_type: str | None = None) -> DataDesignerCheckModelsContext:
    """Build a probe-only engine context.

    Args:
        seed_type: The ``seed_type`` of the config's seed source, if it has one.
            A placeholder reader is registered for it so resource-provider
            construction can find one.
    """
    return DataDesignerCheckModelsContext(seed_type=seed_type)
