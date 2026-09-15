# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconcile claim-scoped uploads against the authoritative archive row."""

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from pathlib import PurePosixPath
from typing import Any

import psycopg

from scaled_evals.api import s3
from scaled_evals.api.repositories.benchmark_archive_repository import BenchmarkArchiveRepository
from scaled_evals.archive_validation import benchmark_archive_object_key, validate_archive_id


def cleanup_benchmark_archives(
    connect: Callable[[], AbstractContextManager[psycopg.Connection[Any]]],
    run_id: str,
    *,
    object_keys: Sequence[str] | None = None,
) -> int:
    """Delete only unpublished, inactive objects, never guessing after a DB error.

    Listing happens outside the transaction. References are checked again under
    an archive-row lock held through deletion, serializing cleanup with claim,
    force-rebuild and publication updates. Unique claim keys cannot be reused by
    a newer worker. Periodic reconciliation also catches uploads completed after
    a crashed/revoked worker's last cleanup attempt.
    """
    prefix = f"benchmark-runs/{validate_archive_id(run_id)}/archives/"
    keys = object_keys if object_keys is not None else [item["key"] for item in s3.list_objects(prefix)]
    candidates = []
    for key in keys:
        if not key.startswith(prefix):
            continue
        relative = key[len(prefix) :]
        path = PurePosixPath(relative)
        if len(path.parts) != 2 or path.as_posix() != relative or not path.name.endswith(".tar.gz"):
            continue
        try:
            validate_archive_id(path.parts[0])
            validate_archive_id(path.name.removesuffix(".tar.gz"))
        except ValueError:
            continue
        candidates.append(key)
    if not candidates:
        return 0
    deleted = 0
    with connect() as conn, conn.transaction():
        row = BenchmarkArchiveRepository(conn).lock_for_cleanup(run_id)
        if row is None:
            return 0
        protected = {row.get("object_key")}
        if row["status"] == "building" and row.get("claim_token"):
            protected.add(benchmark_archive_object_key(run_id, row["generation"], row["claim_token"]))
        for key in candidates:
            if key not in protected:
                s3.delete_object(key)
                deleted += 1
    return deleted
