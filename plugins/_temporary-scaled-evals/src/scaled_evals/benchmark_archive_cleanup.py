# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconcile claim-scoped uploads against the authoritative archive row."""

from collections.abc import Sequence
from pathlib import PurePosixPath

from scaled_evals.api import artifacts
from scaled_evals.api.benchmark_archive_store import BenchmarkArchiveStore
from scaled_evals.archive_validation import benchmark_archive_object_key, validate_archive_id


def cleanup_benchmark_archives(
    run_id: str,
    *,
    object_keys: Sequence[str] | None = None,
) -> int:
    """Delete only unpublished, inactive objects, never guessing after a store error.

    Objects are listed before the archive's references are read, so a claim
    taken after the read uploads under a key that is not a candidate. Claim keys
    are unique and publication is fenced by the claim token, so a key the read
    does not protect can never become the published archive. Periodic
    reconciliation also catches uploads completed after a crashed or revoked
    worker's last cleanup attempt.
    """
    prefix = f"benchmark-runs/{validate_archive_id(run_id)}/archives/"
    keys = object_keys if object_keys is not None else [item["key"] for item in artifacts.list_objects(prefix)]
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
    row = BenchmarkArchiveStore().for_cleanup(run_id)
    if row is None:
        return 0
    protected = {row.get("object_key")}
    if row["status"] == "building" and row.get("claim_token"):
        protected.add(benchmark_archive_object_key(run_id, row["generation"], row["claim_token"]))
    for key in candidates:
        if key not in protected:
            artifacts.delete_object(key)
            deleted += 1
    return deleted
