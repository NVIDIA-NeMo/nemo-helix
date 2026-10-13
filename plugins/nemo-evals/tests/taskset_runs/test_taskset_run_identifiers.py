# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin the public name algorithm, scope separation and random byte budget."""

from hashlib import sha256

import pytest
from nemo_evals.taskset_runs.identifiers import derive_run_id, member_name, random_run_id, task_job_name


def test_identifiers_and_scope() -> None:
    """Pin deterministic names and ensure distinct scope inputs produce distinct IDs.

    Fixed workspace/caller/key tuples must match the null-separated hash contract;
    member and Job names must retain their formats and the exercised length bound.
    These assertions protect stable retries and avoid ambiguous component joins.
    """
    expected = "run-" + sha256(b"default\0user:alice\0sort-suite-ci-2026-10-05").hexdigest()[:16]
    assert derive_run_id("default", "user:alice", "sort-suite-ci-2026-10-05") == expected
    assert len(expected) == 20
    assert member_name(7) == "task-7"
    assert task_job_name(expected, 999) == f"{expected}-task-999"
    assert len(task_job_name(expected, 999)) <= 51
    scopes = [
        ("default", "user:alice", "key"),
        ("other", "user:alice", "key"),
        ("default", "user:bob", "key"),
        ("default", "user:alice", "other"),
        ("ab", "c", "d"),
        ("a", "bc", "d"),
    ]
    assert len({derive_run_id(*scope) for scope in scopes}) == len(scopes)


def test_random_identifiers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use monkeypatch to make entropy deterministic and verify keyless name generation.

    Both random entry points must return the supplied token with a run prefix and
    request eight bytes, protecting the random-name format and entropy budget.
    """
    requests = []

    def token_hex(size: int) -> str:
        requests.append(size)
        return "0123456789abcdef"

    monkeypatch.setattr("nemo_evals.taskset_runs.identifiers.secrets.token_hex", token_hex)
    assert random_run_id() == "run-0123456789abcdef"
    assert derive_run_id("default", "user:alice", None) == "run-0123456789abcdef"
    assert requests == [8, 8]
