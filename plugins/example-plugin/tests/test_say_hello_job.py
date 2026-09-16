# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`~nemo_example_plugin.jobs.say_hello.SayHelloJob`.

Pin the end-to-end wiring: running the job through
:class:`~nemo_helix_plugin.scheduler.NemoJobScheduler` with no clients writes
the greeting to ``ctx.storage.persistent`` and registers it via the
default :class:`~nemo_helix_plugin.job_results.LocalJobResults` with a
``file://`` URL.
"""

from __future__ import annotations

from pathlib import Path

from nemo_example_plugin.jobs.say_hello import (
    DEFAULT_FILE_NAME,
    DEFAULT_RESULT_NAME,
    SayHelloJob,
)
from nemo_helix_plugin.scheduler import NemoJobScheduler


def test_say_hello_job_metadata() -> None:
    assert SayHelloJob.name == "say-hello"
    assert SayHelloJob.description


def test_say_hello_runs_locally_without_clients() -> None:
    scheduler = NemoJobScheduler()
    result = scheduler.run_local(
        SayHelloJob,
        {"name": "Razvan"},
        workspace="dev",
    )
    assert result["result"] == "Hello, Razvan!"
    artefact = result["artifact"]
    assert artefact["name"] == DEFAULT_RESULT_NAME
    artifact_path = Path(artefact["artifact_url"].removeprefix("file://"))
    assert artifact_path.exists()
    assert artifact_path.read_text() == "Hello, Razvan!"


def test_defaults_name_to_world() -> None:
    result = NemoJobScheduler().run_local(SayHelloJob, {}, workspace="dev")
    assert result["result"] == "Hello, world!"


def test_greeting_text_lands_under_persistent(tmp_path: Path) -> None:
    from nemo_helix_plugin.job_context import JobContext, StoragePaths
    from nemo_helix_plugin.job_results import LocalJobResults

    storage = StoragePaths(ephemeral=tmp_path / "e", persistent=tmp_path / "p")
    storage.ephemeral.mkdir()
    storage.persistent.mkdir()
    ctx = JobContext(
        workspace="dev",
        storage=storage,
        results=LocalJobResults(root=storage.persistent / "results"),
    )
    NemoJobScheduler().run_local(
        SayHelloJob,
        {"name": "Razvan"},
        workspace="dev",
        ctx=ctx,
    )
    assert (storage.persistent / DEFAULT_FILE_NAME).read_text() == "Hello, Razvan!"
