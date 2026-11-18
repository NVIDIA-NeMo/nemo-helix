# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The conformance suite against the local backend, so the suite and the protocol are checked in CI without a cluster."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from nhx_sandbox.backends.local import LocalProcessBackend
from nhx_sandbox.conformance import EgressProbe, run_suite
from nhx_sandbox.core.backend import SandboxSpec

# The image's USER decides who the local backend runs as, so there's no backend-wide uid to check, and
# no egress probe is given because the local backend doesn't enforce egress.
EXPECTED_SKIPS = {"workload_user", "egress_denied_by_default"}


async def test_local_backend_passes_the_conformance_suite(tmp_path: Path) -> None:
    backend = LocalProcessBackend(base_dir=tmp_path)
    spec = SandboxSpec(image="local", name_hint="conf", labels={"nhx-conformance-run": uuid.uuid4().hex})

    results = await run_suite(backend, spec)

    failed = [f"{result.name}: {result.detail}" for result in results if result.outcome == "failed"]
    assert not failed, "\n".join(failed)
    assert {result.name for result in results if result.outcome == "skipped"} == EXPECTED_SKIPS
    assert not list(tmp_path.iterdir()), "the suite left sandboxes behind"


async def test_the_default_egress_case_fails_a_backend_that_lets_connections_out(tmp_path: Path) -> None:
    """The local backend enforces no egress, so the case has to catch it reaching an "internal" address."""
    server = await asyncio.start_server(lambda reader, writer: writer.close(), "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    probe = EgressProbe(
        allowed_host="allowed.invalid",
        denied_host="denied.invalid",
        internal_targets=(f"127.0.0.1:{port}",),
        # Pretend every fetch is refused, so only the direct connection can trip the case.
        fetch="false {url}",
    )
    spec = SandboxSpec(image="local", name_hint="conf", labels={"nhx-conformance-run": uuid.uuid4().hex})

    async with server:
        (result,) = await run_suite(
            LocalProcessBackend(base_dir=tmp_path), spec, egress=probe, only=frozenset({"egress_denied_by_default"})
        )

    assert result.outcome == "failed"
    assert f"127.0.0.1:{port}" in result.detail
