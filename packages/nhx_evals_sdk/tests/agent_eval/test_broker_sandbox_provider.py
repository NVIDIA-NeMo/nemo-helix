# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the broker-backed sandbox provider.

Driven against a **real broker app** over an ASGI transport, with the in-memory episode backend
behind it. Mocking the HTTP layer would leave the thing most worth covering untested -- that this
client and that server agree on the wire contract -- while costing no more than a fixture.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from nhx_evals_sdk.agent_eval.runtimes.sandbox.base import (
    SandboxCreateError,
    SandboxSpec,
    SandboxStatus,
)
from nhx_evals_sdk.agent_eval.runtimes.sandbox.providers.broker import (
    BrokerSandboxError,
    BrokerSandboxProvider,
    UnsupportedSandboxOperationError,
)
from sandboxed_gym.backends.memory import InMemoryEpisodeBackend
from sandboxed_gym.config import EpisodeBrokerConfig
from sandboxed_gym.egress import build_egress_policy
from sandboxed_gym.http_app import build_broker_app
from sandboxed_gym.sandbox_types import SandboxStatus as BrokerSandboxStatus
from sandboxed_gym.wire import BROKER_AUTH_HEADER

TOKEN = "test-broker-token"
IMAGE = "registry.example.com/swe/grader:1.0"


@pytest.fixture
async def provider() -> AsyncIterator[BrokerSandboxProvider]:
    config = EpisodeBrokerConfig(
        job_id="eval-job-1",
        backend="memory",
        allow_insecure_memory_backend=True,
        approved_images=(IMAGE,),
    )
    app = build_broker_app(
        backend=InMemoryEpisodeBackend(build_egress_policy(config.egress_allowlist)),
        config=config,
        token=TOKEN,
    )
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://broker",
        headers={BROKER_AUTH_HEADER: TOKEN},
    )
    async with client:
        yield BrokerSandboxProvider(broker_url="http://broker", token=TOKEN, client=client)


async def test_a_workspace_survives_seed_and_harvest_through_the_broker(
    provider: BrokerSandboxProvider, tmp_path: Path
) -> None:
    """The shape Fabric's container runtime needs: upload a staged workspace, get outputs back."""
    staging = tmp_path / "staging"
    (staging / "nested").mkdir(parents=True)
    (staging / "run.sh").write_text("#!/bin/sh\necho hi\n")
    (staging / "nested" / "data.json").write_text('{"k": 1}')

    handle = await provider.create(SandboxSpec(image=IMAGE, workdir="/work"))
    await provider.upload_dir(handle, staging, "/work")

    harvested = tmp_path / "out"
    await provider.download_dir(handle, "/work", harvested)

    assert (harvested / "run.sh").read_text() == "#!/bin/sh\necho hi\n"
    assert (harvested / "nested" / "data.json").read_text() == '{"k": 1}'


async def test_a_single_file_round_trips(provider: BrokerSandboxProvider, tmp_path: Path) -> None:
    source = tmp_path / "input.bin"
    source.write_bytes(b"\x00\x01binary\xff")
    handle = await provider.create(SandboxSpec(image=IMAGE))

    await provider.upload_file(handle, source, "/work/input.bin")
    target = tmp_path / "fetched" / "input.bin"
    await provider.download_file(handle, "/work/input.bin", target)

    assert target.read_bytes() == b"\x00\x01binary\xff"


async def test_exec_reports_the_command_result_rather_than_raising(provider: BrokerSandboxProvider) -> None:
    # The seam's contract: a non-zero command is a result, not an exception.
    handle = await provider.create(SandboxSpec(image=IMAGE))

    result = await provider.exec(handle, "echo hello", cwd="/work", timeout_s=30)

    assert result.return_code == 0
    assert result.ok


async def test_stdin_is_refused_rather_than_silently_dropped(provider: BrokerSandboxProvider) -> None:
    """`stdin` exists nowhere below the wire, so honouring it would mean changing the command."""
    handle = await provider.create(SandboxSpec(image=IMAGE))

    with pytest.raises(UnsupportedSandboxOperationError) as excinfo:
        await provider.exec(handle, "cat", stdin=b"payload")

    assert "upload_file" in str(excinfo.value), "the message should name the supported alternative"


async def test_a_spec_without_an_image_is_refused_before_the_request(
    provider: BrokerSandboxProvider,
) -> None:
    # The broker evaluates its approved-image policy against this field; a request without one is
    # one it cannot evaluate, so there is no point sending it.
    with pytest.raises(SandboxCreateError, match="image is required"):
        await provider.create(SandboxSpec(image=None))


async def test_an_unapproved_image_surfaces_as_a_create_error(provider: BrokerSandboxProvider) -> None:
    # Policy lives on the trusted side, so this is the broker refusing rather than the client
    # pre-checking. It has to arrive as the seam's own create failure.
    with pytest.raises(SandboxCreateError):
        await provider.create(SandboxSpec(image="registry.example.com/not/approved:1.0"))


async def test_status_of_a_closed_episode_is_a_status_not_an_exception(
    provider: BrokerSandboxProvider,
) -> None:
    handle = await provider.create(SandboxSpec(image=IMAGE))
    await provider.close(handle)

    assert await provider.status(handle) == SandboxStatus.UNKNOWN


async def test_close_is_idempotent(provider: BrokerSandboxProvider) -> None:
    # Callers close in `finally`; a second close racing a reaper must not raise.
    handle = await provider.create(SandboxSpec(image=IMAGE))

    await provider.close(handle)
    await provider.close(handle)


async def test_health_refuses_a_broker_speaking_another_protocol_version() -> None:
    """Better to fail at construction than to 404 from `/dirs` partway through a run."""

    async def _v1_broker(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"protocol_version": "1"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(_v1_broker), base_url="http://broker")
    async with client:
        provider = BrokerSandboxProvider(broker_url="http://broker", token=TOKEN, client=client)

        with pytest.raises(BrokerSandboxError, match="protocol"):
            await provider.health()


async def test_an_archive_member_escaping_the_target_is_refused(
    provider: BrokerSandboxProvider, tmp_path: Path
) -> None:
    """The archive is built inside the episode, which runs untrusted code."""
    import base64
    import io
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="../escaped.txt")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"bad"))

    async def _malicious_broker(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"archive_b64": base64.b64encode(buffer.getvalue()).decode()})

    client = httpx.AsyncClient(transport=httpx.MockTransport(_malicious_broker), base_url="http://broker")
    async with client:
        hostile = BrokerSandboxProvider(broker_url="http://broker", token=TOKEN, client=client)
        handle = await provider.create(SandboxSpec(image=IMAGE))

        with pytest.raises(BrokerSandboxError, match="escapes"):
            await hostile.download_dir(handle, "/work", tmp_path / "out")

    assert not (tmp_path / "escaped.txt").exists(), "nothing may be written outside the target"


def test_the_two_sandbox_status_enums_agree() -> None:
    """Both mirror NeMo-Gym's, and the provider maps between them by value.

    They are declared in different packages, so drift is silent: a member added on one side would
    map to ``UNKNOWN`` on the other rather than failing.
    """
    assert {member.value for member in SandboxStatus} == {member.value for member in BrokerSandboxStatus}
