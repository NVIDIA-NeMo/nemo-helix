# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline lifecycle and security tests for best-effort job telemetry."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from pathlib import Path
from time import monotonic
from unittest.mock import patch

import nemo_relay
import pytest
from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.fabric.invocation import AgentConfigInvocationRequest, invoke_agent_config_request_once
from nemo_agents_plugin.fabric.progress import LocalProgressObserver
from nemo_agents_plugin.fabric.runtime import FabricRuntimeResult, FabricRuntimeTimeoutError, run_fabric_agent_once
from nemo_agents_plugin.fabric.translator import translate_agent_config
from nemo_fabric import FabricConfig, RelayAtofConfig, RelayAtofFileSinkConfig
from nemo_fabric_adapters.common.utils import normalize_relay_output_dirs


def _config() -> AgentConfig:
    return AgentConfig.model_validate(
        {
            "config_format": "nemo-agents-spec-v1",
            "name": "progress-test",
            "default_harness": "hermes",
            "harnesses": {"hermes": {"kind": "hermes"}},
            "models": {"default": {"provider": "openai", "model": "test"}},
            "telemetry": {
                "enabled": True,
                "output_dir": "./artifacts/relay",
                "atof": {
                    "enabled": True,
                    "filename": "events.atof.jsonl",
                    "mode": "append",
                },
            },
        }
    )


def _atof(config: FabricConfig) -> RelayAtofConfig:
    assert config.relay is not None
    assert config.relay.observability is not None
    atof = config.relay.observability.atof
    assert isinstance(atof, RelayAtofConfig)
    return atof


def _observer(tmp_path: Path) -> LocalProgressObserver:
    return LocalProgressObserver(translate_agent_config(_config()), tmp_path, tmp_path / "artifacts")


def _file(tmp_path: Path) -> Path:
    path = tmp_path / "artifacts/relay/runtime-1/events.atof.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _record(category: str = "tool", phase: str = "start") -> bytes:
    return (
        json.dumps(
            {
                "kind": "scope",
                "category": category,
                "scope_category": phase,
                "name": "secret-name",
                "data": "secret-payload",
                "metadata": "secret-token",
            }
        ).encode()
        + b"\n"
    )


@pytest.mark.asyncio
async def test_real_relay_export_is_observable_before_tool_and_invocation_finish(tmp_path, caplog, monkeypatch):
    """No force_flush/shutdown/parent end is needed to see SDK file output."""
    caplog.set_level(logging.INFO)
    monkeypatch.setattr("nemo_agents_plugin.fabric.progress._REPORT_SECONDS", 0.05)
    config = _config()
    translated = translate_agent_config(config)
    payload = _atof(translated).model_dump()
    plugin = {"components": [{"kind": "observability", "config": {"version": 4, "atof": payload}}]}
    # Exercise the installed adapter's actual runtime directory normalization.
    normalize_relay_output_dirs(plugin, {"base_dir": str(tmp_path), "runtime_context": {"runtime_id": "runtime-1"}})
    sink = plugin["components"][0]["config"]["atof"]["sinks"][0]
    assert Path(sink["output_directory"]) == _file(tmp_path).parent
    exporter_config = nemo_relay.AtofExporterConfig()
    exporter_config.output_directory = sink["output_directory"]
    exporter_config.filename = sink["filename"]
    exporter = nemo_relay.AtofExporter(exporter_config)
    exporter.register("job-progress-test")
    terminal = FabricRuntimeResult(status="succeeded", response="authoritative", artifacts={"saved": True})

    async def invoke(request):
        assert request.timeout_seconds == 10
        with nemo_relay.scope.scope("secret-agent", nemo_relay.ScopeType.Agent):
            tool = nemo_relay.tools.call("secret-name", {"secret": "secret-payload"})
            try:
                async with asyncio.timeout(3):
                    while "tool_start=1" not in caplog.text:
                        await asyncio.sleep(0.01)
                assert "tool_end=0" in caplog.text
                assert "secret-name" not in caplog.text
                assert "secret-payload" not in caplog.text
            finally:
                nemo_relay.tools.call_end(tool, nemo_relay.ToolExecutionResult({"secret": "secret-result"}))
        return terminal

    try:
        with patch("nemo_agents_plugin.fabric.invocation.run_fabric_agent_once", invoke):
            result = await invoke_agent_config_request_once(
                AgentConfigInvocationRequest(
                    agent_config=config,
                    input="private prompt",
                    base_dir=tmp_path,
                    timeout_seconds=10,
                    observe_progress=True,
                )
            )
        assert result is terminal
        assert "secret-result" not in caplog.text
    finally:
        exporter.deregister("job-progress-test")
        exporter.shutdown()


@pytest.mark.asyncio
async def test_symlinked_base_ancestor_keeps_progress_enabled(tmp_path, caplog):
    """Normalize trusted job storage without allowing symlinked telemetry sinks."""
    caplog.set_level(logging.INFO)
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    base = alias / "fabric"
    terminal = FabricRuntimeResult(status="succeeded")

    async def invoke(request):
        path = _file(real / "fabric")
        path.write_bytes(_record())
        async with asyncio.timeout(2):
            while "tool_start=1" not in caplog.text:
                await asyncio.sleep(0.01)
        return terminal

    with patch("nemo_agents_plugin.fabric.invocation.run_fabric_agent_once", invoke):
        result = await invoke_agent_config_request_once(
            AgentConfigInvocationRequest(
                agent_config=_config(),
                input="private",
                base_dir=base,
                observe_progress=True,
            )
        )
    assert result is terminal
    assert "requires a local ATOF file sink" not in caplog.text
    assert "secret" not in caplog.text


@pytest.mark.asyncio
async def test_normal_observer_shutdown_releases_worker(tmp_path, caplog):
    """Cancelling between polls wakes the worker rather than waiting a poll interval."""
    entered = threading.Event()
    threads = []

    def poll(observer):
        threads.append(threading.current_thread())
        entered.set()

    with patch.object(LocalProgressObserver, "poll", poll):
        task = asyncio.create_task(_observer(tmp_path).run())
        async with asyncio.timeout(1):
            while not entered.is_set():
                await asyncio.sleep(0.001)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    for thread in threads:
        thread.join(timeout=0.1)
    assert len(threads) == 1 and not threads[0].is_alive()
    assert "shutdown timed out" not in caplog.text


@pytest.mark.parametrize("outcome", ["result", "failure", "timeout", "cancel"])
def test_stalled_poll_does_not_delay_invocation_or_asyncio_run_shutdown(tmp_path, caplog, outcome):
    """An in-flight filesystem poll must not join the default executor at exit."""
    caplog.set_level(logging.INFO)
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    threads = []
    original_poll = LocalProgressObserver.poll
    terminal = FabricRuntimeResult(status="succeeded")
    failure = RuntimeError("private invocation error")
    calls = 0
    _file(tmp_path).write_bytes(_record())

    def stalled_poll(observer):
        threads.append(threading.current_thread())
        entered.set()
        try:
            # Self-release on regressions so a default-executor shutdown cannot
            # hang pytest indefinitely. Correct code exits before this expires.
            release.wait(2)
            original_poll(observer)
        finally:
            finished.set()

    async def invoke(request):
        nonlocal calls
        calls += 1
        async with asyncio.timeout(0.5):
            while not entered.is_set():
                await asyncio.sleep(0.001)
        if outcome == "failure":
            raise failure
        if outcome == "timeout":
            try:
                await asyncio.wait_for(asyncio.Future(), timeout=0.02)
            except TimeoutError as error:
                raise FabricRuntimeTimeoutError("invocation timed out") from error
        if outcome == "cancel":
            raise asyncio.CancelledError()
        return terminal

    async def run():
        expected = {"failure": RuntimeError, "timeout": FabricRuntimeTimeoutError, "cancel": asyncio.CancelledError}
        request = AgentConfigInvocationRequest(
            agent_config=_config(),
            input="private",
            base_dir=tmp_path,
            observe_progress=True,
        )
        if outcome == "result":
            assert await invoke_agent_config_request_once(request) is terminal
        else:
            with pytest.raises(expected[outcome]) as caught:
                await invoke_agent_config_request_once(request)
            if outcome == "failure":
                assert caught.value is failure

    try:
        with (
            patch.object(LocalProgressObserver, "poll", stalled_poll),
            patch(
                "nemo_agents_plugin.fabric.invocation.run_fabric_agent_once",
                invoke,
            ),
        ):
            started = monotonic()
            asyncio.run(run())
            assert monotonic() - started < 1
            assert entered.is_set() and not finished.is_set()
            assert calls == 1
            assert len(threads) == 1 and threads[0].daemon
            assert "observer shutdown timed out" in caplog.text
            assert "private invocation error" not in caplog.text
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=1)
        assert all(not thread.is_alive() for thread in threads)
    # A resumed worker must not emit lifecycle records after invocation exit.
    assert "Agent telemetry progress" not in caplog.text


def test_fragmented_oversize_malformed_and_payload_records(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    observer = _observer(tmp_path)
    path = _file(tmp_path)
    record = _record()
    path.write_bytes(record[:10])
    observer.poll()
    assert not observer.counts
    with path.open("ab") as stream:
        stream.write(record[10:] + b"secret-malformed\n" + b"x" * 20000)
    observer.poll()
    with path.open("ab") as stream:
        stream.write(b"x" * 20000 + b"\n" + _record("tool", "end") + _record("llm"))
    observer.poll()
    assert observer.counts["tool_start"] == observer.counts["tool_end"] == observer.counts["llm_start"] == 1
    assert observer.counts["discarded"] >= 2
    assert "secret" not in caplog.text
    assert "last_event_age=" in caplog.text
    assert "succeeded" not in caplog.text  # An end record does not imply success.


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["file", "directory", "fifo"])
async def test_unsafe_file_types_fail_closed_without_content(tmp_path, caplog, target):
    path = _file(tmp_path)
    secret = tmp_path / "secret"
    secret.write_bytes(_record())
    if target == "file":
        path.symlink_to(secret)
    elif target == "directory":
        path.parent.rmdir()
        path.parent.symlink_to(tmp_path, target_is_directory=True)
    else:
        os.mkfifo(path)
    observer = _observer(tmp_path)
    task = asyncio.create_task(observer.run())
    await asyncio.sleep(0.02)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert observer.last_event is None
    assert "secret" not in caplog.text
    if target != "directory":
        assert "observer failed" in caplog.text


def test_paths_outside_artifacts_and_symlinks_are_not_read(tmp_path):
    config = translate_agent_config(_config())
    sinks = _atof(config).sinks
    assert sinks is not None
    sink = sinks[0]
    assert isinstance(sink, RelayAtofFileSinkConfig)
    sink.output_directory = "../outside"
    assert not LocalProgressObserver(config, tmp_path, tmp_path / "artifacts").sinks
    sink.output_directory = str(tmp_path / "other")
    assert not LocalProgressObserver(config, tmp_path, tmp_path / "artifacts").sinks
    _file(tmp_path)
    (tmp_path / "artifacts/relay/runtime-1").rmdir()
    (tmp_path / "artifacts/relay").rmdir()
    (tmp_path / "artifacts/relay").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(OSError):
        _observer(tmp_path).poll()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [RuntimeError("private failure"), FabricRuntimeTimeoutError("timed out"), asyncio.CancelledError()]
)
async def test_observer_never_retries_or_masks_invocation_error(tmp_path, error):
    calls = 0

    async def invoke(request):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        raise error

    with patch("nemo_agents_plugin.fabric.invocation.run_fabric_agent_once", invoke):
        with pytest.raises(type(error)) as caught:
            await invoke_agent_config_request_once(
                AgentConfigInvocationRequest(
                    agent_config=_config(),
                    input="secret",
                    base_dir=tmp_path,
                    observe_progress=True,
                )
            )
    assert caught.value is error
    assert calls == 1
    assert not [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and "LocalProgressObserver.run" in repr(task)
    ]


@pytest.mark.asyncio
async def test_observer_failure_does_not_replace_result(tmp_path, caplog):
    terminal = FabricRuntimeResult(status="succeeded")

    async def invoke(request):
        await asyncio.sleep(0.01)
        return terminal

    with (
        patch("nemo_agents_plugin.fabric.progress.LocalProgressObserver.poll", side_effect=OSError("secret")),
        patch(
            "nemo_agents_plugin.fabric.invocation.run_fabric_agent_once",
            invoke,
        ),
    ):
        result = await invoke_agent_config_request_once(
            AgentConfigInvocationRequest(
                agent_config=_config(),
                input="secret",
                base_dir=tmp_path,
                observe_progress=True,
            )
        )
    assert result is terminal
    assert "observer failed" in caplog.text
    assert "secret" not in caplog.text


@pytest.mark.asyncio
async def test_real_nonstreaming_deadline_cancels_invocation_and_closes_runtime(tmp_path):
    cancelled = asyncio.Event()
    closed = asyncio.Event()

    class Runtime:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            closed.set()

        async def invoke(self, **kwargs):
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

    class Fabric:
        async def start_runtime(self, *args, **kwargs):
            assert kwargs["streaming"] is False
            return Runtime()

    async def invoke(request):
        return await run_fabric_agent_once(request, fabric=Fabric())

    with patch("nemo_agents_plugin.fabric.invocation.run_fabric_agent_once", invoke):
        async with asyncio.timeout(1):
            with pytest.raises(FabricRuntimeTimeoutError):
                await invoke_agent_config_request_once(
                    AgentConfigInvocationRequest(
                        agent_config=_config(),
                        input="secret",
                        base_dir=tmp_path,
                        timeout_seconds=0.02,
                        observe_progress=True,
                    )
                )
    assert cancelled.is_set()
    assert closed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("disabled", [True, False])
async def test_initial_unavailable_notice_distinguishes_disabled_from_missing_sink(tmp_path, caplog, disabled):
    caplog.set_level(logging.INFO)
    config = _config()
    config.telemetry.enabled = not disabled
    translated = translate_agent_config(config)
    if not disabled:
        _atof(translated).sinks = []
    await LocalProgressObserver(translated, tmp_path, tmp_path / "artifacts").run()
    if disabled:
        assert "disabled or unconfigured" in caplog.text
    else:
        assert "requires a local ATOF file sink" in caplog.text
    assert "No live agent telemetry observed" not in caplog.text


def test_absent_file_warning_and_event_age(tmp_path, caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    observer = _observer(tmp_path)
    now = observer.started + 31
    monkeypatch.setattr("nemo_agents_plugin.fabric.progress.monotonic", lambda: now)
    observer.poll()
    observer.poll()
    assert caplog.text.count("No live agent telemetry observed") == 1
    _file(tmp_path).write_bytes(_record("llm"))
    observer.poll()
    now += 10
    observer.poll()
    assert "last_event_age=10.0s" in caplog.text
    assert "llm_start=1 llm_end=0" in caplog.text


def test_volume_memory_and_file_count_are_bounded(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    observer = _observer(tmp_path)
    for index in range(10):
        path = tmp_path / f"artifacts/relay/runtime-{index}/events.atof.jsonl"
        path.parent.mkdir(parents=True)
        path.write_bytes(_record() * 1000)
    # Duplicate sinks must not multiply the per-file/per-poll byte budget.
    observer.sinks *= 4
    with patch("nemo_agents_plugin.fabric.progress.os.read", wraps=os.read) as read:
        observer.poll()
    assert read.call_count == 4
    assert all(call.args[1] == 65536 for call in read.call_args_list)
    for _ in range(9):
        observer.poll()
    assert len(observer.offsets) <= 4
    assert all(len(buffer) <= 16384 for buffer in observer.buffers.values())
    assert len([record for record in caplog.records if "Agent telemetry progress" in record.message]) == 1
