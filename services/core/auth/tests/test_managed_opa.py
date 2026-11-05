# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real native-OPA lifecycle, transport and fail-closed regression tests."""

import asyncio
import copy
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest
from nhx.core.auth.app.embedded_pdp import engine
from nhx.core.auth.app.embedded_pdp.opa_binary import resolve_opa_binary
from nhx.core.auth.app.embedded_pdp.rego_engine import PolicyEngineError, RegoPolicy


@pytest.fixture(scope="session")
def opa_binary() -> str:
    path = os.environ.get("NEMO_TEST_OPA_BINARY") or os.environ.get("NHX_AUTH_OPA_BINARY_PATH")
    return str(resolve_opa_binary(path))


@pytest.fixture
def policy_data() -> dict:
    return {
        "authz": {
            "principals": {"alice": {"workspaces": {"w": ["Viewer"]}}},
            "roles": {"Viewer": {"permissions": ["models.read"]}, "ServiceSystem": {"permissions": ["*"]}},
            "workspaces": {},
            "endpoints": {
                "/apis/models/v2/workspaces/{workspace_id}/models": {"get": {"permissions": ["models.read"]}}
            },
        }
    }


@pytest.fixture
def policy(opa_binary):
    with RegoPolicy(opa_binary) as runtime:
        yield runtime


def request() -> dict:
    return {"principal_id": "alice", "workspace": "w", "role": "Viewer"}


def test_readiness_requires_data_and_socket_is_private(policy, policy_data):
    assert policy.is_running()
    assert not policy.is_ready()
    with pytest.raises(PolicyEngineError, match="Policy data not loaded"):
        policy.evaluate(request(), 2)
    policy.set_data(policy_data)
    assert policy.is_ready()
    assert stat.S_IMODE(policy._directory.stat().st_mode) == 0o700
    assert stat.S_IMODE((policy._directory / "opa.sock").stat().st_mode) == 0o600
    assert policy.evaluate(request(), 2)[0]["result"]["has_role"] is True


def test_atomic_data_replacement_revokes_removed_grants(policy, policy_data):
    policy.set_data(policy_data)
    assert policy.evaluate(request(), 2)[0]["result"]["has_role"] is True
    policy_data["authz"]["principals"] = {}
    policy.set_data(policy_data)
    assert policy.evaluate(request(), 2)[0]["result"]["has_role"] is False


def test_nul_identity_does_not_alias(policy, policy_data):
    policy.set_data(policy_data)
    assert policy.evaluate({**request(), "principal_id": "alice\0attacker"}, 2)[0]["result"]["has_role"] is False


def test_failed_serialization_invalidates_old_grants(policy, policy_data):
    policy.set_data(policy_data)
    with pytest.raises(PolicyEngineError, match="Unable to load"):
        policy.set_data({"authz": {"bad": float("nan")}})
    assert not policy.is_ready()
    with pytest.raises(PolicyEngineError, match="Policy data not loaded"):
        policy.evaluate(request(), 2)
    policy.set_data(policy_data)
    assert policy.is_ready()


def test_shared_process_supports_concurrent_threads(policy, policy_data):
    policy.set_data(policy_data)
    pid = policy.pid
    with ThreadPoolExecutor(max_workers=8) as executor:
        values = list(executor.map(lambda _: policy.evaluate(request(), 2), range(32)))
    assert all(value[0]["result"]["has_role"] is True for value in values)
    assert policy.pid == pid


def test_crash_is_fail_closed_and_explicit_refresh_recovers(policy, policy_data):
    policy.set_data(policy_data)
    previous_pid = policy.pid
    policy._process.kill()
    policy._process.wait(timeout=5)
    assert not policy.is_ready()
    with pytest.raises(PolicyEngineError, match="not running"):
        policy.evaluate(request(), 2)
    policy.set_data(policy_data)
    assert policy.pid != previous_pid
    assert policy.is_ready()
    assert policy.evaluate(request(), 2)[0]["result"]["has_role"] is True


def test_close_reaps_process_and_never_implicitly_restarts(policy, policy_data):
    policy.set_data(policy_data)
    process, directory = policy._process, policy._directory
    policy.close()
    policy.close()
    assert process.poll() is not None
    assert not directory.exists()
    assert not policy.is_running()
    with pytest.raises(PolicyEngineError, match="closed"):
        policy.set_data(policy_data)
    assert policy.pid is None


def test_inflight_decision_is_rechecked_after_revocation(policy, policy_data, monkeypatch):
    policy.set_data(policy_data)
    original = policy._request_json
    started, release = Event(), Event()

    def delayed(client, method, path, data):
        result = original(client, method, path, data)
        if method == "POST":
            started.set()
            assert release.wait(5)
        return result

    monkeypatch.setattr(policy, "_request_json", delayed)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(policy.evaluate, request(), 2)
        assert started.wait(5)
        changed = copy.deepcopy(policy_data)
        changed["authz"]["principals"] = {}
        try:
            policy.set_data(changed)
        finally:
            release.set()
        result = future.result(timeout=5)
        assert result[0]["result"]["has_role"] is False


def test_retry_cannot_be_invalidated_by_a_second_concurrent_refresh(policy, policy_data, monkeypatch):
    policy.set_data(policy_data)
    original = policy._request_json
    first_started, first_release = Event(), Event()
    retry_started, retry_release = Event(), Event()
    writer_started, writer_finished = Event(), Event()
    calls = []

    def delayed(client, method, path, data):
        result = original(client, method, path, data)
        if method == "POST":
            calls.append(None)
            if len(calls) == 1:
                first_started.set()
                assert first_release.wait(5)
            elif len(calls) == 2:
                retry_started.set()
                assert retry_release.wait(5)
        return result

    def regrant():
        writer_started.set()
        policy.set_data(policy_data)
        writer_finished.set()

    monkeypatch.setattr(policy, "_request_json", delayed)
    with ThreadPoolExecutor(max_workers=2) as executor:
        try:
            decision = executor.submit(policy.evaluate, request(), 2)
            assert first_started.wait(5)
            revoked = copy.deepcopy(policy_data)
            revoked["authz"]["principals"] = {}
            policy.set_data(revoked)
            first_release.set()
            assert retry_started.wait(5)
            refresh = executor.submit(regrant)
            assert writer_started.wait(5)
            # The retry observes the revoked snapshot, not the stale initial
            # grant. A later publication must wait until that decision finishes.
            assert not writer_finished.wait(0.1)
            retry_release.set()
            assert decision.result(timeout=5)[0]["result"]["has_role"] is False
            refresh.result(timeout=5)
        finally:
            first_release.set()
            retry_release.set()
    assert len(calls) == 2
    assert policy.evaluate(request(), 2)[0]["result"]["has_role"] is True


def test_manager_successful_refresh_has_no_separate_invalidation_gap(opa_binary, policy_data, monkeypatch):
    manager = engine._PolicyRuntimeManager()
    monkeypatch.setattr(manager, "_create_policy", lambda: RegoPolicy(opa_binary))
    first_started, first_release, decision_done = Event(), Event(), Event()
    try:
        manager.set_data(policy_data)
        policy = manager.get_policy()
        original_request, original_invalidate = policy._request_json, policy.invalidate_data

        def delayed(client, method, path, data):
            result = original_request(client, method, path, data)
            if method == "POST":
                first_started.set()
                assert first_release.wait(5)
            return result

        def separate_invalidation():
            original_invalidate()
            # Exercise the previous manager's publication gap deterministically:
            # its in-flight query sees unloaded data before set_data can begin.
            first_release.set()
            assert decision_done.wait(5)

        def decide():
            try:
                return policy.evaluate(request(), 2)
            finally:
                decision_done.set()

        monkeypatch.setattr(policy, "_request_json", delayed)
        monkeypatch.setattr(policy, "invalidate_data", separate_invalidation)
        with ThreadPoolExecutor(max_workers=1) as executor:
            decision = executor.submit(decide)
            try:
                assert first_started.wait(5)
                revoked = copy.deepcopy(policy_data)
                revoked["authz"]["principals"] = {}
                manager.set_data(revoked)
            finally:
                first_release.set()
            assert decision.result(timeout=5)[0]["result"]["has_role"] is False
    finally:
        first_release.set()
        manager.close()


@pytest.mark.parametrize("value", [None, True, [], {"has_role": "true"}, {"has_role": 1}])
def test_invalid_decisions_are_rejected(policy, policy_data, monkeypatch, value):
    policy.set_data(policy_data)
    monkeypatch.setattr(policy, "_request_json", lambda *args: {"result": value})
    with pytest.raises(PolicyEngineError):
        policy.evaluate(request(), 2)


def test_epoch_invalidation_rejects_queries(policy, policy_data):
    policy.set_data(policy_data)
    policy.invalidate_data()
    assert not policy.is_ready()
    with pytest.raises(PolicyEngineError, match="Policy data not loaded"):
        policy.evaluate(request(), 2)


def test_child_does_not_inherit_service_credentials(opa_binary, tmp_path, monkeypatch):
    monkeypatch.setenv("MANAGED_OPA_TEST_SECRET", "do-not-pass")
    monkeypatch.setenv("OPA_CONFIG_FILE", "/nonexistent/remote-config.yaml")
    monkeypatch.setenv("SSL_CERT_FILE", "/nonexistent/certificate.pem")
    monkeypatch.setenv("SSL_CERT_DIR", "/nonexistent/certificates")
    (tmp_path / "env.rego").write_text('package authz\nallow := {"allowed": true, "environment": opa.runtime().env}')
    with RegoPolicy(opa_binary, policy_directory=tmp_path) as policy:
        policy.set_data({})
        environment = policy.evaluate({})[0]["result"]["environment"]
    assert "MANAGED_OPA_TEST_SECRET" not in environment
    assert "OPA_CONFIG_FILE" not in environment


def test_invalid_sources_fail_before_daemon_start(opa_binary, tmp_path):
    (tmp_path / "invalid.rego").write_text("not valid rego!")
    with pytest.raises(PolicyEngineError, match="validation failed"):
        RegoPolicy(opa_binary, policy_directory=tmp_path)


def test_startup_timeout_reaps_unready_child(tmp_path, monkeypatch):
    pid_file = tmp_path / "pid"
    executable = tmp_path / "fake-opa"
    executable.write_text(
        f"#!{sys.executable}\nimport os,sys,time\n"
        f"if sys.argv[1] == 'run':\n    open({str(pid_file)!r},'w').write(str(os.getpid()))\n    time.sleep(60)\n"
    )
    executable.chmod(0o700)
    monkeypatch.setattr("nhx.core.auth.app.embedded_pdp.rego_engine.resolve_opa_binary", lambda _: executable)
    with pytest.raises(PolicyEngineError, match="Timed out"):
        RegoPolicy(str(executable), startup_timeout_seconds=0.25)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_manager_has_one_process_and_shutdown_latches(opa_binary, policy_data, monkeypatch):
    manager = engine._PolicyRuntimeManager()
    monkeypatch.setattr(manager, "_create_policy", lambda: RegoPolicy(opa_binary))
    try:
        manager.set_data(policy_data)
        with ThreadPoolExecutor(max_workers=4) as executor:
            policies = list(executor.map(lambda _: manager.get_policy(), range(8)))
        assert all(policy is policies[0] for policy in policies)
        assert manager.is_ready()
        process = policies[0]._process
        assert process is not None
        manager.close()
        assert process.poll() is not None
        with pytest.raises(PolicyEngineError, match="stopped"):
            manager.get_policy()
        assert not manager.is_ready()
        manager.start()
        assert not manager.is_ready()
    finally:
        manager.close()


def test_manager_retains_an_immutable_reload_snapshot(opa_binary, policy_data, monkeypatch):
    manager = engine._PolicyRuntimeManager()
    monkeypatch.setattr(manager, "_create_policy", lambda: RegoPolicy(opa_binary))
    try:
        manager.set_data(policy_data)
        policy_data["authz"]["principals"] = {}
        manager.reload()
        assert manager.get_policy().evaluate(request(), 2)[0]["result"]["has_role"] is True
    finally:
        manager.close()


def test_repeated_epoch_changes_fail_closed_after_one_retry(policy, policy_data, monkeypatch):
    policy.set_data(policy_data)
    calls = []

    def changing(*_args):
        calls.append(None)
        policy._generation += 1
        return {"result": {"has_role": True}}

    monkeypatch.setattr(policy, "_request_json", changing)
    with pytest.raises(PolicyEngineError, match="refusing stale"):
        policy.evaluate(request(), 2)
    assert len(calls) == 2


def test_identical_refresh_does_not_interrupt_queries(opa_binary, policy_data, monkeypatch):
    manager = engine._PolicyRuntimeManager()
    monkeypatch.setattr(manager, "_create_policy", lambda: RegoPolicy(opa_binary))
    try:
        policy_data["metadata"] = {"flag": True}
        manager.set_data(policy_data)
        runtime = manager.get_policy()
        generation = runtime._generation
        manager.set_data(copy.deepcopy(policy_data))
        assert runtime._generation == generation
        # Python considers True == 1; authorization JSON must not conflate them.
        policy_data["metadata"]["flag"] = 1
        manager.set_data(policy_data)
        assert runtime._generation > generation
    finally:
        manager.close()


def test_failed_reload_invalidates_old_runtime(opa_binary, policy_data, monkeypatch):
    manager = engine._PolicyRuntimeManager()
    monkeypatch.setattr(manager, "_create_policy", lambda: RegoPolicy(opa_binary))
    try:
        manager.set_data(policy_data)
        old = manager.get_policy()

        def fail_start():
            raise PolicyEngineError("invalid policy source")

        monkeypatch.setattr(manager, "_create_policy", fail_start)
        with pytest.raises(PolicyEngineError):
            manager.reload()
        assert not manager.is_ready()
        with pytest.raises(PolicyEngineError, match="Policy data not loaded"):
            old.evaluate(request(), 2)
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_service_start_and_stop_own_a_real_child(opa_binary, policy_data, monkeypatch):
    from nhx.core.auth.service import AuthService

    manager = engine._PolicyRuntimeManager()
    monkeypatch.setattr(engine, "_policy_runtime", manager)
    monkeypatch.setattr(manager, "_create_policy", lambda: RegoPolicy(opa_binary))
    monkeypatch.setattr(
        "nhx.core.auth.service.get_service_config",
        lambda _: SimpleNamespace(enabled=True, policy_decision_point_provider="embedded"),
    )
    service = AuthService()
    try:
        await service.on_startup()
        runtime = manager.get_policy()
        process = runtime._process
        assert process is not None
        assert runtime.is_running()
        assert not await service.is_ready()
        await asyncio.to_thread(manager.set_data, policy_data)
        service._policy_refresh_healthy = True
        assert await service.is_ready()
        process.kill()
        await asyncio.to_thread(process.wait, 5)
        assert not await service.is_ready()
    finally:
        await service.on_shutdown()
    assert process.poll() is not None
    assert not manager.is_ready()
    with pytest.raises(PolicyEngineError, match="stopped"):
        manager.get_policy()


def test_cleanup_keeps_ownership_when_post_kill_wait_times_out(policy, monkeypatch):
    process = policy._process
    assert process is not None
    original_wait = process.wait
    calls = []

    def delayed_wait(timeout: float | None = None):
        calls.append(None)
        if len(calls) <= 2:
            assert timeout is not None
            raise subprocess.TimeoutExpired(process.args, timeout)
        return original_wait(timeout=timeout)

    monkeypatch.setattr(process, "wait", delayed_wait)
    with pytest.raises(subprocess.TimeoutExpired):
        policy.close()
    assert policy._process is process
    assert policy._finalizer is not None and policy._finalizer.alive
    policy.close()
    assert process.poll() is not None
    assert policy._process is None


def test_foreign_owner_inspection_never_takes_inherited_lock(policy, monkeypatch):
    class InheritedLock:
        def __enter__(self):
            raise AssertionError("must not acquire another process's lock")

        def __exit__(self, *_args):
            pass

    with monkeypatch.context() as patch:
        patch.setattr(policy, "_owner_pid", os.getpid() + 1)
        patch.setattr(policy, "_lock", InheritedLock())
        assert not policy.is_running()
        assert not policy.is_ready()
        assert policy.pid is None
        policy.close()
    assert policy.is_running()


def test_manager_keeps_cleanup_owner_for_retry(opa_binary, policy_data, monkeypatch):
    manager = engine._PolicyRuntimeManager()
    monkeypatch.setattr(manager, "_create_policy", lambda: RegoPolicy(opa_binary))
    try:
        manager.set_data(policy_data)
        policy = manager.get_policy()
        original_close = policy.close
        calls = []

        def fail_once():
            calls.append(None)
            if len(calls) == 1:
                raise OSError("retry cleanup")
            original_close()

        monkeypatch.setattr(policy, "close", fail_once)
        with pytest.raises(OSError, match="retry cleanup"):
            manager.close()
        assert manager._policy is policy
        with pytest.raises(PolicyEngineError, match="stopped"):
            manager.get_policy()
        manager.close()
        assert manager._policy is None
        assert not policy.is_running()
    finally:
        manager.close()
