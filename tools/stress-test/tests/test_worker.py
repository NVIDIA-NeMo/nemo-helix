# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from worker.hammer import OPERATIONS, run_hammer
from worker.history import FLUSH_SECONDS, HistoryLog, classify


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeCalls:
    def __init__(self) -> None:
        self.cleaned = False

    def fileset_create(self, name: str) -> int:
        return 200

    def fileset_get(self, name: str) -> int:
        return 200

    def fileset_list(self) -> int:
        return 200

    def fileset_delete(self, name: str) -> int:
        return 200

    def file_upload(self, name: str, payload: bytes) -> int:
        return 200

    def file_download(self, name: str) -> int:
        return 200

    def file_list(self, name: str) -> int:
        return 200

    def file_delete(self, name: str) -> int:
        return 200

    def secret_create(self, name: str) -> int:
        return 200

    def secret_get(self, name: str) -> int:
        return 200

    def secret_list(self) -> int:
        return 200

    def secret_update(self, name: str) -> int:
        return 200

    def secret_delete(self, name: str) -> int:
        return 200

    def secret_value(self, name: str) -> int:
        return 403

    def discovery(self) -> int:
        return 200

    def authz(self) -> int:
        return 200

    def access_key_create(self) -> tuple[int, str]:
        return 201, "secret-fake-access-key"

    def access_key_list(self) -> int:
        return 200

    def access_key_revoke(self, jti: str) -> int:
        return 200

    def cleanup(self) -> None:
        self.cleaned = True


def test_secret_value_denial_is_expected() -> None:
    assert classify("secrets", "value_read", 403) == "expected_denial"
    assert classify("files", "file_delete", 403) == "result"


def test_history_flushes_within_30_seconds(tmp_path: Path) -> None:
    assert FLUSH_SECONDS <= 30
    clock = Clock()
    path = tmp_path / "service-stress.jsonl"
    log = HistoryLog(path, clock=clock)
    log.write({"record": "operation", "status": 200})
    clock.now = FLUSH_SECONDS
    log.write({"record": "operation", "status": 201})
    assert '"status": 200' in path.read_text()
    log.close()


def test_access_key_secret_is_absent_from_history(tmp_path: Path) -> None:
    clock = Clock()
    log = HistoryLog(tmp_path / "service-stress.jsonl", clock=clock)
    calls = FakeCalls()
    run_hammer(
        service="auth",
        duration=0.05,
        rate=100,
        payload=b"x",
        job="stress-auth-01-1",
        step="auth-01",
        clock=clock,
        sleep=clock.sleep,
        log=log,
        calls=calls,
    )
    log.close()
    text = (tmp_path / "service-stress.jsonl").read_text()
    assert "secret-fake-access-key" not in text
    assert calls.cleaned
    assert "access_key_create" in text


def test_secret_denial_is_recorded_as_a_sample(tmp_path: Path) -> None:
    clock = Clock()
    log = HistoryLog(tmp_path / "service-stress.jsonl", clock=clock)
    run_hammer(
        service="secrets",
        duration=0.07,
        rate=100,
        payload=b"x",
        job="stress-secrets-01-1",
        step="secrets-01",
        clock=clock,
        sleep=clock.sleep,
        log=log,
        calls=FakeCalls(),
    )
    log.close()
    text = (tmp_path / "service-stress.jsonl").read_text()
    assert "expected_denial" in text
    assert set(OPERATIONS["secrets"]) <= {"create", "get", "list", "update", "delete", "value_read"}


class RecordingCalls(FakeCalls):
    """Records the object name each named call receives, in call order."""

    def __init__(self) -> None:
        super().__init__()
        self.named: list[tuple[str, str]] = []

    def _record(self, operation: str, name: str) -> int:
        self.named.append((operation, name))
        return 200

    def fileset_create(self, name: str) -> int:
        return self._record("fileset_create", name)

    def fileset_get(self, name: str) -> int:
        return self._record("fileset_get", name)

    def fileset_delete(self, name: str) -> int:
        return self._record("fileset_delete", name)

    def file_upload(self, name: str, payload: bytes) -> int:
        return self._record("file_upload", name)

    def file_download(self, name: str) -> int:
        return self._record("file_download", name)

    def file_list(self, name: str) -> int:
        return self._record("file_list", name)

    def file_delete(self, name: str) -> int:
        return self._record("file_delete", name)

    def secret_create(self, name: str) -> int:
        return self._record("create", name)

    def secret_get(self, name: str) -> int:
        return self._record("get", name)

    def secret_update(self, name: str) -> int:
        return self._record("update", name)

    def secret_delete(self, name: str) -> int:
        return self._record("delete", name)

    def secret_value(self, name: str) -> int:
        self._record("value_read", name)
        return 403


def _hammer_cycles(service: str, cycles: int, tmp_path: Path) -> list[tuple[str, str]]:
    """Run exactly the given number of full passes through the service's operations."""
    clock = Clock()
    log = HistoryLog(tmp_path / "service-stress.jsonl", clock=clock)
    calls = RecordingCalls()
    run_hammer(
        service=service,
        duration=len(OPERATIONS[service]) * cycles * 0.01 - 0.005,
        rate=100,
        payload=b"x",
        job="stress-job-1",
        step="step-01",
        clock=clock,
        sleep=clock.sleep,
        log=log,
        calls=calls,
    )
    log.close()
    return calls.named


def test_a_secret_cycle_works_on_one_object_and_reads_it_before_deleting_it(tmp_path: Path) -> None:
    named = _hammer_cycles("secrets", 2, tmp_path)
    first, second = named[:5], named[5:]
    assert [operation for operation, _name in first] == ["create", "get", "update", "value_read", "delete"]
    assert len({name for _operation, name in first}) == 1
    assert len({name for _operation, name in second}) == 1
    assert first[0][1] != second[0][1]


def test_a_files_cycle_works_on_one_fileset_and_deletes_it_last(tmp_path: Path) -> None:
    named = _hammer_cycles("files", 2, tmp_path)
    first, second = named[:7], named[7:]
    expected = [
        "fileset_create",
        "fileset_get",
        "file_upload",
        "file_download",
        "file_list",
        "file_delete",
        "fileset_delete",
    ]
    assert [operation for operation, _name in first] == expected
    assert len({name for _operation, name in first}) == 1
    assert len({name for _operation, name in second}) == 1
    assert first[0][1] != second[0][1]


def test_worker_source_does_not_call_role_bindings() -> None:
    source = Path(__file__).resolve().parents[1].joinpath("worker", "hammer.py").read_text()
    assert "role_binding" not in source
