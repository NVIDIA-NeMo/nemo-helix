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

    def file_list(self) -> int:
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


def test_worker_source_does_not_call_role_bindings() -> None:
    source = Path(__file__).resolve().parents[1].joinpath("worker", "hammer.py").read_text()
    assert "role_binding" not in source
