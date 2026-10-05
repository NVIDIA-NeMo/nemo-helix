# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from nemo_helix_ext.cli.telemetry import emit as emit_mod
from nemo_helix_ext.cli.telemetry import session as session_mod
from nemo_helix_ext.cli.telemetry.events import CommandInvokedEvent, TaskStatusEnum


@pytest.fixture(autouse=True)
def _isolate_local_telemetry_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    session_mod._cached_state = None
    monkeypatch.delenv("NEMO_TELEMETRY_ENABLED", raising=False)
    monkeypatch.setenv("NHX_CONFIG_FILE", str(tmp_path / "config.yaml"))
    monkeypatch.setattr(session_mod, "_session_state_path", lambda: tmp_path / "telemetry-state.json")
    yield
    session_mod._cached_state = None


def _event():
    return CommandInvokedEvent(command="docs", task_status=TaskStatusEnum.COMPLETED, duration_sec=0.1)


class TestOptOutLayers:
    def test_env_layer(self, monkeypatch):
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "false")
        assert emit_mod.telemetry_opted_in() is False

    def test_config_layer(self, monkeypatch, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("telemetry_enabled: false\n")
        monkeypatch.setenv("NHX_CONFIG_FILE", str(cfg))
        assert emit_mod.telemetry_opted_in() is False

    def test_invocation_flag_layer(self):
        emit_mod.set_invocation_opt_out(True)
        try:
            assert emit_mod.telemetry_opted_in() is False
        finally:
            emit_mod.set_invocation_opt_out(False)

    def test_default_is_on(self):
        assert emit_mod.telemetry_opted_in() is True

    def test_config_load_error_fails_closed(self, monkeypatch):
        """A broken/parse-error config must fail closed (opted out), not default to on."""
        from nemo_helix_ext.config import config as config_mod

        def boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("broken config")

        monkeypatch.setattr(config_mod.Config, "load", boom)
        assert emit_mod.telemetry_opted_in() is False


class TestEmitEvent:
    @patch.object(emit_mod, "TelemetryHandler")
    def test_emit_enqueues_and_stops(self, handler_cls):
        instance = Mock()
        handler_cls.return_value = instance
        emit_mod.emit_event(_event())
        instance.enqueue.assert_called_once()
        instance.stop.assert_called_once()

    @patch.object(emit_mod, "TelemetryHandler")
    def test_emit_skipped_when_opted_out(self, handler_cls, monkeypatch):
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "0")
        emit_mod.emit_event(_event())
        handler_cls.assert_not_called()

    @patch.object(emit_mod, "TelemetryHandler", side_effect=RuntimeError("boom"))
    def test_emit_never_raises(self, _handler_cls):
        emit_mod.emit_event(_event())  # must not raise

    @patch.object(emit_mod, "TelemetryHandler")
    def test_emit_suppresses_keyboard_interrupt_from_handler_stop(self, handler_cls):
        instance = Mock()
        instance.stop.side_effect = KeyboardInterrupt
        handler_cls.return_value = instance

        emit_mod.emit_event(_event())

        instance.enqueue.assert_called_once()
        instance.stop.assert_called_once()

    @patch.object(emit_mod, "TelemetryHandler")
    def test_emit_suppresses_cancelled_error_from_handler_stop(self, handler_cls):
        instance = Mock()
        instance.stop.side_effect = asyncio.CancelledError
        handler_cls.return_value = instance

        emit_mod.emit_event(_event())

        instance.enqueue.assert_called_once()
        instance.stop.assert_called_once()

    @patch.object(emit_mod, "TelemetryHandler")
    def test_session_id_is_stable_across_calls(self, handler_cls):
        """Every event in one process shares the cached session id."""
        session_mod._cached_state = None
        emit_mod.emit_event(_event())
        emit_mod.emit_event(_event())
        session_ids = [call.kwargs["session_id"] for call in handler_cls.call_args_list]
        assert len(session_ids) == 2
        assert session_ids[0] == session_ids[1]


class TestTelemetrySessionState:
    def _use_state_path(self, monkeypatch, tmp_path):
        session_mod._cached_state = None
        path = tmp_path / "telemetry-state.json"
        monkeypatch.setattr(session_mod, "_session_state_path", lambda: path)
        return path

    def _set_now(self, monkeypatch, now):
        monkeypatch.setattr(session_mod, "_now_utc", lambda: now)

    def _set_next_uuid(self, monkeypatch, value: str):
        monkeypatch.setattr(session_mod.uuid, "uuid4", lambda: uuid.UUID(value))

    def test_missing_state_creates_session_id_with_creation_date(self, monkeypatch, tmp_path):
        path = self._use_state_path(monkeypatch, tmp_path)
        now = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
        self._set_now(monkeypatch, now)
        self._set_next_uuid(monkeypatch, "11111111-1111-4111-8111-111111111111")

        session_id = session_mod.get_session_id()

        assert session_id == "11111111111141118111111111111111"
        assert json.loads(path.read_text()) == {
            "created_at": "2026-07-27T12:00:00Z",
            "session_id": session_id,
        }

    def test_recent_state_is_reused(self, monkeypatch, tmp_path):
        path = self._use_state_path(monkeypatch, tmp_path)
        path.write_text(
            json.dumps({"session_id": "existing", "created_at": "2026-07-01T00:00:00Z"}),
            encoding="utf-8",
        )
        self._set_now(monkeypatch, datetime(2026, 7, 27, 0, 0, tzinfo=timezone.utc))

        assert session_mod.get_session_id() == "existing"
        assert json.loads(path.read_text())["session_id"] == "existing"

    def test_state_rotates_after_thirty_days(self, monkeypatch, tmp_path):
        path = self._use_state_path(monkeypatch, tmp_path)
        now = datetime(2026, 7, 27, 0, 0, tzinfo=timezone.utc)
        path.write_text(
            json.dumps(
                {
                    "session_id": "expired",
                    "created_at": session_mod._format_created_at(now - timedelta(days=30)),
                }
            ),
            encoding="utf-8",
        )
        self._set_now(monkeypatch, now)
        self._set_next_uuid(monkeypatch, "22222222-2222-4222-8222-222222222222")

        session_id = session_mod.get_session_id()

        assert session_id == "22222222222242228222222222222222"
        state = json.loads(path.read_text())
        assert state["session_id"] == session_id
        assert state["created_at"] == "2026-07-27T00:00:00Z"

    def test_state_without_creation_date_rotates(self, monkeypatch, tmp_path):
        path = self._use_state_path(monkeypatch, tmp_path)
        now = datetime(2026, 7, 27, 0, 0, tzinfo=timezone.utc)
        path.write_text(json.dumps({"session_id": "dateless"}), encoding="utf-8")
        self._set_now(monkeypatch, now)
        self._set_next_uuid(monkeypatch, "33333333-3333-4333-8333-333333333333")

        assert session_mod.get_session_id() == "33333333333343338333333333333333"

    def test_malformed_state_rotates(self, monkeypatch, tmp_path):
        path = self._use_state_path(monkeypatch, tmp_path)
        path.write_text("not json", encoding="utf-8")
        self._set_now(monkeypatch, datetime(2026, 7, 27, 0, 0, tzinfo=timezone.utc))
        self._set_next_uuid(monkeypatch, "55555555-5555-4555-8555-555555555555")

        session_id = session_mod.get_session_id()

        assert session_id == "55555555555545558555555555555555"
        assert json.loads(path.read_text())["session_id"] == session_id

    def test_future_dated_state_rotates(self, monkeypatch, tmp_path):
        path = self._use_state_path(monkeypatch, tmp_path)
        now = datetime(2026, 7, 27, 0, 0, tzinfo=timezone.utc)
        path.write_text(
            json.dumps({"session_id": "future", "created_at": "2026-08-01T00:00:00Z"}),
            encoding="utf-8",
        )
        self._set_now(monkeypatch, now)
        self._set_next_uuid(monkeypatch, "44444444-4444-4444-8444-444444444444")

        assert session_mod.get_session_id() == "44444444444444448444444444444444"


class TestFirstRunNotice:
    def test_notice_printed_once_to_stderr(self, capsys, tmp_path, monkeypatch):
        monkeypatch.setattr(emit_mod, "_notice_marker_path", lambda: tmp_path / "telemetry-notice-shown")
        emit_mod.maybe_print_first_run_notice()
        first = capsys.readouterr()
        assert "anonymous usage data" in first.err
        assert first.out == ""  # stderr only; stdout stays machine-clean
        emit_mod.maybe_print_first_run_notice()
        assert capsys.readouterr().err == ""

    def test_no_notice_when_opted_out(self, capsys, tmp_path, monkeypatch):
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "false")
        monkeypatch.setattr(emit_mod, "_notice_marker_path", lambda: tmp_path / "telemetry-notice-shown")
        emit_mod.maybe_print_first_run_notice()
        assert capsys.readouterr().err == ""

    def test_notice_not_printed_when_marker_write_fails(self, capsys, tmp_path, monkeypatch):
        marker_parent = tmp_path / "not-a-directory"
        marker_parent.write_text("already a file")
        monkeypatch.setattr(emit_mod, "_notice_marker_path", lambda: marker_parent / "telemetry-notice-shown")

        emit_mod.maybe_print_first_run_notice()

        assert capsys.readouterr().err == ""
