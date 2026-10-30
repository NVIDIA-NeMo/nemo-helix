# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nhx.common.auth import Principal
from nhx.common.auth.models import NHX_PRINCIPAL_ENVVAR
from nhx.common.jobs.constants import NEMO_JOB_SECRETS_ENVVAR
from nhx.core.jobs.controllers.backends.subprocess_runtime import (
    DEFAULT_LOG_EXPORT_INTERVAL_MS,
    DEFAULT_LOG_EXPORT_MAX_BATCH_SIZE,
    MAX_LOG_EXPORT_MAX_BATCH_SIZE,
    NHX_JOB_LAUNCHER_OTLP_LOGS_SOCKET_PATH_ENVVAR,
    NHX_JOB_LAUNCHER_OTLP_LOGS_TRANSPORT_ENVVAR,
    NHX_JOB_LOG_EXPORT_INTERVAL_MS_ENVVAR,
    NHX_JOB_LOG_EXPORT_MAX_BATCH_SIZE_ENVVAR,
    SERVICE_JOBS_BEARER_HEADERS,
    SubprocessOtelLogger,
    _build_otlp_log_exporter,
    _UnixSocketOTLPSession,
    create_otel_logger,
    inject_secret_env_vars,
    parse_secret_references,
    start_log_capture,
)
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor


def test_parse_secret_references():
    references = parse_secret_references("HF_TOKEN=default/hf-token, WANDB_API_KEY=shared/wandb-key")

    assert [(ref.env_var_name, ref.workspace, ref.secret_name) for ref in references] == [
        ("HF_TOKEN", "default", "hf-token"),
        ("WANDB_API_KEY", "shared", "wandb-key"),
    ]


def test_parse_secret_references_rejects_invalid_format():
    with pytest.raises(ValueError, match="invalid secret reference format"):
        parse_secret_references("bad-format")


def test_parse_secret_references_rejects_whitespace_only_secret_name():
    with pytest.raises(ValueError, match="invalid secret reference format"):
        parse_secret_references("HF_TOKEN=default/   ")


def test_inject_secret_env_vars():
    principal = Principal(
        id="creator@example.com",
        account_id="account-creator",
        authz_aliases=["legacy-creator", "creator@example.com"],
    )
    env = {
        NEMO_JOB_SECRETS_ENVVAR: "HF_TOKEN=default/hf-token",
        NHX_PRINCIPAL_ENVVAR: principal.model_dump_json(exclude_none=True),
        "NHX_SECRETS_URL": "http://secrets.example",
    }

    response = MagicMock()
    response.read.return_value = json.dumps({"value": "secret-value"}).encode("utf-8")
    response.__enter__.return_value = response
    response.__exit__.return_value = None

    with patch("nhx.core.jobs.controllers.backends.subprocess_runtime.urlopen", return_value=response) as mock_urlopen:
        updated_env = inject_secret_env_vars(env.copy())

    assert updated_env["HF_TOKEN"] == "secret-value"
    request = mock_urlopen.call_args.args[0]
    assert request.full_url == "http://secrets.example/apis/secrets/v2/workspaces/default/secrets/hf-token/access"
    assert request.get_header("X-nhx-principal-id") == "service:jobs"
    assert request.get_header("X-nhx-actor-aliases") == "service:jobs"
    assert request.get_header("X-nhx-principal-on-behalf-of") == "creator@example.com"
    assert request.get_header("X-nhx-subject-account-id") == "account-creator"
    assert request.get_header("X-nhx-subject-aliases") == "legacy-creator,creator@example.com"


def test_inject_secret_env_vars_rejects_missing_value_field():
    env = {
        NEMO_JOB_SECRETS_ENVVAR: "HF_TOKEN=default/hf-token",
        "NHX_SECRETS_URL": "http://secrets.example",
    }
    response = MagicMock()
    response.read.return_value = json.dumps({"data": "secret-value"}).encode("utf-8")
    response.__enter__.return_value = response
    response.__exit__.return_value = None

    with patch("nhx.core.jobs.controllers.backends.subprocess_runtime.urlopen", return_value=response):
        with pytest.raises(RuntimeError, match="missing string value field"):
            inject_secret_env_vars(env.copy())


def test_inject_secret_env_vars_rejects_non_http_secrets_url():
    env = {
        NEMO_JOB_SECRETS_ENVVAR: "HF_TOKEN=default/hf-token",
        "NHX_SECRETS_URL": "file:///tmp/secrets",
    }

    with pytest.raises(ValueError, match="absolute http or https URL"):
        inject_secret_env_vars(env)


def test_start_log_capture_writes_file_and_streams_to_otel(tmp_path):
    pipe = StringIO("hello\nproblem\n")
    log_path = tmp_path / "task.log.jsonl"
    mock_otel_logger = MagicMock()
    mock_provider = MagicMock()

    thread = start_log_capture(
        pipe,
        log_path=log_path,
        log_lock=threading.Lock(),
        otel_logger=SubprocessOtelLogger(mock_otel_logger, mock_provider),
        stream_name="stderr",
        job="job-1",
        step="step-1",
        task_id="task-1",
    )

    assert thread is not None
    thread.join(timeout=2)
    assert "[stderr] hello" in log_path.read_text(encoding="utf-8")
    assert mock_otel_logger.emit.call_count == 2
    assert mock_otel_logger.emit.call_args.kwargs["body"] == "[stderr] problem"
    assert mock_otel_logger.emit.call_args.kwargs["severity_text"] == "ERROR"
    assert mock_otel_logger.emit.call_args.kwargs["attributes"] == {"stream": "stderr"}


def test_local_otel_logger_close_flushes_and_shuts_down():
    mock_otel_logger = MagicMock()
    mock_provider = MagicMock()
    otel_logger = SubprocessOtelLogger(mock_otel_logger, mock_provider)

    otel_logger.emit("hello", "stdout")
    otel_logger.close()

    mock_otel_logger.emit.assert_called_once()
    mock_provider.force_flush.assert_called_once()
    mock_provider.shutdown.assert_called_once()


def test_build_otlp_log_exporter_keeps_default_http_exporter_for_tcp():
    with patch("nhx.core.jobs.controllers.backends.subprocess_runtime.OTLPLogExporter") as exporter:
        result = _build_otlp_log_exporter({}, "http://files.example/otlp/v1/logs")

    assert result is exporter.return_value
    exporter.assert_called_once_with(
        endpoint="http://files.example/otlp/v1/logs",
        headers=SERVICE_JOBS_BEARER_HEADERS,
    )


def test_build_otlp_log_exporter_uses_workload_identity_headers_when_configured(tmp_path):
    subject_token_file = tmp_path / "subject.jwt"
    subject_token_file.write_text("subject-token", encoding="utf-8")
    env = {
        WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR: str(subject_token_file),
        "NHX_AUTH_URL": "http://auth.example",
        "NHX_BASE_URL": "http://base.example",
    }
    provider = MagicMock()
    provider.get_access_token.return_value = "access-token"

    with (
        patch("nhx.core.jobs.controllers.backends.subprocess_runtime.OTLPLogExporter") as exporter,
        patch(
            "nhx.core.jobs.controllers.backends.subprocess_runtime.resolve_workload_exchange_provider",
            return_value=provider,
        ) as resolve_provider,
    ):
        result = _build_otlp_log_exporter(env, "http://files.example/otlp/v1/logs")

    assert result is exporter.return_value
    resolve_provider.assert_called_once_with(base_url="http://auth.example", subject_token_file=subject_token_file)
    provider.get_access_token.assert_called_once_with()
    exporter.assert_called_once_with(
        endpoint="http://files.example/otlp/v1/logs",
        headers={"Authorization": "Bearer access-token"},
    )


def test_build_otlp_log_exporter_uses_unix_socket_session_for_uds():
    env = {
        NHX_JOB_LAUNCHER_OTLP_LOGS_TRANSPORT_ENVVAR: "uds",
        NHX_JOB_LAUNCHER_OTLP_LOGS_SOCKET_PATH_ENVVAR: "/tmp/nemo-helix.sock",
    }

    with patch("nhx.core.jobs.controllers.backends.subprocess_runtime.OTLPLogExporter") as exporter:
        result = _build_otlp_log_exporter(env, "http://nemo-helix.local/otlp/v1/logs")

    assert result is exporter.return_value
    kwargs = exporter.call_args.kwargs
    assert kwargs["endpoint"] == "http://nemo-helix.local/otlp/v1/logs"
    assert kwargs["headers"] == SERVICE_JOBS_BEARER_HEADERS
    assert isinstance(kwargs["session"], _UnixSocketOTLPSession)
    kwargs["session"].close()


def test_build_otlp_log_exporter_rejects_uds_without_socket_path():
    with pytest.raises(ValueError, match=NHX_JOB_LAUNCHER_OTLP_LOGS_SOCKET_PATH_ENVVAR):
        _build_otlp_log_exporter(
            {NHX_JOB_LAUNCHER_OTLP_LOGS_TRANSPORT_ENVVAR: "uds"},
            "http://nemo-helix.local/otlp/v1/logs",
        )


def _create_logger_capturing_batch_kwargs(env: dict[str, str]) -> dict:
    env = {"NHX_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT": "http://platform.example/v1/logs", **env}
    with (
        patch("nhx.core.jobs.controllers.backends.subprocess_runtime._build_otlp_log_exporter"),
        patch("nhx.core.jobs.controllers.backends.subprocess_runtime.BatchLogRecordProcessor") as processor,
    ):
        create_otel_logger(env=env, workspace="ws", job="job", attempt_id="a1", step="s", task_id="t")
    return processor.call_args.kwargs


def test_create_otel_logger_uses_low_file_count_batching_defaults():
    kwargs = _create_logger_capturing_batch_kwargs({})

    assert kwargs["schedule_delay_millis"] == DEFAULT_LOG_EXPORT_INTERVAL_MS
    assert kwargs["max_export_batch_size"] == DEFAULT_LOG_EXPORT_MAX_BATCH_SIZE
    assert kwargs["max_queue_size"] >= kwargs["max_export_batch_size"]
    # Guard against regressing to the OTEL SDK defaults (5s / 512 records), which create many tiny files.
    assert kwargs["schedule_delay_millis"] >= 5000


def test_create_otel_logger_batching_is_configurable_via_env():
    kwargs = _create_logger_capturing_batch_kwargs(
        {NHX_JOB_LOG_EXPORT_INTERVAL_MS_ENVVAR: "1500", NHX_JOB_LOG_EXPORT_MAX_BATCH_SIZE_ENVVAR: "100"}
    )

    assert kwargs["schedule_delay_millis"] == 1500
    assert kwargs["max_export_batch_size"] == 100
    assert kwargs["max_queue_size"] >= 100


def test_create_otel_logger_keeps_default_queue_floor_for_small_batches():
    kwargs = _create_logger_capturing_batch_kwargs({NHX_JOB_LOG_EXPORT_MAX_BATCH_SIZE_ENVVAR: "1"})

    assert kwargs["max_export_batch_size"] == 1
    assert kwargs["max_queue_size"] == DEFAULT_LOG_EXPORT_MAX_BATCH_SIZE


def test_create_otel_logger_clamps_oversized_batch_size_env():
    kwargs = _create_logger_capturing_batch_kwargs({NHX_JOB_LOG_EXPORT_MAX_BATCH_SIZE_ENVVAR: "1000000000"})

    assert kwargs["max_export_batch_size"] == MAX_LOG_EXPORT_MAX_BATCH_SIZE
    assert kwargs["max_queue_size"] == MAX_LOG_EXPORT_MAX_BATCH_SIZE * 4


@pytest.mark.parametrize("bad_value", ["abc", "0", "-5", "  "])
def test_create_otel_logger_ignores_invalid_batching_env(bad_value):
    kwargs = _create_logger_capturing_batch_kwargs(
        {NHX_JOB_LOG_EXPORT_INTERVAL_MS_ENVVAR: bad_value, NHX_JOB_LOG_EXPORT_MAX_BATCH_SIZE_ENVVAR: bad_value}
    )

    assert kwargs["schedule_delay_millis"] == DEFAULT_LOG_EXPORT_INTERVAL_MS
    assert kwargs["max_export_batch_size"] == DEFAULT_LOG_EXPORT_MAX_BATCH_SIZE


_BLRP_ENVVARS = ("OTEL_BLRP_SCHEDULE_DELAY", "OTEL_BLRP_MAX_EXPORT_BATCH_SIZE", "OTEL_BLRP_MAX_QUEUE_SIZE")


def _real_processor_settings(env: dict[str, str]) -> tuple[float, int, int]:
    """Build the logger with the real SDK processor and return (interval ms, batch size, queue size)."""
    created: list[BatchLogRecordProcessor] = []

    def _capture(*args, **kwargs):
        processor = BatchLogRecordProcessor(*args, **kwargs)
        created.append(processor)
        return processor

    env = {"NHX_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT": "http://platform.example/v1/logs", **env}
    with (
        patch("nhx.core.jobs.controllers.backends.subprocess_runtime._build_otlp_log_exporter"),
        patch("nhx.core.jobs.controllers.backends.subprocess_runtime.BatchLogRecordProcessor", side_effect=_capture),
    ):
        otel_logger = create_otel_logger(env=env, workspace="ws", job="job", attempt_id="a1", step="s", task_id="t")
    assert otel_logger is not None
    try:
        batch_processor = created[0]._batch_processor
        return (
            batch_processor._schedule_delay_millis,
            batch_processor._max_export_batch_size,
            batch_processor._max_queue_size,
        )
    finally:
        otel_logger.close()


@pytest.mark.parametrize(
    ("otel_env", "nhx_env", "expected"),
    [
        # No overrides anywhere: our defaults.
        ({}, {}, (5000, 2048, 8192)),
        # Each OTEL_BLRP_* setting alone is honored; the others keep our defaults.
        ({"OTEL_BLRP_SCHEDULE_DELAY": "30000"}, {}, (30000, 2048, 8192)),
        ({"OTEL_BLRP_MAX_EXPORT_BATCH_SIZE": "4096"}, {}, (5000, 4096, 16384)),
        ({"OTEL_BLRP_MAX_QUEUE_SIZE": "16384"}, {}, (5000, 2048, 16384)),
        # All three together.
        (
            {
                "OTEL_BLRP_SCHEDULE_DELAY": "30000",
                "OTEL_BLRP_MAX_EXPORT_BATCH_SIZE": "4096",
                "OTEL_BLRP_MAX_QUEUE_SIZE": "16384",
            },
            {},
            (30000, 4096, 16384),
        ),
        # An NHX override wins for its own setting without suppressing the unrelated OTEL settings.
        (
            {"OTEL_BLRP_MAX_EXPORT_BATCH_SIZE": "4096", "OTEL_BLRP_MAX_QUEUE_SIZE": "16384"},
            {NHX_JOB_LOG_EXPORT_INTERVAL_MS_ENVVAR: "1500"},
            (1500, 4096, 16384),
        ),
        (
            {"OTEL_BLRP_SCHEDULE_DELAY": "30000", "OTEL_BLRP_MAX_QUEUE_SIZE": "1024"},
            {NHX_JOB_LOG_EXPORT_MAX_BATCH_SIZE_ENVVAR: "100"},
            (30000, 100, 1024),
        ),
        (
            {"OTEL_BLRP_SCHEDULE_DELAY": "30000", "OTEL_BLRP_MAX_EXPORT_BATCH_SIZE": "4096"},
            {NHX_JOB_LOG_EXPORT_MAX_BATCH_SIZE_ENVVAR: "100"},
            (30000, 100, 2048),
        ),
        # An explicitly configured queue is kept, and a batch that would not fit in it is reduced.
        ({"OTEL_BLRP_MAX_QUEUE_SIZE": "1024"}, {}, (5000, 1024, 1024)),
        ({"OTEL_BLRP_MAX_QUEUE_SIZE": "1024"}, {NHX_JOB_LOG_EXPORT_MAX_BATCH_SIZE_ENVVAR: "4096"}, (5000, 1024, 1024)),
        # Invalid OTEL values fall back to our defaults.
        ({"OTEL_BLRP_SCHEDULE_DELAY": "abc", "OTEL_BLRP_MAX_QUEUE_SIZE": "0"}, {}, (5000, 2048, 8192)),
    ],
)
def test_create_otel_logger_resolves_batching_per_setting_with_real_processor(monkeypatch, otel_env, nhx_env, expected):
    for name in _BLRP_ENVVARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in otel_env.items():
        monkeypatch.setenv(name, value)

    assert _real_processor_settings(nhx_env) == expected


def test_create_otel_logger_honors_short_otel_schedule_delay(monkeypatch):
    """A controller-configured 20 ms OTEL_BLRP_SCHEDULE_DELAY must still export a lone record promptly."""
    for name in _BLRP_ENVVARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OTEL_BLRP_SCHEDULE_DELAY", "20")
    exported = threading.Event()
    exporter = MagicMock()
    exporter.export.side_effect = lambda batch: exported.set() or MagicMock()

    env = {"NHX_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT": "http://platform.example/v1/logs"}
    with patch("nhx.core.jobs.controllers.backends.subprocess_runtime._build_otlp_log_exporter", return_value=exporter):
        otel_logger = create_otel_logger(env=env, workspace="ws", job="job", attempt_id="a1", step="s", task_id="t")
    assert otel_logger is not None
    try:
        otel_logger.emit("one record", "stdout")
        assert exported.wait(timeout=2)
    finally:
        otel_logger.close()


def test_create_otel_logger_close_flushes_buffered_records_before_interval():
    """Records emitted well inside the export interval must still be exported on close."""
    exported: list[str] = []
    exporter = MagicMock()
    exporter.export.side_effect = lambda batch: exported.extend(r.log_record.body for r in batch) or MagicMock()

    env = {"NHX_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT": "http://platform.example/v1/logs"}
    with patch("nhx.core.jobs.controllers.backends.subprocess_runtime._build_otlp_log_exporter", return_value=exporter):
        otel_logger = create_otel_logger(env=env, workspace="ws", job="job", attempt_id="a1", step="s", task_id="t")
    assert otel_logger is not None

    otel_logger.emit("tail line 1", "stdout")
    otel_logger.emit("tail line 2", "stdout")
    assert exported == []  # still buffered; the 5s interval has not elapsed

    otel_logger.close()

    assert exported == ["tail line 1", "tail line 2"]
