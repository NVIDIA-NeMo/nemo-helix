# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from nemo_platform_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nmp.common.auth import Principal
from nmp.common.auth.models import NMP_PRINCIPAL_ENVVAR
from nmp.common.jobs.constants import NEMO_JOB_SECRETS_ENVVAR
from nmp.core.jobs.controllers.backends.subprocess_runtime import (
    NMP_JOB_LAUNCHER_OTLP_LOGS_SOCKET_PATH_ENVVAR,
    NMP_JOB_LAUNCHER_OTLP_LOGS_TRANSPORT_ENVVAR,
    SERVICE_JOBS_BEARER_HEADERS,
    SubprocessOtelLogger,
    _build_otlp_log_exporter,
    _UnixSocketOTLPSession,
    inject_secret_env_vars,
    parse_secret_references,
    start_log_capture,
)


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
        NMP_PRINCIPAL_ENVVAR: principal.model_dump_json(exclude_none=True),
        "NMP_SECRETS_URL": "http://secrets.example",
    }

    response = MagicMock()
    response.read.return_value = json.dumps({"value": "secret-value"}).encode("utf-8")
    response.__enter__.return_value = response
    response.__exit__.return_value = None

    with patch("nmp.core.jobs.controllers.backends.subprocess_runtime.urlopen", return_value=response) as mock_urlopen:
        updated_env = inject_secret_env_vars(env.copy())

    assert updated_env["HF_TOKEN"] == "secret-value"
    request = mock_urlopen.call_args.args[0]
    assert request.full_url == "http://secrets.example/apis/secrets/v2/workspaces/default/secrets/hf-token/access"
    assert request.get_header("X-nmp-principal-id") == "service:jobs"
    assert request.get_header("X-nmp-actor-aliases") == "service:jobs"
    assert request.get_header("X-nmp-principal-on-behalf-of") == "creator@example.com"
    assert request.get_header("X-nmp-subject-account-id") == "account-creator"
    assert request.get_header("X-nmp-subject-aliases") == "legacy-creator,creator@example.com"


def test_inject_secret_env_vars_rejects_missing_value_field():
    env = {
        NEMO_JOB_SECRETS_ENVVAR: "HF_TOKEN=default/hf-token",
        "NMP_SECRETS_URL": "http://secrets.example",
    }
    response = MagicMock()
    response.read.return_value = json.dumps({"data": "secret-value"}).encode("utf-8")
    response.__enter__.return_value = response
    response.__exit__.return_value = None

    with patch("nmp.core.jobs.controllers.backends.subprocess_runtime.urlopen", return_value=response):
        with pytest.raises(RuntimeError, match="missing string value field"):
            inject_secret_env_vars(env.copy())


def test_inject_secret_env_vars_rejects_non_http_secrets_url():
    env = {
        NEMO_JOB_SECRETS_ENVVAR: "HF_TOKEN=default/hf-token",
        "NMP_SECRETS_URL": "file:///tmp/secrets",
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
    with patch("nmp.core.jobs.controllers.backends.subprocess_runtime.OTLPLogExporter") as exporter:
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
        "NMP_AUTH_URL": "http://auth.example",
        "NMP_BASE_URL": "http://base.example",
    }
    provider = MagicMock()
    provider.get_access_token.return_value = "access-token"

    with (
        patch("nmp.core.jobs.controllers.backends.subprocess_runtime.OTLPLogExporter") as exporter,
        patch(
            "nmp.core.jobs.controllers.backends.subprocess_runtime.resolve_workload_exchange_provider",
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
        NMP_JOB_LAUNCHER_OTLP_LOGS_TRANSPORT_ENVVAR: "uds",
        NMP_JOB_LAUNCHER_OTLP_LOGS_SOCKET_PATH_ENVVAR: "/tmp/nemo-platform.sock",
    }

    with patch("nmp.core.jobs.controllers.backends.subprocess_runtime.OTLPLogExporter") as exporter:
        result = _build_otlp_log_exporter(env, "http://nemo-platform.local/otlp/v1/logs")

    assert result is exporter.return_value
    kwargs = exporter.call_args.kwargs
    assert kwargs["endpoint"] == "http://nemo-platform.local/otlp/v1/logs"
    assert kwargs["headers"] == SERVICE_JOBS_BEARER_HEADERS
    assert isinstance(kwargs["session"], _UnixSocketOTLPSession)
    kwargs["session"].close()


def test_build_otlp_log_exporter_rejects_uds_without_socket_path():
    with pytest.raises(ValueError, match=NMP_JOB_LAUNCHER_OTLP_LOGS_SOCKET_PATH_ENVVAR):
        _build_otlp_log_exporter(
            {NMP_JOB_LAUNCHER_OTLP_LOGS_TRANSPORT_ENVVAR: "uds"},
            "http://nemo-platform.local/otlp/v1/logs",
        )
