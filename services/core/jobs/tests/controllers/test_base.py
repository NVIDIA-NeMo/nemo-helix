# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for jobs controller backends base module."""

import datetime
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.parse import urlunsplit

import pytest
from nhx.common.config import PlatformConfig
from nhx.common.jobs.constants import EPHEMERAL_TASK_STORAGE_PATH_ENVVAR, PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nhx.common.jobs.schemas import PlatformJobStatus
from nhx.core.jobs.api.v2.jobs.schemas import PlatformJobStepWithContext
from nhx.core.jobs.app.providers import ContainerSpec, CPUExecutionProvider
from nhx.core.jobs.app.schemas import PlatformJobStepSpec, StepLifecycle
from nhx.core.jobs.controllers.backends.base import (
    JOB_LOGS_ENDPOINT_ENVVAR,
    NHX_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT_ENVVAR,
    NHX_JOB_LAUNCHER_OTLP_LOGS_SOCKET_PATH_ENVVAR,
    NHX_JOB_LAUNCHER_OTLP_LOGS_TRANSPORT_ENVVAR,
    WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR,
    JobExecutionProfileConfig,
    _contains_loopback_address,
    _replace_loopback_address,
    find_reserved_managed_job_environment_variable_names,
    get_job_runtime_shared_envvars,
    get_logs_endpoint_config_from_fileset,
    get_logs_endpoint_from_fileset,
    get_workload_identity_token_audience,
    resolve_task_image,
    validate_no_reserved_managed_job_environment_variable_names,
)
from nhx.core.jobs.controllers.backends.test import MockKubernetesCPUJobBackend
from pydantic import ValidationError

from services.core.jobs.tests.controllers.client_mocks import data_response


class TestManagedJobReservedEnvironment:
    def test_finds_reserved_auth_names_and_prefixes(self):
        assert find_reserved_managed_job_environment_variable_names(
            [
                "NHX_ACCESS_TOKEN",
                WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR,
                "NEMO_WORKFLOW_TOKEN",
                "NHX_WORKFLOW_TOKEN",
                "APP_SETTING",
            ]
        ) == [
            "NEMO_WORKFLOW_TOKEN",
            "NHX_ACCESS_TOKEN",
            "NHX_WORKFLOW_TOKEN",
            WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR,
        ]

    def test_step_validation_allows_storage_path_contract_env_names(self):
        validate_no_reserved_managed_job_environment_variable_names(
            [PERSISTENT_JOB_STORAGE_PATH_ENVVAR, EPHEMERAL_TASK_STORAGE_PATH_ENVVAR],
            source="Job step environment keys",
        )

    def test_step_validation_rejects_workload_identity_prefixes(self):
        with pytest.raises(ValueError, match="NHX_WORKLOAD_TOKEN"):
            validate_no_reserved_managed_job_environment_variable_names(
                ["NHX_WORKLOAD_TOKEN"],
                source="Job step environment keys",
            )

    def test_step_validation_rejects_platform_job_logs_endpoint(self):
        with pytest.raises(ValueError, match=JOB_LOGS_ENDPOINT_ENVVAR):
            validate_no_reserved_managed_job_environment_variable_names(
                [JOB_LOGS_ENDPOINT_ENVVAR],
                source="Job step environment keys",
            )

    def test_profile_env_rejects_access_token(self):
        with pytest.raises(ValidationError, match="NHX_ACCESS_TOKEN"):
            JobExecutionProfileConfig(env={"NHX_ACCESS_TOKEN": "token"})

    def test_profile_env_rejects_auth_service_url(self):
        with pytest.raises(ValidationError, match="NHX_AUTH_URL"):
            JobExecutionProfileConfig(env={"NHX_AUTH_URL": "https://auth.example.com"})

    def test_profile_env_rejects_workflow_typo_prefix(self):
        with pytest.raises(ValidationError, match="NEMO_WORKFLOW_TOKEN"):
            JobExecutionProfileConfig(env={"NEMO_WORKFLOW_TOKEN": "token"})

    def test_profile_env_allows_user_otel_configuration(self):
        config = JobExecutionProfileConfig(
            env={
                "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT": "https://otel.example.com/v1/logs",
                "OTEL_EXPORTER_OTLP_LOGS_HEADERS": "Authorization=Bearer%20external",
                "OTEL_LOGS_EXPORTER": "otlp",
                "OTEL_SERVICE_NAME": "custom-workload",
            }
        )

        assert config.env["OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"] == "https://otel.example.com/v1/logs"
        assert config.env["OTEL_EXPORTER_OTLP_LOGS_HEADERS"] == "Authorization=Bearer%20external"
        assert config.env["OTEL_LOGS_EXPORTER"] == "otlp"
        assert config.env["OTEL_SERVICE_NAME"] == "custom-workload"

    def test_profile_env_rejects_platform_job_logs_endpoint(self):
        with pytest.raises(ValidationError, match=JOB_LOGS_ENDPOINT_ENVVAR):
            JobExecutionProfileConfig(env={JOB_LOGS_ENDPOINT_ENVVAR: "https://files.example.com/v1/logs"})


class TestWorkloadIdentityAudience:
    def test_uses_workload_client_id_for_projected_subject_tokens(self):
        auth_config = SimpleNamespace(
            oidc=SimpleNamespace(
                workload_client_id="nemo-helix-workload",
                client_id="nemo-helix-cli",
                workload_audience="nemo-helix",
                audience="nemo-helix",
            )
        )

        with patch("nhx.common.config.get_auth_config", return_value=auth_config):
            assert get_workload_identity_token_audience() == "nemo-helix-workload"


class TestReplaceLoopbackAddress:
    def test_contains_loopback_address_matches_hostname_exactly(self):
        assert _contains_loopback_address("http://localhost:8080/v1/jobs")
        assert not _contains_loopback_address("http://api.localhost.example:8080/v1/jobs/localhost")

    def test_preserves_url_when_loopback_text_is_not_the_hostname(self):
        url = "http://api.localhost.example:8080/v1/proxy/localhost?next=http://127.0.0.1:8080"

        result = _replace_loopback_address(url, "host.docker.internal")

        assert result == url

    def test_replaces_ipv6_loopback_hostname_with_valid_bracket_formatting(self):
        result = _replace_loopback_address("http://[::1]:8080/v1/jobs", "fd00::10")

        assert result == "http://[fd00::10]:8080/v1/jobs"

    def test_replaces_loopback_hostname_with_userinfo_from_parsed_url(self):
        url = urlunsplit(("http", "user:pass@localhost:8080", "/v1/jobs", "", ""))
        expected = urlunsplit(("http", "user:pass@host.docker.internal:8080", "/v1/jobs", "", ""))

        result = _replace_loopback_address(url, "host.docker.internal")

        assert result == expected


class TestGetLogsEndpointFromFileset:
    """Tests for get_logs_endpoint_from_fileset."""

    def test_returns_expected_path_with_workspace_and_fileset(self):
        """Endpoint URL includes /apis/files prefix and workspace/fileset in path."""
        config = PlatformConfig(  # type: ignore[abstract]
            service_discovery={"files": "http://files.example.com"},
            loopback_address=None,
        )

        with patch(
            "nhx.core.jobs.controllers.backends.base.determine_loopback_override",
            return_value=None,
        ):
            result = get_logs_endpoint_from_fileset(config, workspace="my-workspace", fileset_id="my-fileset-id")

        assert result == (
            "http://files.example.com/apis/files/v2/workspaces/my-workspace/filesets/my-fileset-id/otlp/v1/logs"
        )

    def test_with_loopback_address_replaces_localhost_in_files_url(self):
        """When loopback_address is set, localhost in files URL (base_url) is replaced."""
        config = PlatformConfig(  # type: ignore[abstract]
            base_url="http://localhost:8080",
            loopback_address="host.docker.internal",
        )

        result = get_logs_endpoint_from_fileset(config, workspace="default", fileset_id="job-logs-abc")

        assert result == (
            "http://host.docker.internal:8080/apis/files/v2/workspaces/default/filesets/job-logs-abc/otlp/v1/logs"
        )

    def test_with_loopback_address_replaces_127_0_0_1(self):
        """When loopback_address is set, 127.0.0.1 in files URL is replaced."""
        config = PlatformConfig(  # type: ignore[abstract]
            base_url="http://127.0.0.1:3000",
            loopback_address="host.docker.internal",
        )

        result = get_logs_endpoint_from_fileset(config, workspace="default", fileset_id="fs-123")

        assert result == (
            "http://host.docker.internal:3000/apis/files/v2/workspaces/default/filesets/fs-123/otlp/v1/logs"
        )

    def test_no_loopback_override_when_files_url_has_no_loopback(self):
        """When files URL has no loopback host, URL is unchanged by loopback_address."""
        config = PlatformConfig(  # type: ignore[abstract]
            service_discovery={"files": "https://files.service.svc.cluster.local"},
            loopback_address="host.docker.internal",
        )

        result = get_logs_endpoint_from_fileset(config, workspace="default", fileset_id="fs-456")

        assert result == (
            "https://files.service.svc.cluster.local/apis/files/v2/workspaces/default/filesets/fs-456/otlp/v1/logs"
        )

    def test_uses_determine_loopback_override_when_loopback_address_not_set(self):
        """When loopback_address is None, determine_loopback_override() is used."""
        config = PlatformConfig(  # type: ignore[abstract]
            base_url="http://localhost:8080",
            loopback_address=None,
        )

        with patch(
            "nhx.core.jobs.controllers.backends.base.determine_loopback_override",
            return_value="host.docker.internal",
        ):
            result = get_logs_endpoint_from_fileset(config, workspace="default", fileset_id="job-logs-xyz")

        assert result == (
            "http://host.docker.internal:8080/apis/files/v2/workspaces/default/filesets/job-logs-xyz/otlp/v1/logs"
        )

    def test_no_replacement_when_determine_loopback_override_returns_none(self):
        """When determine_loopback_override returns None, files URL (base_url) is used as-is."""
        config = PlatformConfig(  # type: ignore[abstract]
            base_url="http://localhost:8080",
            loopback_address=None,
        )

        with patch(
            "nhx.core.jobs.controllers.backends.base.determine_loopback_override",
            return_value=None,
        ):
            result = get_logs_endpoint_from_fileset(config, workspace="ns1", fileset_id="fileset-789")

        assert result == ("http://localhost:8080/apis/files/v2/workspaces/ns1/filesets/fileset-789/otlp/v1/logs")

    def test_uses_service_discovery_files_url_when_set(self):
        """When service_discovery has 'files', get_logs_endpoint_from_fileset uses that URL."""
        config = PlatformConfig(  # type: ignore[abstract]
            base_url="http://platform:8080",
            service_discovery={"files": "http://files-service.svc.cluster.local:8080"},
            loopback_address=None,
        )

        with patch(
            "nhx.core.jobs.controllers.backends.base.determine_loopback_override",
            return_value=None,
        ):
            result = get_logs_endpoint_from_fileset(config, workspace="my-workspace", fileset_id="my-fileset-id")

        assert result == (
            "http://files-service.svc.cluster.local:8080/apis/files/v2/workspaces/my-workspace/filesets/my-fileset-id/otlp/v1/logs"
        )

    def test_platform_service_discovery_platform_routes_logs_endpoint(self):
        """Platform service_discovery can route job log export through a workload-facing gateway."""
        config = PlatformConfig(  # type: ignore[abstract]
            base_url="http://platform-api:8080",
            service_discovery={"platform": "https://nemo-gateway:8080"},
            loopback_address=None,
        )

        with patch(
            "nhx.core.jobs.controllers.backends.base.determine_loopback_override",
            return_value=None,
        ):
            result = get_logs_endpoint_from_fileset(
                config,
                workspace="my-workspace",
                fileset_id="my-fileset-id",
            )

        assert result == (
            "https://nemo-gateway:8080/apis/files/v2/workspaces/my-workspace/filesets/my-fileset-id/otlp/v1/logs"
        )

    def test_service_discovery_takes_precedence_over_base_url(self):
        """When service_discovery['files'] is set, it is used instead of base_url for files."""
        config = PlatformConfig(  # type: ignore[abstract]
            base_url="http://platform:8080",
            service_discovery={"files": "http://discovered-files:9000"},
            loopback_address=None,
        )

        with patch(
            "nhx.core.jobs.controllers.backends.base.determine_loopback_override",
            return_value=None,
        ):
            result = get_logs_endpoint_from_fileset(config, workspace="ws1", fileset_id="fs-1")

        # get_service_url("files") returns service_discovery["files"] when present
        assert result == ("http://discovered-files:9000/apis/files/v2/workspaces/ws1/filesets/fs-1/otlp/v1/logs")

    def test_service_discovery_files_takes_precedence_over_local_service_url(self):
        """Job log export uses routable service_discovery even when Files runs in-process."""
        config = PlatformConfig(  # type: ignore[abstract]
            base_url="http://platform:8080",
            services="files",
            service_discovery={"files": "http://files-direct:8080"},
            loopback_address=None,
        )

        with patch(
            "nhx.core.jobs.controllers.backends.base.determine_loopback_override",
            return_value=None,
        ):
            result = get_logs_endpoint_from_fileset(config, workspace="ws1", fileset_id="fs-1")

        assert result == ("http://files-direct:8080/apis/files/v2/workspaces/ws1/filesets/fs-1/otlp/v1/logs")

    def test_resolved_base_url_is_used_when_files_service_discovery_is_not_set(self):
        """Job log export uses resolved runtime base URL, not local in-process service URL lookup."""
        config = PlatformConfig(  # type: ignore[abstract]
            base_url="http://127.0.0.1:59007",
            services="files",
            service_discovery={},
            loopback_address=None,
        )

        with patch(
            "nhx.core.jobs.controllers.backends.base.determine_loopback_override",
            return_value=None,
        ):
            result = get_logs_endpoint_from_fileset(config, workspace="ws1", fileset_id="fs-1")

        assert result == ("http://127.0.0.1:59007/apis/files/v2/workspaces/ws1/filesets/fs-1/otlp/v1/logs")

    def test_service_discovery_files_with_loopback_replacement(self):
        """Loopback replacement works when files URL comes from service_discovery."""
        config = PlatformConfig(  # type: ignore[abstract]
            base_url="http://platform:8080",
            service_discovery={"files": "http://localhost:3000"},
            loopback_address="host.docker.internal",
        )

        result = get_logs_endpoint_from_fileset(config, workspace="default", fileset_id="job-logs-123")

        assert result == (
            "http://host.docker.internal:3000/apis/files/v2/workspaces/default/filesets/job-logs-123/otlp/v1/logs"
        )

    def test_uds_files_url_uses_placeholder_http_origin(self):
        """UDS files URL is normalized to a valid HTTP URL for OTLP request construction."""
        config = PlatformConfig(  # type: ignore[abstract]
            service_discovery={"files": "unix:///tmp/nemo-helix.sock"},
            loopback_address="host.docker.internal",
        )

        result = get_logs_endpoint_from_fileset(config, workspace="default", fileset_id="job-logs-123")

        assert result == ("http://nemo-helix.local/apis/files/v2/workspaces/default/filesets/job-logs-123/otlp/v1/logs")

    def test_uds_files_url_preserves_transport_metadata(self):
        config = PlatformConfig(  # type: ignore[abstract]
            service_discovery={"files": "unix:///tmp/nemo-helix.sock"},
            loopback_address="host.docker.internal",
        )

        result = get_logs_endpoint_config_from_fileset(config, workspace="default", fileset_id="job-logs-123")

        assert result.endpoint == (
            "http://nemo-helix.local/apis/files/v2/workspaces/default/filesets/job-logs-123/otlp/v1/logs"
        )
        assert result.transport == "uds"
        assert result.socket_path == "/tmp/nemo-helix.sock"
        assert result.to_env() == {
            NHX_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT_ENVVAR: result.endpoint,
            NHX_JOB_LAUNCHER_OTLP_LOGS_TRANSPORT_ENVVAR: "uds",
            NHX_JOB_LAUNCHER_OTLP_LOGS_SOCKET_PATH_ENVVAR: "/tmp/nemo-helix.sock",
        }


class TestGetJobRuntimeSharedEnvvars:
    def test_propagates_the_deployments_logging_settings_to_task_containers(self, monkeypatch, tmp_path):
        """A task container gets no platform config file, so a chart set to
        ``service.log_format: json`` would otherwise start its tasks on
        ``plain``. Resolve here, where the file is readable, and pass it down."""
        from nemo_helix_plugin.config import NHX_CONFIG_FILE_PATH_ENV_VAR, Configuration

        config_file = tmp_path / "config.yaml"
        config_file.write_text("service:\n  LOG_FORMAT: json\n  LOG_LEVEL: DEBUG\n")
        monkeypatch.setenv(NHX_CONFIG_FILE_PATH_ENV_VAR, str(config_file))
        monkeypatch.delenv("LOG_FORMAT", raising=False)
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        Configuration.clear_cache()

        try:
            envvars = get_job_runtime_shared_envvars(PlatformConfig())
        finally:
            Configuration.clear_cache()

        assert envvars["LOG_FORMAT"] == "json"
        assert envvars["LOG_LEVEL"] == "DEBUG"

    def test_unreadable_logging_settings_do_not_fail_scheduling(self, monkeypatch, tmp_path):
        """Logging configuration is not worth failing a schedule over; the task
        falls back to its own defaults."""
        from nemo_helix_plugin.config import NHX_CONFIG_FILE_PATH_ENV_VAR, Configuration

        config_file = tmp_path / "config.yaml"
        config_file.write_text("this: is: not: valid: yaml:\n")
        monkeypatch.setenv(NHX_CONFIG_FILE_PATH_ENV_VAR, str(config_file))
        Configuration.clear_cache()

        try:
            envvars = get_job_runtime_shared_envvars(PlatformConfig())
        finally:
            Configuration.clear_cache()

        assert "NHX_BASE_URL" in envvars, "scheduling env was lost over a logging setting"

    def test_uses_service_discovery_gateway_urls_for_job_runtimes(self):
        config = PlatformConfig(  # type: ignore[abstract]
            base_url="http://127.0.0.1:8080",
            services="jobs,files,models,secrets",
            service_discovery={
                "platform": "https://nemo-gateway:8080",
                "jobs": "https://nemo-gateway:8080",
                "files": "https://nemo-gateway:8080",
                "models": "https://nemo-gateway:8080",
                "secrets": "https://nemo-gateway:8080",
            },
            loopback_address="nemo-gateway",
        )

        envvars = get_job_runtime_shared_envvars(config)

        assert envvars == {
            "NHX_BASE_URL": "https://nemo-gateway:8080",
            "NHX_AUTH_URL": "https://nemo-gateway:8080",
            "NHX_JOBS_URL": "https://nemo-gateway:8080",
            "NHX_FILES_URL": "https://nemo-gateway:8080",
            "NHX_MODELS_URL": "https://nemo-gateway:8080",
            "NHX_SECRETS_URL": "https://nemo-gateway:8080",
            "NHX_CONFIG_WARNINGS_DISABLED": "1",
            # Propagated so a task container, which has no platform config
            # file, still logs the way the deployment configured it.
            "LOG_LEVEL": "INFO",
            "LOG_FORMAT": "plain",
        }

    def test_gateway_base_url_is_used_for_job_runtimes_without_service_discovery(self):
        config = PlatformConfig(  # type: ignore[abstract]
            base_url="https://nemo-gateway:8080",
            services="jobs,files,models,secrets",
            service_discovery={},
            loopback_address=None,
        )

        with patch(
            "nhx.core.jobs.controllers.backends.base.determine_loopback_override",
            return_value=None,
        ):
            envvars = get_job_runtime_shared_envvars(config)

        assert envvars == {
            "NHX_BASE_URL": "https://nemo-gateway:8080",
            "NHX_AUTH_URL": "https://nemo-gateway:8080",
            "NHX_JOBS_URL": "https://nemo-gateway:8080",
            "NHX_FILES_URL": "https://nemo-gateway:8080",
            "NHX_MODELS_URL": "https://nemo-gateway:8080",
            "NHX_SECRETS_URL": "https://nemo-gateway:8080",
            "NHX_CONFIG_WARNINGS_DISABLED": "1",
            # Propagated so a task container, which has no platform config
            # file, still logs the way the deployment configured it.
            "LOG_LEVEL": "INFO",
            "LOG_FORMAT": "plain",
        }

    def test_resolved_base_url_is_used_for_job_runtimes_without_service_discovery(self):
        config = PlatformConfig(  # type: ignore[abstract]
            base_url="http://127.0.0.1:59007",
            services="jobs,files,models,secrets",
            service_discovery={},
            loopback_address=None,
        )

        with patch(
            "nhx.core.jobs.controllers.backends.base.determine_loopback_override",
            return_value=None,
        ):
            envvars = get_job_runtime_shared_envvars(config)

        assert envvars == {
            "NHX_BASE_URL": "http://127.0.0.1:59007",
            "NHX_AUTH_URL": "http://127.0.0.1:59007",
            "NHX_JOBS_URL": "http://127.0.0.1:59007",
            "NHX_FILES_URL": "http://127.0.0.1:59007",
            "NHX_MODELS_URL": "http://127.0.0.1:59007",
            "NHX_SECRETS_URL": "http://127.0.0.1:59007",
            "NHX_CONFIG_WARNINGS_DISABLED": "1",
            # Propagated so a task container, which has no platform config
            # file, still logs the way the deployment configured it.
            "LOG_LEVEL": "INFO",
            "LOG_FORMAT": "plain",
        }

    def test_auth_service_url_is_exported_for_job_runtime(self):
        config = PlatformConfig(  # type: ignore[abstract]
            base_url="https://nemo-gateway:8080",
            services="jobs,files,models,secrets",
            service_discovery={"auth": "http://nemo-auth:8080"},
            loopback_address=None,
        )

        with patch(
            "nhx.core.jobs.controllers.backends.base.determine_loopback_override",
            return_value=None,
        ):
            envvars = get_job_runtime_shared_envvars(config)

        assert envvars["NHX_BASE_URL"] == "https://nemo-gateway:8080"
        assert envvars["NHX_AUTH_URL"] == "http://nemo-auth:8080"

    def test_platform_service_discovery_platform_overrides_job_runtime_base_url(self):
        config = PlatformConfig(  # type: ignore[abstract]
            base_url="http://nemo-helix-api:8080",
            services="jobs,files,models,secrets",
            service_discovery={
                "platform": "https://nemo-gateway:8080",
                "auth": "https://nemo-auth:8080",
            },
            loopback_address=None,
        )

        with patch(
            "nhx.core.jobs.controllers.backends.base.determine_loopback_override",
            return_value=None,
        ):
            envvars = get_job_runtime_shared_envvars(config)

        assert envvars["NHX_BASE_URL"] == "https://nemo-gateway:8080"
        assert envvars["NHX_AUTH_URL"] == "https://nemo-auth:8080"
        assert envvars["NHX_JOBS_URL"] == "https://nemo-gateway:8080"
        assert envvars["NHX_FILES_URL"] == "https://nemo-gateway:8080"

    def test_loopback_replacement_applies_to_base_url_fallback(self):
        config = PlatformConfig(  # type: ignore[abstract]
            base_url="http://127.0.0.1:8080",
            service_discovery={},
            loopback_address="host.docker.internal",
        )

        envvars = get_job_runtime_shared_envvars(config)

        assert envvars["NHX_BASE_URL"] == "http://host.docker.internal:8080"
        assert envvars["NHX_AUTH_URL"] == "http://host.docker.internal:8080"
        assert envvars["NHX_FILES_URL"] == "http://host.docker.internal:8080"


def _make_step(
    staleness_timeout: int = 0,
    created_at: datetime.datetime | None = None,
    step_spec: PlatformJobStepSpec | None = ...,  # type: ignore[assignment]
) -> PlatformJobStepWithContext:
    if step_spec is ...:
        step_spec = PlatformJobStepSpec(
            name="test-step",
            executor=CPUExecutionProvider(provider="cpu", profile="default", container=ContainerSpec(image="img")),
            config={},
            lifecycle=StepLifecycle(staleness_timeout_seconds=staleness_timeout),
        )
    return PlatformJobStepWithContext(
        id="step-1",
        job="job-1",
        attempt_id="attempt-1",
        fileset="fileset-1",
        workspace="default",
        name="test-step",
        step_spec=step_spec,
        status=PlatformJobStatus.ACTIVE,
        created_at=created_at or datetime.datetime.now(datetime.timezone.utc),
    )


def _make_task(status: str = "active", updated_at: datetime.datetime | None = None) -> MagicMock:
    task = MagicMock()
    task.status = status
    task.updated_at = updated_at
    return task


def _make_backend(mock_sdk: MagicMock | None = None) -> MockKubernetesCPUJobBackend:
    sdk = mock_sdk or MagicMock()
    return MockKubernetesCPUJobBackend(nhx_sdk=sdk, execution_profile_config=MagicMock(), profile_name="default")


@contextmanager
def _patched_jobs_client(backend):
    """Stub the backend's held ``self._jobs`` handle and yield the mock ``JobsClient``.

    The backend builds its typed Jobs client once in ``JobBackend.__init__`` and
    reuses it as ``self._jobs``, so tests stub that handle directly (rather than
    patching ``client_from_platform``). ``check_step_is_stale`` fetches tasks via
    ``self._jobs.list_job_step_tasks(...).data()``; the returned page exposes the
    task list on its ``.data`` attribute.
    """
    mock_jobs = MagicMock()
    original = backend._jobs
    backend._jobs = mock_jobs
    try:
        yield mock_jobs
    finally:
        backend._jobs = original


def _set_tasks(mock_jobs: MagicMock, tasks: list) -> None:
    mock_jobs.list_job_step_tasks.return_value = data_response(SimpleNamespace(data=tasks))


class TestCheckTaskStaleness:
    """Tests for JobBackend.check_step_is_stale."""

    def test_disabled_when_timeout_is_zero(self):
        backend = _make_backend()
        step = _make_step(staleness_timeout=0)

        assert backend.check_step_is_stale(step) is False

    def test_disabled_when_lifecycle_is_none(self):
        backend = _make_backend()
        step = _make_step()
        step_spec = step.step_spec
        assert step_spec is not None
        step.step_spec = step_spec.model_copy(update={"lifecycle": None})

        assert backend.check_step_is_stale(step) is False

    def test_not_stale_when_step_too_young(self):
        backend = _make_backend()
        step = _make_step(
            staleness_timeout=1800,
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )

        with _patched_jobs_client(backend) as mock_jobs:
            assert backend.check_step_is_stale(step) is False
            mock_jobs.list_job_step_tasks.assert_not_called()

    def test_not_stale_when_no_active_tasks(self):
        backend = _make_backend()
        step = _make_step(
            staleness_timeout=60,
            created_at=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=120),
        )

        with _patched_jobs_client(backend) as mock_jobs:
            _set_tasks(mock_jobs, [_make_task(status="completed"), _make_task(status="error")])
            assert backend.check_step_is_stale(step) is False

    def test_not_stale_when_task_recently_updated(self):
        backend = _make_backend()
        step = _make_step(
            staleness_timeout=60,
            created_at=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=120),
        )

        with _patched_jobs_client(backend) as mock_jobs:
            _set_tasks(
                mock_jobs,
                [_make_task(status="active", updated_at=datetime.datetime.now(datetime.timezone.utc))],
            )
            assert backend.check_step_is_stale(step) is False

    def test_stale_when_all_active_tasks_exceed_threshold(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        backend = _make_backend()
        step = _make_step(
            staleness_timeout=60,
            created_at=now - datetime.timedelta(seconds=120),
        )

        with _patched_jobs_client(backend) as mock_jobs:
            _set_tasks(
                mock_jobs,
                [
                    _make_task(status="active", updated_at=now - datetime.timedelta(seconds=200)),
                    _make_task(status="active", updated_at=now - datetime.timedelta(seconds=300)),
                    _make_task(status="completed"),
                ],
            )
            assert backend.check_step_is_stale(step) is True

    def test_not_stale_when_one_active_task_is_fresh(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        backend = _make_backend()
        step = _make_step(
            staleness_timeout=60,
            created_at=now - datetime.timedelta(seconds=120),
        )

        with _patched_jobs_client(backend) as mock_jobs:
            _set_tasks(
                mock_jobs,
                [
                    _make_task(status="active", updated_at=now - datetime.timedelta(seconds=200)),
                    _make_task(status="active", updated_at=now - datetime.timedelta(seconds=10)),
                ],
            )
            assert backend.check_step_is_stale(step) is False

    def test_returns_false_on_api_failure(self):
        backend = _make_backend()
        step = _make_step(
            staleness_timeout=60,
            created_at=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=120),
        )

        with _patched_jobs_client(backend) as mock_jobs:
            mock_jobs.list_job_step_tasks.side_effect = RuntimeError("connection error")
            assert backend.check_step_is_stale(step) is False

    def test_handles_naive_updated_at_as_utc(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        naive_old = (now - datetime.timedelta(seconds=200)).replace(tzinfo=None)
        backend = _make_backend()
        step = _make_step(
            staleness_timeout=60,
            created_at=now - datetime.timedelta(seconds=120),
        )

        with _patched_jobs_client(backend) as mock_jobs:
            _set_tasks(mock_jobs, [_make_task(status="active", updated_at=naive_old)])
            assert backend.check_step_is_stale(step) is True

    def test_returns_false_when_task_missing_updated_at(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        backend = _make_backend()
        step = _make_step(
            staleness_timeout=60,
            created_at=now - datetime.timedelta(seconds=120),
        )

        with _patched_jobs_client(backend) as mock_jobs:
            _set_tasks(mock_jobs, [_make_task(status="active", updated_at=None)])
            assert backend.check_step_is_stale(step) is False


class TestResolveTaskImage:
    """Tests for resolve_task_image."""

    def test_explicit_image_takes_precedence(self):
        assert resolve_task_image("my-image:v1", "default-image:latest") == "my-image:v1"

    def test_falls_back_to_default_task_image(self):
        assert resolve_task_image(None, "default-image:latest") == "default-image:latest"

    def test_explicit_image_without_default(self):
        assert resolve_task_image("my-image:v1", None) == "my-image:v1"

    def test_falls_back_to_platform_cpu_tasks_image_when_both_none(self):
        with patch("nemo_helix_plugin.jobs.image.get_platform_config") as mock_config:
            mock_config.return_value = MagicMock(image_registry="my-registry", image_tag="v1.0")
            assert resolve_task_image(None, None) == "my-registry/nhx-cpu-tasks:v1.0"
