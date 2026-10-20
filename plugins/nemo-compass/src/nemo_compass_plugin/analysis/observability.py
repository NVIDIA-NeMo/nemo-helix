# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nooa OpenTelemetry setup for Compass self-observability."""

from dataclasses import dataclass
from urllib.parse import urlparse
from uuid import uuid4

from nemo_compass_plugin.platform_client import LOOPBACK_HOSTS
from nemo_helix_ext.config.config import Config
from nooa.tracing import enable_tracing, exporters, flush_traces, set_session

COMPASS_OBSERVABILITY_ENV = "NEMO_COMPASS_OBSERVABILITY"
COMPASS_OBSERVABILITY_AGENT_NAME = "Compass"
COMPASS_OBSERVABILITY_SERVICE_NAMESPACE = "nemo-compass"
OTLP_TRACES_PATH = "/apis/intake/v2/workspaces/{workspace}/ingest/otlp/v1/traces"


@dataclass(frozen=True)
class CompassEvaluationContext:
    """Evaluation identity attached to every span in one Compass run."""

    evaluation_name: str
    test_case_name: str


@dataclass
class CompassObservability:
    """Configured Nooa instrumentation for one Compass run."""

    endpoint: str
    session_id: str

    def shutdown(self) -> None:
        """Flush this process's pending spans without disabling global tracing."""
        flush_traces()


def build_intake_otlp_traces_endpoint(*, base_url: str, workspace: str) -> str:
    """Return Intake's workspace-scoped OTLP/HTTP traces endpoint."""
    parsed = urlparse(base_url)
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    if scheme != "https" and not (scheme == "http" and host in LOOPBACK_HOSTS):
        raise ValueError(
            f"Compass observability endpoint must use HTTPS (got {base_url!r}). "
            "HTTP is only allowed for loopback addresses."
        )
    return f"{base_url.rstrip('/')}{OTLP_TRACES_PATH.format(workspace=workspace)}"


def setup_compass_observability(
    *,
    base_url: str,
    workspace: str,
    target_agent: str,
    evaluation_context: CompassEvaluationContext | None = None,
) -> CompassObservability:
    """Configure native Nooa OTel instrumentation for Intake export.

    By default, Compass sends its own spans to the platform Intake endpoint
    derived from the active ``--base-url`` and ``--workspace``.
    """
    endpoint = build_intake_otlp_traces_endpoint(base_url=base_url, workspace=workspace)
    session_id = f"{COMPASS_OBSERVABILITY_AGENT_NAME}-{uuid4()}"
    resource_attributes = _resource_attributes(
        workspace=workspace,
        target_agent=target_agent,
        session_id=session_id,
        evaluation_context=evaluation_context,
    )

    otlp_headers = _otlp_auth_headers(base_url)
    enable_tracing(
        exporters=[exporters.otlp(endpoint=endpoint, headers=otlp_headers)],
        extra_resource_attrs=resource_attributes,
    )
    set_session(session_id)
    return CompassObservability(
        endpoint=endpoint,
        session_id=session_id,
    )


def _otlp_auth_headers(base_url: str) -> dict[str, str] | None:
    """Return Bearer auth headers for remote Intake OTLP ingest, if available."""
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    config_path = Config.get_default_config_path()
    if parsed.scheme.lower() != "https" or host in LOOPBACK_HOSTS or not config_path.exists():
        return None

    config = Config.load(config_path, overrides={"base_url": base_url})
    user = config.resolve().user
    assert user is not None
    client_config = user.get_client_config()
    headers = client_config.get("default_headers")
    if not isinstance(headers, dict):
        return None
    return {str(k): str(v) for k, v in headers.items()}


def _resource_attributes(
    *,
    workspace: str,
    target_agent: str,
    session_id: str,
    evaluation_context: CompassEvaluationContext | None,
) -> dict[str, str]:
    attributes = {
        "service.name": COMPASS_OBSERVABILITY_AGENT_NAME,
        "service.namespace": COMPASS_OBSERVABILITY_SERVICE_NAMESPACE,
        "gen_ai.agent.name": COMPASS_OBSERVABILITY_AGENT_NAME,
        "gen_ai.agent.id": COMPASS_OBSERVABILITY_AGENT_NAME,
        "project.name": COMPASS_OBSERVABILITY_AGENT_NAME,
        "session.id": session_id,
        "gen_ai.conversation.id": session_id,
        "nemo.compass.workspace": workspace,
        "nemo.compass.target_agent": target_agent,
    }
    if evaluation_context is not None:
        attributes["nemo.evaluation.name"] = evaluation_context.evaluation_name
        attributes["nemo.test_case.name"] = evaluation_context.test_case_name
    return attributes
