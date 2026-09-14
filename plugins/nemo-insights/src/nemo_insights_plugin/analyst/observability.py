# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nooa OpenTelemetry setup for analyst self-observability."""

from dataclasses import dataclass
from urllib.parse import urlparse
from uuid import uuid4

from nemo_insights_plugin.platform_client import LOOPBACK_HOSTS
from nemo_platform_ext.config.config import Config
from nooa.tracing import enable_tracing, exporters, flush_traces, set_session

ANALYST_OBSERVABILITY_ENV = "NEMO_INSIGHTS_ANALYST_OBSERVABILITY"
ANALYST_OBSERVABILITY_AGENT_NAME = "Analyst"
ANALYST_OBSERVABILITY_SERVICE_NAMESPACE = "nemo-insights"
OTLP_TRACES_PATH = "/apis/intake/v2/workspaces/{workspace}/ingest/otlp/v1/traces"


@dataclass(frozen=True)
class AnalystEvaluationContext:
    """Evaluation identity attached to every span in one Analyst run."""

    evaluation_name: str
    test_case_name: str


@dataclass
class AnalystObservability:
    """Configured Nooa instrumentation for one analyst run."""

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
            f"Analyst observability endpoint must use HTTPS (got {base_url!r}). "
            "HTTP is only allowed for loopback addresses."
        )
    return f"{base_url.rstrip('/')}{OTLP_TRACES_PATH.format(workspace=workspace)}"


def setup_analyst_observability(
    *,
    base_url: str,
    workspace: str,
    target_agent: str,
    evaluation_context: AnalystEvaluationContext | None = None,
) -> AnalystObservability:
    """Configure native Nooa OTel instrumentation for Intake export.

    By default, the analyst sends its own spans to the platform Intake endpoint
    derived from the active ``--base-url`` and ``--workspace``.
    """
    endpoint = build_intake_otlp_traces_endpoint(base_url=base_url, workspace=workspace)
    session_id = f"{ANALYST_OBSERVABILITY_AGENT_NAME}-{uuid4()}"
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
    return AnalystObservability(
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
    evaluation_context: AnalystEvaluationContext | None,
) -> dict[str, str]:
    attributes = {
        "service.name": ANALYST_OBSERVABILITY_AGENT_NAME,
        "service.namespace": ANALYST_OBSERVABILITY_SERVICE_NAMESPACE,
        "gen_ai.agent.name": ANALYST_OBSERVABILITY_AGENT_NAME,
        "gen_ai.agent.id": ANALYST_OBSERVABILITY_AGENT_NAME,
        "project.name": ANALYST_OBSERVABILITY_AGENT_NAME,
        "session.id": session_id,
        "gen_ai.conversation.id": session_id,
        "nemo.insights.workspace": workspace,
        "nemo.insights.target_agent": target_agent,
    }
    if evaluation_context is not None:
        attributes["nemo.evaluation.name"] = evaluation_context.evaluation_name
        attributes["nemo.test_case.name"] = evaluation_context.test_case_name
    return attributes
