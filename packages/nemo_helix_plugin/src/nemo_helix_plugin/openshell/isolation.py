# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build environment and network restrictions for sandboxes using Helix-backed providers."""

import re
from fnmatch import fnmatchcase

from google.protobuf import json_format
from google.protobuf.message import Message
from nemo_helix_plugin.entities.base import parse_qualified_name
from nemo_helix_plugin.openshell.types import GatewayProviderBinding


def _segment(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.+@:-]*", value):
        raise ValueError("Invalid workload path segment.")
    return value


def validate_provider_environment(
    bindings: dict[str, GatewayProviderBinding], *, workspace: str, secret_env: dict[str, str], env_names: set[str]
) -> None:
    """Reject process-env secrets and provider credential slots that collide with workload env vars."""
    if secret_env:
        raise ValueError("OpenShell secrets require gateway provider bindings; process-env injection is not supported.")
    keys = [key for binding in bindings.values() for key in binding.credentials]
    if len(keys) != len(set(keys)):
        raise ValueError("Gateway credential slots must be unique across providers.")
    for key in keys:
        if (
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
            or key.startswith(("NHX_", "NEMO_", "OPENAI_"))
            and key != "OPENAI_API_KEY"
            or key in {"PATH", "HOME", "PYTHONPATH", "VIRTUAL_ENV", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"}
            or key in env_names
        ):
            raise ValueError("Gateway credential slot conflicts with the workload environment.")


def restricted_platform_policy[Policy: Message](
    policy: Policy,
    *,
    host: str,
    port: int,
    workspace: str,
    binaries: list[str],
    job: str | None = None,
    attempt: str | None = None,
    step: str | None = None,
    fileset: str | None = None,
    secret_env: dict[str, str] | None = None,
    read_filesets: list[str] | None = None,
) -> Policy:
    """Preserve external-service access and restrict Helix to the workload's required API calls.

    Replace Helix endpoints with inference, telemetry and optional job/fileset access.
    This prevents the sandbox from using the platform connection to read secrets
    directly. OpenShell adds approved provider access and enforces the policy.
    """
    workspace = _segment(workspace)
    rules = [
        {"allow": {"method": "POST", "path": path}}
        for path in (
            f"/apis/intake/v2/workspaces/{workspace}/ingest/atif",
            f"/apis/intake/v2/workspaces/{workspace}/ingest/otlp/v1/traces",
            f"/apis/intake/v2/workspaces/{workspace}/ingest/otlp/v1/logs",
            f"/apis/intake/v2/workspaces/{workspace}/ingest/otlp/v1/metrics",
        )
    ]
    for surface in ("openai", "model/*", "provider/*"):
        for operation in ("chat/completions", "completions", "responses", "embeddings", "messages"):
            rules.append(
                {
                    "allow": {
                        "method": "POST",
                        "path": f"/apis/inference-gateway/v2/workspaces/{workspace}/{surface}/-/v1/{operation}",
                    }
                }
            )
    if job is not None and step is not None:
        if attempt is None:
            raise ValueError("Job sandbox policy requires a concrete attempt.")
        prefix = f"/apis/jobs/v2/workspaces/{workspace}/jobs/{_segment(job)}"
        step_path = f"{prefix}/steps/{_segment(step)}"
        attempt_path = f"{prefix}/attempts/{_segment(attempt)}"
        attempt_step_path = f"{attempt_path}/steps/{_segment(step)}"
        rules.extend(
            {"allow": {"method": method, "path": path}}
            for method, path in (
                ("GET", step_path),
                ("GET", f"{prefix}/status"),
                ("PATCH", f"{attempt_path}/status-details"),
                ("PATCH", f"{step_path}/status"),
                ("PATCH", f"{attempt_step_path}/status"),
                ("PUT", f"{attempt_step_path}/tasks/*"),
                ("GET", f"{attempt_step_path}/tasks/*"),
                ("GET", f"{step_path}/tasks/*"),
                ("POST", f"{prefix}/results/*"),
            )
        )
    if fileset:
        prefix = f"/apis/files/v2/workspaces/{workspace}/filesets/{_segment(fileset)}"
        rules.extend(
            {"allow": {"method": method, "path": path}}
            for method, path in (
                ("GET", prefix),
                ("GET", f"{prefix}/files"),
                ("GET", f"{prefix}/-/**"),
                ("PUT", f"{prefix}/-/**"),
                ("POST", f"{prefix}/otlp/v1/logs"),
            )
        )
    for ref in read_filesets or []:
        input_workspace, name = parse_qualified_name(ref, default_workspace=workspace)
        prefix = f"/apis/files/v2/workspaces/{_segment(input_workspace)}/filesets/{_segment(name)}"
        rules.extend(
            {"allow": {"method": "GET", "path": path}} for path in (prefix, f"{prefix}/files", f"{prefix}/-/**")
        )
    if secret_env:
        raise ValueError("OpenShell sandboxes cannot fetch secret values from the platform.")
    data = json_format.MessageToDict(policy, preserving_proto_field_name=True)
    network = data.setdefault("network_policies", {})
    platform_host = host.lower().rstrip(".")
    for key, existing in list(network.items()):
        endpoints = []
        for endpoint in existing.get("endpoints", []):
            endpoint_host = endpoint.get("host", "").lower().rstrip(".")
            if endpoint_host == platform_host:
                continue
            if fnmatchcase(platform_host, endpoint_host):
                raise ValueError("A wildcard network endpoint overlaps Helix; use explicit external-service hosts.")
            endpoints.append(endpoint)
        if endpoints:
            existing["endpoints"] = endpoints
        else:
            del network[key]
    key = "nemo_workload"
    while key in network:
        key += "_"
    network[key] = {
        "name": "nemo-workload-access",
        "endpoints": [
            {
                "host": host,
                "port": port,
                "protocol": "rest",
                "enforcement": "NETWORK_ENFORCEMENT_MODE_ENFORCE",
                "rules": rules,
            }
        ],
        "binaries": [{"path": path} for path in binaries],
    }
    data.setdefault("landlock", {})["compatibility"] = "hard_requirement"
    restricted = type(policy)()
    json_format.ParseDict(data, restricted)
    return restricted
