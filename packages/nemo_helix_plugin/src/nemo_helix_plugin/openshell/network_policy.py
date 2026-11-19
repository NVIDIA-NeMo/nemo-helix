# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Network rules a workload brings to its OpenShell sandbox, and the agent it runs for.

An agent can carry network rules that were approved for it earlier. They use the policy
YAML shape, keyed by rule name, the same as ``network_policies`` in a sandbox policy file::

    allow_example_com_443:
      name: allow_example_com_443
      endpoints: [{host: example.com, port: 443}]
      binaries: [{path: /usr/local/bin/python3.13}]

The Jobs and Deployments backends add them to the policy of each sandbox they create for
that agent, and label the sandbox with the agent so its requests can be found again.

Prototype scope: rules are added as given. Nothing checks that a rule is narrower than
what a reviewer would approve.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from google.protobuf import json_format

if TYPE_CHECKING:
    from openshell._proto import sandbox_pb2 as sb  # ty: ignore[unresolved-import]

AGENT_LABEL = "nemo.agents/agent"
AGENT_WORKSPACE_LABEL = "nemo.agents/workspace"

NetworkRules = dict[str, dict[str, Any]]

# Policy YAML spells these endpoint fields as short strings; the proto types them as enums.
_ENUM_PREFIXES = {
    "enforcement": "NETWORK_ENFORCEMENT_MODE_",
    "access": "NETWORK_ACCESS_PRESET_",
    "tls": "NETWORK_TLS_MODE_",
}


def agent_labels(workspace: str, agent: str) -> dict[str, str]:
    """Sandbox labels naming the agent a sandbox runs for."""
    return {AGENT_WORKSPACE_LABEL: workspace, AGENT_LABEL: agent}


def agent_from_labels(labels: Mapping[str, str]) -> tuple[str, str] | None:
    """The ``(workspace, agent)`` a sandbox runs for, or None if it is not an agent's."""
    workspace, agent = labels.get(AGENT_WORKSPACE_LABEL), labels.get(AGENT_LABEL)
    return (workspace, agent) if workspace and agent else None


def agent_network_link(workspace: str, agent: str) -> str:
    """Studio path of the agent's network access tab."""
    return f"/workspaces/{workspace}/agents/{agent}?tab=network"


def rule_to_dict(rule: Any) -> dict[str, Any]:
    """A ``NetworkPolicyRule`` proto in the policy YAML shape."""
    data = json_format.MessageToDict(rule, preserving_proto_field_name=True)
    for endpoint in data.get("endpoints", []):
        # The gateway marks rules it proposed itself and repeats a single port as a list.
        endpoint.pop("advisor_proposed", None)
        if endpoint.get("ports") == [endpoint.get("port")]:
            endpoint.pop("ports")
        for field, prefix in _ENUM_PREFIXES.items():
            value = endpoint.get(field)
            if not isinstance(value, str) or not value.startswith(prefix):
                continue
            name = value.removeprefix(prefix)
            if name == "UNSPECIFIED":
                endpoint.pop(field)
            else:
                endpoint[field] = name.lower().replace("_", "-")
    return data


def rule_from_dict(rule: Mapping[str, Any]) -> sb.NetworkPolicyRule:
    """Parse a rule in the policy YAML shape, rejecting anything the proto does not know."""
    from openshell._proto import sandbox_pb2 as sb  # ty: ignore[unresolved-import]

    data = dict(rule)
    endpoints = []
    for endpoint in data.get("endpoints") or []:
        endpoint = dict(endpoint)
        for field, prefix in _ENUM_PREFIXES.items():
            value = endpoint.get(field)
            if isinstance(value, str) and value and not value.startswith(prefix):
                endpoint[field] = prefix + value.upper().replace("-", "_")
        endpoints.append(endpoint)
    data["endpoints"] = endpoints
    try:
        return json_format.ParseDict(data, sb.NetworkPolicyRule())
    except json_format.ParseError as exc:
        raise ValueError(f"invalid network rule: {exc}") from exc


def validate_rules(rules: Mapping[str, Mapping[str, Any]]) -> None:
    """Raise ``ValueError`` if any rule is not a valid network rule naming an endpoint."""
    for key, rule in rules.items():
        parsed = rule_from_dict(rule)
        if not parsed.endpoints:
            raise ValueError(f"invalid network rule {key!r}: at least one endpoint is required")


def with_network_rules(policy: sb.SandboxPolicy, rules: Mapping[str, Mapping[str, Any]]) -> sb.SandboxPolicy:
    """A copy of ``policy`` with ``rules`` added. A rule already in the policy is kept as it is."""
    if not rules:
        return policy
    merged = type(policy)()
    merged.CopyFrom(policy)
    for key, rule in rules.items():
        if key not in merged.network_policies:
            merged.network_policies[key].CopyFrom(rule_from_dict(rule))
    return merged
