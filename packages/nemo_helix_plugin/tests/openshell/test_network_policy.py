# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_helix_plugin.openshell.network_policy import (
    agent_from_labels,
    agent_labels,
    agent_network_link,
    rule_from_dict,
    rule_to_dict,
    validate_rules,
    with_network_rules,
)

pytest.importorskip("openshell")  # platform-restricted extra; skip where not installed (e.g. CI)

from openshell._proto import sandbox_pb2 as sb

EXAMPLE_RULE = {
    "name": "allow_example_com_443",
    "endpoints": [{"host": "example.com", "port": 443}],
    "binaries": [{"path": "/usr/local/bin/python3.13"}],
}


def test_agent_labels_name_the_agent_a_sandbox_runs_for() -> None:
    labels = {"other": "label", **agent_labels("default", "researcher")}

    assert agent_from_labels(labels) == ("default", "researcher")


@pytest.mark.parametrize("labels", [{}, {"nemo.agents/agent": "researcher"}, {"nemo.agents/workspace": "default"}])
def test_sandbox_without_both_agent_labels_belongs_to_no_agent(labels: dict[str, str]) -> None:
    assert agent_from_labels(labels) is None


def test_agent_network_link_opens_the_network_tab() -> None:
    assert agent_network_link("default", "researcher") == "/workspaces/default/agents/researcher?tab=network"


def test_gateway_proposed_rule_becomes_the_policy_yaml_shape() -> None:
    # What the gateway files for a blocked connection: it repeats the port and marks the rule as its own.
    proposed = sb.NetworkPolicyRule(
        name="allow_example_com_443",
        endpoints=[sb.NetworkEndpoint(host="example.com", port=443, ports=[443], advisor_proposed=True)],
        binaries=[sb.NetworkBinary(path="/usr/local/bin/python3.13")],
    )

    assert rule_to_dict(proposed) == EXAMPLE_RULE


def test_rule_with_several_ports_keeps_them() -> None:
    rule = sb.NetworkPolicyRule(
        name="r", endpoints=[sb.NetworkEndpoint(host="example.com", port=443, ports=[443, 8443])]
    )

    assert rule_to_dict(rule)["endpoints"] == [{"host": "example.com", "port": 443, "ports": [443, 8443]}]


def test_endpoint_enums_use_the_policy_yaml_spelling() -> None:
    rule = sb.NetworkPolicyRule(
        name="api",
        endpoints=[
            sb.NetworkEndpoint(
                host="api.example.com",
                port=443,
                protocol="rest",
                enforcement=sb.NETWORK_ENFORCEMENT_MODE_ENFORCE,
                access=sb.NETWORK_ACCESS_PRESET_READ_ONLY,
            )
        ],
    )

    endpoint = rule_to_dict(rule)["endpoints"][0]

    assert (endpoint["enforcement"], endpoint["access"]) == ("enforce", "read-only")
    assert "tls" not in endpoint
    assert rule_from_dict(rule_to_dict(rule)) == rule


def test_rule_round_trips_through_the_policy_yaml_shape() -> None:
    assert rule_to_dict(rule_from_dict(EXAMPLE_RULE)) == EXAMPLE_RULE


def test_rule_with_an_unknown_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid network rule"):
        rule_from_dict({**EXAMPLE_RULE, "endpoint": []})


def test_rule_without_an_endpoint_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one endpoint"):
        validate_rules({"empty": {"name": "empty", "binaries": [{"path": "/usr/bin/curl"}]}})


def test_valid_rules_pass_validation() -> None:
    validate_rules({"allow_example_com_443": EXAMPLE_RULE})


def test_rules_are_added_to_a_copy_of_the_policy() -> None:
    policy = sb.SandboxPolicy(version=1)
    policy.network_policies["nemo_helix"].name = "nemo-helix-egress"

    merged = with_network_rules(policy, {"allow_example_com_443": EXAMPLE_RULE})

    assert sorted(merged.network_policies) == ["allow_example_com_443", "nemo_helix"]
    assert merged.network_policies["allow_example_com_443"].endpoints[0].host == "example.com"
    assert merged.network_policies["allow_example_com_443"].binaries[0].path == "/usr/local/bin/python3.13"
    # The executor's policy is shared by every sandbox, so it must not pick up one agent's rules.
    assert sorted(policy.network_policies) == ["nemo_helix"]


def test_rule_cannot_replace_one_the_policy_already_has() -> None:
    policy = sb.SandboxPolicy(version=1)
    policy.network_policies["nemo_helix"].name = "nemo-helix-egress"

    merged = with_network_rules(policy, {"nemo_helix": EXAMPLE_RULE})

    assert merged.network_policies["nemo_helix"].name == "nemo-helix-egress"
    assert not merged.network_policies["nemo_helix"].endpoints


def test_no_rules_leaves_the_policy_as_it_is() -> None:
    policy = sb.SandboxPolicy(version=1)

    assert with_network_rules(policy, {}) is policy
