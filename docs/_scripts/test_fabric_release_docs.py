# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the public Fabric upgrade path discoverable and release-consistent.

These are documentation contract checks, not substitutes for harness execution.
The coverage table records the user-facing rationale for each contract change.
"""

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
UPGRADE = DOCS / "agents/fabric-upgrade.mdx"
MINI_SWE = DOCS / "agents/mini-swe-agent.mdx"
NAV = DOCS / "fern/versions/latest.yml"


@pytest.mark.parametrize("page", [UPGRADE, MINI_SWE])
def test_new_guides_are_published_and_linked_from_agents(page: Path) -> None:
    assert f"../../agents/{page.name}" in NAV.read_text()
    slug = page.stem
    for source in ("agents/index.mdx", "agents/register-agents.mdx"):
        assert f"/documentation/agents/{slug}" in (DOCS / source).read_text()


@pytest.mark.parametrize("source", ["agents/register-agents.mdx", "agents/fabric-upgrade.mdx"])
def test_registration_migration_states_removal_and_replacement(source: str) -> None:
    text = (DOCS / source).read_text()
    assert "nemo agents create" in text
    assert "nemo agents register" in text
    assert "no compatibility alias" in text.lower()
    assert "No such command 'create'" in text
    assert "AgentsResource.register" in text


def test_published_current_task_snippets_do_not_restore_removed_commands() -> None:
    pages = re.findall(r"path:\s*(\S+\.mdx)\s*$", NAV.read_text(), re.MULTILINE)
    for relative in pages:
        path = (NAV.parent / relative).resolve()
        # Historical release notes describe historical interfaces, not current tasks.
        if "release-notes" in path.parts:
            continue
        text = path.read_text()
        assert not re.search(r"nemo agents (?:optimize-skills|evaluate-suite|analyze)\b", text), path
        for block in re.findall(r"```(?:bash|sh|shell)\b[^\n]*\n(.*?)```", text, re.DOTALL):
            assert not re.search(r"\bnemo agents create\b", block), path


def test_harbor_preflight_and_custom_discovery_have_a_public_path() -> None:
    text = (DOCS / "evals/agent-eval/harbor-fabric-agent.mdx").read_text()
    assert "## Preflight and custom adapters" in text
    for field in (
        "harness.adapter_id",
        "fabric_config",
        "fabric_package",
        "fabric_discovery_paths",
        "fabric_adapter_descriptor",
        "fabric_config_bundle",
        "fabric_config_target",
        "discovery.local_paths",
    ):
        assert field in text
    assert "NemoFabricAgent.preflight" in text
    assert "FabricInstalledAgent" in text
    assert "Requirement ranges are left to the installer" in text
    assert "/documentation/agents/fabric-upgrade" in text
    assert "Nothing Fabric-related needs to be installed on the host" not in text


@pytest.mark.parametrize(
    "section",
    [
        "Persisted Relay observability v3 to v4",
        "Telemetry routing through gateways",
        "Adapter availability is not runtime installation",
        "Pi packaging and prerequisites",
        "Release documentation coverage",
    ],
)
def test_upgrade_contract_sections_are_retained(section: str) -> None:
    assert f"## {section}" in UPGRADE.read_text() or f"### {section}" in UPGRADE.read_text()


def test_release_coverage_map_records_guidance_or_no_docs_rationale() -> None:
    text = UPGRADE.read_text().split("## Release documentation coverage", 1)[1]
    for outcome in (
        "Registration rename/removal",
        "Fabric/Harbor versions",
        "Persisted Relay schema",
        "Mini-SWE runtime",
        "Pi descriptor",
        "Removed hidden improvement prototype",
    ):
        assert outcome in text
    assert "No current user documentation required" in text
    assert "/documentation/agents/fabric-upgrade" in (DOCS / "about/release-notes/index.mdx").read_text()


def test_mini_swe_public_lifecycle_and_capability_boundaries() -> None:
    text = MINI_SWE.read_text()
    for section in (
        "Prerequisites and installation",
        "Create the configuration",
        "Register and deploy",
        "Invoke once",
        "Sessions and streaming",
        "Response correlation and telemetry",
        "Timeouts and unsupported capabilities",
        "Troubleshooting",
        "Cleanup",
        "Validation boundary",
    ):
        assert f"## {section}" in text
    for command in (
        "nemo agents register",
        "nemo agents deploy",
        "nemo agents invoke",
        "nemo agents chat",
        "nemo agents sessions close",
        "nemo agents undeploy",
        "nemo agents delete",
    ):
        assert command in text
    assert "nemo_fabric_request_id" in text
    assert "--session-name" in text and "--session mini-swe-demo-session" in text
    assert "appended system instructions, skills, MCP servers, or tool policy" in text
    assert "not** require complete" in text


def test_mini_swe_configuration_matches_the_shipped_example() -> None:
    block = re.search(r"```yaml\n(.*?)```", MINI_SWE.read_text(), re.DOTALL)
    assert block is not None
    config = yaml.safe_load(block.group(1))
    example = yaml.safe_load(
        (ROOT / "plugins/nemo-agents/examples/nemo-agent-config/agent-mini-swe-agent.yaml").read_text()
    )
    config.pop("name")
    example.pop("name")
    assert config == example


def test_pi_native_capability_boundary_matches_shipped_descriptor() -> None:
    descriptor = json.loads((ROOT / "plugins/nemo-agents/pi.fabric-adapter.json").read_text())
    assert descriptor["adapter_id"] == "nvidia.fabric.pi"
    assert descriptor["capabilities"] == {
        "streaming": False,
        "cancellation": False,
        "updates": False,
        "service": False,
    }


def test_pi_contract_does_not_confuse_descriptor_with_runtime_or_release_gate() -> None:
    text = UPGRADE.read_text()
    assert "opt-in, not end-to-end release-gated" in text
    assert "pi.fabric-adapter.json" in text
    assert "Node >=22.19.0" in text
    assert "nemo-fabric-adapters-pi@^0.5.0" in text
    assert "no `nemo-agents-plugin-pi` Python extra" in text
    assert "relay_extension_path" in text
    assert "native streaming" in text
