# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import yaml
from nemo_helix_plugin.community_plugins.manifest import MAX_MANIFEST_BYTES, parse_manifest_document
from nemo_helix_plugin.community_plugins.registry import ExternalPlugin
from pydantic import ValidationError

YAML_MANIFEST = b"""# An API contract, not a Python distribution or deployment.
schema_version: 1
plugin:
  name: tavily
  title: Tavily
  endpoints:
    - method: POST
      path: /v2/workspaces/{workspace}/mcp
      access: write
      upstream_url: https://mcp.tavily.com/mcp
  credential:
    secret_name: tavily-api-key
"""


def test_yaml_and_json_normalize_to_same_consent_snapshot() -> None:
    raw = parse_manifest_document(YAML_MANIFEST)
    assert isinstance(raw, dict)
    parsed_json = parse_manifest_document(json.dumps(raw).encode())
    assert isinstance(parsed_json, dict)
    first = ExternalPlugin.model_validate(raw["plugin"])
    second = ExternalPlugin.model_validate(parsed_json["plugin"])
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.model_dump() == second.model_dump()


@pytest.mark.parametrize(
    "source",
    [
        b"plugin: {name: first, name: second}",
        b'{"plugin":{"name":"first","name":"second"}}',
        b"plugin: &plugin {name: tavily}\ncopy: *plugin",
        b"plugin: {<<: {name: tavily}}",
        b"!!python/object/apply:os.system ['echo must-not-run']",
        b"value: !!binary /w==",
        b"value: 2026-11-18",
        b"value: .nan",
        b"? [complex, key]\n: value",
        b"---\nfirst\n---\nsecond",
        b"\xff\xfe\x00",
    ],
)
def test_ambiguous_or_unsafe_yaml_is_rejected(source: bytes) -> None:
    with pytest.raises(ValueError):
        parse_manifest_document(source)


def test_yaml_complexity_is_bounded() -> None:
    for source in (b"[" * 60 + b"0" + b"]" * 60, b"[" + b"0," * 10001 + b"0]", b"x" * (MAX_MANIFEST_BYTES + 1)):
        with pytest.raises(ValueError):
            parse_manifest_document(source)


@pytest.mark.parametrize("field", ["wheel", "python", "image", "command", "runtime"])
def test_registry_rejects_deployment_instructions(field: str) -> None:
    raw = parse_manifest_document(YAML_MANIFEST)
    assert isinstance(raw, dict)
    with pytest.raises(ValidationError):
        ExternalPlugin.model_validate(raw["plugin"] | {field: "not-a-registry-concern"})


def test_published_tavily_example_is_deployment_independent() -> None:
    path = Path(__file__).resolve().parents[3] / "packages/nhx_platform_runner/examples/community_plugins/tavily.yaml"
    raw = parse_manifest_document(path.read_bytes())
    assert isinstance(raw, dict)
    plugin = ExternalPlugin.model_validate(raw["plugin"])
    assert plugin.endpoints[0].upstream_url == "https://mcp.tavily.com/mcp"
    assert plugin.credential is not None
    assert plugin.credential.secret_name == "tavily-api-key"
    # A plain safe YAML consumer can read the same external contract.
    assert yaml.safe_load(path.read_bytes())["schema_version"] == 1
