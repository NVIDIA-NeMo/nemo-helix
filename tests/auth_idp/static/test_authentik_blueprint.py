# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

pytestmark = [pytest.mark.auth_idp]

BLUEPRINT = Path("contrib/auth/authentik/helm/files/blueprints/nemo.yaml")
LEGACY_BLUEPRINT = Path("contrib/auth/authentik/blueprints/nemo.yaml")


@dataclass(frozen=True)
class TaggedYamlValue:
    tag: str
    value: Any


class _BlueprintLoader(yaml.SafeLoader):
    pass


def _construct_tagged_value(loader: _BlueprintLoader, tag_suffix: str, node: Node) -> TaggedYamlValue:
    if isinstance(node, ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    elif isinstance(node, MappingNode):
        value = loader.construct_mapping(node, deep=True)
    else:
        raise AssertionError(f"Unsupported YAML node type for !{tag_suffix}: {type(node).__name__}")

    return TaggedYamlValue(f"!{tag_suffix}", value)


_BlueprintLoader.add_multi_constructor("!", _construct_tagged_value)


def _load_blueprint(path: Path = BLUEPRINT) -> Mapping[str, Any]:
    blueprint = yaml.load(path.read_text(encoding="utf-8"), Loader=_BlueprintLoader)
    assert isinstance(blueprint, dict)
    return blueprint


def _entries(blueprint: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries = blueprint["entries"]
    assert isinstance(entries, list)
    assert all(isinstance(entry, dict) for entry in entries)
    return entries


def _entry_by_id(blueprint: Mapping[str, Any], entry_id: str) -> Mapping[str, Any]:
    matches = [entry for entry in _entries(blueprint) if entry.get("id") == entry_id]
    assert len(matches) == 1
    return matches[0]


def _entry_by_identifier(
    blueprint: Mapping[str, Any], model: str, identifier_name: str, identifier_value: str
) -> Mapping[str, Any]:
    matches = [
        entry
        for entry in _entries(blueprint)
        if entry.get("model") == model
        and isinstance(entry.get("identifiers"), dict)
        and entry["identifiers"].get(identifier_name) == identifier_value
    ]
    assert len(matches) == 1
    return matches[0]


def _attrs(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    attrs = entry["attrs"]
    assert isinstance(attrs, dict)
    return attrs


def test_static_authentik_blueprint_declares_workload_provider_defaults() -> None:
    blueprint = _load_blueprint()

    metadata = blueprint["metadata"]
    assert isinstance(metadata, dict)
    labels = metadata["labels"]
    assert isinstance(labels, dict)
    assert labels["blueprints.goauthentik.io/instantiate"] == "true"

    workload_provider = _entry_by_id(blueprint, "nemo-workload-provider")
    assert workload_provider["model"] == "authentik_providers_oauth2.oauth2provider"
    assert workload_provider["identifiers"] == {"name": "nemo-helix-workload"}
    workload_provider_attrs = _attrs(workload_provider)

    assert workload_provider_attrs["name"] == "nemo-helix-workload"
    assert workload_provider_attrs["client_type"] == "public"
    assert workload_provider_attrs["client_id"] == "nemo-helix-workload"
    assert workload_provider_attrs["access_token_validity"] == "minutes=5"

    cli_provider = _entry_by_id(blueprint, "nemo-cli-provider")
    assert _attrs(cli_provider)["access_token_validity"] == "minutes=2"
    assert 'Use a longer value such as "hours=1"' in BLUEPRINT.read_text(encoding="utf-8")

    workload_application = _entry_by_identifier(blueprint, "authentik_core.application", "slug", "nemo-workload")
    assert workload_application["identifiers"] == {"slug": "nemo-workload"}
    workload_application_attrs = _attrs(workload_application)
    assert workload_application_attrs["name"] == "NeMo Helix Workload Identity"
    assert workload_application_attrs["slug"] == "nemo-workload"
    assert workload_application_attrs["provider"] == TaggedYamlValue("!KeyOf", "nemo-workload-provider")


def test_authentik_blueprint_keeps_human_and_workload_groups_separate() -> None:
    blueprint = _load_blueprint()

    editors_group = _entry_by_id(blueprint, "group-nemo-editors")
    workloads_group = _entry_by_id(blueprint, "group-nemo-workloads")
    human_user = _entry_by_id(blueprint, "nemo-user")
    workload_user = _entry_by_id(blueprint, "svc-nemo")

    assert _attrs(editors_group)["name"] == "nemo-editors"
    assert _attrs(workloads_group)["name"] == "nemo-workloads"
    assert _attrs(human_user)["groups"] == [TaggedYamlValue("!KeyOf", "group-nemo-editors")]
    assert _attrs(workload_user)["groups"] == [TaggedYamlValue("!KeyOf", "group-nemo-workloads")]


def test_authentik_blueprint_reads_workload_identity_password_from_env() -> None:
    blueprint = _load_blueprint()
    legacy_secret = "svc-nemo" + "-token-secret-dev"
    token_identifiers = [
        entry["identifiers"]["identifier"]
        for entry in _entries(blueprint)
        if entry.get("model") == "authentik_core.token"
        and isinstance(entry.get("identifiers"), dict)
        and "identifier" in entry["identifiers"]
    ]
    token_keys = [
        _attrs(entry).get("key") for entry in _entries(blueprint) if entry.get("model") == "authentik_core.token"
    ]

    workload_token = _entry_by_identifier(blueprint, "authentik_core.token", "identifier", "svc-nemo-token")
    workload_token_attrs = _attrs(workload_token)

    assert workload_token_attrs["intent"] == "app_password"
    assert workload_token_attrs["user"] == TaggedYamlValue("!KeyOf", "svc-nemo")
    assert workload_token_attrs["key"] == TaggedYamlValue("!Env", "AUTHENTIK_WORKLOAD_IDENTITY_PASSWORD")
    assert "nemo-user-token" not in token_identifiers
    assert legacy_secret not in token_keys


def test_authentik_blueprint_reads_confidential_callback_from_deployment_environment() -> None:
    blueprint = _load_blueprint()
    user_provider = _entry_by_id(blueprint, "nemo-user-provider")

    assert _attrs(user_provider)["redirect_uris"] == [
        {
            "url": TaggedYamlValue("!Env", "NHX_OIDC_LOGIN_REDIRECT_URI"),
            "matching_mode": "strict",
        }
    ]


def test_authentik_blueprint_declares_e2e_setup_identity_as_test_only() -> None:
    blueprint_text = BLUEPRINT.read_text(encoding="utf-8")
    blueprint = _load_blueprint()

    assert "E2E TEST HARNESS ONLY" in blueprint_text

    setup_user = _entry_by_id(blueprint, "nemo-setup")
    setup_user_attrs = _attrs(setup_user)
    assert setup_user["model"] == "authentik_core.user"
    assert setup_user_attrs["type"] == "service_account"
    assert setup_user_attrs["groups"] == [TaggedYamlValue("!KeyOf", "group-nemo-admins")]

    setup_token = _entry_by_identifier(blueprint, "authentik_core.token", "identifier", "nemo-setup-token")
    setup_token_attrs = _attrs(setup_token)
    assert setup_token_attrs["intent"] == "app_password"
    assert setup_token_attrs["user"] == TaggedYamlValue("!KeyOf", "nemo-setup")
    assert setup_token_attrs["key"] == TaggedYamlValue("!Env", "AUTHENTIK_SETUP_PASSWORD")


def test_authentik_blueprint_has_single_canonical_source() -> None:
    assert BLUEPRINT.exists()
    assert not LEGACY_BLUEPRINT.exists()


# The authentik default blueprint that creates each object nemo.yaml looks up with !Find.
_DEFAULT_BLUEPRINT_PROVIDING = {
    ("authentik_flows.flow", "default-authentication-flow"): "Default - Authentication flow",
    (
        "authentik_flows.flow",
        "default-provider-authorization-implicit-consent",
    ): "Default - Provider authorization flow (implicit consent)",
    ("authentik_flows.flow", "default-provider-invalidation-flow"): "Default - Provider invalidation flow",
    ("authentik_providers_oauth2.scopemapping", "openid"): "System - OAuth2 Provider - Scopes",
    ("authentik_providers_oauth2.scopemapping", "email"): "System - OAuth2 Provider - Scopes",
    ("authentik_providers_oauth2.scopemapping", "offline_access"): "System - OAuth2 Provider - Scopes",
}


def _found_defaults(value: object) -> set[tuple[str, str]]:
    if isinstance(value, TaggedYamlValue):
        if value.tag != "!Find":
            return set()
        model, (_, key) = value.value
        return {(model, key)} if model in {"authentik_flows.flow", "authentik_providers_oauth2.scopemapping"} else set()
    if isinstance(value, Mapping):
        return set().union(*(_found_defaults(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_found_defaults(item) for item in value))
    return set()


def test_authentik_blueprint_applies_the_defaults_it_finds_before_using_them() -> None:
    """A !Find that runs before the worker applies a default blueprint resolves to nothing.

    Required fields then fail the apply, but an optional one such as the brand's
    flow_device_code is left empty and /device answers 404. Applying the default blueprint
    first, as a required meta entry, makes the lookup independent of worker timing.
    """
    entries = _entries(_load_blueprint())
    required_meta_positions = {
        _attrs(entry)["identifiers"]["name"]: position
        for position, entry in enumerate(entries)
        if entry.get("model") == "authentik_blueprints.metaapplyblueprint" and _attrs(entry).get("required") is True
    }

    for position, entry in enumerate(entries):
        for found in _found_defaults(entry.get("attrs")):
            assert found in _DEFAULT_BLUEPRINT_PROVIDING, f"map {found} to the default blueprint that creates it"
            providing = _DEFAULT_BLUEPRINT_PROVIDING[found]
            assert providing in required_meta_positions, f"apply {providing!r} as a required meta entry"
            assert required_meta_positions[providing] < position, f"apply {providing!r} before entry {position}"
