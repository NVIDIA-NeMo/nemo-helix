# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml

from tests.auth_idp.providers import load_provider_names_by_mode

REQUIRED_STRIPPED_HEADERS = {
    "x-nhx-principal-id",
    "x-nhx-actor-account-id",
    "x-nhx-principal-email",
    "x-nhx-principal-groups",
    "x-nhx-actor-aliases",
    "x-nhx-principal-on-behalf-of",
    "x-nhx-subject-account-id",
    "x-nhx-subject-aliases",
    "x-nhx-principal-on-behalf-of-email",
    "x-nhx-principal-on-behalf-of-groups",
    "x-nhx-scopes",
}


def _lua_removed_headers(http_filters: list[dict]) -> set[str]:
    removed_headers = set()
    for http_filter in http_filters:
        if http_filter.get("name") != "envoy.filters.http.lua":
            continue
        inline_code = http_filter.get("typed_config", {}).get("inline_code", "").lower()
        for header in REQUIRED_STRIPPED_HEADERS:
            if f'headers:remove("{header}")' in inline_code or f"headers:remove('{header}')" in inline_code:
                removed_headers.add(header)
    return removed_headers


def test_compose_backed_idp_providers_are_discovered_from_manifests():
    providers = load_provider_names_by_mode("compose-ci")
    assert providers
    for provider in providers:
        assert Path(f"contrib/auth/{provider}/gateway/envoy.yaml").exists()


def test_all_compose_backed_gateways_strip_trusted_identity_headers():
    for provider in load_provider_names_by_mode("compose-ci"):
        config = yaml.safe_load(Path(f"contrib/auth/{provider}/gateway/envoy.yaml").read_text())
        typed_config = config["static_resources"]["listeners"][0]["filter_chains"][0]["filters"][0]["typed_config"]
        virtual_host = typed_config["route_config"]["virtual_hosts"][0]
        stripped_headers = set(virtual_host.get("request_headers_to_remove", []))
        stripped_headers.update(_lua_removed_headers(typed_config["http_filters"]))
        assert REQUIRED_STRIPPED_HEADERS.issubset(stripped_headers)


def test_all_compose_backed_gateways_define_minimal_envoy_runtime():
    for provider in load_provider_names_by_mode("compose-ci"):
        config = yaml.safe_load(Path(f"contrib/auth/{provider}/gateway/envoy.yaml").read_text())
        listener = config["static_resources"]["listeners"][0]
        cluster = config["static_resources"]["clusters"][0]
        assert listener["name"]
        assert cluster["name"] == "nemo"
