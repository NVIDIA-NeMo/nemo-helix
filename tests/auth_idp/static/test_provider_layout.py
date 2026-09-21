# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

from tests.auth_idp.providers import load_provider_configs_by_mode, load_provider_names_by_mode

pytestmark = [pytest.mark.auth_idp]

AUTH_CALLOUT_RESPONSE_PRINCIPAL_HEADERS = {
    "x-nmp-principal-id",
    "x-nmp-actor-account-id",
    "x-nmp-principal-email",
    "x-nmp-principal-groups",
    "x-nmp-actor-aliases",
    "x-nmp-principal-on-behalf-of",
    "x-nmp-subject-account-id",
    "x-nmp-subject-aliases",
    "x-nmp-principal-on-behalf-of-email",
    "x-nmp-principal-on-behalf-of-groups",
    "x-nmp-scopes",
}
SEALED_EXTERNAL_PRINCIPAL_HEADERS = AUTH_CALLOUT_RESPONSE_PRINCIPAL_HEADERS


def _exact_header_patterns(patterns: list[dict]) -> set[str]:
    return {pattern["exact"] for pattern in patterns if "exact" in pattern}


def test_provider_name_discovery_does_not_require_grant_secret(monkeypatch):
    monkeypatch.delenv("AUTHENTIK_WORKLOAD_IDENTITY_PASSWORD", raising=False)

    assert "authentik" in load_provider_names_by_mode("compose-ci")


def test_compose_backed_providers_ship_required_assets(monkeypatch):
    monkeypatch.setenv("AUTHENTIK_WORKLOAD_IDENTITY_PASSWORD", "shared-secret")
    for provider in load_provider_configs_by_mode("compose-ci"):
        root = Path(f"contrib/auth/{provider.name}")
        assert provider.compose_file is not None
        assert provider.compose_file.exists()
        assert (root / "gateway").exists()
        assert (root / "README.md").exists()
        assert (root / "manifest.yaml").exists()


def test_reference_only_providers_do_not_require_compose(monkeypatch):
    monkeypatch.setenv("AUTHENTIK_WORKLOAD_IDENTITY_PASSWORD", "shared-secret")
    for provider in load_provider_names_by_mode("reference-only"):
        root = Path(f"contrib/auth/{provider}")
        assert (root / "README.md").exists()
        assert not (root / "docker-compose.yml").exists()


def test_authentik_compose_disables_model_provider_seed_without_ngc_key():
    compose = yaml.safe_load(Path("contrib/auth/authentik/compose/docker-compose.yml").read_text())
    nemo_service = compose["services"]["nemo"]
    nemo_env = nemo_service["environment"]

    assert nemo_env["NMP_SEED_ON_STARTUP"] == "true"
    assert nemo_env["NMP_PLATFORM_SEED_MODEL_PROVIDER_ENABLED"] == "false"
    assert "ports" not in nemo_service


def test_authentik_compose_runs_required_services_and_controllers():
    compose = yaml.safe_load(Path("contrib/auth/authentik/compose/docker-compose.yml").read_text())

    assert compose["services"]["nemo"]["command"] == [
        "--host",
        "0.0.0.0",
        "--services",
        "auth,models,files,inference-gateway,jobs,secrets,entities,deployments",
        "--controllers",
        "jobs,models,entities,deployments",
    ]


def test_authentik_compose_defaults_support_direct_docker_compose_start():
    compose_path = Path("contrib/auth/authentik/compose/docker-compose.yml")
    compose_text = compose_path.read_text(encoding="utf-8")
    compose = yaml.safe_load(compose_text)

    password_default = "${AUTHENTIK_WORKLOAD_IDENTITY_PASSWORD:-svc-nemo-token-secret-dev}"
    blueprint_mount = "${AUTHENTIK_BLUEPRINT_DIR:-../helm/files/blueprints}:/blueprints/custom:ro"

    assert compose["name"] == "${COMPOSE_PROJECT_NAME:-nemo-platform-authentik}"
    assert compose["x-authentik-env"]["AUTHENTIK_WORKLOAD_IDENTITY_PASSWORD"] == password_default
    assert compose["services"]["nemo"]["environment"]["AUTHENTIK_WORKLOAD_IDENTITY_PASSWORD"] == password_default
    assert (
        compose["services"]["nemo"]["environment"]["NMP_AUTH_TOKEN_SIGNING__PRIVATE_KEY_FILE"]
        == "/var/run/secrets/nemo-platform/workload-token-signing/private-key.pem"
    )
    assert (
        "../.generated/workload-token-private-key.pem:"
        "/var/run/secrets/nemo-platform/workload-token-signing/private-key.pem:ro"
    ) in compose["services"]["nemo"]["volumes"]
    assert "../gateway/envoy.yaml:/etc/envoy/envoy.yaml:ro" in compose["services"]["gateway"]["volumes"]
    assert (
        "${AUTHENTIK_GATEWAY_TLS_DIR:-../.generated/gateway-tls}:/source/tls:ro"
        in compose["services"]["gateway-tls-init"]["volumes"]
    )
    assert blueprint_mount not in compose["services"]["authentik-server"]["volumes"]
    assert blueprint_mount not in compose["services"]["authentik-worker"]["volumes"]
    blueprint_init = compose["services"]["authentik-blueprint-init"]
    assert blueprint_init["image"] == "ghcr.io/goauthentik/server:${AUTHENTIK_TAG:-2024.12}"
    assert blueprint_init["environment"] == compose["x-authentik-env"]
    assert blueprint_mount in blueprint_init["volumes"]
    assert blueprint_init["depends_on"]["authentik-server"]["condition"] == "service_healthy"
    assert blueprint_init["depends_on"]["authentik-worker"]["condition"] == "service_healthy"
    assert blueprint_init["entrypoint"][:2] == ["sh", "-euc"]
    blueprint_script = blueprint_init["entrypoint"][2]
    assert "ak apply_blueprint /blueprints/custom/nemo.yaml" in blueprint_script
    assert 'while [ "$$attempt" -le 30 ]' in blueprint_script
    assert "attempt $$attempt/30" in blueprint_script
    assert "attempt=$$((attempt + 1))" in blueprint_script
    assert "sleep 2" in blueprint_script
    assert "./.generated/blueprints" not in compose_text


def test_authentik_compose_platform_config_uses_global_docker_workload_identity_switch_only():
    config = yaml.safe_load(Path("contrib/auth/authentik/config/platform-compose-authentik.yaml").read_text())
    workload_executor = next(
        executor
        for executor in config["jobs"]["executors"]
        if executor["backend"] == "docker" and executor["provider"] == "cpu" and executor["profile"] == "workload"
    )

    assert "workload_identity" not in workload_executor["config"]


def test_authentik_compose_uses_liveness_for_container_health_and_routes_status_through_gateway():
    compose = yaml.safe_load(Path("contrib/auth/authentik/compose/docker-compose.yml").read_text())
    envoy = yaml.safe_load(Path("contrib/auth/authentik/gateway/envoy.yaml").read_text())

    nemo_healthcheck = compose["services"]["nemo"]["healthcheck"]["test"]
    assert "http://127.0.0.1:8080/health/live" in nemo_healthcheck[-1]
    assert "/health/ready" not in nemo_healthcheck[-1]

    http_manager = envoy["static_resources"]["listeners"][0]["filter_chains"][0]["filters"][0]["typed_config"]
    routes = http_manager["route_config"]["virtual_hosts"][0]["routes"]
    clusters = envoy["static_resources"]["clusters"]
    route_matches = [route["match"] for route in routes if route.get("route", {}).get("cluster") == "nemo"]
    forwarded_proto_header = [
        {
            "header": {"key": "x-forwarded-proto", "value": "https"},
            "append_action": "OVERWRITE_IF_EXISTS_OR_ADD",
        }
    ]
    nemo_cluster = next(cluster for cluster in clusters if cluster["name"] == "nemo")
    http_options = nemo_cluster["typed_extension_protocol_options"][
        "envoy.extensions.upstreams.http.v3.HttpProtocolOptions"
    ]
    assert http_options["common_http_protocol_options"] == {"idle_timeout": "4s"}
    assert http_options["explicit_http_config"] == {"http_protocol_options": {}}
    gateway_ready_route = next(route for route in routes if route["match"] == {"path": "/health/gateway/ready"})
    health_route = next(route for route in routes if route["match"] == {"prefix": "/health/"})
    assert routes.index(gateway_ready_route) < routes.index(health_route)
    assert gateway_ready_route["direct_response"] == {
        "status": 503,
        "body": {"inline_string": '{"status":"not_ready"}'},
    }
    assert {"prefix": "/health/"} in route_matches
    assert {"path": "/status"} in route_matches
    for match in (
        {"prefix": "/.well-known/nemo-platform/"},
        {"prefix": "/apis/"},
        {"prefix": "/health/"},
        {"path": "/status"},
        {"prefix": "/studio/"},
    ):
        route = next(
            route for route in routes if route.get("route", {}).get("cluster") == "nemo" and route["match"] == match
        )
        assert route["request_headers_to_add"] == forwarded_proto_header

    lua_filter = next(
        filter_config
        for filter_config in http_manager["http_filters"]
        if filter_config["name"] == "envoy.filters.http.lua"
    )
    lua_code = lua_filter["typed_config"]["inline_code"]
    assert 'headers:get(":path") ~= "/health/gateway/ready"' in lua_code
    assert 'gateway_ready_http_call(request_handle, "nemo", "nemo", "/health/ready")' in lua_code
    assert (
        'gateway_ready_http_call(request_handle, "authentik", "authentik-server", '
        '"/application/o/nemo/.well-known/openid-configuration")'
    ) in lua_code
    assert 'headers:remove("x-nmp-authorized")' in lua_code
    assert 'headers:remove("x-nmp-scopes")' in lua_code
    for header in SEALED_EXTERNAL_PRINCIPAL_HEADERS:
        assert f'headers:remove("{header}")' in lua_code

    assert "claim_to_headers" not in yaml.safe_dump(http_manager)
    assert all(
        filter_config["name"] != "envoy.filters.http.jwt_authn" for filter_config in http_manager["http_filters"]
    )

    ext_authz_filter = next(
        filter_config
        for filter_config in http_manager["http_filters"]
        if filter_config["name"] == "envoy.filters.http.ext_authz"
    )
    ext_authz = ext_authz_filter["typed_config"]
    assert ext_authz["@type"] == "type.googleapis.com/envoy.extensions.filters.http.ext_authz.v3.ExtAuthz"
    assert ext_authz["transport_api_version"] == "V3"
    assert ext_authz["failure_mode_allow"] is False
    assert ext_authz["http_service"]["path_prefix"] == "/apis/auth/ext-authz"
    assert ext_authz["http_service"]["server_uri"] == {
        "uri": "http://nemo:8080",
        "cluster": "nemo",
        "timeout": "5s",
    }
    allowed_headers = ext_authz["http_service"]["authorization_response"]["allowed_upstream_headers"]["patterns"]
    assert AUTH_CALLOUT_RESPONSE_PRINCIPAL_HEADERS.issubset(_exact_header_patterns(allowed_headers))

    protected_api_route = next(route for route in routes if route["match"] == {"prefix": "/apis/"})
    assert "typed_per_filter_config" not in protected_api_route

    public_authenticate_route = next(route for route in routes if route["match"] == {"path": "/apis/auth/authenticate"})
    assert public_authenticate_route["typed_per_filter_config"]["envoy.filters.http.ext_authz"] == {
        "@type": "type.googleapis.com/envoy.extensions.filters.http.ext_authz.v3.ExtAuthzPerRoute",
        "disabled": True,
    }
    public_ext_authz_route = next(route for route in routes if route["match"] == {"prefix": "/apis/auth/ext-authz"})
    assert public_ext_authz_route["typed_per_filter_config"]["envoy.filters.http.ext_authz"] == {
        "@type": "type.googleapis.com/envoy.extensions.filters.http.ext_authz.v3.ExtAuthzPerRoute",
        "disabled": True,
    }
    assert routes.index(public_ext_authz_route) < routes.index(protected_api_route)


def test_authentik_compose_mounts_workload_token_signing_key():
    compose = yaml.safe_load(Path("contrib/auth/authentik/compose/docker-compose.yml").read_text())
    config = yaml.safe_load(Path("contrib/auth/authentik/config/platform-compose-authentik.yaml").read_text())

    key_mount = (
        "../.generated/workload-token-private-key.pem:"
        "/var/run/secrets/nemo-platform/workload-token-signing/private-key.pem:ro"
    )

    assert key_mount in compose["services"]["nemo"]["volumes"]
    assert (
        config["auth"]["token_signing"]["private_key_file"]
        == "/var/run/secrets/nemo-platform/workload-token-signing/private-key.pem"
    )
    assert "workload_token_private_key_file" not in config["auth"]["oidc"]


def test_authentik_compose_uses_https_gateway_for_workloads():
    compose = yaml.safe_load(Path("contrib/auth/authentik/compose/docker-compose.yml").read_text())
    config = yaml.safe_load(Path("contrib/auth/authentik/config/platform-compose-authentik.yaml").read_text())
    nemo = compose["services"]["nemo"]
    gateway = compose["services"]["gateway"]
    gateway_tls_init = compose["services"]["gateway-tls-init"]
    blueprint_init = compose["services"]["authentik-blueprint-init"]

    assert config["platform"]["base_url"] == "https://nemo-gateway:8080"
    assert config["auth"]["policy_decision_point_base_url"] == "http://127.0.0.1:8080"
    assert nemo["environment"]["NMP_AUTH_POLICY_DECISION_POINT_BASE_URL"] == "http://127.0.0.1:8080"
    assert "loopback_address" not in config["platform"]
    assert "service_discovery" not in config["platform"]
    assert config["auth"]["oidc"]["token_endpoint"] == "https://127.0.0.1:18080/application/o/token/"
    assert config["auth"]["token_signing"]["issuer"] == "https://nemo-gateway:8080/apis/auth"
    assert config["auth"]["token_signing"]["key_id"] == "nemo-platform-signing"
    assert config["auth"]["access_keys"]["enabled"] is True
    assert "workload_token_issuer" not in config["auth"]["oidc"]
    assert config["auth"]["oidc"]["workload_token_endpoint"] == "https://nemo-gateway:8080/apis/auth/token"
    assert (
        "https://nemo-gateway:8080/application/o/nemo-workload/" in config["auth"]["oidc"]["workload_subject_issuers"]
    )
    workload_executor = next(
        executor
        for executor in config["jobs"]["executors"]
        if executor["provider"] == "cpu" and executor["profile"] == "workload"
    )
    assert "workload_identity" not in workload_executor["config"]
    assert set(nemo["networks"]) == {"nemo-internal"}
    assert "nemo-direct" not in yaml.safe_dump(nemo)
    assert nemo["depends_on"]["gateway-tls-init"]["condition"] == "service_completed_successfully"
    assert gateway["networks"]["nemo-internal"]["aliases"] == ["nemo-gateway"]
    assert gateway["networks"]["workload"]["aliases"] == ["nemo-gateway"]
    assert gateway["depends_on"]["gateway-tls-init"]["condition"] == "service_completed_successfully"
    assert gateway["depends_on"]["authentik-blueprint-init"]["condition"] == "service_completed_successfully"
    assert "gateway-tls:/etc/envoy/tls:ro" in gateway["volumes"]
    assert blueprint_init["networks"] == ["nemo-internal"]
    assert gateway_tls_init["image"] == (
        "docker.io/library/busybox:1.37.0@sha256:9532d8c39891ca2ecde4d30d7710e01fb739c87a8b9299685c63704296b16028"
    )
    assert gateway_tls_init["user"] == "0:0"
    assert gateway_tls_init["entrypoint"][:2] == ["sh", "-euc"]
    assert "chown 101:101 /target/tls/tls.crt /target/tls/tls.key" in gateway_tls_init["entrypoint"][2]
    assert "chmod 600 /target/tls/tls.key" in gateway_tls_init["entrypoint"][2]
    assert compose["volumes"]["gateway-tls"]["name"] == "${AUTHENTIK_GATEWAY_TLS_VOLUME:-authentik_gateway_tls}"
    assert "gateway-tls:/etc/nmp/gateway-tls:ro" in nemo["volumes"]
    assert nemo["environment"]["SSL_CERT_FILE"] == "/etc/nmp/gateway-tls/tls.crt"
    assert nemo["environment"]["REQUESTS_CA_BUNDLE"] == "/etc/nmp/gateway-tls/tls.crt"


def test_authentik_compose_mounts_gateway_ca_into_docker_workloads():
    config = yaml.safe_load(Path("contrib/auth/authentik/config/platform-compose-authentik.yaml").read_text())
    gateway_tls_volume_name = "authentik_gateway_tls"

    workload_executor = next(
        executor
        for executor in config["jobs"]["executors"]
        if executor["provider"] == "cpu" and executor["profile"] == "workload"
    )
    docker_config = workload_executor["config"]

    assert docker_config["env"] == {
        "SSL_CERT_FILE": "/etc/nmp/gateway-tls/tls.crt",
        "REQUESTS_CA_BUNDLE": "/etc/nmp/gateway-tls/tls.crt",
    }
    assert docker_config["storage"]["additional_volume_mounts"] == [
        {
            "volume_name": gateway_tls_volume_name,
            "mount_path": "/etc/nmp/gateway-tls",
        }
    ]


def test_authentik_runtimes_declare_platform_access_key_capability():
    manifest = yaml.safe_load(Path("contrib/auth/authentik/manifest.yaml").read_text())

    runtimes = {runtime["id"]: runtime for runtime in manifest["test_runtimes"]}

    assert "platform_access_keys" in runtimes["authentik-compose"]["capabilities"]
    assert "platform_access_keys" in runtimes["authentik-kubernetes"]["capabilities"]
