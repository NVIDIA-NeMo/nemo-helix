# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml

from tests.auth_idp.prepare_compose_config import render_authentik_compose_e2e_config


def test_render_authentik_compose_e2e_config_sets_dynamic_docker_resources(tmp_path: Path) -> None:
    source = Path("contrib/auth/authentik/config/platform-compose-authentik.yaml")
    output = tmp_path / "platform.yaml"

    render_authentik_compose_e2e_config(
        source,
        output,
        workload_network="authentik-e2e-test-workload",
        gateway_tls_volume="authentik-e2e-test-gateway-tls",
        gateway_port=19080,
    )

    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    deployment_executor = config["deployments"]["executors"][0]
    job_executor = config["jobs"]["executors"][0]

    assert deployment_executor["config"]["network"] == "authentik-e2e-test-workload"
    assert deployment_executor["config"]["additional_volume_mounts"][0]["volume_name"] == (
        "authentik-e2e-test-gateway-tls"
    )
    assert job_executor["config"]["storage"]["additional_volume_mounts"][0]["volume_name"] == (
        "authentik-e2e-test-gateway-tls"
    )
    assert config["platform"]["advertised_base_url"] == "https://127.0.0.1:19080"
    assert config["auth"]["oidc"]["public_client"]["token_endpoint"] == ("https://127.0.0.1:19080/application/o/token/")
    assert config["auth"]["oidc"]["public_client"]["server_side_sessions"] is False
    assert config["auth"]["oidc"]["server_sessions"] == {"encryption_key_env_var": "NHX_AUTH_SESSION_ENCRYPTION_KEY"}
    assert config["auth"]["oidc"]["confidential_client"]["token_endpoint"] == (
        "http://authentik-server:9000/application/o/token/"
    )
