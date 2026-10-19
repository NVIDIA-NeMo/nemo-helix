# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Additional main-branch consumers of host URLs; transport contracts only."""

import pytest
from nemo_deployments_plugin.backends.openshell.config import HelixEgressConfig
from nemo_helix_plugin.config import Runtime
from nhx.common.config.base import HelixConfig
from nhx.core.jobs.controllers.backends.base import get_job_runtime_shared_envvars, get_logs_endpoint_from_fileset
from nhx.core.models.controllers.backends.deployments_plugin import compiler
from nhx.core.models.controllers.backends.deployments_plugin.resolve import rewrite_loopback_for_docker_container

from tests.integration.network_reachability.probe import DockerProbe

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("native_linux_host")]
BASE = "http://127.0.0.1:8080"
GAP = pytest.mark.skipif(
    "not config.getoption('--include-known-network-gaps')",
    reason="Bridge Jobs retain host-loopback models/secrets/logs URLs; Jobs URL adaptation is outside this fix (default Linux host networking works)",
)


@pytest.mark.parametrize("origin", ["loopback", "external"])
def test_model_weights_files_endpoint(network_probe: DockerProbe, origin, record_property):
    base = BASE if origin == "loopback" else network_probe.external
    endpoint = rewrite_loopback_for_docker_container(base + "/apis/files/v2/hf/probe", runtime=Runtime.DOCKER)
    record_property("selected_endpoint", endpoint)
    network_probe.require_delivery(endpoint, port=8080 if origin == "loopback" else 9090)


def test_model_lora_sidecar_platform_endpoint(network_probe: DockerProbe, monkeypatch, record_property):
    config = HelixConfig(base_url=BASE, runtime=Runtime.NONE, service_discovery={}, loopback_address=None)
    monkeypatch.setattr(compiler, "get_platform_config", lambda: config)
    endpoint = compiler._container_reachable_platform_base_url(runtime=Runtime.DOCKER) + "/apis/models/v1/probe"
    record_property("selected_endpoint", endpoint)
    network_probe.require_delivery(endpoint)


@pytest.mark.parametrize("shared_host_namespace", [True, pytest.param(False, marks=GAP)], ids=["host", "bridge"])
@pytest.mark.parametrize("service", ["models", "secrets", "logs"])
def test_additional_jobs_endpoints(network_probe: DockerProbe, service, shared_host_namespace, record_property):
    config = HelixConfig(base_url=BASE, runtime=Runtime.NONE, service_discovery={}, loopback_address=None)
    if service == "logs":
        endpoint = get_logs_endpoint_from_fileset(config, "default", "probe")
    else:
        endpoint = get_job_runtime_shared_envvars(config)[f"NHX_{service.upper()}_URL"] + f"/apis/{service}/v1/probe"
    record_property("selected_endpoint", endpoint)
    network_probe.require_delivery(endpoint, shared_host_namespace=shared_host_namespace)


def test_openshell_selected_host_endpoint(network_probe: DockerProbe, record_property):
    # Only the chosen destination is probed, not OpenShell's DNS, proxy or policy.
    egress = HelixEgressConfig()
    endpoint = f"http://{egress.host}:{egress.port}/apis/inference-gateway/v2/probe"
    record_property("selected_endpoint", endpoint)
    network_probe.require_delivery(endpoint)


def test_hardener_selected_host_endpoint(network_probe: DockerProbe, record_property):
    from nemo_agent_hardener_plugin import agent_resolver

    # This main-only helper does not exist on the release/0.8 PR branch.
    select = getattr(agent_resolver, "sandbox_reachable", None)
    if select is None:
        pytest.skip("This revision predates Agent Hardener's sandbox_reachable helper")
    endpoint = select(BASE + "/apis/inference-gateway/v2/probe")
    record_property("selected_endpoint", endpoint)
    network_probe.require_delivery(endpoint)
