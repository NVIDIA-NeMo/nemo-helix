# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Desired reachability contracts; unsupported configurations are explicitly skipped.

Endpoint builders are production code. Only the host OS/process context and
platform configuration are supplied by fixtures. HTTP and Docker are real.
"""

import pytest
from nemo_agents_plugin.runner.deployments_backend import resolve_agent_gateway_url, rewrite_fabric_config_base_urls
from nemo_helix_plugin.config import Runtime
from nhx.common.config.base import HelixConfig
from nhx.core.jobs.controllers.backends.base import get_job_runtime_shared_envvars

from tests.integration.network_reachability.probe import DockerProbe

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("native_linux_host")]
BASE = "http://127.0.0.1:8080"
MODEL_PATH = "/apis/inference-gateway/v2/workspaces/default/openai/-/v1/chat/completions"
COLLECTOR_GAP = pytest.mark.skipif(
    "not config.getoption('--include-known-network-gaps')",
    reason="Independent host-loopback collectors need a container-reachable URL; arbitrary collector forwarding is outside this fix",
)
JOBS_GAP = pytest.mark.skipif(
    "not config.getoption('--include-known-network-gaps')",
    reason="Bridge Jobs retain host-loopback service URLs; Jobs URL adaptation is outside this fix (default Linux host networking works)",
)


def _fabric(url: str, *, envelope: str, exporter: str) -> dict:
    if exporter == "atif":
        telemetry = {"atif": {"storage": [{"type": "http", "endpoint": url}]}}
    elif exporter == "otlp":
        telemetry = {"opentelemetry": {"endpoints": [{"endpoint": url, "transport": "http_binary"}]}}
    else:
        telemetry = {"atof": {"endpoints": [{"endpoint": url}]}}
    if envelope == "authoring":
        return {"telemetry": telemetry}
    return {"relay": {"observability": telemetry}}


def _gateway(probe: DockerProbe, route: str) -> str:
    # Only use a forwarder supplied by this checkout. Never import the PR's fix
    # into an older revision. Without it, Harbor uses the ordinary Docker URL.
    if route == "harbor" and probe.gateway is not None:
        return probe.gateway
    return resolve_agent_gateway_url(BASE, mode="docker")


def _telemetry_endpoint(config: dict, *, envelope: str, exporter: str) -> str:
    # Read the known fixture field directly, without depending on a helper that
    # only exists in the fix branch. Production code alone decides any rewrite.
    telemetry = config["telemetry"] if envelope == "authoring" else config["relay"]["observability"]
    if exporter == "atif":
        return telemetry["atif"]["storage"][0]["endpoint"]
    if exporter == "otlp":
        return telemetry["opentelemetry"]["endpoints"][0]["endpoint"]
    return telemetry["atof"]["endpoints"][0]["endpoint"]


@pytest.mark.parametrize("hostname", ["127.0.0.1", "host.docker.internal"])
def test_negative_control_bridge_cannot_reach_loopback(network_probe: DockerProbe, hostname: str):
    result = network_probe.post(f"http://{hostname}:8081/control")
    # A DNS failure would not demonstrate the listener binding defect.
    assert result.get("network_error") == "ConnectionRefusedError", result


def test_positive_control_shared_host_namespace(network_probe: DockerProbe):
    network_probe.require_delivery(BASE + "/control", shared_host_namespace=True)


def test_positive_control_explicit_bridge_listener(network_probe: DockerProbe):
    network_probe.require_delivery(network_probe.external + "/control", port=9090)


@pytest.mark.parametrize("route", ["deployment", "harbor"])
def test_agent_model_endpoint(network_probe: DockerProbe, route: str, record_property):
    gateway = _gateway(network_probe, route)
    source = {"models": {"default": {"base_url": BASE + MODEL_PATH}}}
    config = rewrite_fabric_config_base_urls(source, gateway)
    endpoint = config["models"]["default"]["base_url"]
    record_property("selected_endpoint", endpoint)
    network_probe.require_delivery(endpoint)
    assert source["models"]["default"]["base_url"] == BASE + MODEL_PATH


@pytest.mark.parametrize("route", ["deployment", "harbor"])
@pytest.mark.parametrize("envelope", ["authoring", "translated"])
@pytest.mark.parametrize("exporter", ["atif", "otlp", "atof"])
@pytest.mark.parametrize("destination", ["intake", pytest.param("custom-loopback", marks=COLLECTOR_GAP), "external"])
def test_telemetry_endpoint(network_probe: DockerProbe, route, envelope, exporter, destination, record_property):
    if destination == "intake":
        endpoint = BASE + f"/apis/intake/v2/workspaces/default/ingest/{exporter}?probe=1"
        port = 8080
    elif destination == "custom-loopback":
        endpoint = "http://127.0.0.1:4318/v1/traces?probe=1"
        port = 4318
    else:
        endpoint = network_probe.external + "/v1/traces?probe=1"
        port = 9090
    gateway = _gateway(network_probe, route)
    rewritten = rewrite_fabric_config_base_urls(_fabric(endpoint, envelope=envelope, exporter=exporter), gateway)
    endpoint = _telemetry_endpoint(rewritten, envelope=envelope, exporter=exporter)
    record_property("selected_endpoint", endpoint)
    network_probe.require_delivery(endpoint, port=port)


@pytest.mark.parametrize("shared_host_namespace", [True, pytest.param(False, marks=JOBS_GAP)], ids=["host", "bridge"])
@pytest.mark.parametrize(
    "env_key,path",
    [
        ("NHX_BASE_URL", "/apis/models/v1/workspaces/default/models"),
        ("NHX_JOBS_URL", "/apis/jobs/v1/workspaces/default/jobs/probe"),
        ("NHX_FILES_URL", "/apis/files/v1/workspaces/default/files/probe"),
    ],
    ids=["model-discovery", "job-callback", "file-transfer"],
)
def test_jobs_runtime_endpoints(network_probe: DockerProbe, env_key, path, shared_host_namespace, record_property):
    config = HelixConfig(base_url=BASE, runtime=Runtime.NONE, service_discovery={}, loopback_address=None)
    endpoint = get_job_runtime_shared_envvars(config)[env_key] + path
    record_property("selected_endpoint", endpoint)
    network_probe.require_delivery(endpoint, shared_host_namespace=shared_host_namespace)


@pytest.mark.parametrize("envelope", ["authoring", "translated"])
@pytest.mark.parametrize("exporter", ["atif", "otlp", "atof"])
def test_colocated_collector_loopback_stays_reachable(network_probe: DockerProbe, envelope, exporter):
    """A collector sharing the caller's namespace must retain its loopback URL."""
    endpoint = "http://127.0.0.1:4318/v1/traces"
    config = rewrite_fabric_config_base_urls(
        _fabric(endpoint, envelope=envelope, exporter=exporter), resolve_agent_gateway_url(BASE, mode="docker")
    )
    selected = _telemetry_endpoint(config, envelope=envelope, exporter=exporter)
    assert selected == endpoint
    network_probe.require_delivery(selected, shared_host_namespace=True, port=4318)


def test_direct_provider_origin_is_preserved(network_probe: DockerProbe):
    endpoint = network_probe.external + "/v1/chat/completions"
    config = rewrite_fabric_config_base_urls(
        {"models": {"default": {"base_url": endpoint}}}, resolve_agent_gateway_url(BASE, mode="docker")
    )
    assert config["models"]["default"]["base_url"] == endpoint
    network_probe.require_delivery(config["models"]["default"]["base_url"], port=9090)


def test_jobs_explicit_files_service_origin_is_preserved(network_probe: DockerProbe):
    config = HelixConfig(base_url=BASE, runtime=Runtime.NONE, service_discovery={"files": network_probe.external})
    endpoint = get_job_runtime_shared_envvars(config)["NHX_FILES_URL"] + "/files/probe"
    assert endpoint == network_probe.external + "/files/probe"
    network_probe.require_delivery(endpoint, port=9090)
