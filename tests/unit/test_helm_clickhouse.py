# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent.parent
HELM_DIR = ROOT / "k8s" / "helm"
HELM_TEMPLATE_TIMEOUT_SECONDS = 60


def _helm_template(*args: str) -> list[dict]:
    if shutil.which("helm") is None:
        pytest.skip("helm is required to render the NeMo Helix chart")

    completed = subprocess.run(
        ["helm", "template", "nemo-helix", str(HELM_DIR), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=HELM_TEMPLATE_TIMEOUT_SECONDS,
    )
    return [document for document in yaml.safe_load_all(completed.stdout) if document]


def _helm_template_failure(*args: str) -> str:
    if shutil.which("helm") is None:
        pytest.skip("helm is required to render the NeMo Helix chart")

    completed = subprocess.run(
        ["helm", "template", "nemo-helix", str(HELM_DIR), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=HELM_TEMPLATE_TIMEOUT_SECONDS,
    )
    assert completed.returncode != 0, completed.stdout
    return completed.stderr


def _clickhouse_resources(documents: list[dict]) -> list[dict]:
    return [
        document
        for document in documents
        if document.get("metadata", {}).get("name") == "nemo-helix-clickhouse"
        or document.get("metadata", {}).get("labels", {}).get("app.kubernetes.io/component") == "clickhouse"
    ]


def _api_container(documents: list[dict]) -> dict:
    deployment = next(
        document
        for document in documents
        if document["kind"] == "Deployment" and document["metadata"]["name"] == "nemo-helix-api"
    )
    return deployment["spec"]["template"]["spec"]["containers"][0]


def _controller_container(documents: list[dict]) -> dict:
    deployment = next(
        document
        for document in documents
        if document["kind"] == "Deployment" and document["metadata"]["name"] == "nemo-helix-core-controller"
    )
    return deployment["spec"]["template"]["spec"]["containers"][0]


def _envoy_config(documents: list[dict]) -> dict:
    config_map = next(
        document
        for document in documents
        if document["kind"] == "ConfigMap" and document["metadata"]["name"] == "nemo-helix-envoy"
    )
    return yaml.safe_load(config_map["data"]["envoy.yaml"])


def _env_by_name(container: dict) -> dict[str, dict]:
    return {env["name"]: env for env in container["env"]}


def test_default_intake_selection_renders_embedded_clickhouse_dependency() -> None:
    documents = _helm_template()

    clickhouse_resources = _clickhouse_resources(documents)
    assert {document["kind"] for document in clickhouse_resources} >= {"Secret", "Service", "StatefulSet"}
    assert "--service-group=all" in _api_container(documents)["args"]

    env = _env_by_name(_api_container(documents))
    assert env["NHX_INTAKE_CLICKHOUSE_URL"]["value"] == "http://nemo-helix-clickhouse:8123"


def test_core_service_group_can_skip_embedded_clickhouse_dependency() -> None:
    documents = _helm_template(
        "--set",
        "clickhouse.enabled=false",
        "--set",
        "externalClickhouse.host=unused-clickhouse",
        "--set",
        "externalClickhouse.existingSecret=shared-postgresql",
        "--set",
        "externalClickhouse.existingSecretPasswordKey=nemo-password",
        "--set",
        "api.serviceGroup=core",
    )

    assert _clickhouse_resources(documents) == []
    args = _api_container(documents)["args"]
    assert "--service-group=core" in args
    assert "--service-group=all" not in args


def test_api_services_override_default_service_group() -> None:
    documents = _helm_template(
        "--set",
        "api.services={evaluator,guardrails}",
    )

    args = _api_container(documents)["args"]
    assert "--services=evaluator,guardrails" in args
    assert not any(arg.startswith("--service-group=") for arg in args)


def test_legacy_api_extra_args_service_selection_suppresses_default_service_group() -> None:
    documents = _helm_template(
        "--set-string",
        "api.extraArgs[0]=--service-group=core",
    )

    args = _api_container(documents)["args"]
    assert args.count("--service-group=core") == 1
    assert "--service-group=all" not in args


def test_legacy_api_extra_args_services_selection_suppresses_default_service_group() -> None:
    documents = _helm_template(
        "--set-string",
        "api.extraArgs[0]=--services=evaluator",
    )

    args = _api_container(documents)["args"]
    assert args.count("--services=evaluator") == 1
    assert not any(arg.startswith("--service-group=") for arg in args)


def test_default_controller_selection_renders_controller_group_all() -> None:
    documents = _helm_template()

    assert "--controller-group=all" in _controller_container(documents)["args"]


def test_controller_group_can_be_configured() -> None:
    documents = _helm_template(
        "--set",
        "core.controller.controllerGroup=core",
    )

    args = _controller_container(documents)["args"]
    assert "--controller-group=core" in args
    assert "--controller-group=all" not in args


def test_controllers_override_default_controller_group() -> None:
    documents = _helm_template(
        "--set",
        "core.controller.controllers={models,jobs}",
    )

    args = _controller_container(documents)["args"]
    assert "--controllers=models,jobs" in args
    assert not any(arg.startswith("--controller-group=") for arg in args)


def test_api_services_must_be_a_list() -> None:
    stderr = _helm_template_failure(
        "--set-string",
        "api.services=evaluator",
    )

    assert "api.services must be a list when set" in stderr


def test_falsy_api_services_must_be_a_list() -> None:
    stderr = _helm_template_failure(
        "--set",
        "api.services=false",
    )

    assert "api.services must be a list when set" in stderr


def test_controllers_must_be_a_list() -> None:
    stderr = _helm_template_failure(
        "--set-string",
        "core.controller.controllers=models",
    )

    assert "core.controller.controllers must be a list when set" in stderr


def test_falsy_controllers_must_be_a_list() -> None:
    stderr = _helm_template_failure(
        "--set",
        "core.controller.controllers=false",
    )

    assert "core.controller.controllers must be a list when set" in stderr


def test_legacy_controller_extra_args_selection_suppresses_default_controller_group() -> None:
    documents = _helm_template(
        "--set-string",
        "core.controller.extraArgs[0]=--controller-group=core",
    )

    args = _controller_container(documents)["args"]
    assert args.count("--controller-group=core") == 1
    assert "--controller-group=all" not in args


def test_legacy_controller_extra_args_controllers_selection_suppresses_default_controller_group() -> None:
    documents = _helm_template(
        "--set-string",
        "core.controller.extraArgs[0]=--controllers=models",
    )

    args = _controller_container(documents)["args"]
    assert args.count("--controllers=models") == 1
    assert not any(arg.startswith("--controller-group=") for arg in args)


def test_external_clickhouse_disables_embedded_dependency() -> None:
    documents = _helm_template(
        "--set",
        "clickhouse.enabled=false",
        "--set",
        "externalClickhouse.host=clickhouse.example.internal",
        "--set",
        "externalClickhouse.existingSecret=clickhouse-credentials",
        "--set",
        "externalClickhouse.existingSecretPasswordKey=password",
    )

    assert _clickhouse_resources(documents) == []

    env = _env_by_name(_api_container(documents))
    assert env["NHX_INTAKE_CLICKHOUSE_URL"]["value"] == "http://clickhouse.example.internal:8123"
    password_ref = env["NHX_INTAKE_CLICKHOUSE_PASSWORD"]["valueFrom"]["secretKeyRef"]
    assert password_ref == {"name": "clickhouse-credentials", "key": "password"}


def test_envoy_backend_cluster_uses_short_upstream_idle_timeout() -> None:
    documents = _helm_template(
        "--set",
        "platformConfig.auth.enabled=true",
    )
    envoy_config = _envoy_config(documents)
    backend_cluster = next(
        cluster for cluster in envoy_config["static_resources"]["clusters"] if cluster["name"] == "backend_cluster"
    )
    http_options = backend_cluster["typed_extension_protocol_options"][
        "envoy.extensions.upstreams.http.v3.HttpProtocolOptions"
    ]

    assert http_options["common_http_protocol_options"] == {"idle_timeout": "4s"}
    assert http_options["explicit_http_config"] == {"http_protocol_options": {}}


def test_envoy_upstream_idle_must_be_less_than_api_keep_alive() -> None:
    stderr = _helm_template_failure(
        "--set",
        "platformConfig.auth.enabled=true",
        "--set",
        "api.server.keepAliveTimeoutSeconds=5",
        "--set",
        "envoyProxy.timeouts.upstreamIdle=5s",
    )

    assert ("envoyProxy.timeouts.upstreamIdle (5s) must be less than api.server.keepAliveTimeoutSeconds (5s)") in stderr


def test_api_deployment_passes_keep_alive_timeout_to_services_run() -> None:
    documents = _helm_template(
        "--set",
        "api.server.keepAliveTimeoutSeconds=9",
    )

    assert "--keep-alive-timeout-seconds=9" in _api_container(documents)["args"]
