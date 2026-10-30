# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import os
import shutil
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent.parent
HELM_DIR = ROOT / "k8s" / "helm"
HELM_TEMPLATE_TIMEOUT_SECONDS = 60
HELM_SERVER_DRY_RUN_TIMEOUT_SECONDS = 180
CLICKHOUSE_CREDENTIALS_CHECKSUM = "checksum/clickhouse-credentials"
CHART_MANAGED_CLICKHOUSE_SECRET = "nemo-helix-clickhouse"
CLICKHOUSE_PASSWORD_KEY = "password"
STORED_GENERATED_PASSWORD = "already-installed-password"


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


def _sha256_text(value: str) -> str:
    # Reproduces Helm's sha256sum for the pod annotation. This is a checksum, not a password hash.
    # codeql[py/weak-sensitive-data-hashing]
    return hashlib.sha256(value.encode()).hexdigest()


def _clickhouse_name_key_checksum(secret_name: str, password_key: str = CLICKHOUSE_PASSWORD_KEY) -> str:
    return _sha256_text(f"{secret_name}:{password_key}")


def _explicit_password_checksum(password: str) -> str:
    encoded = base64.b64encode(password.encode()).decode()
    return _sha256_text(encoded)


def _pod_annotation(documents: list[dict], *, kind: str, name: str, annotation: str) -> str:
    document = next(
        item for item in documents if item.get("kind") == kind and item.get("metadata", {}).get("name") == name
    )
    annotations = document["spec"]["template"]["metadata"].get("annotations") or {}
    value = annotations.get(annotation)
    assert isinstance(value, str) and value
    return value


def _clickhouse_credential_checksums(documents: list[dict]) -> tuple[str, str]:
    return (
        _pod_annotation(
            documents,
            kind="Deployment",
            name="nemo-helix-api",
            annotation=CLICKHOUSE_CREDENTIALS_CHECKSUM,
        ),
        _pod_annotation(
            documents,
            kind="StatefulSet",
            name=CHART_MANAGED_CLICKHOUSE_SECRET,
            annotation=CLICKHOUSE_CREDENTIALS_CHECKSUM,
        ),
    )


def _kube_context() -> str | None:
    return os.environ.get("NHX_HELM_KUBE_CONTEXT") or None


def _kube_context_args() -> list[str]:
    context = _kube_context()
    if context is None:
        return []
    return ["--context", context]


def _require_helm_cluster() -> None:
    if shutil.which("helm") is None or shutil.which("kubectl") is None:
        pytest.skip("helm and kubectl are required to render against a live lookup")
    probe = subprocess.run(
        ["kubectl", *_kube_context_args(), "--request-timeout=8s", "get", "ns"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if probe.returncode != 0:
        pytest.skip(f"Kubernetes API is not reachable: {probe.stderr.strip()}")


def _helm_server_dry_run(namespace: str, *args: str) -> list[dict]:
    context = _kube_context()
    command = [
        "helm",
        "upgrade",
        "--install",
        "nemo-helix",
        str(HELM_DIR),
        "--namespace",
        namespace,
        "--dry-run=server",
    ]
    if context:
        command.extend(["--kube-context", context])
    command.extend(args)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=HELM_SERVER_DRY_RUN_TIMEOUT_SECONDS,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = completed.stdout
    if "\nMANIFEST:\n" in manifest:
        manifest = manifest.split("\nMANIFEST:\n", 1)[1]
    if "\nNOTES:\n" in manifest:
        manifest = manifest.split("\nNOTES:\n", 1)[0]
    return [document for document in yaml.safe_load_all(manifest) if isinstance(document, dict)]


def _assert_clickhouse_checksums(documents: list[dict], expected: str) -> None:
    api_checksum, clickhouse_checksum = _clickhouse_credential_checksums(documents)
    assert api_checksum == expected
    assert clickhouse_checksum == expected


def _kubectl(*args: str, input_text: str | None = None, check: bool = True) -> None:
    subprocess.run(
        ["kubectl", *_kube_context_args(), *args],
        check=check,
        capture_output=True,
        input=input_text,
        text=True,
        timeout=30,
    )


def _secret_manifest(name: str, key: str, password: str, *, namespace: str, helm_owned: bool) -> str:
    metadata: dict[str, object] = {"name": name}
    if helm_owned:
        metadata["labels"] = {"app.kubernetes.io/managed-by": "Helm"}
        metadata["annotations"] = {
            "meta.helm.sh/release-name": "nemo-helix",
            "meta.helm.sh/release-namespace": namespace,
        }
    return yaml.safe_dump(
        {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": metadata,
            "type": "Opaque",
            "data": {key: base64.b64encode(password.encode()).decode()},
        }
    )


@pytest.fixture
def clickhouse_lookup_namespace() -> Iterator[str]:
    _require_helm_cluster()
    namespace = f"nhx-ch-cs-{uuid.uuid4().hex[:8]}"
    _kubectl("create", "namespace", namespace)
    try:
        _kubectl(
            "--namespace",
            namespace,
            "apply",
            "-f",
            "-",
            input_text=_secret_manifest(
                CHART_MANAGED_CLICKHOUSE_SECRET,
                CLICKHOUSE_PASSWORD_KEY,
                STORED_GENERATED_PASSWORD,
                namespace=namespace,
                helm_owned=True,
            ),
        )
        _kubectl(
            "--namespace",
            namespace,
            "apply",
            "-f",
            "-",
            input_text=_secret_manifest(
                "custom-clickhouse-credentials",
                "custom-key",
                "external-password",
                namespace=namespace,
                helm_owned=False,
            ),
        )
        yield namespace
    finally:
        _kubectl("delete", "namespace", namespace, "--wait=false", check=False)


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
        "api.services={evals,guardrails}",
    )

    args = _api_container(documents)["args"]
    assert "--services=evals,guardrails" in args
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
        "api.extraArgs[0]=--services=evals",
    )

    args = _api_container(documents)["args"]
    assert args.count("--services=evals") == 1
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
        "api.services=evals",
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


def test_generated_clickhouse_credentials_checksum_hashes_secret_name_and_key() -> None:
    documents = _helm_template()

    _assert_clickhouse_checksums(documents, _clickhouse_name_key_checksum(CHART_MANAGED_CLICKHOUSE_SECRET))


def test_empty_clickhouse_password_keeps_generated_credentials_checksum() -> None:
    documents = _helm_template("--set-string", "clickhouse.auth.password=")

    _assert_clickhouse_checksums(documents, _clickhouse_name_key_checksum(CHART_MANAGED_CLICKHOUSE_SECRET))


def test_generated_clickhouse_checksum_hashes_name_before_any_secret_lookup() -> None:
    """The generated-password hash must be chosen before lookup runs.

    `helm template` in CI never populates lookup, so the rendered tests cannot
    tell this helper from the one that hashed the live Secret after install.
    """
    helper = (HELM_DIR / "templates" / "_helpers.tpl").read_text()
    body = helper.split('{{- define "nemo-common.clickhouse.credentialsChecksum" -}}', 1)[1]
    body = body.split('{{- define "', 1)[0]
    name_key_hash = '{{- printf "%s:%s" $secretName $passwordKey | sha256sum -}}'
    assert body.index(name_key_hash) < body.index('lookup "v1" "Secret"')


def test_existing_clickhouse_secret_checksum_hashes_name_and_key_when_secret_is_absent() -> None:
    documents = _helm_template(
        "--set",
        "clickhouse.auth.existingSecret=custom-clickhouse-credentials",
        "--set",
        "clickhouse.auth.existingSecretPasswordKey=custom-key",
    )

    _assert_clickhouse_checksums(
        documents, _clickhouse_name_key_checksum("custom-clickhouse-credentials", "custom-key")
    )


@pytest.mark.skip_in_ci
def test_chart_managed_clickhouse_checksum_ignores_stored_generated_password(
    clickhouse_lookup_namespace: str,
) -> None:
    """A stored generated password must not change the chart-managed checksum.

    `helm template` never calls lookup, so this renders with a server-side dry-run
    against a Secret that an earlier install would already have stored. The final
    render omits auth.password again without writing the rotated password back to
    the cluster, so it checks that the checksum returns to the name/key hash.
    """
    expected = _clickhouse_name_key_checksum(CHART_MANAGED_CLICKHOUSE_SECRET)
    installed = _helm_server_dry_run(clickhouse_lookup_namespace)
    _assert_clickhouse_checksums(installed, expected)

    unrelated_change = _helm_server_dry_run(
        clickhouse_lookup_namespace,
        "--set-string",
        "api.service.annotations.qa=1",
    )
    _assert_clickhouse_checksums(unrelated_change, expected)

    rotated = _helm_server_dry_run(
        clickhouse_lookup_namespace,
        "--set-string",
        "clickhouse.auth.password=rotated-test-password",
    )
    _assert_clickhouse_checksums(rotated, _explicit_password_checksum("rotated-test-password"))

    cleared = _helm_server_dry_run(clickhouse_lookup_namespace)
    _assert_clickhouse_checksums(cleared, expected)


@pytest.mark.skip_in_ci
def test_existing_clickhouse_secret_checksum_hashes_live_password(clickhouse_lookup_namespace: str) -> None:
    documents = _helm_server_dry_run(
        clickhouse_lookup_namespace,
        "--set",
        "clickhouse.auth.existingSecret=custom-clickhouse-credentials",
        "--set",
        "clickhouse.auth.existingSecretPasswordKey=custom-key",
    )

    _assert_clickhouse_checksums(documents, _explicit_password_checksum("external-password"))
