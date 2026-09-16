# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import contextlib
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import httpx
import pytest
from nemo_helix import DefaultHttpxClient, NeMoHelix
from nemo_helix_ext.client.tls import NHX_CLIENT_SSL_CERT_FILE_ENVVAR, HttpxTLSConfig

from tests.auth_idp.common import jwt_claims
from tests.auth_idp.runtime_contract import AuthIdpCase, DeploymentWorkloadRuntimeConfig, TokenSet

REPO_ROOT = Path(__file__).resolve().parents[2]
NAMESPACE = os.environ.get("NHX_AUTHENTIK_K8S_NAMESPACE", "nemo-authentik")
HELM_RELEASE = os.environ.get("NHX_AUTHENTIK_K8S_HELM_RELEASE", "authentik-demo")
HELM_CHART = Path("contrib/auth/authentik/helm")
ENVOY_TLS_SECRET = "nemo-helix-envoy-tls"
IN_CLUSTER_ENVOY_BASE_URL = f"https://nemo-helix-envoy.{NAMESPACE}.svc.cluster.local:8080"
DEPLOYMENT_WORKLOAD_CA_BUNDLE_FILE = "/etc/nhx/workload-token-ca/ca.crt"
WORKLOAD_AUDIENCE = "nemo-helix"
WORKLOAD_CLIENT_ID = "nemo-helix-workload"
AUTHENTIK_K8S_WORKLOAD_IDENTITY_PASSWORD = "workload-identity-dev-only"
WORKLOAD_TOKEN_PRIVATE_KEY_FILE_ENV = "NHX_AUTHENTIK_K8S_WORKLOAD_TOKEN_PRIVATE_KEY_FILE"
GATEWAY_PORT_ENV = "NHX_AUTHENTIK_K8S_GATEWAY_PORT"
DISCOVERY_PATH = "/application/o/nemo/.well-known/openid-configuration"
GATEWAY_READY_PATH = "/health/gateway/ready"
TOKEN_EXCHANGE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"
JWT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:jwt"
ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"
T = TypeVar("T")


_DURATION_TOKEN_RE = re.compile(r"(\d+)([hms])")


def _duration_seconds(value: str) -> int:
    if value.isdigit():
        return int(value)

    total = 0
    position = 0
    for match in _DURATION_TOKEN_RE.finditer(value):
        if match.start() != position:
            raise ValueError(f"invalid duration: {value}")
        amount, unit = match.groups()
        total += int(amount) * {"h": 3600, "m": 60, "s": 1}[unit]
        position = match.end()

    if position != len(value) or total == 0:
        raise ValueError(f"invalid duration: {value}")
    return total


# Keep timeouts centralized so slow-cluster tuning is a single, visible edit.
PYTEST_TIMEOUT_SECONDS = 2400
DEFAULT_COMMAND_TIMEOUT_SECONDS = 120
DIAGNOSTIC_COMMAND_TIMEOUT_SECONDS = 60
POD_DISCOVERY_TIMEOUT_SECONDS = 60
CLUSTER_CREATE_TIMEOUT_SECONDS = 360
KIND_CREATE_WAIT_TIMEOUT = "180s"
IMAGE_LOAD_TIMEOUT_SECONDS = 300
CLUSTER_DELETE_TIMEOUT_SECONDS = 180
ROLLOUT_STATUS_TIMEOUT = "240s"
ROLLOUT_COMMAND_TIMEOUT_SECONDS = 300
HELM_WAIT_TIMEOUT_DEFAULT = "20m"
HELM_WAIT_TIMEOUT = os.environ.get("NHX_AUTHENTIK_K8S_HELM_WAIT_TIMEOUT", HELM_WAIT_TIMEOUT_DEFAULT)
HELM_UPGRADE_COMMAND_GRACE_SECONDS = 300
try:
    _helm_wait_seconds = _duration_seconds(HELM_WAIT_TIMEOUT)
except ValueError as exc:
    raise RuntimeError(
        "NHX_AUTHENTIK_K8S_HELM_WAIT_TIMEOUT must be a helm duration such as "
        f"'20m', '1h30m', or a plain number of seconds; got {HELM_WAIT_TIMEOUT!r}"
    ) from exc
HELM_UPGRADE_COMMAND_TIMEOUT_SECONDS = _helm_wait_seconds + HELM_UPGRADE_COMMAND_GRACE_SECONDS
HELM_REPO_TIMEOUT_SECONDS = 60
HELM_DEPENDENCY_TIMEOUT_SECONDS = 300
HTTP_RETRY_TIMEOUT_SECONDS = 180
HTTP_REQUEST_TIMEOUT_SECONDS = 10.0
RETRY_SLEEP_SECONDS = 2.0
SECRET_TIMEOUT_SECONDS = 180
SECRET_GET_TIMEOUT_SECONDS = 30
PORT_FORWARD_READY_TIMEOUT_SECONDS = 30
PORT_FORWARD_HTTP_TIMEOUT_SECONDS = 2.0
PORT_FORWARD_RETRY_SLEEP_SECONDS = 0.5
PORT_FORWARD_TERMINATE_TIMEOUT_SECONDS = 10
SUBJECT_TOKEN_DURATION = "10m"
TOKEN_EXCHANGE_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class Cluster:
    name: str
    runtime: str
    context: str
    kubeconfig: Path | None = None
    cleanup_kubeconfig: bool = False


def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        pytest.skip(f"{name} is required for the Authentik Kubernetes E2E test")


def _run(args: list[str], *, timeout: float = DEFAULT_COMMAND_TIMEOUT_SECONDS) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        command = " ".join(args)
        stdout = completed.stdout[-4000:]
        stderr = completed.stderr[-4000:]
        raise AssertionError(
            f"command failed ({completed.returncode}): {command}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return completed


def _temporary_kubeconfig_path(cluster_name: str) -> Path:
    temp_file = tempfile.NamedTemporaryFile(
        prefix=f"nhx-authentik-{cluster_name}-",
        suffix="-kubeconfig.yaml",
        delete=False,
    )
    temp_file.close()
    return Path(temp_file.name)


def _kubectl_command(context: str, args: list[str], kubeconfig: Path | None = None) -> list[str]:
    command = ["kubectl"]
    if kubeconfig is not None:
        command.extend(["--kubeconfig", str(kubeconfig)])
    command.extend(["--context", context, *args])
    return command


def _helm_command(context: str, args: list[str], kubeconfig: Path | None = None) -> list[str]:
    command = ["helm"]
    if kubeconfig is not None:
        command.extend(["--kubeconfig", str(kubeconfig)])
    command.extend(["--kube-context", context, *args])
    return command


def _write_diagnostic_process(
    log_dir: Path,
    name: str,
    args: list[str],
    *,
    timeout: float = DIAGNOSTIC_COMMAND_TIMEOUT_SECONDS,
) -> None:
    completed = subprocess.run(
        args,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    (log_dir / name).write_text(
        "\n".join(
            (
                f"command: {' '.join(args)}",
                f"exit_code: {completed.returncode}",
                "stdout:",
                completed.stdout,
                "stderr:",
                completed.stderr,
            )
        ),
        encoding="utf-8",
    )


def _diagnostic_output_text(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return output


def _diagnostic_text_tail(path: Path, *, limit: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


def _write_diagnostic_timeout(log_dir: Path, name: str, args: list[str], exc: subprocess.TimeoutExpired) -> None:
    (log_dir / name).write_text(
        "\n".join(
            (
                f"command: {' '.join(args)}",
                f"timeout: {exc.timeout}",
                "stdout:",
                _diagnostic_output_text(exc.stdout),
                "stderr:",
                _diagnostic_output_text(exc.stderr),
            )
        ),
        encoding="utf-8",
    )


def _write_diagnostic_command(
    context: str,
    log_dir: Path,
    name: str,
    args: list[str],
    *,
    kubeconfig: Path | None = None,
    timeout: float = DIAGNOSTIC_COMMAND_TIMEOUT_SECONDS,
) -> None:
    _write_diagnostic_process(log_dir, name, _kubectl_command(context, args, kubeconfig), timeout=timeout)


def _port_forward_log_file(service: str) -> Path | None:
    configured_dir = os.environ.get("NHX_AUTHENTIK_K8S_LOG_DIR")
    if not configured_dir:
        return None
    safe_service = service.replace("/", "_")
    return Path(configured_dir) / f"port-forward-{safe_service}.log"


def _port_forward_exit_message(returncode: int, log_file: Path | None) -> str:
    message = f"kubectl port-forward exited early with {returncode}"
    if log_file is None:
        return message
    log_tail = _diagnostic_text_tail(log_file)
    if not log_tail.strip():
        return message
    return f"{message}\nport-forward log ({log_file}):\n{log_tail}"


def _collect_kubernetes_diagnostics(context: str, cluster_name: str, kubeconfig: Path | None = None) -> Path:
    configured_dir = os.environ.get("NHX_AUTHENTIK_K8S_LOG_DIR")
    log_root = Path(configured_dir) if configured_dir else REPO_ROOT / "docker" / "logs"
    log_dir = log_root / f"k8s-authentik-{cluster_name}"
    log_dir.mkdir(parents=True, exist_ok=True)

    for name, args in {
        "helm-list.txt": _helm_command(context, ["-n", NAMESPACE, "list", "--all"], kubeconfig),
        "helm-status.txt": _helm_command(context, ["-n", NAMESPACE, "status", HELM_RELEASE], kubeconfig),
    }.items():
        with contextlib.suppress(Exception):
            _write_diagnostic_process(log_dir, name, args)

    for name, args in {
        "get-nodes.txt": ["get", "nodes", "-o", "wide"],
        "get-all.txt": ["-n", NAMESPACE, "get", "all", "-o", "wide"],
        "get-pods-json.txt": ["-n", NAMESPACE, "get", "pods", "-o", "json"],
        "events.txt": ["-n", NAMESPACE, "get", "events", "--sort-by=.lastTimestamp"],
        "describe-pods.txt": ["-n", NAMESPACE, "describe", "pods"],
        "describe-blueprint-configmap.txt": ["-n", NAMESPACE, "describe", "configmap/authentik-nemo-blueprint"],
    }.items():
        with contextlib.suppress(Exception):
            _write_diagnostic_command(context, log_dir, name, args, kubeconfig=kubeconfig)

    pod_discovery_args = _kubectl_command(
        context,
        [
            "-n",
            NAMESPACE,
            "get",
            "pods",
            "-o",
            "jsonpath={range .items[*]}{.metadata.name}{'\\n'}{end}",
        ],
        kubeconfig,
    )
    try:
        pods = subprocess.run(
            pod_discovery_args,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=POD_DISCOVERY_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        with contextlib.suppress(Exception):
            _write_diagnostic_timeout(log_dir, "get-pods-for-logs.txt", pod_discovery_args, exc)
        return log_dir

    for pod in pods.stdout.splitlines():
        safe_pod = pod.replace("/", "_")
        with contextlib.suppress(Exception):
            _write_diagnostic_command(
                context,
                log_dir,
                f"logs-{safe_pod}.txt",
                ["-n", NAMESPACE, "logs", pod, "--all-containers", "--timestamps"],
                kubeconfig=kubeconfig,
            )
        with contextlib.suppress(Exception):
            _write_diagnostic_command(
                context,
                log_dir,
                f"logs-{safe_pod}-previous.txt",
                ["-n", NAMESPACE, "logs", pod, "--all-containers", "--previous", "--timestamps"],
                kubeconfig=kubeconfig,
            )

    return log_dir


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _configured_gateway_port() -> int | None:
    configured = os.environ.get(GATEWAY_PORT_ENV)
    if not configured:
        return None
    try:
        port = int(configured)
    except ValueError as exc:
        raise AssertionError(f"{GATEWAY_PORT_ENV} must be an integer TCP port") from exc
    if port < 1 or port > 65535:
        raise AssertionError(f"{GATEWAY_PORT_ENV} must be between 1 and 65535")
    return port


def _platform_image() -> str:
    return os.environ.get(
        "NHX_AUTHENTIK_K8S_PLATFORM_IMAGE",
        f"{os.environ.get('IMAGE_REGISTRY', 'my-registry')}/nhx-api:{os.environ.get('BAKE_TAG', 'local')}",
    )


def _create_cluster(runtime: str, name: str) -> Cluster:
    _require_tool("docker")
    _require_tool("kubectl")
    kubeconfig = _temporary_kubeconfig_path(name)

    try:
        if runtime == "k3d":
            _require_tool("k3d")
            _run(
                [
                    "k3d",
                    "cluster",
                    "create",
                    name,
                    "--wait",
                    "--agents",
                    "0",
                    "--k3s-arg",
                    "--disable=traefik@server:0",
                    "--kubeconfig-update-default=false",
                ],
                timeout=CLUSTER_CREATE_TIMEOUT_SECONDS,
            )
            kubeconfig.write_text(_run(["k3d", "kubeconfig", "get", name]).stdout, encoding="utf-8")
            return Cluster(
                name=name,
                runtime=runtime,
                context=f"k3d-{name}",
                kubeconfig=kubeconfig,
                cleanup_kubeconfig=True,
            )

        if runtime == "kind":
            _require_tool("kind")
            _run(
                [
                    "kind",
                    "create",
                    "cluster",
                    "--name",
                    name,
                    "--kubeconfig",
                    str(kubeconfig),
                    "--wait",
                    KIND_CREATE_WAIT_TIMEOUT,
                ],
                timeout=CLUSTER_CREATE_TIMEOUT_SECONDS,
            )
            return Cluster(
                name=name,
                runtime=runtime,
                context=f"kind-{name}",
                kubeconfig=kubeconfig,
                cleanup_kubeconfig=True,
            )
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            kubeconfig.unlink()
        raise

    raise ValueError(f"unsupported NHX_AUTHENTIK_K8S_RUNTIME={runtime!r}; expected kind or k3d")


def _existing_cluster(runtime: str, name: str) -> Cluster | None:
    _require_tool("kubectl")
    kubeconfig = _temporary_kubeconfig_path(name)
    try:
        if runtime == "k3d":
            _require_tool("k3d")
            kubeconfig.write_text(_run(["k3d", "kubeconfig", "get", name]).stdout, encoding="utf-8")
            return Cluster(
                name=name,
                runtime=runtime,
                context=f"k3d-{name}",
                kubeconfig=kubeconfig,
                cleanup_kubeconfig=True,
            )

        if runtime == "kind":
            _require_tool("kind")
            kubeconfig.write_text(_run(["kind", "get", "kubeconfig", "--name", name]).stdout, encoding="utf-8")
            return Cluster(
                name=name,
                runtime=runtime,
                context=f"kind-{name}",
                kubeconfig=kubeconfig,
                cleanup_kubeconfig=True,
            )
    except AssertionError:
        with contextlib.suppress(FileNotFoundError):
            kubeconfig.unlink()
        return None
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            kubeconfig.unlink()
        raise

    with contextlib.suppress(FileNotFoundError):
        kubeconfig.unlink()
    raise ValueError(f"unsupported NHX_AUTHENTIK_K8S_RUNTIME={runtime!r}; expected kind or k3d")


def _reuse_or_create_cluster(runtime: str, name: str) -> Cluster:
    cluster = _existing_cluster(runtime, name)
    if cluster is not None:
        return cluster
    return _create_cluster(runtime, name)


def _load_platform_image(runtime: str, name: str, image: str) -> None:
    if os.environ.get("NHX_AUTHENTIK_K8S_SKIP_IMAGE_LOAD") == "1":
        return
    if runtime == "k3d":
        _run(["k3d", "image", "import", image, "-c", name], timeout=IMAGE_LOAD_TIMEOUT_SECONDS)
        return
    if runtime == "kind":
        _run(["kind", "load", "docker-image", image, "--name", name], timeout=IMAGE_LOAD_TIMEOUT_SECONDS)
        return
    raise ValueError(f"unsupported NHX_AUTHENTIK_K8S_RUNTIME={runtime!r}; expected kind or k3d")


def _delete_cluster(runtime: str, name: str, kubeconfig: Path | None = None) -> None:
    with contextlib.suppress(Exception):
        if runtime == "k3d":
            _run(["k3d", "cluster", "delete", name], timeout=CLUSTER_DELETE_TIMEOUT_SECONDS)
        elif runtime == "kind":
            command = ["kind", "delete", "cluster", "--name", name]
            if kubeconfig is not None:
                command.extend(["--kubeconfig", str(kubeconfig)])
            _run(command, timeout=CLUSTER_DELETE_TIMEOUT_SECONDS)


def _reuse_context(runtime: str, name: str) -> str:
    if runtime == "k3d":
        return f"k3d-{name}"
    if runtime == "kind":
        return f"kind-{name}"
    raise ValueError(f"unsupported NHX_AUTHENTIK_K8S_RUNTIME={runtime!r}; expected kind or k3d")


def _wait_for_authentik(context: str, kubeconfig: Path | None = None) -> None:
    for deployment in (
        "authentik-server",
        "authentik-worker",
        "nemo-helix-api",
        "nemo-helix-envoy",
    ):
        _run(
            _kubectl_command(
                context,
                [
                    "-n",
                    NAMESPACE,
                    "rollout",
                    "status",
                    f"deploy/{deployment}",
                    f"--timeout={ROLLOUT_STATUS_TIMEOUT}",
                ],
                kubeconfig,
            ),
            timeout=ROLLOUT_COMMAND_TIMEOUT_SECONDS,
        )
    _run(
        _kubectl_command(
            context,
            [
                "-n",
                NAMESPACE,
                "rollout",
                "status",
                "statefulset/shared-postgresql",
                f"--timeout={ROLLOUT_STATUS_TIMEOUT}",
            ],
            kubeconfig,
        ),
        timeout=ROLLOUT_COMMAND_TIMEOUT_SECONDS,
    )


def _helm_upgrade_args(context: str, kubeconfig: Path | None = None) -> list[str]:
    image = _platform_image()
    registry, tag = image.rsplit("/nhx-api:", 1)
    args = _helm_command(
        context,
        [
            "upgrade",
            "--install",
            HELM_RELEASE,
            str(HELM_CHART),
            "--namespace",
            NAMESPACE,
            "--create-namespace",
            "--wait",
            "--wait-for-jobs",
            "--timeout",
            HELM_WAIT_TIMEOUT,
            "--set",
            f"nemo-helix.api.image.repository={registry}/nhx-api",
            "--set",
            f"nemo-helix.api.image.tag={tag}",
            "--set",
            f"nemo-helix.core.image.repository={registry}/nhx-api",
            "--set",
            f"nemo-helix.core.image.tag={tag}",
            "--set-string",
            f"nemo-helix.platformConfig.platform.image_registry={registry}",
            "--set-string",
            f"nemo-helix.platformConfig.platform.image_tag={tag}",
            "--set-string",
            "nemo-helix.platformConfig.auth.access_keys.enabled=true",
        ],
        kubeconfig,
    )
    ngc_existing_secret = os.environ.get("NHX_AUTHENTIK_K8S_NGC_EXISTING_SECRET")
    if ngc_existing_secret:
        args.extend(
            [
                "--set-string",
                f"nemo-helix.existingSecret={ngc_existing_secret}",
            ]
        )
    image_pull_secret = os.environ.get("NHX_AUTHENTIK_K8S_IMAGE_PULL_SECRET")
    if image_pull_secret:
        args.extend(
            [
                "--set-string",
                f"nemo-helix.imagePullSecrets[0].name={image_pull_secret}",
            ]
        )
    gateway_port = _configured_gateway_port()
    if gateway_port is not None:
        args.extend(
            [
                "--set-string",
                f"nemo-helix.authentikPublicGateway.port={gateway_port}",
            ]
        )
    workload_token_private_key = os.environ.get(WORKLOAD_TOKEN_PRIVATE_KEY_FILE_ENV)
    if workload_token_private_key:
        args.extend(
            [
                "--set-file",
                f"workloadTokenSigningKey.privateKeyPem={workload_token_private_key}",
            ]
        )
    return args


def _add_platform_helm_repositories() -> None:
    _run(
        ["helm", "repo", "add", "nvidia", "https://helm.ngc.nvidia.com/nvidia", "--force-update"],
        timeout=HELM_REPO_TIMEOUT_SECONDS,
    )
    _run(
        ["helm", "repo", "add", "authentik", "https://charts.goauthentik.io", "--force-update"],
        timeout=HELM_REPO_TIMEOUT_SECONDS,
    )


def _helm_install_authentik_demo(context: str, kubeconfig: Path | None = None) -> None:
    _require_tool("helm")
    _add_platform_helm_repositories()
    _run(["helm", "dependency", "build", "k8s/helm"], timeout=HELM_DEPENDENCY_TIMEOUT_SECONDS)
    _run(["helm", "dependency", "build", str(HELM_CHART)], timeout=HELM_DEPENDENCY_TIMEOUT_SECONDS)

    _run(_helm_upgrade_args(context, kubeconfig), timeout=HELM_UPGRADE_COMMAND_TIMEOUT_SECONDS)


def _retry_until_timeout(
    attempt: Callable[[float], T | None],
    *,
    timeout: float,
    sleep: float,
    retry_exceptions: tuple[type[Exception], ...],
    timeout_message: str,
    raise_last_retry_error: bool = True,
) -> T:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            result = attempt(remaining)
            if result is not None:
                return result
        except retry_exceptions as exc:
            last_error = exc
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(sleep, remaining))
    if last_error is not None:
        if raise_last_retry_error:
            raise last_error
        raise TimeoutError(timeout_message) from last_error
    raise TimeoutError(timeout_message)


def _get_json_with_retries(
    url: str,
    *,
    timeout: float = HTTP_RETRY_TIMEOUT_SECONDS,
    tls_config: HttpxTLSConfig | None = None,
) -> dict:
    request_tls_config: HttpxTLSConfig = {} if tls_config is None else tls_config

    def get_json(remaining: float) -> dict | None:
        response = httpx.get(url, timeout=min(HTTP_REQUEST_TIMEOUT_SECONDS, remaining), **request_tls_config)
        if response.status_code >= 500:
            return None
        response.raise_for_status()
        return response.json()

    return _retry_until_timeout(
        get_json,
        timeout=timeout,
        sleep=RETRY_SLEEP_SECONDS,
        retry_exceptions=(httpx.HTTPError, ValueError),
        timeout_message=f"timed out waiting for {url}",
    )


def _secret_data(
    context: str,
    secret_name: str,
    key: str,
    *,
    kubeconfig: Path | None = None,
    timeout: float = SECRET_TIMEOUT_SECONDS,
) -> bytes:
    def get_secret(remaining: float) -> bytes:
        secret = json.loads(
            _run(
                _kubectl_command(
                    context,
                    [
                        "-n",
                        NAMESPACE,
                        "get",
                        "secret",
                        secret_name,
                        "-o",
                        "json",
                    ],
                    kubeconfig,
                ),
                timeout=min(SECRET_GET_TIMEOUT_SECONDS, remaining),
            ).stdout
        )
        return base64.b64decode(secret["data"][key])

    return _retry_until_timeout(
        get_secret,
        timeout=timeout,
        sleep=RETRY_SLEEP_SECONDS,
        retry_exceptions=(AssertionError, KeyError, json.JSONDecodeError, ValueError),
        timeout_message=f"timed out waiting for secret {NAMESPACE}/{secret_name} key {key}",
    )


def _start_port_forward_service(
    context: str,
    service: str,
    ca_bundle: Path,
    kubeconfig: Path | None = None,
) -> tuple[str, subprocess.Popen[str]]:
    port = _configured_gateway_port() or _free_port()
    command = _kubectl_command(
        context,
        [
            "-n",
            NAMESPACE,
            "port-forward",
            f"svc/{service}",
            f"{port}:8080",
        ],
        kubeconfig,
    )
    log_file = _port_forward_log_file(service)
    log_handle = None
    stdout = subprocess.DEVNULL
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_file.open("w", encoding="utf-8")
        log_handle.write(f"command: {' '.join(command)}\n\n")
        log_handle.flush()
        stdout = log_handle
    try:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        if log_handle is not None:
            log_handle.close()
    gateway_url = f"https://127.0.0.1:{port}"
    tls_config: HttpxTLSConfig = {"verify": str(ca_bundle)}

    def wait_for_gateway_ready(remaining: float) -> httpx.Response:
        if process.poll() is not None:
            assert process.returncode is not None
            raise AssertionError(_port_forward_exit_message(process.returncode, log_file))
        return httpx.get(
            gateway_url + GATEWAY_READY_PATH,
            timeout=min(PORT_FORWARD_HTTP_TIMEOUT_SECONDS, remaining),
            **tls_config,
        )

    try:
        _retry_until_timeout(
            wait_for_gateway_ready,
            timeout=PORT_FORWARD_READY_TIMEOUT_SECONDS,
            sleep=PORT_FORWARD_RETRY_SLEEP_SECONDS,
            retry_exceptions=(httpx.RequestError,),
            timeout_message=f"timed out waiting for port-forward readiness at {gateway_url + GATEWAY_READY_PATH}",
            raise_last_retry_error=False,
        )
    except Exception:
        _terminate_process(process)
        raise
    return gateway_url, process


def _terminate_process(process: subprocess.Popen[str]) -> None:
    process.terminate()
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=PORT_FORWARD_TERMINATE_TIMEOUT_SECONDS)
    if process.poll() is None:
        process.kill()
        process.wait()


def _grant_with_password(grant: dict[str, str], *, default_password: str | None = None) -> dict[str, str]:
    resolved = dict(grant)
    password_env_var = resolved.pop("password_env_var", None)
    if password_env_var and "password" not in resolved:
        password = os.environ.get(password_env_var) or default_password
        if password is None:
            raise AssertionError(f"{password_env_var} must be set for token acquisition")
        resolved["password"] = password
    return resolved


class KubernetesAuthIdpRuntime:
    def __init__(self, case: AuthIdpCase):
        self.case = case
        self.provider = case.provider
        self.cluster: Cluster | None = None
        self.namespace = NAMESPACE
        self.helm_release = HELM_RELEASE
        self.gateway_base_url = ""
        self.discovery_url = ""
        self.token_endpoint: str | None = None
        self.workload_token_endpoint: str | None = None
        self.ca_bundle: Path | None = None
        self._ca_temp_file: tempfile._TemporaryFileWrapper[bytes] | None = None
        self._port_forward_process: subprocess.Popen[str] | None = None
        self._diagnostics_collected = False
        self._reuse_cluster = False
        self._keep_cluster = False
        self._previous_client_ssl_cert_file = os.environ.get(NHX_CLIENT_SSL_CERT_FILE_ENVVAR)
        self._start()

    @property
    def verify(self) -> str:
        assert self.ca_bundle is not None
        return str(self.ca_bundle)

    def e2e_setup_token(self) -> TokenSet:
        assert self.provider.e2e_setup_password_grant is not None
        assert self.token_endpoint is not None
        token = self._exchange_token(self.token_endpoint, self.provider.e2e_setup_password_grant)
        return TokenSet(access_token=token, claims=jwt_claims(token))

    def interactive_user_token(self) -> TokenSet:
        assert self.provider.interactive_user_password_grant is not None
        assert self.token_endpoint is not None
        token = self._exchange_token(self.token_endpoint, self.provider.interactive_user_password_grant)
        return TokenSet(access_token=token, claims=jwt_claims(token))

    def workload_provider_token(self) -> TokenSet:
        assert self.provider.workload_provider_password_grant is not None
        assert self.token_endpoint is not None
        grant = _grant_with_password(
            self.provider.workload_provider_password_grant,
            default_password=AUTHENTIK_K8S_WORKLOAD_IDENTITY_PASSWORD,
        )
        token = self._exchange_token(self.token_endpoint, grant)
        return TokenSet(access_token=token, claims=jwt_claims(token))

    def workload_subject_token(self) -> str:
        assert self.cluster is not None
        return _run(
            _kubectl_command(
                self.cluster.context,
                [
                    "-n",
                    NAMESPACE,
                    "create",
                    "token",
                    "default",
                    "--audience",
                    WORKLOAD_CLIENT_ID,
                    "--duration",
                    SUBJECT_TOKEN_DURATION,
                ],
                self.cluster.kubeconfig,
            ),
        ).stdout.strip()

    def exchange_workload_token(self, subject_token: str) -> TokenSet:
        assert self.workload_token_endpoint is not None
        tls_config: HttpxTLSConfig = {"verify": self.verify}
        response = httpx.post(
            self.workload_token_endpoint,
            data={
                "grant_type": TOKEN_EXCHANGE_GRANT_TYPE,
                "client_id": WORKLOAD_CLIENT_ID,
                "subject_token": subject_token,
                "subject_token_type": JWT_TOKEN_TYPE,
                "requested_token_type": ACCESS_TOKEN_TYPE,
                "audience": WORKLOAD_AUDIENCE,
                "scope": "openid email groups",
            },
            timeout=TOKEN_EXCHANGE_TIMEOUT_SECONDS,
            **tls_config,
        )
        response.raise_for_status()
        token_response = response.json()
        access_token = token_response["access_token"]
        assert token_response.get("token_type", "").lower() == "bearer"
        return TokenSet(access_token=access_token, claims=jwt_claims(access_token))

    def workload_platform_token(self) -> TokenSet:
        return self.exchange_workload_token(self.workload_subject_token())

    def deployment_workload_runtime_config(self) -> DeploymentWorkloadRuntimeConfig:
        assert self.ca_bundle is not None
        ca_bundle = self.ca_bundle.read_text(encoding="utf-8")
        return DeploymentWorkloadRuntimeConfig(
            env=(
                {"name": "NHX_BASE_URL", "value": IN_CLUSTER_ENVOY_BASE_URL},
                {"name": "NHX_CLIENT_SSL_CERT_FILE", "value": DEPLOYMENT_WORKLOAD_CA_BUNDLE_FILE},
                {"name": "SSL_CERT_FILE", "value": DEPLOYMENT_WORKLOAD_CA_BUNDLE_FILE},
                {"name": "REQUESTS_CA_BUNDLE", "value": DEPLOYMENT_WORKLOAD_CA_BUNDLE_FILE},
            ),
            config_files=(
                {
                    "path": DEPLOYMENT_WORKLOAD_CA_BUNDLE_FILE,
                    "content": ca_bundle,
                    "mode": 0o644,
                },
            ),
        )

    def e2e_setup_sdk(self) -> NeMoHelix:
        return self._sdk_for_token(self.e2e_setup_token().access_token)

    def interactive_user_sdk(self) -> NeMoHelix:
        return self._sdk_for_token(self.interactive_user_token().access_token)

    def workload_provider_sdk(self) -> NeMoHelix:
        return self._sdk_for_token(self.workload_platform_token().access_token)

    def workload_role_principals(self) -> list[str]:
        return [f"system:serviceaccounts:{NAMESPACE}"]

    def _collect_diagnostics_best_effort(
        self,
        context: str,
        cluster_name: str,
        kubeconfig: Path | None = None,
    ) -> None:
        if self._diagnostics_collected:
            return
        try:
            with contextlib.suppress(Exception):
                log_dir = _collect_kubernetes_diagnostics(context, cluster_name, kubeconfig)
                print(f"Collected Authentik Kubernetes diagnostics: {log_dir}")
        finally:
            self._diagnostics_collected = True

    def cleanup(self) -> None:
        if self._port_forward_process is not None:
            _terminate_process(self._port_forward_process)
            self._port_forward_process = None
        if self._ca_temp_file is not None:
            with contextlib.suppress(FileNotFoundError):
                Path(self._ca_temp_file.name).unlink()
            self._ca_temp_file = None
        if self._previous_client_ssl_cert_file is None:
            os.environ.pop(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, None)
        else:
            os.environ[NHX_CLIENT_SSL_CERT_FILE_ENVVAR] = self._previous_client_ssl_cert_file
        if self.cluster is not None:
            cluster = self.cluster
            try:
                self._collect_diagnostics_best_effort(cluster.context, cluster.name, cluster.kubeconfig)
            finally:
                try:
                    if not self._reuse_cluster and not self._keep_cluster:
                        _delete_cluster(cluster.runtime, cluster.name, cluster.kubeconfig)
                finally:
                    if cluster.cleanup_kubeconfig and not self._keep_cluster and cluster.kubeconfig is not None:
                        with contextlib.suppress(FileNotFoundError):
                            cluster.kubeconfig.unlink()
                    self.cluster = None

    def _start(self) -> None:
        runtime = os.environ.get("NHX_AUTHENTIK_K8S_RUNTIME", "kind")
        name = os.environ.get("NHX_AUTHENTIK_K8S_CLUSTER_NAME") or f"nhx-authentik-{uuid.uuid4().hex[:8]}"
        self._reuse_cluster = os.environ.get("NHX_AUTHENTIK_K8S_REUSE_CLUSTER") == "1"
        self._keep_cluster = os.environ.get("NHX_AUTHENTIK_K8S_KEEP_CLUSTER") == "1"
        cluster = _reuse_or_create_cluster(runtime, name) if self._reuse_cluster else _create_cluster(runtime, name)
        self.cluster = cluster
        try:
            _load_platform_image(runtime, name, _platform_image())
            _helm_install_authentik_demo(cluster.context, cluster.kubeconfig)
            _wait_for_authentik(cluster.context, cluster.kubeconfig)
            self._write_ca_bundle(cluster.context, cluster.kubeconfig)
            assert self.ca_bundle is not None
            self.gateway_base_url, self._port_forward_process = _start_port_forward_service(
                cluster.context,
                "nemo-helix-envoy",
                self.ca_bundle,
                cluster.kubeconfig,
            )
            self.discovery_url = self.gateway_base_url + DISCOVERY_PATH
            self.token_endpoint = self.gateway_base_url + "/application/o/token/"
            self.workload_token_endpoint = self.gateway_base_url + "/apis/auth/token"
            os.environ[NHX_CLIENT_SSL_CERT_FILE_ENVVAR] = self.verify
        except Exception:
            try:
                self._collect_diagnostics_best_effort(cluster.context, cluster.name, cluster.kubeconfig)
            finally:
                self.cleanup()
            raise

    def _write_ca_bundle(self, context: str, kubeconfig: Path | None = None) -> None:
        temp_file = tempfile.NamedTemporaryFile(suffix="-nhx-ca.crt", delete=False)
        temp_file.write(_secret_data(context, ENVOY_TLS_SECRET, "ca.crt", kubeconfig=kubeconfig))
        temp_file.flush()
        temp_file.close()
        self._ca_temp_file = temp_file
        self.ca_bundle = Path(temp_file.name)

    def _exchange_token(self, token_endpoint: str, grant: dict[str, str]) -> str:
        from tests.auth_idp.conftest import _exchange_token_with_retries

        return _exchange_token_with_retries(token_endpoint, grant, tls_config={"verify": self.verify})

    def _sdk_for_token(self, token: str) -> NeMoHelix:
        return NeMoHelix(
            base_url=self.gateway_base_url,
            default_headers={"Authorization": f"Bearer {token}"},
            max_retries=0,
            http_client=DefaultHttpxClient(verify=self.verify),
        )
