# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live qualification for authoritative sandbox-k8s egress policies."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_SANDBOX_GROUP_PATH = "/apis/agents.x-k8s.io/v1alpha1"
_DEFAULT_ALLOWED = ("1.1.1.1", 443)
_DEFAULT_DENIED = ("1.0.0.1", 443)
_CONTROL_SERVICES = ("scaled-evals-api", "rustfs", "registry", "buildkit")
_PROBE_PORT = 18080


@dataclass(frozen=True)
class ApiResponse:
    status: int
    payload: dict[str, Any]


class KubernetesApi:
    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        ca_path: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.context = ssl.create_default_context(cafile=ca_path) if ca_path else ssl.create_default_context()

    def request(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> ApiResponse:
        data = None
        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(f"{self.base_url}{path}", data=data, headers=headers, method=method)
        with urllib.request.urlopen(request, context=self.context, timeout=20) as response:
            raw = response.read().decode(errors="replace")
            return ApiResponse(response.status, json.loads(raw) if raw else {})


def _in_cluster_api() -> KubernetesApi:
    host = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
    port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
    token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    ca_path = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    with open(token_path, encoding="utf-8") as token_file:
        token = token_file.read().strip()
    return KubernetesApi(f"https://{host}:{port}", token=token, ca_path=ca_path)


def _dns_rule() -> dict[str, Any]:
    return {
        "to": [
            {
                "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "kube-system"}},
                "podSelector": {"matchLabels": {"k8s-app": "kube-dns"}},
            },
            {
                "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "openshift-dns"}},
                "podSelector": {"matchLabels": {"dns.operator.openshift.io/daemonset-dns": "default"}},
            },
        ],
        "ports": [
            {"protocol": "UDP", "port": 53},
            {"protocol": "TCP", "port": 53},
        ],
    }


def _policy(
    namespace: str,
    name: str,
    qualification_id: str,
    probe: str,
    egress: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "podSelector": {
                "matchLabels": {
                    "scaled-evals.nvidia.com/egress-qualification": qualification_id,
                    "scaled-evals.nvidia.com/egress-probe": probe,
                }
            },
            "policyTypes": ["Egress"],
            "egress": egress,
        },
    }


def _sandbox(
    namespace: str,
    name: str,
    image: str,
    pull_secret: str,
    qualification_id: str,
    probe: str,
    command: list[str],
) -> dict[str, Any]:
    labels = {
        "app.kubernetes.io/managed-by": "sandbox-k8s",
        "sandbox-k8s/sandbox": name,
        "scaled-evals.nvidia.com/egress-qualification": qualification_id,
        "scaled-evals.nvidia.com/egress-probe": probe,
    }
    pod_spec: dict[str, Any] = {
        "automountServiceAccountToken": False,
        "containers": [
            {
                "name": "sandbox",
                "image": image,
                "command": command,
                "securityContext": {
                    "readOnlyRootFilesystem": True,
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                },
                "resources": {
                    "requests": {"cpu": "25m", "memory": "64Mi"},
                    "limits": {"cpu": "250m", "memory": "256Mi"},
                },
            }
        ],
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": 1000,
            "seccompProfile": {"type": "RuntimeDefault"},
        },
    }
    if pull_secret:
        pod_spec["imagePullSecrets"] = [{"name": pull_secret}]
    return {
        "apiVersion": "agents.x-k8s.io/v1alpha1",
        "kind": "Sandbox",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "spec": {
            "podTemplate": {"metadata": {"labels": labels}, "spec": pod_spec},
            "shutdownTime": "2099-01-01T00:00:00Z",
            "shutdownPolicy": "Delete",
        },
    }


_CLIENT_SCRIPT = r"""
import json, socket, sys, time
spec = json.loads(sys.argv[1])
results = {"dns": False, "targets": {}}
try:
    socket.getaddrinfo("kubernetes.default.svc", 443)
    results["dns"] = True
except OSError as exc:
    results["dns_error"] = str(exc)
for target in spec["targets"]:
    key = target["name"]
    try:
        connection = socket.create_connection((target["host"], target["port"]), timeout=3)
    except OSError as exc:
        results["targets"][key] = {"reachable": False, "detail": str(exc)}
    else:
        connection.close()
        results["targets"][key] = {"reachable": True}
print("SCALED_EVALS_EGRESS_RESULT=" + json.dumps(results, sort_keys=True), flush=True)
time.sleep(3600)
""".strip()


class Qualification:
    def __init__(
        self,
        api: KubernetesApi,
        *,
        namespace: str,
        image: str,
        pull_secret: str = "",
        timeout: float = 600,
        poll: float = 2,
        allowed: tuple[str, int] = _DEFAULT_ALLOWED,
        denied: tuple[str, int] = _DEFAULT_DENIED,
    ) -> None:
        if allowed == denied:
            raise ValueError("allowed and denied probe endpoints must differ")
        self.api = api
        self.namespace = namespace
        self.image = image
        self.pull_secret = pull_secret
        self.timeout = timeout
        self.poll = poll
        self.allowed = allowed
        self.denied = denied
        self.qualification_id = f"se-{secrets.token_hex(5)}"
        self.sandbox_names: list[str] = []
        self.policy_names: list[str] = []

    @property
    def sandbox_collection(self) -> str:
        return f"{_SANDBOX_GROUP_PATH}/namespaces/{self.namespace}/sandboxes"

    @property
    def policy_collection(self) -> str:
        return f"/apis/networking.k8s.io/v1/namespaces/{self.namespace}/networkpolicies"

    def run(self) -> dict[str, Any]:
        evidence: dict[str, Any] = {
            "schema_version": "scaled-evals-egress-qualification-v1",
            "qualification_id": self.qualification_id,
            "namespace": self.namespace,
            "results": {},
        }
        try:
            server_name = self._create_server()
            server_pod = self._wait_pod(server_name)
            server_ip = str(server_pod["status"]["podIP"])
            targets = self._targets(server_ip)
            self._create_policies()
            expectations = {
                "baseline": {target["name"]: True for target in targets},
                "default-deny": {target["name"]: False for target in targets},
                "scoped": {target["name"]: target["name"] == "declared-public" for target in targets},
            }
            for probe in ("baseline", "default-deny", "scoped"):
                name = self._create_client(probe, targets)
                result = self._wait_result(name)
                self._assert_result(probe, result, expectations[probe])
                evidence["results"][probe] = result
            evidence["status"] = "PASS"
            return evidence
        finally:
            self.cleanup()

    def _targets(self, server_ip: str) -> list[dict[str, Any]]:
        targets = [
            {"name": "declared-public", "host": self.allowed[0], "port": self.allowed[1]},
            {"name": "unlisted-public", "host": self.denied[0], "port": self.denied[1]},
            {"name": "unrelated-sandbox", "host": server_ip, "port": _PROBE_PORT},
        ]
        for service_name in _CONTROL_SERVICES:
            path = f"/api/v1/namespaces/{self.namespace}/services/{service_name}"
            try:
                service = self.api.request("GET", path).payload
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    continue
                raise
            cluster_ip = service.get("spec", {}).get("clusterIP")
            ports = service.get("spec", {}).get("ports") or []
            if cluster_ip and cluster_ip != "None" and ports:
                targets.append(
                    {
                        "name": f"control-plane-{service_name}",
                        "host": cluster_ip,
                        "port": int(ports[0]["port"]),
                    }
                )
        return targets

    def _create_policies(self) -> None:
        public_rule = {
            "to": [{"ipBlock": {"cidr": f"{self.allowed[0]}/32"}}],
            "ports": [{"protocol": "TCP", "port": self.allowed[1]}],
        }
        policies = {
            "baseline": [{}],
            "default-deny": [_dns_rule()],
            "scoped": [_dns_rule(), public_rule],
        }
        for probe, egress in policies.items():
            name = f"{self.qualification_id}-{probe}"
            self.api.request(
                "POST",
                self.policy_collection,
                _policy(self.namespace, name, self.qualification_id, probe, egress),
            )
            self.policy_names.append(name)

    def _create_server(self) -> str:
        name = f"{self.qualification_id}-server"
        body = _sandbox(
            self.namespace,
            name,
            self.image,
            self.pull_secret,
            self.qualification_id,
            "server",
            ["python", "-m", "http.server", str(_PROBE_PORT)],
        )
        self.api.request("POST", self.sandbox_collection, body)
        self.sandbox_names.append(name)
        self._wait_sandbox_ready(name)
        return name

    def _create_client(self, probe: str, targets: list[dict[str, Any]]) -> str:
        name = f"{self.qualification_id}-{probe}"
        body = _sandbox(
            self.namespace,
            name,
            self.image,
            self.pull_secret,
            self.qualification_id,
            probe,
            ["python", "-c", _CLIENT_SCRIPT, json.dumps({"targets": targets})],
        )
        self.api.request("POST", self.sandbox_collection, body)
        self.sandbox_names.append(name)
        self._wait_sandbox_ready(name)
        return name

    def _wait_sandbox_ready(self, name: str) -> None:
        path = f"{self.sandbox_collection}/{name}"
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            current = self.api.request("GET", path).payload
            conditions = current.get("status", {}).get("conditions") or []
            ready = next((item for item in conditions if item.get("type") == "Ready"), None)
            if ready and ready.get("status") == "True":
                return
            if (
                ready
                and ready.get("status") == "False"
                and ready.get("reason")
                in {
                    "CreatePodError",
                    "PodFailed",
                    "ImagePullBackOff",
                    "ErrImagePull",
                    "Forbidden",
                }
            ):
                raise RuntimeError(f"Sandbox {name} failed: {json.dumps(ready, sort_keys=True)}")
            time.sleep(self.poll)
        raise TimeoutError(f"Sandbox {name} did not become ready within {self.timeout}s")

    def _wait_pod(self, sandbox_name: str) -> dict[str, Any]:
        query = urllib.parse.urlencode({"labelSelector": f"sandbox-k8s/sandbox={sandbox_name}"})
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            pods = self.api.request("GET", f"/api/v1/namespaces/{self.namespace}/pods?{query}").payload.get("items", [])
            for pod in pods:
                if pod.get("status", {}).get("podIP"):
                    return pod
            time.sleep(self.poll)
        raise TimeoutError(f"Sandbox pod {sandbox_name} has no IP")

    def _wait_result(self, sandbox_name: str) -> dict[str, Any]:
        pod = self._wait_pod(sandbox_name)
        pod_name = pod["metadata"]["name"]
        query = urllib.parse.urlencode({"container": "sandbox"})
        path = f"/api/v1/namespaces/{self.namespace}/pods/{pod_name}/log?{query}"
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            request = urllib.request.Request(f"{self.api.base_url}{path}")
            if self.api.token:
                request.add_header("Authorization", f"Bearer {self.api.token}")
            try:
                with urllib.request.urlopen(request, context=self.api.context, timeout=20) as response:
                    logs = response.read().decode(errors="replace")
            except urllib.error.HTTPError as exc:
                if exc.code in {400, 404}:
                    time.sleep(self.poll)
                    continue
                raise
            for line in logs.splitlines():
                prefix = "SCALED_EVALS_EGRESS_RESULT="
                if line.startswith(prefix):
                    return json.loads(line.removeprefix(prefix))
            time.sleep(self.poll)
        raise TimeoutError(f"Sandbox {sandbox_name} produced no egress result")

    @staticmethod
    def _assert_result(probe: str, result: Mapping[str, Any], expectations: Mapping[str, bool]) -> None:
        if result.get("dns") is not True:
            raise RuntimeError(f"{probe}: cluster DNS failed: {result.get('dns_error')}")
        observed = result.get("targets") or {}
        failures = {
            name: {"expected": expected, "observed": observed.get(name)}
            for name, expected in expectations.items()
            if not isinstance(observed.get(name), Mapping) or observed[name].get("reachable") is not expected
        }
        if failures:
            raise RuntimeError(f"{probe}: egress mismatch: {json.dumps(failures, sort_keys=True)}")

    def cleanup(self) -> None:
        delete_body = {
            "apiVersion": "v1",
            "kind": "DeleteOptions",
            "propagationPolicy": "Foreground",
        }
        for name in reversed(self.sandbox_names):
            try:
                self.api.request("DELETE", f"{self.sandbox_collection}/{name}", delete_body)
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    print(f"cleanup warning: Sandbox {name}: HTTP {exc.code}", file=sys.stderr)
        deadline = time.monotonic() + min(self.timeout, 120)
        selector = urllib.parse.urlencode(
            {"labelSelector": ("scaled-evals.nvidia.com/egress-qualification=" + self.qualification_id)}
        )
        while time.monotonic() < deadline:
            pods = self.api.request("GET", f"/api/v1/namespaces/{self.namespace}/pods?{selector}").payload.get(
                "items", []
            )
            if not pods:
                break
            time.sleep(self.poll)
        for name in reversed(self.policy_names):
            try:
                self.api.request("DELETE", f"{self.policy_collection}/{name}", delete_body)
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    print(f"cleanup warning: NetworkPolicy {name}: HTTP {exc.code}", file=sys.stderr)


def _parse_endpoint(value: str) -> tuple[str, int]:
    host, separator, port = value.rpartition(":")
    if not separator or not host or not port.isdigit():
        raise argparse.ArgumentTypeError("endpoint must use HOST:PORT")
    try:
        socket.inet_pton(socket.AF_INET, host)
    except OSError as exc:
        raise argparse.ArgumentTypeError("endpoint host must be an IPv4 address") from exc
    parsed_port = int(port)
    if not 1 <= parsed_port <= 65535:
        raise argparse.ArgumentTypeError("endpoint port must be between 1 and 65535")
    return host, parsed_port


def _start_oc_proxy(context: str | None) -> tuple[subprocess.Popen[str], str]:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    command = ["oc"]
    if context:
        command.extend(["--context", context])
    command.extend(["proxy", f"--port={port}", "--accept-hosts=^127\\.0\\.0\\.1$"])
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            detail = "" if process.stderr is None else process.stderr.read().strip()
            raise RuntimeError(f"oc proxy exited before readiness: {detail}")
        try:
            with urllib.request.urlopen(f"{base_url}/version", timeout=1):
                return process, base_url
        except (OSError, urllib.error.URLError):
            time.sleep(0.2)
    process.terminate()
    raise TimeoutError("oc proxy did not become ready")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=os.environ.get("SANDBOX_NAMESPACE"))
    parser.add_argument("--image", default=os.environ.get("SANDBOX_TEST_IMAGE"))
    parser.add_argument("--pull-secret", default=os.environ.get("SANDBOX_TEST_IMAGE_PULL_SECRET", ""))
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--poll", type=float, default=2)
    parser.add_argument("--allowed-endpoint", type=_parse_endpoint, default=_DEFAULT_ALLOWED)
    parser.add_argument("--denied-endpoint", type=_parse_endpoint, default=_DEFAULT_DENIED)
    parser.add_argument("--via-oc-proxy", action="store_true")
    parser.add_argument("--context")
    args = parser.parse_args()
    if not args.namespace or not args.image:
        parser.error("--namespace and --image are required")

    proxy: subprocess.Popen[str] | None = None
    try:
        if args.via_oc_proxy:
            proxy, base_url = _start_oc_proxy(args.context)
            api = KubernetesApi(base_url)
        else:
            api = _in_cluster_api()
        evidence = Qualification(
            api,
            namespace=args.namespace,
            image=args.image,
            pull_secret=args.pull_secret,
            timeout=args.timeout,
            poll=args.poll,
            allowed=args.allowed_endpoint,
            denied=args.denied_endpoint,
        ).run()
        print(json.dumps(evidence, indent=2, sort_keys=True))
    finally:
        if proxy is not None:
            proxy.terminate()
            try:
                proxy.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proxy.kill()


if __name__ == "__main__":
    main()
