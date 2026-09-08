# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Mint a GAR docker config from the Workload Identity token and write it to a
# Secret. GAR needs auth even to read a manifest, and scaled-evals resolves the
# digest of every freshly built image, so this has to stay fresh: the token
# lasts an hour, hence the half-hourly schedule.
#
# Adapted from the standalone chart's inline refresh script. Uses only stdlib so
# the job can run on a bare python image.
import base64
import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request

token_request = urllib.request.Request(
    "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
    headers={"Metadata-Flavor": "Google"},
)
with urllib.request.urlopen(token_request, timeout=30) as response:
    registry_token = json.loads(response.read().decode())["access_token"]

host = os.environ["REGISTRY_HOST"].removeprefix("https://").removeprefix("http://").rstrip("/")
username = os.environ["REGISTRY_USERNAME"]
auth = base64.b64encode(f"{username}:{registry_token}".encode()).decode()
docker_config = json.dumps(
    {"auths": {host: {"username": username, "password": registry_token, "auth": auth}}},
    separators=(",", ":"),
)
docker_config_data = base64.b64encode(docker_config.encode()).decode()

namespace = os.environ["POD_NAMESPACE"]
secret_name = os.environ["REGISTRY_AUTH_SECRET_NAME"]
# The field name the docker config lands under in the Secret's data map, not a
# credential itself.
docker_config_field = os.environ["REGISTRY_AUTH_SECRET_KEY"]
host_port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS") or os.environ["KUBERNETES_SERVICE_PORT"]
api_server = f"https://{os.environ['KUBERNETES_SERVICE_HOST']}:{host_port}"
kube_token = open("/var/run/secrets/kubernetes.io/serviceaccount/token", encoding="utf-8").read().strip()
ssl_context = ssl.create_default_context(cafile="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")


def request(method: str, path: str, body: dict | None = None) -> dict:
    data = None
    headers = {"Authorization": f"Bearer {kube_token}"}
    if body is not None:
        data = json.dumps(body, separators=(",", ":")).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{api_server}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
        payload = response.read()
    return json.loads(payload.decode()) if payload else {}


encoded_namespace = urllib.parse.quote(namespace, safe="")
secret_path = f"/api/v1/namespaces/{encoded_namespace}/secrets/{urllib.parse.quote(secret_name, safe='')}"
secret = {
    "apiVersion": "v1",
    "kind": "Secret",
    "metadata": {"name": secret_name, "namespace": namespace},
    "type": "kubernetes.io/dockerconfigjson",
    "data": {docker_config_field: docker_config_data},
}

try:
    existing = request("GET", secret_path)
except urllib.error.HTTPError as exc:
    if exc.code != 404:
        raise
    request("POST", f"/api/v1/namespaces/{encoded_namespace}/secrets", secret)
else:
    secret["metadata"]["resourceVersion"] = existing["metadata"]["resourceVersion"]
    request("PUT", secret_path, secret)

platform_secrets_url = os.environ.get("PLATFORM_SECRETS_URL")
if platform_secrets_url:
    platform_secret_name = "scaled-evals-registry-auth"
    platform_body = json.dumps({"name": platform_secret_name, "value": docker_config}).encode()
    platform_request = urllib.request.Request(
        platform_secrets_url,
        data=platform_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(platform_request, timeout=30).read()
    except urllib.error.HTTPError as exc:
        if exc.code != 409:
            raise
        patch_request = urllib.request.Request(
            f"{platform_secrets_url}/{platform_secret_name}",
            data=json.dumps({"value": docker_config}).encode(),
            headers={"Content-Type": "application/json"},
            method="PATCH",
        )
        urllib.request.urlopen(patch_request, timeout=30).read()
    except urllib.error.URLError:
        # The deployment script performs the initial sync after API readiness.
        pass

# Deliberately omits the Secret name: it reaches this script through an env var whose
# name matches a credential heuristic, and it is static deployment config anyway.
print(f"wrote docker config for {host} in namespace {namespace}")
