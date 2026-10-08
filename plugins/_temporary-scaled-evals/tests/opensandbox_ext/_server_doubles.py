# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stand-ins for the server integration tests.

The tests run the real opensandbox-server 0.2.1 provider. Only the Kubernetes API is faked,
so manifests come from the stock BatchSandbox create path.
"""

from __future__ import annotations

import types
from typing import Any

import nemo_opensandbox_ext as ext
from _ext_doubles import NAMESPACE
from kubernetes.client import V1Pod
from opensandbox_server.api.schema import ImageSpec, NetworkPolicy, NetworkRule

SANDBOX_ID = "sbx-1"


class FakeK8sClient:
    """The stock server's ``K8sClient``: records created manifests and returns the pods a test sets."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.pods: list[V1Pod] = []
        self.pod_list_calls = 0

    def create_custom_object(self, group: str, version: str, namespace: str, plural: str, body: dict[str, Any]):
        """Record the manifest and answer like the API server would."""
        self.created.append(body)
        return {"metadata": {"name": body["metadata"]["name"], "uid": "uid-1"}}

    def list_pods(self, namespace: str, label_selector: str = "") -> list[V1Pod]:
        """Return ``self.pods``; only ever asked for the test sandbox's pod."""
        self.pod_list_calls += 1
        assert label_selector == f"opensandbox.io/id={SANDBOX_ID}"
        return self.pods

    def get_core_v1_api(self) -> Any:
        """A CoreV1Api whose pod logs are one fixed line."""
        return types.SimpleNamespace(read_namespaced_pod_log=lambda *a, **k: "boom: config missing")


def create(provider: ext.NemoServicesProvider, extensions: dict[str, str] | None, *, egress: bool = True) -> None:
    """Run the stock create path for the test sandbox, with an egress policy unless ``egress`` is False."""
    provider.create_workload(
        sandbox_id=SANDBOX_ID,
        namespace=NAMESPACE,
        image_spec=ImageSpec(uri="registry.example.com/task/main:1"),
        entrypoint=["/app/entrypoint.sh", "sh", "-c", "sleep infinity"],
        env={"A": "1"},
        resource_limits={"cpu": "2", "memory": "4Gi"},
        labels={"opensandbox.io/id": SANDBOX_ID},
        expires_at=None,
        execd_image="registry.example.com/opensandbox/execd:v1",
        extensions=extensions,
        network_policy=NetworkPolicy(defaultAction="deny", egress=[NetworkRule(action="allow", target="pypi.org")])
        if egress
        else None,
        egress_image="registry.example.com/opensandbox/egress:v1" if egress else None,
    )
