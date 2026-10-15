#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the namespace-local OpenSandbox server used by the Harbor POC.

Reads the repo's server config, sandbox pod template, and temporary scaled-evals
extensions, then clones the pod wiring of the existing dev-blue server. Creates
independent ConfigMaps, credentials, ServiceAccount, namespaced RBAC, Deployment,
and Service for the selected namespace. The shared controller and CRDs must
already exist; this script does not install or change them.

Always writes a private multi-document manifest and performs a server dry-run.
Only --apply changes cluster resources, followed by a rollout readiness check.
It requires Kubernetes access even when rendering because it reads the reference
deployment/RBAC and preserves any existing POC API key.
"""

import argparse
import copy
import hashlib
import json
import os
import secrets
import subprocess
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", default="nv-prd-nemo.teleport.sh-nemo-dev-blue")
    parser.add_argument("--namespace", default="ngoncharenko-dev")
    parser.add_argument("--name", default="helix-poc-opensandbox")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    kubectl = ["kubectl", "--context", args.context]

    def get(kind: str, name: str, namespace: str | None = None) -> dict:
        command = [*kubectl, "get", kind, name, "-o", "json"]
        if namespace:
            command += ["-n", namespace]
        return json.loads(subprocess.check_output(command))

    def resource(kind: str, name: str, **fields) -> dict:
        api = "rbac.authorization.k8s.io/v1" if kind in {"Role", "RoleBinding"} else "v1"
        return {"apiVersion": api, "kind": kind, "metadata": {"name": name, "namespace": args.namespace}, **fields}

    # The chart examples provide compatible server settings and sandbox defaults;
    # the extensions add NeMo/Harbor behavior to the stock OpenSandbox server.
    values = yaml.safe_load((root / "k8s/helm/examples/opensandbox/opensandbox-server.yaml").read_text())
    names = {
        "opensandbox-server-config": f"{args.name}-config",
        "opensandbox-batchsandbox-template": f"{args.name}-template",
        "nemo-opensandbox-ext": f"{args.name}-extension",
    }
    config = values["configToml"].replace("REPLACE_WITH_RELEASE_NAMESPACE", args.namespace)
    template_config = yaml.safe_load((root / "k8s/helm/examples/opensandbox/batchsandbox-template.yaml").read_text())
    # Mount the embedded template text, rather than the surrounding ConfigMap YAML.
    # Its volumes, pull secret, and scheduling exclusions must reach real task pods.
    template = template_config["data"]["batchsandbox-template.yaml"]
    ext = root / "plugins/_temporary-scaled-evals"
    extensions = {
        name: (ext / "opensandbox_ext" / name).read_text()
        for name in ("sitecustomize.py", "nemo_ext_server.py", "nemo_ext_pod.py")
    }
    extensions["harbor_opensandbox_services.py"] = (ext / "src/scaled_evals/harbor_opensandbox_services.py").read_text()
    objects = [
        resource("ConfigMap", names["opensandbox-server-config"], data={"config.toml": config}),
        resource(
            "ConfigMap", names["opensandbox-batchsandbox-template"], data={"batchsandbox-template.yaml": template}
        ),
        resource("ConfigMap", names["nemo-opensandbox-ext"], data=extensions),
        resource("ServiceAccount", args.name),
    ]
    # Preserve a previously generated key; rendering never prints it.
    existing = subprocess.run(
        [*kubectl, "get", "secret", f"{args.name}-key", "-n", args.namespace, "-o", "json", "--ignore-not-found"],
        capture_output=True,
        text=True,
        check=True,
    )
    key = resource("Secret", f"{args.name}-key", type="Opaque")
    if existing.stdout.strip():
        key["data"] = json.loads(existing.stdout)["data"]
    else:
        key["stringData"] = {"api-key": secrets.token_hex(32)}
    objects.append(key)
    # Restrict the reference server's permissions to this namespace. Shared-kernel
    # task pods do not need the cluster-scoped RuntimeClass lookup permission.
    rules = [
        rule for rule in get("clusterrole", "opensandbox-server-role")["rules"] if rule["apiGroups"] != ["node.k8s.io"]
    ]
    rules += [
        {"apiGroups": [""], "resources": ["pods/exec"], "verbs": ["get", "create"]},
        {"apiGroups": [""], "resources": ["pods/log"], "verbs": ["get"]},
    ]
    objects += [
        resource("Role", args.name, rules=rules),
        resource(
            "RoleBinding",
            args.name,
            roleRef={"apiGroup": "rbac.authorization.k8s.io", "kind": "Role", "name": args.name},
            subjects=[{"kind": "ServiceAccount", "name": args.name, "namespace": args.namespace}],
        ),
    ]
    # Reuse the proven stock-server pod wiring, with independent names and credentials.
    spec = copy.deepcopy(get("deployment", "opensandbox-server", "jgulabrai-opensandbox-system")["spec"])
    spec["replicas"] = 1
    labels = {"app.kubernetes.io/name": args.name, "app.kubernetes.io/instance": args.name}
    spec["selector"] = {"matchLabels": labels}
    # Mounted ConfigMap changes do not restart a pod by themselves. Include their
    # content hash in the pod template so a later apply rolls out corrected config.
    config_digest = hashlib.sha256((config + template + json.dumps(extensions, sort_keys=True)).encode()).hexdigest()
    spec["template"]["metadata"] = {"labels": labels, "annotations": {"helix-poc/config-sha256": config_digest}}
    pod = spec["template"]["spec"]
    pod["serviceAccountName"] = args.name
    # Remap every mounted config and secret reference to this POC's independent
    # objects. The source deployment remains untouched in its own namespace.
    for volume in pod["volumes"]:
        if cm := volume.get("configMap"):
            cm["name"] = names.get(cm["name"], cm["name"])
            cm["optional"] = False
    for container in pod["containers"]:
        for variable in container.get("env", []):
            if "secretKeyRef" in variable.get("valueFrom", {}):
                variable["valueFrom"]["secretKeyRef"]["name"] = f"{args.name}-key"
    deployment = resource("Deployment", args.name, spec=spec)
    deployment["apiVersion"] = "apps/v1"
    objects += [
        deployment,
        resource(
            "Service",
            args.name,
            spec={"selector": labels, "ports": [{"name": "http", "port": 80, "targetPort": "http"}]},
        ),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # The rendered document contains the API key, so do not print it or make it
    # group/world-readable. Dry-run validates admission/RBAC before a real apply.
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as output:
        yaml.safe_dump_all(objects, output, sort_keys=False)
    subprocess.run([*kubectl, "apply", "--dry-run=server", "-f", str(args.output)], check=True)
    if args.apply:
        subprocess.run([*kubectl, "apply", "-f", str(args.output)], check=True)
        subprocess.run(
            [*kubectl, "-n", args.namespace, "rollout", "status", f"deployment/{args.name}", "--timeout=180s"],
            check=True,
        )
    print(f"OpenSandbox endpoint: {args.name}.{args.namespace}.svc.cluster.local:80")


if __name__ == "__main__":
    main()
