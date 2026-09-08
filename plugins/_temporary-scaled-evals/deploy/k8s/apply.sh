#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Apply the stack, generating the two secrets on first run so no key is ever
# committed. Re-running is safe: existing secrets are left alone, so the Fernet
# key stays stable and previously stored credentials remain decryptable.
#
#   ./apply.sh                 # deploy
#   ./apply.sh --render        # print manifests, touch nothing
#
# The manifests carry ${SE_*} placeholders instead of a hardcoded cloud project.
# Their values come from local.env, which is untracked.
#
# GCP prerequisites are NOT created here. See README.md.
set -euo pipefail

NS=nemo-platform-scaled-evals
cd "$(dirname "$0")"

if [ ! -f local.env ]; then
  cp local.env.example local.env
  echo "created local.env from the example. Fill in the values, then re-run." >&2
  exit 1
fi

# Substitution happens on a copy of the inputs *before* kustomize runs, not on
# its output. configMapGenerator hashes settings.env into the ConfigMap name, and
# that hash has to cover the real values — otherwise editing local.env would
# change a ConfigMap's contents without changing its name, and the pods would
# keep running with the old settings until something restarted them.
#
# Only the keys local.env defines are replaced, so neither the ${HOME} in
# workers.yaml nor the kubelet's own $(VAR) references are touched.
render() {
  local tmp
  tmp="$(mktemp -d)"
  cp ./*.yaml ./settings.env ./sandbox.env ./registry-auth-refresh.py "$tmp/"
  python3 - "$tmp" <<'PY'
import pathlib, re, sys

values = {}
for line in pathlib.Path("local.env").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, _, value = line.partition("=")
    values[key.strip()] = value.strip()

missing = sorted(key for key, value in values.items() if not value)
if missing:
    sys.exit("local.env is missing values for: " + ", ".join(missing))

for path in sorted(pathlib.Path(sys.argv[1]).iterdir()):
    text = original = path.read_text()
    for key, value in values.items():
        text = text.replace("${" + key + "}", value)
    unresolved = sorted(set(re.findall(r"\$\{SE_[A-Z0-9_]+\}", text)))
    if unresolved:
        sys.exit(f"{path.name}: no local.env value for " + ", ".join(unresolved))
    if text != original:
        path.write_text(text)
PY
  kubectl kustomize "$tmp"
  rm -rf "$tmp"
}

if [ "${1:-}" = --render ]; then
  render
  exit 0
fi

echo "==> context: $(kubectl config current-context)"
kubectl get namespace "$NS" >/dev/null 2>&1 || kubectl create namespace "$NS"

# `create --dry-run | apply` would overwrite the Fernet key on every run and
# orphan every credential encrypted under the old one, so only create if absent.
if ! kubectl get secret scaled-evals-credential-encryption -n "$NS" >/dev/null 2>&1; then
  echo "==> generating credential encryption key"
  kubectl create secret generic scaled-evals-credential-encryption -n "$NS" \
    --from-literal=CREDENTIALS_ENCRYPTION_KEY="$(python3 -c \
      'import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())')"
fi

if ! kubectl get secret scaled-evals-postgres -n "$NS" >/dev/null 2>&1; then
  echo "==> generating postgres password"
  kubectl create secret generic scaled-evals-postgres -n "$NS" \
    --from-literal=password="$(python3 -c \
      'import secrets; print(secrets.token_urlsafe(24))')"
fi

echo "==> applying"
render | kubectl apply -f -
kubectl delete deployment -n "$NS" scaled-evals-build-worker --ignore-not-found

# Create the auth Secret before waiting for workloads; the recurring CronJob
# updates it in place without putting an empty credential into the manifests.
echo "==> priming GAR credentials"
JOB="gar-auth-$(date +%s)"
kubectl create job -n "$NS" "$JOB" \
  --from=cronjob/scaled-evals-gar-registry-auth-refresh \
  --dry-run=client -o json |
  python3 -c '
import json
import sys

document = json.load(sys.stdin)
environment = document["spec"]["template"]["spec"]["containers"][0]["env"]
next(item for item in environment if item["name"] == "PLATFORM_SECRETS_SYNC_OPTIONAL")["value"] = "true"
json.dump(document, sys.stdout)
' |
  kubectl create -f -
kubectl wait -n "$NS" --for=condition=complete --timeout=180s "job/$JOB"

echo "==> waiting for the API"
kubectl rollout status -n "$NS" deploy/scaled-evals-api --timeout=300s

# Platform Job pods cannot consume deployment-local Kubernetes Secrets
# directly. Mirror the two required values into Platform Secrets without ever
# placing plaintext in manifests, command arguments, or shell output.
echo "==> configuring Platform Job secrets"
kubectl exec -n "$NS" deploy/scaled-evals-api -- python -c '
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

base = "http://127.0.0.1:8080/apis/secrets/v2/workspaces/default/secrets"
for name, value in {
    "scaled-evals-postgres-password": os.environ["PGPASSWORD"],
    "scaled-evals-credentials-encryption-key": os.environ["CREDENTIALS_ENCRYPTION_KEY"],
    "scaled-evals-registry-auth": Path(os.environ["TASK_IMAGE_REGISTRY_AUTH_FILE"]).read_text(),
}.items():
    body = json.dumps({"name": name, "value": value}).encode()
    request = urllib.request.Request(base, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(request).read()
    except urllib.error.HTTPError as exc:
        if exc.code != 409:
            raise
        patch = urllib.request.Request(
            f"{base}/{name}",
            data=json.dumps({"value": value}).encode(),
            headers={"Content-Type": "application/json"},
            method="PATCH",
        )
        urllib.request.urlopen(patch).read()
'
