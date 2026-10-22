#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

echo "Checking published artifacts for $RELEASE_VERSION"
if [[ "$CHART_REPOSITORY" == oci:* ]]; then
    printf '%s' "$GH_TOKEN" | helm registry login ghcr.io --username "$GITHUB_ACTOR" --password-stdin
    helm pull "$CHART_REPOSITORY" --version "$RELEASE_VERSION" --destination "$RUNNER_TEMP"
else
    helm pull nemo-helix --repo "$CHART_REPOSITORY" --version "$RELEASE_VERSION" --destination "$RUNNER_TEMP"
fi

registry="${IMAGE_REGISTRY%%/*}"
if [[ "$registry" == ghcr.io ]]; then
    printf '%s' "$GH_TOKEN" | docker login "$registry" --username "$GITHUB_ACTOR" --password-stdin
elif [[ -n "${NGC_REGISTRY_READ_TOKEN:-}" ]]; then
    printf '%s' "$NGC_REGISTRY_READ_TOKEN" | docker login "$registry" --username '$oauthtoken' --password-stdin
fi
for image in nhx-api nhx-tasks; do
    docker pull "$IMAGE_REGISTRY/$image:$IMAGE_TAG"
done
kind load docker-image --name "$KIND_CLUSTER_NAME" "$IMAGE_REGISTRY/nhx-api:$IMAGE_TAG"

# Only adapt storage/resources for a single-node CPU cluster. Use the published
# chart and API image, never the chart or packages from this checkout.
helm upgrade --install release-smoke "$RUNNER_TEMP/nemo-helix-$RELEASE_VERSION.tgz" \
    --namespace release-smoke --create-namespace --wait --timeout 8m \
    --set fullnameOverride=release-smoke \
    --set rbac.volcanoEnabled=false \
    --set postgresql.image.repository=docker.io/library/postgres \
    --set clickhouse.persistence.enabled=false \
    --set clickhouse.resources.requests.cpu=500m \
    --set clickhouse.resources.requests.memory=1Gi \
    --set 'core.storage.accessModes[0]=ReadWriteOnce' \
    --set ingress.enabled=false --set httpRoute.enabled=false \
    --set-string "api.image.repository=$IMAGE_REGISTRY/nhx-api" \
    --set-string "api.image.tag=$IMAGE_TAG" \
    --set-string "core.image.repository=$IMAGE_REGISTRY/nhx-api" \
    --set-string "core.image.tag=$IMAGE_TAG" \
    --set-string "platformConfig.platform.image_registry=$IMAGE_REGISTRY" \
    --set-string "platformConfig.platform.image_tag=$IMAGE_TAG"

mkdir -p smoke-logs
kubectl -n release-smoke port-forward svc/release-smoke-api 8080:8080 > smoke-logs/port-forward.log 2>&1 &
forward_pid=$!
trap 'kill "$forward_pid" 2>/dev/null || true' EXIT
curl --fail --silent --show-error --retry 30 --retry-all-errors --retry-delay 2 \
    --max-time 5 --retry-max-time 120 http://127.0.0.1:8080/health/ready \
    | tee smoke-logs/readiness.json | jq -e '.status == "ready"'
