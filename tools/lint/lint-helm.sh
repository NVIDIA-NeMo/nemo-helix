#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


set -xeo pipefail

HELM_FOLDER=${HELM_FOLDER:-k8s/helm}
HELM_RELEASE_NAME=${HELM_RELEASE_NAME:-nemo-helix}
HELM_ENVOY_IMAGE=${HELM_ENVOY_IMAGE:-docker.io/envoyproxy/envoy:v1.37.0}
OPENSHIFT_VERSION=${OPENSHIFT_VERSION:-4.1.0}

# Cache dir for kubeconform so schemas are downloaded once per run instead of per file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${CI_PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
KUBECONFORM_CACHE="${KUBECONFORM_CACHE:-${PROJECT_ROOT}/.kubeconform-cache}"
mkdir -p "${KUBECONFORM_CACHE}"
lint_tmp=$(mktemp -d)
trap 'rm -rf "${lint_tmp}"' EXIT

validate_rendered_envoy_config() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is unavailable; skipping Envoy config validation" >&2
    return
  fi

  docker pull "${HELM_ENVOY_IMAGE}"

  local envoy_config_map="${lint_tmp}/envoy-configmap.yaml"
  local envoy_config="${lint_tmp}/envoy.yaml"

  helm template "${HELM_RELEASE_NAME}" "${HELM_FOLDER}" \
    --set platformConfig.auth.enabled=true \
    --show-only templates/proxy/envoy-configmap.yaml \
    > "${envoy_config_map}"

  awk '
    $0 == "  envoy.yaml: |" { in_block = 1; next }
    in_block && /^  [^[:space:]][^:]*:/ { exit }
    in_block {
      if ($0 == "") {
        print ""
        next
      }
      if (substr($0, 1, 4) != "    ") {
        print "unexpected indentation while extracting envoy.yaml: " $0 > "/dev/stderr"
        exit 1
      }
      print substr($0, 5)
    }
  ' "${envoy_config_map}" > "${envoy_config}"
  test -s "${envoy_config}"

  docker run --rm \
    -v "${envoy_config}:/etc/envoy/envoy.yaml:ro" \
    "${HELM_ENVOY_IMAGE}" \
    --mode validate \
    -c /etc/envoy/envoy.yaml
}

# Fetch chart dependencies so subchart templates (e.g. postgresql) are available during lint/template
helm dependency update "${HELM_FOLDER}"

# Lint the Helm chart
helm lint --strict "${HELM_FOLDER}"
validate_rendered_envoy_config

# StatefulSet volumeClaimTemplates are immutable, so chart metadata must not change them.
postgres_claim_tmp="${lint_tmp}/postgres-claim"
mkdir -p "${postgres_claim_tmp}"

for version in 1 2; do
  helm package "${HELM_FOLDER}" \
    --version "0.0.0-claim-test.${version}" \
    --app-version "claim-test-${version}" \
    --destination "${postgres_claim_tmp}" >/dev/null
  helm template "${HELM_RELEASE_NAME}" \
    "${postgres_claim_tmp}/nemo-helix-0.0.0-claim-test.${version}.tgz" \
    --show-only templates/postgres/postgres-statefulset.yaml \
    | sed -n '/^  volumeClaimTemplates:/,$p' > "${postgres_claim_tmp}/claim-${version}.yaml"
  test -s "${postgres_claim_tmp}/claim-${version}.yaml"
done

diff -u "${postgres_claim_tmp}/claim-1.yaml" "${postgres_claim_tmp}/claim-2.yaml"

# Utilization-based autoscaling must reject missing CPU requests.
api_autoscaling_output=$(helm template "${HELM_RELEASE_NAME}" "${HELM_FOLDER}" \
  --set api.autoscaling.enabled=true 2>&1) && {
  echo "API autoscaling accepted a missing CPU request" >&2
  exit 1
}
grep -Fq "api.resources.requests.cpu is required when API CPU autoscaling is enabled" \
  <<<"${api_autoscaling_output}"

envoy_autoscaling_output=$(helm template "${HELM_RELEASE_NAME}" "${HELM_FOLDER}" \
  --set platformConfig.auth.enabled=true \
  --set envoyProxy.autoscaling.enabled=true 2>&1) && {
  echo "Envoy autoscaling accepted a missing CPU request" >&2
  exit 1
}
grep -Fq "envoyProxy.resources.requests.cpu is required when Envoy CPU autoscaling is enabled" \
  <<<"${envoy_autoscaling_output}"

# Intake must not render with an incomplete external ClickHouse connection.
external_clickhouse_output=$(helm template "${HELM_RELEASE_NAME}" "${HELM_FOLDER}" \
  --set clickhouse.enabled=false 2>&1) && {
  echo "Intake accepted a missing external ClickHouse host" >&2
  exit 1
}
grep -Fq "externalClickhouse.host is required when clickhouse.enabled=false" \
  <<<"${external_clickhouse_output}"

external_clickhouse_secret_output=$(helm template "${HELM_RELEASE_NAME}" "${HELM_FOLDER}" \
  --set clickhouse.enabled=false \
  --set externalClickhouse.host=clickhouse.example.internal 2>&1) && {
  echo "External ClickHouse accepted a missing credentials Secret" >&2
  exit 1
}
grep -Fq "externalClickhouse.existingSecret is required when clickhouse.enabled=false" \
  <<<"${external_clickhouse_secret_output}"

external_clickhouse_password_key_output=$(helm template "${HELM_RELEASE_NAME}" "${HELM_FOLDER}" \
  --set clickhouse.enabled=false \
  --set externalClickhouse.host=clickhouse.example.internal \
  --set externalClickhouse.existingSecret=clickhouse-credentials 2>&1) && {
  echo "External ClickHouse accepted a missing password key" >&2
  exit 1
}
grep -Fq "externalClickhouse.existingSecretPasswordKey is required when clickhouse.enabled=false" \
  <<<"${external_clickhouse_password_key_output}"

# Generated embedded credentials must never use the shipped username as a known
# password, and all consumers must reference the generated `password` key.
generated_clickhouse_output=$(helm template "${HELM_RELEASE_NAME}" "${HELM_FOLDER}" \
  --set clickhouse.auth.existingSecretPasswordKey=not-the-generated-key)
if grep -Fq "bmVtbw==" <<<"${generated_clickhouse_output}"; then
  echo "Embedded ClickHouse rendered the known 'nemo' password" >&2
  exit 1
fi
if grep -Fq "not-the-generated-key" <<<"${generated_clickhouse_output}"; then
  echo "Embedded ClickHouse referenced a key not created by its generated Secret" >&2
  exit 1
fi

# ClickHouse probe overrides must render, and PVC template labels must remain
# stable across chart and application version upgrades.
clickhouse_statefulset_output=$(helm template "${HELM_RELEASE_NAME}" "${HELM_FOLDER}" \
  --show-only templates/clickhouse/clickhouse-statefulset.yaml \
  --set clickhouse.startupProbe.periodSeconds=17)
api_deployment_output=$(helm template "${HELM_RELEASE_NAME}" "${HELM_FOLDER}" \
  --show-only templates/api/api-deployment.yaml)
rotated_api_deployment_output=$(helm template "${HELM_RELEASE_NAME}" "${HELM_FOLDER}" \
  --show-only templates/api/api-deployment.yaml \
  --set-string clickhouse.auth.password=rotated-test-password)
grep -A6 -F "startupProbe:" <<<"${clickhouse_statefulset_output}" \
  | grep -Fq "periodSeconds: 17"
clickhouse_credentials_checksum=$(awk \
  '$1 == "checksum/clickhouse-credentials:" { print $2; exit }' \
  <<<"${clickhouse_statefulset_output}")
api_credentials_checksum=$(awk \
  '$1 == "checksum/clickhouse-credentials:" { print $2; exit }' \
  <<<"${api_deployment_output}")
rotated_credentials_checksum=$(awk \
  '$1 == "checksum/clickhouse-credentials:" { print $2; exit }' \
  <<<"${rotated_api_deployment_output}")
if [[ -z "${clickhouse_credentials_checksum}" || "${clickhouse_credentials_checksum}" != "${api_credentials_checksum}" ]]; then
  echo "API and ClickHouse workloads do not share the same credential checksum" >&2
  exit 1
fi
if [[ -z "${rotated_credentials_checksum}" || "${rotated_credentials_checksum}" == "${api_credentials_checksum}" ]]; then
  echo "ClickHouse credential checksum did not change with the configured password" >&2
  exit 1
fi
clickhouse_pvc_template=$(sed -n '/^  volumeClaimTemplates:/,$p' <<<"${clickhouse_statefulset_output}")
if grep -Eq "helm.sh/chart|app.kubernetes.io/version" <<<"${clickhouse_pvc_template}"; then
  echo "ClickHouse PVC template contains labels that change across chart upgrades" >&2
  exit 1
fi
grep -Fq "app.kubernetes.io/component: clickhouse" <<<"${clickhouse_pvc_template}"
grep -Fq "app.kubernetes.io/instance: ${HELM_RELEASE_NAME}" <<<"${clickhouse_pvc_template}"

# Null telemetry endpoint defaults mean "unset"; they must not render empty env
# vars that override the generic OTLP endpoint. Explicit empty strings remain
# valid user overrides and should render.
otel_default_output=$(helm template "${HELM_RELEASE_NAME}" "${HELM_FOLDER}" \
  --show-only templates/api/api-deployment.yaml)
if grep -Fq "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT" <<<"${otel_default_output}"; then
  echo "Default null OTEL metrics endpoint rendered an empty env var" >&2
  exit 1
fi
if grep -Fq "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" <<<"${otel_default_output}"; then
  echo "Default null OTEL traces endpoint rendered an empty env var" >&2
  exit 1
fi
otel_explicit_empty_output=$(helm template "${HELM_RELEASE_NAME}" "${HELM_FOLDER}" \
  --show-only templates/api/api-deployment.yaml \
  --set-string telemetry.OTEL_EXPORTER_OTLP_METRICS_ENDPOINT= \
  --set-string telemetry.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=)
grep -Fq "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT" <<<"${otel_explicit_empty_output}"
grep -Fq "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" <<<"${otel_explicit_empty_output}"

# Validate the Helm chart by rendering templates with all values files in ci/ directory
shopt -s nullglob
for value_file in "${HELM_FOLDER}"/ci/*.yaml; do
  echo "Validating Helm chart templating with values file: ${value_file}"
  helm template "${HELM_RELEASE_NAME}" "${HELM_FOLDER}" -f "${value_file}" > "${value_file}.output"
  echo "Validating Helm chart kubeconform with values file: ${value_file}"
  helm template "${HELM_RELEASE_NAME}" "${HELM_FOLDER}" -f "${value_file}" \
    | kubeconform -cache "${KUBECONFORM_CACHE}" -schema-location default \
      -schema-location "https://raw.githubusercontent.com/datreeio/CRDs-catalog/main/{{.Group}}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json" \
      -schema-location "https://raw.githubusercontent.com/yannh/kubernetes-json-schema/master/{{.NormalizedKubernetesVersion}}/{{.ResourceKind}}.json" \
      -summary -output json > "${value_file}.kubeconform.json"
done

# If all successful, cleanup the created files
rm -f "${HELM_FOLDER}"/ci/*.output "${HELM_FOLDER}"/ci/*.kubeconform.json "${HELM_FOLDER}"/ci/*.kubeconform-openshift.json
