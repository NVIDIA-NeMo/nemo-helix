#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Use the real chart with only Kubernetes lookup stubbed, so these regressions
# run without a cluster. Persist the rendered install Secret into the next
# render's lookup result, just as Helm sees it on the first upgrade.
chart_tmp=$(mktemp -d)
trap 'rm -rf "${chart_tmp}"' EXIT
cp -R "${HELM_FOLDER:-k8s/helm}/." "${chart_tmp}/"
helpers="${chart_tmp}/templates/_helpers.tpl"
grep -Fq 'lookup "v1" "Secret" .Release.Namespace $secretName' "${helpers}"
sed -i 's/lookup "v1" "Secret" .Release.Namespace $secretName/(include "test.clickhouse.lookup" . | fromJson)/g' "${helpers}"
cat > "${chart_tmp}/templates/_test-lookup.tpl" <<'EOF'
{{- define "test.clickhouse.lookup" -}}
{{- .Values.testClickhouseSecret | default dict | toJson -}}
{{- end -}}
EOF

render() {
  helm template credentials-test "${chart_tmp}" \
    --set secrets.defaultEncryptionKey.existingSecret.name=test-api-env "$@"
}

password_data() {
  awk '/^# Source: .*\/clickhouse\/clickhouse-secret.yaml$/ { secret = 1; next }
       secret && /^  password:/ { gsub(/"/, "", $2); print $2; exit }' <<<"$1"
}

# Check both workload annotations against the Secret data, not just each other.
check_checksums() {
  local output="$1" data="$2" expected count
  expected=$(printf '%s' "${data}" | sha256sum | awk '{ print $1 }')
  count=$(awk -v expected="${expected}" '
    $1 == "checksum/clickhouse-credentials:" {
      gsub(/"/, "", $2)
      if ($2 != expected) exit 1
      count++
    }
    END { print count + 0 }
  ' <<<"${output}")
  test "${count}" = 2
}

install=$(render)
installed_data=$(password_data "${install}")
test -n "${installed_data}"
check_checksums "${install}" "${installed_data}"

upgrade=$(render --is-upgrade \
  --set-string "testClickhouseSecret.data.password=${installed_data}" \
  --set api.service.annotations.qa=1)
test "$(password_data "${upgrade}")" = "${installed_data}"
check_checksums "${upgrade}" "${installed_data}"

# An explicit password takes precedence over the stored password and rolls both
# workloads. Removing it later must reuse the newly stored value, not regenerate.
rotated_data=$(printf '%s' rotated-test-password | base64 | tr -d '\n')
rotated=$(render --is-upgrade \
  --set-string "testClickhouseSecret.data.password=${installed_data}" \
  --set-string clickhouse.auth.password=rotated-test-password)
test "$(password_data "${rotated}")" = "${rotated_data}"
test "${installed_data}" != "${rotated_data}"
check_checksums "${rotated}" "${rotated_data}"
reused=$(render --is-upgrade --set-string "testClickhouseSecret.data.password=${rotated_data}")
check_checksums "${reused}" "${rotated_data}"
explicit_install=$(render --set-string clickhouse.auth.password=rotated-test-password)
check_checksums "${explicit_install}" "${rotated_data}"
explicit_upgrade=$(render --is-upgrade \
  --set-string "testClickhouseSecret.data.password=${rotated_data}" \
  --set-string clickhouse.auth.password=rotated-test-password)
check_checksums "${explicit_upgrade}" "${rotated_data}"

# Live changes to a managed or supplied Secret must still change the hash. An
# existingSecret's custom key takes precedence over an ignored auth.password.
for data in "${installed_data}" "${rotated_data}"; do
  managed=$(render --is-upgrade --set-string "testClickhouseSecret.data.password=${data}")
  test "$(password_data "${managed}")" = "${data}"
  check_checksums "${managed}" "${data}"
  supplied=$(render --is-upgrade \
    --set clickhouse.auth.existingSecret=supplied-credentials \
    --set clickhouse.auth.existingSecretPasswordKey=custom-key \
    --set-string clickhouse.auth.password=ignored-password \
    --set-string "testClickhouseSecret.data.custom-key=${data}")
  test -z "$(password_data "${supplied}")"
  check_checksums "${supplied}" "${data}"
done

echo 'ClickHouse credential install/upgrade regression checks passed'
