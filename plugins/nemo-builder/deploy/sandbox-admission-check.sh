#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Asserts that 25-sandbox-admission.yaml holds: acting as `nhx-build-control`, a sandbox-shaped
# pod is admitted, and each way out of that shape is refused BY THE POLICY.
#
# Server-side dry runs, impersonating the ServiceAccount, so nothing is created. Needs a kubeconfig
# allowed to impersonate it; a cluster admin's is.
#
# A refusal counts only if it names the policy. A pod refused for some other reason -- RBAC, a
# typo in this script -- is not evidence that the policy works, and counting it would let this
# script pass on a cluster where the policy was never applied.
set -uo pipefail
NS="${NS:-nhx-builds}"
AS="system:serviceaccount:${NS}:nhx-build-control"
POLICY="nhx-build-sandbox-shape"

WORK='{ name: work, persistentVolumeClaim: { claimName: nhx-build-work } }'
SECRET='{ name: key, secret: { secretName: cosign-key } }'
SLICE='{ name: work, mountPath: /w, subPath: jobs/default/some-job }'
WHOLE='{ name: work, mountPath: /w }'
KEY_MOUNT='{ name: key, mountPath: /k }'

pod() {  # label, serviceaccount, automount token, volumes, mounts
  cat <<YAML
apiVersion: v1
kind: Pod
metadata:
  generateName: nhx-admission-check-
  namespace: $NS
  labels: { nhx.nvidia.com/sandbox: "$1" }
spec:
  restartPolicy: Never
  serviceAccountName: $2
  automountServiceAccountToken: $3
  enableServiceLinks: false
  containers:
    - name: c
      image: busybox
      securityContext:
        allowPrivilegeEscalation: false
        seccompProfile: { type: RuntimeDefault }
        capabilities: { drop: [ALL] }
      volumeMounts: [$5]
  volumes: [$4]
YAML
}

# The API server loads a new policy asynchronously, and requests in the first seconds after
# `kubectl apply` are admitted without it -- measured. Wait until a pod the policy must refuse is
# refused, for up to 30s. If it never is, carry on: the checks below then fail, which is correct.
for _ in $(seq 1 15); do
  printf '%s\n' "$(pod false default false "$WORK" "$SLICE")" \
    | kubectl --as="$AS" create --dry-run=server -f - 2>&1 | grep -q "$POLICY" && break
  sleep 2
done

fail=0
check() {  # expectation (admit|refuse), description, manifest
  out=$(printf '%s\n' "$3" | kubectl --as="$AS" create --dry-run=server -f - 2>&1)
  rc=$?
  if [ "$1" = admit ]; then
    if [ $rc -eq 0 ]; then echo "ok    admitted: $2"
    else echo "FAIL  $2 -- expected to be admitted: $out"; fail=1; fi
  else
    if [ $rc -ne 0 ] && grep -q "$POLICY" <<<"$out"; then echo "ok    refused:  $2"
    else echo "FAIL  $2 -- expected the policy to refuse it: $out"; fail=1; fi
  fi
}

check admit  "the sandbox's own shape"                    "$(pod true default false "$WORK" "$SLICE")"
check refuse "a pod without the sandbox label"            "$(pod false default false "$WORK" "$SLICE")"
check refuse "a pod running as nhx-build-push"            "$(pod true nhx-build-push false "$WORK" "$SLICE")"
check refuse "a pod with a ServiceAccount token"          "$(pod true default true "$WORK" "$SLICE")"
check refuse "a pod mounting the signing key"             "$(pod true default false "$WORK, $SECRET" "$SLICE, $KEY_MOUNT")"
check refuse "a pod mounting the whole work volume"       "$(pod true default false "$WORK" "$WHOLE")"

echo
if [ "$fail" -eq 0 ]; then echo "PASS -- nhx-build-control can create the sandbox and nothing else."
else echo "FAIL -- see above. nhx-build-control is equivalent to nhx-build-push until this passes."; fi
exit "$fail"
