#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The control that makes a green build mean something.
#
# A successful kaniko build in `nmp-builds` proves nothing on its own -- it is equally consistent
# with the namespace not enforcing at all. So before trusting any positive result, submit the
# posture rootless BuildKit requires and watch admission refuse it. If this script reports
# ADMITTED, every other result in this PoC is void.
#
# Adapted from scratch/builderlab/11-baseline-rejects-buildkit.sh, which measured on this cluster:
#   violates PodSecurity "baseline:latest": forbidden AppArmor profile ...,
#   seccompProfile (container "c" must not set securityContext.seccompProfile.type to "Unconfined")
set -uo pipefail
NS="${NS:-nmp-builds}"
POD="nmp-build-negative-control"

kubectl -n "$NS" delete pod "$POD" --ignore-not-found >/dev/null 2>&1

echo "Submitting the BuildKit posture to namespace '$NS'..."
out=$(kubectl -n "$NS" run "$POD" --image=busybox --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"c","image":"busybox","command":["true"],
    "securityContext":{"seccompProfile":{"type":"Unconfined"},
                       "appArmorProfile":{"type":"Unconfined"},
                       "allowPrivilegeEscalation":true}}]}}' 2>&1)
echo "$out"
echo

if grep -q 'violates PodSecurity "baseline' <<<"$out"; then
  echo "PASS -- refused. The namespace is enforcing baseline; a green build here is meaningful."
  exit 0
fi

echo "FAIL -- the BuildKit posture was ADMITTED (or failed for an unrelated reason)."
echo "        Do not trust any positive build result from this namespace until this passes."
kubectl -n "$NS" delete pod "$POD" --ignore-not-found >/dev/null 2>&1
exit 1
