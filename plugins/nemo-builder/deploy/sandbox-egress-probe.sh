#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Asserts what the sandbox NetworkPolicy actually closes, by running a pod wearing the sandbox
# label and measuring reachability from inside it.
#
# This exists because 30-networkpolicy.yaml is the design's only high-severity risk and it is
# NOT reviewable by eye: NetworkPolicy has no deny verb, so a policy that reads as "deny private,
# allow public" can be a policy that allows everything. The only trustworthy check is a packet.
#
# Run it after any edit to 30-networkpolicy.yaml, and on any new cluster before trusting a build.
#
# Note on shell: every check captures an exit status directly. An earlier version of this script
# piped wget into `head` and read the PIPELINE's status, which is head's -- so a blocked
# metadata server reported as REACHABLE. Do not reintroduce a pipe here.
#
# A caveat this script cannot fix: on a cluster with no metadata server -- any local one -- the
# metadata check passes because the address does not exist, not because the policy denied it.
# It is a real assertion only on a cloud cluster. Nothing here distinguishes "denied" from
# "absent", and pretending otherwise would be worse than saying so.
set -uo pipefail
NS="${NS:-nmp-builds}"
POD="nmp-sandbox-egress-probe"
NODE_SELECTOR="${NODE_SELECTOR:-nmp.nvidia.com/build-node}"

# Derived from the cluster, not hardcoded. The policy's `except` list is written in terms of
# RFC1918 and link-local, which are universal -- but *which* addresses that actually denies is a
# per-cluster fact, and a policy that is sufficient on one cluster can be insufficient on
# another whose CIDRs sit outside those blocks. So the assertions target this cluster's real
# addresses: the API service, a real kube-dns pod, and this cluster's DNS resolver.
API_IP=$(kubectl get svc kubernetes -n default -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
DNS_SVC_IP=$(kubectl get svc -n kube-system -l k8s-app=kube-dns -o jsonpath='{.items[0].spec.clusterIP}' 2>/dev/null)
DNS_POD_IP=$(kubectl get pods -n kube-system -l k8s-app=kube-dns -o jsonpath='{.items[0].status.podIP}' 2>/dev/null)
: "${API_IP:?could not read the kubernetes Service ClusterIP}"
: "${DNS_SVC_IP:?could not read the kube-dns Service ClusterIP}"
: "${DNS_POD_IP:?could not read a kube-dns pod IP}"
echo "cluster under test: api=$API_IP dns-svc=$DNS_SVC_IP dns-pod=$DNS_POD_IP"

kubectl -n "$NS" delete pod "$POD" --ignore-not-found >/dev/null 2>&1

cat <<YAML | kubectl apply -f - >/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: $POD
  namespace: $NS
  labels:
    nmp.nvidia.com/sandbox: "true"
spec:
  restartPolicy: Never
  automountServiceAccountToken: false
  nodeSelector: { $NODE_SELECTOR: "true" }
  dnsPolicy: None
  dnsConfig:
    nameservers: ["8.8.8.8", "1.1.1.1"]
  containers:
    - name: probe
      image: alpine:3.20
      securityContext:
        runAsUser: 0
        allowPrivilegeEscalation: false
        seccompProfile: { type: RuntimeDefault }
        capabilities: { drop: [ALL], add: [CHOWN, DAC_OVERRIDE, FOWNER, SETUID, SETGID] }
      command:
        - sh
        - -c
        - |
          fail=0
          must_work() {  # name, command...
            n="\$1"; shift
            if "\$@" >/dev/null 2>&1; then echo "ok    \$n"
            else echo "FAIL  \$n -- expected to succeed"; fail=1; fi
          }
          must_block() {
            n="\$1"; shift
            if "\$@" >/dev/null 2>&1; then echo "OPEN  \$n -- expected to be BLOCKED"; fail=1
            else echo "ok    \$n (blocked)"; fi
          }
          must_work  "resolve a public name"            nslookup pypi.org
          must_work  "reach the public internet"        wget -q -T 15 -O /dev/null https://pypi.org/simple/
          must_block "metadata server 169.254.169.254"  nc -z -w 6 169.254.169.254 80
          must_block "node-local DNS 169.254.20.10:53"  nc -z -w 6 169.254.20.10 53
          must_block "kubernetes API $API_IP:443"       nc -z -w 6 $API_IP 443
          must_block "kube-dns Service $DNS_SVC_IP:53"  nc -z -w 6 $DNS_SVC_IP 53
          must_block "a Pod CIDR address $DNS_POD_IP:53" nc -z -w 6 $DNS_POD_IP 53
          echo
          if [ "\$fail" -eq 0 ]; then echo "PASS -- the sandbox reaches the public internet and nothing private."
          else echo "FAIL -- see above. Do not run untrusted builds in this namespace."; fi
          exit "\$fail"
YAML

kubectl -n "$NS" wait --for=jsonpath='{.status.phase}'=Succeeded pod/"$POD" --timeout=180s >/dev/null 2>&1
rc=$?
kubectl -n "$NS" logs "$POD" 2>&1
kubectl -n "$NS" delete pod "$POD" --ignore-not-found >/dev/null 2>&1
exit $rc
