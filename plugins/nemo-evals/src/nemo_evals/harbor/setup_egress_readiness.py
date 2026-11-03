# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks that an opened agent setup egress window works before the agent's setup runs.

OpenSandbox's egress sidecar (``dns+nft``) answers the sandbox's DNS itself and allows only addresses it has
answered for. Reading the rules back proves they were stored, not that names resolve through the sidecar yet,
so :data:`RESOLVE_HOSTS_SCRIPT` looks each setup host up from inside the sandbox, with short retries. The
lookup also teaches the sidecar the host's addresses before setup connects to them. A host that never resolves
is only reported: the list serves every agent on the profile, and the selected one may not need that host.
"""

import ipaddress

RESOLVE_ATTEMPTS = 5
RESOLVE_RETRY_DELAY_SEC = 2

# Looks up each of $HOSTS with the first resolver tool the image has: getent, Python or nslookup. Prints each host
# that never resolved, one per line, and exits 1 if any didn't; exits 0 without checking when the image has none.
RESOLVE_HOSTS_SCRIPT = """\
resolve() {
  if command -v getent > /dev/null 2>&1; then
    getent hosts "$1" > /dev/null 2>&1
    return
  fi
  for py in python3 python; do
    if command -v "$py" > /dev/null 2>&1; then
      "$py" -c 'import socket, sys; socket.getaddrinfo(sys.argv[1], None)' "$1" > /dev/null 2>&1
      return
    fi
  done
  if command -v nslookup > /dev/null 2>&1; then
    # An address after the answer's Name line, not the exit status: BusyBox's nslookup can fail for a host it
    # resolved when another search-domain candidate got NXDOMAIN.
    nslookup "$1" 2> /dev/null | awk '/^Name:/ { name = 1 } name && /^Address/ { found = 1 } END { exit !found }'
    return
  fi
  exit 0
}
failed=
for host in $HOSTS; do
  n=1
  until resolve "$host"; do
    if [ "$n" -ge "$ATTEMPTS" ]; then
      failed="$failed $host"
      break
    fi
    n=$((n + 1))
    sleep "$DELAY"
  done
done
[ -z "$failed" ] && exit 0
for host in $failed; do echo "$host"; done
exit 1
"""


def resolvable_hosts(hosts: tuple[str, ...]) -> tuple[str, ...]:
    """The hostnames a lookup can check: not wildcards, IP addresses or CIDRs."""
    names = []
    for host in hosts:
        if "*" in host:
            continue
        try:
            ipaddress.ip_network(host, strict=False)
        except ValueError:
            names.append(host)
    return tuple(names)
