<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Plugin endpoint reachability contracts

These opt-in tests make real HTTP requests to URLs selected by the Agents,
Models and Jobs helpers. They require Docker and a cached probe image, without
provider credentials or a running Helix platform.

## Run

```bash
docker pull python:3.12-slim-bookworm
uv run --frozen pytest tests/integration/network_reachability \
  --run-network-reachability -n 0 -v -rs \
  --junitxml=/tmp/helix-network-reachability.xml -o junit_family=legacy
```

Without `--run-network-reachability`, the suite skips. After opt-in, missing
Docker, missing images, startup errors and unexpected HTTP responses fail.
The tests never pull images automatically.

## Network topologies

The Linux contract tests use one container as the host surrogate and another as
the workload. The surrogate runs a platform listener on port 8080, an independent
loopback collector on 4318 and a reachable external listener on 9090. A separate
loopback listener on 8081 provides the negative control. The workload's host alias
maps explicitly to the surrogate, independently of Docker Desktop routing.
Host-network Jobs are represented by sharing the surrogate's network namespace.

On macOS, `test_desktop_host.py` instead starts a loopback listener directly in
the host Python process and accesses it from Docker Desktop. It checks Desktop
DNS and Harbor's explicit `host-gateway` mapping. It does not use Docker-in-Docker.
This test binds an ephemeral host port; other probes use only isolated container
listeners. Temporary listeners, containers and networks are removed on teardown.

Absolute paths such as `/probe/client.py` and `/platform_local_network.py` are
container mount destinations defined by the fixture. Host source paths are
resolved from the checkout; no fixed host installation path is required.

## Coverage and limits

| Consumer | Contract |
| --- | --- |
| Agents | Model gateway URL selection and rewriting |
| Telemetry | ATIF, OTLP and ATOF destinations in authoring and translated configs; Intake, external and colocated collectors |
| Models | Files/HF download and LoRA callback URLs |
| Jobs | Shared service and log URLs in host and bridge networks |

Every delivered Linux probe checks path/query, request body, authorization header
and destination port. Negative controls require connection refusal, rather than
DNS failure. Endpoint builders are production code; Linux context/configuration
is supplied by fixtures. The echo listeners do not implement plugin workflows,
model inference or telemetry serialization and persistence.

Custom host-loopback collectors and bridge Jobs with unchanged loopback service
URLs are skipped with explicit reasons. Run `--include-known-network-gaps` to
execute those contracts as ordinary tests. A passing suite with skips does not
establish reachability for those configurations.

`--reachability-platform-bind=all` is a diagnostic that widens only the isolated
Linux surrogate's platform listener. It never changes the real host's bind.

Real Fabric/Harbor/gateway/Intake acceptance has a separate opt-in test in
`plugins/nemo-evals/tests/integration/test_local_fabric_network.py`. These endpoint
probes do not certify full native macOS agent evaluation, Kubernetes, real
OpenShell policy, remote Docker, IPv6, TLS or restart/recovery. Local auth is
unsupported and outside this suite's scope.
