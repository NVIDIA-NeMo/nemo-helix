<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native Relay capture provenance

`relay_full_0_9_3.json` is an actual OTLP/HTTP export collected on 2026-11-04
from the installed `nemo-relay==0.9.3` wheel, using
`OpenTelemetryConfig("full", local_http_collector_url)`. The collector decoded
`ExportTraceServiceRequest` protobuf bodies with `MessageToDict` and wrote the
requests as a JSON array. Native attributes, IDs, timestamps, relationships and
request order in this checked-in capture were retained, not hand-authored.

The **workload is synthetic and offline with respect to providers**: two nested
agent scopes, a model callback returning a constructed LangChain `ModelResponse`
(`hello back`, 11 input / 3 output / 14 total tokens), one successful lookup tool,
and one deliberately failing tool. `recorded-model` and `recorded-provider` are
explicit synthetic response metadata, not inferred real identities. No agent,
model provider, judge or external API was invoked. The only runtime HTTP request
is Relay's export to a loopback collector; initial uv dependency installation
may require package-network access.

## Reproduce the capture scenario

From the repository root:

```bash
uv run --no-project services/intake/tests/fixtures/generate_relay_full_capture.py /tmp/relay-full-0.9.3.json
```

The companion script is the retained capture scenario adapted into a standalone,
version-pinned script (Python 3.13, Relay 0.9.3 and the original LangChain/protobuf
versions). It uses a separate uv script environment so the release workspace's
newer Relay does not silently change this compatibility fixture. It requires
permission to bind a loopback socket. It is a manual fixture-maintenance tool,
not a provider-dependent test-suite prerequisite.

A new run **will not be byte-identical**: Relay creates new UUIDs, span/trace IDs,
timestamps, and potentially export batches. Compare normalized span kinds,
messages, identity, usage, topology and raw-attribute preservation rather than
volatile IDs against a previous run. Inspect the new export before intentionally
replacing the checked-in fixture. The regular regression test keeps the original
capture and checks its IDs/timestamps/ancestry against that capture, plus one
expected normalized payload shape. Focused tests separately cover standard
precedence, explicit zero, missing fields and malformed payloads.

The exporter layout was also checked against NVIDIA/NeMo-Relay tag `0.9.3`:
`crates/core/src/observability/{otel.rs,mod.rs}` and
`python/nemo_relay/integrations/langchain/_serialization.py`. This fixture proves
compatibility with that recorded format, not every framework or Relay version.
