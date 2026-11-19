<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phoenix trace viewer, backed by Intake

This is an **independently run FastAPI adapter**, not an installed Helix Python
plugin or an Arize Phoenix server. Helix registers its YAML manifest, proxies
only its declared APIs, and caches its pinned Studio bundle. The registry never
installs, imports, or launches the adapter. No mock gateway is needed.

```text
Studio / producer → Helix manifest gateway → external FastAPI adapter
                                           → typed Intake client → Intake → ClickHouse
```

Reads include existing Intake spans from all sources, not just adapter uploads.
Projects come from Intake's workspace-scoped project aggregation. Unassigned
spans appear as `(unassigned)`. Parent IDs, timestamps, status, nested attributes,
inputs/outputs, models, and token counts are projected into the viewer's
Phoenix-shaped DTO. Writes use Intake's direct span ingestion, including its
validation, replay/upsert behavior, and 90-day retention policy.

## Boundaries and current limits

- One adapter instance is fixed to **one workspace** (`default` initially).
  Other workspace paths fail closed, even if someone copies its secret there.
  A single registration does not yet serve multiple workspace identities.
- `PHOENIX_ADAPTER_KEY` is a capability granting **read and write access to that
  workspace's traces through this adapter**. Protect both it and permission to
  access its Helix workspace secret. It is **not** an Intake/platform bearer.
- `PHOENIX_INTAKE_TOKEN` is a separate, operator-provisioned Intake credential.
  With platform auth enabled, scope it to the configured workspace and required
  span/group read + ingest operations. Individual browser identities are not
  forwarded to Intake: Helix authorizes gateway requests and secret access, then
  the adapter uses its configured identity. This is not per-user delegation.
- API keys are required even for the loopback adapter. Non-loopback service
  URLs require HTTPS. The default listener is loopback; do not expose this demo
  directly to an untrusted network. Production hosting needs TLS, ingress body
  limits, rate limits, credential rotation, and appropriate secret-access policy.
- The viewer loads at most 10 pages of 1,000 spans/projects. It **fails rather
  than displays incomplete traces** above that limit. Offset paging is not a
  snapshot under concurrent writes. Server-side trace navigation is follow-up.
- This is a trace viewer plus a Phoenix-shaped batch upload endpoint, **not** the
  full Phoenix API/OTLP server, annotation editor, or evaluation service.
- Bundle integrity is not a sandbox: approved JS runs inside Studio's React tree.
  Rebuilds require another preview, consent, and registration to update the pin.

## Local demo

Use this branch's platform/Intake (it adds `by=project` grouping). All commands
below run from the repository root in **Bash**. Prerequisites: `uv`, the pinned
Node/pnpm toolchain, `curl`, `jq`, and `openssl`.

### 1. Have Helix and ClickHouse-backed Intake running

Follow [SETUP.md](../../SETUP.md) and the repository's setup skill first. Do not
reset an existing data directory for this demo. Studio must also be built/served
per the normal Studio setup. Verify the existing platform:

```bash
export NHX_BASE_URL=http://127.0.0.1:8080
curl -fsS "$NHX_BASE_URL/health/ready"
```

For auth-enabled Helix, set `PHOENIX_PLATFORM_TOKEN` to the human administrator's
access token in your shell (never in a manifest or committed file). Subsequent
API calls also need permission to read the workspace secret and read/write the
plugin gateway. For the explicitly trusted auth-disabled local demo, start the
platform with `NEMO_PLUGINS_LOCAL_ADMIN=true`; administration still requires a
loopback client/Host and no forwarded headers. This does not bypass auth-enabled
HelixAdmin checks.

```bash
AUTH=()
if [[ -n "${PHOENIX_PLATFORM_TOKEN:-}" ]]; then
  AUTH=(-H "Authorization: Bearer $PHOENIX_PLATFORM_TOKEN")
fi
curl -fsS "${AUTH[@]}" "$NHX_BASE_URL/apis/intake/v2/workspaces/default/spans?page_size=1"
```

### 2. Build the UI, then start the external adapter

The UI uses Common's source types, so bootstrap Studio dependencies/SDK first
(`make bootstrap-studio`, as described in SETUP). Its runtime dependencies remain
externalized; the plugin does not bundle another React/router/query client.

```bash
(cd plugins/phoenix-traces/web && pnpm install --frozen-lockfile && pnpm build)
uv sync --project plugins/phoenix-traces/backend --python 3.13 --frozen

export PHOENIX_WORKSPACE=default
export PHOENIX_INTAKE_URL="$NHX_BASE_URL"
export PHOENIX_PUBLIC_URL=http://127.0.0.1:6007
export PHOENIX_ADAPTER_KEY="$(openssl rand -hex 32)"
# With Intake auth enabled, also export PHOENIX_INTAKE_TOKEN using a distinct,
# workspace-scoped Intake credential. Leave it unset only for auth-disabled Intake.
uv run --project plugins/phoenix-traces/backend --frozen python -m nemo_phoenix_traces \
  > /tmp/phoenix-intake-adapter.log 2>&1 &
ADAPTER_PID=$!

# This checks actual Intake/ClickHouse access, not merely a live HTTP listener.
curl -fsS --retry 10 --retry-connrefused --retry-delay 1 --retry-max-time 20 \
  -H "Authorization: Bearer $PHOENIX_ADAPTER_KEY" \
  "$PHOENIX_PUBLIC_URL/v1/workspaces/default/projects"
```

Stop if verification fails; inspect the adapter log and Intake health/config.
The adapter has its own `.venv`/lockfile and only the public plugin SDK, FastAPI,
and HTTP/client dependencies, not Helix service implementation packages.

### 3. Provision the workspace credential and register by URL

Run in the same shell, so the generated key stays out of command arguments and
files. This creates a new secret; if it already exists, rotate/update it through
the normal Secrets workflow rather than overwriting someone else's credential.

```bash
jq -n '{name:"phoenix-traces-adapter-key",value:env.PHOENIX_ADAPTER_KEY}' |
  curl -fsS "${AUTH[@]}" -H 'Content-Type: application/json' --data-binary @- \
    "$NHX_BASE_URL/apis/secrets/v2/workspaces/default/secrets"

PREVIEW="$(jq -n --arg url "$PHOENIX_PUBLIC_URL/plugin.yaml" '{url:$url}' |
  curl -fsS "${AUTH[@]}" -H 'Content-Type: application/json' --data-binary @- \
    "$NHX_BASE_URL/apis/plugins/-/preview")"
printf '%s\n' "$PREVIEW" | jq .
```

**Stop and review** the upstream URLs, read/write routes, credential reference,
UI SHA-256/sidecar, and trusted same-origin JS warning. Only after approving that
exact snapshot:

```bash
DIGEST="$(printf '%s' "$PREVIEW" | jq -er .manifest_sha256)"
jq -n --arg url "$PHOENIX_PUBLIC_URL/plugin.yaml" --arg sha "$DIGEST" \
  '{url:$url,manifest_sha256:$sha,consent:true}' |
  curl -fsS "${AUTH[@]}" -H 'Content-Type: application/json' --data-binary @- \
    "$NHX_BASE_URL/apis/plugins/-/installed"
```

`/plugin.yaml` adds the actual built bundle's hash and sidecar. Without a build,
it registers APIs only. The checked-in `plugin.yaml` is an API-only example;
register the served URL for the UI. Keep the adapter process running separately.

#### Or register from Studio

Studio has a registry page for the same preview, consent and install flow:
**Settings → Community Plugins** (also **System → Community Plugins** in the side
nav), at `/workspaces/<workspace>/community-plugins`. Click **Add Plugin**, enter
`$PHOENIX_PUBLIC_URL/plugin.yaml`, review the endpoints, secret name, bundle
SHA-256 and trust warning, tick the consent box, and click **Register**. The
workspace secret from this step must exist first. After registering, the side nav
gains **Phoenix** and the viewer is at `/workspaces/<workspace>/plugin/phoenix`.

Notes for trying it locally:

- Plugin administration only accepts **same-origin** requests, so use a Studio
  that the platform serves itself (a built Studio at `/studio`). A Studio dev
  server on another origin gets `403 Plugin administration requires a same-origin
  request`.
- A dev server also does not proxy `/plugin-ui/`, so community bundles fail with
  `PLUGIN_LOAD_FAILED` unless `VITE_PLATFORM_BASE_URL` is the platform's absolute
  URL.
- If a Phoenix container already uses port 6007, run the adapter on another
  port (`python -m nemo_phoenix_traces --port 6008`) and use that port in
  `PHOENIX_PUBLIC_URL`.
- To view traces you already have in Arize Phoenix, post its
  `GET /v1/projects/<project>/spans` output to the registered `spans` route.
  The adapter rejects Phoenix's extra `id` and `events` span fields, so remove
  them first.

### 4. Write a clearly labeled sample trace through the registered gateway

This generates synthetic spans, **not an LLM call**, but stores real records in
Intake/ClickHouse. IDs are fresh on every invocation.

```bash
TRACE_ID="$(openssl rand -hex 16)"
ROOT_ID="$(openssl rand -hex 8)"
CHILD_ID="$(openssl rand -hex 8)"
jq -n --arg trace "$TRACE_ID" --arg root "$ROOT_ID" --arg child "$CHILD_ID" '
  now as $t | {spans:[
    {name:"DEMO agent (persisted)",context:{trace_id:$trace,span_id:$root},
     span_kind:"AGENT",start_time:(($t-3)|todateiso8601),end_time:($t|todateiso8601),
     status_code:"OK",attributes:{"input.value":{"question":"Where are traces stored?"}}},
    {name:"DEMO model call",context:{trace_id:$trace,span_id:$child},parent_id:$root,
     span_kind:"LLM",start_time:(($t-2)|todateiso8601),end_time:(($t-1)|todateiso8601),
     status_code:"OK",attributes:{"llm.model_name":"demo-model",
       "llm.token_count.prompt":7,"llm.token_count.completion":5,"llm.token_count.total":12,
       "output.value":"Intake persists these spans in ClickHouse.","demo.synthetic":true}}
  ]}' |
  curl -fsS "${AUTH[@]}" -H 'Content-Type: application/json' --data-binary @- \
    "$NHX_BASE_URL/apis/plugins/phoenix/v2/workspaces/default/spans?project=phoenix-demo"

curl -fsS "${AUTH[@]}" \
  "$NHX_BASE_URL/apis/plugins/phoenix/v2/workspaces/default/spans?project=phoenix-demo" | jq .
```

Open Studio at `/workspaces/default/plugin/phoenix` and select `phoenix-demo`.
Existing agents can keep emitting real OTLP spans to Intake's standard endpoint:
`/apis/intake/v2/workspaces/default/ingest/otlp/v1/traces`. No agent needs this
adapter in its write path. Assign `project.name` on spans to group them, or use
`(unassigned)`. The Intake skill also documents importing existing Phoenix data.

Stop/restart only the adapter with the same environment; the trace remains in
Intake. `kill "$ADAPTER_PID"` stops the operator-owned process; disabling or
unregistering the manifest does not stop it or delete stored traces. Unregister
with `DELETE /apis/plugins/-/installed/phoenix` when finished; remove/rotate the
demo secret separately after confirming it is not shared.

## Verification

```bash
uv run --project plugins/phoenix-traces/backend --frozen pytest plugins/phoenix-traces/backend/tests -q
uv run --frozen pytest services/intake/tests/integration/spans/test_phoenix_plugin.py -v
(cd plugins/phoenix-traces/web && pnpm test && pnpm typecheck && pnpm build)
```

The integration test uses disposable ClickHouse and temporary registry/SQLite
state, a real external loopback HTTP adapter, preview + digest consent + install,
gateway workspace-secret lookup, and the actual Intake ingestion/read APIs. It
checks nested payloads, semantic fields, parent relationships, cursor paging,
workspace rejection, and a read after adapter restart. Only the test transport
between the typed Intake client and the fixture's Intake ASGI app is bridged;
the span repository/ClickHouse are real. Authorization is disabled in that Intake
fixture; production gateway/admin/OPA guards have their own regression suites.
