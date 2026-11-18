<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Community plugin gateway and backend demo

A community plugin is a **YAML manifest** mapping gateway endpoints to external
APIs. Helix does **not** install Python wheels or launch plugin processes or
containers. The earlier subprocess prototype has been removed.
Studio runs approved UI code inside its own React tree; this is trusted code,
not an iframe or a sandbox.

## API contract for Studio

All management routes are under `/apis/plugins/-/`:

| Method / suffix | Behavior |
| --- | --- |
| `GET installed` | Installed manifest snapshots, including endpoints, navigation, bundle pin, and secret **names**, never values. |
| `GET catalog` | Available URL suggestions from the operator's local `catalog.json`; empty by default. No marketplace or automatic remote discovery. |
| `POST preview` | `{ "url": "https://.../manifest.json" }` → validated manifest, `manifest_sha256`, and trust warning. Fetches metadata only; does not install or execute a bundle. |
| `POST installed` | `{ "url": "...", "manifest_sha256": "...", "consent": true }` → refetch/revalidate, compare consented digest, download/verify bundle, atomically persist. Returns name, digest and `bundleUrl`. |
| `PATCH installed/{name}` | `{ "enabled": false }` (or true). |
| `DELETE installed/{name}` | Remove registration, retaining workspace secrets and content-addressed cached bytes. Removed bytes are no longer served. |

Preview and every mutation require a human **HelixAdmin**, plus the normal
`plugins:write` / `platform:write` scope checks. They have static `@path_rule`
admin permissions, not dynamically registered OPA routes. A conflicting install
returns 409; replace only by an explicit uninstall and newly consented install.
A changed manifest between preview and install also returns 409. Failed downloads
or hash checks never create an installed registration.

For an **auth-disabled, loopback-only throwaway demo**, the operator can explicitly
set `NEMO_PLUGINS_LOCAL_ADMIN=true` before platform startup. It is off by default;
only loopback clients with a loopback Host and no forwarded headers qualify.
Cross-origin browser administration is rejected. Never enable this flag on a
shared deployment or behind a reverse proxy. With auth enabled the flag cannot
bypass HelixAdmin checks. Studio development on a separate origin should use a
same-origin reverse proxy or a built Studio for admin requests.

The add-by-URL screen should show the complete endpoint mapping, navigation,
credential binding and bundle hash, plus the returned trust warning, **before**
sending explicit consent. No frontend management page is added here; this is its
backend contract.

## YAML manifest (version 1)

```yaml
schema_version: 1
plugin:
  name: tavily
  title: Tavily
  endpoints:
    - method: POST
      path: /v2/workspaces/{workspace}/mcp
      access: write
      upstream_url: https://mcp.tavily.com/mcp
  credential:
    secret_name: tavily-api-key
    scheme: bearer
```

JSON remains accepted for compatibility. YAML comments/key ordering/formatting do
not change consent: both formats validate to the same canonical snapshot and
hash. YAML uses a bounded safe loader: no Python constructors, anchors/aliases,
merge keys, duplicate keys, or non-JSON values. Unknown deployment fields such as
`wheel`, `image`, or `command` are rejected, not executed.

For a UI, add `bundle` and optional `navigation` inside `plugin`:

```yaml
bundle:
  url: https://publisher.example/index.js
  sha256: "<64 lowercase hexadecimal characters>"
  sidecar:
    contractVersion: "1.0.0"
    specifiers: [react]
    builtAgainst:
      react: "19.2.8"
navigation:
  - label: Tavily
    path: /workspaces/{workspace}/plugin/tavily
    section: Community
```

The sidecar is the existing plugin build's `plugin.manifest.json` content, embedded
in the consented manifest. Studio still performs its compatibility check before
import. Export the normal `Root` and `navItems`; no separate React root/router.
The bundle is a single built ESM file with Studio-provided shared imports, not a
remotely fetched tree of unpinned chunks or CSS assets.

After installation, existing `GET /apis/plugins` discovery includes enabled
community bundles. It exposes only name and `bundleUrl`. URLs have the form
`/plugin-ui/<name>/<revision>/index.js`, with the sidecar adjacent. The revision
covers **both** bundle digest and sidecar content. The platform verifies cached
JS bytes on every serve, so cache corruption fails closed. Only these two exact
files are served. No public proxy or on-demand remote bundle fetch is involved.
These are public static assets, like existing Python plugin UIs: **never embed
secrets**. Hash pinning verifies integrity, not code safety. Disabling a plugin
blocks new API/asset requests and discovery; it cannot revoke code already loaded
or cached in a browser.

## State, authorization, and upstream credentials

State: `<platform data directory>/plugins/installed.json`, using the standard
`NHX_DATA_DIR` / XDG / `~/.local/share/nemo` resolution. Writers use a POSIX file
lock and atomic replacement with private file permissions. Reads reload state.
The catalog is `<same directory>/catalog.json`, shaped as:

```json
{"entries": [{"name": "tavily", "title": "Tavily", "url": "https://publisher.example/tavily.json"}]}
```

Only declared method/path mappings are callable at
`/apis/plugins/<name>/<endpoint>`. Templates substitute safe single URL segments,
never authorities. Upstream/artifact URLs require HTTPS except explicit loopback
HTTP development servers. Credentials in URLs, static query strings, redirects,
ambiguous mappings, and undeclared calls are rejected. Artifact downloads are
bounded and uncompressed (256 KiB manifests, 16 MiB bundles).

Fixed gateway permissions: GET/HEAD require `plugins.gateway.read`; supported
mutating methods require `plugins.gateway.write`. Their normal scope gates are
`plugins:read` / `plugins:write` (or platform equivalents). GET/HEAD mappings must
be declared read, other methods write; POST-as-read is conservatively unsupported.
Permissions are currently gateway-wide, not per-plugin.

Use `/v2/workspaces/{workspace}/...` for workspace operations. Static OPA patterns
extract and check that workspace. Other paths are platform/system-scoped; query
parameters and arbitrary headers do not establish workspace authorization.
Wildcard workspace `-` is rejected. Mapping an authorized workspace to the correct
external tenant remains the administrator's responsibility.

An optional `credential` binding is restricted to workspace-shaped endpoints.
The gateway uses `AsyncSecretsClient.from_client(request_client)` to access the
named secret **as the caller in the URL's workspace**, then sends only
`Authorization: Bearer <secret value>` to the approved upstream. It never uses a
service-wide privileged identity. Missing keys return 412; denied access returns
403 without contacting the provider. Scoped keys need the permissions/scopes for
both the plugin call and secret access. No implicit environment-based credentials
are forwarded.

Caller platform tokens, cookies, identity/forwarded headers, and ambient client
cookies/proxy settings are never relayed externally. JSON/NDJSON/SSE responses
stream; redirects, upstream cookies and unsafe response headers are blocked.
The API proxy cannot serve unpinned executable JS/HTML: active/other content is a
non-executable attachment. Pinned bundles use the separate static route above.

## Reproducible demo

### Automated, no provider key required

```bash
uv run --frozen pytest packages/nhx_platform_runner/tests/test_community_plugin_installation.py -v
```

This covers preview/consent, change detection, hash failures, durable registration,
Studio discovery and pinned assets, workspace secret access/denial, proxying, and
teardown. `test_demo_lifecycle_against_real_loopback_http_provider` uses actual HTTP
to the demo server and downloads the real example UI bundle/sidecar. Platform
secret access is fixture-backed; it does **not** claim a live Tavily call or a
browser-rendered demo.

### Manual backend / Studio demo

First follow repository `SETUP.md` (do not wipe existing data). The platform must
have configured entities/secrets, the `plugins` service, and Studio. For a new
foreground process after setup, a minimal explicit service selection is:

```bash
export NHX_BASE_URL=http://localhost:8080
# Only for an auth-disabled local demo, BEFORE starting the platform:
export NEMO_PLUGINS_LOCAL_ADMIN=true
uv run nemo services run --services auth,entities,secrets,plugins,studio
```

Do not start a second process over an existing platform. Use a human HelixAdmin
bearer instead of local demo mode when authentication is enabled. In another
terminal, start the operator-owned manifest server:

```bash
uv run python -m nhx.platform_runner.community_plugins.demo_provider
```

Default mode is a clearly labeled **LOCAL Tavily STUB**; it makes no web calls.
Use the following shell flow against the running platform (set `HELIX_TOKEN` for
an authenticated deployment):

```bash
BASE=http://localhost:8080
AUTH=()
if [ -n "${HELIX_TOKEN:-}" ]; then AUTH=(-H "Authorization: Bearer $HELIX_TOKEN"); fi
SOURCE=http://127.0.0.1:8099/tavily.yaml

curl -fsS "${AUTH[@]}" -H 'Content-Type: application/json' \
  -d '{"name":"tavily-api-key","value":"demo-tavily-key"}' \
  "$BASE/apis/secrets/v2/workspaces/default/secrets"

curl -fsS "${AUTH[@]}" -H 'Content-Type: application/json' \
  -d "$(jq -n --arg url "$SOURCE" '{url:$url}')" \
  "$BASE/apis/plugins/-/preview" > /tmp/community-plugin-preview.json
jq . /tmp/community-plugin-preview.json
# STOP and review the endpoints, credential name, hash, and trust warning.
SHA=$(jq -r .manifest_sha256 /tmp/community-plugin-preview.json)
curl -fsS "${AUTH[@]}" -H 'Content-Type: application/json' \
  -d "$(jq -n --arg url "$SOURCE" --arg sha "$SHA" '{url:$url,manifest_sha256:$sha,consent:true}')" \
  "$BASE/apis/plugins/-/installed"

curl -fsS "${AUTH[@]}" -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"tavily_search","arguments":{"query":"demo"}}}' \
  "$BASE/apis/plugins/tavily/v2/workspaces/default/mcp"
```

Creating an existing secret returns 409; do not overwrite a real Tavily key with
the dummy key. Use a throwaway workspace for the stub in that case, changing both
secret and proxy paths. To demonstrate pinned UI installation from a source
checkout, repeat the preview/review/install flow with `/example.yaml`; refresh
Studio and open the Example navigation. It reuses the existing example UI (not
a new Tavily UI). An already installed Python Example UI intentionally conflicts.

For **live Tavily**, stop the demo manifest server and restart it with
`--live-tavily`. Preview again; the manifest now maps to `https://mcp.tavily.com/mcp`.
Store a real Tavily key in the selected workspace, explicitly uninstall the stub
registration, review the changed mapping, and install with the new consent digest.
The legacy Tavily Python plugin may coexist; its `/apis/tavily/...` routes are
not used by the new gateway. No live provider key is bundled in this repository.

Disable and uninstall:

```bash
curl -fsS "${AUTH[@]}" -X PATCH -H 'Content-Type: application/json' \
  -d '{"enabled":false}' "$BASE/apis/plugins/-/installed/tavily"
curl -fsS "${AUTH[@]}" -X DELETE "$BASE/apis/plugins/-/installed/tavily"
```

## Deployment is composed separately

The YAML is a registry contract, not a workload definition. An external publisher
can serve it as a static file. Running its API (a hosted API, a Python service,
an image, or another deployment system) is a separate concern. Registering a
manifest only connects the running service to the gateway; it never starts it.
Tavily needs no Helix-managed deployment because its MCP service is hosted.
The local stub demonstrates the same separation with an operator-started server.

For a future Phoenix trace plugin, define the adapter's API and Intake integration
separately, then map that running adapter's endpoints in YAML. Access to Intake
will need its own explicit integration credential/authorization contract, not a
forwarded user's platform token. No Phoenix/Intake endpoint mappings are claimed
or guessed by this demo.

The installed-plugins/add-by-URL/consent **UI** remains Nathan's work. This change
provides the backend and existing Studio loader integration; it does not add a
marketplace, automatic upgrades, container deployment, or security sandbox.
