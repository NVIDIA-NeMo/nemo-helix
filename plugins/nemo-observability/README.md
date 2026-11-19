<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Observability plugin (POC)

Embeds a trace explorer the customer already runs (MLflow or Phoenix) in NeMo
Studio, as **Observability → Trace Explorer**. Helix never hosts the platform; it
proxies the platform's UI through its own origin and tells agents how to export
traces to it.

Status: proof of concept, tracked in ASTD-939. The plugin is a dev dependency only.
It loads in a source checkout's `nemo services run`, but it is not in
`enabled-plugins` and no shipped image carries it.

## How it works

```text
Studio ── Trace Explorer page ── POST .../connections/<name>/session  (bearer)  → sets cookie
   └── <iframe src="/apis/observability/ui/<name>/...">  (cookie)  ──proxy──▶  upstream UI
Agent (Relay) ── OTLP/HTTP ─────────────────────────────────────────────────▶  upstream /v1/traces
```

- **Same-origin proxy.** `/apis/observability/ui/<name>/…` streams to the upstream.
  The upstream must be configured to serve under that exact prefix, as shown below.
  Studio's CSP allows `frame-src 'self'` because the plugin's `StudioSpec` declares it.
- **Session cookie.** A frame cannot send a bearer token, so the page mints an
  HMAC-signed, `HttpOnly`, `SameSite=Strict` cookie scoped by `Path` to the one
  connection, and re-mints it before it expires.
- **Proxy behavior.** The proxy strips the caller's `Authorization`, the platform's
  cookies and the `X-NHX-*` headers. It injects the upstream credential from
  `secret_env`, rewrites `Origin` and `Location` to the upstream's origin, and keeps
  `Content-Length`.
- **Trace export.** Export is a `telemetry.opentelemetry` stanza in the agent's
  `agent.yaml`. Relay sends OTLP straight to the upstream, alongside Helix's own
  ATIF telemetry to Intake. `GET .../connections/<name>/otlp-snippet` (the page's
  **Send agent traces here** button) generates it.

### Known limits (POC)

- **Auth-enabled deployments are not supported yet.** The platform auth middleware
  authenticates only bearer tokens or trusted headers, so it rejects the frame's
  cookie-only requests before they reach the plugin. Local mode (auth disabled)
  works. See ASTD-939 for the follow-up.
- **The embedded UI runs as trusted code on Studio's origin.** Its JavaScript can do
  anything a Studio page can, so connect only platforms you trust.
- **Studio and the API must share an origin.** A cross-origin `apiBaseUrl` would need
  that origin in `frame-src`.
- **The session key is per process** unless you set `session_secret`. With more than
  one replica, set it.

## Configure a connection

Add the connection to the platform config file:

```yaml
observability:
  connections:
    - name: mlflow            # fixes the prefix /apis/observability/ui/mlflow
      provider: mlflow        # mlflow | phoenix
      base_url: http://127.0.0.1:5000   # origin the platform reaches; no path
      default_project: "0"    # MLflow experiment id / Phoenix project name
      # secret_env: MLFLOW_AUTH   # env var with the upstream credential
      # otlp_endpoint: ...        # if agents reach the upstream by another host
  # session_ttl_seconds: 900
  # session_secret: <random>      # required with more than one replica
```

Or set it in the environment:
`NEMO_OBSERVABILITY_CONNECTIONS='[{"name":"mlflow","provider":"mlflow","base_url":"http://127.0.0.1:5000","default_project":"0"}]'`.

## Runbooks

### MLflow (Apache-2.0), verified

```bash
mlflow server --host 127.0.0.1 --port 5000 \
  --static-prefix /apis/observability/ui/mlflow \
  --backend-store-uri sqlite:///mlflow.db
```

- `--static-prefix` moves **every** route under the prefix: the UI, the REST API,
  `/health`, and OTLP at `<prefix>/v1/traces`. The generated snippet accounts for
  this.
- MLflow 3's security middleware rejects a non-localhost `Origin` on writes with
  403 "Cross-origin request blocked", and it validates `Host`. The proxy rewrites
  `Origin` and sends the upstream `Host`, so the defaults work. If MLflow listens
  on a non-loopback address, configure `--allowed-hosts` for the address the
  platform uses.
- MLflow's WSGI layer reads a chunked request body as empty. That is why the proxy
  forwards `Content-Length`.
- MLflow sends `X-Frame-Options: SAMEORIGIN`, which is fine because the frame is
  same-origin through the proxy.
- OTLP: `type: gen_ai`, `transport: http_binary`, header
  `x-mlflow-experiment-id: <id>`. MLflow derives token usage from the
  `gen_ai.usage.*` attributes.

### Phoenix (Elastic License 2.0), verified

ELv2 lets a customer self-host Phoenix and use it this way. It does **not** let
Helix offer Phoenix to third parties as a hosted or managed service.

```bash
PHOENIX_HOST_ROOT_PATH=/apis/observability/ui/phoenix \
PHOENIX_ROOT_URL=http://<studio-host>/apis/observability/ui/phoenix \
phoenix serve
```

- Set **both** variables. Phoenix serves at its own root and only prefixes the URLs
  it emits, so this adapter strips the prefix.
- GraphQL subscriptions stream as HTTP multipart over `POST /graphql`, with no
  WebSockets. The proxy streams them.
- Setting `PHOENIX_HOST` to a non-loopback address turns Phoenix auth on unless
  `PHOENIX_ENABLE_AUTH` says otherwise. In that case, put an API key in `secret_env`.
- OTLP: `type: openinference` to `<phoenix>/v1/traces`. The project comes from the
  `openinference.project.name` resource attribute. The startup banner advertises
  `<prefix>/v1/traces`, but that path answers 405 to POST; use the root path.

### Langfuse (MIT core), not yet an adapter

The base path `NEXT_PUBLIC_BASE_PATH` is a build-time setting, so it needs a custom
image built with `/apis/observability/ui/<name>`. Langfuse sends
`X-Frame-Options: SAMEORIGIN`, which works same-origin. OTLP goes to
`/api/public/otel/v1/traces` with Basic auth, which `header_env` can carry. It also
supports OIDC SSO. Adding Langfuse means writing an adapter plus this runbook.

## Develop

```bash
uv run --frozen pytest plugins/nemo-observability/tests
cd plugins/nemo-observability/web && pnpm install && pnpm typecheck && pnpm build
```
