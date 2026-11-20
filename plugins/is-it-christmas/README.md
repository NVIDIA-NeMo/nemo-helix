<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Is it Christmas? (a Rust community plugin)

An example NeMo Helix community plugin whose **backend is written in Rust**. It shows
that a plugin is a manifest plus HTTP, not Python: Helix never runs or imports this
code, it only proxies to it.

```
browser ──▶ Studio ──▶ Helix gateway ──▶ Rust server (axum, :6010)
                       /apis/plugins/is-it-christmas/...
```

| Piece | Where | What it does |
| --- | --- | --- |
| Backend | `backend/` | axum server: the API, the manifest, and the UI bundle |
| UI | `web/` | A page in Studio's React tree, built with the Studio plugin helper |

The server exposes:

- `GET /plugin.json`: the manifest Helix registers. The UI bundle's SHA-256 is computed
  from the file it serves, so the pin always matches.
- `GET /index.js`: the built UI bundle.
- `GET /v1/workspaces/{workspace}/is-it-christmas?date=YYYY-MM-DD`: the API. `date` is
  the caller's local date; it defaults to today in UTC.

- `GET /v1/workspaces/{workspace}/panic`: panics on purpose, to show what a Rust failure looks
  like end to end. The handler does `"ho ho ho".parse::<u32>().unwrap()`, catches the panic so
  the server keeps serving, and answers HTTP 500 with the text Rust itself prints, including
  `ParseIntError { kind: InvalidDigit }` and the `RUST_BACKTRACE` note. Rust also prints it to its
  own stderr. Try `nemo plugins call is-it-christmas panic`, or the "Trigger a Rust panic" link in
  the Studio page. The CLI retries 5xx responses, so one call shows up as three panics.

- `GET /v1/workspaces/{workspace}/stefan?mood=happy|sad` and `.../stefan-animated`: a still JPEG (the
  waving one for YES, the crying one for NO; shown under the answer) and an animated GIF (shown on the panic page), compiled into the binary from
  `backend/assets/` (web-sized copies of the originals). They answer **JSON with a `data:` URL**, not
  raw `image/jpeg`: the Helix gateway relays only JSON and text as-is and rewrites any other content
  type to an `application/octet-stream` download, which an `<img>` can't render. Studio's content
  security policy allows `data:` images. The animation is its own endpoint so the big image never
  rides in the panic's 500 body. `nemo plugins call is-it-christmas stefan` shortens the long value;
  add `-f json` for all of it.

There is no credential: the endpoint is public on loopback. A real plugin would add a
`credential` (a workspace secret sent as a bearer token), as the Phoenix plugin does.

## Run it yourself, and change the answer

The server is an ordinary process: run it in your own terminal and restart it freely. Helix
only needs the URL to answer, so a restart needs no re-registration.

```bash
cd plugins/is-it-christmas/backend
CHRISTMAS_OVERRIDE=no cargo run       # or yes, or auto (the default: follow the calendar)
```

| Variable | Default | What it does |
| --- | --- | --- |
| `CHRISTMAS_OVERRIDE` | `auto` | `yes` or `no` forces the answer (and the happy or sad Stefan). Anything else is an error. |
| `PORT` | `6010` | Port to listen on (loopback only). |
| `PUBLIC_URL` | `http://127.0.0.1:$PORT` | The address Helix uses in the manifest. Set it to a tap's address to route through one. |
| `BUNDLE_DIR` | `../dist` | Where the built UI lives. |

A forced answer says so: the JSON has `"forced": true` and the page notes it. Without the server
override you can still try dates: `nemo plugins call is-it-christmas -q date=2026-12-25`.

Changing `PUBLIC_URL`, the endpoints, or the UI changes the manifest, so after that:

```bash
nemo plugins preview  http://127.0.0.1:6010/plugin.json            # note the manifest sha256
nemo plugins uninstall is-it-christmas --yes
nemo plugins install  http://127.0.0.1:6010/plugin.json --manifest-sha256 <sha256> --yes
```

## Run it

```bash
# 1. Build the UI (once, and after any change under web/).
cd plugins/is-it-christmas/web && pnpm install && pnpm build

# 2. Start the backend (loopback only; PORT and PUBLIC_URL are optional).
cd ../backend && cargo run

# 3. Register it with a running platform, then open Studio.
nemo plugins preview http://127.0.0.1:6010/plugin.json
nemo plugins install http://127.0.0.1:6010/plugin.json
# Studio: Community -> Is it Christmas?
```

Without a built UI the manifest is API-only: no bundle, no nav entry.

Try the API directly, and through the gateway once installed:

```bash
curl "http://127.0.0.1:6010/v1/workspaces/default/is-it-christmas?date=2026-12-25"
curl "http://127.0.0.1:8080/apis/plugins/is-it-christmas/v2/workspaces/default/is-it-christmas"
```

## Test

```bash
cd backend && cargo test
cd ../web && pnpm typecheck && pnpm test
```
